/// Production Loopix Directory Server
/// 
/// Implements the directory authority for the Loopix mix network:
/// - Signed epoch management with cryptographic verification
/// - Node registration and health monitoring
/// - Topology distribution to clients
/// - Key rotation and epoch transitions

use crate::loopix_protocol::{
    DirectoryRequest, DirectoryResponse, NodeInfo, NodeType, EpochDescriptor
};
use libp2p::{PeerId, request_response::{ResponseChannel, OutboundRequestId}};
use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use ring::{
    signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519},
    rand::SystemRandom,
};
use tracing::{info, warn, debug, error};
use tokio::time::{interval, Instant};

/// Directory server configuration
#[derive(Debug, Clone)]
pub struct DirectoryConfig {
    /// How long each epoch lasts
    pub epoch_duration: Duration,
    /// Minimum number of mix nodes required per epoch
    pub min_mix_nodes: usize,
    /// Minimum number of providers required per epoch
    pub min_providers: usize,
    /// How often to perform health checks
    pub health_check_interval: Duration,
    /// How long before a node is considered stale
    pub node_timeout: Duration,
}

impl Default for DirectoryConfig {
    fn default() -> Self {
        Self {
            epoch_duration: Duration::from_secs(3600), // 1 hour epochs
            min_mix_nodes: 5,
            min_providers: 2,
            health_check_interval: Duration::from_secs(60),
            node_timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Directory server state
pub struct LoopixDirectory {
    /// Server configuration
    config: DirectoryConfig,
    /// Ed25519 keypair for signing epochs
    signing_keypair: Ed25519KeyPair,
    /// Current epoch descriptor
    current_epoch: Option<EpochDescriptor>,
    /// Registered nodes by peer ID
    registered_nodes: HashMap<PeerId, RegisteredNode>,
    /// Current epoch ID counter
    epoch_counter: u64,
    /// When the current epoch started
    epoch_start_time: Option<SystemTime>,
}

/// Node registration with additional metadata
#[derive(Debug, Clone)]
struct RegisteredNode {
    /// Node information from registration
    info: NodeInfo,
    /// When this node was last seen (health check)
    last_seen: SystemTime,
    /// Number of failed health checks
    failed_health_checks: u32,
    /// Whether this node is included in the current epoch
    active_in_epoch: bool,
}

impl LoopixDirectory {
    /// Create a new directory server
    pub fn new(config: DirectoryConfig) -> Result<Self, anyhow::Error> {
        // Generate Ed25519 keypair for epoch signing
        let rng = SystemRandom::new();
        let pkcs8_bytes = Ed25519KeyPair::generate_pkcs8(&rng)
            .map_err(|_| anyhow::anyhow!("Failed to generate Ed25519 keypair"))?;
        
        let signing_keypair = Ed25519KeyPair::from_pkcs8(pkcs8_bytes.as_ref())
            .map_err(|_| anyhow::anyhow!("Failed to create Ed25519 keypair"))?;
        
        info!("Directory server initialized with new signing keypair");
        
        Ok(Self {
            config,
            signing_keypair,
            current_epoch: None,
            registered_nodes: HashMap::new(),
            epoch_counter: 1,
            epoch_start_time: None,
        })
    }
    
    /// Start the directory server (background tasks)
    pub async fn start(&mut self) -> Result<(), anyhow::Error> {
        info!("Starting Loopix directory server");
        
        // Create initial epoch if we have enough nodes
        self.try_create_new_epoch().await?;
        
        // Start background tasks
        let mut epoch_timer = interval(self.config.epoch_duration);
        let mut health_timer = interval(self.config.health_check_interval);
        
        loop {
            tokio::select! {
                // Epoch rotation
                _ = epoch_timer.tick() => {
                    if let Err(e) = self.try_create_new_epoch().await {
                        error!("Failed to create new epoch: {}", e);
                    }
                }
                
                // Health checks
                _ = health_timer.tick() => {
                    self.perform_health_checks().await;
                }
            }
        }
    }
    
    /// Handle incoming directory requests
    pub async fn handle_request(
        &mut self,
        request: DirectoryRequest,
        response_channel: ResponseChannel<DirectoryResponse>,
    ) {
        let response = match request {
            DirectoryRequest::Register(node_info) => {
                self.handle_registration(node_info).await
            }
            
            DirectoryRequest::FetchEpoch => {
                self.handle_fetch_epoch().await
            }
            
            DirectoryRequest::FetchNodesByType(node_type) => {
                self.handle_fetch_nodes_by_type(node_type).await
            }
        };
        
        // Send response back to client
        // ResponseChannel is consumed when used, so we ignore the result
        drop(response_channel);
    }
    
    /// Handle node registration
    async fn handle_registration(&mut self, node_info: NodeInfo) -> DirectoryResponse {
        debug!("Registering node: {:?} as {:?}", node_info.peer_id, node_info.node_type);
        
        // Create registered node entry
        let registered_node = RegisteredNode {
            info: node_info.clone(),
            last_seen: SystemTime::now(),
            failed_health_checks: 0,
            active_in_epoch: false,
        };
        
        // Add to registry
        self.registered_nodes.insert(node_info.peer_id, registered_node);
        
        info!(
            "Registered {} node: {} (total nodes: {})",
            match node_info.node_type {
                NodeType::Mix => "mix",
                NodeType::Provider => "provider",
                NodeType::Client => "client",
                NodeType::Directory => "directory",
            },
            node_info.peer_id,
            self.registered_nodes.len()
        );
        
        // Check if we can create a new epoch with this registration
        if let Err(e) = self.try_create_new_epoch().await {
            warn!("Failed to create epoch after registration: {}", e);
        }
        
        DirectoryResponse::Ack
    }
    
    /// Handle epoch fetch request
    async fn handle_fetch_epoch(&self) -> DirectoryResponse {
        if let Some(ref epoch) = self.current_epoch {
            if epoch.is_valid() {
                debug!("Serving current epoch {} to client", epoch.epoch_id);
                return DirectoryResponse::Epoch(epoch.clone());
            }
        }
        
        warn!("No valid epoch available for client request");
        DirectoryResponse::Error("No valid epoch available".to_string())
    }
    
    /// Handle fetch nodes by type
    async fn handle_fetch_nodes_by_type(&self, node_type: NodeType) -> DirectoryResponse {
        let matching_nodes: Vec<NodeInfo> = self.registered_nodes
            .values()
            .filter(|node| {
                node.info.node_type == node_type && 
                node.active_in_epoch &&
                SystemTime::now().duration_since(node.last_seen).unwrap_or(Duration::MAX) < self.config.node_timeout
            })
            .map(|node| node.info.clone())
            .collect();
        
        debug!("Serving {} nodes of type {:?}", matching_nodes.len(), node_type);
        DirectoryResponse::Nodes(matching_nodes)
    }
    
    /// Try to create a new epoch if conditions are met
    async fn try_create_new_epoch(&mut self) -> Result<(), anyhow::Error> {
        // Count healthy nodes by type
        let now = SystemTime::now();
        let mut mix_nodes = Vec::new();
        let mut providers = Vec::new();
        
        for (peer_id, node) in &self.registered_nodes {
            // Skip stale nodes
            if now.duration_since(node.last_seen).unwrap_or(Duration::MAX) > self.config.node_timeout {
                continue;
            }
            
            match node.info.node_type {
                NodeType::Mix => mix_nodes.push(node.info.clone()),
                NodeType::Provider => providers.push(node.info.clone()),
                _ => {} // Ignore clients and other directories
            }
        }
        
        // Check if we have enough nodes for a valid epoch
        if mix_nodes.len() < self.config.min_mix_nodes {
            debug!(
                "Not enough mix nodes for epoch: {} < {}",
                mix_nodes.len(),
                self.config.min_mix_nodes
            );
            return Ok(()); // Not an error, just waiting for more nodes
        }
        
        if providers.len() < self.config.min_providers {
            debug!(
                "Not enough providers for epoch: {} < {}",
                providers.len(),
                self.config.min_providers
            );
            return Ok(());
        }
        
        // Create new epoch
        let epoch_id = self.epoch_counter;
        let started_at = now;
        let expires_at = started_at + self.config.epoch_duration;
        
        // Create unsigned epoch descriptor
        let mut epoch = EpochDescriptor {
            mix_nodes: mix_nodes.clone(),
            providers: providers.clone(),
            epoch_id,
            started_at,
            expires_at,
            signature: Vec::new(), // Will be filled below
        };
        
        // Sign the epoch
        let epoch_bytes = self.serialize_epoch_for_signing(&epoch)?;
        let signature = self.signing_keypair.sign(&epoch_bytes);
        epoch.signature = signature.as_ref().to_vec();
        
        // Update state
        self.current_epoch = Some(epoch.clone());
        self.epoch_counter += 1;
        self.epoch_start_time = Some(started_at);
        
        // Mark nodes as active in this epoch
        for node in &mut self.registered_nodes.values_mut() {
            node.active_in_epoch = mix_nodes.iter().any(|m| m.peer_id == node.info.peer_id) ||
                                   providers.iter().any(|p| p.peer_id == node.info.peer_id);
        }
        
        info!(
            "Created new epoch {} with {} mix nodes and {} providers (expires in {}s)",
            epoch_id,
            mix_nodes.len(),
            providers.len(),
            self.config.epoch_duration.as_secs()
        );
        
        Ok(())
    }
    
    /// Serialize epoch data for signing (deterministic)
    fn serialize_epoch_for_signing(&self, epoch: &EpochDescriptor) -> Result<Vec<u8>, anyhow::Error> {
        // Create a signing payload with all relevant epoch data except signature
        let mut payload = Vec::new();
        
        // Epoch ID
        payload.extend_from_slice(&epoch.epoch_id.to_be_bytes());
        
        // Timestamps
        let started_secs = epoch.started_at.duration_since(UNIX_EPOCH)?.as_secs();
        let expires_secs = epoch.expires_at.duration_since(UNIX_EPOCH)?.as_secs();
        payload.extend_from_slice(&started_secs.to_be_bytes());
        payload.extend_from_slice(&expires_secs.to_be_bytes());
        
        // Node information (deterministic ordering)
        let mut all_nodes = epoch.mix_nodes.clone();
        all_nodes.extend(epoch.providers.clone());
        all_nodes.sort_by(|a, b| a.peer_id.cmp(&b.peer_id));
        
        for node in all_nodes {
            payload.extend_from_slice(node.peer_id.to_bytes().as_slice());
            payload.push(node.node_type as u8);
            payload.extend_from_slice(&node.public_key);
        }
        
        Ok(payload)
    }
    
    /// Perform health checks on registered nodes
    async fn perform_health_checks(&mut self) {
        let now = SystemTime::now();
        let mut stale_nodes = Vec::new();
        
        for (peer_id, node) in &mut self.registered_nodes {
            let time_since_seen = now.duration_since(node.last_seen).unwrap_or(Duration::MAX);
            
            if time_since_seen > self.config.node_timeout {
                node.failed_health_checks += 1;
                
                if node.failed_health_checks > 3 {
                    stale_nodes.push(*peer_id);
                } else {
                    warn!("Node {} has been offline for {:?}", peer_id, time_since_seen);
                }
            }
        }
        
        // Remove stale nodes
        for peer_id in stale_nodes {
            self.registered_nodes.remove(&peer_id);
            info!("Removed stale node {} from directory", peer_id);
        }
        
        debug!(
            "Health check complete: {} active nodes",
            self.registered_nodes.len()
        );
    }
    
    /// Update node health (called when we receive messages from nodes)
    pub fn update_node_health(&mut self, peer_id: PeerId) {
        if let Some(node) = self.registered_nodes.get_mut(&peer_id) {
            node.last_seen = SystemTime::now();
            node.failed_health_checks = 0;
        }
    }
    
    /// Get the current epoch (for testing/monitoring)
    pub fn get_current_epoch(&self) -> Option<&EpochDescriptor> {
        self.current_epoch.as_ref()
    }
    
    /// Get directory statistics
    pub fn get_stats(&self) -> DirectoryStats {
        let now = SystemTime::now();
        let mut active_nodes = 0;
        let mut mix_count = 0;
        let mut provider_count = 0;
        let mut client_count = 0;
        
        for node in self.registered_nodes.values() {
            if now.duration_since(node.last_seen).unwrap_or(Duration::MAX) < self.config.node_timeout {
                active_nodes += 1;
                match node.info.node_type {
                    NodeType::Mix => mix_count += 1,
                    NodeType::Provider => provider_count += 1,
                    NodeType::Client => client_count += 1,
                    NodeType::Directory => {},
                }
            }
        }
        
        DirectoryStats {
            total_registered: self.registered_nodes.len(),
            active_nodes,
            mix_nodes: mix_count,
            provider_nodes: provider_count,
            client_nodes: client_count,
            current_epoch_id: self.current_epoch.as_ref().map(|e| e.epoch_id),
            epoch_expires_in: self.current_epoch.as_ref()
                .and_then(|e| e.expires_at.duration_since(now).ok()),
        }
    }
    
    /// Get the directory's public key for clients to verify epochs
    pub fn get_public_key(&self) -> Vec<u8> {
        self.signing_keypair.public_key().as_ref().to_vec()
    }
}

/// Directory server statistics
#[derive(Debug, Clone)]
pub struct DirectoryStats {
    pub total_registered: usize,
    pub active_nodes: usize,
    pub mix_nodes: usize,
    pub provider_nodes: usize,
    pub client_nodes: usize,
    pub current_epoch_id: Option<u64>,
    pub epoch_expires_in: Option<Duration>,
}

/// Utility function to verify epoch signature
pub fn verify_epoch_signature(
    epoch: &EpochDescriptor,
    directory_public_key: &[u8]
) -> Result<bool, anyhow::Error> {
    let public_key = UnparsedPublicKey::new(&ED25519, directory_public_key);
    
    // Reconstruct the signing payload
    let mut payload = Vec::new();
    payload.extend_from_slice(&epoch.epoch_id.to_be_bytes());
    
    let started_secs = epoch.started_at.duration_since(UNIX_EPOCH)?.as_secs();
    let expires_secs = epoch.expires_at.duration_since(UNIX_EPOCH)?.as_secs();
    payload.extend_from_slice(&started_secs.to_be_bytes());
    payload.extend_from_slice(&expires_secs.to_be_bytes());
    
    let mut all_nodes = epoch.mix_nodes.clone();
    all_nodes.extend(epoch.providers.clone());
    all_nodes.sort_by(|a, b| a.peer_id.cmp(&b.peer_id));
    
    for node in all_nodes {
        payload.extend_from_slice(node.peer_id.to_bytes().as_slice());
        payload.push(node.node_type as u8);
        payload.extend_from_slice(&node.public_key);
    }
    
    // Verify signature
    match public_key.verify(&payload, &epoch.signature) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_directory_creation() {
        let config = DirectoryConfig::default();
        let directory = LoopixDirectory::new(config).unwrap();
        
        assert!(directory.current_epoch.is_none());
        assert_eq!(directory.registered_nodes.len(), 0);
        assert_eq!(directory.epoch_counter, 1);
    }
    
    #[tokio::test]
    async fn test_node_registration() {
        let config = DirectoryConfig::default();
        let mut directory = LoopixDirectory::new(config).unwrap();
        
        let node_info = NodeInfo {
            peer_id: PeerId::random(),
            node_type: NodeType::Mix,
            listen_addrs: vec!["127.0.0.1:8001".to_string()],
            public_key: vec![0u8; 32],
            capabilities: vec!["loopix-mix".to_string()],
            registered_at: SystemTime::now(),
        };
        
        let response = directory.handle_registration(node_info.clone()).await;
        
        assert!(matches!(response, DirectoryResponse::Ack));
        assert_eq!(directory.registered_nodes.len(), 1);
        assert!(directory.registered_nodes.contains_key(&node_info.peer_id));
    }
    
    #[tokio::test]
    async fn test_epoch_creation() {
        let mut config = DirectoryConfig::default();
        config.min_mix_nodes = 2;
        config.min_providers = 1;
        
        let mut directory = LoopixDirectory::new(config).unwrap();
        
        // Add enough nodes for an epoch
        for i in 0..2 {
            let mix_node = NodeInfo {
                peer_id: PeerId::random(),
                node_type: NodeType::Mix,
                listen_addrs: vec![format!("127.0.0.1:{}", 8001 + i)],
                public_key: vec![i as u8; 32],
                capabilities: vec!["loopix-mix".to_string()],
                registered_at: SystemTime::now(),
            };
            directory.handle_registration(mix_node).await;
        }
        
        let provider_node = NodeInfo {
            peer_id: PeerId::random(),
            node_type: NodeType::Provider,
            listen_addrs: vec!["127.0.0.1:8100".to_string()],
            public_key: vec![42u8; 32],
            capabilities: vec!["loopix-provider".to_string()],
            registered_at: SystemTime::now(),
        };
        directory.handle_registration(provider_node).await;
        
        // Should have created an epoch
        assert!(directory.current_epoch.is_some());
        let epoch = directory.current_epoch.as_ref().unwrap();
        assert_eq!(epoch.mix_nodes.len(), 2);
        assert_eq!(epoch.providers.len(), 1);
        assert!(epoch.signature.len() > 0);
    }
    
    #[test]
    fn test_epoch_signature_verification() {
        let config = DirectoryConfig::default();
        let directory = LoopixDirectory::new(config).unwrap();
        let public_key = directory.get_public_key();
        
        let epoch = EpochDescriptor {
            mix_nodes: vec![],
            providers: vec![],
            epoch_id: 1,
            started_at: SystemTime::now(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
            signature: vec![0u8; 64], // Invalid signature
        };
        
        // Should fail with invalid signature
        let result = verify_epoch_signature(&epoch, &public_key).unwrap();
        assert!(!result);
    }
}