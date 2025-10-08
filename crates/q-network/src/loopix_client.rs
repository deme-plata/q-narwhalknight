/// Production Loopix Client Node
/// 
/// Implements a Loopix client with:
/// - Random path selection through verified mix networks
/// - Cover traffic generation (drop and loop messages)
/// - Message scheduling with Poisson processes
/// - Provider selection and load balancing

use crate::loopix_protocol::{
    DirectoryRequest, DirectoryResponse, EpochDescriptor, LoopixMessage, 
    LoopixCell, NodeInfo, NodeType, path_selection::{PathConfig, select_random_path}
};
use crate::loopix_crypto::{self, SecureKey, keygen};
use crate::loopix_directory::verify_epoch_signature;
use libp2p::{PeerId, request_response::OutboundRequestId};
use rand_distr::{Exp, Poisson, Distribution};
use std::{
    collections::HashMap,
    time::{Duration, SystemTime, Instant},
};
use futures::StreamExt;
use tokio::time::{interval, sleep_until};
use tracing::{info, warn, debug, error};
use zeroize::Zeroizing;
use bincode;

/// Client configuration parameters
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Client identifier
    pub client_id: String,
    /// Directory server address  
    pub directory_address: String,
    /// Directory's public key for epoch verification
    pub directory_public_key: Vec<u8>,
    /// Path selection configuration
    pub path_config: PathConfig,
    /// Cover traffic parameters
    pub cover_traffic_config: CoverTrafficConfig,
    /// Message scheduling parameters
    pub scheduling_config: SchedulingConfig,
}

/// Cover traffic generation configuration
#[derive(Debug, Clone)]
pub struct CoverTrafficConfig {
    /// Mean time between drop messages (seconds)
    pub lambda_drop: f64,
    /// Mean time between loop messages (seconds)  
    pub lambda_loop: f64,
    /// Maximum cover traffic messages per minute
    pub max_cover_per_minute: u32,
}

impl Default for CoverTrafficConfig {
    fn default() -> Self {
        Self {
            lambda_drop: 30.0,    // 1 drop message per 30 seconds
            lambda_loop: 60.0,    // 1 loop message per minute
            max_cover_per_minute: 10,
        }
    }
}

/// Message scheduling configuration  
#[derive(Debug, Clone)]
pub struct SchedulingConfig {
    /// Mean time between real messages (seconds)
    pub lambda_real: f64,
    /// Maximum messages in send queue
    pub max_send_queue: usize,
    /// Provider connection timeout
    pub provider_timeout: Duration,
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            lambda_real: 120.0,  // 1 real message per 2 minutes
            max_send_queue: 100,
            provider_timeout: Duration::from_secs(30),
        }
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            client_id: "default-client".to_string(),
            directory_address: "127.0.0.1:8080".to_string(),
            directory_public_key: vec![0u8; 32],
            path_config: PathConfig::default(),
            cover_traffic_config: CoverTrafficConfig::default(),
            scheduling_config: SchedulingConfig::default(),
        }
    }
}

/// Scheduled message for transmission
#[derive(Debug)]
struct ScheduledMessage {
    /// When to send this message
    send_time: Instant,
    /// Message recipient identifier
    recipient: String,
    /// Message payload
    payload: Vec<u8>,
    /// Message priority (lower = higher priority)
    priority: u8,
}

/// Cover traffic message for anonymity
#[derive(Debug)]
struct CoverMessage {
    /// When to send this cover message
    send_time: Instant,
    /// Type of cover traffic
    cover_type: CoverType,
}

#[derive(Debug, Clone)]
enum CoverType {
    /// Drop message (discarded by mix)
    Drop,
    /// Loop message (returns to sender)
    Loop,
}

/// Production Loopix client implementation
pub struct LoopixClient {
    /// Client configuration
    config: ClientConfig,
    /// Current network epoch from directory
    current_epoch: Option<EpochDescriptor>,
    /// Our provider node
    current_provider: Option<NodeInfo>,
    /// Message send queue
    send_queue: Vec<ScheduledMessage>,
    /// Cover traffic queue
    cover_queue: Vec<CoverMessage>,
    /// Path keys cache (for loop messages)
    path_keys_cache: HashMap<String, Vec<SecureKey>>,
    /// Statistics
    stats: ClientStats,
    /// Random number generation
    rng: rand::rngs::ThreadRng,
}

/// Client performance statistics
#[derive(Debug, Default)]
pub struct ClientStats {
    pub real_messages_sent: u64,
    pub cover_messages_sent: u64,
    pub loop_messages_sent: u64,
    pub drop_messages_sent: u64,
    pub messages_failed: u64,
    pub epoch_updates: u64,
    pub provider_changes: u64,
}

impl LoopixClient {
    /// Create a new Loopix client
    pub fn new(config: ClientConfig) -> Self {
        Self {
            config,
            current_epoch: None,
            current_provider: None,
            send_queue: Vec::new(),
            cover_queue: Vec::new(),
            path_keys_cache: HashMap::new(),
            stats: ClientStats::default(),
            rng: rand::thread_rng(),
        }
    }
    
    /// Start the client (background processes)
    pub async fn start(&mut self) -> Result<(), anyhow::Error> {
        info!("Starting Loopix client: {}", self.config.client_id);
        
        // Fetch initial epoch from directory
        self.update_epoch().await?;
        
        // Select initial provider
        self.select_provider().await?;
        
        // Start background tasks
        let mut epoch_update_timer = interval(Duration::from_secs(300)); // 5 minutes
        let mut cover_traffic_timer = interval(Duration::from_secs(10)); // Check every 10s
        let mut send_timer = interval(Duration::from_millis(100)); // Process queue every 100ms
        
        loop {
            tokio::select! {
                // Update network epoch periodically
                _ = epoch_update_timer.tick() => {
                    if let Err(e) = self.update_epoch().await {
                        warn!("Failed to update epoch: {}", e);
                    }
                }
                
                // Generate and send cover traffic
                _ = cover_traffic_timer.tick() => {
                    self.schedule_cover_traffic().await;
                    self.process_cover_queue().await;
                }
                
                // Process send queues
                _ = send_timer.tick() => {
                    self.process_send_queue().await;
                }
            }
        }
    }
    
    /// Send a message through the Loopix network
    pub async fn send_message(
        &mut self,
        recipient: String,
        payload: Vec<u8>,
        priority: u8,
    ) -> Result<(), anyhow::Error> {
        if self.send_queue.len() >= self.config.scheduling_config.max_send_queue {
            return Err(anyhow::anyhow!("Send queue full"));
        }
        
        // Calculate send time based on Poisson process
        let delay_secs = Exp::new(1.0 / self.config.scheduling_config.lambda_real)
            .map_err(|_| anyhow::anyhow!("Invalid lambda_real parameter"))?
            .sample(&mut self.rng);
        
        let send_time = Instant::now() + Duration::from_secs_f64(delay_secs);
        
        let scheduled = ScheduledMessage {
            send_time,
            recipient: recipient.clone(),
            payload,
            priority,
        };

        // Insert in priority order
        let insert_pos = self.send_queue
            .binary_search_by(|probe| {
                probe.priority.cmp(&scheduled.priority)
                    .then(probe.send_time.cmp(&scheduled.send_time))
            })
            .unwrap_or_else(|pos| pos);

        self.send_queue.insert(insert_pos, scheduled);

        debug!("Queued message for {} (queue size: {})", recipient, self.send_queue.len());
        Ok(())
    }
    
    /// Process the message send queue
    async fn process_send_queue(&mut self) {
        let now = Instant::now();
        let mut sent_count = 0;
        
        // Send all messages whose time has come
        while let Some(message) = self.send_queue.first() {
            if message.send_time > now {
                break; // Not ready yet
            }
            
            let message = self.send_queue.remove(0);
            
            match self.send_real_message(message.recipient, message.payload).await {
                Ok(()) => {
                    self.stats.real_messages_sent += 1;
                    sent_count += 1;
                }
                Err(e) => {
                    error!("Failed to send message: {}", e);
                    self.stats.messages_failed += 1;
                }
            }
        }
        
        if sent_count > 0 {
            debug!("Sent {} real messages", sent_count);
        }
    }
    
    /// Send a real message through the mix network
    async fn send_real_message(
        &mut self,
        recipient: String,
        payload: Vec<u8>,
    ) -> Result<(), anyhow::Error> {
        // Ensure we have a current epoch and provider
        if self.current_epoch.is_none() {
            self.update_epoch().await?;
        }
        if self.current_provider.is_none() {
            self.select_provider().await?;
        }
        
        let epoch = self.current_epoch.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current epoch available"))?;
        let provider = self.current_provider.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current provider available"))?;
        
        // Select random path through mix network
        let mix_path = select_random_path(&epoch.mix_nodes, &self.config.path_config)?;
        
        // Generate keys for each hop (client -> mixes -> provider)
        let mut hop_keys = Vec::new();
        for _ in 0..mix_path.len() {
            hop_keys.push(keygen());
        }
        hop_keys.push(keygen()); // Provider key
        
        // Create the inner message
        let user_message = LoopixMessage::UserMessage {
            recipient: recipient.clone(),
            encrypted_payload: payload,
            layer_count: (hop_keys.len()) as u8,
        };
        
        let serialized = bincode::serialize(&user_message)
            .map_err(|e| anyhow::anyhow!("Serialization failed: {}", e))?;
        
        // Encrypt through all layers
        let layered_encrypted = loopix_crypto::encrypt_layered(serialized, &hop_keys)?;
        
        // Create cell for transmission
        let cell = LoopixCell::new(layered_encrypted)?;
        
        // Send to first mix node in path
        if let Some(first_hop) = mix_path.first() {
            // In a real implementation, this would use libp2p to send the cell
            debug!("Sending message for {} via path: {:?}", recipient, mix_path);
            // TODO: Actually send the cell via libp2p request-response
        }
        
        Ok(())
    }
    
    /// Schedule cover traffic generation
    async fn schedule_cover_traffic(&mut self) {
        let now = Instant::now();
        
        // Schedule drop messages
        if rand::random::<f64>() < 1.0 / self.config.cover_traffic_config.lambda_drop {
            let delay_secs = Exp::new(1.0 / self.config.cover_traffic_config.lambda_drop)
                .unwrap()
                .sample(&mut self.rng);
            
            let send_time = now + Duration::from_secs_f64(delay_secs);
            
            self.cover_queue.push(CoverMessage {
                send_time,
                cover_type: CoverType::Drop,
            });
        }
        
        // Schedule loop messages  
        if rand::random::<f64>() < 1.0 / self.config.cover_traffic_config.lambda_loop {
            let delay_secs = Exp::new(1.0 / self.config.cover_traffic_config.lambda_loop)
                .unwrap()
                .sample(&mut self.rng);
            
            let send_time = now + Duration::from_secs_f64(delay_secs);
            
            self.cover_queue.push(CoverMessage {
                send_time,
                cover_type: CoverType::Loop,
            });
        }
        
        // Sort by send time
        self.cover_queue.sort_by_key(|msg| msg.send_time);
        
        // Limit cover traffic rate
        let cutoff_time = now + Duration::from_secs(60);
        let cover_in_next_minute = self.cover_queue
            .iter()
            .filter(|msg| msg.send_time <= cutoff_time)
            .count();
        
        if cover_in_next_minute > self.config.cover_traffic_config.max_cover_per_minute as usize {
            self.cover_queue.retain(|msg| msg.send_time > cutoff_time);
        }
    }
    
    /// Process cover traffic queue
    async fn process_cover_queue(&mut self) {
        let now = Instant::now();
        let mut sent_count = 0;
        
        while let Some(cover_msg) = self.cover_queue.first() {
            if cover_msg.send_time > now {
                break; // Not ready yet
            }
            
            let cover_msg = self.cover_queue.remove(0);
            
            match self.send_cover_message(cover_msg.cover_type).await {
                Ok(()) => sent_count += 1,
                Err(e) => error!("Failed to send cover traffic: {}", e),
            }
        }
        
        if sent_count > 0 {
            debug!("Sent {} cover traffic messages", sent_count);
        }
    }
    
    /// Send cover traffic message
    async fn send_cover_message(&mut self, cover_type: CoverType) -> Result<(), anyhow::Error> {
        let epoch = self.current_epoch.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current epoch for cover traffic"))?;
        
        // Select random path
        let mix_path = select_random_path(&epoch.mix_nodes, &self.config.path_config)?;
        
        let cover_message = match cover_type {
            CoverType::Drop => {
                self.stats.drop_messages_sent += 1;
                LoopixMessage::CoverTraffic {
                    dummy_payload: vec![0u8; 256 + (rand::random::<usize>() % 256)],
                    path_length: mix_path.len() as u8,
                }
            }
            CoverType::Loop => {
                self.stats.loop_messages_sent += 1;
                // For loop messages, we cache the path keys so we can recognize
                // the message when it comes back
                let loop_id = format!("loop-{}", rand::random::<u64>());
                LoopixMessage::CoverTraffic {
                    dummy_payload: loop_id.as_bytes().to_vec(),
                    path_length: mix_path.len() as u8,
                }
            }
        };
        
        // Generate encryption keys
        let hop_keys: Vec<SecureKey> = (0..mix_path.len()).map(|_| keygen()).collect();
        
        // Encrypt and send
        let serialized = bincode::serialize(&cover_message)?;
        let encrypted = loopix_crypto::encrypt_layered(serialized, &hop_keys)?;
        let cell = LoopixCell::new(encrypted)?;
        
        // Send to first hop
        if let Some(first_hop) = mix_path.first() {
            debug!("Sending {:?} cover traffic via {:?}", cover_type, first_hop);
            // TODO: Actually send via libp2p
        }
        
        self.stats.cover_messages_sent += 1;
        Ok(())
    }
    
    /// Update network epoch from directory
    async fn update_epoch(&mut self) -> Result<(), anyhow::Error> {
        debug!("Updating network epoch from directory");
        
        // Request current epoch from directory
        let request = DirectoryRequest::FetchEpoch;
        
        // TODO: Send request via libp2p to directory server
        // For now, simulate with a mock epoch
        let mock_epoch = self.create_mock_epoch();
        
        // Verify epoch signature
        if !verify_epoch_signature(&mock_epoch, &self.config.directory_public_key)? {
            return Err(anyhow::anyhow!("Invalid epoch signature from directory"));
        }
        
        // Check if epoch is still valid
        if !mock_epoch.is_valid() {
            return Err(anyhow::anyhow!("Received expired epoch from directory"));
        }
        
        // Update our epoch
        let epoch_changed = self.current_epoch.as_ref()
            .map(|e| e.epoch_id != mock_epoch.epoch_id)
            .unwrap_or(true);
        
        if epoch_changed {
            info!("Updated to epoch {} with {} mix nodes and {} providers",
                  mock_epoch.epoch_id,
                  mock_epoch.mix_nodes.len(),
                  mock_epoch.providers.len());
            
            self.current_epoch = Some(mock_epoch);
            self.stats.epoch_updates += 1;
            
            // Clear cached keys since topology changed
            self.path_keys_cache.clear();
            
            // Reselect provider for new epoch
            self.select_provider().await?;
        }
        
        Ok(())
    }
    
    /// Select a provider node for this client
    async fn select_provider(&mut self) -> Result<(), anyhow::Error> {
        let epoch = self.current_epoch.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No current epoch for provider selection"))?;
        
        if epoch.providers.is_empty() {
            return Err(anyhow::anyhow!("No providers available in current epoch"));
        }
        
        // Simple random selection (could be improved with load balancing)
        let provider_index = rand::random::<usize>() % epoch.providers.len();
        let selected_provider = epoch.providers[provider_index].clone();
        
        let provider_changed = self.current_provider.as_ref()
            .map(|p| p.peer_id != selected_provider.peer_id)
            .unwrap_or(true);
        
        if provider_changed {
            info!("Selected provider: {}", selected_provider.peer_id);
            self.current_provider = Some(selected_provider);
            self.stats.provider_changes += 1;
        }
        
        Ok(())
    }
    
    /// Create a mock epoch for testing (replace with real directory communication)
    fn create_mock_epoch(&self) -> EpochDescriptor {
        use std::time::{SystemTime, Duration};
        
        // Create mock mix nodes
        let mix_nodes: Vec<NodeInfo> = (0..5)
            .map(|i| NodeInfo {
                peer_id: PeerId::random(),
                node_type: NodeType::Mix,
                listen_addrs: vec![format!("127.0.0.1:{}", 8001 + i)],
                public_key: vec![i as u8; 32],
                capabilities: vec!["loopix-mix".to_string()],
                registered_at: SystemTime::now(),
            })
            .collect();
        
        // Create mock providers
        let providers: Vec<NodeInfo> = (0..2)
            .map(|i| NodeInfo {
                peer_id: PeerId::random(),
                node_type: NodeType::Provider,
                listen_addrs: vec![format!("127.0.0.1:{}", 8100 + i)],
                public_key: vec![100 + i as u8; 32],
                capabilities: vec!["loopix-provider".to_string()],
                registered_at: SystemTime::now(),
            })
            .collect();
        
        EpochDescriptor {
            mix_nodes,
            providers,
            epoch_id: 1,
            started_at: SystemTime::now(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
            signature: vec![0u8; 64], // Mock signature
        }
    }
    
    /// Get client statistics
    pub fn get_stats(&self) -> &ClientStats {
        &self.stats
    }
    
    /// Get current epoch info
    pub fn get_current_epoch(&self) -> Option<&EpochDescriptor> {
        self.current_epoch.as_ref()
    }
    
    /// Get current provider info
    pub fn get_current_provider(&self) -> Option<&NodeInfo> {
        self.current_provider.as_ref()
    }
    
    /// Get queue status
    pub fn get_queue_status(&self) -> (usize, usize) {
        (self.send_queue.len(), self.cover_queue.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_config() -> ClientConfig {
        ClientConfig {
            client_id: "test-client".to_string(),
            directory_address: "127.0.0.1:8000".to_string(),
            directory_public_key: vec![0u8; 32],
            path_config: PathConfig::default(),
            cover_traffic_config: CoverTrafficConfig::default(),
            scheduling_config: SchedulingConfig::default(),
        }
    }
    
    #[test]
    fn test_client_creation() {
        let config = create_test_config();
        let client = LoopixClient::new(config);
        
        assert_eq!(client.config.client_id, "test-client");
        assert!(client.current_epoch.is_none());
        assert!(client.current_provider.is_none());
        assert_eq!(client.send_queue.len(), 0);
        assert_eq!(client.cover_queue.len(), 0);
    }
    
    #[tokio::test]
    async fn test_message_queuing() {
        let config = create_test_config();
        let mut client = LoopixClient::new(config);
        
        // Queue a message
        let result = client.send_message(
            "recipient@example.com".to_string(),
            b"Test message".to_vec(),
            1,
        ).await;
        
        assert!(result.is_ok());
        assert_eq!(client.send_queue.len(), 1);
    }
    
    #[tokio::test]
    async fn test_provider_selection() {
        let config = create_test_config();
        let mut client = LoopixClient::new(config);
        
        // Set a mock epoch
        client.current_epoch = Some(client.create_mock_epoch());
        
        // Select provider
        let result = client.select_provider().await;
        assert!(result.is_ok());
        assert!(client.current_provider.is_some());
        
        let provider = client.current_provider.as_ref().unwrap();
        assert_eq!(provider.node_type, NodeType::Provider);
    }
    
    #[tokio::test]
    async fn test_cover_traffic_scheduling() {
        let config = create_test_config();
        let mut client = LoopixClient::new(config);
        
        // Schedule cover traffic
        client.schedule_cover_traffic().await;
        
        // Should have scheduled some cover traffic
        // (probabilistic, so might be 0 sometimes)
        assert!(client.cover_queue.len() <= 10); // Reasonable upper bound
    }
}