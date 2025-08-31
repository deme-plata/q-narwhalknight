/// Quantum-enhanced peer discovery for Q-Network
/// Supports capability negotiation and post-quantum algorithm discovery

use q_types::*;
use anyhow::Result;
use libp2p::{
    core::ProtocolName,
    request_response::{self, ProtocolSupport, ResponseChannel},
    swarm::{NetworkBehaviour, SwarmEvent},
    PeerId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Peer discovery behavior
#[derive(NetworkBehaviour)]
pub struct Behaviour {
    request_response: request_response::Behaviour<PeerDiscoveryCodec>,
}

impl Behaviour {
    pub fn new() -> Self {
        let protocols = std::iter::once((PeerDiscoveryProtocol(), ProtocolSupport::Full));
        let cfg = request_response::Config::default();
        
        Self {
            request_response: request_response::Behaviour::new(protocols, cfg),
        }
    }

    /// Request peer capabilities
    pub fn request_capabilities(&mut self, peer_id: PeerId) {
        let request = PeerDiscoveryRequest::CapabilityQuery;
        self.request_response.send_request(&peer_id, request);
        debug!("Requested capabilities from peer {}", peer_id);
    }

    /// Announce our capabilities
    pub fn announce_capabilities(&mut self, peer_id: PeerId, capabilities: Vec<String>) {
        let request = PeerDiscoveryRequest::CapabilityAnnouncement { capabilities };
        self.request_response.send_request(&peer_id, request);
        info!("Announced capabilities to peer {}: {:?}", peer_id, request);
    }
}

/// Peer discovery events
#[derive(Debug)]
pub enum Event {
    PeerDiscovered {
        peer_id: PeerId,
        capabilities: Vec<String>,
    },
    CapabilityUpdate {
        peer_id: PeerId,
        capabilities: Vec<String>,
    },
    PeerOffline {
        peer_id: PeerId,
    },
}

/// Peer discovery protocol identifier
#[derive(Debug, Clone)]
pub struct PeerDiscoveryProtocol();

impl ProtocolName for PeerDiscoveryProtocol {
    fn protocol_name(&self) -> &[u8] {
        b"/q-narwhal-knight/peer-discovery/1.0.0"
    }
}

/// Peer discovery request/response messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PeerDiscoveryRequest {
    CapabilityQuery,
    CapabilityAnnouncement { capabilities: Vec<String> },
    QuantumCapabilityProbe { phase: Phase },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PeerDiscoveryResponse {
    Capabilities { capabilities: Vec<String> },
    QuantumSupport { 
        supported_phases: Vec<Phase>,
        crypto_algorithms: Vec<String>,
        qrng_available: bool,
    },
    Error { message: String },
}

/// Codec for peer discovery protocol
#[derive(Debug, Clone)]
pub struct PeerDiscoveryCodec();

impl request_response::Codec for PeerDiscoveryCodec {
    type Protocol = PeerDiscoveryProtocol;
    type Request = PeerDiscoveryRequest;
    type Response = PeerDiscoveryResponse;

    async fn read_request<T>(
        &mut self,
        _: &PeerDiscoveryProtocol,
        io: &mut T,
    ) -> std::io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        let mut reader = io.take(1024 * 1024); // 1MB limit
        reader.read_to_end(&mut buf).await?;
        
        postcard::from_bytes(&buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _: &PeerDiscoveryProtocol,
        io: &mut T,
    ) -> std::io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        let mut reader = io.take(1024 * 1024); // 1MB limit
        reader.read_to_end(&mut buf).await?;
        
        postcard::from_bytes(&buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _: &PeerDiscoveryProtocol,
        io: &mut T,
        req: Self::Request,
    ) -> std::io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        let data = postcard::to_allocvec(&req)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        io.write_all(&data).await?;
        io.close().await
    }

    async fn write_response<T>(
        &mut self,
        _: &PeerDiscoveryProtocol,
        io: &mut T,
        res: Self::Response,
    ) -> std::io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        let data = postcard::to_allocvec(&res)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        io.write_all(&data).await?;
        io.close().await
    }
}

/// Peer information and capabilities
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub peer_id: PeerId,
    pub capabilities: Vec<String>,
    pub supported_phases: Vec<Phase>,
    pub crypto_algorithms: Vec<String>,
    pub agent_version: Option<String>,
    pub protocol_version: Option<String>,
    pub supported_protocols: Vec<String>,
    pub qrng_available: bool,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}

impl PeerInfo {
    pub fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            capabilities: Vec::new(),
            supported_phases: vec![Phase::Phase0], // Default to Phase 0
            crypto_algorithms: vec!["ed25519".to_string()], // Default classical
            agent_version: None,
            protocol_version: None,
            supported_protocols: Vec::new(),
            qrng_available: false,
            last_seen: chrono::Utc::now(),
        }
    }

    /// Check if peer supports a specific phase
    pub fn supports_phase(&self, phase: Phase) -> bool {
        self.supported_phases.contains(&phase)
    }

    /// Check if peer supports post-quantum cryptography
    pub fn supports_post_quantum(&self) -> bool {
        self.supported_phases.iter().any(|p| *p >= Phase::Phase1)
    }

    /// Check if peer has quantum capabilities
    pub fn has_quantum_capabilities(&self) -> bool {
        self.qrng_available || self.supported_phases.iter().any(|p| *p >= Phase::Phase2)
    }

    /// Get the highest supported phase
    pub fn max_supported_phase(&self) -> Phase {
        self.supported_phases.iter().max().copied().unwrap_or(Phase::Phase0)
    }

    /// Update capabilities from discovery response
    pub fn update_capabilities(&mut self, capabilities: Vec<String>) {
        self.capabilities = capabilities;
        self.last_seen = chrono::Utc::now();
        
        // Parse quantum capabilities
        for cap in &self.capabilities {
            match cap.as_str() {
                "phase0" => {
                    if !self.supported_phases.contains(&Phase::Phase0) {
                        self.supported_phases.push(Phase::Phase0);
                    }
                }
                "phase1" | "post-quantum" => {
                    if !self.supported_phases.contains(&Phase::Phase1) {
                        self.supported_phases.push(Phase::Phase1);
                    }
                }
                "phase2" | "quantum-random" => {
                    if !self.supported_phases.contains(&Phase::Phase2) {
                        self.supported_phases.push(Phase::Phase2);
                    }
                }
                "qrng" | "quantum-rng" => {
                    self.qrng_available = true;
                }
                "dilithium5" | "kyber1024" | "falcon1024" => {
                    if !self.crypto_algorithms.contains(&cap) {
                        self.crypto_algorithms.push(cap.clone());
                    }
                }
                _ => {}
            }
        }
        
        self.supported_phases.sort();
        self.supported_phases.dedup();
    }
}

/// Peer discovery manager
pub struct PeerDiscovery {
    known_peers: RwLock<HashMap<PeerId, PeerInfo>>,
    bootstrap_peers: Vec<String>,
    our_capabilities: Vec<String>,
    current_phase: Phase,
}

impl PeerDiscovery {
    pub fn new() -> Self {
        Self {
            known_peers: RwLock::new(HashMap::new()),
            bootstrap_peers: Vec::new(),
            our_capabilities: vec![
                "phase0".to_string(),
                "ed25519".to_string(),
                "x25519".to_string(),
                "gossipsub".to_string(),
                "narwhal-vertices".to_string(),
                "dag-knight".to_string(),
            ],
            current_phase: Phase::Phase0,
        }
    }

    /// Add bootstrap peer
    pub fn add_bootstrap_peer(&mut self, multiaddr: String) {
        self.bootstrap_peers.push(multiaddr);
        info!("Added bootstrap peer: {}", multiaddr);
    }

    /// Update our capabilities for current phase
    pub fn update_phase_capabilities(&mut self, phase: Phase) {
        self.current_phase = phase;
        
        match phase {
            Phase::Phase0 => {
                self.our_capabilities = vec![
                    "phase0".to_string(),
                    "ed25519".to_string(),
                    "x25519".to_string(),
                    "gossipsub".to_string(),
                    "narwhal-vertices".to_string(),
                    "dag-knight".to_string(),
                ];
            }
            Phase::Phase1 => {
                self.our_capabilities = vec![
                    "phase0".to_string(),
                    "phase1".to_string(),
                    "post-quantum".to_string(),
                    "dilithium5".to_string(),
                    "kyber1024".to_string(),
                    "crypto-agile".to_string(),
                    "gossipsub".to_string(),
                    "narwhal-vertices".to_string(),
                    "dag-knight".to_string(),
                ];
            }
            Phase::Phase2 => {
                self.our_capabilities.extend(vec![
                    "phase2".to_string(),
                    "quantum-random".to_string(),
                    "qrng".to_string(),
                    "l-vrf".to_string(),
                ]);
            }
            _ => {}
        }
        
        info!("Updated capabilities for {:?}: {:?}", phase, self.our_capabilities);
    }

    /// Get our current capabilities
    pub fn get_capabilities(&self) -> Vec<String> {
        self.our_capabilities.clone()
    }

    /// Handle incoming discovery request
    pub async fn handle_request(&self, request: PeerDiscoveryRequest) -> PeerDiscoveryResponse {
        match request {
            PeerDiscoveryRequest::CapabilityQuery => {
                PeerDiscoveryResponse::Capabilities {
                    capabilities: self.get_capabilities(),
                }
            }
            PeerDiscoveryRequest::CapabilityAnnouncement { capabilities } => {
                debug!("Received capability announcement: {:?}", capabilities);
                PeerDiscoveryResponse::Capabilities {
                    capabilities: self.get_capabilities(),
                }
            }
            PeerDiscoveryRequest::QuantumCapabilityProbe { phase } => {
                let supported_phases = match self.current_phase {
                    Phase::Phase0 => vec![Phase::Phase0],
                    Phase::Phase1 => vec![Phase::Phase0, Phase::Phase1],
                    Phase::Phase2 => vec![Phase::Phase0, Phase::Phase1, Phase::Phase2],
                    _ => vec![self.current_phase],
                };

                PeerDiscoveryResponse::QuantumSupport {
                    supported_phases,
                    crypto_algorithms: self.our_capabilities.clone(),
                    qrng_available: self.current_phase >= Phase::Phase2,
                }
            }
        }
    }

    /// Add discovered peer
    pub async fn add_peer(&self, peer_id: PeerId, capabilities: Vec<String>) {
        let mut peers = self.known_peers.write().await;
        
        if let Some(peer_info) = peers.get_mut(&peer_id) {
            peer_info.update_capabilities(capabilities);
        } else {
            let mut peer_info = PeerInfo::new(peer_id);
            peer_info.update_capabilities(capabilities);
            peers.insert(peer_id, peer_info);
            info!("Added new peer {}", peer_id);
        }
    }

    /// Get peer information
    pub async fn get_peer(&self, peer_id: &PeerId) -> Option<PeerInfo> {
        let peers = self.known_peers.read().await;
        peers.get(peer_id).cloned()
    }

    /// Get all peers
    pub async fn get_all_peers(&self) -> Vec<PeerInfo> {
        let peers = self.known_peers.read().await;
        peers.values().cloned().collect()
    }

    /// Get peers supporting a specific phase
    pub async fn get_peers_by_phase(&self, phase: Phase) -> Vec<PeerInfo> {
        let peers = self.known_peers.read().await;
        peers
            .values()
            .filter(|p| p.supports_phase(phase))
            .cloned()
            .collect()
    }

    /// Get quantum-capable peers
    pub async fn get_quantum_peers(&self) -> Vec<PeerInfo> {
        let peers = self.known_peers.read().await;
        peers
            .values()
            .filter(|p| p.has_quantum_capabilities())
            .cloned()
            .collect()
    }

    /// Remove offline peer
    pub async fn remove_peer(&self, peer_id: &PeerId) {
        let mut peers = self.known_peers.write().await;
        if peers.remove(peer_id).is_some() {
            info!("Removed offline peer {}", peer_id);
        }
    }

    /// Cleanup old peers
    pub async fn cleanup_old_peers(&self, timeout_seconds: u64) {
        let cutoff = chrono::Utc::now() - chrono::Duration::seconds(timeout_seconds as i64);
        let mut peers = self.known_peers.write().await;
        
        let old_peers: Vec<PeerId> = peers
            .iter()
            .filter(|(_, info)| info.last_seen < cutoff)
            .map(|(id, _)| *id)
            .collect();
        
        for peer_id in old_peers {
            peers.remove(&peer_id);
            debug!("Removed stale peer {}", peer_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::PeerId;

    #[tokio::test]
    async fn test_peer_discovery_creation() {
        let discovery = PeerDiscovery::new();
        let capabilities = discovery.get_capabilities();
        
        assert!(capabilities.contains(&"phase0".to_string()));
        assert!(capabilities.contains(&"ed25519".to_string()));
    }

    #[tokio::test]
    async fn test_phase_upgrade() {
        let mut discovery = PeerDiscovery::new();
        
        // Initial capabilities
        let caps = discovery.get_capabilities();
        assert!(!caps.contains(&"phase1".to_string()));
        
        // Upgrade to Phase 1
        discovery.update_phase_capabilities(Phase::Phase1);
        let caps = discovery.get_capabilities();
        
        assert!(caps.contains(&"phase1".to_string()));
        assert!(caps.contains(&"post-quantum".to_string()));
        assert!(caps.contains(&"dilithium5".to_string()));
    }

    #[tokio::test]
    async fn test_peer_info_capabilities() {
        let peer_id = PeerId::random();
        let mut peer_info = PeerInfo::new(peer_id);
        
        assert_eq!(peer_info.max_supported_phase(), Phase::Phase0);
        assert!(!peer_info.supports_post_quantum());
        
        // Update with post-quantum capabilities
        peer_info.update_capabilities(vec![
            "phase1".to_string(),
            "dilithium5".to_string(),
        ]);
        
        assert!(peer_info.supports_phase(Phase::Phase1));
        assert!(peer_info.supports_post_quantum());
        assert_eq!(peer_info.max_supported_phase(), Phase::Phase1);
    }

    #[tokio::test]
    async fn test_peer_filtering() {
        let discovery = PeerDiscovery::new();
        
        // Add peers with different capabilities
        let peer1 = PeerId::random();
        let peer2 = PeerId::random();
        
        discovery.add_peer(peer1, vec!["phase0".to_string()]).await;
        discovery.add_peer(peer2, vec!["phase1".to_string(), "post-quantum".to_string()]).await;
        
        let phase1_peers = discovery.get_peers_by_phase(Phase::Phase1).await;
        assert_eq!(phase1_peers.len(), 1);
        assert_eq!(phase1_peers[0].peer_id, peer2);
    }
}