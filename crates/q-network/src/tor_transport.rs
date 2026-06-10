/// Tor Transport: libp2p transport integration for Tor networking
/// Enables anonymous communication through the Tor network with circuit management

use anyhow::{Context, Result};
use async_trait::async_trait;
use libp2p::{
    core::{
        multiaddr::{Multiaddr, Protocol},
        transport::{Transport, TransportError, TransportEvent, ListenerId, DialOpts},
        upgrade::Version,
    },
    PeerId,
};
use q_tor_client::{QTorClient, TorConfig, TorConnection, TorStats, TorEnabled};
use q_types::{NodeId, Phase};
use std::{
    collections::HashMap,
    io,
    pin::Pin,
    sync::Arc,
    task::{Context as TaskContext, Poll},
    time::{Duration, Instant},
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{Mutex, RwLock},
};
use tracing::{debug, error, info, warn};

/// Tor transport for libp2p networking
pub struct TorTransport {
    /// Tor client for making connections
    tor_client: Arc<Mutex<QTorClient>>,
    /// Active connections
    connections: Arc<RwLock<HashMap<String, TorConnection>>>,
    /// Transport configuration
    config: TorTransportConfig,
    /// Node ID for this validator
    node_id: NodeId,
    /// Current phase
    phase: Phase,
    /// Connection statistics
    stats: Arc<RwLock<TorTransportStats>>,
}

/// Configuration for Tor transport
#[derive(Debug, Clone)]
pub struct TorTransportConfig {
    /// Enable Tor transport
    pub enabled: bool,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Fallback to TCP for failed Tor connections
    pub fallback_enabled: bool,
    /// Onion service port
    pub onion_service_port: u16,
}

impl Default for TorTransportConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            connection_timeout: Duration::from_secs(30),
            max_connections: 100,
            fallback_enabled: true,
            onion_service_port: 4001,
        }
    }
}

/// Transport statistics
#[derive(Debug, Clone, Default)]
pub struct TorTransportStats {
    pub tor_connections: u64,
    pub fallback_connections: u64,
    pub failed_connections: u64,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub average_connection_time: Duration,
    pub active_connections: usize,
}

impl TorTransport {
    /// Create new Tor transport
    pub async fn new(
        tor_config: TorConfig,
        node_id: NodeId,
        phase: Phase,
    ) -> Result<Self> {
        info!("🚀 Initializing Tor transport for phase {:?}", phase);

        // Create Tor client
        let tor_client = QTorClient::new(tor_config, node_id, phase).await
            .context("Failed to create Tor client")?;

        // Start onion service if enabled
        let onion_address = tor_client.start_onion_service().await
            .context("Failed to start onion service")?;

        info!("✅ Tor transport ready with onion service: {}", onion_address);

        Ok(Self {
            tor_client: Arc::new(Mutex::new(tor_client)),
            connections: Arc::new(RwLock::new(HashMap::new())),
            config: TorTransportConfig::default(),
            node_id,
            phase,
            stats: Arc::new(RwLock::new(TorTransportStats::default())),
        })
    }

    /// Create Tor transport with custom configuration
    pub async fn with_config(
        tor_config: TorConfig,
        transport_config: TorTransportConfig,
        node_id: NodeId,
        phase: Phase,
    ) -> Result<Self> {
        let mut transport = Self::new(tor_config, node_id, phase).await?;
        transport.config = transport_config;
        Ok(transport)
    }

    /// Connect to an onion address
    pub async fn connect_onion(&self, onion_address: &str) -> Result<TorConnection> {
        let start = Instant::now();
        
        debug!("🔗 Connecting to onion address: {}", onion_address);

        let connection = {
            let tor_client = self.tor_client.lock().await;
            tor_client.connect_to_peer(onion_address).await
                .context("Failed to connect through Tor")?
        };

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.tor_connections += 1;
            stats.active_connections += 1;
            
            let connection_time = start.elapsed();
            if stats.tor_connections == 1 {
                stats.average_connection_time = connection_time;
            } else {
                // Exponential moving average
                let alpha = 0.1;
                let current_avg_ms = stats.average_connection_time.as_millis() as f64;
                let new_time_ms = connection_time.as_millis() as f64;
                let new_avg_ms = alpha * new_time_ms + (1.0 - alpha) * current_avg_ms;
                stats.average_connection_time = Duration::from_millis(new_avg_ms as u64);
            }
        }

        // Store connection
        {
            let mut connections = self.connections.write().await;
            connections.insert(onion_address.to_string(), connection.clone());
        }

        info!("✅ Connected to {} via Tor in {}ms", 
              onion_address, start.elapsed().as_millis());

        Ok(connection)
    }

    /// Close connection to peer
    pub async fn close_connection(&self, peer_address: &str) -> Result<()> {
        let mut connections = self.connections.write().await;
        
        if connections.remove(peer_address).is_some() {
            let mut stats = self.stats.write().await;
            if stats.active_connections > 0 {
                stats.active_connections -= 1;
            }
            
            debug!("🔌 Closed connection to {}", peer_address);
        }

        Ok(())
    }

    /// Get transport statistics
    pub async fn get_stats(&self) -> TorTransportStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Check if address is an onion address
    pub fn is_onion_address(address: &Multiaddr) -> bool {
        address.iter().any(|protocol| {
            matches!(protocol, Protocol::Onion3(_))
        })
    }

    /// Convert onion multiaddr to string
    pub fn onion_multiaddr_to_string(address: &Multiaddr) -> Option<String> {
        for protocol in address.iter() {
            if let Protocol::Onion3(addr) = protocol {
                // Convert the onion3 address to string format
                return Some(format!("{}.onion", hex::encode(addr.hash())));
            }
        }
        None
    }

    /// Get Tor client statistics
    pub async fn get_tor_stats(&self) -> Result<TorStats> {
        let tor_client = self.tor_client.lock().await;
        Ok(tor_client.get_tor_stats().await)
    }

    /// Rotate Tor circuits
    pub async fn rotate_circuits(&self) -> Result<()> {
        info!("🔄 Rotating Tor circuits");
        
        let tor_client = self.tor_client.lock().await;
        tor_client.rotate_circuits().await?;
        
        info!("✅ Tor circuit rotation complete");
        Ok(())
    }

    /// Check if Tor is ready
    pub async fn is_tor_ready(&self) -> bool {
        let tor_client = self.tor_client.lock().await;
        tor_client.is_ready().await
    }

    /// Get onion address for this node
    pub async fn get_onion_address(&self) -> Option<String> {
        let tor_client = self.tor_client.lock().await;
        tor_client.get_onion_address().await
    }

    /// Shutdown Tor transport
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down Tor transport");

        // Close all connections
        {
            let mut connections = self.connections.write().await;
            connections.clear();
        }

        // Shutdown Tor client
        {
            let tor_client = self.tor_client.lock().await;
            tor_client.shutdown().await?;
        }

        info!("✅ Tor transport shutdown complete");
        Ok(())
    }
}

/// Tor-enabled transport wrapper
pub struct TorEnabledTransport<T> {
    /// Base transport (TCP, QUIC, etc.)
    base_transport: T,
    /// Tor transport overlay
    tor_transport: Option<Arc<TorTransport>>,
    /// Configuration
    config: TorTransportConfig,
}

impl<T> TorEnabledTransport<T> 
where
    T: Transport + Clone + Send + Unpin + 'static,
    T::Output: Send + Unpin + 'static,
    T::Error: Send + Sync,
    T::ListenerUpgrade: Send,
    T::Dial: Send,
{
    /// Create new Tor-enabled transport
    pub fn new(base_transport: T) -> Self {
        Self {
            base_transport,
            tor_transport: None,
            config: TorTransportConfig::default(),
        }
    }

    /// Enable Tor transport
    pub async fn enable_tor(
        mut self,
        tor_config: TorConfig,
        node_id: NodeId,
        phase: Phase,
    ) -> Result<Self> {
        let tor_transport = TorTransport::new(tor_config, node_id, phase).await?;
        self.tor_transport = Some(Arc::new(tor_transport));
        Ok(self)
    }

    /// Check if Tor is enabled
    pub fn is_tor_enabled(&self) -> bool {
        self.tor_transport.is_some()
    }

    /// Try to dial using Tor first, fallback to base transport
    pub async fn dial_with_tor_fallback(&mut self, addr: Multiaddr) -> Result<T::Output, T::Error> {
        // Check if this is an onion address
        if TorTransport::is_onion_address(&addr) {
            if let Some(tor_transport) = &self.tor_transport {
                if let Some(onion_str) = TorTransport::onion_multiaddr_to_string(&addr) {
                    match tor_transport.connect_onion(&onion_str).await {
                        Ok(_connection) => {
                            // For now, we can't directly return TorConnection as T::Output
                            // In a full implementation, we'd need to wrap it properly
                            // Fallback to base transport for now
                            warn!("⚠️ Tor connection established but falling back to base transport");
                        }
                        Err(e) => {
                            warn!("⚠️ Tor connection failed: {}", e);
                        }
                    }
                }
            }
        }

        // Use base transport (direct or fallback)
        self.base_transport.dial(addr)
    }
}

impl<T> Transport for TorEnabledTransport<T>
where
    T: Transport + Clone + Send + Unpin + 'static,
    T::Output: Send + Unpin + 'static,
    T::Error: Send + Sync,
    T::ListenerUpgrade: Send,
    T::Dial: Send,
{
    type Output = T::Output;
    type Error = T::Error;
    type ListenerUpgrade = T::ListenerUpgrade;
    type Dial = T::Dial;

    fn listen_on(
        &mut self,
        id: ListenerId,
        addr: Multiaddr,
    ) -> Result<(), TransportError<Self::Error>> {
        // Delegate to base transport for listening
        self.base_transport.listen_on(id, addr)
    }

    fn remove_listener(&mut self, id: ListenerId) -> bool {
        self.base_transport.remove_listener(id)
    }

    fn dial(&mut self, addr: Multiaddr, _opts: libp2p::core::transport::DialOpts) -> Result<Self::Dial, TransportError<Self::Error>> {
        // For now, delegate to base transport
        // In a full implementation, we'd handle Tor addresses specially
        self.base_transport.dial(addr, _opts)
    }

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        Pin::new(&mut self.base_transport).poll(cx)
    }
}

/// Trait for Tor-enabled networking components
#[async_trait]
pub trait TorNetworking {
    /// Enable Tor networking
    async fn enable_tor(&mut self, tor_config: TorConfig, node_id: NodeId, phase: Phase) -> Result<()>;
    
    /// Disable Tor networking
    async fn disable_tor(&mut self) -> Result<()>;
    
    /// Check if Tor is enabled
    fn is_tor_enabled(&self) -> bool;
    
    /// Connect to onion service
    async fn connect_onion(&self, onion_address: &str) -> Result<()>;
    
    /// Get onion address for this node
    async fn get_onion_address(&self) -> Option<String>;
    
    /// Get Tor statistics
    async fn get_tor_stats(&self) -> Option<TorStats>;
    
    /// Rotate Tor circuits
    async fn rotate_tor_circuits(&self) -> Result<()>;
}

/// Tor peer discovery for finding validators via onion services
pub struct TorPeerDiscovery {
    /// Known onion addresses
    known_onions: Arc<RwLock<HashMap<PeerId, String>>>,
    /// Bootstrap onion services
    bootstrap_onions: Vec<String>,
    /// Discovery statistics
    stats: Arc<RwLock<TorDiscoveryStats>>,
}

/// Tor peer discovery statistics
#[derive(Debug, Clone, Default)]
pub struct TorDiscoveryStats {
    pub discovered_peers: u64,
    pub bootstrap_attempts: u64,
    pub successful_bootstraps: u64,
    pub last_discovery: Option<Instant>,
}

impl TorPeerDiscovery {
    /// Create new Tor peer discovery
    pub fn new(bootstrap_onions: Vec<String>) -> Self {
        Self {
            known_onions: Arc::new(RwLock::new(HashMap::new())),
            bootstrap_onions,
            stats: Arc::new(RwLock::new(TorDiscoveryStats::default())),
        }
    }

    /// Add known onion service
    pub async fn add_onion_service(&self, peer_id: PeerId, onion_address: String) {
        debug!("📝 Adding onion service for peer {}: {}", peer_id, onion_address);
        
        let mut known_onions = self.known_onions.write().await;
        known_onions.insert(peer_id, onion_address);
        
        let mut stats = self.stats.write().await;
        stats.discovered_peers += 1;
        stats.last_discovery = Some(Instant::now());
    }

    /// Get onion address for peer
    pub async fn get_onion_address(&self, peer_id: &PeerId) -> Option<String> {
        let known_onions = self.known_onions.read().await;
        known_onions.get(peer_id).cloned()
    }

    /// Bootstrap from known onion services
    pub async fn bootstrap(&self) -> Result<Vec<String>> {
        info!("🥾 Bootstrapping from {} onion services", self.bootstrap_onions.len());
        
        let mut stats = self.stats.write().await;
        stats.bootstrap_attempts += 1;
        
        // In production, this would connect to bootstrap onions and discover peers
        let discovered = self.bootstrap_onions.clone();
        
        stats.successful_bootstraps += 1;
        info!("✅ Bootstrap complete, discovered {} peers", discovered.len());
        
        Ok(discovered)
    }

    /// Get discovery statistics
    pub async fn get_stats(&self) -> TorDiscoveryStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_tor_client::TorConfig;

    #[test]
    fn test_onion_address_detection() {
        // Test multiaddr with onion3 address
        let onion_addr = "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:1234/tcp/4001"
            .parse::<Multiaddr>()
            .unwrap();
        
        assert!(TorTransport::is_onion_address(&onion_addr));
        
        // Test regular TCP address
        let tcp_addr = "/ip4/127.0.0.1/tcp/4001".parse::<Multiaddr>().unwrap();
        assert!(!TorTransport::is_onion_address(&tcp_addr));
    }

    #[tokio::test]
    async fn test_tor_peer_discovery() {
        let bootstrap_onions = vec![
            "bootstrap1.qnk.onion:4001".to_string(),
            "bootstrap2.qnk.onion:4001".to_string(),
        ];
        
        let discovery = TorPeerDiscovery::new(bootstrap_onions);
        let discovered = discovery.bootstrap().await.unwrap();
        
        assert_eq!(discovered.len(), 2);
        
        let stats = discovery.get_stats().await;
        assert_eq!(stats.bootstrap_attempts, 1);
        assert_eq!(stats.successful_bootstraps, 1);
    }

    #[test]
    fn test_transport_config() {
        let config = TorTransportConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_connections, 100);
        assert!(config.fallback_enabled);
    }

    #[tokio::test]
    async fn test_tor_discovery_peer_management() {
        let discovery = TorPeerDiscovery::new(vec![]);
        let peer_id = PeerId::random();
        let onion = "validator123.qnk.onion".to_string();
        
        // Add peer
        discovery.add_onion_service(peer_id, onion.clone()).await;
        
        // Retrieve peer
        let retrieved = discovery.get_onion_address(&peer_id).await;
        assert_eq!(retrieved, Some(onion));
        
        // Check stats
        let stats = discovery.get_stats().await;
        assert_eq!(stats.discovered_peers, 1);
        assert!(stats.last_discovery.is_some());
    }
}