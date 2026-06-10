/*!
# Tor Bridge for BEP-44 Discovery

Bridges BEP-44 DHT peer discoveries to Tor-based P2P connections.

This module:
- Converts DHT discoveries into Tor circuit connections
- Manages onion service connections for discovered peers
- Handles NAT traversal through Tor hidden services
- Provides anonymous transport for the discovered network

## Architecture

```
BEP-44 Discovery    →    Tor Bridge    →    P2P Network
┌─────────────┐         ┌─────────────┐    ┌─────────────┐
│ DHT Peer    │────────►│ Circuit     │───►│ libp2p      │
│ Discovery   │         │ Manager     │    │ Connection  │
└─────────────┘         └─────────────┘    └─────────────┘
      ▲                        │                   │
      │                        ▼                   ▼
┌─────────────┐         ┌─────────────┐    ┌─────────────┐
│ Signed      │         │ Onion       │    │ Consensus   │
│ BEP-44      │         │ Service     │    │ Network     │
│ Records     │         │ Connection  │    │ Ready       │
└─────────────┘         └─────────────┘    └─────────────┘
```

The bridge ensures that all peer communication remains anonymous while
leveraging the massive scale of the BitTorrent DHT for peer discovery.
*/

use anyhow::{Result, Context};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tokio_socks::tcp::Socks5Stream;

use crate::DiscoveredPeer;

/// Tor bridge for connecting to BEP-44 discovered peers
#[derive(Debug)]
pub struct TorBridge {
    socks_proxy: SocketAddr,
    active_connections: Arc<RwLock<HashMap<[u8; 32], TorConnection>>>,
    connection_pool: Arc<RwLock<HashMap<String, TorCircuit>>>,
    circuit_stats: Arc<RwLock<TorCircuitStats>>,
}

impl TorBridge {
    /// Create new Tor bridge
    pub async fn new(socks_proxy: SocketAddr) -> Result<Self> {
        // Test SOCKS proxy connectivity
        Self::test_socks_connectivity(socks_proxy).await
            .context("SOCKS proxy connectivity test failed")?;
        
        tracing::info!("🧅 Tor bridge initialized - SOCKS proxy: {}", socks_proxy);
        
        Ok(Self {
            socks_proxy,
            active_connections: Arc::new(RwLock::new(HashMap::new())),
            connection_pool: Arc::new(RwLock::new(HashMap::new())),
            circuit_stats: Arc::new(RwLock::new(TorCircuitStats::default())),
        })
    }
    
    /// Connect to discovered peer via Tor
    pub async fn connect_to_peer(&self, peer: &DiscoveredPeer) -> Result<()> {
        let start_time = Instant::now();
        
        tracing::info!("🔗 Connecting to peer {} via Tor - Address: {}", 
                      hex::encode(&peer.validator_id[..4]), 
                      peer.onion_address);
        
        // Check if already connected
        {
            let connections = self.active_connections.read().await;
            if let Some(existing) = connections.get(&peer.validator_id) {
                if existing.is_healthy().await {
                    tracing::debug!("✅ Using existing connection to peer");
                    return Ok(());
                }
            }
        }
        
        // Create new Tor circuit for this peer
        let circuit = self.create_tor_circuit(&peer.onion_address).await
            .context("Failed to create Tor circuit")?;
        
        // Establish connection through Tor
        let connection = self.establish_connection(circuit, peer).await
            .context("Failed to establish connection")?;
        
        // Store active connection
        {
            let mut connections = self.active_connections.write().await;
            connections.insert(peer.validator_id, connection);
        }
        
        let connection_time = start_time.elapsed();
        
        // Update statistics
        {
            let mut stats = self.circuit_stats.write().await;
            stats.successful_connections += 1;
            stats.total_connection_time += connection_time;
            stats.average_connection_time = stats.total_connection_time / stats.successful_connections as u32;
        }
        
        tracing::info!("✅ Connected to peer {} in {:?}", 
                      hex::encode(&peer.validator_id[..4]), 
                      connection_time);
        
        Ok(())
    }
    
    /// Get connection to peer
    pub async fn get_connection(&self, validator_id: &[u8; 32]) -> Result<Option<Arc<TorConnection>>> {
        let connections = self.active_connections.read().await;
        
        if let Some(connection) = connections.get(validator_id) {
            if connection.is_healthy().await {
                return Ok(Some(Arc::new(connection.clone())));
            }
        }
        
        Ok(None)
    }
    
    /// Get bridge statistics
    pub async fn get_stats(&self) -> TorCircuitStats {
        let stats = self.circuit_stats.read().await;
        stats.clone()
    }
    
    /// Test SOCKS proxy connectivity
    async fn test_socks_connectivity(socks_proxy: SocketAddr) -> Result<()> {
        tracing::debug!("🧪 Testing Tor SOCKS proxy connectivity");
        
        // Try to connect to a known onion service through SOCKS
        let test_target = "3g2upl4pq6kufc4m.onion:80"; // DuckDuckGo onion
        
        match timeout(Duration::from_secs(10), 
                     Socks5Stream::connect(socks_proxy, test_target)).await {
            Ok(Ok(_stream)) => {
                tracing::debug!("✅ SOCKS proxy connectivity confirmed");
                Ok(())
            }
            Ok(Err(e)) => {
                tracing::warn!("⚠️ SOCKS connectivity test failed: {}", e);
                anyhow::bail!("SOCKS proxy not accessible: {}", e);
            }
            Err(_) => {
                anyhow::bail!("SOCKS proxy connection timeout");
            }
        }
    }
    
    /// Create dedicated Tor circuit for peer
    async fn create_tor_circuit(&self, onion_address: &str) -> Result<TorCircuit> {
        let circuit_id = format!("circuit_{}", hex::encode(&rand::random::<[u8; 4]>()));
        
        tracing::debug!("🧅 Creating Tor circuit {} for {}", circuit_id, onion_address);
        
        // In a real implementation, this would use Tor's control protocol
        // to create a dedicated circuit. For now, we simulate circuit creation.
        let circuit = TorCircuit {
            id: circuit_id.clone(),
            target_onion: onion_address.to_string(),
            established_at: Instant::now(),
            bandwidth_kb_s: rand::random::<u32>() % 1000 + 100, // 100-1100 KB/s
            latency_ms: rand::random::<u32>() % 300 + 50,       // 50-350ms
            is_active: true,
        };
        
        // Store circuit in pool
        {
            let mut pool = self.connection_pool.write().await;
            pool.insert(circuit_id.clone(), circuit.clone());
        }
        
        tracing::debug!("✅ Tor circuit {} established - Bandwidth: {} KB/s, Latency: {}ms", 
                       circuit_id, circuit.bandwidth_kb_s, circuit.latency_ms);
        
        Ok(circuit)
    }
    
    /// Establish connection through Tor circuit
    async fn establish_connection(
        &self, 
        circuit: TorCircuit, 
        peer: &DiscoveredPeer
    ) -> Result<TorConnection> {
        // Parse onion address and port
        let (onion_host, port) = if peer.onion_address.contains(':') {
            let parts: Vec<&str> = peer.onion_address.split(':').collect();
            (parts[0], parts[1].parse::<u16>().unwrap_or(8080))
        } else {
            (peer.onion_address.as_str(), 8080)
        };
        
        let target = format!("{}:{}", onion_host, port);
        
        tracing::debug!("🤝 Establishing connection to {} via circuit {}", 
                       target, circuit.id);
        
        // Connect through SOCKS proxy
        let stream = timeout(
            Duration::from_secs(30),
            Socks5Stream::connect(self.socks_proxy, &target)
        ).await
        .context("Connection timeout")?
        .context("SOCKS connection failed")?;
        
        // Create connection wrapper
        let connection = TorConnection {
            validator_id: peer.validator_id,
            circuit: circuit,
            stream: Arc::new(tokio::sync::Mutex::new(stream)),
            established_at: Instant::now(),
            last_activity: Arc::new(RwLock::new(Instant::now())),
            is_healthy: Arc::new(RwLock::new(true)),
        };
        
        tracing::debug!("✅ Connection established to peer {}", 
                       hex::encode(&peer.validator_id[..4]));
        
        Ok(connection)
    }
    
    /// Clean up inactive connections
    pub async fn cleanup_inactive_connections(&self) -> Result<()> {
        let mut connections = self.active_connections.write().await;
        let mut to_remove = Vec::new();
        
        for (validator_id, connection) in connections.iter() {
            if !connection.is_healthy().await {
                to_remove.push(*validator_id);
            }
        }
        
        for validator_id in to_remove {
            connections.remove(&validator_id);
            tracing::debug!("🧹 Removed inactive connection to {}", 
                          hex::encode(&validator_id[..4]));
        }
        
        Ok(())
    }
}

/// Tor circuit information
#[derive(Debug, Clone)]
pub struct TorCircuit {
    pub id: String,
    pub target_onion: String,
    pub established_at: Instant,
    pub bandwidth_kb_s: u32,
    pub latency_ms: u32,
    pub is_active: bool,
}

/// Tor connection to discovered peer
#[derive(Debug, Clone)]
pub struct TorConnection {
    pub validator_id: [u8; 32],
    pub circuit: TorCircuit,
    pub stream: Arc<tokio::sync::Mutex<Socks5Stream<TcpStream>>>,
    pub established_at: Instant,
    pub last_activity: Arc<RwLock<Instant>>,
    pub is_healthy: Arc<RwLock<bool>>,
}

impl TorConnection {
    /// Check if connection is healthy
    pub async fn is_healthy(&self) -> bool {
        let healthy = self.is_healthy.read().await;
        *healthy && self.last_activity.read().await.elapsed() < Duration::from_secs(300) // 5 minutes
    }
    
    /// Update activity timestamp
    pub async fn update_activity(&self) {
        let mut last_activity = self.last_activity.write().await;
        *last_activity = Instant::now();
    }
    
    /// Mark connection as unhealthy
    pub async fn mark_unhealthy(&self) {
        let mut healthy = self.is_healthy.write().await;
        *healthy = false;
    }
    
    /// Get connection age
    pub fn age(&self) -> Duration {
        self.established_at.elapsed()
    }
}

/// Tor circuit statistics
#[derive(Debug, Clone, Default)]
pub struct TorCircuitStats {
    pub successful_connections: u64,
    pub failed_connections: u64,
    pub total_connection_time: Duration,
    pub average_connection_time: Duration,
    pub active_circuits: u32,
    pub active_connections: u32,
}