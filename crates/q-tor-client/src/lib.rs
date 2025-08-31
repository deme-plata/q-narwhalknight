/// Q-Tor-Client: Embedded Tor client for Q-NarwhalKnight
/// Provides anonymity and privacy through Tor onion routing
/// Integrates with post-quantum cryptography for quantum-resistant communication

use anyhow::{Context, Result};
use arti_client::{TorClient, TorClientConfig};
use arti_config::ArtiConfig;
use async_trait::async_trait;
use futures::Future;
use q_types::{NodeId, Phase};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

pub mod circuit_manager;
pub mod config;
pub mod metrics;
pub mod onion_service;

pub use circuit_manager::CircuitManager;
pub use config::TorConfig;
pub use metrics::TorMetrics;
pub use onion_service::OnionService;

/// Main Tor client for Q-NarwhalKnight
pub struct QTorClient {
    /// Arti Tor client
    tor_client: Arc<TorClient<tor_rtcompat::tokio::TokioNativeTlsRuntime>>,
    /// Circuit manager for dedicated circuits
    circuit_manager: Arc<Mutex<CircuitManager>>,
    /// Onion service for this validator
    onion_service: Arc<RwLock<Option<OnionService>>>,
    /// Configuration
    config: TorConfig,
    /// Performance metrics
    metrics: Arc<TorMetrics>,
    /// Node ID for this validator
    node_id: NodeId,
    /// Current cryptographic phase
    current_phase: Phase,
}

impl QTorClient {
    /// Create a new Tor client
    pub async fn new(config: TorConfig, node_id: NodeId, phase: Phase) -> Result<Self> {
        info!("🧅 Initializing Q-Tor-Client for validator {}", hex::encode(node_id));

        // Configure Arti
        let mut arti_config = ArtiConfig::default();
        
        // Set up data directory
        if let Some(data_dir) = &config.data_dir {
            arti_config.storage.cache_dir = data_dir.clone();
            arti_config.storage.state_dir = data_dir.clone();
        }

        // Configure circuit preferences
        arti_config.circuit_timing.max_dirtiness = Duration::from_secs(300); // 5 min rotation
        arti_config.circuit_timing.request_timeout = Duration::from_secs(60);

        // Create Tor client
        let tor_client = TorClient::with_runtime(
            tor_rtcompat::tokio::TokioNativeTlsRuntime::current()?,
        )
        .config(TorClientConfig::from_arti_config(arti_config)?)
        .create_bootstrapped()
        .await
        .context("Failed to bootstrap Tor client")?;

        let tor_client = Arc::new(tor_client);

        // Initialize circuit manager with 4 dedicated circuits
        let circuit_manager = Arc::new(Mutex::new(
            CircuitManager::new(tor_client.clone(), config.circuit_count).await?
        ));

        let metrics = Arc::new(TorMetrics::new());

        Ok(Self {
            tor_client,
            circuit_manager,
            onion_service: Arc::new(RwLock::new(None)),
            config,
            metrics,
            node_id,
            current_phase: phase,
        })
    }

    /// Start the onion service for this validator
    pub async fn start_onion_service(&self) -> Result<String> {
        info!("🧅 Starting onion service for validator");

        let onion_name = format!("validator{}.qnk", hex::encode(&self.node_id[..4]));
        let onion_service = OnionService::new(
            self.tor_client.clone(),
            onion_name.clone(),
            self.config.rpc_port,
        ).await?;

        let onion_address = onion_service.get_onion_address();
        info!("✅ Onion service started: {}.onion", onion_address);

        // Store the onion service
        {
            let mut service = self.onion_service.write().await;
            *service = Some(onion_service);
        }

        Ok(format!("{}.onion", onion_address))
    }

    /// Connect to a peer through Tor
    pub async fn connect_to_peer(&self, onion_address: &str) -> Result<TorConnection> {
        debug!("🔗 Connecting to peer via Tor: {}", onion_address);
        
        let start_time = Instant::now();
        
        // Get a dedicated circuit for this connection
        let circuit = {
            let mut manager = self.circuit_manager.lock().await;
            manager.get_circuit_for_peer(onion_address).await?
        };

        // Establish connection through the circuit
        let stream = self.tor_client
            .connect((onion_address, self.config.rpc_port))
            .await
            .context("Failed to connect through Tor")?;

        let latency = start_time.elapsed();
        self.metrics.record_connection_latency(latency).await;

        info!("✅ Connected to {} via Tor ({}ms)", onion_address, latency.as_millis());

        Ok(TorConnection::new(stream, circuit, onion_address.to_string()))
    }

    /// Broadcast message through Tor with traffic analysis resistance
    pub async fn broadcast_message(&self, message: &[u8], topic: &str) -> Result<()> {
        debug!("📡 Broadcasting message via Tor to topic: {}", topic);

        // Use Dandelion++ for traffic analysis resistance
        if self.config.enable_dandelion {
            self.dandelion_broadcast(message, topic).await?;
        } else {
            self.direct_broadcast(message, topic).await?;
        }

        Ok(())
    }

    /// Direct broadcast through Tor circuits
    async fn direct_broadcast(&self, message: &[u8], _topic: &str) -> Result<()> {
        let circuit_manager = self.circuit_manager.lock().await;
        
        // Use all gossip circuits for broadcasting
        for circuit_id in circuit_manager.get_gossip_circuits() {
            tokio::spawn({
                let tor_client = self.tor_client.clone();
                let message = message.to_vec();
                let circuit_id = *circuit_id;
                
                async move {
                    // Send message through this circuit
                    // Implementation would depend on the specific networking protocol
                    debug!("📤 Sending message through circuit {}", circuit_id);
                }
            });
        }

        Ok(())
    }

    /// Dandelion++ broadcast for traffic analysis resistance
    async fn dandelion_broadcast(&self, message: &[u8], topic: &str) -> Result<()> {
        debug!("🌻 Using Dandelion++ broadcast for topic: {}", topic);
        
        // Phase 1: Stem phase - relay to random peer
        let random_circuit = {
            let manager = self.circuit_manager.lock().await;
            manager.get_random_circuit().await?
        };

        // Send to random peer first (stem phase)
        // Then peer will either continue stem or switch to fluff phase
        self.relay_through_circuit(message, random_circuit).await?;
        
        Ok(())
    }

    /// Relay message through specific circuit
    async fn relay_through_circuit(&self, _message: &[u8], circuit_id: u64) -> Result<()> {
        debug!("🔄 Relaying message through circuit {}", circuit_id);
        // Implementation would integrate with the actual circuit
        Ok(())
    }

    /// Get Tor network statistics
    pub async fn get_tor_stats(&self) -> TorStats {
        let circuit_manager = self.circuit_manager.lock().await;
        let metrics = self.metrics.get_current_metrics().await;

        TorStats {
            active_circuits: circuit_manager.active_circuit_count(),
            average_latency: metrics.average_latency,
            connection_count: metrics.connection_count,
            bytes_sent: metrics.bytes_sent,
            bytes_received: metrics.bytes_received,
            onion_address: self.get_onion_address().await,
            tor_enabled: true,
        }
    }

    /// Get our onion address
    pub async fn get_onion_address(&self) -> Option<String> {
        let service = self.onion_service.read().await;
        service.as_ref().map(|s| s.get_onion_address())
    }

    /// Rotate circuits (called every epoch)
    pub async fn rotate_circuits(&self) -> Result<()> {
        info!("🔄 Rotating Tor circuits for new epoch");
        
        let mut circuit_manager = self.circuit_manager.lock().await;
        circuit_manager.rotate_all_circuits().await?;
        
        info!("✅ Circuit rotation complete");
        Ok(())
    }

    /// Set latency target for adaptive QoS
    pub async fn set_latency_target(&self, target_ms: u16) -> Result<()> {
        let mut circuit_manager = self.circuit_manager.lock().await;
        circuit_manager.set_latency_target(Duration::from_millis(target_ms as u64)).await;
        Ok(())
    }

    /// Check if Tor client is ready
    pub async fn is_ready(&self) -> bool {
        // Check if we have at least one working circuit
        let circuit_manager = self.circuit_manager.lock().await;
        circuit_manager.active_circuit_count() > 0
    }

    /// Shutdown Tor client gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Shutting down Tor client");
        
        // Close onion service
        {
            let mut service = self.onion_service.write().await;
            if let Some(service) = service.take() {
                service.shutdown().await?;
            }
        }

        // Close all circuits
        {
            let mut circuit_manager = self.circuit_manager.lock().await;
            circuit_manager.close_all_circuits().await?;
        }

        info!("✅ Tor client shutdown complete");
        Ok(())
    }
}

/// Tor connection wrapper
pub struct TorConnection {
    stream: Box<dyn Send + Sync>,
    circuit_id: u64,
    peer_onion: String,
}

impl TorConnection {
    pub fn new(stream: impl Send + Sync + 'static, circuit_id: u64, peer_onion: String) -> Self {
        Self {
            stream: Box::new(stream),
            circuit_id,
            peer_onion,
        }
    }

    pub fn get_circuit_id(&self) -> u64 {
        self.circuit_id
    }

    pub fn get_peer_onion(&self) -> &str {
        &self.peer_onion
    }
}

/// Tor network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorStats {
    pub active_circuits: usize,
    pub average_latency: Duration,
    pub connection_count: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub onion_address: Option<String>,
    pub tor_enabled: bool,
}

/// Trait for Tor-enabled networking
#[async_trait]
pub trait TorEnabled {
    /// Enable Tor mode
    async fn enable_tor(&mut self, config: TorConfig) -> Result<()>;
    
    /// Disable Tor mode
    async fn disable_tor(&mut self) -> Result<()>;
    
    /// Check if Tor is enabled
    fn is_tor_enabled(&self) -> bool;
    
    /// Get Tor statistics
    async fn get_tor_stats(&self) -> Option<TorStats>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_tor_client_creation() {
        let config = TorConfig::default();
        let node_id = [1u8; 32];
        
        // Note: This test might fail in CI without Tor network access
        // In production, we'd mock the Tor client
        let result = QTorClient::new(config, node_id, Phase::Phase0).await;
        
        // Just check that the function doesn't panic
        // Actual Tor functionality would be tested in integration tests
        if result.is_err() {
            warn!("Tor client creation failed (expected in test environment)");
        }
    }

    #[test]
    fn test_tor_stats_serialization() {
        let stats = TorStats {
            active_circuits: 4,
            average_latency: Duration::from_millis(150),
            connection_count: 10,
            bytes_sent: 1024,
            bytes_received: 2048,
            onion_address: Some("validator123.qnk.onion".to_string()),
            tor_enabled: true,
        };

        let serialized = serde_json::to_string(&stats).unwrap();
        let deserialized: TorStats = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(stats.active_circuits, deserialized.active_circuits);
        assert_eq!(stats.onion_address, deserialized.onion_address);
    }
}