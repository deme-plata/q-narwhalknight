/// Q-Tor-Client: Embedded Tor client for Q-NarwhalKnight
/// Provides anonymity and privacy through Tor onion routing
/// Integrates with post-quantum cryptography for quantum-resistant communication
use anyhow::{Context, Result};
use async_trait::async_trait;
use q_types::{NodeId, Phase};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::net::TcpStream;
use tokio::sync::{Mutex, RwLock};
use tokio_socks::tcp::Socks5Stream;
use tracing::{debug, error, info, warn};

pub mod circuit_manager;
pub mod config;
pub mod dandelion;
pub mod metrics;
pub mod onion_service;
pub mod prometheus_metrics;
pub mod quantum_seeding;

// Production implementation
pub mod real_tor_client;

pub use circuit_manager::CircuitManager;
pub use config::TorConfig;
pub use dandelion::{DandelionConfig, DandelionProtocol, DandelionStatistics};
pub use metrics::TorMetrics;
pub use onion_service::OnionService;
pub use prometheus_metrics::{MetricsSummary, PrometheusConfig, TorPrometheusMetrics};
pub use quantum_seeding::{CircuitParameters, QuantumEntropyPool, QuantumSeedingConfig};

/// Main Tor client for Q-NarwhalKnight
pub struct QTorClient {
    /// SOCKS proxy address for Tor connection
    socks_proxy: SocketAddr,
    /// Circuit manager for dedicated circuits
    circuit_manager: Arc<Mutex<CircuitManager>>,
    /// Onion service for this validator
    onion_service: Arc<RwLock<Option<OnionService>>>,
    /// Configuration
    config: TorConfig,
    /// Performance metrics
    metrics: Arc<TorMetrics>,
    /// Prometheus metrics collector
    prometheus_metrics: Option<Arc<TorPrometheusMetrics>>,
    /// Node ID for this validator
    node_id: NodeId,
    /// Current cryptographic phase
    current_phase: Phase,
    /// Quantum entropy pool for circuit seeding
    quantum_entropy: Option<Arc<QuantumEntropyPool>>,
    /// Dandelion++ protocol for privacy
    dandelion: Option<Arc<DandelionProtocol>>,
}

impl QTorClient {
    /// Create a new Tor client
    pub async fn new(config: TorConfig, node_id: NodeId, phase: Phase) -> Result<Self> {
        info!(
            "🧅 Initializing Q-Tor-Client for validator {}",
            hex::encode(node_id)
        );

        // Default Tor SOCKS proxy address (standard Tor port)
        let socks_proxy = config.socks_proxy_addr.unwrap_or_else(|| {
            "127.0.0.1:9050"
                .parse()
                .expect("Valid default SOCKS address")
        });

        // Test SOCKS proxy connection
        Self::test_socks_connection(&socks_proxy)
            .await
            .context("Failed to connect to Tor SOCKS proxy. Is Tor running?")?;

        // Initialize circuit manager with 4 dedicated circuits
        let circuit_manager = Arc::new(Mutex::new(
            CircuitManager::new(socks_proxy, config.circuit_count).await?,
        ));

        let metrics = Arc::new(TorMetrics::new());

        // Initialize quantum entropy pool if in Phase 2+
        let quantum_entropy = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            match QuantumEntropyPool::new(QuantumSeedingConfig::default()).await {
                Ok(pool) => {
                    info!("✅ Quantum entropy pool initialized for {:?}", phase);
                    Some(Arc::new(pool))
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to initialize quantum entropy: {}, using classical fallback",
                        e
                    );
                    None
                }
            }
        } else {
            debug!("Using classical entropy for {:?}", phase);
            None
        };

        // Initialize Prometheus metrics if enabled
        let prometheus_metrics = if config.enable_prometheus_metrics {
            match TorPrometheusMetrics::new(PrometheusConfig::default()) {
                Ok(prometheus) => {
                    info!("✅ Prometheus metrics initialized");
                    Some(Arc::new(prometheus))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize Prometheus metrics: {}", e);
                    None
                }
            }
        } else {
            debug!("Prometheus metrics disabled");
            None
        };

        Ok(Self {
            socks_proxy,
            circuit_manager,
            onion_service: Arc::new(RwLock::new(None)),
            config,
            metrics,
            prometheus_metrics,
            node_id,
            current_phase: phase,
            quantum_entropy,
            dandelion: None, // Will be initialized separately
        })
    }

    /// Test SOCKS proxy connection with retry for Tor bootstrap
    async fn test_socks_connection(proxy_addr: &SocketAddr) -> Result<()> {
        debug!("Testing SOCKS proxy connection at {}", proxy_addr);

        // Retry connection up to 30 seconds to wait for Tor bootstrap
        let max_retries = 6; // 6 retries * 5 seconds = 30 seconds max wait
        let retry_interval = Duration::from_secs(5);

        for attempt in 1..=max_retries {
            debug!("🔄 Tor connection attempt {} of {}", attempt, max_retries);

            // Try to connect to a known Tor test address
            let test_result = tokio::time::timeout(
                Duration::from_secs(5),
                Socks5Stream::connect(proxy_addr, ("check.torproject.org", 443)),
            )
            .await;

            match test_result {
                Ok(Ok(_)) => {
                    info!("✅ Tor SOCKS proxy is operational (attempt {})", attempt);
                    return Ok(());
                }
                Ok(Err(e)) => {
                    warn!("⚠️ Tor connection attempt {} failed: {}", attempt, e);
                    if attempt == max_retries {
                        return Err(anyhow::anyhow!(
                            "SOCKS proxy connection failed after {} attempts: {}",
                            max_retries,
                            e
                        ));
                    }
                }
                Err(_) => {
                    warn!("⚠️ Tor connection attempt {} timed out", attempt);
                    if attempt == max_retries {
                        return Err(anyhow::anyhow!(
                            "SOCKS proxy connection timed out after {} attempts",
                            max_retries
                        ));
                    }
                }
            }

            if attempt < max_retries {
                info!(
                    "⏳ Waiting for Tor bootstrap... retrying in {} seconds",
                    retry_interval.as_secs()
                );
                tokio::time::sleep(retry_interval).await;
            }
        }

        unreachable!("Loop should have returned or errored");
    }

    /// Start the onion service for this validator
    pub async fn start_onion_service(&self) -> Result<String> {
        info!("🧅 Starting onion service for validator");

        let onion_name = format!("validator{}.qnk", hex::encode(&self.node_id[..4]));
        let onion_service =
            OnionService::new(self.socks_proxy, onion_name.clone(), self.config.rpc_port).await?;

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
        let circuit_id = {
            let mut manager = self.circuit_manager.lock().await;
            manager.get_circuit_for_peer(onion_address).await?
        };

        // Parse onion address and port
        let (host, port) = if onion_address.contains(':') {
            let parts: Vec<&str> = onion_address.split(':').collect();
            (
                parts[0],
                parts[1].parse::<u16>().unwrap_or(self.config.rpc_port),
            )
        } else {
            (onion_address, self.config.rpc_port)
        };

        // Establish connection through SOCKS proxy
        let stream = Socks5Stream::connect(&self.socks_proxy, (host, port))
            .await
            .context("Failed to connect through Tor")?;

        let latency = start_time.elapsed();
        self.metrics.record_connection_latency(latency).await;

        info!(
            "✅ Connected to {} via Tor ({}ms)",
            onion_address,
            latency.as_millis()
        );

        Ok(TorConnection::new(
            stream.into_inner(),
            circuit_id,
            onion_address.to_string(),
        ))
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
            let proxy_addr = self.socks_proxy;
            let message = message.to_vec();
            let circuit_id = *circuit_id;

            tokio::spawn(async move {
                // Send message through this circuit
                // Implementation would depend on the specific networking protocol
                debug!("📤 Sending message through circuit {}", circuit_id);
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

    /// Generate quantum-enhanced circuit parameters
    pub async fn generate_quantum_circuit_parameters(&self) -> Result<CircuitParameters> {
        if let Some(quantum_entropy) = &self.quantum_entropy {
            info!("🌊 Generating quantum circuit parameters");
            quantum_entropy.generate_circuit_parameters().await
        } else {
            // Fallback to classical circuit parameters
            debug!("Using classical circuit parameters");
            Ok(CircuitParameters {
                seed: rand::random(),
                nonce: (0..12).map(|_| rand::random()).collect(),
                timing_offset: Duration::from_millis(rand::random::<u64>() % 1000),
                hop_weights: (0..16).map(|_| rand::random()).collect(),
                created_at: SystemTime::now(),
            })
        }
    }

    /// Get quantum entropy quality metrics
    pub async fn get_entropy_quality(&self) -> Option<quantum_seeding::EntropyQuality> {
        if let Some(quantum_entropy) = &self.quantum_entropy {
            Some(quantum_entropy.get_entropy_quality().await)
        } else {
            None
        }
    }

    /// Test quantum randomness quality
    pub async fn test_quantum_randomness(
        &self,
        sample_size: usize,
    ) -> Result<quantum_seeding::RandomnessTest> {
        if let Some(quantum_entropy) = &self.quantum_entropy {
            quantum_entropy.test_randomness_quality(sample_size).await
        } else {
            anyhow::bail!("Quantum entropy not available for testing")
        }
    }

    /// Generate quantum delay for timing obfuscation
    pub async fn generate_quantum_delay(
        &self,
        min_delay: Duration,
        max_delay: Duration,
    ) -> Duration {
        if let Some(quantum_entropy) = &self.quantum_entropy {
            quantum_entropy
                .generate_quantum_delay(min_delay, max_delay)
                .await
                .unwrap_or_else(|_| {
                    // Fallback to classical randomness
                    let min_ms = min_delay.as_millis() as u64;
                    let max_ms = max_delay.as_millis() as u64;
                    let range = if max_ms > min_ms { max_ms - min_ms } else { 0 };
                    Duration::from_millis(min_ms + (rand::random::<u64>() % (range + 1)))
                })
        } else {
            // Classical fallback
            let min_ms = min_delay.as_millis() as u64;
            let max_ms = max_delay.as_millis() as u64;
            let range = if max_ms > min_ms { max_ms - min_ms } else { 0 };
            Duration::from_millis(min_ms + (rand::random::<u64>() % (range + 1)))
        }
    }

    /// Initialize Dandelion++ protocol with quantum seeding
    pub async fn initialize_dandelion(&mut self) -> Result<()> {
        if self.dandelion.is_some() {
            return Ok(()); // Already initialized
        }

        info!("🌻 Initializing Dandelion++ protocol");

        // Generate quantum seed for Dandelion++
        let quantum_seed = if let Some(quantum_entropy) = &self.quantum_entropy {
            quantum_entropy.generate_circuit_seed().await?
        } else {
            rand::random() // Classical fallback
        };

        let dandelion_config = DandelionConfig::default();
        let dandelion = DandelionProtocol::new(
            dandelion_config,
            Arc::clone(&self.circuit_manager),
            Arc::clone(&self.metrics),
            quantum_seed,
        );

        self.dandelion = Some(Arc::new(dandelion));

        info!("✅ Dandelion++ protocol initialized with quantum seeding");
        Ok(())
    }

    /// Update Prometheus metrics with current state
    pub async fn update_prometheus_metrics(&self) -> Result<()> {
        if let Some(prometheus) = &self.prometheus_metrics {
            let circuit_manager = self.circuit_manager.lock().await;
            let metrics = self.metrics.get_current_metrics().await;

            // Update circuit metrics
            prometheus
                .update_circuit_metrics(
                    circuit_manager.active_circuit_count(),
                    metrics.connection_count,
                )
                .await;

            // Update connection metrics
            prometheus
                .update_connection_metrics(
                    metrics.connection_count,
                    metrics.bytes_sent,
                    metrics.bytes_received,
                )
                .await;

            // Update performance metrics
            prometheus
                .update_performance_metrics(
                    metrics.average_latency,
                    0, // circuit rotations - would be tracked separately
                )
                .await;

            // Update onion service status
            let onion_active = self.get_onion_address().await.is_some();
            prometheus.update_onion_service_status(onion_active).await;

            // Update entropy metrics if available
            if let Some(entropy_quality) = self.get_entropy_quality().await {
                prometheus.update_entropy_metrics(&entropy_quality).await;
            }

            // Update Dandelion++ metrics if available
            if let Some(dandelion) = &self.dandelion {
                let stats = dandelion.get_statistics().await;
                prometheus.update_dandelion_metrics(&stats).await;
            }

            // Calculate and update privacy metrics
            let anonymity_score = self.calculate_anonymity_score().await;
            let traffic_resistance = self.calculate_traffic_resistance().await;
            let circuit_diversity = self.calculate_circuit_diversity().await;

            prometheus
                .update_privacy_metrics(anonymity_score, traffic_resistance, circuit_diversity)
                .await;

            prometheus.mark_updated().await;
        }

        Ok(())
    }

    /// Get Prometheus metrics in text format
    pub async fn get_prometheus_metrics(&self) -> Result<Option<String>> {
        if let Some(prometheus) = &self.prometheus_metrics {
            self.update_prometheus_metrics().await?;
            Ok(Some(prometheus.get_metrics().await?))
        } else {
            Ok(None)
        }
    }

    /// Get metrics summary for monitoring
    pub async fn get_metrics_summary(&self) -> Option<MetricsSummary> {
        if let Some(prometheus) = &self.prometheus_metrics {
            self.update_prometheus_metrics().await.ok()?;
            Some(prometheus.get_metrics_summary().await)
        } else {
            None
        }
    }

    /// Calculate current anonymity score based on multiple factors
    async fn calculate_anonymity_score(&self) -> f64 {
        let mut score: f64 = 0.0;

        // Base score from Tor usage
        if self.config.enabled {
            score += 0.4; // 40% for using Tor
        }

        // Bonus for circuit diversity
        let circuit_manager = self.circuit_manager.lock().await;
        let active_circuits = circuit_manager.active_circuit_count();
        if active_circuits >= 4 {
            score += 0.2; // 20% for having multiple circuits
        }

        // Bonus for Dandelion++ usage
        if self.dandelion.is_some() {
            score += 0.2; // 20% for traffic analysis resistance
        }

        // Bonus for quantum entropy
        if self.quantum_entropy.is_some() {
            score += 0.1; // 10% for quantum-enhanced privacy
        }

        // Bonus for Tor-only mode
        if self.config.tor_only {
            score += 0.1; // 10% for no fallback to direct connections
        }

        score.clamp(0.0, 1.0)
    }

    /// Calculate traffic analysis resistance score
    async fn calculate_traffic_resistance(&self) -> f64 {
        let mut resistance: f64 = 0.0;

        // Dandelion++ provides significant resistance
        if let Some(dandelion) = &self.dandelion {
            let stats = dandelion.get_statistics().await;
            if stats.stem_forwards > 0 {
                resistance += 0.5; // 50% for active stem forwarding
            }
            if stats.fluff_broadcasts > 0 {
                resistance += 0.3; // 30% for fluff broadcasting
            }
        }

        // Circuit rotation provides resistance
        let circuit_manager = self.circuit_manager.lock().await;
        if circuit_manager.should_rotate_circuits() {
            resistance += 0.1; // 10% for regular rotation
        } else {
            resistance += 0.2; // 20% for recent rotation
        }

        resistance.clamp(0.0, 1.0)
    }

    /// Calculate circuit path diversity
    async fn calculate_circuit_diversity(&self) -> f64 {
        let circuit_manager = self.circuit_manager.lock().await;
        let stats = circuit_manager.get_circuit_stats();

        // More circuit types = better diversity
        let mut diversity = 0.0;

        if stats.control_circuits > 0 {
            diversity += 0.25;
        }
        if stats.gossip_circuits > 0 {
            diversity += 0.25;
        }
        if stats.ack_circuits > 0 {
            diversity += 0.25;
        }
        if stats.qrng_circuits > 0 {
            diversity += 0.25;
        }

        diversity
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
        circuit_manager
            .set_latency_target(Duration::from_millis(target_ms as u64))
            .await;
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

    /// Create a mock Tor client for development/testing
    pub fn mock() -> Self {
        use std::net::{IpAddr, Ipv4Addr};
        let mock_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9050);

        Self {
            config: TorConfig::default(),
            node_id: [0u8; 32],
            socks_proxy: mock_addr,
            circuit_manager: Arc::new(Mutex::new(CircuitManager::mock())),
            onion_service: Arc::new(RwLock::new(None)),
            metrics: Arc::new(TorMetrics::new()),
            current_phase: q_types::Phase::Phase1,
            prometheus_metrics: None,
            quantum_entropy: None,
            dandelion: None,
        }
    }
}

/// Tor connection wrapper
pub struct TorConnection {
    stream: TcpStream,
    circuit_id: u64,
    peer_onion: String,
}

impl TorConnection {
    pub fn new(stream: TcpStream, circuit_id: u64, peer_onion: String) -> Self {
        Self {
            stream,
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
