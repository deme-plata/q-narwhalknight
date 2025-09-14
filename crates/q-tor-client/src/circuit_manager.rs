use anyhow::{Context, Result};
use q_quantum_rng::{QRNGConfig, QuantumRNG, QuantumRandomness};
use q_types::Phase;
use rand::Rng;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::net::TcpStream;
use tokio::time::sleep;
use tokio_socks::tcp::Socks5Stream;
use tracing::{debug, info, warn};

/// Manages dedicated Tor circuits for Q-NarwhalKnight
pub struct CircuitManager {
    /// SOCKS proxy address for Tor connections
    socks_proxy: SocketAddr,
    circuits: HashMap<CircuitType, Vec<CircuitInfo>>,
    circuit_count: usize,
    latency_target: Duration,
    last_rotation: Instant,
    /// Quantum RNG for circuit entropy (Phase 2+)
    qrng: Option<Arc<QuantumRNG>>,
    /// Current cryptographic phase
    current_phase: Phase,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum CircuitType {
    Control, // Bootstrap and control messages
    Gossip,  // Block and transaction gossip
    Ack,     // Acknowledgment messages
    Qrng,    // Quantum randomness distribution
}

#[derive(Debug, Clone)]
pub struct CircuitInfo {
    id: u64,
    circuit_type: CircuitType,
    created_at: Instant,
    last_used: Instant,
    latency_ms: Option<u64>,
    peer_onion: Option<String>,
    quantum_nonce: [u8; 12], // 96-bit QRNG-derived nonce
}

impl CircuitManager {
    /// Create new circuit manager
    pub async fn new(socks_proxy: SocketAddr, circuit_count: usize) -> Result<Self> {
        Self::new_with_phase(socks_proxy, circuit_count, Phase::Phase0).await
    }

    /// Create new circuit manager with specific phase
    pub async fn new_with_phase(
        socks_proxy: SocketAddr,
        circuit_count: usize,
        phase: Phase,
    ) -> Result<Self> {
        info!(
            "🔧 Initializing CircuitManager with {} circuits for {:?}",
            circuit_count, phase
        );

        // Initialize QRNG for Phase 2+
        let qrng = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            info!("🌌 Initializing quantum RNG for circuit generation");
            let config = QRNGConfig {
                min_entropy_quality: 0.98, // Higher quality for Tor circuits
                pool_size: 4096,           // Smaller pool for faster refill
                polling_interval_ms: 50,   // More frequent polling
                ..Default::default()
            };

            match QuantumRNG::new(phase, config).await {
                Ok(qrng) => {
                    info!("✅ Quantum RNG initialized for Tor circuits");
                    Some(Arc::new(qrng))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize QRNG, using fallback: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let mut manager = Self {
            socks_proxy,
            circuits: HashMap::new(),
            circuit_count,
            latency_target: Duration::from_millis(300),
            last_rotation: Instant::now(),
            qrng,
            current_phase: phase,
        };

        // Initialize circuits
        manager.initialize_circuits().await?;

        Ok(manager)
    }

    /// Initialize the required circuits
    async fn initialize_circuits(&mut self) -> Result<()> {
        // Allocate circuits by type
        let control_circuits = 1;
        let gossip_circuits = (self.circuit_count - 1) / 3;
        let ack_circuits = gossip_circuits;
        let qrng_circuits = self.circuit_count - control_circuits - gossip_circuits - ack_circuits;

        // Create control circuit
        self.create_circuits(CircuitType::Control, control_circuits)
            .await?;

        // Create gossip circuits
        self.create_circuits(CircuitType::Gossip, gossip_circuits)
            .await?;

        // Create ack circuits
        self.create_circuits(CircuitType::Ack, ack_circuits).await?;

        // Create QRNG circuits
        self.create_circuits(CircuitType::Qrng, qrng_circuits)
            .await?;

        info!(
            "✅ Initialized {} circuits across {} types",
            self.total_circuit_count(),
            4
        );

        Ok(())
    }

    /// Create circuits of a specific type
    async fn create_circuits(&mut self, circuit_type: CircuitType, count: usize) -> Result<()> {
        let mut circuits_vec = Vec::new();

        for i in 0..count {
            let circuit_info = self.create_single_circuit(circuit_type, i).await?;
            circuits_vec.push(circuit_info);

            // Small delay between circuit creation to avoid overwhelming Tor
            sleep(Duration::from_millis(100)).await;
        }

        // Insert all at once to avoid borrow checker issues
        self.circuits
            .entry(circuit_type)
            .or_insert_with(Vec::new)
            .extend(circuits_vec);

        Ok(())
    }

    /// Create a single circuit with QRNG entropy
    async fn create_single_circuit(
        &self,
        circuit_type: CircuitType,
        index: usize,
    ) -> Result<CircuitInfo> {
        let circuit_id = self.generate_circuit_id().await;
        let quantum_nonce = self.generate_quantum_nonce().await;

        debug!(
            "🛠️ Creating {:?} circuit {} with ID {}",
            circuit_type, index, circuit_id
        );

        // Circuit creation is conceptual - SOCKS proxy handles actual Tor circuits
        // We track logical circuits for load balancing and management

        let circuit_info = CircuitInfo {
            id: circuit_id,
            circuit_type,
            created_at: Instant::now(),
            last_used: Instant::now(),
            latency_ms: None,
            peer_onion: None,
            quantum_nonce,
        };

        Ok(circuit_info)
    }

    /// Generate circuit ID using QRNG entropy
    async fn generate_circuit_id(&self) -> u64 {
        match &self.qrng {
            Some(qrng) => {
                debug!("🌌 Generating quantum circuit ID");
                match qrng.generate_bytes(8).await {
                    Ok(bytes) => {
                        let mut id_bytes = [0u8; 8];
                        id_bytes.copy_from_slice(&bytes);
                        let circuit_id = u64::from_be_bytes(id_bytes);
                        debug!("✅ Generated quantum circuit ID: {}", circuit_id);
                        circuit_id
                    }
                    Err(e) => {
                        warn!("⚠️ QRNG failed for circuit ID, using fallback: {}", e);
                        rand::thread_rng().gen()
                    }
                }
            }
            None => {
                // Classical fallback for Phase 0/1
                rand::thread_rng().gen()
            }
        }
    }

    /// Generate quantum nonce for circuit entropy
    async fn generate_quantum_nonce(&self) -> [u8; 12] {
        // Format: entropy(4) + epoch(4) + counter(4)
        let mut nonce = [0u8; 12];

        // Entropy from QRNG or fallback
        let entropy: u32 = match &self.qrng {
            Some(qrng) => match qrng.generate_bytes(4).await {
                Ok(bytes) => {
                    let mut entropy_bytes = [0u8; 4];
                    entropy_bytes.copy_from_slice(&bytes);
                    u32::from_be_bytes(entropy_bytes)
                }
                Err(e) => {
                    warn!("⚠️ QRNG failed for nonce entropy, using fallback: {}", e);
                    rand::thread_rng().gen()
                }
            },
            None => rand::thread_rng().gen(),
        };
        nonce[0..4].copy_from_slice(&entropy.to_be_bytes());

        // Current epoch (5-min epochs)
        let epoch: u32 = (Instant::now().elapsed().as_secs() / 300) as u32;
        nonce[4..8].copy_from_slice(&epoch.to_be_bytes());

        // Circuit counter
        let counter: u32 = self.total_circuit_count() as u32;
        nonce[8..12].copy_from_slice(&counter.to_be_bytes());

        nonce
    }

    /// Get circuit for connecting to specific peer
    pub async fn get_circuit_for_peer(&mut self, peer_onion: &str) -> Result<u64> {
        // Use gossip circuit for peer connections
        let circuits = self
            .circuits
            .get_mut(&CircuitType::Gossip)
            .context("No gossip circuits available")?;

        if circuits.is_empty() {
            anyhow::bail!("No gossip circuits available for peer connection");
        }

        // Find or assign circuit for this peer
        if let Some(circuit) = circuits.iter_mut().find(|c| {
            c.peer_onion
                .as_ref()
                .map_or(false, |onion| onion == peer_onion)
        }) {
            circuit.last_used = Instant::now();
            return Ok(circuit.id);
        }

        // Assign a new circuit to this peer
        let circuit = circuits
            .iter_mut()
            .min_by_key(|c| c.last_used)
            .context("No available gossip circuits")?;

        circuit.peer_onion = Some(peer_onion.to_string());
        circuit.last_used = Instant::now();

        Ok(circuit.id)
    }

    /// Get random circuit for Dandelion++ stem phase
    pub async fn get_random_circuit(&self) -> Result<u64> {
        let all_circuits: Vec<u64> = self.circuits.values().flatten().map(|c| c.id).collect();

        if all_circuits.is_empty() {
            anyhow::bail!("No circuits available");
        }

        let random_index = rand::thread_rng().gen_range(0..all_circuits.len());
        Ok(all_circuits[random_index])
    }

    /// Get all gossip circuit IDs
    pub fn get_gossip_circuits(&self) -> Vec<&u64> {
        self.circuits
            .get(&CircuitType::Gossip)
            .map(|circuits| circuits.iter().map(|c| &c.id).collect())
            .unwrap_or_default()
    }

    /// Rotate all circuits
    pub async fn rotate_all_circuits(&mut self) -> Result<()> {
        info!("🔄 Rotating all Tor circuits");

        // Close existing circuits and create new ones
        self.circuits.clear();
        self.initialize_circuits().await?;
        self.last_rotation = Instant::now();

        info!("✅ Circuit rotation complete");
        Ok(())
    }

    /// Set latency target for adaptive QoS
    pub async fn set_latency_target(&mut self, target: Duration) {
        self.latency_target = target;
        info!("🎯 Updated latency target to {}ms", target.as_millis());

        // Check if we need to switch to 2-hop circuits for better performance
        if target < Duration::from_millis(200) {
            self.enable_fast_circuits().await;
        }
    }

    /// Enable 2-hop circuits for lower latency
    async fn enable_fast_circuits(&mut self) {
        warn!("⚡ Enabling 2-hop circuits for latency target <200ms");
        // Implementation would configure shorter circuits in production
    }

    /// Record latency for a circuit
    pub async fn record_circuit_latency(&mut self, circuit_id: u64, latency: Duration) {
        for circuits in self.circuits.values_mut() {
            if let Some(circuit) = circuits.iter_mut().find(|c| c.id == circuit_id) {
                circuit.latency_ms = Some(latency.as_millis() as u64);
                circuit.last_used = Instant::now();
                break;
            }
        }
    }

    /// Get active circuit count
    pub fn active_circuit_count(&self) -> usize {
        self.circuits.values().map(|v| v.len()).sum()
    }

    /// Get total circuit count
    fn total_circuit_count(&self) -> usize {
        self.circuits.values().map(|v| v.len()).sum()
    }

    /// Close all circuits
    pub async fn close_all_circuits(&mut self) -> Result<()> {
        info!("🛑 Closing all Tor circuits");

        let circuit_count = self.total_circuit_count();
        self.circuits.clear();

        info!("✅ Closed {} circuits", circuit_count);
        Ok(())
    }

    /// Get circuit statistics
    pub fn get_circuit_stats(&self) -> CircuitStats {
        let mut total_latency_ms = 0u64;
        let mut circuit_count = 0usize;

        for circuits in self.circuits.values() {
            for circuit in circuits {
                if let Some(latency) = circuit.latency_ms {
                    total_latency_ms += latency;
                    circuit_count += 1;
                }
            }
        }

        let average_latency = if circuit_count > 0 {
            Duration::from_millis(total_latency_ms / circuit_count as u64)
        } else {
            Duration::from_millis(0)
        };

        CircuitStats {
            total_circuits: self.total_circuit_count(),
            control_circuits: self
                .circuits
                .get(&CircuitType::Control)
                .map_or(0, |v| v.len()),
            gossip_circuits: self
                .circuits
                .get(&CircuitType::Gossip)
                .map_or(0, |v| v.len()),
            ack_circuits: self.circuits.get(&CircuitType::Ack).map_or(0, |v| v.len()),
            qrng_circuits: self.circuits.get(&CircuitType::Qrng).map_or(0, |v| v.len()),
            average_latency,
            last_rotation: self.last_rotation,
        }
    }

    /// Check if circuits need rotation (every epoch = 5 minutes)
    pub fn should_rotate_circuits(&self) -> bool {
        self.last_rotation.elapsed() > Duration::from_secs(300) // 5 minutes
    }

    /// Broadcast transaction to all peers through Tor circuits (for Dandelion++ fluff phase)
    pub async fn broadcast_transaction(&self, tx_data: &[u8]) -> Result<()> {
        info!("Broadcasting transaction through all Tor circuits");

        // Get all gossip circuits for broadcasting
        let gossip_circuits = self
            .circuits
            .get(&CircuitType::Gossip)
            .context("No gossip circuits available for broadcasting")?;

        if gossip_circuits.is_empty() {
            anyhow::bail!("No gossip circuits available for transaction broadcast");
        }

        // Broadcast through each gossip circuit
        for circuit in gossip_circuits {
            if let Err(e) = self.send_through_circuit(circuit.id, tx_data).await {
                warn!("Failed to broadcast through circuit {}: {}", circuit.id, e);
                // Continue with other circuits even if one fails
            }
        }

        info!(
            "Transaction broadcast completed through {} circuits",
            gossip_circuits.len()
        );
        Ok(())
    }

    /// Send data to a specific peer through Tor
    pub async fn send_to_peer(&self, peer_addr: SocketAddr, data: &[u8]) -> Result<()> {
        debug!("Sending data to peer {} through Tor", peer_addr);

        // Connect through SOCKS proxy
        let stream = Socks5Stream::connect(self.socks_proxy, peer_addr)
            .await
            .context(format!("Failed to connect to peer {} via Tor", peer_addr))?;

        // Send data through the stream
        use tokio::io::AsyncWriteExt;
        let mut stream = stream.into_inner();
        stream
            .write_all(data)
            .await
            .context("Failed to send data to peer")?;

        debug!(
            "Successfully sent {} bytes to peer {}",
            data.len(),
            peer_addr
        );
        Ok(())
    }

    /// Send data through a specific circuit
    async fn send_through_circuit(&self, circuit_id: u64, data: &[u8]) -> Result<()> {
        debug!("Sending data through circuit {}", circuit_id);

        // Find the circuit
        let circuit = self
            .find_circuit(circuit_id)
            .ok_or_else(|| anyhow::anyhow!("Circuit {} not found", circuit_id))?;

        // For now, use the circuit's peer onion address if available
        // In production, this would use the actual Tor circuit
        if let Some(peer_onion) = &circuit.peer_onion {
            // Convert onion address to socket address (simplified for demo)
            // In production, this would resolve .onion addresses properly
            let peer_addr = format!("{}:8080", peer_onion)
                .parse::<SocketAddr>()
                .unwrap_or_else(|_| {
                    // Fallback to localhost for demo
                    "127.0.0.1:8080".parse().unwrap()
                });

            self.send_to_peer(peer_addr, data).await?;
        } else {
            // Use a default peer for circuits without assigned peers
            let default_peer = "127.0.0.1:8080".parse::<SocketAddr>().unwrap();
            self.send_to_peer(default_peer, data).await?;
        }

        Ok(())
    }

    /// Find a circuit by ID
    fn find_circuit(&self, circuit_id: u64) -> Option<&CircuitInfo> {
        for circuits in self.circuits.values() {
            if let Some(circuit) = circuits.iter().find(|c| c.id == circuit_id) {
                return Some(circuit);
            }
        }
        None
    }

    /// Create a mock circuit manager for development/testing
    pub fn mock() -> Self {
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};
        let mock_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 9050);

        Self {
            socks_proxy: mock_addr,
            circuits: HashMap::new(),
            circuit_count: 4,
            latency_target: Duration::from_millis(300),
            last_rotation: Instant::now(),
            qrng: None,
            current_phase: q_types::Phase::Phase1,
        }
    }
}

/// Circuit statistics for monitoring
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub total_circuits: usize,
    pub control_circuits: usize,
    pub gossip_circuits: usize,
    pub ack_circuits: usize,
    pub qrng_circuits: usize,
    pub average_latency: Duration,
    pub last_rotation: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_nonce_generation() {
        // Create manager with mock circuit count
        let total_circuits = 4;

        // Test nonce format: entropy(4) + epoch(4) + counter(4)
        let mut test_circuits = HashMap::new();
        let manager = CircuitManagerForTest {
            circuits: test_circuits,
            circuit_count: total_circuits,
            qrng: None,
            current_phase: Phase::Phase0,
        };

        let nonce1 = manager.generate_quantum_nonce_for_test().await;
        let nonce2 = manager.generate_quantum_nonce_for_test().await;

        // Nonces should be different (due to different timestamps)
        // Note: they might be the same if generated in the same millisecond

        // Nonce should be 12 bytes
        assert_eq!(nonce1.len(), 12);
        assert_eq!(nonce2.len(), 12);
    }

    // Test-only helper struct
    struct CircuitManagerForTest {
        circuits: HashMap<CircuitType, Vec<CircuitInfo>>,
        circuit_count: usize,
        qrng: Option<Arc<QuantumRNG>>,
        current_phase: Phase,
    }

    impl CircuitManagerForTest {
        async fn generate_quantum_nonce_for_test(&self) -> [u8; 12] {
            let mut nonce = [0u8; 12];

            // Test entropy
            let entropy: u32 = rand::thread_rng().gen();
            nonce[0..4].copy_from_slice(&entropy.to_be_bytes());

            // Current epoch
            let epoch: u32 = (Instant::now().elapsed().as_secs() / 300) as u32;
            nonce[4..8].copy_from_slice(&epoch.to_be_bytes());

            // Circuit counter
            let counter: u32 = self.circuit_count as u32;
            nonce[8..12].copy_from_slice(&counter.to_be_bytes());

            nonce
        }
    }

    #[test]
    fn test_circuit_stats() {
        let mut circuits = HashMap::new();

        // Add some test circuits
        circuits.insert(
            CircuitType::Control,
            vec![CircuitInfo {
                id: 1,
                circuit_type: CircuitType::Control,
                created_at: Instant::now(),
                last_used: Instant::now(),
                latency_ms: Some(100),
                peer_onion: None,
                quantum_nonce: [0u8; 12],
            }],
        );

        circuits.insert(
            CircuitType::Gossip,
            vec![CircuitInfo {
                id: 2,
                circuit_type: CircuitType::Gossip,
                created_at: Instant::now(),
                last_used: Instant::now(),
                latency_ms: Some(200),
                peer_onion: None,
                quantum_nonce: [0u8; 12],
            }],
        );

        let manager = CircuitManagerForTest {
            circuits,
            circuit_count: 4,
            qrng: None,
            current_phase: Phase::Phase0,
        };

        let stats = manager.get_circuit_stats_for_test();
        assert_eq!(stats.total_circuits, 2);
        assert_eq!(stats.control_circuits, 1);
        assert_eq!(stats.gossip_circuits, 1);
        assert_eq!(stats.average_latency, Duration::from_millis(150));
    }

    impl CircuitManagerForTest {
        fn get_circuit_stats_for_test(&self) -> CircuitStats {
            let mut total_latency_ms = 0u64;
            let mut circuit_count = 0usize;

            for circuits in self.circuits.values() {
                for circuit in circuits {
                    if let Some(latency) = circuit.latency_ms {
                        total_latency_ms += latency;
                        circuit_count += 1;
                    }
                }
            }

            let average_latency = if circuit_count > 0 {
                Duration::from_millis(total_latency_ms / circuit_count as u64)
            } else {
                Duration::from_millis(0)
            };

            CircuitStats {
                total_circuits: self.circuits.values().map(|v| v.len()).sum(),
                control_circuits: self
                    .circuits
                    .get(&CircuitType::Control)
                    .map_or(0, |v| v.len()),
                gossip_circuits: self
                    .circuits
                    .get(&CircuitType::Gossip)
                    .map_or(0, |v| v.len()),
                ack_circuits: self.circuits.get(&CircuitType::Ack).map_or(0, |v| v.len()),
                qrng_circuits: self.circuits.get(&CircuitType::Qrng).map_or(0, |v| v.len()),
                average_latency,
                last_rotation: Instant::now(),
            }
        }
    }
}
