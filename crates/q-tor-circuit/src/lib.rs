/// Q-Tor-Circuit: Dedicated circuit management for Q-NarwhalKnight
/// Manages 4 dedicated circuits per validator:
/// - 1 Control circuit (bootstrap, discovery)
/// - 3 Gossip circuits (blocks, acks, quantum randomness)

use anyhow::{Context, Result};
use async_trait::async_trait;
use q_tor_client::{TorConfig, TorMetrics};
use q_types::{NodeId, Phase};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

pub mod pool;
pub mod qos;
pub mod rotation;

pub use pool::CircuitPool;
pub use qos::AdaptiveQoS;
pub use rotation::CircuitRotator;

/// Circuit purpose classification
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum CircuitPurpose {
    /// Bootstrap and peer discovery
    Control,
    /// Block and transaction gossip
    BlockGossip,
    /// Acknowledgment messages
    AckGossip,
    /// Quantum randomness distribution
    QuantumBeacon,
}

impl CircuitPurpose {
    /// Get priority level (lower = higher priority)
    pub fn priority(&self) -> u8 {
        match self {
            Self::Control => 0,       // Highest priority
            Self::QuantumBeacon => 1, // Quantum data is critical
            Self::BlockGossip => 2,   // Block data
            Self::AckGossip => 3,     // Acknowledgments
        }
    }

    /// Get expected message size
    pub fn expected_message_size(&self) -> usize {
        match self {
            Self::Control => 1024,       // Small control messages
            Self::QuantumBeacon => 512,  // Quantum entropy data
            Self::BlockGossip => 8192,   // Block data
            Self::AckGossip => 256,      // Small ack messages
        }
    }

    /// Get circuit rotation interval
    pub fn rotation_interval(&self) -> Duration {
        match self {
            Self::Control => Duration::from_secs(600),    // 10 minutes
            Self::QuantumBeacon => Duration::from_secs(300), // 5 minutes (more sensitive)
            Self::BlockGossip => Duration::from_secs(450),   // 7.5 minutes
            Self::AckGossip => Duration::from_secs(450),     // 7.5 minutes
        }
    }
}

/// Individual circuit information
#[derive(Debug, Clone)]
pub struct CircuitInfo {
    pub id: u64,
    pub purpose: CircuitPurpose,
    pub created_at: Instant,
    pub last_used: Instant,
    pub latency_ms: Option<u64>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub hop_count: u8,
    pub quantum_nonce: [u8; 12], // QRNG-derived entropy
    pub peer_assignments: Vec<String>, // Onion addresses using this circuit
}

impl CircuitInfo {
    /// Create new circuit info
    pub fn new(id: u64, purpose: CircuitPurpose, quantum_nonce: [u8; 12]) -> Self {
        Self {
            id,
            purpose,
            created_at: Instant::now(),
            last_used: Instant::now(),
            latency_ms: None,
            bytes_sent: 0,
            bytes_received: 0,
            hop_count: 3, // Default 3-hop circuit
            quantum_nonce,
            peer_assignments: Vec::new(),
        }
    }

    /// Check if circuit needs rotation
    pub fn needs_rotation(&self) -> bool {
        self.created_at.elapsed() > self.purpose.rotation_interval()
    }

    /// Update usage statistics
    pub fn update_usage(&mut self, bytes_sent: u64, bytes_received: u64, latency: Duration) {
        self.last_used = Instant::now();
        self.bytes_sent += bytes_sent;
        self.bytes_received += bytes_received;
        self.latency_ms = Some(latency.as_millis() as u64);
    }

    /// Get utilization score (for load balancing)
    pub fn utilization_score(&self) -> f64 {
        let age_factor = self.created_at.elapsed().as_secs_f64() / 300.0; // Normalize to 5 minutes
        let usage_factor = (self.bytes_sent + self.bytes_received) as f64 / 1_000_000.0; // Normalize to 1MB
        let peer_factor = self.peer_assignments.len() as f64 / 10.0; // Normalize to 10 peers
        
        (age_factor + usage_factor + peer_factor) / 3.0
    }
}

/// Dedicated circuit manager for validator networking
pub struct DedicatedCircuitManager {
    circuits: Arc<RwLock<HashMap<CircuitPurpose, Vec<CircuitInfo>>>>,
    circuit_pool: Arc<Mutex<CircuitPool>>,
    qos_manager: Arc<Mutex<AdaptiveQoS>>,
    rotator: Arc<Mutex<CircuitRotator>>,
    metrics: Arc<TorMetrics>,
    config: TorConfig,
    node_id: NodeId,
}

impl DedicatedCircuitManager {
    /// Create new dedicated circuit manager
    pub async fn new(config: TorConfig, node_id: NodeId) -> Result<Self> {
        info!("🔧 Initializing DedicatedCircuitManager for validator {}", 
              hex::encode(&node_id[..4]));

        let circuits = Arc::new(RwLock::new(HashMap::new()));
        let metrics = Arc::new(TorMetrics::new());
        
        let circuit_pool = Arc::new(Mutex::new(
            CircuitPool::new(config.circuit_count, node_id).await?
        ));
        
        let qos_manager = Arc::new(Mutex::new(
            AdaptiveQoS::new(config.latency_target_ms.unwrap_or(300))
        ));
        
        let rotator = Arc::new(Mutex::new(
            CircuitRotator::new(Duration::from_secs(300)) // 5-minute default rotation
        ));

        let mut manager = Self {
            circuits,
            circuit_pool,
            qos_manager,
            rotator,
            metrics,
            config,
            node_id,
        };

        // Initialize circuits for each purpose
        manager.initialize_all_circuits().await?;

        Ok(manager)
    }

    /// Initialize circuits for all purposes
    async fn initialize_all_circuits(&mut self) -> Result<()> {
        let purposes = [
            CircuitPurpose::Control,
            CircuitPurpose::BlockGossip,
            CircuitPurpose::AckGossip,
            CircuitPurpose::QuantumBeacon,
        ];

        let circuits_per_purpose = self.config.circuit_count / purposes.len();
        let mut circuits_map = self.circuits.write().await;

        for purpose in &purposes {
            let count = if *purpose == CircuitPurpose::Control { 
                1 // Only one control circuit needed
            } else { 
                circuits_per_purpose.max(1) // At least one circuit per purpose
            };

            let mut purpose_circuits = Vec::new();

            for i in 0..count {
                let circuit_info = self.create_circuit_for_purpose(*purpose, i).await?;
                purpose_circuits.push(circuit_info);
            }

            circuits_map.insert(*purpose, purpose_circuits);
            info!("✅ Created {} circuits for {:?}", count, purpose);
        }

        Ok(())
    }

    /// Create circuit for specific purpose
    async fn create_circuit_for_purpose(&self, purpose: CircuitPurpose, index: usize) -> Result<CircuitInfo> {
        let circuit_id = self.generate_circuit_id();
        let quantum_nonce = self.generate_quantum_nonce(purpose, index);

        debug!("🛠️ Creating {:?} circuit {} with ID {}", purpose, index, circuit_id);

        // Create circuit info
        let circuit_info = CircuitInfo::new(circuit_id, purpose, quantum_nonce);

        // Register with circuit pool
        {
            let mut pool = self.circuit_pool.lock().await;
            pool.register_circuit(circuit_id, purpose).await?;
        }

        Ok(circuit_info)
    }

    /// Generate quantum-seeded circuit ID
    fn generate_circuit_id(&self) -> u64 {
        // In production, this would use actual QRNG from quantum hardware
        // For now, use cryptographically secure random with node_id seed
        let mut rng = rand::thread_rng();
        let base: u64 = rng.gen();
        
        // XOR with node_id for deterministic element
        let node_hash = self.node_id.iter().fold(0u64, |acc, &b| acc ^ (b as u64));
        
        base ^ node_hash
    }

    /// Generate quantum nonce for circuit entropy
    fn generate_quantum_nonce(&self, purpose: CircuitPurpose, index: usize) -> [u8; 12] {
        let mut nonce = [0u8; 12];
        
        // Entropy from QRNG (4 bytes)
        let entropy: u32 = rand::thread_rng().gen();
        nonce[0..4].copy_from_slice(&entropy.to_be_bytes());
        
        // Current epoch (4 bytes) - 5-minute epochs
        let epoch: u32 = (Instant::now().elapsed().as_secs() / 300) as u32;
        nonce[4..8].copy_from_slice(&epoch.to_be_bytes());
        
        // Circuit identifier (4 bytes): purpose + index
        let circuit_id: u32 = ((purpose.priority() as u32) << 16) | (index as u32);
        nonce[8..12].copy_from_slice(&circuit_id.to_be_bytes());
        
        nonce
    }

    /// Get circuit for specific purpose
    pub async fn get_circuit_for_purpose(&self, purpose: CircuitPurpose) -> Result<u64> {
        let circuits = self.circuits.read().await;
        
        if let Some(purpose_circuits) = circuits.get(&purpose) {
            if let Some(circuit) = purpose_circuits.first() {
                return Ok(circuit.id);
            }
        }
        
        anyhow::bail!("No circuits available for purpose {:?}", purpose);
    }

    /// Get best circuit for peer connection
    pub async fn get_best_circuit_for_peer(&self, peer_onion: &str, purpose: CircuitPurpose) -> Result<u64> {
        let mut circuits = self.circuits.write().await;
        
        if let Some(purpose_circuits) = circuits.get_mut(&purpose) {
            // Find circuit already assigned to this peer
            if let Some(circuit) = purpose_circuits.iter_mut().find(|c| {
                c.peer_assignments.contains(&peer_onion.to_string())
            }) {
                circuit.last_used = Instant::now();
                return Ok(circuit.id);
            }

            // Find least utilized circuit
            if let Some(circuit) = purpose_circuits.iter_mut()
                .min_by(|a, b| a.utilization_score().partial_cmp(&b.utilization_score()).unwrap()) {
                
                circuit.peer_assignments.push(peer_onion.to_string());
                circuit.last_used = Instant::now();
                return Ok(circuit.id);
            }
        }
        
        anyhow::bail!("No circuits available for peer {} with purpose {:?}", peer_onion, purpose);
    }

    /// Record circuit usage
    pub async fn record_circuit_usage(
        &self, 
        circuit_id: u64, 
        bytes_sent: u64, 
        bytes_received: u64, 
        latency: Duration
    ) -> Result<()> {
        let mut circuits = self.circuits.write().await;
        
        for purpose_circuits in circuits.values_mut() {
            if let Some(circuit) = purpose_circuits.iter_mut().find(|c| c.id == circuit_id) {
                circuit.update_usage(bytes_sent, bytes_received, latency);
                
                // Record metrics
                self.metrics.record_bytes_sent(bytes_sent).await;
                self.metrics.record_bytes_received(bytes_received).await;
                self.metrics.record_connection_latency(latency).await;
                
                // Check QoS targets
                {
                    let mut qos = self.qos_manager.lock().await;
                    qos.update_latency_measurement(circuit_id, latency).await;
                }
                
                return Ok(());
            }
        }
        
        warn!("⚠️ Circuit {} not found for usage recording", circuit_id);
        Ok(())
    }

    /// Rotate circuits if needed
    pub async fn check_and_rotate_circuits(&mut self) -> Result<()> {
        let should_rotate = {
            let rotator = self.rotator.lock().await;
            rotator.should_rotate_now()
        };

        if should_rotate {
            self.rotate_expired_circuits().await?;
        }

        Ok(())
    }

    /// Rotate expired circuits
    async fn rotate_expired_circuits(&mut self) -> Result<()> {
        info!("🔄 Checking for circuits that need rotation");

        let mut circuits = self.circuits.write().await;
        let mut rotated_count = 0;

        for (purpose, purpose_circuits) in circuits.iter_mut() {
            let mut new_circuits = Vec::new();
            
            for (index, circuit) in purpose_circuits.iter().enumerate() {
                if circuit.needs_rotation() {
                    debug!("🔄 Rotating {:?} circuit {} (age: {}s)", 
                           purpose, circuit.id, circuit.created_at.elapsed().as_secs());
                    
                    // Create replacement circuit
                    let new_circuit = self.create_circuit_for_purpose(*purpose, index).await?;
                    new_circuits.push(new_circuit);
                    rotated_count += 1;
                } else {
                    new_circuits.push(circuit.clone());
                }
            }
            
            *purpose_circuits = new_circuits;
        }

        if rotated_count > 0 {
            info!("✅ Rotated {} circuits", rotated_count);
            
            // Update rotator
            {
                let mut rotator = self.rotator.lock().await;
                rotator.mark_rotation_complete().await;
            }
        }

        Ok(())
    }

    /// Get circuit statistics by purpose
    pub async fn get_circuit_stats(&self) -> HashMap<CircuitPurpose, CircuitPurposeStats> {
        let circuits = self.circuits.read().await;
        let mut stats = HashMap::new();

        for (purpose, purpose_circuits) in circuits.iter() {
            let total_sent: u64 = purpose_circuits.iter().map(|c| c.bytes_sent).sum();
            let total_received: u64 = purpose_circuits.iter().map(|c| c.bytes_received).sum();
            
            let latencies: Vec<u64> = purpose_circuits.iter()
                .filter_map(|c| c.latency_ms)
                .collect();
            
            let avg_latency = if latencies.is_empty() {
                Duration::from_millis(0)
            } else {
                Duration::from_millis(latencies.iter().sum::<u64>() / latencies.len() as u64)
            };

            let purpose_stats = CircuitPurposeStats {
                circuit_count: purpose_circuits.len(),
                total_bytes_sent: total_sent,
                total_bytes_received: total_received,
                average_latency: avg_latency,
                active_connections: purpose_circuits.iter()
                    .map(|c| c.peer_assignments.len())
                    .sum(),
            };

            stats.insert(*purpose, purpose_stats);
        }

        stats
    }

    /// Get total active circuit count
    pub async fn active_circuit_count(&self) -> usize {
        let circuits = self.circuits.read().await;
        circuits.values().map(|v| v.len()).sum()
    }

    /// Shutdown all circuits
    pub async fn shutdown_all_circuits(&mut self) -> Result<()> {
        info!("🛑 Shutting down all dedicated circuits");

        let circuit_count = self.active_circuit_count().await;
        
        // Clear circuit tracking
        {
            let mut circuits = self.circuits.write().await;
            circuits.clear();
        }

        // Shutdown circuit pool
        {
            let mut pool = self.circuit_pool.lock().await;
            pool.shutdown().await?;
        }

        info!("✅ Shutdown {} circuits", circuit_count);
        Ok(())
    }
}

/// Statistics for circuits of a specific purpose
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitPurposeStats {
    pub circuit_count: usize,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub average_latency: Duration,
    pub active_connections: usize,
}

/// Trait for circuit-aware networking
#[async_trait]
pub trait CircuitAware {
    /// Send message through appropriate circuit
    async fn send_via_circuit(&self, message: &[u8], purpose: CircuitPurpose, target: Option<&str>) -> Result<()>;
    
    /// Get circuit statistics
    async fn get_circuit_stats(&self) -> HashMap<CircuitPurpose, CircuitPurposeStats>;
    
    /// Force circuit rotation
    async fn force_circuit_rotation(&mut self) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_purpose_priority() {
        assert_eq!(CircuitPurpose::Control.priority(), 0);
        assert_eq!(CircuitPurpose::QuantumBeacon.priority(), 1);
        assert_eq!(CircuitPurpose::BlockGossip.priority(), 2);
        assert_eq!(CircuitPurpose::AckGossip.priority(), 3);
    }

    #[test]
    fn test_circuit_info_creation() {
        let circuit_id = 12345;
        let purpose = CircuitPurpose::BlockGossip;
        let quantum_nonce = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        let circuit = CircuitInfo::new(circuit_id, purpose, quantum_nonce);

        assert_eq!(circuit.id, circuit_id);
        assert_eq!(circuit.purpose, purpose);
        assert_eq!(circuit.quantum_nonce, quantum_nonce);
        assert_eq!(circuit.bytes_sent, 0);
        assert_eq!(circuit.peer_assignments.len(), 0);
    }

    #[test]
    fn test_circuit_utilization_score() {
        let mut circuit = CircuitInfo::new(1, CircuitPurpose::BlockGossip, [0u8; 12]);
        
        let initial_score = circuit.utilization_score();
        
        // Update usage
        circuit.update_usage(1000, 2000, Duration::from_millis(100));
        circuit.peer_assignments.push("peer1.onion".to_string());
        
        let updated_score = circuit.utilization_score();
        
        // Score should change after usage update
        assert_ne!(initial_score, updated_score);
    }

    #[test]
    fn test_purpose_rotation_intervals() {
        assert_eq!(CircuitPurpose::Control.rotation_interval(), Duration::from_secs(600));
        assert_eq!(CircuitPurpose::QuantumBeacon.rotation_interval(), Duration::from_secs(300));
        assert_eq!(CircuitPurpose::BlockGossip.rotation_interval(), Duration::from_secs(450));
        assert_eq!(CircuitPurpose::AckGossip.rotation_interval(), Duration::from_secs(450));
    }
}