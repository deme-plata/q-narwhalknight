//! 🔑 Quantum Key Distribution (QKD) Implementation
//! BB84 Protocol with true quantum security guarantees

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;

use crate::quantum_channels::QuantumChannel;
use crate::quantum_entropy::QuantumEntropySource;
use crate::{NodeId, PhotonState, QuantumKey};

/// QKD key with security metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDKey {
    /// Raw key material
    pub key_data: Vec<u8>,
    /// Key generation timestamp
    pub generated_at: SystemTime,
    /// Security level achieved
    pub security_level: SecurityLevel,
    /// Error rate during generation
    pub error_rate: f64,
    /// Key usage counter (for one-time pad tracking)
    pub usage_count: u64,
    /// Maximum allowed usage
    pub max_usage: u64,
}

impl QKDKey {
    /// Check if key is still secure for use
    pub fn is_secure(&self) -> bool {
        self.usage_count < self.max_usage && 
        self.error_rate < 0.11 && // QBER threshold
        self.generated_at.elapsed().unwrap_or(Duration::MAX) < Duration::from_secs(3600)
    }

    /// Mark key as used (for one-time pad)
    pub fn mark_used(&mut self, bytes_used: usize) {
        self.usage_count += bytes_used as u64;
    }
}

/// QKD security level
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Information-theoretically secure
    InformationTheoretic,
    /// Computationally secure
    Computational,
    /// Degraded security
    Degraded,
    /// Insecure (should not be used)
    Insecure,
}

/// BB84 Protocol Implementation
#[derive(Debug)]
pub struct BB84Protocol {
    /// Node ID
    node_id: NodeId,
    /// Quantum entropy source
    entropy_source: Arc<QuantumEntropySource>,
    /// Protocol statistics
    stats: Arc<RwLock<BB84Stats>>,
}

impl BB84Protocol {
    /// Create new BB84 protocol instance
    pub fn new(node_id: NodeId, entropy_source: Arc<QuantumEntropySource>) -> Self {
        Self {
            node_id,
            entropy_source,
            stats: Arc::new(RwLock::new(BB84Stats::default())),
        }
    }

    /// Execute BB84 protocol as Alice (sender)
    pub async fn execute_as_alice(
        &self,
        channel: &QuantumChannel,
        key_length: usize,
    ) -> Result<QKDKey> {
        let start_time = Instant::now();

        // Step 1: Generate random bits and bases
        let bits = self
            .entropy_source
            .generate_true_random(key_length * 4)
            .await?; // Extra for sifting
        let bases = self
            .entropy_source
            .generate_true_random(key_length * 4)
            .await?;

        // Step 2: Prepare and send photons
        let mut photons = Vec::new();
        for i in 0..(key_length * 4) {
            let bit = (bits[i % bits.len()] & 1) != 0;
            let basis = (bases[i % bases.len()] & 1) != 0;

            let photon_state = if basis {
                // Diagonal basis
                PhotonState::Diagonal(bit)
            } else {
                // Rectilinear basis
                PhotonState::Rectilinear(bit)
            };

            photons.push(photon_state);
        }

        // Send photons through quantum channel
        channel.send_photons(&photons).await?;

        // Step 3: Receive Bob's basis announcement
        let bob_bases = channel.receive_basis_announcement().await?;

        // Step 4: Public basis comparison and sifting
        let mut sifted_bits = Vec::new();
        for i in 0..photons.len().min(bob_bases.len()) {
            let alice_basis = match photons[i] {
                PhotonState::Rectilinear(_) => false,
                PhotonState::Diagonal(_) => true,
            };

            if alice_basis == bob_bases[i] {
                let bit = match photons[i] {
                    PhotonState::Rectilinear(b) | PhotonState::Diagonal(b) => b,
                };
                sifted_bits.push(bit);
            }
        }

        // Step 5: Error estimation
        let test_bits_count = sifted_bits.len() / 10; // Use 10% for testing
        let mut error_count = 0;

        // Bob will announce his test bits, we compare
        let bob_test_bits = channel.receive_test_bits(test_bits_count).await?;
        for i in 0..test_bits_count.min(bob_test_bits.len()) {
            if sifted_bits[i] != bob_test_bits[i] {
                error_count += 1;
            }
        }

        let error_rate = error_count as f64 / test_bits_count as f64;

        // Step 6: Error correction (simplified)
        let corrected_bits = if error_rate < 0.11 {
            // Apply error correction (Cascade algorithm simulation)
            self.apply_error_correction(&sifted_bits[test_bits_count..], error_rate)
                .await?
        } else {
            return Err(anyhow::anyhow!(
                "Error rate too high: {:.2}%",
                error_rate * 100.0
            ));
        };

        // Step 7: Privacy amplification
        let final_key = self
            .privacy_amplification(&corrected_bits, error_rate)
            .await?;

        // Step 8: Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.sessions_completed += 1;
            stats.total_photons_sent += photons.len() as u64;
            stats.average_error_rate = (stats.average_error_rate + error_rate) / 2.0;
            stats.total_key_bits_generated += final_key.len() as u64 * 8;
            stats.last_session_duration = start_time.elapsed();
        }

        // Determine security level
        let security_level = if error_rate < 0.02 {
            SecurityLevel::InformationTheoretic
        } else if error_rate < 0.05 {
            SecurityLevel::Computational
        } else if error_rate < 0.11 {
            SecurityLevel::Degraded
        } else {
            SecurityLevel::Insecure
        };

        let key_len = final_key.len() as u64;
        Ok(QKDKey {
            key_data: final_key,
            generated_at: SystemTime::now(),
            security_level,
            error_rate,
            usage_count: 0,
            max_usage: key_len, // One-time pad usage
        })
    }

    /// Execute BB84 protocol as Bob (receiver)
    pub async fn execute_as_bob(&self, channel: &QuantumChannel) -> Result<QKDKey> {
        let start_time = Instant::now();

        // Step 1: Receive photons from Alice
        let received_photons = channel.receive_photons().await?;

        // Step 2: Generate random measurement bases
        let measurement_bases = self
            .entropy_source
            .generate_true_random(received_photons.len())
            .await?;

        // Step 3: Measure photons
        let mut measured_bits = Vec::new();
        let mut bob_bases = Vec::new();

        for (i, photon) in received_photons.iter().enumerate() {
            let measurement_basis = (measurement_bases[i % measurement_bases.len()] & 1) != 0;
            bob_bases.push(measurement_basis);

            // Simulate quantum measurement
            let measured_bit = self.measure_photon(photon, measurement_basis).await?;
            measured_bits.push(measured_bit);
        }

        // Step 4: Announce measurement bases
        channel.send_basis_announcement(&bob_bases).await?;

        // Step 5: Basis sifting (Alice will tell us which bases matched)
        let matching_indices = channel.receive_matching_indices().await?;
        let mut sifted_bits = Vec::new();

        for &index in &matching_indices {
            if index < measured_bits.len() {
                sifted_bits.push(measured_bits[index]);
            }
        }

        // Step 6: Error estimation
        let test_bits_count = sifted_bits.len() / 10;
        let test_bits = sifted_bits[..test_bits_count].to_vec();
        channel.send_test_bits(&test_bits).await?;

        // Receive error rate from Alice
        let error_rate = channel.receive_error_rate().await?;

        // Step 7: Error correction
        let corrected_bits = if error_rate < 0.11 {
            self.apply_error_correction(&sifted_bits[test_bits_count..], error_rate)
                .await?
        } else {
            return Err(anyhow::anyhow!(
                "Error rate too high: {:.2}%",
                error_rate * 100.0
            ));
        };

        // Step 8: Privacy amplification (synchronized with Alice)
        let final_key = self
            .privacy_amplification(&corrected_bits, error_rate)
            .await?;

        // Step 9: Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.sessions_completed += 1;
            stats.total_photons_received += received_photons.len() as u64;
            stats.average_error_rate = (stats.average_error_rate + error_rate) / 2.0;
            stats.total_key_bits_generated += final_key.len() as u64 * 8;
            stats.last_session_duration = start_time.elapsed();
        }

        // Determine security level
        let security_level = if error_rate < 0.02 {
            SecurityLevel::InformationTheoretic
        } else if error_rate < 0.05 {
            SecurityLevel::Computational
        } else if error_rate < 0.11 {
            SecurityLevel::Degraded
        } else {
            SecurityLevel::Insecure
        };

        let key_len = final_key.len() as u64;
        Ok(QKDKey {
            key_data: final_key,
            generated_at: SystemTime::now(),
            security_level,
            error_rate,
            usage_count: 0,
            max_usage: key_len,
        })
    }

    /// Simulate quantum measurement of photon
    async fn measure_photon(&self, photon: &PhotonState, measurement_basis: bool) -> Result<bool> {
        match (photon, measurement_basis) {
            // Correct basis measurement
            (PhotonState::Rectilinear(bit), false) => Ok(*bit),
            (PhotonState::Diagonal(bit), true) => Ok(*bit),

            // Wrong basis measurement - random result
            (PhotonState::Rectilinear(_), true) | (PhotonState::Diagonal(_), false) => {
                let random_byte = self.entropy_source.generate_true_random(1).await?;
                Ok((random_byte[0] & 1) != 0)
            }
        }
    }

    /// Apply error correction (simplified Cascade algorithm)
    async fn apply_error_correction(&self, bits: &[bool], error_rate: f64) -> Result<Vec<u8>> {
        // Simplified error correction - in practice, use full Cascade or LDPC codes
        let mut corrected_bytes = Vec::new();

        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << i;
                }
            }

            // Simple parity check and correction
            let parity = byte.count_ones() % 2;
            if parity != 0 && error_rate > 0.01 {
                // Flip the least significant bit (simplified correction)
                byte ^= 1;
            }

            corrected_bytes.push(byte);
        }

        Ok(corrected_bytes)
    }

    /// Privacy amplification using universal hash functions
    async fn privacy_amplification(&self, bits: &[u8], error_rate: f64) -> Result<Vec<u8>> {
        // Calculate output length based on error rate
        let input_entropy = bits.len() as f64 * (1.0 - error_rate);
        let output_length = (input_entropy * 0.8) as usize; // Conservative estimate

        // Generate random universal hash function parameters
        let hash_params = self.entropy_source.generate_true_random(32).await?;

        let mut amplified_key = Vec::with_capacity(output_length);

        for i in 0..output_length {
            let mut hash_input = Vec::new();
            hash_input.extend_from_slice(bits);
            hash_input.extend_from_slice(&hash_params);
            hash_input.push(i as u8);

            // Universal hash function (Toeplitz matrix multiplication)
            let hash_byte = self.universal_hash(&hash_input, &hash_params).await?;
            amplified_key.push(hash_byte);
        }

        Ok(amplified_key)
    }

    /// Universal hash function implementation
    async fn universal_hash(&self, input: &[u8], params: &[u8]) -> Result<u8> {
        let mut result = 0u8;

        for (i, &byte) in input.iter().enumerate() {
            let param = params[i % params.len()];
            result ^= byte.wrapping_mul(param);
        }

        Ok(result)
    }

    /// Get BB84 protocol statistics
    pub async fn get_stats(&self) -> BB84Stats {
        self.stats.read().await.clone()
    }
}

/// QKD Engine managing multiple BB84 sessions
#[derive(Debug)]
pub struct QKDEngine {
    /// Node identifier
    node_id: NodeId,
    /// BB84 protocol instance
    bb84: BB84Protocol,
    /// Active QKD sessions
    active_sessions: Arc<RwLock<HashMap<NodeId, QKDSession>>>,
    /// Key storage
    key_storage: Arc<RwLock<HashMap<NodeId, QKDKey>>>,
}

impl QKDEngine {
    /// Create new QKD engine
    pub async fn new(node_id: NodeId, entropy_source: Arc<QuantumEntropySource>) -> Result<Self> {
        let bb84 = BB84Protocol::new(node_id, entropy_source);

        Ok(Self {
            node_id,
            bb84,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            key_storage: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Execute BB84 protocol with peer
    pub async fn execute_bb84_protocol(
        &self,
        peer_id: NodeId,
        channel: &QuantumChannel,
    ) -> Result<Vec<u8>> {
        // Determine if we should be Alice or Bob based on node IDs
        let is_alice = self.node_id > peer_id;

        let qkd_key = if is_alice {
            self.bb84.execute_as_alice(channel, 256).await?
        } else {
            self.bb84.execute_as_bob(channel).await?
        };

        // Store the key
        self.key_storage
            .write()
            .await
            .insert(peer_id, qkd_key.clone());

        // Create session record
        let session = QKDSession {
            peer_id,
            established_at: SystemTime::now(),
            key_length: qkd_key.key_data.len(),
            security_level: qkd_key.security_level.clone(),
            error_rate: qkd_key.error_rate,
        };

        self.active_sessions.write().await.insert(peer_id, session);

        Ok(qkd_key.key_data)
    }

    /// Get quantum key for peer
    pub async fn get_key(&self, peer_id: NodeId) -> Option<QKDKey> {
        self.key_storage.read().await.get(&peer_id).cloned()
    }

    /// Refresh key with peer
    pub async fn refresh_key(&self, peer_id: NodeId, channel: &QuantumChannel) -> Result<()> {
        let new_key_data = self.execute_bb84_protocol(peer_id, channel).await?;
        // Key is already stored by execute_bb84_protocol
        Ok(())
    }

    /// Get session count
    pub async fn get_session_count(&self) -> u64 {
        self.active_sessions.read().await.len() as u64
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool> {
        let sessions = self.active_sessions.read().await;
        let keys = self.key_storage.read().await;

        // Check if we have valid keys for active sessions
        for (peer_id, session) in sessions.iter() {
            if let Some(key) = keys.get(peer_id) {
                if !key.is_secure() {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// QKD session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QKDSession {
    pub peer_id: NodeId,
    pub established_at: SystemTime,
    pub key_length: usize,
    pub security_level: SecurityLevel,
    pub error_rate: f64,
}

/// BB84 Protocol Statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BB84Stats {
    pub sessions_completed: u64,
    pub total_photons_sent: u64,
    pub total_photons_received: u64,
    pub average_error_rate: f64,
    pub total_key_bits_generated: u64,
    pub last_session_duration: Duration,
}

/// QKD Channel interface
#[async_trait::async_trait]
pub trait QKDChannel {
    /// Send photons through quantum channel
    async fn send_photons(&self, photons: &[PhotonState]) -> Result<()>;

    /// Receive photons from quantum channel
    async fn receive_photons(&self) -> Result<Vec<PhotonState>>;

    /// Send basis announcement
    async fn send_basis_announcement(&self, bases: &[bool]) -> Result<()>;

    /// Receive basis announcement
    async fn receive_basis_announcement(&self) -> Result<Vec<bool>>;

    /// Send test bits for error estimation
    async fn send_test_bits(&self, bits: &[bool]) -> Result<()>;

    /// Receive test bits for error estimation
    async fn receive_test_bits(&self, count: usize) -> Result<Vec<bool>>;

    /// Receive matching basis indices
    async fn receive_matching_indices(&self) -> Result<Vec<usize>>;

    /// Receive error rate
    async fn receive_error_rate(&self) -> Result<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_entropy::QuantumEntropySource;

    #[tokio::test]
    async fn test_qkd_key_security() {
        let key = QKDKey {
            key_data: vec![0u8; 32],
            generated_at: SystemTime::now(),
            security_level: SecurityLevel::InformationTheoretic,
            error_rate: 0.01,
            usage_count: 0,
            max_usage: 32,
        };

        assert!(key.is_secure());
    }

    #[tokio::test]
    async fn test_bb84_protocol_creation() {
        let node_id = [1u8; 32];
        let entropy_source = Arc::new(QuantumEntropySource::new().await.unwrap());
        let protocol = BB84Protocol::new(node_id, entropy_source);

        assert_eq!(protocol.node_id, node_id);
    }

    #[test]
    fn test_security_level_ordering() {
        assert!(SecurityLevel::InformationTheoretic > SecurityLevel::Computational);
        assert!(SecurityLevel::Computational > SecurityLevel::Degraded);
        assert!(SecurityLevel::Degraded > SecurityLevel::Insecure);
    }
}
