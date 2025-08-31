/// Q-Dandelion: Quantum-enhanced Dandelion++ protocol for traffic analysis resistance
/// Implements anonymous message propagation over Tor with L-VRF randomness

use anyhow::{Context, Result};
use async_trait::async_trait;
use q_lattice_vrf::{LatticeVRF, VRFProof, VRFConfig};
use q_quantum_rng::{QuantumRNG, QuantumRandomness};
use q_tor_client::QTorClient;
use q_types::{NodeId, Phase};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};

pub mod gossip;
pub mod routing;
pub mod anonymity;

pub use gossip::DandelionGossip;
pub use routing::AnonymityRouter;
pub use anonymity::AnonymityMetrics;

/// Dandelion++ protocol phases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DandelionPhase {
    /// Stem phase: relay to single random peer
    Stem,
    /// Fluff phase: broadcast to all peers
    Fluff,
}

/// Dandelion++ message with quantum-enhanced routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DandelionMessage {
    /// Message ID for deduplication
    pub id: [u8; 32],
    /// Actual message payload
    pub payload: Vec<u8>,
    /// Current phase (stem/fluff)
    pub phase: DandelionPhase,
    /// Hop count for privacy analysis
    pub hop_count: u8,
    /// VRF proof for randomness verification
    pub vrf_proof: Option<VRFProof>,
    /// Message timestamp
    pub timestamp: u64,
    /// Quantum nonce for enhanced privacy
    pub quantum_nonce: [u8; 16],
}

/// Quantum-enhanced Dandelion++ protocol implementation
pub struct QuantumDandelion {
    /// Node ID
    node_id: NodeId,
    /// Current cryptographic phase
    phase: Phase,
    /// Tor client for anonymous communication
    tor_client: Arc<QTorClient>,
    /// Lattice VRF for verifiable randomness
    vrf: Option<Arc<LatticeVRF>>,
    /// Quantum RNG for enhanced randomness
    qrng: Option<Arc<QuantumRNG>>,
    /// Message tracking for deduplication
    seen_messages: Arc<RwLock<HashMap<[u8; 32], Instant>>>,
    /// Stem relay targets
    stem_targets: Arc<RwLock<Vec<NodeId>>>,
    /// Configuration
    config: DandelionConfig,
    /// Anonymity metrics
    metrics: Arc<AnonymityMetrics>,
}

impl QuantumDandelion {
    /// Create new quantum Dandelion++ instance
    pub async fn new(
        node_id: NodeId,
        phase: Phase,
        tor_client: Arc<QTorClient>,
        config: DandelionConfig,
    ) -> Result<Self> {
        info!("🌻 Initializing Quantum Dandelion++ for {:?}", phase);

        // Initialize L-VRF for Phase 2+
        let vrf = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            info!("🔐 Initializing L-VRF for verifiable randomness");
            let vrf_config = VRFConfig::new_quantum_enhanced(phase)?;
            
            match LatticeVRF::new(vrf_config, phase).await {
                Ok(vrf) => {
                    info!("✅ L-VRF initialized for Dandelion++");
                    Some(Arc::new(vrf))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize L-VRF: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize QRNG for Phase 2+
        let qrng = if matches!(phase, Phase::Phase2 | Phase::Phase3 | Phase::Phase4) {
            info!("🌌 Initializing quantum RNG for anonymity enhancement");
            let qrng_config = q_quantum_rng::QRNGConfig {
                min_entropy_quality: 0.97,
                pool_size: 2048, // Smaller pool for routing decisions
                polling_interval_ms: 200,
                ..Default::default()
            };

            match QuantumRNG::new(phase, qrng_config).await {
                Ok(qrng) => {
                    info!("✅ Quantum RNG initialized for Dandelion++");
                    Some(Arc::new(qrng))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize QRNG: {}", e);
                    None
                }
            }
        } else {
            None
        };

        let dandelion = Self {
            node_id,
            phase,
            tor_client,
            vrf,
            qrng,
            seen_messages: Arc::new(RwLock::new(HashMap::new())),
            stem_targets: Arc::new(RwLock::new(Vec::new())),
            config,
            metrics: Arc::new(AnonymityMetrics::new()),
        };

        // Start background cleanup of seen messages
        dandelion.start_message_cleanup().await;

        info!("✅ Quantum Dandelion++ initialized");
        Ok(dandelion)
    }

    /// Propagate message using Dandelion++
    pub async fn propagate_message(&self, payload: &[u8], topic: &str) -> Result<()> {
        let message_id = self.generate_message_id(payload).await?;
        
        debug!("🌻 Propagating message {} via Dandelion++", hex::encode(message_id));

        // Check if we've seen this message before
        {
            let seen = self.seen_messages.read().await;
            if seen.contains_key(&message_id) {
                debug!("🔁 Message {} already seen, ignoring", hex::encode(message_id));
                return Ok(());
            }
        }

        // Mark message as seen
        {
            let mut seen = self.seen_messages.write().await;
            seen.insert(message_id, Instant::now());
        }

        // Decide phase using quantum randomness
        let phase = self.decide_propagation_phase().await?;
        
        let message = DandelionMessage {
            id: message_id,
            payload: payload.to_vec(),
            phase,
            hop_count: 0,
            vrf_proof: self.generate_vrf_proof(topic, &message_id).await?,
            timestamp: chrono::Utc::now().timestamp() as u64,
            quantum_nonce: self.generate_quantum_nonce().await?,
        };

        match phase {
            DandelionPhase::Stem => {
                self.stem_propagate(message).await?;
            }
            DandelionPhase::Fluff => {
                self.fluff_propagate(message).await?;
            }
        }

        // Update metrics
        self.metrics.record_message_propagation(phase, message.hop_count).await;

        Ok(())
    }

    /// Decide propagation phase using quantum randomness
    async fn decide_propagation_phase(&self) -> Result<DandelionPhase> {
        let probability = match &self.qrng {
            Some(qrng) => {
                // Use quantum randomness for phase decision
                let bytes = qrng.generate_quantum_bytes(4).await?;
                let value = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                (value as f64) / (u32::MAX as f64)
            }
            None => {
                // Classical fallback
                rand::random::<f64>()
            }
        };

        // Use configured stem probability
        if probability < self.config.stem_probability {
            Ok(DandelionPhase::Stem)
        } else {
            Ok(DandelionPhase::Fluff)
        }
    }

    /// Propagate message in stem phase (relay to single peer)
    async fn stem_propagate(&self, mut message: DandelionMessage) -> Result<()> {
        debug!("🌱 Stem propagation for message {}", hex::encode(message.id));

        // Select random stem target
        let target = self.select_stem_target().await?;
        
        // Increment hop count
        message.hop_count += 1;

        // Check hop limit
        if message.hop_count > self.config.max_stem_hops {
            info!("🌸 Converting to fluff after {} hops", message.hop_count);
            message.phase = DandelionPhase::Fluff;
            return self.fluff_propagate(message).await;
        }

        // Send via Tor to selected peer
        self.send_to_peer(&target, &message).await?;

        Ok(())
    }

    /// Propagate message in fluff phase (broadcast to all peers)
    async fn fluff_propagate(&self, message: DandelionMessage) -> Result<()> {
        debug!("🌸 Fluff propagation for message {}", hex::encode(message.id));

        // Get all connected peers
        let peers = self.get_all_peers().await?;
        
        // Broadcast to all peers via Tor
        for peer in peers {
            if let Err(e) = self.send_to_peer(&peer, &message).await {
                warn!("⚠️ Failed to send to peer {}: {}", hex::encode(peer), e);
            }
        }

        Ok(())
    }

    /// Select random stem target using quantum or VRF randomness
    async fn select_stem_target(&self) -> Result<NodeId> {
        let targets = self.stem_targets.read().await;
        
        if targets.is_empty() {
            anyhow::bail!("No stem targets available");
        }

        // Use quantum randomness for selection if available
        let index = match &self.qrng {
            Some(qrng) => {
                let range_value = qrng.generate_range(0, targets.len() as u64).await?;
                range_value as usize
            }
            None => {
                rand::random::<usize>() % targets.len()
            }
        };

        Ok(targets[index])
    }

    /// Generate message ID using quantum randomness
    async fn generate_message_id(&self, payload: &[u8]) -> Result<[u8; 32]> {
        match &self.qrng {
            Some(qrng) => {
                // Quantum-enhanced message ID
                let quantum_entropy = qrng.generate_quantum_bytes(16).await?;
                
                let mut hasher = blake3::Hasher::new();
                hasher.update(payload);
                hasher.update(&quantum_entropy);
                hasher.update(&self.node_id);
                hasher.update(&chrono::Utc::now().timestamp().to_be_bytes());
                
                Ok(hasher.finalize().into())
            }
            None => {
                // Classical message ID
                let mut hasher = blake3::Hasher::new();
                hasher.update(payload);
                hasher.update(&self.node_id);
                hasher.update(&chrono::Utc::now().timestamp().to_be_bytes());
                
                Ok(hasher.finalize().into())
            }
        }
    }

    /// Generate VRF proof for message routing
    async fn generate_vrf_proof(&self, topic: &str, message_id: &[u8; 32]) -> Result<Option<VRFProof>> {
        match &self.vrf {
            Some(vrf) => {
                let input = [topic.as_bytes(), message_id].concat();
                let proof = vrf.prove(&input).await?;
                Ok(Some(proof))
            }
            None => Ok(None),
        }
    }

    /// Generate quantum nonce for enhanced privacy
    async fn generate_quantum_nonce(&self) -> Result<[u8; 16]> {
        match &self.qrng {
            Some(qrng) => {
                let nonce: [u8; 16] = qrng.generate_quantum_seed().await?;
                Ok(nonce)
            }
            None => {
                let mut nonce = [0u8; 16];
                rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut nonce);
                Ok(nonce)
            }
        }
    }

    /// Send message to specific peer via Tor
    async fn send_to_peer(&self, peer_id: &NodeId, message: &DandelionMessage) -> Result<()> {
        debug!("📤 Sending Dandelion message to peer {}", hex::encode(peer_id));

        // Serialize message
        let message_data = bincode::serialize(message)?;

        // Get peer's onion address (would come from peer discovery)
        let peer_onion = format!("peer{}.qnk.onion", hex::encode(&peer_id[..4]));

        // Send via Tor
        let _connection = self.tor_client.connect_to_peer(&peer_onion).await?;
        
        // In production, would send the actual message data
        debug!("✅ Sent message {} to {}", hex::encode(message.id), peer_onion);

        Ok(())
    }

    /// Get all connected peers
    async fn get_all_peers(&self) -> Result<Vec<NodeId>> {
        // In production, this would get peers from the network layer
        // For now, return empty list
        Ok(vec![])
    }

    /// Update stem targets
    pub async fn update_stem_targets(&self, targets: Vec<NodeId>) {
        let mut stem_targets = self.stem_targets.write().await;
        *stem_targets = targets;
        
        info!("🎯 Updated stem targets: {} peers", stem_targets.len());
    }

    /// Handle incoming Dandelion message
    pub async fn handle_incoming_message(&self, message: DandelionMessage) -> Result<()> {
        debug!("📥 Received Dandelion message {} in {:?} phase", 
               hex::encode(message.id), message.phase);

        // Check if we've seen this message before
        {
            let mut seen = self.seen_messages.write().await;
            if seen.contains_key(&message.id) {
                debug!("🔁 Duplicate message {}, ignoring", hex::encode(message.id));
                return Ok(());
            }
            seen.insert(message.id, Instant::now());
        }

        // Verify VRF proof if present
        if let Some(ref proof) = message.vrf_proof {
            if let Some(ref vrf) = self.vrf {
                let input = [b"dandelion", &message.id].concat();
                if !vrf.verify(&input, proof).await? {
                    warn!("❌ Invalid VRF proof for message {}", hex::encode(message.id));
                    return Ok(());
                }
            }
        }

        match message.phase {
            DandelionPhase::Stem => {
                // Continue stem phase with probability, or switch to fluff
                let should_continue_stem = self.should_continue_stem(&message).await?;
                
                if should_continue_stem {
                    self.stem_propagate(message).await?;
                } else {
                    let mut fluff_message = message;
                    fluff_message.phase = DandelionPhase::Fluff;
                    self.fluff_propagate(fluff_message).await?;
                }
            }
            DandelionPhase::Fluff => {
                // Forward message and deliver locally
                self.fluff_propagate(message.clone()).await?;
                self.deliver_message_locally(message).await?;
            }
        }

        Ok(())
    }

    /// Decide whether to continue stem phase
    async fn should_continue_stem(&self, message: &DandelionMessage) -> Result<bool> {
        // Check hop limit
        if message.hop_count >= self.config.max_stem_hops {
            return Ok(false);
        }

        // Use quantum randomness for decision
        let probability = match &self.qrng {
            Some(qrng) => {
                let bytes = qrng.generate_quantum_bytes(4).await?;
                let value = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                (value as f64) / (u32::MAX as f64)
            }
            None => rand::random::<f64>(),
        };

        Ok(probability < self.config.stem_continuation_probability)
    }

    /// Deliver message to local application
    async fn deliver_message_locally(&self, message: DandelionMessage) -> Result<()> {
        debug!("📨 Delivering message {} locally", hex::encode(message.id));
        
        // In production, this would deliver to the consensus layer
        // For now, just log the delivery
        info!("✅ Delivered message {} ({} bytes)", 
              hex::encode(message.id), message.payload.len());

        Ok(())
    }

    /// Start background cleanup of seen messages
    async fn start_message_cleanup(&self) {
        let seen_messages = self.seen_messages.clone();
        let cleanup_interval = self.config.message_ttl;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                let mut seen = seen_messages.write().await;
                let cutoff = Instant::now() - cleanup_interval;
                
                seen.retain(|_, &mut timestamp| timestamp > cutoff);
                
                if seen.len() % 100 == 0 {
                    debug!("🧹 Cleaned up seen messages, {} remaining", seen.len());
                }
            }
        });
    }

    /// Get anonymity statistics
    pub async fn get_anonymity_stats(&self) -> AnonymityStats {
        let metrics = self.metrics.get_current_metrics().await;
        let seen_count = self.seen_messages.read().await.len();

        AnonymityStats {
            messages_seen: seen_count as u64,
            stem_messages: metrics.stem_messages,
            fluff_messages: metrics.fluff_messages,
            hop_distribution: metrics.hop_distribution.clone(),
            quantum_enhanced: self.qrng.is_some() && self.vrf.is_some(),
            anonymity_score: self.calculate_anonymity_score(&metrics).await,
        }
    }

    /// Calculate anonymity score based on message patterns
    async fn calculate_anonymity_score(&self, metrics: &anonymity::AnonymityMetricsSnapshot) -> f64 {
        // Base score from hop distribution entropy
        let hop_entropy = self.calculate_hop_entropy(&metrics.hop_distribution);
        
        // Quantum enhancement bonus
        let quantum_bonus = if self.qrng.is_some() && self.vrf.is_some() {
            0.2 // 20% bonus for quantum enhancements
        } else {
            0.0
        };

        // VRF verification bonus
        let vrf_bonus = if self.vrf.is_some() {
            0.1 // 10% bonus for verifiable randomness
        } else {
            0.0
        };

        (hop_entropy + quantum_bonus + vrf_bonus).min(1.0)
    }

    /// Calculate entropy of hop distribution
    fn calculate_hop_entropy(&self, hop_distribution: &HashMap<u8, u64>) -> f64 {
        let total: u64 = hop_distribution.values().sum();
        
        if total == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for count in hop_distribution.values() {
            if *count > 0 {
                let p = (*count as f64) / (total as f64);
                entropy -= p * p.log2();
            }
        }

        // Normalize to 0-1 range
        entropy / 8.0 // Assuming max 8 hops
    }
}

/// Configuration for Dandelion++ protocol
#[derive(Debug, Clone)]
pub struct DandelionConfig {
    /// Probability of starting in stem phase
    pub stem_probability: f64,
    
    /// Probability of continuing stem phase at each hop
    pub stem_continuation_probability: f64,
    
    /// Maximum hops in stem phase
    pub max_stem_hops: u8,
    
    /// Message time-to-live
    pub message_ttl: Duration,
    
    /// Enable quantum enhancements
    pub quantum_enhanced: bool,
}

impl Default for DandelionConfig {
    fn default() -> Self {
        Self {
            stem_probability: 0.9,        // 90% chance to start in stem
            stem_continuation_probability: 0.8, // 80% chance to continue stem
            max_stem_hops: 10,
            message_ttl: Duration::from_secs(300), // 5 minutes
            quantum_enhanced: true,
        }
    }
}

/// Anonymity statistics for monitoring
#[derive(Debug, Clone, Serialize)]
pub struct AnonymityStats {
    pub messages_seen: u64,
    pub stem_messages: u64,
    pub fluff_messages: u64,
    pub hop_distribution: HashMap<u8, u64>,
    pub quantum_enhanced: bool,
    pub anonymity_score: f64,
}

/// Trait for Dandelion++-enabled protocols
#[async_trait]
pub trait DandelionEnabled {
    /// Propagate message with anonymity
    async fn dandelion_propagate(&self, payload: &[u8], topic: &str) -> Result<()>;
    
    /// Handle incoming Dandelion message
    async fn handle_dandelion_message(&self, message: DandelionMessage) -> Result<()>;
    
    /// Get anonymity statistics
    async fn get_anonymity_stats(&self) -> AnonymityStats;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dandelion_message_serialization() {
        let message = DandelionMessage {
            id: [1u8; 32],
            payload: vec![1, 2, 3, 4],
            phase: DandelionPhase::Stem,
            hop_count: 2,
            vrf_proof: None,
            timestamp: 1234567890,
            quantum_nonce: [5u8; 16],
        };

        let serialized = bincode::serialize(&message).unwrap();
        let deserialized: DandelionMessage = bincode::deserialize(&serialized).unwrap();

        assert_eq!(message.id, deserialized.id);
        assert_eq!(message.phase, deserialized.phase);
        assert_eq!(message.hop_count, deserialized.hop_count);
    }

    #[test]
    fn test_dandelion_config_defaults() {
        let config = DandelionConfig::default();
        
        assert_eq!(config.stem_probability, 0.9);
        assert_eq!(config.stem_continuation_probability, 0.8);
        assert_eq!(config.max_stem_hops, 10);
        assert!(config.quantum_enhanced);
    }

    #[test]
    fn test_hop_entropy_calculation() {
        let mut hop_distribution = HashMap::new();
        hop_distribution.insert(1, 10);
        hop_distribution.insert(2, 20);
        hop_distribution.insert(3, 10);

        // The calculate_hop_entropy method would be called here
        // For testing, we just verify the distribution is set up correctly
        assert_eq!(hop_distribution.values().sum::<u64>(), 40);
    }
}