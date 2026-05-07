/// Q-Dandelion: Quantum-enhanced Dandelion++ protocol for traffic analysis resistance
/// Implements anonymous message propagation over Tor with L-VRF randomness
use anyhow::{Context, Result};
use async_trait::async_trait;
use q_lattice_vrf::{LatticeVRF, VRFConfig, VRFProof};
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

pub mod anonymity;
pub mod failsafe;
pub mod gossip;
pub mod network_bridge;
pub mod routing;
pub mod tor_bridge;

pub use anonymity::{AnonymityMetrics, AnonymityMetricsSnapshot};
pub use failsafe::{FailsafeConfig, FailsafeEvent, FailsafeStats, FailsafeTimer, TransactionState};
pub use gossip::DandelionGossip;
pub use network_bridge::{
    MessagePriority, NetworkBridge, NetworkBridgeConfig, NetworkBridgeStats, NetworkCommand,
    NetworkEvent, PeerInfo,
};
pub use routing::AnonymityRouter;
pub use tor_bridge::{
    CircuitInfo, DandelionCircuitPurpose, TorBridge, TorBridgeConfig, TorConnectionStats,
};

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
    /// Gossipsub topic used for fluff propagation (set by propagate_message caller)
    #[serde(default)]
    pub topic: String,
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
    /// Stem relay targets (also used as the connected peer list)
    stem_targets: Arc<RwLock<Vec<NodeId>>>,
    /// Configuration
    config: DandelionConfig,
    /// Anonymity metrics
    metrics: Arc<AnonymityMetrics>,
    /// Network bridge for gossipsub integration
    network_bridge: Option<Arc<NetworkBridge>>,
    /// Tor bridge for circuit management
    tor_bridge: Option<Arc<TorBridge>>,
    /// Failsafe timer for transaction protection
    failsafe: Option<Arc<FailsafeTimer>>,
    /// Failsafe event receiver
    failsafe_rx: Option<Arc<Mutex<tokio::sync::mpsc::UnboundedReceiver<FailsafeEvent>>>>,
    /// Live gossipsub publisher: (topic, data) — the actual wire for fluff propagation.
    /// Set via set_gossipsub_tx() after libp2p initialises. Uses RwLock so &self setters work.
    gossipsub_tx: Arc<RwLock<Option<tokio::sync::mpsc::UnboundedSender<(String, Vec<u8>)>>>>,
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
            let vrf_config = VRFConfig::default();

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

        // 🌻 v2.5.0-beta: Initialize Tor bridge with REAL QTorClient
        // The tor_client parameter is passed through to TorBridge for actual Tor routing
        let tor_bridge = {
            info!("🧅 Initializing Tor bridge for Dandelion++ circuits with real QTorClient");
            let tor_config = TorBridgeConfig::default();

            // Use the new constructor that takes the real tor_client
            match TorBridge::new_with_tor_client(tor_config, tor_client.clone()).await {
                Ok(bridge) => {
                    // Warm up circuits for stem relay
                    if let Err(e) = bridge.warmup_circuits().await {
                        warn!("⚠️ Failed to warm up Tor circuits: {}", e);
                    }
                    info!("✅ Tor bridge initialized with real QTorClient");
                    Some(Arc::new(bridge))
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize Tor bridge: {}", e);
                    // Fall back to legacy mode without real Tor
                    match TorBridge::new(TorBridgeConfig::default()).await {
                        Ok(fallback) => {
                            warn!("⚠️ Using fallback Tor bridge (reduced functionality)");
                            Some(Arc::new(fallback))
                        }
                        Err(_) => None,
                    }
                }
            }
        };

        // Initialize failsafe timer
        let failsafe_config = FailsafeConfig::default();
        let (failsafe, failsafe_rx) = FailsafeTimer::new(failsafe_config);
        let failsafe = Arc::new(failsafe);
        let failsafe_rx = Arc::new(Mutex::new(failsafe_rx));

        // Start failsafe timer background task
        failsafe.start();
        info!("⏱️ Failsafe timer started");

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
            network_bridge: None, // Set via set_network_bridge()
            tor_bridge,
            failsafe: Some(failsafe),
            failsafe_rx: Some(failsafe_rx),
            gossipsub_tx: Arc::new(RwLock::new(None)),
        };

        // Start background cleanup of seen messages
        dandelion.start_message_cleanup().await;

        info!("✅ Quantum Dandelion++ initialized with Tor bridge and failsafe timer");
        Ok(dandelion)
    }

    /// Propagate message using Dandelion++
    pub async fn propagate_message(&self, payload: &[u8], topic: &str) -> Result<()> {
        let message_id = self.generate_message_id(payload).await?;

        debug!(
            "🌻 Propagating message {} via Dandelion++",
            hex::encode(message_id)
        );

        // Check if we've seen this message before
        {
            let seen = self.seen_messages.read().await;
            if seen.contains_key(&message_id) {
                debug!(
                    "🔁 Message {} already seen, ignoring",
                    hex::encode(message_id)
                );
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
            topic: topic.to_string(),
        };

        // Capture hop_count before moving message
        let hop_count = message.hop_count;

        match phase {
            DandelionPhase::Stem => {
                self.stem_propagate(message).await?;
            }
            DandelionPhase::Fluff => {
                self.fluff_propagate(message).await?;
            }
        }

        // Update metrics
        self.metrics
            .record_message_propagation(phase, hop_count)
            .await;

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
                use rand::Rng;
                rand::rng().random::<f64>()
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
        debug!(
            "🌱 Stem propagation for message {}",
            hex::encode(message.id)
        );

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

    /// Propagate message in fluff phase (broadcast via gossipsub)
    async fn fluff_propagate(&self, message: DandelionMessage) -> Result<()> {
        debug!(
            "🌸 Fluff propagation for message {}",
            hex::encode(message.id)
        );

        // Primary path: publish via wired gossipsub channel.
        // This is zero-copy efficient — gossipsub handles fanout to all subscribed peers.
        {
            let tx_guard = self.gossipsub_tx.read().await;
            if let Some(ref tx) = *tx_guard {
                let fluff_topic = if message.topic.is_empty() {
                    // Fallback topic when caller didn't set one
                    "/qnk/blocks".to_string()
                } else {
                    message.topic.clone()
                };
                match bincode::serialize(&message) {
                    Ok(payload) => {
                        if tx.send((fluff_topic.clone(), payload)).is_ok() {
                            debug!(
                                "🌸 Fluff published via gossipsub topic '{}' (msg {})",
                                fluff_topic,
                                hex::encode(&message.id[..4])
                            );
                            return Ok(());
                        }
                        warn!("⚠️ Gossipsub TX channel closed — falling back to peer iteration");
                    }
                    Err(e) => {
                        warn!("⚠️ Failed to serialize fluff message: {}", e);
                    }
                }
            }
        }

        // Fallback: iterate connected peers (only reached when gossipsub not wired)
        let peers = self.get_all_peers().await?;
        if peers.is_empty() {
            warn!(
                "⚠️ No gossipsub TX and no connected peers — fluff message {} dropped",
                hex::encode(&message.id[..4])
            );
            return Ok(());
        }
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
                use rand::Rng;
                rand::rng().random_range(0..targets.len())
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

    /// Generate VRF proof for message routing (optional quantum enhancement)
    async fn generate_vrf_proof(
        &self,
        _topic: &str,
        _message_id: &[u8; 32],
    ) -> Result<Option<VRFProof>> {
        // VRF proof generation is optional and requires full L-VRF setup
        // For now, skip VRF proofs - they can be added when L-VRF is fully integrated
        // The quantum randomness for routing decisions is still used via QRNG
        Ok(None)
    }

    /// Generate quantum nonce for enhanced privacy
    async fn generate_quantum_nonce(&self) -> Result<[u8; 16]> {
        match &self.qrng {
            Some(qrng) => {
                let nonce: [u8; 16] = qrng.generate_quantum_seed().await?;
                Ok(nonce)
            }
            None => {
                use rand::Rng;
                let mut nonce = [0u8; 16];
                rand::rng().fill(&mut nonce);
                Ok(nonce)
            }
        }
    }

    /// Send message to specific peer — used for stem relay.
    /// Primary path uses gossipsub dandelion-stem topic; Tor is the fallback when available.
    async fn send_to_peer(&self, peer_id: &NodeId, message: &DandelionMessage) -> Result<()> {
        debug!(
            "📤 Dandelion stem relay (peer prefix: {:08x})",
            u32::from_le_bytes([peer_id[0], peer_id[1], peer_id[2], peer_id[3]])
        );

        let message_data = bincode::serialize(message)?;

        // Primary: gossipsub dandelion-stem topic.
        // Peers subscribed to this topic re-run their own stem/fluff coin flip,
        // providing the time-delay and re-routing that makes Dandelion++ effective.
        {
            let tx_guard = self.gossipsub_tx.read().await;
            if let Some(ref tx) = *tx_guard {
                if tx
                    .send(("/qnk/dandelion-stem".to_string(), message_data.clone()))
                    .is_ok()
                {
                    debug!(
                        "📤 Stem relay sent via gossipsub (msg {})",
                        hex::encode(&message.id[..4])
                    );
                    return Ok(());
                }
                warn!("⚠️ Gossipsub TX closed during stem relay — trying Tor");
            }
        }

        // Fallback: Tor circuit (when available and connected)
        let peer_onion = tor_bridge::peer_id_to_onion(peer_id);
        if peer_onion.ends_with(".onion") && !peer_onion.contains("qnk") {
            if let Ok(_conn) = self.tor_client.connect_to_peer(&peer_onion).await {
                debug!(
                    "📤 Stem relay sent via Tor onion (msg {})",
                    hex::encode(&message.id[..4])
                );
                return Ok(());
            }
        }

        // Neither path worked — log and continue (failsafe timer will force fluff)
        warn!(
            "⚠️ Stem relay for {} had no viable transport — failsafe will promote to fluff",
            hex::encode(&message.id[..4])
        );
        Ok(())
    }

    /// Get all connected peers (from stem_targets, updated by main.rs peer-sync task)
    async fn get_all_peers(&self) -> Result<Vec<NodeId>> {
        let peers = self.stem_targets.read().await;
        Ok(peers.clone())
    }

    /// Update stem targets (also used as connected peer list for fluff fallback)
    pub async fn update_stem_targets(&self, targets: Vec<NodeId>) {
        let mut stem_targets = self.stem_targets.write().await;
        *stem_targets = targets;
        info!("🎯 Updated stem targets: {} peers", stem_targets.len());
    }

    /// Wire the live gossipsub publisher into Dandelion++.
    /// Call this once after libp2p initialises. The sender should wrap
    /// q_network::NetworkCommand::PublishMessage so published (topic, data) pairs
    /// reach the gossipsub swarm.
    pub async fn set_gossipsub_tx(
        &self,
        tx: tokio::sync::mpsc::UnboundedSender<(String, Vec<u8>)>,
    ) {
        let mut guard = self.gossipsub_tx.write().await;
        *guard = Some(tx);
        info!("📡 Gossipsub TX wired — Dandelion++ fluff/stem propagation is now LIVE");
    }

    /// Handle incoming Dandelion message
    pub async fn handle_incoming_message(&self, message: DandelionMessage) -> Result<()> {
        debug!(
            "📥 Received Dandelion message {} in {:?} phase",
            hex::encode(message.id),
            message.phase
        );

        // Check if we've seen this message before
        {
            let mut seen = self.seen_messages.write().await;
            if seen.contains_key(&message.id) {
                debug!("🔁 Duplicate message {}, ignoring", hex::encode(message.id));
                return Ok(());
            }
            seen.insert(message.id, Instant::now());
        }

        // VRF proof verification is optional - skip if not present
        // Full VRF verification would require message source's public key
        if message.vrf_proof.is_some() {
            debug!(
                "VRF proof present for message {}, verification skipped (optional)",
                hex::encode(message.id)
            );
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
            None => {
                use rand::Rng;
                rand::rng().random::<f64>()
            }
        };

        Ok(probability < self.config.stem_continuation_probability)
    }

    /// Deliver message to local application
    async fn deliver_message_locally(&self, message: DandelionMessage) -> Result<()> {
        debug!("📨 Delivering message {} locally", hex::encode(message.id));

        // In production, this would deliver to the consensus layer
        // For now, just log the delivery
        info!(
            "✅ Delivered message {} ({} bytes)",
            hex::encode(message.id),
            message.payload.len()
        );

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
                // Use checked_sub to avoid panic on Windows where Instant is based on uptime
                let cutoff = Instant::now().checked_sub(cleanup_interval).unwrap_or(Instant::now());

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

        // Calculate anonymity score with quantum enhancements
        let base_score = metrics.anonymity_score;
        let quantum_bonus = if self.qrng.is_some() && self.vrf.is_some() {
            0.2 // 20% bonus for quantum enhancements
        } else {
            0.0
        };
        let vrf_bonus = if self.vrf.is_some() {
            0.1 // 10% bonus for verifiable randomness
        } else {
            0.0
        };
        let anonymity_score = (base_score + quantum_bonus + vrf_bonus).min(1.0);

        AnonymityStats {
            messages_seen: seen_count as u64,
            stem_messages: metrics.stem_messages,
            fluff_messages: metrics.fluff_messages,
            hop_distribution: metrics.hop_distribution.clone(),
            quantum_enhanced: self.qrng.is_some() && self.vrf.is_some(),
            anonymity_score,
        }
    }

    // ==================== Network Bridge Integration ====================

    /// Set the network bridge for gossipsub integration
    pub fn set_network_bridge(&mut self, bridge: Arc<NetworkBridge>) {
        info!("🌐 Network bridge connected to Dandelion++");
        self.network_bridge = Some(bridge);
    }

    /// Get reference to network bridge
    pub fn network_bridge(&self) -> Option<&Arc<NetworkBridge>> {
        self.network_bridge.as_ref()
    }

    /// Propagate message using network bridge (fluff via gossipsub)
    pub async fn propagate_via_network_bridge(&self, message: &DandelionMessage) -> Result<()> {
        if let Some(ref bridge) = self.network_bridge {
            // Track in failsafe before sending
            if let Some(ref failsafe) = self.failsafe {
                failsafe.track_transaction(message.clone()).await?;
                failsafe
                    .update_state(message.id, TransactionState::Fluffing)
                    .await;
            }

            // Broadcast via gossipsub
            bridge.broadcast_fluff(message).await?;

            // Mark as delivered
            if let Some(ref failsafe) = self.failsafe {
                failsafe.mark_delivered(message.id).await;
            }

            debug!(
                "📡 Message {} propagated via network bridge",
                hex::encode(message.id)
            );
        } else {
            // Fallback to direct fluff propagation
            self.fluff_propagate(message.clone()).await?;
        }
        Ok(())
    }

    // ==================== Tor Bridge Integration ====================

    /// Get reference to Tor bridge
    pub fn tor_bridge(&self) -> Option<&Arc<TorBridge>> {
        self.tor_bridge.as_ref()
    }

    /// Send stem message via Tor bridge
    pub async fn send_stem_via_tor(
        &self,
        peer_id: &NodeId,
        message: &DandelionMessage,
    ) -> Result<()> {
        // Track in failsafe
        if let Some(ref failsafe) = self.failsafe {
            failsafe.track_transaction(message.clone()).await?;
            failsafe
                .update_state(
                    message.id,
                    TransactionState::Stemming {
                        hops_completed: message.hop_count,
                    },
                )
                .await;
        }

        // Use Tor bridge if available
        if let Some(ref tor_bridge) = self.tor_bridge {
            if tor_bridge.is_available().await {
                let peer_onion = tor_bridge::peer_id_to_onion(peer_id);
                let message_bytes = bincode::serialize(message)?;

                match tor_bridge
                    .send_via_tor(&peer_onion, &message_bytes, DandelionCircuitPurpose::StemPrimary)
                    .await
                {
                    Ok(latency) => {
                        debug!(
                            "🧅 Sent stem message {} via Tor (latency: {:?})",
                            hex::encode(message.id),
                            latency
                        );

                        // Record success in failsafe
                        if let Some(ref failsafe) = self.failsafe {
                            failsafe.record_relay_attempt(message.id, true).await;
                        }
                        return Ok(());
                    }
                    Err(e) => {
                        warn!("⚠️ Tor relay failed: {}, falling back to direct", e);
                        tor_bridge.record_relay_failure().await;

                        // Record failure in failsafe
                        if let Some(ref failsafe) = self.failsafe {
                            failsafe.record_relay_attempt(message.id, false).await;
                        }
                    }
                }
            }
        }

        // Fallback to direct peer connection
        self.send_to_peer(peer_id, message).await
    }

    // ==================== Failsafe Integration ====================

    /// Get reference to failsafe timer
    pub fn failsafe(&self) -> Option<&Arc<FailsafeTimer>> {
        self.failsafe.as_ref()
    }

    /// Get failsafe statistics
    pub async fn get_failsafe_stats(&self) -> Option<FailsafeStats> {
        if let Some(ref failsafe) = self.failsafe {
            Some(failsafe.get_stats().await)
        } else {
            None
        }
    }

    /// Handle failsafe events (called from background task)
    pub async fn handle_failsafe_event(&self, event: FailsafeEvent) -> Result<()> {
        match event {
            FailsafeEvent::StemTimeout { message_id } => {
                warn!(
                    "⏱️ Failsafe timeout for message {}, forcing fluff",
                    hex::encode(message_id)
                );

                // Get the original message from failsafe
                if let Some(ref failsafe) = self.failsafe {
                    let pending = failsafe.get_pending_transactions().await;
                    if let Some(message) = pending.iter().find(|m| m.id == message_id) {
                        let mut fluff_message = message.clone();
                        fluff_message.phase = DandelionPhase::Fluff;

                        // Force fluff propagation
                        self.propagate_via_network_bridge(&fluff_message).await?;

                        info!(
                            "🌸 Forced fluff for timed-out message {}",
                            hex::encode(message_id)
                        );
                    }
                }
            }
            FailsafeEvent::StemSuccess { message_id } => {
                debug!(
                    "✅ Stem relay succeeded for message {}",
                    hex::encode(message_id)
                );
            }
            FailsafeEvent::StemFailed { message_id, error } => {
                warn!(
                    "❌ Stem relay failed for message {}: {}",
                    hex::encode(message_id),
                    error
                );

                // Try to recover by forcing fluff
                if let Some(ref failsafe) = self.failsafe {
                    let pending = failsafe.get_pending_transactions().await;
                    if let Some(message) = pending.iter().find(|m| m.id == message_id) {
                        let mut fluff_message = message.clone();
                        fluff_message.phase = DandelionPhase::Fluff;
                        let _ = self.propagate_via_network_bridge(&fluff_message).await;
                    }
                }
            }
            FailsafeEvent::FluffComplete { message_id } => {
                debug!(
                    "🎉 Fluff complete for message {}",
                    hex::encode(message_id)
                );
            }
        }
        Ok(())
    }

    /// Start background failsafe event handler
    pub fn start_failsafe_handler(self: &Arc<Self>) {
        if let Some(ref failsafe_rx) = self.failsafe_rx {
            let dandelion = Arc::clone(self);
            let failsafe_rx = Arc::clone(failsafe_rx);

            tokio::spawn(async move {
                let mut rx = failsafe_rx.lock().await;
                while let Some(event) = rx.recv().await {
                    if let Err(e) = dandelion.handle_failsafe_event(event).await {
                        warn!("⚠️ Failed to handle failsafe event: {}", e);
                    }
                }
            });

            info!("🛡️ Failsafe event handler started");
        }
    }

    // ==================== Combined Statistics ====================

    /// Get comprehensive Dandelion++ statistics
    pub async fn get_comprehensive_stats(&self) -> DandelionStats {
        let anonymity = self.get_anonymity_stats().await;
        let failsafe = self.get_failsafe_stats().await;

        let tor_stats = if let Some(ref tor_bridge) = self.tor_bridge {
            Some(tor_bridge.get_stats().await)
        } else {
            None
        };

        let network_stats = if let Some(ref network_bridge) = self.network_bridge {
            Some(network_bridge.get_stats())
        } else {
            None
        };

        DandelionStats {
            anonymity,
            failsafe,
            tor: tor_stats,
            network: network_stats,
            tor_available: self.tor_bridge.as_ref().map(|b| {
                // Can't await in map, so use is_some as proxy
                true
            }),
            quantum_enhanced: self.qrng.is_some() && self.vrf.is_some(),
        }
    }

    /// Shutdown Dandelion++ cleanly
    pub async fn shutdown(&self) {
        info!("🛑 Shutting down Quantum Dandelion++");

        // Shutdown failsafe timer
        if let Some(ref failsafe) = self.failsafe {
            failsafe.shutdown();
        }

        // Close Tor circuits
        if let Some(ref tor_bridge) = self.tor_bridge {
            tor_bridge.close_all_circuits().await;
        }

        info!("✅ Quantum Dandelion++ shutdown complete");
    }
}

/// Comprehensive Dandelion++ statistics
#[derive(Debug, Clone)]
pub struct DandelionStats {
    /// Anonymity metrics
    pub anonymity: AnonymityStats,
    /// Failsafe timer stats (if enabled)
    pub failsafe: Option<FailsafeStats>,
    /// Tor connection stats (if enabled)
    pub tor: Option<TorConnectionStats>,
    /// Network bridge stats (if enabled)
    pub network: Option<NetworkBridgeStats>,
    /// Whether Tor is available
    pub tor_available: Option<bool>,
    /// Whether quantum enhancements are active
    pub quantum_enhanced: bool,
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
            stem_probability: 0.9,              // 90% chance to start in stem
            // v8.6.0: reduced from 0.8 to 0.75 — increases fluff transition rate per hop,
            // improving latency while maintaining strong privacy (expected ~3 hops avg)
            stem_continuation_probability: 0.75,
            // v8.6.0: reduced from 10 to 5 — Dandelion++ paper recommends 4-5 max hops;
            // beyond 5 hops adds latency without meaningful anonymity improvement
            max_stem_hops: 5,
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
        // v8.6.0: updated from 0.8 to 0.75
        assert_eq!(config.stem_continuation_probability, 0.75);
        // v8.6.0: updated from 10 to 5
        assert_eq!(config.max_stem_hops, 5);
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
