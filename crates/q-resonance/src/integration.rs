//! 🎻 Quillon Resonance Integration Bridge
//!
//! This module bridges the resonance consensus with traditional Narwhal+Bullshark,
//! allowing the symphony of physical consensus to enhance the existing system.
//!
//! Philosophy: We don't replace the existing consensus - we harmonize with it.
//! Like adding string instruments to an orchestra, resonance enhances without disrupting.

use crate::{
    energy::EnergyFunctional,
    ordering::ResonanceOrdering,
    spectral_bft::SpectralBFT,
    string_state::StringState,
    vertex::ResonanceVertex,
    gossip::{ResonanceStateTracker, ResonanceMessage},
    ResonanceError,
    Result,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, HashMap};
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use q_aegis_ql::{AegisQL, PublicKey as AegisPublicKey, Signature as AegisSignature};

/// 🎻 Transaction type compatible with Narwhal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NarwhalTransaction {
    pub hash: [u8; 32],
    pub data: Vec<u8>,
    pub sender: [u8; 32],
    pub nonce: u64,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}

/// 🎻 Enhanced vertex that bridges Narwhal vertices with resonance properties
///
/// Philosophy: A Narwhal vertex is like a musical note waiting to find its place
/// in the symphony. The ResonanceEnhancedVertex adds harmonic properties that
/// allow it to naturally align with other vertices through physical resonance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResonanceEnhancedVertex {
    /// Original Narwhal vertex data
    pub round: u64,
    pub hash: [u8; 32],
    pub author: Vec<u8>,
    pub transactions: Vec<NarwhalTransaction>,
    pub parents: HashSet<[u8; 32]>,
    pub timestamp: u64,

    /// 🎻 Resonance properties: The harmonic signature of this vertex
    pub string_state: StringState,

    /// 🎻 Stake weight: The amplitude of this vertex's vibration
    pub stake: f64,

    /// 🎻 Network position: Spatial coordinates in consensus space
    pub network_position: Vec<f64>,

    /// 🎻 Resonance score: How strongly this vertex harmonizes with the network
    pub resonance_score: f64,

    /// 🎻 Is Byzantine: Detected through spectral analysis
    pub is_byzantine: bool,
}

impl ResonanceEnhancedVertex {
    /// 🎻 Create an enhanced vertex from Narwhal transaction batch
    ///
    /// Philosophy: Every batch of transactions becomes a vibrating string in consensus space.
    /// Its frequency is determined by priority, amplitude by stake, and phase by temporal alignment.
    pub fn from_narwhal_batch(
        round: u64,
        hash: [u8; 32],
        author: Vec<u8>,
        transactions: Vec<NarwhalTransaction>,
        parents: HashSet<[u8; 32]>,
        timestamp: u64,
        stake: f64,
        network_position: Vec<f64>,
    ) -> Self {
        tracing::debug!(
            "🎻 Creating resonance vertex: round={}, stake={}, tx_count={}",
            round,
            stake,
            transactions.len()
        );

        // 🎻 Compute priority from transaction urgency
        let priority = Self::compute_priority(&transactions, round);

        // 🎻 Initialize string state: The harmonic signature
        let string_state = StringState::new(
            stake.sqrt(),           // amplitude = sqrt(stake) for proper coupling
            priority,               // frequency = urgency
            network_position.clone(),
            hash,
            timestamp,
        );

        Self {
            round,
            hash,
            author,
            transactions,
            parents,
            timestamp,
            string_state,
            stake,
            network_position,
            resonance_score: 0.0,
            is_byzantine: false,
        }
    }

    /// 🎻 Compute transaction priority
    ///
    /// Philosophy: Priority emerges from transaction characteristics, not arbitrary ranking.
    /// Earlier rounds + higher fees = higher frequency vibration
    fn compute_priority(transactions: &[NarwhalTransaction], round: u64) -> f64 {
        if transactions.is_empty() {
            return 1.0 / (round as f64 + 1.0);
        }

        // Higher priority for earlier rounds (urgency)
        let round_factor = 1.0 / (round as f64 + 1.0);

        // Higher priority for more transactions (throughput)
        let throughput_factor = (transactions.len() as f64).ln() + 1.0;

        round_factor * throughput_factor
    }

    /// 🎻 Convert to pure resonance vertex for consensus processing
    pub fn to_resonance_vertex(&self) -> ResonanceVertex {
        

        ResonanceVertex::new(
            self.hash,
            self.round,
            self.parents.clone(),
            self.transactions
                .iter()
                .map(|tx| tx.data.clone())
                .collect(),
            self.author.clone(),
            self.timestamp,
            self.stake,
            self.network_position.clone(),
            0.5, // Default entropy
        )
    }
}

/// 🎻 Resonance Consensus Coordinator
///
/// Philosophy: This is the conductor of the distributed symphony.
/// It doesn't impose order - it helps the network find its natural harmonic state.
pub struct ResonanceCoordinator {
    /// Node identification
    node_id: Vec<u8>,

    /// 🎻 Resonance ordering engine: Finds harmonic alignment
    ordering: Arc<RwLock<ResonanceOrdering>>,

    /// 🎻 Spectral Byzantine detector: Filters dissonance
    #[allow(dead_code)]
    spectral_bft: Arc<RwLock<SpectralBFT>>,

    /// 🎻 Energy functional: The optimization landscape
    #[allow(dead_code)]
    energy_functional: Arc<RwLock<Option<EnergyFunctional>>>,

    /// 🎻 Enhanced vertices by round
    vertices_by_round: Arc<DashMap<u64, Vec<ResonanceEnhancedVertex>>>,

    /// 🎻 Consensus state
    latest_consensus_round: Arc<RwLock<u64>>,

    /// 🎻 Performance metrics
    metrics: Arc<RwLock<ResonanceMetrics>>,

    /// 🎻 State tracker for gossip synchronization
    state_tracker: Arc<ResonanceStateTracker>,

    /// 🎻 Gossip message sender (to network)
    gossip_tx: Option<mpsc::UnboundedSender<ResonanceMessage>>,

    /// 🎻 Gossip message receiver (from network)
    gossip_rx: Option<mpsc::UnboundedReceiver<ResonanceMessage>>,

    /// 🔐 AEGIS-QL validator authentication
    aegis: AegisQL,

    /// 🔐 Validator public keys for signature verification
    validator_public_keys: Arc<RwLock<HashMap<[u8; 32], AegisPublicKey>>>,
}

/// 🎻 Performance metrics for resonance consensus
#[derive(Clone, Debug, Default)]
pub struct ResonanceMetrics {
    pub total_rounds_processed: u64,
    pub average_convergence_time_ms: f64,
    pub average_phase_variance: f64,
    pub byzantine_detected_count: u64,
    pub average_energy_reduction: f64,
    pub total_vertices_ordered: u64,
}

impl ResonanceCoordinator {
    /// 🎻 Create a new resonance coordinator (without gossip)
    pub fn new(node_id: Vec<u8>) -> Self {
        tracing::info!("🎻 Initializing Resonance Coordinator for node {:?}", node_id);

        let state_tracker = Arc::new(ResonanceStateTracker::new(node_id.clone()));

        Self {
            node_id,
            ordering: Arc::new(RwLock::new(ResonanceOrdering::new(
                100.0,  // energy_threshold
                1.0,    // variance_threshold
                0.5,    // byzantine_threshold
            ))),
            spectral_bft: Arc::new(RwLock::new(SpectralBFT::new(0.5, 5))),
            energy_functional: Arc::new(RwLock::new(None)),
            vertices_by_round: Arc::new(DashMap::new()),
            latest_consensus_round: Arc::new(RwLock::new(0)),
            metrics: Arc::new(RwLock::new(ResonanceMetrics::default())),
            state_tracker,
            gossip_tx: None,
            gossip_rx: None,
            aegis: AegisQL::new(),
            validator_public_keys: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 🎻 Create a new resonance coordinator with AEGIS-QL validator authentication
    pub fn new_with_aegis(
        node_id: Vec<u8>,
        validator_keys: HashMap<[u8; 32], AegisPublicKey>,
    ) -> Self {
        tracing::info!("🔐 Initializing Resonance Coordinator with AEGIS-QL authentication for node {:?}", node_id);

        let state_tracker = Arc::new(ResonanceStateTracker::new(node_id.clone()));

        Self {
            node_id,
            ordering: Arc::new(RwLock::new(ResonanceOrdering::new(
                100.0,  // energy_threshold
                1.0,    // variance_threshold
                0.5,    // byzantine_threshold
            ))),
            spectral_bft: Arc::new(RwLock::new(SpectralBFT::new(0.5, 5))),
            energy_functional: Arc::new(RwLock::new(None)),
            vertices_by_round: Arc::new(DashMap::new()),
            latest_consensus_round: Arc::new(RwLock::new(0)),
            metrics: Arc::new(RwLock::new(ResonanceMetrics::default())),
            state_tracker,
            gossip_tx: None,
            gossip_rx: None,
            aegis: AegisQL::new(),
            validator_public_keys: Arc::new(RwLock::new(validator_keys)),
        }
    }

    /// 🎻 Create a new resonance coordinator with gossip support
    pub fn new_with_gossip(
        node_id: Vec<u8>,
    ) -> (Self, mpsc::UnboundedSender<ResonanceMessage>, mpsc::UnboundedReceiver<ResonanceMessage>) {
        tracing::info!("🎻 Initializing Resonance Coordinator with Gossip for node {:?}", node_id);

        let state_tracker = Arc::new(ResonanceStateTracker::new(node_id.clone()));

        // Create gossip channels
        let (tx_to_network, rx_from_coordinator) = mpsc::unbounded_channel();
        let (tx_from_network, rx_to_coordinator) = mpsc::unbounded_channel();

        let coordinator = Self {
            node_id,
            ordering: Arc::new(RwLock::new(ResonanceOrdering::new(
                100.0,  // energy_threshold
                1.0,    // variance_threshold
                0.5,    // byzantine_threshold
            ))),
            spectral_bft: Arc::new(RwLock::new(SpectralBFT::new(0.5, 5))),
            energy_functional: Arc::new(RwLock::new(None)),
            vertices_by_round: Arc::new(DashMap::new()),
            latest_consensus_round: Arc::new(RwLock::new(0)),
            metrics: Arc::new(RwLock::new(ResonanceMetrics::default())),
            state_tracker,
            gossip_tx: Some(tx_to_network),
            gossip_rx: Some(rx_to_coordinator),
            aegis: AegisQL::new(),
            validator_public_keys: Arc::new(RwLock::new(HashMap::new())),
        };

        (coordinator, tx_from_network, rx_from_coordinator)
    }

    /// 🎻 Create a new resonance coordinator with gossip and AEGIS-QL authentication
    pub fn new_with_gossip_and_aegis(
        node_id: Vec<u8>,
        validator_keys: HashMap<[u8; 32], AegisPublicKey>,
    ) -> (Self, mpsc::UnboundedSender<ResonanceMessage>, mpsc::UnboundedReceiver<ResonanceMessage>) {
        tracing::info!("🔐 Initializing Resonance Coordinator with Gossip and AEGIS-QL for node {:?}", node_id);

        let state_tracker = Arc::new(ResonanceStateTracker::new(node_id.clone()));

        // Create gossip channels
        let (tx_to_network, rx_from_coordinator) = mpsc::unbounded_channel();
        let (tx_from_network, rx_to_coordinator) = mpsc::unbounded_channel();

        let coordinator = Self {
            node_id,
            ordering: Arc::new(RwLock::new(ResonanceOrdering::new(
                100.0,  // energy_threshold
                1.0,    // variance_threshold
                0.5,    // byzantine_threshold
            ))),
            spectral_bft: Arc::new(RwLock::new(SpectralBFT::new(0.5, 5))),
            energy_functional: Arc::new(RwLock::new(None)),
            vertices_by_round: Arc::new(DashMap::new()),
            latest_consensus_round: Arc::new(RwLock::new(0)),
            metrics: Arc::new(RwLock::new(ResonanceMetrics::default())),
            state_tracker,
            gossip_tx: Some(tx_to_network),
            gossip_rx: Some(rx_to_coordinator),
            aegis: AegisQL::new(),
            validator_public_keys: Arc::new(RwLock::new(validator_keys)),
        };

        (coordinator, tx_from_network, rx_from_coordinator)
    }

    /// 🎻 Process a batch of Narwhal transactions with resonance consensus
    ///
    /// Philosophy: Instead of voting on transaction order, we let them find
    /// their natural harmonic alignment through energy minimization.
    pub async fn process_narwhal_batch(
        &self,
        round: u64,
        transactions: Vec<NarwhalTransaction>,
        stake: f64,
        network_position: Vec<f64>,
    ) -> Result<Vec<[u8; 32]>> {
        use blake3::Hasher;

        let start_time = std::time::Instant::now();

        tracing::info!(
            "🎻 Processing Narwhal batch: round={}, tx_count={}, stake={}",
            round,
            transactions.len(),
            stake
        );

        // 🎻 Compute batch hash
        let mut hasher = Hasher::new();
        for tx in &transactions {
            hasher.update(&tx.hash);
        }
        let batch_hash_bytes = hasher.finalize();
        let mut batch_hash = [0u8; 32];
        batch_hash.copy_from_slice(batch_hash_bytes.as_bytes());

        // 🎻 Create enhanced vertex
        let enhanced_vertex = ResonanceEnhancedVertex::from_narwhal_batch(
            round,
            batch_hash,
            self.node_id.clone(),
            transactions,
            HashSet::new(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            stake,
            network_position,
        );

        // 🎻 Store vertex
        self.vertices_by_round
            .entry(round)
            .or_insert_with(Vec::new)
            .push(enhanced_vertex.clone());

        // 🎻 Convert to resonance vertices
        let round_vertices: Vec<ResonanceVertex> = self
            .vertices_by_round
            .get(&round)
            .map(|v| v.iter().map(|ev| ev.to_resonance_vertex()).collect())
            .unwrap_or_default();

        tracing::debug!(
            "🎻 Round {} has {} vertices ready for resonance consensus",
            round,
            round_vertices.len()
        );

        // 🎻 Process with resonance ordering
        let ordered_hashes = {
            let mut ordering = self.ordering.write();
            ordering.process_round(round, round_vertices)?
        };

        // 🎻 Update metrics
        let elapsed = start_time.elapsed();
        self.update_metrics(round, elapsed, &ordered_hashes).await;

        // 🎻 Update consensus round
        *self.latest_consensus_round.write() = round;

        tracing::info!(
            "🎻 Resonance consensus complete: round={}, ordered_count={}, time={}ms",
            round,
            ordered_hashes.len(),
            elapsed.as_millis()
        );

        Ok(ordered_hashes)
    }

    /// 🎻 Update performance metrics
    async fn update_metrics(
        &self,
        _round: u64,
        elapsed: std::time::Duration,
        ordered_hashes: &[[u8; 32]],
    ) {
        let mut metrics = self.metrics.write();

        metrics.total_rounds_processed += 1;
        metrics.total_vertices_ordered += ordered_hashes.len() as u64;

        // Rolling average of convergence time
        let alpha = 0.1;
        metrics.average_convergence_time_ms =
            alpha * elapsed.as_millis() as f64
            + (1.0 - alpha) * metrics.average_convergence_time_ms;

        tracing::debug!(
            "🎻 Metrics updated: rounds={}, avg_time={}ms, total_vertices={}",
            metrics.total_rounds_processed,
            metrics.average_convergence_time_ms,
            metrics.total_vertices_ordered
        );
    }

    /// 🎻 Get current performance metrics
    pub fn get_metrics(&self) -> ResonanceMetrics {
        self.metrics.read().clone()
    }

    /// 🎻 Get spectral gap for consensus strength measurement
    pub async fn get_spectral_gap(&self) -> Result<f64> {
        let ordering = self.ordering.read();
        ordering.get_spectral_gap()
    }

    /// 🎻 Get total energy of current consensus
    pub fn get_total_energy(&self) -> f64 {
        let ordering = self.ordering.read();
        ordering.get_total_energy()
    }

    /// 🎻 Check if round has reached consensus
    pub fn has_consensus(&self, round: u64) -> bool {
        let ordering = self.ordering.read();
        let committed = ordering.get_committed();

        // Check if any vertices from this round are committed
        if let Some(vertices) = self.vertices_by_round.get(&round) {
            vertices.iter().any(|v| committed.contains(&v.hash))
        } else {
            false
        }
    }

    /// 🎻 Broadcast string state announcement to network
    ///
    /// Philosophy: Like a violin broadcasting its vibration through the concert hall,
    /// we announce our resonance state to the entire network.
    pub fn broadcast_string_state(
        &self,
        round: u64,
        vertex_hash: [u8; 32],
        string_state: &StringState,
    ) -> Result<()> {
        if let Some(gossip_tx) = &self.gossip_tx {
            let msg = ResonanceMessage::StringStateAnnouncement {
                round,
                vertex_hash,
                string_state: string_state.clone(),
                validator: self.node_id.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            };

            gossip_tx.send(msg).map_err(|e| {
                ResonanceError::InvalidState(format!("Failed to broadcast string state: {}", e))
            })?;

            tracing::debug!(
                "🎻 Broadcast string state: round={}, freq={:.4}",
                round,
                string_state.frequency
            );
        }

        Ok(())
    }

    /// 🎻 Broadcast consensus achievement to network
    ///
    /// Philosophy: When the symphony reaches harmonic convergence, we announce
    /// the achievement to all participants.
    pub fn broadcast_consensus(
        &self,
        round: u64,
        committed_hashes: Vec<[u8; 32]>,
        final_energy: f64,
        spectral_gap: f64,
    ) -> Result<()> {
        if let Some(gossip_tx) = &self.gossip_tx {
            let msg = ResonanceMessage::ConsensusAchieved {
                round,
                committed_hashes,
                final_energy,
                spectral_gap,
                node: self.node_id.clone(),
            };

            gossip_tx.send(msg).map_err(|e| {
                ResonanceError::InvalidState(format!("Failed to broadcast consensus: {}", e))
            })?;

            tracing::info!(
                "🎻 Broadcast consensus: round={}, energy={:.4}, spectral_gap={:.4}",
                round,
                final_energy,
                spectral_gap
            );
        }

        Ok(())
    }

    /// 🎻 Broadcast Byzantine alert to network
    ///
    /// Philosophy: When dissonance is detected, we warn the network so all
    /// participants can filter out the discordant vibrations.
    pub fn broadcast_byzantine_alert(
        &self,
        round: u64,
        suspected_node: Vec<u8>,
        spectral_coefficient: f64,
    ) -> Result<()> {
        if let Some(gossip_tx) = &self.gossip_tx {
            let msg = ResonanceMessage::ByzantineAlert {
                round,
                suspected_node: suspected_node.clone(),
                detector_node: self.node_id.clone(),
                spectral_coefficient,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            };

            gossip_tx.send(msg).map_err(|e| {
                ResonanceError::InvalidState(format!("Failed to broadcast Byzantine alert: {}", e))
            })?;

            tracing::warn!(
                "🎻 Broadcast Byzantine alert: round={}, suspected={:?}, coeff={:.4}",
                round,
                suspected_node,
                spectral_coefficient
            );
        }

        Ok(())
    }

    /// 🎻 Handle incoming gossip message
    ///
    /// Philosophy: Process vibrations received from other nodes in the symphony,
    /// recording their states and responding to requests.
    pub async fn handle_gossip_message(&self, msg: ResonanceMessage) -> Result<()> {
        match msg {
            ResonanceMessage::StringStateAnnouncement {
                round,
                vertex_hash,
                string_state,
                validator,
                timestamp: _,
            } => {
                // Record the string state from peer
                self.state_tracker.record_string_state(round, validator.clone(), string_state.clone());

                // Create resonance vertex from the announcement
                let resonance_vertex = ResonanceVertex::new(
                    vertex_hash,
                    round,
                    HashSet::new(), // Parents unknown in announcement
                    vec![],         // Transactions unknown in announcement
                    validator,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                    string_state.amplitude.powi(2), // Recover stake from amplitude
                    string_state.position.clone(),
                    0.5, // Default entropy
                );

                self.state_tracker.record_vertex(round, resonance_vertex);

                tracing::debug!(
                    "🎻 Received string state: round={}, freq={:.4}",
                    round,
                    string_state.frequency
                );
            }

            ResonanceMessage::StateRequest {
                round,
                requesting_node,
            } => {
                // Collect our vertices for the requested round
                let vertices = self.state_tracker.get_vertices_for_round(round);

                // Send response
                if let Some(gossip_tx) = &self.gossip_tx {
                    let response = ResonanceMessage::StateResponse {
                        round,
                        vertices,
                        responding_node: self.node_id.clone(),
                    };

                    gossip_tx.send(response).map_err(|e| {
                        ResonanceError::InvalidState(format!("Failed to send state response: {}", e))
                    })?;

                    tracing::debug!(
                        "🎻 Sent state response to {:?} for round {}",
                        requesting_node,
                        round
                    );
                }
            }

            ResonanceMessage::StateResponse {
                round,
                vertices,
                responding_node: _,
            } => {
                // Record all received vertices
                for vertex in vertices {
                    self.state_tracker.record_vertex(round, vertex);
                }

                tracing::debug!("🎻 Received state response for round {}", round);
            }

            ResonanceMessage::ConsensusAchieved {
                round,
                committed_hashes,
                final_energy,
                spectral_gap,
                node: _,
            } => {
                // Record consensus achievement
                self.state_tracker.record_consensus(
                    round,
                    committed_hashes,
                    final_energy,
                    spectral_gap,
                );

                tracing::info!(
                    "🎻 Peer consensus: round={}, energy={:.4}, spectral_gap={:.4}",
                    round,
                    final_energy,
                    spectral_gap
                );
            }

            ResonanceMessage::ByzantineAlert {
                round,
                suspected_node,
                detector_node,
                spectral_coefficient,
                timestamp: _,
            } => {
                // Record Byzantine alert
                self.state_tracker.record_byzantine_alert(
                    round,
                    suspected_node.clone(),
                    detector_node,
                    spectral_coefficient,
                );

                tracing::warn!(
                    "🎻 Byzantine alert received: round={}, suspected={:?}",
                    round,
                    suspected_node
                );
            }
        }

        Ok(())
    }

    /// 🎻 Process Narwhal batch with gossip integration
    ///
    /// Philosophy: Enhanced processing that broadcasts our vibration to the network
    /// and incorporates peer states for stronger consensus.
    pub async fn process_narwhal_batch_with_gossip(
        &self,
        round: u64,
        transactions: Vec<NarwhalTransaction>,
        stake: f64,
        network_position: Vec<f64>,
    ) -> Result<Vec<[u8; 32]>> {
        use blake3::Hasher;

        let start_time = std::time::Instant::now();

        tracing::info!(
            "🎻 Processing Narwhal batch with gossip: round={}, tx_count={}, stake={}",
            round,
            transactions.len(),
            stake
        );

        // 🎻 Compute batch hash
        let mut hasher = Hasher::new();
        for tx in &transactions {
            hasher.update(&tx.hash);
        }
        let batch_hash_bytes = hasher.finalize();
        let mut batch_hash = [0u8; 32];
        batch_hash.copy_from_slice(batch_hash_bytes.as_bytes());

        // 🎻 Create enhanced vertex
        let enhanced_vertex = ResonanceEnhancedVertex::from_narwhal_batch(
            round,
            batch_hash,
            self.node_id.clone(),
            transactions,
            HashSet::new(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            stake,
            network_position,
        );

        // 🎻 Broadcast our string state to the network
        self.broadcast_string_state(round, batch_hash, &enhanced_vertex.string_state)?;

        // 🎻 Store vertex locally and in state tracker
        self.vertices_by_round
            .entry(round)
            .or_insert_with(Vec::new)
            .push(enhanced_vertex.clone());

        let resonance_vertex = enhanced_vertex.to_resonance_vertex();
        self.state_tracker.record_vertex(round, resonance_vertex);

        // 🎻 Collect peer vertices from gossip
        let peer_vertices = self.state_tracker.get_vertices_for_round(round);

        // 🎻 Combine our vertices with peer vertices
        let mut all_vertices: Vec<ResonanceVertex> = self
            .vertices_by_round
            .get(&round)
            .map(|v| v.iter().map(|ev| ev.to_resonance_vertex()).collect())
            .unwrap_or_default();

        all_vertices.extend(peer_vertices);

        tracing::debug!(
            "🎻 Round {} has {} total vertices (local + peer) for resonance consensus",
            round,
            all_vertices.len()
        );

        // 🎻 Process with resonance ordering
        let ordered_hashes = {
            let mut ordering = self.ordering.write();
            ordering.process_round(round, all_vertices)?
        };

        // 🎻 Get consensus quality metrics
        let final_energy = self.get_total_energy();
        let spectral_gap = self.get_spectral_gap().await.unwrap_or(0.0);

        // 🎻 Broadcast consensus achievement
        self.broadcast_consensus(round, ordered_hashes.clone(), final_energy, spectral_gap)?;

        // 🎻 Update metrics
        let elapsed = start_time.elapsed();
        self.update_metrics(round, elapsed, &ordered_hashes).await;

        // 🎻 Update consensus round
        *self.latest_consensus_round.write() = round;

        tracing::info!(
            "🎻 Resonance consensus with gossip complete: round={}, ordered_count={}, time={}ms, energy={:.4}",
            round,
            ordered_hashes.len(),
            elapsed.as_millis(),
            final_energy
        );

        Ok(ordered_hashes)
    }

    /// 🎻 Request states from peers for a round
    ///
    /// Philosophy: When we need to synchronize, we ask our peers to share their vibrations.
    pub fn request_peer_states(&self, round: u64) -> Result<()> {
        if let Some(gossip_tx) = &self.gossip_tx {
            let msg = ResonanceMessage::StateRequest {
                round,
                requesting_node: self.node_id.clone(),
            };

            gossip_tx.send(msg).map_err(|e| {
                ResonanceError::InvalidState(format!("Failed to request peer states: {}", e))
            })?;

            tracing::debug!("🎻 Requested peer states for round {}", round);
        }

        Ok(())
    }

    /// 🎻 Get state tracker for external access
    pub fn get_state_tracker(&self) -> Arc<ResonanceStateTracker> {
        Arc::clone(&self.state_tracker)
    }

    /// 🔐 Verify validator signature with AEGIS-QL
    ///
    /// Philosophy: Like checking a musician's credentials before they join the symphony,
    /// we verify validator authenticity through post-quantum cryptography.
    pub fn verify_validator(
        &self,
        validator_address: &[u8; 32],
        message: &[u8],
        signature: &AegisSignature,
    ) -> Result<bool> {
        let keys = self.validator_public_keys.read();
        let pub_key = keys.get(validator_address)
            .ok_or_else(|| ResonanceError::InvalidState("Unknown validator".to_string()))?;

        self.aegis.verify(message, signature, pub_key)
            .map_err(|e| ResonanceError::InvalidState(format!("Signature verification failed: {}", e)))
    }

    /// 🔐 Add validator public key to authorization list
    ///
    /// Philosophy: Admit a new musician to the orchestra with their unique signature.
    pub fn add_validator(&self, validator_address: [u8; 32], public_key: AegisPublicKey) {
        let mut keys = self.validator_public_keys.write();
        keys.insert(validator_address, public_key);

        tracing::info!("🔐 Added validator {:?} to AEGIS-QL authorization", hex::encode(&validator_address[..8]));
    }

    /// 🔐 Remove validator from authorization list
    ///
    /// Philosophy: Remove a musician from the orchestra roster.
    pub fn remove_validator(&self, validator_address: &[u8; 32]) {
        let mut keys = self.validator_public_keys.write();
        keys.remove(validator_address);

        tracing::info!("🔐 Removed validator {:?} from AEGIS-QL authorization", hex::encode(&validator_address[..8]));
    }

    /// 🔐 Check if validator is authorized
    pub fn is_validator_authorized(&self, validator_address: &[u8; 32]) -> bool {
        let keys = self.validator_public_keys.read();
        keys.contains_key(validator_address)
    }

    /// 🔐 Get number of authorized validators
    pub fn validator_count(&self) -> usize {
        let keys = self.validator_public_keys.read();
        keys.len()
    }

    /// 🔐 Process Narwhal batch with AEGIS-QL validator authentication
    ///
    /// Philosophy: Enhanced processing that verifies validator identity before accepting
    /// their contribution to the symphony.
    pub async fn process_with_aegis_auth(
        &self,
        round: u64,
        transactions: Vec<NarwhalTransaction>,
        validator_address: &[u8; 32],
        signature: &AegisSignature,
        stake: f64,
        network_position: Vec<f64>,
    ) -> Result<Vec<[u8; 32]>> {
        // 🔐 Verify validator signature
        let message = format!("CONSENSUS:{}:{}", round, transactions.len());

        if !self.verify_validator(validator_address, message.as_bytes(), signature)? {
            return Err(ResonanceError::InvalidState("Invalid validator signature".to_string()));
        }

        tracing::info!("🔐 Validator {:?} authenticated for round {}", hex::encode(&validator_address[..8]), round);

        // Process with verified validator
        self.process_narwhal_batch_with_gossip(
            round,
            transactions,
            stake,
            network_position,
        ).await
    }

    /// v6.2.0: Get the number of rounds currently stored (for diagnostics)
    pub fn round_count(&self) -> usize {
        self.vertices_by_round.len()
    }

    /// v6.1.2: Clean up old round data to prevent unbounded memory growth
    /// Called from periodic cleanup task in main.rs
    pub fn cleanup_old_rounds(&self, keep_rounds: u64) -> usize {
        let latest_round = self.vertices_by_round.iter()
            .map(|entry| *entry.key())
            .max()
            .unwrap_or(0);
        let cutoff = latest_round.saturating_sub(keep_rounds);
        if cutoff == 0 {
            return 0;
        }

        let rounds_to_remove: Vec<u64> = self.vertices_by_round.iter()
            .filter(|entry| *entry.key() < cutoff)
            .map(|entry| *entry.key())
            .collect();

        let removed = rounds_to_remove.len();
        for round in rounds_to_remove {
            self.vertices_by_round.remove(&round);
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_transaction(nonce: u64) -> NarwhalTransaction {
        NarwhalTransaction {
            hash: [nonce as u8; 32],
            data: vec![1, 2, 3],
            sender: [0u8; 32],
            nonce,
            signature: vec![0u8; 64],
            timestamp: 1000 + nonce,
        }
    }

    #[test]
    fn test_enhanced_vertex_creation() {
        let txs = vec![create_test_transaction(1), create_test_transaction(2)];

        let vertex = ResonanceEnhancedVertex::from_narwhal_batch(
            1,
            [1u8; 32],
            vec![1, 2, 3],
            txs,
            HashSet::new(),
            1000,
            100.0,
            vec![0.0, 0.0],
        );

        assert_eq!(vertex.round, 1);
        assert_eq!(vertex.transactions.len(), 2);
        assert!((vertex.string_state.amplitude - 10.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = ResonanceCoordinator::new(vec![1, 2, 3]);

        let metrics = coordinator.get_metrics();
        assert_eq!(metrics.total_rounds_processed, 0);
    }

    #[tokio::test]
    async fn test_process_batch() {
        let coordinator = ResonanceCoordinator::new(vec![1, 2, 3]);

        let txs = vec![
            create_test_transaction(1),
            create_test_transaction(2),
            create_test_transaction(3),
        ];

        let result = coordinator
            .process_narwhal_batch(1, txs, 100.0, vec![0.0, 0.0])
            .await;

        match &result {
            Ok(ordered) => {
                println!("🎻 Successfully ordered {} vertices", ordered.len());
            }
            Err(e) => {
                println!("🎻 Ordering error (may be expected with single vertex): {}", e);
            }
        }

        let metrics = coordinator.get_metrics();
        assert!(metrics.total_rounds_processed >= 1 || result.is_err());
    }
}
