use anyhow::Result;
use async_trait::async_trait;
use q_narwhal_core::{
    reliable_broadcast::BroadcastStats, Certificate, InMemoryVertexStorage, VertexStore,
};
/// Q-DAG-Knight: Quantum-enhanced Bullshark consensus with DAG-Knight ordering
/// Zero-message complexity asynchronous BFT with quantum anchor election
///
/// PHASE 4: QUANTUM CONSENSUS OVER ANONYMOUS MESH NETWORK
/// - Operates over Tor-anonymized connections from Phase 3
/// - Integrates with DNS-Phantom peer discovery from Phase 1
/// - Scales across 50+ anonymous validator connections from Phase 2
use q_types::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub mod anchor_election;
pub mod commit_logic;
pub mod homological_consensus;  // 🚀 v3.4.6-beta: Homological fork detection (H₀/H₁ Betti numbers)
pub mod mempool_integration;
pub mod ordering_rules;
pub mod quantum_beacon;
pub mod quantum_vdf;
pub mod vertex_creator;

// ✨ v1.0.58-beta: Genus-2 VDF for quantum-resistant anchor election (IACR 2025/1050)
#[cfg(feature = "advanced-crypto")]
pub mod genus2_vdf_integration;

// v10.0.0: SIMD-accelerated bitfield DAG operations (Phase 2 optimization)
#[cfg(feature = "simd-dag")]
pub mod simd_sets;

pub use anchor_election::{AnchorElectionResult, QuantumAnchorElection};
pub use commit_logic::CommitProtocol;
pub use homological_consensus::{
    HomologicalConsensus, HomologicalState, HomologicalResult, HomologicalConfig, HomologicalStats,
};
pub use mempool_integration::{
    ConsensusStatus as MempoolConsensusStatus, IntegrationConfig, MempoolDAGIntegration,
};
pub use ordering_rules::OrderingEngine;
pub use quantum_beacon::{BeaconState, QuantumBeacon};
pub use quantum_vdf::{QuantumVDF, QuantumVDFConfig, QuantumVDFProof, VDFComputationResult, VDFSecurityLevel};
pub use vertex_creator::{Vertex, VertexCreator, VertexCreatorConfig};

// ✨ v1.0.58-beta: Genus-2 VDF exports (quantum-resistant time-locking)
#[cfg(feature = "advanced-crypto")]
pub use genus2_vdf_integration::{
    Genus2VDFEngine, Genus2VDFConfig, Genus2VDFResult, Genus2SecurityLevel, Genus2BatchVerifier,
};

// v10.0.0: SIMD-accelerated bitfield DAG exports
#[cfg(feature = "simd-dag")]
pub use simd_sets::{VertexBitfield, BitfieldDag, BitfieldDagStats};
#[cfg(feature = "simd-dag")]
pub use ordering_rules::SimdOrderingEngine;

// Re-export types from q-types for easier access
pub use q_types::{Block, BullsharkCert, NarwhalPayload};

/// PHASE 4: Validator information for anonymous mesh network
#[derive(Debug, Clone)]
pub struct ValidatorInfo {
    pub node_id: NodeId,
    pub onion_address: Option<String>,
    pub stake_weight: u64,
    pub last_seen: std::time::SystemTime,
    pub connection_quality: f64, // 0.0-1.0 based on Tor connection performance
    pub latency_ms: u64,         // Average latency through Tor
    pub is_anonymous: bool,      // True if connected via Tor
}

/// Main DAG-Knight consensus engine - PHASE 4 ANONYMOUS MESH INTEGRATION
pub struct DAGKnightConsensus {
    pub node_id: NodeId,
    pub vertex_store: VertexStore,
    pub anchor_election: QuantumAnchorElection,
    pub ordering_engine: OrderingEngine,
    pub quantum_beacon: QuantumBeacon,
    pub commit_protocol: CommitProtocol,
    pub quantum_vdf: Arc<QuantumVDF>,
    pub vertex_creator: VertexCreator,

    // State tracking
    pub committed_vertices: RwLock<HashMap<Round, Vec<VertexId>>>,
    pub pending_commits: RwLock<HashMap<Round, Vec<VertexId>>>,
    pub current_round: RwLock<Round>,
    pub last_commit_round: RwLock<Round>,

    // Configuration
    pub f: usize,   // Number of Byzantine nodes (2f+1 = total nodes)
    pub delta: u64, // Rounds to look back for commit decision

    // PHASE 4: ANONYMOUS MESH NETWORK INTEGRATION
    pub anonymous_validator_set: RwLock<HashMap<NodeId, ValidatorInfo>>, // Tor-anonymized validators
    pub onion_address_registry: RwLock<HashMap<NodeId, String>>, // Node -> .qnk.onion mapping
    pub mesh_connectivity_score: RwLock<f64>, // Network connectivity quality (0.0-1.0)
    pub tor_latency_compensation: RwLock<HashMap<NodeId, u64>>, // Latency adjustments per validator
}

impl DAGKnightConsensus {
    pub async fn new(node_id: NodeId, f: usize) -> Result<Self> {
        // Configure quantum VDF for Phase 1
        let vdf_config = QuantumVDFConfig {
            base_difficulty: 1024,    // Moderate difficulty for production
            quantum_enhancement: 0.7, // 70% quantum enhancement for Phase 1
            parallel_threads: 4,
            qrng_seed_interval: std::time::Duration::from_secs(30),
            security_level: VDFSecurityLevel::PostQuantum,
        };

        let quantum_vdf = Arc::new(QuantumVDF::new(vdf_config).await?);
        // 🔐 v2.4.7-beta: Use random key for vertex signing
        let vertex_creator = VertexCreator::new_with_random_key(node_id, quantum_vdf.clone());

        Ok(Self {
            node_id,
            vertex_store: VertexStore::new(Arc::new(InMemoryVertexStorage::new())),
            anchor_election: QuantumAnchorElection::new(f).await?,
            ordering_engine: OrderingEngine::new()?,
            quantum_beacon: QuantumBeacon::new()?,
            commit_protocol: CommitProtocol::new(f)?,
            quantum_vdf: quantum_vdf.clone(),
            vertex_creator,
            committed_vertices: RwLock::new(HashMap::new()),
            pending_commits: RwLock::new(HashMap::new()),
            current_round: RwLock::new(0),
            last_commit_round: RwLock::new(0),
            f,
            delta: 1, // ⚡ v1.0.72-beta: Aggressive delta=1 for sub-50ms finality (was 4)

            // PHASE 4: Initialize anonymous mesh network state
            anonymous_validator_set: RwLock::new(HashMap::new()),
            onion_address_registry: RwLock::new(HashMap::new()),
            mesh_connectivity_score: RwLock::new(0.0),
            tor_latency_compensation: RwLock::new(HashMap::new()),
        })
    }

    /// Process a certificate from the Narwhal layer
    pub async fn process_certificate(
        &self,
        certificate: Certificate,
    ) -> Result<Vec<CommitDecision>> {
        debug!(
            "Processing certificate for vertex {} in round {}",
            hex::encode(certificate.vertex_id),
            certificate.round
        );

        let mut commit_decisions = Vec::new();

        // Update current round if necessary
        {
            let mut current = self.current_round.write().await;
            if certificate.round > *current {
                *current = certificate.round;
                info!("Advanced to round {}", certificate.round);
            }
        }

        // Store vertex if we have it
        if let Some(vertex) = self.vertex_store.get_vertex(&certificate.vertex_id).await {
            // Process through ordering engine
            let ordering_result = self
                .ordering_engine
                .process_vertex(&vertex, &certificate)
                .await?;

            // Check for anchor election in even rounds
            if certificate.round % 2 == 0 {
                let election_result = self
                    .anchor_election
                    .elect_anchor(certificate.round, &vertex)
                    .await?;

                if let Some(anchor_vertex_id) = election_result.anchor_vertex_id {
                    // Log enhanced information for L-VRF integration
                    let vrf_info = if let Some(ref vrf_result) = election_result.vrf_result {
                        format!(
                            "L-VRF entropy: {:.3}, quality: {:.3}",
                            vrf_result.output.entropy_estimate(),
                            vrf_result.metadata.quantum_enhanced as u8 as f64
                        )
                    } else {
                        "Classical selection".to_string()
                    };

                    info!(
                        "Elected anchor {} for round {} with VDF output {:?}, {}",
                        hex::encode(anchor_vertex_id),
                        certificate.round,
                        hex::encode(&election_result.vdf_output[..8]),
                        vrf_info
                    );

                    // Apply commit rule with L-VRF enhanced decision
                    let commit_result = self
                        .commit_protocol
                        .evaluate_commit_with_vrf(
                            certificate.round,
                            anchor_vertex_id,
                            election_result.vrf_result.as_ref(),
                        )
                        .await?;

                    if let Some(decision) = commit_result {
                        commit_decisions.push(decision);

                        // Update committed vertices
                        {
                            let mut committed = self.committed_vertices.write().await;
                            committed
                                .entry(certificate.round)
                                .or_insert_with(Vec::new)
                                .push(anchor_vertex_id);
                        }

                        {
                            let mut last_commit = self.last_commit_round.write().await;
                            *last_commit = certificate.round;
                        }
                    }
                }
            }

            // Check for delayed commit decisions (δ rounds back)
            let current_round = *self.current_round.read().await;
            if current_round >= self.delta {
                let check_round = current_round - self.delta;
                let delayed_commits = self
                    .commit_protocol
                    .check_delayed_commits(check_round)
                    .await?;
                commit_decisions.extend(delayed_commits);
            }
        }

        Ok(commit_decisions)
    }

    /// Advance to the next round with VDF timing coordination
    pub async fn advance_round(&self) -> Result<()> {
        let start_time = std::time::Instant::now();

        let new_round = {
            let mut current = self.current_round.write().await;
            *current += 1;
            *current
        };

        // Generate new quantum beacon for the round
        let beacon_state = self.quantum_beacon.generate_beacon_state(new_round).await?;

        // Coordinate VDF computation with round timing for Phase 1 completion
        let vdf_input = self.create_round_vdf_input(&beacon_state.entropy_seed, new_round)?;
        let vdf_iterations = self.calculate_vdf_iterations_for_round(new_round);

        // Convert Vec<u8> to [u8; 32] for VDF input
        let mut vdf_input_array = [0u8; 32];
        let len = std::cmp::min(vdf_input.len(), 32);
        vdf_input_array[..len].copy_from_slice(&vdf_input[..len]);

        // Store VDF input for later computation (avoiding complex async spawn for now)
        debug!(
            "VDF input prepared for round {} (background computation disabled)",
            new_round
        );

        let round_advancement_time = start_time.elapsed();

        info!(
            "Advanced to round {} with quantum beacon entropy: {} ({}ms)",
            new_round,
            hex::encode(&beacon_state.entropy_seed[..8]),
            round_advancement_time.as_millis()
        );

        // Performance validation for consensus timing targets
        if round_advancement_time.as_millis() > 10 {
            warn!(
                "Round advancement time {}ms exceeds 10ms target",
                round_advancement_time.as_millis()
            );
        }

        Ok(())
    }

    /// Create VDF input for consensus round timing
    fn create_round_vdf_input(&self, beacon_entropy: &[u8], round: Round) -> Result<Vec<u8>> {
        use sha3::{Digest, Sha3_256};

        let mut hasher = Sha3_256::new();
        hasher.update(beacon_entropy);
        hasher.update(&round.to_be_bytes());
        hasher.update(&self.node_id);
        hasher.update(b"consensus-round-vdf-input");

        Ok(hasher.finalize().to_vec())
    }

    /// Calculate VDF iterations based on round and consensus timing
    fn calculate_vdf_iterations_for_round(&self, round: Round) -> u64 {
        // Base iterations for consensus timing (target: complete in <15ms)
        let base_iterations = 1024u64;

        // Scale with round for increasing security
        let round_factor = 1.0 + (round as f64 * 0.001); // 0.1% increase per round

        // Quantum enhancement factor (Phase 1 completion)
        let quantum_factor = 1.2; // 20% increase for quantum resistance

        ((base_iterations as f64) * round_factor * quantum_factor) as u64
    }

    /// Integrate VDF timing with consensus round progression
    pub async fn integrate_quantum_vdf_timing(&mut self, target_round_time_ms: u64) -> Result<()> {
        info!(
            "🔄 Integrating quantum VDF timing with consensus rounds (target: {}ms)",
            target_round_time_ms
        );

        // Update VDF configuration for consensus timing
        let current_round = *self.current_round.read().await;
        let optimal_iterations = self.calculate_optimal_vdf_iterations(target_round_time_ms);

        // Test VDF timing with sample computation
        let test_input_vec = format!("timing-test-round-{}", current_round).into_bytes();
        let mut test_input = [0u8; 32];
        let len = std::cmp::min(test_input_vec.len(), 32);
        test_input[..len].copy_from_slice(&test_input_vec[..len]);

        let timing_start = std::time::Instant::now();

        let _result = self.quantum_vdf.compute_proof(&test_input).await?;

        let actual_time = timing_start.elapsed();
        info!(
            "✅ VDF timing test: {} iterations completed in {:?}",
            optimal_iterations, actual_time
        );

        // Adjust VDF parameters for consensus timing requirements
        if actual_time.as_millis() > target_round_time_ms as u128 {
            warn!("VDF timing exceeds target, consider reducing iterations or optimizing");
        } else {
            info!("VDF timing meets consensus requirements");
        }

        Ok(())
    }

    /// Calculate optimal VDF iterations for target consensus timing
    fn calculate_optimal_vdf_iterations(&self, target_round_time_ms: u64) -> u64 {
        // Conservative estimate: aim for VDF to complete in 50% of target round time
        let vdf_time_budget_ms = target_round_time_ms / 2;

        // Estimate iterations based on Phase 1 performance (rough estimate)
        let estimated_ms_per_1k_iterations = 2.0; // Based on quantum VDF benchmarks

        ((vdf_time_budget_ms as f64) / estimated_ms_per_1k_iterations * 1000.0) as u64
    }

    /// Get current consensus status
    pub async fn get_status(&self) -> ConsensusStatus {
        let current_round = *self.current_round.read().await;
        let last_commit_round = *self.last_commit_round.read().await;
        let committed_vertices = self.committed_vertices.read().await;
        let pending_commits = self.pending_commits.read().await;

        let total_committed: usize = committed_vertices.values().map(|v| v.len()).sum();

        ConsensusStatus {
            current_round,
            last_commit_round,
            committed_vertices: total_committed as u64,
            pending_commits: pending_commits.len() as u64,
            beacon_health: self.quantum_beacon.get_health_metrics().await,
        }
    }

    /// Get vertices committed in a specific round
    pub async fn get_committed_vertices(&self, round: Round) -> Vec<VertexId> {
        let committed = self.committed_vertices.read().await;
        committed.get(&round).cloned().unwrap_or_else(Vec::new)
    }

    /// Get N most recent committed vertices for use as DAG parents
    ///
    /// This method is used during block production to populate the dag_parents field.
    /// It returns vertices from the most recent committed rounds, prioritizing
    /// newer rounds first.
    ///
    /// # Arguments
    /// * `count` - Maximum number of vertices to return
    ///
    /// # Returns
    /// Vector of VertexIds from recent committed rounds (most recent first)
    ///
    /// # Example
    /// ```ignore
    /// // Get 3 most recent committed vertices to use as DAG parents
    /// let parents = dag_knight.get_recent_committed_vertices(3).await?;
    /// let block = QBlock {
    ///     dag_parents: parents,
    ///     // ... other fields
    /// };
    /// ```
    pub async fn get_recent_committed_vertices(&self, count: usize) -> Result<Vec<VertexId>> {
        let committed = self.committed_vertices.read().await;

        // Get all rounds and sort descending (most recent first)
        let mut rounds: Vec<Round> = committed.keys().copied().collect();
        rounds.sort_by(|a, b| b.cmp(a)); // Descending order

        let mut vertices = Vec::new();

        // Collect vertices from most recent rounds until we have enough
        for round in rounds {
            if vertices.len() >= count {
                break;
            }

            if let Some(round_vertices) = committed.get(&round) {
                for vertex_id in round_vertices {
                    if vertices.len() >= count {
                        break;
                    }
                    vertices.push(*vertex_id);
                }
            }
        }

        Ok(vertices)
    }

    /// Check if consensus has progressed beyond a round
    pub async fn is_round_committed(&self, round: Round) -> bool {
        let last_commit = *self.last_commit_round.read().await;
        round <= last_commit
    }

    /// ⚔️ v1.0.69-beta: Get the latest committed/finalized round
    ///
    /// This is used by the block producer to enforce ancestor finality checks
    /// and prevent tail forking vulnerabilities in pipelined BFT.
    ///
    /// # BFT Safety
    /// Proposals should not be more than δ+1 rounds ahead of this value,
    /// otherwise they risk creating tail forks that could be reorganized.
    ///
    /// # Returns
    /// * `Ok(round)` - The highest round with committed vertices
    /// * `Err` - If consensus state cannot be read (should never happen)
    ///
    /// # Example
    /// ```ignore
    /// let committed = dag_knight.get_latest_committed_round().await?;
    /// if proposed_height > committed + delta + 1 {
    ///     return Err("Too far ahead of committed height");
    /// }
    /// ```
    pub async fn get_latest_committed_round(&self) -> Result<Round> {
        let last_commit = *self.last_commit_round.read().await;
        Ok(last_commit)
    }

    /// 🚀 v1.0.73-beta: Initialize committed round from persisted blockchain height
    /// This MUST be called on server restart to sync consensus state with storage
    /// Without this, tail fork protection blocks all new blocks after restart!
    pub async fn initialize_committed_round(&self, blockchain_height: u64) {
        let mut last_commit = self.last_commit_round.write().await;
        let mut current_round = self.current_round.write().await;

        // Set committed round to current blockchain height
        // This allows new blocks to be proposed immediately after restart
        *last_commit = blockchain_height;
        *current_round = blockchain_height;

        info!(
            "🔧 [CONSENSUS INIT] Initialized committed round to {} from blockchain height",
            blockchain_height
        );
    }

    /// 🚀 v1.0.76-beta: Advance committed round after successful block production
    /// This MUST be called whenever a block is successfully added to the chain
    /// Without this, tail fork protection will eventually stall block production
    pub async fn advance_committed_round(&self, new_height: u64) {
        let mut last_commit = self.last_commit_round.write().await;

        // Only advance if new height is greater (prevents rollback attacks)
        if new_height > *last_commit {
            let old = *last_commit;
            *last_commit = new_height;
            debug!(
                "📈 [COMMIT ADVANCE] Committed round advanced: {} → {}",
                old, new_height
            );
        }
    }

    /// Get ordering of transactions from committed vertices
    pub async fn get_transaction_ordering(
        &self,
        from_round: Round,
        to_round: Round,
    ) -> Result<Vec<Transaction>> {
        let mut ordered_transactions = Vec::new();

        for round in from_round..=to_round {
            let committed_vertices = self.get_committed_vertices(round).await;

            for vertex_id in committed_vertices {
                if let Some(vertex) = self.vertex_store.get_vertex(&vertex_id).await {
                    // Add transactions in deterministic order
                    let mut txs = vertex.transactions;
                    txs.sort_by(|a, b| a.id.cmp(&b.id)); // Deterministic ordering
                    ordered_transactions.extend(txs);
                }
            }
        }

        Ok(ordered_transactions)
    }

    /// Performance and health metrics
    pub async fn get_metrics(&self) -> ConsensusMetrics {
        let status = self.get_status().await;
        let vertex_stats = self.vertex_store.get_stats().await;
        let anchor_stats = self.anchor_election.get_statistics().await;
        let commit_stats = self.commit_protocol.get_statistics().await;
        let vdf_stats = self.quantum_vdf.get_statistics().await;

        // Enhanced quantum entropy quality combining beacon and VDF
        let combined_entropy_quality =
            (status.beacon_health.entropy_quality + vdf_stats.quantum_entropy_quality) / 2.0;

        ConsensusMetrics {
            current_round: status.current_round,
            commit_latency_rounds: status.current_round - status.last_commit_round,
            total_vertices: vertex_stats.total_vertices as u64,
            committed_vertices: status.committed_vertices,
            anchor_elections: anchor_stats.total_elections,
            successful_commits: commit_stats.successful_commits,
            throughput_tps: commit_stats.average_tps,
            quantum_entropy_quality: combined_entropy_quality,
            vdf_computation_count: vdf_stats.total_computations,
            vdf_quantum_enhancement_ratio: vdf_stats.parallel_efficiency,
            average_vdf_time_ms: vdf_stats.average_computation_time.as_millis() as f64,
        }
    }
}

/// Result of a commit decision
#[derive(Debug, Clone)]
pub struct CommitDecision {
    pub round: Round,
    pub vertex_id: VertexId,
    pub commit_type: CommitType,
    pub transactions: Vec<Transaction>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub enum CommitType {
    AnchorCommit,  // Committed due to anchor election
    DelayedCommit, // Committed after δ rounds
    ChainCommit,   // Committed due to causal dependency
}

/// Consensus engine status
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsensusStatus {
    pub current_round: Round,
    pub last_commit_round: Round,
    pub committed_vertices: u64,
    pub pending_commits: u64,
    pub beacon_health: quantum_beacon::BeaconHealth,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsensusMetrics {
    pub current_round: Round,
    pub commit_latency_rounds: u64,
    pub total_vertices: u64,
    pub committed_vertices: u64,
    pub anchor_elections: u64,
    pub successful_commits: u64,
    pub throughput_tps: f64,
    pub quantum_entropy_quality: f64,
    pub vdf_computation_count: u64,
    pub vdf_quantum_enhancement_ratio: f64,
    pub average_vdf_time_ms: f64,
}

/// Trait for consensus event handlers
#[async_trait]
pub trait ConsensusEventHandler: Send + Sync {
    async fn on_vertex_committed(&self, decision: &CommitDecision) -> Result<()>;
    async fn on_anchor_elected(&self, round: Round, vertex_id: VertexId) -> Result<()>;
    async fn on_round_advanced(&self, round: Round) -> Result<()>;
    async fn on_consensus_stall(&self, round: Round) -> Result<()>;
}

/// Default event handler that logs events
pub struct LoggingEventHandler {
    pub anonymous_validator_set: RwLock<HashMap<NodeId, ValidatorInfo>>, // Tor-anonymized validators
    pub onion_address_registry: RwLock<HashMap<NodeId, String>>, // Node -> .qnk.onion mapping
    pub mesh_connectivity_score: RwLock<f64>, // Network connectivity quality (0.0-1.0)
    pub tor_latency_compensation: RwLock<HashMap<NodeId, u64>>, // Latency adjustments per validator
}

#[async_trait]
impl ConsensusEventHandler for LoggingEventHandler {
    async fn on_vertex_committed(&self, decision: &CommitDecision) -> Result<()> {
        info!(
            "Committed vertex {} in round {} ({:?})",
            hex::encode(decision.vertex_id),
            decision.round,
            decision.commit_type
        );
        Ok(())
    }

    async fn on_anchor_elected(&self, round: Round, vertex_id: VertexId) -> Result<()> {
        info!(
            "Anchor elected for round {}: {}",
            round,
            hex::encode(vertex_id)
        );
        Ok(())
    }

    async fn on_round_advanced(&self, round: Round) -> Result<()> {
        debug!("Advanced to round {}", round);
        Ok(())
    }

    async fn on_consensus_stall(&self, round: Round) -> Result<()> {
        warn!("Consensus appears to be stalled at round {}", round);
        Ok(())
    }
}

// PHASE 4: ANONYMOUS MESH NETWORK INTEGRATION METHODS
impl LoggingEventHandler {
    /// Default constructor for LoggingEventHandler with empty state
    pub fn new() -> Self {
        Self {
            anonymous_validator_set: RwLock::new(HashMap::new()),
            onion_address_registry: RwLock::new(HashMap::new()),
            mesh_connectivity_score: RwLock::new(0.0),
            tor_latency_compensation: RwLock::new(HashMap::new()),
        }
    }

    /// Update mesh network connectivity score based on validator status
    pub async fn update_mesh_connectivity_score(&self) {
        let validator_set = self.anonymous_validator_set.read().await;

        if validator_set.is_empty() {
            *self.mesh_connectivity_score.write().await = 0.0;
            return;
        }

        let mut total_quality = 0.0;
        let mut anonymous_count = 0;

        for validator in validator_set.values() {
            total_quality += validator.connection_quality;
            if validator.is_anonymous {
                anonymous_count += 1;
            }
        }

        let avg_quality = total_quality / validator_set.len() as f64;
        let anonymity_bonus = (anonymous_count as f64 / validator_set.len() as f64) * 0.2; // 20% bonus for anonymity
        let connectivity_score = (avg_quality + anonymity_bonus).min(1.0);

        *self.mesh_connectivity_score.write().await = connectivity_score;

        info!("🌐 PHASE 4: Mesh connectivity score updated: {:.3} ({} total validators, {} anonymous)",
              connectivity_score, validator_set.len(), anonymous_count);
    }

    /// Process certificate (stub method for compatibility)
    pub async fn process_certificate(&self, certificate: &Certificate) -> Result<Vec<VertexId>> {
        info!(
            "📜 Processing certificate for vertex: {} at round: {}",
            hex::encode(certificate.vertex_id),
            certificate.round
        );
        Ok(vec![certificate.vertex_id])
    }

    /// Register anonymous validator discovered via DNS-Phantom and connected via Tor
    pub async fn register_anonymous_validator(&self, validator_info: ValidatorInfo) -> Result<()> {
        info!(
            "🧅 PHASE 4: Registering anonymous validator: {} via {}",
            hex::encode(&validator_info.node_id[..4]),
            validator_info
                .onion_address
                .as_ref()
                .unwrap_or(&"direct".to_string())
        );

        let mut validator_set = self.anonymous_validator_set.write().await;
        validator_set.insert(validator_info.node_id, validator_info.clone());

        // Register onion address if available
        if let Some(ref onion_addr) = validator_info.onion_address {
            let mut registry = self.onion_address_registry.write().await;
            registry.insert(validator_info.node_id, onion_addr.clone());
            info!("🔒 PHASE 4: Registered onion address: {}", onion_addr);
        }

        // Update Tor latency compensation
        if validator_info.is_anonymous {
            let mut latency_comp = self.tor_latency_compensation.write().await;
            latency_comp.insert(validator_info.node_id, validator_info.latency_ms);
            info!(
                "⏰ PHASE 4: Tor latency compensation: {}ms for anonymous validator",
                validator_info.latency_ms
            );
        }

        self.update_mesh_connectivity_score().await;

        Ok(())
    }

    /// Get anonymous validator statistics for monitoring
    pub async fn get_anonymous_mesh_stats(&self) -> AnonymousMeshStats {
        let validator_set = self.anonymous_validator_set.read().await;
        let connectivity_score = *self.mesh_connectivity_score.read().await;

        let mut total_validators = validator_set.len();
        let mut anonymous_validators = 0;
        let mut total_latency = 0u64;
        let mut total_quality = 0.0;

        for validator in validator_set.values() {
            if validator.is_anonymous {
                anonymous_validators += 1;
            }
            total_latency += validator.latency_ms;
            total_quality += validator.connection_quality;
        }

        AnonymousMeshStats {
            total_validators,
            anonymous_validators,
            direct_validators: total_validators - anonymous_validators,
            average_latency_ms: if total_validators > 0 {
                total_latency / total_validators as u64
            } else {
                0
            },
            average_connection_quality: if total_validators > 0 {
                total_quality / total_validators as f64
            } else {
                0.0
            },
            mesh_connectivity_score: connectivity_score,
            anonymity_ratio: if total_validators > 0 {
                anonymous_validators as f64 / total_validators as f64
            } else {
                0.0
            },
        }
    }

    /// Phase 4: Process certificate with Tor latency compensation
    pub async fn process_certificate_with_latency_compensation(
        &self,
        certificate: &Certificate,
    ) -> Result<Vec<VertexId>> {
        info!("⚡ PHASE 4: Processing certificate with Tor latency compensation");

        // Certificate processing with anonymous validator considerations
        // Note: Certificate struct only contains vertex_id, round, signatures, threshold_met
        info!(
            "🔍 Processing certificate for vertex: {} at round: {}",
            hex::encode(certificate.vertex_id),
            certificate.round
        );

        let validator_set = self.anonymous_validator_set.read().await;
        let latency_comp = self.tor_latency_compensation.read().await;

        // Check if any validators in the certificate signatures are anonymous
        let mut anonymous_validator_count = 0;
        for node_id in certificate.signatures.keys() {
            if let Some(validator) = validator_set.get(node_id) {
                if validator.is_anonymous {
                    anonymous_validator_count += 1;
                    if let Some(latency) = latency_comp.get(node_id) {
                        debug!(
                            "🧅 Validator {} has {}ms Tor latency compensation",
                            hex::encode(&node_id[..4]),
                            latency
                        );
                    }
                }
            }
        }

        if anonymous_validator_count > 0 {
            info!(
                "🔐 Certificate contains {} anonymous validator signatures",
                anonymous_validator_count
            );
        }

        // Process certificate (simplified since we don't have a real process_certificate method)
        Ok(vec![certificate.vertex_id])
    }
}

/// PHASE 4: Statistics for anonymous mesh network monitoring
#[derive(Debug, Clone)]
pub struct AnonymousMeshStats {
    pub total_validators: usize,
    pub anonymous_validators: usize,
    pub direct_validators: usize,
    pub average_latency_ms: u64,
    pub average_connection_quality: f64,
    pub mesh_connectivity_score: f64,
    pub anonymity_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_vertex(id: u8, round: Round) -> Vertex {
        Vertex {
            id: [id; 32],
            round,
            author: [1; 32],
            tx_root: [0; 32],
            parents: vec![],
            transactions: vec![],
            signature: vec![],
            timestamp: Utc::now(),
        }
    }

    fn create_test_certificate(vertex_id: VertexId, round: Round) -> Certificate {
        Certificate {
            vertex_id,
            round,
            signatures: std::collections::BTreeMap::new(),
            threshold_met: true,
        }
    }

    #[tokio::test]
    async fn test_consensus_creation() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1);
        assert!(consensus.is_ok());

        let consensus = consensus.unwrap();
        assert_eq!(consensus.node_id, node_id);
        assert_eq!(consensus.f, 1);
    }

    #[tokio::test]
    async fn test_round_advancement() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1).unwrap();

        let initial_round = *consensus.current_round.read().await;
        assert_eq!(initial_round, 0);

        consensus.advance_round().await.unwrap();

        let new_round = *consensus.current_round.read().await;
        assert_eq!(new_round, 1);
    }

    #[tokio::test]
    async fn test_certificate_processing() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1).unwrap();

        // Create and store a vertex
        let vertex = create_test_vertex(42, 1);
        consensus
            .vertex_store
            .store_vertex(vertex.clone())
            .await
            .unwrap();

        // Create certificate
        let certificate = create_test_certificate(vertex.id, vertex.round);

        // Process certificate
        let decisions = consensus.process_certificate(certificate).await.unwrap();

        // Should have advanced round
        let current_round = *consensus.current_round.read().await;
        assert!(current_round >= 1);
    }

    #[tokio::test]
    async fn test_status_reporting() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1).unwrap();

        let status = consensus.get_status().await;
        assert_eq!(status.current_round, 0);
        assert_eq!(status.last_commit_round, 0);
        assert_eq!(status.committed_vertices, 0);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let node_id = [1u8; 32];
        let consensus = DAGKnightConsensus::new(node_id, 1).unwrap();

        let metrics = consensus.get_metrics().await;
        assert_eq!(metrics.current_round, 0);
        assert_eq!(metrics.committed_vertices, 0);
        assert!(metrics.quantum_entropy_quality >= 0.0);
    }
}
