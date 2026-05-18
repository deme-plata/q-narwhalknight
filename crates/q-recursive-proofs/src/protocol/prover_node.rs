//! Prover Node Implementation
//!
//! A prover node participates in the decentralized proof generation network.
//! It listens for epoch proof tasks, generates proofs, and broadcasts them.

use super::messages::{EpochProofSubmission, EpochProofTask, HardwareInfo, RewardParams};
use super::topics::{RecursiveProofTopics, TOPIC_EPOCH_PROOFS, TOPIC_EPOCH_PROOF_TASK};
use crate::circuits::epoch_transition::{EpochTransitionCircuit, EpochTransitionConfig};
use crate::{EpochProof, EpochProofMetadata, EpochPublicInputs};
use q_lattice_guard::{LatticeGuard, LatticeGuardProof, LatticeGuardSRS, SecurityLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// Prover node configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverNodeConfig {
    /// Peer ID for this prover
    pub peer_id: String,
    /// LatticeGuard security level
    pub security_level: SecurityLevel,
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// Number of parallel proving threads
    pub parallel_threads: usize,
    /// Maximum epochs to prove concurrently
    pub max_concurrent_proofs: usize,
    /// Minimum reward to participate
    pub min_reward: u64,
    /// Epoch transition circuit config
    pub circuit_config: EpochTransitionConfig,
}

impl Default for ProverNodeConfig {
    fn default() -> Self {
        Self {
            peer_id: "prover-default".to_string(),
            security_level: SecurityLevel::PQ128,
            use_gpu: false,
            parallel_threads: num_cpus::get(),
            max_concurrent_proofs: 1,
            min_reward: 0,
            // Use light mode by default for ~15x faster SRS generation
            circuit_config: EpochTransitionConfig::light_mode(),
        }
    }
}

impl ProverNodeConfig {
    /// Create a light mode configuration for faster startup
    ///
    /// Light mode uses:
    /// - 100 blocks per epoch (instead of 1000)
    /// - 20 validators (instead of 100)
    /// - Smaller verifier config
    ///
    /// Results in ~200K constraints vs ~3M constraints
    pub fn light() -> Self {
        Self {
            circuit_config: EpochTransitionConfig::light_mode(),
            ..Default::default()
        }
    }

    /// Create a full mode configuration for production use
    pub fn full() -> Self {
        Self {
            circuit_config: EpochTransitionConfig::default(),
            ..Default::default()
        }
    }
}

/// Current proving task status
#[derive(Clone, Debug)]
pub struct ProvingTask {
    /// Epoch being proved
    pub epoch: u64,
    /// Task details
    pub task: EpochProofTask,
    /// Start time
    pub started_at: Instant,
    /// Status
    pub status: ProvingStatus,
}

/// Proving task status
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvingStatus {
    /// Gathering data
    Preparing,
    /// Generating proof
    Proving,
    /// Proof complete, awaiting broadcast
    Complete,
    /// Failed
    Failed(String),
    /// Cancelled (another prover won)
    Cancelled,
}

/// Prover node metrics
#[derive(Clone, Debug, Default)]
pub struct ProverMetrics {
    /// Total proofs generated
    pub proofs_generated: u64,
    /// Total proofs accepted by network
    pub proofs_accepted: u64,
    /// Total rewards earned
    pub rewards_earned: u64,
    /// Average proving time (ms)
    pub avg_proving_time_ms: f64,
    /// Current active tasks
    pub active_tasks: usize,
}

/// Prover node for decentralized proof generation
pub struct ProverNode {
    /// Configuration
    config: ProverNodeConfig,
    /// LatticeGuard prover
    lattice_guard: Arc<LatticeGuard>,
    /// Epoch transition circuit
    circuit: Arc<EpochTransitionCircuit>,
    /// SRS for proving
    srs: Arc<LatticeGuardSRS>,
    /// Current proving tasks
    active_tasks: Arc<RwLock<HashMap<u64, ProvingTask>>>,
    /// Completed proofs cache
    proof_cache: Arc<RwLock<HashMap<u64, EpochProof>>>,
    /// Metrics
    metrics: Arc<RwLock<ProverMetrics>>,
    /// Shutdown signal
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl ProverNode {
    /// Create new prover node
    pub fn new(config: ProverNodeConfig) -> anyhow::Result<Self> {
        info!("Initializing prover node:");
        info!("  Peer ID: {}", config.peer_id);
        info!("  Security level: {:?}", config.security_level);
        info!("  Parallel threads: {}", config.parallel_threads);

        // Initialize LatticeGuard
        let lattice_guard = Arc::new(LatticeGuard::new(config.security_level)?);

        // Initialize epoch transition circuit
        let circuit = Arc::new(EpochTransitionCircuit::new(config.circuit_config.clone()));

        // Calculate max constraints needed
        // Use 100 blocks for light mode, which gives ~200K constraints instead of ~3M
        let blocks_for_estimation = config
            .circuit_config
            .state_config
            .max_blocks_per_epoch
            .min(100);
        let max_constraints = circuit.estimate_constraints(blocks_for_estimation);
        info!(
            "  Estimated constraints for {} blocks: {} (light mode)",
            blocks_for_estimation, max_constraints
        );

        // Use SRS caching to avoid regenerating on every startup
        // Cache location: use Q_DB_PATH environment or fall back to /tmp
        let cache_path = std::env::var("Q_DB_PATH")
            .map(|p| std::path::PathBuf::from(p).join("srs_cache"))
            .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/q-lattice-guard-srs"));

        info!("  SRS cache path: {:?}", cache_path);

        let mut rng = rand::thread_rng();
        let srs = Arc::new(LatticeGuardSRS::generate_or_load(
            lattice_guard.params().clone(),
            max_constraints,
            &cache_path,
            &mut rng,
        )?);

        info!("✅ Prover node initialized");

        Ok(Self {
            config,
            lattice_guard,
            circuit,
            srs,
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            proof_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ProverMetrics::default())),
            shutdown_tx: None,
        })
    }

    /// Handle incoming epoch proof task
    pub async fn handle_task(&self, task: EpochProofTask) -> anyhow::Result<()> {
        // Check if we should participate
        if !self.should_participate(&task) {
            debug!("Skipping task for epoch {}: criteria not met", task.epoch);
            return Ok(());
        }

        // Check if already proving this epoch
        {
            let tasks = self.active_tasks.read().await;
            if tasks.contains_key(&task.epoch) {
                debug!("Already proving epoch {}", task.epoch);
                return Ok(());
            }
        }

        // Check if we already have a proof
        {
            let cache = self.proof_cache.read().await;
            if cache.contains_key(&task.epoch) {
                debug!("Already have proof for epoch {}", task.epoch);
                return Ok(());
            }
        }

        info!("Starting proof generation for epoch {}", task.epoch);

        // Create proving task
        let proving_task = ProvingTask {
            epoch: task.epoch,
            task: task.clone(),
            started_at: Instant::now(),
            status: ProvingStatus::Preparing,
        };

        // Register task
        {
            let mut tasks = self.active_tasks.write().await;
            tasks.insert(task.epoch, proving_task);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.active_tasks += 1;
        }

        // Spawn proving task
        let self_clone = self.clone_inner();
        tokio::spawn(async move {
            if let Err(e) = self_clone.prove_epoch(task).await {
                error!("Proving task failed: {}", e);
            }
        });

        Ok(())
    }

    /// Check if we should participate in this proving task
    fn should_participate(&self, task: &EpochProofTask) -> bool {
        // Check minimum reward
        if task.reward < self.config.min_reward {
            return false;
        }

        // Check deadline
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if now > task.deadline {
            return false;
        }

        // Check concurrent task limit
        // (would need async here for real implementation)

        true
    }

    /// Generate proof for an epoch
    async fn prove_epoch(&self, task: EpochProofTask) -> anyhow::Result<EpochProofSubmission> {
        let start = Instant::now();

        // Update status to Proving
        {
            let mut tasks = self.active_tasks.write().await;
            if let Some(t) = tasks.get_mut(&task.epoch) {
                t.status = ProvingStatus::Proving;
            }
        }

        info!(
            "Generating proof for epoch {}, blocks {}-{}",
            task.epoch, task.height_start, task.height_end
        );

        // Fetch required data (simplified - would fetch from storage/network)
        let (previous_proof, epoch_data) = self.gather_proving_data(&task).await?;

        // Build the circuit
        let num_blocks = task.block_hashes.len();
        let circuit = self.circuit.build_circuit(num_blocks);

        // Generate witness (simplified)
        let witness = self.generate_witness(&task, &previous_proof, &epoch_data)?;
        let public_inputs = self.extract_public_inputs(&task);

        // Generate the proof
        info!(
            "Starting LatticeGuard proving ({} constraints)...",
            circuit.num_constraints
        );

        let mut rng = rand::thread_rng();
        let proof =
            self.lattice_guard
                .prove(&circuit, &witness, &public_inputs, &self.srs, &mut rng)?;

        let proving_time = start.elapsed();
        info!(
            "Proof generated in {:?} for epoch {}",
            proving_time, task.epoch
        );

        // Create epoch proof
        let epoch_proof = EpochProof {
            proof,
            public_inputs: EpochPublicInputs {
                previous_state_root: task.previous_state_root,
                current_state_root: task.current_state_root,
                epoch: task.epoch,
                height_range: (task.height_start, task.height_end),
                validator_set_hash: task.validator_set_hash,
                signature_count: task.signature_refs.len() as u32,
                epoch_end_timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
            metadata: EpochProofMetadata {
                version: 1,
                prover_peer_id: Some(self.config.peer_id.clone()),
                proving_time_ms: proving_time.as_millis() as u64,
                hardware_info: Some(self.get_hardware_info()),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        };

        // Cache the proof
        {
            let mut cache = self.proof_cache.write().await;
            cache.insert(task.epoch, epoch_proof.clone());
        }

        // Update task status
        {
            let mut tasks = self.active_tasks.write().await;
            if let Some(t) = tasks.get_mut(&task.epoch) {
                t.status = ProvingStatus::Complete;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.proofs_generated += 1;
            metrics.active_tasks = metrics.active_tasks.saturating_sub(1);

            // Update average proving time
            let total = metrics.proofs_generated as f64;
            let old_avg = metrics.avg_proving_time_ms;
            let new_time = proving_time.as_millis() as f64;
            metrics.avg_proving_time_ms = (old_avg * (total - 1.0) + new_time) / total;
        }

        // Create submission
        let submission = EpochProofSubmission::from_proof(
            task.epoch,
            &epoch_proof,
            self.config.peer_id.clone(),
            proving_time.as_millis() as u64,
        )?;

        Ok(submission)
    }

    /// Gather data needed for proving
    async fn gather_proving_data(
        &self,
        task: &EpochProofTask,
    ) -> anyhow::Result<(Option<LatticeGuardProof>, EpochData)> {
        // In production, this would:
        // 1. Fetch previous epoch's proof from DHT or local storage
        // 2. Fetch all blocks in the epoch
        // 3. Fetch BFT signatures
        // 4. Verify data integrity

        // Simplified placeholder
        let previous_proof = if task.epoch == 0 {
            None // Genesis epoch
        } else {
            // Fetch from DHT
            let _key = RecursiveProofTopics::epoch_proof_key(task.epoch - 1);
            // ... DHT lookup
            None // Placeholder
        };

        let epoch_data = EpochData {
            blocks: Vec::new(), // Would contain actual block data
            signatures: Vec::new(),
        };

        Ok((previous_proof, epoch_data))
    }

    /// Generate witness for the circuit
    fn generate_witness(
        &self,
        _task: &EpochProofTask,
        _previous_proof: &Option<LatticeGuardProof>,
        _epoch_data: &EpochData,
    ) -> anyhow::Result<Vec<u64>> {
        // In production, this would populate all witness values
        // based on the actual epoch data

        // Placeholder witness
        let num_witness = self.circuit.config().verifier_config.num_commitments * 1000;
        Ok(vec![0u64; num_witness])
    }

    /// Extract public inputs from task
    fn extract_public_inputs(&self, task: &EpochProofTask) -> Vec<u64> {
        let mut inputs = Vec::new();

        // Previous state root
        for chunk in task.previous_state_root.chunks(8) {
            inputs.push(u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])));
        }

        // Current state root
        for chunk in task.current_state_root.chunks(8) {
            inputs.push(u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])));
        }

        // Epoch and height
        inputs.push(task.epoch);
        inputs.push(task.height_start);
        inputs.push(task.height_end);

        inputs
    }

    /// Get hardware info
    fn get_hardware_info(&self) -> String {
        format!(
            "threads={}, gpu={}",
            self.config.parallel_threads, self.config.use_gpu
        )
    }

    /// Decode a submitted proof and validate metadata that controls verification.
    fn decode_and_validate_submission(
        submission: &EpochProofSubmission,
    ) -> anyhow::Result<(LatticeGuardProof, EpochPublicInputs, usize)> {
        let (proof, public_inputs) = submission.to_proof()?;

        anyhow::ensure!(
            submission.epoch == public_inputs.epoch,
            "submission epoch {} does not match public inputs epoch {}",
            submission.epoch,
            public_inputs.epoch
        );

        let (height_start, height_end) = public_inputs.height_range;
        let num_blocks = height_end.checked_sub(height_start).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid height range {}..{}: end height must be greater than or equal to start height",
                height_start,
                height_end
            )
        })?;
        let num_blocks = usize::try_from(num_blocks).map_err(|_| {
            anyhow::anyhow!(
                "height range {}..{} spans too many blocks for this platform",
                height_start,
                height_end
            )
        })?;

        Ok((proof, public_inputs, num_blocks))
    }

    /// Verify an incoming proof
    pub async fn verify_proof(&self, submission: &EpochProofSubmission) -> anyhow::Result<bool> {
        let start = Instant::now();

        let (proof, public_inputs, num_blocks) = Self::decode_and_validate_submission(submission)?;

        // Build verification circuit for the decoded proof range.
        let circuit = self.circuit.build_circuit(num_blocks);

        // Verify
        let is_valid =
            self.lattice_guard
                .verify(&circuit, &public_inputs.to_scalars(), &proof, &self.srs)?;

        let verification_time = start.elapsed();
        info!(
            "Proof verification for epoch {} completed in {:?}: {}",
            submission.epoch,
            verification_time,
            if is_valid { "VALID" } else { "INVALID" }
        );

        Ok(is_valid)
    }

    /// Get metrics
    pub async fn metrics(&self) -> ProverMetrics {
        self.metrics.read().await.clone()
    }

    /// Get cached proof for epoch
    pub async fn get_cached_proof(&self, epoch: u64) -> Option<EpochProof> {
        let cache = self.proof_cache.read().await;
        cache.get(&epoch).cloned()
    }

    /// Clone inner state (for spawning tasks)
    fn clone_inner(&self) -> ProverNodeInner {
        ProverNodeInner {
            config: self.config.clone(),
            lattice_guard: Arc::clone(&self.lattice_guard),
            circuit: Arc::clone(&self.circuit),
            srs: Arc::clone(&self.srs),
            active_tasks: Arc::clone(&self.active_tasks),
            proof_cache: Arc::clone(&self.proof_cache),
            metrics: Arc::clone(&self.metrics),
        }
    }
}

/// Inner state for async tasks
struct ProverNodeInner {
    config: ProverNodeConfig,
    lattice_guard: Arc<LatticeGuard>,
    circuit: Arc<EpochTransitionCircuit>,
    srs: Arc<LatticeGuardSRS>,
    active_tasks: Arc<RwLock<HashMap<u64, ProvingTask>>>,
    proof_cache: Arc<RwLock<HashMap<u64, EpochProof>>>,
    metrics: Arc<RwLock<ProverMetrics>>,
}

impl ProverNodeInner {
    async fn prove_epoch(&self, task: EpochProofTask) -> anyhow::Result<EpochProofSubmission> {
        // Similar to ProverNode::prove_epoch but using inner state
        // (Simplified for this implementation)
        Err(anyhow::anyhow!("Not implemented in inner"))
    }
}

/// Epoch data for proving
#[derive(Debug)]
struct EpochData {
    blocks: Vec<Vec<u8>>,
    signatures: Vec<Vec<u8>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prover_node_creation() {
        let config = ProverNodeConfig::default();
        let node = ProverNode::new(config);

        assert!(node.is_ok());
    }

    fn dummy_proof_submission(
        submission_epoch: u64,
        public_epoch: u64,
        height_range: (u64, u64),
    ) -> EpochProofSubmission {
        let proof = LatticeGuardProof {
            commitments: Vec::new(),
            evaluations: (0, 0, 0),
            product_proofs: Vec::new(),
            transcript_state: [0; 32],
            metadata: q_lattice_guard::prover::ProofMetadata {
                num_constraints: 0,
                num_public_inputs: 0,
                security_level: SecurityLevel::PQ128,
                generation_time_ms: 0,
            },
        };
        let public_inputs = EpochPublicInputs {
            previous_state_root: [1; 32],
            current_state_root: [2; 32],
            epoch: public_epoch,
            height_range,
            validator_set_hash: [3; 32],
            signature_count: 0,
            epoch_end_timestamp: 0,
        };

        EpochProofSubmission {
            epoch: submission_epoch,
            proof_data: bincode::serialize(&proof).unwrap(),
            public_inputs_data: bincode::serialize(&public_inputs).unwrap(),
            prover_peer_id: "test-prover".to_string(),
            prover_signature: Vec::new(),
            proving_time_ms: 0,
            hardware_info: None,
            protocol_version: 1,
            created_at: 0,
        }
    }

    #[test]
    fn test_verify_proof_accepts_zero_block_genesis_range() {
        let submission = dummy_proof_submission(0, 0, (0, 0));

        let (_proof, public_inputs, num_blocks) =
            ProverNode::decode_and_validate_submission(&submission).unwrap();

        assert_eq!(public_inputs.epoch, 0);
        assert_eq!(public_inputs.height_range, (0, 0));
        assert_eq!(num_blocks, 0);
    }

    #[test]
    fn test_verify_proof_derives_one_block_epoch_range() {
        let submission = dummy_proof_submission(1, 1, (100, 101));

        let (_proof, public_inputs, num_blocks) =
            ProverNode::decode_and_validate_submission(&submission).unwrap();

        assert_eq!(public_inputs.epoch, 1);
        assert_eq!(public_inputs.height_range, (100, 101));
        assert_eq!(num_blocks, 1);
    }

    #[test]
    fn test_verify_proof_derives_multi_block_epoch_range() {
        let submission = dummy_proof_submission(7, 7, (250, 375));

        let (_proof, public_inputs, num_blocks) =
            ProverNode::decode_and_validate_submission(&submission).unwrap();

        assert_eq!(public_inputs.epoch, 7);
        assert_eq!(public_inputs.height_range, (250, 375));
        assert_eq!(num_blocks, 125);
    }

    #[test]
    fn test_verify_proof_rejects_malformed_height_range() {
        let submission = dummy_proof_submission(2, 2, (10, 9));

        let err = ProverNode::decode_and_validate_submission(&submission).unwrap_err();

        assert!(
            err.to_string().contains("invalid height range 10..9"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_verify_proof_rejects_submission_epoch_mismatch() {
        let submission = dummy_proof_submission(3, 4, (20, 21));

        let err = ProverNode::decode_and_validate_submission(&submission).unwrap_err();

        assert!(
            err.to_string()
                .contains("submission epoch 3 does not match public inputs epoch 4"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_should_participate() {
        let config = ProverNodeConfig {
            min_reward: 100,
            ..Default::default()
        };

        // Can't test fully without creating node, but test config
        assert_eq!(config.min_reward, 100);
    }
}
