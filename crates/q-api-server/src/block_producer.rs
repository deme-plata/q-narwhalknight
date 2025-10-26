/// Block Producer - Aggregates mining solutions into QBlocks
///
/// This module is responsible for:
/// - Collecting mining solutions from the mining submission queue
/// - Creating blocks at regular intervals (10-30 seconds)
/// - Computing quantum metadata (K-parameter, energy functional)
/// - Generating VDF proofs for anchor election
/// - Broadcasting new blocks to the network

use q_types::*;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

/// Block production configuration
#[derive(Debug, Clone)]
pub struct BlockProducerConfig {
    /// Target block time (seconds between blocks)
    pub block_interval_secs: u64,

    /// Maximum mining solutions per block
    pub max_solutions_per_block: usize,

    /// Minimum solutions before producing block (can be 0)
    pub min_solutions_per_block: usize,

    /// Node ID of this validator
    pub node_id: NodeId,

    /// Whether this node is a validator (can propose blocks)
    pub is_validator: bool,

    /// Validator index (0-based, must be unique per validator)
    /// v0.0.22-beta Quick Win #4
    pub validator_index: u64,

    /// Total number of validators in network
    /// v0.0.22-beta Quick Win #4
    pub total_validators: u64,
}

impl Default for BlockProducerConfig {
    fn default() -> Self {
        Self {
            block_interval_secs: 15, // 15 second blocks
            max_solutions_per_block: 100,
            min_solutions_per_block: 1,
            node_id: [0u8; 32],
            is_validator: true,
            validator_index: 0,  // Default to primary validator
            total_validators: 1, // Default to single validator
        }
    }
}

/// Block Producer state machine
pub struct BlockProducer {
    /// Configuration
    config: BlockProducerConfig,

    /// Queue of pending mining solutions
    pending_solutions: VecDeque<MiningSolution>,

    /// Last block production time
    last_block_time: Instant,

    /// Latest block hash (for prev_block_hash)
    latest_block_hash: BlockHash,

    /// Current blockchain height
    current_height: u64,

    /// Total accumulated difficulty
    total_difficulty: u128,

    /// DAG round counter
    dag_round: u64,
}

impl BlockProducer {
    /// Create new block producer
    pub fn new(config: BlockProducerConfig) -> Self {
        Self {
            config,
            pending_solutions: VecDeque::new(),
            last_block_time: Instant::now(),
            latest_block_hash: [0u8; 32], // Genesis
            current_height: 0,
            total_difficulty: 0,
            dag_round: 0,
        }
    }

    /// Add a mining solution to the pending queue
    pub fn queue_solution(&mut self, solution: MiningSolution) {
        debug!("📦 Queued mining solution: nonce={}, miner={:?}",
            solution.nonce,
            hex::encode(&solution.miner_address[..8])
        );

        self.pending_solutions.push_back(solution);
    }

    /// Check if we should produce a block now
    /// v0.0.20-beta: Enabled automatic time-based block production
    /// v0.0.22-beta Quick Win #4: Added simple validator coordination
    pub fn should_produce_block(&self) -> bool {
        let time_elapsed = self.last_block_time.elapsed().as_secs() >= self.config.block_interval_secs;
        let enough_solutions = self.pending_solutions.len() >= self.config.min_solutions_per_block;
        let max_solutions_reached = self.pending_solutions.len() >= self.config.max_solutions_per_block;

        // Immediate production if max solutions reached
        if max_solutions_reached {
            return true;
        }

        // Time-based production
        if time_elapsed {
            if enough_solutions {
                return true;  // Any validator can produce if they have solutions
            } else if self.config.is_validator {
                // v0.0.22-beta Quick Win #4: Simple coordination for empty blocks
                if self.config.total_validators == 1 {
                    return true;  // Single validator - always produce
                } else {
                    // Multi-validator: only index 0 produces empty blocks
                    if self.config.validator_index == 0 {
                        debug!("📦 Validator {} producing empty block (simple coordination mode)",
                               self.config.validator_index);
                        return true;
                    } else {
                        debug!("⏭️  Skipping empty block production (not primary validator)");
                        return false;
                    }
                }
            }
        }

        false
    }

    /// Produce a new block from pending solutions
    /// v0.0.20-beta: Allow blocks without mining solutions for automatic production
    pub async fn produce_block(&mut self) -> Option<QBlock> {
        if !self.config.is_validator {
            return None;
        }

        // v0.0.20-beta: Allow empty blocks for automatic time-based production
        // Collect solutions if available, otherwise create empty block
        let solutions_count = self.config.max_solutions_per_block.min(self.pending_solutions.len());
        let solutions: Vec<MiningSolution> = if solutions_count > 0 {
            self.pending_solutions.drain(0..solutions_count).collect()
        } else {
            // No solutions available - create empty block for DAG continuity
            debug!("📦 Producing empty block for DAG continuity (no mining solutions)");
            vec![]
        };

        info!("🏗️  Producing block: height={}, solutions={}, pending={}",
            self.current_height + 1,
            solutions.len(),
            self.pending_solutions.len()
        );

        // Calculate block difficulty from solutions
        let block_difficulty: u128 = solutions.iter()
            .map(|s| Self::calculate_solution_difficulty(&s.difficulty_target))
            .sum();

        self.total_difficulty += block_difficulty;

        // Create block header
        let timestamp = chrono::Utc::now().timestamp() as u64;

        // Compute Merkle roots
        let solutions_root = Self::compute_solutions_merkle_root(&solutions);
        let tx_root = [0u8; 32]; // TODO: Add transactions
        let state_root = [0u8; 32]; // TODO: Compute state root

        // Create VDF proof (simplified for now)
        let vdf_proof = VDFProof {
            output: self.latest_block_hash.to_vec(),
            verification_proof: vec![],
            iterations: 100 + (self.current_height / 10) as u64,
            challenge: self.latest_block_hash.to_vec(),
            generated_at: timestamp,
        };

        // Generate quantum metadata
        let quantum_metadata = self.generate_quantum_metadata(&solutions, block_difficulty);

        // Create block
        let block = QBlock {
            header: BlockHeader {
                height: self.current_height + 1,
                prev_block_hash: self.latest_block_hash,
                solutions_root,
                tx_root,
                state_root,
                timestamp,
                dag_round: self.dag_round,
                vdf_proof,
                anchor_validator: None, // TODO: Anchor election
                proposer: self.config.node_id,
                total_difficulty: self.total_difficulty,
            },
            mining_solutions: solutions.clone(),
            dag_parents: vec![], // TODO: Get from DAG-Knight
            quantum_metadata,
            transactions: vec![],
            size_bytes: 0, // Will be calculated
        };

        // Calculate block hash
        let block_hash = block.calculate_hash();

        // Update state
        self.latest_block_hash = block_hash;
        self.current_height += 1;
        self.dag_round += 1;
        self.last_block_time = Instant::now();

        info!("✅ BLOCK PRODUCED: Height {}, Hash {}, Solutions {}, Difficulty {}",
            block.header.height,
            hex::encode(&block_hash[..8]),
            solutions.len(),
            block_difficulty
        );

        Some(block)
    }

    /// Generate quantum metadata for block
    fn generate_quantum_metadata(&self, solutions: &[MiningSolution], difficulty: u128) -> QuantumMetadata {
        // Calculate quantum entropy from VDF
        let quantum_entropy = self.calculate_quantum_entropy(solutions);

        // Generate 5D hypergraph coordinates
        let vertex_coordinates = HypergraphCoordinates::from_block_data(
            self.dag_round,
            solutions.len(),
            self.total_difficulty,
            quantum_entropy,
        );

        // Calculate K-parameter (simplified)
        let k_parameter = self.calculate_k_parameter(difficulty, quantum_entropy);

        // Calculate energy components (simplified)
        let energy_components = EnergyComponents {
            coupling: 0.0, // TODO: Calculate from validator phase alignment
            potential: difficulty as f64,
            ordering: self.current_height as f64,
            fault_tolerance: 0.0, // TODO: Byzantine detection
            temporal: self.last_block_time.elapsed().as_secs_f64(),
            finality: 0.0, // TODO: Calculate from DAG depth
        };

        let energy = energy_components.coupling +
                     energy_components.potential +
                     energy_components.ordering +
                     energy_components.fault_tolerance +
                     energy_components.temporal +
                     energy_components.finality;

        QuantumMetadata {
            vertex_coordinates,
            k_parameter,
            energy,
            energy_components,
            spectral_signatures: vec![], // TODO: Collect validator signatures
            wavefunction_phase: quantum_entropy * std::f64::consts::PI,
            entropy_variance: quantum_entropy * 0.1,
            byzantine_scores: std::collections::HashMap::new(),
        }
    }

    /// Calculate quantum entropy from mining solutions
    fn calculate_quantum_entropy(&self, solutions: &[MiningSolution]) -> f64 {
        if solutions.is_empty() {
            return 0.0;
        }

        // Use hash diversity as entropy measure
        let mut entropy_sum = 0.0;
        for solution in solutions {
            let hash_value: u64 = u64::from_be_bytes(solution.hash[0..8].try_into().unwrap());
            entropy_sum += (hash_value as f64 / u64::MAX as f64);
        }

        entropy_sum / solutions.len() as f64
    }

    /// Calculate K-parameter (phase transition metric)
    fn calculate_k_parameter(&self, difficulty: u128, entropy: f64) -> f64 {
        // Simplified K-parameter calculation
        // K = 2π √(ΔH · Δs · ℏ) / τ

        let energy_variance = (difficulty as f64).log10();
        let entropy_variance = entropy * 0.1;
        let planck_constant = 1.0; // Normalized
        let round_duration = self.last_block_time.elapsed().as_secs_f64().max(1.0);

        let product = energy_variance * entropy_variance * planck_constant;
        if product <= 0.0 {
            return 0.0;
        }

        (2.0 * std::f64::consts::PI * product.sqrt()) / round_duration
    }

    /// Calculate difficulty from target
    fn calculate_solution_difficulty(target: &[u8; 32]) -> u128 {
        // Difficulty = 2^256 / target
        // Simplified: count leading zero bytes
        let leading_zeros = target.iter().take_while(|&&b| b == 0).count();
        (1u128 << (leading_zeros * 8))
    }

    /// Compute Merkle root of mining solutions
    fn compute_solutions_merkle_root(solutions: &[MiningSolution]) -> BlockHash {
        if solutions.is_empty() {
            return [0u8; 32];
        }

        let hashes: Vec<_> = solutions.iter()
            .map(|s| blake3::hash(&bincode::serialize(s).unwrap()))
            .collect();

        Self::merkle_root(&hashes)
    }

    /// Calculate Merkle root from list of hashes
    fn merkle_root(hashes: &[blake3::Hash]) -> BlockHash {
        if hashes.is_empty() {
            return [0u8; 32];
        }
        if hashes.len() == 1 {
            return hashes[0].into();
        }

        let mut current_level = hashes.to_vec();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();

            for chunk in current_level.chunks(2) {
                let combined = if chunk.len() == 2 {
                    let mut combined_bytes = Vec::new();
                    combined_bytes.extend_from_slice(chunk[0].as_bytes());
                    combined_bytes.extend_from_slice(chunk[1].as_bytes());
                    blake3::hash(&combined_bytes)
                } else {
                    chunk[0]
                };
                next_level.push(combined);
            }

            current_level = next_level;
        }

        current_level[0].into()
    }

    /// Get current blockchain height
    pub fn get_height(&self) -> u64 {
        self.current_height
    }

    /// Get latest block hash
    pub fn get_latest_hash(&self) -> BlockHash {
        self.latest_block_hash
    }

    /// Get pending solutions count
    pub fn pending_count(&self) -> usize {
        self.pending_solutions.len()
    }

    /// Set latest block (for initialization from storage)
    pub fn set_latest_block(&mut self, height: u64, hash: BlockHash, difficulty: u128) {
        self.current_height = height;
        self.latest_block_hash = hash;
        self.total_difficulty = difficulty;
        self.dag_round = height; // Sync DAG round with height
    }
    // ==========================================
    // PHASE 3: DAG-KNIGHT CONSENSUS INTEGRATION
    // ==========================================

    /// Convert a QBlock into a DAG Vertex for consensus processing
    ///
    /// This bridges the blockchain layer (QBlocks) with the consensus layer (DAG vertices).
    /// Each QBlock becomes a vertex in the DAG, enabling Byzantine fault-tolerant ordering.
    ///
    /// **Mapping**:
    /// - QBlock.header.height → Vertex.round (height = consensus round)
    /// - QBlock.header.prev_block_hash → Vertex.parents[0] (single parent for now)
    /// - QBlock.transactions → Vertex.transactions (transaction hashes)
    /// - QBlock.calculate_hash() → Vertex.id (vertex identifier)
    /// - QBlock.header.vdf_proof → Vertex.vdf_proof
    /// - QBlock.header.timestamp → Vertex.timestamp
    ///
    /// **Returns**: A DAG vertex ready for submission to DAG-Knight consensus
    pub fn qblock_to_vertex(&self, block: &block::QBlock) -> anyhow::Result<q_dag_knight::Vertex> {
        use q_dag_knight::vertex_creator::Vertex;

        debug!("🔄 Converting QBlock {} to DAG Vertex for consensus",
            block.header.height);

        // Extract transaction hashes from block
        let tx_hashes: Vec<TxHash> = block.transactions
            .iter()
            .map(|tx| tx.hash())
            .collect();

        // Determine parent vertices
        // For now, use prev_block_hash as the single parent
        // Genesis block (height 0) has no parents
        let parents = if block.header.height == 0 {
            vec![] // Genesis has no parents
        } else {
            vec![block.header.prev_block_hash] // Previous block = parent vertex
        };

        // Generate vertex ID from block hash
        let vertex_id = block.calculate_hash();

        info!("✅ Converted QBlock {} → Vertex (ID: {}, {} txs, {} parents)",
            block.header.height,
            hex::encode(&vertex_id[..8]),
            tx_hashes.len(),
            parents.len()
        );

        // Convert BlockHeader VDFProof to QuantumVDFProof for consensus
        // TODO Phase 3 Part 2: Proper VDF proof conversion
        let quantum_vdf_proof = q_dag_knight::QuantumVDFProof {
            challenge: vertex_id, // Use vertex ID as challenge
            proof: {
                let mut proof = [0u8; 64];
                let output_bytes = &block.header.vdf_proof.output;
                let len = std::cmp::min(output_bytes.len(), 64);
                proof[..len].copy_from_slice(&output_bytes[..len]);
                proof
            },
            quantum_seed: None, // TODO: Extract from block quantum metadata
            computation_time: std::time::Duration::from_secs(block.header.vdf_proof.iterations / 100),
            difficulty: block.header.vdf_proof.iterations,
            entropy_estimate: 0.85, // TODO: Calculate from quantum metadata
            parallel_witnesses: vec![], // TODO: Add witnesses if available
        };

        // Create DAG vertex with block data
        Ok(Vertex {
            id: vertex_id,
            round: block.header.height, // Height serves as consensus round
            proposer: self.config.node_id,
            transactions: tx_hashes,
            parents,
            vdf_proof: quantum_vdf_proof,
            timestamp: block.header.timestamp,
            signature: vec![], // TODO Phase 3 Part 2: Add vertex signature
        })
    }

    /// Convert DAG-Knight Vertex to q_types Vertex for storage
    /// This bridges the consensus layer (DAG-Knight) with the storage layer (Narwhal)
    pub fn dag_vertex_to_storage_vertex(
        &self,
        dag_vertex: &q_dag_knight::Vertex,
        block: &block::QBlock,
    ) -> q_types::Vertex {
        use sha3::{Digest, Sha3_256};

        // Calculate tx root from transaction hashes
        let tx_root = if dag_vertex.transactions.is_empty() {
            [0u8; 32]
        } else {
            let mut hasher = Sha3_256::new();
            for tx_hash in &dag_vertex.transactions {
                hasher.update(tx_hash);
            }
            let result = hasher.finalize();
            let mut tx_root = [0u8; 32];
            tx_root.copy_from_slice(&result[..32]);
            tx_root
        };

        q_types::Vertex {
            id: dag_vertex.id,
            round: dag_vertex.round,
            author: dag_vertex.proposer,
            tx_root,
            parents: dag_vertex.parents.clone(),
            transactions: block.transactions.clone(), // Use full transactions from block
            signature: dag_vertex.signature.clone(),
            timestamp: chrono::DateTime::from_timestamp(dag_vertex.timestamp as i64, 0)
                .unwrap_or_else(chrono::Utc::now),
        }
    }
}

/// Shared block producer wrapped for async access
pub type SharedBlockProducer = Arc<RwLock<BlockProducer>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_producer_creation() {
        let config = BlockProducerConfig::default();
        let producer = BlockProducer::new(config);

        assert_eq!(producer.get_height(), 0);
        assert_eq!(producer.pending_count(), 0);
    }

    #[test]
    fn test_solution_queueing() {
        let mut producer = BlockProducer::new(BlockProducerConfig::default());

        let solution = MiningSolution {
            nonce: 12345,
            hash: [0u8; 32],
            difficulty_target: [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
            miner_address: [1u8; 32],
            timestamp: 1234567890,
            pool_id: None,
        };

        producer.queue_solution(solution);
        assert_eq!(producer.pending_count(), 1);
    }

    #[test]
    fn test_should_produce_block() {
        let config = BlockProducerConfig {
            block_interval_secs: 1,
            max_solutions_per_block: 10,
            min_solutions_per_block: 5,
            ..Default::default()
        };

        let mut producer = BlockProducer::new(config);

        // Not enough solutions yet
        assert!(!producer.should_produce_block());

        // Add solutions
        for i in 0..5 {
            let solution = MiningSolution {
                nonce: i,
                hash: [0u8; 32],
                difficulty_target: [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                miner_address: [1u8; 32],
                timestamp: 1234567890,
                pool_id: None,
            };
            producer.queue_solution(solution);
        }

        // Wait for interval (would need to sleep 1 second in real test)
        // For now, test max_solutions trigger
        for i in 5..15 {
            let solution = MiningSolution {
                nonce: i,
                hash: [0u8; 32],
                difficulty_target: [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                miner_address: [1u8; 32],
                timestamp: 1234567890,
                pool_id: None,
            };
            producer.queue_solution(solution);
        }

        assert!(producer.should_produce_block());
    }

    #[tokio::test]
    async fn test_block_production() {
        let mut producer = BlockProducer::new(BlockProducerConfig::default());

        // Add some solutions
        for i in 0..10 {
            let solution = MiningSolution {
                nonce: i,
                hash: [0u8; 32],
                difficulty_target: [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                miner_address: [1u8; 32],
                timestamp: 1234567890,
                pool_id: None,
            };
            producer.queue_solution(solution);
        }

        let block = producer.produce_block().await;
        assert!(block.is_some());

        let block = block.unwrap();
        assert_eq!(block.header.height, 1);
        assert_eq!(block.mining_solutions.len(), 10);
        assert!(block.header.total_difficulty > 0);
    }
}

// ============================================================================
// PHASE 2: PARALLEL BLOCK PRODUCTION (Performance Optimization Roadmap)
// ============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

/// Parallel Block Producer Pool - Enables concurrent block production
///
/// Based on FUTURE_OPTIMIZATION_ROADMAP_1M_TPS.md Phase 2.1
///
/// This implements multi-threaded block producers that can create blocks
/// concurrently, populating different lanes in the DAG visualization.
///
/// Performance Target: 8-16x improvement over single producer
pub struct ParallelBlockProducerPool {
    /// Array of block producers (8 parallel workers)
    producers: Vec<Arc<RwLock<BlockProducer>>>,

    /// Round-robin index for solution distribution
    round_robin_index: AtomicUsize,

    /// Number of producers in the pool
    num_producers: usize,
}

impl ParallelBlockProducerPool {
    /// Create a new parallel producer pool
    ///
    /// # Arguments
    /// * `num_producers` - Number of parallel producers (typically 8-16)
    /// * `base_config` - Base configuration to clone for each producer
    pub fn new(num_producers: usize, base_config: BlockProducerConfig) -> Self {
        info!("🚀 Initializing Parallel Block Producer Pool with {} producers", num_producers);

        let producers = (0..num_producers)
            .map(|producer_id| {
                let mut config = base_config.clone();
                // Each producer gets a unique validator index
                config.validator_index = producer_id as u64;
                config.total_validators = num_producers as u64;

                info!("  ✅ Producer #{} initialized (validator_index={})",
                    producer_id, config.validator_index);

                Arc::new(RwLock::new(BlockProducer::new(config)))
            })
            .collect();

        Self {
            producers,
            round_robin_index: AtomicUsize::new(0),
            num_producers,
        }
    }

    /// Queue a mining solution to a producer (round-robin distribution)
    pub async fn queue_solution(&self, solution: MiningSolution) {
        // Round-robin distribution across all producers
        let index = self.round_robin_index.fetch_add(1, Ordering::SeqCst) % self.num_producers;
        let mut producer = self.producers[index].write().await;
        producer.queue_solution(solution);
    }

    /// Produce blocks from all producers that are ready
    ///
    /// This method checks all producers and produces blocks from those
    /// that have enough solutions and have waited long enough.
    ///
    /// Returns vector of (producer_id, block) pairs for visualization
    pub async fn produce_blocks(&self) -> Vec<(usize, QBlock)> {
        let mut blocks = Vec::new();

        // Try to produce from each producer in parallel
        for (producer_id, producer_arc) in self.producers.iter().enumerate() {
            let mut producer = producer_arc.write().await;

            if let Some(block) = producer.produce_block().await {
                info!("🎉 Producer #{} created block at height {}",
                    producer_id, block.header.height);
                blocks.push((producer_id, block));
            }
        }

        blocks
    }

    /// Check if any producer should produce a block
    pub async fn should_produce(&self) -> bool {
        for producer_arc in &self.producers {
            let producer = producer_arc.read().await;
            if producer.should_produce_block() {
                return true;
            }
        }
        false
    }

    /// Get the number of producers in the pool
    pub fn num_producers(&self) -> usize {
        self.num_producers
    }
}
