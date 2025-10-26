/// Q-NarwhalKnight Block Types for DAG-Knight Consensus
/// Comprehensive block structure integrating:
/// - Mining proof-of-work solutions
/// - DAG vertex references
/// - Quantum consensus metadata
/// - VDF-based anchor election

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Block hash type (blake3)
pub type BlockHash = [u8; 32];

/// DAG round number
pub type DagRound = u64;

/// Complete Q-NarwhalKnight block with quantum consensus integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QBlock {
    /// Block header containing hashes and metadata
    pub header: BlockHeader,

    /// Mining proof-of-work solutions included in this block
    pub mining_solutions: Vec<MiningSolution>,

    /// DAG vertex parent references (for DAG-Knight ordering)
    pub dag_parents: Vec<super::VertexId>,

    /// Quantum consensus metadata
    pub quantum_metadata: QuantumMetadata,

    /// Transactions included in this block
    pub transactions: Vec<super::Transaction>,

    /// Block size in bytes (for performance monitoring)
    pub size_bytes: usize,
}

/// Block header with all cryptographic commitments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    /// Block height (monotonically increasing, Bitcoin-style chain)
    pub height: u64,

    /// Previous block hash (forms Bitcoin-style chain backbone)
    pub prev_block_hash: BlockHash,

    /// Merkle root of all mining solutions
    pub solutions_root: BlockHash,

    /// Merkle root of all transactions
    pub tx_root: BlockHash,

    /// State root (world state after applying this block)
    pub state_root: BlockHash,

    /// Block creation timestamp (Unix epoch seconds)
    pub timestamp: u64,

    /// DAG round number (for DAG-Knight consensus ordering)
    pub dag_round: DagRound,

    /// Quantum VDF proof for anchor election
    pub vdf_proof: VDFProof,

    /// Anchor validator elected for this round (PeerId as string)
    pub anchor_validator: Option<String>,

    /// Block proposer (validator who created this block)
    pub proposer: super::NodeId,

    /// Total difficulty accumulated to this block
    pub total_difficulty: u128,
}

/// Quantum VDF (Verifiable Delay Function) proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VDFProof {
    /// VDF output (deterministic function of input + time)
    pub output: Vec<u8>,

    /// Wesolowski verification proof (2048x speedup)
    pub verification_proof: Vec<u8>,

    /// Number of sequential iterations (time parameter)
    pub iterations: u64,

    /// Challenge input (previous block hash + quantum seed)
    pub challenge: Vec<u8>,

    /// Proof generation timestamp
    pub generated_at: u64,
}

/// Mining proof-of-work solution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MiningSolution {
    /// Nonce that produces valid hash
    pub nonce: u64,

    /// Resulting hash (must meet difficulty target)
    pub hash: [u8; 32],

    /// Difficulty target this solution meets
    pub difficulty_target: [u8; 32],

    /// Miner wallet address (receives reward)
    pub miner_address: [u8; 32],

    /// Solution submission timestamp
    pub timestamp: u64,

    /// Optional: Mining pool information
    pub pool_id: Option<String>,
}

/// Quantum consensus metadata (Q-NarwhalKnight innovations)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetadata {
    /// 5D hypergraph vertex coordinates
    pub vertex_coordinates: HypergraphCoordinates,

    /// Kristensen K-parameter (phase transition metric)
    pub k_parameter: f64,

    /// Total energy functional value (minimized via gradient descent)
    pub energy: f64,

    /// Energy components breakdown
    pub energy_components: EnergyComponents,

    /// Spectral BFT signatures from validators
    pub spectral_signatures: Vec<SpectralSignature>,

    /// String-theoretic wavefunction phase
    pub wavefunction_phase: f64,

    /// Entropy variance (for K-parameter calculation)
    pub entropy_variance: f64,

    /// Byzantine node detection scores
    pub byzantine_scores: HashMap<String, f64>, // PeerId -> deviation score
}

/// 5D Hypergraph coordinates (quantum vertex positioning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphCoordinates {
    /// Temporal dimension (round number, causal ordering)
    pub temporal: f64,

    /// Spatial dimensions (network topology, RTT-based positioning)
    pub spatial: Vec<f64>, // Typically 3D: [x, y, z]

    /// Energetic dimension (stake weight, transaction fees)
    pub energetic: f64,

    /// Entropic dimension (quantum randomness from VDF)
    pub entropic: f64,

    /// Metadata dimensions (ZK-proofs, oracle data, etc.)
    pub metadata: HashMap<String, f64>,
}

/// Energy functional components (for physics-based consensus)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyComponents {
    /// Coupling energy: Σ J_ij |ψ_i - ψ_j|² (phase alignment penalty)
    pub coupling: f64,

    /// Potential energy: V(state) (validator stake weights)
    pub potential: f64,

    /// Ordering energy: temporal causality constraints
    pub ordering: f64,

    /// Fault tolerance energy: Byzantine deviation penalty
    pub fault_tolerance: f64,

    /// Temporal energy: time-based decay function
    pub temporal: f64,

    /// Finality energy: commitment barrier (prevents rollbacks)
    pub finality: f64,
}

/// Spectral BFT signature (quantum-enhanced Byzantine detection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralSignature {
    /// Validator public key
    pub validator: super::NodeId,

    /// Classical signature (Ed25519 in Phase 0, Dilithium in Phase 1)
    pub classical_sig: Vec<u8>,

    /// Spectral decomposition coefficient (for Byzantine detection)
    pub spectral_coefficient: f64,

    /// Phase deviation from consensus (3-sigma threshold)
    pub phase_deviation: f64,

    /// Signature timestamp
    pub timestamp: u64,
}

/// Block finality status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinalityStatus {
    /// Block is pending consensus
    Pending,

    /// Block is in consensus but not yet finalized
    InConsensus { round: DagRound },

    /// Block is finalized (committed via DAG-Knight 2f+1 rule)
    Finalized { commit_round: DagRound },

    /// Block is orphaned (not part of canonical chain)
    Orphaned,
}

/// Extended block with finality information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalizedBlock {
    /// The block itself
    pub block: QBlock,

    /// Finality status
    pub finality_status: FinalityStatus,

    /// Finality certificate (2f+1 signatures)
    pub finality_cert: Option<FinalityCertificate>,

    /// Confirmation count (number of blocks building on top)
    pub confirmations: u64,
}

/// Finality certificate (DAG-Knight 2f+1 commit proof)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalityCertificate {
    /// Block hash being committed
    pub block_hash: BlockHash,

    /// Commit round (when 2f+1 threshold reached)
    pub commit_round: DagRound,

    /// Validator signatures proving commitment
    pub validator_signatures: HashMap<String, Vec<u8>>, // PeerId -> signature

    /// Total stake weight of signers
    pub total_stake: u64,

    /// Byzantine fault tolerance threshold reached
    pub bft_threshold_met: bool,

    /// Merkle proof of commit path in DAG
    pub commit_path_proof: Vec<BlockHash>,
}

impl QBlock {
    /// Calculate block hash (blake3 of header)
    pub fn calculate_hash(&self) -> BlockHash {
        let header_bytes = bincode::serialize(&self.header).expect("Failed to serialize header");
        blake3::hash(&header_bytes).into()
    }

    /// Verify block integrity
    pub fn verify(&self) -> Result<(), String> {
        // 1. Verify mining solutions meet difficulty
        for solution in &self.mining_solutions {
            if !Self::verify_difficulty(&solution.hash, &solution.difficulty_target) {
                return Err(format!("Mining solution nonce {} does not meet difficulty", solution.nonce));
            }
        }

        // 2. Verify solutions Merkle root
        let computed_root = Self::compute_solutions_merkle_root(&self.mining_solutions);
        if computed_root != self.header.solutions_root {
            return Err("Solutions Merkle root mismatch".to_string());
        }

        // 3. Verify transactions Merkle root
        let computed_tx_root = Self::compute_tx_merkle_root(&self.transactions);
        if computed_tx_root != self.header.tx_root {
            return Err("Transaction Merkle root mismatch".to_string());
        }

        // 4. Verify timestamp is reasonable (not too far in future)
        let now = chrono::Utc::now().timestamp() as u64;
        if self.header.timestamp > now + 300 {
            return Err("Block timestamp too far in future".to_string());
        }

        Ok(())
    }

    /// Verify mining difficulty
    fn verify_difficulty(hash: &[u8; 32], target: &[u8; 32]) -> bool {
        hash < target
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

    /// Compute Merkle root of transactions
    fn compute_tx_merkle_root(transactions: &[super::Transaction]) -> BlockHash {
        if transactions.is_empty() {
            return [0u8; 32];
        }

        let hashes: Vec<_> = transactions.iter()
            .map(|tx| blake3::hash(&bincode::serialize(tx).unwrap()))
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

    /// Get block size estimate
    pub fn estimate_size(&self) -> usize {
        bincode::serialize(self)
            .map(|bytes| bytes.len())
            .unwrap_or(0)
    }
}

impl HypergraphCoordinates {
    /// Calculate Euclidean distance between two vertices in 5D space
    pub fn distance(&self, other: &HypergraphCoordinates) -> f64 {
        let temporal_dist = (self.temporal - other.temporal).powi(2);

        let spatial_dist: f64 = self.spatial.iter()
            .zip(&other.spatial)
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        let energetic_dist = (self.energetic - other.energetic).powi(2);
        let entropic_dist = (self.entropic - other.entropic).powi(2);

        (temporal_dist + spatial_dist + energetic_dist + entropic_dist).sqrt()
    }

    /// Create coordinates from block data
    pub fn from_block_data(
        round: u64,
        solutions_count: usize,
        total_difficulty: u128,
        quantum_entropy: f64,
    ) -> Self {
        Self {
            temporal: round as f64,
            spatial: vec![
                (solutions_count as f64).sqrt(), // x: mining activity
                (total_difficulty as f64).log10(), // y: difficulty growth
                quantum_entropy, // z: randomness
            ],
            energetic: total_difficulty as f64,
            entropic: quantum_entropy,
            metadata: HashMap::new(),
        }
    }
}

impl Default for VDFProof {
    fn default() -> Self {
        Self {
            output: vec![],
            verification_proof: vec![],
            iterations: 100,
            challenge: vec![],
            generated_at: chrono::Utc::now().timestamp() as u64,
        }
    }
}

impl Default for QuantumMetadata {
    fn default() -> Self {
        Self {
            vertex_coordinates: HypergraphCoordinates {
                temporal: 0.0,
                spatial: vec![0.0, 0.0, 0.0],
                energetic: 0.0,
                entropic: 0.0,
                metadata: HashMap::new(),
            },
            k_parameter: 0.0,
            energy: 0.0,
            energy_components: EnergyComponents::default(),
            spectral_signatures: vec![],
            wavefunction_phase: 0.0,
            entropy_variance: 0.0,
            byzantine_scores: HashMap::new(),
        }
    }
}

impl Default for EnergyComponents {
    fn default() -> Self {
        Self {
            coupling: 0.0,
            potential: 0.0,
            ordering: 0.0,
            fault_tolerance: 0.0,
            temporal: 0.0,
            finality: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_hash_calculation() {
        let block = QBlock {
            header: BlockHeader {
                height: 1,
                prev_block_hash: [0u8; 32],
                solutions_root: [0u8; 32],
                tx_root: [0u8; 32],
                state_root: [0u8; 32],
                timestamp: 1234567890,
                dag_round: 1,
                vdf_proof: VDFProof::default(),
                anchor_validator: None,
                proposer: [1u8; 32],
                total_difficulty: 1000,
            },
            mining_solutions: vec![],
            dag_parents: vec![],
            quantum_metadata: QuantumMetadata::default(),
            transactions: vec![],
            size_bytes: 0,
        };

        let hash = block.calculate_hash();
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_hypergraph_distance() {
        let coord1 = HypergraphCoordinates {
            temporal: 1.0,
            spatial: vec![1.0, 2.0, 3.0],
            energetic: 100.0,
            entropic: 0.5,
            metadata: HashMap::new(),
        };

        let coord2 = HypergraphCoordinates {
            temporal: 2.0,
            spatial: vec![2.0, 3.0, 4.0],
            energetic: 200.0,
            entropic: 0.7,
            metadata: HashMap::new(),
        };

        let distance = coord1.distance(&coord2);
        assert!(distance > 0.0);
    }

    #[test]
    fn test_mining_difficulty_verification() {
        let easy_hash = [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                         0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                         0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                         0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

        let target = [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

        assert!(QBlock::verify_difficulty(&easy_hash, &target));
    }
}
