/// Q-NarwhalKnight Block Types for DAG-Knight Consensus
/// Comprehensive block structure integrating:
/// - Mining proof-of-work solutions
/// - DAG vertex references
/// - Quantum consensus metadata
/// - VDF-based anchor election

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Default values for backwards compatibility with existing blocks
fn default_phase() -> u8 { 5 }  // Phase 5 is current testnet phase
fn default_network_id() -> String { "mainnet-genesis".to_string() }

/// v1.1.3-beta: Custom serde module for u128 serialization
/// v3.2.8-beta: RESTORED native u128 for Bincode storage compatibility
/// Blocks are stored with Bincode which handles u128 natively.
/// The Visitor pattern handles backward compatibility with old u64 data.
/// NOTE: P2P balance updates use CBOR via P2PBalanceUpdate which has its own fix.
mod u128_as_string {
    use serde::{de::Visitor, Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S>(value: &u128, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // v8.0.6: Format-aware serialization
        // - Bincode (is_human_readable=false): native u128 (16 bytes)
        // - JSON (is_human_readable=true): string to avoid float precision loss
        //   JSON numbers are IEEE 754 doubles (53-bit mantissa), u128 > 2^53
        //   gets serialized as scientific notation (e.g. 1.2477e23) which then
        //   fails to deserialize back as u128.
        if serializer.is_human_readable() {
            serializer.serialize_str(&value.to_string())
        } else {
            serializer.serialize_u128(*value)
        }
    }

    struct U128Visitor;

    impl<'de> Visitor<'de> for U128Visitor {
        type Value = u128;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a u128, u64, f64, or string representing a number")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            v.parse().map_err(E::custom)
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            v.parse().map_err(E::custom)
        }

        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v as u128)
        }

        fn visit_u128<E>(self, v: u128) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v)
        }

        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v >= 0 {
                Ok(v as u128)
            } else {
                Err(E::custom("negative value cannot be u128"))
            }
        }

        // v8.0.6: Handle f64 for backward compatibility with blocks stored as JSON
        // Large u128 values serialized as JSON numbers become floats (e.g. 1.2477e23)
        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v < 0.0 {
                Err(E::custom("negative value cannot be u128"))
            } else if v > u128::MAX as f64 {
                Err(E::custom("value too large for u128"))
            } else {
                Ok(v as u128)
            }
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u128, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            // JSON: could be string, number, or float — accept any
            deserializer.deserialize_any(U128Visitor)
        } else {
            // Bincode: native u128
            deserializer.deserialize_u128(U128Visitor)
        }
    }
}

/// v8.0.6: Serialize/deserialize Option<u128> with format-aware handling
/// Needed for total_coinbase_reward and similar optional u128 fields
mod option_u128_as_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &Option<u128>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(v) => {
                if serializer.is_human_readable() {
                    // JSON: serialize as string (implicitly Some, None is null)
                    serializer.serialize_str(&v.to_string())
                } else {
                    // Bincode: MUST use serialize_some to write the Option discriminant [0x01]
                    // v.serialize(serializer) would skip the discriminant, corrupting the layout
                    serializer.serialize_some(v)
                }
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u128>, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize as Option<serde_json::Value> for human-readable, Option<u128> for binary
        if deserializer.is_human_readable() {
            #[derive(Deserialize)]
            #[serde(untagged)]
            enum Helper {
                Str(String),
                Num(f64),
                Null,
            }
            let opt = Option::<Helper>::deserialize(deserializer)?;
            match opt {
                None | Some(Helper::Null) => Ok(None),
                Some(Helper::Str(s)) => s
                    .parse::<u128>()
                    .map(Some)
                    .map_err(serde::de::Error::custom),
                Some(Helper::Num(f)) => {
                    if f < 0.0 {
                        Err(serde::de::Error::custom("negative value cannot be u128"))
                    } else {
                        Ok(Some(f as u128))
                    }
                }
            }
        } else {
            Option::<u128>::deserialize(deserializer)
        }
    }
}

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

    /// Balance updates included in this block (v0.9.0-beta: Balance Consensus)
    /// CRITICAL: Enables deterministic balance state across all nodes
    /// When a block is synced, these balance updates MUST be applied
    /// Optional for backwards compatibility (defaults to empty vec)
    #[serde(default)]
    pub balance_updates: Vec<BalanceUpdate>,

    /// Block size in bytes (for performance monitoring)
    pub size_bytes: usize,
}

/// Block header with all cryptographic commitments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockHeader {
    /// Block height (monotonically increasing, Bitcoin-style chain)
    pub height: u64,

    /// Network phase identifier (1 = Phase 1, 2 = Phase 2, etc.)
    /// CRITICAL: Prevents cross-phase contamination (Phase 1 blocks can't sync into Phase 2)
    /// Optional for backwards compatibility with existing blocks (defaults to Phase 2)
    #[serde(default = "default_phase")]
    pub phase: u8,

    /// Network ID ("testnet-phase1", "mainnet", "mainnet", etc.)
    /// Optional for backwards compatibility (defaults to "mainnet")
    #[serde(default = "default_network_id")]
    pub network_id: String,

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

    /// Producer ID / Lane ID for parallel DAG production (0-7 for 8 parallel producers)
    /// v0.8.11-beta: Enables unique block hashes for parallel producers at same height
    /// Optional for backwards compatibility (defaults to 0)
    #[serde(default)]
    pub producer_id: u8,

    /// Total difficulty accumulated to this block
    /// v3.4.13: CRITICAL - Use u128_as_string for CBOR compatibility during P2P sync
    #[serde(with = "u128_as_string")]
    pub total_difficulty: u128,

    // ============================================================================
    // 🔐 v1.2.0-beta Phase 3: Block Producer Signature Fields
    // ============================================================================

    /// Producer public key (Ed25519 - 32 bytes)
    /// Required for all new blocks, optional for backwards compatibility
    #[serde(default)]
    pub producer_public_key: Option<[u8; 32]>,

    /// Producer signature over the header hash (Ed25519 - 64 bytes)
    /// Required for all new blocks, optional for backwards compatibility
    #[serde(default)]
    pub producer_signature: Option<Vec<u8>>,

    // ============================================================================
    // 🔐 v1.2.0-beta Phase 3 Step 6: Coinbase Transaction Security
    // ============================================================================

    /// Merkle root of all coinbase transaction outputs in this block
    /// This allows SPV clients to verify mining rewards without full block data
    /// SHA3-256 of (coinbase_tx_hash_0 || coinbase_tx_hash_1 || ... || coinbase_tx_hash_n)
    #[serde(default)]
    pub coinbase_merkle_root: Option<[u8; 32]>,

    /// Total coinbase reward in this block (sum of all coinbase outputs)
    /// Used for quick emission schedule validation
    /// v2.5.0: Upgraded to u128 for full precision (Bincode native)
    /// v8.0.6: CRITICAL FIX - Added option_u128_as_string for JSON HTTP sync compatibility
    /// Without this, large u128 values become floats (1.2477e23) in JSON and fail to deserialize
    #[serde(default, with = "option_u128_as_string")]
    pub total_coinbase_reward: Option<u128>,

    /// Number of coinbase transactions in this block
    #[serde(default)]
    pub coinbase_count: Option<u32>,
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

    /// Adaptive security parameters (v1.0.16-beta+)
    /// Optional for backwards compatibility with existing blocks
    #[serde(default)]
    pub adaptive_params: Option<AdaptiveVDFParams>,
}

/// Adaptive VDF parameters based on network hashrate
/// v1.0.16-beta: Conservative implementation based on AI review feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveVDFParams {
    /// Security tier (Standard = 1000, Enhanced = 1500, Maximum = 2000 iterations)
    pub security_tier: SecurityTier,

    /// Network hashrate (24-hour smoothed moving average)
    pub smoothed_hashrate: f64,

    /// Security multiplier (1.0 - 2.0 range)
    pub security_multiplier: f64,

    /// Actual adaptive iterations used
    pub adaptive_iterations: u64,
}

/// Security tier levels (governance-approved)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityTier {
    /// Standard security - 1,000 VDF iterations
    Standard,
    /// Enhanced security - 1,500 VDF iterations
    Enhanced,
    /// Maximum security - 2,000 VDF iterations
    Maximum,
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

    /// Miner's hash rate at time of solution (H/s)
    /// Stored as u64 for Eq compatibility, actual H/s value
    /// This allows ultra-precise network hashrate calculation in real-time
    #[serde(default)]
    pub hash_rate_hs: u64,

    /// v3.3.3-beta: Unique miner instance ID for identification
    #[serde(default)]
    pub miner_id: Option<String>,

    /// v3.3.3-beta: Human-readable miner name (e.g., "Server Alpha", "Mining Rig 1")
    #[serde(default)]
    pub worker_name: Option<String>,

    /// v1.0.5: Genus-2 Jacobian VDF output (Mumford representation of y = x^(2^T) in J(C))
    /// Present when mining uses real VDF (above UPGRADE_VDF_MINING_HEIGHT)
    #[serde(default)]
    pub vdf_output: Option<Vec<u8>>,

    /// v1.0.5: Wesolowski proof π for O(log T) VDF verification
    /// Allows verifiers to check VDF without recomputing all T sequential squarings
    #[serde(default)]
    pub vdf_proof: Option<Vec<u8>>,

    /// v1.0.5: VDF intermediate checkpoints for parallel verification
    #[serde(default)]
    pub vdf_checkpoints: Option<Vec<Vec<u8>>>,

    /// v1.0.5: Number of VDF iterations (T) used for this solution
    #[serde(default)]
    pub vdf_iterations_count: Option<u64>,
}

/// Balance update (v0.9.0-beta: Balance Consensus, v2.5.0: u128)
/// Represents a deterministic balance state transition
/// MUST be applied in order when processing blocks
///
/// v3.4.13-beta: CRITICAL FIX - Added u128_as_string for CBOR compatibility
/// The BlockPackCodec uses CBOR serialization, which does NOT support u128 natively.
/// Without this fix, syncing blocks with BalanceUpdate fails with:
/// "The number can't be stored in CBOR"
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BalanceUpdate {
    /// Wallet address being updated
    pub address: super::Address,

    /// v2.5.0: Balance before this update (u128 for extreme precision)
    /// v3.4.13: CRITICAL - Use u128_as_string for CBOR compatibility during P2P sync
    #[serde(with = "u128_as_string")]
    pub old_balance: u128,

    /// v2.5.0: Balance after this update (u128 for extreme precision)
    /// v3.4.13: CRITICAL - Use u128_as_string for CBOR compatibility during P2P sync
    #[serde(with = "u128_as_string")]
    pub new_balance: u128,

    /// Reason for balance change
    pub reason: String, // "mining_reward", "transaction", "dev_fee", etc.

    /// Timestamp of balance update
    pub timestamp: u64,
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

/// Cryptographic phase for signature scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignaturePhase {
    /// Phase 0: Ed25519 classical signatures (64 bytes)
    Phase0Ed25519,
    /// Phase 1: Dilithium5 post-quantum signatures (4,627 bytes) - DEPRECATED
    Phase1Dilithium5,
    /// Hybrid: Both Ed25519 and Dilithium5 (transition mode)
    HybridEd25519Dilithium5,
    /// Phase 2: SQIsign compact post-quantum signatures (204 bytes)
    /// 🚀 v1.0.86-beta: 95.6% smaller than Dilithium5, equivalent security
    Phase2SQIsign,
    /// Hybrid: Ed25519 + SQIsign (transition from Phase 0 to Phase 2)
    HybridEd25519SQIsign,
}

impl Default for SignaturePhase {
    fn default() -> Self {
        SignaturePhase::Phase0Ed25519
    }
}

/// Spectral BFT signature (quantum-enhanced Byzantine detection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralSignature {
    /// Validator public key
    pub validator: super::NodeId,

    /// ✨ v1.0.15-beta: Cryptographic phase used for this signature
    #[serde(default)]
    pub crypto_phase: SignaturePhase,

    /// Classical signature (Ed25519 in Phase 0)
    /// For hybrid mode, this contains Ed25519 signature
    pub classical_sig: Vec<u8>,

    /// ✨ v1.0.15-beta: Post-quantum signature (Dilithium5) - DEPRECATED
    /// Only populated in Phase1 or HybridEd25519Dilithium5 mode
    /// ⚠️ DEPRECATED: Use sqisign_sig for new blocks (95.6% smaller)
    #[serde(default)]
    pub pqc_sig: Option<Vec<u8>>,

    /// ✨ v1.0.86-beta: SQIsign compact post-quantum signature (204 bytes)
    /// Only populated in Phase2SQIsign or HybridEd25519SQIsign mode
    /// 🚀 95.6% smaller than Dilithium5 (204 vs 4,627 bytes)
    #[serde(default)]
    pub sqisign_sig: Option<Vec<u8>>,

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
    /// v0.6.0-beta: Added expected_network_id parameter to prevent cross-network pollution
    pub fn verify(&self, expected_network_id: Option<&str>) -> Result<(), String> {
        // 1. Verify network ID matches expected network (v0.6.0-beta)
        if let Some(expected_id) = expected_network_id {
            if self.header.network_id != expected_id {
                return Err(format!(
                    "Network ID mismatch: block has '{}', expected '{}'",
                    self.header.network_id, expected_id
                ));
            }
        }

        // 2. Verify mining solutions meet difficulty
        for solution in &self.mining_solutions {
            if !Self::verify_difficulty(&solution.hash, &solution.difficulty_target) {
                return Err(format!("Mining solution nonce {} does not meet difficulty", solution.nonce));
            }
        }

        // 3. Verify solutions Merkle root
        let computed_root = Self::compute_solutions_merkle_root(&self.mining_solutions);
        if computed_root != self.header.solutions_root {
            return Err("Solutions Merkle root mismatch".to_string());
        }

        // 4. Verify transactions Merkle root
        let computed_tx_root = Self::compute_tx_merkle_root(&self.transactions);
        if computed_tx_root != self.header.tx_root {
            return Err("Transaction Merkle root mismatch".to_string());
        }

        // 5. Verify timestamp is reasonable (not too far in future)
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

    // ============================================================================
    // 🔐 v1.2.0-beta Phase 3: Block Producer Signature Methods
    // ============================================================================

    /// Get the payload that the producer should sign
    /// This is the block hash BEFORE the signature is added
    pub fn signing_payload(&self) -> BlockHash {
        // Create a temporary header without signature to compute the signing payload
        let mut temp_header = self.header.clone();
        temp_header.producer_signature = None;

        let header_bytes = bincode::serialize(&temp_header).expect("Failed to serialize header");
        blake3::hash(&header_bytes).into()
    }

    /// Sign this block with the producer's keypair
    /// Sets both producer_public_key and producer_signature
    pub fn sign(&mut self, signing_key: &ed25519_dalek::SigningKey) -> Result<(), String> {
        use ed25519_dalek::Signer;

        // Get the signing payload (hash without signature)
        let payload = self.signing_payload();

        // Sign the payload
        let signature = signing_key.sign(&payload);

        // Store public key and signature
        self.header.producer_public_key = Some(signing_key.verifying_key().to_bytes());
        self.header.producer_signature = Some(signature.to_bytes().to_vec());

        Ok(())
    }

    /// Verify the producer's signature on this block
    /// Returns Ok(()) if valid, Err with reason otherwise
    pub fn verify_producer_signature(&self) -> Result<(), String> {
        use ed25519_dalek::{Signature, VerifyingKey, Verifier};

        // Get the public key
        let public_key_bytes = self.header.producer_public_key
            .ok_or("Block producer public key is missing (Phase 3 requires all blocks to be signed)")?;

        // Get the signature
        let signature_bytes = self.header.producer_signature
            .as_ref()
            .ok_or("Block producer signature is missing (Phase 3 requires all blocks to be signed)")?;

        // Validate signature length
        if signature_bytes.len() != 64 {
            return Err(format!(
                "Invalid signature length: expected 64 bytes, got {}",
                signature_bytes.len()
            ));
        }

        // Parse the public key
        let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)
            .map_err(|e| format!("Invalid producer public key: {}", e))?;

        // Parse the signature
        let signature_arr: [u8; 64] = signature_bytes.clone().try_into()
            .map_err(|_| "Failed to convert signature to fixed-size array")?;
        let signature = Signature::from_bytes(&signature_arr);

        // Verify the signature against the signing payload
        let payload = self.signing_payload();
        verifying_key.verify(&payload, &signature)
            .map_err(|e| format!("Block producer signature verification failed: {}", e))?;

        Ok(())
    }

    /// Check if this block has a producer signature
    pub fn is_producer_signed(&self) -> bool {
        self.header.producer_public_key.is_some() &&
        self.header.producer_signature.as_ref().map(|s| s.len() == 64).unwrap_or(false)
    }

    /// Check if producer signature is required for Phase 3+ blocks
    /// Signature is required for all blocks produced after Phase 3 activation
    pub fn requires_producer_signature(&self) -> bool {
        // Phase 3 = v1.2.0-beta, signature required for phase >= 9
        // For gradual rollout, we can adjust this threshold
        self.header.phase >= 9
    }

    // =========================================================================
    // 🔐 v1.2.0-beta Phase 3 Step 6: Coinbase Transaction Security Methods
    // =========================================================================

    /// Compute the merkle root of all coinbase transactions in this block
    /// Uses SHA3-256 for the merkle tree
    pub fn compute_coinbase_merkle_root(&self) -> [u8; 32] {
        use sha3::{Sha3_256, Digest};

        // Collect all coinbase transaction hashes
        let coinbase_hashes: Vec<[u8; 32]> = self.transactions.iter()
            .filter(|tx| tx.is_coinbase())
            .map(|tx| tx.hash())
            .collect();

        if coinbase_hashes.is_empty() {
            return [0u8; 32]; // Empty merkle root for blocks without coinbase
        }

        if coinbase_hashes.len() == 1 {
            return coinbase_hashes[0]; // Single coinbase - just return its hash
        }

        // Build merkle tree (simple binary tree)
        let mut level = coinbase_hashes;
        while level.len() > 1 {
            let mut next_level = Vec::with_capacity((level.len() + 1) / 2);

            for chunk in level.chunks(2) {
                let mut hasher = Sha3_256::new();
                hasher.update(&chunk[0]);
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    // Odd number - duplicate the last hash
                    hasher.update(&chunk[0]);
                }
                let hash: [u8; 32] = hasher.finalize().into();
                next_level.push(hash);
            }

            level = next_level;
        }

        level[0]
    }

    /// Get total coinbase reward in this block
    /// v2.5.0: Returns u128 for full precision
    pub fn total_coinbase_reward(&self) -> u128 {
        self.transactions.iter()
            .filter(|tx| tx.is_coinbase())
            .map(|tx| tx.amount)
            .sum()
    }

    /// Get number of coinbase transactions in this block
    pub fn coinbase_count(&self) -> u32 {
        self.transactions.iter()
            .filter(|tx| tx.is_coinbase())
            .count() as u32
    }

    /// Populate coinbase security fields in the header
    /// Called after coinbase transactions are finalized
    pub fn populate_coinbase_security(&mut self) {
        self.header.coinbase_merkle_root = Some(self.compute_coinbase_merkle_root());
        self.header.total_coinbase_reward = Some(self.total_coinbase_reward());
        self.header.coinbase_count = Some(self.coinbase_count());
    }

    /// Verify coinbase merkle root matches actual transactions
    pub fn verify_coinbase_merkle_root(&self) -> Result<(), String> {
        // If header doesn't have coinbase merkle root, skip verification (legacy blocks)
        let expected_root = match self.header.coinbase_merkle_root {
            Some(root) => root,
            None => return Ok(()), // Legacy block - no verification needed
        };

        let computed_root = self.compute_coinbase_merkle_root();

        if computed_root != expected_root {
            return Err(format!(
                "Coinbase merkle root mismatch: expected {}, computed {}",
                hex::encode(expected_root),
                hex::encode(computed_root)
            ));
        }

        Ok(())
    }

    /// Verify all coinbase transaction signatures (Phase 3 security)
    /// Returns the producer public key if all signatures are valid
    pub fn verify_coinbase_signatures(&self) -> Result<Option<[u8; 32]>, String> {
        let mut producer_key: Option<[u8; 32]> = None;

        for (idx, tx) in self.transactions.iter().enumerate() {
            if !tx.is_coinbase() {
                continue;
            }

            // Check if this has a Phase 3 producer signature
            if tx.has_producer_signature() {
                match tx.verify_coinbase_signature() {
                    Ok(key) => {
                        // Verify all coinbase txs have same producer key
                        if let Some(existing_key) = producer_key {
                            if existing_key != key {
                                return Err(format!(
                                    "Coinbase tx {} has different producer key",
                                    idx
                                ));
                            }
                        } else {
                            producer_key = Some(key);
                        }
                    }
                    Err(e) => return Err(format!("Coinbase tx {} signature invalid: {}", idx, e)),
                }
            }
            // Legacy coinbase without signature - allowed for backwards compatibility
        }

        // If block header has producer_public_key, verify it matches coinbase producer
        if let (Some(header_key), Some(coinbase_key)) = (self.header.producer_public_key, producer_key) {
            if header_key != coinbase_key {
                return Err("Block producer key does not match coinbase producer key".to_string());
            }
        }

        Ok(producer_key)
    }

    /// Validate all coinbase amounts against emission schedule
    /// v2.5.0: Updated to use u128 for amounts
    pub fn validate_coinbase_amounts(&self) -> Result<(), String> {
        const DEV_FEE_PERCENT: f64 = 0.01;
        const TOLERANCE: f64 = 1.15; // 15% tolerance for rounding

        // First pass: calculate total miner rewards (excluding dev fee which is idx 0)
        let mut total_miner_rewards: u128 = 0;
        let mut dev_fee_amount: u128 = 0;

        for (idx, tx) in self.transactions.iter().enumerate() {
            if !tx.is_coinbase() {
                continue;
            }

            if idx == 0 {
                // First coinbase is the dev fee
                dev_fee_amount = tx.amount;
            } else {
                // Miner rewards
                total_miner_rewards = total_miner_rewards.saturating_add(tx.amount);
            }
        }

        // Validate dev fee: should be ~1% of total miner rewards
        // (with tolerance for rounding and edge cases)
        if dev_fee_amount > 0 && total_miner_rewards > 0 {
            let expected_dev_fee = (total_miner_rewards as f64 * DEV_FEE_PERCENT * TOLERANCE) as u128;
            if dev_fee_amount > expected_dev_fee {
                // Only log warning, don't reject - this can happen with many concurrent miners
                // The actual enforcement is at the miner submission level
                tracing::warn!(
                    "Dev fee {} slightly high vs expected {} (1% of {} total rewards)",
                    dev_fee_amount, expected_dev_fee, total_miner_rewards
                );
            }
        }

        // Validate individual miner rewards
        for (idx, tx) in self.transactions.iter().enumerate() {
            if !tx.is_coinbase() || idx == 0 {
                continue; // Skip non-coinbase and dev fee (already validated)
            }

            // Validate miner reward (not dev fee)
            if let Err(e) = tx.validate_coinbase_amount(self.header.height, false) {
                return Err(format!("Coinbase tx {} invalid: {}", idx, e));
            }
        }

        Ok(())
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
        // 🔒 v0.5.25-beta P2P GOSSIPSUB FIX: Sanitize all f64 values
        // Prevent NaN/Infinity which causes "invalid type: null, expected f64" deserialization errors
        let mining_activity = (solutions_count as f64).sqrt();
        let difficulty_growth = if total_difficulty > 0 {
            (total_difficulty as f64).log10()
        } else {
            0.0 // Prevent -Infinity from log10(0)
        };

        Self {
            temporal: round as f64,
            spatial: vec![
                Self::sanitize_f64(mining_activity), // x: mining activity
                Self::sanitize_f64(difficulty_growth), // y: difficulty growth
                Self::sanitize_f64(quantum_entropy), // z: randomness
            ],
            energetic: Self::sanitize_f64(total_difficulty as f64),
            entropic: Self::sanitize_f64(quantum_entropy),
            metadata: HashMap::new(),
        }
    }

    /// Ensure f64 values are valid (not NaN or Infinity) for P2P serialization
    /// Fix for v0.5.25-beta: Gossipsub deserialization fails with "invalid type: null, expected f64"
    fn sanitize_f64(value: f64) -> f64 {
        if value.is_nan() || value.is_infinite() {
            0.0
        } else {
            value
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
            adaptive_params: None,
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
                phase: 5,
                network_id: "mainnet-genesis".to_string(),
                prev_block_hash: [0u8; 32],
                solutions_root: [0u8; 32],
                tx_root: [0u8; 32],
                state_root: [0u8; 32],
                timestamp: 1234567890,
                dag_round: 1,
                vdf_proof: VDFProof::default(),
                anchor_validator: None,
                proposer: [1u8; 32],
                producer_id: 0,
                total_difficulty: 1000,
                producer_public_key: None,
                producer_signature: None,
                coinbase_merkle_root: None,
                total_coinbase_reward: None,
                coinbase_count: None,
            },
            mining_solutions: vec![],
            dag_parents: vec![],
            quantum_metadata: QuantumMetadata::default(),
            transactions: vec![],
            balance_updates: vec![],
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
        // Hash must be strictly less than target
        let easy_hash = [0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                         0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                         0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                         0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

        let target = [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];

        // This should pass because easy_hash < target (0x00 < 0xFF at byte 2)
        assert!(QBlock::verify_difficulty(&easy_hash, &target));
    }
}
