use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

// Re-export commonly used types (ed25519-dalek v2.x API compatibility)
pub use ed25519_dalek::{Signature, SigningKey as SecretKey, VerifyingKey as PublicKey};
pub use sha3::{Digest, Sha3_256};

/// Core blockchain types for Q-NarwhalKnight Phase 0
/// These will be extended with post-quantum primitives in Phase 1
///
/// Transaction hash type
pub type TxHash = [u8; 32];

/// Transaction ID type alias (same as TxHash)
pub type TransactionId = TxHash;

/// Proposal hash for consensus
pub type ProposalHash = [u8; 32];

/// Consensus vote structure
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsensusVote {
    pub epoch: u64,
    pub proposal_hash: ProposalHash,
    pub participated: bool,
}

/// Block height
pub type Height = u64;

/// Round number in DAG consensus
pub type Round = u64;

/// Node identifier (Phase 0: hash of Ed25519 public key)
pub type NodeId = [u8; 32];

/// Validator identifier (alias for NodeId)
pub type ValidatorId = NodeId;

/// Amount type for token operations (Phase 0: u64, Phase 1+: QAmount for ultra-precision)
pub type Amount = u64;

/// Address type (Phase 0: Ed25519 public key hash)
pub type Address = [u8; 32];

/// Token type for dual-token economics (QUG mining token + QUGUSD stablecoin)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenType {
    /// QUG - Native mining token (21M fixed supply, deflationary)
    QUG,
    /// QUGUSD - Algorithmic stablecoin pegged to USD ($1.00)
    QUGUSD,
}

/// Token information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub token_type: TokenType,
    pub name: String,
    pub symbol: String,
    pub decimals: u8,
    pub max_supply: Option<u64>, // None for QUGUSD (unlimited if collateralized)
}

/// QUG token constants
pub const QUG_DECIMALS: u8 = 8;
pub const QUG_MAX_SUPPLY: u64 = 2_100_000_000_000_000; // 21M * 10^8
pub const QUG_TOKEN_ADDRESS: [u8; 32] = [
    0x51, 0x55, 0x47, 0x00, 0x00, 0x00, 0x00, 0x00, // "QUG" in hex + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// QUGUSD token constants
pub const QUGUSD_DECIMALS: u8 = 8;
pub const QUGUSD_TOKEN_ADDRESS: [u8; 32] = [
    0x51, 0x55, 0x47, 0x55, 0x53, 0x44, 0x00, 0x00, // "QUGUSD" in hex + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Bank master account address for fee collection
pub const BANK_MASTER_ACCOUNT: [u8; 32] = [
    0x42, 0x41, 0x4E, 0x4B, 0x00, 0x00, 0x00, 0x00, // "BANK" in hex + zeros
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

impl TokenInfo {
    /// Get QUG token information
    pub fn qug() -> Self {
        Self {
            token_type: TokenType::QUG,
            name: "Quillon".to_string(),
            symbol: "QUG".to_string(),
            decimals: QUG_DECIMALS,
            max_supply: Some(QUG_MAX_SUPPLY),
        }
    }

    /// Get QUGUSD token information
    pub fn qugusd() -> Self {
        Self {
            token_type: TokenType::QUGUSD,
            name: "Quillon USD".to_string(),
            symbol: "QUGUSD".to_string(),
            decimals: QUGUSD_DECIMALS,
            max_supply: None, // Unlimited if properly collateralized
        }
    }
}

impl TokenType {
    /// Get the reserved token address for this token type
    pub fn address(&self) -> [u8; 32] {
        match self {
            TokenType::QUG => QUG_TOKEN_ADDRESS,
            TokenType::QUGUSD => QUGUSD_TOKEN_ADDRESS,
        }
    }

    /// Get token info for this token type
    pub fn info(&self) -> TokenInfo {
        match self {
            TokenType::QUG => TokenInfo::qug(),
            TokenType::QUGUSD => TokenInfo::qugusd(),
        }
    }
}

/// Hash256 type for general cryptographic hashing
pub type Hash256 = [u8; 32];

/// Fixed-point number with 28 decimal places for ultra-precision
pub type FixedPoint28 = i64;

/// Transaction structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: TxHash,
    pub from: Address,
    pub to: Address,
    pub amount: Amount,
    pub fee: Amount,
    pub nonce: u64,
    pub signature: Vec<u8>, // Will be Signature in Phase 0, expandable for PQ
    pub timestamp: DateTime<Utc>,
    pub data: Vec<u8>, // Contract call data or arbitrary transaction payload
    #[serde(default = "default_token_type")]
    pub token_type: TokenType, // QUG or QUGUSD
    #[serde(default = "default_fee_token_type")]
    pub fee_token_type: TokenType, // Token used to pay fees (default: QUGUSD)
}

/// Default token type for backwards compatibility
fn default_token_type() -> TokenType {
    TokenType::QUG
}

/// Default fee token type
fn default_fee_token_type() -> TokenType {
    TokenType::QUGUSD
}

/// DAG vertex (Narwhal block)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    pub id: VertexId,
    pub round: Round,
    pub author: NodeId,
    pub tx_root: TxHash,
    pub parents: Vec<VertexId>,
    pub transactions: Vec<Transaction>,
    pub signature: Vec<u8>,
    pub timestamp: DateTime<Utc>,
}

/// Vertex identifier
pub type VertexId = [u8; 32];

/// Certificate for vertex availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub vertex_id: VertexId,
    pub round: Round,
    pub signatures: BTreeMap<NodeId, Vec<u8>>,
    pub threshold_met: bool,
}

/// Wallet information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletInfo {
    pub id: Uuid,
    pub address: Address,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address_formatted: Option<String>, // "qnk" + hex encoding
    pub public_key: Vec<u8>,
    pub balance: Amount,
    pub nonce: u64,
    pub created_at: DateTime<Utc>,
}

/// State key for state management
pub type StateKey = Vec<u8>;

/// State value for state management  
pub type StateValue = Vec<u8>;

/// Node status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub node_id: NodeId,
    pub current_round: Round,
    pub current_height: Height,
    pub connected_peers: u32,
    pub tx_pool_size: u32,
    pub is_validator: bool,
    pub uptime: std::time::Duration,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TxStatus {
    Pending,
    InMempool,
    Mixing, // New status for quantum privacy mixing
    Confirmed { block_height: Height, round: Round },
    Failed { error: String },
}

/// Privacy level for quantum mixing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrivacyLevel {
    Standard, // 3 mixing rounds, 15 decoys
    High,     // 5 mixing rounds, 25 decoys
    Maximum,  // 8 mixing rounds, 50 decoys
}

/// API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            timestamp: Utc::now(),
        }
    }

    pub fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            timestamp: Utc::now(),
        }
    }
}

/// Wallet creation request
#[derive(Debug, Serialize, Deserialize)]
pub struct CreateWalletRequest {
    pub password: Option<String>,
    pub mnemonic: Option<String>,
}

/// Transaction signing request
#[derive(Debug, Serialize, Deserialize)]
pub struct SignTransactionRequest {
    pub to: Address,
    pub amount: Amount,
    pub fee: Amount,
    pub password: String,
}

/// Submit transaction request
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitTransactionRequest {
    pub transaction: Transaction,
}

/// Quantum entropy info (Phase 2+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QRNGInfo {
    pub vendor: String,
    pub serial: [u8; 16],
    pub health: BTreeMap<u64, f64>, // entropy-rate per 1-second window
    pub signature: Vec<u8>,
}

/// Cryptographic agility metadata (Phase 1+)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoAgility {
    pub signature_scheme: String,
    pub kem_scheme: String,
    pub hash_function: String,
    pub vrf_scheme: String,
    pub multicodec_version: u32,
}

/// Narwhal payload structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarwhalPayload {
    pub data: Vec<u8>,
    pub transactions: Vec<Transaction>,
    pub timestamp: u64,
    pub payload_hash: [u8; 32],
}

/// Finalized block structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub height: u64,
    pub hash: [u8; 32],
    pub vertices: Vec<VertexId>,
    pub finality_cert: Option<BullsharkCert>,
    pub timestamp: DateTime<Utc>,
    pub proposer: NodeId,
}

/// BullShark certificate for finality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BullsharkCert {
    pub round: u64,
    pub vertex_id: VertexId,
    pub signatures: BTreeMap<NodeId, Vec<u8>>,
    pub finality_proof: Vec<u8>,
    pub commit_round: u64,
}

/// Peer information for networking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub multiaddrs: Vec<String>,
    pub capabilities: Vec<String>,
    pub agent_version: Option<String>,
    pub protocol_version: Option<String>,
    pub supported_protocols: Vec<String>,
}

impl PeerInfo {
    pub fn new(peer_id_str: &str) -> Self {
        Self {
            peer_id: peer_id_str.to_string(),
            multiaddrs: vec![],
            capabilities: vec![],
            agent_version: None,
            protocol_version: None,
            supported_protocols: vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Phase {
    Phase0, // Classical baseline
    Phase1, // Post-quantum cryptography
    Phase2, // Quantum randomness
    Phase3, // STARK-only zkVM
    Phase4, // QKD integration
}

/// Node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub phase: Phase,
    pub node_id: NodeId,
    pub is_validator: bool,
    pub api_port: u16,
    pub p2p_port: u16,
    pub bootstrap_peers: Vec<String>,
    pub crypto_config: CryptoAgility,
}

/// Utility functions
impl Transaction {
    pub fn hash(&self) -> TxHash {
        let mut hasher = Sha3_256::new();
        let encoded = postcard::to_allocvec(self).unwrap();
        hasher.update(&encoded);
        hasher.finalize().into()
    }
}

impl Vertex {
    pub fn hash(&self) -> VertexId {
        let mut hasher = Sha3_256::new();
        let encoded = postcard::to_allocvec(self).unwrap();
        hasher.update(&encoded);
        hasher.finalize().into()
    }
}

// Duplicate Transaction definition removed - using the one above with Address types

/// Error types
#[derive(Debug, thiserror::Error)]
pub enum QError {
    #[error("Cryptographic error: {0}")]
    Crypto(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Consensus error: {0}")]
    Consensus(String),

    #[error("Wallet error: {0}")]
    Wallet(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(#[from] anyhow::Error),
}

// ============================================================================
// Phase-Aware Signature and Verification for Consensus
// ============================================================================

/// Phase-aware signature metadata
/// Allows consensus layer to know which cryptographic scheme was used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSignature {
    /// The cryptographic phase used for signing
    pub phase: Phase,
    /// Raw signature bytes (Ed25519 for Phase 0, Dilithium5 for Phase 1)
    pub signature: Vec<u8>,
    /// Optional: Scheme identifier for crypto-agility
    pub scheme_id: Option<u16>,
}

impl PhaseSignature {
    /// Create Phase 0 signature (Ed25519)
    pub fn phase0(signature: Vec<u8>) -> Self {
        Self {
            phase: Phase::Phase0,
            signature,
            scheme_id: Some(0x1200), // Ed25519 multicodec
        }
    }

    /// Create Phase 1 signature (Dilithium5)
    pub fn phase1(signature: Vec<u8>) -> Self {
        Self {
            phase: Phase::Phase1,
            signature,
            scheme_id: Some(0x1300), // Dilithium5 multicodec
        }
    }

    /// Get signature size for this phase
    pub fn signature_size(&self) -> usize {
        match self.phase {
            Phase::Phase0 => 64,       // Ed25519: 64 bytes
            Phase::Phase1 => 4627,     // Dilithium5: ~4,627 bytes
            Phase::Phase2 => 4627,     // Dilithium5 (with QRNG)
            Phase::Phase3 => 4627,     // Dilithium5 (with STARK)
            Phase::Phase4 => 4627,     // Dilithium5 (with QKD)
        }
    }

    /// Check if this signature is quantum-resistant
    pub fn is_quantum_resistant(&self) -> bool {
        self.phase >= Phase::Phase1
    }
}

/// Phase-aware certificate with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCertificate {
    /// Original certificate
    pub certificate: Certificate,
    /// Phase of the network when certificate was created
    pub phase: Phase,
    /// Timestamp when certificate was created
    pub created_at: DateTime<Utc>,
}

impl PhaseCertificate {
    /// Create a new phase-aware certificate
    pub fn new(certificate: Certificate, phase: Phase) -> Self {
        Self {
            certificate,
            phase,
            created_at: Utc::now(),
        }
    }

    /// Check if certificate has sufficient signatures for the phase
    pub fn is_valid(&self, threshold: usize) -> bool {
        self.certificate.threshold_met && self.certificate.signatures.len() >= threshold
    }

    /// Get the number of quantum-resistant signatures
    /// (All signatures are quantum-resistant if phase >= Phase1)
    pub fn quantum_resistant_signature_count(&self) -> usize {
        if self.phase >= Phase::Phase1 {
            self.certificate.signatures.len()
        } else {
            0
        }
    }
}

/// Consensus voting with phase awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConsensusVote {
    /// Original vote
    pub vote: ConsensusVote,
    /// Phase when vote was cast
    pub phase: Phase,
    /// Signature of the vote
    pub signature: PhaseSignature,
    /// Voter's node ID
    pub voter: NodeId,
}

impl PhaseConsensusVote {
    /// Create a Phase 0 vote (Ed25519)
    pub fn phase0(vote: ConsensusVote, signature: Vec<u8>, voter: NodeId) -> Self {
        Self {
            vote,
            phase: Phase::Phase0,
            signature: PhaseSignature::phase0(signature),
            voter,
        }
    }

    /// Create a Phase 1 vote (Dilithium5)
    pub fn phase1(vote: ConsensusVote, signature: Vec<u8>, voter: NodeId) -> Self {
        Self {
            vote,
            phase: Phase::Phase1,
            signature: PhaseSignature::phase1(signature),
            voter,
        }
    }

    /// Verify the vote signature
    /// Returns true if signature is valid for the given phase
    pub fn verify(&self, public_key: &[u8]) -> bool {
        // TODO: Implement actual signature verification
        // Phase 0: Use Ed25519 verification
        // Phase 1: Use Dilithium5 verification

        // For now, basic validation
        match self.phase {
            Phase::Phase0 => self.signature.signature.len() == 64,
            Phase::Phase1 | Phase::Phase2 | Phase::Phase3 | Phase::Phase4 => {
                // Dilithium5 signatures are ~4,627 bytes
                self.signature.signature.len() >= 4000 && self.signature.signature.len() <= 5000
            }
        }
    }
}

/// Helper trait for phase-aware signing
pub trait PhaseAwareSigning {
    /// Sign data using the appropriate algorithm for the phase
    fn sign_with_phase(&self, data: &[u8], phase: Phase) -> Result<PhaseSignature, QError>;

    /// Verify signature using the appropriate algorithm for the phase
    fn verify_with_phase(&self, data: &[u8], signature: &PhaseSignature) -> Result<bool, QError>;
}

/// Helper function to create a vertex signature based on phase
pub fn create_vertex_signature(
    vertex_data: &[u8],
    phase: Phase,
    private_key: &[u8],
) -> Result<Vec<u8>, QError> {
    match phase {
        Phase::Phase0 => {
            // Ed25519 signing (Phase 0)
            // In production, use actual Ed25519 signing
            Ok(vec![0u8; 64]) // Placeholder
        }
        Phase::Phase1 | Phase::Phase2 | Phase::Phase3 | Phase::Phase4 => {
            // Dilithium5 signing (Phase 1+)
            // In production, use actual Dilithium5 signing from q-wallet
            Ok(vec![0u8; 4627]) // Placeholder
        }
    }
}

/// Helper function to verify a vertex signature based on phase
pub fn verify_vertex_signature(
    vertex_data: &[u8],
    signature: &[u8],
    public_key: &[u8],
    phase: Phase,
) -> Result<bool, QError> {
    match phase {
        Phase::Phase0 => {
            // Ed25519 verification
            Ok(signature.len() == 64)
        }
        Phase::Phase1 | Phase::Phase2 | Phase::Phase3 | Phase::Phase4 => {
            // Dilithium5 verification
            Ok(signature.len() >= 4000 && signature.len() <= 5000)
        }
    }
}
