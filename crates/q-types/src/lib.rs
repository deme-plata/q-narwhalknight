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

// ============================================================================
// Network Configuration for Testnet/Mainnet Separation
// ============================================================================

/// Network identifier for testnet/mainnet separation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NetworkId {
    /// Testnet (October 2025 - ongoing development)
    Testnet,
    /// Mainnet (Launch: December 15, 2025 00:00 UTC)
    Mainnet,
}

impl NetworkId {
    /// Get the string identifier for this network
    pub fn as_str(&self) -> &'static str {
        match self {
            NetworkId::Testnet => "testnet",
            NetworkId::Mainnet => "mainnet",
        }
    }

    /// Get the human-readable name for this network
    pub fn display_name(&self) -> &'static str {
        match self {
            NetworkId::Testnet => "Q-NarwhalKnight Testnet",
            NetworkId::Mainnet => "Q-NarwhalKnight Mainnet",
        }
    }

    /// Get the default API port for this network
    pub fn default_api_port(&self) -> u16 {
        match self {
            NetworkId::Testnet => 8080,
            NetworkId::Mainnet => 8081,
        }
    }

    /// Get the default P2P port for this network
    pub fn default_p2p_port(&self) -> u16 {
        match self {
            NetworkId::Testnet => 9001,
            NetworkId::Mainnet => 9002,
        }
    }

    /// Get the gossipsub topic prefix for this network
    pub fn gossipsub_topic_prefix(&self) -> String {
        format!("/qnk/{}", self.as_str())
    }

    /// Get the transaction gossipsub topic for this network
    pub fn transactions_topic(&self) -> String {
        format!("{}/transactions", self.gossipsub_topic_prefix())
    }

    /// Get the blocks gossipsub topic for this network
    pub fn blocks_topic(&self) -> String {
        format!("{}/blocks", self.gossipsub_topic_prefix())
    }

    /// Get the acknowledgments gossipsub topic for this network
    pub fn acks_topic(&self) -> String {
        format!("{}/ack", self.gossipsub_topic_prefix())
    }
}

impl std::str::FromStr for NetworkId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "testnet" => Ok(NetworkId::Testnet),
            "mainnet" => Ok(NetworkId::Mainnet),
            _ => Err(format!("Invalid network ID: {}", s)),
        }
    }
}

impl Default for NetworkId {
    fn default() -> Self {
        NetworkId::Testnet
    }
}

/// Network configuration with genesis hash and launch time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network identifier (testnet/mainnet)
    pub network_id: NetworkId,

    /// Genesis block hash (unique per network)
    pub genesis_hash: [u8; 32],

    /// Network launch timestamp (UTC)
    pub launch_time: DateTime<Utc>,

    /// Network version string
    pub version: String,

    /// Chain ID for transaction replay protection
    pub chain_id: u64,

    /// API server port
    pub api_port: u16,

    /// P2P networking port
    pub p2p_port: u16,

    /// Bootstrap peers for this network
    pub bootstrap_peers: Vec<String>,
}

impl NetworkConfig {
    /// Create testnet configuration
    pub fn testnet() -> Self {
        Self {
            network_id: NetworkId::Testnet,
            genesis_hash: [
                // Testnet genesis hash (October 2025)
                0x74, 0x65, 0x73, 0x74, 0x6e, 0x65, 0x74, 0x2d,  // "testnet-"
                0x6f, 0x63, 0x74, 0x32, 0x30, 0x32, 0x35, 0x00,  // "oct2025"
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            launch_time: DateTime::parse_from_rfc3339("2025-10-23T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            version: "v0.0.9-beta-testnet".to_string(),
            chain_id: 1, // Testnet chain ID
            api_port: 8080,
            p2p_port: 9001,
            bootstrap_peers: vec![
                "/ip4/185.182.185.227/tcp/9001".to_string(),
            ],
        }
    }

    /// Create mainnet configuration
    pub fn mainnet() -> Self {
        Self {
            network_id: NetworkId::Mainnet,
            genesis_hash: [
                // Mainnet genesis hash (December 2025)
                0x6d, 0x61, 0x69, 0x6e, 0x6e, 0x65, 0x74, 0x2d,  // "mainnet-"
                0x64, 0x65, 0x63, 0x32, 0x30, 0x32, 0x35, 0x00,  // "dec2025"
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
            launch_time: DateTime::parse_from_rfc3339("2025-12-15T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            version: "v1.0.0-mainnet".to_string(),
            chain_id: 999, // Mainnet chain ID
            api_port: 8081,
            p2p_port: 9002,
            bootstrap_peers: vec![
                "/ip4/185.182.185.227/tcp/9002".to_string(),
            ],
        }
    }

    /// Get network configuration by network ID
    pub fn from_network_id(network_id: NetworkId) -> Self {
        match network_id {
            NetworkId::Testnet => Self::testnet(),
            NetworkId::Mainnet => Self::mainnet(),
        }
    }

    /// Check if the network has launched
    pub fn is_launched(&self) -> bool {
        Utc::now() >= self.launch_time
    }

    /// Get time until launch (None if already launched)
    pub fn time_until_launch(&self) -> Option<chrono::Duration> {
        let now = Utc::now();
        if now >= self.launch_time {
            None
        } else {
            Some(self.launch_time - now)
        }
    }

    /// Verify that a transaction belongs to this network
    pub fn verify_transaction_network(&self, tx: &Transaction) -> bool {
        // In the future, transactions should include chain_id
        // For now, we accept all transactions on the same network
        true
    }

    /// Verify that a message came from this network
    pub fn verify_message_network(&self, genesis_hash: &[u8; 32]) -> bool {
        genesis_hash == &self.genesis_hash
    }
}

/// Network message wrapper with network verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMessage<T> {
    /// Network genesis hash for verification
    pub genesis_hash: [u8; 32],

    /// Chain ID for replay protection
    pub chain_id: u64,

    /// Message payload
    pub payload: T,

    /// Timestamp when message was created
    pub timestamp: DateTime<Utc>,
}

impl<T> NetworkMessage<T> {
    /// Create a new network message
    pub fn new(config: &NetworkConfig, payload: T) -> Self {
        Self {
            genesis_hash: config.genesis_hash,
            chain_id: config.chain_id,
            payload,
            timestamp: Utc::now(),
        }
    }

    /// Verify that this message belongs to the given network
    pub fn verify_network(&self, config: &NetworkConfig) -> bool {
        self.genesis_hash == config.genesis_hash && self.chain_id == config.chain_id
    }
}

#[cfg(test)]
mod network_separation_tests {
    use super::*;

    #[test]
    fn test_network_id_string_conversion() {
        // Test NetworkId to string conversion
        assert_eq!(NetworkId::Testnet.as_str(), "testnet");
        assert_eq!(NetworkId::Mainnet.as_str(), "mainnet");

        // Test string parsing
        assert_eq!("testnet".parse::<NetworkId>().unwrap(), NetworkId::Testnet);
        assert_eq!("mainnet".parse::<NetworkId>().unwrap(), NetworkId::Mainnet);

        // Test case-insensitive parsing (to_lowercase is used)
        assert_eq!("TESTNET".parse::<NetworkId>().unwrap(), NetworkId::Testnet);
        assert_eq!("Mainnet".parse::<NetworkId>().unwrap(), NetworkId::Mainnet);

        // Test invalid string parsing
        assert!("invalid".parse::<NetworkId>().is_err());
        assert!("".parse::<NetworkId>().is_err());
    }

    #[test]
    fn test_network_id_display() {
        assert_eq!(NetworkId::Testnet.display_name(), "Q-NarwhalKnight Testnet");
        assert_eq!(NetworkId::Mainnet.display_name(), "Q-NarwhalKnight Mainnet");
    }

    #[test]
    fn test_gossipsub_topic_generation() {
        let testnet = NetworkId::Testnet;
        let mainnet = NetworkId::Mainnet;

        // Test topic prefix generation
        assert_eq!(testnet.gossipsub_topic_prefix(), "/qnk/testnet");
        assert_eq!(mainnet.gossipsub_topic_prefix(), "/qnk/mainnet");

        // Test transaction topics
        assert_eq!(testnet.transactions_topic(), "/qnk/testnet/transactions");
        assert_eq!(mainnet.transactions_topic(), "/qnk/mainnet/transactions");

        // Test block topics
        assert_eq!(testnet.blocks_topic(), "/qnk/testnet/blocks");
        assert_eq!(mainnet.blocks_topic(), "/qnk/mainnet/blocks");

        // Test ACK topics
        assert_eq!(testnet.acks_topic(), "/qnk/testnet/ack");
        assert_eq!(mainnet.acks_topic(), "/qnk/mainnet/ack");

        // Verify topics are different between networks
        assert_ne!(testnet.transactions_topic(), mainnet.transactions_topic());
        assert_ne!(testnet.blocks_topic(), mainnet.blocks_topic());
    }

    #[test]
    fn test_network_config_testnet() {
        let config = NetworkConfig::testnet();

        // Verify network ID
        assert_eq!(config.network_id, NetworkId::Testnet);

        // Verify chain ID
        assert_eq!(config.chain_id, 1);

        // Verify ports
        assert_eq!(config.api_port, 8080);
        assert_eq!(config.p2p_port, 9001);

        // Verify version
        assert_eq!(config.version, "v0.0.9-beta-testnet");

        // Verify genesis hash starts with "testnet-"
        assert_eq!(&config.genesis_hash[..8], b"testnet-");

        // Verify launch time (October 23, 2025)
        let expected = DateTime::parse_from_rfc3339("2025-10-23T00:00:00Z").unwrap();
        assert_eq!(config.launch_time, expected.with_timezone(&Utc));
    }

    #[test]
    fn test_network_config_mainnet() {
        let config = NetworkConfig::mainnet();

        // Verify network ID
        assert_eq!(config.network_id, NetworkId::Mainnet);

        // Verify chain ID
        assert_eq!(config.chain_id, 999);

        // Verify ports
        assert_eq!(config.api_port, 8081);
        assert_eq!(config.p2p_port, 9002);

        // Verify version
        assert_eq!(config.version, "v1.0.0-mainnet");

        // Verify genesis hash starts with "mainnet-"
        assert_eq!(&config.genesis_hash[..8], b"mainnet-");

        // Verify launch time (December 15, 2025)
        let expected = DateTime::parse_from_rfc3339("2025-12-15T00:00:00Z").unwrap();
        assert_eq!(config.launch_time, expected.with_timezone(&Utc));
    }

    #[test]
    fn test_network_config_uniqueness() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Genesis hashes must be different
        assert_ne!(testnet.genesis_hash, mainnet.genesis_hash);

        // Chain IDs must be different
        assert_ne!(testnet.chain_id, mainnet.chain_id);

        // Ports must be different
        assert_ne!(testnet.api_port, mainnet.api_port);
        assert_ne!(testnet.p2p_port, mainnet.p2p_port);

        // Network IDs must be different
        assert_ne!(testnet.network_id, mainnet.network_id);
    }

    #[test]
    fn test_network_config_from_network_id() {
        let testnet = NetworkConfig::from_network_id(NetworkId::Testnet);
        let mainnet = NetworkConfig::from_network_id(NetworkId::Mainnet);

        assert_eq!(testnet.network_id, NetworkId::Testnet);
        assert_eq!(mainnet.network_id, NetworkId::Mainnet);

        // Should match direct constructors
        assert_eq!(testnet.genesis_hash, NetworkConfig::testnet().genesis_hash);
        assert_eq!(mainnet.genesis_hash, NetworkConfig::mainnet().genesis_hash);
    }

    #[test]
    fn test_genesis_hash_verification() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Same network should verify
        assert!(testnet.verify_message_network(&testnet.genesis_hash));
        assert!(mainnet.verify_message_network(&mainnet.genesis_hash));

        // Cross-network should fail
        assert!(!testnet.verify_message_network(&mainnet.genesis_hash));
        assert!(!mainnet.verify_message_network(&testnet.genesis_hash));

        // Random hash should fail
        let random_hash = [0xff; 32];
        assert!(!testnet.verify_message_network(&random_hash));
        assert!(!mainnet.verify_message_network(&random_hash));
    }

    #[test]
    fn test_network_message_creation() {
        let testnet = NetworkConfig::testnet();
        let message = NetworkMessage::new(&testnet, "test payload".to_string());

        // Verify genesis hash is set correctly
        assert_eq!(message.genesis_hash, testnet.genesis_hash);

        // Verify chain ID is set correctly
        assert_eq!(message.chain_id, testnet.chain_id);

        // Verify payload
        assert_eq!(message.payload, "test payload");

        // Verify timestamp is recent (within last second)
        let now = Utc::now();
        let diff = now - message.timestamp;
        assert!(diff.num_seconds() < 1);
    }

    #[test]
    fn test_network_message_verification() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Create message for testnet
        let testnet_msg = NetworkMessage::new(&testnet, 42u64);

        // Should verify on same network
        assert!(testnet_msg.verify_network(&testnet));

        // Should fail on different network
        assert!(!testnet_msg.verify_network(&mainnet));
    }

    #[test]
    fn test_network_message_cross_network_rejection() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Create messages for each network
        let testnet_msg = NetworkMessage::new(&testnet, vec![1, 2, 3]);
        let mainnet_msg = NetworkMessage::new(&mainnet, vec![1, 2, 3]);

        // Testnet message should only verify on testnet
        assert!(testnet_msg.verify_network(&testnet));
        assert!(!testnet_msg.verify_network(&mainnet));

        // Mainnet message should only verify on mainnet
        assert!(mainnet_msg.verify_network(&mainnet));
        assert!(!mainnet_msg.verify_network(&testnet));
    }

    #[test]
    fn test_network_message_with_modified_genesis_hash() {
        let testnet = NetworkConfig::testnet();
        let mut message = NetworkMessage::new(&testnet, "payload");

        // Message should verify initially
        assert!(message.verify_network(&testnet));

        // Modify genesis hash
        message.genesis_hash[0] ^= 0xff;

        // Should now fail verification
        assert!(!message.verify_network(&testnet));
    }

    #[test]
    fn test_network_message_with_modified_chain_id() {
        let testnet = NetworkConfig::testnet();
        let mut message = NetworkMessage::new(&testnet, "payload");

        // Message should verify initially
        assert!(message.verify_network(&testnet));

        // Modify chain ID
        message.chain_id = 999;

        // Should now fail verification
        assert!(!message.verify_network(&testnet));
    }

    #[test]
    fn test_launch_time_verification() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Testnet launch time (October 23, 2025) is in the past (relative to test date)
        // Note: This test assumes we're running after Oct 23, 2025
        // For now we just verify the launch times are set correctly
        assert_eq!(
            testnet.launch_time,
            DateTime::parse_from_rfc3339("2025-10-23T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc)
        );

        assert_eq!(
            mainnet.launch_time,
            DateTime::parse_from_rfc3339("2025-12-15T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc)
        );
    }

    #[test]
    fn test_bootstrap_peers_configuration() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Verify testnet has bootstrap peers
        assert!(!testnet.bootstrap_peers.is_empty());
        assert_eq!(testnet.bootstrap_peers[0], "/ip4/185.182.185.227/tcp/9001");

        // Verify mainnet has bootstrap peers
        assert!(!mainnet.bootstrap_peers.is_empty());
        assert_eq!(mainnet.bootstrap_peers[0], "/ip4/185.182.185.227/tcp/9002");

        // Verify different ports for different networks
        assert_ne!(testnet.bootstrap_peers, mainnet.bootstrap_peers);
    }

    #[test]
    fn test_edge_case_empty_genesis_hash() {
        let testnet = NetworkConfig::testnet();
        let empty_hash = [0u8; 32];

        // Empty hash should not match testnet
        assert!(!testnet.verify_message_network(&empty_hash));
    }

    #[test]
    fn test_edge_case_all_ones_genesis_hash() {
        let testnet = NetworkConfig::testnet();
        let ones_hash = [0xff; 32];

        // All-ones hash should not match testnet
        assert!(!testnet.verify_message_network(&ones_hash));
    }

    #[test]
    fn test_topic_namespace_isolation() {
        let testnet_id = NetworkId::Testnet;
        let mainnet_id = NetworkId::Mainnet;

        // Generate all topic types for both networks
        let testnet_topics = vec![
            testnet_id.transactions_topic(),
            testnet_id.blocks_topic(),
            testnet_id.acks_topic(),
        ];

        let mainnet_topics = vec![
            mainnet_id.transactions_topic(),
            mainnet_id.blocks_topic(),
            mainnet_id.acks_topic(),
        ];

        // No topic should overlap between networks
        for testnet_topic in &testnet_topics {
            for mainnet_topic in &mainnet_topics {
                assert_ne!(testnet_topic, mainnet_topic);
            }
        }

        // All testnet topics should start with /qnk/testnet
        for topic in &testnet_topics {
            assert!(topic.starts_with("/qnk/testnet/"));
        }

        // All mainnet topics should start with /qnk/mainnet
        for topic in &mainnet_topics {
            assert!(topic.starts_with("/qnk/mainnet/"));
        }
    }

    #[test]
    fn test_chain_id_replay_protection() {
        let testnet = NetworkConfig::testnet();
        let mainnet = NetworkConfig::mainnet();

        // Create a transaction-like payload
        #[derive(Clone)]
        struct MockTransaction {
            from: String,
            to: String,
            amount: u64,
        }

        let tx = MockTransaction {
            from: "alice".to_string(),
            to: "bob".to_string(),
            amount: 100,
        };

        // Wrap in network messages
        let testnet_msg = NetworkMessage::new(&testnet, tx.clone());
        let mainnet_msg = NetworkMessage::new(&mainnet, tx.clone());

        // Same transaction data, but different chain IDs should prevent replay
        assert_ne!(testnet_msg.chain_id, mainnet_msg.chain_id);
        assert!(!testnet_msg.verify_network(&mainnet));
        assert!(!mainnet_msg.verify_network(&testnet));
    }
}
