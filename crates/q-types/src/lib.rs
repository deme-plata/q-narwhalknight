use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

// Re-export commonly used types
pub use ed25519_dalek::{PublicKey, SecretKey, Signature};
pub use sha3::{Digest, Sha3_256};

/// Core blockchain types for Q-NarwhalKnight Phase 0
/// These will be extended with post-quantum primitives in Phase 1

/// Transaction hash type
pub type TxHash = [u8; 32];

/// Block height
pub type Height = u64;

/// Round number in DAG consensus
pub type Round = u64;

/// Node identifier (Phase 0: hash of Ed25519 public key)
pub type NodeId = [u8; 32];

/// Amount type for token operations
pub type Amount = u64;

/// Address type (Phase 0: Ed25519 public key hash)
pub type Address = [u8; 32];

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
    pub public_key: Vec<u8>,
    pub balance: Amount,
    pub nonce: u64,
    pub created_at: DateTime<Utc>,
}

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
    Confirmed { block_height: Height, round: Round },
    Failed { error: String },
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
#[derive(Debug, Deserialize)]
pub struct CreateWalletRequest {
    pub password: Option<String>,
    pub mnemonic: Option<String>,
}

/// Transaction signing request
#[derive(Debug, Deserialize)]
pub struct SignTransactionRequest {
    pub to: Address,
    pub amount: Amount,
    pub fee: Amount,
    pub password: String,
}

/// Submit transaction request
#[derive(Debug, Deserialize)]
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

/// Phase identifiers for feature flagging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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