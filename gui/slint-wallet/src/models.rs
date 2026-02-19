use serde::{Deserialize, Serialize};

/// Server status response - matches /api/v1/status "data" fields
#[derive(Debug, Deserialize)]
pub struct StatusResponse {
    #[serde(default)]
    pub current_height: u64,
    #[serde(default)]
    pub highest_network_height: u64,
    #[serde(default)]
    pub peer_count: u64,
    #[serde(default)]
    pub status: String,
}

/// Wallet balance response
/// Server returns balance as String (u128), balance_qnk as f64 (display-ready)
#[derive(Debug, Deserialize)]
pub struct BalanceResponse {
    #[serde(default)]
    pub balance_qnk: f64,
}

/// Token balance entry
#[derive(Debug, Deserialize)]
pub struct TokenBalance {
    #[serde(default)]
    pub token_address: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub symbol: String,
    #[serde(default)]
    pub balance: String,
    #[serde(default)]
    pub decimals: u32,
    #[serde(default)]
    pub price_usd: f64,
}

/// Multi-token balance response
#[derive(Debug, Deserialize)]
pub struct MultiTokenBalanceResponse {
    #[serde(default)]
    pub tokens: Vec<TokenBalance>,
}

/// Mining challenge from server
#[derive(Debug, Deserialize)]
pub struct MiningChallenge {
    pub challenge_hash: String,
    pub difficulty_target: String,
    pub block_height: u64,
    #[serde(default = "default_vdf_iterations")]
    pub vdf_iterations: u32,
    #[serde(default)]
    pub block_reward: f64,
}

fn default_vdf_iterations() -> u32 {
    100
}

/// Mining submission to server
#[derive(Debug, Serialize)]
pub struct MiningSubmission {
    pub miner_address: String,
    pub nonce: u64,
    pub hash: String,
    pub difficulty_target: String,
    pub challenge_hash: String,
    pub hash_rate: f64,
}

/// Transaction to submit
#[derive(Debug, Serialize)]
pub struct TransactionRequest {
    pub from: String,
    pub to: String,
    pub amount: String,
    pub fee: String,
    pub nonce: u64,
    pub signature: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memo: Option<String>,
}

/// Transaction history entry
#[derive(Debug, Deserialize)]
pub struct TransactionRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub from: String,
    #[serde(default)]
    pub to: String,
    #[serde(default)]
    pub amount: f64,
    #[serde(default)]
    pub fee: f64,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub tx_type: String,
}

/// Generic API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(default)]
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}
