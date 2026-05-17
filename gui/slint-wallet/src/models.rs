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

/// Token balance entry (used for UI display)
#[derive(Debug, Clone)]
pub struct TokenBalanceDisplay {
    pub symbol: String,
    pub name: String,
    pub balance: String,
    pub usd_value: f64,
}

/// Token info from the public /api/v1/dex/tokens endpoint.
/// Lists ALL live tokens: QUG, QUGUSD, bridge tokens, custom deployed contracts.
#[derive(Debug, Clone, Deserialize)]
pub struct SupportedToken {
    #[serde(default)]
    pub address: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub symbol: String,
    #[serde(default)]
    pub decimals: u8,
    #[serde(default)]
    pub total_supply: String,
    #[serde(default)]
    pub contract_type: String,
    #[serde(default)]
    pub verified: bool,
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
/// Fields must match MiningSolutionRequest in handlers.rs for stats to work
#[derive(Debug, Clone, Serialize)]
pub struct MiningSubmission {
    pub miner_address: String,
    pub nonce: u64,
    pub hash: String,
    pub difficulty_target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub miner_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub miner_version: Option<String>,

    /// v1.0.5: Genus-2 VDF output (hex-encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_output: Option<String>,

    /// v1.0.5: Wesolowski proof (hex-encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_proof: Option<String>,

    /// v1.0.5: VDF checkpoints (hex-encoded list)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_checkpoints: Option<Vec<String>>,

    /// v1.0.5: Number of Genus-2 VDF iterations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vdf_iterations_count: Option<u64>,
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

/// Transaction history entry (matches server's UnifiedTransactionEntry)
#[derive(Debug, Deserialize)]
pub struct TransactionRecord {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub from: String,
    #[serde(default)]
    pub to: String,
    #[serde(default, deserialize_with = "deserialize_amount")]
    pub amount: f64,
    #[serde(default)]
    pub fee: f64,
    #[serde(default, deserialize_with = "deserialize_timestamp")]
    pub timestamp: String,
    #[serde(default)]
    pub tx_type: String,
    #[serde(default)]
    pub block_height: u64,
    #[serde(default)]
    pub token_symbol: Option<String>,
    #[serde(default)]
    pub direction: Option<String>,
    #[serde(default)]
    pub token_in: Option<String>,
    #[serde(default)]
    pub token_out: Option<String>,
    #[serde(default)]
    pub amount_out: Option<String>,
}

/// Deserialize amount from either String or f64
fn deserialize_amount<'de, D>(deserializer: D) -> Result<f64, D::Error>
where D: serde::Deserializer<'de> {
    use serde::de;
    struct AmountVisitor;
    impl<'de> de::Visitor<'de> for AmountVisitor {
        type Value = f64;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a number or numeric string")
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> { Ok(v) }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<f64, E> { Ok(v as f64) }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<f64, E> { Ok(v as f64) }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<f64, E> {
            Ok(v.parse::<f64>().unwrap_or(0.0))
        }
    }
    deserializer.deserialize_any(AmountVisitor)
}

/// Deserialize timestamp from either i64 (Unix) or String
fn deserialize_timestamp<'de, D>(deserializer: D) -> Result<String, D::Error>
where D: serde::Deserializer<'de> {
    use serde::de;
    struct TsVisitor;
    impl<'de> de::Visitor<'de> for TsVisitor {
        type Value = String;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a timestamp (number or string)")
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<String, E> {
            Ok(chrono::DateTime::from_timestamp(v, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| v.to_string()))
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<String, E> {
            self.visit_i64(v as i64)
        }
        fn visit_f64<E: de::Error>(self, v: f64) -> Result<String, E> {
            self.visit_i64(v as i64)
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<String, E> {
            if v.is_empty() { Ok("Unknown".to_string()) } else { Ok(v.to_string()) }
        }
    }
    deserializer.deserialize_any(TsVisitor)
}

/// OAuth2 token exchange response
#[derive(Debug, Deserialize)]
pub struct OAuthTokenResponse {
    pub access_token: String,
    #[serde(default = "default_token_type")]
    pub token_type: String,
    #[serde(default)]
    pub expires_in: i64,
    #[serde(default)]
    pub scope: String,
}

fn default_token_type() -> String {
    "Bearer".to_string()
}

/// OAuth2 userinfo response
#[derive(Debug, Deserialize)]
pub struct OAuthUserInfo {
    #[serde(default)]
    pub wallet_address: String,
    #[serde(default)]
    pub sub: String,
}

/// Generic API response wrapper
#[derive(Debug, Deserialize)]
pub struct ApiResponse<T> {
    #[serde(default)]
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

/// SSE balance-updated event data
#[derive(Debug, Deserialize)]
pub struct SseBalanceUpdate {
    #[serde(default)]
    pub wallet_address: String,
    #[serde(default)]
    pub old_balance: f64,
    #[serde(default)]
    pub new_balance: f64,
    #[serde(default)]
    pub change_reason: String,
}

/// Address book entry from server API
#[derive(Debug, Deserialize, Clone)]
pub struct AddressBookEntry {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub address: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub favorite: bool,
    #[serde(default)]
    pub notes: String,
}

/// SSE token-balance-updated event data
#[derive(Debug, Deserialize)]
pub struct SseTokenBalanceUpdate {
    #[serde(default)]
    pub wallet_address: String,
    #[serde(default)]
    pub token_address: String,
    #[serde(default)]
    pub token_symbol: String,
    #[serde(default)]
    pub old_balance: f64,
    #[serde(default)]
    pub new_balance: f64,
}
