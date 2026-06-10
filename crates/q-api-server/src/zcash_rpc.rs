//! Zcash RPC Client - Integration with Zebra Node
//!
//! Provides wallet functionality for Zcash (ZEC) via Zebra RPC:
//! - Address generation (t-addresses and z-addresses)
//! - Balance queries (transparent and shielded)
//! - Transaction sending (shielded transactions)
//! - Transaction history
//!
//! Zebra Configuration:
//! - RPC Endpoint: http://127.0.0.1:8232
//! - Network: Mainnet
//! - Shielded pool: Sapling/Orchard support

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Zebra RPC endpoint configuration
#[derive(Debug, Clone)]
pub struct ZcashRpcConfig {
    /// RPC endpoint URL (default: http://127.0.0.1:8232)
    pub rpc_url: String,

    /// Optional RPC username for authentication
    pub rpc_user: Option<String>,

    /// Optional RPC password for authentication
    pub rpc_password: Option<String>,

    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for ZcashRpcConfig {
    fn default() -> Self {
        Self {
            rpc_url: "http://127.0.0.1:8232".to_string(),
            rpc_user: None,
            rpc_password: None,
            timeout_secs: 30,
        }
    }
}

/// Zcash address types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ZcashAddressType {
    /// Transparent address (t-address, similar to Bitcoin)
    Transparent,
    /// Shielded Sapling address (z-address)
    Sapling,
    /// Shielded Orchard address (unified address)
    Orchard,
}

/// Zcash balance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZcashBalance {
    /// Transparent balance (publicly visible)
    pub transparent: f64,
    /// Sapling shielded balance
    pub sapling: f64,
    /// Orchard shielded balance
    pub orchard: f64,
    /// Total balance
    pub total: f64,
}

/// Zcash transaction details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZcashTransaction {
    /// Transaction ID (txid)
    pub txid: String,
    /// Block height (None if unconfirmed)
    pub height: Option<u64>,
    /// Transaction amount (negative for sends, positive for receives)
    pub amount: f64,
    /// Transaction fee
    pub fee: f64,
    /// Confirmations count
    pub confirmations: u64,
    /// Transaction timestamp
    pub timestamp: u64,
    /// Transaction type (send/receive/shield/deshield)
    pub tx_type: String,
}

/// Zcash RPC client for wallet operations
#[derive(Clone)]
pub struct ZcashRpcClient {
    config: ZcashRpcConfig,
    http_client: reqwest::Client,
    /// Cached wallet addresses (address type -> address)
    wallet_cache: Arc<RwLock<std::collections::HashMap<String, String>>>,
}

impl ZcashRpcClient {
    /// Create a new Zcash RPC client with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ZcashRpcConfig::default())
    }

    /// Create a new Zcash RPC client with custom configuration
    pub fn with_config(config: ZcashRpcConfig) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        info!(
            "✅ [ZCASH RPC] Client initialized with endpoint: {}",
            config.rpc_url
        );

        Ok(Self {
            config,
            http_client,
            wallet_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Make an RPC call to Zebra node
    async fn call_rpc(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        debug!(
            "🔗 [ZCASH RPC] Calling method: {} with params: {}",
            method, params
        );

        let request_body = json!({
            "jsonrpc": "2.0",
            "id": "quillon-wallet",
            "method": method,
            "params": params,
        });

        let mut req = self
            .http_client
            .post(&self.config.rpc_url)
            .json(&request_body);

        // Add HTTP Basic Auth if configured
        if let (Some(user), Some(pass)) = (&self.config.rpc_user, &self.config.rpc_password) {
            req = req.basic_auth(user, Some(pass));
        }

        let response = req
            .send()
            .await
            .map_err(|e| anyhow!("RPC request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow!("RPC returned error status: {}", response.status()));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse RPC response: {}", e))?;

        // Check for RPC error
        if let Some(error) = response_json.get("error") {
            if !error.is_null() {
                return Err(anyhow!("RPC error: {}", error));
            }
        }

        // Extract result
        response_json
            .get("result")
            .ok_or_else(|| anyhow!("Missing 'result' field in RPC response"))
            .map(|v| v.clone())
    }

    /// Check if Zebra node is synced and ready
    pub async fn check_node_status(&self) -> Result<(bool, u64, f64)> {
        let info = self.call_rpc("getblockchaininfo", json!([])).await?;

        let blocks = info
            .get("blocks")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("Missing 'blocks' field"))?;

        let headers = info
            .get("headers")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("Missing 'headers' field"))?;

        let verification_progress = info
            .get("verificationprogress")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let is_synced = blocks >= headers && verification_progress > 0.99;

        info!(
            "📊 [ZCASH RPC] Node status: blocks={}, headers={}, progress={:.2}%, synced={}",
            blocks,
            headers,
            verification_progress * 100.0,
            is_synced
        );

        Ok((is_synced, blocks, verification_progress))
    }

    /// Generate a new Zcash address
    pub async fn generate_address(&self, addr_type: ZcashAddressType) -> Result<String> {
        // Check cache first
        let cache_key = format!("{:?}", addr_type);
        {
            let cache = self.wallet_cache.read().await;
            if let Some(cached_addr) = cache.get(&cache_key) {
                debug!("✅ [ZCASH RPC] Using cached address: {}", cached_addr);
                return Ok(cached_addr.clone());
            }
        }

        // Generate new address via RPC
        let method = match addr_type {
            ZcashAddressType::Transparent => "getnewaddress",
            ZcashAddressType::Sapling => "z_getnewaddress",
            ZcashAddressType::Orchard => "z_getnewaddress", // Orchard uses same method
        };

        let params = match addr_type {
            ZcashAddressType::Transparent => json!([]),
            ZcashAddressType::Sapling => json!(["sapling"]),
            ZcashAddressType::Orchard => json!(["orchard"]),
        };

        let result = self.call_rpc(method, params).await?;
        let address = result
            .as_str()
            .ok_or_else(|| anyhow!("RPC returned non-string address"))?
            .to_string();

        info!(
            "✅ [ZCASH RPC] Generated {:?} address: {}",
            addr_type, address
        );

        // Cache the address
        {
            let mut cache = self.wallet_cache.write().await;
            cache.insert(cache_key, address.clone());
        }

        Ok(address)
    }

    /// Get wallet balance (all address types)
    pub async fn get_balance(&self) -> Result<ZcashBalance> {
        // Get transparent balance
        let transparent_result = self.call_rpc("getbalance", json!([])).await?;
        let transparent = transparent_result.as_f64().unwrap_or(0.0);

        // Get shielded balance
        let shielded_result = self
            .call_rpc("z_gettotalbalance", json!([]))
            .await
            .unwrap_or_else(|_| json!({"transparent": "0", "private": "0", "total": "0"}));

        let sapling = shielded_result
            .get("private")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        // Orchard balance (not all nodes support this yet)
        let orchard = 0.0; // TODO: Add when Zebra supports Orchard balance queries

        let total = transparent + sapling + orchard;

        debug!(
            "💰 [ZCASH RPC] Balance: transparent={} ZEC, sapling={} ZEC, total={} ZEC",
            transparent, sapling, total
        );

        Ok(ZcashBalance {
            transparent,
            sapling,
            orchard,
            total,
        })
    }

    /// Send shielded transaction
    pub async fn send_shielded(
        &self,
        from_address: &str,
        to_address: &str,
        amount: f64,
        memo: Option<String>,
    ) -> Result<String> {
        info!(
            "📤 [ZCASH RPC] Sending shielded tx: {} ZEC from {} to {}",
            amount, from_address, to_address
        );

        // Construct operation
        let mut operation = json!({
            "address": to_address,
            "amount": amount,
        });

        if let Some(memo_text) = memo {
            // Convert memo to hex
            let memo_hex = hex::encode(memo_text.as_bytes());
            operation["memo"] = json!(memo_hex);
        }

        // Send operation
        let params = json!([
            from_address,
            [operation],
            1,      // minconf
            0.0001  // fee
        ]);

        let result = self.call_rpc("z_sendmany", params).await?;
        let opid = result
            .as_str()
            .ok_or_else(|| anyhow!("RPC returned non-string operation ID"))?
            .to_string();

        // Wait for operation to complete
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        let status_params = json!([[opid.clone()]]);
        let status_result = self.call_rpc("z_getoperationstatus", status_params).await?;

        let status_array = status_result
            .as_array()
            .ok_or_else(|| anyhow!("Invalid operation status response"))?;

        if status_array.is_empty() {
            return Err(anyhow!("Operation status not found"));
        }

        let status_obj = &status_array[0];
        let status_str = status_obj
            .get("status")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing status field"))?;

        if status_str == "success" {
            let txid = status_obj
                .get("result")
                .and_then(|r| r.get("txid"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("Missing txid in result"))?
                .to_string();

            info!("✅ [ZCASH RPC] Transaction sent: {}", txid);
            Ok(txid)
        } else if status_str == "failed" {
            let error_msg = status_obj
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            Err(anyhow!("Transaction failed: {}", error_msg))
        } else {
            Err(anyhow!(
                "Transaction still processing (status: {})",
                status_str
            ))
        }
    }

    /// Get transaction history
    pub async fn get_transaction_history(&self, count: usize) -> Result<Vec<ZcashTransaction>> {
        let params = json!([count, 0, true]);
        let result = self.call_rpc("listtransactions", params).await?;

        let tx_array = result
            .as_array()
            .ok_or_else(|| anyhow!("Invalid transaction list response"))?;

        let mut transactions = Vec::new();

        for tx_json in tx_array {
            if let Ok(tx) = Self::parse_transaction(tx_json) {
                transactions.push(tx);
            } else {
                warn!("⚠️  [ZCASH RPC] Failed to parse transaction: {:?}", tx_json);
            }
        }

        debug!(
            "📜 [ZCASH RPC] Retrieved {} transactions",
            transactions.len()
        );

        Ok(transactions)
    }

    /// Parse transaction from RPC JSON
    fn parse_transaction(json: &serde_json::Value) -> Result<ZcashTransaction> {
        let txid = json
            .get("txid")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing txid"))?
            .to_string();

        let amount = json.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let fee = json.get("fee").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let confirmations = json
            .get("confirmations")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let height = json.get("blockheight").and_then(|v| v.as_u64());

        let timestamp = json.get("time").and_then(|v| v.as_u64()).unwrap_or(0);

        let category = json
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let tx_type = match category {
            "send" => "send",
            "receive" => "receive",
            "generate" => "mined",
            _ => "other",
        }
        .to_string();

        Ok(ZcashTransaction {
            txid,
            height,
            amount,
            fee,
            confirmations,
            timestamp,
            tx_type,
        })
    }
}

impl Default for ZcashRpcClient {
    fn default() -> Self {
        Self::new().expect("Failed to create default ZcashRpcClient")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_zcash_rpc_client_creation() {
        let client = ZcashRpcClient::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_zcash_balance_serialization() {
        let balance = ZcashBalance {
            transparent: 1.5,
            sapling: 2.3,
            orchard: 0.0,
            total: 3.8,
        };

        let json = serde_json::to_string(&balance).unwrap();
        assert!(json.contains("\"transparent\":1.5"));
        assert!(json.contains("\"sapling\":2.3"));
    }
}
