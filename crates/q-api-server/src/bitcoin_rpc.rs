//! Bitcoin RPC Client - Integration with Bitcoin Knots Node
//!
//! Provides wallet functionality for Bitcoin (BTC) via Bitcoin Knots JSON-RPC:
//! - Address generation (bech32/segwit and legacy)
//! - Balance queries (confirmed + unconfirmed)
//! - Transaction sending
//! - Transaction history
//! - UTXO management
//! - Node sync status
//!
//! Bitcoin Knots Configuration:
//! - RPC Endpoint: http://5.79.79.158:8332 (Delta)
//! - Network: Mainnet
//! - Auth: rpcauth (v28+ format)
//!
//! v9.6.5: Initial implementation (parity with zcash_rpc.rs)

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Bitcoin RPC endpoint configuration
#[derive(Debug, Clone)]
pub struct BitcoinRpcConfig {
    /// RPC endpoint URL (default: http://5.79.79.158:8332)
    pub rpc_url: String,

    /// RPC username for authentication
    pub rpc_user: String,

    /// RPC password for authentication
    pub rpc_password: String,

    /// Request timeout in seconds
    pub timeout_secs: u64,

    /// Bitcoin network (mainnet, testnet, regtest)
    pub network: String,
}

impl Default for BitcoinRpcConfig {
    fn default() -> Self {
        Self {
            rpc_url: std::env::var("BTC_RPC_URL")
                .unwrap_or_else(|_| "http://5.79.79.158:8332".to_string()),
            rpc_user: std::env::var("BTC_RPC_USER")
                .unwrap_or_else(|_| "qnk".to_string()),
            rpc_password: std::env::var("BTC_RPC_PASS")
                .unwrap_or_default(),
            timeout_secs: 30,
            network: "mainnet".to_string(),
        }
    }
}

/// Bitcoin address types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BitcoinAddressType {
    /// Legacy P2PKH address (1...)
    Legacy,
    /// SegWit P2SH-P2WPKH address (3...)
    P2shSegwit,
    /// Native SegWit bech32 address (bc1q...)
    Bech32,
    /// Taproot bech32m address (bc1p...)
    Bech32m,
}

impl BitcoinAddressType {
    pub fn as_rpc_str(&self) -> &str {
        match self {
            BitcoinAddressType::Legacy => "legacy",
            BitcoinAddressType::P2shSegwit => "p2sh-segwit",
            BitcoinAddressType::Bech32 => "bech32",
            BitcoinAddressType::Bech32m => "bech32m",
        }
    }
}

/// Bitcoin balance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinBalance {
    /// Confirmed balance in satoshis
    pub confirmed_sats: u64,
    /// Unconfirmed (pending) balance in satoshis
    pub unconfirmed_sats: i64,
    /// Total balance in BTC (display)
    pub confirmed_btc: f64,
    /// Unconfirmed balance in BTC (display)
    pub unconfirmed_btc: f64,
    /// Total (confirmed + unconfirmed) in BTC
    pub total_btc: f64,
}

/// Bitcoin UTXO (unspent transaction output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinUtxo {
    pub txid: String,
    pub vout: u32,
    pub address: String,
    pub amount_sats: u64,
    pub confirmations: u64,
    pub spendable: bool,
}

/// Bitcoin transaction details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinTransaction {
    pub txid: String,
    pub address: Option<String>,
    pub amount_btc: f64,
    pub amount_sats: i64,
    pub fee_btc: Option<f64>,
    pub confirmations: i64,
    pub block_height: Option<u64>,
    pub timestamp: u64,
    pub category: String, // send, receive, generate
    pub label: Option<String>,
}

/// Bitcoin blockchain info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinNodeInfo {
    pub chain: String,
    pub blocks: u64,
    pub headers: u64,
    pub verification_progress: f64,
    pub is_synced: bool,
    pub size_on_disk: u64,
    pub pruned: bool,
    pub difficulty: f64,
}

/// Bitcoin RPC client for wallet operations
#[derive(Clone)]
pub struct BitcoinRpcClient {
    config: BitcoinRpcConfig,
    http_client: reqwest::Client,
    /// Cached wallet addresses (label -> address)
    address_cache: Arc<RwLock<std::collections::HashMap<String, String>>>,
}

impl BitcoinRpcClient {
    /// Create a new Bitcoin RPC client with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(BitcoinRpcConfig::default())
    }

    /// Create a new Bitcoin RPC client with custom configuration
    pub fn with_config(config: BitcoinRpcConfig) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        info!(
            "₿ [BTC RPC] Client initialized with endpoint: {}",
            config.rpc_url
        );

        Ok(Self {
            config,
            http_client,
            address_cache: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Make a JSON-RPC call to Bitcoin Knots
    async fn call_rpc(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        debug!("₿ [BTC RPC] Calling method: {} with params: {}", method, params);

        let request_body = json!({
            "jsonrpc": "1.0",
            "id": "quillon-bridge",
            "method": method,
            "params": params,
        });

        let response = self
            .http_client
            .post(&self.config.rpc_url)
            .basic_auth(&self.config.rpc_user, Some(&self.config.rpc_password))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| anyhow!("BTC RPC request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("BTC RPC returned error status {}: {}", status, body));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse BTC RPC response: {}", e))?;

        // Check for RPC error
        if let Some(error) = response_json.get("error") {
            if !error.is_null() {
                let code = error.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
                let message = error.get("message").and_then(|m| m.as_str()).unwrap_or("Unknown error");
                return Err(anyhow!("BTC RPC error ({}): {}", code, message));
            }
        }

        response_json
            .get("result")
            .ok_or_else(|| anyhow!("Missing 'result' field in BTC RPC response"))
            .map(|v| v.clone())
    }

    /// Check if Bitcoin Knots node is synced and ready
    pub async fn get_node_info(&self) -> Result<BitcoinNodeInfo> {
        let info = self.call_rpc("getblockchaininfo", json!([])).await?;

        let blocks = info.get("blocks").and_then(|v| v.as_u64()).unwrap_or(0);
        let headers = info.get("headers").and_then(|v| v.as_u64()).unwrap_or(0);
        let verification_progress = info
            .get("verificationprogress")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let chain = info
            .get("chain")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let size_on_disk = info.get("size_on_disk").and_then(|v| v.as_u64()).unwrap_or(0);
        let pruned = info.get("pruned").and_then(|v| v.as_bool()).unwrap_or(false);
        let difficulty = info.get("difficulty").and_then(|v| v.as_f64()).unwrap_or(0.0);

        let is_synced = blocks >= headers.saturating_sub(1) && verification_progress > 0.999;

        info!(
            "₿ [BTC RPC] Node: chain={}, blocks={}, headers={}, progress={:.2}%, synced={}",
            chain, blocks, headers, verification_progress * 100.0, is_synced
        );

        Ok(BitcoinNodeInfo {
            chain,
            blocks,
            headers,
            verification_progress,
            is_synced,
            size_on_disk,
            pruned,
            difficulty,
        })
    }

    /// Generate a new Bitcoin address
    pub async fn generate_address(
        &self,
        label: &str,
        addr_type: BitcoinAddressType,
    ) -> Result<String> {
        // Check cache first
        let cache_key = format!("{}:{:?}", label, addr_type);
        {
            let cache = self.address_cache.read().await;
            if let Some(cached_addr) = cache.get(&cache_key) {
                debug!("₿ [BTC RPC] Using cached address: {}", cached_addr);
                return Ok(cached_addr.clone());
            }
        }

        // Generate new address via RPC
        let result = self
            .call_rpc(
                "getnewaddress",
                json!([label, addr_type.as_rpc_str()]),
            )
            .await?;

        let address = result
            .as_str()
            .ok_or_else(|| anyhow!("BTC RPC returned non-string address"))?
            .to_string();

        info!("₿ [BTC RPC] Generated {:?} address for '{}': {}", addr_type, label, address);

        // Cache the address
        {
            let mut cache = self.address_cache.write().await;
            cache.insert(cache_key, address.clone());
        }

        Ok(address)
    }

    /// Get addresses by label (returns all addresses for a given label)
    pub async fn get_addresses_by_label(&self, label: &str) -> Result<Vec<String>> {
        match self.call_rpc("getaddressesbylabel", json!([label])).await {
            Ok(result) => {
                if let Some(obj) = result.as_object() {
                    Ok(obj.keys().cloned().collect())
                } else {
                    Ok(Vec::new())
                }
            }
            Err(_) => Ok(Vec::new()),
        }
    }

    /// Get wallet balance (confirmed + unconfirmed)
    pub async fn get_balance(&self) -> Result<BitcoinBalance> {
        // Get confirmed balance
        let confirmed_result = self.call_rpc("getbalance", json!(["*", 1])).await?;
        let confirmed_btc = confirmed_result.as_f64().unwrap_or(0.0);

        // Get unconfirmed balance
        let unconfirmed_result = self.call_rpc("getunconfirmedbalance", json!([])).await
            .unwrap_or(json!(0.0));
        let unconfirmed_btc = unconfirmed_result.as_f64().unwrap_or(0.0);

        let confirmed_sats = (confirmed_btc * 100_000_000.0).round() as u64;
        let unconfirmed_sats = (unconfirmed_btc * 100_000_000.0).round() as i64;
        let total_btc = confirmed_btc + unconfirmed_btc;

        debug!(
            "₿ [BTC RPC] Balance: confirmed={:.8} BTC ({} sats), unconfirmed={:.8} BTC",
            confirmed_btc, confirmed_sats, unconfirmed_btc
        );

        Ok(BitcoinBalance {
            confirmed_sats,
            unconfirmed_sats,
            confirmed_btc,
            unconfirmed_btc,
            total_btc,
        })
    }

    /// Get balance for a specific label (e.g., per-user balance tracking)
    pub async fn get_balance_for_label(&self, label: &str) -> Result<BitcoinBalance> {
        // getbalance with label filter
        let confirmed_result = self.call_rpc("getbalance", json!([label, 1])).await
            .unwrap_or(json!(0.0));
        let confirmed_btc = confirmed_result.as_f64().unwrap_or(0.0);

        // For per-label unconfirmed, we need to check receivedbyaddress
        let confirmed_sats = (confirmed_btc * 100_000_000.0).round() as u64;

        Ok(BitcoinBalance {
            confirmed_sats,
            unconfirmed_sats: 0,
            confirmed_btc,
            unconfirmed_btc: 0.0,
            total_btc: confirmed_btc,
        })
    }

    /// List unspent transaction outputs (UTXOs)
    pub async fn list_unspent(
        &self,
        min_conf: u32,
        max_conf: u32,
        addresses: Option<&[String]>,
    ) -> Result<Vec<BitcoinUtxo>> {
        let params = if let Some(addrs) = addresses {
            json!([min_conf, max_conf, addrs])
        } else {
            json!([min_conf, max_conf])
        };

        let result = self.call_rpc("listunspent", params).await?;

        let utxo_array = result
            .as_array()
            .ok_or_else(|| anyhow!("Invalid UTXO list response"))?;

        let mut utxos = Vec::new();
        for utxo_json in utxo_array {
            let txid = utxo_json.get("txid").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let vout = utxo_json.get("vout").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let address = utxo_json.get("address").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let amount = utxo_json.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let confirmations = utxo_json.get("confirmations").and_then(|v| v.as_u64()).unwrap_or(0);
            let spendable = utxo_json.get("spendable").and_then(|v| v.as_bool()).unwrap_or(false);

            utxos.push(BitcoinUtxo {
                txid,
                vout,
                address,
                amount_sats: (amount * 100_000_000.0).round() as u64,
                confirmations,
                spendable,
            });
        }

        debug!("₿ [BTC RPC] Found {} UTXOs", utxos.len());
        Ok(utxos)
    }

    /// Send BTC to an address
    pub async fn send_to_address(
        &self,
        address: &str,
        amount_btc: f64,
        comment: Option<&str>,
    ) -> Result<String> {
        info!("₿ [BTC RPC] Sending {:.8} BTC to {}", amount_btc, address);

        let params = json!([
            address,
            amount_btc,
            comment.unwrap_or("QNK Bridge"),   // comment
            "QNK Bridge withdrawal",           // comment_to
            false,                             // subtractfeefromamount
            true,                              // replaceable (RBF)
        ]);

        let result = self.call_rpc("sendtoaddress", params).await?;
        let txid = result
            .as_str()
            .ok_or_else(|| anyhow!("BTC RPC returned non-string txid"))?
            .to_string();

        info!("₿ [BTC RPC] Transaction sent: {}", txid);
        Ok(txid)
    }

    /// Get transaction history
    pub async fn get_transaction_history(&self, count: usize) -> Result<Vec<BitcoinTransaction>> {
        let params = json!([count, 0, true]);
        let result = self.call_rpc("listtransactions", params).await?;

        let tx_array = result
            .as_array()
            .ok_or_else(|| anyhow!("Invalid transaction list response"))?;

        let mut transactions = Vec::new();
        for tx_json in tx_array {
            match Self::parse_transaction(tx_json) {
                Ok(tx) => transactions.push(tx),
                Err(e) => warn!("₿ [BTC RPC] Failed to parse tx: {}", e),
            }
        }

        // Reverse so most recent first
        transactions.reverse();

        debug!("₿ [BTC RPC] Retrieved {} transactions", transactions.len());
        Ok(transactions)
    }

    /// Get transaction details by txid
    pub async fn get_transaction(&self, txid: &str) -> Result<BitcoinTransaction> {
        let result = self.call_rpc("gettransaction", json!([txid, true])).await?;
        Self::parse_wallet_transaction(&result)
    }

    /// Get current estimated fee rate (sat/vB)
    pub async fn estimate_fee(&self, conf_target: u32) -> Result<f64> {
        let result = self
            .call_rpc("estimatesmartfee", json!([conf_target]))
            .await?;

        let fee_rate = result
            .get("feerate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.00001); // Default ~1 sat/vB

        // Convert BTC/kB to sat/vB
        let sat_per_vb = fee_rate * 100_000.0; // BTC/kB * 100_000_000 sats/BTC / 1000 bytes

        debug!("₿ [BTC RPC] Estimated fee: {:.1} sat/vB (target {} blocks)", sat_per_vb, conf_target);
        Ok(sat_per_vb)
    }

    /// Get block count (current height)
    pub async fn get_block_count(&self) -> Result<u64> {
        let result = self.call_rpc("getblockcount", json!([])).await?;
        result.as_u64().ok_or_else(|| anyhow!("Invalid block count"))
    }

    /// Validate a Bitcoin address
    pub async fn validate_address(&self, address: &str) -> Result<bool> {
        let result = self.call_rpc("validateaddress", json!([address])).await?;
        Ok(result.get("isvalid").and_then(|v| v.as_bool()).unwrap_or(false))
    }

    /// Get total amount received at a specific address (min_conf = min confirmations)
    pub async fn get_received_by_address(&self, address: &str, min_conf: u32) -> Result<f64> {
        // Use wallet endpoint for getreceivedbyaddress
        let url = format!("{}/wallet/qug-bridge", self.config.rpc_url);
        let body = json!({
            "jsonrpc": "1.0",
            "id": "donation",
            "method": "getreceivedbyaddress",
            "params": [address, min_conf],
        });
        let response = self.http_client.post(&url)
            .basic_auth(&self.config.rpc_user, Some(&self.config.rpc_password))
            .header("Content-Type", "application/json")
            .json(&body)
            .send().await?;
        let json: serde_json::Value = response.json().await?;
        Ok(json.get("result").and_then(|v| v.as_f64()).unwrap_or(0.0))
    }

    /// Parse transaction from listtransactions response
    fn parse_transaction(json: &serde_json::Value) -> Result<BitcoinTransaction> {
        let txid = json.get("txid").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing txid"))?.to_string();
        let address = json.get("address").and_then(|v| v.as_str()).map(|s| s.to_string());
        let amount_btc = json.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let fee_btc = json.get("fee").and_then(|v| v.as_f64());
        let confirmations = json.get("confirmations").and_then(|v| v.as_i64()).unwrap_or(0);
        let block_height = json.get("blockheight").and_then(|v| v.as_u64());
        let timestamp = json.get("time").and_then(|v| v.as_u64()).unwrap_or(0);
        let category = json.get("category").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
        let label = json.get("label").and_then(|v| v.as_str()).map(|s| s.to_string());

        Ok(BitcoinTransaction {
            txid,
            address,
            amount_btc,
            amount_sats: (amount_btc * 100_000_000.0).round() as i64,
            fee_btc,
            confirmations,
            block_height,
            timestamp,
            category,
            label,
        })
    }

    /// Parse transaction from gettransaction response (slightly different format)
    fn parse_wallet_transaction(json: &serde_json::Value) -> Result<BitcoinTransaction> {
        let txid = json.get("txid").and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing txid"))?.to_string();
        let amount_btc = json.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let fee_btc = json.get("fee").and_then(|v| v.as_f64());
        let confirmations = json.get("confirmations").and_then(|v| v.as_i64()).unwrap_or(0);
        let block_height = json.get("blockheight").and_then(|v| v.as_u64());
        let timestamp = json.get("time").and_then(|v| v.as_u64()).unwrap_or(0);

        // Get first detail entry for address/category
        let details = json.get("details").and_then(|v| v.as_array());
        let (address, category) = if let Some(d) = details.and_then(|a| a.first()) {
            (
                d.get("address").and_then(|v| v.as_str()).map(|s| s.to_string()),
                d.get("category").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
            )
        } else {
            (None, "unknown".to_string())
        };

        Ok(BitcoinTransaction {
            txid,
            address,
            amount_btc,
            amount_sats: (amount_btc * 100_000_000.0).round() as i64,
            fee_btc,
            confirmations,
            block_height,
            timestamp,
            category,
            label: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bitcoin_rpc_client_creation() {
        let client = BitcoinRpcClient::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_bitcoin_balance_serialization() {
        let balance = BitcoinBalance {
            confirmed_sats: 150000,
            unconfirmed_sats: 5000,
            confirmed_btc: 0.0015,
            unconfirmed_btc: 0.00005,
            total_btc: 0.00155,
        };

        let json = serde_json::to_string(&balance).unwrap();
        assert!(json.contains("\"confirmed_sats\":150000"));
    }

    #[test]
    fn test_address_type_rpc_str() {
        assert_eq!(BitcoinAddressType::Legacy.as_rpc_str(), "legacy");
        assert_eq!(BitcoinAddressType::Bech32.as_rpc_str(), "bech32");
        assert_eq!(BitcoinAddressType::Bech32m.as_rpc_str(), "bech32m");
    }
}
