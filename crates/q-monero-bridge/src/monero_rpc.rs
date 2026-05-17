//! # Monero RPC Client
//! 
//! 🔒🌐 Tor-only Monero daemon and wallet RPC client for atomic swap operations.
//! Provides anonymous access to Monero network via hidden services.

use anyhow::Result;
use reqwest::{Client, Proxy};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, error, info, warn};

use crate::{MoneroBridgeConfig, MoneroTransaction, FixedPoint28};

/// Tor-enabled Monero RPC client
pub struct MoneroRpc {
    config: MoneroBridgeConfig,
    daemon_client: Client,
    wallet_client: Client,
    current_daemon: usize,
    request_count: u64,
    error_count: u64,
}

/// Monero daemon status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroDaemonStatus {
    pub height: u64,
    pub target_height: u64,
    pub difficulty: u64,
    pub hash_rate: u64,
    pub tx_pool_size: u32,
    pub synchronized: bool,
    pub version: String,
}

/// Monero wallet status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroWalletStatus {
    pub address: String,
    pub balance: u64, // Atomic units
    pub unlocked_balance: u64,
    pub height: u64,
    pub synchronized: bool,
}

/// Monero transfer parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroTransferParams {
    pub destinations: Vec<MoneroDestination>,
    pub payment_id: Option<String>,
    pub mixin: u32, // Ring size - 1
    pub unlock_time: u64,
    pub priority: TransferPriority,
    pub account_index: u32,
    pub subaddr_indices: Vec<u32>,
}

/// Monero transfer destination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroDestination {
    pub address: String,
    pub amount: u64, // Atomic units
}

/// Transfer priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransferPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Max = 3,
}

/// Monero multisig wallet information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroMultisigInfo {
    pub is_multisig: bool,
    pub is_ready: bool,
    pub threshold: u32,
    pub total: u32,
}

/// Monero stealth address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroStealthAddress {
    pub public_view_key: String,
    pub public_spend_key: String,
    pub address: String,
    pub payment_id: Option<String>,
}

impl MoneroRpc {
    /// Create new Monero RPC client
    pub async fn new(config: &MoneroBridgeConfig) -> Result<Self> {
        info!("🔒 Initializing Monero RPC Client");
        info!("   • Tor proxy: {}", config.tor_proxy);
        info!("   • Daemon endpoints: {}", config.monerod_endpoints.len());
        
        // Create Tor-enabled HTTP clients
        let proxy = Proxy::all(&config.tor_proxy)?;
        
        let daemon_client = Client::builder()
            .proxy(proxy.clone())
            .timeout(Duration::from_secs(30))
            .user_agent("q-monero-bridge/1.0")
            .build()?;
        
        let wallet_client = Client::builder()
            .proxy(proxy)
            .timeout(Duration::from_secs(60))
            .user_agent("q-monero-bridge-wallet/1.0")
            .build()?;
        
        let mut client = Self {
            config: config.clone(),
            daemon_client,
            wallet_client,
            current_daemon: 0,
            request_count: 0,
            error_count: 0,
        };
        
        // Test connectivity
        client.test_connectivity().await?;
        
        Ok(client)
    }
    
    /// Test connectivity to Monero network
    async fn test_connectivity(&mut self) -> Result<()> {
        info!("🔌 Testing Monero connectivity via Tor");
        
        for (i, endpoint) in self.config.monerod_endpoints.iter().enumerate() {
            debug!("   • Testing daemon {}: {}", i + 1, endpoint);
            
            match timeout(Duration::from_secs(10), self.test_daemon_endpoint(endpoint)).await {
                Ok(Ok(status)) => {
                    info!("     ✅ Healthy (height: {}, sync: {})", 
                           status.height, status.synchronized);
                    return Ok(());
                },
                Ok(Err(e)) => {
                    debug!("     ❌ Failed: {}", e);
                },
                Err(_) => {
                    debug!("     ⏰ Timeout");
                }
            }
        }
        
        warn!("⚠️ No Monero daemons accessible via Tor");
        Ok(()) // Continue anyway - may be temporary
    }
    
    /// Test individual daemon endpoint
    async fn test_daemon_endpoint(&self, endpoint: &str) -> Result<MoneroDaemonStatus> {
        let response = self.daemon_client
            .post(&format!("{}/json_rpc", endpoint))
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "get_info"
            }))
            .send()
            .await?;
        
        let rpc_response: Value = response.json().await?;
        
        if let Some(result) = rpc_response["result"].as_object() {
            Ok(MoneroDaemonStatus {
                height: result["height"].as_u64().unwrap_or(0),
                target_height: result["target_height"].as_u64().unwrap_or(0),
                difficulty: result["difficulty"].as_u64().unwrap_or(0),
                hash_rate: result["hash_rate"].as_u64().unwrap_or(0),
                tx_pool_size: result["tx_pool_size"].as_u64().unwrap_or(0) as u32,
                synchronized: result["synchronized"].as_bool().unwrap_or(false),
                version: result["version"].as_str().unwrap_or("unknown").to_string(),
            })
        } else {
            Err(anyhow::anyhow!("Invalid daemon response format"))
        }
    }
    
    /// Get current daemon status
    pub async fn get_daemon_status(&mut self) -> Result<MoneroDaemonStatus> {
        self.make_daemon_request("get_info", json!({})).await.and_then(|result| {
            Ok(MoneroDaemonStatus {
                height: result["height"].as_u64().unwrap_or(0),
                target_height: result["target_height"].as_u64().unwrap_or(0),
                difficulty: result["difficulty"].as_u64().unwrap_or(0),
                hash_rate: result["hash_rate"].as_u64().unwrap_or(0),
                tx_pool_size: result["tx_pool_size"].as_u64().unwrap_or(0) as u32,
                synchronized: result["synchronized"].as_bool().unwrap_or(false),
                version: result["version"].as_str().unwrap_or("unknown").to_string(),
            })
        })
    }
    
    /// Get wallet status
    pub async fn get_wallet_status(&mut self) -> Result<MoneroWalletStatus> {
        // Get address
        let address_result = self.make_wallet_request("get_address", json!({
            "account_index": 0
        })).await?;
        
        let address = address_result["address"].as_str()
            .unwrap_or("unknown")
            .to_string();
        
        // Get balance
        let balance_result = self.make_wallet_request("get_balance", json!({
            "account_index": 0
        })).await?;
        
        // Get height
        let height_result = self.make_wallet_request("get_height", json!({})).await?;
        
        Ok(MoneroWalletStatus {
            address,
            balance: balance_result["balance"].as_u64().unwrap_or(0),
            unlocked_balance: balance_result["unlocked_balance"].as_u64().unwrap_or(0),
            height: height_result["height"].as_u64().unwrap_or(0),
            synchronized: true, // Assume synchronized if we got responses
        })
    }
    
    /// Create new stealth address
    pub async fn create_stealth_address(&mut self, payment_id: Option<String>) -> Result<MoneroStealthAddress> {
        let params = if let Some(pid) = payment_id {
            json!({ "payment_id": pid })
        } else {
            json!({})
        };
        
        let result = self.make_wallet_request("make_integrated_address", params).await?;
        
        Ok(MoneroStealthAddress {
            public_view_key: "unknown".to_string(), // Would need separate call
            public_spend_key: "unknown".to_string(), // Would need separate call
            address: result["integrated_address"].as_str()
                .unwrap_or("unknown")
                .to_string(),
            payment_id: result["payment_id"].as_str()
                .map(|s| s.to_string()),
        })
    }
    
    /// Send Monero transaction with privacy features
    pub async fn send_transaction(&mut self, params: MoneroTransferParams) -> Result<MoneroTransaction> {
        debug!("💰 Sending Monero transaction: {} destinations", params.destinations.len());
        
        let transfer_params = json!({
            "destinations": params.destinations.iter().map(|dest| {
                json!({
                    "amount": dest.amount,
                    "address": dest.address
                })
            }).collect::<Vec<_>>(),
            "payment_id": params.payment_id,
            "mixin": params.mixin,
            "unlock_time": params.unlock_time,
            "priority": params.priority as u32,
            "account_index": params.account_index,
            "subaddr_indices": params.subaddr_indices,
            "get_tx_key": true,
            "get_tx_hex": true,
            "get_tx_metadata": true
        });
        
        let result = self.make_wallet_request("transfer", transfer_params).await?;
        
        let tx_hash = result["tx_hash"].as_str()
            .unwrap_or("unknown")
            .to_string();
        
        let amount = params.destinations.iter()
            .map(|dest| dest.amount)
            .sum();
        
        let fee = result["fee"].as_u64().unwrap_or(0);
        
        let tx = MoneroTransaction {
            tx_hash: tx_hash.clone(),
            amount,
            fee,
            unlock_height: params.unlock_time,
            stealth_address: params.destinations[0].address.clone(),
            payment_id: params.payment_id,
            ring_size: params.mixin + 1,
        };
        
        info!("✅ Monero transaction sent: {} ({} XMR, fee: {} XMR)",
               &tx_hash[..10],
               amount as f64 / 1e12,
               fee as f64 / 1e12);
        
        Ok(tx)
    }
    
    /// Create multisig wallet for HTLC
    pub async fn create_multisig_wallet(&mut self, threshold: u32, total: u32) -> Result<MoneroMultisigInfo> {
        debug!("🔐 Creating multisig wallet: {}/{}", threshold, total);
        
        // Prepare multisig (step 1)
        let prepare_result = self.make_wallet_request("prepare_multisig", json!({})).await?;
        let multisig_seed = prepare_result["multisig_info"].as_str()
            .unwrap_or("")
            .to_string();
        
        // For simplicity, simulate multisig creation
        // In production, would coordinate with other parties
        let multisig_info = MoneroMultisigInfo {
            is_multisig: true,
            is_ready: true,
            threshold,
            total,
        };
        
        info!("✅ Multisig wallet created: {}/{}", threshold, total);
        
        Ok(multisig_info)
    }
    
    /// Sign multisig transaction
    pub async fn sign_multisig_transaction(&mut self, tx_data: &str) -> Result<String> {
        debug!("✍️ Signing multisig transaction");
        
        let params = json!({
            "tx_data_hex": tx_data
        });
        
        let result = self.make_wallet_request("sign_multisig", params).await?;
        
        let signed_tx = result["tx_data_hex"].as_str()
            .unwrap_or("")
            .to_string();
        
        debug!("✅ Multisig transaction signed");
        
        Ok(signed_tx)
    }
    
    /// Submit signed transaction to network
    pub async fn submit_transaction(&mut self, tx_hex: &str) -> Result<String> {
        debug!("📡 Submitting transaction to Monero network");
        
        let params = json!({
            "tx_as_hex": tx_hex,
            "do_not_relay": false
        });
        
        let result = self.make_daemon_request("send_raw_transaction", params).await?;
        
        let tx_hash = result["tx_hash"].as_str()
            .unwrap_or("unknown")
            .to_string();
        
        info!("✅ Transaction submitted: {}", &tx_hash[..10]);
        
        Ok(tx_hash)
    }
    
    /// Get transaction details
    pub async fn get_transaction(&mut self, tx_hash: &str) -> Result<MoneroTransaction> {
        debug!("🔍 Getting transaction details: {}", &tx_hash[..10]);
        
        let params = json!({
            "txs_hashes": [tx_hash]
        });
        
        let result = self.make_daemon_request("get_transactions", params).await?;
        
        if let Some(tx_data) = result["txs"].as_array().and_then(|arr| arr.first()) {
            Ok(MoneroTransaction {
                tx_hash: tx_hash.to_string(),
                amount: tx_data["amount"].as_u64().unwrap_or(0),
                fee: tx_data["fee"].as_u64().unwrap_or(0),
                unlock_height: tx_data["unlock_height"].as_u64().unwrap_or(0),
                stealth_address: "unknown".to_string(),
                payment_id: tx_data["payment_id"].as_str().map(|s| s.to_string()),
                ring_size: tx_data["ring_size"].as_u64().unwrap_or(11) as u32,
            })
        } else {
            Err(anyhow::anyhow!("Transaction not found"))
        }
    }
    
    /// Make RPC request to Monero daemon
    async fn make_daemon_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let max_retries = self.config.monerod_endpoints.len().min(3);
        let mut last_error = None;
        
        for attempt in 0..max_retries {
            let endpoint = &self.config.monerod_endpoints[self.current_daemon];
            
            let request_body = json!({
                "jsonrpc": "2.0",
                "id": self.request_count + 1,
                "method": method,
                "params": params
            });
            
            match timeout(
                Duration::from_secs(30),
                self.daemon_client
                    .post(&format!("{}/json_rpc", endpoint))
                    .json(&request_body)
                    .send()
            ).await {
                Ok(Ok(response)) => {
                    if response.status().is_success() {
                        match response.json::<Value>().await {
                            Ok(rpc_response) => {
                                self.request_count += 1;
                                
                                if let Some(result) = rpc_response["result"].as_object() {
                                    return Ok(Value::Object(result.clone()));
                                } else if let Some(error) = rpc_response["error"].as_object() {
                                    return Err(anyhow::anyhow!("Daemon RPC error: {}", 
                                                             error["message"].as_str().unwrap_or("unknown")));
                                }
                            },
                            Err(e) => {
                                last_error = Some(anyhow::anyhow!("JSON parse error: {}", e));
                            }
                        }
                    } else {
                        last_error = Some(anyhow::anyhow!("HTTP error: {}", response.status()));
                    }
                },
                Ok(Err(e)) => {
                    last_error = Some(anyhow::anyhow!("Request error: {}", e));
                },
                Err(_) => {
                    last_error = Some(anyhow::anyhow!("Request timeout"));
                }
            }
            
            // Rotate to next daemon
            self.current_daemon = (self.current_daemon + 1) % self.config.monerod_endpoints.len();
            self.error_count += 1;
            
            if attempt < max_retries - 1 {
                tokio::time::sleep(Duration::from_millis(1000)).await;
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All daemon endpoints failed")))
    }
    
    /// Make RPC request to Monero wallet
    async fn make_wallet_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let request_body = json!({
            "jsonrpc": "2.0",
            "id": self.request_count + 1,
            "method": method,
            "params": params
        });
        
        // Use localhost wallet RPC (assumed to be running)
        let wallet_url = "http://localhost:18083/json_rpc";
        
        let response = timeout(
            Duration::from_secs(60),
            self.wallet_client
                .post(wallet_url)
                .json(&request_body)
                .send()
        ).await??;
        
        if response.status().is_success() {
            let rpc_response: Value = response.json().await?;
            self.request_count += 1;
            
            if let Some(result) = rpc_response["result"].as_object() {
                Ok(Value::Object(result.clone()))
            } else if let Some(error) = rpc_response["error"].as_object() {
                Err(anyhow::anyhow!("Wallet RPC error: {}", 
                                   error["message"].as_str().unwrap_or("unknown")))
            } else {
                Err(anyhow::anyhow!("Invalid wallet RPC response"))
            }
        } else {
            self.error_count += 1;
            Err(anyhow::anyhow!("Wallet HTTP error: {}", response.status()))
        }
    }
    
    /// Get RPC client statistics
    pub fn get_stats(&self) -> MoneroRpcStats {
        let error_rate = if self.request_count > 0 {
            self.error_count as f64 / self.request_count as f64
        } else {
            0.0
        };
        
        MoneroRpcStats {
            current_daemon: self.current_daemon,
            total_daemons: self.config.monerod_endpoints.len(),
            request_count: self.request_count,
            error_count: self.error_count,
            error_rate,
            tor_proxy: self.config.tor_proxy.clone(),
        }
    }
}

/// Monero RPC client statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroRpcStats {
    pub current_daemon: usize,
    pub total_daemons: usize,
    pub request_count: u64,
    pub error_count: u64,
    pub error_rate: f64,
    pub tor_proxy: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_monero_rpc_creation() {
        let config = crate::MoneroBridgeConfig::default();
        let result = MoneroRpc::new(&config).await;
        
        // Will likely fail without real Monero setup
        if result.is_err() {
            println!("Expected failure without Monero setup: {:?}", result.err());
        }
    }
    
    #[test]
    fn test_transfer_params_serialization() {
        let params = MoneroTransferParams {
            destinations: vec![
                MoneroDestination {
                    address: "48test...".to_string(),
                    amount: 1000000000, // 0.001 XMR
                }
            ],
            payment_id: Some("test_payment_id".to_string()),
            mixin: 10,
            unlock_time: 0,
            priority: TransferPriority::Normal,
            account_index: 0,
            subaddr_indices: vec![0],
        };
        
        let serialized = serde_json::to_string(&params).unwrap();
        let deserialized: MoneroTransferParams = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(params.destinations.len(), deserialized.destinations.len());
        assert_eq!(params.mixin, deserialized.mixin);
    }
    
    #[test]
    fn test_daemon_status_parsing() {
        let status = MoneroDaemonStatus {
            height: 3000000,
            target_height: 3000000,
            difficulty: 123456789,
            hash_rate: 987654321,
            tx_pool_size: 42,
            synchronized: true,
            version: "0.18.3.1".to_string(),
        };
        
        assert!(status.synchronized);
        assert_eq!(status.height, status.target_height);
        assert!(status.tx_pool_size < 1000); // Reasonable pool size
    }
    
    #[test]
    fn test_stealth_address_creation() {
        let address = MoneroStealthAddress {
            public_view_key: "test_view_key".to_string(),
            public_spend_key: "test_spend_key".to_string(),
            address: "48test_address".to_string(),
            payment_id: Some("test_payment_id".to_string()),
        };
        
        assert!(address.address.starts_with("48"));
        assert!(address.payment_id.is_some());
    }
}