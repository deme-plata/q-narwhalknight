//! # Tor-Only Solana RPC Client
//! 
//! 🧅🌞 Anonymous Solana RPC client that routes all requests through Tor hidden services.
//! Provides reliable access to Solana network data without IP leakage.

use anyhow::Result;
use reqwest::{Client, Proxy};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, info, warn, error};

use crate::{SolanaBridgeConfig, SolanaAccount};

/// Tor-enabled Solana RPC client
pub struct TorSolanaRpc {
    client: Client,
    config: SolanaBridgeConfig,
    current_endpoint: usize,
    request_count: u64,
    error_count: u64,
    last_health_check: Option<Instant>,
}

/// Solana RPC response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcResponse<T> {
    pub jsonrpc: String,
    pub result: Option<T>,
    pub error: Option<RpcError>,
    pub id: u64,
}

/// Solana RPC error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

/// Slot information response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotInfo {
    pub slot: u64,
    pub blockhash: String,
    pub parent_slot: u64,
    pub leader: String,
    pub timestamp: u64,
    pub transaction_count: u32,
}

impl TorSolanaRpc {
    /// Create new Tor-enabled Solana RPC client
    pub async fn new(config: &SolanaBridgeConfig) -> Result<Self> {
        info!("🧅 Initializing Tor-Only Solana RPC Client");
        info!("   • Tor proxy: {}", config.tor_proxy);
        info!("   • Solana endpoints: {} nodes", config.solana_rpc_endpoints.len());
        info!("   • Commitment level: {}", config.commitment_level);
        
        // Create HTTP client with Tor SOCKS5 proxy
        let proxy = Proxy::all(&config.tor_proxy)?;
        
        let client = Client::builder()
            .proxy(proxy)
            .timeout(Duration::from_secs(30))
            .user_agent("q-narwhal-solana-bridge/1.0")
            .danger_accept_invalid_certs(false) // Only accept valid TLS certs
            .build()?;
        
        // Test connectivity
        info!("🔌 Testing Tor connectivity to Solana nodes...");
        let mut client_instance = Self {
            client,
            config: config.clone(),
            current_endpoint: 0,
            request_count: 0,
            error_count: 0,
            last_health_check: None,
        };
        
        // Perform initial health check
        match client_instance.health_check().await {
            Ok(true) => info!("✅ Tor-Solana connectivity established"),
            Ok(false) => warn!("⚠️ Limited Tor-Solana connectivity"),
            Err(e) => {
                error!("❌ Failed to establish Tor-Solana connectivity: {}", e);
                // Continue anyway - may be temporary network issue
            }
        }
        
        Ok(client_instance)
    }
    
    /// Perform health check on Solana endpoints
    pub async fn health_check(&mut self) -> Result<bool> {
        let start_time = Instant::now();
        let mut successful_endpoints = 0;
        
        debug!("🏥 Performing health check on {} endpoints", 
               self.config.solana_rpc_endpoints.len());
        
        for (i, endpoint) in self.config.solana_rpc_endpoints.iter().enumerate() {
            debug!("   • Testing endpoint {}: {}", i + 1, endpoint);
            
            match timeout(Duration::from_secs(10), self.test_endpoint(endpoint)).await {
                Ok(Ok(slot)) => {
                    debug!("     ✅ Healthy (slot: {})", slot);
                    successful_endpoints += 1;
                },
                Ok(Err(e)) => {
                    debug!("     ❌ Unhealthy: {}", e);
                },
                Err(_) => {
                    debug!("     ⏰ Timeout");
                }
            }
        }
        
        let health_ratio = successful_endpoints as f64 / self.config.solana_rpc_endpoints.len() as f64;
        let check_duration = start_time.elapsed();
        
        info!("🏥 Health check completed in {:.1}s: {}/{} endpoints healthy ({:.1}%)",
               check_duration.as_secs_f64(),
               successful_endpoints, 
               self.config.solana_rpc_endpoints.len(),
               health_ratio * 100.0);
        
        self.last_health_check = Some(start_time);
        
        // Consider healthy if >50% of endpoints work
        Ok(health_ratio > 0.5)
    }
    
    /// Test individual endpoint connectivity
    async fn test_endpoint(&self, endpoint: &str) -> Result<u64> {
        let response = self.make_rpc_request_to_endpoint(
            endpoint,
            "getSlot",
            json!([{"commitment": "finalized"}]),
        ).await?;
        
        if let Some(slot) = response.result {
            if let Some(slot_num) = slot.as_u64() {
                Ok(slot_num)
            } else {
                Err(anyhow::anyhow!("Invalid slot format in response"))
            }
        } else if let Some(error) = response.error {
            Err(anyhow::anyhow!("RPC error: {}", error.message))
        } else {
            Err(anyhow::anyhow!("Empty response"))
        }
    }
    
    /// Make RPC request with automatic endpoint rotation
    async fn make_rpc_request(&mut self, method: &str, params: Value) -> Result<RpcResponse<Value>> {
        let max_retries = self.config.solana_rpc_endpoints.len().min(3);
        let mut last_error = None;
        
        for attempt in 0..max_retries {
            let endpoint = &self.config.solana_rpc_endpoints[self.current_endpoint].clone();
            
            match self.make_rpc_request_to_endpoint(endpoint, method, params.clone()).await {
                Ok(response) => {
                    self.request_count += 1;
                    return Ok(response);
                },
                Err(e) => {
                    warn!("❌ RPC request failed on endpoint {} (attempt {}): {}", 
                           self.current_endpoint + 1, attempt + 1, e);
                    
                    last_error = Some(e);
                    self.error_count += 1;
                    
                    // Rotate to next endpoint
                    self.current_endpoint = (self.current_endpoint + 1) % self.config.solana_rpc_endpoints.len();
                    
                    // Brief delay before retry
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All endpoints failed")))
    }
    
    /// Make RPC request to specific endpoint
    async fn make_rpc_request_to_endpoint(&self, endpoint: &str, method: &str, params: Value) -> Result<RpcResponse<Value>> {
        let request_id = self.request_count + 1;
        
        let request_body = json!({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        });
        
        debug!("🌐 {} -> {} (via Tor)", method, endpoint);
        
        let response = timeout(
            Duration::from_secs(20),
            self.client.post(endpoint)
                .json(&request_body)
                .send()
        ).await??;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("HTTP error: {}", response.status()));
        }
        
        let rpc_response: RpcResponse<Value> = response.json().await?;
        
        if let Some(error) = &rpc_response.error {
            debug!("🚫 RPC error: {} (code: {})", error.message, error.code);
        }
        
        Ok(rpc_response)
    }
    
    /// Get current slot information
    pub async fn get_slot_info(&mut self) -> Result<SlotInfo> {
        // Get current slot
        let slot_response = self.make_rpc_request(
            "getSlot",
            json!([{"commitment": self.config.commitment_level}])
        ).await?;
        
        let slot = slot_response.result
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("Invalid slot response"))?;
        
        // Get block information for this slot
        let block_response = self.make_rpc_request(
            "getBlock",
            json!([slot, {
                "encoding": "json",
                "transactionDetails": "none",
                "rewards": false,
                "commitment": self.config.commitment_level
            }])
        ).await?;
        
        if let Some(block_data) = block_response.result {
            Ok(SlotInfo {
                slot,
                blockhash: block_data["blockhash"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string(),
                parent_slot: block_data["parentSlot"]
                    .as_u64()
                    .unwrap_or(slot.saturating_sub(1)),
                leader: "unknown".to_string(), // Would need getSlotLeader call
                timestamp: block_data["blockTime"]
                    .as_u64()
                    .unwrap_or_else(|| {
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    }),
                transaction_count: block_data["transactions"]
                    .as_array()
                    .map(|arr| arr.len() as u32)
                    .unwrap_or(0),
            })
        } else {
            Err(anyhow::anyhow!("Failed to get block information"))
        }
    }
    
    /// Get finalized slot number
    pub async fn get_finalized_slot(&mut self) -> Result<u64> {
        let response = self.make_rpc_request(
            "getSlot",
            json!([{"commitment": "finalized"}])
        ).await?;
        
        response.result
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("Invalid finalized slot response"))
    }
    
    /// Get account information
    pub async fn get_account(&mut self, pubkey: &str) -> Result<SolanaAccount> {
        let response = self.make_rpc_request(
            "getAccountInfo",
            json!([pubkey, {
                "encoding": "base64",
                "commitment": self.config.commitment_level
            }])
        ).await?;
        
        if let Some(account_data) = response.result {
            if account_data.is_null() {
                return Err(anyhow::anyhow!("Account not found"));
            }
            
            let value = &account_data["value"];
            
            // Decode base64 data
            let data_array = value["data"].as_array()
                .ok_or_else(|| anyhow::anyhow!("Invalid account data format"))?;
            
            let data_b64 = data_array[0].as_str()
                .ok_or_else(|| anyhow::anyhow!("Missing account data"))?;
            
            let data = base64::decode(data_b64)
                .map_err(|e| anyhow::anyhow!("Failed to decode account data: {}", e))?;
            
            Ok(SolanaAccount {
                pubkey: pubkey.to_string(),
                lamports: value["lamports"].as_u64().unwrap_or(0),
                data,
                owner: value["owner"].as_str().unwrap_or("unknown").to_string(),
                executable: value["executable"].as_bool().unwrap_or(false),
                rent_epoch: value["rentEpoch"].as_u64().unwrap_or(0),
            })
        } else if let Some(error) = response.error {
            Err(anyhow::anyhow!("RPC error: {}", error.message))
        } else {
            Err(anyhow::anyhow!("Empty account response"))
        }
    }
    
    /// Get blockhash for specific slot
    pub async fn get_blockhash_for_slot(&mut self, slot: u64) -> Result<String> {
        let response = self.make_rpc_request(
            "getBlock",
            json!([slot, {
                "encoding": "json", 
                "transactionDetails": "none",
                "rewards": false,
                "commitment": "finalized"
            }])
        ).await?;
        
        if let Some(block_data) = response.result {
            block_data["blockhash"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow::anyhow!("Missing blockhash in response"))
        } else {
            Err(anyhow::anyhow!("Failed to get block for slot {}", slot))
        }
    }
    
    /// Get slot leader for specific slot
    pub async fn get_slot_leader(&mut self, slot: u64) -> Result<String> {
        let response = self.make_rpc_request(
            "getSlotLeader",
            json!([{"commitment": "finalized"}])
        ).await?;
        
        response.result
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("Invalid slot leader response"))
    }
    
    /// Get recent blockhash
    pub async fn get_recent_blockhash(&mut self) -> Result<String> {
        let response = self.make_rpc_request(
            "getRecentBlockhash",
            json!([{"commitment": self.config.commitment_level}])
        ).await?;
        
        if let Some(result) = response.result {
            result["value"]["blockhash"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| anyhow::anyhow!("Missing recent blockhash"))
        } else {
            Err(anyhow::anyhow!("Failed to get recent blockhash"))
        }
    }
    
    /// Get multiple accounts in batch
    pub async fn get_multiple_accounts(&mut self, pubkeys: Vec<String>) -> Result<Vec<Option<SolanaAccount>>> {
        if pubkeys.is_empty() {
            return Ok(Vec::new());
        }
        
        // Solana RPC limits batch requests to 100 accounts
        const BATCH_SIZE: usize = 100;
        let mut results = Vec::new();
        
        for chunk in pubkeys.chunks(BATCH_SIZE) {
            let response = self.make_rpc_request(
                "getMultipleAccounts",
                json!([chunk, {
                    "encoding": "base64",
                    "commitment": self.config.commitment_level
                }])
            ).await?;
            
            if let Some(result) = response.result {
                if let Some(accounts) = result["value"].as_array() {
                    for (i, account_data) in accounts.iter().enumerate() {
                        if account_data.is_null() {
                            results.push(None);
                        } else {
                            let pubkey = &chunk[i];
                            let value = account_data;
                            
                            // Decode account data
                            let data_array = value["data"].as_array()
                                .unwrap_or(&Vec::new());
                            
                            let data = if data_array.is_empty() {
                                Vec::new()
                            } else {
                                let data_b64 = data_array[0].as_str().unwrap_or("");
                                base64::decode(data_b64).unwrap_or_default()
                            };
                            
                            results.push(Some(SolanaAccount {
                                pubkey: pubkey.clone(),
                                lamports: value["lamports"].as_u64().unwrap_or(0),
                                data,
                                owner: value["owner"].as_str().unwrap_or("unknown").to_string(),
                                executable: value["executable"].as_bool().unwrap_or(false),
                                rent_epoch: value["rentEpoch"].as_u64().unwrap_or(0),
                            }));
                        }
                    }
                } else {
                    return Err(anyhow::anyhow!("Invalid multiple accounts response format"));
                }
            } else {
                return Err(anyhow::anyhow!("Failed to get multiple accounts"));
            }
        }
        
        debug!("📦 Fetched {}/{} accounts via Tor", 
               results.iter().filter(|r| r.is_some()).count(),
               pubkeys.len());
        
        Ok(results)
    }
    
    /// Get RPC client statistics
    pub fn get_stats(&self) -> TorRpcStats {
        let health_check_age = self.last_health_check
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(u64::MAX);
        
        let error_rate = if self.request_count > 0 {
            self.error_count as f64 / self.request_count as f64
        } else {
            0.0
        };
        
        TorRpcStats {
            current_endpoint: self.current_endpoint,
            total_endpoints: self.config.solana_rpc_endpoints.len(),
            request_count: self.request_count,
            error_count: self.error_count,
            error_rate,
            health_check_age_seconds: health_check_age,
            proxy_url: self.config.tor_proxy.clone(),
            commitment_level: self.config.commitment_level.clone(),
        }
    }
}

/// Tor RPC client statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorRpcStats {
    pub current_endpoint: usize,
    pub total_endpoints: usize,
    pub request_count: u64,
    pub error_count: u64,
    pub error_rate: f64,
    pub health_check_age_seconds: u64,
    pub proxy_url: String,
    pub commitment_level: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SolanaBridgeConfig;
    
    #[tokio::test]
    async fn test_tor_rpc_creation() {
        let config = SolanaBridgeConfig::default();
        
        // This will likely fail without real Tor setup
        let result = TorSolanaRpc::new(&config).await;
        
        if result.is_err() {
            println!("Expected failure without Tor setup: {:?}", result.err());
        }
    }
    
    #[test]
    fn test_rpc_error_serialization() {
        let error = RpcError {
            code: -32603,
            message: "Internal error".to_string(),
            data: Some(json!({"details": "Connection timeout"})),
        };
        
        let serialized = serde_json::to_string(&error).unwrap();
        let deserialized: RpcError = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(error.code, deserialized.code);
        assert_eq!(error.message, deserialized.message);
    }
    
    #[test]
    fn test_slot_info_structure() {
        let slot_info = SlotInfo {
            slot: 123456789,
            blockhash: "11111111111111111111111111111111111111111111".to_string(),
            parent_slot: 123456788,
            leader: "validator_pubkey".to_string(),
            timestamp: 1703097600,
            transaction_count: 1234,
        };
        
        // Should serialize without issues
        let serialized = serde_json::to_string(&slot_info).unwrap();
        let deserialized: SlotInfo = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(slot_info.slot, deserialized.slot);
        assert_eq!(slot_info.transaction_count, deserialized.transaction_count);
    }
    
    #[test]
    fn test_tor_rpc_stats() {
        let stats = TorRpcStats {
            current_endpoint: 1,
            total_endpoints: 3,
            request_count: 100,
            error_count: 5,
            error_rate: 0.05,
            health_check_age_seconds: 60,
            proxy_url: "socks5://127.0.0.1:9050".to_string(),
            commitment_level: "finalized".to_string(),
        };
        
        assert_eq!(stats.error_rate, 0.05);
        assert!(stats.health_check_age_seconds < 120);
    }
}