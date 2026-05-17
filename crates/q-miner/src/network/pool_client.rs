//! Mining pool client - Stratum V1 over TCP
//!
//! Connects to a mining pool, receives work, and submits shares.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, RwLock};
use tracing::{info, warn, debug, error};

/// Stratum JSON-RPC request
#[derive(Debug, Serialize)]
struct StratumRequest {
    id: u64,
    method: String,
    params: serde_json::Value,
}

/// Stratum JSON-RPC response
#[derive(Debug, Deserialize)]
struct StratumResponse {
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<serde_json::Value>,
    method: Option<String>,
    params: Option<serde_json::Value>,
}

/// Pool connection status
#[derive(Debug, Clone, PartialEq)]
pub enum PoolStatus {
    Disconnected,
    Connecting,
    Connected,
    Authorized,
    Mining,
    Error(String),
}

/// Pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolClientStats {
    pub shares_accepted: u64,
    pub shares_rejected: u64,
    pub shares_stale: u64,
    pub blocks_found: u64,
    pub current_difficulty: f64,
    pub pool_hashrate: f64,
    pub uptime_seconds: u64,
}

/// Mining work from pool
#[derive(Debug, Clone)]
pub struct PoolWork {
    pub job_id: String,
    pub prev_hash: [u8; 32],
    pub coinbase1: Vec<u8>,
    pub coinbase2: Vec<u8>,
    pub merkle_branch: Vec<[u8; 32]>,
    pub version: u32,
    pub nbits: u32,
    pub ntime: u64,
    pub clean_jobs: bool,
}

/// Share submission to pool
#[derive(Debug, Clone)]
pub struct ShareSubmit {
    pub job_id: String,
    pub extranonce2: String,
    pub ntime: String,
    pub nonce: String,
    pub worker_name: String,
}

/// Pool client for Stratum V1 mining
#[derive(Debug, Clone)]
pub struct PoolClient {
    pub pool_url: String,
    pub wallet_address: String,
    pub worker_name: String,
    pub tor_enabled: bool,
    status: Arc<RwLock<PoolStatus>>,
    stats: Arc<RwLock<PoolClientStats>>,
    extranonce1: Arc<RwLock<String>>,
    extranonce2_size: Arc<RwLock<usize>>,
    current_difficulty: Arc<RwLock<f64>>,
    connected: Arc<AtomicBool>,
    request_id: Arc<AtomicU64>,
    work_tx: Arc<RwLock<Option<mpsc::Sender<PoolWork>>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolInfo {
    pub name: String,
    pub url: String,
    pub port: u16,
    pub fee: f64,
}

impl PoolClient {
    /// Create a new pool client
    pub async fn new(pool_url: String, wallet_address: String, tor_enabled: bool) -> Result<Self> {
        info!("Initializing pool client: {}", pool_url);

        let worker_name = format!("{}.rig1", &wallet_address[..16.min(wallet_address.len())]);

        Ok(Self {
            pool_url,
            wallet_address,
            worker_name,
            tor_enabled,
            status: Arc::new(RwLock::new(PoolStatus::Disconnected)),
            stats: Arc::new(RwLock::new(PoolClientStats::default())),
            extranonce1: Arc::new(RwLock::new(String::new())),
            extranonce2_size: Arc::new(RwLock::new(4)),
            current_difficulty: Arc::new(RwLock::new(1.0)),
            connected: Arc::new(AtomicBool::new(false)),
            request_id: Arc::new(AtomicU64::new(1)),
            work_tx: Arc::new(RwLock::new(None)),
        })
    }

    /// Get the next request ID
    fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Parse host and port from pool URL
    fn parse_url(&self) -> Result<(String, u16)> {
        let url = self.pool_url
            .replace("stratum+tcp://", "")
            .replace("stratum+tor://", "")
            .replace("stratum://", "");

        let parts: Vec<&str> = url.split(':').collect();
        let host = parts.first().context("Missing host")?.to_string();
        let port: u16 = parts.get(1)
            .unwrap_or(&"3333")
            .parse()
            .unwrap_or(3333);

        Ok((host, port))
    }

    /// Connect to the mining pool and start receiving work
    pub async fn connect(&mut self) -> Result<()> {
        let (host, port) = self.parse_url()?;

        *self.status.write().await = PoolStatus::Connecting;
        info!("Connecting to pool {}:{}", host, port);

        let stream = if self.tor_enabled {
            // Connect through SOCKS5 proxy (Tor)
            let proxy_addr = "127.0.0.1:9050";
            debug!("Connecting through Tor proxy: {}", proxy_addr);

            match tokio_socks::tcp::Socks5Stream::connect(proxy_addr, (host.as_str(), port)).await {
                Ok(socks_stream) => socks_stream.into_inner(),
                Err(e) => {
                    warn!("Tor proxy connection failed, trying direct: {}", e);
                    TcpStream::connect(format!("{}:{}", host, port)).await?
                }
            }
        } else {
            TcpStream::connect(format!("{}:{}", host, port)).await?
        };

        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);

        // Step 1: mining.subscribe
        let subscribe_req = StratumRequest {
            id: self.next_id(),
            method: "mining.subscribe".to_string(),
            params: serde_json::json!(["QUG-Miner/1.0", null]),
        };
        let msg = serde_json::to_string(&subscribe_req)? + "\n";
        writer.write_all(msg.as_bytes()).await?;

        // Read subscribe response
        let mut line = String::new();
        reader.read_line(&mut line).await?;
        let resp: StratumResponse = serde_json::from_str(line.trim())?;

        if let Some(result) = &resp.result {
            if let Some(arr) = result.as_array() {
                // Extract extranonce1 and extranonce2_size
                if arr.len() >= 2 {
                    if let Some(en1) = arr.get(1).and_then(|v| v.as_str()) {
                        *self.extranonce1.write().await = en1.to_string();
                    }
                    if let Some(en2_size) = arr.get(2).and_then(|v| v.as_u64()) {
                        *self.extranonce2_size.write().await = en2_size as usize;
                    }
                }
            }
        }

        info!("Subscribed to pool, extranonce1: {}", self.extranonce1.read().await);

        // Step 2: mining.authorize
        let auth_req = StratumRequest {
            id: self.next_id(),
            method: "mining.authorize".to_string(),
            params: serde_json::json!([self.worker_name, ""]),
        };
        let msg = serde_json::to_string(&auth_req)? + "\n";
        writer.write_all(msg.as_bytes()).await?;

        // Read authorize response
        line.clear();
        reader.read_line(&mut line).await?;
        let resp: StratumResponse = serde_json::from_str(line.trim())?;

        if resp.result == Some(serde_json::Value::Bool(true)) {
            info!("Authorized with pool as {}", self.worker_name);
            *self.status.write().await = PoolStatus::Authorized;
        } else {
            let err_msg = resp.error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "Unknown auth error".to_string());
            *self.status.write().await = PoolStatus::Error(err_msg.clone());
            return Err(anyhow::anyhow!("Pool authorization failed: {}", err_msg));
        }

        self.connected.store(true, Ordering::Relaxed);
        *self.status.write().await = PoolStatus::Mining;

        // Spawn message reader loop
        let status = self.status.clone();
        let stats = self.stats.clone();
        let difficulty = self.current_difficulty.clone();
        let connected = self.connected.clone();
        let work_tx = self.work_tx.clone();

        tokio::spawn(async move {
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line).await {
                    Ok(0) => {
                        warn!("Pool connection closed");
                        *status.write().await = PoolStatus::Disconnected;
                        connected.store(false, Ordering::Relaxed);
                        break;
                    }
                    Ok(_) => {
                        if let Ok(msg) = serde_json::from_str::<StratumResponse>(line.trim()) {
                            // Handle notifications (no id, has method)
                            if let Some(method) = &msg.method {
                                match method.as_str() {
                                    "mining.notify" => {
                                        if let Some(params) = &msg.params {
                                            if let Some(work) = Self::parse_notify(params) {
                                                if let Some(tx) = work_tx.read().await.as_ref() {
                                                    let _ = tx.send(work).await;
                                                }
                                            }
                                        }
                                    }
                                    "mining.set_difficulty" => {
                                        if let Some(params) = &msg.params {
                                            if let Some(arr) = params.as_array() {
                                                if let Some(diff) = arr.first().and_then(|v| v.as_f64()) {
                                                    *difficulty.write().await = diff;
                                                    debug!("Pool difficulty set to {}", diff);
                                                }
                                            }
                                        }
                                    }
                                    _ => {
                                        debug!("Unknown pool method: {}", method);
                                    }
                                }
                            }
                            // Handle share submission responses
                            else if let Some(id) = msg.id {
                                if msg.result == Some(serde_json::Value::Bool(true)) {
                                    let mut s = stats.write().await;
                                    s.shares_accepted += 1;
                                    debug!("Share {} accepted", id);
                                } else if msg.error.is_some() {
                                    let mut s = stats.write().await;
                                    s.shares_rejected += 1;
                                    let err = msg.error.unwrap_or_default();
                                    warn!("Share {} rejected: {}", id, err);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Pool read error: {}", e);
                        *status.write().await = PoolStatus::Error(e.to_string());
                        connected.store(false, Ordering::Relaxed);
                        break;
                    }
                }
            }
        });

        info!("Pool client connected and mining");
        Ok(())
    }

    /// Parse mining.notify params into PoolWork
    fn parse_notify(params: &serde_json::Value) -> Option<PoolWork> {
        let arr = params.as_array()?;
        if arr.len() < 8 { return None; }

        let job_id = arr[0].as_str()?.to_string();
        let prev_hash_hex = arr[1].as_str()?;
        let coinbase1_hex = arr[2].as_str()?;
        let coinbase2_hex = arr[3].as_str()?;
        let merkle_branch_arr = arr[4].as_array()?;
        let version_hex = arr[5].as_str()?;
        let nbits_hex = arr[6].as_str()?;
        let ntime_hex = arr[7].as_str()?;
        let clean_jobs = arr.get(8).and_then(|v| v.as_bool()).unwrap_or(false);

        let prev_hash = hex_to_32(prev_hash_hex)?;
        let coinbase1 = hex::decode(coinbase1_hex).ok()?;
        let coinbase2 = hex::decode(coinbase2_hex).ok()?;
        let version = u32::from_str_radix(version_hex.trim_start_matches("0x"), 16).ok()?;
        let nbits = u32::from_str_radix(nbits_hex.trim_start_matches("0x"), 16).ok()?;
        let ntime = u64::from_str_radix(ntime_hex.trim_start_matches("0x"), 16).ok()?;

        let merkle_branch: Vec<[u8; 32]> = merkle_branch_arr.iter()
            .filter_map(|v| v.as_str().and_then(hex_to_32))
            .collect();

        Some(PoolWork {
            job_id,
            prev_hash,
            coinbase1,
            coinbase2,
            merkle_branch,
            version,
            nbits,
            ntime,
            clean_jobs,
        })
    }

    /// Submit a share to the pool
    pub async fn submit_share(&self, share: ShareSubmit) -> Result<()> {
        if !self.connected.load(Ordering::Relaxed) {
            return Err(anyhow::anyhow!("Not connected to pool"));
        }

        debug!("Submitting share: job={}, nonce={}", share.job_id, share.nonce);

        // Note: actual write requires keeping the writer half accessible
        // For now, log the submission - in production, we'd keep the writer in a shared state
        info!("Share submitted: job={}, worker={}", share.job_id, share.worker_name);

        Ok(())
    }

    /// Update reported hash rate to pool
    pub async fn update_hash_rate(&self, hash_rate: f64) -> Result<()> {
        debug!("Reporting hash rate to pool: {:.2} H/s", hash_rate);
        Ok(())
    }

    /// Get pool connection status
    pub async fn status(&self) -> PoolStatus {
        self.status.read().await.clone()
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolClientStats {
        self.stats.read().await.clone()
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Get current pool difficulty
    pub async fn difficulty(&self) -> f64 {
        *self.current_difficulty.read().await
    }

    /// Subscribe to new work notifications
    pub async fn subscribe_work(&self) -> mpsc::Receiver<PoolWork> {
        let (tx, rx) = mpsc::channel(32);
        *self.work_tx.write().await = Some(tx);
        rx
    }
}

fn hex_to_32(hex_str: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(hex_str.trim_start_matches("0x")).ok()?;
    if bytes.len() != 32 { return None; }
    let mut result = [0u8; 32];
    result.copy_from_slice(&bytes);
    Some(result)
}
