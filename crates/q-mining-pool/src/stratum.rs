//! Stratum V1 protocol implementation for mining pool

use bytes::{Buf, BytesMut};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};
use parking_lot::RwLock;

use crate::config::StratumConfig;
use crate::error::{PoolResult, StratumError};
use crate::job::MiningJob;
use crate::share::ShareSubmission;
use crate::worker::{Worker, WorkerId, WorkerManager};

/// Stratum JSON-RPC request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumMessage {
    /// Request ID
    pub id: Option<u64>,

    /// Method name
    pub method: String,

    /// Parameters
    pub params: Vec<Value>,
}

/// Stratum JSON-RPC response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumResponse {
    /// Request ID
    pub id: Option<u64>,

    /// Result (null if error)
    pub result: Option<Value>,

    /// Error (null if success)
    pub error: Option<Value>,
}

impl StratumResponse {
    /// Create success response
    pub fn success(id: Option<u64>, result: Value) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create error response
    pub fn error(id: Option<u64>, code: i32, message: &str) -> Self {
        Self {
            id,
            result: None,
            error: Some(json!([code, message, null])),
        }
    }

    /// Serialize to JSON line
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_default() + "\n"
    }
}

/// Stratum notification (server -> client)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumNotification {
    /// Always null for notifications
    pub id: Option<()>,

    /// Method name
    pub method: String,

    /// Parameters
    pub params: Vec<Value>,
}

impl StratumNotification {
    /// Create job notification
    pub fn mining_notify(job: &MiningJob, extranonce1: &str) -> Self {
        Self {
            id: None,
            method: "mining.notify".to_string(),
            params: job.to_stratum_params(extranonce1),
        }
    }

    /// Create difficulty notification
    pub fn mining_set_difficulty(difficulty: f64) -> Self {
        Self {
            id: None,
            method: "mining.set_difficulty".to_string(),
            params: vec![json!(difficulty)],
        }
    }

    /// Serialize to JSON line
    pub fn to_json_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_default() + "\n"
    }
}

/// Stratum connection handler
pub struct StratumConnection {
    /// Worker for this connection
    worker: Option<Arc<Worker>>,

    /// Connection address
    addr: SocketAddr,

    /// Subscribed
    subscribed: bool,

    /// Authorized
    authorized: bool,

    /// Extranonce1
    extranonce1: String,

    /// Extranonce2 size
    extranonce2_size: u8,
}

impl StratumConnection {
    /// Create new connection
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            worker: None,
            addr,
            subscribed: false,
            authorized: false,
            extranonce1: String::new(),
            extranonce2_size: 4,
        }
    }
}

/// Pool event for broadcasting to connections
#[derive(Debug, Clone)]
pub enum PoolEvent {
    /// New job available
    NewJob(Arc<MiningJob>),

    /// Difficulty update
    DifficultyUpdate(f64),

    /// Block found — tell all workers to abandon current work (POOL-004)
    CleanJobs,

    /// Pool shutdown
    Shutdown,
}

/// Stratum server
pub struct StratumServer {
    /// Configuration
    config: StratumConfig,

    /// Minimum share difficulty (from VardiffConfig) — used for POOL-002 synchronous floor check
    min_difficulty: f64,

    /// Worker manager
    worker_manager: Arc<WorkerManager>,

    /// Event broadcaster
    event_tx: broadcast::Sender<PoolEvent>,

    /// Active connections count
    connection_count: Arc<RwLock<usize>>,

    /// Connections per IP
    connections_per_ip: Arc<RwLock<HashMap<String, usize>>>,
}

impl StratumServer {
    /// Create new stratum server
    pub fn new(config: StratumConfig, min_difficulty: f64, worker_manager: Arc<WorkerManager>) -> Self {
        let (event_tx, _) = broadcast::channel(1024);

        Self {
            config,
            min_difficulty,
            worker_manager,
            event_tx,
            connection_count: Arc::new(RwLock::new(0)),
            connections_per_ip: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get event sender for broadcasting
    pub fn event_sender(&self) -> broadcast::Sender<PoolEvent> {
        self.event_tx.clone()
    }

    /// Broadcast new job to all workers
    pub fn broadcast_job(&self, job: Arc<MiningJob>) {
        let _ = self.event_tx.send(PoolEvent::NewJob(job));
    }

    /// Broadcast clean-jobs notification to all workers (call after block found + invalidate_all)
    pub fn broadcast_clean_jobs(&self) {
        tracing::debug!("[POOL] broadcast_clean_jobs: signaling all workers to abandon stale work");
        let _ = self.event_tx.send(PoolEvent::CleanJobs);
    }

    /// Start the stratum server
    pub async fn start(
        self: Arc<Self>,
        mut share_tx: mpsc::Sender<(WorkerId, ShareSubmission)>,
    ) -> PoolResult<()> {
        let addr = format!("{}:{}", self.config.bind_address, self.config.port);
        let listener = TcpListener::bind(&addr).await?;

        tracing::info!(
            addr = %addr,
            max_connections = self.config.max_connections,
            "Stratum server listening"
        );

        loop {
            let (stream, addr) = listener.accept().await?;

            // Check connection limits
            if !self.can_accept_connection(&addr) {
                tracing::warn!(addr = %addr, "Connection rejected: limit exceeded");
                continue;
            }

            let server = Arc::clone(&self);
            let share_tx = share_tx.clone();
            let event_rx = self.event_tx.subscribe();

            tokio::spawn(async move {
                if let Err(e) = server.handle_connection(stream, addr, share_tx, event_rx).await {
                    tracing::debug!(addr = %addr, error = %e, "Connection error");
                }
                server.on_connection_closed(&addr);
            });
        }
    }

    /// Check if we can accept a new connection
    fn can_accept_connection(&self, addr: &SocketAddr) -> bool {
        let count = *self.connection_count.read();
        if count >= self.config.max_connections {
            return false;
        }

        let ip = addr.ip().to_string();
        let per_ip = self.connections_per_ip.read();
        let ip_count = per_ip.get(&ip).copied().unwrap_or(0);

        if ip_count >= 50 { // Max 50 connections per IP
            return false;
        }

        // Check if IP is banned
        if self.worker_manager.is_ip_banned(&ip) {
            return false;
        }

        // Update counters
        *self.connection_count.write() += 1;
        *self.connections_per_ip.write().entry(ip).or_insert(0) += 1;

        true
    }

    /// Called when connection closes
    fn on_connection_closed(&self, addr: &SocketAddr) {
        *self.connection_count.write() -= 1;

        let ip = addr.ip().to_string();
        let mut per_ip = self.connections_per_ip.write();
        if let Some(count) = per_ip.get_mut(&ip) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                per_ip.remove(&ip);
            }
        }
    }

    /// Handle a single connection
    async fn handle_connection(
        &self,
        stream: TcpStream,
        addr: SocketAddr,
        share_tx: mpsc::Sender<(WorkerId, ShareSubmission)>,
        mut event_rx: broadcast::Receiver<PoolEvent>,
    ) -> PoolResult<()> {
        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);
        let mut line = String::new();

        let mut conn = StratumConnection::new(addr);

        tracing::debug!(addr = %addr, "New stratum connection");

        loop {
            tokio::select! {
                // Read from client
                result = reader.read_line(&mut line) => {
                    match result {
                        Ok(0) => {
                            // Connection closed
                            break;
                        }
                        Ok(_) => {
                            if let Some(response) = self.handle_message(&line, &mut conn, &share_tx).await? {
                                writer.write_all(response.to_json_line().as_bytes()).await?;
                            }
                            line.clear();
                        }
                        Err(e) => {
                            return Err(e.into());
                        }
                    }
                }

                // Handle pool events
                event = event_rx.recv() => {
                    match event {
                        Ok(PoolEvent::NewJob(job)) => {
                            if conn.subscribed && conn.authorized {
                                let notification = StratumNotification::mining_notify(&job, &conn.extranonce1);
                                writer.write_all(notification.to_json_line().as_bytes()).await?;
                            }
                        }
                        Ok(PoolEvent::DifficultyUpdate(diff)) => {
                            if conn.subscribed {
                                let notification = StratumNotification::mining_set_difficulty(diff);
                                writer.write_all(notification.to_json_line().as_bytes()).await?;
                            }
                        }
                        Ok(PoolEvent::CleanJobs) => {
                            // POOL-004: Tell workers to abandon stale work after a block is found.
                            // Send mining.notify with clean_jobs=true so they stop submitting against
                            // the now-invalid job. Workers will wait for the next real notify.
                            if conn.subscribed && conn.authorized {
                                tracing::debug!("[STRATUM] sending clean_jobs=true to worker");
                                let notification = StratumNotification {
                                    id: None,
                                    method: "mining.notify".to_string(),
                                    params: vec![
                                        serde_json::json!("clean"),  // job_id placeholder
                                        serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000"),
                                        serde_json::json!(""),
                                        serde_json::json!(""),
                                        serde_json::json!([]),
                                        serde_json::json!("00000001"),
                                        serde_json::json!("1d00ffff"),
                                        serde_json::json!(format!("{:08x}",
                                            std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .unwrap_or_default()
                                                .as_secs() as u32)),
                                        serde_json::json!(true),  // clean_jobs = true
                                    ],
                                };
                                writer.write_all(notification.to_json_line().as_bytes()).await?;
                            }
                        }
                        Ok(PoolEvent::Shutdown) => {
                            break;
                        }
                        Err(_) => {
                            // Channel closed
                            break;
                        }
                    }
                }
            }
        }

        if let Some(worker) = &conn.worker {
            worker.disconnect();
        }

        Ok(())
    }

    /// Handle incoming message
    async fn handle_message(
        &self,
        line: &str,
        conn: &mut StratumConnection,
        share_tx: &mpsc::Sender<(WorkerId, ShareSubmission)>,
    ) -> PoolResult<Option<StratumResponse>> {
        let msg: StratumMessage = match serde_json::from_str(line.trim()) {
            Ok(m) => m,
            Err(_) => {
                return Ok(Some(StratumResponse::error(None, 20, "Invalid JSON")));
            }
        };

        match msg.method.as_str() {
            "mining.subscribe" => {
                self.handle_subscribe(&msg, conn).await
            }
            "mining.authorize" => {
                self.handle_authorize(&msg, conn).await
            }
            "mining.submit" => {
                self.handle_submit(&msg, conn, share_tx).await
            }
            "mining.extranonce.subscribe" => {
                // Optional extension - acknowledge but don't do anything special
                Ok(Some(StratumResponse::success(msg.id, json!(true))))
            }
            _ => {
                tracing::warn!(method = %msg.method, "Unknown stratum method");
                Ok(Some(StratumResponse::error(msg.id, 20, "Unknown method")))
            }
        }
    }

    /// Handle mining.subscribe
    async fn handle_subscribe(
        &self,
        msg: &StratumMessage,
        conn: &mut StratumConnection,
    ) -> PoolResult<Option<StratumResponse>> {
        // Generate extranonce1
        let mut bytes = [0u8; 4];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
        conn.extranonce1 = hex::encode(bytes);
        conn.extranonce2_size = 4;
        conn.subscribed = true;

        // Response format: [[["mining.set_difficulty", subscription_id], ["mining.notify", subscription_id]], extranonce1, extranonce2_size]
        let result = json!([
            [
                ["mining.set_difficulty", conn.extranonce1.clone()],
                ["mining.notify", conn.extranonce1.clone()]
            ],
            conn.extranonce1,
            conn.extranonce2_size
        ]);

        tracing::debug!(
            addr = %conn.addr,
            extranonce1 = %conn.extranonce1,
            "Worker subscribed"
        );

        Ok(Some(StratumResponse::success(msg.id, result)))
    }

    /// Handle mining.authorize
    async fn handle_authorize(
        &self,
        msg: &StratumMessage,
        conn: &mut StratumConnection,
    ) -> PoolResult<Option<StratumResponse>> {
        if !conn.subscribed {
            return Ok(Some(StratumResponse::error(msg.id, 25, "Not subscribed")));
        }

        // Parse worker name (format: wallet.worker or just wallet)
        let worker_str = msg.params.get(0)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let (wallet, worker_name) = WorkerId::parse(worker_str)?;

        // Register worker
        match self.worker_manager.register(wallet.clone(), worker_name.clone()) {
            Ok(worker) => {
                conn.worker = Some(worker.clone());
                conn.extranonce1 = worker.extranonce1.clone();
                conn.authorized = true;

                tracing::info!(
                    addr = %conn.addr,
                    wallet = %wallet,
                    worker = %worker_name,
                    "Worker authorized"
                );

                Ok(Some(StratumResponse::success(msg.id, json!(true))))
            }
            Err(e) => {
                tracing::warn!(
                    addr = %conn.addr,
                    wallet = %wallet,
                    error = %e,
                    "Worker authorization failed"
                );
                Ok(Some(StratumResponse::error(msg.id, 25, &e.to_string())))
            }
        }
    }

    /// Handle mining.submit
    async fn handle_submit(
        &self,
        msg: &StratumMessage,
        conn: &mut StratumConnection,
        share_tx: &mpsc::Sender<(WorkerId, ShareSubmission)>,
    ) -> PoolResult<Option<StratumResponse>> {
        if !conn.authorized {
            return Ok(Some(StratumResponse::error(msg.id, 25, "Not authorized")));
        }

        let worker = match &conn.worker {
            Some(w) => w,
            None => return Ok(Some(StratumResponse::error(msg.id, 25, "No worker"))),
        };

        // Parse submission
        // params: [worker_name, job_id, extranonce2, ntime, nonce]
        let job_id = msg.params.get(1)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let extranonce2 = msg.params.get(2)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let ntime: u64 = msg.params.get(3)
            .and_then(|v| v.as_str())
            .and_then(|s| u64::from_str_radix(s, 16).ok())
            .unwrap_or(0);

        let nonce: u64 = msg.params.get(4)
            .and_then(|v| v.as_str())
            .and_then(|s| u64::from_str_radix(s, 16).ok())
            .unwrap_or(0);

        let submission = ShareSubmission {
            job_id,
            extranonce2,
            nonce,
            ntime,
            worker_name: worker.worker_name.clone(),
        };

        // POOL-002: Synchronous difficulty floor — reject shares before they enter the queue
        let worker_diff = worker.current_difficulty();
        let min_diff = self.min_difficulty;
        tracing::trace!("[STRATUM] submit pre-validate job={} worker_diff={:.4}", submission.job_id, worker_diff);
        if worker_diff < min_diff {
            tracing::debug!("[STRATUM] submit DIFF_LOW job={} worker_diff={:.4} min_diff={:.4}",
                submission.job_id, worker_diff, min_diff);
            return Ok(Some(StratumResponse::error(msg.id, 23, "Difficulty below pool minimum")));
        }

        // Send to share processor
        let worker_id = worker.id.clone();
        tracing::trace!("[STRATUM] submit queued job={}", submission.job_id);
        if share_tx.send((worker_id, submission)).await.is_err() {
            return Ok(Some(StratumResponse::error(msg.id, 20, "Share processing failed")));
        }

        Ok(Some(StratumResponse::success(msg.id, json!(true))))
    }

    /// Get connection statistics
    pub fn stats(&self) -> StratumStats {
        StratumStats {
            active_connections: *self.connection_count.read(),
            max_connections: self.config.max_connections,
            unique_ips: self.connections_per_ip.read().len(),
        }
    }
}

/// Stratum server statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumStats {
    pub active_connections: usize,
    pub max_connections: usize,
    pub unique_ips: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratum_response_success() {
        let response = StratumResponse::success(Some(1), json!(true));
        let json = response.to_json_line();
        assert!(json.contains("\"result\":true"));
        assert!(json.contains("\"id\":1"));
    }

    #[test]
    fn test_stratum_response_error() {
        let response = StratumResponse::error(Some(1), 21, "Stale share");
        let json = response.to_json_line();
        assert!(json.contains("\"error\""));
        assert!(json.contains("21"));
    }

    #[test]
    fn test_stratum_notification() {
        let notification = StratumNotification::mining_set_difficulty(1.5);
        let json = notification.to_json_line();
        assert!(json.contains("\"method\":\"mining.set_difficulty\""));
        assert!(json.contains("1.5"));
    }
}
