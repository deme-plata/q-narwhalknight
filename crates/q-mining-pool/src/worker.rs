//! Worker management for mining pool

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::error::{PoolResult, WorkerError};
use crate::vardiff::VardiffController;

/// Unique worker identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerId(pub String);

impl WorkerId {
    /// Create worker ID from wallet address and worker name
    pub fn new(wallet: &str, worker_name: &str) -> Self {
        WorkerId(format!("{}.{}", wallet, worker_name))
    }

    /// Parse worker ID from string (format: wallet.worker_name)
    pub fn parse(s: &str) -> PoolResult<(String, String)> {
        let parts: Vec<&str> = s.splitn(2, '.').collect();
        if parts.len() == 2 {
            Ok((parts[0].to_string(), parts[1].to_string()))
        } else if parts.len() == 1 {
            // Default worker name if not provided
            Ok((parts[0].to_string(), "default".to_string()))
        } else {
            Err(WorkerError::InvalidName(s.to_string()).into())
        }
    }

    /// Get wallet address from worker ID
    pub fn wallet(&self) -> &str {
        self.0.split('.').next().unwrap_or(&self.0)
    }

    /// Get worker name from worker ID
    pub fn worker_name(&self) -> &str {
        self.0.split('.').nth(1).unwrap_or("default")
    }
}

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Worker connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerState {
    /// Connected and active
    Active,
    /// Connected but idle (no shares recently)
    Idle,
    /// Disconnected
    Disconnected,
    /// Banned
    Banned,
}

/// Mining worker
#[derive(Debug)]
pub struct Worker {
    /// Worker identifier
    pub id: WorkerId,

    /// Wallet address for payouts
    pub wallet_address: String,

    /// Worker name (e.g., "rig01")
    pub worker_name: String,

    /// Current state
    pub state: RwLock<WorkerState>,

    /// Current difficulty
    pub difficulty: RwLock<f64>,

    /// Vardiff controller
    pub vardiff: RwLock<VardiffController>,

    /// Extranonce1 (unique per worker)
    pub extranonce1: String,

    /// Connection time
    pub connected_at: DateTime<Utc>,

    /// Last activity time
    pub last_activity: RwLock<Instant>,

    /// Statistics
    pub stats: WorkerStats,

    /// Session ID for this connection
    pub session_id: String,
}

impl Worker {
    /// Create new worker
    pub fn new(
        wallet_address: String,
        worker_name: String,
        initial_difficulty: f64,
        vardiff_config: crate::config::VardiffConfig,
    ) -> Self {
        let id = WorkerId::new(&wallet_address, &worker_name);
        let extranonce1 = Self::generate_extranonce1();
        let session_id = Self::generate_session_id();

        Self {
            id,
            wallet_address,
            worker_name,
            state: RwLock::new(WorkerState::Active),
            difficulty: RwLock::new(initial_difficulty),
            vardiff: RwLock::new(VardiffController::new(vardiff_config)),
            extranonce1,
            connected_at: Utc::now(),
            last_activity: RwLock::new(Instant::now()),
            stats: WorkerStats::new(),
            session_id,
        }
    }

    /// Generate unique extranonce1
    fn generate_extranonce1() -> String {
        let mut bytes = [0u8; 4];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
        hex::encode(bytes)
    }

    /// Generate session ID
    fn generate_session_id() -> String {
        let mut bytes = [0u8; 16];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut bytes);
        hex::encode(bytes)
    }

    /// Get current difficulty
    pub fn current_difficulty(&self) -> f64 {
        *self.difficulty.read()
    }

    /// Record share (convenience alias for pool.rs)
    pub fn record_share(&self, difficulty: f64) {
        self.stats.total_shares.fetch_add(1, Ordering::Relaxed);
        self.stats.accepted_shares.fetch_add(1, Ordering::Relaxed);
        self.stats.add_difficulty(difficulty);
        *self.last_activity.write() = Instant::now();

        // Update vardiff
        self.vardiff.write().on_share_accepted();
    }

    /// Record stale share
    pub fn record_stale(&self) {
        self.stats.total_shares.fetch_add(1, Ordering::Relaxed);
        self.stats.stale_shares.fetch_add(1, Ordering::Relaxed);
        *self.last_activity.write() = Instant::now();
    }

    /// Record invalid share
    pub fn record_invalid(&self) {
        self.stats.total_shares.fetch_add(1, Ordering::Relaxed);
        self.stats.rejected_shares.fetch_add(1, Ordering::Relaxed);
        *self.last_activity.write() = Instant::now();
    }

    /// Record block found
    pub fn record_block(&self) {
        self.stats.blocks_found.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            worker = %self.id,
            blocks = self.stats.blocks_found.load(Ordering::Relaxed),
            "Block found!"
        );
    }

    /// Record share accepted (alias)
    pub fn on_share_accepted(&self, difficulty: f64) {
        self.record_share(difficulty);
    }

    /// Record share rejected
    pub fn on_share_rejected(&self, reason: &str) {
        self.stats.total_shares.fetch_add(1, Ordering::Relaxed);
        self.stats.rejected_shares.fetch_add(1, Ordering::Relaxed);
        *self.last_activity.write() = Instant::now();

        tracing::debug!(
            worker = %self.id,
            reason = reason,
            "Share rejected"
        );
    }

    /// Record block found (alias)
    pub fn on_block_found(&self) {
        self.record_block();
    }

    /// Get current hashrate estimate (H/s)
    pub fn hashrate(&self) -> f64 {
        let vardiff = self.vardiff.read();
        let difficulty = *self.difficulty.read();
        let shares_per_second = 1.0 / vardiff.average_share_time();

        // hashrate = difficulty * 2^32 / target_time
        // Simplified: hashrate ≈ difficulty * shares_per_second * base_difficulty_hashrate
        difficulty * shares_per_second * 4_294_967_296.0 // 2^32
    }

    /// Check if worker should update difficulty
    pub fn should_update_difficulty(&self) -> bool {
        self.vardiff.read().should_retarget()
    }

    /// Update worker difficulty
    pub fn update_difficulty(&self) -> Option<f64> {
        let mut vardiff = self.vardiff.write();
        if let Some(new_diff) = vardiff.calculate_new_difficulty() {
            *self.difficulty.write() = new_diff;
            Some(new_diff)
        } else {
            None
        }
    }

    /// Get time since last activity
    pub fn idle_time(&self) -> Duration {
        self.last_activity.read().elapsed()
    }

    /// Check if worker is idle (no activity for 5 minutes)
    pub fn is_idle(&self) -> bool {
        self.idle_time() > Duration::from_secs(300)
    }

    /// Check if worker is stale (no activity for 30 minutes)
    pub fn is_stale(&self) -> bool {
        self.idle_time() > Duration::from_secs(1800)
    }

    /// Disconnect worker
    pub fn disconnect(&self) {
        *self.state.write() = WorkerState::Disconnected;
    }

    /// Ban worker
    pub fn ban(&self) {
        *self.state.write() = WorkerState::Banned;
        tracing::warn!(worker = %self.id, "Worker banned");
    }

    /// Get worker info for API
    pub fn info(&self) -> WorkerInfo {
        WorkerInfo {
            id: self.id.clone(),
            wallet_address: self.wallet_address.clone(),
            worker_name: self.worker_name.clone(),
            state: *self.state.read(),
            difficulty: *self.difficulty.read(),
            hashrate: self.hashrate(),
            connected_at: self.connected_at,
            last_activity: Utc::now() - chrono::Duration::from_std(self.idle_time()).unwrap_or_default(),
            stats: self.stats.snapshot(),
        }
    }
}

/// Worker statistics
#[derive(Debug)]
pub struct WorkerStats {
    /// Total shares submitted
    pub total_shares: AtomicU64,

    /// Accepted shares
    pub accepted_shares: AtomicU64,

    /// Rejected shares
    pub rejected_shares: AtomicU64,

    /// Stale shares
    pub stale_shares: AtomicU64,

    /// Blocks found
    pub blocks_found: AtomicU64,

    /// Total difficulty submitted
    pub total_difficulty: RwLock<f64>,

    /// Last hour difficulty (for hashrate calculation)
    pub last_hour_difficulty: RwLock<f64>,

    /// Stats reset time
    pub stats_since: DateTime<Utc>,
}

impl WorkerStats {
    /// Create new stats
    pub fn new() -> Self {
        Self {
            total_shares: AtomicU64::new(0),
            accepted_shares: AtomicU64::new(0),
            rejected_shares: AtomicU64::new(0),
            stale_shares: AtomicU64::new(0),
            blocks_found: AtomicU64::new(0),
            total_difficulty: RwLock::new(0.0),
            last_hour_difficulty: RwLock::new(0.0),
            stats_since: Utc::now(),
        }
    }

    /// Add difficulty to stats
    pub fn add_difficulty(&self, diff: f64) {
        *self.total_difficulty.write() += diff;
        *self.last_hour_difficulty.write() += diff;
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        let total = self.total_shares.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0;
        }
        let accepted = self.accepted_shares.load(Ordering::Relaxed);
        accepted as f64 / total as f64
    }

    /// Get stats snapshot
    pub fn snapshot(&self) -> WorkerStatsSnapshot {
        WorkerStatsSnapshot {
            total_shares: self.total_shares.load(Ordering::Relaxed),
            accepted_shares: self.accepted_shares.load(Ordering::Relaxed),
            rejected_shares: self.rejected_shares.load(Ordering::Relaxed),
            stale_shares: self.stale_shares.load(Ordering::Relaxed),
            blocks_found: self.blocks_found.load(Ordering::Relaxed),
            total_difficulty: *self.total_difficulty.read(),
            acceptance_rate: self.acceptance_rate(),
            stats_since: self.stats_since,
        }
    }
}

impl Default for WorkerStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Worker stats snapshot (serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatsSnapshot {
    pub total_shares: u64,
    pub accepted_shares: u64,
    pub rejected_shares: u64,
    pub stale_shares: u64,
    pub blocks_found: u64,
    pub total_difficulty: f64,
    pub acceptance_rate: f64,
    pub stats_since: DateTime<Utc>,
}

/// Worker info for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: WorkerId,
    pub wallet_address: String,
    pub worker_name: String,
    pub state: WorkerState,
    pub difficulty: f64,
    pub hashrate: f64,
    pub connected_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub stats: WorkerStatsSnapshot,
}

/// Worker manager
pub struct WorkerManager {
    /// Active workers by ID
    workers: DashMap<WorkerId, Arc<Worker>>,

    /// Workers by wallet (for aggregated stats)
    workers_by_wallet: DashMap<String, Vec<WorkerId>>,

    /// Banned IPs
    banned_ips: DashMap<String, Instant>,

    /// Vardiff configuration
    vardiff_config: crate::config::VardiffConfig,

    /// Security configuration
    security_config: crate::config::SecurityConfig,
}

impl WorkerManager {
    /// Create new worker manager with VardiffConfig (for stratum.rs)
    pub fn new(vardiff_config: crate::config::VardiffConfig) -> Self {
        Self {
            workers: DashMap::new(),
            workers_by_wallet: DashMap::new(),
            banned_ips: DashMap::new(),
            vardiff_config,
            security_config: crate::config::SecurityConfig::default(),
        }
    }

    /// Create new worker manager with full config
    pub fn with_config(
        vardiff_config: crate::config::VardiffConfig,
        security_config: crate::config::SecurityConfig,
    ) -> Self {
        Self {
            workers: DashMap::new(),
            workers_by_wallet: DashMap::new(),
            banned_ips: DashMap::new(),
            vardiff_config,
            security_config,
        }
    }

    /// Register new worker
    pub fn register(&self, wallet: String, worker_name: String) -> PoolResult<Arc<Worker>> {
        // Validate wallet format
        if self.security_config.validate_wallet_format && !self.is_valid_wallet(&wallet) {
            return Err(WorkerError::InvalidWallet(wallet).into());
        }

        let worker_id = WorkerId::new(&wallet, &worker_name);

        // Check if already exists
        if let Some(existing) = self.workers.get(&worker_id) {
            return Ok(Arc::clone(&existing));
        }

        // Create new worker
        let worker = Arc::new(Worker::new(
            wallet.clone(),
            worker_name,
            self.vardiff_config.initial_difficulty,
            self.vardiff_config.clone(),
        ));

        self.workers.insert(worker_id.clone(), Arc::clone(&worker));

        // Track by wallet
        self.workers_by_wallet
            .entry(wallet)
            .or_insert_with(Vec::new)
            .push(worker_id);

        tracing::info!(
            worker = %worker.id,
            difficulty = worker.difficulty.read().clone(),
            "Worker registered"
        );

        Ok(worker)
    }

    /// Get worker by ID
    pub fn get(&self, id: &WorkerId) -> Option<Arc<Worker>> {
        self.workers.get(id).map(|w| Arc::clone(&w))
    }

    /// Get worker by ID (alias for pool.rs compatibility)
    pub fn get_worker(&self, id: &WorkerId) -> Option<Arc<Worker>> {
        self.get(id)
    }

    /// Get wallet address for worker
    pub fn get_wallet(&self, id: &WorkerId) -> Option<String> {
        self.workers.get(id).map(|w| w.wallet_address.clone())
    }

    /// Get all workers for a wallet
    pub fn get_by_wallet(&self, wallet: &str) -> Vec<Arc<Worker>> {
        self.workers_by_wallet
            .get(wallet)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.workers.get(id).map(|w| Arc::clone(&w)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Disconnect worker
    pub fn disconnect(&self, id: &WorkerId) {
        if let Some(worker) = self.workers.get(id) {
            worker.disconnect();
        }
    }

    /// Ban worker
    pub fn ban_worker(&self, id: &WorkerId) {
        if let Some(worker) = self.workers.get(id) {
            worker.ban();
        }
    }

    /// Ban IP address
    pub fn ban_ip(&self, ip: String) {
        self.banned_ips.insert(ip.clone(), Instant::now());
        tracing::warn!(ip = %ip, "IP banned");
    }

    /// Check if IP is banned
    pub fn is_ip_banned(&self, ip: &str) -> bool {
        if let Some(banned_at) = self.banned_ips.get(ip) {
            if banned_at.elapsed() < self.security_config.ban_duration() {
                return true;
            }
            // Ban expired, remove
            self.banned_ips.remove(ip);
        }
        false
    }

    /// Validate wallet address format
    fn is_valid_wallet(&self, wallet: &str) -> bool {
        // QNK wallet format: qnk + 64 hex chars = 67 chars total
        wallet.len() == 67 && wallet.starts_with("qnk") && wallet[3..].chars().all(|c| c.is_ascii_hexdigit())
    }

    /// Get active worker count
    pub fn active_count(&self) -> usize {
        self.workers
            .iter()
            .filter(|w| *w.state.read() == WorkerState::Active)
            .count()
    }

    /// Get total hashrate
    pub fn total_hashrate(&self) -> f64 {
        self.workers
            .iter()
            .filter(|w| *w.state.read() == WorkerState::Active)
            .map(|w| w.hashrate())
            .sum()
    }

    /// Get all worker infos
    pub fn all_workers(&self) -> Vec<WorkerInfo> {
        self.workers.iter().map(|w| w.info()).collect()
    }

    /// Get all worker infos (alias for pool.rs)
    pub fn get_all_workers(&self) -> Vec<WorkerInfo> {
        self.all_workers()
    }

    /// Cleanup stale workers
    pub fn cleanup_stale(&self) {
        let stale_ids: Vec<_> = self.workers
            .iter()
            .filter(|w| w.is_stale())
            .map(|w| w.id.clone())
            .collect();

        for id in stale_ids {
            self.workers.remove(&id);
            tracing::debug!(worker = %id, "Removed stale worker");
        }
    }

    /// Cleanup stale workers (alias for pool.rs)
    pub fn cleanup_stale_workers(&self) {
        self.cleanup_stale()
    }

    /// Ban a wallet (wrapper for ban_worker)
    pub fn ban(&self, wallet: &str, reason: &str, duration: Duration) {
        tracing::warn!(wallet = %wallet, reason = %reason, "Wallet banned");
        // Ban all workers for this wallet
        if let Some(worker_ids) = self.workers_by_wallet.get(wallet) {
            for id in worker_ids.iter() {
                if let Some(worker) = self.workers.get(id) {
                    worker.ban();
                }
            }
        }
    }

    /// Unban a wallet
    pub fn unban(&self, wallet: &str) -> bool {
        // Unban is handled by removing banned state
        if let Some(worker_ids) = self.workers_by_wallet.get(wallet) {
            for id in worker_ids.iter() {
                if let Some(worker) = self.workers.get(id) {
                    *worker.state.write() = WorkerState::Active;
                }
            }
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_id_parse() {
        let (wallet, name) = WorkerId::parse("qnk123abc.rig01").unwrap();
        assert_eq!(wallet, "qnk123abc");
        assert_eq!(name, "rig01");

        let (wallet, name) = WorkerId::parse("qnk123abc").unwrap();
        assert_eq!(wallet, "qnk123abc");
        assert_eq!(name, "default");
    }

    #[test]
    fn test_worker_creation() {
        let vardiff_config = crate::config::VardiffConfig::default();
        let worker = Worker::new(
            "qnk1234567890".to_string(),
            "rig01".to_string(),
            0.001,
            vardiff_config,
        );

        assert_eq!(worker.wallet_address, "qnk1234567890");
        assert_eq!(worker.worker_name, "rig01");
        assert_eq!(*worker.state.read(), WorkerState::Active);
    }
}
