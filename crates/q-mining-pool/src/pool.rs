//! Mining pool orchestrator - coordinates all pool components

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use parking_lot::RwLock;

use crate::config::PoolConfig;
use crate::error::{PoolError, PoolResult};
use crate::job::{BlockTemplate, JobManager, MiningJob};
use crate::payout::PayoutProcessor;
use crate::pplns::PPLNSCalculator;
use crate::share::{Share, ShareSubmission, ShareValidationResult, ShareValidator};
use crate::stratum::{PoolEvent, StratumServer};
use crate::worker::{WorkerId, WorkerManager};

/// Mining pool statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoolStats {
    /// Pool name
    pub name: String,

    /// Total hashrate (H/s)
    pub hashrate: f64,

    /// Connected workers
    pub workers: usize,

    /// Total shares submitted
    pub total_shares: u64,

    /// Valid shares
    pub valid_shares: u64,

    /// Invalid shares
    pub invalid_shares: u64,

    /// Stale shares
    pub stale_shares: u64,

    /// Blocks found
    pub blocks_found: u64,

    /// Total paid out
    pub total_paid: u64,

    /// Pending payouts
    pub pending_payouts: u64,

    /// Pool fee percentage
    pub pool_fee_percent: f64,

    /// Network difficulty
    pub network_difficulty: f64,

    /// Current block height
    pub block_height: u64,

    /// Pool uptime (seconds)
    pub uptime_seconds: u64,
}

/// Block found notification
#[derive(Debug, Clone)]
pub struct BlockFound {
    /// Block hash
    pub hash: [u8; 32],

    /// Block height
    pub height: u64,

    /// Block header
    pub header: Vec<u8>,

    /// Worker who found it
    pub found_by: WorkerId,

    /// Total reward
    pub reward: u64,
}

/// Mining pool - main orchestrator
pub struct MiningPool {
    /// Configuration
    config: PoolConfig,

    /// Stratum server
    stratum_server: Arc<StratumServer>,

    /// Job manager
    job_manager: Arc<JobManager>,

    /// Share validator
    share_validator: Arc<RwLock<ShareValidator>>,

    /// Worker manager
    worker_manager: Arc<WorkerManager>,

    /// PPLNS calculator
    pplns: Arc<PPLNSCalculator>,

    /// Payout processor
    payout_processor: Arc<PayoutProcessor>,

    /// Statistics
    stats: Arc<RwLock<PoolStatsInternal>>,

    /// Pool start time
    start_time: std::time::Instant,

    /// Block found callback
    on_block_found: Option<Arc<dyn Fn(BlockFound) + Send + Sync>>,

    /// Current block reward (updated from emission controller)
    current_block_reward: Arc<RwLock<u64>>,

    /// Payout handler - called to credit miner balances on-chain
    /// Takes Vec<(wallet_address, amount_atomic)>, returns tx_hash
    payout_handler: Arc<RwLock<Option<Arc<dyn Fn(Vec<(String, u64)>) -> String + Send + Sync>>>>,
}

/// Internal statistics tracking
struct PoolStatsInternal {
    total_shares: u64,
    valid_shares: u64,
    invalid_shares: u64,
    stale_shares: u64,
    blocks_found: u64,
    network_difficulty: f64,
    block_height: u64,
}

impl Default for PoolStatsInternal {
    fn default() -> Self {
        Self {
            total_shares: 0,
            valid_shares: 0,
            invalid_shares: 0,
            stale_shares: 0,
            blocks_found: 0,
            network_difficulty: 1.0,
            block_height: 0,
        }
    }
}

impl MiningPool {
    /// Create new mining pool
    pub fn new(config: PoolConfig) -> Self {
        let worker_manager = Arc::new(WorkerManager::with_config(
            config.vardiff.clone(),
            config.security.clone(),
        ));
        let stratum_server = Arc::new(StratumServer::new(
            config.stratum.clone(),
            config.vardiff.min_difficulty,
            Arc::clone(&worker_manager),
        ));
        let job_manager = Arc::new(JobManager::new(config.pool_wallet.clone()));
        let share_validator = Arc::new(RwLock::new(ShareValidator::new(1.0)));
        let pplns = Arc::new(PPLNSCalculator::new(
            config.pplns.clone(),
            config.fees.pool_fee_bps,
        ));
        let payout_processor = Arc::new(PayoutProcessor::new(config.payout.clone()));

        Self {
            config,
            stratum_server,
            job_manager,
            share_validator,
            worker_manager,
            pplns,
            payout_processor,
            stats: Arc::new(RwLock::new(PoolStatsInternal::default())),
            start_time: std::time::Instant::now(),
            on_block_found: None,
            current_block_reward: Arc::new(RwLock::new(290_000)), // ~0.00029 QUG default
            payout_handler: Arc::new(RwLock::new(None)),
        }
    }

    /// Set block found callback
    pub fn on_block_found<F>(&mut self, callback: F)
    where
        F: Fn(BlockFound) + Send + Sync + 'static,
    {
        self.on_block_found = Some(Arc::new(callback));
    }

    /// Start the mining pool
    pub async fn start(self: Arc<Self>) -> PoolResult<()> {
        tracing::info!(
            name = %self.config.name,
            stratum_port = self.config.stratum.port,
            pool_fee = self.config.fees.pool_fee_bps as f64 / 100.0,
            "Starting mining pool"
        );

        // Create share processing channel
        let (share_tx, share_rx) = mpsc::channel(10_000);

        // Start share processor
        let pool_clone = Arc::clone(&self);
        tokio::spawn(async move {
            pool_clone.share_processor(share_rx).await;
        });

        // Start worker cleanup task
        let worker_manager_clone = Arc::clone(&self.worker_manager);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                worker_manager_clone.cleanup_stale_workers();
            }
        });

        // Start payout processor (if auto-payout enabled)
        if self.config.payout.auto_payout {
            let pool_clone = Arc::clone(&self);
            tokio::spawn(async move {
                pool_clone.payout_loop().await;
            });
        }

        // Start stratum server (blocking)
        let stratum = Arc::clone(&self.stratum_server);
        stratum.start(share_tx).await?;

        Ok(())
    }

    /// Process incoming shares
    async fn share_processor(
        &self,
        mut share_rx: mpsc::Receiver<(WorkerId, ShareSubmission)>,
    ) {
        while let Some((worker_id, submission)) = share_rx.recv().await {
            if let Err(e) = self.process_share(&worker_id, &submission).await {
                tracing::warn!(
                    worker = %worker_id,
                    error = %e,
                    "Share processing error"
                );
            }
        }
    }

    /// Process a single share submission
    async fn process_share(
        &self,
        worker_id: &WorkerId,
        submission: &ShareSubmission,
    ) -> PoolResult<()> {
        // Get worker
        let worker = self.worker_manager
            .get_worker(worker_id)
            .ok_or_else(|| PoolError::InvalidWorker(worker_id.to_string()))?;

        // Get job
        let job = self.job_manager
            .get_job(&submission.job_id)
            .ok_or_else(|| PoolError::InvalidJob(submission.job_id.clone()))?;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_shares += 1;
        }

        // Validate share
        let worker_difficulty = worker.current_difficulty();
        let result = self.share_validator.read().validate(
            &job,
            submission,
            worker_difficulty,
        )?;

        match result {
            ShareValidationResult::ValidShare { difficulty } => {
                self.handle_valid_share(worker_id, &job, difficulty).await?;
            }
            ShareValidationResult::BlockFound { hash, header, height } => {
                self.handle_block_found(worker_id, &job, hash, header, height).await?;
            }
            ShareValidationResult::Stale => {
                self.stats.write().stale_shares += 1;
                worker.record_stale();
                tracing::debug!(worker = %worker_id, job = %submission.job_id, "Stale share");
            }
            ShareValidationResult::LowDifficulty { got, need } => {
                self.stats.write().invalid_shares += 1;
                worker.record_invalid();
                tracing::debug!(
                    worker = %worker_id,
                    got = got,
                    need = need,
                    "Low difficulty share"
                );
            }
            ShareValidationResult::Duplicate => {
                self.stats.write().invalid_shares += 1;
                worker.record_invalid();
                tracing::debug!(worker = %worker_id, "Duplicate share");
            }
            ShareValidationResult::InvalidHash | ShareValidationResult::InvalidNonce => {
                self.stats.write().invalid_shares += 1;
                worker.record_invalid();
                tracing::warn!(worker = %worker_id, "Invalid share hash/nonce");
            }
        }

        Ok(())
    }

    /// Handle valid share
    async fn handle_valid_share(
        &self,
        worker_id: &WorkerId,
        job: &MiningJob,
        difficulty: f64,
    ) -> PoolResult<()> {
        // Update stats
        self.stats.write().valid_shares += 1;

        // Record share with worker
        if let Some(worker) = self.worker_manager.get_worker(worker_id) {
            worker.record_share(difficulty);
        }

        // Add to PPLNS window
        let share = Share::new(
            worker_id.clone(),
            job.job_id.clone(),
            difficulty,
            [0; 32], // Hash not needed for PPLNS
            0,       // Nonce not needed
            false,
        );
        self.pplns.add_share(share);

        tracing::trace!(
            worker = %worker_id,
            difficulty = difficulty,
            "Valid share recorded"
        );

        Ok(())
    }

    /// Handle block found
    async fn handle_block_found(
        &self,
        worker_id: &WorkerId,
        job: &MiningJob,
        hash: [u8; 32],
        header: Vec<u8>,
        height: u64,
    ) -> PoolResult<()> {
        tracing::info!(
            worker = %worker_id,
            height = height,
            hash = %hex::encode(&hash[..8]),
            "BLOCK FOUND!"
        );

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.valid_shares += 1;
            stats.blocks_found += 1;
        }

        // Record with worker
        if let Some(worker) = self.worker_manager.get_worker(worker_id) {
            worker.record_block();
        }

        // Calculate rewards using PPLNS
        let block_reward = self.get_block_reward(height);
        let round = self.pplns.calculate_rewards(
            block_reward,
            height,
            hash,
            worker_id.clone(),
            |wid| self.worker_manager.get_wallet(wid).unwrap_or_default(),
        );

        // Credit rewards to pending balances
        self.payout_processor.credit_rewards(&round.payouts, round.round_id);

        // Start new round
        self.pplns.new_round();

        // Invalidate stale jobs and immediately notify workers to abandon their work (POOL-004)
        tracing::debug!("[POOL] block_found invalidating_all_jobs height={}", height);
        self.job_manager.invalidate_all();
        // broadcast_clean_jobs closes the race window: without this, workers keep submitting
        // against the now-invalid job until the next block template arrives
        self.stratum_server.broadcast_clean_jobs();
        tracing::debug!("[POOL] block_found clean_jobs broadcast sent height={}", height);

        // Notify callback
        if let Some(callback) = &self.on_block_found {
            let block_found = BlockFound {
                hash,
                height,
                header,
                found_by: worker_id.clone(),
                reward: block_reward,
            };
            callback(block_found);
        }

        // Auto-trigger payout if configured
        if self.config.payout.auto_payout {
            match self.process_payouts().await {
                Ok(Some(batch)) => {
                    tracing::info!(
                        batch_id = batch.id,
                        recipients = batch.payouts.len(),
                        "Auto-payout triggered after block found"
                    );
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::error!(error = %e, "Auto-payout failed after block found");
                }
            }
        }

        Ok(())
    }

    /// Get block reward for height
    fn get_block_reward(&self, _height: u64) -> u64 {
        // Use the dynamically-set block reward from emission controller
        // Falls back to 290_000 (~0.00029 QUG) which is the current emission rate
        let reward = *self.current_block_reward.read();
        if reward > 0 { reward } else { 290_000 }
    }

    /// Update block reward from emission controller
    pub fn set_block_reward(&self, reward: u64) {
        *self.current_block_reward.write() = reward;
    }

    /// Set payout handler for crediting miner balances on-chain
    pub fn set_payout_handler(&self, handler: Arc<dyn Fn(Vec<(String, u64)>) -> String + Send + Sync>) {
        *self.payout_handler.write() = Some(handler);
    }

    /// Payout processing loop
    async fn payout_loop(&self) {
        let interval = match self.config.payout.interval {
            crate::config::PayoutInterval::Immediate => Duration::from_secs(60),
            crate::config::PayoutInterval::Hourly => Duration::from_secs(3600),
            crate::config::PayoutInterval::Daily => Duration::from_secs(86400),
            crate::config::PayoutInterval::EveryNBlocks(_) => Duration::from_secs(300),
            crate::config::PayoutInterval::Threshold(_threshold) => Duration::from_secs(300),
        };

        loop {
            tokio::time::sleep(interval).await;

            match self.process_payouts().await {
                Ok(Some(batch)) => {
                    tracing::info!(
                        batch_id = batch.id,
                        recipients = batch.payouts.len(),
                        total = batch.total_amount,
                        "Batch payout completed"
                    );
                }
                Ok(None) => {
                    tracing::debug!("No payouts ready");
                }
                Err(e) => {
                    tracing::error!(error = %e, "Payout processing failed");
                }
            }
        }
    }

    /// Process pending payouts
    pub async fn process_payouts(&self) -> PoolResult<Option<crate::payout::BatchPayout>> {
        let handler = self.payout_handler.read().clone();
        let result = self.payout_processor.process_payouts(|outputs| {
            let handler = handler.clone();
            async move {
                match handler {
                    Some(h) => {
                        let tx_hash = h(outputs.clone());
                        tracing::info!(
                            tx_hash = %tx_hash,
                            outputs = outputs.len(),
                            "Pool payout transaction submitted"
                        );
                        Ok(tx_hash)
                    }
                    None => {
                        // Fallback: log and return placeholder
                        let tx_hash = format!("pool_payout_{}", chrono::Utc::now().timestamp());
                        tracing::warn!(
                            tx_hash = %tx_hash,
                            outputs = outputs.len(),
                            "Pool payout handler not set - rewards credited internally only"
                        );
                        Ok(tx_hash)
                    }
                }
            }
        }).await?;

        Ok(result)
    }

    /// Update block template from node
    pub fn update_template(&self, template: BlockTemplate) -> PoolResult<()> {
        // Update network difficulty
        let difficulty = self.target_to_difficulty(&template.target);
        self.share_validator.write().update_network_difficulty(difficulty);
        self.pplns.update_network_difficulty(difficulty);

        {
            let mut stats = self.stats.write();
            stats.network_difficulty = difficulty;
            stats.block_height = template.height;
        }

        // Create new job
        let job = self.job_manager.create_job(template, false)?;

        // Broadcast to all workers
        self.stratum_server.broadcast_job(job);

        Ok(())
    }

    /// Update block template and signal clean jobs (new block found on network)
    pub fn update_template_clean(&self, template: BlockTemplate) -> PoolResult<()> {
        // Update network difficulty
        let difficulty = self.target_to_difficulty(&template.target);
        self.share_validator.write().update_network_difficulty(difficulty);
        self.pplns.update_network_difficulty(difficulty);

        {
            let mut stats = self.stats.write();
            stats.network_difficulty = difficulty;
            stats.block_height = template.height;
        }

        // Invalidate existing jobs
        self.job_manager.invalidate_all();

        // Create new job with clean flag
        let job = self.job_manager.create_job(template, true)?;

        // Broadcast to all workers
        self.stratum_server.broadcast_job(job);

        Ok(())
    }

    /// Convert target to difficulty
    fn target_to_difficulty(&self, target: &[u8; 32]) -> f64 {
        // Count leading zero bytes
        let mut leading_zeros = 0;
        for &byte in target.iter() {
            if byte == 0 {
                leading_zeros += 1;
            } else {
                break;
            }
        }

        let base_difficulty = 256.0_f64.powi(leading_zeros as i32);

        if leading_zeros < 32 {
            let first_nonzero = target[leading_zeros] as f64;
            if first_nonzero > 0.0 {
                base_difficulty * (255.0 / first_nonzero)
            } else {
                base_difficulty
            }
        } else {
            f64::MAX
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let internal = self.stats.read();
        let stratum_stats = self.stratum_server.stats();
        let payout_stats = self.payout_processor.stats();

        PoolStats {
            name: self.config.name.clone(),
            hashrate: self.worker_manager.total_hashrate(),
            workers: stratum_stats.active_connections,
            total_shares: internal.total_shares,
            valid_shares: internal.valid_shares,
            invalid_shares: internal.invalid_shares,
            stale_shares: internal.stale_shares,
            blocks_found: internal.blocks_found,
            total_paid: payout_stats.total_paid,
            pending_payouts: payout_stats.pending_total,
            pool_fee_percent: self.config.fees.pool_fee_bps as f64 / 100.0,
            network_difficulty: internal.network_difficulty,
            block_height: internal.block_height,
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }

    /// Get worker statistics
    pub fn worker_stats(&self) -> Vec<crate::worker::WorkerInfo> {
        self.worker_manager.get_all_workers()
    }

    /// Get PPLNS statistics
    pub fn pplns_stats(&self) -> crate::pplns::PPLNSStats {
        self.pplns.stats()
    }

    /// Get payout statistics
    pub fn payout_stats(&self) -> crate::payout::PayoutStats {
        self.payout_processor.stats()
    }

    /// Get pending balance for wallet
    pub fn get_pending_balance(&self, wallet: &str) -> u64 {
        self.payout_processor.get_pending_balance(wallet)
    }

    /// Get payout history for wallet
    pub fn get_wallet_payouts(&self, wallet: &str) -> Vec<crate::payout::Payout> {
        self.payout_processor.get_wallet_history(wallet)
    }

    /// Get round history
    pub fn get_round_history(&self) -> Vec<crate::pplns::Round> {
        self.pplns.round_history()
    }

    /// Ban worker
    pub fn ban_worker(&self, wallet: &str, reason: &str, duration: Duration) {
        self.worker_manager.ban(wallet, reason, duration);
    }

    /// Unban worker
    pub fn unban_worker(&self, wallet: &str) -> bool {
        self.worker_manager.unban(wallet)
    }

    /// Get pool configuration
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// Get worker manager reference
    pub fn worker_manager(&self) -> &WorkerManager {
        &self.worker_manager
    }

    /// Get payout processor reference
    pub fn payout_processor(&self) -> &PayoutProcessor {
        &self.payout_processor
    }

    /// Get current round ID
    pub fn current_round_id(&self) -> u64 {
        self.pplns.current_round_id()
    }

    /// Get current difficulty (for API)
    pub fn current_difficulty(&self) -> f64 {
        self.stats.read().network_difficulty
    }

    /// Record an HTTP mining share in the PPLNS window (bypasses Stratum).
    /// Called by the batch mining processor when a valid solution is accepted.
    pub fn record_http_share(&self, share: Share) {
        self.pplns.add_share(share);
        let mut stats = self.stats.write();
        stats.valid_shares += 1;
        stats.total_shares += 1;
    }

    /// Notify the pool that a block has been produced by the local block producer.
    ///
    /// Must be called every time the block producer successfully creates a block so that:
    /// - The PPLNS round advances (sliding window is preserved, but round counter increments)
    /// - The blocks_found stat is incremented
    ///
    /// This is the HTTP-path equivalent of `handle_block_found` (which is only reached via
    /// the Stratum submit path). Without this call the pool's round never advances and the
    /// `blocks_found` counter stays at zero even though blocks are being produced.
    pub fn notify_block_produced(&self, block_height: u64, block_hash: [u8; 32]) {
        tracing::info!(
            height = block_height,
            hash = %hex::encode(&block_hash[..8]),
            shares = self.pplns.share_count(),
            "🏊 [POOL] Block produced — advancing PPLNS round",
        );
        // Advance the PPLNS round (rolls the sliding window; shares are kept per PPLNS rules)
        self.pplns.new_round();
        // Increment blocks_found stat
        let mut stats = self.stats.write();
        stats.blocks_found += 1;
        stats.block_height = block_height;
    }

    /// Get PPLNS share proportions for coinbase distribution.
    /// Returns Vec<(wallet_hex, proportion)> or None if no shares in window.
    /// Proportions are fee-free — block producer handles dev fee separately.
    pub fn get_pplns_proportions(&self) -> Option<Vec<(String, f64)>> {
        if self.pplns.share_count() == 0 {
            return None;
        }
        let proportions = self.pplns.get_share_proportions(|wid| wid.wallet().to_string());
        if proportions.is_empty() { None } else { Some(proportions) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn test_config() -> PoolConfig {
        PoolConfig {
            name: "Test Pool".to_string(),
            pool_wallet: "qnk_test_wallet".to_string(),
            stratum: StratumConfig {
                bind_address: "127.0.0.1".to_string(),
                port: 13333,
                max_connections: 100,
            },
            fees: FeeConfig {
                pool_fee_bps: 150,
                dev_fee_enabled: true,
                promotional_period: false,
                promotional_end: None,
            },
            pplns: PPLNSConfig {
                n_factor: 2.0,
                max_shares_in_memory: 1000,
                persist_shares: false,
            },
            vardiff: VardiffConfig {
                enabled: true,
                initial_difficulty: 0.001,
                target_time_seconds: 20.0,
                variance_percent: 0.25,
                min_difficulty: 0.001,
                max_difficulty: 1_000_000.0,
                retarget_interval_seconds: 120.0,
            },
            payout: PayoutConfig {
                min_payout: 10_000_000,
                interval: PayoutInterval::Immediate,
                max_batch_size: 100,
                auto_payout: false,
            },
            security: SecurityConfig {
                rate_limit_shares_per_second: 10.0,
                max_invalid_shares_before_ban: 100,
                ban_duration_seconds: 3600,
                require_worker_names: false,
                max_connections_per_ip: 50,
                detect_block_withholding: false,
                withholding_threshold_sigma: 3.0,
                validate_wallet_format: false,
            },
        }
    }

    #[test]
    fn test_pool_creation() {
        let config = test_config();
        let pool = MiningPool::new(config);

        let stats = pool.stats();
        assert_eq!(stats.name, "Test Pool");
        assert_eq!(stats.workers, 0);
        assert_eq!(stats.blocks_found, 0);
    }

    #[test]
    fn test_target_to_difficulty() {
        let config = test_config();
        let pool = MiningPool::new(config);

        // Easy target (high values = low difficulty)
        let easy_target = [0xff_u8; 32];
        let diff = pool.target_to_difficulty(&easy_target);
        assert!(diff < 2.0);

        // Hard target (leading zeros = high difficulty)
        let mut hard_target = [0xff_u8; 32];
        hard_target[0] = 0x00;
        hard_target[1] = 0x00;
        let diff = pool.target_to_difficulty(&hard_target);
        assert!(diff > 100.0);
    }
}
