//! 🔨 v2.3.0-beta: Integrated Mining Module
//!
//! Single unified node with built-in mining capability.
//! No separate miner binary needed - mining runs directly in the API server.
//!
//! Features:
//! - Direct access to blockchain state (no HTTP overhead)
//! - Automatic reward crediting to wallet
//! - Multi-threaded CPU mining with configurable thread count

use crate::AppState;
use blake3::Hasher;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Mining statistics for monitoring
pub struct MiningStats {
    pub hashes_computed: AtomicU64,
    pub shares_found: AtomicU64,
    pub blocks_found: AtomicU64,
    pub start_time: std::time::Instant,
}

impl MiningStats {
    pub fn new() -> Self {
        Self {
            hashes_computed: AtomicU64::new(0),
            shares_found: AtomicU64::new(0),
            blocks_found: AtomicU64::new(0),
            start_time: std::time::Instant::now(),
        }
    }

    pub fn hashrate(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.hashes_computed.load(Ordering::Relaxed) as f64 / elapsed / 1000.0
        } else {
            0.0
        }
    }
}

/// Mining configuration
pub struct MiningConfig {
    pub wallet_address: String,
    pub thread_count: usize,
    pub enabled: bool,
}

/// Spawn the integrated mining loop
pub fn spawn_integrated_mining(
    app_state: Arc<AppState>,
    config: MiningConfig,
) {
    if !config.enabled {
        return;
    }

    let wallet = config.wallet_address.clone();
    let threads = config.thread_count;

    info!("⛏️  [INTEGRATED MINING] Starting {} mining threads", threads);
    info!("   💰 Wallet: {}", wallet);

    // Spawn the main mining coordinator
    tokio::spawn(async move {
        run_mining_coordinator(app_state, wallet, threads).await;
    });
}

/// Main mining coordinator
async fn run_mining_coordinator(
    app_state: Arc<AppState>,
    wallet: String,
    thread_count: usize,
) {
    let stats = Arc::new(MiningStats::new());
    let running = Arc::new(AtomicBool::new(true));

    // Current challenge state (shared between threads)
    let current_challenge: Arc<RwLock<Option<MiningChallenge>>> = Arc::new(RwLock::new(None));

    // Spawn challenge updater task
    let app_state_updater = app_state.clone();
    let challenge_ref = current_challenge.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
        loop {
            interval.tick().await;

            // Get current block height from local state
            let height = app_state_updater
                .current_height_atomic
                .load(Ordering::SeqCst);

            // Generate challenge from current blockchain state
            let challenge = generate_local_challenge(height);

            // Update shared challenge
            let mut guard = challenge_ref.write().await;
            *guard = Some(challenge);
        }
    });

    // Get the tokio runtime handle to pass to mining threads
    let runtime_handle = tokio::runtime::Handle::current();

    // Spawn mining threads
    for thread_id in 0..thread_count {
        let app_state_thread = app_state.clone();
        let wallet_thread = wallet.clone();
        let stats_thread = stats.clone();
        let running_thread = running.clone();
        let challenge_thread = current_challenge.clone();
        let rt_handle = runtime_handle.clone();

        std::thread::spawn(move || {
            run_mining_thread(
                thread_id,
                app_state_thread,
                wallet_thread,
                stats_thread,
                running_thread,
                challenge_thread,
                rt_handle,
            );
        });
    }

    // Stats reporting loop
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
    loop {
        interval.tick().await;

        let hashrate = stats.hashrate();
        let shares = stats.shares_found.load(Ordering::Relaxed);
        let blocks = stats.blocks_found.load(Ordering::Relaxed);

        info!(
            "⛏️  [MINING STATS] {:.2} KH/s | Shares: {} | Blocks: {}",
            hashrate, shares, blocks
        );
    }
}

/// Local mining challenge (no HTTP needed)
#[derive(Clone)]
struct MiningChallenge {
    challenge_hash: [u8; 32],
    difficulty_target: [u8; 32],
    block_height: u64,
}

/// Generate challenge from local blockchain state
fn generate_local_challenge(height: u64) -> MiningChallenge {
    // Generate challenge hash based on height and time
    let mut hasher = Hasher::new();
    hasher.update(&height.to_le_bytes());
    hasher.update(&chrono::Utc::now().timestamp().to_le_bytes());
    let challenge_hash: [u8; 32] = *hasher.finalize().as_bytes();

    // Use a reasonable difficulty (adjustable)
    // Higher difficulty = lower target = harder to find
    let difficulty = 1.0;
    let difficulty_target = difficulty_to_target(difficulty);

    MiningChallenge {
        challenge_hash,
        difficulty_target,
        block_height: height + 1,
    }
}

/// Convert difficulty to target bytes
fn difficulty_to_target(difficulty: f64) -> [u8; 32] {
    let mut target = [0xff_u8; 32];

    // Simple difficulty adjustment
    let adjusted_difficulty = difficulty.max(1.0);
    let leading_zeros = (adjusted_difficulty.log2() / 8.0).floor() as usize;

    for i in 0..leading_zeros.min(32) {
        target[i] = 0;
    }

    if leading_zeros < 32 {
        let fraction = 255.0 / (2.0_f64.powf(adjusted_difficulty.log2() % 8.0));
        target[leading_zeros] = fraction as u8;
    }

    target
}

/// Single mining thread
fn run_mining_thread(
    thread_id: usize,
    app_state: Arc<AppState>,
    wallet: String,
    stats: Arc<MiningStats>,
    running: Arc<AtomicBool>,
    challenge: Arc<RwLock<Option<MiningChallenge>>>,
    runtime: tokio::runtime::Handle,
) {
    info!("⛏️  [THREAD {}] Mining thread started for wallet: {}", thread_id, &wallet);

    let mut nonce: u64 = thread_id as u64 * 1_000_000_000;
    let mut last_height = 0u64;
    let mut current_challenge_cache: Option<MiningChallenge> = None;

    while running.load(Ordering::Relaxed) {
        // Check for new challenge periodically
        if nonce % 10000 == 0 {
            if let Ok(guard) = challenge.try_read() {
                if let Some(ref c) = *guard {
                    if c.block_height != last_height {
                        last_height = c.block_height;
                        current_challenge_cache = Some(c.clone());
                        debug!(
                            "[THREAD {}] New challenge at height {}",
                            thread_id, last_height
                        );
                    }
                }
            }
        }

        // Skip if no challenge yet
        let Some(ref challenge) = current_challenge_cache else {
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        };

        // Compute hash
        let mut hasher = Hasher::new();
        hasher.update(&challenge.challenge_hash);
        hasher.update(wallet.as_bytes());
        hasher.update(&nonce.to_le_bytes());
        let hash: [u8; 32] = *hasher.finalize().as_bytes();

        stats.hashes_computed.fetch_add(1, Ordering::Relaxed);

        // Check if hash meets difficulty target
        if hash < challenge.difficulty_target {
            info!(
                "⛏️  [THREAD {}] 🎉 SHARE FOUND! Height: {}, Nonce: {}",
                thread_id, challenge.block_height, nonce
            );
            stats.shares_found.fetch_add(1, Ordering::Relaxed);

            // Submit solution - credit rewards directly
            let app_state_submit = app_state.clone();
            let wallet_submit = wallet.clone();

            runtime.spawn(async move {
                credit_mining_reward(app_state_submit, wallet_submit).await;
            });
        }

        nonce = nonce.wrapping_add(1);
    }

    info!("⛏️  [THREAD {}] Mining thread stopped", thread_id);
}

/// Credit mining reward directly to wallet balance
///
/// Uses the same mechanism as submit_mining_solution handler:
/// 1. Update in-memory wallet_balances
/// 2. Persist to RocksDB via storage_engine
async fn credit_mining_reward(app_state: Arc<AppState>, wallet: String) {
    // Genesis timestamp for reward calculation (network birth: Dec 12, 2024)
    const GENESIS_TIMESTAMP: u64 = 1733980800;

    // Calculate block reward based on emission schedule
    let current_timestamp = chrono::Utc::now().timestamp() as u64;
    let block_reward = calculate_block_reward(GENESIS_TIMESTAMP, current_timestamp);

    // 1% dev fee
    const DEV_FEE_BPS: u128 = 100;
    const BPS_DIVISOR: u128 = 10_000;
    let dev_fee = block_reward.saturating_mul(DEV_FEE_BPS) / BPS_DIVISOR;
    let miner_reward = block_reward.saturating_sub(dev_fee);

    // Convert wallet address to bytes for balance lookup
    let wallet_bytes = if wallet.starts_with("qnk") && wallet.len() == 67 {
        let hex_part = &wallet[3..];
        match hex::decode(hex_part) {
            Ok(bytes) if bytes.len() == 32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                Some(arr)
            }
            _ => None,
        }
    } else {
        None
    };

    let Some(miner_address) = wallet_bytes else {
        error!("❌ Invalid wallet address format: {}", wallet);
        return;
    };

    // Update in-memory balance
    let new_balance = {
        let mut balances = app_state.wallet_balances.write().await;
        let current = balances.get(&miner_address).copied().unwrap_or(0);
        let new = current.saturating_add(miner_reward);
        balances.insert(miner_address, new);
        new
    };

    // Persist to RocksDB
    if let Err(e) = app_state
        .storage_engine
        .save_wallet_balance(&miner_address, new_balance)
        .await
    {
        warn!("⚠️ Failed to persist mining reward: {}", e);
    }

    info!(
        "💰 [INTEGRATED MINING] Credited {:.8} QUG to {} (total: {:.8} QUG)",
        miner_reward as f64 / 1e24,
        &wallet[..16],
        new_balance as f64 / 1e24
    );
}

/// Calculate block reward based on emission schedule
/// Halving every ~4 years (same as Bitcoin)
fn calculate_block_reward(genesis_timestamp: u64, current_timestamp: u64) -> u128 {
    // v3.0.6-beta: Updated for 24 decimals (1 QUG = 10^24 base units)
    const ONE_QUG: u128 = 1_000_000_000_000_000_000_000_000;
    const INITIAL_REWARD: u128 = 50 * ONE_QUG; // 50 QUG in base units
    const HALVING_INTERVAL_SECS: u64 = 4 * 365 * 24 * 60 * 60; // ~4 years

    let elapsed = current_timestamp.saturating_sub(genesis_timestamp);
    let halvings = (elapsed / HALVING_INTERVAL_SECS) as u32;

    if halvings >= 64 {
        0 // After 64 halvings, reward is essentially zero
    } else {
        INITIAL_REWARD >> halvings
    }
}
