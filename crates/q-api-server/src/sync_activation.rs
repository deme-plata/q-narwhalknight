use std::sync::Arc;
/// Sync Activation Module - v1.0.15-beta
///
/// Implements timeout-based sync activation to break the "stuck at genesis" deadlock.
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone)]
pub struct SyncActivationConfig {
    pub cold_start_timeout: Duration,
    pub retry_interval: Duration,
    pub min_peers: usize,
    pub aggressive_mode: bool,
    /// Configurable threshold for determining if node is stagnant (default: 100 blocks behind network)
    /// ✅ KIMI AI FIX: Removed hardcoded 13000 threshold - now dynamically checks if behind network
    pub stagnant_threshold_blocks: u64,
}

impl Default for SyncActivationConfig {
    fn default() -> Self {
        Self {
            cold_start_timeout: Duration::from_secs(30),
            // v1.4.13-beta: CRITICAL FIX - Reduced from 60s to 2s
            // BUG FOUND: 60s retry_interval caused sync to stall after initial burst.
            // Docker test showed node stuck at 7999 blocks because it waited 60s between syncs.
            // Fix: Continuous sync with 2s interval for maximum throughput.
            retry_interval: Duration::from_secs(2),
            min_peers: 1,
            aggressive_mode: false,
            stagnant_threshold_blocks: 100, // Node is stagnant if >100 blocks behind network
        }
    }
}

pub struct TimeoutBasedSyncActivation {
    startup_time: Instant,
    last_sync_attempt: Arc<RwLock<Option<Instant>>>,
    config: SyncActivationConfig,
}

impl TimeoutBasedSyncActivation {
    pub fn new(config: SyncActivationConfig) -> Self {
        info!("🚀 [SYNC ACTIVATION] Initializing timeout-based sync activation");
        info!("   Cold start timeout: {:?}", config.cold_start_timeout);
        info!("   Retry interval: {:?}", config.retry_interval);
        Self {
            startup_time: Instant::now(),
            last_sync_attempt: Arc::new(RwLock::new(None)),
            config,
        }
    }

    pub async fn should_force_sync(
        &self,
        current_height: u64,
        peer_count: usize,
        network_height: u64,
    ) -> bool {
        // v1.0.65-beta: Removed diagnostic spam - only log occasionally
        // This function is called frequently, so avoid logging on every call

        let now = Instant::now();
        let since_startup = now.duration_since(self.startup_time);

        let clearly_behind = network_height > current_height + 5;
        let gap = network_height.saturating_sub(current_height);

        // ✅ KIMI AI FIX: Remove hardcoded 13000 threshold
        // Dynamic stagnation check: node is stagnant if significantly behind network OR at very low height
        let is_stagnant = if network_height > 0 {
            // If we know network height, check if we're behind by threshold
            current_height + self.config.stagnant_threshold_blocks < network_height
        } else {
            // Bootstrap case: consider stagnant if below threshold (for cold start)
            current_height < self.config.stagnant_threshold_blocks
        };

        let cold_start_expired = is_stagnant && since_startup > self.config.cold_start_timeout;

        let since_last_attempt = {
            let guard = self.last_sync_attempt.read().await;
            guard.map(|t| now.duration_since(t))
        };
        let retry_due = match since_last_attempt {
            None => true,
            Some(delta) => delta > self.config.retry_interval,
        };

        // 🔧 v1.0.18-beta: CRITICAL FIX - Inverted logic bug
        // If node is clearly behind (network_height > current_height + 5),
        // we should IMMEDIATELY activate sync, not return false!
        // Previous bug: returned false when clearly_behind=true, preventing all sync!
        if clearly_behind {
            debug!(
                "🚀 [SYNC ACTIVATION] Node clearly behind: current={}, network={}, gap={}",
                current_height,
                network_height,
                network_height.saturating_sub(current_height)
            );
            return true;  // ✅ FIX: Return TRUE to activate sync when behind!
        }

        let have_enough_peers = peer_count >= self.config.min_peers;
        let should_force =
            cold_start_expired && retry_due && (have_enough_peers || self.config.aggressive_mode);

        if should_force {
            warn!(
                "⏰ [SYNC ACTIVATION] Forcing sync from height={}",
                current_height
            );
        }

        should_force
    }

    pub async fn record_sync_attempt(&self) {
        let mut guard = self.last_sync_attempt.write().await;
        *guard = Some(Instant::now());
    }
}
