//! Nova-fold tip watcher — integration point for Phase 2 recursive folding.
//!
//! ## Purpose
//!
//! This is a **scaffold**. The actual Nova recursive SNARK folding wrapper
//! (Phase 2 of the whitepaper) does not exist yet — see
//! `docs/deepseek-job-board-nova-phase2-2026-05-14.md`, jobs N1..N8.
//!
//! The `TipWatcher` defined here owns the *integration point* so that when
//! Phase 2 lands, wiring the real fold driver in is a one-line change to
//! [`TipWatcher::fold_block`] (see the `PHASE2-WIRE-POINT` marker below).
//!
//! ## Behavior today (Phase 1, phase2_enabled = false)
//!
//! - Subscribes to `Receiver<u64>` of new-block heights.
//! - Increments `blocks_observed` on every event.
//! - Logs at TRACE level "would-fold block N" once every 100 blocks (no spam).
//! - `fold_block(_)` returns `Err("Phase 2 not yet implemented ...")`.
//!
//! ## Behavior after Phase 2 lands (phase2_enabled = true)
//!
//! - On each new block, calls `fold_block(height)`.
//! - On Ok, increments `folds_succeeded` and updates `last_folded_height`.
//! - On Err, logs WARN with the error string; only `folds_attempted` is bumped.
//!
//! Flipping `phase2_enabled` should be driven by the upgrade gate when
//! `Upgrade::NovaPhase2` activates (see job N3 on the board). Until then it
//! stays `false`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use tokio::sync::{mpsc, watch};
use tracing::{trace, warn};

/// How often to emit a "would-fold" trace line while Phase 2 is disabled.
///
/// We only log once per N blocks to keep logs quiet — the watcher is meant
/// to be invisible until Phase 2 actually activates.
const WOULD_FOLD_LOG_EVERY: u64 = 100;

/// Error string returned by the Phase 2 stub. Kept as a `&'static str` so
/// it round-trips through `Result<(), &'static str>` without allocation.
pub const PHASE2_NOT_IMPL_ERR: &str =
    "Phase 2 not yet implemented — see docs/deepseek-job-board-nova-phase2-2026-05-14.md";

/// Observable state for the tip watcher.
///
/// Wrapped in `Arc` and exposed via [`TipWatcher::state`] so external
/// observers (metrics, tests, the upgrade-gate flipper) can read counters
/// and toggle `phase2_enabled` without owning the watcher itself.
#[derive(Debug, Default)]
pub struct TipWatcherState {
    /// Total new-block events received from the channel.
    pub blocks_observed: AtomicU64,
    /// Number of times `fold_block` was called (success OR failure).
    pub folds_attempted: AtomicU64,
    /// Number of successful folds (`fold_block` returned Ok).
    pub folds_succeeded: AtomicU64,
    /// Height of the most recent successful fold, or 0 if none yet.
    pub last_folded_height: AtomicU64,
    /// Whether Phase 2 folding is active. Stays `false` until N1..N3 ship
    /// and the upgrade-gate flips it.
    pub phase2_enabled: AtomicBool,
}

/// Tip watcher — subscribes to new-block heights and dispatches them to
/// the Nova fold driver (when Phase 2 is active).
///
/// Today this is a counter + logger. The shape of the API is final so that
/// when Phase 2 lands, the only change is the body of [`Self::fold_block`].
pub struct TipWatcher {
    state: Arc<TipWatcherState>,
}

impl Default for TipWatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl TipWatcher {
    /// Create a new watcher with all counters at zero and `phase2_enabled = false`.
    pub fn new() -> Self {
        Self {
            state: Arc::new(TipWatcherState::default()),
        }
    }

    /// Clone of the shared state handle. Useful for metrics endpoints and
    /// tests that need to inspect counters while `run` is executing.
    pub fn state(&self) -> Arc<TipWatcherState> {
        Arc::clone(&self.state)
    }

    /// Phase 2 dispatch stub.
    ///
    /// ## PHASE2-WIRE-POINT
    ///
    /// When jobs N1..N3 land, the entire body below — from the first line
    /// of the function down to the closing brace — gets replaced with:
    ///
    /// ```ignore
    /// self.nova_folder.fold_block(block_height).map_err(|_| "fold failed")
    /// ```
    ///
    /// (Plus a `nova_folder: Arc<NovaFolder>` field on `TipWatcher` and an
    /// updated constructor.) That is the entire diff for activation.
    ///
    /// Returning a clear "not yet" error — instead of `Ok(())` pretending —
    /// is intentional: it surfaces in WARN logs the moment someone flips
    /// `phase2_enabled` prematurely, which is what we want.
    pub fn fold_block(&self, _block_height: u64) -> Result<(), &'static str> {
        Err(PHASE2_NOT_IMPL_ERR)
    }

    /// Run the watcher loop. Exits when the `shutdown` watch flips to
    /// `true` or when `incoming_blocks` is closed.
    ///
    /// This consumes `self` because the watcher's whole job is to live for
    /// the duration of the task; if you need to read counters from outside,
    /// call [`Self::state`] first and keep that `Arc`.
    pub async fn run(
        self,
        mut incoming_blocks: mpsc::Receiver<u64>,
        mut shutdown: watch::Receiver<bool>,
    ) {
        // Snapshot at start so it doesn't move into the loop.
        let state = self.state.clone();

        loop {
            tokio::select! {
                biased;

                // Shutdown takes priority — exit promptly on signal.
                changed = shutdown.changed() => {
                    // If the sender was dropped, treat that as shutdown too.
                    if changed.is_err() || *shutdown.borrow() {
                        break;
                    }
                }

                maybe_height = incoming_blocks.recv() => {
                    let Some(height) = maybe_height else {
                        // Channel closed — sender side gone, exit cleanly.
                        break;
                    };

                    let observed = state.blocks_observed.fetch_add(1, Ordering::Relaxed) + 1;

                    if state.phase2_enabled.load(Ordering::Relaxed) {
                        // Phase 2 path. We always count the attempt, then
                        // dispatch and either record success or warn.
                        state.folds_attempted.fetch_add(1, Ordering::Relaxed);
                        match self.fold_block(height) {
                            Ok(()) => {
                                state.folds_succeeded.fetch_add(1, Ordering::Relaxed);
                                state.last_folded_height.store(height, Ordering::Relaxed);
                            }
                            Err(e) => {
                                warn!(
                                    block_height = height,
                                    error = %e,
                                    "Nova fold_block returned error"
                                );
                            }
                        }
                    } else {
                        // Phase 1 path — quiet trace, throttled.
                        if observed.is_multiple_of(WOULD_FOLD_LOG_EVERY) {
                            trace!(
                                block_height = height,
                                blocks_observed = observed,
                                "would-fold block (Phase 2 not yet active)"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Build a watcher, channels, and spawn the run loop. Returns the
    /// state handle, the block sender, the shutdown sender, and the join
    /// handle so each test can drive and then await termination.
    fn spawn_watcher() -> (
        Arc<TipWatcherState>,
        mpsc::Sender<u64>,
        watch::Sender<bool>,
        tokio::task::JoinHandle<()>,
    ) {
        let watcher = TipWatcher::new();
        let state = watcher.state();
        let (block_tx, block_rx) = mpsc::channel(64);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let handle = tokio::spawn(watcher.run(block_rx, shutdown_rx));
        (state, block_tx, shutdown_tx, handle)
    }

    /// Spin until `cond` returns true or `timeout` elapses. Used so tests
    /// don't depend on a fixed sleep to observe atomic increments.
    async fn wait_until<F: Fn() -> bool>(cond: F, timeout: Duration) -> bool {
        let start = tokio::time::Instant::now();
        while start.elapsed() < timeout {
            if cond() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        cond()
    }

    #[tokio::test]
    async fn test_phase2_disabled_no_op() {
        let (state, block_tx, shutdown_tx, handle) = spawn_watcher();

        // phase2_enabled defaults to false — confirm before driving traffic.
        assert!(!state.phase2_enabled.load(Ordering::Relaxed));

        for h in 1..=10u64 {
            block_tx.send(h).await.expect("send block height");
        }

        // Wait for the watcher to drain the queue.
        let drained = wait_until(
            || state.blocks_observed.load(Ordering::Relaxed) == 10,
            Duration::from_secs(2),
        )
        .await;
        assert!(drained, "blocks_observed never reached 10");

        assert_eq!(state.blocks_observed.load(Ordering::Relaxed), 10);
        assert_eq!(
            state.folds_attempted.load(Ordering::Relaxed),
            0,
            "no folds should be attempted while phase2_enabled = false"
        );
        assert_eq!(state.folds_succeeded.load(Ordering::Relaxed), 0);
        assert_eq!(state.last_folded_height.load(Ordering::Relaxed), 0);

        shutdown_tx.send(true).expect("send shutdown");
        handle.await.expect("watcher task joined");
    }

    #[tokio::test]
    async fn test_phase2_enabled_stub_returns_err() {
        let (state, block_tx, shutdown_tx, handle) = spawn_watcher();

        // Flip the gate BEFORE sending blocks so every event takes the
        // Phase 2 path.
        state.phase2_enabled.store(true, Ordering::Relaxed);

        for h in 1..=5u64 {
            block_tx.send(h).await.expect("send block height");
        }

        let attempted = wait_until(
            || state.folds_attempted.load(Ordering::Relaxed) == 5,
            Duration::from_secs(2),
        )
        .await;
        assert!(attempted, "folds_attempted never reached 5");

        assert_eq!(state.blocks_observed.load(Ordering::Relaxed), 5);
        assert_eq!(state.folds_attempted.load(Ordering::Relaxed), 5);
        assert_eq!(
            state.folds_succeeded.load(Ordering::Relaxed),
            0,
            "stub must not report success — Phase 2 isn't implemented"
        );
        assert_eq!(state.last_folded_height.load(Ordering::Relaxed), 0);

        shutdown_tx.send(true).expect("send shutdown");
        handle.await.expect("watcher task joined");
    }

    #[tokio::test]
    async fn test_shutdown_clean() {
        let (_state, _block_tx, shutdown_tx, handle) = spawn_watcher();

        // Give the watcher a moment to park on select{}.
        tokio::time::sleep(Duration::from_millis(10)).await;

        let t0 = tokio::time::Instant::now();
        shutdown_tx.send(true).expect("send shutdown");

        let result = tokio::time::timeout(Duration::from_millis(100), handle).await;
        let elapsed = t0.elapsed();

        assert!(
            result.is_ok(),
            "watcher did not exit within 100ms (elapsed = {:?})",
            elapsed
        );
        result
            .expect("timeout")
            .expect("watcher task panicked");
        assert!(
            elapsed < Duration::from_millis(100),
            "shutdown took too long: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_state_observable() {
        let watcher = TipWatcher::new();
        let state = watcher.state();

        // Externally bump a counter BEFORE the task ever runs. This proves
        // the state Arc is genuinely shared, not a snapshot.
        state.blocks_observed.fetch_add(1, Ordering::Relaxed);
        assert_eq!(state.blocks_observed.load(Ordering::Relaxed), 1);

        let (block_tx, block_rx) = mpsc::channel(8);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let handle = tokio::spawn(watcher.run(block_rx, shutdown_rx));

        block_tx.send(42).await.expect("send block height");

        // Should observe baseline (1) + new event (1) = 2.
        let reached = wait_until(
            || state.blocks_observed.load(Ordering::Relaxed) == 2,
            Duration::from_secs(2),
        )
        .await;
        assert!(reached, "blocks_observed never reached 2 after external bump");

        shutdown_tx.send(true).expect("send shutdown");
        handle.await.expect("watcher task joined");
    }

    /// Direct unit test for the stub — does not depend on the run loop.
    /// Locks in the Err string so a careless edit gets caught.
    #[test]
    fn test_fold_block_stub_returns_err_string() {
        let watcher = TipWatcher::new();
        let result = watcher.fold_block(12345);
        assert_eq!(result, Err(PHASE2_NOT_IMPL_ERR));
    }
}
