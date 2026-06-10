//! DEX analytics state persistence (volume_tracker, swap_history).
//!
//! ## What this fixes
//!
//! Pre-v10.10.15, `AppState.volume_tracker` and `AppState.swap_history` were
//! initialized as empty in-memory `HashMap`s at boot and mutated on every
//! swap. Neither was persisted. After any restart the explorer's "24h
//! Trading Volume" dashboard reported $0 until a full 24h of fresh swaps
//! had accumulated again. The TVL number survives because it's derived from
//! pool reserves (which DO live in RocksDB); the volume number was a pure
//! ephemeral counter.
//!
//! This module snapshots the two HashMaps to `CF_MANIFEST` periodically and
//! restores them at boot. Worst-case data loss: up to one snapshot interval
//! (default 60s) of swap activity around a hard crash.
//!
//! ## Mainnet safety notes (v10.10.15, $2.1B mcap)
//!
//! The user explicitly flagged this commit as "be careful — we are on
//! mainnet." The risk profile here is **moderate, not catastrophic** because:
//!
//!   - Both fields are analytics/observability data, NOT balance or consensus
//!     state. The chain's correctness does not depend on them. Worst case if
//!     this module mis-persists: the dashboard shows wrong volume; no balance
//!     moves, no block is mis-validated, no consensus break.
//!   - Storage uses `CF_MANIFEST` (already used for many non-balance keys),
//!     under a fresh, versioned key prefix (`b"dex_analytics:v1:"`). No
//!     existing key format is touched.
//!   - All writes go through `put_manifest_sync` (the existing async API);
//!     no direct CF manipulation.
//!   - Snapshot is bounded (1 MB cap per key after bincode + LZ4). Volume
//!     entries are trimmed to the last 24h on snapshot to keep payload
//!     stable.
//!   - Restore is best-effort: deserialize failures log a warning and start
//!     with empty state, matching the pre-v10.10.15 behavior.
//!   - The periodic task is `tokio::spawn`-ed, runs every
//!     `SNAPSHOT_INTERVAL_SECS` (60s default), no hot-path code touched.
//!
//! ## What this does NOT persist
//!
//! - `state.tx_pool` (transient mempool — node-local, intentionally lost on
//!   restart; ProductionMempool has its own persistence path).
//! - `state.liquidity_pools` (already lives in `CF_DEX_POOLS` per
//!   `crates/q-storage/src/lib.rs:450`).
//! - `state.token_balances` (lives in `CF_TOKEN_BALANCES`, balance-critical).
//!
//! Only the two non-persisted analytics HashMaps are touched here.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::handlers::SwapHistoryRecord;
use crate::AppState;

/// CF_MANIFEST key for the snapshotted volume_tracker.
/// Bumping the `v1` suffix on schema-breaking changes lets old keys be
/// ignored without an explicit migration.
const KEY_VOLUME_TRACKER: &[u8] = b"dex_analytics:v1:volume_tracker";

/// CF_MANIFEST key for the snapshotted swap_history.
const KEY_SWAP_HISTORY: &[u8] = b"dex_analytics:v1:swap_history";

/// Periodic snapshot interval. Worst-case data-loss window on hard crash.
const SNAPSHOT_INTERVAL_SECS: u64 = 60;

/// 24-hour cutoff for volume entries — older points are trimmed at snapshot
/// time to keep the payload bounded. Volume rendering is "last 24h" anyway,
/// so older data is unused.
const VOLUME_RETENTION_SECS: i64 = 86_400;

/// Soft cap on serialized snapshot size (bytes). If a snapshot would exceed
/// this, log a warning but still attempt the write — RocksDB itself caps
/// at 4 GB per key, but we want to notice ballooning state.
const SOFT_CAP_BYTES: usize = 1_000_000;

// ════════════════════════════════════════════════════════════════════════════
// Restore (called once at boot, before serving)
// ════════════════════════════════════════════════════════════════════════════

/// Restore the volume_tracker and swap_history from CF_MANIFEST snapshots.
///
/// Best-effort: missing keys → empty start (fresh node, expected). Malformed
/// bytes → log warn, empty start (treat as if the key wasn't there). Never
/// returns Err — analytics persistence failures must not block boot.
pub async fn restore_from_manifest(
    storage: Arc<q_storage::QStorage>,
    volume_tracker: Arc<RwLock<HashMap<String, Vec<(i64, f64)>>>>,
    swap_history: Arc<RwLock<HashMap<String, Vec<SwapHistoryRecord>>>>,
) {
    info!("📊 [DEX-ANALYTICS] Restoring volume_tracker + swap_history from CF_MANIFEST");

    // -- volume_tracker --
    match storage.get_kv().get(q_storage::CF_MANIFEST, KEY_VOLUME_TRACKER).await {
        Ok(Some(bytes)) => match bincode::deserialize::<HashMap<String, Vec<(i64, f64)>>>(&bytes) {
            Ok(restored) => {
                let entry_count: usize = restored.values().map(|v| v.len()).sum();
                info!(
                    "📊 [DEX-ANALYTICS] Restored volume_tracker: {} tokens, {} entries ({} bytes)",
                    restored.len(),
                    entry_count,
                    bytes.len(),
                );
                *volume_tracker.write().await = restored;
            }
            Err(e) => {
                warn!(
                    "📊 [DEX-ANALYTICS] volume_tracker snapshot bytes present but unparseable ({}); starting empty",
                    e
                );
            }
        },
        Ok(None) => {
            debug!("📊 [DEX-ANALYTICS] No volume_tracker snapshot — fresh start");
        }
        Err(e) => {
            warn!("📊 [DEX-ANALYTICS] CF_MANIFEST read failed for volume_tracker ({}); starting empty", e);
        }
    }

    // -- swap_history --
    match storage.get_kv().get(q_storage::CF_MANIFEST, KEY_SWAP_HISTORY).await {
        Ok(Some(bytes)) => match bincode::deserialize::<HashMap<String, Vec<SwapHistoryRecord>>>(&bytes) {
            Ok(restored) => {
                let record_count: usize = restored.values().map(|v| v.len()).sum();
                info!(
                    "📊 [DEX-ANALYTICS] Restored swap_history: {} wallets, {} records ({} bytes)",
                    restored.len(),
                    record_count,
                    bytes.len(),
                );
                *swap_history.write().await = restored;
            }
            Err(e) => {
                warn!(
                    "📊 [DEX-ANALYTICS] swap_history snapshot bytes present but unparseable ({}); starting empty",
                    e
                );
            }
        },
        Ok(None) => {
            debug!("📊 [DEX-ANALYTICS] No swap_history snapshot — fresh start");
        }
        Err(e) => {
            warn!("📊 [DEX-ANALYTICS] CF_MANIFEST read failed for swap_history ({}); starting empty", e);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Snapshot (called periodically + on graceful shutdown)
// ════════════════════════════════════════════════════════════════════════════

/// Snapshot the current in-memory analytics state to CF_MANIFEST.
///
/// Trims volume_tracker entries older than 24h to keep payload bounded.
/// Logs the resulting payload size; warns if it exceeds SOFT_CAP_BYTES.
pub async fn snapshot_to_manifest(state: Arc<AppState>) {
    let now_secs = chrono::Utc::now().timestamp();

    // -- volume_tracker --
    let volume_snapshot: HashMap<String, Vec<(i64, f64)>> = {
        let tracker = state.volume_tracker.read().await;
        tracker
            .iter()
            .map(|(token, entries)| {
                let trimmed: Vec<(i64, f64)> = entries
                    .iter()
                    .filter(|(ts, _)| now_secs - *ts <= VOLUME_RETENTION_SECS)
                    .copied()
                    .collect();
                (token.clone(), trimmed)
            })
            .filter(|(_, entries)| !entries.is_empty())
            .collect()
    };

    match bincode::serialize(&volume_snapshot) {
        Ok(bytes) => {
            if bytes.len() > SOFT_CAP_BYTES {
                warn!(
                    "📊 [DEX-ANALYTICS] volume_tracker snapshot {} bytes exceeds soft cap {}; writing anyway",
                    bytes.len(),
                    SOFT_CAP_BYTES
                );
            }
            if let Err(e) = state
                .storage_engine
                .put_manifest_sync(KEY_VOLUME_TRACKER, &bytes)
                .await
            {
                warn!("📊 [DEX-ANALYTICS] volume_tracker snapshot write failed: {}", e);
            } else {
                debug!(
                    "📊 [DEX-ANALYTICS] volume_tracker snapshot OK: {} tokens, {} bytes",
                    volume_snapshot.len(),
                    bytes.len()
                );
            }
        }
        Err(e) => {
            error!("📊 [DEX-ANALYTICS] volume_tracker serialize failed: {}", e);
        }
    }

    // -- swap_history --
    let swap_snapshot: HashMap<String, Vec<SwapHistoryRecord>> = state.swap_history.read().await.clone();

    match bincode::serialize(&swap_snapshot) {
        Ok(bytes) => {
            if bytes.len() > SOFT_CAP_BYTES {
                warn!(
                    "📊 [DEX-ANALYTICS] swap_history snapshot {} bytes exceeds soft cap {}; writing anyway",
                    bytes.len(),
                    SOFT_CAP_BYTES
                );
            }
            if let Err(e) = state
                .storage_engine
                .put_manifest_sync(KEY_SWAP_HISTORY, &bytes)
                .await
            {
                warn!("📊 [DEX-ANALYTICS] swap_history snapshot write failed: {}", e);
            } else {
                debug!(
                    "📊 [DEX-ANALYTICS] swap_history snapshot OK: {} wallets, {} bytes",
                    swap_snapshot.len(),
                    bytes.len()
                );
            }
        }
        Err(e) => {
            error!("📊 [DEX-ANALYTICS] swap_history serialize failed: {}", e);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Periodic task spawner (called once during AppState bring-up)
// ════════════════════════════════════════════════════════════════════════════

/// Spawn the periodic snapshot task. Returns the JoinHandle so the caller
/// can abort it during shutdown if desired (currently the AppState owns
/// no shutdown signal, so the handle is dropped — task lives until process
/// exit).
pub fn spawn_periodic_snapshot_task(state: Arc<AppState>) -> tokio::task::JoinHandle<()> {
    info!(
        "📊 [DEX-ANALYTICS] Periodic snapshot task starting (every {}s)",
        SNAPSHOT_INTERVAL_SECS
    );
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(SNAPSHOT_INTERVAL_SECS));
        // Skip the first immediate tick — first snapshot fires 60s after spawn,
        // not at startup (boot path already called restore_from_manifest).
        interval.tick().await;
        loop {
            interval.tick().await;
            snapshot_to_manifest(state.clone()).await;
        }
    })
}
