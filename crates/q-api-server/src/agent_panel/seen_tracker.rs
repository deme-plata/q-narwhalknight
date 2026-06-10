//! Previously-seen task tracking for the agent activity panel.
//!
//! Without this, an agent polling `/api/v1/agent/panel/{wallet}` every 5s
//! would see the same tasks forever, defeating the panel's purpose. Twitter's
//! `home-mixer/filters/previously_seen_posts_filter.rs` solves the exact same
//! problem on their side; we mirror the pattern.
//!
//! v10.10.10 implementation: in-process `RwLock<HashMap<wallet, ...>>`. Not
//! persisted across restarts — a node restart resets the seen-set, so the
//! next pipeline run shows everything again. That's acceptable for now;
//! v10.10.11 will swap in a RocksDB CF_PANEL_SEEN column family for cross-
//! restart durability.
//!
//! Memory bound: per wallet, we keep up to 5000 task_ids. When the cap is
//! hit we drop the oldest half. Across N wallets the worst case is
//! `N × 5000 × 64 bytes ≈ 320 KB per wallet × N`. At N=1000 active wallets
//! that's 320 MB — bounded.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

/// Per-wallet seen-set with bounded LRU-ish eviction.
struct WalletSeenSet {
    /// Fast membership lookup.
    set: HashSet<String>,
    /// Insertion-order queue for eviction. When `set.len()` exceeds the cap,
    /// we drain half from the front and rebuild `set` from the remaining.
    order: VecDeque<String>,
}

impl WalletSeenSet {
    fn new() -> Self {
        Self {
            set: HashSet::new(),
            order: VecDeque::new(),
        }
    }

    fn has(&self, task_id: &str) -> bool {
        self.set.contains(task_id)
    }

    fn insert(&mut self, task_id: String, max_size: usize) {
        if self.set.contains(&task_id) {
            return;
        }
        self.set.insert(task_id.clone());
        self.order.push_back(task_id);

        // Evict oldest half if we're over cap.
        if self.set.len() > max_size {
            let to_drop = self.set.len() / 2;
            for _ in 0..to_drop {
                if let Some(old) = self.order.pop_front() {
                    self.set.remove(&old);
                }
            }
        }
    }

    fn len(&self) -> usize {
        self.set.len()
    }
}

/// Shared tracker — clone the `Arc<SeenTracker>` into both the
/// `PreviouslySeenFilter` and the `SeenRecorderSideEffect`.
pub struct SeenTracker {
    /// Keyed by wallet hex (no `qnk` prefix).
    by_wallet: RwLock<HashMap<String, WalletSeenSet>>,
    /// Per-wallet cap. Default 5000.
    max_per_wallet: usize,
}

impl SeenTracker {
    pub fn new() -> Self {
        Self {
            by_wallet: RwLock::new(HashMap::new()),
            max_per_wallet: 5000,
        }
    }

    pub fn with_capacity(max_per_wallet: usize) -> Self {
        Self {
            by_wallet: RwLock::new(HashMap::new()),
            max_per_wallet,
        }
    }

    /// Check whether the wallet has already seen the task. O(1).
    pub fn has_seen(&self, wallet_hex: &str, task_id: &str) -> bool {
        let map = self.by_wallet.read();
        match map.get(wallet_hex) {
            Some(set) => set.has(task_id),
            None => false,
        }
    }

    /// Record that the wallet has seen the task. O(1) amortised; O(n)
    /// when eviction triggers (rare — once per 5000 insertions per wallet).
    pub fn mark_seen(&self, wallet_hex: &str, task_id: &str) {
        let mut map = self.by_wallet.write();
        let set = map
            .entry(wallet_hex.to_string())
            .or_insert_with(WalletSeenSet::new);
        set.insert(task_id.to_string(), self.max_per_wallet);
    }

    /// Total entries across all wallets — useful for /metrics.
    pub fn total_entries(&self) -> usize {
        self.by_wallet.read().values().map(|s| s.len()).sum()
    }

    /// Number of wallets being tracked.
    pub fn wallet_count(&self) -> usize {
        self.by_wallet.read().len()
    }

    /// Clear entries older than `_age_secs`. v10.10.10 stub — we don't
    /// track per-entry timestamps yet, so this currently clears NOTHING.
    /// v10.10.11 will add a `(timestamp, task_id)` representation that
    /// supports time-based eviction.
    pub fn prune_older_than(&self, _age_secs: u64) {
        // intentional no-op
    }
}

impl Default for SeenTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global tracker so handler.rs and other call sites share state
/// without threading it through AppState. v10.10.11 may move this to
/// AppState if we end up needing per-environment configuration; for
/// v10.10.10 the singleton is the right complexity tradeoff.
static GLOBAL_TRACKER: OnceLock<Arc<SeenTracker>> = OnceLock::new();

pub fn global() -> Arc<SeenTracker> {
    GLOBAL_TRACKER.get_or_init(|| Arc::new(SeenTracker::new())).clone()
}

// ════════════════════════════════════════════════════════════════════════════
// v10.10.10: file-based persistence
//
// Same pragma as score_history: persist to a JSON file under Q_DB_PATH so
// the agent's "seen" set survives restarts. Without persistence, every
// node restart would resurface every previously-shown task to every polling
// agent — defeating the entire point of the filter.
//
// Persist trigger: handler.rs spawns persistence after each pipeline run.
// Also periodic auto-persist every 60s as belt-and-suspenders.
// ════════════════════════════════════════════════════════════════════════════

impl SeenTracker {
    /// Resolve the persistence file path: `$Q_DB_PATH/agent_panel_seen_tracker.json`.
    pub fn persist_path() -> PathBuf {
        let base = std::env::var("Q_DB_PATH")
            .unwrap_or_else(|_| "./data-mainnet-genesis".to_string());
        PathBuf::from(base).join("agent_panel_seen_tracker.json")
    }

    /// Snapshot internal state to a serialisable form.
    fn export_all(&self) -> SerializedSeenTracker {
        let map = self.by_wallet.read();
        let mut by_wallet = HashMap::with_capacity(map.len());
        for (wallet, set) in map.iter() {
            // Preserve insertion order via `order` so the LRU semantics
            // survive restart (oldest task_ids will be evicted first
            // after restart, matching pre-restart behaviour).
            by_wallet.insert(
                wallet.clone(),
                set.order.iter().cloned().collect::<Vec<_>>(),
            );
        }
        SerializedSeenTracker {
            version: 1,
            max_per_wallet: self.max_per_wallet,
            by_wallet,
        }
    }

    /// Persist to JSON file via atomic rename. Async.
    pub async fn persist_to_file(&self) -> Result<usize, String> {
        let payload = self.export_all();
        let n = payload.by_wallet.values().map(|v| v.len()).sum();
        let path = Self::persist_path();
        let tmp = path.with_extension("json.tmp");

        let bytes = serde_json::to_vec_pretty(&payload)
            .map_err(|e| format!("encode: {}", e))?;

        if let Some(parent) = path.parent() {
            if !parent.exists() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|e| format!("create_dir_all: {}", e))?;
            }
        }
        tokio::fs::write(&tmp, &bytes)
            .await
            .map_err(|e| format!("write tmp: {}", e))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .map_err(|e| format!("rename: {}", e))?;
        tracing::debug!(
            entries = n,
            wallets = payload.by_wallet.len(),
            path = %path.display(),
            "seen_tracker persisted",
        );
        Ok(n)
    }

    /// Spawn a fire-and-forget persist. Used by the panel handler.
    pub fn spawn_persist(self: Arc<Self>) {
        tokio::spawn(async move {
            if let Err(e) = self.persist_to_file().await {
                tracing::warn!(error = %e, "seen_tracker persist_to_file failed");
            }
        });
    }

    /// Load from disk on startup. Returns total entries loaded.
    pub async fn load_from_file(&self) -> usize {
        let path = Self::persist_path();
        if !path.exists() {
            tracing::debug!(
                path = %path.display(),
                "seen_tracker persist file not found (first boot)",
            );
            return 0;
        }
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "seen_tracker file read failed");
                return 0;
            }
        };
        let payload: SerializedSeenTracker = match serde_json::from_slice(&bytes) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "seen_tracker file parse failed (schema mismatch?)");
                return 0;
            }
        };
        if payload.version != 1 {
            tracing::warn!(
                version = payload.version,
                "seen_tracker file version unknown — refusing to load",
            );
            return 0;
        }
        let mut total = 0usize;
        let mut map = self.by_wallet.write();
        for (wallet, task_ids) in payload.by_wallet {
            let mut set = WalletSeenSet::new();
            for tid in &task_ids {
                set.insert(tid.clone(), self.max_per_wallet);
            }
            total += set.set.len();
            map.insert(wallet, set);
        }
        tracing::info!(
            total,
            wallets = map.len(),
            path = %path.display(),
            "seen_tracker loaded from disk",
        );
        total
    }

    /// Spawn a periodic background persist (every `interval_secs`).
    pub fn spawn_periodic_persist(self: Arc<Self>, interval_secs: u64) {
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            tick.tick().await; // skip first tick
            loop {
                tick.tick().await;
                if let Err(e) = self.persist_to_file().await {
                    tracing::warn!(error = %e, "seen_tracker periodic persist failed");
                }
            }
        });
    }
}

/// On-disk schema. Versioned for forward-compat.
#[derive(Serialize, Deserialize)]
struct SerializedSeenTracker {
    version: u32,
    max_per_wallet: usize,
    by_wallet: HashMap<String, Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_tracker_reports_unseen() {
        let t = SeenTracker::new();
        assert!(!t.has_seen("alice", "task-1"));
    }

    #[test]
    fn mark_then_check() {
        let t = SeenTracker::new();
        t.mark_seen("alice", "task-1");
        assert!(t.has_seen("alice", "task-1"));
        assert!(!t.has_seen("alice", "task-2"));
        assert!(!t.has_seen("bob", "task-1"));
    }

    #[test]
    fn eviction_keeps_set_bounded() {
        let t = SeenTracker::with_capacity(10);
        for i in 0..25 {
            t.mark_seen("alice", &format!("task-{}", i));
        }
        // After exceeding cap we evict half; final size should be ≤ cap.
        assert!(t.total_entries() <= 10);
        // The most recent ID should still be present.
        assert!(t.has_seen("alice", "task-24"));
    }

    #[test]
    fn dedup_within_wallet() {
        let t = SeenTracker::new();
        t.mark_seen("alice", "task-1");
        t.mark_seen("alice", "task-1");
        t.mark_seen("alice", "task-1");
        // Three inserts of the same id should add just one entry.
        assert_eq!(t.total_entries(), 1);
    }

    #[test]
    fn metrics() {
        let t = SeenTracker::new();
        t.mark_seen("alice", "x");
        t.mark_seen("bob", "y");
        t.mark_seen("bob", "z");
        assert_eq!(t.wallet_count(), 2);
        assert_eq!(t.total_entries(), 3);
    }
}
