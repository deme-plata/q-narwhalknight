//! Score persistence — write `ScoreReport`s to disk so we can train + calibrate
//! later without rerunning the pipeline against historical state.
//!
//! Why this exists (the "killer next move" from
//! docs/x-algorithm-deeper-dive-2026-05-20.md §2.6 / §A.3):
//!
//! Without persisted scores, we can never:
//!   - Verify our hand-tuned weights are right (calibration audit)
//!   - Mine hard-negative examples for future ML training
//!   - A/B-test scorer variants meaningfully
//!   - Audit information leaks against historical data
//!
//! v10.10.10 design — pragmatic, no new RocksDB column family:
//!
//! For v10.10.10 we keep score history in process memory (bounded
//! `VecDeque<ScoreEntry>` per wallet, capped at 1000 entries each). On every
//! pipeline run the `ScoreHistorySideEffect` records the top-K selected
//! candidates' scores. A new `GET /api/v1/agent/score-history/:addr`
//! endpoint returns the last 1000 for the wallet.
//!
//! This is intentionally NOT durable across restarts — we get the "history
//! exists somewhere we can query" property without touching the storage
//! engine's column-family schema (which has migration implications).
//! v10.10.11 will swap in a real RocksDB CF_SCORE_HISTORY for cross-restart
//! durability.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

use super::scorers::ScoreReport;

/// One row of persisted scoring data — a single candidate's score at the
/// moment a pipeline run selected it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreEntry {
    /// Unix-seconds when this score was computed.
    pub at_unix: i64,
    /// Wallet whose panel was being computed.
    pub viewer_wallet: String,
    /// Task that received this score.
    pub task_id: String,
    /// `TaskType` as a stable string so historical entries survive enum
    /// renaming.
    pub task_type: String,
    /// `TaskStatus` as a stable string.
    pub status: String,
    /// Full breakdown — components + total.
    pub score: ScoreReport,
    /// Whether the candidate made the final top-K cut.
    pub selected: bool,
}

/// In-memory score history shared across the pipeline. Per-wallet ring buffer
/// of recent entries.
pub struct ScoreHistory {
    /// Keyed by viewer wallet hex (no `qnk` prefix).
    by_wallet: RwLock<HashMap<String, VecDeque<ScoreEntry>>>,
    /// Per-wallet cap. Default 1000 entries.
    max_per_wallet: usize,
}

impl ScoreHistory {
    pub fn new() -> Self {
        Self {
            by_wallet: RwLock::new(HashMap::new()),
            max_per_wallet: 1000,
        }
    }

    pub fn with_capacity(max_per_wallet: usize) -> Self {
        Self {
            by_wallet: RwLock::new(HashMap::new()),
            max_per_wallet,
        }
    }

    /// Append one entry; evict oldest if over cap.
    pub fn record(&self, entry: ScoreEntry) {
        let mut map = self.by_wallet.write();
        let q = map
            .entry(entry.viewer_wallet.clone())
            .or_insert_with(VecDeque::new);
        q.push_back(entry);
        while q.len() > self.max_per_wallet {
            q.pop_front();
        }
    }

    /// Append many entries in one critical section — faster than a loop of
    /// `record` calls when SideEffects produce a batch.
    pub fn record_batch(&self, entries: Vec<ScoreEntry>) {
        if entries.is_empty() {
            return;
        }
        let mut map = self.by_wallet.write();
        for entry in entries {
            let q = map
                .entry(entry.viewer_wallet.clone())
                .or_insert_with(VecDeque::new);
            q.push_back(entry);
            while q.len() > self.max_per_wallet {
                q.pop_front();
            }
        }
    }

    /// Return the last `limit` entries for a wallet, newest first.
    pub fn get_recent(&self, wallet_hex: &str, limit: usize) -> Vec<ScoreEntry> {
        let map = self.by_wallet.read();
        match map.get(wallet_hex) {
            Some(q) => q.iter().rev().take(limit).cloned().collect(),
            None => Vec::new(),
        }
    }

    /// Total entries across all wallets — for /metrics.
    pub fn total_entries(&self) -> usize {
        self.by_wallet.read().values().map(|q| q.len()).sum()
    }

    /// Count distinct wallets being tracked.
    pub fn wallet_count(&self) -> usize {
        self.by_wallet.read().len()
    }

    /// Bulk-export all entries for a wallet (for the API endpoint).
    /// Returns a clone of the deque as a Vec, newest last.
    pub fn export_wallet(&self, wallet_hex: &str) -> Vec<ScoreEntry> {
        let map = self.by_wallet.read();
        match map.get(wallet_hex) {
            Some(q) => q.iter().cloned().collect(),
            None => Vec::new(),
        }
    }
}

impl Default for ScoreHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global history so the score-history GET endpoint and the pipeline
/// SideEffect that writes records share state without going through AppState.
static GLOBAL_HISTORY: OnceLock<Arc<ScoreHistory>> = OnceLock::new();

pub fn global() -> Arc<ScoreHistory> {
    GLOBAL_HISTORY.get_or_init(|| Arc::new(ScoreHistory::new())).clone()
}

// ════════════════════════════════════════════════════════════════════════════
// v10.10.10: file-based persistence
//
// Persists ScoreHistory state to a JSON file in the chain's data directory
// (resolved from `Q_DB_PATH` env var, falling back to `./data-mainnet-genesis`
// to match the systemd defaults).
//
// Why file + not RocksDB CF: hot_db is private inside q-storage and we don't
// want to widen its API (or do a schema migration for a new CF) in v10.10.10.
// A flat JSON file under the data-dir lives next to the chain data, gets
// included in backups automatically, and is debuggable via `jq`. v10.10.11
// will promote this to a CF_SCORE_HISTORY column family once we can do the
// migration properly.
//
// Persist policy: on every record_batch the handler spawns a background task
// that overwrites the file (write-once-per-pipeline-run, not write-per-entry,
// so disk I/O cost stays bounded). The full per-wallet rings (up to 1000 each)
// are dumped — at ~500 bytes per entry × 1000 × 10 active wallets = ~5 MB,
// well under any reasonable disk budget.
// ════════════════════════════════════════════════════════════════════════════

impl ScoreHistory {
    /// Resolve the persistence file path: `$Q_DB_PATH/agent_panel_score_history.json`.
    /// Falls back to `./data-mainnet-genesis/...` if the env var is unset, matching
    /// the default in the systemd service file.
    pub fn persist_path() -> PathBuf {
        let base = std::env::var("Q_DB_PATH")
            .unwrap_or_else(|_| "./data-mainnet-genesis".to_string());
        PathBuf::from(base).join("agent_panel_score_history.json")
    }

    /// Snapshot the full state as a single JSON blob. Used by persist().
    fn export_all(&self) -> SerializedScoreHistory {
        let map = self.by_wallet.read();
        let mut by_wallet = HashMap::with_capacity(map.len());
        for (wallet, q) in map.iter() {
            by_wallet.insert(wallet.clone(), q.iter().cloned().collect::<Vec<_>>());
        }
        SerializedScoreHistory {
            version: 1,
            max_per_wallet: self.max_per_wallet,
            by_wallet,
        }
    }

    /// Persist the full ring to disk as a single JSON file. Atomically renames
    /// a temp file over the target so a crash mid-write doesn't leave a torn
    /// file. Async — uses tokio::fs.
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
            "score_history persisted",
        );
        Ok(n)
    }

    /// Convenience wrapper that spawns a tokio task so the caller doesn't block.
    /// Used by the panel handler after record_batch so the request returns
    /// immediately while the disk write happens in the background.
    pub fn spawn_persist(self: Arc<Self>) {
        tokio::spawn(async move {
            if let Err(e) = self.persist_to_file().await {
                tracing::warn!(error = %e, "score_history persist_to_file failed");
            }
        });
    }

    /// Load score history from disk on startup. If the file doesn't exist or
    /// is malformed, returns 0 (treated as "nothing to load"). Counts the
    /// total entries loaded across all wallets.
    pub async fn load_from_file(&self) -> usize {
        let path = Self::persist_path();
        if !path.exists() {
            tracing::debug!(
                path = %path.display(),
                "score_history persist file not found (first boot or fresh DB)",
            );
            return 0;
        }
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "score_history file read failed");
                return 0;
            }
        };
        let payload: SerializedScoreHistory = match serde_json::from_slice(&bytes) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "score_history file parse failed (schema mismatch?)");
                return 0;
            }
        };
        if payload.version != 1 {
            tracing::warn!(
                version = payload.version,
                "score_history file version unknown — refusing to load",
            );
            return 0;
        }
        let mut total = 0usize;
        let mut map = self.by_wallet.write();
        for (wallet, entries) in payload.by_wallet {
            // Sort by at_unix ascending so newest ends up at the back of
            // the VecDeque (matches the live insertion order).
            let mut sorted = entries;
            sorted.sort_by_key(|e| e.at_unix);
            let q: VecDeque<ScoreEntry> = sorted.into_iter().collect();
            total += q.len();
            map.insert(wallet, q);
        }
        tracing::info!(
            total,
            wallets = map.len(),
            path = %path.display(),
            "score_history loaded from disk",
        );
        total
    }

    /// Spawn a periodic background task that persists every `interval_secs`.
    /// Useful as a fallback so we don't lose more than `interval_secs` of
    /// data on hard kill. handler.rs already calls spawn_persist after every
    /// pipeline run; this is belt-and-suspenders for periods of low traffic.
    pub fn spawn_periodic_persist(self: Arc<Self>, interval_secs: u64) {
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            // Skip the immediate first tick — startup already loaded from disk.
            tick.tick().await;
            loop {
                tick.tick().await;
                if let Err(e) = self.persist_to_file().await {
                    tracing::warn!(error = %e, "score_history periodic persist failed");
                }
            }
        });
    }
}

/// On-disk schema. Versioned so we can evolve later without losing data.
#[derive(Serialize, Deserialize)]
struct SerializedScoreHistory {
    version: u32,
    max_per_wallet: usize,
    by_wallet: HashMap<String, Vec<ScoreEntry>>,
}

/// Summary statistics over a wallet's score history — what a calibration
/// run actually needs.
#[derive(Debug, Clone, Serialize)]
pub struct ScoreHistorySummary {
    pub wallet: String,
    pub entries: usize,
    pub oldest_at_unix: i64,
    pub newest_at_unix: i64,
    /// Mean total-score for entries that ended up selected.
    pub mean_total_selected: f64,
    /// Mean total-score for entries that did NOT make the cut.
    pub mean_total_unselected: f64,
    /// Per-component mean for selected entries — what calibration would
    /// regress against future "did this tx confirm or get orphaned" labels.
    pub component_means_selected: HashMap<String, f64>,
}

impl ScoreHistory {
    /// Compute summary statistics over the cached entries for a wallet.
    /// Useful for the calibration audit (task §3.a / §3.b).
    pub fn summary(&self, wallet_hex: &str) -> Option<ScoreHistorySummary> {
        let map = self.by_wallet.read();
        let q = map.get(wallet_hex)?;
        if q.is_empty() {
            return None;
        }
        let mut sel_count = 0usize;
        let mut unsel_count = 0usize;
        let mut sel_total = 0.0f64;
        let mut unsel_total = 0.0f64;
        let mut component_sum: HashMap<String, (f64, usize)> = HashMap::new();
        let mut oldest = i64::MAX;
        let mut newest = i64::MIN;
        for e in q.iter() {
            if e.at_unix < oldest { oldest = e.at_unix; }
            if e.at_unix > newest { newest = e.at_unix; }
            if e.selected {
                sel_count += 1;
                sel_total += e.score.total;
                for c in &e.score.components {
                    let slot = component_sum.entry(c.name.clone()).or_insert((0.0, 0));
                    slot.0 += c.value;
                    slot.1 += 1;
                }
            } else {
                unsel_count += 1;
                unsel_total += e.score.total;
            }
        }
        let mean_total_selected = if sel_count > 0 { sel_total / sel_count as f64 } else { 0.0 };
        let mean_total_unselected = if unsel_count > 0 { unsel_total / unsel_count as f64 } else { 0.0 };
        let component_means_selected: HashMap<String, f64> = component_sum
            .into_iter()
            .map(|(k, (sum, n))| (k, if n > 0 { sum / n as f64 } else { 0.0 }))
            .collect();
        Some(ScoreHistorySummary {
            wallet: wallet_hex.to_string(),
            entries: q.len(),
            oldest_at_unix: if oldest == i64::MAX { 0 } else { oldest },
            newest_at_unix: if newest == i64::MIN { 0 } else { newest },
            mean_total_selected,
            mean_total_unselected,
            component_means_selected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_panel::scorers::{ScoreComponent, ScoreReport};

    fn make_entry(viewer: &str, task_id: &str, total: f64, selected: bool, at: i64) -> ScoreEntry {
        ScoreEntry {
            at_unix: at,
            viewer_wallet: viewer.to_string(),
            task_id: task_id.to_string(),
            task_type: "MempoolTx".to_string(),
            status: "Executing".to_string(),
            score: ScoreReport {
                total,
                components: vec![ScoreComponent {
                    name: "recency".to_string(),
                    value: 0.5,
                    weight: 0.4,
                    explanation: "test".to_string(),
                }],
            },
            selected,
        }
    }

    #[test]
    fn record_and_get_recent() {
        let h = ScoreHistory::new();
        h.record(make_entry("alice", "tx-1", 0.7, true, 100));
        h.record(make_entry("alice", "tx-2", 0.5, false, 200));
        let recent = h.get_recent("alice", 10);
        assert_eq!(recent.len(), 2);
        // newest first
        assert_eq!(recent[0].task_id, "tx-2");
    }

    #[test]
    fn eviction() {
        let h = ScoreHistory::with_capacity(5);
        for i in 0..20 {
            h.record(make_entry("alice", &format!("tx-{}", i), 0.5, false, i));
        }
        let recent = h.get_recent("alice", 100);
        assert_eq!(recent.len(), 5);
        // Eviction is FIFO: oldest go first, so newest tx-15..19 should remain.
        assert_eq!(recent[0].task_id, "tx-19");
        assert_eq!(recent[4].task_id, "tx-15");
    }

    #[test]
    fn summary_computes_means() {
        let h = ScoreHistory::new();
        h.record(make_entry("alice", "tx-1", 0.8, true, 100));
        h.record(make_entry("alice", "tx-2", 0.7, true, 110));
        h.record(make_entry("alice", "tx-3", 0.3, false, 120));
        let s = h.summary("alice").unwrap();
        assert_eq!(s.entries, 3);
        assert!((s.mean_total_selected - 0.75).abs() < 1e-9);
        assert!((s.mean_total_unselected - 0.30).abs() < 1e-9);
    }
}
