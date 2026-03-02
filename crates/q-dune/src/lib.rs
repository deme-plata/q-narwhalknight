//! # q-dune: Dune Analytics Data Pipeline for Q-NarwhalKnight
//!
//! Pushes blockchain data (blocks, transactions, mining rewards, DEX swaps,
//! token supply, top holders, network stats, emission schedule) to Dune
//! Analytics custom tables via their CSV upload API.
//!
//! ## Usage
//! ```ignore
//! if std::env::var("DUNE_ENABLED").unwrap_or_default() == "1" {
//!     let config = q_dune::DuneConfig::from_env()?;
//!     tokio::spawn(q_dune::start_dune_sync_task(state, config));
//! }
//! ```

pub mod client;
pub mod csv_formatter;
pub mod extractors;
pub mod schema;
pub mod sync_engine;
pub mod sync_state;

use q_storage::{BalanceConsensusEngine, StorageEngine};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

pub use sync_engine::{DuneSyncProgress, NetworkSnapshot, NetworkSnapshotFn};

/// Configuration for the Dune Analytics pipeline.
#[derive(Debug, Clone)]
pub struct DuneConfig {
    pub api_key: String,
    pub namespace: String,
}

impl DuneConfig {
    /// Load config from environment variables.
    /// Required: `DUNE_API_KEY`, `DUNE_NAMESPACE`.
    pub fn from_env() -> anyhow::Result<Self> {
        let api_key = std::env::var("DUNE_API_KEY")
            .map_err(|_| anyhow::anyhow!("DUNE_API_KEY env var not set"))?;
        let namespace = std::env::var("DUNE_NAMESPACE")
            .unwrap_or_else(|_| "qnk".to_string());
        Ok(Self { api_key, namespace })
    }
}

/// Start the background Dune sync task.
/// Returns the shared progress handle for the status API.
pub async fn start_dune_sync_task(
    storage: Arc<StorageEngine>,
    bce: Arc<BalanceConsensusEngine>,
    height_atomic: Arc<std::sync::atomic::AtomicU64>,
    network_snapshot_fn: NetworkSnapshotFn,
    config: DuneConfig,
) -> Arc<RwLock<DuneSyncProgress>> {
    let progress = Arc::new(RwLock::new(DuneSyncProgress::default()));
    let progress_clone = Arc::clone(&progress);

    info!("[Dune] Starting Dune Analytics sync pipeline (namespace: {})", config.namespace);

    tokio::spawn(sync_engine::run_sync_loop(
        config,
        storage,
        bce,
        height_atomic,
        network_snapshot_fn,
        progress_clone,
    ));

    progress
}

/// Handler: GET /api/v1/dune/status — returns sync progress as JSON.
pub async fn dune_status(
    progress: Arc<RwLock<DuneSyncProgress>>,
) -> serde_json::Value {
    let p = progress.read().await;
    serde_json::json!({
        "is_running": p.is_running,
        "last_pushed_height": p.last_pushed_height,
        "chain_tip": p.chain_tip,
        "tables_created": p.tables_created,
        "last_error": p.last_error,
        "total_rows_pushed": p.total_rows_pushed,
        "sync_pct": if p.chain_tip > 0 {
            (p.last_pushed_height as f64 / p.chain_tip as f64 * 100.0).min(100.0)
        } else { 0.0 },
    })
}
