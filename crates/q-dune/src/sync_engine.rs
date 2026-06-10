/// Dune sync orchestrator: backfill + incremental push loop.

use crate::client::DuneClient;
use crate::extractors;
use crate::schema;
use crate::sync_state::SyncState;
use crate::DuneConfig;

use q_storage::{BalanceConsensusEngine, StorageEngine};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// How many blocks to extract per CSV upload batch.
/// Keep at 500 to avoid Dune API timeouts on free plan.
const BACKFILL_BATCH_SIZE: u64 = 500;

/// Interval between incremental syncs.
const INCREMENTAL_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes

/// Network stats snapshot passed from the caller (q-api-server).
/// This avoids q-dune depending on q-api-server types.
#[derive(Debug, Clone, Default)]
pub struct NetworkSnapshot {
    pub block_height: u64,
    pub peer_count: u32,
    pub active_miners: u32,
    pub total_hashrate_khs: f64,
    pub difficulty: f64,
    pub blocks_per_minute: f64,
    pub nakamoto_coefficient: u32,
}

/// Shared sync progress visible to the status API.
pub struct DuneSyncProgress {
    pub is_running: bool,
    pub last_pushed_height: u64,
    pub chain_tip: u64,
    pub tables_created: bool,
    pub last_error: Option<String>,
    pub total_rows_pushed: u64,
}

impl Default for DuneSyncProgress {
    fn default() -> Self {
        Self {
            is_running: false,
            last_pushed_height: 0,
            chain_tip: 0,
            tables_created: false,
            last_error: None,
            total_rows_pushed: 0,
        }
    }
}

/// Callback to fetch network stats from q-api-server without a direct dependency.
pub type NetworkSnapshotFn = Arc<dyn Fn() -> NetworkSnapshot + Send + Sync>;

pub async fn run_sync_loop(
    config: DuneConfig,
    storage: Arc<StorageEngine>,
    bce: Arc<BalanceConsensusEngine>,
    height_atomic: Arc<std::sync::atomic::AtomicU64>,
    network_snapshot_fn: NetworkSnapshotFn,
    progress: Arc<RwLock<DuneSyncProgress>>,
) {
    let client = DuneClient::new(config.api_key.clone(), config.namespace.clone());
    let sync_state = SyncState::new(Arc::clone(&storage));

    // 1. Create all tables
    info!("[Dune] Ensuring all {} tables exist in namespace '{}'...", schema::ALL_TABLES.len(), config.namespace);
    for table_def in schema::ALL_TABLES {
        match client.create_table(table_def).await {
            Ok(_) => {}
            Err(e) => {
                error!("[Dune] Failed to create table {}: {}", table_def.name, e);
                progress.write().await.last_error = Some(format!("Table creation failed: {}", e));
                if !e.to_string().contains("already exists") {
                    return;
                }
            }
        }
    }
    {
        let mut p = progress.write().await;
        p.tables_created = true;
        p.is_running = true;
    }
    info!("[Dune] All tables ready. Starting sync loop.");

    // Reset cursor if requested (e.g., after deleting/recreating tables)
    if std::env::var("DUNE_RESET_CURSOR").unwrap_or_default() == "1" {
        warn!("[Dune] DUNE_RESET_CURSOR=1 — resetting all sync cursors to 0");
        let _ = sync_state.set_last_pushed_height(0).await;
    }

    // 2. Push emission schedule once (static data)
    if let Ok(false) = sync_state.is_emission_schedule_pushed().await {
        let stats = bce.get_emission_stats().await.unwrap_or_else(|_| {
            q_storage::emission_controller::EmissionStats {
                current_era: 0,
                era_target_emission: 0,
                total_emitted_this_era: 0,
                phase: q_storage::emission_controller::EmissionPhase::Bootstrap,
                current_block_rate: 0.0,
                window_count: 0,
            }
        });
        let csv = extractors::emission_schedule::extract_emission_schedule(stats.current_era);
        match client.insert_csv("qnk_emission_schedule", &csv).await {
            Ok(rows) => {
                info!("[Dune] Pushed emission schedule ({} rows)", rows);
                let _ = sync_state.set_emission_schedule_pushed().await;
                progress.write().await.total_rows_pushed += rows;
            }
            Err(e) => warn!("[Dune] Failed to push emission schedule: {}", e),
        }
    }

    // 3. Main sync loop
    loop {
        let chain_tip = height_atomic.load(Ordering::Relaxed);
        let last_pushed = sync_state.get_last_pushed_height().await.unwrap_or(0);
        {
            let mut p = progress.write().await;
            p.chain_tip = chain_tip;
            p.last_pushed_height = last_pushed;
        }

        // --- Block-based tables: backfill + incremental ---
        // v9.1.7: Reduced from 50K→10K blocks to stay well under 100MB free plan quota.
        // At ~100 bytes/block CSV row × 10K blocks × 4 tables ≈ 4MB per full resync.
        // Set DUNE_FULL_BACKFILL=1 to sync from genesis (requires paid plan).
        if last_pushed < chain_tip {
            let full_backfill = std::env::var("DUNE_FULL_BACKFILL").unwrap_or_default() == "1";
            let start_from = if last_pushed == 0 && !full_backfill {
                // Skip to recent blocks to stay under storage quota
                chain_tip.saturating_sub(10_000).max(1)
            } else if last_pushed == 0 {
                1
            } else {
                last_pushed + 1
            };
            let mut cursor = start_from;

            while cursor <= chain_tip {
                let batch_end = (cursor + BACKFILL_BATCH_SIZE - 1).min(chain_tip);
                info!("[Dune] Syncing blocks {}-{} (tip: {})", cursor, batch_end, chain_tip);

                let genesis_ts = q_storage::emission_controller::GENESIS_TIMESTAMP as u64;

                // Extract and push blocks (critical — skip batch on failure)
                let blocks_ok = match extractors::blocks::extract_blocks(&storage, cursor, batch_end).await {
                    Ok(csv) if csv.lines().count() > 1 => {
                        match client.insert_csv("qnk_blocks", &csv).await {
                            Ok(rows) => {
                                progress.write().await.total_rows_pushed += rows;
                                true
                            }
                            Err(e) => {
                                warn!("[Dune] blocks insert failed: {}", e);
                                progress.write().await.last_error = Some(e.to_string());
                                false
                            }
                        }
                    }
                    Ok(_) => {
                        // v9.1.2: Empty batch (header only) — skip without stalling
                        warn!("[Dune] blocks batch {}-{} has no data rows, advancing cursor", cursor, batch_end);
                        true
                    }
                    Err(e) => {
                        warn!("[Dune] blocks extraction failed: {}", e);
                        false
                    }
                };
                // If blocks failed, skip this batch but continue — don't stall entire backfill
                if !blocks_ok {
                    warn!("[Dune] Skipping batch {}-{}, will retry next cycle", cursor, batch_end);
                    // Don't advance cursor — will retry this batch next loop
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    break;
                }

                // Extract and push transactions
                match extractors::transactions::extract_transactions(&storage, cursor, batch_end).await {
                    Ok(csv) if csv.lines().count() > 1 => {
                        match client.insert_csv("qnk_transactions", &csv).await {
                            Ok(rows) => progress.write().await.total_rows_pushed += rows,
                            Err(e) => warn!("[Dune] transactions insert failed: {}", e),
                        }
                    }
                    Ok(_) => {}
                    Err(e) => warn!("[Dune] transactions extraction failed: {}", e),
                }

                // Extract and push mining rewards
                match extractors::mining_rewards::extract_mining_rewards(&storage, cursor, batch_end, genesis_ts).await {
                    Ok(csv) if csv.lines().count() > 1 => {
                        match client.insert_csv("qnk_mining_rewards", &csv).await {
                            Ok(rows) => progress.write().await.total_rows_pushed += rows,
                            Err(e) => warn!("[Dune] mining_rewards insert failed: {}", e),
                        }
                    }
                    Ok(_) => {}
                    Err(e) => warn!("[Dune] mining_rewards extraction failed: {}", e),
                }

                // Extract and push DEX swaps
                match extractors::dex_swaps::extract_dex_swaps(&storage, cursor, batch_end).await {
                    Ok(csv) if csv.lines().count() > 1 => {
                        match client.insert_csv("qnk_dex_swaps", &csv).await {
                            Ok(rows) => progress.write().await.total_rows_pushed += rows,
                            Err(e) => warn!("[Dune] dex_swaps insert failed: {}", e),
                        }
                    }
                    Ok(_) => {}
                    Err(e) => warn!("[Dune] dex_swaps extraction failed: {}", e),
                }

                // Save cursor (fsync)
                if let Err(e) = sync_state.set_last_pushed_height(batch_end).await {
                    error!("[Dune] Failed to persist sync cursor: {}", e);
                    break;
                }
                progress.write().await.last_pushed_height = batch_end;

                cursor = batch_end + 1;

                // Pause between batches — Dune free plan allows ~20 req/min
                tokio::time::sleep(Duration::from_secs(10)).await;
            }
        }

        // --- Periodic tables ---
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Get circulating supply once for reuse by multiple extractors
        let circulating_supply = bce.get_emission_summary().await
            .map(|s| s.total_supply)
            .unwrap_or(0);

        // Token supply snapshot (hourly)
        let last_supply_ts = sync_state.get_last_supply_timestamp().await.unwrap_or(0);
        if now - last_supply_ts >= 3600 {
            match extractors::token_supply::extract_token_supply(&bce, now).await {
                Ok(csv) => {
                    match client.insert_csv("qnk_token_supply", &csv).await {
                        Ok(rows) => {
                            let _ = sync_state.set_last_supply_timestamp(now).await;
                            progress.write().await.total_rows_pushed += rows;
                        }
                        Err(e) => warn!("[Dune] token_supply insert failed: {}", e),
                    }
                }
                Err(e) => warn!("[Dune] token_supply extraction failed: {}", e),
            }
        }

        // Top holders (daily)
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let last_holders_date = sync_state.get_last_holders_date().await.unwrap_or(None);
        if last_holders_date.as_deref() != Some(&today) {
            match extractors::top_holders::extract_top_holders(&storage, &today).await {
                Ok(csv) => {
                    match client.insert_csv("qnk_top_holders", &csv).await {
                        Ok(rows) => {
                            let _ = sync_state.set_last_holders_date(&today).await;
                            progress.write().await.total_rows_pushed += rows;
                        }
                        Err(e) => warn!("[Dune] top_holders insert failed: {}", e),
                    }
                }
                Err(e) => warn!("[Dune] top_holders extraction failed: {}", e),
            }
        }

        // Network stats (every 5 min)
        let last_net_ts = sync_state.get_last_network_stats_ts().await.unwrap_or(0);
        if now - last_net_ts >= 300 {
            let snap = (network_snapshot_fn)();
            let csv = extractors::network_stats::extract_network_stats(
                now,
                snap.block_height,
                snap.peer_count,
                snap.active_miners,
                snap.total_hashrate_khs,
                snap.difficulty,
                snap.blocks_per_minute,
                snap.nakamoto_coefficient,
            );
            match client.insert_csv("qnk_network_stats", &csv).await {
                Ok(rows) => {
                    let _ = sync_state.set_last_network_stats_ts(now).await;
                    progress.write().await.total_rows_pushed += rows;
                }
                Err(e) => warn!("[Dune] network_stats insert failed: {}", e),
            }
        }

        // Daily metrics (check if yesterday is done)
        let yesterday = (chrono::Utc::now() - chrono::Duration::days(1)).format("%Y-%m-%d").to_string();
        let last_daily = sync_state.get_last_daily_date().await.unwrap_or(None);
        if last_daily.as_deref() != Some(&yesterday) {
            let tip = height_atomic.load(Ordering::Relaxed);
            if tip > 0 {
                match extractors::daily_metrics::extract_daily_metrics(&storage, 1, tip, &yesterday, circulating_supply).await {
                    Ok(csv) if csv.lines().count() > 1 => {
                        match client.insert_csv("qnk_daily_metrics", &csv).await {
                            Ok(rows) => {
                                let _ = sync_state.set_last_daily_date(&yesterday).await;
                                progress.write().await.total_rows_pushed += rows;
                            }
                            Err(e) => warn!("[Dune] daily_metrics insert failed: {}", e),
                        }
                    }
                    Ok(_) => {
                        let _ = sync_state.set_last_daily_date(&yesterday).await;
                    }
                    Err(e) => warn!("[Dune] daily_metrics extraction failed: {}", e),
                }
            }
        }

        // --- NEW: Miner Economics (daily) ---
        let last_miner_econ = sync_state.get_last_miner_economics_date().await.unwrap_or(None);
        if last_miner_econ.as_deref() != Some(&yesterday) {
            let tip = height_atomic.load(Ordering::Relaxed);
            if tip > 0 {
                match extractors::miner_economics::extract_miner_economics(&storage, 1, tip, &yesterday).await {
                    Ok(csv) if csv.lines().count() > 1 => {
                        match client.insert_csv("qnk_miner_economics", &csv).await {
                            Ok(rows) => {
                                let _ = sync_state.set_last_miner_economics_date(&yesterday).await;
                                progress.write().await.total_rows_pushed += rows;
                            }
                            Err(e) => warn!("[Dune] miner_economics insert failed: {}", e),
                        }
                    }
                    Ok(_) => {
                        let _ = sync_state.set_last_miner_economics_date(&yesterday).await;
                    }
                    Err(e) => warn!("[Dune] miner_economics extraction failed: {}", e),
                }
            }
        }

        // --- NEW: Wealth Distribution (daily) ---
        let last_wealth = sync_state.get_last_wealth_distribution_date().await.unwrap_or(None);
        if last_wealth.as_deref() != Some(&today) {
            match extractors::wealth_distribution::extract_wealth_distribution(&storage, &today).await {
                Ok(csv) if csv.lines().count() > 1 => {
                    match client.insert_csv("qnk_wealth_distribution", &csv).await {
                        Ok(rows) => {
                            let _ = sync_state.set_last_wealth_distribution_date(&today).await;
                            progress.write().await.total_rows_pushed += rows;
                        }
                        Err(e) => warn!("[Dune] wealth_distribution insert failed: {}", e),
                    }
                }
                Ok(_) => {
                    let _ = sync_state.set_last_wealth_distribution_date(&today).await;
                }
                Err(e) => warn!("[Dune] wealth_distribution extraction failed: {}", e),
            }
        }

        // --- NEW: Block Time Analysis (daily) ---
        let last_bt = sync_state.get_last_block_time_analysis_date().await.unwrap_or(None);
        if last_bt.as_deref() != Some(&yesterday) {
            let tip = height_atomic.load(Ordering::Relaxed);
            if tip > 0 {
                match extractors::block_time_analysis::extract_block_time_analysis(&storage, 1, tip, &yesterday).await {
                    Ok(csv) if csv.lines().count() > 1 => {
                        match client.insert_csv("qnk_block_time_analysis", &csv).await {
                            Ok(rows) => {
                                let _ = sync_state.set_last_block_time_analysis_date(&yesterday).await;
                                progress.write().await.total_rows_pushed += rows;
                            }
                            Err(e) => warn!("[Dune] block_time_analysis insert failed: {}", e),
                        }
                    }
                    Ok(_) => {
                        let _ = sync_state.set_last_block_time_analysis_date(&yesterday).await;
                    }
                    Err(e) => warn!("[Dune] block_time_analysis extraction failed: {}", e),
                }
            }
        }

        info!(
            "[Dune] Sync cycle complete. Height: {}/{}, rows pushed: {}",
            progress.read().await.last_pushed_height,
            progress.read().await.chain_tip,
            progress.read().await.total_rows_pushed,
        );

        tokio::time::sleep(INCREMENTAL_INTERVAL).await;
    }
}
