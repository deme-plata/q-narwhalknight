//! Block Production Loop v1.0.3-beta - Comprehensive Stall Protection

use std::sync::{atomic::{AtomicU64, AtomicU8, Ordering}, Arc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::{self, MissedTickBehavior};
use tracing::{debug, error, info, warn};
use serde_json::json;
use crate::AppState;

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum ProductionPhase {
    BeforeTick = 0,
    AfterTick = 1,
    BeforeShouldProduce = 2,
    AfterShouldProduce = 3,
    BeforeProduceBlocks = 4,
    AfterProduceBlocks = 5,
    BeforeSaveBlocks = 6,
    AfterSaveBlocks = 7,
    BeforeBroadcast = 8,
    AfterBroadcast = 9,
}

impl ProductionPhase {
    fn from_u8(val: u8) -> &'static str {
        match val {
            0 => "BeforeTick", 1 => "AfterTick", 2 => "BeforeShouldProduce",
            3 => "AfterShouldProduce", 4 => "BeforeProduceBlocks", 5 => "AfterProduceBlocks",
            6 => "BeforeSaveBlocks", 7 => "AfterSaveBlocks", 8 => "BeforeBroadcast",
            9 => "AfterBroadcast", _ => "Unknown",
        }
    }
}

pub const USE_SLEEP_INSTEAD_OF_INTERVAL: bool = false;

pub fn spawn_block_production_loop(app_state: Arc<AppState>) {
    info!("🚀 [v1.0.3-beta] Block production with comprehensive stall protection");
    info!("   Mode: {}", if USE_SLEEP_INSTEAD_OF_INTERVAL { "SLEEP" } else { "INTERVAL" });

    let heartbeat = Arc::new(AtomicU64::new(0));
    let phase = Arc::new(AtomicU8::new(ProductionPhase::BeforeTick as u8));

    let (hb_loop, hb_watch, hb_panic) = (heartbeat.clone(), heartbeat.clone(), heartbeat.clone());
    let (ph_loop, ph_watch, ph_panic) = (phase.clone(), phase.clone(), phase.clone());
    let state_loop = app_state.clone();

    let handle = tokio::spawn(async move {
        production_loop(state_loop, hb_loop, ph_loop, USE_SLEEP_INSTEAD_OF_INTERVAL).await
    });

    tokio::spawn(async move { monitor_panics(handle, hb_panic, ph_panic).await; });
    tokio::spawn(async move { run_watchdog(hb_watch, ph_watch).await; });

    info!("✅ Block production initialized with watchdog");
}

async fn production_loop(
    app: Arc<AppState>, hb: Arc<AtomicU64>, ph: Arc<AtomicU8>, sleep: bool
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut iter = 0u64;
    let start = std::time::Instant::now();
    let mut interval = if !sleep {
        let mut i = time::interval(Duration::from_secs(1));
        i.set_missed_tick_behavior(MissedTickBehavior::Skip);
        Some(i)
    } else { None };

    loop {
        ph.store(ProductionPhase::BeforeTick as u8, Ordering::SeqCst);
        if let Some(ref mut i) = interval { i.tick().await; } else { time::sleep(Duration::from_secs(1)).await; }
        ph.store(ProductionPhase::AfterTick as u8, Ordering::SeqCst);

        hb.fetch_add(1, Ordering::SeqCst);
        iter += 1;

        if iter % 30 == 0 {
            // v1.0.62-beta: Use current_height_atomic for accurate height in heartbeat
            let h = app.current_height_atomic.load(Ordering::SeqCst);
            info!("💓 {} HB: iter={}, height={}, uptime={}s, phase={}",
                if sleep {"SLEEP"} else {"INTERVAL"}, iter, h, start.elapsed().as_secs(),
                ProductionPhase::from_u8(ph.load(Ordering::SeqCst)));
        }

        // v1.0.62-beta: Use current_height_atomic (updated by P2P sync) instead of stale node_status
        let cur_h = app.current_height_atomic.load(Ordering::SeqCst);
        let net_h = app.highest_network_height.load(Ordering::SeqCst);

        // v1.0.62-beta: Auto-correct highest_network_height if we've caught up
        // This fixes the bug where node shows "Production DISABLED: N behind" even when synced
        if cur_h > 0 && net_h > 0 && cur_h >= net_h {
            // We've caught up or surpassed the network - update highest_network_height
            app.highest_network_height.store(cur_h, Ordering::SeqCst);
        }

        if net_h > 0 {
            let gap = net_h.saturating_sub(cur_h);
            if gap > 1000 { if iter % 30 == 0 { warn!("🚫 Production DISABLED: {} behind (local={}, net={})", gap, cur_h, net_h); } continue; }
            if gap > 10 { if iter % 30 == 0 { debug!("⏸️  Syncing {} behind", gap); } continue; }
        }

        ph.store(ProductionPhase::BeforeShouldProduce as u8, Ordering::SeqCst);
        let should = match time::timeout(Duration::from_secs(10), app.block_producer_pool.should_produce()).await {
            Ok(Ok(r)) => r, Ok(Err(e)) => { error!("🚨 should_produce: {}", e); false }
            Err(_) => { error!("🚨 should_produce TIMEOUT"); false }
        };
        ph.store(ProductionPhase::AfterShouldProduce as u8, Ordering::SeqCst);
        if !should { continue; }

        ph.store(ProductionPhase::BeforeProduceBlocks as u8, Ordering::SeqCst);
        let blocks = match time::timeout(Duration::from_secs(30), app.block_producer_pool.produce_blocks()).await {
            Ok(b) => b,
            Err(_) => { error!("🚨 produce_blocks TIMEOUT"); vec![] }
        };
        ph.store(ProductionPhase::AfterProduceBlocks as u8, Ordering::SeqCst);
        if blocks.is_empty() { continue; }

        for (pid, blk) in &blocks {
            ph.store(ProductionPhase::BeforeSaveBlocks as u8, Ordering::SeqCst);
            match time::timeout(Duration::from_secs(30), app.storage_engine.save_qblock(blk)).await {
                Ok(Ok(_)) => {
                    debug!("✅ Block saved: h={}, p={}", blk.header.height, pid);

                    // ✨ v1.4.0-beta: Epoch boundary detection for recursive proofs
                    // Check if this block crosses an epoch boundary (every 1000 blocks)
                    const EPOCH_BLOCKS: u64 = 1000;
                    let current_epoch = blk.header.height / EPOCH_BLOCKS;
                    let prev_epoch = blk.header.height.saturating_sub(1) / EPOCH_BLOCKS;

                    if current_epoch > prev_epoch && blk.header.height >= EPOCH_BLOCKS {
                        info!("🔐 Epoch boundary crossed! Block {} completes epoch {}",
                              blk.header.height, prev_epoch);

                        // Trigger epoch proof generation if recursive proofs service is enabled
                        if let Some(ref service) = app.recursive_proofs_service {
                            let service = service.clone();
                            let epoch = prev_epoch;
                            let height = blk.header.height;

                            // Spawn async task to trigger proof generation
                            tokio::spawn(async move {
                                info!("🔐 Triggering epoch proof generation for epoch {} (blocks 0-{})",
                                      epoch, height);
                                service.set_epoch(epoch + 1).await;

                                // If this node has a prover, start generating proof
                                if let Some(prover) = service.get_prover().await {
                                    let task = q_recursive_proofs::protocol::messages::EpochProofTask {
                                        epoch,
                                        deadline: std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_secs() + 300, // 5 minute deadline
                                        ..Default::default()
                                    };

                                    if let Err(e) = prover.handle_task(task).await {
                                        warn!("⚠️  Failed to start epoch proof task: {}", e);
                                    } else {
                                        info!("✅ Epoch {} proof generation task started", epoch);
                                    }
                                } else {
                                    info!("🔐 Prover not ready, epoch proof will be generated by network peers");
                                }
                            });
                        }
                    }
                },
                Ok(Err(e)) => { error!("🚨 save_qblock: {}", e); continue; }
                Err(_) => { error!("🚨 save_qblock TIMEOUT"); continue; }
            }
            ph.store(ProductionPhase::AfterSaveBlocks as u8, Ordering::SeqCst);

            ph.store(ProductionPhase::BeforeBroadcast as u8, Ordering::SeqCst);
            let block_hash = blk.calculate_hash();
            let _ = app.event_broadcaster.broadcast(
                q_api_server::streaming::StreamEvent::NewBlock {
                    height: blk.header.height,
                    hash: hex::encode(&block_hash),
                    prev_hash: hex::encode(&blk.header.prev_block_hash),
                    solutions_count: blk.mining_solutions.len(),
                    total_difficulty: blk.header.total_difficulty,
                    dag_round: blk.header.height,
                    miner_count: blk.mining_solutions.len(),
                    tx_count: blk.transactions.len(),
                    block_reward: 0.0, // TODO: Calculate actual reward
                    producer_id: *pid,
                    timestamp: chrono::DateTime::from_timestamp(blk.header.timestamp as i64, 0)
                        .unwrap_or_else(|| chrono::Utc::now()),
                }
            ).await;
            ph.store(ProductionPhase::AfterBroadcast as u8, Ordering::SeqCst);
        }
    }
}

async fn run_watchdog(hb: Arc<AtomicU64>, ph: Arc<AtomicU8>) {
    info!("🐶 Watchdog: 60s check, 120s crash threshold");
    let (mut last, mut fails) = (0u64, 0u32);
    loop {
        time::sleep(Duration::from_secs(60)).await;
        let cur = hb.load(Ordering::SeqCst);
        if cur == last {
            fails += 1;
            let phase_str = ProductionPhase::from_u8(ph.load(Ordering::SeqCst));
            error!("🚨 WATCHDOG: DEAD! last={}, stall={}s, phase={}", last, 60*fails, phase_str);
            if fails >= 2 {
                error!("🚨 FATAL: DEAD 120s+, phase={}", phase_str);
                let _ = write_forensics(last, 60*fails as u64, phase_str, false);
                error!("   Crashing...");
                std::process::abort();
            }
        } else {
            if fails > 0 { info!("✅ WATCHDOG: Recovered after {} fails", fails); fails = 0; }
            last = cur;
        }
    }
}

async fn monitor_panics(
    h: tokio::task::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>>,
    hb: Arc<AtomicU64>, ph: Arc<AtomicU8>
) {
    match h.await {
        Ok(Ok(())) => { error!("🚨 FATAL: Loop exited normally"); std::process::abort(); }
        Ok(Err(e)) => { error!("🚨 FATAL: Loop error: {}", e); std::process::abort(); }
        Err(e) => {
            let (last, phase_str) = (hb.load(Ordering::SeqCst), ProductionPhase::from_u8(ph.load(Ordering::SeqCst)));
            if e.is_panic() {
                error!("🚨 FATAL: PANIC! last={}, phase={}, err={:?}", last, phase_str, e);
                let _ = write_forensics(last, 0, phase_str, true);
                std::process::abort();
            } else {
                error!("🚨 FATAL: Cancelled: {:?}", e);
                std::process::abort();
            }
        }
    }
}

fn write_forensics(hb: u64, dur: u64, phase: &str, panic: bool) -> Result<(), Box<dyn std::error::Error>> {
    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let path = format!("/tmp/stall-forensics-{}.json", ts);
    let dump = json!({"timestamp": ts, "last_heartbeat": hb, "stall_duration_seconds": dur,
        "last_phase": phase, "is_panic": panic, "use_sleep_mode": USE_SLEEP_INSTEAD_OF_INTERVAL});
    std::fs::write(&path, serde_json::to_string_pretty(&dump)?)?;
    Ok(())
}
