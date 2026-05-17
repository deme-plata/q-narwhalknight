//! Block Production Loop v1.0.3-beta - Comprehensive Stall Protection
//!
//! v3.4.3-beta: CRITICAL FIX - Added balance processing after block save
//! Previously, blocks saved here bypassed balance_consensus entirely!
//! This caused P2P transactions to confirm but balances never update.

use std::sync::{atomic::{AtomicU64, AtomicU8, Ordering}, Arc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::time::{self, MissedTickBehavior};
use tracing::{debug, error, info, warn};
use serde_json::json;
use crate::AppState;

/// Display divisor for QUG amounts (1 QUG = 1e24 base units, 24 decimal precision)
/// v3.6.1-beta: CRITICAL FIX - was incorrectly 1e9, causing balance display to be 1e15x too high!
const QUG_DISPLAY_DIVISOR: f64 = 1e24;

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

    // 🦈 SharkGod: Clone the wake notify for the select! loop
    let sharkgod_wake = app.sharkgod_block_wake.clone();

    loop {
        ph.store(ProductionPhase::BeforeTick as u8, Ordering::SeqCst);

        // 🦈 SharkGod: Listen for both normal tick AND SharkGod wake signal
        // If SharkGod fires, we skip the interval wait and produce immediately
        let _sharkgod_triggered = if let Some(ref wake) = sharkgod_wake {
            tokio::select! {
                _ = async {
                    if let Some(ref mut i) = interval { i.tick().await; } else { time::sleep(Duration::from_secs(1)).await; }
                } => false,
                _ = wake.notified() => {
                    info!("🦈 [SHARKGOD] Block producer WOKEN — producing block immediately");
                    true
                }
            }
        } else {
            if let Some(ref mut i) = interval { i.tick().await; } else { time::sleep(Duration::from_secs(1)).await; }
            false
        };

        ph.store(ProductionPhase::AfterTick as u8, Ordering::SeqCst);

        hb.fetch_add(1, Ordering::SeqCst);
        iter += 1;

        // 🚨 v3.3.3-beta: EMERGENCY PAUSE CHECK - Skip block production if paused
        if app.emergency_paused.load(Ordering::SeqCst) {
            if iter % 60 == 0 {
                warn!("🚨 [EMERGENCY PAUSE] Block production HALTED - System is in emergency pause mode");
            }
            continue;
        }

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

        // v7.1.4: MINING QUEUE DRAIN FIX
        // Previous threshold (gap > 3) was too strict — Beta is consistently 4-6 blocks behind
        // due to P2P block arrival timing, causing production to happen only ~1/min instead of
        // every 2 seconds. Combined with max_solutions_per_block=100, this created a massive
        // solution queue backlog where new miners' rewards were delayed by HOURS.
        //
        // New thresholds:
        // - gap > 1000: Emergency brake - node is severely behind, don't produce
        // - gap > 50: Moderate lag - pause production to prioritize sync
        // - gap <= 50: Normal — produce blocks to drain the solution queue
        //
        // v7.3.7: Bug #29 fix — ignore peer heights BEFORE genesis timestamp.
        // A rogue pre-launch node can announce height 311,755 on mainnet2026.2 before Feb 22,
        // causing gap > 1000 and disabling production permanently on canary nodes.
        // If we haven't reached genesis yet, peer heights are meaningless — skip the gap check.
        let pre_genesis = {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
            let genesis_ts = q_storage::balance_consensus::active_genesis_timestamp();
            now < genesis_ts
        };
        if !pre_genesis && net_h > 0 {
            let gap = net_h.saturating_sub(cur_h);
            if gap > 1000 { if iter % 30 == 0 { warn!("🚫 Production DISABLED: {} behind (local={}, net={})", gap, cur_h, net_h); } continue; }
            if gap > 50 { if iter % 30 == 0 { debug!("⏸️  Syncing {} behind (threshold=50)", gap); } continue; }
        } else if pre_genesis && net_h > 0 {
            let gap = net_h.saturating_sub(cur_h);
            if gap > 1000 {
                if iter % 30 == 0 {
                    let genesis_ts = q_storage::balance_consensus::active_genesis_timestamp();
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
                    warn!("🕐 [PRE-GENESIS] Ignoring rogue peer height {} (gap={}) — genesis in {}s",
                          net_h, gap, genesis_ts.saturating_sub(now));
                }
            }
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

                    // 🚨 v3.4.3-beta: CRITICAL FIX - Process balance updates after block save!
                    // Previously this entire code path skipped balance_consensus, causing
                    // P2P transactions to confirm but balances never update.
                    // See: Block 1648786 bug - transaction 69a6934385d82c47 confirmed but funds never arrived

                    // 1. Process via balance_consensus for RocksDB persistence
                    // Note: signature is (storage, block) - use &* to deref Arc
                    match app.balance_consensus_engine.process_block_mining_rewards(
                        &*app.storage_engine,
                        blk
                    ).await {
                        Ok(updates) => {
                            debug!("✅ [BLOCK_PROD_V2] Balance consensus processed block {} ({} updates)",
                                   blk.header.height, updates.len());
                        }
                        Err(q_storage::BalanceConsensusError::AlreadyProcessed(_)) => {
                            debug!("[BLOCK_PROD_V2] Block {} already processed (safe)", blk.header.height);
                        }
                        Err(e) => {
                            error!("❌ [BLOCK_PROD_V2] Balance consensus failed for block {}: {:?}",
                                   blk.header.height, e);
                        }
                    }

                    // 2. Update in-memory wallet_balances for BOTH coinbase AND transfers
                    let balance_updates = {
                        let mut balances = app.wallet_balances.write().await;
                        let mut updates = Vec::new();

                        for tx in &blk.transactions {
                            if tx.from == [0u8; 32] {
                                // Coinbase transaction - credit recipient
                                let current = balances.get(&tx.to).copied().unwrap_or(0);
                                let new_balance = current + tx.amount;
                                balances.insert(tx.to, new_balance);

                                info!("💰 [BLOCK_PROD_V2] Coinbase: {} → {} (balance: {} → {})",
                                      q_log_privacy::mask_amt_display(tx.amount as f64 / QUG_DISPLAY_DIVISOR),
                                      q_log_privacy::mask_addr(&hex::encode(&tx.to[..8])),
                                      q_log_privacy::mask_amt_display(current as f64 / QUG_DISPLAY_DIVISOR),
                                      q_log_privacy::mask_amt_display(new_balance as f64 / QUG_DISPLAY_DIVISOR));

                                updates.push((tx.to, current, new_balance, "coinbase".to_string()));
                            } else {
                                // Transfer transaction - debit sender, credit receiver
                                let sender_current = balances.get(&tx.from).copied().unwrap_or(0);
                                let sender_new = sender_current.saturating_sub(tx.amount);
                                balances.insert(tx.from, sender_new);

                                let receiver_current = balances.get(&tx.to).copied().unwrap_or(0);
                                let receiver_new = receiver_current.saturating_add(tx.amount);
                                balances.insert(tx.to, receiver_new);

                                info!("🔄 [BLOCK_PROD_V2] Transfer: {} {} → {} (sender: {} → {}, receiver: {} → {})",
                                      q_log_privacy::mask_amt_display(tx.amount as f64 / QUG_DISPLAY_DIVISOR),
                                      q_log_privacy::mask_addr(&hex::encode(&tx.from[..8])),
                                      q_log_privacy::mask_addr(&hex::encode(&tx.to[..8])),
                                      q_log_privacy::mask_amt_display(sender_current as f64 / QUG_DISPLAY_DIVISOR),
                                      q_log_privacy::mask_amt_display(sender_new as f64 / QUG_DISPLAY_DIVISOR),
                                      q_log_privacy::mask_amt_display(receiver_current as f64 / QUG_DISPLAY_DIVISOR),
                                      q_log_privacy::mask_amt_display(receiver_new as f64 / QUG_DISPLAY_DIVISOR));

                                updates.push((tx.from, sender_current, sender_new, "transfer_sent".to_string()));
                                updates.push((tx.to, receiver_current, receiver_new, "transfer_received".to_string()));
                            }
                        }
                        updates
                    };

                    // 3. Broadcast SSE balance update events
                    for (wallet_addr, old_balance, new_balance, mut change_reason) in balance_updates {
                        let wallet_addr_hex = hex::encode(wallet_addr);

                        // Dev fee wallet gets special label
                        const MASTER_ACCOUNT_HEX: &str =
                            "efca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";
                        if wallet_addr_hex == MASTER_ACCOUNT_HEX && change_reason == "coinbase" {
                            change_reason = "DevelopmentFee".to_string();
                        } else if change_reason == "coinbase" {
                            change_reason = "mining_reward".to_string();
                        }

                        let wallet_with_prefix = format!("qnk{}", wallet_addr_hex);
                        let balance_event = crate::streaming::StreamEvent::BalanceUpdated {
                            wallet_address: wallet_with_prefix,
                            old_balance: old_balance as f64 / QUG_DISPLAY_DIVISOR,
                            new_balance: new_balance as f64 / QUG_DISPLAY_DIVISOR,
                            change_reason,
                            timestamp: chrono::Utc::now(),
                            block_hash: Some(hex::encode(blk.calculate_hash())),
                            block_height: Some(blk.header.height),
                            confirmation_status: "confirmed".to_string(),
                            from_address: None,
                            tx_hash: None,
                            memo: None,
                        };

                        if let Err(e) = app.event_broadcaster.broadcast(balance_event).await {
                            warn!("[BLOCK_PROD_V2] Failed to broadcast balance SSE for {}: {}",
                                  &wallet_addr_hex[..16], e);
                        }
                    }

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

            // v7.1.1: Advance producer height after block save.
            // Without this, the v2 loop never advances the producer's internal height,
            // which breaks should_produce_block() and last_produced_height tracking.
            if let Err(e) = app.block_producer_pool.sync_from_storage(&app.storage_engine).await {
                error!("❌ [BLOCK_PROD_V2] Failed to sync producers after block save: {}", e);
            }

            // Update current_height_atomic so mining API and sync logic have accurate height
            let cur = app.current_height_atomic.load(Ordering::SeqCst);
            if blk.header.height > cur {
                app.current_height_atomic.store(blk.header.height, Ordering::SeqCst);
            }

            // v7.1.4 FIX: Also update node_status.current_height for Explorer/status API
            // Previously this was only updated by sync code (gossipsub block reception),
            // so bootstrap nodes that produce blocks locally showed "current height: 0"
            // in the Explorer UI and status API.
            {
                let mut status = app.node_status.write().await;
                if blk.header.height > status.current_height {
                    status.current_height = blk.header.height;
                }
            }

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
