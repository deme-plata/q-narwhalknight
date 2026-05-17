//! Balance Consensus Engine
//!
//! **CRITICAL MAINNET BLOCKER FIX**
//!
//! This module implements deterministic balance updates triggered by block reception,
//! ensuring ALL nodes process identical state transitions.
//!
//! ## The Problem
//!
//! Mining rewards were processed LOCALLY per node, causing balance divergence:
//! - Server Alpha (miner): 5000 QNK ✅
//! - Server Beta (bootstrap): 0 QNK ❌
//!
//! ## The Solution
//!
//! Every node processes mining rewards when receiving blocks via gossipsub:
//! 1. Block broadcast via gossipsub
//! 2. ALL nodes receive block
//! 3. BalanceConsensusEngine processes rewards deterministically
//! 4. ALL nodes update balances identically
//! 5. Result: Network-wide consensus on account state
//!
//! ## Design Principles
//!
//! - **Deterministic**: Same input → Same output on ALL nodes
//! - **Atomic**: Balance updates succeed or fail as a unit
//! - **Idempotent**: Processing same block twice is safe
//! - **Verifiable**: All nodes reach identical state

use anyhow::{anyhow, Result};
use q_types::{QBlock, MiningSolution};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace, warn};
use lru::LruCache;
use crate::emission_controller::EmissionController;

/// Genesis timestamp for reward calculation (Feb 22, 2026 12:00:00 UTC - Mainnet 2026.2)
pub const GENESIS_TIMESTAMP: u64 = 1771761600;

/// v7.3.2: Get the active genesis timestamp based on current network
/// v8.0.1: Added mainnet2026.1.3 support
pub fn active_genesis_timestamp() -> u64 {
    let network = std::env::var("Q_NETWORK_ID").unwrap_or_default();
    match network.as_str() {
        "mainnet2026.1.1" => crate::emission_controller::REHEARSAL_GENESIS_TIMESTAMP,
        "mainnet2026.1.3" => crate::emission_controller::REHEARSAL3_GENESIS_TIMESTAMP,
        _ => GENESIS_TIMESTAMP,
    }
}

// v1.4.5-beta: Use integer basis points instead of floating-point for cross-platform determinism
// 100 basis points = 1% (10_000 bps = 100%)
pub const DEV_FEE_BPS: u128 = 190; // 1.9% = 190 basis points (mainnet)
pub const BPS_DIVISOR: u128 = 10_000; // Basis points divisor for percentage calculation

/// Development fee percentage (1%) - DEPRECATED, use DEV_FEE_BPS for calculations
#[deprecated(since = "1.4.5", note = "Use DEV_FEE_BPS/BPS_DIVISOR for integer math")]
pub const DEV_FEE_PERCENT: f64 = 0.01;

/// Founder wallet address (receives 1% dev fee) - Quillon Bank Master Account
pub const FOUNDER_WALLET: &str = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

/// Balance update operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BalanceUpdate {
    /// Wallet address that received update
    pub address: String,
    /// Amount added to balance (in base units, u128 for 24 decimal precision)
    pub amount: u128,
    /// Reason for balance change
    pub reason: ChangeReason,
    /// Block height where this update occurred
    pub block_height: u64,
    /// Index of mining solution in block (if applicable)
    pub solution_index: usize,
    /// v10.2.0: Token address for non-QUG transfers (QUGUSD, custom tokens).
    /// None = QUG native transfer (update wallet_balances).
    /// Some(addr) = token transfer (update token_balances).
    pub token_address: Option<[u8; 32]>,
}

/// Reason for balance change (for auditing)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeReason {
    /// Mining reward (99% of block reward)
    MiningReward,
    /// Development fee (1% of block reward)
    DevelopmentFee,
    /// Transfer sent (debited from sender) - v3.5.14-beta
    TransferSent,
    /// Transfer received (credited to receiver) - v3.5.14-beta
    TransferReceived,
    /// Transfer failed — insufficient funds or invalid tx (shadow mode: recorded but no balance change applied)
    /// Full determinism (TxResult in block format) requires Phase 0 hard fork.
    TransferFailed,
}

/// Balance consensus error types
#[derive(Debug, thiserror::Error)]
pub enum BalanceConsensusError {
    #[error("Block {0:?} already processed (double-processing prevented)")]
    AlreadyProcessed([u8; 32]),

    #[error("Invalid mining solution in block {block_height} at index {solution_index}")]
    InvalidSolution {
        block_height: u64,
        solution_index: usize,
    },

    #[error("Zero reward calculated at block height {0} (halving complete?)")]
    ZeroReward(u64),

    #[error("Storage error: {0}")]
    Storage(#[from] anyhow::Error),

    #[error("Batch operation failed: {0}")]
    BatchOperation(String),

    #[error("Invalid timestamp: current={current}, genesis={genesis}")]
    InvalidTimestamp { current: u64, genesis: u64 },
}

/// Core balance consensus engine
///
/// Processes mining rewards deterministically across all nodes
#[derive(Debug, Clone)]
pub struct BalanceConsensusEngine {
    /// Genesis timestamp for reward calculation
    genesis_timestamp: u64,

    /// Development wallet address
    dev_wallet: String,

    /// Track processed blocks to prevent double-processing
    /// **SECURITY FIX (v0.8.0-beta)**: Now uses LRU cache with bounded memory (100k entries ≈ 5MB)
    /// - Previously: Unbounded HashMap grew forever (DoS vulnerability)
    /// - Now: LRU cache auto-evicts oldest entries when full
    processed_blocks: std::sync::Arc<RwLock<LruCache<[u8; 32], bool>>>,

    /// Maximum cache size for processed blocks
    max_cache_size: usize,

    /// Statistics (optional)
    stats: std::sync::Arc<RwLock<ConsensusStats>>,

    /// ✅ v0.9.99-beta: Adaptive block reward controller
    /// Ensures constant annual emission (2,625,000 QUG/year Era 0, halving every 4 years) regardless of throughput
    emission_controller: std::sync::Arc<RwLock<EmissionController>>,

    /// ✅ v0.9.99-beta: Cached total supply for 10,000 bps performance
    /// Reduces I/O from 10,000 queries/sec to 1 query/sec
    /// Format: (supply, last_updated) - v2.10.0: u128 for 24 decimal precision
    cached_total_supply: std::sync::Arc<RwLock<(u128, std::time::Instant)>>,

    /// v8.5.0: Persistent balance watermark — highest block height whose balance
    /// effects are already persisted in RocksDB. On restart, blocks at or below
    /// this height are skipped by all process_block_* functions. This prevents
    /// the LRU dedup cache (in-memory only, lost on restart) from allowing
    /// turbo_sync to re-credit coinbase transactions, which was the root cause
    /// of the 111x inflation bug.
    balance_processed_watermark: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

/// Consensus statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ConsensusStats {
    /// Total blocks processed
    pub blocks_processed: u64,
    /// Total balance updates applied
    pub updates_applied: u64,
    /// Total mining rewards distributed (u128 for 24 decimal precision)
    pub total_rewards: u128,
    /// Total dev fees collected (u128 for 24 decimal precision)
    pub total_dev_fees: u128,
    /// Blocks rejected (already processed)
    pub blocks_rejected: u64,
}

impl BalanceConsensusEngine {
    /// Create new balance consensus engine
    ///
    /// # Arguments
    /// * `genesis_timestamp` - Unix timestamp of network genesis (for halving)
    /// * `dev_wallet` - Development wallet address (receives 1% fee)
    pub fn new(genesis_timestamp: u64, dev_wallet: String) -> Self {
        // SECURITY FIX (v0.8.0-beta): Bounded cache size to prevent memory exhaustion
        const MAX_CACHE_SIZE: usize = 500_000;  // v8.6.0: 5x increase (was 100_000), ~25 MB memory

        info!("💰 Initializing Balance Consensus Engine");
        info!("   Genesis timestamp: {} ({})", genesis_timestamp,
              chrono::DateTime::from_timestamp(genesis_timestamp as i64, 0)
                  .map(|dt| dt.to_rfc3339())
                  .unwrap_or_else(|| "Invalid timestamp".to_string()));
        info!("   Dev wallet: {}", dev_wallet);
        info!("   Dev fee: {}% ({} bps)", DEV_FEE_BPS as f64 / 100.0, DEV_FEE_BPS);
        info!("   🛡️  Memory protection: LRU cache limited to {} entries (~25 MB)", MAX_CACHE_SIZE);
        info!("   ✅ Adaptive rewards: Emission scales with throughput for 256-year timeline");

        // ✅ v0.9.99-beta: Initialize adaptive emission controller
        let emission_controller = EmissionController::new(genesis_timestamp);

        Self {
            genesis_timestamp,
            dev_wallet,
            processed_blocks: std::sync::Arc::new(RwLock::new(
                LruCache::new(NonZeroUsize::new(MAX_CACHE_SIZE).unwrap())
            )),
            max_cache_size: MAX_CACHE_SIZE,
            stats: std::sync::Arc::new(RwLock::new(ConsensusStats::default())),
            emission_controller: std::sync::Arc::new(RwLock::new(emission_controller)),
            cached_total_supply: std::sync::Arc::new(RwLock::new((0, std::time::Instant::now()))),
            balance_processed_watermark: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Process mining rewards from a block
    ///
    /// **CRITICAL**: This MUST be called by ALL nodes when receiving a block via gossipsub
    ///
    /// # Returns
    /// Vec of balance updates that were applied
    ///
    /// # Errors
    /// - `AlreadyProcessed` if block was already processed (safe to ignore)
    /// - `InvalidSolution` if mining solution doesn't meet difficulty
    /// - `ZeroReward` if halving has made rewards negligible
    pub async fn process_block_mining_rewards(
        &self,
        storage: &dyn BalanceStorage,
        block: &QBlock,
    ) -> Result<Vec<BalanceUpdate>, BalanceConsensusError> {
        // v7.1.3: Reject pre-genesis blocks (testnet contamination)
        {
            let active_genesis = active_genesis_timestamp();
            if block.header.timestamp > 0 && block.header.timestamp < active_genesis {
                tracing::debug!("🧹 [GENESIS FILTER] Skipping pre-genesis block reward h={}", block.header.height);
                return Ok(Vec::new());
            }
        }

        // v10.2.1: REMOVED watermark early-return (was v8.5.0).
        // The watermark caused a CRITICAL race condition: batch sync advances the
        // watermark every ~15s to current tip. Locally-produced blocks (which contain
        // user transfers) would have their height <= watermark, causing ALL balance
        // processing to be skipped — transfers never applied.
        // The processed_blocks LRU hash check below is the correct dedup mechanism.

        // ✅ v1.0.75-beta: FIXED - Process EXISTING coinbase transactions from synced blocks
        //
        // Previous Issue (v0.9.77-beta Phase 7):
        // - balance_consensus was CREATING new rewards, causing DOUBLE REWARDS
        // - block_producer.rs already creates coinbase transactions
        //
        // NEW Approach (v1.0.75-beta):
        // - Do NOT create new rewards
        // - INSTEAD, process existing coinbase transactions from block.transactions
        // - This ensures synced blocks update balances correctly on all nodes

        let mut updates = Vec::new();

        // =========================================================================
        // 🔐 v10.3.2: PERSISTENT DEDUP — survives restarts (fixes balance reset bug)
        // =========================================================================
        // The LRU cache (below) is volatile — lost on restart. After restart,
        // balance_consensus reprocesses historical blocks and re-applies all
        // Swap transactions, causing balances to drop to zero.
        //
        // FIX: Check a PERSISTENT key in RocksDB FIRST. If the block was already
        // processed in a previous session, skip it. The key survives restarts.
        //
        // Key format: "processed_balance_block:{block_hash_hex}" → "1"
        // Storage: CF_MANIFEST (same as wallet_balance_, zero schema change)
        let block_hash = self.calculate_block_hash(block);
        let block_hash_hex = hex::encode(&block_hash);
        let persistent_key = format!("processed_balance_block:{}", &block_hash_hex);

        // Check persistent store FIRST (survives restart)
        match storage.get_processed_block_flag(&persistent_key).await {
            Ok(true) => {
                trace!("⏭️ Block {} already processed (PERSISTENT, height {}), skipping",
                       &block_hash_hex[..16], block.header.height);
                return Ok(Vec::new());
            }
            Ok(false) => {} // Not yet processed — continue
            Err(e) => {
                warn!("⚠️ [PERSISTENT DEDUP] Failed to check key {}: {} — falling through to LRU",
                      &persistent_key[..40], e);
                // Fall through to LRU check as backup
            }
        }

        // v7.1.3: LRU check (fast in-memory, backup for persistent check)
        {
            let mut processed = self.processed_blocks.write().await;
            if processed.contains(&block_hash) {
                debug!("⏭️ Block {} already processed (LRU, height {}), skipping",
                       &block_hash_hex[..16], block.header.height);
                return Ok(Vec::new());
            }
            processed.put(block_hash, true);
        }

        // =========================================================================
        // 🔐 v1.2.0-beta Phase 3 Step 6: Verify Coinbase Security
        // =========================================================================
        // Verify coinbase merkle root if present (Phase 3+ blocks)
        if let Err(e) = block.verify_coinbase_merkle_root() {
            warn!("🚫 [Phase 3] Block {} coinbase merkle root verification failed: {}",
                  block.header.height, e);
            return Err(BalanceConsensusError::BatchOperation(
                format!("Coinbase merkle root invalid: {}", e)
            ));
        }

        // Verify coinbase signatures if present (Phase 3+ blocks)
        match block.verify_coinbase_signatures() {
            Ok(Some(producer_key)) => {
                debug!("✅ [Phase 3] Block {} coinbase signatures verified (producer: {}...)",
                       block.header.height, hex::encode(&producer_key[..8]));
            }
            Ok(None) => {
                // Legacy block without coinbase signatures - allowed for backwards compatibility
                debug!("📋 [Legacy] Block {} has legacy coinbase transactions (no producer signature)",
                       block.header.height);
            }
            Err(e) => {
                warn!("🚫 [Phase 3] Block {} coinbase signature verification failed: {}",
                      block.header.height, e);
                return Err(BalanceConsensusError::BatchOperation(
                    format!("Coinbase signature invalid: {}", e)
                ));
            }
        }

        // Validate coinbase amounts against emission schedule
        if let Err(e) = block.validate_coinbase_amounts() {
            warn!("🚫 [Phase 3] Block {} coinbase amount validation failed: {}",
                  block.header.height, e);
            return Err(BalanceConsensusError::BatchOperation(
                format!("Coinbase amount invalid: {}", e)
            ));
        }

        // ✅ v6.2.0-beta: CRITICAL FIX - Track ALL blocks for emission rate calculation
        // Previously only locally-produced blocks were tracked (block_producer.rs:1024).
        // With N nodes each producing blocks, the emission controller only saw 1/N of
        // the network block rate, causing N× emission overshoot.
        // FIX: Track every block here (called for ALL blocks: local + P2P received).
        // This gives the emission controller the TRUE network-wide block rate.
        let has_txs = !block.transactions.is_empty();
        if let Err(e) = self.track_block_for_emission(
            block.header.height,
            block.header.timestamp,
            has_txs,
        ).await {
            warn!("⚠️ Failed to track block {} for emission: {}", block.header.height, e);
        }

        // Process existing coinbase transactions from the block
        info!("🔍 [BALANCE] Block {} has {} transactions to process",
              block.header.height, block.transactions.len());

        for (idx, block_tx) in block.transactions.iter().enumerate() {
            // v3.5.14-beta: Debug log each transaction type
            let is_coinbase_by_from = block_tx.is_coinbase();
            let is_coinbase_by_type = block_tx.tx_type.is_coinbase();
            info!("🔍 [TX {}] from={}, to={}, amount={}, is_coinbase_from={}, is_coinbase_type={}, tx_type={:?}",
                  idx,
                  q_log_privacy::mask_addr(&hex::encode(&block_tx.from[..8])),
                  q_log_privacy::mask_addr(&hex::encode(&block_tx.to[..8])),
                  q_log_privacy::mask_amt(block_tx.amount),
                  is_coinbase_by_from,
                  is_coinbase_by_type,
                  block_tx.tx_type);

            // Check if this is a coinbase transaction (mining reward)
            if is_coinbase_by_from || is_coinbase_by_type {
                let miner_address = hex::encode(&block_tx.to);
                let reward_amount = block_tx.amount;

                // Apply the mining reward to the miner's balance
                storage.add_balance(&miner_address, reward_amount).await
                    .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                // v6.2.4: Record daily emission for audit trail
                {
                    let mut controller = self.emission_controller.write().await;
                    controller.record_emission(reward_amount);
                    controller.record_daily_emission(block.header.timestamp, reward_amount);
                }

                updates.push(BalanceUpdate {
                    address: miner_address.clone(),
                    amount: reward_amount,
                    reason: ChangeReason::MiningReward,
                    block_height: block.header.height,
                    solution_index: idx,
                    token_address: None,
                });

                debug!("💰 [SYNC] Processed coinbase tx at height {}: {} → {} QUG",
                       block.header.height, &miner_address[..16], reward_amount);
            } else {
                // 📦 v3.5.14-beta: Process Transfer transactions (user P2P transactions)
                // v10.2.0: CRITICAL FIX - Check token_type to route QUGUSD/custom to token_balances
                // v10.2.4: Heavy debug logging for token routing
                let from_address = hex::encode(&block_tx.from);
                let to_address = hex::encode(&block_tx.to);
                let transfer_amount = block_tx.amount;

                // Skip if amount is 0
                if transfer_amount == 0 {
                    continue;
                }

                // v10.2.0: Determine if this is a token transfer (QUGUSD or Custom)
                let token_addr = match block_tx.token_type {
                    q_types::TokenType::QUGUSD => Some(q_types::QUGUSD_TOKEN_ADDRESS),
                    q_types::TokenType::Custom(addr) => Some(addr),
                    q_types::TokenType::QUG => None,
                };
                // v10.2.4: Heavy debug logging for token routing (primary path)
                info!(
                    "🔬 [BAL-ROUTE] height={} token_type={:?} → {} | from={} to={} amount={}",
                    block.header.height, block_tx.token_type,
                    if token_addr.is_some() { "token_balances" } else { "wallet_balances" },
                    &from_address[..std::cmp::min(from_address.len(), 16)],
                    &to_address[..std::cmp::min(to_address.len(), 16)],
                    transfer_amount
                );

                if let Some(tok_addr) = token_addr {
                    // ═══════════════════════════════════════════
                    // TOKEN TRANSFER (QUGUSD / Custom) — use token_balances CF
                    // ═══════════════════════════════════════════
                    let token_label = match block_tx.token_type {
                        q_types::TokenType::QUGUSD => "QUGUSD",
                        _ => "TOKEN",
                    };
                    match storage.subtract_token_balance(&block_tx.from, &tok_addr, transfer_amount).await {
                        Ok(_) => {
                            debug!("💸 [TOKEN TRANSFER] Debited {} {} from {} at height {}",
                                   transfer_amount, token_label, &from_address[..16], block.header.height);
                        }
                        Err(e) => {
                            error!("🚫 [TOKEN TRANSFER] INSUFFICIENT FUNDS at height {} — tx skipped (shadow: recording TransferFailed, no balance change): {} {} from {}: {}",
                                  block.header.height, transfer_amount, token_label, &from_address[..16], e);
                            updates.push(BalanceUpdate {
                                address: from_address.clone(),
                                amount: transfer_amount,
                                reason: ChangeReason::TransferFailed,
                                block_height: block.header.height,
                                solution_index: idx,
                                token_address: Some(tok_addr),
                            });
                            continue; // No balance change applied — deterministic receipt in block format is Phase 0
                        }
                    }
                    storage.add_token_balance(&block_tx.to, &tok_addr, transfer_amount).await
                        .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                    updates.push(BalanceUpdate {
                        address: from_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferSent,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: Some(tok_addr),
                    });
                    updates.push(BalanceUpdate {
                        address: to_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferReceived,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: Some(tok_addr),
                    });
                    info!("💸 [TOKEN TRANSFER] Processed {} transfer at height {}: {} → {} ({} units)",
                           token_label, block.header.height, &from_address[..16], &to_address[..16], transfer_amount);
                } else {
                    // ═══════════════════════════════════════════
                    // QUG NATIVE TRANSFER — use wallet_balances
                    // ═══════════════════════════════════════════
                    match storage.subtract_balance(&from_address, transfer_amount).await {
                        Ok(_) => {
                            debug!("💸 [TRANSFER] Debited {} from {} at height {}",
                                   transfer_amount, &from_address[..16], block.header.height);
                        }
                        Err(e) => {
                            error!("🚫 [TRANSFER] INSUFFICIENT FUNDS at height {} — tx skipped (shadow: recording TransferFailed, no balance change): {} QUG from {}: {}",
                                  block.header.height, transfer_amount, &from_address[..16], e);
                            updates.push(BalanceUpdate {
                                address: from_address.clone(),
                                amount: transfer_amount,
                                reason: ChangeReason::TransferFailed,
                                block_height: block.header.height,
                                solution_index: idx,
                                token_address: None,
                            });
                            continue; // No balance change applied — deterministic receipt in block format is Phase 0
                        }
                    }
                    storage.add_balance(&to_address, transfer_amount).await
                        .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                    updates.push(BalanceUpdate {
                        address: from_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferSent,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: None,
                    });
                    updates.push(BalanceUpdate {
                        address: to_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferReceived,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: None,
                    });
                    info!("💸 [TRANSFER] Processed transfer at height {}: {} → {} ({} QUG)",
                           block.header.height, &from_address[..16], &to_address[..16], transfer_amount);
                }
            }
        }

        // Update stats if we processed any coinbase transactions
        if !updates.is_empty() {
            let mut stats = self.stats.write().await;
            stats.blocks_processed = stats.blocks_processed.saturating_add(1);
            stats.updates_applied = stats.updates_applied.saturating_add(updates.len() as u64);
            stats.total_rewards = stats.total_rewards.saturating_add(
                updates.iter().map(|u| u.amount).sum::<u128>()
            );
        }

        // v10.3.2: Mark block as processed PERSISTENTLY (survives restart)
        // Written AFTER successful balance processing — if we crash before this,
        // the block will be reprocessed on restart (safe: idempotent adds are fine,
        // and the LRU prevents in-session reprocessing).
        if let Err(e) = storage.set_processed_block_flag(&persistent_key).await {
            warn!("⚠️ [PERSISTENT DEDUP] Failed to write key {}: {} — block will be reprocessed on restart",
                  &persistent_key[..40], e);
        }

        // Return updates
        Ok(updates)

        // COMMENTED OUT - Old double-reward code below:
        /*
        // 1. Prevent double-processing
        let block_hash = self.calculate_block_hash(block);

        {
            let processed = self.processed_blocks.read().await;
            if processed.contains(&block_hash) {
                debug!("Block {} already processed (height {})",
                       hex::encode(&block_hash[..8]), block.header.height);

                // Update stats
                let mut stats = self.stats.write().await;
                stats.blocks_rejected += 1;

                return Err(BalanceConsensusError::AlreadyProcessed(block_hash));
            }
        }

        // 2. Calculate block reward (deterministic across all nodes)
        let block_reward = self.calculate_block_reward(block.header.timestamp)?;
        */

        /*  // COMMENTED OUT - Phase 7 double-reward fix
        if block_reward == 0 {
            warn!("⚠️ Zero reward at height {} (halving complete)", block.header.height);
            return Err(BalanceConsensusError::ZeroReward(block.header.height));
        }

        // 3. Calculate dev fee split
        let dev_fee = (block_reward as f64 * DEV_FEE_PERCENT) as u64;
        let miner_reward = block_reward.saturating_sub(dev_fee);

        debug!("💰 Block {} rewards: total={}, miner={}, dev={}",
               block.header.height, block_reward, miner_reward, dev_fee);

        // 4. Process each mining solution in the block
        let mut updates = Vec::new();

        for (index, solution) in block.mining_solutions.iter().enumerate() {
            // Validate solution meets block difficulty (safety check)
            if !self.verify_solution_for_block(solution, block) {
                error!("❌ Invalid solution in block {} at index {}",
                       block.header.height, index);
                return Err(BalanceConsensusError::InvalidSolution {
                    block_height: block.header.height,
                    solution_index: index,
                });
            }

            let miner_address = hex::encode(&solution.miner_address);

            // Update miner balance
            storage.add_balance(&miner_address, miner_reward).await
                .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

            updates.push(BalanceUpdate {
                address: miner_address.clone(),
                amount: miner_reward,
                reason: ChangeReason::MiningReward,
                block_height: block.header.height,
                solution_index: index,
                token_address: None,
            });

            // Update dev wallet balance (strip "qnk" prefix to get raw hex)
            let dev_wallet_hex = self.dev_wallet.strip_prefix("qnk").unwrap_or(&self.dev_wallet);
            storage.add_balance(dev_wallet_hex, dev_fee).await
                .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

            updates.push(BalanceUpdate {
                address: self.dev_wallet.clone(),
                amount: dev_fee,
                reason: ChangeReason::DevelopmentFee,
                block_height: block.header.height,
                solution_index: index,
                token_address: None,
            });

            debug!("   ✅ Solution {}: miner={}, reward={}",
                   index, &miner_address[..16], miner_reward);
        }

        // 5. Mark block as processed (LRU cache with bounded memory)
        {
            let mut processed = self.processed_blocks.write().await;
            processed.push(block_hash, true);

            // Warn if cache is getting full (potential attack or rapid block production)
            let cache_size = processed.len();
            if cache_size >= self.max_cache_size * 90 / 100 {  // 90% full
                warn!("🧹 Balance consensus cache at {}% capacity ({}/{} entries)",
                      cache_size * 100 / self.max_cache_size, cache_size, self.max_cache_size);
                warn!("   LRU eviction active - oldest entries being removed");
            }
        }

        // 6. Update statistics (with overflow protection)
        {
            let mut stats = self.stats.write().await;

            // Use saturating arithmetic to prevent panic on overflow
            stats.blocks_processed = stats.blocks_processed.saturating_add(1);
            stats.updates_applied = stats.updates_applied.saturating_add(updates.len() as u64);

            // Safely calculate reward totals with saturating multiplication and addition
            let solutions_count = block.mining_solutions.len() as u64;
            stats.total_rewards = stats.total_rewards.saturating_add(
                miner_reward.saturating_mul(solutions_count)
            );
            stats.total_dev_fees = stats.total_dev_fees.saturating_add(
                dev_fee.saturating_mul(solutions_count)
            );

            // Warn if statistics have saturated (indicates potential attack or bug)
            if stats.total_rewards == u64::MAX || stats.total_dev_fees == u64::MAX {
                warn!("🚨 CRITICAL: Balance consensus statistics have saturated at u64::MAX");
                warn!("   This may indicate an overflow attack or excessive block processing");
            }
        }

        info!("💰 Processed {} balance updates for block {} ({} solutions)",
              updates.len(), block.header.height, block.mining_solutions.len());

        Ok(updates)
        */  // END COMMENTED OUT - Phase 7 double-reward fix
    }

    /// Process block mining rewards within a transaction (v0.8.1-beta)
    ///
    /// **SECURITY FIX (v0.8.1-beta)**: Transaction-aware version prevents
    /// CRITICAL-1 race condition where balances are updated but block is not saved.
    ///
    /// This method performs the same logic as `process_block_mining_rewards()` but
    /// operates within a QTransaction, ensuring atomicity.
    ///
    /// # Performance
    /// - Maintains sub-50ms DAG-Knight finality ✅
    /// - WriteBatch reduces overhead by ~25% vs separate writes
    ///
    /// # Arguments
    /// - `tx`: QTransaction to write balance updates to
    /// - `block`: Block to process
    ///
    /// # Returns
    /// Vec of balance updates (not yet committed - caller must commit transaction)
    pub async fn process_block_mining_rewards_tx(
        &self,
        tx: &crate::transaction::QTransaction,
        block: &QBlock,
    ) -> Result<Vec<BalanceUpdate>, BalanceConsensusError> {
        // v7.1.3: Reject pre-genesis blocks (testnet contamination)
        {
            let active_genesis = active_genesis_timestamp();
            if block.header.timestamp > 0 && block.header.timestamp < active_genesis {
                tracing::debug!("🧹 [GENESIS FILTER] Skipping pre-genesis block reward TX h={}", block.header.height);
                return Ok(Vec::new());
            }
        }

        // v10.2.1: REMOVED watermark early-return (was v8.5.0) — same race condition fix
        // as process_block_mining_rewards(). processed_blocks LRU is the correct dedup.

        let mut updates = Vec::new();

        // =========================================================================
        // 🔐 v10.3.2: PERSISTENT DEDUP (TX path) — same fix as non-TX path
        // =========================================================================
        let block_hash = self.calculate_block_hash(block);
        let block_hash_hex = hex::encode(&block_hash);
        let persistent_key = format!("processed_balance_block:{}", &block_hash_hex);

        // Check persistent store FIRST (survives restart)
        match tx.get("manifest", persistent_key.as_bytes()).await {
            Ok(Some(_)) => {
                trace!("⏭️ Block {} already processed (PERSISTENT TX, height {}), skipping",
                       &block_hash_hex[..16], block.header.height);
                return Ok(Vec::new());
            }
            Ok(None) => {} // Not yet processed
            Err(e) => {
                warn!("⚠️ [PERSISTENT DEDUP TX] Check failed: {} — falling through to LRU", e);
            }
        }

        // v7.1.3: LRU check (fast in-memory backup)
        {
            let mut processed = self.processed_blocks.write().await;
            if processed.contains(&block_hash) {
                debug!("⏭️ Block {} already processed via TX path (LRU, height {}), skipping",
                       &block_hash_hex[..16], block.header.height);
                return Ok(Vec::new());
            }
            processed.put(block_hash, true);
        }

        // v1.3.10-beta: Reduced to trace! to avoid log spam during sync
        // Empty blocks are normal in DAG-based consensus (not every block has coinbase)
        trace!(
            "[SYNC TX] Block {} has {} transactions",
            block.header.height, block.transactions.len()
        );

        // ✅ v6.2.0-beta: Track ALL blocks for emission rate (same fix as process_block_mining_rewards)
        let has_txs = !block.transactions.is_empty();
        if let Err(e) = self.track_block_for_emission(
            block.header.height,
            block.header.timestamp,
            has_txs,
        ).await {
            warn!("⚠️ Failed to track block {} for emission: {}", block.header.height, e);
        }

        // Process existing coinbase transactions from the block
        for (idx, block_tx) in block.transactions.iter().enumerate() {
            let is_coinbase_by_from = block_tx.is_coinbase();
            let is_coinbase_by_type = block_tx.tx_type.is_coinbase();

            // Check if this is a coinbase transaction (mining reward)
            if is_coinbase_by_from || is_coinbase_by_type {
                let miner_address = hex::encode(&block_tx.to);
                let reward_amount = block_tx.amount;

                // Skip if already processed (check processed_blocks cache)
                // Note: The block hash check happens at the block level, not per-tx

                // Apply the mining reward to the miner's balance
                self.add_balance_tx(tx, &miner_address, reward_amount).await
                    .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                // v8.2.9: Persistent per-wallet mining stats (blockchain-derived)
                // These are deterministic — same blocks = same stats on ANY node.
                // Stored alongside balances so mining stats survive restarts and
                // are consistent across all servers in the HA cluster.
                {
                    const CF: &str = "manifest";
                    let blocks_key = format!("mining_blocks_{}", miner_address);
                    let rewards_key = format!("mining_rewards_{}", miner_address);

                    // Read current values and increment
                    let current_blocks: u64 = tx.get(CF, blocks_key.as_bytes())
                        .await
                        .ok()
                        .flatten()
                        .and_then(|v| if v.len() == 8 { Some(u64::from_le_bytes(v[..8].try_into().unwrap())) } else { None })
                        .unwrap_or(0);
                    let current_rewards: u128 = tx.get(CF, rewards_key.as_bytes())
                        .await
                        .ok()
                        .flatten()
                        .and_then(|v| if v.len() == 16 { Some(u128::from_le_bytes(v[..16].try_into().unwrap())) } else { None })
                        .unwrap_or(0);

                    tx.put(CF, blocks_key.as_bytes(), &(current_blocks + 1).to_le_bytes())
                        .await
                        .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;
                    tx.put(CF, rewards_key.as_bytes(), &(current_rewards + reward_amount).to_le_bytes())
                        .await
                        .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;
                }

                // v6.2.4: Record daily emission for audit trail
                {
                    let mut controller = self.emission_controller.write().await;
                    controller.record_emission(reward_amount);
                    controller.record_daily_emission(block.header.timestamp, reward_amount);
                }

                updates.push(BalanceUpdate {
                    address: miner_address.clone(),
                    amount: reward_amount,
                    reason: ChangeReason::MiningReward,
                    block_height: block.header.height,
                    solution_index: idx,
                    token_address: None,
                });

                debug!("💰 [SYNC] Processed coinbase tx at height {}: {} → {} QUG",
                       block.header.height, &miner_address[..16], reward_amount);
            } else {
                // ✅ v3.5.17-beta: Process Transfer transactions in _tx version too!
                // v10.2.0: CRITICAL FIX - Check token_type to route QUGUSD/custom to token_balances
                let from_address = hex::encode(&block_tx.from);
                let to_address = hex::encode(&block_tx.to);
                let transfer_amount = block_tx.amount;

                // Skip if amount is 0
                if transfer_amount == 0 {
                    continue;
                }

                // v10.2.0: Determine if this is a token transfer (QUGUSD or Custom)
                let token_addr = match block_tx.token_type {
                    q_types::TokenType::QUGUSD => Some(q_types::QUGUSD_TOKEN_ADDRESS),
                    q_types::TokenType::Custom(addr) => Some(addr),
                    q_types::TokenType::QUG => None,
                };
                // v10.2.4: Heavy debug logging for token routing (_tx variant)
                info!(
                    "🔬 [BAL-ROUTE-TX] height={} token_type={:?} → {} | from={} to={} amount={}",
                    block.header.height, block_tx.token_type,
                    if token_addr.is_some() { "token_balances" } else { "wallet_balances" },
                    &from_address[..std::cmp::min(from_address.len(), 16)],
                    &to_address[..std::cmp::min(to_address.len(), 16)],
                    transfer_amount
                );

                if let Some(tok_addr) = token_addr {
                    // ═══════════════════════════════════════════
                    // TOKEN TRANSFER (QUGUSD / Custom) — use token_balances CF
                    // ═══════════════════════════════════════════
                    let token_label = match block_tx.token_type {
                        q_types::TokenType::QUGUSD => "QUGUSD",
                        _ => "TOKEN",
                    };
                    match self.subtract_token_balance_tx(tx, &block_tx.from, &tok_addr, transfer_amount).await {
                        Ok(_) => {
                            debug!("💸 [TOKEN TRANSFER TX] Debited {} {} from {} at height {}",
                                   transfer_amount, token_label, &from_address[..16], block.header.height);
                        }
                        Err(e) => {
                            error!("🚫 [TOKEN TRANSFER TX] INSUFFICIENT FUNDS at height {} — tx skipped (shadow: recording TransferFailed, no balance change): {} {} from {}: {}",
                                  block.header.height, transfer_amount, token_label, &from_address[..16], e);
                            updates.push(BalanceUpdate {
                                address: from_address.clone(),
                                amount: transfer_amount,
                                reason: ChangeReason::TransferFailed,
                                block_height: block.header.height,
                                solution_index: idx,
                                token_address: Some(tok_addr),
                            });
                            continue; // No balance change applied — deterministic receipt in block format is Phase 0
                        }
                    }
                    self.add_token_balance_tx(tx, &block_tx.to, &tok_addr, transfer_amount).await
                        .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                    updates.push(BalanceUpdate {
                        address: from_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferSent,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: Some(tok_addr),
                    });
                    updates.push(BalanceUpdate {
                        address: to_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferReceived,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: Some(tok_addr),
                    });
                    info!("💸 [TOKEN TRANSFER TX v10.2.0] Processed {} transfer at height {}: {} → {} ({} units)",
                           token_label, block.header.height, &from_address[..16], &to_address[..16], transfer_amount);
                } else {
                    // ═══════════════════════════════════════════
                    // QUG NATIVE TRANSFER — use wallet_balances
                    // ═══════════════════════════════════════════
                    match self.subtract_balance_tx(tx, &from_address, transfer_amount).await {
                        Ok(_) => {
                            debug!("💸 [TRANSFER TX] Debited {} from {} at height {}",
                                   transfer_amount, &from_address[..16], block.header.height);
                        }
                        Err(e) => {
                            error!("🚫 [TRANSFER TX] INSUFFICIENT FUNDS at height {} — tx skipped (shadow: recording TransferFailed, no balance change): {} QUG from {}: {}",
                                  block.header.height, transfer_amount, &from_address[..16], e);
                            updates.push(BalanceUpdate {
                                address: from_address.clone(),
                                amount: transfer_amount,
                                reason: ChangeReason::TransferFailed,
                                block_height: block.header.height,
                                solution_index: idx,
                                token_address: None,
                            });
                            continue; // No balance change applied — deterministic receipt in block format is Phase 0
                        }
                    }
                    self.add_balance_tx(tx, &to_address, transfer_amount).await
                        .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                    updates.push(BalanceUpdate {
                        address: from_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferSent,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: None,
                    });
                    updates.push(BalanceUpdate {
                        address: to_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferReceived,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: None,
                    });
                    info!("💸 [TRANSFER TX v3.5.17] Processed transfer at height {}: {} → {} ({} QUG)",
                           block.header.height, &from_address[..16], &to_address[..16], transfer_amount);
                }
            }
        }

        // Update stats if we processed any coinbase transactions
        if !updates.is_empty() {
            let mut stats = self.stats.write().await;
            stats.blocks_processed = stats.blocks_processed.saturating_add(1);
            stats.updates_applied = stats.updates_applied.saturating_add(updates.len() as u64);
            stats.total_rewards = stats.total_rewards.saturating_add(
                updates.iter().map(|u| u.amount).sum::<u128>()
            );
        }

        // v7.1.1: Mark block as processed to prevent double-crediting
        // v7.1.3: Block already marked as processed at entry (atomic check-and-set)

        // v10.3.2: Mark block as processed PERSISTENTLY (TX path)
        if let Err(e) = tx.put("manifest", persistent_key.as_bytes(), b"1").await {
            warn!("⚠️ [PERSISTENT DEDUP TX] Failed to write key: {} — block may reprocess on restart", e);
        }

        // Return updates (caller can use for SSE broadcast, logging, etc.)
        Ok(updates)

        /*  // COMMENTED OUT - Phase 7 double-reward fix
        // 1. Check if block already processed (prevent double-processing)
        let block_hash = self.calculate_block_hash(block);

        {
            let processed = self.processed_blocks.read().await;
            if processed.contains(&block_hash) {
                warn!("⚠️  Block {} already processed (hash: {})",
                      block.header.height, hex::encode(&block_hash[..8]));
                return Err(BalanceConsensusError::AlreadyProcessed(block_hash));
            }
        }

        // 2. Calculate time-based block reward
        let block_reward = self.calculate_block_reward(block.header.timestamp)?;

        if block_reward == 0 {
            warn!("⚠️ Zero reward at height {} (halving complete)", block.header.height);
            return Err(BalanceConsensusError::ZeroReward(block.header.height));
        }

        // 3. Calculate dev fee split (v1.4.5-beta: integer math for determinism)
        let dev_fee = block_reward.saturating_mul(DEV_FEE_BPS) / BPS_DIVISOR;
        let miner_reward = block_reward.saturating_sub(dev_fee);

        debug!("💰 Block {} rewards (TX): total={}, miner={}, dev={}",
               block.header.height, block_reward, miner_reward, dev_fee);

        // 4. Process each mining solution in the block
        let mut updates = Vec::new();

        for (index, solution) in block.mining_solutions.iter().enumerate() {
            // Validate solution meets block difficulty (safety check)
            if !self.verify_solution_for_block(solution, block) {
                error!("❌ Invalid solution in block {} at index {}",
                       block.header.height, index);
                return Err(BalanceConsensusError::InvalidSolution {
                    block_height: block.header.height,
                    solution_index: index,
                });
            }

            let miner_address = hex::encode(&solution.miner_address);

            // Update miner balance via transaction
            self.add_balance_tx(tx, &miner_address, miner_reward).await
                .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

            updates.push(BalanceUpdate {
                address: miner_address.clone(),
                amount: miner_reward,
                reason: ChangeReason::MiningReward,
                block_height: block.header.height,
                solution_index: index,
                token_address: None,
            });

            // Update dev wallet balance via transaction
            self.add_balance_tx(tx, &self.dev_wallet, dev_fee).await
                .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

            updates.push(BalanceUpdate {
                address: self.dev_wallet.clone(),
                amount: dev_fee,
                reason: ChangeReason::DevelopmentFee,
                block_height: block.header.height,
                solution_index: index,
                token_address: None,
            });

            debug!("   ✅ Solution {} (TX): miner={}, reward={}",
                   index, &miner_address[..16], miner_reward);
        }

        // 5. Mark block as processed (LRU cache with bounded memory)
        {
            let mut processed = self.processed_blocks.write().await;
            processed.push(block_hash, true);

            let cache_size = processed.len();
            if cache_size >= self.max_cache_size * 90 / 100 {
                warn!("🧹 Balance consensus cache at {}% capacity ({}/{} entries)",
                      cache_size * 100 / self.max_cache_size, cache_size, self.max_cache_size);
            }
        }

        // 6. Update statistics (with overflow protection)
        {
            let mut stats = self.stats.write().await;

            stats.blocks_processed = stats.blocks_processed.saturating_add(1);
            stats.updates_applied = stats.updates_applied.saturating_add(updates.len() as u64);

            let solutions_count = block.mining_solutions.len() as u64;
            stats.total_rewards = stats.total_rewards.saturating_add(
                miner_reward.saturating_mul(solutions_count)
            );
            stats.total_dev_fees = stats.total_dev_fees.saturating_add(
                dev_fee.saturating_mul(solutions_count)
            );

            if stats.total_rewards == u64::MAX || stats.total_dev_fees == u64::MAX {
                warn!("🚨 CRITICAL: Balance consensus statistics saturated at u64::MAX");
            }
        }

        // Track balance updates in transaction for logging
        for update in &updates {
            tx.track_balance_update(update.clone()).await
                .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;
        }

        info!("💰 Processed {} balance updates (TX) for block {} ({} solutions)",
              updates.len(), block.header.height, block.mining_solutions.len());

        Ok(updates)
        */  // END COMMENTED OUT - Phase 7 double-reward fix (TX variant)
    }

    /// Process ONLY coinbase (mining reward) transactions from a block
    ///
    /// v7.1.2: Lightweight version used during skip_balances sync mode.
    /// Processes mining rewards so miners get credited even during fast sync,
    /// but skips transfer transactions for speed (transfers are rebuiltlater).
    ///
    /// This ensures:
    /// 1. Mining rewards are never lost during sync
    /// 2. Emission tracking stays accurate (all blocks counted)
    /// 3. Sync speed remains high (no transfer validation overhead)
    pub async fn process_block_coinbase_only_tx(
        &self,
        tx: &crate::transaction::QTransaction,
        block: &QBlock,
    ) -> Result<Vec<BalanceUpdate>, BalanceConsensusError> {
        // v7.1.3: Reject pre-genesis blocks (testnet contamination)
        {
            let active_genesis = active_genesis_timestamp();
            if block.header.timestamp > 0 && block.header.timestamp < active_genesis {
                tracing::debug!("🧹 [GENESIS FILTER] Skipping pre-genesis coinbase h={}", block.header.height);
                return Ok(Vec::new());
            }
        }

        // v10.2.1: REMOVED watermark early-return (was v8.5.0) — same race condition fix.

        let mut updates = Vec::new();

        // =========================================================================
        // v10.3.6: PERSISTENT DEDUP — fixes balance inflation on restart
        // The identical pattern already runs in process_block_mining_rewards_tx()
        // since v10.3.2. This function was missed, causing catch-up blocks
        // after restart to be double-counted (406 QUG → 5000+).
        // Uses tx.get/tx.put (same as process_block_mining_rewards_tx line 707).
        // =========================================================================
        let block_hash = self.calculate_block_hash(block);
        let block_hash_hex = hex::encode(&block_hash);
        let persistent_key = format!("processed_balance_block:{}", &block_hash_hex);

        // Check 1: Persistent RocksDB flag via transaction (survives restart)
        match tx.get("manifest", persistent_key.as_bytes()).await {
            Ok(Some(_)) => {
                trace!("⏭️ Block {} already processed (PERSISTENT, height {}), skipping coinbase",
                       &block_hash_hex[..16], block.header.height);
                return Ok(Vec::new());
            }
            Ok(None) => {} // Not yet processed — continue
            Err(e) => {
                warn!("⚠️ [PERSISTENT DEDUP] Check failed: {} — falling through to LRU", e);
            }
        }

        // Check 2: In-memory LRU (fast backup)
        {
            let mut processed = self.processed_blocks.write().await;
            if processed.contains(&block_hash) {
                return Ok(Vec::new());
            }
            processed.put(block_hash, true);
        }

        // Track ALL blocks for emission rate calculation (critical for correct emission)
        let has_txs = !block.transactions.is_empty();
        if let Err(e) = self.track_block_for_emission(
            block.header.height,
            block.header.timestamp,
            has_txs,
        ).await {
            warn!("⚠️ Failed to track block {} for emission: {}", block.header.height, e);
        }

        // v1.0.2: Process coinbase AND transfer transactions
        // Previously skipped transfers for speed, but this caused transfer-only wallets
        // to never be created in RocksDB on fast-syncing nodes, leading to balance divergence.
        // Transfer processing is cheap (just subtract+add), the expensive part (DEX/token replay)
        // is still skipped.
        for (idx, block_tx) in block.transactions.iter().enumerate() {
            let is_coinbase = block_tx.is_coinbase() || block_tx.tx_type.is_coinbase();

            if is_coinbase {
                let miner_address = hex::encode(&block_tx.to);
                let reward_amount = block_tx.amount;

                self.add_balance_tx(tx, &miner_address, reward_amount).await
                    .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                // Record emission
                {
                    let mut controller = self.emission_controller.write().await;
                    controller.record_emission(reward_amount);
                    controller.record_daily_emission(block.header.timestamp, reward_amount);
                }

                updates.push(BalanceUpdate {
                    address: miner_address.clone(),
                    amount: reward_amount,
                    reason: ChangeReason::MiningReward,
                    block_height: block.header.height,
                    solution_index: idx,
                    token_address: None,
                });

                trace!("💰 [COINBASE-ONLY] height {}: {} → {} QUG",
                       block.header.height, &miner_address[..16.min(miner_address.len())], reward_amount);
            } else {
                // v1.0.2 / v10.2.0: Process transfer transactions with token_type awareness
                let from_address = hex::encode(&block_tx.from);
                let to_address = hex::encode(&block_tx.to);
                let transfer_amount = block_tx.amount;

                if transfer_amount == 0 {
                    continue;
                }

                // v10.2.0: Determine if this is a token transfer (QUGUSD or Custom)
                let token_addr = match block_tx.token_type {
                    q_types::TokenType::QUGUSD => Some(q_types::QUGUSD_TOKEN_ADDRESS),
                    q_types::TokenType::Custom(addr) => Some(addr),
                    q_types::TokenType::QUG => None,
                };
                // v10.2.4: Heavy debug logging for token routing (fast-sync variant)
                info!(
                    "🔬 [BAL-ROUTE-FAST] height={} token_type={:?} → {} | from={} to={} amount={}",
                    block.header.height, block_tx.token_type,
                    if token_addr.is_some() { "token_balances" } else { "wallet_balances" },
                    &from_address[..std::cmp::min(from_address.len(), 16)],
                    &to_address[..std::cmp::min(to_address.len(), 16)],
                    transfer_amount
                );

                if let Some(tok_addr) = token_addr {
                    // TOKEN TRANSFER (QUGUSD / Custom) — use token_balances CF
                    match self.subtract_token_balance_tx(tx, &block_tx.from, &tok_addr, transfer_amount).await {
                        Ok(_) => {
                            trace!("💸 [FAST-SYNC TOKEN TRANSFER] Debited {} from {} at height {}",
                                   transfer_amount, &from_address[..16.min(from_address.len())], block.header.height);
                        }
                        Err(e) => {
                            warn!("⚠️ [FAST-SYNC TOKEN TRANSFER] Failed to debit {} from {}: {}",
                                  transfer_amount, &from_address[..16.min(from_address.len())], e);
                            continue;
                        }
                    }
                    self.add_token_balance_tx(tx, &block_tx.to, &tok_addr, transfer_amount).await
                        .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                    updates.push(BalanceUpdate {
                        address: from_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferSent,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: Some(tok_addr),
                    });
                    updates.push(BalanceUpdate {
                        address: to_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferReceived,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: Some(tok_addr),
                    });
                } else {
                    // QUG NATIVE TRANSFER — use wallet_balances
                    match self.subtract_balance_tx(tx, &from_address, transfer_amount).await {
                        Ok(_) => {
                            trace!("💸 [FAST-SYNC TRANSFER] Debited {} from {} at height {}",
                                   transfer_amount, &from_address[..16.min(from_address.len())], block.header.height);
                        }
                        Err(e) => {
                            warn!("⚠️ [FAST-SYNC TRANSFER] Failed to debit {} from {}: {}",
                                  transfer_amount, &from_address[..16.min(from_address.len())], e);
                            continue;
                        }
                    }
                    self.add_balance_tx(tx, &to_address, transfer_amount).await
                        .map_err(|e| BalanceConsensusError::BatchOperation(e.to_string()))?;

                    updates.push(BalanceUpdate {
                        address: from_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferSent,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: None,
                    });
                    updates.push(BalanceUpdate {
                        address: to_address.clone(),
                        amount: transfer_amount,
                        reason: ChangeReason::TransferReceived,
                        block_height: block.header.height,
                        solution_index: idx,
                        token_address: None,
                    });
                }

                trace!("💸 [FAST-SYNC TRANSFER] height {}: {} → {} ({} QUG)",
                       block.header.height, &from_address[..16.min(from_address.len())],
                       &to_address[..16.min(to_address.len())], transfer_amount);
            }
        }

        // v10.3.6: Mark block as processed PERSISTENTLY via transaction (survives restart)
        // Written AFTER successful balance processing — if we crash before this,
        // the block will be reprocessed on restart (safe: balance writes already persisted).
        if let Err(e) = tx.put("manifest", persistent_key.as_bytes(), b"1").await {
            warn!("⚠️ [PERSISTENT DEDUP] Failed to write flag for block {} at height {}: {} — block may re-process on restart",
                  &block_hash_hex[..16], block.header.height, e);
        }

        Ok(updates)
    }

    /// Add balance within a transaction
    ///
    /// **SECURITY FIX (v0.8.1-beta)**: Helper method for transaction-aware balance updates
    /// v2.10.0: Updated to u128 for 24 decimal precision
    async fn add_balance_tx(
        &self,
        tx: &crate::transaction::QTransaction,
        address: &str,
        amount: u128,
    ) -> Result<()> {
        // ✅ v3.5.17-beta: CRITICAL FIX - Use same storage format as BalanceStorage trait
        //
        // PREVIOUS BUG: This wrote to "balances" CF with raw address bytes as key,
        // but get_balance() reads from "wallet_balance_<hex>" keys in CF_MANIFEST.
        // Mining rewards were stored in a location that was NEVER read by API endpoints!
        //
        // FIX: Use the same "manifest" CF and "wallet_balance_<hex>" key format as
        // save_wallet_balance() in lib.rs. This ensures mining rewards are visible
        // to API balance queries.
        //
        // Storage format (must match lib.rs save_wallet_balance/load_wallet_balance):
        // - CF: "manifest" (CF_MANIFEST)
        // - Key: "wallet_balance_<64-char-hex-address>"
        // - Value: u128 as 16-byte little-endian
        let key = format!("wallet_balance_{}", address);
        const CF_MANIFEST: &str = "manifest";

        // Get current balance using the correct key format
        // v2.10.0: Support both u128 (16 bytes) and legacy u64 (8 bytes)
        let current_balance = tx
            .get(CF_MANIFEST, key.as_bytes())
            .await?
            .and_then(|bytes| {
                if bytes.len() == 16 {
                    // New u128 format (little-endian, same as lib.rs)
                    Some(u128::from_le_bytes(bytes[..16].try_into().unwrap()))
                } else if bytes.len() == 8 {
                    // Legacy u64 format - convert to u128 with decimal upgrade
                    let legacy = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                    // Upgrade from 8 to 24 decimals: multiply by 10^16
                    Some((legacy as u128) * 10u128.pow(16))
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // Calculate new balance with overflow protection
        let new_balance = current_balance.saturating_add(amount);

        // v10.3.8: Balance write audit log (privacy-masked)
        tracing::debug!(
            "💰 [BALANCE] add: wallet={} delta=+{} caller=balance_consensus height=IN_TX",
            &address[..16.min(address.len())],
            q_log_privacy::mask_amt(amount)
        );

        // Write new balance to transaction using wallet_balance_ key format (little-endian)
        tx.put(CF_MANIFEST, key.as_bytes(), &new_balance.to_le_bytes()).await?;

        info!("💰 [BALANCE TX v3.5.17] {} += {} → {} (CF: manifest)",
              q_log_privacy::mask_addr(&address[..16.min(address.len())]), q_log_privacy::mask_amt(amount), q_log_privacy::mask_amt(new_balance));

        Ok(())
    }

    /// Subtract balance within a transaction
    ///
    /// v3.5.17-beta: Added for transfer processing in _tx version
    /// Uses same storage format as BalanceStorage::subtract_balance
    async fn subtract_balance_tx(
        &self,
        tx: &crate::transaction::QTransaction,
        address: &str,
        amount: u128,
    ) -> Result<()> {
        // Use same key format as add_balance_tx / BalanceStorage
        let key = format!("wallet_balance_{}", address);
        const CF_MANIFEST: &str = "manifest";

        // Get current balance
        let current_balance = tx
            .get(CF_MANIFEST, key.as_bytes())
            .await?
            .and_then(|bytes| {
                if bytes.len() == 16 {
                    Some(u128::from_le_bytes(bytes[..16].try_into().unwrap()))
                } else if bytes.len() == 8 {
                    let legacy = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                    Some((legacy as u128) * 10u128.pow(16))
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // Check sufficient balance
        if current_balance < amount {
            return Err(anyhow::anyhow!(
                "Insufficient balance: {} < {} for address {}",
                current_balance, amount, &address[..16]
            ));
        }

        // Calculate new balance
        let new_balance = current_balance.saturating_sub(amount);

        // v10.3.8: Balance write audit log (privacy-masked)
        tracing::debug!(
            "💸 [BALANCE] subtract: wallet={} delta=-{} caller=balance_consensus height=IN_TX",
            &address[..16.min(address.len())],
            q_log_privacy::mask_amt(amount)
        );

        // Write new balance
        tx.put(CF_MANIFEST, key.as_bytes(), &new_balance.to_le_bytes()).await?;

        info!("💸 [BALANCE TX v3.5.17] {} -= {} → {} (CF: manifest)",
              q_log_privacy::mask_addr(&address[..16.min(address.len())]), q_log_privacy::mask_amt(amount), q_log_privacy::mask_amt(new_balance));

        Ok(())
    }

    /// v10.2.0: Add token balance within a transaction (for QUGUSD / custom token transfers)
    ///
    /// Uses same key format as StorageEngine::save_token_balance:
    /// CF: "manifest", Key: "token_balance_{wallet_hex}_{token_hex}", Value: u128 LE
    async fn add_token_balance_tx(
        &self,
        tx: &crate::transaction::QTransaction,
        wallet: &[u8; 32],
        token: &[u8; 32],
        amount: u128,
    ) -> Result<()> {
        let key = format!("token_balance_{}_{}", hex::encode(wallet), hex::encode(token));
        const CF_MANIFEST: &str = "manifest";

        let current = tx
            .get(CF_MANIFEST, key.as_bytes())
            .await?
            .and_then(|bytes| {
                if bytes.len() == 16 {
                    Some(u128::from_le_bytes(bytes[..16].try_into().unwrap()))
                } else if bytes.len() == 8 {
                    let legacy = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                    Some((legacy as u128) * 10u128.pow(16))
                } else {
                    None
                }
            })
            .unwrap_or(0);

        let new_balance = current.saturating_add(amount);
        tx.put(CF_MANIFEST, key.as_bytes(), &new_balance.to_le_bytes()).await?;

        debug!("💰 [TOKEN BAL TX v10.2.0] {}:{} += {} → {}",
              &hex::encode(&wallet[..8]), &hex::encode(&token[..4]), amount, new_balance);
        Ok(())
    }

    /// v10.2.0: Subtract token balance within a transaction (for QUGUSD / custom token transfers)
    async fn subtract_token_balance_tx(
        &self,
        tx: &crate::transaction::QTransaction,
        wallet: &[u8; 32],
        token: &[u8; 32],
        amount: u128,
    ) -> Result<()> {
        let key = format!("token_balance_{}_{}", hex::encode(wallet), hex::encode(token));
        const CF_MANIFEST: &str = "manifest";

        let current = tx
            .get(CF_MANIFEST, key.as_bytes())
            .await?
            .and_then(|bytes| {
                if bytes.len() == 16 {
                    Some(u128::from_le_bytes(bytes[..16].try_into().unwrap()))
                } else if bytes.len() == 8 {
                    let legacy = u64::from_le_bytes(bytes[..8].try_into().unwrap());
                    Some((legacy as u128) * 10u128.pow(16))
                } else {
                    None
                }
            })
            .unwrap_or(0);

        if current < amount {
            return Err(anyhow::anyhow!(
                "Insufficient token balance: {} < {} for wallet {} token {}",
                current, amount, &hex::encode(&wallet[..8]), &hex::encode(&token[..4])
            ));
        }

        let new_balance = current.saturating_sub(amount);
        tx.put(CF_MANIFEST, key.as_bytes(), &new_balance.to_le_bytes()).await?;

        debug!("💸 [TOKEN BAL TX v10.2.0] {}:{} -= {} → {}",
              &hex::encode(&wallet[..8]), &hex::encode(&token[..4]), amount, new_balance);
        Ok(())
    }

    /// Calculate block reward using adaptive emission controller
    ///
    /// ✅ v0.9.99-beta: Adaptive Block Reward System
    ///
    /// This replaces the old FIXED reward system with ADAPTIVE rewards that scale
    /// inversely with network throughput, ensuring constant annual emission.
    ///
    /// # Formula
    /// reward_per_block = (annual_target_emission / blocks_produced_this_year)
    ///
    /// # Benefits
    /// - At 10 blocks/sec: 0.00832 QUG/block → 2,625,000 QUG/year → 256 years to 21M
    /// - At 10,000 blocks/sec: 0.00000832 QUG/block → 2,625,000 QUG/year → 256 years to 21M
    /// - Network can scale to ANY throughput without affecting emission timeline!
    ///
    /// # Arguments
    /// * `current_timestamp` - Current block timestamp
    /// * `total_supply` - Current total QUG supply (for safety cap)
    ///
    /// # Returns
    /// Adaptive reward in base units (100,000,000 base units = 1 QUG)
    ///
    /// # Errors
    /// - Returns error if emission controller calculation fails
    /// - **CRITICAL**: Caller MUST propagate this error - do NOT produce 0-reward block!
    ///
    /// # Note
    /// The emission controller tracks block rate internally using add_block() calls.
    /// Make sure to call track_block_for_emission() after adding each block.
    /// v2.10.0: Returns u128 for 24 decimal precision
    pub async fn calculate_block_reward(
        &self,
        current_timestamp: u64,
        total_supply: u128,
    ) -> Result<u128, BalanceConsensusError> {
        let mut controller = self.emission_controller.write().await;

        controller
            .calculate_block_reward(current_timestamp, total_supply)
            .map_err(|e| {
                error!("🚨 CRITICAL: EmissionController calculation failed: {}", e);
                error!("   Block production MUST abort - cannot produce block without valid reward!");
                BalanceConsensusError::Storage(e)
            })
    }

    /// Verify mining solution meets difficulty target
    ///
    /// This is a redundant safety check - blocks should already be validated
    /// before reaching consensus engine, but this protects against malicious blocks
    fn verify_solution_for_block(&self, solution: &MiningSolution, _block: &QBlock) -> bool {
        // For now, we trust that blocks have been validated by the time they reach us
        // In future, can add actual difficulty verification here
        // Note: difficulty_target is on the MiningSolution itself, not on BlockHeader
        solution.hash <= solution.difficulty_target
    }

    /// Calculate deterministic hash of block for deduplication
    ///
    /// **SECURITY FIX (v0.8.0-beta)**: Now hashes ALL block data to prevent collision attacks
    /// - Previously only hashed header, allowing blocks with different solutions to collide
    /// - Now includes mining solutions and transactions for complete uniqueness
    fn calculate_block_hash(&self, block: &QBlock) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // Hash block header
        hasher.update(&block.header.height.to_be_bytes());
        hasher.update(&block.header.timestamp.to_be_bytes());
        hasher.update(&block.header.prev_block_hash);
        hasher.update(&block.header.solutions_root);

        // CRITICAL SECURITY FIX: Hash mining solutions to prevent collision attacks
        // Without this, two blocks with same header but different solutions would have SAME hash
        // This would allow attackers to censor legitimate miners
        for solution in &block.mining_solutions {
            hasher.update(&solution.miner_address);
            hasher.update(&solution.nonce.to_be_bytes());
            hasher.update(&solution.difficulty_target);  // Already [u8; 32]
            hasher.update(&solution.timestamp.to_be_bytes());
            hasher.update(&solution.hash);  // Already [u8; 32]
        }

        // CRITICAL SECURITY FIX: Hash transactions to prevent collision attacks
        // Ensures blocks with different transaction sets cannot collide
        for tx in &block.transactions {
            // Use postcard serialization for deterministic byte representation
            if let Ok(tx_bytes) = postcard::to_allocvec(tx) {
                hasher.update(&tx_bytes);
            }
        }

        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(&hash);
        result
    }

    /// Get consensus statistics (for monitoring)
    pub async fn get_stats(&self) -> ConsensusStats {
        self.stats.read().await.clone()
    }

    /// Clear processed blocks cache (for testing or memory management)
    ///
    /// **WARNING**: Only use this in tests or if you're certain blocks won't be re-processed
    #[cfg(test)]
    pub async fn clear_processed_blocks(&self) {
        let mut processed = self.processed_blocks.write().await;
        processed.clear();
    }

    /// Rollback balance state to specific height for chain reorganization
    ///
    /// **NETWORK UNIFICATION (v0.9.37-beta Phase 3)**: Balance State Migration
    ///
    /// This clears the processed blocks cache to allow reprocessing from the fork point.
    /// Balances will be rebuilt by replaying blocks from storage after reorganization.
    ///
    /// # Safety
    /// This should only be called during chain reorganization with proper coordination.
    /// The blockchain must be rolled back BEFORE calling this to ensure consistency.
    ///
    /// # Arguments
    /// * `target_height` - Height to roll back to (blocks after this will be reprocessed)
    pub async fn rollback_to_height(&self, target_height: u64) -> anyhow::Result<()> {
        warn!("⏪ Rolling back balance consensus to height {}", target_height);

        // Clear processed blocks cache
        // This allows blocks to be reprocessed during chain reorganization
        let mut processed = self.processed_blocks.write().await;
        let original_size = processed.len();
        processed.clear();

        info!("💰 Balance consensus rollback complete:");
        info!("   Cleared {} processed blocks from cache", original_size);
        info!("   Ready to replay blocks from height {}", target_height + 1);
        info!("   Balances will be rebuilt during chain reorganization");

        Ok(())
    }

    /// Get all current balances (for debugging and verification)
    ///
    /// **NOTE**: This method requires access to storage to retrieve balances.
    /// The BalanceConsensusEngine itself doesn't store balances - they're in the database.
    /// Use storage.get_all_balances() instead when available.
    ///
    /// For now, we return an empty map as placeholder.
    /// Real implementation should query the CF_BALANCES column family.
    /// v2.10.0: Returns u128 for 24 decimal precision
    pub async fn get_all_balances(&self) -> anyhow::Result<std::collections::HashMap<String, u128>> {
        // TODO: This should query the database's CF_BALANCES column family
        // For now, return empty map
        warn!("get_all_balances() called on BalanceConsensusEngine - requires storage access");
        Ok(std::collections::HashMap::new())
    }

    /// Track block for adaptive reward calculation
    ///
    /// ✅ v0.9.99-beta: Adaptive Block Reward System
    ///
    /// This should be called after each block is added to update the emission controller's
    /// block rate tracking. The controller uses weighted averaging to calculate recent throughput.
    ///
    /// # Arguments
    /// * `height` - Block height
    /// * `timestamp` - Block timestamp
    /// * `has_transactions` - Whether block has transactions (for spam filtering)
    pub async fn track_block_for_emission(
        &self,
        height: u64,
        timestamp: u64,
        has_transactions: bool,
    ) -> anyhow::Result<()> {
        let mut controller = self.emission_controller.write().await;
        controller.add_block(height, timestamp, has_transactions);
        Ok(())
    }

    /// Get current emission statistics for monitoring
    ///
    /// ✅ v0.9.99-beta: Returns emission controller state including:
    /// - Current era (0-63)
    /// - Total emitted this era
    /// - Recent block rate
    /// - Emission phase (Bootstrap/Mature)
    pub async fn get_emission_stats(&self) -> anyhow::Result<crate::emission_controller::EmissionStats> {
        let controller = self.emission_controller.read().await;
        Ok(controller.get_stats())
    }

    /// v6.2.4: Record daily emission for a mined block
    pub async fn record_daily_emission(&self, timestamp: u64, reward_amount: u128) -> anyhow::Result<()> {
        let mut controller = self.emission_controller.write().await;
        controller.record_daily_emission(timestamp, reward_amount);
        Ok(())
    }

    /// v6.2.4: Get daily emission history
    pub async fn get_daily_emission_history(&self, days: usize) -> anyhow::Result<Vec<crate::emission_controller::DailyEmissionRecord>> {
        let controller = self.emission_controller.read().await;
        Ok(controller.get_daily_history(days))
    }

    /// v6.2.4: Get emission summary
    pub async fn get_emission_summary(&self) -> anyhow::Result<crate::emission_controller::EmissionSummary> {
        let controller = self.emission_controller.read().await;
        Ok(controller.get_emission_summary())
    }

    /// v7.0.0: Get correction factor from emission controller (for API display)
    pub async fn get_correction_factor(&self) -> f64 {
        let controller = self.emission_controller.read().await;
        controller.correction_factor()
    }

    /// v8.0.3: Get rate measurement diagnostics for ultra-advanced analytics
    pub async fn get_rate_diagnostics(&self) -> crate::emission_controller::RateDiagnostics {
        let controller = self.emission_controller.read().await;
        controller.get_rate_diagnostics()
    }

    /// v10.3.15: Get attosecond opto-physics emission diagnostics
    pub async fn get_attophysics_metrics(&self) -> crate::emission_controller::AttoPhysicsMetrics {
        let controller = self.emission_controller.read().await;
        controller.get_attophysics_metrics()
    }

    /// v8.8.4: Get mutable access to emission controller for migration sync
    pub async fn emission_controller_write(&self) -> tokio::sync::RwLockWriteGuard<'_, crate::emission_controller::EmissionController> {
        self.emission_controller.write().await
    }

    /// v7.1.0: Serialize emission controller state for persistence
    pub async fn serialize_emission_state(&self) -> anyhow::Result<Vec<u8>> {
        let controller = self.emission_controller.read().await;
        controller.serialize_state().map_err(|e| anyhow::anyhow!("Failed to serialize emission state: {}", e))
    }

    /// v7.1.0: Restore emission controller from persisted state
    pub async fn restore_emission_state(&self, bytes: &[u8]) -> anyhow::Result<()> {
        let restored = crate::emission_controller::EmissionController::restore_from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize emission state: {}", e))?;
        let mut controller = self.emission_controller.write().await;
        *controller = restored;
        // v7.4.2: Always override genesis_timestamp with current correct value
        // (persisted state may have stale genesis from before bug fix)
        controller.set_genesis_timestamp(self.genesis_timestamp);
        info!("💰 Emission controller state restored from disk");
        Ok(())
    }

    /// v8.5.0: Set the balance processed watermark.
    /// All blocks at or below this height will be skipped by process_block_* functions.
    pub fn set_balance_watermark(&self, height: u64) {
        let old = self.balance_processed_watermark.load(std::sync::atomic::Ordering::Relaxed);
        if height > old {
            self.balance_processed_watermark.store(height, std::sync::atomic::Ordering::Relaxed);
            info!("🛡️ [WATERMARK] Balance watermark set: {} → {}", old, height);
        }
    }

    /// v8.5.0: Get the current balance processed watermark.
    pub fn get_balance_watermark(&self) -> u64 {
        self.balance_processed_watermark.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get total supply with 1-second caching for 10,000 bps performance
    ///
    /// ✅ v0.9.99-beta: Performance Optimization
    ///
    /// # Performance Impact
    /// - Without cache: 10,000 disk I/O ops/sec at 10,000 bps
    /// - With cache: 1 disk I/O op/sec (>99% reduction)
    /// - Cache lifetime: 1 second (balance between accuracy and performance)
    ///
    /// # Arguments
    /// * `storage` - Storage trait to query if cache is stale
    ///
    /// # Returns
    /// Current total supply in atomic units (10^24 = 1 QUG with 24 decimals)
    /// v2.10.0: Returns u128 for 24 decimal precision
    ///
    /// # Errors
    /// Returns error if storage query fails (fail-fast pattern)
    pub async fn get_total_supply_cached(
        &self,
        storage: &dyn BalanceStorage,
    ) -> anyhow::Result<u128> {
        use std::time::Duration;

        // Check cache first (read lock)
        {
            let cache = self.cached_total_supply.read().await;
            if cache.1.elapsed() < Duration::from_secs(1) {
                debug!("📊 Total supply cache hit: {} QUG", cache.0 as f64 / 1e24);
                return Ok(cache.0);
            }
        } // Release read lock

        // Cache stale - query storage and update (write lock)
        // Note: This requires storage to have get_total_supply() method
        // For now, we'll return a placeholder that counts from balances

        warn!("📊 Total supply cache miss - querying storage (this should be rare)");

        // TODO: Implement actual storage.get_total_supply() method
        // For now, approximate from emission controller
        let controller = self.emission_controller.read().await;
        let stats = controller.get_stats();
        let supply = stats.total_emitted_this_era as u128; // Approximation

        // Update cache
        {
            let mut cache = self.cached_total_supply.write().await;
            *cache = (supply, std::time::Instant::now());
        }

        Ok(supply)
    }

    /// Get approximate total supply from emission controller
    ///
    /// This is a fast approximation that doesn't require storage access.
    /// Uses the emission controller's tracking to estimate total supply.
    /// Accurate for reward calculations but may not reflect burned tokens.
    /// v2.10.0: Returns u128 for 24 decimal precision
    pub async fn get_total_supply_approx(&self) -> anyhow::Result<u128> {
        let controller = self.emission_controller.read().await;
        // v7.1.3: Return TOTAL cumulative emission across all eras, not just this era.
        // Using only this-era emission would undercount supply after era transitions,
        // causing block rewards to exceed the remaining supply cap.
        Ok(controller.total_cumulative_emission())
    }
}

// =============================================================================
// v8.7.3: Deterministic Block State Replay
//
// After STATE_REPLAY_ACTIVATION_HEIGHT, ALL transaction types (DEX swaps,
// token creates, stablecoin mints, governance, etc.) are deterministically
// replayed from blocks using StateProcessor + StateApplicator.
//
// This enables true P2P state decentralization: every node that has the same
// blocks computes the same state. No new P2P messages needed.
//
// Coinbase and basic Transfer transactions are SKIPPED here because they are
// already handled by the balance consensus engine above.
// =============================================================================

/// Replay non-coinbase/non-transfer state changes from a block (v8.7.3)
///
/// This function processes all "rich" transaction types (DEX, tokens, stablecoin,
/// governance, AI credits, staking, contracts) through the StateProcessor pipeline,
/// producing deterministic state changes that are applied atomically to RocksDB.
///
/// # Arguments
/// * `db` - RocksDB handle (shared Arc)
/// * `block` - Block to process
///
/// # Safety
/// - Height-gated: only activates for blocks >= STATE_REPLAY_ACTIVATION_HEIGHT
/// - Fail-open: errors are logged but don't block consensus
/// - Watermark-protected: blocks at or below state_replay_watermark are skipped
///   to prevent double-credit (StateApplicator uses incremental add/sub, NOT absolute writes)
/// - Skips coinbase + transfer (already handled by balance consensus)
#[cfg(not(target_os = "windows"))]
pub fn replay_block_state_changes(
    db: &std::sync::Arc<rocksdb::DB>,
    block: &QBlock,
) {
    use q_types::{STATE_REPLAY_ACTIVATION_HEIGHT, TransactionType};
    use crate::state_processor::StateProcessor;
    use crate::state_applicator::StateApplicator;
    use crate::block_state_processor::RocksDbStateReader;

    // Height gate: only process blocks at or above activation height
    if block.header.height < STATE_REPLAY_ACTIVATION_HEIGHT {
        return;
    }

    // v8.9.1: State replay watermark — prevents double-credit from replaying
    // the same block twice. StateApplicator uses incremental balance updates
    // (current + delta), so replaying a block twice DOUBLES the effect.
    // This watermark is persisted in RocksDB and survives restarts.
    let watermark_key = b"state_replay_watermark";
    if let Some(cf) = db.cf_handle("manifest") {
        if let Ok(Some(bytes)) = db.get_cf(&cf, watermark_key) {
            if bytes.len() == 8 {
                let watermark = u64::from_le_bytes(bytes[..8].try_into().unwrap_or([0u8; 8]));
                if block.header.height <= watermark {
                    return; // Already replayed — skip to prevent double-credit
                }
            }
        }
    }

    // Skip blocks with no transactions (common in DAG consensus)
    if block.transactions.is_empty() {
        return;
    }

    // Count non-coinbase/non-transfer transactions to avoid unnecessary setup
    // v8.7.4: Also skip StableMint/StableBurn/VaultLiquidate — these are propagated in blocks
    // for record-keeping, but the actual vault state updates happen via CollateralVault
    // (synced through vault_data in StateSnapshotResponse). Processing them through
    // StateProcessor would double-credit QUGUSD to token_balances, causing balance inflation.
    let rich_tx_count = block.transactions.iter().filter(|tx| {
        if tx.is_coinbase() || tx.effective_tx_type() == TransactionType::Transfer {
            return false;
        }
        // Skip vault/stablecoin types — handled by CollateralVault path
        !matches!(
            tx.effective_tx_type(),
            TransactionType::StableMint
                | TransactionType::StableBurn
                | TransactionType::VaultLiquidate
                | TransactionType::VaultLock
                | TransactionType::VaultUnlock
        )
    }).count();

    if rich_tx_count == 0 {
        return;
    }

    // Set up the processing pipeline
    let state_reader = RocksDbStateReader::new(db.clone());
    let mut processor = StateProcessor::new(1, 1000); // gas_price=1, chain_id=1000
    processor.set_block_context(block.header.height, block.header.timestamp as i64);
    let applicator = StateApplicator::new(db.clone(), true);

    let mut applied_count = 0u32;
    let mut error_count = 0u32;

    for block_tx in &block.transactions {
        let tx_type = block_tx.effective_tx_type();

        // Skip coinbase and basic transfers — already handled by balance consensus
        if block_tx.is_coinbase() || tx_type == TransactionType::Transfer {
            continue;
        }

        // v8.7.4: Skip vault/stablecoin types — handled by CollateralVault sync path
        if matches!(
            tx_type,
            TransactionType::StableMint
                | TransactionType::StableBurn
                | TransactionType::VaultLiquidate
                | TransactionType::VaultLock
                | TransactionType::VaultUnlock
        ) {
            debug!(
                "🏦 [STATE REPLAY v8.7.4] Skipping {:?} tx at h={} (handled by vault sync)",
                tx_type, block.header.height
            );
            continue;
        }

        match processor.process_transaction(block_tx, &state_reader) {
            Ok(result) if result.error.is_none() && !result.changes.is_empty() => {
                if let Err(e) = applicator.apply_changes(&result.changes, block.header.height) {
                    warn!(
                        "⚠️ [STATE REPLAY] Failed to apply {:?} state changes at h={}: {}",
                        tx_type, block.header.height, e
                    );
                    error_count += 1;
                } else {
                    applied_count += 1;
                    debug!(
                        "🔄 [STATE REPLAY] Applied {:?} ({} changes) at h={}",
                        tx_type, result.changes.len(), block.header.height
                    );
                }
            }
            Ok(result) if result.error.is_some() => {
                debug!(
                    "⚠️ [STATE REPLAY] Tx {:?} execution error at h={}: {:?}",
                    tx_type, block.header.height, result.error
                );
                error_count += 1;
            }
            Ok(_) => {
                // No changes produced (e.g., no-op transaction) — skip silently
            }
            Err(e) => {
                warn!(
                    "⚠️ [STATE REPLAY] Tx {:?} processing error at h={}: {}",
                    tx_type, block.header.height, e
                );
                error_count += 1;
            }
        }
    }

    if applied_count > 0 {
        info!(
            "🔄 [STATE REPLAY] Block h={}: replayed {} rich txs ({} errors)",
            block.header.height, applied_count, error_count
        );
    }

    // v8.9.1: Update state replay watermark AFTER all state changes are applied.
    // Only advance the watermark (never go backwards) to maintain monotonicity.
    // This prevents double-credit if the same block is replayed after restart.
    if applied_count > 0 || error_count == 0 {
        if let Some(cf) = db.cf_handle("manifest") {
            let _ = db.put_cf(&cf, watermark_key, &block.header.height.to_le_bytes());
        }
    }
}

/// v8.9.1: After replay_block_state_changes(), extract updated token balances
/// for wallets affected by this block's "rich" transactions.
///
/// Returns a Vec of ((wallet, token), balance) pairs read from CF_TOKEN_BALANCES.
/// The caller should merge these into the in-memory token_balances HashMap.
#[cfg(not(target_os = "windows"))]
pub fn get_updated_token_balances_for_block(
    db: &std::sync::Arc<rocksdb::DB>,
    block: &QBlock,
) -> Vec<(([u8; 32], [u8; 32]), u128)> {
    use q_types::TransactionType;

    let cf = match db.cf_handle(crate::CF_TOKEN_BALANCES) {
        Some(cf) => cf,
        None => return Vec::new(),
    };

    // Collect unique wallet addresses involved in rich transactions
    let mut wallets: Vec<[u8; 32]> = Vec::new();
    let mut token_addresses: Vec<[u8; 32]> = Vec::new();
    for tx in &block.transactions {
        let tx_type = tx.effective_tx_type();
        if tx.is_coinbase() || tx_type == TransactionType::Transfer {
            continue;
        }
        // Skip vault types (handled separately)
        if matches!(tx_type,
            TransactionType::StableMint | TransactionType::StableBurn
            | TransactionType::VaultLiquidate | TransactionType::VaultLock
            | TransactionType::VaultUnlock
        ) {
            continue;
        }
        // The `to` field is typically the contract address for DEX/token txs
        if !wallets.contains(&tx.from) { wallets.push(tx.from); }
        if !wallets.contains(&tx.to) { wallets.push(tx.to); }
        if tx.to != [0u8; 32] && !token_addresses.contains(&tx.to) {
            token_addresses.push(tx.to);
        }
    }

    if wallets.is_empty() || token_addresses.is_empty() {
        return Vec::new();
    }

    // Read current balances for all (wallet, token) pairs from CF_TOKEN_BALANCES
    let mut results = Vec::new();
    for wallet in &wallets {
        for token in &token_addresses {
            let mut key = Vec::with_capacity(64);
            key.extend_from_slice(wallet);
            key.extend_from_slice(token);

            if let Ok(Some(value)) = db.get_cf(&cf, &key) {
                let balance = if value.len() >= 16 {
                    u128::from_le_bytes(value[..16].try_into().unwrap_or([0u8; 16]))
                } else if value.len() >= 8 {
                    (u64::from_le_bytes(value[..8].try_into().unwrap_or([0u8; 8])) as u128) * 10u128.pow(16)
                } else {
                    0u128
                };
                if balance > 0 {
                    results.push(((*wallet, *token), balance));
                }
            }
        }
    }

    results
}

/// No-op stub for Windows builds (RocksDB not available)
#[cfg(target_os = "windows")]
pub fn replay_block_state_changes(
    _db: &std::sync::Arc<()>,
    _block: &QBlock,
) {
    // State replay requires RocksDB — not available on Windows
}

/// Windows stub for get_updated_token_balances_for_block
#[cfg(target_os = "windows")]
pub fn get_updated_token_balances_for_block(
    _db: &std::sync::Arc<()>,
    _block: &QBlock,
) -> Vec<(([u8; 32], [u8; 32]), u128)> {
    Vec::new()
}

// =============================================================================
// v8.7.3: Historical State Migration
//
// On first startup after upgrade, replays ALL existing blocks through the
// StateProcessor to build complete DEX/token/stablecoin/governance state.
// This ensures every node that has the same blocks computes identical state,
// even for transactions that happened before this code existed.
//
// The migration is idempotent and runs once (flagged in CF_MANIFEST).
// It processes blocks sequentially (lowest to highest) so state builds up
// correctly (e.g., token must be created before it can be transferred).
// =============================================================================

/// Migration flag key in CF_MANIFEST
const STATE_REPLAY_MIGRATION_FLAG: &[u8] = b"state_replay_migration_v1_complete";

/// Replay all historical blocks through StateProcessor to build complete state.
///
/// This is called once on startup after upgrade. It iterates ALL blocks from
/// height 1 to the current tip and replays non-coinbase/non-transfer transactions
/// through StateProcessor + StateApplicator.
///
/// After completion, a flag is set in CF_MANIFEST to skip on future startups.
///
/// # Arguments
/// * `storage` - QStorage instance for block retrieval and flag management
///
/// # Returns
/// Number of blocks that had state changes applied
#[cfg(not(target_os = "windows"))]
pub async fn migrate_historical_state(
    storage: &crate::QStorage,
) -> anyhow::Result<u64> {
    use q_types::TransactionType;
    use crate::state_processor::StateProcessor;
    use crate::state_applicator::StateApplicator;
    use crate::block_state_processor::RocksDbStateReader;

    // Check if migration already completed
    if storage.has_migration_flag(STATE_REPLAY_MIGRATION_FLAG).await {
        info!("✅ [STATE MIGRATION] Historical state replay already completed — skipping");
        return Ok(0);
    }

    // Get the RocksDB handle
    let db = match storage.get_rocks_db_handle() {
        Some(db) => db,
        None => {
            warn!("⚠️ [STATE MIGRATION] No RocksDB handle available — skipping migration");
            return Ok(0);
        }
    };

    // Get current chain tip
    let tip_height = storage
        .get_latest_qblock_height()
        .await?
        .unwrap_or(0);

    if tip_height == 0 {
        info!("✅ [STATE MIGRATION] No blocks in database — nothing to migrate");
        storage.set_migration_flag(STATE_REPLAY_MIGRATION_FLAG).await;
        return Ok(0);
    }

    info!("🔄 [STATE MIGRATION] Starting historical state replay: {} blocks to process", tip_height);
    info!("   This rebuilds DEX pools, token balances, stablecoin vaults, and all other state");
    info!("   from the block history. This is a ONE-TIME operation.");

    let state_reader = RocksDbStateReader::new(db.clone());
    let mut processor = StateProcessor::new(1, 1000); // gas_price=1, chain_id=1000
    let applicator = StateApplicator::new(db.clone(), true);

    let mut blocks_with_state = 0u64;
    let mut total_changes = 0u64;
    let mut total_errors = 0u64;
    let start_time = std::time::Instant::now();
    let mut last_log_time = std::time::Instant::now();

    // Process blocks sequentially from genesis to tip
    // Use batched reads for performance (100 blocks at a time)
    let batch_size = 100usize;
    let mut height = 1u64;

    while height <= tip_height {
        // Read batch of blocks using get_qblocks_range(start_height, limit)
        let blocks = storage.get_qblocks_range(height, batch_size).await.unwrap_or_default();
        let batch_end = if blocks.is_empty() {
            // No blocks found — skip ahead
            height + batch_size as u64 - 1
        } else {
            // Use highest block in batch
            blocks.last().map(|b| b.header.height).unwrap_or(height + batch_size as u64 - 1)
        };

        for block in &blocks {
            // Skip blocks with no transactions
            if block.transactions.is_empty() {
                continue;
            }

            // Count non-coinbase/non-transfer transactions
            let rich_txs: Vec<_> = block.transactions.iter().filter(|tx| {
                !tx.is_coinbase() && tx.effective_tx_type() != TransactionType::Transfer
            }).collect();

            if rich_txs.is_empty() {
                continue;
            }

            // Set block context for the processor
            processor.set_block_context(block.header.height, block.header.timestamp as i64);

            let mut block_changes = 0u32;
            for block_tx in &rich_txs {
                match processor.process_transaction(block_tx, &state_reader) {
                    Ok(result) if result.error.is_none() && !result.changes.is_empty() => {
                        match applicator.apply_changes(&result.changes, block.header.height) {
                            Ok(_) => {
                                block_changes += result.changes.len() as u32;
                            }
                            Err(e) => {
                                // Don't fail the migration — log and continue
                                trace!(
                                    "⚠️ [STATE MIGRATION] Apply error at h={} {:?}: {}",
                                    block.header.height, block_tx.effective_tx_type(), e
                                );
                                total_errors += 1;
                            }
                        }
                    }
                    Ok(result) if result.error.is_some() => {
                        // Transaction execution error (e.g., insufficient balance) — expected for some txs
                        trace!(
                            "[STATE MIGRATION] Tx error at h={}: {:?}",
                            block.header.height, result.error
                        );
                        total_errors += 1;
                    }
                    Err(e) => {
                        trace!(
                            "⚠️ [STATE MIGRATION] Process error at h={}: {}",
                            block.header.height, e
                        );
                        total_errors += 1;
                    }
                    _ => {} // No changes produced
                }
            }

            if block_changes > 0 {
                blocks_with_state += 1;
                total_changes += block_changes as u64;
            }
        }

        height = batch_end + 1;

        // Progress logging every 5 seconds
        if last_log_time.elapsed().as_secs() >= 5 {
            let progress_pct = (height as f64 / tip_height as f64 * 100.0).min(100.0);
            let elapsed = start_time.elapsed();
            let blocks_per_sec = height as f64 / elapsed.as_secs_f64();
            let eta_secs = if blocks_per_sec > 0.0 {
                ((tip_height - height) as f64 / blocks_per_sec) as u64
            } else {
                0
            };
            info!(
                "🔄 [STATE MIGRATION] {:.1}% — h={}/{} — {} state blocks, {} changes, {} errors — {:.0} blocks/s — ETA {}s",
                progress_pct, height, tip_height, blocks_with_state, total_changes, total_errors,
                blocks_per_sec, eta_secs
            );
            last_log_time = std::time::Instant::now();
        }
    }

    let elapsed = start_time.elapsed();

    // Set migration flag so this doesn't run again
    storage.set_migration_flag(STATE_REPLAY_MIGRATION_FLAG).await;

    info!("✅ [STATE MIGRATION] Historical state replay complete!");
    info!("   Blocks processed: {}", tip_height);
    info!("   Blocks with state changes: {}", blocks_with_state);
    info!("   Total state changes applied: {}", total_changes);
    info!("   Errors (non-fatal): {}", total_errors);
    info!("   Duration: {:.1}s ({:.0} blocks/s)", elapsed.as_secs_f64(), tip_height as f64 / elapsed.as_secs_f64());

    Ok(blocks_with_state)
}

/// Windows stub
#[cfg(target_os = "windows")]
pub async fn migrate_historical_state(
    _storage: &crate::QStorage,
) -> anyhow::Result<u64> {
    Ok(0)
}

/// Storage trait for balance updates
///
/// This abstracts the actual storage implementation (RocksDB, etc.)
/// v2.5.0: Updated to u128 for extreme precision (24 decimals)
#[async_trait::async_trait]
pub trait BalanceStorage: Send + Sync {
    /// Add amount to wallet balance (atomic operation)
    /// v2.5.0: amount is now u128
    async fn add_balance(&self, address: &str, amount: u128) -> Result<()>;

    /// Subtract amount from wallet balance (atomic operation) - v3.5.14-beta
    /// Returns error if insufficient balance
    async fn subtract_balance(&self, address: &str, amount: u128) -> Result<()>;

    /// Get current balance for wallet
    /// v2.5.0: returns u128
    async fn get_balance(&self, address: &str) -> Result<u128>;

    /// Set balance for wallet (used in tests)
    /// v2.5.0: balance is now u128
    async fn set_balance(&self, address: &str, balance: u128) -> Result<()>;

    /// v10.2.0: Add amount to token balance (QUGUSD / custom tokens)
    /// Default implementation is no-op (for backwards compatibility with MockStorage in tests)
    async fn add_token_balance(&self, _wallet: &[u8; 32], _token: &[u8; 32], _amount: u128) -> Result<()> {
        Ok(())
    }

    /// v10.2.0: Subtract amount from token balance (QUGUSD / custom tokens)
    async fn subtract_token_balance(&self, _wallet: &[u8; 32], _token: &[u8; 32], _amount: u128) -> Result<()> {
        Ok(())
    }

    /// v10.3.2: Check if a block has been processed (persistent dedup, survives restart)
    async fn get_processed_block_flag(&self, _key: &str) -> Result<bool> {
        Ok(false) // Default: not processed (for test MockStorage compatibility)
    }

    /// v10.3.2: Mark a block as processed (persistent dedup, survives restart)
    async fn set_processed_block_flag(&self, _key: &str) -> Result<()> {
        Ok(()) // Default: no-op (for test MockStorage compatibility)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    /// Mock storage for testing (v2.10.0: u128 for 24 decimal precision)
    struct MockStorage {
        balances: Arc<RwLock<HashMap<String, u128>>>,
    }

    impl MockStorage {
        fn new() -> Self {
            Self {
                balances: Arc::new(RwLock::new(HashMap::new())),
            }
        }

        async fn get_all_balances(&self) -> HashMap<String, u128> {
            self.balances.read().await.clone()
        }
    }

    #[async_trait::async_trait]
    impl BalanceStorage for MockStorage {
        async fn add_balance(&self, address: &str, amount: u128) -> Result<()> {
            let mut balances = self.balances.write().await;
            *balances.entry(address.to_string()).or_insert(0) += amount;
            Ok(())
        }

        async fn subtract_balance(&self, address: &str, amount: u128) -> Result<()> {
            let mut balances = self.balances.write().await;
            let current = *balances.get(address).unwrap_or(&0);
            if current < amount {
                return Err(anyhow!("Insufficient balance: {} < {}", current, amount));
            }
            balances.insert(address.to_string(), current - amount);
            Ok(())
        }

        async fn get_balance(&self, address: &str) -> Result<u128> {
            let balances = self.balances.read().await;
            Ok(*balances.get(address).unwrap_or(&0))
        }

        async fn set_balance(&self, address: &str, balance: u128) -> Result<()> {
            let mut balances = self.balances.write().await;
            balances.insert(address.to_string(), balance);
            Ok(())
        }
    }

    fn create_test_block(height: u64, timestamp: u64, num_solutions: usize) -> QBlock {
        let solutions: Vec<MiningSolution> = (0..num_solutions)
            .map(|i| MiningSolution {
                nonce: i as u64,
                hash: [0u8; 32],
                difficulty_target: [0xFF; 32],
                miner_address: [i as u8; 32],
                timestamp,
                pool_id: None,
                hash_rate_hs: 10000 + (i as u64 * 1000),
                miner_id: None,
                worker_name: None,
                vdf_output: None,
                vdf_proof: None,
                vdf_checkpoints: None,
                vdf_iterations_count: None,
            })
            .collect();

        QBlock {
            header: q_types::BlockHeader {
                height,
                timestamp,
                prev_hash: [0u8; 32],
                merkle_root: [0u8; 32],
                difficulty_target: 1000,
                nonce: 0,
            },
            mining_solutions: solutions,
            transactions: vec![],
        }
    }

    #[tokio::test]
    async fn test_balance_consensus_determinism() {
        // Two nodes processing same block should get identical results
        let storage1 = MockStorage::new();
        let storage2 = MockStorage::new();

        let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());
        let block = create_test_block(1, GENESIS_TIMESTAMP + 100, 1);

        let updates1 = engine.process_block_mining_rewards(&storage1, &block).await.unwrap();

        // Clear processed blocks to allow re-processing
        engine.clear_processed_blocks().await;

        let updates2 = engine.process_block_mining_rewards(&storage2, &block).await.unwrap();

        // Updates should be identical
        assert_eq!(updates1, updates2);

        // Balances should match
        let balances1 = storage1.get_all_balances().await;
        let balances2 = storage2.get_all_balances().await;
        assert_eq!(balances1, balances2);
    }

    #[tokio::test]
    async fn test_double_processing_prevention() {
        let storage = MockStorage::new();
        let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());
        let block = create_test_block(1, GENESIS_TIMESTAMP + 100, 1);

        // First processing should succeed
        let result1 = engine.process_block_mining_rewards(&storage, &block).await;
        assert!(result1.is_ok());

        // Second processing should fail
        let result2 = engine.process_block_mining_rewards(&storage, &block).await;
        assert!(matches!(result2, Err(BalanceConsensusError::AlreadyProcessed(_))));
    }

    #[tokio::test]
    async fn test_dev_fee_split() {
        let storage = MockStorage::new();
        let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());
        let block = create_test_block(1, GENESIS_TIMESTAMP + 100, 1);

        let updates = engine.process_block_mining_rewards(&storage, &block).await.unwrap();

        // Should have 2 updates: 1 miner + 1 dev
        assert_eq!(updates.len(), 2);

        let miner_update = updates.iter().find(|u| u.reason == ChangeReason::MiningReward).unwrap();
        let dev_update = updates.iter().find(|u| u.reason == ChangeReason::DevelopmentFee).unwrap();

        // Dev fee should be 1% of total (v1.4.5-beta: integer math)
        let total = miner_update.amount + dev_update.amount;
        // expected_dev_fee = total * DEV_FEE_BPS / (BPS_DIVISOR - DEV_FEE_BPS)
        let expected_dev_fee = total.saturating_mul(DEV_FEE_BPS) / (BPS_DIVISOR - DEV_FEE_BPS);

        // Allow small rounding error
        assert!((dev_update.amount as i64 - expected_dev_fee as i64).abs() <= 1);
    }

    #[tokio::test]
    async fn test_reward_calculation() {
        let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

        // Year 0: Full reward
        let reward_y0 = engine.calculate_block_reward(GENESIS_TIMESTAMP + 100).unwrap();
        assert_eq!(reward_y0, 100_000);

        // Year 1: Half reward
        let reward_y1 = engine.calculate_block_reward(GENESIS_TIMESTAMP + 31_536_000 + 100).unwrap();
        assert_eq!(reward_y1, 50_000);

        // Year 2: Quarter reward
        let reward_y2 = engine.calculate_block_reward(GENESIS_TIMESTAMP + 2 * 31_536_000 + 100).unwrap();
        assert_eq!(reward_y2, 25_000);

        // Year 64+: Zero reward
        let reward_y64 = engine.calculate_block_reward(GENESIS_TIMESTAMP + 64 * 31_536_000 + 100).unwrap();
        assert_eq!(reward_y64, 0);
    }

    #[tokio::test]
    async fn test_invalid_timestamp() {
        let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

        // Timestamp before genesis should fail
        let result = engine.calculate_block_reward(GENESIS_TIMESTAMP - 1000);
        assert!(matches!(result, Err(BalanceConsensusError::InvalidTimestamp { .. })));
    }
}
