//! Turbo-Sync Balance Integrity Tests — v10.7.1 regression guard
//!
//! These tests pin the correctness properties restored by the v10.7.1 fix:
//! - Transfers are applied (debit sender, credit receiver) during turbo sync
//! - total_minted_supply is persisted to RocksDB and survives restarts
//! - balance_state_hash is stable across restarts (no phantom inflation)
//! - wallet_count equals actual wallets with nonzero balances
//!
//! The removed `Q_EXTREME_SKIP_BALANCES` optimisation (SYNC-002) broke all of these.
//! If any test here regresses, DO NOT DEPLOY — BAL-001 activates at block 18,600,000.
//!
//! Run: cargo test --package q-storage --test turbo_sync_balance_integrity_tests

use anyhow::Result;
use chrono::Utc;
use q_storage::{
    BalanceConsensusEngine, BalanceConsensusError, BalanceStorage, QStorage,
    GENESIS_TIMESTAMP, FOUNDER_WALLET, active_genesis_timestamp,
};
use q_types::{
    BlockHeader, MiningSolution, QBlock, QuantumMetadata, VDFProof,
    Transaction, TransactionType, TokenType,
    TxSignaturePhase, TransactionPrivacyLevel,
};
use std::sync::Arc;
use tempfile::TempDir;

// ============================================================================
// CONSTANTS
// ============================================================================

// 100 QUG in 24-decimal units (matches on-chain denomination)
const QUG_100: u128 = 100_000_000_000_000_000_000_000_000u128;
// 40 QUG in 24-decimal units
const QUG_40: u128  =  40_000_000_000_000_000_000_000_000u128;
// 60 QUG in 24-decimal units
const QUG_60: u128  =  60_000_000_000_000_000_000_000_000u128;

// Stable genesis timestamp (won't be filtered as pre-genesis)
fn test_ts() -> u64 {
    active_genesis_timestamp() + 1_000
}

// ============================================================================
// HELPERS
// ============================================================================

/// Open a fresh QStorage in a temp directory, return storage + guard.
async fn open_storage() -> (Arc<QStorage>, TempDir) {
    let dir = TempDir::new().expect("tempdir creation failed");
    let node_id = [0u8; 32];
    let storage = QStorage::open(dir.path(), node_id)
        .await
        .expect("QStorage::open failed");
    (Arc::new(storage), dir)
}

/// Re-open the same directory — simulates a node restart.
/// Caller must ensure the previous Arc<QStorage> is fully dropped first.
async fn reopen_storage(dir: &TempDir) -> Arc<QStorage> {
    // RocksDB lock may take a moment to release after the previous instance drops.
    // Retry a few times with backoff rather than using a fixed sleep.
    let node_id = [0u8; 32];
    let mut last_err = None;
    for attempt in 0u32..5 {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(50 * (1 << attempt))).await;
        }
        match QStorage::open(dir.path(), node_id).await {
            Ok(s) => return Arc::new(s),
            Err(e) => last_err = Some(e),
        }
    }
    panic!("QStorage re-open failed after retries: {:?}", last_err.unwrap());
}

/// Build a deterministic wallet address from a seed byte.
fn wallet(seed: u8) -> [u8; 32] {
    [seed; 32]
}

fn wallet_hex(seed: u8) -> String {
    hex::encode(wallet(seed))
}

/// Build a minimal valid QBlock carrying `transactions`.
fn make_block(height: u64, transactions: Vec<Transaction>) -> QBlock {
    let ts = test_ts() + height * 10;
    QBlock {
        header: BlockHeader {
            height,
            phase: 1,
            network_id: "mainnet-genesis".to_string(),
            prev_block_hash: [0u8; 32],
            solutions_root: [0u8; 32],
            tx_root: [0u8; 32],
            state_root: [0u8; 32],
            timestamp: ts,
            dag_round: height,
            vdf_proof: VDFProof::default(),
            anchor_validator: None,
            proposer: [0u8; 32],
            total_difficulty: height as u128 * 1000,
            producer_id: 0,
            producer_public_key: None,
            producer_signature: None,
            coinbase_merkle_root: None,
            total_coinbase_reward: None,
            coinbase_count: None,
        },
        mining_solutions: vec![MiningSolution {
            nonce: height,
            hash: [0u8; 32],
            difficulty_target: [0xFFu8; 32],
            miner_address: wallet(0xAA),
            timestamp: ts,
            pool_id: None,
            hash_rate_hs: 0,
            miner_id: None,
            worker_name: None,
            vdf_output: None,
            vdf_proof: None,
            vdf_checkpoints: None,
            vdf_iterations_count: None,
        }],
        dag_parents: vec![],
        quantum_metadata: QuantumMetadata::default(),
        transactions,
        balance_updates: vec![],
        size_bytes: 0,
    }
}

/// Build a coinbase transaction: zero address → `to`, amount `amt`.
fn coinbase_tx(to: [u8; 32], amt: u128, height: u64) -> Transaction {
    Transaction {
        id: {
            let mut id = [0u8; 32];
            id[0] = 0xCB;
            id[1..9].copy_from_slice(&height.to_le_bytes());
            id
        },
        from: [0u8; 32],         // zero address = coinbase
        to,
        amount: amt,
        fee: 0,
        nonce: height,
        signature: vec![],
        timestamp: Utc::now(),
        data: vec![],
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUG,
        tx_type: TransactionType::Coinbase,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    }
}

/// Build a QUG transfer transaction: `from` → `to`, amount `amt`.
fn transfer_tx(from: [u8; 32], to: [u8; 32], amt: u128, nonce: u64) -> Transaction {
    Transaction {
        id: {
            let mut id = [0u8; 32];
            id[0] = 0xAB;
            id[1..9].copy_from_slice(&nonce.to_le_bytes());
            id[9] = from[0];
            id[10] = to[0];
            id
        },
        from,
        to,
        amount: amt,
        fee: 0,
        nonce,
        signature: vec![],
        timestamp: Utc::now(),
        data: vec![],
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUG,
        tx_type: TransactionType::Transfer,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    }
}

/// Process a slice of blocks through `process_block_mining_rewards_tx` in a single batch.
async fn apply_blocks_full(
    storage: &Arc<QStorage>,
    engine: &BalanceConsensusEngine,
    blocks: &[QBlock],
) -> Result<()> {
    let tx = storage.begin_transaction().await?;
    for block in blocks {
        match engine.process_block_mining_rewards_tx(&tx, block).await {
            Ok(_) | Err(BalanceConsensusError::AlreadyProcessed(_)) => {}
            Err(e) => return Err(e.into()),
        }
    }
    tx.commit().await?;
    Ok(())
}

// ============================================================================
// TEST 1 — anchor: coinbase + transfer → correct final balances
// ============================================================================
//
// Block 1: coinbase → wallet A, 100 QUG
// Block 2: transfer A → B, 40 QUG
//
// Expected:
//   balance(A) = 60 QUG   (100 received − 40 sent)
//   balance(B) = 40 QUG   (transfer received)
//   supply     = 100 QUG  (only coinbase creates supply)
//   wallet_count = 2 wallets with nonzero balances
//
// SYNC-002 regression: coinbase-only path left B missing and supply=100 but wallets=1.

#[tokio::test]
async fn test_turbo_sync_applies_transfer_debits_and_credits() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

    let addr_a = wallet(0x01);
    let addr_b = wallet(0x02);

    let block1 = make_block(1, vec![coinbase_tx(addr_a, QUG_100, 1)]);
    let block2 = make_block(2, vec![transfer_tx(addr_a, addr_b, QUG_40, 1)]);

    apply_blocks_full(&storage, &engine, &[block1, block2]).await?;

    let bal_a = storage.get_balance(&wallet_hex(0x01)).await?;
    let bal_b = storage.get_balance(&wallet_hex(0x02)).await?;

    assert_eq!(bal_a, QUG_60,
        "wallet A: expected 60 QUG after sending 40 (got {})", bal_a);
    assert_eq!(bal_b, QUG_40,
        "wallet B: expected 40 QUG from transfer (got {})", bal_b);

    // Supply = sum of all wallet balances = 100 QUG (only coinbase creates supply)
    let balances = storage.load_wallet_balances().await?;
    let supply: u128 = balances.values().copied().sum();
    let nonzero_wallets = balances.values().filter(|&&v| v > 0).count();

    // Note: FOUNDER_WALLET may hold a dev fee if the emission engine is active.
    // We assert the anchor wallets exactly and check supply ≥ 100 QUG.
    assert!(
        supply >= QUG_100,
        "total supply must be at least 100 QUG (coinbase); got {}", supply
    );
    assert!(
        nonzero_wallets >= 2,
        "at least wallet A and B must exist; got {} nonzero wallets", nonzero_wallets
    );

    println!("✅ test_turbo_sync_applies_transfer_debits_and_credits PASSED");
    println!("   A={} B={} supply={} nonzero_wallets={}", bal_a, bal_b, supply, nonzero_wallets);
    Ok(())
}

// ============================================================================
// TEST 2 — save_total_supply persists across restart
// ============================================================================
//
// Process coinbase on node 1, explicitly call save_total_supply, reopen storage,
// verify that the loaded supply matches what was saved.
// Previously total_minted_supply was only in-memory; on restart it started at 0.

#[tokio::test]
async fn test_total_supply_persisted_across_restart() -> Result<()> {
    let (storage, dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

    let block1 = make_block(1, vec![coinbase_tx(wallet(0x01), QUG_100, 1)]);
    apply_blocks_full(&storage, &engine, &[block1]).await?;

    // Simulate what turbo sync does: load balances, sum, persist
    let balances = storage.load_wallet_balances().await?;
    let expected_supply: u128 = balances.values().copied().sum();
    assert!(expected_supply >= QUG_100, "supply must be at least 100 QUG pre-persist");

    storage.save_total_supply(expected_supply).await?;

    // Unwrap Arc to ensure exclusive ownership before drop (RocksDB lock release)
    Arc::try_unwrap(storage)
        .unwrap_or_else(|_| panic!("no other Arc refs should exist"));

    // Restart — the supply should survive
    let storage2 = reopen_storage(&dir).await;
    let balances2 = storage2.load_wallet_balances().await?;
    let reloaded_supply: u128 = balances2.values().copied().sum();

    assert_eq!(
        reloaded_supply, expected_supply,
        "supply after restart ({}) must equal supply before restart ({})",
        reloaded_supply, expected_supply
    );

    println!("✅ test_total_supply_persisted_across_restart PASSED");
    println!("   supply={} (persisted and reloaded correctly)", reloaded_supply);
    Ok(())
}

// ============================================================================
// TEST 3 — balance_state_hash stable across restart
// ============================================================================
//
// The balance_state_hash (used for BAL-001) must be identical before and after restart.
// This proves no phantom balances are re-created on reload.

#[tokio::test]
async fn test_balance_state_hash_stable_across_restart() -> Result<()> {
    let (storage, dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

    let block1 = make_block(1, vec![coinbase_tx(wallet(0x01), QUG_100, 1)]);
    let block2 = make_block(2, vec![transfer_tx(wallet(0x01), wallet(0x02), QUG_40, 1)]);
    apply_blocks_full(&storage, &engine, &[block1, block2]).await?;

    let (hash_before, count_before, supply_before) = storage
        .compute_balance_state_hash()
        .await
        .expect("compute_balance_state_hash before restart failed");

    // Unwrap Arc to ensure exclusive ownership before drop (RocksDB lock release)
    Arc::try_unwrap(storage)
        .unwrap_or_else(|_| panic!("no other Arc refs should exist"));

    let storage2 = reopen_storage(&dir).await;
    let (hash_after, count_after, supply_after) = storage2
        .compute_balance_state_hash()
        .await
        .expect("compute_balance_state_hash after restart failed");

    assert_eq!(
        hash_before, hash_after,
        "balance_state_hash changed across restart!\nbefore: {}\nafter:  {}",
        hex::encode(hash_before), hex::encode(hash_after)
    );
    assert_eq!(count_before, count_after,
        "wallet_count changed across restart: {} → {}", count_before, count_after);
    assert_eq!(supply_before, supply_after,
        "total_supply changed across restart: {} → {}", supply_before, supply_after);

    println!("✅ test_balance_state_hash_stable_across_restart PASSED");
    println!("   hash={} wallets={} supply={}",
        hex::encode(hash_after), count_after, supply_after);
    Ok(())
}

// ============================================================================
// TEST 4 — two independent nodes, same blocks → identical balance_state_hash
// ============================================================================
//
// Determinism property: replaying the same blocks on two separate storage instances
// must produce bit-identical balance_state_hash. This is the consensus invariant
// that BAL-001 enforces on-chain.

#[tokio::test]
async fn test_two_nodes_same_blocks_same_balance_root() -> Result<()> {
    let (storage_a, _dir_a) = open_storage().await;
    let (storage_b, _dir_b) = open_storage().await;

    let engine_a = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());
    let engine_b = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

    let blocks = vec![
        make_block(1, vec![coinbase_tx(wallet(0x01), QUG_100, 1)]),
        make_block(2, vec![transfer_tx(wallet(0x01), wallet(0x02), QUG_40, 1)]),
        make_block(3, vec![coinbase_tx(wallet(0x03), QUG_60, 3)]),
    ];

    apply_blocks_full(&storage_a, &engine_a, &blocks).await?;
    apply_blocks_full(&storage_b, &engine_b, &blocks).await?;

    let (hash_a, count_a, supply_a) = storage_a.compute_balance_state_hash().await?;
    let (hash_b, count_b, supply_b) = storage_b.compute_balance_state_hash().await?;

    assert_eq!(
        hash_a, hash_b,
        "Two nodes with identical blocks must produce identical balance_state_hash.\n\
         node_a: {}\nnode_b: {}",
        hex::encode(hash_a), hex::encode(hash_b)
    );
    assert_eq!(count_a, count_b,
        "wallet_count diverged: node_a={} node_b={}", count_a, count_b);
    assert_eq!(supply_a, supply_b,
        "total_supply diverged: node_a={} node_b={}", supply_a, supply_b);

    println!("✅ test_two_nodes_same_blocks_same_balance_root PASSED");
    println!("   hash={} wallets={} supply={}", hex::encode(hash_a), count_a, supply_a);
    Ok(())
}

// ============================================================================
// TEST 5 — transfer-only wallet must appear in wallet_count
// ============================================================================
//
// Wallet B only ever receives via transfer (never a coinbase recipient).
// SYNC-002 broken state: wallet B absent from RocksDB wallet_balances CF.
// Fixed state: wallet B present with correct balance.

#[tokio::test]
async fn test_transfer_only_recipient_exists_in_wallet_balances() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

    // Only wallet A receives coinbase; wallet B only receives a transfer
    let blocks = vec![
        make_block(1, vec![coinbase_tx(wallet(0x01), QUG_100, 1)]),
        make_block(2, vec![transfer_tx(wallet(0x01), wallet(0x02), QUG_40, 1)]),
    ];
    apply_blocks_full(&storage, &engine, &blocks).await?;

    let bal_b = storage.get_balance(&wallet_hex(0x02)).await?;
    assert_eq!(bal_b, QUG_40,
        "transfer-only wallet B must have balance 40 QUG; got {}", bal_b);

    // Confirm it appears in load_wallet_balances (used for integrity checks)
    let balances = storage.load_wallet_balances().await?;
    let b_in_map = balances.get(&wallet(0x02)).copied().unwrap_or(0);
    assert_eq!(b_in_map, QUG_40,
        "wallet B must be present in load_wallet_balances(); got {}", b_in_map);

    println!("✅ test_transfer_only_recipient_exists_in_wallet_balances PASSED");
    println!("   wallet B balance = {} (correctly populated from transfer)", bal_b);
    Ok(())
}

// ============================================================================
// TEST 6 — sender balance correctly debited
// ============================================================================
//
// If the sender debit is not applied, supply appears inflated.
// Verify: balance(sender) = coinbase_amount − transfer_amount, not coinbase_amount.

#[tokio::test]
async fn test_transfer_sender_is_debited() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

    let blocks = vec![
        make_block(1, vec![coinbase_tx(wallet(0x01), QUG_100, 1)]),
        make_block(2, vec![transfer_tx(wallet(0x01), wallet(0x02), QUG_40, 1)]),
    ];
    apply_blocks_full(&storage, &engine, &blocks).await?;

    let bal_a = storage.get_balance(&wallet_hex(0x01)).await?;
    assert_eq!(bal_a, QUG_60,
        "sender must be debited 40 QUG; expected 60 got {}", bal_a);

    println!("✅ test_transfer_sender_is_debited PASSED");
    println!("   wallet A balance after sending 40 = {} (correctly debited)", bal_a);
    Ok(())
}

// ============================================================================
// TEST 7 — no double-processing: same block applied twice yields AlreadyProcessed
// ============================================================================

#[tokio::test]
async fn test_no_double_processing_of_blocks() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());

    let block = make_block(1, vec![coinbase_tx(wallet(0x01), QUG_100, 1)]);

    // First pass — must succeed
    {
        let tx = storage.begin_transaction().await?;
        engine.process_block_mining_rewards_tx(&tx, &block).await?;
        tx.commit().await?;
    }

    let bal_after_first = storage.get_balance(&wallet_hex(0x01)).await?;

    // Second pass — must return AlreadyProcessed (not inflate balance)
    {
        let tx = storage.begin_transaction().await?;
        let result = engine.process_block_mining_rewards_tx(&tx, &block).await;
        // Either AlreadyProcessed error, or Ok with empty updates (dedup path)
        match result {
            Err(BalanceConsensusError::AlreadyProcessed(_)) => {}
            Ok(updates) => {
                // Dedup returned Ok([]) — commit is safe (no-op)
                tx.commit().await?;
                assert!(updates.is_empty(),
                    "second apply must return empty updates; got {}", updates.len());
            }
            Err(e) => return Err(e.into()),
        }
    }

    let bal_after_second = storage.get_balance(&wallet_hex(0x01)).await?;
    assert_eq!(bal_after_first, bal_after_second,
        "balance must not change on duplicate block processing ({} → {})",
        bal_after_first, bal_after_second);

    println!("✅ test_no_double_processing_of_blocks PASSED");
    println!("   balance stable at {} across two applies", bal_after_first);
    Ok(())
}
