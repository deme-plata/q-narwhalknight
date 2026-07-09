//! Phase 0 DS-1 gap regression tests — 2026-07-08
//!
//! Pins two historical-incident invariants against the LIVE authoritative
//! balance path (`BalanceConsensusEngine::process_block_mining_rewards_tx`,
//! the same function `q-api-server`'s live mining pipeline calls via
//! `block_producer.rs:2188` → `get_transactions_for_block` → block apply):
//!
//!   1. Self-mix (tx.from == tx.to) must net to exactly zero balance change,
//!      not double-credit and not zero-out the wallet. The live `_tx` path
//!      debits first (`subtract_balance_tx`) then credits
//!      (`add_balance_tx`), both against the RocksDB-backed `QTransaction`
//!      (sequential await, no up-front balance snapshot), so this is
//!      currently safe — this test is a PIN, not a new-bug repro. It exists
//!      so a future "read all balances once up front" batching refactor
//!      cannot silently reintroduce the old "old_recipient snapshotted
//!      before debit" clobber shape without a red test.
//!
//!   2. The v10.11.18 max-wins-drops-a-debit fix: `BalanceStorage::subtract_balance`
//!      (on `QStorage`) must actually apply a debit (uses
//!      `save_wallet_balance_authoritative`, which bypasses the max-wins
//!      guard), while the underlying `QStorage::save_wallet_balance` must
//!      STILL refuse to silently lower a balance when called directly
//!      (max-wins is still enforced there — this is the intentional
//!      asymmetry documented at lib.rs:9956-9963). Both directions are
//!      pinned so a future "consolidate these two balance-write paths"
//!      refactor cannot quietly restore the old free-money bug.
//!
//! Uses the same fixture helpers/conventions as
//! sync_integrity_regression_tests.rs (open_storage/make_block/transfer_tx/
//! wallet) so this file stays consistent with the existing regression-test
//! style in this crate.
//!
//! Run: cargo test --package q-storage --test phase0_ds1_gap_regression_tests

use anyhow::Result;
use q_storage::{
    BalanceConsensusEngine, BalanceConsensusError, BalanceStorage, QStorage,
    active_genesis_timestamp,
};
use q_types::{
    BlockHeader, MiningSolution, QBlock, QuantumMetadata, VDFProof,
    Transaction, TransactionType, TokenType,
    TxSignaturePhase, TransactionPrivacyLevel,
};
use std::sync::Arc;
use tempfile::TempDir;

// Stable genesis timestamp so blocks pass the pre-genesis filter.
fn test_ts() -> u64 { active_genesis_timestamp() + 1_000 }

const QUG_100: u128 = 100_000_000_000_000_000_000_000_000u128;
const QUG_30: u128 = 30_000_000_000_000_000_000_000_000u128;
const QUG_70: u128 = 70_000_000_000_000_000_000_000_000u128;

async fn open_storage() -> (Arc<QStorage>, TempDir) {
    let dir = TempDir::new().expect("tempdir creation failed");
    let node_id = [0u8; 32];
    let storage = QStorage::open(dir.path(), node_id)
        .await
        .expect("QStorage::open failed");
    (Arc::new(storage), dir)
}

fn wallet(seed: u8) -> [u8; 32] { [seed; 32] }

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

fn coinbase_tx(to: [u8; 32], amt: u128, height: u64) -> Transaction {
    Transaction {
        id: { let mut id = [0u8; 32]; id[0] = 0xCB; id[1..9].copy_from_slice(&height.to_le_bytes()); id },
        from: [0u8; 32],
        to,
        amount: amt,
        fee: 0,
        nonce: height,
        signature: vec![],
        timestamp: chrono::Utc::now(),
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

fn transfer_tx(from: [u8; 32], to: [u8; 32], amt: u128, nonce: u64, tag: u8) -> Transaction {
    Transaction {
        id: { let mut id = [0u8; 32]; id[0] = 0xABu8.wrapping_add(tag); id[1..9].copy_from_slice(&nonce.to_le_bytes()); id[9] = from[0]; id[10] = to[0]; id[11] = tag; id },
        from,
        to,
        amount: amt,
        fee: 0,
        nonce,
        signature: vec![],
        timestamp: chrono::Utc::now(),
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
// TEST 1 — Self-mix (tx.from == tx.to) nets to exactly zero balance change
//
// Regression guard for the "old_recipient snapshotted before debit" clobber
// shape (double-credit or zero-out on a self-transfer). Targets the LIVE
// authoritative path: BalanceConsensusEngine::process_block_mining_rewards_tx,
// which is what q-api-server's mining pipeline calls in production
// (main.rs's mining pipeline -> block_producer.rs:2188
// get_transactions_for_block -> ... -> balance_consensus_tx apply).
// ============================================================================

#[tokio::test]
async fn test_self_mix_transfer_nets_to_zero_balance_change() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(test_ts(), "0".repeat(64));

    let a = wallet(0xA1);

    // Fund A with a coinbase reward, then A sends to itself.
    let blocks = vec![
        make_block(1, vec![coinbase_tx(a, QUG_100, 1)]),
        make_block(2, vec![transfer_tx(a, a, QUG_30, 1, 0)]),
    ];
    apply_blocks_full(&storage, &engine, &blocks).await?;

    let balances = storage.load_wallet_balances().await?;
    let balance_a = balances.get(&a).copied().unwrap_or(0);

    assert_eq!(
        balance_a, QUG_100,
        "self-mix transfer must net to zero balance change: expected exactly the pre-transfer \
         balance ({}), got {} (double-credit if higher, clobbered-to-zero-net if lower)",
        QUG_100, balance_a
    );

    println!("✅ test_self_mix_transfer_nets_to_zero_balance_change PASSED (balance={})", balance_a);
    Ok(())
}

// ============================================================================
// TEST 1b — Two concurrent self-mix txs for the SAME address in one block
//
// Catches a future "read all balances once up front, batch-apply" refactor
// that could reintroduce the clobber even though today's sequential-await
// code (subtract_balance_tx immediately followed by add_balance_tx, each a
// fresh read against the in-flight QTransaction) is safe.
// ============================================================================

#[tokio::test]
async fn test_two_self_mix_transfers_same_address_one_block_nets_to_zero() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(test_ts(), "0".repeat(64));

    let a = wallet(0xA2);

    let blocks = vec![
        make_block(1, vec![coinbase_tx(a, QUG_100, 1)]),
        make_block(2, vec![
            transfer_tx(a, a, QUG_30, 1, 0),
            transfer_tx(a, a, QUG_70, 2, 1),
        ]),
    ];
    apply_blocks_full(&storage, &engine, &blocks).await?;

    let balances = storage.load_wallet_balances().await?;
    let balance_a = balances.get(&a).copied().unwrap_or(0);

    assert_eq!(
        balance_a, QUG_100,
        "two same-block self-mix transfers must still net to zero total balance change: \
         expected {}, got {}",
        QUG_100, balance_a
    );

    println!("✅ test_two_self_mix_transfers_same_address_one_block_nets_to_zero PASSED (balance={})", balance_a);
    Ok(())
}

// ============================================================================
// TEST 2 — Pin: BalanceStorage::subtract_balance actually debits (v10.11.18)
//
// Pre-v10.11.18, subtract_balance called save_wallet_balance (max-wins
// guarded): a debit is old > new, so the guard silently skipped the write,
// the debit never landed, and the recipient side of a transfer still
// credited — net effect: money minted from thin air per transfer.
// The fix routes debits through save_wallet_balance_authoritative, which
// bypasses max-wins. This test pins that the debit actually lands.
// ============================================================================

#[tokio::test]
async fn test_subtract_balance_actually_debits_not_silently_skipped() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let addr = wallet(0xB1);
    let addr_hex = hex::encode(&addr);

    // Seed balance to 100 QUG via the authoritative path (bypasses max-wins,
    // safe for test setup regardless of guard behavior).
    storage.save_wallet_balance(&addr, QUG_100).await?;

    let pre = storage.load_wallet_balance(&addr).await?.unwrap_or(0);
    assert_eq!(pre, QUG_100, "test setup: seeded balance did not persist");

    // subtract_balance(addr, 30) — the BalanceStorage trait method,
    // exercised the same way balance_consensus.rs's transfer-processing
    // code calls it.
    storage.subtract_balance(&addr_hex, QUG_30).await?;

    let post = storage.load_wallet_balance(&addr).await?.unwrap_or(0);
    assert_eq!(
        post, QUG_70,
        "subtract_balance must actually apply the debit: expected {} (100 - 30), got {} \
         (100 would mean the debit was silently skipped by a max-wins guard — the v10.11.18 bug)",
        QUG_70, post
    );

    println!("✅ test_subtract_balance_actually_debits_not_silently_skipped PASSED (post={})", post);
    Ok(())
}

// ============================================================================
// TEST 2b — Positive control: QStorage::save_wallet_balance (direct, NOT via
// subtract_balance) still enforces max-wins for a lower value.
//
// Pins the intentional asymmetry documented at lib.rs:9956-9963: stale/
// lower-value writes from state-sync/replay/reconciliation callers must
// still be refused by the plain save_wallet_balance path. Only the
// dedicated authoritative debit path (subtract_balance, test above) is
// meant to bypass this guard.
// ============================================================================

#[tokio::test]
async fn test_save_wallet_balance_still_enforces_max_wins_for_direct_lower_write() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let addr = wallet(0xB2);

    storage.save_wallet_balance(&addr, QUG_100).await?;
    let pre = storage.load_wallet_balance(&addr).await?.unwrap_or(0);
    assert_eq!(pre, QUG_100, "test setup: seeded balance did not persist");

    // Direct call to the plain (non-authoritative) writer with a LOWER
    // value — must be refused (balance stays at 100), per the max-wins
    // guard's designed behavior for this entry point.
    storage.save_wallet_balance(&addr, QUG_30).await?;

    let post = storage.load_wallet_balance(&addr).await?.unwrap_or(0);
    assert_eq!(
        post, QUG_100,
        "save_wallet_balance must still refuse a stale lower-value write (max-wins): \
         expected balance to remain {}, got {} (guard regression — would let a stale replay \
         destroy a higher authoritative balance)",
        QUG_100, post
    );

    println!("✅ test_save_wallet_balance_still_enforces_max_wins_for_direct_lower_write PASSED (post={})", post);
    Ok(())
}
