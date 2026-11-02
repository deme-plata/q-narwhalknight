//! 2026-11-02 — "make the follower equal to the authority at height H, once, then pay only after H".
//!
//! Measured live: the Stockholm follower sits 486,831 QUG below Epsilon at equal height, in wallets
//! whose balances came from a one-time import on trust (10-30) — while every block since is credited
//! exactly. These tests drive the real import (`QStorage::apply_authority_import`) and the real two
//! pay paths (gossip engine + turbo-sync chunk bridge) and grade by reading the balances back the way
//! /integrity/compare does.
//!
//! Run (the credit floor is process-wide, so one thread):
//!   cargo test --package q-storage --test authority_import_tests -- --test-threads=1

use anyhow::Result;
use chrono::Utc;
use q_storage::authority_import::AuthoritySnapshot;
use q_storage::balance_consensus::{claim_block_credit_at, credit_floor, set_credit_floor};
use q_storage::{
    active_genesis_timestamp, BalanceConsensusEngine, BalanceConsensusError, BlockStateProcessor,
    QStorage, FOUNDER_WALLET, GENESIS_TIMESTAMP,
};
use q_types::{
    BlockHeader, MiningSolution, QBlock, QuantumMetadata, TokenType, Transaction,
    TransactionPrivacyLevel, TransactionType, TxSignaturePhase, VDFProof,
};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

const QUG: u128 = 1_000_000_000_000_000_000_000_000;
const A: [u8; 32] = [0xA1; 32];
const B: [u8; 32] = [0xB2; 32];
const C: [u8; 32] = [0xC3; 32];
const X: [u8; 32] = [0xE9; 32]; // a wallet only the follower has
const USD: [u8; 32] = [0x77; 32];
const OTHER: [u8; 32] = [0x66; 32];

fn qug_copy() -> [u8; 32] {
    let mut t = [0u8; 32];
    t[..3].copy_from_slice(b"QUG");
    t
}

async fn open_storage() -> (Arc<QStorage>, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let storage = QStorage::open(dir.path(), [0u8; 32]).await.expect("QStorage::open");
    (Arc::new(storage), dir)
}

async fn set_pointer(s: &QStorage, h: u64) -> Result<()> {
    s.db_put("blocks", b"qblock:latest", &h.to_be_bytes()).await
}

/// The authority's export, in the exact shape `GET /api/v1/sync/full-state` serves.
fn export(height: u64, network: &str, wallets: &[([u8; 32], u128)], tokens: &[([u8; 32], [u8; 32], u128)],
          debited: &[([u8; 32], u128)], credited: &[([u8; 32], u128)]) -> Vec<u8> {
    let m = |xs: &[([u8; 32], u128)]| -> serde_json::Map<String, serde_json::Value> {
        xs.iter().map(|(w, a)| (hex::encode(w), serde_json::Value::String(a.to_string()))).collect()
    };
    let t: serde_json::Map<String, serde_json::Value> = tokens.iter()
        .map(|(w, t, a)| (format!("{}_{}", hex::encode(w), hex::encode(t)), serde_json::Value::String(a.to_string())))
        .collect();
    serde_json::to_vec(&serde_json::json!({"success": true, "data": {
        "block_height": height, "network_id": network, "version": "10.11.114",
        "wallet_balances": m(wallets), "token_balances": t,
        "dex_qug_debited": m(debited), "dex_qug_credited": m(credited),
        "contracts": {}, "liquidity_pools": {}, "symbol_to_address": {}, "timestamp": 0
    }})).unwrap()
}

async fn seed_follower(s: &QStorage) -> Result<()> {
    s.save_wallet_balance(&A, 5 * QUG).await?;
    s.save_wallet_balance(&B, 7 * QUG).await?;
    s.save_wallet_balance(&X, 9 * QUG).await?;
    s.save_token_balance(&A, &USD, 3 * QUG).await?;
    s.save_token_balance(&A, &qug_copy(), 99 * QUG).await?;
    s.save_token_balance(&B, &OTHER, 8 * QUG).await?;
    s.record_dex_qug_debit(&hex::encode(X), QUG).await?; // a counter the authority does not have
    Ok(())
}

fn the_authority(height: u64) -> Vec<u8> {
    export(height, "mainnet-genesis",
        &[(A, 10 * QUG), (B, 7 * QUG), (C, QUG)],
        &[(A, USD, 4 * QUG), (A, qug_copy(), 12_345 * QUG), (B, USD, 2 * QUG)],
        &[(A, 2 * QUG)],
        &[(A, 5 * QUG), (C, QUG)])
}

#[tokio::test]
async fn import_makes_every_balance_exactly_the_authoritys() -> Result<()> {
    let (s, _d) = open_storage().await;
    seed_follower(&s).await?;
    set_pointer(&s, 1_000).await?;
    let snap = AuthoritySnapshot::from_full_state_json(&the_authority(1_005))?;
    let rep = s.apply_authority_import(&snap, "aa01", "mainnet-genesis").await?;

    let w = s.load_wallet_balances().await?;
    let want: HashMap<[u8; 32], u128> = [(A, 10 * QUG), (B, 7 * QUG), (C, QUG)].into_iter().collect();
    assert_eq!(w, want, "wallets must be exactly the authority's (the follower-only wallet X removed)");

    let t = s.load_token_balances().await?;
    assert_eq!(t.get(&(A, USD)).copied(), Some(4 * QUG));
    assert_eq!(t.get(&(B, USD)).copied(), Some(2 * QUG));
    assert_eq!(t.get(&(A, qug_copy())).copied(), Some(99 * QUG), "the 'QUG' token copy is never touched");
    assert_eq!(t.get(&(B, OTHER)).copied().unwrap_or(0), 0, "a token the authority does not hold reads 0");

    assert_eq!((rep.wallets_written, rep.wallets_changed, rep.wallets_removed), (3, 2, 1));
    assert_eq!((rep.token_copies_skipped, rep.tokens_zeroed, rep.dex_wallets), (1, 1, 2));
    assert_eq!(rep.supply_after, (18 * QUG).to_string());
    assert_eq!(credit_floor(), 1_005);

    // The 5-second DEX reconciler must find NOTHING to do: the imported balances already contain
    // every trade. (Import the counters without dex_applied_net and it would replay them all.)
    let adjusted = s.apply_dex_qug_adjustments().await?;
    assert_eq!(adjusted, 0, "the DEX reconciler moved {adjusted} wallets after the import");
    assert_eq!(s.load_wallet_balances().await?, want, "balances moved after the DEX reconciler ran");
    Ok(())
}

#[tokio::test]
async fn a_snapshot_below_the_local_pointer_is_refused_and_nothing_changes() -> Result<()> {
    let (s, _d) = open_storage().await;
    seed_follower(&s).await?;
    set_pointer(&s, 2_000).await?;
    let before = s.load_wallet_balances().await?;
    let snap = AuthoritySnapshot::from_full_state_json(&the_authority(1_005))?;
    let err = s.apply_authority_import(&snap, "aa02", "mainnet-genesis").await.unwrap_err();
    assert!(err.to_string().contains("BELOW"), "{err}");
    assert_eq!(s.load_wallet_balances().await?, before);
    Ok(())
}

#[tokio::test]
async fn a_snapshot_of_another_network_is_refused() -> Result<()> {
    let (s, _d) = open_storage().await;
    seed_follower(&s).await?;
    let before = s.load_wallet_balances().await?;
    let bytes = export(1_005, "testnet", &[(A, QUG)], &[], &[], &[]);
    let snap = AuthoritySnapshot::from_full_state_json(&bytes)?;
    assert!(s.apply_authority_import(&snap, "aa03", "mainnet-genesis").await.is_err());
    assert_eq!(s.load_wallet_balances().await?, before);
    Ok(())
}

#[tokio::test]
async fn the_same_file_twice_is_a_no_op_even_after_blocks_moved_balances() -> Result<()> {
    let (s, _d) = open_storage().await;
    seed_follower(&s).await?;
    let snap = AuthoritySnapshot::from_full_state_json(&the_authority(1_005))?;
    let first = s.apply_authority_import(&snap, "aa04", "mainnet-genesis").await?;
    assert!(!first.already_applied);
    s.save_wallet_balance(&A, 11 * QUG).await?; // a block after H paid A
    let second = s.apply_authority_import(&snap, "aa04", "mainnet-genesis").await?;
    assert!(second.already_applied);
    assert_eq!(s.load_wallet_balances().await?.get(&A).copied(), Some(11 * QUG),
        "a restart with the import still configured must not roll balances back to H");
    Ok(())
}

// ---- the credit floor against the two real pay paths ----

const PAY: [u128; 3] = [200_000_000_000_000_000_000_000, 140_000_000_000_000_000_000_000, 100_602_133_898_843_654_193_152];

fn coinbase_tx(to: [u8; 32], amount: u128, height: u64, idx: u8) -> Transaction {
    let mut id = [0u8; 32];
    id[0] = 0xCB;
    id[1..9].copy_from_slice(&height.to_le_bytes());
    id[9] = idx;
    id[10] = to[0];
    Transaction {
        id, from: [0u8; 32], to, amount, fee: 0, nonce: height, signature: vec![], timestamp: Utc::now(),
        data: vec![], token_type: TokenType::QUG, fee_token_type: TokenType::QUG, tx_type: TransactionType::Coinbase,
        pqc_signature: None, signature_phase: TxSignaturePhase::Phase0Ed25519, pqc_public_key: None,
        zk_proof_bundle: None, privacy_level: TransactionPrivacyLevel::Transparent, bulletproof: None,
        nullifier: None, memo: None,
    }
}

fn make_block(height: u64, to: [[u8; 32]; 3]) -> QBlock {
    let ts = active_genesis_timestamp() + 1_000 + height * 10;
    QBlock {
        header: BlockHeader {
            height, phase: 1, network_id: "mainnet-genesis".to_string(), prev_block_hash: [0u8; 32],
            solutions_root: [0u8; 32], tx_root: [0u8; 32], state_root: [0u8; 32], timestamp: ts, dag_round: height,
            vdf_proof: VDFProof::default(), anchor_validator: None, proposer: [0u8; 32],
            total_difficulty: height as u128 * 1000, producer_id: 0, producer_public_key: None, producer_signature: None,
            coinbase_merkle_root: None, total_coinbase_reward: Some(PAY.iter().sum()), coinbase_count: Some(3),
        },
        mining_solutions: vec![MiningSolution {
            nonce: height, hash: [0u8; 32], difficulty_target: [0xFFu8; 32], miner_address: to[0], timestamp: ts,
            pool_id: None, hash_rate_hs: 0, miner_id: None, worker_name: None, vdf_output: None, vdf_proof: None,
            vdf_checkpoints: None, vdf_iterations_count: None,
        }],
        dag_parents: vec![], quantum_metadata: QuantumMetadata::default(),
        transactions: (0..3).map(|i| coinbase_tx(to[i], PAY[i], height, i as u8)).collect(),
        balance_updates: vec![], size_bytes: 0,
    }
}

async fn pay_gossip(s: &Arc<QStorage>, e: &BalanceConsensusEngine, b: &QBlock) -> Result<()> {
    let tx = s.begin_transaction().await?;
    match e.process_block_mining_rewards_tx(&tx, b).await {
        Ok(_) | Err(BalanceConsensusError::AlreadyProcessed(_)) => {}
        Err(err) => return Err(err.into()),
    }
    tx.commit().await?;
    Ok(())
}

async fn holds(s: &Arc<QStorage>, who: &[[u8; 32]; 3]) -> Result<[u128; 3]> {
    let all = s.load_wallet_balances().await?;
    Ok([0, 1, 2].map(|i| all.get(&who[i]).copied().unwrap_or(0)))
}

#[tokio::test]
async fn blocks_at_or_below_the_floor_are_never_paid_and_the_next_block_is_paid_once() -> Result<()> {
    let (s, _d) = open_storage().await;
    let h = 24_500_000u64;
    let snap = AuthoritySnapshot::from_full_state_json(&export(h, "mainnet-genesis", &[(A, QUG)], &[], &[], &[]))?;
    s.apply_authority_import(&snap, "aa05", "mainnet-genesis").await?;
    assert_eq!(credit_floor(), h);

    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());
    let processor = BlockStateProcessor::new(s.get_rocks_db_handle().expect("rocksdb"));

    // Block H (inside the import) arrives by BOTH paths after the restart: nobody is paid.
    let old_miners = [[0x31; 32], [0x32; 32], [0x33; 32]];
    let at_floor = make_block(h, old_miners);
    pay_gossip(&s, &engine, &at_floor).await?;
    processor.process_block_with_bridge(&at_floor, true)?;
    assert_eq!(holds(&s, &old_miners).await?, [0, 0, 0], "a block inside the imported state was paid again");

    // Block H+1 arrives by both paths: paid exactly once.
    let new_miners = [[0x41; 32], [0x42; 32], [0x43; 32]];
    let next = make_block(h + 1, new_miners);
    pay_gossip(&s, &engine, &next).await?;
    processor.process_block_with_bridge(&next, true)?;
    assert_eq!(holds(&s, &new_miners).await?, PAY, "the first block after the floor must be paid exactly once");
    Ok(())
}

#[test]
fn the_floor_rule_itself() {
    set_credit_floor(100);
    assert!(!claim_block_credit_at(&[0x51; 32], 99));
    assert!(!claim_block_credit_at(&[0x52; 32], 100));
    assert!(claim_block_credit_at(&[0x53; 32], 101));
    assert!(!claim_block_credit_at(&[0x53; 32], 101), "the paid-once claim still holds above the floor");
    set_credit_floor(0);
    assert!(claim_block_credit_at(&[0x54; 32], 5), "floor 0 = a node that never imported");
}
