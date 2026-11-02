//! 2026-10-30 — "one block, two delivery paths, count the payments".
//!
//! Measured live on the Stockholm follower (sigil-se1, q-api-server 10.11.114f-sync2):
//! its total supply grew 1.99× what the block headers mint. Every block was paid
//! once by the gossip path (BalanceConsensusEngine::process_block_mining_rewards_tx,
//! which sets `processed_balance_block:<hash>`) and again, seconds later, by the
//! turbo-sync chunk path (BlockStateProcessor::process_block_with_bridge with
//! bridge_legacy = true, i.e. balance_engine = None), which never read that flag.
//! A 1-Hz supply trace showed the second payment arriving in lumps of exactly the
//! block counts of each `[TURBO BATCH] Saved N blocks` line (6, 22, 6, 7, 6).
//!
//! These tests drive the REAL two writers with the SAME block and grade with
//! `load_wallet_balances()` — the wallet_balance_* scan that /integrity/compare,
//! the balance root and total supply all read.
//!
//! Run: cargo test --package q-storage --test double_pay_two_paths_tests

use anyhow::Result;
use chrono::Utc;
use q_storage::balance_consensus::{balance_block_hash, processed_balance_key};
use q_storage::{
    active_genesis_timestamp, BalanceConsensusEngine, BalanceConsensusError, BlockStateProcessor,
    QStorage, CF_MANIFEST, FOUNDER_WALLET, GENESIS_TIMESTAMP,
};
use q_types::{
    BlockHeader, MiningSolution, QBlock, QuantumMetadata, TokenType, Transaction,
    TransactionPrivacyLevel, TransactionType, TxSignaturePhase, VDFProof,
};
use std::sync::Arc;
use tempfile::TempDir;

/// One Stockholm block's coinbase, split over three miners (24-decimal QUG units):
/// 0.440602133898843654193152 QUG = the supply step measured at h=24,409,581.
const PAY: [u128; 3] = [
    200_000_000_000_000_000_000_000,
    140_000_000_000_000_000_000_000,
    100_602_133_898_843_654_193_152,
];

async fn open_storage() -> (Arc<QStorage>, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let storage = QStorage::open(dir.path(), [0u8; 32]).await.expect("QStorage::open");
    (Arc::new(storage), dir)
}

/// Three distinct miner addresses per test (`seed` keeps tests apart: the
/// in-process claim set is global, so blocks must differ between tests).
fn miners(seed: u8) -> [[u8; 32]; 3] {
    [[seed; 32], [seed.wrapping_add(1); 32], [seed.wrapping_add(2); 32]]
}

fn coinbase_tx(to: [u8; 32], amount: u128, height: u64, idx: u8) -> Transaction {
    let mut id = [0u8; 32];
    id[0] = 0xCB;
    id[1..9].copy_from_slice(&height.to_le_bytes());
    id[9] = idx;
    id[10] = to[0];
    Transaction {
        id,
        from: [0u8; 32],
        to,
        amount,
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

fn make_block(height: u64, to: [[u8; 32]; 3]) -> QBlock {
    let ts = active_genesis_timestamp() + 1_000 + height * 10;
    let transactions = (0..3).map(|i| coinbase_tx(to[i], PAY[i], height, i as u8)).collect();
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
            total_coinbase_reward: Some(PAY.iter().sum()),
            coinbase_count: Some(3),
        },
        mining_solutions: vec![MiningSolution {
            nonce: height,
            hash: [0u8; 32],
            difficulty_target: [0xFFu8; 32],
            miner_address: to[0],
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

/// Delivery path 1 — gossip: the engine pays the block inside a storage transaction.
async fn pay_via_gossip_engine(storage: &Arc<QStorage>, engine: &BalanceConsensusEngine, block: &QBlock) -> Result<()> {
    let tx = storage.begin_transaction().await?;
    match engine.process_block_mining_rewards_tx(&tx, block).await {
        Ok(_) | Err(BalanceConsensusError::AlreadyProcessed(_)) => {}
        Err(e) => return Err(e.into()),
    }
    tx.commit().await?;
    Ok(())
}

/// Delivery path 2 — turbo-sync chunk: the state processor with the legacy bridge
/// ON, exactly as turbo_sync.rs calls it when balance_engine is None.
fn pay_via_turbo_chunk_bridge(processor: &BlockStateProcessor, block: &QBlock) -> Result<()> {
    processor.process_block_with_bridge(block, true)?;
    Ok(())
}

/// What /integrity/compare sums: the three miners' wallet_balance_* entries.
async fn paid_to(storage: &Arc<QStorage>, who: &[[u8; 32]; 3]) -> Result<[u128; 3]> {
    let all = storage.load_wallet_balances().await?;
    Ok([0, 1, 2].map(|i| all.get(&who[i]).copied().unwrap_or(0)))
}

fn assert_paid_once(got: [u128; 3], times_label: &str) {
    for i in 0..3 {
        assert_eq!(
            got[i], PAY[i],
            "{}: miner {} holds {} base units, the block paid it {} — {:.2}× the header",
            times_label, i, got[i], PAY[i], got[i] as f64 / PAY[i] as f64
        );
    }
}

/// The shape measured at the tip: gossip pays first, turbo-sync re-delivers the
/// same block ~20 s later.
#[tokio::test]
async fn gossip_then_turbo_chunk_pays_once() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());
    let processor = BlockStateProcessor::new(storage.get_rocks_db_handle().expect("rocksdb"));
    let who = miners(0x10);
    let block = make_block(24_409_581, who);

    pay_via_gossip_engine(&storage, &engine, &block).await?;
    assert_paid_once(paid_to(&storage, &who).await?, "after gossip alone");

    pay_via_turbo_chunk_bridge(&processor, &block)?;
    assert_paid_once(paid_to(&storage, &who).await?, "after gossip + turbo-chunk");
    Ok(())
}

/// The reverse order (a block gossip missed, or the 23:39:18 case: turbo batch
/// first, then the BATCH-SYNC engine pass over the same blocks).
#[tokio::test]
async fn turbo_chunk_then_gossip_pays_once() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string());
    let db = storage.get_rocks_db_handle().expect("rocksdb");
    let processor = BlockStateProcessor::new(db.clone());
    let who = miners(0x20);
    let block = make_block(24_409_599, who);

    pay_via_turbo_chunk_bridge(&processor, &block)?;
    assert_paid_once(paid_to(&storage, &who).await?, "after turbo-chunk alone");

    // The chunk path must leave the same mark the engine leaves.
    let cf = db.cf_handle(CF_MANIFEST).expect("manifest cf");
    let key = processed_balance_key(&balance_block_hash(&block));
    assert!(db.get_cf(&cf, key.as_bytes())?.is_some(),
        "the turbo-chunk payment must write processed_balance_block:<hash>, or the engine pays again");

    pay_via_gossip_engine(&storage, &engine, &block).await?;
    assert_paid_once(paid_to(&storage, &who).await?, "after turbo-chunk + gossip");
    Ok(())
}

/// Catch-up shape: the chunk path is the only writer and must still pay (once).
#[tokio::test]
async fn turbo_chunk_alone_still_pays() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let processor = BlockStateProcessor::new(storage.get_rocks_db_handle().expect("rocksdb"));
    let who = miners(0x30);
    for h in 24_100_000..24_100_010u64 {
        pay_via_turbo_chunk_bridge(&processor, &make_block(h, who))?;
    }
    let got = paid_to(&storage, &who).await?;
    for i in 0..3 {
        assert_eq!(got[i], PAY[i] * 10, "ten blocks through the chunk path alone must pay ten times");
    }
    Ok(())
}

/// Both paths fired at the same moment for every block (gossip handler task vs the
/// turbo-sync task) — the window in which neither flag is committed yet.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn both_paths_at_once_pay_once() -> Result<()> {
    let (storage, _dir) = open_storage().await;
    let engine = Arc::new(BalanceConsensusEngine::new(GENESIS_TIMESTAMP, FOUNDER_WALLET.to_string()));
    let processor = Arc::new(BlockStateProcessor::new(storage.get_rocks_db_handle().expect("rocksdb")));
    let who = miners(0x40);
    const N: u64 = 64;
    for h in 24_500_000..24_500_000 + N {
        let block = make_block(h, who);
        let (b1, b2) = (block.clone(), block);
        let (s, e, p) = (storage.clone(), engine.clone(), processor.clone());
        let gossip = tokio::spawn(async move { pay_via_gossip_engine(&s, &e, &b1).await });
        let chunk = tokio::task::spawn_blocking(move || pay_via_turbo_chunk_bridge(&p, &b2));
        gossip.await??;
        chunk.await??;
    }
    let got = paid_to(&storage, &who).await?;
    for i in 0..3 {
        assert_eq!(got[i], PAY[i] * N as u128,
            "{} blocks delivered by both paths at once: miner {} got {:.3}× what the headers mint",
            N, i, got[i] as f64 / (PAY[i] * N as u128) as f64);
    }
    Ok(())
}
