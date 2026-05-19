//! v10.9.43 item 14: chain-linkage validation integration tests.
//!
//! These tests live in `tests/` (as a separate compile unit) rather than the
//! `sha3_data_integrity.rs::tests` inline module because the q-storage lib
//! test build has pre-existing schema-drift errors in unrelated test
//! fixtures (`balance_consensus.rs`, `manifest.rs`, etc.) that block the
//! entire `cargo test --lib` build. Integration tests in `tests/` compile
//! against the public API only, bypassing the broken sibling fixtures.
//!
//! Coverage:
//! - A correctly-chained 10-block sequence passes chain-linkage validation.
//! - A forged `prev_block_hash` is detected and rejected with a useful
//!   error message that mentions "chain-linkage".
//! - Single-block packs are trivially valid (nothing to chain).

use q_storage::sha3_data_integrity::{Sha3DataIntegrity, Sha3IntegrityConfig};
use q_types::block::{BlockHeader, QBlock, QuantumMetadata, VDFProof};

fn make_block_for_linkage(height: u64, prev_hash: [u8; 32]) -> QBlock {
    QBlock {
        header: BlockHeader {
            height,
            phase: 5,
            network_id: "mainnet-genesis".to_string(),
            prev_block_hash: prev_hash,
            solutions_root: [0u8; 32],
            tx_root: [0u8; 32],
            state_root: [(height & 0xff) as u8; 32],
            timestamp: 1_700_000_000 + height,
            dag_round: height,
            vdf_proof: VDFProof::default(),
            anchor_validator: None,
            proposer: [(height ^ 0x11) as u8; 32],
            producer_id: 0,
            total_difficulty: 1000u128 + height as u128,
            producer_public_key: None,
            producer_signature: None,
            coinbase_merkle_root: None,
            total_coinbase_reward: None,
            coinbase_count: None,
        },
        mining_solutions: vec![],
        dag_parents: vec![],
        quantum_metadata: QuantumMetadata::default(),
        transactions: vec![],
        balance_updates: vec![],
        size_bytes: 0,
    }
}

#[test]
fn test_chain_linkage_accepts_valid_chain() {
    let verifier = Sha3DataIntegrity::new(Sha3IntegrityConfig::default());
    let mut blocks = Vec::new();
    let mut prev_hash = [0u8; 32];
    for h in 1..=10u64 {
        let b = make_block_for_linkage(h, prev_hash);
        prev_hash = b.calculate_hash();
        blocks.push(b);
    }
    assert!(
        verifier.verify_chain_linkage(&blocks).is_ok(),
        "valid chain should pass linkage check"
    );
}

#[test]
fn test_chain_linkage_rejects_forged_prev_hash() {
    let verifier = Sha3DataIntegrity::new(Sha3IntegrityConfig::default());
    let mut blocks = Vec::new();
    let mut prev_hash = [0u8; 32];
    for h in 1..=5u64 {
        let b = make_block_for_linkage(h, prev_hash);
        prev_hash = b.calculate_hash();
        blocks.push(b);
    }
    // Forge: change block 3's prev_block_hash to a non-matching value.
    blocks[3].header.prev_block_hash = [0xCC; 32];
    let res = verifier.verify_chain_linkage(&blocks);
    assert!(res.is_err(), "forgery should be rejected");
    let msg = res.unwrap_err();
    assert!(
        msg.contains("chain-linkage mismatch"),
        "error message must mention chain-linkage: {}",
        msg
    );
}

#[test]
fn test_chain_linkage_single_block_ok() {
    let verifier = Sha3DataIntegrity::new(Sha3IntegrityConfig::default());
    let blocks = vec![make_block_for_linkage(1, [0u8; 32])];
    assert!(verifier.verify_chain_linkage(&blocks).is_ok());
}

#[test]
fn test_chain_linkage_empty_pack_ok() {
    let verifier = Sha3DataIntegrity::new(Sha3IntegrityConfig::default());
    let blocks: Vec<QBlock> = vec![];
    assert!(verifier.verify_chain_linkage(&blocks).is_ok());
}
