//! Balance Determinism Tests — Phase 1 prerequisite for BalanceRootV1
//!
//! Proves both `compute_balance_state_hash()` and `compute_balance_root_for_block()`
//! are fully deterministic across:
//! - Two independent DB instances processing identical state
//! - Node restarts (DB persist/reload)
//! - Different wallet insertion orders
//! - Zero balance filtering
//! - Single-unit balance sensitivity
//! - Empty state
//!
//! These tests MUST pass before activating BalanceRootV1 on mainnet (height 18,600,000).
//! If any test here fails, DO NOT DEPLOY.
//!
//! Run: cargo test --package q-storage --test balance_determinism_tests

use q_storage::{BalanceStorage, QStorage};
use std::sync::Arc;
use tempfile::TempDir;

// ============================================================================
// HELPERS
// ============================================================================

/// Open a fresh QStorage in a temp directory.
/// Uses the canonical open() path (same as production).
async fn open_test_storage() -> (Arc<QStorage>, TempDir) {
    let dir = TempDir::new().expect("failed to create tempdir");
    let node_id = [0u8; 32];
    let storage = QStorage::open(dir.path(), node_id)
        .await
        .expect("failed to open QStorage");
    (Arc::new(storage), dir)
}

/// Build a deterministic 64-hex wallet address from a seed byte.
/// The address is a 32-byte value with `seed` in each byte, encoded as 64 hex chars.
fn wallet_addr(seed: u8) -> String {
    hex::encode([seed; 32])
}

// ============================================================================
// TEST 1: Two nodes, same blocks → same hash
// ============================================================================
//
// Two independent nodes that apply identical balances must arrive at:
// - identical compute_balance_state_hash() output (hash, wallet_count, total_supply)
// - identical compute_balance_root_for_block() output
//
// This is the fundamental network consensus property.

#[tokio::test]
async fn test_two_nodes_same_blocks_same_hash() {
    let (engine_a, _dir_a) = open_test_storage().await;
    let (engine_b, _dir_b) = open_test_storage().await;

    let wallets: Vec<(String, u128)> = vec![
        (wallet_addr(0x01), 1_000_000_000_000_000_000_000_000u128),
        (wallet_addr(0x02), 2_500_000_000_000_000_000_000_000u128),
        (wallet_addr(0x03), 750_000_000_000_000_000_000_000u128),
    ];

    for (addr, balance) in &wallets {
        engine_a.add_balance(addr, *balance).await
            .expect("add_balance engine_a failed");
        engine_b.add_balance(addr, *balance).await
            .expect("add_balance engine_b failed");
    }

    let (hash_a, count_a, supply_a) = engine_a.compute_balance_state_hash().await
        .expect("compute_balance_state_hash engine_a failed");
    let (hash_b, count_b, supply_b) = engine_b.compute_balance_state_hash().await
        .expect("compute_balance_state_hash engine_b failed");

    assert_eq!(hash_a, hash_b,
        "Two nodes with identical balances must produce identical balance_state_hash.\n\
         hash_a: {}\n\
         hash_b: {}",
        hex::encode(hash_a), hex::encode(hash_b));
    assert_eq!(count_a, count_b,
        "Wallet counts must match: {} vs {}", count_a, count_b);
    assert_eq!(supply_a, supply_b,
        "Total supply must match: {} vs {}", supply_a, supply_b);
    assert_eq!(count_a, wallets.len(),
        "Expected {} wallets, got {}", wallets.len(), count_a);

    let root_a = engine_a.compute_balance_root_for_block().await
        .expect("compute_balance_root_for_block engine_a failed");
    let root_b = engine_b.compute_balance_root_for_block().await
        .expect("compute_balance_root_for_block engine_b failed");
    assert_ne!(root_a, [0u8; 32], "root must not be zero with non-zero balances");
    assert_eq!(root_a, root_b,
        "compute_balance_root_for_block must be identical across nodes with identical state.\n\
         root_a: {}\n\
         root_b: {}",
        hex::encode(root_a), hex::encode(root_b));
}

// ============================================================================
// TEST 2: Restart preserves the balance hash
// ============================================================================
//
// After writing balances and dropping the storage (simulating node restart),
// reopening the same directory must yield bit-for-bit identical hashes.
// This validates RocksDB persistence and the scan-then-sort-then-hash pattern.

#[tokio::test]
async fn test_restart_preserves_balance_hash() {
    let dir = TempDir::new().expect("failed to create tempdir");
    // Save the path before drop — TempDir lives for the full test.
    let dir_path = dir.path().to_path_buf();
    let node_id = [1u8; 32];

    let wallets: Vec<(String, u128)> = vec![
        (wallet_addr(0xAA), 5_000_000_000_000_000_000_000_000u128),
        (wallet_addr(0xBB), 3_333_333_333_333_333_333_333_333u128),
    ];

    // Phase 1: open, add balances, capture hashes.
    // Do NOT wrap in Arc — we need the RocksDB lock to release synchronously when
    // `engine` drops. An Arc could delay destruction if the runtime holds a ref.
    let (hash_before, root_before) = {
        let engine = QStorage::open(&dir_path, node_id).await
            .expect("open pre-restart failed");
        for (addr, balance) in &wallets {
            engine.add_balance(addr, *balance).await.expect("add_balance pre-restart failed");
        }
        let (h, _, _) = engine.compute_balance_state_hash().await
            .expect("compute_balance_state_hash pre-restart failed");
        let r = engine.compute_balance_root_for_block().await
            .expect("compute_balance_root_for_block pre-restart failed");
        (h, r)
        // engine drops here — RocksDB LOCK released synchronously
    };

    // Yield to the async runtime so RocksDB's destructor fully completes and
    // the OS file lock is released before we attempt to re-open the same path.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Phase 2: reopen same path, verify identical hashes (simulates node restart)
    let engine2 = Arc::new(QStorage::open(&dir_path, node_id).await
        .expect("open post-restart failed"));
    let (hash_after, _, _) = engine2.compute_balance_state_hash().await
        .expect("compute_balance_state_hash post-restart failed");
    let root_after = engine2.compute_balance_root_for_block().await
        .expect("compute_balance_root_for_block post-restart failed");

    assert_ne!(hash_before, [0u8; 32], "pre-restart hash must not be zero");
    assert_eq!(hash_before, hash_after,
        "compute_balance_state_hash must be identical after node restart.\n\
         Before: {}\n\
         After:  {}",
        hex::encode(hash_before), hex::encode(hash_after));

    assert_ne!(root_before, [0u8; 32], "pre-restart root must not be zero");
    assert_eq!(root_before, root_after,
        "compute_balance_root_for_block must be identical after node restart.\n\
         Before: {}\n\
         After:  {}",
        hex::encode(root_before), hex::encode(root_after));
}

// ============================================================================
// TEST 3: Insertion order does not affect the hash (order independence)
// ============================================================================
//
// The hash sorts (address, balance) pairs by address before hashing, so the
// order of DB writes must not affect the result.
// This guards against HashMap non-determinism leaking into consensus.

#[tokio::test]
async fn test_order_independence() {
    let (engine_fwd, _dir_fwd) = open_test_storage().await;
    let (engine_rev, _dir_rev) = open_test_storage().await;

    let wallets: Vec<(String, u128)> = vec![
        (wallet_addr(0xA0), 1_000_000_000_000_000_000_000_000u128),
        (wallet_addr(0xB0), 2_000_000_000_000_000_000_000_000u128),
        (wallet_addr(0xC0), 3_000_000_000_000_000_000_000_000u128),
        (wallet_addr(0xD0), 4_000_000_000_000_000_000_000_000u128),
        (wallet_addr(0xE0), 5_000_000_000_000_000_000_000_000u128),
    ];

    // Forward order: A → B → C → D → E
    for (addr, balance) in wallets.iter() {
        engine_fwd.add_balance(addr, *balance).await.expect("add_balance fwd failed");
    }

    // Reverse order: E → D → C → B → A
    for (addr, balance) in wallets.iter().rev() {
        engine_rev.add_balance(addr, *balance).await.expect("add_balance rev failed");
    }

    let (hash_fwd, count_fwd, supply_fwd) = engine_fwd.compute_balance_state_hash().await
        .expect("compute fwd failed");
    let (hash_rev, count_rev, supply_rev) = engine_rev.compute_balance_state_hash().await
        .expect("compute rev failed");

    assert_eq!(hash_fwd, hash_rev,
        "Balance state hash must be insertion-order independent.\n\
         Forward:  {}\n\
         Reversed: {}",
        hex::encode(hash_fwd), hex::encode(hash_rev));
    assert_eq!(count_fwd, count_rev);
    assert_eq!(supply_fwd, supply_rev);
    assert_eq!(count_fwd, wallets.len());

    let root_fwd = engine_fwd.compute_balance_root_for_block().await.expect("root fwd failed");
    let root_rev = engine_rev.compute_balance_root_for_block().await.expect("root rev failed");
    assert_eq!(root_fwd, root_rev,
        "compute_balance_root_for_block must be insertion-order independent.\n\
         Forward:  {}\n\
         Reversed: {}",
        hex::encode(root_fwd), hex::encode(root_rev));
}

// ============================================================================
// TEST 4: Zero-balance wallets are excluded
// ============================================================================
//
// Wallets with balance=0 are filtered out before hashing. A node that has
// a zero-balance record for a wallet must produce the same hash as a node
// that never saw that wallet at all (as long as non-zero balances agree).

#[tokio::test]
async fn test_zero_balance_excluded() {
    let (engine_with_zero, _dir_a) = open_test_storage().await;
    let (engine_without, _dir_b) = open_test_storage().await;

    let addr_live = wallet_addr(0x55);
    let addr_zero = wallet_addr(0x66); // written only to engine_with_zero
    let live_balance = 9_999_000_000_000_000_000_000_000u128;

    // engine_with_zero gets both wallets (one with 0 balance)
    engine_with_zero.add_balance(&addr_live, live_balance).await
        .expect("add live A failed");
    // add_balance with 0 is a no-op (adds 0 to current = 0), but the key may still appear
    engine_with_zero.add_balance(&addr_zero, 0u128).await
        .expect("add zero A failed");

    // engine_without gets only the live wallet
    engine_without.add_balance(&addr_live, live_balance).await
        .expect("add live B failed");

    let (hash_with_zero, count_with_zero, _) = engine_with_zero.compute_balance_state_hash().await
        .expect("compute with_zero failed");
    let (hash_without, count_without, _) = engine_without.compute_balance_state_hash().await
        .expect("compute without failed");

    assert_eq!(hash_with_zero, hash_without,
        "Zero-balance wallets must be excluded from the balance state hash.\n\
         With zero: {}\n\
         Without:   {}",
        hex::encode(hash_with_zero), hex::encode(hash_without));
    assert_eq!(count_with_zero, count_without,
        "Wallet count must exclude zero-balance entries: {} vs {}", count_with_zero, count_without);

    let root_with_zero = engine_with_zero.compute_balance_root_for_block().await
        .expect("root with_zero failed");
    let root_without = engine_without.compute_balance_root_for_block().await
        .expect("root without failed");
    assert_eq!(root_with_zero, root_without,
        "compute_balance_root_for_block must exclude zero-balance wallets.\n\
         With zero: {}\n\
         Without:   {}",
        hex::encode(root_with_zero), hex::encode(root_without));
}

// ============================================================================
// TEST 5: Changing a balance changes the hash
// ============================================================================
//
// A 1-base-unit change in any wallet's balance must produce a completely
// different hash. This ensures consensus can detect the smallest possible
// balance discrepancy.

#[tokio::test]
async fn test_balance_change_changes_hash() {
    let (engine, _dir) = open_test_storage().await;

    let addr_a = wallet_addr(0x11);
    let addr_b = wallet_addr(0x22);
    engine.add_balance(&addr_a, 1_000_000_000_000_000_000_000_000u128).await
        .expect("add balance_a failed");
    engine.add_balance(&addr_b, 2_000_000_000_000_000_000_000_000u128).await
        .expect("add balance_b failed");

    let (hash_1, _, supply_1) = engine.compute_balance_state_hash().await
        .expect("compute hash_1 failed");
    let root_1 = engine.compute_balance_root_for_block().await
        .expect("compute root_1 failed");

    // Add exactly 1 base unit to addr_a
    engine.add_balance(&addr_a, 1u128).await.expect("add +1 failed");

    let (hash_2, _, supply_2) = engine.compute_balance_state_hash().await
        .expect("compute hash_2 failed");
    let root_2 = engine.compute_balance_root_for_block().await
        .expect("compute root_2 failed");

    assert_ne!(hash_1, hash_2,
        "A 1-unit balance change must change compute_balance_state_hash.\n\
         hash_1: {}\n\
         hash_2: {}",
        hex::encode(hash_1), hex::encode(hash_2));
    assert_ne!(root_1, root_2,
        "A 1-unit balance change must change compute_balance_root_for_block.\n\
         root_1: {}\n\
         root_2: {}",
        hex::encode(root_1), hex::encode(root_2));
    assert_eq!(supply_2 - supply_1, 1u128,
        "Total supply must increase by exactly 1 unit after adding 1");
}

// ============================================================================
// TEST 6: Empty state behavior
// ============================================================================
//
// For compute_balance_state_hash():
//   - Returns (blake3_of_empty_input, 0, 0). The hash is deterministic but NOT [0;32].
//   - Two independent fresh engines must return the same hash value.
//
// For compute_balance_root_for_block():
//   - Explicitly returns [0u8; 32] when no non-zero balances exist.
//   - [0;32] is the sentinel used by validators to detect "missing root" post-BalanceRootV1.

#[tokio::test]
async fn test_empty_state_returns_zero_hash() {
    let (engine, _dir) = open_test_storage().await;

    // compute_balance_root_for_block: must return [0u8;32] for empty state
    let root = engine.compute_balance_root_for_block().await
        .expect("compute_balance_root_for_block empty failed");
    assert_eq!(root, [0u8; 32],
        "compute_balance_root_for_block must return [0u8;32] for empty state, got: {}",
        hex::encode(root));

    // compute_balance_state_hash: for empty input, the hasher is finalized with nothing fed in.
    // This produces the blake3 hash of the empty string, which is a specific non-zero constant.
    // We verify determinism across two fresh engines rather than hardcoding the constant.
    let (hash_1, count_1, supply_1) = engine.compute_balance_state_hash().await
        .expect("compute_balance_state_hash empty_1 failed");
    assert_eq!(count_1, 0, "Empty state must have 0 wallets");
    assert_eq!(supply_1, 0u128, "Empty state must have 0 total supply");

    let (engine2, _dir2) = open_test_storage().await;
    let (hash_2, count_2, supply_2) = engine2.compute_balance_state_hash().await
        .expect("compute_balance_state_hash empty_2 failed");
    assert_eq!(count_2, 0);
    assert_eq!(supply_2, 0u128);

    assert_eq!(hash_1, hash_2,
        "Two empty engines must return the same balance_state_hash.\n\
         Engine 1: {}\n\
         Engine 2: {}",
        hex::encode(hash_1), hex::encode(hash_2));
}
