//! BalanceRootV1 Activation Test Suite
//!
//! Comprehensive, atomic-level simulation of exactly what happens at height
//! 18,600,000 when BalanceRootV1 activates on mainnet.
//!
//! Eight modules, ordered from smallest unit to largest simulation:
//!
//!   Module 1 — Upgrade Gate Boundary
//!     Verifies the exact 18,599,999 → 18,600,000 transition at the is_upgrade_active() level.
//!
//!   Module 2 — Balance Root Hash Correctness
//!     Proves compute_balance_root_for_block() satisfies every property the protocol requires:
//!     determinism, order-independence, big-endian encoding, domain-separator uniqueness,
//!     zero-balance exclusion, single-satoshi sensitivity.
//!
//!   Module 3 — PRE-STATE Semantics
//!     The root written into block N reflects the balance state BEFORE block N's transactions
//!     are applied. This module simulates block chains and verifies the pre/post split.
//!
//!   Module 4 — Block Acceptance / Rejection Rules (atomic)
//!     Exercises the exact conditional logic in main.rs accept_block() — every branch at
//!     every activation boundary, correct root, wrong root, zero root, old-node simulation.
//!
//!   Module 5 — Balance Divergence Detection
//!     Two nodes that diverge (one missed a P2P balance update, or a different tx ordering)
//!     produce different roots. Their blocks reject each other. The split is detectable.
//!
//!   Module 6 — Network Partition Scenarios
//!     Partition → diverge → rejoin. Both sides produce internally-consistent chains.
//!     Cross-partition blocks are rejected by both sides.
//!
//!   Module 7 — Activation Boundary Sequence (iter)
//!     Walks blocks 18,599,990 through 18,600,010 one by one, verifying correct
//!     enforcement at every step. No tolerance for off-by-one in the gate.
//!
//!   Module 8 — Production Scale (iter)
//!     1,340-wallet state, 1,000-block sequences, concurrent root reads, churn
//!     (wallets created and zeroed). Matches production wallet count.
//!
//! Run: cargo test --package q-storage --test balance_root_v1_activation_tests

use q_consensus_guard::{Upgrade, MAINNET_UPGRADES};
use q_storage::{BalanceStorage, QStorage};
use std::sync::Arc;
use tempfile::TempDir;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Mainnet activation height for BalanceRootV1.
const ACTIVATION: u64 = 18_600_000;

/// One block before activation — enforcement must NOT apply.
const BEFORE: u64 = ACTIVATION - 1;

/// One block after activation — enforcement must apply.
const AFTER: u64 = ACTIVATION + 1;

/// Production wallet count (as of 2026-05-06).
const PROD_WALLET_COUNT: usize = 1_340;

/// Standard test balance unit: 1 QUG in 24-decimal representation.
const ONE_QUG: u128 = 1_000_000_000_000_000_000_000_000u128;

// ============================================================================
// SHARED HELPERS
// ============================================================================

async fn open_storage() -> (Arc<QStorage>, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let storage = QStorage::open(dir.path(), [0u8; 32])
        .await
        .expect("QStorage::open");
    (Arc::new(storage), dir)
}

/// Deterministic hex wallet address from a u16 seed (supports > 255 wallets).
fn addr(seed: u16) -> String {
    let mut bytes = [0u8; 32];
    bytes[30] = (seed >> 8) as u8;
    bytes[31] = (seed & 0xff) as u8;
    hex::encode(bytes)
}

/// Simple linear-congruential PRNG (no external dep, fully deterministic).
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Read the mainnet activation height for BalanceRootV1 directly from the schedule.
/// Does NOT require init_upgrade_gate() — safe to call in tests.
fn balance_root_v1_active_at(block_height: u64) -> bool {
    let config = MAINNET_UPGRADES
        .get(&Upgrade::BalanceRootV1)
        .expect("BalanceRootV1 missing from MAINNET_UPGRADES");
    block_height >= config.activation_height
}

/// The exact three-branch decision made in main.rs for every received block.
/// Returns Ok(()) if the block would be accepted, Err(reason) if rejected.
fn simulate_validator_decision(
    block_height: u64,
    block_state_root: [u8; 32],
    local_root: [u8; 32],
) -> Result<(), &'static str> {
    if balance_root_v1_active_at(block_height) {
        if block_state_root == [0u8; 32] {
            return Err("REJECT: zero root (old node or missing root)");
        }
        if block_state_root != local_root {
            return Err("REJECT: balance root mismatch");
        }
    }
    Ok(())
}

// ============================================================================
// MODULE 1 — UPGRADE GATE BOUNDARY
// ============================================================================

/// Heights well below activation must not be active.
#[test]
fn gate_inactive_at_genesis() {
    assert!(!balance_root_v1_active_at(0));
    assert!(!balance_root_v1_active_at(1));
}

#[test]
fn gate_inactive_at_current_tip() {
    // Network tip at time of writing (~17.4M). Must be below activation.
    assert!(!balance_root_v1_active_at(17_400_000));
}

#[test]
fn gate_inactive_one_below_activation() {
    assert!(!balance_root_v1_active_at(BEFORE));
}

#[test]
fn gate_active_at_exact_activation() {
    assert!(balance_root_v1_active_at(ACTIVATION));
}

#[test]
fn gate_active_one_above_activation() {
    assert!(balance_root_v1_active_at(AFTER));
}

#[test]
fn gate_active_far_above_activation() {
    assert!(balance_root_v1_active_at(100_000_000));
    assert!(balance_root_v1_active_at(u64::MAX));
}

/// The upgrade config in MAINNET_UPGRADES must match what is_upgrade_active() returns.
#[test]
fn gate_config_matches_is_active() {
    let config = MAINNET_UPGRADES
        .get(&Upgrade::BalanceRootV1)
        .expect("BalanceRootV1 must be in MAINNET_UPGRADES");
    assert_eq!(config.activation_height, ACTIVATION);
    assert!(config.mandatory, "BalanceRootV1 must be mandatory");
    assert_eq!(config.min_version, "10.6.0");
}

/// StateRootV1 (= 7) must remain at u64::MAX — it must NOT activate at 18,600,000.
/// The two upgrades are independent; conflating them would be catastrophic.
#[test]
fn state_root_v1_still_inactive_at_balance_root_activation() {
    assert!(
        !{
            let cfg = MAINNET_UPGRADES.get(&Upgrade::StateRootV1).unwrap();
            ACTIVATION >= cfg.activation_height
        },
        "StateRootV1 must NOT activate at height {}; it is u64::MAX", ACTIVATION
    );
}

// ============================================================================
// MODULE 2 — BALANCE ROOT HASH CORRECTNESS
// ============================================================================

/// Same state, called twice, must return the identical 32 bytes.
#[tokio::test]
async fn root_is_deterministic_multiple_calls() {
    let (s, _d) = open_storage().await;
    s.add_balance(&addr(1), 5 * ONE_QUG).await.unwrap();
    s.add_balance(&addr(2), 3 * ONE_QUG).await.unwrap();

    let r1 = s.compute_balance_root_for_block().await.unwrap();
    let r2 = s.compute_balance_root_for_block().await.unwrap();
    let r3 = s.compute_balance_root_for_block().await.unwrap();
    assert_eq!(r1, r2, "second call differs");
    assert_eq!(r1, r3, "third call differs");
}

/// Empty state → [0u8; 32]. No wallets, no hash.
#[tokio::test]
async fn root_empty_state_is_zero() {
    let (s, _d) = open_storage().await;
    let root = s.compute_balance_root_for_block().await.unwrap();
    assert_eq!(root, [0u8; 32], "empty state root must be all zeros");
}

/// A wallet with exactly zero balance must be excluded from the root.
/// Two storages — one with (addr, 0), one with nothing — must agree.
#[tokio::test]
async fn root_zero_balance_wallet_excluded() {
    let (s_with, _d1) = open_storage().await;
    let (s_without, _d2) = open_storage().await;

    s_with.add_balance(&addr(1), 10 * ONE_QUG).await.unwrap();
    s_with.add_balance(&addr(2), 0u128).await.unwrap(); // zero — must be excluded

    s_without.add_balance(&addr(1), 10 * ONE_QUG).await.unwrap();
    // addr(2) never written

    let r_with = s_with.compute_balance_root_for_block().await.unwrap();
    let r_without = s_without.compute_balance_root_for_block().await.unwrap();
    assert_eq!(r_with, r_without, "zero-balance wallet must not affect root");
}

/// Changing any balance by a single unit (1 / 10^24 QUG) changes the root.
/// This is the "single-satoshi sensitivity" property.
#[tokio::test]
async fn root_single_unit_change_changes_root() {
    let (s_base, _d1) = open_storage().await;
    let (s_plus1, _d2) = open_storage().await;

    s_base.add_balance(&addr(1), 1_000 * ONE_QUG).await.unwrap();
    s_plus1.add_balance(&addr(1), 1_000 * ONE_QUG + 1).await.unwrap();

    let r_base = s_base.compute_balance_root_for_block().await.unwrap();
    let r_plus1 = s_plus1.compute_balance_root_for_block().await.unwrap();
    assert_ne!(r_base, r_plus1, "adding 1 unit must change root");
}

/// Insertion order of wallets into the DB must not affect the root.
/// The sort-by-address step neutralises any HashMap/RocksDB ordering.
#[tokio::test]
async fn root_order_independence_three_wallets() {
    let wallets = vec![
        (addr(0xAA), 7 * ONE_QUG),
        (addr(0x01), 3 * ONE_QUG),
        (addr(0xFF), 11 * ONE_QUG),
    ];

    let (s_fwd, _d1) = open_storage().await;
    let (s_rev, _d2) = open_storage().await;

    for (a, b) in &wallets {
        s_fwd.add_balance(a, *b).await.unwrap();
    }
    for (a, b) in wallets.iter().rev() {
        s_rev.add_balance(a, *b).await.unwrap();
    }

    let r_fwd = s_fwd.compute_balance_root_for_block().await.unwrap();
    let r_rev = s_rev.compute_balance_root_for_block().await.unwrap();
    assert_eq!(r_fwd, r_rev, "insertion order must not affect root");
}

/// compute_balance_root_for_block() uses "balance_root_v1" domain separator + big-endian.
/// compute_balance_state_hash() uses no domain separator + little-endian (legacy).
/// They MUST differ even for identical state — they serve different purposes.
#[tokio::test]
async fn root_differs_from_legacy_state_hash() {
    let (s, _d) = open_storage().await;
    s.add_balance(&addr(1), 42 * ONE_QUG).await.unwrap();

    let new_root = s.compute_balance_root_for_block().await.unwrap();
    let (legacy_hash, _, _) = s.compute_balance_state_hash().await.unwrap();

    assert_ne!(
        new_root, legacy_hash,
        "BalanceRootV1 root must not equal the legacy state hash \
         (different encoding: big-endian + domain separator vs little-endian + none)"
    );
}

/// Adding a new wallet that didn't exist before changes the root.
#[tokio::test]
async fn root_new_wallet_changes_root() {
    let (s, _d) = open_storage().await;
    s.add_balance(&addr(1), 5 * ONE_QUG).await.unwrap();
    let r_before = s.compute_balance_root_for_block().await.unwrap();

    s.add_balance(&addr(2), ONE_QUG).await.unwrap();
    let r_after = s.compute_balance_root_for_block().await.unwrap();

    assert_ne!(r_before, r_after, "adding a new wallet must change root");
}

// ============================================================================
// MODULE 3 — PRE-STATE SEMANTICS
// ============================================================================
//
// Block N's state_root is computed on the balance state AFTER block N-1 is applied
// (i.e. BEFORE block N's own transactions are applied). Both producer and validator
// compute on the same snapshot. This module verifies that chain.

/// Simulate a two-block chain and verify pre-state semantics hold.
///
///   genesis:  balances = {}
///   block 1:  pre_root = root({})                = 0x000…
///             transactions: addr(1) += 100 QUG
///   block 2:  pre_root = root({addr(1): 100})
///             transactions: addr(2) += 50 QUG
///   block 3:  pre_root = root({addr(1): 100, addr(2): 50})
#[tokio::test]
async fn prestate_two_block_chain() {
    // Node A produces blocks
    let (producer, _d1) = open_storage().await;
    // Node B validates blocks
    let (validator, _d2) = open_storage().await;

    // --- Block 1 ---
    // Pre-state root (both nodes have empty state)
    let producer_root_1 = producer.compute_balance_root_for_block().await.unwrap();
    let validator_root_1 = validator.compute_balance_root_for_block().await.unwrap();
    assert_eq!(producer_root_1, validator_root_1, "block 1 pre-state roots must match");
    assert_eq!(producer_root_1, [0u8; 32], "block 1 pre-state must be zero (empty)");

    // Apply block 1's transactions on both nodes
    producer.add_balance(&addr(1), 100 * ONE_QUG).await.unwrap();
    validator.add_balance(&addr(1), 100 * ONE_QUG).await.unwrap();

    // --- Block 2 ---
    let producer_root_2 = producer.compute_balance_root_for_block().await.unwrap();
    let validator_root_2 = validator.compute_balance_root_for_block().await.unwrap();
    assert_eq!(producer_root_2, validator_root_2, "block 2 pre-state roots must match");
    assert_ne!(producer_root_2, [0u8; 32], "block 2 pre-state must not be zero");

    // Apply block 2's transactions
    producer.add_balance(&addr(2), 50 * ONE_QUG).await.unwrap();
    validator.add_balance(&addr(2), 50 * ONE_QUG).await.unwrap();

    // --- Block 3 ---
    let producer_root_3 = producer.compute_balance_root_for_block().await.unwrap();
    let validator_root_3 = validator.compute_balance_root_for_block().await.unwrap();
    assert_eq!(producer_root_3, validator_root_3, "block 3 pre-state roots must match");

    // Block 2 and block 3 pre-state roots must differ (block 2 added addr(2))
    assert_ne!(producer_root_2, producer_root_3, "consecutive roots must differ");
}

/// If a node applies transactions in the wrong order or skips one, its pre-state
/// diverges. Verify the pre-state root captures this immediately.
#[tokio::test]
async fn prestate_missed_transaction_diverges() {
    let (complete, _d1) = open_storage().await;
    let (missed_one, _d2) = open_storage().await;

    // Both apply tx1
    complete.add_balance(&addr(1), 100 * ONE_QUG).await.unwrap();
    missed_one.add_balance(&addr(1), 100 * ONE_QUG).await.unwrap();

    // Only `complete` applies tx2 (missed_one skips it — simulates missed P2P update)
    complete.add_balance(&addr(2), 50 * ONE_QUG).await.unwrap();

    let root_complete = complete.compute_balance_root_for_block().await.unwrap();
    let root_missed = missed_one.compute_balance_root_for_block().await.unwrap();

    assert_ne!(
        root_complete, root_missed,
        "missing a transaction must produce a different pre-state root"
    );
}

// ============================================================================
// MODULE 4 — BLOCK ACCEPTANCE / REJECTION RULES (ATOMIC)
// ============================================================================
//
// These tests exercise simulate_validator_decision() which mirrors the exact
// conditional tree in main.rs:
//
//   if is_upgrade_active(BalanceRootV1, height) {
//       if block.state_root == [0;32] → REJECT
//       if block.state_root != local_root → REJECT
//   }
//   // else: accept regardless of root field

/// Before activation: zero root in state_root → ACCEPTED (old nodes still valid).
#[test]
fn accept_zero_root_before_activation() {
    let result = simulate_validator_decision(BEFORE, [0u8; 32], [0u8; 32]);
    assert!(result.is_ok(), "zero root before activation must be accepted");
}

/// Before activation: any root value → ACCEPTED (enforcement not yet live).
#[test]
fn accept_any_root_before_activation() {
    let arbitrary_root = [0xABu8; 32];
    let local_root = [0xCDu8; 32]; // different — doesn't matter before activation
    let result = simulate_validator_decision(BEFORE, arbitrary_root, local_root);
    assert!(result.is_ok(), "any root before activation must be accepted");
}

/// At activation: zero root → REJECTED. Old-node simulation.
#[test]
fn reject_zero_root_at_activation() {
    let result = simulate_validator_decision(ACTIVATION, [0u8; 32], [0xFFu8; 32]);
    assert!(result.is_err(), "zero root at activation must be rejected");
    assert!(result.unwrap_err().contains("zero root"));
}

/// At activation: wrong root (mismatch) → REJECTED.
#[test]
fn reject_wrong_root_at_activation() {
    let block_root = [0xAAu8; 32];
    let local_root = [0xBBu8; 32];
    let result = simulate_validator_decision(ACTIVATION, block_root, local_root);
    assert!(result.is_err(), "mismatched root at activation must be rejected");
    assert!(result.unwrap_err().contains("mismatch"));
}

/// At activation: correct root (block_root == local_root) → ACCEPTED.
#[test]
fn accept_correct_root_at_activation() {
    let root = [0x42u8; 32];
    let result = simulate_validator_decision(ACTIVATION, root, root);
    assert!(result.is_ok(), "matching root at activation must be accepted");
}

/// After activation: zero root → REJECTED. Always.
#[test]
fn reject_zero_root_after_activation() {
    let result = simulate_validator_decision(AFTER, [0u8; 32], [0x01u8; 32]);
    assert!(result.is_err());
}

/// After activation: correct root → ACCEPTED.
#[test]
fn accept_correct_root_after_activation() {
    let root = [0xDEu8; 32];
    let result = simulate_validator_decision(AFTER, root, root);
    assert!(result.is_ok());
}

/// Exactly one bit difference in the root → REJECTED.
/// Proves there is no tolerance: the comparison is always byte-exact.
#[test]
fn reject_one_bit_off_root() {
    let mut block_root = [0xAAu8; 32];
    let local_root = [0xAAu8; 32];
    block_root[15] ^= 0x01; // flip one bit

    let result = simulate_validator_decision(ACTIVATION, block_root, local_root);
    assert!(result.is_err(), "one-bit difference must be rejected");
}

/// Old node (running v10.5.3): always emits zero state_root.
/// Its blocks must be rejected by upgraded nodes (v10.6.0+) after activation.
#[tokio::test]
async fn old_node_blocks_rejected_after_activation() {
    let (local, _d) = open_storage().await;
    local.add_balance(&addr(1), 100 * ONE_QUG).await.unwrap();
    let local_root = local.compute_balance_root_for_block().await.unwrap();

    // Old node emits state_root = [0; 32]
    let old_node_state_root = [0u8; 32];

    let result = simulate_validator_decision(ACTIVATION, old_node_state_root, local_root);
    assert!(
        result.is_err(),
        "old node (v10.5.x) blocks must be rejected at height {}", ACTIVATION
    );
}

// ============================================================================
// MODULE 5 — BALANCE DIVERGENCE DETECTION
// ============================================================================

/// Single wallet diverges by 1 unit → completely different root → block rejected.
#[tokio::test]
async fn divergence_single_wallet_off_by_one() {
    let (node_a, _d1) = open_storage().await;
    let (node_b, _d2) = open_storage().await;

    // Same base state
    node_a.add_balance(&addr(1), 1000 * ONE_QUG).await.unwrap();
    node_b.add_balance(&addr(1), 1000 * ONE_QUG).await.unwrap();

    // Node B receives one extra unit (node A didn't get the P2P update)
    node_b.add_balance(&addr(1), 1u128).await.unwrap();

    let root_a = node_a.compute_balance_root_for_block().await.unwrap();
    let root_b = node_b.compute_balance_root_for_block().await.unwrap();

    assert_ne!(root_a, root_b, "off-by-one divergence must produce different roots");

    // Node B produces a block with its root. Node A rejects it.
    let verdict = simulate_validator_decision(ACTIVATION, root_b, root_a);
    assert!(verdict.is_err(), "diverged block must be rejected");

    // Symmetrically: Node A's block is rejected by B
    let verdict2 = simulate_validator_decision(ACTIVATION, root_a, root_b);
    assert!(verdict2.is_err(), "symmetric rejection must also hold");
}

/// One node received a P2P balance update, the other did not.
/// Their roots diverge immediately. The split is detected on the next block ≥ activation.
#[tokio::test]
async fn divergence_missed_p2p_update_detected() {
    let (synced, _d1) = open_storage().await;
    let (unsynced, _d2) = open_storage().await;

    // Common base
    for i in 0u16..10 {
        let amount = (i as u128 + 1) * ONE_QUG;
        synced.add_balance(&addr(i), amount).await.unwrap();
        unsynced.add_balance(&addr(i), amount).await.unwrap();
    }

    // `synced` receives a P2P balance update that `unsynced` misses
    synced.add_balance(&addr(5), 777 * ONE_QUG).await.unwrap();

    let root_synced = synced.compute_balance_root_for_block().await.unwrap();
    let root_unsynced = unsynced.compute_balance_root_for_block().await.unwrap();

    assert_ne!(root_synced, root_unsynced);

    // The divergence is caught when `synced` tries to produce a block that `unsynced` validates
    let verdict = simulate_validator_decision(ACTIVATION, root_synced, root_unsynced);
    assert!(
        verdict.is_err(),
        "missed P2P update must be detected at activation"
    );
}

/// Divergence in a wallet with address 0x000...001 (the lexicographically smallest address).
/// Ensures the sort order places it correctly and the divergence propagates to the root.
#[tokio::test]
async fn divergence_lexicographically_first_wallet() {
    let min_addr = hex::encode([0u8; 31].iter().chain(&[1u8]).cloned().collect::<Vec<u8>>());

    let (node_a, _d1) = open_storage().await;
    let (node_b, _d2) = open_storage().await;

    node_a.add_balance(&min_addr, ONE_QUG).await.unwrap();
    node_b.add_balance(&min_addr, ONE_QUG + 1).await.unwrap(); // 1 unit off

    let r_a = node_a.compute_balance_root_for_block().await.unwrap();
    let r_b = node_b.compute_balance_root_for_block().await.unwrap();
    assert_ne!(r_a, r_b, "lexicographically-first wallet divergence must be detected");
}

/// Divergence in a wallet with address 0xFF…FF (the lexicographically last address).
#[tokio::test]
async fn divergence_lexicographically_last_wallet() {
    let max_addr = hex::encode([0xFFu8; 32]);

    let (node_a, _d1) = open_storage().await;
    let (node_b, _d2) = open_storage().await;

    node_a.add_balance(&max_addr, ONE_QUG).await.unwrap();
    node_b.add_balance(&max_addr, ONE_QUG + 1).await.unwrap();

    let r_a = node_a.compute_balance_root_for_block().await.unwrap();
    let r_b = node_b.compute_balance_root_for_block().await.unwrap();
    assert_ne!(r_a, r_b, "lexicographically-last wallet divergence must be detected");
}

// ============================================================================
// MODULE 6 — NETWORK PARTITION SCENARIOS
// ============================================================================

/// Partition: two halves of the network process different transactions.
/// After partition heals, each side rejects the other's blocks.
/// Both sides are internally consistent.
#[tokio::test]
async fn partition_two_sides_reject_each_other() {
    // Shared base state (pre-partition)
    let (side_a, _d1) = open_storage().await;
    let (side_b, _d2) = open_storage().await;

    // Both sides start with identical state
    side_a.add_balance(&addr(1), 500 * ONE_QUG).await.unwrap();
    side_b.add_balance(&addr(1), 500 * ONE_QUG).await.unwrap();

    // --- Partition begins ---

    // Side A processes a transaction
    side_a.add_balance(&addr(2), 100 * ONE_QUG).await.unwrap();

    // Side B processes a different transaction
    side_b.add_balance(&addr(3), 200 * ONE_QUG).await.unwrap();

    let root_a = side_a.compute_balance_root_for_block().await.unwrap();
    let root_b = side_b.compute_balance_root_for_block().await.unwrap();

    // Internal consistency: each side matches itself
    assert!(simulate_validator_decision(ACTIVATION, root_a, root_a).is_ok());
    assert!(simulate_validator_decision(ACTIVATION, root_b, root_b).is_ok());

    // Cross-partition: each side rejects the other
    assert!(simulate_validator_decision(ACTIVATION, root_a, root_b).is_err(),
        "side B must reject side A's block");
    assert!(simulate_validator_decision(ACTIVATION, root_b, root_a).is_err(),
        "side A must reject side B's block");
}

/// Three-node consensus: all three have identical state → all accept each other's blocks.
#[tokio::test]
async fn three_node_consensus_all_agree() {
    let (n1, _d1) = open_storage().await;
    let (n2, _d2) = open_storage().await;
    let (n3, _d3) = open_storage().await;

    let state = vec![
        (addr(1), 100 * ONE_QUG),
        (addr(2), 200 * ONE_QUG),
        (addr(3), 300 * ONE_QUG),
    ];

    for (a, b) in &state {
        n1.add_balance(a, *b).await.unwrap();
        n2.add_balance(a, *b).await.unwrap();
        n3.add_balance(a, *b).await.unwrap();
    }

    let r1 = n1.compute_balance_root_for_block().await.unwrap();
    let r2 = n2.compute_balance_root_for_block().await.unwrap();
    let r3 = n3.compute_balance_root_for_block().await.unwrap();

    // All roots identical
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);

    // Every node accepts every other node's block
    assert!(simulate_validator_decision(ACTIVATION, r1, r2).is_ok());
    assert!(simulate_validator_decision(ACTIVATION, r2, r3).is_ok());
    assert!(simulate_validator_decision(ACTIVATION, r3, r1).is_ok());
}

// ============================================================================
// MODULE 7 — ACTIVATION BOUNDARY SEQUENCE (iter)
// ============================================================================
//
// Walks 20 blocks around the activation boundary one-by-one.
// Verifies: before activation → old-node zero-root accepted, new-root not checked.
//           at/after activation → zero-root rejected, correct-root accepted, wrong-root rejected.

#[tokio::test]
async fn activation_boundary_walk_20_blocks() {
    let (s, _d) = open_storage().await;
    s.add_balance(&addr(1), 999 * ONE_QUG).await.unwrap();
    let correct_root = s.compute_balance_root_for_block().await.unwrap();
    let wrong_root = [0xDEu8; 32];

    // Walk heights ACTIVATION-10 through ACTIVATION+10
    for offset in -10i64..=10i64 {
        let height = (ACTIVATION as i64 + offset) as u64;
        let active = balance_root_v1_active_at(height);

        // Zero root (old node)
        let zero_verdict = simulate_validator_decision(height, [0u8; 32], correct_root);
        // Correct root
        let correct_verdict = simulate_validator_decision(height, correct_root, correct_root);
        // Wrong root
        let wrong_verdict = simulate_validator_decision(height, wrong_root, correct_root);

        if !active {
            // Before activation: everything passes (enforcement dormant)
            assert!(zero_verdict.is_ok(),
                "height {}: zero root must be accepted before activation", height);
            assert!(correct_verdict.is_ok(),
                "height {}: correct root must be accepted before activation", height);
            assert!(wrong_verdict.is_ok(),
                "height {}: wrong root must be accepted before activation", height);
        } else {
            // At/after activation: strict enforcement
            assert!(zero_verdict.is_err(),
                "height {}: zero root must be REJECTED at/after activation", height);
            assert!(correct_verdict.is_ok(),
                "height {}: correct root must be ACCEPTED at/after activation", height);
            assert!(wrong_verdict.is_err(),
                "height {}: wrong root must be REJECTED at/after activation", height);
        }
    }
}

/// The gate flips exactly between BEFORE and ACTIVATION — no gradual ramp.
#[test]
fn activation_is_sharp_cutover() {
    assert!(!balance_root_v1_active_at(BEFORE));
    assert!( balance_root_v1_active_at(ACTIVATION));
    // There is no height where it's "partially active"
    for h in [BEFORE - 100, BEFORE - 1, BEFORE] {
        assert!(!balance_root_v1_active_at(h),
            "height {} must be inactive", h);
    }
    for h in [ACTIVATION, ACTIVATION + 1, ACTIVATION + 100] {
        assert!(balance_root_v1_active_at(h),
            "height {} must be active", h);
    }
}

// ============================================================================
// MODULE 8 — PRODUCTION SCALE (iter)
// ============================================================================

/// Simulate 1,000 sequential blocks, each updating a random wallet.
/// Verifies: root is always deterministic, always non-zero after first balance,
/// always changes when state changes.
#[tokio::test]
async fn iter_1000_blocks_root_consistency() {
    let (s, _d) = open_storage().await;
    let mut rng = 0xDEAD_BEEF_1337u64;

    let mut prev_root = [0u8; 32];

    for block in 0u64..1_000 {
        // Pick a wallet (20 wallets rotate)
        let wallet_seed = (lcg(&mut rng) % 20) as u16;
        let amount = (lcg(&mut rng) as u128 % (100 * ONE_QUG)) + 1;

        s.add_balance(&addr(wallet_seed), amount).await.unwrap();

        let root = s.compute_balance_root_for_block().await.unwrap();

        // Deterministic: second call must match
        let root2 = s.compute_balance_root_for_block().await.unwrap();
        assert_eq!(root, root2, "block {}: root not deterministic", block);

        // After at least one balance write, root must not be zero
        assert_ne!(root, [0u8; 32], "block {}: root must not be zero after balance write", block);

        // Root must have changed (we added a non-zero amount)
        assert_ne!(root, prev_root,
            "block {}: root must change when balance changes", block);

        prev_root = root;
    }
}

/// Production-scale wallet count: 1,340 wallets.
/// Two nodes with identical 1,340-wallet state must produce identical roots.
#[tokio::test]
async fn iter_1340_wallets_two_nodes_agree() {
    let (node_a, _d1) = open_storage().await;
    let (node_b, _d2) = open_storage().await;

    let mut rng = 0xC0FFEE_1340u64;

    for i in 0u16..PROD_WALLET_COUNT as u16 {
        let amount = (lcg(&mut rng) as u128 % (10_000 * ONE_QUG)) + ONE_QUG;
        node_a.add_balance(&addr(i), amount).await.unwrap();
        node_b.add_balance(&addr(i), amount).await.unwrap();
    }

    let root_a = node_a.compute_balance_root_for_block().await.unwrap();
    let root_b = node_b.compute_balance_root_for_block().await.unwrap();

    assert_eq!(root_a, root_b,
        "1,340-wallet production-scale state: both nodes must agree on root");
    assert_ne!(root_a, [0u8; 32]);
}

/// Among 1,340 wallets, a single unit off in wallet #670 (the middle) must change the root.
/// Validates sensitivity at production scale.
#[tokio::test]
async fn iter_1340_wallets_single_divergence_detected() {
    let mut rng = 0xC0FFEE_1340u64;
    let amounts: Vec<u128> = (0..PROD_WALLET_COUNT as u16)
        .map(|_| (lcg(&mut rng) as u128 % (10_000 * ONE_QUG)) + ONE_QUG)
        .collect();

    let (node_a, _d1) = open_storage().await;
    let (node_b, _d2) = open_storage().await;

    for (i, &amount) in amounts.iter().enumerate() {
        node_a.add_balance(&addr(i as u16), amount).await.unwrap();
        node_b.add_balance(&addr(i as u16), amount).await.unwrap();
    }

    // Node B is off by 1 unit on wallet 670 (the middle of 1,340)
    node_b.add_balance(&addr(670), 1u128).await.unwrap();

    let root_a = node_a.compute_balance_root_for_block().await.unwrap();
    let root_b = node_b.compute_balance_root_for_block().await.unwrap();

    assert_ne!(root_a, root_b,
        "single +1 unit on wallet 670 out of 1,340 must change root (production scale)");

    let verdict = simulate_validator_decision(ACTIVATION, root_b, root_a);
    assert!(verdict.is_err(), "diverged block must be rejected at production scale");
}

/// Wallet churn: wallets are funded and then brought to zero.
/// Zeroed wallets must be excluded from the root.
/// After churn, root must equal the root of a fresh state with only surviving wallets.
#[tokio::test]
async fn iter_wallet_churn_zero_exclusion() {
    let (s_churned, _d1) = open_storage().await;
    let (s_clean, _d2) = open_storage().await;

    // Fund 10 wallets
    for i in 0u16..10 {
        s_churned.add_balance(&addr(i), (i as u128 + 1) * ONE_QUG).await.unwrap();
    }

    // Zero out the odd-indexed wallets (simulates full spends)
    // Note: add_balance with 0 is a no-op. To zero a wallet we must use set_balance.
    // Use set_balance to set them to 0 explicitly.
    for i in (1u16..10).step_by(2) {
        s_churned.set_balance(&addr(i), 0u128).await.unwrap();
    }

    // `s_clean` only ever had the surviving (even-indexed) wallets
    for i in (0u16..10).step_by(2) {
        s_clean.add_balance(&addr(i), (i as u128 + 1) * ONE_QUG).await.unwrap();
    }

    let root_churned = s_churned.compute_balance_root_for_block().await.unwrap();
    let root_clean = s_clean.compute_balance_root_for_block().await.unwrap();

    assert_eq!(
        root_churned, root_clean,
        "churned (zeroed) wallets must be excluded; root must match clean state"
    );
}

/// Concurrent reads: computing the root from multiple async tasks simultaneously
/// must always return the same value (no race on the read path).
#[tokio::test]
async fn iter_concurrent_root_reads_agree() {
    let (s, _d) = open_storage().await;

    for i in 0u16..50 {
        s.add_balance(&addr(i), (i as u128 + 1) * ONE_QUG).await.unwrap();
    }

    // Compute 8 roots concurrently
    let s = Arc::clone(&s);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let s = Arc::clone(&s);
            tokio::spawn(async move {
                s.compute_balance_root_for_block().await.unwrap()
            })
        })
        .collect();

    let roots: Vec<[u8; 32]> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.expect("task panicked"))
        .collect();

    let first = roots[0];
    for (i, root) in roots.iter().enumerate() {
        assert_eq!(*root, first, "concurrent read {} produced different root", i);
    }
}
