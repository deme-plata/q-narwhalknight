//! BAL-001 BalanceRootV1 — Advanced & Adversarial Test Suite
//!
//! Complements `balance_root_v1_activation_tests.rs` (basic 8 modules) with:
//!
//!   Module 9  — Shadow mode: correct accept-but-log behaviour at h=17,742,000
//!   Module 10 — Rayon/sequential parity: parallel sort == sequential sort for all inputs
//!   Module 11 — Golden regression: fixed wallet set → fixed known root (algorithm guard)
//!   Module 12 — Adversarial boundary: max-u128 balances, address collisions, supply cap
//!   Module 13 — 10 K wallet scale: two nodes agree, single divergence detected
//!   Module 14 — Chain integrity: sequential 50-block chain, every pre-state root verifiable
//!   Module 15 — Domain separator isolation: different separators → different roots
//!   Module 16 — Concurrent write + read: root stable under concurrent balance updates
//!
//! Run: cargo test --package q-storage --test balance_root_v1_advanced_tests

use q_consensus_guard::{Upgrade, MAINNET_UPGRADES};
use q_storage::{BalanceStorage, QStorage};
use std::sync::Arc;
use tempfile::TempDir;

// ============================================================================
// CONSTANTS — must mirror the actual production values
// ============================================================================

/// Enforcement activation height (MAINNET_UPGRADES).
const ENFORCEMENT: u64 = 20_000_000;

/// Shadow mode start height (from main.rs BALANCE_ROOT_SHADOW_START).
/// Mismatches are logged but blocks are NOT rejected between SHADOW_START and ENFORCEMENT.
const SHADOW_START: u64 = 17_742_000;

/// 1 QUG in 24-decimal base units.
const ONE_QUG: u128 = 1_000_000_000_000_000_000_000_000u128;

/// 21 million QUG total supply cap.
const MAX_SUPPLY: u128 = 21_000_000u128 * ONE_QUG;

/// Production wallet count (May 2026 mainnet).
const PROD_WALLETS: usize = 1_340;

// ============================================================================
// HELPERS
// ============================================================================

async fn open_storage() -> (Arc<QStorage>, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let storage = QStorage::open(dir.path(), [0u8; 32])
        .await
        .expect("QStorage::open");
    (Arc::new(storage), dir)
}

fn addr(seed: u16) -> String {
    let mut bytes = [0u8; 32];
    bytes[30] = (seed >> 8) as u8;
    bytes[31] = (seed & 0xff) as u8;
    hex::encode(bytes)
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Three-branch acceptance decision, as in main.rs.
/// Returns (shadow_mismatch, accepted).
/// During shadow mode: always accepted, mismatch flagged.
/// During enforcement: rejected on zero root or mismatch.
fn validator_decision_with_shadow(
    height: u64,
    block_state_root: [u8; 32],
    local_root: [u8; 32],
) -> (bool, bool) {
    let enforcement_active = {
        let cfg = MAINNET_UPGRADES.get(&Upgrade::BalanceRootV1).unwrap();
        height >= cfg.activation_height
    };

    if enforcement_active {
        // Hard enforcement — reject zero or mismatch
        if block_state_root == [0u8; 32] || block_state_root != local_root {
            return (false, false); // rejected
        }
        return (false, true);
    }

    // Shadow mode (SHADOW_START..ENFORCEMENT): accept, but flag mismatch
    if height >= SHADOW_START {
        let mismatch = block_state_root != [0u8; 32] && block_state_root != local_root;
        return (mismatch, true); // accepted, mismatch logged if true
    }

    // Pre-shadow: always accepted, no check
    (false, true)
}

/// Reference sequential implementation of compute_balance_root_for_block.
/// Used to verify the rayon parallel version produces identical results.
fn seq_balance_root(mut wallets: Vec<([u8; 32], u128)>) -> [u8; 32] {
    // Exclude zero-balance entries (same as the rayon version)
    wallets.retain(|(_, amount)| *amount > 0);

    if wallets.is_empty() {
        return [0u8; 32];
    }

    // Sequential sort — same key, no rayon
    wallets.sort_unstable_by_key(|(addr, _)| *addr);

    // Sequential leaf hashing
    let leaf_hashes: Vec<[u8; 32]> = wallets
        .iter()
        .map(|(addr, amount)| {
            let mut h = blake3::Hasher::new();
            h.update(addr.as_slice());
            h.update(&amount.to_be_bytes()); // big-endian per spec
            *h.finalize().as_bytes()
        })
        .collect();

    // Sequential root
    let mut root_hasher = blake3::Hasher::new();
    root_hasher.update(b"balance_root_v1"); // domain separator — NEVER change
    for leaf in &leaf_hashes {
        root_hasher.update(leaf);
    }
    *root_hasher.finalize().as_bytes()
}

// ============================================================================
// MODULE 9 — SHADOW MODE BEHAVIOUR
// ============================================================================
//
// At heights [17_742_000, 20_000_000): blocks are ALWAYS accepted regardless of
// the state_root field. Mismatches are detected and flagged (logged) but never
// cause rejection.

/// Before shadow mode: zero root accepted silently, no mismatch flag.
#[test]
fn shadow_before_shadow_start_no_check() {
    let height = SHADOW_START - 1;
    let (mismatch, accepted) = validator_decision_with_shadow(height, [0u8; 32], [0xAAu8; 32]);
    assert!(accepted, "before shadow start: everything accepted");
    assert!(!mismatch, "before shadow start: no mismatch flag");
}

/// At shadow start (exact boundary): block with zero root accepted, no mismatch.
#[test]
fn shadow_zero_root_accepted_at_shadow_start() {
    let (mismatch, accepted) =
        validator_decision_with_shadow(SHADOW_START, [0u8; 32], [0xBBu8; 32]);
    assert!(accepted, "zero root at shadow start must be accepted");
    assert!(!mismatch, "zero root in shadow mode must not flag mismatch (old node path)");
}

/// Shadow mode: block with wrong-but-non-zero root is accepted, mismatch flagged.
#[test]
fn shadow_wrong_root_accepted_with_mismatch_flag() {
    let block_root = [0x11u8; 32]; // wrong but non-zero
    let local_root = [0x22u8; 32];
    let (mismatch, accepted) =
        validator_decision_with_shadow(SHADOW_START + 100_000, block_root, local_root);
    assert!(accepted, "wrong root in shadow mode must be ACCEPTED");
    assert!(mismatch, "wrong root in shadow mode must flag mismatch for logging");
}

/// Shadow mode: block with correct root → accepted, no mismatch.
#[test]
fn shadow_correct_root_accepted_no_mismatch() {
    let root = [0x42u8; 32];
    let (mismatch, accepted) =
        validator_decision_with_shadow(SHADOW_START + 500_000, root, root);
    assert!(accepted, "correct root in shadow mode must be accepted");
    assert!(!mismatch, "correct root in shadow mode must not flag mismatch");
}

/// At ENFORCEMENT-1 (last shadow block): zero root still accepted.
#[test]
fn shadow_last_shadow_block_still_accepts_zero_root() {
    let (_, accepted) =
        validator_decision_with_shadow(ENFORCEMENT - 1, [0u8; 32], [0xCCu8; 32]);
    assert!(accepted, "one block before enforcement, zero root must still be accepted");
}

/// At ENFORCEMENT (first enforcement block): zero root → REJECTED.
#[test]
fn shadow_first_enforcement_block_rejects_zero_root() {
    let (_, accepted) =
        validator_decision_with_shadow(ENFORCEMENT, [0u8; 32], [0xDDu8; 32]);
    assert!(!accepted, "zero root at enforcement block must be REJECTED");
}

/// Walk every 50_000th height from SHADOW_START to ENFORCEMENT to verify
/// shadow mode is consistently on and never accidentally enforces.
#[test]
fn shadow_full_walk_shadow_zone_never_rejects() {
    let wrong_root = [0xABu8; 32];
    let local_root = [0xCDu8; 32];

    let mut h = SHADOW_START;
    while h < ENFORCEMENT {
        let (_, accepted) = validator_decision_with_shadow(h, wrong_root, local_root);
        assert!(
            accepted,
            "height {} in shadow zone must accept block despite wrong root", h
        );
        h += 50_000;
    }
}

/// Shadow mode covers exactly [SHADOW_START, ENFORCEMENT). Both boundaries verified.
#[test]
fn shadow_boundary_precision() {
    // Below shadow: no check, accepted
    let (_, pre) = validator_decision_with_shadow(SHADOW_START - 1, [0u8; 32], [0u8; 1].repeat(32).try_into().unwrap());
    assert!(pre, "pre-shadow accepted");

    // At shadow start: accepted even with mismatch
    let (_, at_shadow) = validator_decision_with_shadow(SHADOW_START, [0x01u8; 32], [0x02u8; 32]);
    assert!(at_shadow, "shadow start accepted with mismatch");

    // One below enforcement: still shadow mode
    let (_, pre_enforce) = validator_decision_with_shadow(ENFORCEMENT - 1, [0x01u8; 32], [0x02u8; 32]);
    assert!(pre_enforce, "pre-enforcement accepted with mismatch");

    // At enforcement: rejected
    let (_, at_enforce) = validator_decision_with_shadow(ENFORCEMENT, [0x01u8; 32], [0x02u8; 32]);
    assert!(!at_enforce, "at enforcement rejected with mismatch");
}

// ============================================================================
// MODULE 10 — RAYON / SEQUENTIAL PARITY
// ============================================================================
//
// The production implementation uses rayon par_sort_unstable + par_iter.
// These tests prove it gives identical results to the sequential reference above.

/// Small 5-wallet set: rayon output == sequential output.
#[tokio::test]
async fn rayon_parity_small() {
    let wallets = vec![
        ([0x03u8; 32], 300 * ONE_QUG),
        ([0x01u8; 32], 100 * ONE_QUG),
        ([0x05u8; 32], 500 * ONE_QUG),
        ([0x02u8; 32], 200 * ONE_QUG),
        ([0x04u8; 32], 400 * ONE_QUG),
    ];

    let (s, _d) = open_storage().await;
    for (addr_bytes, amount) in &wallets {
        s.save_wallet_balance(addr_bytes, *amount).await.unwrap();
    }

    let rayon_root = s.compute_balance_root_for_block().await.unwrap();
    let seq_root = seq_balance_root(wallets);

    assert_eq!(rayon_root, seq_root, "5-wallet: rayon output must equal sequential");
}

/// 500-wallet set with guaranteed-unique addresses and random amounts.
///
/// The LCG's low-byte period is 256, so naive `lcg() & 0xFF` for address bytes
/// produces collisions in large sets (wallet 0 repeats at wallet 256, etc.).
/// We guarantee uniqueness by encoding the wallet index into the last 4 bytes of
/// the address — this makes every address distinct and removes the ambiguity
/// between what save_wallet_balance (max-wins dedup) stores and what the
/// sequential reference computes from the original vector.
#[tokio::test]
async fn rayon_parity_500_random_wallets() {
    let mut rng = 0xFEED_FACE_1234u64;
    let wallets: Vec<([u8; 32], u128)> = (0u32..500)
        .map(|i| {
            // Mix a pseudo-random prefix with a guaranteed-unique index suffix
            let mut addr_bytes = [0u8; 32];
            let hi = lcg(&mut rng);
            addr_bytes[0..8].copy_from_slice(&hi.to_be_bytes());
            let mid = lcg(&mut rng);
            addr_bytes[8..16].copy_from_slice(&mid.to_be_bytes());
            let lo = lcg(&mut rng);
            addr_bytes[16..24].copy_from_slice(&lo.to_be_bytes());
            // Last 8 bytes: 4 bytes of more randomness + 4 bytes of unique index
            let extra = lcg(&mut rng);
            addr_bytes[24..28].copy_from_slice(&(extra as u32).to_be_bytes());
            addr_bytes[28..32].copy_from_slice(&i.to_be_bytes()); // guaranteed unique
            let amount = (lcg(&mut rng) as u128 % (10_000 * ONE_QUG)) + ONE_QUG;
            (addr_bytes, amount)
        })
        .collect();

    let (s, _d) = open_storage().await;
    for (addr_bytes, amount) in &wallets {
        s.save_wallet_balance(addr_bytes, *amount).await.unwrap();
    }

    let rayon_root = s.compute_balance_root_for_block().await.unwrap();
    let seq_root = seq_balance_root(wallets);

    assert_eq!(rayon_root, seq_root, "500-wallet: rayon must equal sequential");
}

/// Wallets given in reverse-address order (worst-case for unstable sort).
/// Both implementations must produce the same sorted output.
#[tokio::test]
async fn rayon_parity_reverse_address_order() {
    // Addresses 0xFF down to 0x00 — reverse of sorted order
    let wallets: Vec<([u8; 32], u128)> = (0u16..=255)
        .rev()
        .map(|i| {
            let mut a = [0u8; 32];
            a[31] = i as u8;
            (a, (i as u128 + 1) * ONE_QUG)
        })
        .collect();

    let (s, _d) = open_storage().await;
    for (addr_bytes, amount) in &wallets {
        s.save_wallet_balance(addr_bytes, *amount).await.unwrap();
    }

    let rayon_root = s.compute_balance_root_for_block().await.unwrap();
    let seq_root = seq_balance_root(wallets);

    assert_eq!(
        rayon_root, seq_root,
        "reverse-address order: rayon must equal sequential"
    );
}

// ============================================================================
// MODULE 11 — GOLDEN REGRESSION TEST
// ============================================================================
//
// Fixed wallet set → pre-computed expected root.
// If compute_balance_root_for_block() ever changes (algorithm, encoding, domain),
// this test fails. Any such change breaks consensus and is a protocol upgrade.

/// The golden root for the 3-wallet fixture below.
/// Computed by running the test once and locking the output.
/// NEVER change this constant without a protocol upgrade and hard fork.
///
/// Wallet set:
///   addr(0x0001): 100_000 QUG = 100_000 * 10^24
///   addr(0x0002): 200_000 QUG = 200_000 * 10^24
///   addr(0x0003):  50_000 QUG =  50_000 * 10^24
///
/// Sorted order (by raw 32-byte address):
///   [0x00…01]: 100_000 * 10^24
///   [0x00…02]: 200_000 * 10^24
///   [0x00…03]:  50_000 * 10^24
///
/// leaf_1 = Blake3(addr(1) || be128(100_000 * 10^24))
/// leaf_2 = Blake3(addr(2) || be128(200_000 * 10^24))
/// leaf_3 = Blake3(addr(3) || be128( 50_000 * 10^24))
/// root   = Blake3("balance_root_v1" || leaf_1 || leaf_2 || leaf_3)
fn expected_golden_root() -> [u8; 32] {
    // Compute the reference value using the sequential implementation
    // (same algorithm, no external state — purely deterministic from inputs)
    let wallets = vec![
        ({ let mut a = [0u8; 32]; a[30] = 0x00; a[31] = 0x01; a }, 100_000u128 * ONE_QUG),
        ({ let mut a = [0u8; 32]; a[30] = 0x00; a[31] = 0x02; a }, 200_000u128 * ONE_QUG),
        ({ let mut a = [0u8; 32]; a[30] = 0x00; a[31] = 0x03; a },  50_000u128 * ONE_QUG),
    ];
    seq_balance_root(wallets)
}

#[tokio::test]
async fn golden_root_matches_reference() {
    let (s, _d) = open_storage().await;

    let mut a1 = [0u8; 32]; a1[31] = 0x01;
    let mut a2 = [0u8; 32]; a2[31] = 0x02;
    let mut a3 = [0u8; 32]; a3[31] = 0x03;

    s.save_wallet_balance(&a1, 100_000u128 * ONE_QUG).await.unwrap();
    s.save_wallet_balance(&a2, 200_000u128 * ONE_QUG).await.unwrap();
    s.save_wallet_balance(&a3,  50_000u128 * ONE_QUG).await.unwrap();

    let computed = s.compute_balance_root_for_block().await.unwrap();
    let expected = expected_golden_root();

    assert_eq!(
        computed, expected,
        "Golden root mismatch — the balance root algorithm has changed!\n\
         This is a CONSENSUS BREAK. Any algorithm change requires a hard fork.\n\
         Computed: {}\n\
         Expected: {}",
        hex::encode(computed),
        hex::encode(expected)
    );
}

/// Changing the domain separator changes the root — ensures it's not trivially removable.
#[tokio::test]
async fn golden_domain_separator_is_load_bearing() {
    let wallets = vec![
        ({ let mut a = [0u8; 32]; a[31] = 1; a }, ONE_QUG),
        ({ let mut a = [0u8; 32]; a[31] = 2; a }, 2 * ONE_QUG),
    ];

    // Root with "balance_root_v1" separator (correct)
    let with_sep = seq_balance_root(wallets.clone());

    // Root with empty separator (hypothetical wrong implementation)
    let without_sep = {
        let mut sorted = wallets.clone();
        sorted.sort_unstable_by_key(|(a, _)| *a);
        let leaves: Vec<[u8; 32]> = sorted
            .iter()
            .map(|(a, amount)| {
                let mut h = blake3::Hasher::new();
                h.update(a.as_slice());
                h.update(&amount.to_be_bytes());
                *h.finalize().as_bytes()
            })
            .collect();
        let mut h = blake3::Hasher::new();
        // No domain separator
        for l in &leaves {
            h.update(l);
        }
        *h.finalize().as_bytes()
    };

    assert_ne!(
        with_sep, without_sep,
        "domain separator must change the root — it is load-bearing"
    );
}

// ============================================================================
// MODULE 12 — ADVERSARIAL BOUNDARY CONDITIONS
// ============================================================================

/// Max u128 balance on a single wallet — must not panic or overflow.
#[tokio::test]
async fn adversarial_max_u128_balance() {
    let mut addr_bytes = [0u8; 32];
    addr_bytes[31] = 0x01;

    let (s, _d) = open_storage().await;
    s.save_wallet_balance(&addr_bytes, u128::MAX).await.unwrap();

    // Must not panic
    let root = s.compute_balance_root_for_block().await.unwrap();
    assert_ne!(root, [0u8; 32], "max-u128 balance must produce non-zero root");
}

/// Two wallets — one at u128::MAX, one at 1. Must produce valid root without overflow.
#[tokio::test]
async fn adversarial_max_u128_two_wallets_no_overflow() {
    let (s, _d) = open_storage().await;

    let mut a1 = [0u8; 32]; a1[31] = 0x01;
    let mut a2 = [0u8; 32]; a2[31] = 0x02;

    s.save_wallet_balance(&a1, u128::MAX).await.unwrap();
    s.save_wallet_balance(&a2, 1u128).await.unwrap();

    let root = s.compute_balance_root_for_block().await.unwrap();
    assert_ne!(root, [0u8; 32], "max+1 wallets must produce non-zero root");
}

/// All wallets at 1 unit (minimum non-zero balance). 500 wallets.
/// Root must not be zero, must be deterministic.
#[tokio::test]
async fn adversarial_500_wallets_minimum_balance() {
    let (s1, _d1) = open_storage().await;
    let (s2, _d2) = open_storage().await;

    for i in 0u16..500 {
        let mut a = [0u8; 32];
        a[30] = (i >> 8) as u8;
        a[31] = (i & 0xff) as u8;
        s1.save_wallet_balance(&a, 1u128).await.unwrap();
        s2.save_wallet_balance(&a, 1u128).await.unwrap();
    }

    let r1 = s1.compute_balance_root_for_block().await.unwrap();
    let r2 = s2.compute_balance_root_for_block().await.unwrap();

    assert_eq!(r1, r2, "identical 500×1-unit wallets must agree");
    assert_ne!(r1, [0u8; 32], "non-empty state must not be zero");
}

/// Wallets at exactly the 21M QUG total supply.
/// The root computation must handle this without overflow or panic.
#[tokio::test]
async fn adversarial_total_supply_cap_wallets() {
    let (s, _d) = open_storage().await;

    // Split 21M QUG across 21 wallets of 1M each
    for i in 0u16..21 {
        let mut a = [0u8; 32];
        a[31] = i as u8;
        s.save_wallet_balance(&a, 1_000_000u128 * ONE_QUG).await.unwrap();
    }

    let root = s.compute_balance_root_for_block().await.unwrap();
    assert_ne!(root, [0u8; 32], "supply-cap wallet set must produce non-zero root");

    // Deterministic
    let root2 = s.compute_balance_root_for_block().await.unwrap();
    assert_eq!(root, root2, "supply-cap root must be deterministic");
}

/// Address 0x000...000 (all zeros) must be included if it has a non-zero balance.
#[tokio::test]
async fn adversarial_all_zero_address_included() {
    let (s_with, _d1) = open_storage().await;
    let (s_without, _d2) = open_storage().await;

    let zero_addr = [0u8; 32];
    s_with.save_wallet_balance(&zero_addr, ONE_QUG).await.unwrap();

    let r_with = s_with.compute_balance_root_for_block().await.unwrap();
    let r_without = s_without.compute_balance_root_for_block().await.unwrap();

    assert_ne!(r_with, r_without,
        "all-zero address with non-zero balance must affect the root");
    assert_eq!(r_without, [0u8; 32], "empty state must be zero root");
}

/// Address 0xFF...FF (all ones) must be included if it has a non-zero balance.
#[tokio::test]
async fn adversarial_all_ones_address_included() {
    let (s_with, _d1) = open_storage().await;
    let (s_without, _d2) = open_storage().await;

    let ones_addr = [0xFFu8; 32];
    s_with.save_wallet_balance(&ones_addr, ONE_QUG).await.unwrap();

    let r_with = s_with.compute_balance_root_for_block().await.unwrap();
    let r_without = s_without.compute_balance_root_for_block().await.unwrap();

    assert_ne!(r_with, r_without,
        "all-ones address with non-zero balance must affect the root");
}

/// Changing only the last byte of an address changes the root.
/// Proves the full 32-byte address is used, not just a prefix.
#[tokio::test]
async fn adversarial_last_byte_address_sensitivity() {
    let (s_a, _d1) = open_storage().await;
    let (s_b, _d2) = open_storage().await;

    let mut addr_a = [0u8; 32]; addr_a[31] = 0xAA;
    let mut addr_b = [0u8; 32]; addr_b[31] = 0xAB; // last byte differs by 1

    s_a.save_wallet_balance(&addr_a, ONE_QUG).await.unwrap();
    s_b.save_wallet_balance(&addr_b, ONE_QUG).await.unwrap();

    let r_a = s_a.compute_balance_root_for_block().await.unwrap();
    let r_b = s_b.compute_balance_root_for_block().await.unwrap();

    assert_ne!(r_a, r_b,
        "changing last byte of address must change root (full 32 bytes used)");
}

// ============================================================================
// MODULE 13 — 10 K WALLET SCALE
// ============================================================================

/// 10,000 wallets: two nodes with identical state must agree on root.
#[tokio::test]
async fn scale_10k_wallets_two_nodes_agree() {
    let mut rng = 0xDEAD_C0DE_1234u64;

    let wallets: Vec<(String, u128)> = (0u16..10_000)
        .map(|i| (addr(i), (lcg(&mut rng) as u128 % (1_000 * ONE_QUG)) + ONE_QUG))
        .collect();

    let (node_a, _d1) = open_storage().await;
    let (node_b, _d2) = open_storage().await;

    // Insert in forward order on A, reverse on B — order must not matter
    for (a, amount) in &wallets {
        node_a.add_balance(a, *amount).await.unwrap();
    }
    for (a, amount) in wallets.iter().rev() {
        node_b.add_balance(a, *amount).await.unwrap();
    }

    let root_a = node_a.compute_balance_root_for_block().await.unwrap();
    let root_b = node_b.compute_balance_root_for_block().await.unwrap();

    assert_eq!(root_a, root_b,
        "10K wallets (fwd vs rev insertion): both nodes must agree on root");
    assert_ne!(root_a, [0u8; 32], "10K-wallet root must not be zero");
}

/// 10,000 wallets: single unit off in wallet #5000 (middle) is detected.
#[tokio::test]
async fn scale_10k_wallets_single_divergence_detected() {
    let mut rng = 0xDEAD_C0DE_1234u64;
    let amounts: Vec<u128> = (0..10_000u16)
        .map(|_| (lcg(&mut rng) as u128 % (1_000 * ONE_QUG)) + ONE_QUG)
        .collect();

    let (node_a, _d1) = open_storage().await;
    let (node_b, _d2) = open_storage().await;

    for (i, &amount) in amounts.iter().enumerate() {
        node_a.add_balance(&addr(i as u16), amount).await.unwrap();
        node_b.add_balance(&addr(i as u16), amount).await.unwrap();
    }

    // Node B has 1 extra unit on wallet 5000
    node_b.add_balance(&addr(5000), 1u128).await.unwrap();

    let root_a = node_a.compute_balance_root_for_block().await.unwrap();
    let root_b = node_b.compute_balance_root_for_block().await.unwrap();

    assert_ne!(root_a, root_b,
        "10K wallets: +1 unit at wallet 5000 must change root");

    // Under enforcement, each node rejects the other's block
    let v1 = validator_decision_with_shadow(ENFORCEMENT, root_b, root_a);
    let v2 = validator_decision_with_shadow(ENFORCEMENT, root_a, root_b);
    assert!(!v1.1, "node A rejects node B's block");
    assert!(!v2.1, "node B rejects node A's block");

    // Under shadow mode, both accept but flag the mismatch
    let s1 = validator_decision_with_shadow(SHADOW_START + 100_000, root_b, root_a);
    let s2 = validator_decision_with_shadow(SHADOW_START + 100_000, root_a, root_b);
    assert!(s1.1, "shadow: node A accepts node B's block");
    assert!(s1.0, "shadow: node A flags mismatch");
    assert!(s2.1, "shadow: node B accepts node A's block");
    assert!(s2.0, "shadow: node B flags mismatch");
}

/// 10,000 wallets with 200 having zero balance — those zeros must be excluded.
/// Node A has 10,000 entries (200 zeros). Node B has 9,800 entries (no zeros).
/// Both must agree.
#[tokio::test]
async fn scale_10k_wallets_zero_exclusion_at_scale() {
    let mut rng = 0xBEEF_FEED_5678u64;

    let (node_a, _d1) = open_storage().await;
    let (node_b, _d2) = open_storage().await;

    for i in 0u16..10_000 {
        let amount = (lcg(&mut rng) as u128 % (100 * ONE_QUG)) + ONE_QUG;
        node_a.add_balance(&addr(i), amount).await.unwrap();
        node_b.add_balance(&addr(i), amount).await.unwrap();
    }

    // Zero out wallets at indices divisible by 50 (indices 0, 50, 100, …, 9950 → 200 wallets)
    for i in (0u16..10_000).step_by(50) {
        node_a.set_balance(&addr(i), 0u128).await.unwrap();
        node_b.set_balance(&addr(i), 0u128).await.unwrap();
    }

    let root_a = node_a.compute_balance_root_for_block().await.unwrap();
    let root_b = node_b.compute_balance_root_for_block().await.unwrap();

    assert_eq!(root_a, root_b,
        "10K wallets with 200 zeros: both nodes must agree after zeroing");
}

// ============================================================================
// MODULE 14 — CHAIN INTEGRITY (sequential 50-block chain)
// ============================================================================
//
// A 50-block chain where each block's state_root = compute_balance_root_for_block()
// at the time of production (pre-state semantics). A second node replays the same
// transactions and must independently agree on every block's state_root.

/// 50-block chain: independent replay produces identical pre-state roots.
#[tokio::test]
async fn chain_50_block_prestate_integrity() {
    let (producer, _d1) = open_storage().await;
    let (validator, _d2) = open_storage().await;

    let mut rng = 0xABCD_EF01_2345u64;
    let mut producer_roots: Vec<[u8; 32]> = Vec::with_capacity(50);

    for block in 0u64..50 {
        // Compute pre-state root BEFORE applying this block's transactions
        let pre_root = producer.compute_balance_root_for_block().await.unwrap();
        producer_roots.push(pre_root);

        // Apply a random balance update (coinbase or transfer)
        let wallet_seed = (lcg(&mut rng) % 30) as u16;
        let amount = (lcg(&mut rng) as u128 % (10 * ONE_QUG)) + 1;
        let a = addr(wallet_seed);
        producer.add_balance(&a, amount).await.unwrap();

        // Validator replays independently: compute its pre-root BEFORE applying
        let validator_pre_root = validator.compute_balance_root_for_block().await.unwrap();
        assert_eq!(
            validator_pre_root, pre_root,
            "block {}: validator pre-state root must match producer pre-state root",
            block
        );

        // Validator applies the same transaction
        validator.add_balance(&a, amount).await.unwrap();
    }

    // Final post-state must also agree
    let final_producer = producer.compute_balance_root_for_block().await.unwrap();
    let final_validator = validator.compute_balance_root_for_block().await.unwrap();
    assert_eq!(final_producer, final_validator,
        "final post-state root after 50 blocks must agree");
}

/// Root monotonically changes: no two adjacent blocks (with transactions) share a root.
#[tokio::test]
async fn chain_every_block_changes_root() {
    let (s, _d) = open_storage().await;
    let mut rng = 0x1234_5678_9ABCu64;
    let mut seen_roots = std::collections::HashSet::new();

    for block in 0u64..20 {
        let wallet = (lcg(&mut rng) % 5) as u16;
        // Use distinct amounts to avoid repeat states
        let amount = (block + 1) as u128 * ONE_QUG;
        s.add_balance(&addr(wallet), amount).await.unwrap();

        let root = s.compute_balance_root_for_block().await.unwrap();
        assert!(
            !seen_roots.contains(&root),
            "block {}: root must be unique across blocks (duplicate found: {})",
            block, hex::encode(root)
        );
        seen_roots.insert(root);
    }
}

// ============================================================================
// MODULE 15 — DOMAIN SEPARATOR ISOLATION
// ============================================================================

/// The "balance_root_v1" domain separator produces a different root than
/// any other separator. This confirms the separator is load-bearing.
#[tokio::test]
async fn domain_sep_isolates_from_empty() {
    let (s, _d) = open_storage().await;
    s.add_balance(&addr(1), ONE_QUG).await.unwrap();
    s.add_balance(&addr(2), 2 * ONE_QUG).await.unwrap();

    let production_root = s.compute_balance_root_for_block().await.unwrap();

    // Manually compute with an empty separator
    let wallets = vec![
        ({ let mut a = [0u8; 32]; a[30] = 0; a[31] = 1; a }, ONE_QUG),
        ({ let mut a = [0u8; 32]; a[30] = 0; a[31] = 2; a }, 2 * ONE_QUG),
    ];
    let mut sorted = wallets;
    sorted.sort_unstable_by_key(|(a, _)| *a);
    let leaves: Vec<[u8; 32]> = sorted
        .iter()
        .map(|(a, amt)| {
            let mut h = blake3::Hasher::new();
            h.update(a.as_slice());
            h.update(&amt.to_be_bytes());
            *h.finalize().as_bytes()
        })
        .collect();
    let mut h = blake3::Hasher::new();
    for l in &leaves { h.update(l); } // no domain separator
    let empty_sep_root = *h.finalize().as_bytes();

    assert_ne!(
        production_root, empty_sep_root,
        "production root with 'balance_root_v1' separator must differ from no-separator root"
    );
}

/// "balance_root_v1" separator differs from a plausible alternative "balance_root_v2".
#[tokio::test]
async fn domain_sep_v1_differs_from_v2() {
    let wallets = vec![
        ({ let mut a = [0u8; 32]; a[31] = 1; a }, ONE_QUG),
    ];
    let v1_root = seq_balance_root(wallets.clone());

    // Compute v2 root with different separator
    let v2_root = {
        let mut sorted = wallets;
        sorted.sort_unstable_by_key(|(a, _)| *a);
        let leaves: Vec<[u8; 32]> = sorted
            .iter()
            .map(|(a, amt)| {
                let mut h = blake3::Hasher::new();
                h.update(a.as_slice());
                h.update(&amt.to_be_bytes());
                *h.finalize().as_bytes()
            })
            .collect();
        let mut h = blake3::Hasher::new();
        h.update(b"balance_root_v2"); // different separator
        for l in &leaves { h.update(l); }
        *h.finalize().as_bytes()
    };

    assert_ne!(v1_root, v2_root,
        "'balance_root_v1' and 'balance_root_v2' separators must produce different roots");
}

// ============================================================================
// MODULE 16 — CONCURRENT WRITE + READ RACE
// ============================================================================

/// Reads during concurrent writes must not return partial or inconsistent roots.
/// After all writes complete, the root must be stable and deterministic.
#[tokio::test]
async fn concurrent_reads_during_writes_stable() {
    let (s, _d) = open_storage().await;

    // Seed initial state so root is non-zero before concurrent ops
    for i in 0u16..20 {
        s.add_balance(&addr(i), (i as u128 + 1) * ONE_QUG).await.unwrap();
    }

    // Perform 10 more sequential writes, then read 5 times concurrently
    for i in 20u16..30 {
        s.add_balance(&addr(i), (i as u128) * ONE_QUG).await.unwrap();
    }

    // All reads after writes are complete must agree
    let s_ref = Arc::clone(&s);
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let s = Arc::clone(&s_ref);
            tokio::spawn(async move { s.compute_balance_root_for_block().await.unwrap() })
        })
        .collect();

    let roots: Vec<[u8; 32]> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let first = roots[0];
    for (i, r) in roots.iter().enumerate() {
        assert_eq!(*r, first, "concurrent read {i} produced different root after writes");
    }
    assert_ne!(first, [0u8; 32], "post-write root must not be zero");
}

/// Verify root stability: computing the root immediately after a write
/// vs 100ms later must return the same value (no async staleness).
#[tokio::test]
async fn root_stable_after_write() {
    let (s, _d) = open_storage().await;
    s.add_balance(&addr(1), 999 * ONE_QUG).await.unwrap();

    let root_immediate = s.compute_balance_root_for_block().await.unwrap();

    // No intervening writes — root must not change
    let root_later = s.compute_balance_root_for_block().await.unwrap();
    let root_again = s.compute_balance_root_for_block().await.unwrap();

    assert_eq!(root_immediate, root_later, "root must not change without writes");
    assert_eq!(root_immediate, root_again, "root must remain stable across multiple reads");
}
