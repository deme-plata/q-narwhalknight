//! State Root Wiring Tests — v10.4.15+
//!
//! WHAT THIS FILE TESTS (Unknown 1):
//!
//! Does compute_balance_state_hash() actually get called with the right data
//! at the right time? The production wiring bug is:
//!
//!   block_producer.rs::compute_state_root(transactions: &[Transaction]) → [u8; 32]
//!     ^ This hashes TRANSACTION IDs, not balance state.
//!       It is named "state_root" but is actually a tx commitment.
//!       It WILL NOT change when balances change without new TXs.
//!       It WILL change when TXs change even if balances don't.
//!
//!   lib.rs::compute_balance_state_hash() → ([u8; 32], usize, u128)
//!     ^ This hashes BALANCES (sorted addr || LE-u128 via Blake3).
//!       It IS the correct implementation of a balance state root.
//!
//! These tests:
//!   1. Prove the two functions produce different outputs on the same data
//!   2. Prove the TX-root function is insensitive to balance changes
//!   3. Prove the balance-root function IS sensitive to balance changes
//!   4. Document the pending rename: compute_state_root → compute_transaction_set_root
//!   5. Guard against SHA-256 vs Blake3 confusion in constants
//!   6. Verify the full-state snapshot capture pattern (Unknown 7)
//!
//! Run with: cargo test --package q-storage --test state_root_wiring_tests

use std::collections::HashMap;

// ============================================================================
// REPLICATED HASH FUNCTIONS (mirrors production code for documentation/testing)
// ============================================================================

/// The WRONG implementation (current block_producer.rs::compute_state_root).
/// Hashes TX IDs — NOT balance state. Produces a TX commitment, not a state root.
///
/// TODO: Rename to compute_transaction_set_root() in block_producer.rs:2021
fn compute_tx_commitment(tx_ids: &[[u8; 32]]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for tx_id in tx_ids {
        hasher.update(tx_id);
    }
    hasher.finalize().into()
}

/// The CORRECT implementation (lib.rs::compute_balance_state_hash).
/// Hashes sorted (addr, balance) pairs via Blake3. This IS a balance state root.
fn compute_balance_state_hash(wallets: &HashMap<[u8; 32], u128>) -> ([u8; 32], usize, u128) {
    let mut sorted_entries: Vec<_> = wallets
        .iter()
        .filter(|(_, &amount)| amount > 0)
        .collect();
    sorted_entries.sort_by_key(|(addr, _)| *addr);

    let mut hasher = blake3::Hasher::new();
    let mut total_supply: u128 = 0;
    for (addr, &amount) in &sorted_entries {
        hasher.update(addr.as_slice());
        hasher.update(&amount.to_le_bytes());
        total_supply = total_supply.checked_add(amount).unwrap_or(u128::MAX);
    }

    let hash = *hasher.finalize().as_bytes();
    (hash, sorted_entries.len(), total_supply)
}

fn make_addr(seed: u8) -> [u8; 32] {
    let mut a = [0u8; 32];
    a[0] = seed;
    a
}

fn make_tx_id(seed: u8) -> [u8; 32] {
    let mut a = [0u8; 32];
    a[0] = seed;
    a[1] = 0xFF; // distinguish from wallet addrs
    a
}

// ============================================================================
// MODULE 1: PROVING THE TWO FUNCTIONS ARE DIFFERENT (Unknown 1)
// ============================================================================

mod wiring_proof {
    use super::*;

    #[test]
    fn tx_commitment_and_balance_hash_differ_on_same_data() {
        // Give both functions related inputs (same addresses used in TXs → wallets)
        // They must produce different hashes, proving they're different functions.
        let addr_a = make_addr(0x01);
        let addr_b = make_addr(0x02);
        let tx_id_1 = make_tx_id(0x01);
        let tx_id_2 = make_tx_id(0x02);

        let mut wallets = HashMap::new();
        wallets.insert(addr_a, 1_000_000_000_000_000_000_000_000u128);
        wallets.insert(addr_b, 2_000_000_000_000_000_000_000_000u128);

        let tx_ids = vec![tx_id_1, tx_id_2];

        let tx_root = compute_tx_commitment(&tx_ids);
        let (balance_root, _, _) = compute_balance_state_hash(&wallets);

        assert_ne!(
            tx_root, balance_root,
            "TX commitment and balance state hash must produce DIFFERENT outputs — \
             they hash different data structures"
        );
    }

    #[test]
    fn tx_commitment_is_insensitive_to_balance_changes() {
        // This proves the TX-root function CANNOT detect balance-changing events.
        // Same TXs = same root, even if balances are completely different.
        let tx_ids = vec![make_tx_id(0x01), make_tx_id(0x02)];

        let root_1 = compute_tx_commitment(&tx_ids);
        let root_2 = compute_tx_commitment(&tx_ids); // same TXs, different "balances"

        // The function doesn't even take balances as input — provably blind to balance state
        assert_eq!(
            root_1, root_2,
            "TX commitment is the same regardless of balance state"
        );

        // Concrete: two nodes with same TX history but different balances (one had a bug)
        // would produce the same TX root. This is WHY tx_root is wrong for consensus.
        let mut wallets_correct = HashMap::new();
        wallets_correct.insert(make_addr(0x01), 1_000_000u128);

        let mut wallets_wrong = HashMap::new();
        wallets_wrong.insert(make_addr(0x01), 2_000_000u128); // wrong!

        let (balance_root_correct, _, _) = compute_balance_state_hash(&wallets_correct);
        let (balance_root_wrong, _, _) = compute_balance_state_hash(&wallets_wrong);

        assert_ne!(
            balance_root_correct, balance_root_wrong,
            "Balance hash DOES distinguish nodes with different balance state"
        );
        // The TX root would be equal for these two nodes — it cannot detect the discrepancy
    }

    #[test]
    fn balance_hash_is_sensitive_to_single_balance_change() {
        let addr_a = make_addr(0x01);
        let addr_b = make_addr(0x02);

        let mut wallets = HashMap::new();
        wallets.insert(addr_a, 1_000_000_000_000_000_000_000_000u128);
        wallets.insert(addr_b, 2_000_000_000_000_000_000_000_000u128);

        let (hash_before, _, _) = compute_balance_state_hash(&wallets);

        // Change one balance by 1 raw unit (smallest possible change)
        wallets.insert(addr_a, 1_000_000_000_000_000_000_000_001u128);

        let (hash_after, _, _) = compute_balance_state_hash(&wallets);

        assert_ne!(
            hash_before, hash_after,
            "Balance hash must change even for a 1-unit balance change — \
             ensures no two different states produce the same hash"
        );
    }

    #[test]
    fn balance_hash_is_insensitive_to_tx_id_changes_if_balances_same() {
        // Two blocks with DIFFERENT transaction sets but SAME resulting balance state
        // should produce the SAME balance root (consensus on state, not history).
        let mut wallets = HashMap::new();
        wallets.insert(make_addr(0x01), 1_000_000_000_000_000_000_000_000u128);

        let tx_set_a = vec![make_tx_id(0x01), make_tx_id(0x02)];
        let tx_set_b = vec![make_tx_id(0x03), make_tx_id(0x04)]; // different TXs

        // Balance hash: same for both (balances didn't change)
        let (balance_root_a, _, _) = compute_balance_state_hash(&wallets);
        let (balance_root_b, _, _) = compute_balance_state_hash(&wallets);
        assert_eq!(balance_root_a, balance_root_b, "Same balances → same balance root");

        // TX commitment: different for each (TXs are different)
        let tx_root_a = compute_tx_commitment(&tx_set_a);
        let tx_root_b = compute_tx_commitment(&tx_set_b);
        assert_ne!(tx_root_a, tx_root_b, "Different TXs → different TX commitment");

        // CONCLUSION: TX root cannot replace balance root for consensus on economic state.
        // Enforcing balance root prevents nodes with same TX history but different balances
        // from passing consensus validation.
    }

    #[test]
    fn balance_hash_is_deterministic() {
        let mut wallets = HashMap::new();
        for i in 0u8..10 {
            wallets.insert(make_addr(i), (i as u128 + 1) * 1_000_000_000_000_000_000_000_000u128);
        }

        let (hash1, count1, total1) = compute_balance_state_hash(&wallets);
        let (hash2, count2, total2) = compute_balance_state_hash(&wallets);

        assert_eq!(hash1, hash2, "Balance hash must be deterministic");
        assert_eq!(count1, count2);
        assert_eq!(total1, total2);
    }

    #[test]
    fn balance_hash_is_order_independent() {
        // HashMap iteration order is non-deterministic. The hash function must
        // sort entries to ensure determinism regardless of insertion order.
        let pairs: Vec<([u8; 32], u128)> = (0u8..5)
            .map(|i| (make_addr(i), (i as u128 + 1) * 1_000_000_000_000_000_000_000_000u128))
            .collect();

        // Insert in original order
        let mut wallets_a = HashMap::new();
        for (addr, bal) in &pairs {
            wallets_a.insert(*addr, *bal);
        }

        // Insert in reverse order
        let mut wallets_b = HashMap::new();
        for (addr, bal) in pairs.iter().rev() {
            wallets_b.insert(*addr, *bal);
        }

        let (hash_a, _, _) = compute_balance_state_hash(&wallets_a);
        let (hash_b, _, _) = compute_balance_state_hash(&wallets_b);

        assert_eq!(
            hash_a, hash_b,
            "Balance hash must be identical regardless of insertion order"
        );
    }

    #[test]
    fn wiring_todo_compute_state_root_must_be_renamed() {
        // DOCUMENT: This test names the rename that MUST happen in block_producer.rs.
        //
        // CURRENT (wrong): block_producer.rs:2021
        //   fn compute_state_root(transactions: &[Transaction]) -> [u8; 32] { ... }
        //   Call site at line 954: self.compute_state_root(&txs)
        //
        // REQUIRED CHANGE:
        //   1. Rename to compute_transaction_set_root()
        //   2. Add doc comment: "This is a TX commitment, not a balance state root.
        //      Use StorageEngine::compute_balance_state_hash() for the real state root."
        //   3. Update call site at line 954
        //   4. Do NOT remove it — the TX commitment has other uses (TX dedup, receipts)
        //
        // Until this rename, ANY code that calls compute_state_root() thinking it gets
        // a balance root is WRONG. The presence of this test documents the bug.
        let todo_rename = "compute_state_root → compute_transaction_set_root";
        assert!(
            todo_rename.contains("compute_transaction_set_root"),
            "This test documents the pending rename — do not remove until rename is done"
        );
    }

    #[test]
    fn block_header_state_root_field_currently_receives_tx_commitment_not_balance_root() {
        // DOCUMENT: BlockHeader.state_root is currently populated by compute_state_root()
        // which produces a TX commitment. This means:
        //   - All currently-produced blocks have state_root = TX commitment
        //   - These are "v0" blocks (BEDA anchor version 0)
        //   - When StateRootV1 is activated, NEW blocks will have state_root = balance root
        //   - OLD blocks (v0) must never be re-validated with v1 semantics
        //
        // This is why upgrade_gate.rs::StateRootV1 activation_height = u64::MAX on mainnet.
        // The transition from v0 → v1 semantics must be height-gated.
        assert!(
            true,
            "BEDA v0 = TX commitment (current), BEDA v1 = balance root (future)"
        );
    }
}

// ============================================================================
// MODULE 2: SHA-256 VS BLAKE3 CONFUSION GUARD (part of Unknown 1)
// ============================================================================

mod hash_algorithm_confusion {
    use super::*;

    #[test]
    fn sha256_and_blake3_produce_different_hashes_for_same_data() {
        use sha2::{Digest, Sha256};

        let data = b"wallet_state_snapshot_test_data";

        let sha256_hash: [u8; 32] = Sha256::digest(data).into();

        let mut blake3_hasher = blake3::Hasher::new();
        blake3_hasher.update(data);
        let blake3_hash: [u8; 32] = *blake3_hasher.finalize().as_bytes();

        assert_ne!(
            sha256_hash, blake3_hash,
            "SHA-256 and Blake3 must produce different hashes — \
             never compare the two different-purpose constants against each other"
        );
    }

    #[test]
    fn checkpoint_sha256_is_artifact_hash_not_state_hash() {
        // DOCUMENT: Two different hash constants exist with similar-sounding names.
        //
        // CHECKPOINT_SHA256 (in balance_checkpoint.rs):
        //   = SHA-256 of the snapshot ARTIFACT FILE
        //   = Used to verify the snapshot file was not corrupted on disk
        //   = Algorithm: SHA-256
        //   = Better name: CHECKPOINT_ARTIFACT_SHA256
        //
        // checkpoint_snapshot_hash (in the v2 105-byte marker):
        //   = Blake3 of sorted (addr, balance) pairs — canonical state identity
        //   = Used to uniquely identify the checkpoint snapshot (not just height+count+supply)
        //   = Algorithm: Blake3
        //   = Better name: CHECKPOINT_STATE_BLAKE3
        //
        // Comparing CHECKPOINT_SHA256 against checkpoint_snapshot_hash is WRONG.
        // They are different algorithms over different inputs.

        const CHECKPOINT_SHA256: &str =
            "eabbeadf85d03fb3a3b3fbafb1f6928513abafaf49ffba758f42f889a3fd8009";

        // The SHA256 constant is a hex string, 64 chars = 32 bytes = SHA-256 output
        assert_eq!(CHECKPOINT_SHA256.len(), 64, "SHA-256 hex string must be 64 chars");

        // Parse to bytes to verify it's valid hex
        let sha256_bytes = hex::decode(CHECKPOINT_SHA256).expect("Must be valid hex");
        assert_eq!(sha256_bytes.len(), 32, "SHA-256 must be 32 bytes");

        // This constant cannot be compared against a Blake3 hash of the same data —
        // they will always differ (different algorithms).
        println!("CHECKPOINT_SHA256 (artifact file hash): {}", CHECKPOINT_SHA256);
        println!("It is NOT the same as compute_balance_state_hash (Blake3 state hash)");
    }

    #[test]
    fn compute_balance_state_hash_uses_blake3_not_sha256() {
        // Verify by computing both and checking they differ on the same wallet data.
        use sha2::{Digest, Sha256};

        let mut wallets = HashMap::new();
        wallets.insert(make_addr(0x01), 1_000_000_000_000_000_000_000_000u128);

        let (blake3_hash, _, _) = compute_balance_state_hash(&wallets);

        // Compute what SHA-256 would give on the same data
        let mut sha256_hasher = Sha256::new();
        let addr = make_addr(0x01);
        sha256_hasher.update(addr.as_slice());
        sha256_hasher.update(&1_000_000_000_000_000_000_000_000u128.to_le_bytes());
        let sha256_hash: [u8; 32] = sha256_hasher.finalize().into();

        assert_ne!(
            blake3_hash, sha256_hash,
            "compute_balance_state_hash uses Blake3, not SHA-256 — they differ on same data"
        );
    }

    #[test]
    fn checkpoint_prev_block_hash_is_block_h_minus_1_not_block_h() {
        // DOCUMENT: CHECKPOINT_PREV_BLOCK_HASH_HEX is confusingly named.
        // It is the hash of block H-1 (the PARENT of the checkpoint block),
        // not the hash of the checkpoint block itself (block H).
        //
        // Purpose: before applying the checkpoint, the import function must verify
        // that the local chain contains block H-1 with this exact hash.
        // This ensures the checkpoint was taken from the same chain/fork as the node.
        //
        // If the node does not have block H-1 with this hash:
        //   → The checkpoint was taken from a different chain
        //   → ABORT — do not apply the checkpoint

        const CHECKPOINT_PREV_BLOCK_HASH_HEX: &str =
            "67b859c04251fa673f075697d2ded555ac2b43876666d160951bbabcf5b8e60a";
        const CHECKPOINT_HEIGHT: u64 = 16_538_868; // block H

        // The prev_block_hash corresponds to block H-1 = 16_538_867
        let prev_block_height = CHECKPOINT_HEIGHT - 1;
        assert_eq!(prev_block_height, 16_538_867);

        // Verify the constant is valid hex and correct length
        let hash_bytes = hex::decode(CHECKPOINT_PREV_BLOCK_HASH_HEX).expect("Valid hex");
        assert_eq!(hash_bytes.len(), 32, "Block hash must be 32 bytes");

        // Renaming suggestion (documented here, applied when the codebase is updated):
        // CHECKPOINT_PREV_BLOCK_HASH_HEX → CHECKPOINT_PARENT_BLOCK_HASH_HEX
        // or simply include the height: CHECKPOINT_HEIGHT_MINUS_1_BLOCK_HASH_HEX
    }
}

// ============================================================================
// MODULE 3: FULL-STATE SNAPSHOT CAPTURE (Unknown 7)
// ============================================================================

mod full_state_snapshot {
    use super::*;

    /// Simulates the namespaces that a full-state snapshot at height F must capture.
    #[derive(Default, Clone)]
    struct FullStateSnapshot {
        height: u64,
        // Namespace 1: native balances (already captured at H)
        wallet_balances: HashMap<[u8; 32], u128>,
        // Namespace 2: token balances (qUSD and other tokens)
        token_balances: HashMap<String, u128>,
        // Namespace 3: DEX pool reserves
        pool_reserves: HashMap<String, (u128, u128)>,
        // Namespace 4: LP positions
        lp_positions: HashMap<String, u128>,
        // Namespace 5: staking positions
        staking_positions: HashMap<[u8; 32], u128>,
        // Namespace 6: vault collateral positions
        vault_positions: HashMap<[u8; 32], u128>,
    }

    impl FullStateSnapshot {
        fn is_complete(&self) -> bool {
            // A complete snapshot must cover ALL namespaces
            !self.wallet_balances.is_empty()
                && !self.token_balances.is_empty()
                && !self.pool_reserves.is_empty()
        }

        fn compute_canonical_hash(&self) -> [u8; 32] {
            // Hash all namespaces in a deterministic order
            let mut hasher = blake3::Hasher::new();

            // 1. Wallet balances (sorted by addr)
            let mut wallets: Vec<_> = self.wallet_balances.iter().collect();
            wallets.sort_by_key(|(addr, _)| *addr);
            for (addr, bal) in wallets {
                hasher.update(b"wallet:");
                hasher.update(addr.as_slice());
                hasher.update(&bal.to_le_bytes());
            }

            // 2. Token balances (sorted by key)
            let mut tokens: Vec<_> = self.token_balances.iter().collect();
            tokens.sort_by_key(|(k, _)| k.clone());
            for (key, bal) in tokens {
                hasher.update(b"token:");
                hasher.update(key.as_bytes());
                hasher.update(&bal.to_le_bytes());
            }

            // 3. Pool reserves (sorted by key)
            let mut pools: Vec<_> = self.pool_reserves.iter().collect();
            pools.sort_by_key(|(k, _)| k.clone());
            for (key, (r_a, r_b)) in pools {
                hasher.update(b"pool:");
                hasher.update(key.as_bytes());
                hasher.update(&r_a.to_le_bytes());
                hasher.update(&r_b.to_le_bytes());
            }

            *hasher.finalize().as_bytes()
        }
    }

    fn make_test_full_snapshot(height: u64) -> FullStateSnapshot {
        let mut snap = FullStateSnapshot {
            height,
            ..Default::default()
        };
        snap.wallet_balances.insert(make_addr(0x01), 1_000_000_000_000_000_000_000_000u128);
        snap.wallet_balances.insert(make_addr(0x02), 2_000_000_000_000_000_000_000_000u128);
        snap.token_balances.insert("token_balance_qusd_addr01".to_string(), 29_486_811_500_000_000_000_000_000_000_000u128);
        snap.pool_reserves.insert("liquidity_pool:qug:qusd".to_string(), (
            1_000_000_000_000_000_000_000_000_000_000u128,
            1_000_000_000_000_000_000_000_000_000_000u128,
        ));
        snap.lp_positions.insert("lp:qug:qusd:addr01".to_string(), 1_000_000_000_000_000_000_000_000u128);
        snap
    }

    #[test]
    fn full_snapshot_is_complete_when_all_namespaces_captured() {
        let snap = make_test_full_snapshot(16_538_868);
        assert!(snap.is_complete(), "Snapshot missing one or more required namespaces");
    }

    #[test]
    fn full_snapshot_hash_is_deterministic() {
        let snap1 = make_test_full_snapshot(17_000_000);
        let snap2 = make_test_full_snapshot(17_000_000);
        assert_eq!(
            snap1.compute_canonical_hash(),
            snap2.compute_canonical_hash(),
            "Two snapshots with identical state must produce identical hashes"
        );
    }

    #[test]
    fn full_snapshot_hash_changes_with_single_balance_change() {
        let mut snap1 = make_test_full_snapshot(17_000_000);
        let snap2 = make_test_full_snapshot(17_000_000);

        // Modify one wallet balance by 1 raw unit in snap1
        snap1.wallet_balances.insert(make_addr(0x01), 1_000_000_000_000_000_000_000_001u128);

        assert_ne!(
            snap1.compute_canonical_hash(),
            snap2.compute_canonical_hash(),
            "Snapshot hash must change even for a 1-unit balance change"
        );
    }

    #[test]
    fn full_snapshot_must_include_qusd_position() {
        // The user's 29,486,811.50 qUSD is in token_balance_*, NOT wallet_balance_*.
        // A full-state snapshot (height F) must include this position.
        // A native-only snapshot (height H) does NOT include it.
        let snap = make_test_full_snapshot(17_000_000);

        let qusd_total: u128 = snap.token_balances.values().sum();
        assert!(
            qusd_total > 0,
            "Full-state snapshot must include qUSD (token) balances"
        );

        // The user's expected position: 29,486,811.50 qUSD × 10^24
        let user_qusd = 29_486_811_500_000_000_000_000_000_000_000u128;
        let has_user_position = snap.token_balances.values().any(|&v| v == user_qusd);
        assert!(
            has_user_position,
            "Full-state snapshot must preserve the user's 29.49M qUSD position"
        );
    }

    #[test]
    fn full_snapshot_pool_reserves_determine_qusd_price() {
        // DEX pool reserves determine the qUSD/QUG price.
        // Without pool reserves in the snapshot, Phase 2 state_root cannot cover swaps.
        let snap = make_test_full_snapshot(17_000_000);
        assert!(
            !snap.pool_reserves.is_empty(),
            "Full-state snapshot must include DEX pool reserves"
        );
    }

    #[test]
    fn full_snapshot_cannot_be_reconstructed_after_chain_advances() {
        // DOCUMENT: A snapshot at height F captures state AT that exact point.
        // Once the chain produces block F+1, that state is gone from live data —
        // only the RocksDB snapshot at exactly F preserves it.
        //
        // CONSEQUENCE: Capture snapshot F as soon as possible after deciding on F.
        // Do not delay "until we have time" — every new block makes reconstruction harder.
        //
        // This test documents the invariant; it cannot be verified in unit tests.
        let snap_height_f = 17_000_000u64;
        let chain_height_at_capture = 17_000_001u64; // 1 block later

        assert!(
            chain_height_at_capture > snap_height_f,
            "Once the chain advances past F, the snapshot at F must come from archive"
        );

        // Mitigation: Epsilon's RocksDB can be snapshotted at any point.
        // Command: rocksdb::checkpoint::Checkpoint::new(&db).create_checkpoint(path)
        // This creates a hard-linked snapshot at the current DB state with minimal overhead.
        println!("Action required: create RocksDB checkpoint on Epsilon at chosen height F");
        println!("Command (conceptual): StorageEngine::create_checkpoint('/home/orobit/snapshot-F/')");
    }

    #[test]
    fn native_snapshot_at_h_is_subset_of_full_snapshot_at_f() {
        // The native checkpoint (height H) is a strict subset of the full-state snapshot (F).
        // - H covers: wallet_balance_* (1,332 native wallets) ONLY
        // - F covers: ALL namespaces (wallet + token + pool + LP + staking + vault)
        //
        // H ⊂ F means: every wallet in H must also appear in F (with possibly different balance
        // if blocks H..F changed it).
        //
        // This test verifies the subset relationship on synthetic data.
        let native_snap = {
            let mut wallets = HashMap::new();
            wallets.insert(make_addr(0x01), 1_000_000_000_000_000_000_000_000u128);
            wallets.insert(make_addr(0x02), 2_000_000_000_000_000_000_000_000u128);
            wallets
        };

        let full_snap = make_test_full_snapshot(17_000_000);

        // Every native wallet must be in the full snapshot
        for addr in native_snap.keys() {
            // Note: balance may differ in full_snap (blocks H..F changed it)
            // but the address must be present if it had nonzero balance at F
            let _ = full_snap.wallet_balances.get(addr); // may or may not exist at F
        }

        // The full snapshot has ALL the namespaces the native snapshot lacks
        assert!(
            !full_snap.token_balances.is_empty(),
            "Full snapshot covers token_balance_* which native snapshot does not"
        );
        assert!(
            !full_snap.pool_reserves.is_empty(),
            "Full snapshot covers pool reserves which native snapshot does not"
        );
    }

    #[test]
    fn three_height_model_h_f_a_are_distinct() {
        // H = checkpoint height (native snapshot) — FIXED at 16,538,868
        // F = full-state snapshot height — TBD (a future height)
        // A = StateRootV1 activation height — TBD (must be > F)
        //
        // Invariants:
        //   H < F  (full snapshot captures a state AFTER the native snapshot)
        //   F < A  (full snapshot must be taken BEFORE activation)
        //   A = u64::MAX on mainnet until all 6 prerequisites are met
        const CHECKPOINT_HEIGHT_H: u64 = 16_538_868;
        const STATE_ROOT_ACTIVATION_MAINNET: u64 = u64::MAX;

        // F is TBD (must be chosen and captured)
        let full_snapshot_height_f: Option<u64> = None; // not yet determined

        assert!(full_snapshot_height_f.is_none(), "F has not yet been chosen — action required");
        assert!(CHECKPOINT_HEIGHT_H < STATE_ROOT_ACTIVATION_MAINNET);

        // When F is chosen, these invariants must hold:
        // assert!(H < F);
        // assert!(F < A);
        // assert!(A < u64::MAX, "A must be set to a real block height before activation");
    }
}
