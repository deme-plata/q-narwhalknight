//! `balance_root_v1` as a pure function.
//!
//! The producer writes this root into `BlockHeader.state_root` from height 17,742,000.
//! Measured on production 2026-10-01: the live root at height H equals the `state_root`
//! of header **H+1** — a header commits to the balance state *before* its own block.
//!
//! Extracted so that the node's own root (`StorageEngine::compute_balance_root_for_block`)
//! and a peer's balance SNAPSHOT (fresh-node bootstrap import, `q-api-server`
//! `bootstrap_verify`) are hashed by the very same code — the precondition for checking a
//! snapshot against the chain's own headers, the way SIGIL checks its state roots.
//!
//! Spec (BAL-001, NEVER change — headers on mainnet commit to it):
//!   * drop zero balances;
//!   * sort by raw 32-byte address;
//!   * leaf = BLAKE3(address || amount as u128 big-endian);
//!   * root = BLAKE3("balance_root_v1" || leaf_0 || leaf_1 || ...);
//!   * no non-zero balance at all -> root is 32 zero bytes.

use rayon::prelude::*;

/// Root of an arbitrary (address, amount) set. Duplicate addresses are the caller's
/// problem (a HashMap-backed set has none).
pub fn balance_root_v1<I>(entries: I) -> [u8; 32]
where
    I: IntoIterator<Item = ([u8; 32], u128)>,
{
    let mut sorted: Vec<([u8; 32], u128)> =
        entries.into_iter().filter(|(_, amount)| *amount > 0).collect();
    if sorted.is_empty() {
        return [0u8; 32];
    }
    sorted.par_sort_unstable_by_key(|(addr, _)| *addr);

    let leaf_hashes: Vec<[u8; 32]> = sorted
        .par_iter()
        .map(|(addr, amount)| {
            let mut leaf_hasher = blake3::Hasher::new();
            leaf_hasher.update(addr.as_slice());
            leaf_hasher.update(&amount.to_be_bytes()); // big-endian per spec — NEVER change
            *leaf_hasher.finalize().as_bytes()
        })
        .collect();

    let mut root_hasher = blake3::Hasher::new();
    root_hasher.update(b"balance_root_v1"); // domain separator — NEVER change
    for leaf in &leaf_hashes {
        root_hasher.update(leaf);
    }
    *root_hasher.finalize().as_bytes()
}

// Tests: crates/q-storage/tests/integrity_guards_tests.rs (the lib unit-test target does not
// compile at 1e36503e4, so in-module tests would never run).
