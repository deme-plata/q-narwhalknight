//! 2026-08-22: verifies the fix for a real custom-token read bug found during
//! the new-node data-integrity sweep. CF_TOKEN_BALANCES (the state-sync write
//! target for token balances, key = account(32)‖token(32)) is written
//! LITTLE-endian by its only writer, `StateApplicator::apply_balance_credit`/
//! `apply_balance_debit` (state_applicator.rs, `to_le_bytes()`). Three reader
//! fallbacks in lib.rs — `get_token_balance`, `load_token_balances`, and
//! `load_token_balances_for_wallet` — decoded that same CF BIG-endian, so any
//! custom-token balance sourced purely from CF_TOKEN_BALANCES (i.e. not yet
//! mirrored into CF_MANIFEST by a live API write) came back byte-swapped into
//! an astronomically wrong number. Fixed by matching the writer's endianness.
//!
//! All 3 tests are `#[ignore]`d: `StorageEngine::open()` on a bare tempdir hits
//! a pre-existing environment constraint (a winter-prover ZK trace-size assertion
//! panics on a too-small DB) already documented and worked around the same way by
//! `checkpoint_rocksdb_integration_tests.rs`. The fix itself is a one-line
//! endianness match against a single, unambiguous writer — verified by direct
//! code inspection rather than blocked on this pre-existing limitation.
//!
//! Run with: cargo test --package q-storage --test token_balance_endianness_tests -- --ignored

use q_storage::{CF_TOKEN_BALANCES, StorageEngine};

async fn open_test_storage() -> (StorageEngine, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().expect("failed to create tempdir");
    let node_id = [7u8; 32];
    let storage = StorageEngine::open(dir.path(), node_id)
        .await
        .expect("failed to open StorageEngine");
    (storage, dir)
}

/// Write directly into CF_TOKEN_BALANCES exactly as StateApplicator does:
/// key = account‖token, value = amount.to_le_bytes(). Bypasses the applicator
/// itself so this test is a pure reader-side regression check, independent of
/// any state_applicator.rs change.
fn seed_token_balance(storage: &StorageEngine, account: &[u8; 32], token: &[u8; 32], amount: u128) {
    let db = storage.get_rocks_db_handle().expect("RocksDB handle required for this test");
    let cf = db.cf_handle(CF_TOKEN_BALANCES).expect("CF_TOKEN_BALANCES must exist");
    let mut key = account.to_vec();
    key.extend_from_slice(token);
    db.put_cf(&cf, &key, &amount.to_le_bytes()).expect("seed write failed");
}

#[tokio::test]
#[ignore = "requires real blockchain environment (winter-prover ZK trace too small in tests) \
            — same StorageEngine::open() constraint checkpoint_rocksdb_integration_tests.rs \
            already documents and ignores for; the fix under test (lib.rs get_token_balance's \
            CF_TOKEN_BALANCES fallback) was verified by direct code inspection: the only writer \
            (StateApplicator::apply_balance_credit, state_applicator.rs) uses to_le_bytes(), and \
            this reader now matches with from_le_bytes()"]
async fn get_token_balance_reads_le_written_cf_token_balances_correctly() {
    let (storage, _dir) = open_test_storage().await;
    let account = [21u8; 32];
    let token = [22u8; 32]; // a custom (non-QUG, non-QUGUSD) token
    // Chosen to be unambiguous under a BE/LE mixup: byte-swapping this value
    // does NOT coincidentally equal itself, and lands nowhere near the
    // correct magnitude if the bug were still present.
    let amount: u128 = 123_456_789_012_345;

    seed_token_balance(&storage, &account, &token, amount);

    let read_back = storage.get_token_balance(&account, &token).await.unwrap();
    assert_eq!(read_back, amount,
        "get_token_balance must decode CF_TOKEN_BALANCES little-endian, matching \
         the only writer (StateApplicator) — a BE mismatch here returns a \
         byte-swapped, wildly wrong amount instead of the real balance");
}

#[tokio::test]
#[ignore = "requires real blockchain environment (winter-prover ZK trace too small in tests) \
            — see get_token_balance_reads_le_written_cf_token_balances_correctly above"]
async fn load_token_balances_reads_le_written_cf_token_balances_correctly() {
    let (storage, _dir) = open_test_storage().await;
    let account = [23u8; 32];
    let token = [24u8; 32];
    let amount: u128 = 987_654_321_098_765;

    seed_token_balance(&storage, &account, &token, amount);

    let all = storage.load_token_balances().await.unwrap();
    assert_eq!(all.get(&(account, token)).copied(), Some(amount),
        "load_token_balances (the all-wallets scan) must also decode LE, \
         matching the only writer");
}

#[tokio::test]
#[ignore = "requires real blockchain environment (winter-prover ZK trace too small in tests) \
            — see get_token_balance_reads_le_written_cf_token_balances_correctly above"]
async fn load_token_balances_for_wallet_reads_le_written_cf_token_balances_correctly() {
    let (storage, _dir) = open_test_storage().await;
    let account = [25u8; 32];
    let token = [26u8; 32];
    let amount: u128 = 555_000_111_222_333;

    seed_token_balance(&storage, &account, &token, amount);

    let per_wallet = storage.load_token_balances_for_wallet(&account).await.unwrap();
    assert_eq!(per_wallet.get(&token).copied(), Some(amount),
        "load_token_balances_for_wallet (the per-wallet prefix scan) must also \
         decode LE, matching the only writer");
}
