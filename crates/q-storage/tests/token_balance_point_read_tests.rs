//! v10.11.90 — Equivalence tests for `load_token_balances_for_wallet()`.
//!
//! The per-wallet prefix read replaces full-CF scans on prod's hot paths
//! (swap/mining confirmation + get_multi_token_balance). It MUST return exactly
//! what `load_token_balances()` returns filtered to one wallet, across every
//! storage quirk the full loader handles:
//! - CF_MANIFEST text keys (`token_balance_{wallet}_{token}`), LE u128 AND legacy LE u64
//! - CF_TOKEN_BALANCES binary 64-byte `wallet‖token` keys, BE u128 AND legacy BE u64
//! - manifest-is-authoritative merge (state sync fills only MISSING entries)
//! - QUGUSD excluded from the state-sync CF (the 172K-ghost rule)
//! - zero-amount state-sync entries skipped
//!
//! Run: cargo test --package q-storage --test token_balance_point_read_tests

use q_storage::{QStorage, CF_TOKEN_BALANCES};
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::TempDir;

async fn open_test_storage() -> (Arc<QStorage>, TempDir) {
    let dir = TempDir::new().expect("failed to create tempdir");
    let node_id = [0u8; 32];
    let storage = QStorage::open(dir.path(), node_id)
        .await
        .expect("failed to open QStorage");
    (Arc::new(storage), dir)
}

fn addr(seed: u8) -> [u8; 32] {
    [seed; 32]
}

/// Write a state-sync entry the way turbo sync does: binary wallet‖token key, BE value.
async fn put_state_sync(storage: &QStorage, wallet: [u8; 32], token: [u8; 32], amount: u128, legacy_u64: bool) {
    let mut key = Vec::with_capacity(64);
    key.extend_from_slice(&wallet);
    key.extend_from_slice(&token);
    let value: Vec<u8> = if legacy_u64 {
        (amount as u64).to_be_bytes().to_vec()
    } else {
        amount.to_be_bytes().to_vec()
    };
    storage.db_put(CF_TOKEN_BALANCES, &key, &value).await.expect("state-sync put");
}

/// The invariant under test: for every wallet, the point read equals the full
/// scan filtered to that wallet.
async fn assert_equivalent(storage: &QStorage, wallets: &[[u8; 32]]) {
    let full = storage.load_token_balances().await.expect("full load");
    for w in wallets {
        let expected: HashMap<[u8; 32], u128> = full
            .iter()
            .filter(|((fw, _), _)| fw == w)
            .map(|((_, t), amt)| (*t, *amt))
            .collect();
        let point = storage
            .load_token_balances_for_wallet(w)
            .await
            .expect("point load");
        assert_eq!(
            point, expected,
            "per-wallet load diverged from filtered full load for wallet {}",
            hex::encode(w)
        );
    }
}

#[tokio::test]
async fn test_manifest_only_equivalence() {
    let (storage, _dir) = open_test_storage().await;
    let (w1, w2) = (addr(0x11), addr(0x22));
    let (t1, t2) = (addr(0xa1), addr(0xa2));

    let mut balances: HashMap<([u8; 32], [u8; 32]), u128> = HashMap::new();
    balances.insert((w1, t1), 1_000_000_000_000_000_000_000_000u128); // 1.0 token, 24-dec
    balances.insert((w1, t2), 42u128);
    balances.insert((w2, t1), 7_500_000_000_000_000_000_000_000u128);
    storage.save_token_balances(&balances).await.expect("save manifest");

    assert_equivalent(&storage, &[w1, w2, addr(0x33)]).await; // 0x33 = wallet with nothing
}

#[tokio::test]
async fn test_state_sync_fill_and_manifest_priority() {
    let (storage, _dir) = open_test_storage().await;
    let w = addr(0x44);
    let (t_manifest, t_sync_only, t_conflict) = (addr(0xb1), addr(0xb2), addr(0xb3));

    // Manifest holds t_manifest and t_conflict (authoritative).
    let mut balances: HashMap<([u8; 32], [u8; 32]), u128> = HashMap::new();
    balances.insert((w, t_manifest), 500u128);
    balances.insert((w, t_conflict), 111u128);
    storage.save_token_balances(&balances).await.expect("save manifest");

    // State sync holds t_sync_only (fills in), t_conflict with a DIFFERENT stale
    // value (must lose to manifest), and a zero entry (must be skipped).
    put_state_sync(&storage, w, t_sync_only, 999u128, false).await;
    put_state_sync(&storage, w, t_conflict, 99_999u128, false).await;
    put_state_sync(&storage, w, addr(0xb4), 0u128, false).await;

    let point = storage.load_token_balances_for_wallet(&w).await.expect("point load");
    assert_eq!(point.get(&t_manifest), Some(&500u128));
    assert_eq!(point.get(&t_sync_only), Some(&999u128), "state sync must fill missing entries");
    assert_eq!(point.get(&t_conflict), Some(&111u128), "manifest must win over state sync");
    assert_eq!(point.get(&addr(0xb4)), None, "zero state-sync entries must be skipped");

    assert_equivalent(&storage, &[w]).await;
}

#[tokio::test]
async fn test_qugusd_excluded_from_state_sync() {
    let (storage, _dir) = open_test_storage().await;
    let w = addr(0x55);

    // A QUGUSD entry in the state-sync CF is the documented ghost source — both
    // loaders must refuse it.
    put_state_sync(&storage, w, q_types::QUGUSD_TOKEN_ADDRESS, 172_000u128, false).await;
    put_state_sync(&storage, w, addr(0xc1), 5u128, false).await;

    let point = storage.load_token_balances_for_wallet(&w).await.expect("point load");
    assert_eq!(point.get(&q_types::QUGUSD_TOKEN_ADDRESS), None, "QUGUSD must never load from state sync");
    assert_eq!(point.get(&addr(0xc1)), Some(&5u128));

    assert_equivalent(&storage, &[w]).await;
}

#[tokio::test]
async fn test_legacy_u64_values_and_prefix_isolation() {
    let (storage, _dir) = open_test_storage().await;
    // Adjacent wallets in key order: the prefix walk for w_mid must not leak
    // into w_lo (before) or w_hi (after).
    let (w_lo, w_mid, w_hi) = (addr(0x60), addr(0x61), addr(0x62));

    put_state_sync(&storage, w_lo, addr(0xd1), 10u128, true).await; // legacy 8-byte BE
    put_state_sync(&storage, w_mid, addr(0xd2), 20u128, true).await;
    put_state_sync(&storage, w_mid, addr(0xd3), u64::MAX as u128 + 5, false).await; // needs u128
    put_state_sync(&storage, w_hi, addr(0xd4), 30u128, true).await;

    let point = storage.load_token_balances_for_wallet(&w_mid).await.expect("point load");
    assert_eq!(point.len(), 2, "prefix walk leaked adjacent wallets: {:?}", point.keys().collect::<Vec<_>>());
    assert_eq!(point.get(&addr(0xd2)), Some(&20u128), "legacy u64 BE value misread");
    assert_eq!(point.get(&addr(0xd3)), Some(&(u64::MAX as u128 + 5)), "u128 BE value misread");

    assert_equivalent(&storage, &[w_lo, w_mid, w_hi]).await;
}

/// Real-database equivalence + timing. Skipped unless QNK_REAL_DB points at a
/// (COPY of a) node database — never point this at a live node's DB.
/// For every wallet found by the full loader, the point read must agree; prints
/// full-scan vs point-read timings so the speedup is measured, not assumed.
#[tokio::test]
async fn test_equivalence_and_timing_on_real_db_if_provided() {
    let Ok(db_path) = std::env::var("QNK_REAL_DB") else {
        eprintln!("QNK_REAL_DB not set — skipping real-DB equivalence test");
        return;
    };
    let node_id = [0u8; 32];
    let storage = QStorage::open(std::path::Path::new(&db_path), node_id)
        .await
        .expect("failed to open real DB copy");

    let t0 = std::time::Instant::now();
    let full = storage.load_token_balances().await.expect("full load");
    let full_elapsed = t0.elapsed();

    let mut wallets: Vec<[u8; 32]> = full.keys().map(|(w, _)| *w).collect();
    wallets.sort();
    wallets.dedup();
    assert!(!wallets.is_empty(), "real DB has no token balances — wrong path?");

    let t1 = std::time::Instant::now();
    for w in &wallets {
        let expected: HashMap<[u8; 32], u128> = full
            .iter()
            .filter(|((fw, _), _)| fw == w)
            .map(|((_, t), amt)| (*t, *amt))
            .collect();
        let point = storage
            .load_token_balances_for_wallet(w)
            .await
            .expect("point load");
        assert_eq!(point, expected, "diverged for wallet {}", hex::encode(w));
    }
    let per_wallet_avg = t1.elapsed() / wallets.len() as u32;

    println!(
        "REAL-DB EQUIVALENCE: {} balances / {} wallets ALL MATCH. full scan = {:?}, point read avg = {:?} ({}x)",
        full.len(),
        wallets.len(),
        full_elapsed,
        per_wallet_avg,
        (full_elapsed.as_micros().max(1) / per_wallet_avg.as_micros().max(1))
    );
}
