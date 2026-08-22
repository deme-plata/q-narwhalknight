//! 2026-08-13: verifies the fix for the "new nodes don't agree on balances with
//! Epsilon" root cause — CF_TOKEN_BALANCES (written by the bulk/turbo-sync
//! StateProcessor pipeline) and the legacy wallet_balance_* storage (read by
//! compute_balance_root_for_block / /api/v1/integrity/balance-root /
//! total_supply_qug) were completely disconnected. state_applicator.rs now
//! bridges native-QUG BalanceCredit/BalanceDebit into both.
//!
//! Written as a standalone integration test (not an embedded `#[cfg(test)]`
//! unit test) because q-storage's existing unit-test aggregate has 68
//! pre-existing, unrelated compile errors that block the whole `--lib` test
//! binary — integration tests under tests/ compile as independent binaries,
//! so this can run regardless of that separate, pre-existing problem.
//!
//! Run with: cargo test --package q-storage --test balance_dual_storage_bridge_tests

use q_storage::{StateApplicator, CF_TOKEN_BALANCES, CF_MANIFEST, STATE_SYNC_COLUMN_FAMILIES};
use q_types::{StateChange, QUG_TOKEN_ADDRESS};
use rocksdb::DB;
use std::sync::Arc;

fn create_test_db(name: &str) -> Arc<DB> {
    let dir = std::env::temp_dir().join(format!(
        "qnk-bridge-test-{}-{}-{}",
        name,
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));

    let mut opts = rocksdb::Options::default();
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);

    // Include CF_MANIFEST — the bridge writes there, and it is not in
    // STATE_SYNC_COLUMN_FAMILIES (a real production DB always has it via
    // save_wallet_balance, a bare STATE_SYNC_COLUMN_FAMILIES-only DB would not).
    let mut cf_names: Vec<&str> = STATE_SYNC_COLUMN_FAMILIES.to_vec();
    cf_names.push(CF_MANIFEST);
    let cfs: Vec<rocksdb::ColumnFamilyDescriptor> = cf_names
        .iter()
        .map(|n| rocksdb::ColumnFamilyDescriptor::new(*n, rocksdb::Options::default()))
        .collect();

    Arc::new(DB::open_cf_descriptors(&opts, dir, cfs).unwrap())
}

fn read_token_balance(db: &Arc<DB>, account: &[u8; 32], token: &[u8; 32]) -> u128 {
    let cf = db.cf_handle(CF_TOKEN_BALANCES).unwrap();
    let mut key = account.to_vec();
    key.extend_from_slice(token);
    db.get_cf(&cf, &key).unwrap()
        .map(|v| u128::from_le_bytes(v[..16].try_into().unwrap()))
        .unwrap_or(0)
}

fn read_legacy_wallet_balance(db: &Arc<DB>, account: &[u8; 32]) -> u128 {
    let cf = db.cf_handle(CF_MANIFEST).unwrap();
    let key = format!("wallet_balance_{}", hex::encode(account));
    db.get_cf(&cf, key.as_bytes()).unwrap()
        .map(|v| u128::from_le_bytes(v[..16].try_into().unwrap()))
        .unwrap_or(0)
}

#[test]
fn qug_credit_lands_in_both_stores() {
    let db = create_test_db("credit");
    let applicator = StateApplicator::new(db.clone(), false).with_legacy_bridge(true);
    let account = [11u8; 32];

    applicator.apply_changes(&[StateChange::BalanceCredit {
        account, token: QUG_TOKEN_ADDRESS, amount: 750,
    }], 1).unwrap();

    assert_eq!(read_token_balance(&db, &account, &QUG_TOKEN_ADDRESS), 750,
        "CF_TOKEN_BALANCES side (unchanged behavior) must still work");
    assert_eq!(read_legacy_wallet_balance(&db, &account), 750,
        "legacy wallet_balance_* — what /api/v1/integrity/balance-root actually reads — \
         must now ALSO reflect the credit. This is the fix under test.");
}

#[test]
fn qug_debit_lands_in_both_stores() {
    let db = create_test_db("debit");
    let applicator = StateApplicator::new(db.clone(), false).with_legacy_bridge(true);
    let account = [12u8; 32];

    applicator.apply_changes(&[StateChange::BalanceCredit {
        account, token: QUG_TOKEN_ADDRESS, amount: 2000,
    }], 1).unwrap();
    applicator.apply_changes(&[StateChange::BalanceDebit {
        account, token: QUG_TOKEN_ADDRESS, amount: 800,
    }], 2).unwrap();

    assert_eq!(read_token_balance(&db, &account, &QUG_TOKEN_ADDRESS), 1200);
    assert_eq!(read_legacy_wallet_balance(&db, &account), 1200,
        "debit must be mirrored into legacy storage too, not just credits");
}

#[test]
fn non_qug_token_never_touches_legacy_wallet_balance() {
    let db = create_test_db("nonqug");
    let applicator = StateApplicator::new(db.clone(), false).with_legacy_bridge(true);
    let account = [13u8; 32];
    let custom_token = [77u8; 32]; // definitely not QUG_TOKEN_ADDRESS

    applicator.apply_changes(&[StateChange::BalanceCredit {
        account, token: custom_token, amount: 4242,
    }], 1).unwrap();

    assert_eq!(read_token_balance(&db, &account, &custom_token), 4242);
    assert_eq!(read_legacy_wallet_balance(&db, &account), 0,
        "the bridge is QUG-specific by design — a custom/LP token credit must \
         not leak into wallet_balance_*");
}

/// The real-world scenario this fix exists for: a node whose history flowed
/// through bulk/turbo-sync already has CF_TOKEN_BALANCES ahead of the legacy
/// store (that gap IS the divergence measured live on qnk-wedge-v96 — 762,365
/// QUG behind Epsilon and growing). The bridge must apply new deltas on top of
/// whatever's currently on disk, converging the gap going forward, not assume
/// a synced starting point (which would silently re-introduce fresh drift).
#[test]
fn bridge_converges_from_an_already_diverged_starting_point() {
    let db = create_test_db("diverged");
    let applicator = StateApplicator::new(db.clone(), false).with_legacy_bridge(true);
    let account = [14u8; 32];

    {
        let cf = db.cf_handle(CF_TOKEN_BALANCES).unwrap();
        let mut key = account.to_vec();
        key.extend_from_slice(&QUG_TOKEN_ADDRESS);
        db.put_cf(&cf, &key, &50_000u128.to_le_bytes()).unwrap();

        let mcf = db.cf_handle(CF_MANIFEST).unwrap();
        let legacy_key = format!("wallet_balance_{}", hex::encode(&account));
        db.put_cf(&mcf, legacy_key.as_bytes(), &10_000u128.to_le_bytes()).unwrap();
    }

    applicator.apply_changes(&[StateChange::BalanceCredit {
        account, token: QUG_TOKEN_ADDRESS, amount: 500,
    }], 1).unwrap();

    assert_eq!(read_token_balance(&db, &account, &QUG_TOKEN_ADDRESS), 50_500);
    assert_eq!(read_legacy_wallet_balance(&db, &account), 10_500,
        "must apply the delta on top of whatever legacy currently holds — \
         the whole point is to stop the gap from growing further, starting now");
}

/// 2026-08-22 P0 redesign: the bridge must be OFF by default (the original
/// 2026-08-13 version was unconditional). A caller that shares StateApplicator
/// with another legacy-writing path (e.g. BalanceConsensusEngine, gossip
/// block-pack apply) must get the OFF default so it does not double-apply.
#[test]
fn bridge_off_by_default_matches_shared_engine_caller() {
    let db = create_test_db("default-off");
    let applicator = StateApplicator::new(db.clone(), false); // no .with_legacy_bridge
    let account = [15u8; 32];

    applicator.apply_changes(&[StateChange::BalanceCredit {
        account, token: QUG_TOKEN_ADDRESS, amount: 999,
    }], 1).unwrap();

    assert_eq!(read_token_balance(&db, &account, &QUG_TOKEN_ADDRESS), 999,
        "CF_TOKEN_BALANCES must still be written regardless of the bridge flag");
    assert_eq!(read_legacy_wallet_balance(&db, &account), 0,
        "with the bridge off (the safe default), legacy must stay untouched — \
         a caller that ALSO runs BalanceConsensusEngine over the same blocks \
         relies on this to avoid crediting wallet_balance_* twice");
}

/// Simulates the confirmed-unsafe path from the 2026-08-22 sweep
/// (main.rs apply_block_pack with Some(balance_engine)): BOTH the state
/// pipeline (via StateApplicator, bridge OFF) and a stand-in "legacy engine"
/// write wallet_balance_* for the same coinbase credit. With the bridge off,
/// StateApplicator contributes NOTHING to legacy, so only the engine's write
/// lands — exactly once, matching pre-bridge behavior on that path.
#[test]
fn bridge_off_prevents_double_credit_when_a_second_writer_shares_the_block() {
    let db = create_test_db("shared-engine");
    let applicator = StateApplicator::new(db.clone(), false); // bridge_legacy=false, as turbo_sync
                                                               // now sets when balance_engine is Some
    let account = [16u8; 32];
    let reward = 1_000u128;

    // 1) StateApplicator processes the block (bridge OFF — this is the fix)
    applicator.apply_changes(&[StateChange::BalanceCredit {
        account, token: QUG_TOKEN_ADDRESS, amount: reward,
    }], 1).unwrap();

    // 2) The stand-in "legacy engine" (BalanceConsensusEngine in production)
    //    independently credits the SAME coinbase reward into legacy storage —
    //    this is real, pre-existing behavior on the Some(engine) path.
    {
        let mcf = db.cf_handle(CF_MANIFEST).unwrap();
        let key = format!("wallet_balance_{}", hex::encode(&account));
        let current = db.get_cf(&mcf, key.as_bytes()).unwrap()
            .map(|v| u128::from_le_bytes(v[..16].try_into().unwrap()))
            .unwrap_or(0);
        db.put_cf(&mcf, key.as_bytes(), &(current + reward).to_le_bytes()).unwrap();
    }

    assert_eq!(read_legacy_wallet_balance(&db, &account), reward,
        "exactly ONE credit of the reward must land in legacy storage — if the \
         bridge were ON here, this would read 2*reward (the double-credit the \
         2026-08-22 sweep found on the unsolicited gossip block-pack path)");
}
