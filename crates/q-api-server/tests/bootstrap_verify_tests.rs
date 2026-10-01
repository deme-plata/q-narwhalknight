//! v10.11.109 bootstrap snapshot verification + equal-height divergence record.
//! Integration-test target on purpose (see q-storage/tests/integrity_guards_tests.rs).
//!
//!   cargo test -p q-api-server --test bootstrap_verify_tests

use q_api_server::bootstrap_verify::{
    claim_and_verify_import, divergence_stats, note_divergence_check, note_divergence_skipped,
    parse_pin_ids, pinned_producer_ids, snapshot_root, verify_snapshot, Verdict,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn baked_pin_yields_the_production_producer_id() {
    if std::env::var("Q_PRODUCER_DILITHIUM_PUBKEY_HEX").is_ok() {
        return; // operator override in this environment — nothing to assert about the baked pin
    }
    let ids = pinned_producer_ids();
    assert_eq!(ids.len(), 1, "the baked pin names exactly one producer");
    assert_eq!(hex::encode(&ids[0][..4]), "cdf9bda1");
}

#[test]
fn pin_parser_skips_garbage() {
    assert!(parse_pin_ids("").is_empty());
    assert!(parse_pin_ids("zz:aa").is_empty());
    assert!(parse_pin_ids("00ff:aa").is_empty()); // wrong id length
    let one = format!("{}:abcd", "11".repeat(32));
    assert_eq!(parse_pin_ids(&one), vec![[0x11u8; 32]]);
}

#[test]
fn snapshot_root_is_the_nodes_own_root_function() {
    let mut snap = HashMap::new();
    snap.insert("22".repeat(32), "500".to_string());
    snap.insert("11".repeat(32), "7".to_string());
    snap.insert("33".repeat(32), "0".to_string()); // zero: excluded, as in the node
    snap.insert("not-hex".to_string(), "9".to_string()); // unparsable: skipped, as in the importer
    let want = q_storage::balance_root::balance_root_v1(vec![([0x11; 32], 7u128), ([0x22; 32], 500u128)]);
    assert_eq!(snapshot_root(&snap), want);
}

#[tokio::test]
async fn a_snapshot_below_the_first_rooted_header_is_unverifiable_without_network() {
    let snap: HashMap<String, String> = HashMap::new();
    match verify_snapshot(&snap, 21_001).await {
        Verdict::Unverifiable(why) => assert!(why.contains("below"), "{why}"),
        other => panic!("expected Unverifiable, got {:?}", other),
    }
}

#[test]
fn divergence_record_counts_and_keeps_the_last_mismatch() {
    let before = divergence_stats();
    note_divergence_skipped();
    note_divergence_check(true, 10, 10, "aa", "aa");
    note_divergence_check(false, 11, 11, "bb", "cc");
    let after = divergence_stats();
    assert_eq!(after.skipped_unequal_height, before.skipped_unequal_height + 1);
    assert_eq!(after.matches, before.matches + 1);
    assert_eq!(after.mismatches, before.mismatches + 1);
    let last = after.last_mismatch.expect("mismatch recorded");
    assert_eq!((last.height, last.our_hash.as_str(), last.peer_hash.as_str()), (11, "bb", "cc"));
}

#[tokio::test]
async fn the_one_time_import_can_only_be_claimed_once() {
    std::env::set_var("Q_BOOTSTRAP_VERIFY", "off"); // no network in this test
    let done = AtomicBool::new(false);
    let snap: HashMap<String, String> = HashMap::new();
    assert!(claim_and_verify_import(&done, &snap, 5).await, "first claim imports");
    assert!(!claim_and_verify_import(&done, &snap, 5).await, "a second, concurrent response must not import again");
    assert!(done.load(Ordering::SeqCst));
}
