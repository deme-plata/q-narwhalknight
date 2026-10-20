//! v10.11.114c added five optional fields to `MultisigProposal` (proposer, executed_at_unix,
//! executed_by, cancelled_at_unix, cancelled_by). Proposals already on disk
//! (`$Q_DB_PATH/multisig/store.json`, written by 114b and earlier) lack them. Pin that such a
//! proposal still loads, gets `None` for every new field, and keeps the SAME payload hash —
//! the hash every collected signature covers.

use q_multisig::proposal::{MultisigAction, MultisigProposal, ProposalStatus};

fn proposal() -> MultisigProposal {
    MultisigProposal::new(
        [7u8; 32],
        MultisigAction::Transfer {
            token: "QUG".into(),
            recipient: [9u8; 32],
            amount_raw: 1_000_000_000_000_000_000_000_000,
            memo: Some("test".into()),
        },
        None,
        2,
        2,
    )
}

#[test]
fn proposal_stored_by_114b_on_mainnet_loads_with_the_hash_its_signers_signed() {
    // The real record from $Q_DB_PATH/multisig/store.json (2026-10-20, 1 QUG, 2/2 signed,
    // written by v10.11.114b) and the payload_hash the live 114b node reported for it.
    let raw = include_str!("fixtures/proposal-stored-by-114b.json");
    assert!(!raw.contains("proposer"), "fixture must be the pre-114c shape");
    let p: MultisigProposal = serde_json::from_str(raw).expect("a pre-114c proposal must load");
    assert_eq!(
        hex::encode(p.payload_hash()),
        "848450f2c9456e2fdb6635bf36ca90b3819dcbe1292396927dec1eb957dd1c31"
    );
    assert_eq!(p.status, ProposalStatus::Threshold);
    assert_eq!(p.signatures.len(), 2);
    assert!(p.proposer.is_none() && p.executed_at_unix.is_none() && p.executed_by.is_none());
    assert!(p.cancelled_at_unix.is_none() && p.cancelled_by.is_none());
}

#[test]
fn round_trip_through_the_store_format_keeps_the_hash() {
    // The store writes with serde_json::to_vec_pretty (u128 amounts are fine there; only
    // serde_json::Value cannot hold them).
    let mut p = proposal();
    p.proposer = Some([1u8; 32]);
    let bytes = serde_json::to_vec_pretty(&p).unwrap();
    let back: MultisigProposal = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back.payload_hash(), p.payload_hash());
    assert_eq!(back.proposer, Some([1u8; 32]));
}

#[test]
fn new_fields_do_not_enter_the_signed_hash() {
    let p = proposal();
    let mut q = p.clone();
    q.proposer = Some([1u8; 32]);
    q.executed_at_unix = Some(1_792_480_615);
    q.executed_by = Some([2u8; 32]);
    q.cancelled_at_unix = Some(1);
    q.cancelled_by = Some([3u8; 32]);
    assert_eq!(p.payload_hash(), q.payload_hash());
}
