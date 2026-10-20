//! v10.11.114c: the store must survive a node restart. `persist_to_file` writes
//! `$Q_DB_PATH/multisig/store.json`; a FRESH store (what `global()` gives after a restart)
//! must get everything back from `load_from_file` — which had no caller before 114c, so
//! every restart forgot every org and the next write saved the empty state over the file.
//! This is the only test in this binary that sets Q_DB_PATH (env is process-wide).

use q_multisig::proposal::{MultisigAction, MultisigProposal, ProposalStatus};
use q_multisig::{MultisigStore, OrgPolicy, OrgPolicyMember};

#[tokio::test]
async fn a_fresh_store_loads_what_the_old_one_persisted() {
    let dir = std::env::temp_dir().join(format!("q-multisig-reload-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::env::set_var("Q_DB_PATH", &dir);

    let before = MultisigStore::new();
    let mut p = MultisigProposal::new(
        [7u8; 32],
        MultisigAction::Transfer { token: "QUG".into(), recipient: [9u8; 32], amount_raw: 1, memo: Some("rent".into()) },
        None,
        2,
        2,
    );
    p.status = ProposalStatus::Threshold;
    p.proposer = Some([1u8; 32]);
    let (id, hash) = (p.id, p.payload_hash());
    before.insert_proposal(p);
    before.set_org_policy([7u8; 32], OrgPolicy {
        org_name: "Rig Treasury".into(),
        ceo_address_hex: hex::encode([1u8; 32]),
        members: vec![OrgPolicyMember {
            address_hex: hex::encode([1u8; 32]), name: "Alice".into(), role: "ceo".into(),
            per_tx_limit: 0, daily_limit: 0, approval_threshold: 0, approvals_required: 0,
        }],
        updated_at_unix: 1,
    });
    before.persist_to_file().await.expect("persist");

    let after = MultisigStore::new(); // = the process-global store right after a restart
    assert!(after.get_proposal(&id).is_none());
    assert!(after.load_from_file().await > 0, "load_from_file found nothing");
    let back = after.get_proposal(&id).expect("proposal survives a restart");
    assert_eq!(back.payload_hash(), hash);
    assert_eq!(back.status, ProposalStatus::Threshold);
    assert_eq!(back.proposer, Some([1u8; 32]));
    assert_eq!(after.get_org_policy(&[7u8; 32]).expect("policy survives").org_name, "Rig Treasury");

    let _ = std::fs::remove_dir_all(&dir);
}
