//! Phase B1 producer ↔ consumer end-to-end pipeline test.
//!
//! Exercises the full path:
//!   1. Producer (block-producer simulator) extends a [`TipProofService`].
//!   2. Service serializes via `current_proof_bytes()`.
//!   3. Bytes travel over a simulated network (Vec<u8> shuffle).
//!   4. Consumer [`TipProofClient`] (fresh wallet) ingests bytes.
//!   5. Consumer's cached tip equals producer's chain state.
//!
//! Plus adversarial coverage:
//!   - Tampered bytes mid-transit
//!   - Stale proof replay (anti-rollback)
//!   - Anchor-swap forgery defeated end-to-end
//!   - Multiple concurrent consumers see consistent state

use std::sync::Arc;
use std::thread;

use q_ivc::recursion::{LatticeStepProof, StepIO};
use q_lattice_guard::{params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof};
use q_recursive_proofs::{
    LatticeTipProofV2, TipProofClient, TipProofClientError, TipProofService,
    TipProofServiceConfig, VerifyErrorV2,
};

fn dummy_lattice_proof() -> LatticeGuardProof {
    LatticeGuardProof {
        commitments: Vec::new(),
        evaluations: (0, 0, 0),
        product_proofs: Vec::new(),
        transcript_state: [0u8; 32],
        metadata: ProofMetadata {
            num_constraints: 0,
            num_public_inputs: 0,
            security_level: SecurityLevel::PQ128,
            generation_time_ms: 0,
        },
    }
}

fn step(z_in: StepIO, z_out: StepIO) -> LatticeStepProof {
    LatticeStepProof {
        proof: dummy_lattice_proof(),
        z_in: z_in.pack(),
        z_out: z_out.pack(),
        public_input_count: 9,
    }
}

fn root(seed: u8) -> [u8; 32] {
    let mut r = [0u8; 32];
    for (i, b) in r.iter_mut().enumerate() {
        *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
    }
    r
}

/// Producer simulator: extend the service by N blocks. Returns the
/// last z_out so the caller can keep extending if desired.
fn produce_blocks(
    service: &TipProofService,
    starting_prev: [u8; 32],
    starting_height: u64,
    n: u64,
) -> [u8; 32] {
    let mut prev = starting_prev;
    for h in starting_height..starting_height + n {
        let next = root((h + 1) as u8);
        service
            .extend(step(StepIO::new(prev, h), StepIO::new(next, h + 1)))
            .expect("honest producer extend");
        prev = next;
    }
    prev
}

#[test]
fn honest_producer_to_honest_consumer_round_trip() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());
    let client = TipProofClient::genesis();

    // Producer builds a 10-block chain.
    produce_blocks(&service, [0u8; 32], 0, 10);
    let bytes = service.current_proof_bytes().expect("serialise");

    // Consumer ingests via the wire bytes.
    let tip = client
        .ingest_bytes(&bytes)
        .expect("honest proof must pass consumer verification");
    assert_eq!(tip, 10);
    assert_eq!(client.current_tip_height(), Some(10));
    assert_eq!(
        client.current_tip_state(),
        Some(service.current_proof().tip_state)
    );

    // Both sides agree on the proof bytes.
    let consumer_proof = client.current_proof().expect("cached");
    assert_eq!(consumer_proof.tip_height, service.tip_height());
    assert_eq!(consumer_proof.anchor_height, 0);
    assert_eq!(consumer_proof.anchor_state, [0u8; 32]);
}

#[test]
fn tampered_bytes_rejected_by_consumer() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());
    let client = TipProofClient::genesis();

    produce_blocks(&service, [0u8; 32], 0, 5);
    let mut bytes = service.current_proof_bytes().expect("serialise");

    // Flip a byte in the middle of the payload (likely lands inside a
    // step proof's commitment region — corrupts deserialization or
    // chain structure).
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;

    let result = client.ingest_bytes(&bytes);
    assert!(
        result.is_err(),
        "tampered bytes must be rejected, got {result:?}"
    );

    // Consumer cache unchanged (was empty, stays empty).
    assert_eq!(client.current_tip_height(), None);
    let stats = client.stats();
    assert_eq!(stats.total_proofs_rejected, 1);
}

#[test]
fn stale_proof_replay_rejected_by_consumer() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());
    let client = TipProofClient::genesis();

    // Producer at height 3.
    produce_blocks(&service, [0u8; 32], 0, 3);
    let bytes_at_3 = service.current_proof_bytes().expect("serialise");

    // Consumer ingests.
    client.ingest_bytes(&bytes_at_3).expect("ingest tip=3");
    assert_eq!(client.current_tip_height(), Some(3));

    // Producer advances to height 5.
    let prev = service.current_proof().tip_state;
    produce_blocks(&service, prev, 3, 2);

    // Consumer ingests the newer proof.
    let bytes_at_5 = service.current_proof_bytes().expect("serialise");
    client.ingest_bytes(&bytes_at_5).expect("ingest tip=5");
    assert_eq!(client.current_tip_height(), Some(5));

    // Replay attack: malicious upstream serves the OLD bytes again.
    let err = client.ingest_bytes(&bytes_at_3).unwrap_err();
    assert!(
        matches!(
            err,
            TipProofClientError::RollbackRejected { fetched: 3, cached: 5 }
        ),
        "replay must be rejected, got {err:?}"
    );

    // Cache still at the newer state.
    assert_eq!(client.current_tip_height(), Some(5));
}

#[test]
fn anchor_swap_forgery_defeated_end_to_end() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());
    let client = TipProofClient::genesis();

    produce_blocks(&service, [0u8; 32], 0, 3);
    let mut proof = service.current_proof();

    // Attacker rewrites the proof header to claim a non-genesis anchor.
    proof.anchor_height = 100;
    proof.anchor_state = root(99);
    proof.tip_height = 103; // adjust to keep step count math consistent
    proof.tip_state = root(99);

    let err = client.ingest_proof(proof).unwrap_err();
    // Either AnchorHeightMismatch (the proof's stored anchor disagrees
    // with the client's) or AnchorMismatch (the first step's z_in
    // disagrees) — both surface the attack.
    assert!(
        matches!(
            err,
            TipProofClientError::Verify(
                VerifyErrorV2::AnchorHeightMismatch { .. }
                    | VerifyErrorV2::AnchorMismatch { .. }
                    | VerifyErrorV2::StepCountMismatch { .. }
            )
        ),
        "anchor-swap forgery must be rejected, got {err:?}"
    );
}

#[test]
fn upgrade_gate_simulation_full_lifecycle() {
    // Pre-upgrade phase: service + client both anchored at genesis.
    let service = TipProofService::new(TipProofServiceConfig::genesis());
    let client = TipProofClient::genesis();
    produce_blocks(&service, [0u8; 32], 0, 5);
    client
        .ingest_bytes(&service.current_proof_bytes().unwrap())
        .expect("pre-upgrade ingest");
    assert_eq!(client.current_tip_height(), Some(5));

    // Upgrade-gate flips — new trust root is (100, root(50)).
    service.reset_anchor(100, root(50));
    let new_client = TipProofClient::new(100, root(50));
    // Old client at old anchor still has its cached proof, but if it
    // tries to ingest the new producer's output it'll reject (anchor
    // mismatch).
    produce_blocks(&service, root(50), 100, 3);
    let post_upgrade_bytes = service.current_proof_bytes().expect("serialise");

    // Old client rejects.
    let err = client.ingest_bytes(&post_upgrade_bytes).unwrap_err();
    assert!(matches!(
        err,
        TipProofClientError::Verify(VerifyErrorV2::AnchorHeightMismatch { .. })
            | TipProofClientError::Verify(VerifyErrorV2::AnchorStateMismatch)
    ));

    // New client (correctly upgraded) accepts.
    let tip = new_client
        .ingest_bytes(&post_upgrade_bytes)
        .expect("upgraded client must accept post-upgrade proof");
    assert_eq!(tip, 103);
    assert_eq!(new_client.current_tip_height(), Some(103));
}

#[test]
fn multiple_concurrent_clients_observe_consistent_state() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());
    let client = Arc::new(TipProofClient::genesis());

    // Producer builds a 20-block chain.
    produce_blocks(&service, [0u8; 32], 0, 20);
    let bytes = Arc::new(service.current_proof_bytes().expect("serialise"));

    // Spawn 8 concurrent consumers; each ingests the same bytes.
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let c = Arc::clone(&client);
            let b = Arc::clone(&bytes);
            thread::spawn(move || {
                // Repeated ingest_bytes on the same payload: first one
                // accepts (tip=20), subsequent are rolled back (cached=20).
                let result = c.ingest_bytes(&b);
                match result {
                    Ok(20) => true,
                    Err(TipProofClientError::RollbackRejected {
                        fetched: 20,
                        cached: 20,
                    }) => true,
                    other => panic!("unexpected concurrent ingest result: {other:?}"),
                }
            })
        })
        .collect();

    for h in handles {
        assert!(h.join().expect("thread joined"));
    }

    // Exactly one acceptance + 7 rollback-rejections.
    let stats = client.stats();
    assert_eq!(stats.total_proofs_accepted, 1);
    assert_eq!(stats.total_rollbacks_rejected, 7);
    assert_eq!(client.current_tip_height(), Some(20));
}

/// Service + Client are both Sync + Send: spawn-friendly without
/// awkward `Arc<Mutex>` wrappers at the caller's site.
#[test]
fn service_and_client_satisfy_send_sync_bounds() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TipProofService>();
    assert_send_sync::<TipProofClient>();
    assert_send_sync::<LatticeTipProofV2>();
}
