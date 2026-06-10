//! `TipProofService` end-to-end integration tests.
//!
//! Exercises a realistic call pattern: block-producer task ingests step
//! proofs into the service; API handler snapshots the proof; periodic
//! integrity audit runs `verify_chain_self_consistent`. Concurrent
//! reader/writer access is exercised to surface any locking regressions.

use std::sync::Arc;
use std::thread;

use q_ivc::recursion::{LatticeStepProof, StepIO};
use q_lattice_guard::{params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof};
use q_recursive_proofs::{
    tip_verify_v2, TipProofService, TipProofServiceConfig, VerifyErrorV2,
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

/// Realistic producer pattern: build a 10-block chain, snapshot at each
/// step, observe stats, run integrity audit, serialize for HTTP response.
#[test]
fn producer_to_api_full_lifecycle() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());
    let mut prev_root = [0u8; 32];

    for h in 0u64..10 {
        let next = root((h + 1) as u8);
        let z_in = StepIO::new(prev_root, h);
        let z_out = StepIO::new(next, h + 1);
        let stats = service
            .extend(step(z_in, z_out))
            .expect("extend must succeed for honest producer");

        assert_eq!(stats.current_tip_height, h + 1);
        assert_eq!(stats.current_step_count, (h + 1) as usize);

        prev_root = next;
    }

    // API server: snapshot for HTTP response.
    let bytes = service.current_proof_bytes().expect("serialise");
    assert!(!bytes.is_empty(), "serialised proof must be non-empty");

    // External verifier (e.g. fresh node) consumes the proof.
    let proof = service.current_proof();
    tip_verify_v2(&proof, 0, [0u8; 32]).expect("fresh-node verify must pass");

    // Operational integrity audit.
    service
        .verify_chain_self_consistent()
        .expect("self-check must pass");

    let stats = service.stats();
    assert_eq!(stats.total_extends_succeeded, 10);
    assert_eq!(stats.total_extends_rejected, 0);
}

/// Adversarial producer mid-stream: alternating valid + invalid extends.
/// Service must reject the bad ones without corrupting the chain state,
/// and continue accepting good ones afterward.
#[test]
fn service_resilient_against_intermittent_bad_extends() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());

    // Honest step 1.
    service
        .extend(step(StepIO::new([0u8; 32], 0), StepIO::new(root(1), 1)))
        .expect("honest 1");

    // Adversarial: height jump.
    let result = service.extend(step(StepIO::new(root(99), 50), StepIO::new(root(100), 51)));
    assert!(matches!(result, Err(VerifyErrorV2::HeightDiscontinuity { .. })));

    // Adversarial: chain discontinuity (z_in disagrees with current tip state).
    let result = service.extend(step(StepIO::new(root(77), 1), StepIO::new(root(2), 2)));
    assert!(matches!(result, Err(VerifyErrorV2::ChainDiscontinuity { .. })));

    // Honest step 2 (continues from root(1), height 1).
    service
        .extend(step(StepIO::new(root(1), 1), StepIO::new(root(2), 2)))
        .expect("honest 2");

    // Final state must reflect only the honest extends.
    assert_eq!(service.tip_height(), 2);
    assert_eq!(service.step_count(), 2);

    let stats = service.stats();
    assert_eq!(stats.total_extends_attempted, 4);
    assert_eq!(stats.total_extends_succeeded, 2);
    assert_eq!(stats.total_extends_rejected, 2);
    assert!(stats.last_failure_reason.is_none(), "last_failure_reason cleared by most-recent success");
}

/// Concurrent readers + one writer. Verifies the `RwLock` doesn't
/// surface as a logical race — every reader sees a monotonically
/// non-decreasing tip height.
#[test]
fn concurrent_readers_observe_monotonic_tip_height() {
    let service = Arc::new(TipProofService::new(TipProofServiceConfig::genesis()));
    let writer_service = Arc::clone(&service);

    let writer = thread::spawn(move || {
        let mut prev = [0u8; 32];
        for h in 0u64..50 {
            let next = root((h + 1) as u8);
            writer_service
                .extend(step(StepIO::new(prev, h), StepIO::new(next, h + 1)))
                .expect("writer extend");
            prev = next;
        }
    });

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let s = Arc::clone(&service);
            thread::spawn(move || {
                let mut last = 0u64;
                for _ in 0..200 {
                    let h = s.tip_height();
                    assert!(
                        h >= last,
                        "concurrent reader observed tip height regression: {last} → {h}"
                    );
                    last = h;
                }
                last
            })
        })
        .collect();

    writer.join().expect("writer joined");
    for r in readers {
        let last = r.join().expect("reader joined");
        assert!(last <= 50, "reader's last observed height must be ≤ writer's max");
    }

    // After all threads complete, final state must be deterministic.
    assert_eq!(service.tip_height(), 50);
    assert_eq!(service.step_count(), 50);
    service
        .verify_chain_self_consistent()
        .expect("final state self-consistent");
}

/// Upgrade-gate flip simulation: producer is mid-chain when the verifier's
/// trust root changes. Service resets the anchor; subsequent extends use
/// the new anchor as the chain's origin.
#[test]
fn upgrade_gate_anchor_reset_mid_chain() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());

    // Pre-upgrade chain.
    service
        .extend(step(StepIO::new([0u8; 32], 0), StepIO::new(root(1), 1)))
        .expect("pre-upgrade extend");

    // Upgrade-gate flips — new trust root is (100, root(50)).
    service.reset_anchor(100, root(50));
    assert_eq!(service.tip_height(), 100);
    assert_eq!(service.step_count(), 0);

    // Post-upgrade producer continues from the new anchor.
    service
        .extend(step(StepIO::new(root(50), 100), StepIO::new(root(51), 101)))
        .expect("post-upgrade extend");

    assert_eq!(service.tip_height(), 101);
    let stats = service.stats();
    assert!(stats.total_anchor_resets >= 1);
    assert_eq!(stats.total_extends_succeeded, 2, "pre + post upgrade extends both counted");

    // Fresh node connecting after the upgrade verifies against the new anchor.
    let proof = service.current_proof();
    tip_verify_v2(&proof, 100, root(50)).expect("post-upgrade verify");
}

/// Max-steps-retained backpressure under sustained load: producer extends
/// faster than retention cap allows; oldest steps drop, anchor advances,
/// chain remains self-consistent at all times.
#[test]
fn max_steps_retained_backpressure_under_sustained_load() {
    const CAP: usize = 10;
    const N: u64 = 100;
    let config = TipProofServiceConfig::genesis().with_max_steps_retained(CAP);
    let service = TipProofService::new(config);

    let mut prev = [0u8; 32];
    for h in 0u64..N {
        let next = root((h + 1) as u8);
        service
            .extend(step(StepIO::new(prev, h), StepIO::new(next, h + 1)))
            .expect("extend");
        prev = next;

        // Invariant: chain length never exceeds CAP after extend.
        assert!(
            service.step_count() <= CAP,
            "step_count {} exceeded cap {CAP} at h={h}",
            service.step_count()
        );
    }

    assert_eq!(service.tip_height(), N);
    assert_eq!(service.step_count(), CAP);
    assert_eq!(
        service.anchor().0,
        N - CAP as u64,
        "anchor advanced to (tip - cap)"
    );

    let stats = service.stats();
    assert!(stats.total_prefix_drops > 0);
    service
        .verify_chain_self_consistent()
        .expect("backpressured chain remains self-consistent");
}

/// Proof size grows monotonically with chain length (no backpressure)
/// and ApproximateWireSize tracks roughly with actual bincode size.
#[test]
fn proof_size_grows_with_chain_length() {
    let service = TipProofService::new(TipProofServiceConfig::genesis());

    let size_0 = service.current_proof_size_estimate();
    let bincode_0 = service.current_proof_bytes().unwrap().len();
    assert!(size_0 > 0);
    assert!(bincode_0 > 0);

    let mut prev = [0u8; 32];
    for h in 0u64..3 {
        let next = root((h + 1) as u8);
        service
            .extend(step(StepIO::new(prev, h), StepIO::new(next, h + 1)))
            .expect("extend");
        prev = next;
    }

    let size_3 = service.current_proof_size_estimate();
    let bincode_3 = service.current_proof_bytes().unwrap().len();
    assert!(size_3 > size_0, "estimate must grow: {size_0} → {size_3}");
    assert!(bincode_3 > bincode_0, "bincode must grow: {bincode_0} → {bincode_3}");
}
