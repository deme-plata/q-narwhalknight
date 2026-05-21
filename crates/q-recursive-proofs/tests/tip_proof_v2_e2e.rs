//! Phase B1 `LatticeTipProofV2` end-to-end integration tests.
//!
//! Exercises chain integrity + the public anchor/extend/verify API. The
//! mandatory DeepSeek §0 anchor-swap forgery regression is here (lifted
//! from v1 but adapted to v2's chain-integrity model — see test comment).

use q_ivc::recursion::{LatticeStepProof, StepIO};
use q_lattice_guard::{params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof};
use q_recursive_proofs::{
    tip_anchor_v2, tip_extend_v2, tip_verify_v2, LatticeTipProofV2, VerifyErrorV2,
    TIP_PROOF_V2_VERSION,
};

fn root(seed: u8) -> [u8; 32] {
    let mut r = [0u8; 32];
    for (i, b) in r.iter_mut().enumerate() {
        *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
    }
    r
}

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

fn step_proof(z_in: StepIO, z_out: StepIO) -> LatticeStepProof {
    LatticeStepProof {
        proof: dummy_lattice_proof(),
        z_in: z_in.pack(),
        z_out: z_out.pack(),
        public_input_count: 9,
    }
}

#[test]
fn anchor_only_passes() {
    let p = tip_anchor_v2(0, root(1));
    tip_verify_v2(&p, 0, root(1)).expect("anchor-only proof must verify structure");
    assert_eq!(p.tip_height, 0);
    assert_eq!(p.tip_state, root(1));
    assert_eq!(p.delta_proofs.len(), 0);
}

#[test]
fn version_string_is_pinned() {
    assert_eq!(TIP_PROOF_V2_VERSION, "latticeguard-rlwe-v1");
    let p = tip_anchor_v2(0, [0u8; 32]);
    assert_eq!(p.version_str(), "latticeguard-rlwe-v1");
}

#[test]
fn extend_three_step_chain_round_trip() {
    let mut p = tip_anchor_v2(0, root(9));
    let z0 = StepIO::new(root(9), 0);
    let z1 = StepIO::new(root(10), 1);
    let z2 = StepIO::new(root(11), 2);
    let z3 = StepIO::new(root(12), 3);
    p = tip_extend_v2(&p, step_proof(z0, z1)).expect("step 1");
    p = tip_extend_v2(&p, step_proof(z1, z2)).expect("step 2");
    p = tip_extend_v2(&p, step_proof(z2, z3)).expect("step 3");

    tip_verify_v2(&p, 0, root(9)).expect("3-step chain must verify structure");
    assert_eq!(p.tip_height, 3);
    assert_eq!(p.tip_state, root(12));
    assert_eq!(p.delta_proofs.len(), 3);
}

#[test]
fn extend_chain_rejects_height_jump_at_append() {
    let p = tip_anchor_v2(0, root(1));
    let bad = step_proof(StepIO::new(root(1), 5), StepIO::new(root(2), 6));
    let err = tip_extend_v2(&p, bad).unwrap_err();
    assert!(
        matches!(err, VerifyErrorV2::HeightDiscontinuity { .. }),
        "got {err:?}"
    );
}

#[test]
fn extend_chain_rejects_state_root_break_at_append() {
    let p = tip_anchor_v2(0, root(1));
    // z_in claims state root(99) but anchor is root(1).
    let bad = step_proof(StepIO::new(root(99), 0), StepIO::new(root(2), 1));
    let err = tip_extend_v2(&p, bad).unwrap_err();
    assert!(matches!(err, VerifyErrorV2::ChainDiscontinuity { step: 0 }));
}

/// **Mandatory regression** — DeepSeek §0 anchor-swap forgery rejected.
///
/// Lifted from `tip_proof_v1::tests::rejects_deepseek_anchor_swap_forgery`
/// but adapted to v2's chain-integrity model: instead of a monolithic
/// transcript+commitment to forge, v2 has a vec of immutable per-step
/// proofs. The attacker substitutes `anchor_*` and `tip_*` in the proof
/// struct, but the first step's `z_in` still references the original
/// anchor — `verify_chain_structure` surfaces this as `AnchorMismatch`.
#[test]
fn rejects_deepseek_anchor_swap_forgery() {
    let honest = tip_anchor_v2(100, root(11));
    let s = step_proof(StepIO::new(root(11), 100), StepIO::new(root(12), 101));
    let honest_ext = tip_extend_v2(&honest, s).expect("honest extend");

    // Attacker rewrites the proof header to claim genesis anchor + a
    // different tip. delta_proofs is unchanged — the attacker cannot
    // forge step proofs (each carries an immutable LatticeGuardProof).
    let forged = LatticeTipProofV2 {
        anchor_height: 0,
        anchor_state: [0u8; 32],
        tip_height: 101,
        tip_state: root(12),
        ..honest_ext
    };

    let err = tip_verify_v2(&forged, 0, [0u8; 32]).unwrap_err();
    assert!(
        matches!(
            err,
            VerifyErrorV2::AnchorMismatch { anchor_height: 0 }
                | VerifyErrorV2::StepCountMismatch { .. }
        ),
        "DeepSeek §0 anchor-swap MUST be rejected, got {err:?}"
    );
}

#[test]
fn rejects_tip_height_inflation() {
    let p = tip_anchor_v2(50, root(7));
    let s = step_proof(StepIO::new(root(7), 50), StepIO::new(root(8), 51));
    let ext = tip_extend_v2(&p, s).expect("extend");
    let inflated = LatticeTipProofV2 {
        tip_height: 1_000_000,
        ..ext
    };
    let err = tip_verify_v2(&inflated, 50, root(7)).unwrap_err();
    assert!(
        matches!(err, VerifyErrorV2::StepCountMismatch { .. }),
        "tip_height inflation must be REJECTED, got {err:?}"
    );
}

#[test]
fn rejects_anchor_state_mismatch_against_verifier() {
    let p = tip_anchor_v2(0, root(1));
    let err = tip_verify_v2(&p, 0, root(99)).unwrap_err();
    assert!(matches!(err, VerifyErrorV2::AnchorStateMismatch));
}

#[test]
fn rejects_anchor_height_mismatch_against_verifier() {
    let p = tip_anchor_v2(0, root(1));
    let err = tip_verify_v2(&p, 7, root(1)).unwrap_err();
    assert!(matches!(
        err,
        VerifyErrorV2::AnchorHeightMismatch { expected: 7, got: 0 }
    ));
}

#[test]
fn rejects_wrong_version_string() {
    let mut p = tip_anchor_v2(0, root(1));
    p.version = "tip-blake3-fs-v1".to_string(); // pretending to be v1
    let err = tip_verify_v2(&p, 0, root(1)).unwrap_err();
    assert!(matches!(err, VerifyErrorV2::VersionMismatch { .. }));
}
