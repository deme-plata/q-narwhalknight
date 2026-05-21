//! Tip proof v2 (LatticeGuard per-step) — Phase B1 lattice-backed proof.
//!
//! Replaces [`tip_proof_v1`](super::tip_proof_v1)'s BLAKE3 Fiat-Shamir
//! hash-chain (whose soundness reduces only to BLAKE3 preimage hardness)
//! with a **chain of per-step LatticeGuard proofs**, each one proving
//! one block's δ-transition over the RLWE-based zk-SNARK in
//! `q-lattice-guard`. The wire format carries `Vec<LatticeStepProof>` so
//! proof size grows linearly with chain length — Phase C swaps the vec
//! for a single constant-size `FoldedInstance` via Module-SIS folding.
//!
//! `proof_version = "latticeguard-rlwe-v1"`.
//!
//! # What v2 binds (vs v1)
//!
//! - **Chain integrity:** every step's `z_in` must equal the previous
//!   step's `z_out` (or the anchor, for the first step). An attacker who
//!   swaps the anchor invalidates the first step's `z_in` reference.
//!   This defeats the DeepSeek §0 anchor-swap class structurally —
//!   there's no monolithic commitment to forge.
//! - **Tip integrity:** the final step's `z_out` must equal
//!   `(tip_state, tip_height)`. An attacker who inflates `tip_height`
//!   breaks this equality.
//! - **Per-step transition correctness:** each `LatticeStepProof.proof`
//!   attests that `δ(z_in) → z_out` was satisfied by some witness inside
//!   the lattice-guard prover's R1CS. Subject to plan risk R5 (witness
//!   projection truncation), this is post-quantum sound.
//!
//! # Soundness regime: ADVISORY through v10.9.x
//!
//! Per plan risk R5 (documented at
//! [`crate::tip_proof_v2::SOUNDNESS_NOTES`]), the lattice-guard bridge
//! truncates BLS12-381 Fr witnesses to 64-bit `Scalar`. Effective
//! soundness drops from BLS12-381's ~125 bits to the RLWE modulus' ~32
//! bits at `pq128`. The upgrade gate `Upgrade::TipProofV2` MUST NOT
//! activate binding consensus use until Phase C ships the wider modulus.
//!
//! # Verification cost
//!
//! Linear in `delta_proofs.len()` — each step verify is one
//! `LatticeGuardVerifier::verify` call (10-100ms at pq128). For a
//! 1,000-block soak this is ~10-100 seconds, too slow for ≤10ms tip
//! verification but acceptable as a soak/spot-check during the advisory
//! window. Phase C's constant-size accumulator restores the ≤10ms target.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use q_ivc::recursion::{LatticeStepFolder, LatticeStepProof, StepIO};

/// 32-byte BLAKE3 state root.
pub type StateRoot = [u8; 32];

/// Wire-format proof_version string identifying this proof system.
/// Embedded in every `LatticeTipProofV2` so the API server can serve
/// v1 (`tip-blake3-fs-v1`) or v2 (`latticeguard-rlwe-v1`) interchangeably
/// during the migration window. Wallets render this in the verifier banner.
pub const PROOF_VERSION: &str = "latticeguard-rlwe-v1";

/// Phase B1 lattice-backed tip proof.
///
/// Wire size: 32 + 8 + 32 + 8 + N * sizeof(LatticeStepProof). At pq128
/// each step proof is ~10-50KB. A 1,000-step chain serialises to ~10-50MB.
/// Phase C compresses this to a single `FoldedInstance` (~8KB constant).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatticeTipProofV2 {
    /// Per-block step proofs in chain order. `delta_proofs[i].z_in`
    /// chains from the previous step (or the anchor at `i = 0`); the
    /// final step's `z_out` must equal `(tip_state, tip_height)`.
    pub delta_proofs: Vec<LatticeStepProof>,
    /// Anchor (verifier-known trust root). Genesis is `(0, [0u8; 32])`.
    pub anchor_state: StateRoot,
    pub anchor_height: u64,
    /// Tip claim (prover's assertion of current chain state).
    pub tip_state: StateRoot,
    pub tip_height: u64,
    /// Stored on the wire for forward-compat across proof-system versions.
    pub version: String,
}

impl LatticeTipProofV2 {
    /// Number of step proofs in the chain (= `tip_height - anchor_height`).
    pub fn step_count(&self) -> usize {
        self.delta_proofs.len()
    }

    /// Quick header inspection without deserialising the proofs — used by
    /// API servers to short-circuit version-mismatched clients.
    pub fn version_str(&self) -> &str {
        &self.version
    }
}

/// Errors surfaced by [`verify`]. Each carries enough context to debug
/// without leaking witness data.
#[derive(Debug, Error)]
pub enum VerifyErrorV2 {
    /// Wire-format version string didn't match [`PROOF_VERSION`].
    #[error("proof version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: String, got: String },

    /// The first step's `z_in` doesn't reference the verifier's expected
    /// anchor — defeats the DeepSeek §0 anchor-swap class.
    #[error("first step z_in does not anchor to (anchor_height={anchor_height}, anchor_state)")]
    AnchorMismatch { anchor_height: u64 },

    /// The proof's stored anchor disagrees with the verifier's expected
    /// anchor. Independent surface from `AnchorMismatch` so the diagnosis
    /// is unambiguous.
    #[error("proof.anchor_height {got} differs from expected {expected}")]
    AnchorHeightMismatch { expected: u64, got: u64 },

    /// Same as above but for the state root.
    #[error("proof.anchor_state does not match expected anchor state")]
    AnchorStateMismatch,

    /// A step's `z_in` doesn't match the previous step's `z_out`. Catches
    /// any attempted chain splice / reorder.
    #[error("chain discontinuity at step {step}: z_in disagrees with prior z_out")]
    ChainDiscontinuity { step: usize },

    /// Heights skipped or non-monotonic. Caught at chain integrity time
    /// before the (expensive) per-proof crypto verification.
    #[error("height discontinuity at step {step}: expected {expected}, got {got}")]
    HeightDiscontinuity { step: usize, expected: u64, got: u64 },

    /// The final step's `z_out` doesn't match the claimed tip.
    #[error("final step z_out does not match claimed tip (height={tip_height})")]
    TipMismatch { tip_height: u64 },

    /// `proof.tip_height` is below `proof.anchor_height` — chain runs
    /// backward, which is structurally impossible.
    #[error("tip_height {tip} below anchor_height {anchor}")]
    TipBelowAnchor { tip: u64, anchor: u64 },

    /// `proof.delta_proofs.len()` doesn't equal `tip_height - anchor_height`.
    /// Catches a proof that claims to span N blocks while shipping K ≠ N
    /// step proofs.
    #[error("step_count mismatch: vec len {vec_len} but tip - anchor = {derived}")]
    StepCountMismatch { vec_len: usize, derived: u64 },

    /// A per-step lattice-guard verification returned `Ok(false)` — the
    /// proof bytes parsed but the lattice predicate didn't satisfy.
    #[error("step {step}: lattice-guard verifier rejected the proof")]
    StepProofInvalid { step: usize },

    /// A per-step lattice-guard verification failed structurally
    /// (folder error, bad SRS, etc.).
    #[error("step {step}: lattice-guard folder error: {source}")]
    FolderError {
        step: usize,
        #[source]
        source: q_ivc::recursion::FolderError,
    },
}

/// Construct an anchor (empty-chain) proof. Used at genesis or when a
/// fresh chain segment opens.
///
/// No step proofs, no cryptographic work. Subsequent
/// [`extend_with_step_proof`] calls append one step at a time.
pub fn anchor(anchor_height: u64, anchor_state: StateRoot) -> LatticeTipProofV2 {
    LatticeTipProofV2 {
        delta_proofs: Vec::new(),
        anchor_state,
        anchor_height,
        tip_state: anchor_state,
        tip_height: anchor_height,
        version: PROOF_VERSION.to_string(),
    }
}

/// Extend a proof by attaching an already-generated step proof.
///
/// Validates chain consistency at append time (so a misshapen chain is
/// caught at the prover, not at the verifier hours later):
///   - `step_proof.z_in.height == prev.tip_height`
///   - `step_proof.z_in.state_root == prev.tip_state`
///   - `step_proof.z_out.height == prev.tip_height + 1`
///
/// Caller is responsible for producing `step_proof` via
/// [`LatticeStepFolder::prove_step`] on the actual δ-circuit; this
/// function just stitches them into the chain.
pub fn extend_with_step_proof(
    prev: &LatticeTipProofV2,
    step_proof: LatticeStepProof,
) -> Result<LatticeTipProofV2, VerifyErrorV2> {
    let z_in = step_proof.z_in_unpacked();
    let z_out = step_proof.z_out_unpacked();

    // Chain continuity at append time.
    if z_in.height != prev.tip_height {
        return Err(VerifyErrorV2::HeightDiscontinuity {
            step: prev.delta_proofs.len(),
            expected: prev.tip_height,
            got: z_in.height,
        });
    }
    if z_in.state_root != prev.tip_state {
        return Err(VerifyErrorV2::ChainDiscontinuity {
            step: prev.delta_proofs.len(),
        });
    }
    if z_out.height != prev.tip_height + 1 {
        return Err(VerifyErrorV2::HeightDiscontinuity {
            step: prev.delta_proofs.len(),
            expected: prev.tip_height + 1,
            got: z_out.height,
        });
    }

    let mut next = prev.clone();
    next.delta_proofs.push(step_proof);
    next.tip_state = z_out.state_root;
    next.tip_height = z_out.height;
    Ok(next)
}

/// Phase B1 verification — chain integrity check only. The per-step
/// cryptographic verification is gated behind [`verify_with_folder`]
/// because it requires an SRS-loaded `LatticeStepFolder` (the SRS is
/// not part of the proof and must be reconstructed by the verifier).
///
/// Returns `Ok(())` if the chain integrity passes; per-step proofs are
/// NOT cryptographically verified by this function. Use this in
/// API-server response handlers where the structural check is enough to
/// reject malformed proofs cheaply.
pub fn verify_chain_structure(
    proof: &LatticeTipProofV2,
    expected_anchor_height: u64,
    expected_anchor_state: StateRoot,
) -> Result<(), VerifyErrorV2> {
    if proof.version != PROOF_VERSION {
        return Err(VerifyErrorV2::VersionMismatch {
            expected: PROOF_VERSION.to_string(),
            got: proof.version.clone(),
        });
    }
    if proof.anchor_height != expected_anchor_height {
        return Err(VerifyErrorV2::AnchorHeightMismatch {
            expected: expected_anchor_height,
            got: proof.anchor_height,
        });
    }
    if proof.anchor_state != expected_anchor_state {
        return Err(VerifyErrorV2::AnchorStateMismatch);
    }
    if proof.tip_height < proof.anchor_height {
        return Err(VerifyErrorV2::TipBelowAnchor {
            tip: proof.tip_height,
            anchor: proof.anchor_height,
        });
    }

    let derived = proof.tip_height - proof.anchor_height;
    if (proof.delta_proofs.len() as u64) != derived {
        return Err(VerifyErrorV2::StepCountMismatch {
            vec_len: proof.delta_proofs.len(),
            derived,
        });
    }

    // Walk the chain — every step's z_in must reference the previous
    // tip; the first step must reference the anchor.
    let mut cursor_height = proof.anchor_height;
    let mut cursor_state = proof.anchor_state;
    for (i, step) in proof.delta_proofs.iter().enumerate() {
        let z_in = step.z_in_unpacked();
        let z_out = step.z_out_unpacked();

        if i == 0 {
            // Defeats the DeepSeek §0 anchor-swap class: the first
            // step's z_in must match the verifier's expected anchor.
            if z_in.state_root != expected_anchor_state
                || z_in.height != expected_anchor_height
            {
                return Err(VerifyErrorV2::AnchorMismatch {
                    anchor_height: expected_anchor_height,
                });
            }
        } else if z_in.state_root != cursor_state || z_in.height != cursor_height {
            return Err(VerifyErrorV2::ChainDiscontinuity { step: i });
        }

        if z_out.height != cursor_height + 1 {
            return Err(VerifyErrorV2::HeightDiscontinuity {
                step: i,
                expected: cursor_height + 1,
                got: z_out.height,
            });
        }
        cursor_height = z_out.height;
        cursor_state = z_out.state_root;
    }

    // Final step's z_out must match the claimed tip.
    if cursor_height != proof.tip_height || cursor_state != proof.tip_state {
        return Err(VerifyErrorV2::TipMismatch {
            tip_height: proof.tip_height,
        });
    }

    Ok(())
}

/// Per-step cryptographic verification. Calls
/// [`LatticeStepFolder::verify_step`] for each step proof, using
/// `circuit_builder` to reconstruct the matching verification-side
/// constraint shape.
///
/// `circuit_builder` is a closure `FnMut(StepIO, StepIO) -> C` —
/// receives the expected `(z_in, z_out)` for each step and returns the
/// circuit `C` whose R1CS shape must match what the prover saw.
///
/// Wraps [`verify_chain_structure`] internally so a malformed chain is
/// rejected before paying any crypto cost.
///
/// **Phase B1 caveat:** this assumes the verifier can rebuild each
/// step's R1CS shape deterministically from `(z_in, z_out)`. Real
/// δ-circuits also depend on per-block witness shape (tx count, etc.)
/// — Phase B2 wires the SRS + witness-shape registry; until then this
/// path is exercised only by test fixtures whose circuits are
/// witness-independent.
pub fn verify_with_folder<F, B, C>(
    proof: &LatticeTipProofV2,
    expected_anchor_height: u64,
    expected_anchor_state: StateRoot,
    folder: Arc<LatticeStepFolder>,
    mut circuit_builder: B,
) -> Result<(), VerifyErrorV2>
where
    F: ark_ff::PrimeField,
    B: FnMut(StepIO, StepIO) -> C,
    C: ark_relations::r1cs::ConstraintSynthesizer<F>,
{
    verify_chain_structure(proof, expected_anchor_height, expected_anchor_state)?;

    for (i, step) in proof.delta_proofs.iter().enumerate() {
        let z_in = step.z_in_unpacked();
        let z_out = step.z_out_unpacked();
        let builder = || circuit_builder(z_in, z_out);

        let verified = folder
            .verify_step::<F, _, _>(step, builder, z_in, z_out)
            .map_err(|source| VerifyErrorV2::FolderError { step: i, source })?;
        if !verified {
            return Err(VerifyErrorV2::StepProofInvalid { step: i });
        }
    }

    Ok(())
}

// ════════════════════════════════════════════════════════════════════════════
// Soundness notes (committed inline so anyone reading the file sees them)
// ════════════════════════════════════════════════════════════════════════════

/// Plain-text soundness narrative. Surfaced as a constant so it survives
/// rustdoc + grep / strings on the binary.
pub const SOUNDNESS_NOTES: &str = "\
PHASE B1 tip_proof_v2 — ADVISORY, NOT BINDING.

Bridge truncation (plan risk R5): the R1csBridge in q-ivc projects \
BLS12-381 Fr witnesses to q-lattice-guard's u64 Scalar via low-64-bit \
truncation modulo the RLWE prime. Effective soundness drops from ~125 \
bits (BLS12-381) to ~32 bits at pq128. An adversary who finds a \
witness that satisfies the arkworks system but whose 64-bit projection \
violates the lattice constraints wins. Phase C's pq128_folding profile \
(60-bit NTT-friendly modulus + wider Scalar) closes this gap.

Linear-size proof (plan): wire format carries Vec<LatticeStepProof> so \
a 1000-block chain serialises to ~10-50MB. Verifier cost is also \
linear (one LatticeGuardVerifier::verify per step). Phase C's \
constant-size FoldedInstance fixes both. \

Upgrade gate: Upgrade::TipProofV2 MUST NOT activate binding consensus \
use until Phase C ships. The API server may serve v2 proofs for \
diagnostic purposes during the advisory window; wallets render \
'lattice-guard rlwe v1, advisory'.";

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use q_ivc::recursion::StepIO;
    use q_lattice_guard::{
        params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof,
    };

    fn dummy_proof() -> LatticeGuardProof {
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
            proof: dummy_proof(),
            z_in: z_in.pack(),
            z_out: z_out.pack(),
            public_input_count: 9,
        }
    }

    fn root(seed: u8) -> StateRoot {
        let mut r = [0u8; 32];
        for (i, b) in r.iter_mut().enumerate() {
            *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
        }
        r
    }

    #[test]
    fn anchor_only_passes_structure_check() {
        let p = anchor(0, root(1));
        verify_chain_structure(&p, 0, root(1)).expect("anchor-only proof must verify");
    }

    #[test]
    fn extend_advances_tip() {
        let p = anchor(100, root(7));
        let s = step(
            StepIO::new(root(7), 100),
            StepIO::new(root(8), 101),
        );
        let ext = extend_with_step_proof(&p, s).expect("append");
        assert_eq!(ext.tip_height, 101);
        assert_eq!(ext.tip_state, root(8));
        assert_eq!(ext.delta_proofs.len(), 1);
    }

    #[test]
    fn extend_chain_3_blocks_round_trip() {
        let mut p = anchor(0, root(9));
        for h in 0..3u64 {
            let z_in = StepIO::new(if h == 0 { root(9) } else { root(10 + h as u8 - 1) }, h);
            let z_out = StepIO::new(root(10 + h as u8), h + 1);
            p = extend_with_step_proof(&p, step(z_in, z_out)).expect("append");
        }
        verify_chain_structure(&p, 0, root(9)).expect("3-block chain must verify structure");
        assert_eq!(p.tip_height, 3);
    }

    #[test]
    fn rejects_wrong_anchor_state() {
        let p = anchor(0, root(5));
        let err = verify_chain_structure(&p, 0, root(6)).unwrap_err();
        assert!(matches!(err, VerifyErrorV2::AnchorStateMismatch));
    }

    #[test]
    fn rejects_wrong_anchor_height() {
        let p = anchor(0, root(5));
        let err = verify_chain_structure(&p, 1, root(5)).unwrap_err();
        assert!(
            matches!(err, VerifyErrorV2::AnchorHeightMismatch { expected: 1, got: 0 }),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_wrong_version_string() {
        let mut p = anchor(0, root(1));
        p.version = "wrong-version-string".to_string();
        let err = verify_chain_structure(&p, 0, root(1)).unwrap_err();
        assert!(matches!(err, VerifyErrorV2::VersionMismatch { .. }));
    }

    /// Phase B1 successor to the DeepSeek §0 anchor-swap regression
    /// (lifted from `tip_proof_v1::tests::rejects_deepseek_anchor_swap_forgery`).
    ///
    /// The attack model in v1 was: substitute (anchor_height, anchor_state)
    /// in the proof while keeping the same commitment + transcript. v1's
    /// fix folded the anchor into the commitment hash so this fails.
    ///
    /// In v2 the chain integrity check makes the attack structurally
    /// impossible: even if the attacker substitutes the proof's anchor
    /// fields, the first step's `z_in` still references the original
    /// chain's anchor (because step proofs are immutable bytes signed by
    /// the lattice prover). The verifier's expected anchor disagreement
    /// surfaces as `AnchorMismatch`.
    #[test]
    fn rejects_deepseek_anchor_swap_forgery() {
        // Honest chain: anchors at (100, root(11)), advances by one block.
        let honest = anchor(100, root(11));
        let s = step(
            StepIO::new(root(11), 100),
            StepIO::new(root(12), 101),
        );
        let honest_ext = extend_with_step_proof(&honest, s).expect("honest extend");

        // Attacker rewrites the proof to claim it anchors at genesis with
        // a different tip. The delta_proofs vec is unchanged — the
        // attacker can't forge step proofs.
        let forged = LatticeTipProofV2 {
            anchor_height: 0,
            anchor_state: [0u8; 32],
            tip_height: 101,
            tip_state: root(12),
            ..honest_ext
        };
        let err = verify_chain_structure(&forged, 0, [0u8; 32]).unwrap_err();
        assert!(
            matches!(
                err,
                VerifyErrorV2::AnchorMismatch { anchor_height: 0 }
                    | VerifyErrorV2::StepCountMismatch { .. }
            ),
            "DeepSeek §0 anchor-swap must be REJECTED, got {err:?}"
        );
    }

    #[test]
    fn rejects_tip_height_inflation() {
        // Honest extend produces (anchor=50, tip=51). Attacker inflates
        // tip_height while leaving delta_proofs unchanged.
        let p = anchor(50, root(7));
        let s = step(
            StepIO::new(root(7), 50),
            StepIO::new(root(8), 51),
        );
        let ext = extend_with_step_proof(&p, s).expect("extend");
        let inflated = LatticeTipProofV2 {
            tip_height: 1_000_000, // attacker lies about tip
            ..ext
        };
        let err = verify_chain_structure(&inflated, 50, root(7)).unwrap_err();
        assert!(
            matches!(err, VerifyErrorV2::StepCountMismatch { .. }),
            "tip_height inflation must be REJECTED, got {err:?}"
        );
    }

    #[test]
    fn rejects_step_count_mismatch() {
        // Two step proofs but tip - anchor = 1 — should fail.
        let mut p = anchor(0, root(1));
        let s1 = step(
            StepIO::new(root(1), 0),
            StepIO::new(root(2), 1),
        );
        p = extend_with_step_proof(&p, s1).expect("first extend");
        // Pretend we extended again without bumping tip_height.
        let s2 = step(
            StepIO::new(root(2), 1),
            StepIO::new(root(3), 2),
        );
        p.delta_proofs.push(s2);
        // tip stays at 1 but delta_proofs.len() = 2 — mismatch.
        let err = verify_chain_structure(&p, 0, root(1)).unwrap_err();
        assert!(
            matches!(err, VerifyErrorV2::StepCountMismatch { vec_len: 2, derived: 1 }),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_chain_discontinuity() {
        // step[0].z_out = root(8); step[1].z_in claims root(99) — spliced.
        let mut p = anchor(0, root(1));
        let s1 = step(
            StepIO::new(root(1), 0),
            StepIO::new(root(8), 1),
        );
        p = extend_with_step_proof(&p, s1).expect("first extend");

        // Manually push a discontinuous step (bypassing extend_with_step_proof).
        let bad_step = step(
            StepIO::new(root(99), 1), // z_in.state_root != root(8)
            StepIO::new(root(10), 2),
        );
        p.delta_proofs.push(bad_step);
        p.tip_height = 2;
        p.tip_state = root(10);

        let err = verify_chain_structure(&p, 0, root(1)).unwrap_err();
        assert!(
            matches!(err, VerifyErrorV2::ChainDiscontinuity { step: 1 }),
            "spliced chain must be REJECTED, got {err:?}"
        );
    }

    #[test]
    fn rejects_first_step_anchor_mismatch() {
        // first step's z_in claims root(99), not the anchor's root(1).
        let mut p = anchor(0, root(1));
        let s = step(
            StepIO::new(root(99), 0), // z_in.state_root != anchor.state_root
            StepIO::new(root(2), 1),
        );
        p.delta_proofs.push(s);
        p.tip_height = 1;
        p.tip_state = root(2);
        let err = verify_chain_structure(&p, 0, root(1)).unwrap_err();
        assert!(
            matches!(err, VerifyErrorV2::AnchorMismatch { anchor_height: 0 }),
            "first-step anchor mismatch must be REJECTED, got {err:?}"
        );
    }

    #[test]
    fn extend_with_step_proof_catches_height_jump() {
        // Attacker tries to append a step at height 5 onto a chain at tip 1.
        let p = anchor(0, root(1));
        let s = step(
            StepIO::new(root(99), 5), // z_in.height != prev.tip_height
            StepIO::new(root(100), 6),
        );
        let err = extend_with_step_proof(&p, s).unwrap_err();
        assert!(matches!(err, VerifyErrorV2::HeightDiscontinuity { .. }));
    }

    #[test]
    fn proof_version_string_is_stable() {
        // The wire-format version string is part of the public ABI —
        // changing it breaks wallets. Pin it.
        assert_eq!(PROOF_VERSION, "latticeguard-rlwe-v1");
        let p = anchor(0, [0u8; 32]);
        assert_eq!(p.version_str(), "latticeguard-rlwe-v1");
    }

    #[test]
    fn soundness_notes_mention_phase_c() {
        // The advisory-only disclosure must survive cargo strings and grep —
        // pin it so a future refactor doesn't quietly drop the disclosure.
        assert!(SOUNDNESS_NOTES.contains("ADVISORY"));
        assert!(SOUNDNESS_NOTES.contains("Phase C"));
        assert!(SOUNDNESS_NOTES.contains("R1csBridge") || SOUNDNESS_NOTES.contains("bridge"));
    }
}
