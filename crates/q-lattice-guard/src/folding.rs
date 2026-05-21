//! Phase C scaffold — Module-SIS folding scheme (LatticeFold / LaBRADOR family).
//!
//! **Status:** TYPE-LEVEL SCAFFOLD ONLY. Method bodies are `todo!()`;
//! constructors that don't depend on the cryptography return placeholder
//! values. The point of shipping this file now is to:
//!
//!   1. Lock the public API shape so downstream crates
//!      (`q-recursive-proofs::tip_proof_v3`) can compile their own
//!      scaffolds against it.
//!   2. Make the Phase C boundary visible in the codebase — anyone
//!      reading sees what's pending and where the cryptography needs
//!      to land.
//!   3. Pin the soundness contract via `SOUNDNESS_NOTES` so the
//!      research risks (plan §R1-R5) survive any refactor.
//!
//! **Do NOT call any method on `LatticeFolder` outside of a test marked
//! `#[ignore]`** — the bodies are `todo!()` and will panic.
//!
//! # Cryptographic references
//!
//! - **LatticeFold** (Boneh-Chen 2024): https://eprint.iacr.org/2024/257
//!   The reference design for module-SIS-based folding. Provides the
//!   accumulator structure + folding transform but lands at ~25-50 KB
//!   proofs at PQ128.
//!
//! - **LaBRADOR** (Beullens-Seiler 2023): https://eprint.iacr.org/2022/1341
//!   The decomposition + base-2 folding trick that brings LatticeFold's
//!   25-50 KB proof down toward the 8 KB target. Reference implementation
//!   exists in Sage but not in production Rust.
//!
//! - **Greyhound** (research code): module-SIS commitment scheme used by
//!   LatticeFold for the Ajtai-style commit step.
//!
//! No production Rust implementation of any of these exists today —
//! Phase C ships vendored Sage/Python reference fixtures under
//! `crates/q-lattice-guard/reference/` and cross-checks our Rust impl
//! byte-for-byte against them.
//!
//! # The folding flow
//!
//! ```text
//!     ┌──── per-step LatticeGuard proof (~10-50 KB)
//!     │
//!     ▼
//!  ┌─────────────────────────────────────────────────┐
//!  │ LatticeFolder                                    │
//!  │                                                  │
//!  │  fresh: LatticeGuardProof  ──►  fold(acc, fresh) │
//!  │  acc:   FoldedInstance         w/ challenge α    │
//!  │                                  ▼               │
//!  │                              new_acc             │
//!  │                              (size O(1))         │
//!  └──────────────────────────────────────┬──────────┘
//!                                          ▼
//!                                    LatticeTipProofV3
//!                                    (constant-size:
//!                                    one FoldedInstance,
//!                                    target ≤ 8 KB)
//! ```
//!
//! Verification: `LatticeFolder::verify(acc, expected_z)` runs the
//! relaxed Module-SIS check on the accumulator + asserts the public IO
//! matches `expected_z`. Target ≤10 ms on Pi 4.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    params::{RlweParams, SecurityLevel},
    Polynomial, Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Error)]
pub enum FoldingError {
    #[error("witness norm exceeded bound {bound} (got {got})")]
    WitnessNormExceeded { bound: u64, got: u64 },

    #[error("accumulator error vector exceeded bound {bound} (got {got})")]
    ErrorNormExceeded { bound: u64, got: u64 },

    #[error("public IO mismatch: accumulator z = {acc:?}, expected = {expected:?}")]
    PublicIoMismatch { acc: Vec<u32>, expected: Vec<u32> },

    #[error("commitment verification failed")]
    CommitmentInvalid,

    #[error("folding challenge derivation failed (transcript shape mismatch)")]
    ChallengeError,

    #[error("scaffold method not implemented: {0}")]
    NotImplemented(&'static str),
}

pub type FoldingResult<T> = Result<T, FoldingError>;

// ════════════════════════════════════════════════════════════════════════════
// Parameters
// ════════════════════════════════════════════════════════════════════════════

/// Folding-specific parameters layered on top of [`RlweParams`].
///
/// Values here are **provisional**. The final Phase C parameter set
/// requires:
///   1. LWE-Estimator analysis confirming `(n, q, β_witness, β_error)`
///      yields ≥ 128-bit PQ Module-SIS hardness.
///   2. Folding-error-growth analysis confirming the accumulator stays
///      within `q/4` across ≥ 2^30 fold steps.
///   3. Empirical soak measuring real proof sizes match the ≤ 8 KB
///      target (the LatticeFold reference implementation can be vendored
///      to produce a baseline).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldingParams {
    /// Base RLWE parameters (typically [`RlweParams::pq128_folding`]).
    pub rlwe: RlweParams,
    /// Ajtai-commitment matrix dimensions: `A ∈ R_q^{kappa × m}`.
    /// `kappa` rows = commitment size; `m` cols = witness vector length.
    pub kappa: usize,
    pub m: usize,
    /// Maximum infinity-norm of an honestly-formed witness vector.
    /// `β_witness` in the soundness analysis.
    pub beta_witness: u64,
    /// Maximum infinity-norm of the accumulator's error vector.
    /// `β_error` in the soundness analysis. Folding inflates the error
    /// each step — exceeding this bound necessitates re-aggregation.
    pub beta_error: u64,
    /// Maximum number of fold steps before forced re-aggregation (the
    /// LatticeFold §5 decomposition trick). After N folds the accumulator
    /// is compressed via a terminal SNARK + a fresh accumulator starts.
    pub max_folds_before_reagg: u64,
}

impl FoldingParams {
    /// Provisional 128-bit PQ folding parameters. **NOT CRYPTOGRAPHICALLY
    /// VALIDATED** — see [`ParameterNotes`].
    pub fn pq128_provisional() -> Self {
        Self {
            rlwe: RlweParams::pq128_folding(),
            kappa: 8,
            m: 64,
            beta_witness: 1u64 << 12,
            beta_error: 1u64 << 20,
            max_folds_before_reagg: 1u64 << 20, // ~12 days at 1 bps
        }
    }
}

/// Documentation type — exists to surface the parameter-selection
/// methodology in `cargo doc` output + as a place to hang the LWE
/// Estimator output when it's computed.
pub struct ParameterNotes;

impl ParameterNotes {
    /// Run-this-by-hand command for the LWE Estimator. Produces the
    /// concrete-security bit-count for [`FoldingParams::pq128_provisional`].
    ///
    /// ```text
    /// sage: from estimator import *
    /// sage: LWE.estimate(LWE.Parameters(
    ///         n=1024, q=0x0FFFFFFFEFFE0001,
    ///         Xs=DG(stddev=3.2), Xe=DG(stddev=3.2)))
    /// ```
    ///
    /// Expected output (estimated, pending real run): primal-uSVP ≥ 128 bits,
    /// dual ≥ 130 bits at the chosen `(n, q)`.
    pub const ESTIMATOR_COMMAND: &'static str =
        "sage -c 'from estimator import *; LWE.estimate(...)'";
}

// ════════════════════════════════════════════════════════════════════════════
// AjtaiCommitment
// ════════════════════════════════════════════════════════════════════════════

/// Public Ajtai-style module-SIS commitment matrix.
///
/// `A ∈ R_q^{kappa × m}` — uniformly sampled, structurally fixed for
/// the lifetime of the folder. The commitment of a witness `x ∈ R_q^m`
/// is `A · x mod q`.
///
/// Phase C implementation note: the matrix is derived deterministically
/// from a public seed (similar to Dilithium's `ExpandA`) so the verifier
/// reconstructs it without transmission.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AjtaiCommitment {
    /// Public matrix in row-major order — `kappa` rows × `m` cols,
    /// each entry a `Polynomial` in `R_q`.
    pub matrix: Vec<Vec<Polynomial>>,
    /// 32-byte seed that deterministically expanded into `matrix`.
    /// Verifier re-expands from this seed.
    pub seed: [u8; 32],
    /// Parameters this commitment was built for.
    pub params: FoldingParams,
}

impl AjtaiCommitment {
    /// Deterministically expand a fresh commitment matrix from `seed`.
    ///
    /// **Phase C TODO:** SHAKE-256(seed ‖ row ‖ col) sampled into each
    /// `Polynomial`, modular-reduced into `Z_q`. Mirrors Dilithium's
    /// `ExpandA` shape so the in-circuit verifier (Phase D) can reuse
    /// the existing SHAKE gadget.
    pub fn expand(_seed: [u8; 32], _params: FoldingParams) -> Self {
        todo!(
            "Phase C: SHAKE-256-based deterministic matrix expansion. \
             See LatticeFold §3.1 + Dilithium FIPS-204 §3.2 ExpandA pattern."
        )
    }

    /// Compute the commitment of a witness vector `x ∈ R_q^m`:
    /// `commit = A · x mod q`.
    ///
    /// **Phase C TODO:** matrix-vector multiply over `R_q`, leveraging
    /// the existing NTT (`crate::ntt`) for fast pointwise products.
    pub fn commit(&self, _witness: &[Polynomial]) -> FoldingResult<Vec<Polynomial>> {
        Err(FoldingError::NotImplemented(
            "AjtaiCommitment::commit — vendor LatticeFold reference + port to Rust",
        ))
    }

    /// Verify a commitment opening: check that `A · x ≡ claimed`.
    ///
    /// **Phase C TODO:** same matrix-vector multiply + equality check
    /// over `R_q`.
    pub fn verify_opening(
        &self,
        _witness: &[Polynomial],
        _claimed: &[Polynomial],
    ) -> FoldingResult<()> {
        Err(FoldingError::NotImplemented(
            "AjtaiCommitment::verify_opening — paired with commit() above",
        ))
    }
}

// ════════════════════════════════════════════════════════════════════════════
// FoldedInstance
// ════════════════════════════════════════════════════════════════════════════

/// Relaxed Module-SIS instance — the constant-size accumulator that
/// `LatticeTipProofV3` carries instead of `Vec<LatticeStepProof>`.
///
/// "Relaxed" in the sense of the LatticeFold §4 relaxed-R1CS instance:
/// the accumulator carries the linear combination of all folded steps
/// + a per-step error vector, both bounded by `(β_witness, β_error)`.
///
/// Wire size at PQ128 (provisional): `kappa × n × log2(q) / 8`
///   = 8 × 1024 × 60 / 8 ≈ 7.5 KB for the commitment vector
///   + similar for the witness + error.
/// **Total target: ≤ 8 KB.** Achieved? Pending Phase C implementation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldedInstance {
    /// `A · x` — the commitment to the running witness.
    pub commitment: Vec<Polynomial>,
    /// Slack scalar `μ` from the relaxed instance (LatticeFold §4).
    pub mu: Scalar,
    /// Running witness vector `x`.
    pub witness: Vec<Polynomial>,
    /// Running error vector `e`.
    pub error: Vec<Polynomial>,
    /// Public IO that the next fold step chains against. Mirrors
    /// `q_ivc::recursion::STEP_Z_LEN` so the IVC chain integrates
    /// cleanly.
    pub public_z: [u32; 9],
    /// Number of folds applied since `init`. The folder enforces this
    /// stays below `params.max_folds_before_reagg`.
    pub fold_count: u64,
    /// Parameters this instance was built for.
    pub params: FoldingParams,
}

impl FoldedInstance {
    /// Construct a fresh (empty) accumulator. The commitment is the
    /// zero vector; the witness/error are zero; fold_count is 0.
    pub fn init(params: FoldingParams, anchor_z: [u32; 9]) -> Self {
        let dim = params.rlwe.dimension;
        let kappa = params.kappa;
        let m = params.m;
        let zero_poly = || Polynomial::zero(dim - 1);
        Self {
            commitment: (0..kappa).map(|_| zero_poly()).collect(),
            mu: 0,
            witness: (0..m).map(|_| zero_poly()).collect(),
            error: (0..kappa).map(|_| zero_poly()).collect(),
            public_z: anchor_z,
            fold_count: 0,
            params,
        }
    }

    /// True if the accumulator has hit the re-aggregation threshold —
    /// downstream callers must terminal-SNARK + restart.
    pub fn needs_reaggregation(&self) -> bool {
        self.fold_count >= self.params.max_folds_before_reagg
    }
}

// ════════════════════════════════════════════════════════════════════════════
// LatticeFolder
// ════════════════════════════════════════════════════════════════════════════

/// The Phase C folder. Holds the public Ajtai commitment and provides
/// the `fold` / `verify` operations.
///
/// **All methods on this struct are `todo!()` scaffolds.** The fields
/// + constructor work so downstream crates can typecheck against the
/// real shape.
pub struct LatticeFolder {
    pub params: FoldingParams,
    pub commitment_matrix: Arc<AjtaiCommitment>,
}

impl LatticeFolder {
    /// Construct a folder with a freshly-expanded Ajtai matrix.
    ///
    /// **Phase C TODO:** call [`AjtaiCommitment::expand`] (currently
    /// `todo!()`), then return `Self { params, commitment_matrix }`.
    /// For Phase C scaffold tests, callers should construct
    /// [`AjtaiCommitment`] manually with empty matrix fields and pass
    /// it to [`Self::with_commitment`].
    pub fn new(_params: FoldingParams, _seed: [u8; 32]) -> Self {
        todo!(
            "Phase C: build via AjtaiCommitment::expand(seed, params.clone()). \
             For scaffold tests use Self::with_commitment."
        )
    }

    /// Test-friendly constructor that takes a pre-built commitment.
    /// Used by Phase C scaffold tests to exercise the API shape without
    /// invoking the unimplemented matrix expansion.
    pub fn with_commitment(commitment: Arc<AjtaiCommitment>) -> Self {
        let params = commitment.params.clone();
        Self {
            params,
            commitment_matrix: commitment,
        }
    }

    /// Fold a fresh instance into the accumulator.
    ///
    /// **Phase C algorithm (LatticeFold §4):**
    ///   1. Derive a Fiat-Shamir challenge `α ∈ R_q` from a transcript
    ///      over `(acc.commitment, fresh.commitment, public_z)`.
    ///   2. New witness: `x' = α · acc.witness + (1-α) · fresh.witness`.
    ///   3. New error:   `e' = α · acc.error    + (1-α) · fresh.error`
    ///      + cross-terms from the relaxed-instance linearisation.
    ///   4. New commitment: `A · x' = α · acc.commitment + (1-α) · fresh.commitment`.
    ///   5. Norm checks: assert `||x'||_∞ ≤ β_witness`, `||e'||_∞ ≤ β_error`.
    ///   6. Increment `fold_count`.
    pub fn fold(
        &self,
        _acc: &FoldedInstance,
        _fresh_witness: &[Polynomial],
        _fresh_commitment: &[Polynomial],
        _next_public_z: [u32; 9],
    ) -> FoldingResult<FoldedInstance> {
        Err(FoldingError::NotImplemented(
            "LatticeFolder::fold — implement the LatticeFold §4 transform with norm checks",
        ))
    }

    /// Verify the accumulator is well-formed against the public IO.
    ///
    /// **Phase C verification algorithm:**
    ///   1. Check `||witness||_∞ ≤ β_witness`.
    ///   2. Check `||error||_∞ ≤ β_error`.
    ///   3. Check `A · witness ≡ commitment + μ · error  (mod q)`.
    ///   4. Check `public_z == expected_z`.
    pub fn verify(
        &self,
        acc: &FoldedInstance,
        expected_z: &[u32; 9],
    ) -> FoldingResult<()> {
        // The only piece testable today: public IO mismatch.
        if acc.public_z != *expected_z {
            return Err(FoldingError::PublicIoMismatch {
                acc: acc.public_z.to_vec(),
                expected: expected_z.to_vec(),
            });
        }
        // Everything else is Phase C.
        Err(FoldingError::NotImplemented(
            "LatticeFolder::verify — norm checks + relaxed-SIS check pending Phase C",
        ))
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Soundness notes (constant — survives strings/grep on the binary)
// ════════════════════════════════════════════════════════════════════════════

/// Survives `cargo strings` + `grep` on the binary — used by the build
/// to assert the soundness disclaimer is present.
pub const SOUNDNESS_NOTES: &str = "\
PHASE C folding scaffold — SCAFFOLD ONLY, NOT IMPLEMENTED.

Method bodies on AjtaiCommitment::commit, LatticeFolder::fold, and \
LatticeFolder::verify return FoldingError::NotImplemented. Calling \
them in production will fail at runtime.

Cryptographic risks (plan §R1-R5):
- R1: Module-SIS at PQ128 may not fit ≤8 KB at the chosen (n, q). \
  LatticeFold reference lands at 25-50 KB; LaBRADOR amortization \
  required to reach 8 KB.
- R2: ZK noise flooding may blow the 10 ms verify budget.
- R3: No production Rust crate for LatticeFold/LaBRADOR — vendor Sage \
  reference + cross-check via fixtures.
- R4: Folding error norm grows super-linearly → soundness break at \
  chain length > 2^20. Mitigation: periodic re-aggregation.
- R5: Phase B1 R1csBridge truncates Fr to u64 Scalar. Phase C's \
  pq128_folding (60-bit modulus) closes this to ≥60-bit residual \
  hardness post-projection.

Upgrade gate: Upgrade::TipProofV3 MUST NOT activate binding consensus \
use until Phase C ships + external cryptanalysis confirms parameter \
choices.";

// ════════════════════════════════════════════════════════════════════════════
// Tests — type-shape only, do NOT call todo!() methods
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_params_pq128_provisional_constructs() {
        let p = FoldingParams::pq128_provisional();
        assert_eq!(p.rlwe.dimension, 1024);
        assert_eq!(p.rlwe.modulus_bits, 60);
        assert_eq!(p.kappa, 8);
        assert_eq!(p.m, 64);
        assert!(p.beta_witness > 0);
        assert!(p.beta_error > 0);
        assert!(p.max_folds_before_reagg > 0);
    }

    #[test]
    fn folded_instance_init_has_zero_fold_count() {
        let p = FoldingParams::pq128_provisional();
        let anchor = [0u32; 9];
        let inst = FoldedInstance::init(p, anchor);
        assert_eq!(inst.fold_count, 0);
        assert_eq!(inst.public_z, anchor);
        assert!(!inst.needs_reaggregation());
    }

    #[test]
    fn folded_instance_init_has_kappa_commitment_polys() {
        let p = FoldingParams::pq128_provisional();
        let kappa = p.kappa;
        let m = p.m;
        let inst = FoldedInstance::init(p, [0u32; 9]);
        assert_eq!(inst.commitment.len(), kappa);
        assert_eq!(inst.witness.len(), m);
        assert_eq!(inst.error.len(), kappa);
    }

    #[test]
    fn folded_instance_needs_reaggregation_at_threshold() {
        let mut p = FoldingParams::pq128_provisional();
        p.max_folds_before_reagg = 5;
        let mut inst = FoldedInstance::init(p, [0u32; 9]);
        assert!(!inst.needs_reaggregation());
        inst.fold_count = 4;
        assert!(!inst.needs_reaggregation());
        inst.fold_count = 5;
        assert!(inst.needs_reaggregation());
        inst.fold_count = 100;
        assert!(inst.needs_reaggregation());
    }

    #[test]
    fn folded_instance_round_trips_through_bincode() {
        let p = FoldingParams::pq128_provisional();
        let inst = FoldedInstance::init(p, [7u32; 9]);
        let bytes = bincode::serialize(&inst).expect("serialise");
        let parsed: FoldedInstance = bincode::deserialize(&bytes).expect("deserialise");
        assert_eq!(parsed.public_z, inst.public_z);
        assert_eq!(parsed.fold_count, inst.fold_count);
        assert_eq!(parsed.commitment.len(), inst.commitment.len());
    }

    #[test]
    fn folder_with_commitment_constructor_works() {
        // Build a dummy commitment without invoking AjtaiCommitment::expand
        // (which is todo!()). We only need the API shape to check out.
        let params = FoldingParams::pq128_provisional();
        let dummy_commit = Arc::new(AjtaiCommitment {
            matrix: Vec::new(),
            seed: [0u8; 32],
            params: params.clone(),
        });
        let folder = LatticeFolder::with_commitment(dummy_commit.clone());
        assert_eq!(folder.params.kappa, params.kappa);
        assert!(Arc::ptr_eq(&folder.commitment_matrix, &dummy_commit));
    }

    #[test]
    fn folder_verify_surfaces_public_io_mismatch_first() {
        // verify() is mostly todo!(), but it DOES check public IO
        // before erroring out — that path is testable.
        let params = FoldingParams::pq128_provisional();
        let dummy_commit = Arc::new(AjtaiCommitment {
            matrix: Vec::new(),
            seed: [0u8; 32],
            params: params.clone(),
        });
        let folder = LatticeFolder::with_commitment(dummy_commit);
        let inst = FoldedInstance::init(params, [1u32; 9]);
        let result = folder.verify(&inst, &[99u32; 9]);
        assert!(
            matches!(result, Err(FoldingError::PublicIoMismatch { .. })),
            "public IO mismatch must surface before the NotImplemented error path"
        );
    }

    #[test]
    fn folder_verify_with_matching_io_hits_not_implemented_after_io_check() {
        let params = FoldingParams::pq128_provisional();
        let dummy_commit = Arc::new(AjtaiCommitment {
            matrix: Vec::new(),
            seed: [0u8; 32],
            params: params.clone(),
        });
        let folder = LatticeFolder::with_commitment(dummy_commit);
        let inst = FoldedInstance::init(params, [1u32; 9]);
        let result = folder.verify(&inst, &[1u32; 9]);
        assert!(
            matches!(result, Err(FoldingError::NotImplemented(_))),
            "after IO match, verify must surface NotImplemented (Phase C boundary)"
        );
    }

    #[test]
    fn ajtai_commit_returns_not_implemented_for_phase_c_scaffold() {
        let params = FoldingParams::pq128_provisional();
        let commit = AjtaiCommitment {
            matrix: Vec::new(),
            seed: [0u8; 32],
            params,
        };
        let result = commit.commit(&[]);
        assert!(matches!(result, Err(FoldingError::NotImplemented(_))));
    }

    #[test]
    fn folder_fold_returns_not_implemented_for_phase_c_scaffold() {
        let params = FoldingParams::pq128_provisional();
        let dummy_commit = Arc::new(AjtaiCommitment {
            matrix: Vec::new(),
            seed: [0u8; 32],
            params: params.clone(),
        });
        let folder = LatticeFolder::with_commitment(dummy_commit);
        let inst = FoldedInstance::init(params, [0u32; 9]);
        let result = folder.fold(&inst, &[], &[], [1u32; 9]);
        assert!(matches!(result, Err(FoldingError::NotImplemented(_))));
    }

    #[test]
    fn soundness_notes_mention_phase_c_and_risks() {
        // The Phase C disclosure MUST survive any refactor; pin its
        // content so a future cleanup doesn't quietly drop the disclaimer.
        assert!(SOUNDNESS_NOTES.contains("SCAFFOLD ONLY"));
        assert!(SOUNDNESS_NOTES.contains("R1"));
        assert!(SOUNDNESS_NOTES.contains("R5"));
        assert!(SOUNDNESS_NOTES.contains("LatticeFold"));
        assert!(SOUNDNESS_NOTES.contains("Upgrade::TipProofV3"));
    }

    #[test]
    fn parameter_notes_estimator_command_is_pinned() {
        // Pins the LWE Estimator command path; a future restructure
        // that drops this constant breaks the test.
        assert!(ParameterNotes::ESTIMATOR_COMMAND.contains("estimator"));
    }
}
