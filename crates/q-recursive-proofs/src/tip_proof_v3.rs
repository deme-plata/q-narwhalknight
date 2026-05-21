//! Phase C scaffold — tip proof v3 with constant-size Module-SIS
//! folded accumulator.
//!
//! **Status:** TYPE-LEVEL SCAFFOLD. The proof type compiles + serializes
//! + structurally verifies. The CRYPTOGRAPHIC verification path
//! ([`verify_with_folder_v3`]) delegates to
//! [`q_lattice_guard::LatticeFolder`], whose `fold` / `verify` bodies
//! are `todo!()`. Calling those at runtime will panic.
//!
//! The point of shipping this file now:
//!
//!   1. Pin the v3 wire format so the API server can serve `v1` /
//!      `v2` / `v3` interchangeably during the rolling migration.
//!   2. Pin the mandatory DeepSeek §0 anchor-swap regression at the
//!      v3 boundary — when Phase C cryptography lands, the regression
//!      must STILL pass (the structural check is unchanged across
//!      v1 → v2 → v3).
//!   3. Make the Phase B1 → Phase C migration visible in the codebase
//!      via [`PROOF_VERSION_V3`].
//!
//! # The v2 → v3 migration shape
//!
//! Phase B1's `LatticeTipProofV2` carries `Vec<LatticeStepProof>` —
//! proof size grows linearly with chain length (10-50 KB per step).
//! Phase C's `LatticeTipProofV3` replaces the vec with a single
//! [`FoldedInstance`](q_lattice_guard::FoldedInstance) — constant
//! size (target ≤ 8 KB) regardless of chain length.
//!
//! Wire-format change:
//!
//! ```text
//!     V2: { delta_proofs: Vec<LatticeStepProof>, anchor, tip, version }
//!     V3: { accumulator: FoldedInstance,         anchor, tip, version }
//! ```
//!
//! Verifier upgrade flow:
//!
//!   1. Phase B1 → Phase B2: q-ivc-verifier-wasm wired to v2.
//!   2. Phase C (this scaffold): v3 type lands behind upgrade gate
//!      `Upgrade::TipProofV3`. Producers emit v2 + v3 simultaneously
//!      during the upgrade window; consumers consume whichever they
//!      can verify.
//!   3. Phase C activation: consumers prefer v3 (constant-size, ≤10 ms
//!      verify). Producers drop v2 emission once stake threshold confirmed.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use q_lattice_guard::{FoldedInstance, FoldingError, LatticeFolder};

use crate::tip_proof_v2::StateRoot;

// ════════════════════════════════════════════════════════════════════════════
// Wire format
// ════════════════════════════════════════════════════════════════════════════

/// Phase C wire-format version string. Pinned across the entire v3 line.
pub const PROOF_VERSION_V3: &str = "latticefold-modulesis-v1";

/// Constant-size lattice-folded tip proof.
///
/// Compared to [`LatticeTipProofV2`](crate::LatticeTipProofV2): the
/// `Vec<LatticeStepProof>` is replaced by one [`FoldedInstance`], which
/// stays the same size regardless of how many blocks were folded into it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatticeTipProofV3 {
    /// The folded accumulator — constant-size carrier of the entire
    /// proven chain history since the anchor.
    pub accumulator: FoldedInstance,
    /// Verifier-known trust root (genesis or hardcoded checkpoint).
    pub anchor_state: StateRoot,
    pub anchor_height: u64,
    /// Prover's claimed tip.
    pub tip_state: StateRoot,
    pub tip_height: u64,
    /// Wire-format version. Pinned to [`PROOF_VERSION_V3`].
    pub version: String,
}

impl LatticeTipProofV3 {
    /// Number of folds applied (i.e., number of blocks proven since the
    /// anchor). Tracked inside the accumulator's `fold_count`.
    pub fn fold_count(&self) -> u64 {
        self.accumulator.fold_count
    }

    /// Wire-format version this proof carries.
    pub fn version_str(&self) -> &str {
        &self.version
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Error)]
pub enum VerifyErrorV3 {
    #[error("proof version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: String, got: String },

    #[error("proof.anchor_height {got} differs from expected {expected}")]
    AnchorHeightMismatch { expected: u64, got: u64 },

    #[error("proof.anchor_state does not match expected anchor state")]
    AnchorStateMismatch,

    #[error("tip_height {tip} below anchor_height {anchor}")]
    TipBelowAnchor { tip: u64, anchor: u64 },

    #[error("accumulator's public_z does not match (tip_state, tip_height)")]
    PublicZMismatch,

    #[error("accumulator needs re-aggregation (fold_count {fold_count} >= threshold)")]
    NeedsReaggregation { fold_count: u64 },

    #[error("lattice folder error: {0}")]
    Folder(#[from] FoldingError),
}

// ════════════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════════════

/// Anchor a fresh v3 proof at the given trust root. Used at genesis or
/// post-upgrade-flip when starting a new chain segment.
pub fn anchor_v3(
    anchor_height: u64,
    anchor_state: StateRoot,
    folding_params: q_lattice_guard::FoldingParams,
) -> LatticeTipProofV3 {
    let public_z = anchor_z_packed(anchor_height, anchor_state);
    LatticeTipProofV3 {
        accumulator: FoldedInstance::init(folding_params, public_z),
        anchor_state,
        anchor_height,
        tip_state: anchor_state,
        tip_height: anchor_height,
        version: PROOF_VERSION_V3.to_string(),
    }
}

/// Pack a `(state_root, height)` pair into the 9-element `public_z`
/// vector format that matches [`q_ivc::recursion::StepIO::pack`].
///
/// Layout (mirrors [`q_ivc::recursion::StepIO::pack`]):
///   `public_z[0..8]` = state_root as 8 little-endian u32s
///   `public_z[8]`    = height truncated to u32
fn anchor_z_packed(height: u64, state: StateRoot) -> [u32; 9] {
    let mut out = [0u32; 9];
    for (i, c) in state.chunks(4).enumerate() {
        out[i] = u32::from_le_bytes(c.try_into().expect("4 bytes per word"));
    }
    out[8] = (height & 0xFFFF_FFFF) as u32;
    out
}

// ════════════════════════════════════════════════════════════════════════════
// Structural verification (fast — chain header check, no cryptography)
// ════════════════════════════════════════════════════════════════════════════

/// Structural-only verification: version, anchor binding, accumulator
/// public_z match, monotonicity, re-aggregation threshold. Does NOT
/// run the [`LatticeFolder::verify`] cryptographic check (that's
/// `todo!()` in the Phase C scaffold).
///
/// Suitable for the API server's `/api/v1/proof/tip` content-negotiation
/// fast path: reject malformed proofs cheaply before any crypto.
pub fn verify_chain_structure_v3(
    proof: &LatticeTipProofV3,
    expected_anchor_height: u64,
    expected_anchor_state: StateRoot,
) -> Result<(), VerifyErrorV3> {
    if proof.version != PROOF_VERSION_V3 {
        return Err(VerifyErrorV3::VersionMismatch {
            expected: PROOF_VERSION_V3.to_string(),
            got: proof.version.clone(),
        });
    }
    if proof.anchor_height != expected_anchor_height {
        return Err(VerifyErrorV3::AnchorHeightMismatch {
            expected: expected_anchor_height,
            got: proof.anchor_height,
        });
    }
    if proof.anchor_state != expected_anchor_state {
        return Err(VerifyErrorV3::AnchorStateMismatch);
    }
    if proof.tip_height < proof.anchor_height {
        return Err(VerifyErrorV3::TipBelowAnchor {
            tip: proof.tip_height,
            anchor: proof.anchor_height,
        });
    }

    // The accumulator's public_z must reflect the claimed tip.
    let expected_z = anchor_z_packed(proof.tip_height, proof.tip_state);
    if proof.accumulator.public_z != expected_z {
        return Err(VerifyErrorV3::PublicZMismatch);
    }

    // Re-aggregation threshold — past this, the producer should have
    // terminal-SNARK'd the accumulator before serving.
    if proof.accumulator.needs_reaggregation() {
        return Err(VerifyErrorV3::NeedsReaggregation {
            fold_count: proof.accumulator.fold_count,
        });
    }

    Ok(())
}

/// Full cryptographic verification — wraps `verify_chain_structure_v3`
/// then calls [`LatticeFolder::verify`].
///
/// **Phase C boundary:** `LatticeFolder::verify` returns
/// `FoldingError::NotImplemented` until the Module-SIS cryptography
/// lands. Until then this function is structurally-correct but will
/// always surface `VerifyErrorV3::Folder(NotImplemented)` after the
/// structural checks pass.
pub fn verify_with_folder_v3(
    proof: &LatticeTipProofV3,
    expected_anchor_height: u64,
    expected_anchor_state: StateRoot,
    folder: &LatticeFolder,
) -> Result<(), VerifyErrorV3> {
    verify_chain_structure_v3(proof, expected_anchor_height, expected_anchor_state)?;

    let expected_z = anchor_z_packed(proof.tip_height, proof.tip_state);
    folder
        .verify(&proof.accumulator, &expected_z)
        .map_err(VerifyErrorV3::Folder)
}

// ════════════════════════════════════════════════════════════════════════════
// Tests — type shape + structural verify
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use q_lattice_guard::FoldingParams;

    fn root(seed: u8) -> StateRoot {
        let mut r = [0u8; 32];
        for (i, b) in r.iter_mut().enumerate() {
            *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
        }
        r
    }

    #[test]
    fn proof_version_string_is_pinned() {
        assert_eq!(PROOF_VERSION_V3, "latticefold-modulesis-v1");
        let p = anchor_v3(0, [0u8; 32], FoldingParams::pq128_provisional());
        assert_eq!(p.version_str(), "latticefold-modulesis-v1");
    }

    #[test]
    fn anchor_v3_starts_with_zero_folds() {
        let p = anchor_v3(0, [0u8; 32], FoldingParams::pq128_provisional());
        assert_eq!(p.fold_count(), 0);
        assert_eq!(p.anchor_height, 0);
        assert_eq!(p.tip_height, 0);
        assert_eq!(p.anchor_state, [0u8; 32]);
        assert_eq!(p.tip_state, [0u8; 32]);
    }

    #[test]
    fn anchor_v3_at_non_genesis_height() {
        let p = anchor_v3(7777, root(42), FoldingParams::pq128_provisional());
        assert_eq!(p.anchor_height, 7777);
        assert_eq!(p.tip_height, 7777);
        assert_eq!(p.anchor_state, root(42));
    }

    #[test]
    fn structural_verify_accepts_honest_anchor() {
        let p = anchor_v3(0, [0u8; 32], FoldingParams::pq128_provisional());
        verify_chain_structure_v3(&p, 0, [0u8; 32])
            .expect("anchor-only proof must structurally verify");
    }

    #[test]
    fn structural_verify_rejects_wrong_anchor_state() {
        let p = anchor_v3(0, root(1), FoldingParams::pq128_provisional());
        let err = verify_chain_structure_v3(&p, 0, root(99)).unwrap_err();
        assert!(matches!(err, VerifyErrorV3::AnchorStateMismatch));
    }

    #[test]
    fn structural_verify_rejects_wrong_anchor_height() {
        let p = anchor_v3(0, root(1), FoldingParams::pq128_provisional());
        let err = verify_chain_structure_v3(&p, 7, root(1)).unwrap_err();
        assert!(matches!(
            err,
            VerifyErrorV3::AnchorHeightMismatch { expected: 7, got: 0 }
        ));
    }

    #[test]
    fn structural_verify_rejects_wrong_version_string() {
        let mut p = anchor_v3(0, root(1), FoldingParams::pq128_provisional());
        p.version = "latticeguard-rlwe-v1".to_string(); // pretending to be v2
        let err = verify_chain_structure_v3(&p, 0, root(1)).unwrap_err();
        assert!(matches!(err, VerifyErrorV3::VersionMismatch { .. }));
    }

    #[test]
    fn structural_verify_rejects_tip_below_anchor() {
        let mut p = anchor_v3(100, root(1), FoldingParams::pq128_provisional());
        // Force tip below anchor — structurally impossible state.
        p.tip_height = 50;
        // public_z must also reflect tip=50 for the binding check to fail
        // on the height comparison rather than the public_z check.
        p.accumulator.public_z = anchor_z_packed(50, p.tip_state);
        let err = verify_chain_structure_v3(&p, 100, root(1)).unwrap_err();
        assert!(matches!(
            err,
            VerifyErrorV3::TipBelowAnchor { tip: 50, anchor: 100 }
        ));
    }

    /// **Mandatory regression** — the DeepSeek §0 anchor-swap forgery
    /// must remain rejected at the v3 boundary. The class of attack is
    /// the same as in v1 and v2 (substitute the proof's anchor fields
    /// to claim a different trust root). At v3, the structural check
    /// catches it via `AnchorHeightMismatch` / `AnchorStateMismatch` /
    /// `PublicZMismatch`.
    #[test]
    fn rejects_deepseek_anchor_swap_forgery_v3() {
        // Honest proof anchored at (100, root(11)).
        let honest = anchor_v3(100, root(11), FoldingParams::pq128_provisional());

        // Attacker rewrites the header to claim genesis anchor + same tip.
        let forged = LatticeTipProofV3 {
            anchor_height: 0,
            anchor_state: [0u8; 32],
            ..honest
        };

        // Genesis-anchored verifier rejects: either anchor mismatch OR
        // public_z mismatch (because the accumulator's public_z still
        // reflects the original anchor).
        let err = verify_chain_structure_v3(&forged, 0, [0u8; 32]).unwrap_err();
        assert!(
            matches!(
                err,
                VerifyErrorV3::AnchorStateMismatch
                    | VerifyErrorV3::AnchorHeightMismatch { .. }
                    | VerifyErrorV3::PublicZMismatch
            ),
            "DeepSeek §0 anchor-swap MUST be rejected at v3, got {err:?}"
        );
    }

    #[test]
    fn structural_verify_surfaces_reaggregation_threshold() {
        let mut params = FoldingParams::pq128_provisional();
        params.max_folds_before_reagg = 3;
        let mut p = anchor_v3(0, [0u8; 32], params);
        // Simulate an over-folded accumulator that the producer should
        // have terminal-SNARK'd.
        p.accumulator.fold_count = 5;
        let err = verify_chain_structure_v3(&p, 0, [0u8; 32]).unwrap_err();
        assert!(
            matches!(err, VerifyErrorV3::NeedsReaggregation { fold_count: 5 }),
            "over-folded accumulator must surface NeedsReaggregation, got {err:?}"
        );
    }

    #[test]
    fn proof_round_trips_through_bincode() {
        let p = anchor_v3(42, root(7), FoldingParams::pq128_provisional());
        let bytes = bincode::serialize(&p).expect("serialise");
        let parsed: LatticeTipProofV3 = bincode::deserialize(&bytes).expect("deserialise");
        assert_eq!(parsed.anchor_height, 42);
        assert_eq!(parsed.tip_height, 42);
        assert_eq!(parsed.anchor_state, root(7));
        assert_eq!(parsed.version, PROOF_VERSION_V3);
    }
}
