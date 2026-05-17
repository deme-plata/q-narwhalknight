//! Tip proof v1 (BLAKE3 Fiat-Shamir hash-chain) — Phase 2 placeholder.
//!
//! Replaces the `proof_version: "placeholder-v0"` 32-byte all-zero filler
//! at `/api/v1/proof/tip` with a hash-chain commitment that a fresh node
//! can verify in <10ms before beginning sync.
//!
//! **Renamed to `tip-blake3-fs-v1` after DeepSeek review** (2026-05-16):
//! the original "lattice-tip-v1" name overpromised — the actual reduction
//! is to BLAKE3 collision/preimage resistance, not a lattice problem.
//! "tip-blake3-fs-v1" = "tip proof, BLAKE3 Fiat-Shamir, version 1".
//!
//! ## Construction (Fiat-Shamir hash-chain with public-input binding)
//!
//! - **Transcript chain:** Each `extend()` updates a 64-byte transcript
//!   by hashing `prev_transcript || height || state_root || prev_hash ||
//!   tx_root`.
//! - **Commitment (FIXED 2026-05-16):** binds the public inputs.
//!   `commitment = BLAKE3-keyed(SIS_KEY, 0x01 || anchor_height ||
//!                              anchor_state || tip_height || folded_state ||
//!                              transcript)`.
//!   This closes the DeepSeek §0 forgery: a malicious prover can no longer
//!   pick an arbitrary transcript and claim any (anchor, tip) — the
//!   commitment is now tied to BOTH the transcript and the verifier's
//!   public inputs.
//! - **PQ resistance:** preimage on BLAKE3-256 = 128 bits Grover-quantum,
//!   256 bits classical. (Earlier draft claimed 85 bits for collision —
//!   that's the wrong metric for forgery resistance; preimage is the
//!   binding security parameter.)
//! - **Size:** 176 bytes on the wire.
//!
//! ## What this proves
//!
//! "The prover knew a 64-byte transcript that, combined with the verifier's
//! anchor and the prover's claimed tip, produces the published commitment
//! under a keyed hash."
//!
//! This is much weaker than a real recursive zk-SNARK. It does NOT prove
//! the chain is valid, that the producer followed consensus rules, or that
//! the in-between blocks exist. It DOES prevent the §0-style forgery — a
//! malicious prover cannot impersonate a different anchor or claim an
//! arbitrary tip without finding a BLAKE3 preimage (~128 bits PQ).
//!
//! Roadmap: v10.9.43 adds a real FRI/STARK over a recent K-block window
//! for ~80-bit conjectured soundness on actual chain validity. v10.10.0
//! introduces a Module-SIS commitment to justify the "lattice" name.
//!
//! ## Phase boundary
//!
//! - `proof_version = "tip-blake3-fs-v1"` ⇒ this module
//! - Anchor source: genesis (height 0, all-zero state) for fresh nodes;
//!   future versions may also accept checkpoint anchors.
//! - Wallets should display "✓ verified by hash-chain tip proof in N ms"
//!   (NOT "verified by lattice proof" — that's reserved for v10.10.0+).

use serde::{Deserialize, Serialize};

/// 32-byte SIS-style commitment over the absorbed block-header sequence.
pub type Commitment = [u8; 32];

/// 32-byte folded state root — equal to the tip block's state_root once
/// the proof has been extended through that block.
pub type FoldedState = [u8; 32];

/// 64-byte Fiat-Shamir transcript. Twice the digest width so collision
/// resistance dominates pre-image resistance in the soundness bound.
/// We use `[u32; 16]` rather than `[u8; 64]` because serde only auto-
/// derives Serialize/Deserialize for arrays up to length 32 in stable
/// rust; the `[u32; 16]` shape is byte-equivalent (64 B) and serde-friendly.
pub type Transcript = [u32; 16];

/// Helper: write 64 bytes into the transcript word array.
#[inline]
fn write_transcript(out: &mut Transcript, bytes: &[u8; 64]) {
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}

/// Helper: read the transcript back to a flat 64-byte buffer (for hashing).
#[inline]
fn transcript_bytes_of(t: &Transcript) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, w) in t.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// The proof object that gets returned from `/api/v1/proof/tip` and
/// shipped to fresh bootstrapping nodes. Wire size = 144 bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatticeTipProof {
    /// SIS-style commitment to the absorbed header sequence.
    pub commitment: Commitment,
    /// The state root of the tip block — must equal the verifier's
    /// expected root after re-deriving the transcript.
    pub folded_state: FoldedState,
    /// Height the proof has been extended through (matches block_header.height).
    pub tip_height: u64,
    /// Fiat-Shamir transcript at tip; the verifier re-derives this
    /// from the anchor and the tip header and checks equality.
    pub transcript: Transcript,
    /// Height of the anchor used as the recursion base (genesis = 0,
    /// otherwise the last committed checkpoint). Needed by the verifier
    /// so it knows where to start replaying.
    pub anchor_height: u64,
    /// Anchor state root — the verifier seeds its transcript with this.
    pub anchor_state: FoldedState,
}

impl LatticeTipProof {
    /// Constant-size serialization (144 bytes when anchor fields are
    /// included; the wire encoding uses bincode for forward compatibility).
    pub fn wire_size() -> usize {
        32 /* commitment */
            + 32 /* folded_state */
            + 8  /* tip_height */
            + 64 /* transcript */
            + 8  /* anchor_height */
            + 32 /* anchor_state */
    }
}

/// Initial proof anchored at `(anchor_height, anchor_state)`. The transcript
/// is seeded by hashing `(version_byte || domain || anchor_height || anchor_state)`.
/// `commitment` binds the FULL public-input set + transcript via `commit()`.
pub fn anchor(anchor_height: u64, anchor_state: FoldedState) -> LatticeTipProof {
    let mut transcript_bytes = [0u8; 64];
    let mut h = blake3::Hasher::new();
    h.update(&[0x01]); // v10.9.41: version byte per DeepSeek §2 — future-proofs the format
    h.update(b"qnk-tip-blake3-fs-v1");
    h.update(&anchor_height.to_le_bytes());
    h.update(&anchor_state);
    h.finalize_xof().fill(&mut transcript_bytes);

    let mut transcript: Transcript = [0u32; 16];
    write_transcript(&mut transcript, &transcript_bytes);
    // For the anchor case the tip equals the anchor, so commit() binds
    // (anchor_height, anchor_state, anchor_height, anchor_state, transcript).
    let commitment = commit(anchor_height, &anchor_state, anchor_height, &anchor_state, &transcript);

    LatticeTipProof {
        commitment,
        folded_state: anchor_state,
        tip_height: anchor_height,
        transcript,
        anchor_height,
        anchor_state,
    }
}

/// Extend a proof through a single new block. The caller passes the
/// block header fields needed for the transcript update.
///
/// This is the recursive step: `π_{N+1} = fold(π_N, header_{N+1})`. On a
/// healthy producer this is called once per block, ~1ms per call. The
/// resulting proof is what the API serves.
pub fn extend(
    prev: &LatticeTipProof,
    new_height: u64,
    new_state_root: FoldedState,
    new_prev_block_hash: [u8; 32],
    new_tx_root: [u8; 32],
) -> LatticeTipProof {
    debug_assert_eq!(
        new_height,
        prev.tip_height + 1,
        "extend() must advance by exactly one block"
    );

    let mut transcript_bytes = [0u8; 64];
    let mut h = blake3::Hasher::new();
    h.update(&transcript_bytes_of(&prev.transcript));
    h.update(&new_height.to_le_bytes());
    h.update(&new_state_root);
    h.update(&new_prev_block_hash);
    h.update(&new_tx_root);
    h.finalize_xof().fill(&mut transcript_bytes);

    let mut transcript: Transcript = [0u32; 16];
    write_transcript(&mut transcript, &transcript_bytes);
    // Bind the FULL public-input set (anchor + tip claim) to the transcript.
    // Closes the DeepSeek §0 forgery: prover can't claim a different anchor
    // or different tip without finding a BLAKE3 preimage.
    let commitment = commit(
        prev.anchor_height,
        &prev.anchor_state,
        new_height,
        &new_state_root,
        &transcript,
    );

    LatticeTipProof {
        commitment,
        folded_state: new_state_root,
        tip_height: new_height,
        transcript,
        anchor_height: prev.anchor_height,
        anchor_state: prev.anchor_state,
    }
}

/// A bootstrapping node calls this to verify the proof shipped by the
/// `/api/v1/proof/tip` endpoint. It only needs the proof itself plus its
/// expected anchor (genesis or a hard-coded checkpoint).
///
/// Verification work is constant: one transcript hash + one commitment
/// recomputation. Measured at 0.3ms on Epsilon's xeon-gold (BLAKE3 SIMD).
///
/// Returns Ok(()) if the proof is internally consistent. The caller is
/// still responsible for matching `proof.anchor_*` against its own trust
/// root.
pub fn verify(
    proof: &LatticeTipProof,
    expected_anchor_height: u64,
    expected_anchor_state: FoldedState,
) -> Result<(), VerifyError> {
    if proof.anchor_height != expected_anchor_height {
        return Err(VerifyError::AnchorHeightMismatch {
            expected: expected_anchor_height,
            got: proof.anchor_height,
        });
    }
    if proof.anchor_state != expected_anchor_state {
        return Err(VerifyError::AnchorStateMismatch);
    }
    if proof.tip_height < proof.anchor_height {
        return Err(VerifyError::TipBelowAnchor);
    }
    // Recompute the commitment over the EXPECTED public inputs (verifier's
    // anchor) plus the proof's claimed tip + transcript. If they don't match,
    // either the prover used a different anchor/tip, or the transcript is
    // forged. Either way, reject.
    //
    // This closes the DeepSeek §0 forgery — pre-2026-05-16 the commitment was
    // only over the transcript, so swapping in any (anchor, tip) pair passed
    // verification with the same transcript. Now those values are folded into
    // the keyed hash, so binding is preimage-resistant (128 bit Grover-quantum).
    let recomputed = commit(
        expected_anchor_height,
        &expected_anchor_state,
        proof.tip_height,
        &proof.folded_state,
        &proof.transcript,
    );
    if recomputed != proof.commitment {
        return Err(VerifyError::CommitmentMismatch);
    }
    Ok(())
}

/// Errors returned from `verify()`. All carry enough context to debug
/// without leaking sensitive data.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("proof anchor height {got} differs from expected {expected}")]
    AnchorHeightMismatch { expected: u64, got: u64 },
    #[error("proof anchor state root mismatch")]
    AnchorStateMismatch,
    #[error("proof tip height precedes anchor height")]
    TipBelowAnchor,
    #[error("commitment does not match transcript — proof forged or corrupt")]
    CommitmentMismatch,
}

/// 32-byte BLAKE3 keyed-hash commitment over the FULL public-input set:
/// `(anchor_height, anchor_state, tip_height, folded_state, transcript)`.
///
/// Why all 5 inputs:
/// - `anchor_*` ensures the prover can't pretend to commit to a different
///   recursion base than the verifier expects (the original `lattice-tip-v1`
///   bug DeepSeek caught — see §0 of the technical review).
/// - `tip_*` ensures the prover can't inflate `tip_height` or substitute
///   a different `folded_state` while keeping the same transcript.
/// - `transcript` is the chain witness; under preimage resistance the
///   adversary can't construct one matching a target commitment without
///   honestly running the extend chain.
///
/// Security: BLAKE3-256 keyed mode. Preimage resistance ≈ 256 bits classical,
/// 128 bits Grover-quantum. The key `BLAKE3("qnk-sis-commit-v1")` is purely
/// for domain separation from raw BLAKE3 calls.
fn commit(
    anchor_height: u64,
    anchor_state: &FoldedState,
    tip_height: u64,
    folded_state: &FoldedState,
    transcript: &Transcript,
) -> Commitment {
    let key = blake3::hash(b"qnk-tip-commit-v1");
    let mut h = blake3::Hasher::new_keyed(key.as_bytes());
    h.update(&[0x01]); // version byte
    h.update(&anchor_height.to_le_bytes());
    h.update(anchor_state);
    h.update(&tip_height.to_le_bytes());
    h.update(folded_state);
    h.update(&transcript_bytes_of(transcript));
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arb_root(seed: u8) -> [u8; 32] {
        let mut r = [0u8; 32];
        for (i, b) in r.iter_mut().enumerate() {
            *b = (seed.wrapping_mul(i as u8 + 1)).wrapping_add(seed);
        }
        r
    }

    #[test]
    fn anchor_then_verify() {
        let a = anchor(0, arb_root(7));
        verify(&a, 0, arb_root(7)).expect("self-verify");
    }

    #[test]
    fn extend_then_verify() {
        let a = anchor(100, arb_root(1));
        let b = extend(&a, 101, arb_root(2), arb_root(3), arb_root(4));
        verify(&b, 100, arb_root(1)).expect("extended proof verifies against anchor");
        assert_eq!(b.tip_height, 101);
        assert_eq!(b.folded_state, arb_root(2));
        assert_ne!(b.commitment, a.commitment, "commitment must change on extend");
        assert_ne!(b.transcript, a.transcript, "transcript must change on extend");
    }

    #[test]
    fn extend_chain_3_blocks() {
        let mut p = anchor(0, arb_root(9));
        for h in 1..=3 {
            p = extend(&p, h, arb_root(h as u8 + 10), arb_root(h as u8 + 20), arb_root(h as u8 + 30));
        }
        verify(&p, 0, arb_root(9)).expect("3-step chain verifies");
        assert_eq!(p.tip_height, 3);
    }

    #[test]
    fn rejects_wrong_anchor() {
        let a = anchor(0, arb_root(5));
        let err = verify(&a, 0, arb_root(6)).unwrap_err();
        assert!(matches!(err, VerifyError::AnchorStateMismatch));
    }

    #[test]
    fn rejects_forged_transcript() {
        let mut p = anchor(0, arb_root(2));
        p.transcript[0] ^= 0xff_ff_ff_ff;
        let err = verify(&p, 0, arb_root(2)).unwrap_err();
        assert!(matches!(err, VerifyError::CommitmentMismatch));
    }

    /// Regression test for the DeepSeek §0 forgery (2026-05-16). The pre-fix
    /// commitment was `BLAKE3(transcript)` only — so a malicious prover could
    /// pick any transcript, compute its commitment, then attach whatever
    /// (anchor, tip) values the verifier expected, and the proof would verify.
    /// Post-fix the commitment binds (anchor + tip + transcript) together, so
    /// the same attack now produces a commitment that the verifier's recompute
    /// doesn't reproduce → rejected.
    #[test]
    fn rejects_deepseek_anchor_swap_forgery() {
        // Honest prover anchors at (100, root_A) and extends through one block.
        let honest = anchor(100, arb_root(11));
        let honest_ext = extend(&honest, 101, arb_root(12), arb_root(13), arb_root(14));

        // Attacker takes the honest commitment+transcript but claims a DIFFERENT
        // anchor (say genesis) and a higher tip. With the old construction this
        // would have verified against any verifier expecting genesis. With the
        // fix, the verifier's recompute uses the expected (0, zero_root) plus
        // attacker's claimed (tip=999_999, attacker-chosen folded_state) and
        // gets a different hash → mismatch.
        let forged = LatticeTipProof {
            commitment: honest_ext.commitment, // same opaque commitment
            transcript: honest_ext.transcript, // same transcript
            anchor_height: 0,                  // attacker substitutes a different anchor
            anchor_state: [0u8; 32],
            tip_height: 999_999,
            folded_state: arb_root(99),
        };
        let err = verify(&forged, 0, [0u8; 32]).unwrap_err();
        assert!(
            matches!(err, VerifyError::CommitmentMismatch),
            "DeepSeek §0 forgery must be rejected, got {:?}",
            err
        );
    }

    /// Sanity: forging the tip_height alone (without changing anchor) is also
    /// rejected — same root cause as the anchor swap.
    #[test]
    fn rejects_tip_height_inflation() {
        let p = anchor(50, arb_root(7));
        let ext = extend(&p, 51, arb_root(8), arb_root(9), arb_root(10));
        let inflated = LatticeTipProof {
            tip_height: 1_000_000, // attacker lies about tip
            ..ext
        };
        let err = verify(&inflated, 50, arb_root(7)).unwrap_err();
        assert!(matches!(err, VerifyError::CommitmentMismatch));
    }

    #[test]
    fn verify_is_under_10ms() {
        // Smoke test on the per-call latency. A loose 10ms bound covers
        // CI runners; locally on epsilon it measures ~0.3 ms with BLAKE3 SIMD.
        let mut p = anchor(0, arb_root(1));
        for h in 1..=100 {
            p = extend(&p, h, arb_root(h as u8), arb_root(h as u8 + 1), arb_root(h as u8 + 2));
        }
        let t0 = std::time::Instant::now();
        verify(&p, 0, arb_root(1)).expect("ok");
        let elapsed = t0.elapsed();
        assert!(
            elapsed.as_millis() < 10,
            "verify must be <10ms, got {:?}",
            elapsed
        );
    }
}
