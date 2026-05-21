//! WASM bindings for the Quillon Graph recursive proof verifier.
//!
//! This crate is the **browser-side** verifier loaded by wallets at
//! `quillon.xyz`. Compiles to `wasm32-unknown-unknown` and is consumed
//! via wasm-bindgen.
//!
//! # Phase B2 — real LatticeGuard structural verifier
//!
//! [`verify_proof_bytes`] now performs the full Phase B1 chain-integrity
//! verification: deserialize the bincode payload into a
//! [`LatticeTipProofV2`](q_recursive_proofs::LatticeTipProofV2), check
//! anchor binding + monotonicity + version pinning + per-step chain
//! continuity. Returns `true` iff the proof verifies against the
//! caller-supplied anchor (genesis or a hardcoded checkpoint).
//!
//! **Soundness regime:** the Phase B1 structural verifier proves
//! end-to-end chain integrity (defeats DeepSeek §0 anchor-swap, replay,
//! splicing) but does NOT cryptographically verify each step's
//! `LatticeGuardProof`. Per-step crypto is Phase B3 (browser-side
//! LatticeGuardVerifier — possible but currently bandwidth-prohibitive
//! at 10-50KB per step). Wallets render the verifier banner as
//! "advisory — chain integrity verified" through the Phase B1 window.
//!
//! # API stability
//!
//! - `verifier_version()`: returns the `proof_version` string the verifier
//!   accepts. Bumped from `"placeholder-v0"` → `"latticeguard-rlwe-v1"`.
//!   Wallets that string-match on the old value will see the change.
//! - `verify_proof_bytes(state_root, tip_height, proof_bytes)`:
//!   unchanged shape; semantics now real.
//! - `verifier_ready()`, `hex_encode/decode`: unchanged.

#![forbid(unsafe_code)]

use wasm_bindgen::prelude::*;

use q_recursive_proofs::{
    tip_verify_v2, LatticeTipProofV2, TIP_PROOF_V2_VERSION as ACCEPTED_VERSION,
};

/// Verifier proof-system version.
///
/// Returns the wire-format string the verifier accepts on
/// [`verify_proof_bytes`]. Wallets MUST render the verifier banner with
/// this value so users see the actual security regime.
///
/// Version history:
/// - `"placeholder-v0"` — pre-Phase-B2 scaffolding (true for any non-empty body)
/// - `"latticeguard-rlwe-v1"` — Phase B2 chain-integrity verifier (this build)
/// - `"latticefold-modulesis-v1"` — Phase C, with constant-size folded proof
/// - `"latticesnark-pq128-v1"` — Phase D, terminal SNARK over the fold
#[wasm_bindgen]
pub fn verifier_version() -> String {
    ACCEPTED_VERSION.to_string()
}

/// Health-check the JS side calls after `init()` returns.
#[wasm_bindgen]
pub fn verifier_ready() -> bool {
    true
}

/// Verify a recursive proof against an expected genesis anchor + tip
/// height.
///
/// # Arguments
///
/// * `state_root_bytes` — 32-byte BLAKE3 state root the wallet expects
///   the tip to commit to. The verifier additionally enforces this
///   matches `proof.tip_state`.
/// * `tip_height` — `u64` from the proof's claimed tip. The verifier
///   enforces this matches the proof's `tip_height` field.
/// * `proof_bytes` — bincode'd `LatticeTipProofV2`.
///
/// # Returns
///
/// `true` iff:
/// 1. `proof_bytes` deserializes as a `LatticeTipProofV2`
/// 2. `proof.tip_height == tip_height` (caller's claim matches)
/// 3. `proof.tip_state == state_root_bytes` (caller's claim matches)
/// 4. `tip_verify_v2(&proof, 0, [0u8; 32])` passes (genesis-anchored
///    chain integrity: version, endpoints, monotonicity, no splices)
///
/// Genesis is hardcoded as the anchor — Phase B2 wallets bootstrap
/// from `(0, [0u8; 32])`. Checkpoint-anchored bootstrap is a Phase B3
/// API addition.
#[wasm_bindgen]
pub fn verify_proof_bytes(
    state_root_bytes: &[u8],
    tip_height: u64,
    proof_bytes: &[u8],
) -> bool {
    // ── Defensive input checks (stay across all phases) ──────────
    if state_root_bytes.len() != 32 {
        return false;
    }
    if proof_bytes.is_empty() {
        return false;
    }
    if tip_height == u64::MAX {
        return false;
    }

    // ── Deserialize ─────────────────────────────────────────────
    let proof: LatticeTipProofV2 = match bincode::deserialize(proof_bytes) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // ── Caller's claims must match the proof's header ───────────
    if proof.tip_height != tip_height {
        return false;
    }
    let claimed_root: [u8; 32] = match state_root_bytes.try_into() {
        Ok(r) => r,
        Err(_) => return false,
    };
    if proof.tip_state != claimed_root {
        return false;
    }

    // ── Chain integrity against the hardcoded genesis anchor ────
    // Phase B2 wallets pin to genesis. The structural verifier
    // checks version + anchor binding + monotonicity + chain splice.
    tip_verify_v2(&proof, 0, [0u8; 32]).is_ok()
}

/// Hex-encode bytes. Debugging helper exposed to JS.
#[wasm_bindgen]
pub fn hex_encode(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Hex-decode a string. Returns an empty Vec on invalid input.
#[wasm_bindgen]
pub fn hex_decode(s: &str) -> Vec<u8> {
    hex::decode(s).unwrap_or_default()
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════
//
// These run on the host (cargo test) — wasm-bindgen-test would run them
// in a headless browser via `wasm-pack test --headless`. Both paths are
// supported because the underlying logic is identical on wasm32 and
// host targets (bincode + tip_verify_v2 are pure-Rust).

#[cfg(test)]
mod tests {
    use super::*;
    use q_recursive_proofs::{
        tip_anchor_v2 as v2_anchor, tip_extend_v2 as v2_extend,
        LatticeTipProofV2,
    };

    use q_ivc::recursion::{LatticeStepProof, StepIO};
    use q_lattice_guard::{
        params::SecurityLevel, prover::ProofMetadata, LatticeGuardProof,
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

    fn honest_proof_to_height(n: u64) -> LatticeTipProofV2 {
        let mut p = v2_anchor(0, [0u8; 32]);
        let mut prev = [0u8; 32];
        for h in 0..n {
            let next = root((h + 1) as u8);
            p = v2_extend(&p, step(StepIO::new(prev, h), StepIO::new(next, h + 1)))
                .expect("honest extend");
            prev = next;
        }
        p
    }

    #[test]
    fn version_is_latticeguard_rlwe_v1() {
        assert_eq!(verifier_version(), "latticeguard-rlwe-v1");
    }

    #[test]
    fn verifier_ready_returns_true() {
        assert!(verifier_ready());
    }

    #[test]
    fn rejects_wrong_state_root_length() {
        let too_short = [0u8; 16];
        let proof = vec![1, 2, 3];
        assert!(!verify_proof_bytes(&too_short, 100, &proof));
    }

    #[test]
    fn rejects_empty_proof() {
        let sr = [0u8; 32];
        let empty: Vec<u8> = vec![];
        assert!(!verify_proof_bytes(&sr, 100, &empty));
    }

    #[test]
    fn rejects_u64_max_tip_height() {
        let sr = [0u8; 32];
        let proof = vec![1, 2, 3];
        assert!(!verify_proof_bytes(&sr, u64::MAX, &proof));
    }

    #[test]
    fn rejects_malformed_proof_bytes() {
        let sr = [0u8; 32];
        let garbage = vec![0xFFu8; 100];
        assert!(!verify_proof_bytes(&sr, 1, &garbage));
    }

    #[test]
    fn accepts_honest_proof_to_height_3() {
        let proof = honest_proof_to_height(3);
        let bytes = bincode::serialize(&proof).expect("serialise");
        assert!(verify_proof_bytes(&proof.tip_state, 3, &bytes));
    }

    #[test]
    fn rejects_proof_with_wrong_claimed_tip_height() {
        let proof = honest_proof_to_height(3);
        let bytes = bincode::serialize(&proof).expect("serialise");
        // Caller claims a different tip height than what the proof
        // actually attests.
        assert!(!verify_proof_bytes(&proof.tip_state, 99, &bytes));
    }

    #[test]
    fn rejects_proof_with_wrong_claimed_tip_state() {
        let proof = honest_proof_to_height(3);
        let bytes = bincode::serialize(&proof).expect("serialise");
        let wrong_root = [0xAAu8; 32];
        assert!(!verify_proof_bytes(&wrong_root, 3, &bytes));
    }

    #[test]
    fn rejects_anchor_swap_forgery_end_to_end() {
        // Attacker constructs a "valid-looking" proof with a swapped
        // anchor. The wasm verifier (anchored at genesis) must reject.
        let honest = honest_proof_to_height(3);
        let forged = LatticeTipProofV2 {
            anchor_height: 100,
            anchor_state: root(99),
            ..honest.clone()
        };
        let bytes = bincode::serialize(&forged).expect("serialise");
        assert!(!verify_proof_bytes(&forged.tip_state, 3, &bytes));
    }

    #[test]
    fn hex_round_trip() {
        let bytes = vec![0xab, 0xcd, 0xef];
        let hex_str = hex_encode(&bytes);
        assert_eq!(hex_str, "abcdef");
        assert_eq!(hex_decode(&hex_str), bytes);
    }

    #[test]
    fn hex_decode_invalid_returns_empty() {
        assert_eq!(hex_decode("not-hex"), Vec::<u8>::new());
    }
}
