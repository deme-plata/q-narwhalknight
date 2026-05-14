//! WASM bindings for the Quillon Graph recursive proof verifier.
//!
//! This crate is the **browser-side** verifier for the trustless light-client
//! bootstrap described in the whitepaper. It compiles to wasm32-unknown-unknown
//! and is loaded by the js-libp2p browser wallet at quillon.xyz.
//!
//! ## Phase 1 — scaffolding (current)
//!
//! The WASM build pipeline, the JS API, and IndexedDB caching all work end-to-end.
//! The verifier function `verify_proof_bytes` is a placeholder that returns true
//! for any non-empty proof. This is intentional — the JS-side wallet plumbing
//! can be wired and tested without depending on Nova being ready. **Phase 1
//! provides NO cryptographic security on its own.** Browser UI must show a
//! prominent warning while `verifier_version()` returns "placeholder-v0".
//!
//! ## Phase 2 — real Nova verification (months out)
//!
//! Once the Nova IVC wrapper lands (`crates/q-ivc/src/recursion/`), this file's
//! `verify_proof_bytes` body is replaced with a real `nova_snark::RecursiveSNARK::verify`
//! call against bundled public parameters. The JS API is unchanged. `verifier_version()`
//! becomes "nova-bn254-v1". Phase 4 swaps to a lattice scheme; the JS API still
//! does not change.
//!
//! See `docs/deepseek-handoff-wasm-browser-verifier-2026-05-13.md` for the full spec.

#![forbid(unsafe_code)]

use wasm_bindgen::prelude::*;

/// Verifier proof-system version.
///
/// * `"placeholder-v0"` — Phase 1 scaffolding. Returns true for any non-empty proof.
///   Wallets MUST show a warning when this is the active version.
/// * `"nova-bn254-v1"` — Phase 2. Real Nova recursive verification over BN254.
/// * `"latticefold-modulesis-v1"` — Phase 4. Post-quantum lattice scheme.
#[wasm_bindgen]
pub fn verifier_version() -> String {
    "placeholder-v0".to_string()
}

/// Health-check the JS side calls after `init()` returns.
#[wasm_bindgen]
pub fn verifier_ready() -> bool {
    true
}

/// Verify a recursive proof against an expected state-root + tip height.
///
/// # Arguments
///
/// * `state_root_bytes` — 32-byte BLAKE3 SMT root v2
/// * `tip_height` — `u64` from the block header
/// * `proof_bytes` — serialized recursive proof (Nova/lattice depending on phase)
///
/// # Returns
///
/// `true` iff the proof verifies against `state_root_bytes` at `tip_height`.
///
/// # ⚠️  Phase 1 PLACEHOLDER
///
/// Returns `true` for any non-empty `proof_bytes` paired with a 32-byte state root.
/// This is wire-protocol-shape correct but provides ZERO cryptographic security.
/// Wallets must check `verifier_version()` and show a warning banner while it
/// returns `"placeholder-v0"`.
#[wasm_bindgen]
pub fn verify_proof_bytes(
    state_root_bytes: &[u8],
    tip_height: u64,
    proof_bytes: &[u8],
) -> bool {
    // Defensive input checks — these stay across all phases.
    if state_root_bytes.len() != 32 {
        return false;
    }
    if proof_bytes.is_empty() {
        return false;
    }
    // tip_height bounds: u64::MAX would indicate corruption / non-finite block
    if tip_height == u64::MAX {
        return false;
    }
    let _ = tip_height; // currently unused; Phase 2 uses it as a public input

    // ════════════════════════════════════════════════════════════════════════
    // PHASE 2 REPLACE: real Nova/BN254 verification.
    //
    // let recursive_snark: nova_snark::RecursiveSNARK<G1, G2, C1, C2> =
    //     bincode::deserialize(proof_bytes).map_err(...)?;
    // let z_final = pack_state_root_to_field_pair(state_root_bytes);
    // recursive_snark.verify(&PUBLIC_PARAMS, tip_height as usize, &[z_final])
    //     .map(|_| true)
    //     .unwrap_or(false)
    //
    // PHASE 1 placeholder:
    // ════════════════════════════════════════════════════════════════════════
    true
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_placeholder_in_phase_1() {
        assert_eq!(verifier_version(), "placeholder-v0");
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
    fn placeholder_accepts_wellformed_input() {
        let sr = [0u8; 32];
        let proof = vec![1, 2, 3];
        assert!(verify_proof_bytes(&sr, 100, &proof));
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
