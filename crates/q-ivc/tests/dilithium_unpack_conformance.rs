//! Phase A task #7 conformance: cross-check our FIPS-204 byte unpackers
//! against real Dilithium5 signatures from `pqcrypto-dilithium`.
//!
//! The host-side unpackers
//! ([`DilithiumKeyBytes`](q_ivc::host::dilithium_witness::DilithiumKeyBytes) /
//! [`DilithiumSigBytes`](q_ivc::host::dilithium_witness::DilithiumSigBytes))
//! parse the wire format produced by any FIPS-204 conformant
//! implementation. The tip-of-spec reference is `pqcrypto-dilithium`'s
//! `dilithium5` (NIST ML-DSA-87). Generating a real signature here gives
//! us:
//!
//! 1. Byte-length validation — `from_slice` accepts the exact length
//!    pqcrypto emits.
//! 2. Per-coefficient range validation — `unpack_t1_native` /
//!    `unpack_z_native` produce values within the spec's bounds.
//! 3. Per-hint sparsity validation — `unpack_h_native` returns ≤ω = 75
//!    set bits total across all `K = 8` polynomials.
//!
//! What this file does NOT do:
//!
//! - Re-pack the unpacked values to bytes (no packer exists in the host
//!   helpers; the in-circuit verifier consumes the unpacked form).
//! - Cross-check `expand_a_native` digests — that requires an external
//!   reference matrix and is gated in `expand_a_conformance.rs`.
//! - Run the in-circuit verifier itself — that's task #6's scope and
//!   needs ~1.5M constraints/tx of in-R1CS work.

use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey};

use q_ivc::host::dilithium_witness::{
    DilithiumKeyBytes, DilithiumSigBytes, DILITHIUM5_PK_BYTES,
    DILITHIUM5_SIG_BYTES, GAMMA1, K, L, N, OMEGA,
};

/// Sign a message and return the wire bytes from the production
/// Dilithium5 implementation.
fn sign_message(message: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let (pk, sk) = dilithium5::keypair();
    let sig = dilithium5::detached_sign(message, &sk);
    (pk.as_bytes().to_vec(), sig.as_bytes().to_vec())
}

#[test]
fn pqcrypto_pk_length_matches_our_constant() {
    let (pk, _) = sign_message(b"length check");
    assert_eq!(
        pk.len(),
        DILITHIUM5_PK_BYTES,
        "pqcrypto pk bytes ({}) must equal DILITHIUM5_PK_BYTES ({})",
        pk.len(),
        DILITHIUM5_PK_BYTES
    );
}

#[test]
fn pqcrypto_sig_length_matches_our_constant() {
    let (_, sig) = sign_message(b"length check");
    assert_eq!(
        sig.len(),
        DILITHIUM5_SIG_BYTES,
        "pqcrypto sig bytes ({}) must equal DILITHIUM5_SIG_BYTES ({})",
        sig.len(),
        DILITHIUM5_SIG_BYTES
    );
}

#[test]
fn dilithium_key_bytes_accepts_pqcrypto_pk() {
    let (pk, _) = sign_message(b"accept pk");
    let wrapped =
        DilithiumKeyBytes::from_slice(&pk).expect("DilithiumKeyBytes::from_slice must accept pqcrypto pk");

    // rho() must extract bytes 0..32 verbatim.
    let rho = wrapped.rho();
    assert_eq!(&pk[..32], &rho[..], "rho extraction must match raw bytes");
}

#[test]
fn dilithium_sig_bytes_accepts_pqcrypto_sig() {
    let (_, sig) = sign_message(b"accept sig");
    DilithiumSigBytes::from_slice(&sig)
        .expect("DilithiumSigBytes::from_slice must accept pqcrypto sig");
}

#[test]
fn unpack_t1_yields_k_polynomials_in_10bit_range() {
    let (pk, _) = sign_message(b"t1 range check");
    let wrapped = DilithiumKeyBytes::from_slice(&pk).expect("accept pk");
    let t1 = wrapped.unpack_t1_native();

    // K = 8 polynomials.
    assert_eq!(t1.len(), K);
    // Each polynomial has N = 256 coefficients.
    for poly in &t1 {
        assert_eq!(poly.len(), N);
    }

    // FIPS-204 t1 = upper bits of the secret key polynomial t; each
    // coefficient lives in [0, 2^10) = [0, 1024).
    let bound: u32 = 1 << 10;
    for (poly_idx, poly) in t1.iter().enumerate() {
        for (coeff_idx, &coeff) in poly.iter().enumerate() {
            assert!(
                coeff < bound,
                "t1[{}][{}] = {} out of [0, 2^10) range",
                poly_idx,
                coeff_idx,
                coeff
            );
        }
    }
}

#[test]
fn unpack_z_yields_l_polynomials_in_encoded_range() {
    let (_, sig) = sign_message(b"z range check");
    let wrapped = DilithiumSigBytes::from_slice(&sig).expect("accept sig");
    let z = wrapped.unpack_z_native();

    // L = 7 polynomials.
    assert_eq!(z.len(), L);
    for poly in &z {
        assert_eq!(poly.len(), N);
    }

    // FIPS-204 z encoding uses 20 bits per coefficient (γ1 = 2^19), so
    // the encoded representation lives in [0, 2*γ1) = [0, 2^20).
    let bound: u32 = 2 * GAMMA1;
    for (poly_idx, poly) in z.iter().enumerate() {
        for (coeff_idx, &coeff) in poly.iter().enumerate() {
            assert!(
                coeff < bound,
                "z[{}][{}] = {} out of [0, 2*γ1) range (γ1 = {})",
                poly_idx,
                coeff_idx,
                coeff,
                GAMMA1
            );
        }
    }
}

#[test]
fn unpack_h_yields_hint_bits_respecting_omega_bound() {
    let (_, sig) = sign_message(b"h sparsity check");
    let wrapped = DilithiumSigBytes::from_slice(&sig).expect("accept sig");
    let h = wrapped
        .unpack_h_native()
        .expect("hint vector must unpack from a valid signature");

    // K = 8 polynomials.
    assert_eq!(h.len(), K);
    for poly in &h {
        assert_eq!(poly.len(), N);
    }

    // FIPS-204 ω-bound: total number of set hint bits across all K
    // polynomials must be ≤ ω = 75.
    let total_set: usize = h
        .iter()
        .map(|poly| poly.iter().filter(|&&b| b).count())
        .sum();
    assert!(
        total_set <= OMEGA,
        "hint bit count {} exceeds ω = {}",
        total_set,
        OMEGA
    );
}

#[test]
fn distinct_messages_produce_distinct_signatures() {
    let (_, sig_a) = sign_message(b"message A");
    let (_, sig_b) = sign_message(b"message B");
    assert_ne!(sig_a, sig_b, "different messages must yield different sigs");

    // Both signatures must unpack cleanly.
    DilithiumSigBytes::from_slice(&sig_a)
        .expect("sig_a")
        .unpack_z_native();
    DilithiumSigBytes::from_slice(&sig_b)
        .expect("sig_b")
        .unpack_z_native();
}

#[test]
fn empty_message_signature_still_unpacks() {
    let (pk, sig) = sign_message(b"");
    DilithiumKeyBytes::from_slice(&pk).expect("pk for empty-msg sig");
    let wrapped = DilithiumSigBytes::from_slice(&sig).expect("sig for empty msg");
    let z = wrapped.unpack_z_native();
    assert_eq!(z.len(), L);
}

#[test]
fn long_message_signature_unpacks() {
    // 64 KB message — exercises any chunked SHAKE absorption in the
    // signer + verifies our unpackers don't care about message length.
    let long_msg = vec![0xA5u8; 64 * 1024];
    let (_, sig) = sign_message(&long_msg);
    let wrapped = DilithiumSigBytes::from_slice(&sig).expect("sig for long msg");
    wrapped.unpack_z_native();
    wrapped.unpack_h_native().expect("h for long msg");
}

#[test]
fn ten_keypairs_each_unpack_cleanly() {
    // Loop a few keypairs to catch any rng-sensitive edge case in the
    // unpackers (e.g. an off-by-one in bit-extraction that fires only
    // when certain coefficient patterns appear).
    for round in 0..10 {
        let msg = format!("round {round}");
        let (pk, sig) = sign_message(msg.as_bytes());

        let k = DilithiumKeyBytes::from_slice(&pk).expect("pk round");
        let _t1 = k.unpack_t1_native();
        let _rho = k.rho();

        let s = DilithiumSigBytes::from_slice(&sig).expect("sig round");
        let _z = s.unpack_z_native();
        let _h = s.unpack_h_native().expect("h round");
    }
}

#[test]
fn pqcrypto_verifies_its_own_signatures() {
    // Sanity: pqcrypto's own verifier accepts what its signer produces.
    // Establishes baseline correctness independent of our unpackers
    // (if THIS fails, the pqcrypto dep is broken, not our code).
    use pqcrypto_traits::sign::SecretKey;
    let (pk, sk) = dilithium5::keypair();
    let msg = b"self-verify check";
    let sig = dilithium5::detached_sign(msg, &sk);
    let _ = (sk.as_bytes().len(), pk.as_bytes().len()); // touch both sides
    let result = dilithium5::verify_detached_signature(&sig, msg, &pk);
    assert!(result.is_ok(), "pqcrypto self-verify must succeed");
}
