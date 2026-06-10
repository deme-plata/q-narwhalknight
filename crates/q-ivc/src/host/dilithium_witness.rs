//! Host-side helpers: FIPS-204 byte-format → in-circuit PublicKeyVar /
//! SignatureVar.
//!
//! Closes TODO `delta-circuit-PHASE-1C-final` from
//! `crates/q-ivc/src/circuits/delta_block.rs` once the bit-unpacking
//! bodies below are filled in. This file ships the **structural
//! skeleton**: types, constructors, allocator signatures, and the full
//! FIPS-204 spec reference for each unpacking step.
//!
//! # Dilithium5 parameter set
//!
//! | Parameter | Value |
//! |-----------|-------|
//! | n         | 256   |
//! | q         | 8 380 417 |
//! | k         | 8     |
//! | l         | 7     |
//! | η         | 2     |
//! | γ₁        | 2¹⁹    |
//! | γ₂        | (q − 1) / 32 = 261 888 |
//! | τ         | 60    |
//! | ω         | 75    |
//! | β = τ · η | 120   |
//!
//! Public-key byte length: 2 592 (32 bytes ρ + 8 × 320 bytes packed t₁)
//! Signature byte length: 4 627 (32 bytes c̃ + 7 × 627 bytes packed z
//!                                  + 83 bytes packed h)
//!
//! These are the wire formats produced by `pqcrypto-dilithium`'s
//! Dilithium5 signing and consumed by its verification — the same crate
//! is used by the production `crates/q-crypto-simd/` AVX-512 batched
//! verifier. The in-circuit verifier must accept the SAME byte format
//! so a transaction signed by any FIPS-204 conformant implementation
//! can be re-verified inside the δ-circuit.

use ark_ff::PrimeField;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::Boolean;
use ark_r1cs_std::alloc::AllocVar;
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use sha3::{Shake128, Shake256, digest::{Update, ExtendableOutput, XofReader}};

use crate::gadgets::dilithium::{NttRoots, PublicKeyVar, SignatureVar};

/// τ = number of non-zero coefficients in the SampleInBall output for
/// Dilithium5. Each non-zero coefficient is ±1.
pub const TAU: usize = 60;

/// Dilithium5 packed public-key bytes (2 592 bytes).
///
/// Layout (FIPS-204 §5.2.1):
///   bytes  0..32  ρ — seed for matrix A
///   bytes 32..2592 packed t₁ (8 polynomials × 320 bytes each)
pub const DILITHIUM5_PK_BYTES: usize = 2_592;

/// Dilithium5 packed signature bytes (4 627 bytes).
///
/// Layout (FIPS-204 §5.2.3, ML-DSA-87):
///   bytes    0..64  c̃ — 64-byte challenge seed (2·λ/8 with λ=256)
///   bytes   64..4544 packed z (7 polynomials × 640 bytes each)
///   bytes 4544..4627 packed h (hint indices, ω + k = 75 + 8 = 83 bytes)
pub const DILITHIUM5_SIG_BYTES: usize = 4_627;

/// Dilithium parameters used by the host-side unpacker. Match
/// `gadgets/dilithium.rs::DILITHIUM_Q`.
pub const N: usize = 256;
pub const K: usize = 8;
pub const L: usize = 7;
pub const Q: u64 = 8_380_417;
/// γ₁ = 2^19 = 524 288. Sets the per-coefficient bound on z.
pub const GAMMA1: u32 = 1 << 19;
/// Hint weight bound (max set bits across all h polynomials).
pub const OMEGA: usize = 75;

/// Signature subsection byte lengths (must sum to `DILITHIUM5_SIG_BYTES`).
pub const C_TILDE_BYTES: usize = 64;
/// Bits-per-coefficient for z encoding: γ₁ = 2^19 → encoded value fits in 20 bits.
pub const Z_BITS_PER_COEFF: usize = 20;
/// Bytes per packed z polynomial: ceil(256 × 20 / 8) = 640.
pub const Z_PACKED_POLY_BYTES: usize = 640;
/// Total bytes for h: ω indices + k poly-end-position bytes.
pub const H_PACKED_BYTES: usize = OMEGA + K;

/// Raw FIPS-204 packed public-key bytes wrapped for type safety.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DilithiumKeyBytes(pub [u8; DILITHIUM5_PK_BYTES]);

/// Raw FIPS-204 packed signature bytes wrapped for type safety.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DilithiumSigBytes(pub [u8; DILITHIUM5_SIG_BYTES]);

impl DilithiumKeyBytes {
    /// Construct from a slice. Returns `None` if the length is wrong.
    pub fn from_slice(s: &[u8]) -> Option<Self> {
        if s.len() != DILITHIUM5_PK_BYTES {
            return None;
        }
        let mut bytes = [0u8; DILITHIUM5_PK_BYTES];
        bytes.copy_from_slice(s);
        Some(Self(bytes))
    }

    /// Extract ρ, the 32-byte matrix seed.
    pub fn rho(&self) -> [u8; 32] {
        let mut rho = [0u8; 32];
        rho.copy_from_slice(&self.0[..32]);
        rho
    }

    /// Extract the packed t₁ region (bytes 32..2592).
    pub fn t1_bytes(&self) -> &[u8] {
        &self.0[32..]
    }

    /// Unpack the t₁ region (bytes 32..2592) into 8 polynomials of 256
    /// 10-bit coefficients each, per FIPS-204 §5.2.1 SimpleBitPack.
    ///
    /// `t₁` values live in [0, 2¹⁰) = [0, 1024) — the truncated upper bits
    /// of the public-key polynomial `t = t₁·2^d + t₀` where d=13 for
    /// Dilithium5. The lower 13 bits (`t₀`) are in the secret key, not
    /// the public key.
    ///
    /// Returns `[u32; N]` × K = `[u32; 256] × 8`. Caller allocates each
    /// coefficient as an `FpVar` witness in the circuit.
    pub fn unpack_t1_native(&self) -> [[u32; N]; K] {
        let mut out = [[0u32; N]; K];
        for poly_idx in 0..K {
            let start = 32 + poly_idx * T1_PACKED_POLY_BYTES;
            let end = start + T1_PACKED_POLY_BYTES;
            simple_bit_unpack_generic(&self.0[start..end], 10, &mut out[poly_idx]);
        }
        out
    }

    /// Allocate as in-circuit `PublicKeyVar`.
    ///
    /// Implements all three sub-tasks listed in this file's docstring:
    ///   • t₁ unpacking via `unpack_t1_native` + per-coefficient `FpVar`
    ///     allocation as witnesses.
    ///   • a_mat: ExpandA(ρ) per FIPS-204 §3.2. Produces NTT-DOMAIN A
    ///     matrix (k×l = 56 polynomials of 256 coefficients each).
    ///     **The downstream `DilithiumVerifierGadget::compute_az_minus_ct`
    ///     applies poly_mul that internally NTTs the inputs — so a_mat
    ///     must be in STANDARD polynomial form for that path. This
    ///     function currently emits NTT-domain values and the gadget
    ///     adapter (tracked as `dilithium-witness-a-mat-domain-bridge`)
    ///     will need to either (a) inverse-NTT before passing to
    ///     compute_az_minus_ct, or (b) switch to a pointwise-multiply
    ///     gadget that consumes NTT-form inputs directly. Decision
    ///     deferred to the Dilithium-end-to-end test commit.**
    ///
    /// **Soundness note**: the DilithiumVerifierGadget::verify_structured
    /// implementation in `gadgets/dilithium.rs` consumes `a_mat` directly
    /// as witness coefficients — there's no in-circuit check that
    /// `a_mat == ExpandA(ρ)`. That means the prover providing a CORRUPT
    /// `a_mat` (random polynomials unrelated to ρ) could produce a
    /// "valid" w' but the signature wouldn't actually verify against
    /// the intended public key. This is a known soundness gap for Phase 1
    /// and is closed by either (a) in-circuit ExpandA verification or
    /// (b) committing to `a_mat` via Poseidon and binding the commitment
    /// to ρ via a separate ZK-friendly AIR. Phase 5 mandatory
    /// activation requires this closed.
    pub fn allocate<F: PrimeField>(
        &self,
        cs: ConstraintSystemRef<F>,
    ) -> Result<PublicKeyVar<F>, SynthesisError> {
        // ─── t_vec: k polynomials × n coefficients ────────────────────
        let t1_native = self.unpack_t1_native();
        let mut t_vec: Vec<Vec<FpVar<F>>> = Vec::with_capacity(K);
        for poly in t1_native.iter() {
            let mut fp_poly: Vec<FpVar<F>> = Vec::with_capacity(N);
            for &coeff in poly.iter() {
                // Range constraint: t₁ coefficients fit in 10 bits. The
                // FpVar allocation itself doesn't enforce this — callers
                // who care about strict spec conformance should add
                // `NttVerifierGadget::verify_infinity_norm` with
                // bound = 1024 over t_vec[i]. The signature verifier in
                // verify_structured doesn't currently do that (it relies
                // on the q-range check inside compute_az_minus_ct).
                fp_poly.push(FpVar::new_witness(cs.clone(), || Ok(F::from(coeff)))?);
            }
            t_vec.push(fp_poly);
        }

        // ─── a_mat: ExpandA(ρ) ─────────────────────────────────────────
        // 56 polynomials × 256 coefficients in NTT domain.
        let rho = self.rho();
        let a_native = expand_a_native(&rho);
        let mut a_mat: Vec<Vec<FpVar<F>>> = Vec::with_capacity(K * L);
        for poly in a_native.iter() {
            let mut fp_poly: Vec<FpVar<F>> = Vec::with_capacity(N);
            for &coeff in poly.iter() {
                fp_poly.push(FpVar::new_witness(cs.clone(), || Ok(F::from(coeff)))?);
            }
            a_mat.push(fp_poly);
        }

        Ok(PublicKeyVar { a_mat, t_vec })
    }
}

/// Bytes per packed t₁ polynomial: ceil(256 × 10 / 8) = 320.
const T1_PACKED_POLY_BYTES: usize = 320;

/// Native (off-circuit) implementation of FIPS-204 §3.2 Algorithm 8
/// "ExpandA" (Algorithm 4 RejBoundedPoly per the August 2024 final
/// version). Given a 32-byte seed ρ, produces the public-key matrix
/// A — k × l = 56 polynomials of 256 coefficients each in [0, q).
///
/// **Output domain**: NTT-domain coefficients per FIPS-204. The reference
/// signer/verifier keeps A in NTT form throughout; consumers of this
/// function that expect standard-polynomial form must apply an inverse
/// NTT first. See `DilithiumKeyBytes::allocate` for the gadget-bridge
/// caveat.
///
/// Algorithm per polynomial (rows×cols indexed (i,j)):
///   1. SHAKE-128 seeded with **ρ ∥ IntegerToBytes(i, 1) ∥ IntegerToBytes(j, 1)**
///      — row index first, then column. (Caught in DeepSeek peer review
///      2026-05-15; previous version had columns first, which would have
///      silently bypassed advisory-mode validation but broken the future
///      in-circuit SHAKE-128 binding `A == ExpandA(ρ)`.)
///   2. Rejection-sample 256 coefficients: read 3 bytes, parse as a
///      23-bit value (mask off top bit), accept if < q.
///
/// Returns `Vec<[u32; N]>` of length K * L = 56, row-major ordered.
pub fn expand_a_native(rho: &[u8; 32]) -> Vec<[u32; N]> {
    let mut out: Vec<[u32; N]> = Vec::with_capacity(K * L);
    for i in 0..K {
        for j in 0..L {
            // SHAKE-128(ρ ∥ i ∥ j) per FIPS-204 §3.2 Algorithm 4.
            let mut shake = Shake128::default();
            Update::update(&mut shake, rho);
            Update::update(&mut shake, &[i as u8]);
            Update::update(&mut shake, &[j as u8]);
            let mut reader = shake.finalize_xof();

            let mut poly = [0u32; N];
            let mut filled = 0usize;
            while filled < N {
                // Read 3 bytes at a time.
                let mut buf = [0u8; 3];
                reader.read(&mut buf);
                // Parse as 23-bit unsigned integer (top bit of the third
                // byte is masked off per FIPS-204 §A.1).
                let v: u32 = (buf[0] as u32)
                    | ((buf[1] as u32) << 8)
                    | (((buf[2] & 0x7F) as u32) << 16);
                if (v as u64) < Q {
                    poly[filled] = v;
                    filled += 1;
                }
            }
            out.push(poly);
        }
    }
    out
}

/// Native (off-circuit) implementation of FIPS-204 §4 Algorithm 3
/// "SampleInBall". Given a 64-byte challenge seed c̃, produces the
/// challenge polynomial c ∈ {-1, 0, 1}^256 with exactly τ=60 non-zero
/// coefficients.
///
/// Algorithm:
///   1. Initialize a SHAKE-256 XOF over c̃.
///   2. Read 8 bytes; interpret as 64 sign bits.
///   3. Fisher-Yates loop for i in (n - τ)..n:
///      a. Rejection-sample j ∈ [0, i] by reading bytes one at a time.
///      b. Set c[i] = c[j], c[j] = ±1 driven by the next sign bit.
///
/// The output is byte-identical between this native impl and any
/// FIPS-204 reference implementation given the same c̃.
pub fn sample_in_ball_native(c_tilde: &[u8; C_TILDE_BYTES]) -> [i32; N] {
    let mut shake = Shake256::default();
    Update::update(&mut shake, c_tilde);
    let mut reader = shake.finalize_xof();

    // Read 8 bytes of signs into a u64. The bits are consumed LSB-first
    // as the algorithm progresses.
    let mut sign_bytes = [0u8; 8];
    reader.read(&mut sign_bytes);
    let mut signs = u64::from_le_bytes(sign_bytes);

    let mut c = [0i32; N];
    for i in (N - TAU)..N {
        // Rejection-sample j ∈ [0, i] by reading one byte at a time
        // until a value ≤ i appears. Per FIPS-204 §4 Algorithm 3
        // step 6, bytes > i are simply discarded.
        let j: usize = loop {
            let mut b = [0u8; 1];
            reader.read(&mut b);
            if (b[0] as usize) <= i {
                break b[0] as usize;
            }
        };

        c[i] = c[j];
        // Sign: 0 → +1, 1 → -1. (FIPS-204 §4 step 8 specifies the
        // sign bit ordering: low bit of the running signs value.)
        c[j] = if signs & 1 == 0 { 1 } else { -1 };
        signs >>= 1;
    }
    c
}

/// Inverse of `simple_bit_unpack_generic` — packs `n = values.len()`
/// d-bit unsigned integers into a little-endian bit-stream.
///
/// Test helper (not used by the production verifier path; the signer
/// produces the packed bytes externally). Lives here so the unpacker
/// tests have a way to construct synthetic packed inputs.
fn simple_bit_pack_generic(values: &[u32], d: usize, out: &mut [u8]) {
    out.fill(0);
    for (k, &v) in values.iter().enumerate() {
        let bit_pos = k * d;
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let span_bytes = (bit_offset + d + 7) / 8;
        let shifted: u64 = (v as u64) << bit_offset;
        for i in 0..span_bytes {
            let byte_shift = i * 8;
            let byte_chunk = ((shifted >> byte_shift) & 0xFF) as u8;
            out[byte_idx + i] |= byte_chunk;
        }
    }
}

impl DilithiumSigBytes {
    pub fn from_slice(s: &[u8]) -> Option<Self> {
        if s.len() != DILITHIUM5_SIG_BYTES {
            return None;
        }
        let mut bytes = [0u8; DILITHIUM5_SIG_BYTES];
        bytes.copy_from_slice(s);
        Some(Self(bytes))
    }

    /// Extract c̃, the 64-byte challenge seed (ML-DSA-87).
    pub fn c_tilde(&self) -> [u8; C_TILDE_BYTES] {
        let mut c = [0u8; C_TILDE_BYTES];
        c.copy_from_slice(&self.0[..C_TILDE_BYTES]);
        c
    }

    /// Unpack the z region into 7 polynomials × 256 coefficients each,
    /// per FIPS-204 §5.2.3 BitPack(z, γ₁-1, γ₁) with d=20.
    ///
    /// Encoding (signer side): each signed coefficient z[i] ∈ (-γ₁, γ₁]
    /// is stored as `enc = γ₁ - z[i]` ∈ [0, 2γ₁), packed as 20-bit
    /// little-endian within the 640-byte polynomial slab.
    ///
    /// Decoding (this function): `z[i] = γ₁ - enc`. Negative results are
    /// converted to their Z_q representation `q - |z[i]|` so the
    /// downstream FpVar arithmetic matches the gadget's expectation
    /// (positives stored as-is; negatives stored as q-complement).
    ///
    /// Returns u32 because Z_q values < 2^23 < 2^32. Caller allocates
    /// each as an `FpVar` witness.
    pub fn unpack_z_native(&self) -> [[u32; N]; L] {
        let mut out = [[0u32; N]; L];
        for poly_idx in 0..L {
            let start = C_TILDE_BYTES + poly_idx * Z_PACKED_POLY_BYTES;
            let end = start + Z_PACKED_POLY_BYTES;
            let mut encoded = [0u32; N];
            simple_bit_unpack_generic(&self.0[start..end], Z_BITS_PER_COEFF, &mut encoded);
            for (i, &enc) in encoded.iter().enumerate() {
                // signed_z lives in (-γ₁, γ₁]; |signed_z| < γ₁ < q so the
                // i64 intermediate doesn't overflow.
                let signed_z: i64 = GAMMA1 as i64 - enc as i64;
                let z_in_zq: u32 = if signed_z < 0 {
                    (Q as i64 + signed_z) as u32
                } else {
                    signed_z as u32
                };
                out[poly_idx][i] = z_in_zq;
            }
        }
        out
    }

    /// Unpack the h region into 8 polynomials × 256 hint-bits, per
    /// FIPS-204 §5.2.3 HintBitUnpack.
    ///
    /// h is packed as 83 bytes total:
    ///   bytes 0..ω  (=75): coefficient indices in [0, 256) where hint bits
    ///                       are set, packed contiguously per polynomial.
    ///   bytes ω..ω+k (=83): cumulative end-positions — byte i tells the
    ///                       reader the END index (exclusive) of polynomial
    ///                       i's hint-index list within bytes 0..ω.
    ///
    /// Returns `[[bool; N]; K]`. Caller allocates each as `Boolean<F>`.
    ///
    /// **Malformed-witness handling**: per FIPS-204 §4 algorithm 7
    /// Verify step 4, a hint vector with > ω total set bits OR with
    /// non-monotonic length bytes OR with out-of-range coefficient
    /// indices MUST cause signature rejection. This function returns
    /// `None` on any such violation; the gadget allocates a Boolean
    /// `valid_hint` from `result.is_some()` and ANDs it into the
    /// overall signature-validity result.
    pub fn unpack_h_native(&self) -> Option<[[bool; N]; K]> {
        let hint_start = C_TILDE_BYTES + L * Z_PACKED_POLY_BYTES;
        if hint_start + H_PACKED_BYTES > DILITHIUM5_SIG_BYTES {
            return None;
        }
        let hint_bytes = &self.0[hint_start..hint_start + H_PACKED_BYTES];
        let length_bytes = &hint_bytes[OMEGA..OMEGA + K];

        let mut out = [[false; N]; K];
        let mut cursor: usize = 0;
        for poly_idx in 0..K {
            let end_pos = length_bytes[poly_idx] as usize;
            // Length bytes must be monotone non-decreasing AND ≤ ω.
            if end_pos < cursor || end_pos > OMEGA {
                return None;
            }
            for idx_byte_pos in cursor..end_pos {
                let coef_idx = hint_bytes[idx_byte_pos] as usize;
                if coef_idx >= N {
                    return None;
                }
                // Indices within one poly must be strictly increasing.
                // FIPS-204 §4.3 Algorithm 7 (UnpackHint) requires that
                // "the indices are sorted in increasing order"; since
                // the polynomial coefficients are binary, sorted +
                // no-duplicate is equivalent to strictly-increasing.
                // Confirmed in DeepSeek peer review 2026-05-15 as
                // implied by spec, not an extra invariant.
                if idx_byte_pos > cursor {
                    let prev = hint_bytes[idx_byte_pos - 1] as usize;
                    if coef_idx <= prev {
                        return None;
                    }
                }
                out[poly_idx][coef_idx] = true;
            }
            cursor = end_pos;
        }
        // All bytes past `cursor` up to OMEGA must be zero-padded.
        for &b in &hint_bytes[cursor..OMEGA] {
            if b != 0 {
                return None;
            }
        }
        Some(out)
    }

    /// Allocate as in-circuit `SignatureVar`.
    ///
    /// Implements two of the three sub-tasks listed in this file's docstring:
    ///   • z unpacking via `unpack_z_native` + per-coefficient `FpVar`
    ///     witness allocation.
    ///   • h unpacking via `unpack_h_native` + per-bit `Boolean<F>` witness
    ///     allocation. Returns `AssignmentMissing` on malformed hint
    ///     packing (the gadget caller wraps the entire signature
    ///     verification in an outer Boolean that captures this).
    ///   • c_poly: PARTIAL. Returns an n-zero polynomial.
    ///     The proper fill-in needs `SampleInBall(c̃)` (FIPS-204 §4),
    ///     which is rejection-sampling driven by SHAKE-256. SHAKE-256
    ///     is available via the workspace `sha3` crate, so this is a
    ///     finite follow-up (~80 LOC). Tracked as
    ///     `dilithium-witness-sample-in-ball`.
    ///
    /// **Soundness note**: until `SampleInBall` is wired,
    /// the prover supplies `c_poly` as a witness without
    /// constraint binding to `c̃`. A malicious prover could supply a
    /// c_poly that satisfies the verifier's algebra but doesn't actually
    /// commit to the signed message's challenge. Phase 5 mandatory
    /// activation requires this closed.
    pub fn allocate<F: PrimeField>(
        &self,
        cs: ConstraintSystemRef<F>,
    ) -> Result<SignatureVar<F>, SynthesisError> {
        // ─── z: l polynomials × n coefficients ─────────────────────────
        let z_native = self.unpack_z_native();
        let mut z: Vec<Vec<FpVar<F>>> = Vec::with_capacity(L);
        for poly in z_native.iter() {
            let mut fp_poly: Vec<FpVar<F>> = Vec::with_capacity(N);
            for &coeff in poly.iter() {
                fp_poly.push(FpVar::new_witness(cs.clone(), || Ok(F::from(coeff)))?);
            }
            z.push(fp_poly);
        }

        // ─── h: k polynomials × n Booleans ─────────────────────────────
        let h_native = self.unpack_h_native()
            .ok_or(SynthesisError::AssignmentMissing)?;
        let mut h: Vec<Vec<Boolean<F>>> = Vec::with_capacity(K);
        for poly in h_native.iter() {
            let mut bool_poly: Vec<Boolean<F>> = Vec::with_capacity(N);
            for &bit in poly.iter() {
                bool_poly.push(Boolean::new_witness(cs.clone(), || Ok(bit))?);
            }
            h.push(bool_poly);
        }

        // ─── c_poly: SampleInBall(c̃) ───────────────────────────────────
        //
        // FIPS-204 §4 Algorithm 3. Deterministic — c̃ uniquely
        // determines c_poly. We compute it natively here and allocate
        // each coefficient as a witness; the in-circuit relationship
        // "c_poly = SampleInBall(c̃)" is enforced by a separate
        // sub-circuit (in-circuit SHAKE-256 + Fisher-Yates), tracked
        // as `dilithium-witness-sample-in-ball-incircuit`. Until that
        // lands, the recursive-proof level trusts the prover supplied
        // the correct c_poly for the c̃ — block-by-block validation in
        // the API server independently checks the signature.
        let c_native = sample_in_ball_native(&self.c_tilde());
        let c_poly: Vec<FpVar<F>> = c_native
            .iter()
            .map(|&signed| {
                // -1 → q - 1, 0 → 0, 1 → 1 in the Z_q hosting field.
                let value: u64 = if signed < 0 {
                    Q - (-(signed as i64)) as u64
                } else {
                    signed as u64
                };
                FpVar::new_witness(cs.clone(), || Ok(F::from(value)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // ─── c_tilde: allocate the 64-byte challenge seed as field
        // elements. The gadget's SignatureVar has `c_tilde: Vec<FpVar<F>>`
        // — pack as 16 little-endian u32 words (64 bytes = 16 words).
        let c_tilde_bytes = self.c_tilde();
        let c_tilde: Vec<FpVar<F>> = c_tilde_bytes
            .chunks(4)
            .map(|c| {
                let w = u32::from_le_bytes(c.try_into().expect("4 bytes per word"));
                FpVar::new_witness(cs.clone(), || Ok(F::from(w)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(SignatureVar { z, h, c_tilde, c_poly })
    }
}

/// Generic SimpleBitPack: read `bytes` and write `n = out.len()` values
/// of `d` bits each into `out`. Each value is little-endian bit-packed.
fn simple_bit_unpack_generic(bytes: &[u8], d: usize, out: &mut [u32]) {
    let mask: u64 = (1u64 << d) - 1;
    for k in 0..out.len() {
        let bit_pos = k * d;
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        // Pull enough bytes to cover (bit_offset + d) bits.
        let span_bytes = (bit_offset + d + 7) / 8;
        let mut combined: u64 = 0;
        for i in 0..span_bytes {
            let b = *bytes.get(byte_idx + i).unwrap_or(&0) as u64;
            combined |= b << (i * 8);
        }
        out[k] = ((combined >> bit_offset) & mask) as u32;
    }
}

/// Compute μ = SHAKE-256(SHAKE-256(pk, 64) ∥ M, 64) for the signed
/// message. Native (off-circuit) version.
///
/// FIPS-204 §5.1 Sign step 6:
///   tr = SHAKE-256(pk, 64)
///   μ  = SHAKE-256(tr ∥ M, 64)
///
/// Returns 64 bytes. The signer used exactly this μ as the signed
/// digest fed into SampleInBall and the challenge-binding equation,
/// so the in-circuit verifier must derive the SAME μ from the same
/// pk and M.
pub fn message_hash_native(pk_bytes: &[u8], message: &[u8]) -> [u8; 64] {
    // Step 1: tr = SHAKE-256(pk, 64)
    let mut tr = [0u8; 64];
    let mut shake_tr = Shake256::default();
    Update::update(&mut shake_tr, pk_bytes);
    shake_tr.finalize_xof().read(&mut tr);

    // Step 2: μ = SHAKE-256(tr ∥ M, 64)
    let mut mu = [0u8; 64];
    let mut shake_mu = Shake256::default();
    Update::update(&mut shake_mu, &tr);
    Update::update(&mut shake_mu, message);
    shake_mu.finalize_xof().read(&mut mu);

    mu
}

/// In-circuit allocation of μ as 16 little-endian u32 FpVar words.
/// Computes μ NATIVELY then allocates each word as a witness. The
/// constraint that `μ = SHAKE-256(SHAKE-256(pk)∥M)` is enforced by an
/// in-circuit SHAKE-256 sub-circuit (separate AIR, tracked as
/// `dilithium-witness-message-hash-incircuit`).
///
/// During the advisory window the prover supplies μ as a witness and
/// the verifier's algebra uses it directly. Block-by-block validation
/// in the API server independently re-hashes pk and M to verify
/// against the signature.
pub fn message_hash<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    pk_bytes: &[u8],
    message: &[u8],
) -> Result<Vec<FpVar<F>>, SynthesisError> {
    let mu_bytes = message_hash_native(pk_bytes, message);
    mu_bytes
        .chunks(4)
        .map(|c| {
            let w = u32::from_le_bytes(c.try_into().expect("4 bytes per word"));
            FpVar::new_witness(cs.clone(), || Ok(F::from(w)))
        })
        .collect()
}

/// Construct the standard FIPS-204 NTT roots for the Dilithium5
/// parameter set (n=256, q=8 380 417).
///
/// The primitive 512-th root of unity in Z_q is ψ = 1753 per FIPS-204
/// §A.4. From ψ we derive:
///
///   ω = ψ²              (primitive 256-th root of unity)
///   fwd[k] = ω^(bit_rev₈(k))   for k = 0..256, with 8-bit reversal
///   inv[k] = ω^(−bit_rev₈(k))  = fwd[k]⁻¹ mod q
///   n_inv  = 256⁻¹ mod q
///
/// All arithmetic happens natively in u64 (q fits comfortably; ψ^k for
/// k < 512 stays well under u64::MAX with proper modular reduction).
/// The resulting `F::from(...)` allocations are constants in any
/// `PrimeField<F>` whose modulus is larger than q (BN254, BLS12-381,
/// pasta, etc.) — i.e., any field the recursive proof might use.
///
/// The output table is ~4 KB; cache it once at startup.
pub fn standard_ntt_roots<F: PrimeField>() -> NttRoots<F> {
    const PSI: u64 = 1753;

    // ω = ψ² mod q
    let omega = mul_mod(PSI, PSI, Q);

    // ω^bit_rev_8(k) for k in 0..N
    let mut fwd_native: Vec<u64> = Vec::with_capacity(N);
    for k in 0..N {
        let exp = bit_reverse_8(k as u8) as u64;
        fwd_native.push(pow_mod(omega, exp, Q));
    }

    // inv[k] = fwd[k]⁻¹ mod q  via Fermat's little theorem (q is prime).
    // a⁻¹ ≡ a^(q−2) mod q.
    let mut inv_native: Vec<u64> = Vec::with_capacity(N);
    for &f in &fwd_native {
        inv_native.push(pow_mod(f, Q - 2, Q));
    }

    // n_inv = 256⁻¹ mod q
    let n_inv_native = pow_mod(N as u64, Q - 2, Q);

    // ψ⁻¹ via Fermat's little theorem.
    let psi_inv = pow_mod(PSI, Q - 2, Q);

    NttRoots {
        fwd: fwd_native.into_iter().map(F::from).collect(),
        inv: inv_native.into_iter().map(F::from).collect(),
        n_inv: F::from(n_inv_native),
        psi: F::from(PSI),
        psi_inv: F::from(psi_inv),
    }
}

/// Bit-reverse the bottom 8 bits of `x`. For the n=256 NTT, indices
/// are reversed within 8 bits (i.e., reflect across a 4-bit midpoint).
fn bit_reverse_8(x: u8) -> u8 {
    let mut r = 0u8;
    for i in 0..8 {
        r |= ((x >> i) & 1) << (7 - i);
    }
    r
}

/// (a × b) mod m using u128 intermediate to avoid u64 overflow.
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    (((a as u128) * (b as u128)) % (m as u128)) as u64
}

/// (base ^ exp) mod m using square-and-multiply.
fn pow_mod(base: u64, exp: u64, m: u64) -> u64 {
    let mut result: u64 = 1;
    let mut base = base % m;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = mul_mod(result, base, m);
        }
        e >>= 1;
        base = mul_mod(base, base, m);
    }
    result
}

/// One-shot convenience that bundles pk-unpack + sig-unpack + message-hash
/// for a single transaction's signature verification call. The output
/// can be fed directly into
/// `DilithiumVerifierGadget::verify_structured`.
///
/// **STUB.** Wires the four sub-stubs above. When they're filled in,
/// this function is trivial.
pub fn allocate_dilithium_witness_for_tx<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    pubkey_bytes: &[u8],
    signature_bytes: &[u8],
    signing_message: &[u8],
) -> Result<DilithiumTxWitness<F>, SynthesisError> {
    let pk_pack = DilithiumKeyBytes::from_slice(pubkey_bytes)
        .ok_or(SynthesisError::AssignmentMissing)?;
    let sig_pack = DilithiumSigBytes::from_slice(signature_bytes)
        .ok_or(SynthesisError::AssignmentMissing)?;

    let pk = pk_pack.allocate::<F>(cs.clone())?;
    let sig = sig_pack.allocate::<F>(cs.clone())?;
    let msg = message_hash::<F>(cs.clone(), &pk_pack.0, signing_message)?;
    let roots = standard_ntt_roots::<F>();

    Ok(DilithiumTxWitness { pk, sig, msg, roots })
}

/// Bundle of allocated in-circuit witnesses for one transaction's
/// signature verification.
pub struct DilithiumTxWitness<F: PrimeField> {
    pub pk: PublicKeyVar<F>,
    pub sig: SignatureVar<F>,
    pub msg: Vec<FpVar<F>>,
    pub roots: NttRoots<F>,
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_bytes_length_constant_is_fips_204_correct() {
        assert_eq!(DILITHIUM5_PK_BYTES, 2_592);
        assert_eq!(DILITHIUM5_SIG_BYTES, 4_627);
    }

    #[test]
    fn key_bytes_from_slice_rejects_wrong_length() {
        assert!(DilithiumKeyBytes::from_slice(&[0u8; 100]).is_none());
        assert!(DilithiumKeyBytes::from_slice(&[0u8; DILITHIUM5_PK_BYTES]).is_some());
    }

    #[test]
    fn sig_bytes_from_slice_rejects_wrong_length() {
        assert!(DilithiumSigBytes::from_slice(&[0u8; 100]).is_none());
        assert!(DilithiumSigBytes::from_slice(&[0u8; DILITHIUM5_SIG_BYTES]).is_some());
    }

    #[test]
    fn key_bytes_rho_extracts_first_32() {
        let mut bytes = [0u8; DILITHIUM5_PK_BYTES];
        for i in 0..32 {
            bytes[i] = i as u8;
        }
        let pk = DilithiumKeyBytes(bytes);
        let rho = pk.rho();
        for i in 0..32 {
            assert_eq!(rho[i], i as u8);
        }
    }

    #[test]
    fn parameter_constants_match_fips_204() {
        // Spot-check that our local copies of the Dilithium5 parameters
        // match the values the gadgets crate uses.
        use crate::gadgets::dilithium::DILITHIUM_Q;
        assert_eq!(Q, DILITHIUM_Q);
        assert_eq!(N, 256);
        assert_eq!(K, 8);
        assert_eq!(L, 7);
    }

    #[test]
    fn bit_reverse_8_is_self_inverse() {
        // Bit-reversal twice = identity.
        for x in 0..=255u8 {
            assert_eq!(bit_reverse_8(bit_reverse_8(x)), x);
        }
    }

    #[test]
    fn bit_reverse_8_specific_values() {
        // Known values: bit_rev(0b00000001) = 0b10000000 = 128.
        assert_eq!(bit_reverse_8(1), 128);
        assert_eq!(bit_reverse_8(128), 1);
        // bit_rev(0b11110000) = 0b00001111 = 15.
        assert_eq!(bit_reverse_8(0b11110000), 0b00001111);
    }

    #[test]
    fn pow_mod_satisfies_fermats_little_theorem() {
        // For any a coprime to q (and q prime), a^(q-1) ≡ 1 mod q.
        // Test on a few small bases.
        for base in [2u64, 3, 5, 7, 1753] {
            let result = pow_mod(base, Q - 1, Q);
            assert_eq!(result, 1, "Fermat's little theorem fails for base {}", base);
        }
    }

    #[test]
    fn pow_mod_inverse_property() {
        // a × a^(q-2) ≡ 1 mod q for prime q.
        for a in [2u64, 7, 1753, 8_380_416] {
            let a_inv = pow_mod(a, Q - 2, Q);
            assert_eq!(mul_mod(a, a_inv, Q), 1);
        }
    }

    #[test]
    fn psi_is_primitive_512th_root_of_unity() {
        // ψ = 1753, q = 8_380_417. ψ should satisfy ψ^512 ≡ 1 mod q
        // and ψ^256 ≡ −1 mod q (i.e., q−1) but ψ^k ≠ 1 for any k < 512.
        const PSI: u64 = 1753;

        // ψ^512 = 1
        assert_eq!(pow_mod(PSI, 512, Q), 1, "ψ^512 must be 1");

        // ψ^256 = -1 = q - 1
        assert_eq!(pow_mod(PSI, 256, Q), Q - 1, "ψ^256 must be q-1 (=-1 mod q)");

        // ψ^d ≠ 1 for d ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}
        // (i.e., 256 is the multiplicative order of ψ² = ω, so ψ has order 512)
        for d in [1u64, 2, 4, 8, 16, 32, 64, 128, 256] {
            let v = pow_mod(PSI, d, Q);
            assert_ne!(v, 1, "ψ has order < {}", d);
        }
    }

    #[test]
    fn omega_is_primitive_256th_root_of_unity() {
        // ω = ψ² has order n = 256.
        const PSI: u64 = 1753;
        let omega = mul_mod(PSI, PSI, Q);

        assert_eq!(pow_mod(omega, 256, Q), 1, "ω^256 must be 1");
        // Should not be 1 for any proper divisor of 256.
        for d in [1u64, 2, 4, 8, 16, 32, 64, 128] {
            assert_ne!(pow_mod(omega, d, Q), 1, "ω^{} must not be 1", d);
        }
    }

    #[test]
    fn standard_ntt_roots_table_has_correct_shape() {
        use ark_bls12_381::Fr;
        let roots = standard_ntt_roots::<Fr>();
        assert_eq!(roots.fwd.len(), N, "fwd must have N=256 entries");
        assert_eq!(roots.inv.len(), N, "inv must have N=256 entries");
        // fwd and inv must be elementwise inverses:
        //   fwd[k] × inv[k] ≡ 1 mod q  for all k
        for k in 0..N {
            let prod = roots.fwd[k] * roots.inv[k];
            assert_eq!(prod, Fr::from(1u64), "fwd[{}] × inv[{}] != 1", k, k);
        }
        // n_inv × n ≡ 1
        let n_check = roots.n_inv * Fr::from(N as u64);
        assert_eq!(n_check, Fr::from(1u64));
        // psi × psi_inv ≡ 1
        let psi_check = roots.psi * roots.psi_inv;
        assert_eq!(psi_check, Fr::from(1u64));
    }

    #[test]
    fn standard_ntt_roots_first_entry_is_one() {
        // fwd[0] = ω^(bit_rev_8(0)) = ω^0 = 1.
        use ark_bls12_381::Fr;
        let roots = standard_ntt_roots::<Fr>();
        assert_eq!(roots.fwd[0], Fr::from(1u64));
        assert_eq!(roots.inv[0], Fr::from(1u64));
    }

    // ─── Generic bit-pack / unpack round-trip ─────────────────────────

    #[test]
    fn simple_bit_pack_unpack_round_trip_d10() {
        // 256 random values in [0, 1024) should round-trip through
        // pack + unpack identically.
        let mut values = [0u32; 256];
        for i in 0..256 {
            values[i] = ((i * 31 + 7) % 1024) as u32;
        }
        let mut packed = [0u8; 320];
        simple_bit_pack_generic(&values, 10, &mut packed);
        let mut unpacked = [0u32; 256];
        simple_bit_unpack_generic(&packed, 10, &mut unpacked);
        for i in 0..256 {
            assert_eq!(unpacked[i], values[i], "d=10 round-trip failed at index {}", i);
        }
    }

    #[test]
    fn simple_bit_pack_unpack_round_trip_d20() {
        // 256 random values in [0, 2^20) should round-trip.
        let mut values = [0u32; 256];
        for i in 0..256 {
            // Use a mix of small + large values to exercise high bits.
            values[i] = ((i as u32 * 1009 + 41).wrapping_mul(31337)) & 0xFFFFF;
        }
        let mut packed = [0u8; 640];
        simple_bit_pack_generic(&values, 20, &mut packed);
        let mut unpacked = [0u32; 256];
        simple_bit_unpack_generic(&packed, 20, &mut unpacked);
        for i in 0..256 {
            assert_eq!(unpacked[i], values[i], "d=20 round-trip failed at index {}", i);
        }
    }

    // ─── PK t₁ unpack tests ───────────────────────────────────────────

    #[test]
    fn pk_t1_unpack_returns_all_zeros_for_zero_input() {
        let pk = DilithiumKeyBytes([0u8; DILITHIUM5_PK_BYTES]);
        let t1 = pk.unpack_t1_native();
        for poly in &t1 {
            for &coeff in poly.iter() {
                assert_eq!(coeff, 0);
            }
        }
    }

    #[test]
    fn pk_t1_unpack_round_trip_with_known_coeffs() {
        // Construct a packed PK where each poly i has coefficient j = (i*256 + j) % 1024.
        let mut pk_bytes = [0u8; DILITHIUM5_PK_BYTES];
        for poly_idx in 0..K {
            let mut values = [0u32; N];
            for j in 0..N {
                values[j] = ((poly_idx * N + j) % 1024) as u32;
            }
            let start = 32 + poly_idx * T1_PACKED_POLY_BYTES;
            let end = start + T1_PACKED_POLY_BYTES;
            simple_bit_pack_generic(&values, 10, &mut pk_bytes[start..end]);
        }
        let pk = DilithiumKeyBytes(pk_bytes);
        let t1 = pk.unpack_t1_native();
        for poly_idx in 0..K {
            for j in 0..N {
                let expected = ((poly_idx * N + j) % 1024) as u32;
                assert_eq!(t1[poly_idx][j], expected,
                    "t₁[{}][{}] mismatch", poly_idx, j);
            }
        }
    }

    // ─── SIG z unpack tests ───────────────────────────────────────────

    #[test]
    fn sig_z_unpack_all_zero_packed_means_z_equals_gamma1() {
        // Decoding rule: z[i] = γ₁ - enc[i]. enc = 0 → z = γ₁, which
        // is the maximum positive z value (one above the spec bound,
        // but the unpacker just decodes; range-checking happens
        // separately in the gadget).
        let sig = DilithiumSigBytes([0u8; DILITHIUM5_SIG_BYTES]);
        let z = sig.unpack_z_native();
        for poly in &z {
            for &coeff in poly.iter() {
                // z = γ₁. Stored in Z_q as γ₁ (positive, < q).
                assert_eq!(coeff, GAMMA1);
            }
        }
    }

    #[test]
    fn sig_z_unpack_round_trip_with_known_coeffs() {
        let mut sig_bytes = [0u8; DILITHIUM5_SIG_BYTES];
        let mut expected_z = [[0u32; N]; L];
        for poly_idx in 0..L {
            let mut encoded = [0u32; N];
            for j in 0..N {
                // Choose encoded values that map to a known z.
                // We pick z values in (-γ₁, γ₁]; pick enc = (poly_idx * 17 + j) mod 2γ₁.
                let enc = ((poly_idx as u64 * 17 + j as u64) % (2 * GAMMA1 as u64)) as u32;
                encoded[j] = enc;
                let signed_z = GAMMA1 as i64 - enc as i64;
                expected_z[poly_idx][j] = if signed_z < 0 {
                    (Q as i64 + signed_z) as u32
                } else {
                    signed_z as u32
                };
            }
            let start = C_TILDE_BYTES + poly_idx * Z_PACKED_POLY_BYTES;
            let end = start + Z_PACKED_POLY_BYTES;
            simple_bit_pack_generic(&encoded, Z_BITS_PER_COEFF, &mut sig_bytes[start..end]);
        }
        let sig = DilithiumSigBytes(sig_bytes);
        let z = sig.unpack_z_native();
        for poly_idx in 0..L {
            for j in 0..N {
                assert_eq!(z[poly_idx][j], expected_z[poly_idx][j],
                    "z[{}][{}] mismatch", poly_idx, j);
            }
        }
    }

    // ─── SIG h unpack tests ───────────────────────────────────────────

    #[test]
    fn sig_h_unpack_all_zero_means_no_hints() {
        // All-zero hint region: all 8 length bytes are 0 (no indices per poly),
        // and the index region is all zeros (unused). Should parse as
        // 8 all-false hint vectors.
        let sig = DilithiumSigBytes([0u8; DILITHIUM5_SIG_BYTES]);
        let h = sig.unpack_h_native().expect("all-zero hint must parse");
        for poly in &h {
            for &bit in poly.iter() {
                assert_eq!(bit, false);
            }
        }
    }

    #[test]
    fn sig_h_unpack_rejects_non_monotone_length_bytes() {
        // Build a sig where length bytes are [5, 3, ...] — second poly
        // claims FEWER cumulative indices than the first. Must be
        // rejected as malformed.
        let mut sig_bytes = [0u8; DILITHIUM5_SIG_BYTES];
        let hint_start = C_TILDE_BYTES + L * Z_PACKED_POLY_BYTES;
        // Indices 0..5 are valid (e.g., 0,1,2,3,4).
        for i in 0..5 {
            sig_bytes[hint_start + i] = i as u8;
        }
        // Length bytes: first poly ends at index 5, second poly ends at
        // index 3 (regression — not allowed).
        sig_bytes[hint_start + OMEGA] = 5;
        sig_bytes[hint_start + OMEGA + 1] = 3;
        let sig = DilithiumSigBytes(sig_bytes);
        assert!(sig.unpack_h_native().is_none(),
            "non-monotone length bytes MUST fail to parse");
    }

    #[test]
    fn sig_h_unpack_rejects_out_of_range_coefficient_index() {
        // Coefficient index ≥ N (256). The actual byte value is u8 so
        // max storable is 255 — which is in range. We can't construct
        // an out-of-range index with a single u8. But we CAN test that
        // the OMEGA limit is enforced: try to claim 76 indices.
        let mut sig_bytes = [0u8; DILITHIUM5_SIG_BYTES];
        let hint_start = C_TILDE_BYTES + L * Z_PACKED_POLY_BYTES;
        sig_bytes[hint_start + OMEGA] = (OMEGA + 1) as u8; // 76 > ω=75
        let sig = DilithiumSigBytes(sig_bytes);
        assert!(sig.unpack_h_native().is_none(),
            "end_pos > ω MUST fail to parse");
    }

    #[test]
    fn sig_h_unpack_rejects_non_increasing_indices_within_poly() {
        // Indices within ONE poly must be strictly increasing.
        // [3, 5, 4] is non-monotone — must reject.
        let mut sig_bytes = [0u8; DILITHIUM5_SIG_BYTES];
        let hint_start = C_TILDE_BYTES + L * Z_PACKED_POLY_BYTES;
        sig_bytes[hint_start] = 3;
        sig_bytes[hint_start + 1] = 5;
        sig_bytes[hint_start + 2] = 4;
        // Poly 0 has 3 indices.
        sig_bytes[hint_start + OMEGA] = 3;
        let sig = DilithiumSigBytes(sig_bytes);
        assert!(sig.unpack_h_native().is_none(),
            "non-increasing indices within poly MUST fail to parse");
    }

    // ─── SampleInBall tests ───────────────────────────────────────────

    #[test]
    fn sample_in_ball_produces_exactly_tau_non_zeros() {
        let c_tilde = [0x42u8; C_TILDE_BYTES];
        let c = sample_in_ball_native(&c_tilde);
        let non_zero_count = c.iter().filter(|&&x| x != 0).count();
        assert_eq!(non_zero_count, TAU,
            "Sample-in-ball must produce exactly τ=60 non-zero coefficients");
    }

    #[test]
    fn sample_in_ball_only_emits_plus_minus_one() {
        let c_tilde = [0xAAu8; C_TILDE_BYTES];
        let c = sample_in_ball_native(&c_tilde);
        for (i, &v) in c.iter().enumerate() {
            assert!(
                v == 0 || v == 1 || v == -1,
                "c[{}] = {} not in {{-1, 0, 1}}",
                i, v
            );
        }
    }

    #[test]
    fn sample_in_ball_is_deterministic() {
        // Same c̃ → same c_poly. SampleInBall is a deterministic
        // function of c̃; this is critical for soundness — both
        // signer and verifier must derive the same c.
        let c_tilde = [0x11u8; C_TILDE_BYTES];
        let c1 = sample_in_ball_native(&c_tilde);
        let c2 = sample_in_ball_native(&c_tilde);
        assert_eq!(c1, c2);
    }

    // ─── ExpandA tests ────────────────────────────────────────────────

    #[test]
    fn expand_a_produces_k_times_l_polynomials() {
        let rho = [0x42u8; 32];
        let a = expand_a_native(&rho);
        assert_eq!(a.len(), K * L, "ExpandA must produce K×L=56 polynomials");
        for poly in &a {
            assert_eq!(poly.len(), N);
        }
    }

    #[test]
    fn expand_a_coefficients_are_in_zq() {
        // Every coefficient must satisfy 0 ≤ v < q. The rejection
        // sampling step in ExpandA enforces this; we double-check.
        let rho = [0xAAu8; 32];
        let a = expand_a_native(&rho);
        for (idx, poly) in a.iter().enumerate() {
            for (j, &v) in poly.iter().enumerate() {
                assert!(v < Q as u32, "A[{}][{}] = {} ≥ q = {}", idx, j, v, Q);
            }
        }
    }

    #[test]
    fn expand_a_is_deterministic() {
        let rho = [0x11u8; 32];
        let a1 = expand_a_native(&rho);
        let a2 = expand_a_native(&rho);
        assert_eq!(a1, a2);
    }

    #[test]
    fn expand_a_different_seeds_produce_different_matrices() {
        let mut rho_a = [0u8; 32];
        let mut rho_b = [0u8; 32];
        rho_b[0] = 1;
        let a = expand_a_native(&rho_a);
        let b = expand_a_native(&rho_b);
        assert_ne!(a, b, "Different ρ MUST produce different A");
        // Address compiler complaint about unused mut.
        let _ = &mut rho_a;
    }

    #[test]
    fn expand_a_coefficients_appear_uniformly_distributed() {
        // Light statistical check: across 56 polys × 256 coeffs = 14336
        // samples, count how many fall in each of 4 equal q-quartile
        // buckets. Each bucket should hold ~3584 ± noise. Reject if
        // any bucket is < 60% or > 140% of expected (very lax bounds
        // — this is a sanity test, not a real chi-squared).
        let rho = [0x55u8; 32];
        let a = expand_a_native(&rho);
        let bucket_size = Q as u32 / 4;
        let mut buckets = [0usize; 4];
        for poly in &a {
            for &v in poly.iter() {
                let b = (v / bucket_size).min(3) as usize;
                buckets[b] += 1;
            }
        }
        let expected = (K * L * N) / 4;
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                count > expected * 6 / 10 && count < expected * 14 / 10,
                "bucket {} count {} far from expected {} (60–140% bounds)",
                i, count, expected
            );
        }
    }

    #[test]
    fn message_hash_native_is_deterministic() {
        let pk = vec![0x42u8; DILITHIUM5_PK_BYTES];
        let m = b"some transaction bytes";
        let mu1 = message_hash_native(&pk, m);
        let mu2 = message_hash_native(&pk, m);
        assert_eq!(mu1, mu2);
        assert_eq!(mu1.len(), 64);
    }

    #[test]
    fn message_hash_native_depends_on_pk() {
        let mut pk_a = vec![0u8; DILITHIUM5_PK_BYTES];
        let mut pk_b = vec![0u8; DILITHIUM5_PK_BYTES];
        pk_b[0] = 1;
        let m = b"same message";
        let mu_a = message_hash_native(&pk_a, m);
        let mu_b = message_hash_native(&pk_b, m);
        assert_ne!(mu_a, mu_b, "Different pk MUST produce different μ");
        // Address compiler complaint about unused mut on pk_a.
        let _ = &mut pk_a;
    }

    #[test]
    fn message_hash_native_depends_on_message() {
        let pk = vec![0u8; DILITHIUM5_PK_BYTES];
        let mu_a = message_hash_native(&pk, b"message A");
        let mu_b = message_hash_native(&pk, b"message B");
        assert_ne!(mu_a, mu_b, "Different message MUST produce different μ");
    }

    #[test]
    fn message_hash_native_handles_empty_message() {
        // SHAKE-256 over (tr ∥ <empty>) is well-defined; should not panic.
        let pk = vec![0u8; DILITHIUM5_PK_BYTES];
        let mu = message_hash_native(&pk, &[]);
        // Output is non-zero (extremely unlikely for SHAKE on tr ∥ ∅).
        assert!(mu.iter().any(|&b| b != 0));
    }

    #[test]
    fn sample_in_ball_different_seeds_produce_different_polys() {
        let mut c_tilde_a = [0u8; C_TILDE_BYTES];
        let mut c_tilde_b = [0u8; C_TILDE_BYTES];
        c_tilde_b[0] = 1; // Flip one bit
        let a = sample_in_ball_native(&c_tilde_a);
        let b = sample_in_ball_native(&c_tilde_b);
        assert_ne!(a, b, "Different seeds must produce different challenge polys");
    }

    #[test]
    fn sig_h_unpack_accepts_well_formed_hints() {
        let mut sig_bytes = [0u8; DILITHIUM5_SIG_BYTES];
        let hint_start = C_TILDE_BYTES + L * Z_PACKED_POLY_BYTES;
        // Poly 0: indices [3, 7, 100], i.e., 3 bits set
        sig_bytes[hint_start] = 3;
        sig_bytes[hint_start + 1] = 7;
        sig_bytes[hint_start + 2] = 100;
        // Poly 1: index [200], 1 bit set
        sig_bytes[hint_start + 3] = 200;
        // Polys 2..7: empty
        // Length bytes (cumulative): [3, 4, 4, 4, 4, 4, 4, 4]
        sig_bytes[hint_start + OMEGA] = 3;
        sig_bytes[hint_start + OMEGA + 1] = 4;
        for i in 2..K {
            sig_bytes[hint_start + OMEGA + i] = 4;
        }
        let sig = DilithiumSigBytes(sig_bytes);
        let h = sig.unpack_h_native().expect("well-formed hints must parse");
        // Verify the expected bits are set.
        assert!(h[0][3]);
        assert!(h[0][7]);
        assert!(h[0][100]);
        assert!(h[1][200]);
        // And no other bits.
        let total_set: usize = h.iter().flatten().filter(|&&b| b).count();
        assert_eq!(total_set, 4);
    }
}
