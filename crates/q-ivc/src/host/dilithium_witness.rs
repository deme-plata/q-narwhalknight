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

use crate::gadgets::dilithium::{NttRoots, PublicKeyVar, SignatureVar};

/// Dilithium5 packed public-key bytes (2 592 bytes).
///
/// Layout (FIPS-204 §5.2.1):
///   bytes  0..32  ρ — seed for matrix A
///   bytes 32..2592 packed t₁ (8 polynomials × 320 bytes each)
pub const DILITHIUM5_PK_BYTES: usize = 2_592;

/// Dilithium5 packed signature bytes (4 627 bytes).
///
/// Layout (FIPS-204 §5.2.3):
///   bytes    0..32  c̃ — 32-byte challenge seed
///   bytes   32..2304 packed z (7 polynomials × ~324 bytes each)
///   bytes 4544..4627 packed h (hint indices, ω + k bytes)
pub const DILITHIUM5_SIG_BYTES: usize = 4_627;

/// Dilithium parameters used by the host-side unpacker. Match
/// `gadgets/dilithium.rs::DILITHIUM_Q`.
pub const N: usize = 256;
pub const K: usize = 8;
pub const L: usize = 7;
pub const Q: u64 = 8_380_417;

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

    /// Allocate as in-circuit `PublicKeyVar`.
    ///
    /// **STUB.** The body needs to:
    ///   1. Expand ρ into A (the k×l matrix) via ExpandA (FIPS-204 §3).
    ///      ExpandA uses SHAKE-128 keyed by (ρ, j||i) for each polynomial
    ///      in row i, column j. Each polynomial is rejection-sampled
    ///      into the range [0, q-1] via the algorithm in FIPS-204 §A.1.
    ///      Two implementation paths:
    ///        a. Compute A natively (off-circuit), pass as witness, trust
    ///           the prover — soundness requires VERIFYING the SHAKE
    ///           expansion in-circuit OR using rho as a public input and
    ///           reproducing A in-circuit. Neither is cheap.
    ///        b. Make A a public input (one allocation per circuit
    ///           setup) — works if the validator set is bounded and
    ///           keys are public anyway. ~14 336 FpVar public inputs.
    ///      Decision pending; see whitepaper §3.4.
    ///   2. Unpack the t₁ region: 8 polynomials, each 256 coefficients
    ///      of 10 bits packed bit-by-bit in 320-byte chunks (per FIPS-204
    ///      §5.2.1 SimpleBitPack with d=10). Each unpacked coefficient
    ///      is allocated as an FpVar witness.
    ///   3. Wrap into PublicKeyVar { a_mat, t_vec }.
    pub fn allocate<F: PrimeField>(
        &self,
        _cs: ConstraintSystemRef<F>,
    ) -> Result<PublicKeyVar<F>, SynthesisError> {
        // TODO(dilithium-witness-pk-allocate): see module docstring +
        // FIPS-204 §3 / §5.2.1.
        Err(SynthesisError::AssignmentMissing)
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

    /// Extract c̃, the 32-byte challenge seed.
    pub fn c_tilde(&self) -> [u8; 32] {
        let mut c = [0u8; 32];
        c.copy_from_slice(&self.0[..32]);
        c
    }

    /// Allocate as in-circuit `SignatureVar`.
    ///
    /// **STUB.** The body needs to:
    ///   1. SampleInBall(c̃) → c_poly (FIPS-204 §4 "Sample in ball").
    ///      Deterministic: takes the 32-byte seed and produces a
    ///      polynomial with exactly τ=60 non-zero coefficients (each ±1)
    ///      via rejection sampling driven by SHAKE-256. Re-implementing
    ///      this in-circuit requires SHAKE-256 (Keccak-f[1600] permutation,
    ///      ~190 K constraints per call). Alternative: pass c_poly as
    ///      witness, hash c̃ in-circuit and verify the SampleInBall
    ///      relation via a dedicated AIR (separate sub-circuit, ~250 K
    ///      constraints).
    ///   2. Unpack z: 7 polynomials, each 256 coefficients in [-(γ₁-1),
    ///      γ₁-1] = [-524 287, 524 287]. FIPS-204 §5.2.3 BitPack with
    ///      d=20. Each unpacked coefficient is stored as a positive FpVar
    ///      (negative values represented as q-|v|).
    ///   3. Unpack h: 8 polynomials each holding 256 hint bits, packed
    ///      as ω=75 indices + k=8 length bytes per FIPS-204 §5.2.3
    ///      HintBitPack. Allocate as Boolean<F>.
    ///   4. Wrap into SignatureVar { z, h, c_poly }.
    pub fn allocate<F: PrimeField>(
        &self,
        _cs: ConstraintSystemRef<F>,
    ) -> Result<SignatureVar<F>, SynthesisError> {
        // TODO(dilithium-witness-sig-allocate): see module docstring +
        // FIPS-204 §5.2.3.
        Err(SynthesisError::AssignmentMissing)
    }
}

/// Compute μ = H(H(pk) || M) for the signed message.
///
/// FIPS-204 §5.1 step 6. `pk_bytes` is the full DILITHIUM5_PK_BYTES
/// packed key; `message` is the canonical bytes of the transaction
/// payload. Returns the SHAKE-256 output as 32 FpVar bytes packed into
/// 8 u32 words.
///
/// **STUB.** Needs in-circuit SHAKE-256 — substantial. Defer to a
/// follow-up commit that adds the SHAKE-256 AIR or imports a vetted
/// crate's gadget.
pub fn message_hash<F: PrimeField>(
    _cs: ConstraintSystemRef<F>,
    _pk_bytes: &[u8],
    _message: &[u8],
) -> Result<Vec<FpVar<F>>, SynthesisError> {
    // TODO(dilithium-witness-message-hash): SHAKE-256 over (H(pk) || M).
    Err(SynthesisError::AssignmentMissing)
}

/// Construct the standard FIPS-204 NTT roots for the Dilithium5
/// parameter set (n=256, q=8 380 417).
///
/// These are CONSTANTS — every call produces the same root table. The
/// gadget allocates them as `F::from(constant)` so they're zero-cost
/// in-circuit.
///
/// **STUB.** Needs the 256-element forward + inverse root tables.
/// They're derivable from the primitive 512-th root of unity ψ = 1 753
/// (the FIPS-204 reference value) but require careful bit-reversal
/// indexing. Land via a separate constants commit; the table is
/// 256 × 2 × 8 bytes = 4 KB at runtime.
pub fn standard_ntt_roots<F: PrimeField>() -> NttRoots<F> {
    // TODO(dilithium-witness-ntt-roots): populate from FIPS-204 §A.4
    // primitive root ψ=1753; bit-reversal-permuted root array.
    NttRoots {
        fwd: Vec::new(),
        inv: Vec::new(),
        n_inv: F::one(),
        psi: F::zero(),
        psi_inv: F::zero(),
    }
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
}
