//! EpochTransitionCircuit — top-level IVC composition circuit.
//!
//! Composes four sub-circuits enforcing one epoch transition:
//!
//!   1. **Block-header BLAKE3 hash chain.** For every block in the epoch,
//!      `BLAKE3(block_headers[i]) == block_hashes[i]` is enforced via
//!      `Blake3Gadget::verify_hash`.
//!
//!   2. **BFT 2f+1 threshold.** At least `BFT_THRESHOLD` valid Dilithium5
//!      signatures over `bft_message_hash`. Skipped for genesis epochs
//!      (`prev_epoch_proof_commitment.is_none()`) so the chain can start
//!      without an attesting prior validator set.
//!
//!   3. **Recursive prior-epoch binding (Phase A — Poseidon hash).** The
//!      prior commitment is `Poseidon(prev_state_root || epoch_no ||
//!      validator_set_hash)`. Substituting any of the three values
//!      without re-running Poseidon over a matching preimage off-chain
//!      fails the in-circuit equality check. Off-chain witness producers
//!      use [`compute_prior_commitment_native`] to compute the matching
//!      value. The future recursive-verifier upgrade (cycle-of-curves or
//!      lattice folding) replaces this binding with an in-circuit proof
//!      check — the function signature is final, the upgrade is a
//!      single-edit change.
//!
//!   4. **State-transition chain endpoint binding (Phase A).** When the
//!      caller supplies a `block_state_roots` chain (length must equal
//!      `block_headers.len() + 1`), the first entry is enforced equal to
//!      `prev_state_root` and the last equal to `next_state_root`
//!      (word-by-word over 8 little-endian `u32`s each). Per-block
//!      transition correctness (the rule that `block_state_roots[i+1]`
//!      is the result of applying block `i`'s transactions to
//!      `block_state_roots[i]`) is delegated to the per-block δ-circuit
//!      proof attached by the recursive-folding layer in Phase B.
//!
//!      The empty-vec sentinel (`block_state_roots: Vec::new()`) skips
//!      sub-circuit 4 entirely — backward compat for pre-Phase-A test
//!      fixtures that don't yet supply a chain.
//!
//! ## Status (Phase A — sub-circuits 1/2/3 live, sub-circuit 4 endpoints live)
//!
//! For genesis epoch (epoch_no = 0): set
//! `prev_epoch_proof_commitment = None`.
//!
//! For subsequent epochs: include the prior epoch's commitment computed
//! via [`compute_prior_commitment_native`] over the tuple
//! `(prev_state_root, epoch_no, validator_set_hash)`.

use ark_ff::PrimeField;
use ark_r1cs_std::{
    boolean::Boolean,
    fields::fp::FpVar,
    prelude::*,
    uint32::UInt32,
    R1CSVar,
};
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError,
};

use crate::gadgets::{
    blake3::Blake3Gadget,
    dilithium::DilithiumVerifierGadget,
    poseidon::PoseidonGadget,
};

/// BFT minimum threshold constant (2f+1 for f=1 Byzantine faults with 4
/// validators). Update based on actual validator set size:
/// `threshold = floor(2n/3) + 1`.
pub const BFT_THRESHOLD: usize = 3;

/// Number of `u32` words a 32-byte state root chunks into.
pub const STATE_ROOT_WORDS: usize = 8;

/// Number of `u64` chunks a 32-byte validator-set hash chunks into.
pub const VSH_CHUNKS: usize = 4;

/// Input data for one validator's signature (BFT sub-circuit).
#[derive(Clone)]
pub struct ValidatorSignatureInput<F: PrimeField> {
    /// Dilithium5 public-key components `(t, ρ)` as field elements.
    pub public_key: Vec<FpVar<F>>,
    /// Signature `z`-component (L×N field elements).
    pub sig_z: Vec<FpVar<F>>,
    /// Signature `h`-component (K×N bits).
    pub sig_h: Vec<Boolean<F>>,
    /// Challenge `c̃` (8 field elements).
    pub sig_c_tilde: Vec<FpVar<F>>,
}

/// Complete input to the [`EpochTransitionCircuit`].
pub struct EpochTransitionInputs<F: PrimeField> {
    // ── State-root endpoints (public, 32-byte BLAKE3 outputs) ─────────
    /// State root at the start of this epoch.
    pub prev_state_root: [u8; 32],
    /// State root at the end of this epoch.
    pub next_state_root: [u8; 32],

    // ── Epoch identity (witness, bound via sub-circuit 3) ─────────────
    /// Epoch number — part of the prior-commitment Poseidon preimage.
    pub epoch_no: u64,
    /// 32-byte hash of the validator set active during this epoch —
    /// part of the prior-commitment Poseidon preimage.
    pub validator_set_hash: [u8; 32],

    // ── Prior-epoch commitment (Some for non-genesis, None for genesis)
    /// Prior epoch's terminal Poseidon commitment. `None` for genesis;
    /// otherwise the value returned by [`compute_prior_commitment_native`]
    /// applied to (`prev_state_root`, `epoch_no`, `validator_set_hash`).
    pub prev_epoch_proof_commitment: Option<F>,

    // ── Block chain (witness) ─────────────────────────────────────────
    /// Block header bytes for each block in this epoch (64 bytes each).
    pub block_headers: Vec<Vec<u8>>,
    /// BLAKE3 hashes of each block header — sub-circuit 1 enforces
    /// `BLAKE3(block_headers[i]) == block_hashes[i]`.
    pub block_hashes: Vec<[u8; 32]>,
    /// Intermediate state roots through this epoch. Length must equal
    /// `block_headers.len() + 1`. Index 0 is enforced equal to
    /// `prev_state_root` and the final entry to `next_state_root`. Set
    /// to `Vec::new()` to disable sub-circuit 4 (backward-compat for
    /// pre-Phase-A fixtures).
    pub block_state_roots: Vec<[u8; 32]>,

    // ── BFT signatures (witness, threshold-checked) ───────────────────
    /// BFT validator signatures for this epoch. Use `None` for absent.
    pub validator_signatures: Vec<Option<ValidatorSignatureInput<F>>>,
    /// The message signed by BFT validators (e.g., Poseidon hash of the
    /// epoch block root).
    pub bft_message_hash: Vec<F>,
}

/// IVC epoch transition circuit.
///
/// Implements `ConstraintSynthesizer` so it composes with any
/// arkworks-compatible proving backend (Groth16, PLONK, Marlin).
pub struct EpochTransitionCircuit<F: PrimeField> {
    pub inputs: EpochTransitionInputs<F>,
}

// ════════════════════════════════════════════════════════════════════════════
// Decoders + helpers (shared between in-circuit and native paths)
// ════════════════════════════════════════════════════════════════════════════

/// Decode a 32-byte hash into `STATE_ROOT_WORDS` little-endian `u32`s —
/// the same word order used by `delta_block::alloc_root_input` and
/// `recursion::step_circuit::StepIO::pack`, so a single canonical
/// representation flows through every IVC layer.
fn root_words(root: &[u8; 32]) -> [u32; STATE_ROOT_WORDS] {
    let mut out = [0u32; STATE_ROOT_WORDS];
    for (i, chunk) in root.chunks(4).enumerate() {
        out[i] = u32::from_le_bytes(chunk.try_into().expect("4-byte chunk"));
    }
    out
}

/// Decode a 32-byte hash into `VSH_CHUNKS` little-endian `u64`s.
fn vsh_chunks(vsh: &[u8; 32]) -> [u64; VSH_CHUNKS] {
    let mut out = [0u64; VSH_CHUNKS];
    for (i, chunk) in vsh.chunks(8).enumerate() {
        out[i] = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
    }
    out
}

/// Allocate a 32-byte state root as 8 `UInt32<F>`s — public inputs when
/// `public = true`, witnesses otherwise. Little-endian word order.
fn alloc_state_root<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    root: &[u8; 32],
    public: bool,
) -> Result<Vec<UInt32<F>>, SynthesisError> {
    let words = root_words(root);
    words
        .iter()
        .map(|&w| {
            if public {
                UInt32::new_input(cs.clone(), || Ok(w))
            } else {
                UInt32::new_witness(cs.clone(), || Ok(w))
            }
        })
        .collect()
}

/// View-convert a `UInt32<F>` to an `FpVar<F>` via its little-endian bit
/// decomposition. No fresh witness allocation — adds a linear-combination
/// constraint over the SAME constrained bits already on the wire.
fn uint32_to_fp_var<F: PrimeField>(word: &UInt32<F>) -> Result<FpVar<F>, SynthesisError> {
    Boolean::le_bits_to_fp_var(&word.to_bits_le())
}

/// Enforce two 8-word state roots are equal via bit-by-bit equality on
/// each word's little-endian decomposition. ~256 constraints (8 words ×
/// 32 bits).
fn enforce_root_eq<F: PrimeField>(
    a: &[UInt32<F>],
    b: &[UInt32<F>],
) -> Result<(), SynthesisError> {
    debug_assert_eq!(a.len(), STATE_ROOT_WORDS);
    debug_assert_eq!(b.len(), STATE_ROOT_WORDS);
    for (aw, bw) in a.iter().zip(b.iter()) {
        let a_bits = aw.to_bits_le();
        let b_bits = bw.to_bits_le();
        for (ab, bb) in a_bits.iter().zip(b_bits.iter()) {
            ab.enforce_equal(bb)?;
        }
    }
    Ok(())
}

/// Build the canonical sub-circuit 3 Poseidon preimage:
/// `[prev_state_root_words (8 × u32 as FpVar),
///   epoch_no (1 × u64 as FpVar),
///   validator_set_hash_chunks (4 × u64 as FpVar)]`.
///
/// Used by both the in-circuit binding and the native helper so the
/// two stay in lockstep.
fn build_prior_preimage<F: PrimeField>(
    prev_root_words_fp: Vec<FpVar<F>>,
    epoch_no_var: FpVar<F>,
    vsh_vars: Vec<FpVar<F>>,
) -> Vec<FpVar<F>> {
    let mut preimage =
        Vec::with_capacity(STATE_ROOT_WORDS + 1 + VSH_CHUNKS);
    preimage.extend(prev_root_words_fp);
    preimage.push(epoch_no_var);
    preimage.extend(vsh_vars);
    preimage
}

/// Compute the prior-epoch Poseidon commitment off-circuit.
///
/// Mirrors sub-circuit 3's in-circuit Poseidon binding exactly. Use this
/// when constructing `EpochTransitionInputs::prev_epoch_proof_commitment`
/// for a non-genesis epoch: the value must equal what the in-circuit
/// gadget would compute from the same
/// `(prev_state_root, epoch_no, validator_set_hash)` triple.
///
/// Runs the in-circuit gadget against a throwaway constraint system and
/// extracts the resulting witness value — the only correct way to keep
/// native + in-circuit Poseidon outputs in lockstep without duplicating
/// the permutation logic.
pub fn compute_prior_commitment_native<F: PrimeField>(
    prev_state_root: &[u8; 32],
    epoch_no: u64,
    validator_set_hash: &[u8; 32],
) -> F {
    let cs = ConstraintSystem::<F>::new_ref();
    let prev_words = root_words(prev_state_root);
    let vsh_words = vsh_chunks(validator_set_hash);

    let prev_root_fp: Vec<FpVar<F>> = prev_words
        .iter()
        .map(|&w| FpVar::new_witness(cs.clone(), || Ok(F::from(w))))
        .collect::<Result<_, _>>()
        .expect("native prev-root allocation must not fail");
    let epoch_no_var = FpVar::new_witness(cs.clone(), || Ok(F::from(epoch_no)))
        .expect("native epoch-no allocation must not fail");
    let vsh_vars: Vec<FpVar<F>> = vsh_words
        .iter()
        .map(|&w| FpVar::new_witness(cs.clone(), || Ok(F::from(w))))
        .collect::<Result<_, _>>()
        .expect("native vsh allocation must not fail");

    let preimage = build_prior_preimage(prev_root_fp, epoch_no_var, vsh_vars);
    let computed = PoseidonGadget::hash_many::<F>(cs, &preimage)
        .expect("native Poseidon hash must not fail");
    computed
        .value()
        .expect("native FpVar value must be set")
}

// ════════════════════════════════════════════════════════════════════════════
// Constraint synthesizer
// ════════════════════════════════════════════════════════════════════════════

impl<F: PrimeField> ConstraintSynthesizer<F> for EpochTransitionCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        let inputs = self.inputs;

        // ── Allocate the public state-root endpoints as 8 u32 words each ─
        let prev_root_words = alloc_state_root(cs.clone(), &inputs.prev_state_root, true)?;
        let next_root_words = alloc_state_root(cs.clone(), &inputs.next_state_root, true)?;

        // ╔═══════════════════════════════════════════════════════════════╗
        // ║  Sub-circuit 1 — block-header BLAKE3 hash chain               ║
        // ╚═══════════════════════════════════════════════════════════════╝
        for (header, claimed_hash) in inputs
            .block_headers
            .iter()
            .zip(inputs.block_hashes.iter())
        {
            let preimage = Blake3Gadget::alloc_bytes_as_words(cs.clone(), header)?;
            let expected = Blake3Gadget::alloc_hash(cs.clone(), claimed_hash)?;
            Blake3Gadget::verify_hash(cs.clone(), &preimage, &expected)?;
        }

        // ╔═══════════════════════════════════════════════════════════════╗
        // ║  Sub-circuit 2 — BFT 2f+1 threshold (genesis carve-out)       ║
        // ╚═══════════════════════════════════════════════════════════════╝
        let is_genesis = inputs.prev_epoch_proof_commitment.is_none();

        let bft_msg: Vec<FpVar<F>> = inputs
            .bft_message_hash
            .iter()
            .map(|v| FpVar::new_witness(cs.clone(), || Ok(*v)))
            .collect::<Result<_, _>>()?;

        let validator_data: Vec<
            Option<(
                Vec<FpVar<F>>,
                Vec<FpVar<F>>,
                Vec<Boolean<F>>,
                Vec<FpVar<F>>,
            )>,
        > = inputs
            .validator_signatures
            .into_iter()
            .map(|opt| opt.map(|s| (s.public_key, s.sig_z, s.sig_h, s.sig_c_tilde)))
            .collect();

        let bft_valid = DilithiumVerifierGadget::verify_threshold(
            cs.clone(),
            BFT_THRESHOLD,
            &bft_msg,
            &validator_data,
        )?;
        if !is_genesis {
            bft_valid.enforce_equal(&Boolean::constant(true))?;
        }

        // ╔═══════════════════════════════════════════════════════════════╗
        // ║  Sub-circuit 3 — recursive prior-epoch Poseidon binding       ║
        // ╚═══════════════════════════════════════════════════════════════╝
        if let Some(prior_commit) = inputs.prev_epoch_proof_commitment {
            let epoch_no_var =
                FpVar::new_witness(cs.clone(), || Ok(F::from(inputs.epoch_no)))?;
            let vsh_words = vsh_chunks(&inputs.validator_set_hash);
            let vsh_vars: Vec<FpVar<F>> = vsh_words
                .iter()
                .map(|&w| FpVar::new_witness(cs.clone(), || Ok(F::from(w))))
                .collect::<Result<_, _>>()?;

            // View-convert the public prev_root UInt32 words to FpVar.
            let prev_root_fp: Vec<FpVar<F>> = prev_root_words
                .iter()
                .map(uint32_to_fp_var)
                .collect::<Result<_, _>>()?;

            let preimage = build_prior_preimage(prev_root_fp, epoch_no_var, vsh_vars);
            let computed = PoseidonGadget::hash_many::<F>(cs.clone(), &preimage)?;
            let prior_var = FpVar::new_witness(cs.clone(), || Ok(prior_commit))?;
            computed.enforce_equal(&prior_var)?;
        }

        // ╔═══════════════════════════════════════════════════════════════╗
        // ║  Sub-circuit 4 — state-transition chain endpoint binding      ║
        // ╚═══════════════════════════════════════════════════════════════╝
        if !inputs.block_state_roots.is_empty() {
            if inputs.block_state_roots.len() != inputs.block_headers.len() + 1 {
                // Misshaped witness — fail loud rather than silently
                // accept a partial chain.
                return Err(SynthesisError::AssignmentMissing);
            }

            let intermediates: Vec<Vec<UInt32<F>>> = inputs
                .block_state_roots
                .iter()
                .map(|root| alloc_state_root(cs.clone(), root, false))
                .collect::<Result<_, _>>()?;

            // Endpoint equality. Per-block transition correctness (i.e.,
            // intermediates[i+1] = δ(intermediates[i], block_i)) is the
            // per-block δ-circuit's job, attached by the Phase B fold.
            enforce_root_eq(&intermediates[0], &prev_root_words)?;
            let last = intermediates.len() - 1;
            enforce_root_eq(&intermediates[last], &next_root_words)?;
        }

        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    fn hash_of(bytes: &[u8]) -> [u8; 32] {
        *blake3::hash(bytes).as_bytes()
    }

    fn empty_64() -> Vec<u8> {
        vec![0u8; 64]
    }

    fn empty_64_hash() -> [u8; 32] {
        hash_of(&empty_64())
    }

    fn genesis_inputs() -> EpochTransitionInputs<Fr> {
        EpochTransitionInputs {
            prev_state_root: [1u8; 32],
            next_state_root: [1u8; 32],
            epoch_no: 0,
            validator_set_hash: [0u8; 32],
            prev_epoch_proof_commitment: None,
            block_headers: vec![empty_64()],
            block_hashes: vec![empty_64_hash()],
            block_state_roots: Vec::new(),
            validator_signatures: vec![None; 3],
            bft_message_hash: vec![Fr::from(42u64)],
        }
    }

    #[test]
    fn test_epoch_circuit_genesis_satisfiable() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        EpochTransitionCircuit { inputs: genesis_inputs() }
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(
            cs.is_satisfied().unwrap(),
            "genesis epoch circuit must be satisfiable"
        );
    }

    #[test]
    fn test_epoch_circuit_genesis_rejects_tampered_block_hash() {
        let mut inputs = genesis_inputs();
        inputs.block_hashes[0][0] ^= 1; // flip one bit
        let cs = ConstraintSystem::<Fr>::new_ref();
        EpochTransitionCircuit { inputs }
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "BLAKE3 hash chain must reject a tampered block_hash"
        );
    }

    #[test]
    fn test_non_genesis_with_zero_sigs_fails_bft_threshold() {
        let prev_root = [3u8; 32];
        let vsh = [5u8; 32];
        let epoch_no = 7u64;
        let prior_commit =
            compute_prior_commitment_native::<Fr>(&prev_root, epoch_no, &vsh);
        let inputs = EpochTransitionInputs {
            prev_state_root: prev_root,
            next_state_root: prev_root,
            epoch_no,
            validator_set_hash: vsh,
            prev_epoch_proof_commitment: Some(prior_commit),
            block_headers: vec![empty_64()],
            block_hashes: vec![empty_64_hash()],
            block_state_roots: Vec::new(),
            validator_signatures: vec![None; 3],
            bft_message_hash: vec![Fr::from(1u64)],
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        EpochTransitionCircuit { inputs }
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "non-genesis epoch with zero signatures must fail the 2f+1 threshold"
        );
    }

    #[test]
    fn test_prior_commitment_binds_prev_state_root() {
        // Honest commit for prev_root=[1;32]; circuit is built with
        // prev_root=[2;32] — in-circuit Poseidon recomputes to a
        // different value, equality with the honest commit fails,
        // circuit must reject.
        let honest_prev = [1u8; 32];
        let dishonest_prev = [2u8; 32];
        let vsh = [9u8; 32];
        let epoch_no = 7u64;
        let prior_commit =
            compute_prior_commitment_native::<Fr>(&honest_prev, epoch_no, &vsh);
        let inputs = EpochTransitionInputs {
            prev_state_root: dishonest_prev,
            next_state_root: dishonest_prev,
            epoch_no,
            validator_set_hash: vsh,
            prev_epoch_proof_commitment: Some(prior_commit),
            block_headers: vec![empty_64()],
            block_hashes: vec![empty_64_hash()],
            block_state_roots: Vec::new(),
            validator_signatures: vec![None; 3],
            bft_message_hash: vec![Fr::from(1u64)],
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        EpochTransitionCircuit { inputs }
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "sub-circuit 3 must REJECT a prev_state_root that does not match the prior commitment"
        );
    }

    #[test]
    fn test_prior_commitment_binds_epoch_no() {
        // Honest commit for epoch_no=7; circuit built with epoch_no=99.
        let prev_root = [3u8; 32];
        let vsh = [5u8; 32];
        let honest_commit =
            compute_prior_commitment_native::<Fr>(&prev_root, 7, &vsh);
        let inputs = EpochTransitionInputs {
            prev_state_root: prev_root,
            next_state_root: prev_root,
            epoch_no: 99,
            validator_set_hash: vsh,
            prev_epoch_proof_commitment: Some(honest_commit),
            block_headers: vec![empty_64()],
            block_hashes: vec![empty_64_hash()],
            block_state_roots: Vec::new(),
            validator_signatures: vec![None; 3],
            bft_message_hash: vec![Fr::from(1u64)],
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        EpochTransitionCircuit { inputs }
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "sub-circuit 3 must REJECT a mismatched epoch_no"
        );
    }

    #[test]
    fn test_state_root_chain_endpoint_binding_satisfied() {
        let g = [1u8; 32];
        let inputs = EpochTransitionInputs {
            prev_state_root: g,
            next_state_root: g,
            epoch_no: 0,
            validator_set_hash: [0u8; 32],
            prev_epoch_proof_commitment: None,
            block_headers: vec![empty_64()],
            block_hashes: vec![empty_64_hash()],
            block_state_roots: vec![g, g],
            validator_signatures: vec![None; 3],
            bft_message_hash: vec![Fr::from(1u64)],
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        EpochTransitionCircuit { inputs }
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(
            cs.is_satisfied().unwrap(),
            "chain with endpoints matching prev/next must be satisfiable"
        );
    }

    #[test]
    fn test_state_root_chain_rejects_terminal_mismatch() {
        let prev = [1u8; 32];
        let claimed_next = [2u8; 32];
        let chain_terminal = [9u8; 32]; // ≠ claimed_next
        let inputs = EpochTransitionInputs {
            prev_state_root: prev,
            next_state_root: claimed_next,
            epoch_no: 0,
            validator_set_hash: [0u8; 32],
            prev_epoch_proof_commitment: None,
            block_headers: vec![empty_64()],
            block_hashes: vec![empty_64_hash()],
            block_state_roots: vec![prev, chain_terminal],
            validator_signatures: vec![None; 3],
            bft_message_hash: vec![Fr::from(1u64)],
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        EpochTransitionCircuit { inputs }
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "chain terminus disagreeing with claimed next_state_root must be REJECTED"
        );
    }

    #[test]
    fn test_state_root_chain_rejects_initial_mismatch() {
        let claimed_prev = [1u8; 32];
        let next = [9u8; 32];
        let chain_start = [2u8; 32]; // ≠ claimed_prev
        let inputs = EpochTransitionInputs {
            prev_state_root: claimed_prev,
            next_state_root: next,
            epoch_no: 0,
            validator_set_hash: [0u8; 32],
            prev_epoch_proof_commitment: None,
            block_headers: vec![empty_64()],
            block_hashes: vec![empty_64_hash()],
            block_state_roots: vec![chain_start, next],
            validator_signatures: vec![None; 3],
            bft_message_hash: vec![Fr::from(1u64)],
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        EpochTransitionCircuit { inputs }
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "chain start disagreeing with claimed prev_state_root must be REJECTED"
        );
    }

    #[test]
    fn test_state_root_chain_rejects_length_mismatch() {
        // 1-block epoch with a 3-entry chain (must be N+1=2).
        let g = [1u8; 32];
        let inputs = EpochTransitionInputs {
            prev_state_root: g,
            next_state_root: g,
            epoch_no: 0,
            validator_set_hash: [0u8; 32],
            prev_epoch_proof_commitment: None,
            block_headers: vec![empty_64()],
            block_hashes: vec![empty_64_hash()],
            block_state_roots: vec![g, g, g],
            validator_signatures: vec![None; 3],
            bft_message_hash: vec![Fr::from(1u64)],
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        let result = EpochTransitionCircuit { inputs }.generate_constraints(cs);
        assert!(
            result.is_err(),
            "block_state_roots length != block_headers.len() + 1 must fail loudly"
        );
    }

    #[test]
    fn test_compute_prior_commitment_is_deterministic() {
        // The native helper must be a pure function of its inputs.
        let prev = [7u8; 32];
        let vsh = [11u8; 32];
        let a = compute_prior_commitment_native::<Fr>(&prev, 42, &vsh);
        let b = compute_prior_commitment_native::<Fr>(&prev, 42, &vsh);
        assert_eq!(a, b, "compute_prior_commitment_native must be deterministic");

        let c = compute_prior_commitment_native::<Fr>(&prev, 43, &vsh);
        assert_ne!(
            a, c,
            "different epoch_no must produce a different commitment"
        );

        let d = compute_prior_commitment_native::<Fr>(&[8u8; 32], 42, &vsh);
        assert_ne!(
            a, d,
            "different prev_state_root must produce a different commitment"
        );
    }
}
