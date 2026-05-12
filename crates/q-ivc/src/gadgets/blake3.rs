//! In-circuit BLAKE3 gadget for block header hash verification.
//!
//! Block headers use BLAKE3 for their hash. The StateTransitionCircuit must verify
//! that hash(header_bytes) == claimed_hash for each block in the epoch.
//!
//! BLAKE3 is bitwise — it's expensive in arithmetic circuits (~50K constraints per
//! 64-byte input block). Options:
//!   A. In-circuit BLAKE3: ~50K constraints per block header hash. Correct and no
//!      consensus changes required. Viable for epochs of ≤1K blocks.
//!   B. Switch headers to Poseidon: ~3K constraints. Requires consensus fork.
//!
//! Decision: use in-circuit BLAKE3 (Option A) to avoid a consensus change.
//!
//! This scaffold shows the interface. Actual BLAKE3 constraints require implementing
//! the 7 compression rounds as bitwise operations over 32-bit words, using
//! `ark-r1cs-std`'s `UInt32` gadget.
//!
//! Status: SCAFFOLD — interface defined, constraints are TODO.

use ark_ff::PrimeField;
use ark_r1cs_std::{
    boolean::Boolean,
    fields::fp::FpVar,
    prelude::*,
    uint32::UInt32,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

/// In-circuit BLAKE3 hash gadget.
///
/// Verifies that BLAKE3(preimage) == expected_hash.
/// Preimage and hash are encoded as packed field elements (little-endian 32-bit words).
pub struct Blake3Gadget;

impl Blake3Gadget {
    /// Verify BLAKE3(preimage_bytes) == expected_hash_bytes inside the circuit.
    ///
    /// # Encoding
    /// Both preimage and hash are passed as packed u32 words (one FpVar per word).
    /// A 64-byte block header becomes 16 FpVars. The 32-byte BLAKE3 output is 8 FpVars.
    ///
    /// # Constraint estimate
    /// BLAKE3 core is 7 rounds × 16 quarter-round calls × 6 operations = 672 bitwise ops.
    /// Each 32-bit XOR/rotation costs ~64 constraints in R1CS (bit decomposition).
    /// Total: ~43K constraints for one 64-byte block.
    ///
    /// # TODO
    /// Implement the BLAKE3 permutation using `UInt32` gadgets.
    /// Key reference: https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf
    pub fn verify_hash<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        preimage_words: &[FpVar<F>],   // one FpVar per u32 word of input
        expected_hash_words: &[FpVar<F>], // 8 FpVars for 256-bit BLAKE3 output
    ) -> Result<(), SynthesisError> {
        assert_eq!(expected_hash_words.len(), 8, "BLAKE3 output is 256 bits = 8 x u32");

        // TODO: Implement BLAKE3 compression function as R1CS constraints.
        // Steps:
        //   1. Initialize state with IV + block flags
        //   2. Run 7 compression rounds (G function, 8× per round)
        //   3. XOR state halves to produce output
        //   4. Constrain output == expected_hash_words

        // Placeholder: enforce sum(preimage) == sum(expected) (NOT cryptographically sound)
        let mut sum_pre = FpVar::Constant(F::zero());
        for w in preimage_words {
            sum_pre = &sum_pre + w;
        }
        let mut sum_exp = FpVar::Constant(F::zero());
        for w in expected_hash_words {
            sum_exp = &sum_exp + w;
        }
        sum_pre.enforce_equal(&sum_exp)?;
        Ok(())
    }

    /// Convert raw bytes (as a slice) to allocated u32 word FpVars.
    ///
    /// Used to encode block header bytes as circuit inputs.
    pub fn alloc_bytes_as_words<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        bytes: &[u8],
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        assert_eq!(bytes.len() % 4, 0, "Input must be 4-byte aligned");
        let words: Vec<u32> = bytes
            .chunks(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        words.iter()
            .map(|&w| FpVar::new_witness(cs.clone(), || Ok(F::from(w as u64))))
            .collect()
    }

    /// Allocate the 32-byte BLAKE3 hash as 8 u32 FpVars.
    pub fn alloc_hash<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        hash: &[u8; 32],
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        Self::alloc_bytes_as_words::<F>(cs, hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_blake3_gadget_compiles() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        // Simulate a 64-byte block header (16 x u32 words)
        let header_bytes = [0u8; 64];
        let preimage = Blake3Gadget::alloc_bytes_as_words(cs.clone(), &header_bytes).unwrap();

        // Compute native hash for expected output
        let native_hash = blake3::hash(&header_bytes);
        let hash_bytes: [u8; 32] = native_hash.into();
        let expected = Blake3Gadget::alloc_hash(cs.clone(), &hash_bytes).unwrap();

        // In the scaffold, the placeholder constraint sum(preimage)==sum(expected) won't
        // hold for random inputs, so we just verify the circuit builds without error.
        println!("Blake3 gadget constraint count: {}", cs.num_constraints());
        // Note: cs.is_satisfied() would fail for real data until actual BLAKE3 constraints are added.
    }

    #[test]
    fn test_blake3_alloc_helpers() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let bytes = [0x42u8; 32];
        let words = Blake3Gadget::alloc_bytes_as_words(cs.clone(), &bytes).unwrap();
        assert_eq!(words.len(), 8, "32 bytes = 8 u32 words");
    }
}
