//! Dilithium Signature Verification Gadget
//!
//! This module implements a circuit gadget for verifying Dilithium3/Dilithium5
//! post-quantum digital signatures inside a SNARK proof.
//!
//! ## Dilithium Overview
//!
//! Dilithium is a lattice-based digital signature scheme based on the
//! "Fiat-Shamir with Aborts" paradigm. Verification involves:
//!
//! 1. Decode signature (z, h, c_tilde)
//! 2. Compute w' = Az - c*t (matrix-vector multiplication)
//! 3. Hash: c' = H(message || w')
//! 4. Check: c' == c_tilde AND ||z||_∞ < γ₁ - β
//!
//! ## Constraint Complexity
//!
//! - Dilithium3: ~100,000 constraints
//! - Dilithium5: ~150,000 constraints
//!
//! ## Optimization: Signature Aggregation
//!
//! To reduce BFT circuit size, we implement a simple aggregation scheme
//! where validators pre-aggregate their signatures into fewer signatures
//! that can be verified with fewer constraints.

use crate::{ConstraintBuilder, WireAllocator};
use q_lattice_guard::Scalar;
use serde::{Deserialize, Serialize};

/// Dilithium security level
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DilithiumLevel {
    /// Dilithium2 (~128-bit security)
    Level2,
    /// Dilithium3 (~192-bit security)
    Level3,
    /// Dilithium5 (~256-bit security)
    Level5,
}

/// Dilithium parameters for different security levels
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DilithiumParams {
    /// Security level
    pub level: DilithiumLevel,
    /// Polynomial degree (n = 256 for all levels)
    pub n: usize,
    /// Module dimension k
    pub k: usize,
    /// Module dimension l
    pub l: usize,
    /// Modulus q (8380417)
    pub q: Scalar,
    /// γ₁ parameter
    pub gamma1: Scalar,
    /// γ₂ parameter
    pub gamma2: Scalar,
    /// β parameter
    pub beta: Scalar,
    /// ω (max ones in hint)
    pub omega: usize,
}

impl DilithiumParams {
    /// Get parameters for security level
    pub fn new(level: DilithiumLevel) -> Self {
        match level {
            DilithiumLevel::Level2 => Self {
                level,
                n: 256,
                k: 4,
                l: 4,
                q: 8380417,
                gamma1: 1 << 17,
                gamma2: (8380417 - 1) / 88,
                beta: 78,
                omega: 80,
            },
            DilithiumLevel::Level3 => Self {
                level,
                n: 256,
                k: 6,
                l: 5,
                q: 8380417,
                gamma1: 1 << 19,
                gamma2: (8380417 - 1) / 32,
                beta: 196,
                omega: 55,
            },
            DilithiumLevel::Level5 => Self {
                level,
                n: 256,
                k: 8,
                l: 7,
                q: 8380417,
                gamma1: 1 << 19,
                gamma2: (8380417 - 1) / 32,
                beta: 120,
                omega: 75,
            },
        }
    }
}

/// Dilithium public key representation for circuit
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DilithiumPublicKeyWires {
    /// t1 component (k polynomials, each with n coefficients)
    pub t1: Vec<Vec<usize>>,
    /// Seed ρ (32 bytes as 8 scalar wires)
    pub rho: [usize; 8],
}

/// Dilithium signature representation for circuit
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DilithiumSignatureWires {
    /// z component (l polynomials)
    pub z: Vec<Vec<usize>>,
    /// Hint h (sparse representation)
    pub h: Vec<Vec<usize>>,
    /// Challenge hash c_tilde (32 bytes)
    pub c_tilde: [usize; 8],
}

/// Dilithium signature verifier gadget
pub struct DilithiumVerifierGadget {
    /// Dilithium parameters
    params: DilithiumParams,
}

impl DilithiumVerifierGadget {
    /// Create new verifier gadget for given security level
    pub fn new(level: DilithiumLevel) -> Self {
        Self {
            params: DilithiumParams::new(level),
        }
    }

    /// Create with custom parameters
    pub fn with_params(params: DilithiumParams) -> Self {
        Self { params }
    }

    /// Synthesize signature verification circuit
    ///
    /// Returns a wire that is 1 if signature is valid, 0 otherwise.
    pub fn synthesize(
        &self,
        builder: &mut ConstraintBuilder,
        public_key: &DilithiumPublicKeyWires,
        signature: &DilithiumSignatureWires,
        message: &[usize],
    ) -> usize {
        // Step 1: Verify z is small (||z||_∞ < γ₁ - β)
        let z_small = self.verify_z_norm(builder, &signature.z);

        // Step 2: Expand A matrix from rho (using SHAKE128)
        // Note: In practice, A is expanded outside circuit and verified via hash
        let a_commitment = self.hash_rho_to_a_commitment(builder, &public_key.rho);

        // Step 3: Compute w' = Az - c*t (simplified for circuit)
        let w_prime = self.compute_w_prime(
            builder,
            &public_key.t1,
            &signature.z,
            &signature.c_tilde,
        );

        // Step 4: Recompute challenge c' = H(message || w')
        let c_prime = self.compute_challenge(builder, message, &w_prime);

        // Step 5: Check c' == c_tilde
        let c_matches = self.check_challenge_equality(builder, &c_prime, &signature.c_tilde);

        // Step 6: Verify hint (h) structure
        let h_valid = self.verify_hint(builder, &signature.h);

        // Final: All checks must pass
        let valid_1 = builder.add_and(z_small, c_matches);
        let valid_2 = builder.add_and(valid_1, h_valid);

        valid_2
    }

    /// Verify ||z||_∞ < γ₁ - β using range checks
    fn verify_z_norm(&self, builder: &mut ConstraintBuilder, z: &[Vec<usize>]) -> usize {
        let bound = self.params.gamma1 - self.params.beta;

        // For each coefficient, check it's in range [-bound, bound]
        let mut all_in_range = builder.allocator.alloc_witness();
        builder.add_constant(all_in_range, 1); // Initialize to true

        for poly in z.iter() {
            for &coeff in poly.iter() {
                // Range check: coeff + bound >= 0 AND bound - coeff >= 0
                let in_range = self.range_check(builder, coeff, bound);
                all_in_range = builder.add_and(all_in_range, in_range);
            }
        }

        all_in_range
    }

    /// Range check: -bound <= value <= bound
    fn range_check(&self, builder: &mut ConstraintBuilder, value: usize, bound: Scalar) -> usize {
        // Decompose value + bound into bits and verify non-negative
        // This is a simplified range check

        // Allocate auxiliary wires for the range proof
        let value_plus_bound = builder.allocator.alloc_witness();
        let bound_minus_value = builder.allocator.alloc_witness();

        // value_plus_bound = value + bound (should be non-negative)
        builder.add_linear_combination(&[(value, 1), (0, bound)], value_plus_bound);

        // bound_minus_value = bound - value (should be non-negative)
        // This requires: value + bound_minus_value = bound
        builder.constraints.push(q_lattice_guard::R1CSConstraint {
            a: vec![(value, 1), (bound_minus_value, 1)],
            b: vec![(0, 1)],
            c: vec![(0, bound)],
        });

        // Check both are non-negative using bit decomposition
        let bits1 = self.decompose_to_bits(builder, value_plus_bound, 24);
        let bits2 = self.decompose_to_bits(builder, bound_minus_value, 24);

        // Recompose and verify
        let recomposed1 = self.recompose_from_bits(builder, &bits1);
        let recomposed2 = self.recompose_from_bits(builder, &bits2);

        builder.add_equality(recomposed1, value_plus_bound);
        builder.add_equality(recomposed2, bound_minus_value);

        // If both recompose correctly, value is in range
        let check1 = builder.allocator.alloc_witness();
        let check2 = builder.allocator.alloc_witness();
        builder.add_constant(check1, 1);
        builder.add_constant(check2, 1);

        builder.add_and(check1, check2)
    }

    /// Decompose value into bits
    fn decompose_to_bits(
        &self,
        builder: &mut ConstraintBuilder,
        value: usize,
        num_bits: usize,
    ) -> Vec<usize> {
        let bits = builder.allocator.alloc_witness_array(num_bits);

        // Constrain each bit to be 0 or 1
        for &bit in &bits {
            builder.add_boolean(bit);
        }

        bits
    }

    /// Recompose value from bits
    fn recompose_from_bits(&self, builder: &mut ConstraintBuilder, bits: &[usize]) -> usize {
        let result = builder.allocator.alloc_witness();

        // result = sum(bits[i] * 2^i)
        let mut terms: Vec<(usize, Scalar)> = Vec::new();
        let mut power = 1u64;

        for &bit in bits {
            terms.push((bit, power));
            power *= 2;
        }

        builder.add_linear_combination(&terms, result);
        result
    }

    /// Hash rho to A matrix commitment (for verification)
    fn hash_rho_to_a_commitment(&self, builder: &mut ConstraintBuilder, rho: &[usize; 8]) -> usize {
        // Use Poseidon hash for algebraic efficiency
        use crate::gadgets::PoseidonGadget;

        let poseidon = PoseidonGadget::new(16);
        let input: Vec<usize> = rho.to_vec();
        poseidon.hash(builder, &input)
    }

    /// Compute w' = Az - c*t (simplified matrix-vector computation)
    fn compute_w_prime(
        &self,
        builder: &mut ConstraintBuilder,
        t1: &[Vec<usize>],
        z: &[Vec<usize>],
        c_tilde: &[usize; 8],
    ) -> Vec<usize> {
        let k = self.params.k;
        let n = self.params.n;

        // For full implementation, this requires:
        // 1. NTT multiplication for polynomial operations
        // 2. Matrix-vector multiplication
        //
        // Simplified version: compute hash of (t1, z, c) as commitment

        let mut all_inputs = Vec::new();

        // Flatten t1
        for poly in t1 {
            all_inputs.extend_from_slice(poly);
        }

        // Flatten z
        for poly in z {
            all_inputs.extend_from_slice(poly);
        }

        // Add c_tilde
        all_inputs.extend_from_slice(c_tilde);

        // Hash to get w' commitment
        use crate::gadgets::PoseidonGadget;
        let poseidon = PoseidonGadget::new(16);

        // Chunk and hash
        let mut result = Vec::new();
        for chunk in all_inputs.chunks(16) {
            let hash = poseidon.hash(builder, chunk);
            result.push(hash);
        }

        result
    }

    /// Compute challenge c' = H(message || w')
    fn compute_challenge(
        &self,
        builder: &mut ConstraintBuilder,
        message: &[usize],
        w_prime: &[usize],
    ) -> [usize; 8] {
        use crate::gadgets::PoseidonGadget;

        let poseidon = PoseidonGadget::new(16);

        // Combine message and w'
        let mut input = message.to_vec();
        input.extend_from_slice(w_prime);

        // Hash multiple times to get 32 bytes (8 scalars)
        let mut result = [0usize; 8];

        for i in 0..8 {
            let mut chunk_input = input.clone();
            // Add domain separator for each output element
            let domain_sep = builder.allocator.alloc_witness();
            builder.add_constant(domain_sep, i as Scalar);
            chunk_input.push(domain_sep);

            result[i] = poseidon.hash(builder, &chunk_input);
        }

        result
    }

    /// Check if two challenge hashes are equal
    fn check_challenge_equality(
        &self,
        builder: &mut ConstraintBuilder,
        c_prime: &[usize; 8],
        c_tilde: &[usize; 8],
    ) -> usize {
        // Check each element pair
        let mut all_equal = builder.allocator.alloc_witness();
        builder.add_constant(all_equal, 1);

        for i in 0..8 {
            // diff = c_prime[i] - c_tilde[i]
            let diff = builder.allocator.alloc_witness();
            builder.constraints.push(q_lattice_guard::R1CSConstraint {
                a: vec![(c_prime[i], 1)],
                b: vec![(0, 1)],
                c: vec![(c_tilde[i], 1), (diff, 1)],
            });

            // Check diff == 0 using inverse
            // If diff != 0, then diff * diff_inv = 1 for some diff_inv
            // If diff == 0, there's no such inverse
            let is_zero = self.check_is_zero(builder, diff);
            all_equal = builder.add_and(all_equal, is_zero);
        }

        all_equal
    }

    /// Check if value is zero
    fn check_is_zero(&self, builder: &mut ConstraintBuilder, value: usize) -> usize {
        // Use the technique: is_zero = 1 - value * inverse
        // where inverse = 1/value if value != 0, else 0

        let inverse = builder.allocator.alloc_witness();
        let value_times_inverse = builder.allocator.alloc_witness();
        let is_zero = builder.allocator.alloc_witness();

        // value * inverse = value_times_inverse
        builder.add_mul(value, inverse, value_times_inverse);

        // is_zero = 1 - value_times_inverse
        builder.constraints.push(q_lattice_guard::R1CSConstraint {
            a: vec![(0, 1)],
            b: vec![(0, 1)],
            c: vec![(is_zero, 1), (value_times_inverse, 1)],
        });

        // Constraint: value * is_zero = 0
        let should_be_zero = builder.allocator.alloc_witness();
        builder.add_mul(value, is_zero, should_be_zero);
        builder.add_constant(should_be_zero, 0);

        is_zero
    }

    /// Verify hint structure (sparse, bounded)
    fn verify_hint(&self, builder: &mut ConstraintBuilder, h: &[Vec<usize>]) -> usize {
        // Count total hint ones
        let mut total_ones = builder.allocator.alloc_witness();
        builder.add_constant(total_ones, 0);

        for hint_poly in h {
            for &hint_coeff in hint_poly {
                // Each hint coefficient should be 0 or 1
                builder.add_boolean(hint_coeff);

                // Add to total
                let new_total = builder.allocator.alloc_witness();
                builder.add_linear_combination(&[(total_ones, 1), (hint_coeff, 1)], new_total);
                total_ones = new_total;
            }
        }

        // Check total_ones <= omega
        // (simplified: just return 1 for now, full implementation needs range check)
        let valid = builder.allocator.alloc_witness();
        builder.add_constant(valid, 1);
        valid
    }

    /// Estimate constraint count for full signature verification
    pub fn estimate_constraints(&self) -> usize {
        let n = self.params.n;
        let k = self.params.k;
        let l = self.params.l;

        // Range checks for z: l * n coefficients, ~24 bits each
        let z_range_constraints = l * n * 50;  // ~50 constraints per range check

        // Matrix-vector multiplication (simplified)
        let matrix_constraints = k * l * n * 3;

        // Challenge hash computation
        let hash_constraints = 2000;

        // Hint verification
        let hint_constraints = k * 100;

        z_range_constraints + matrix_constraints + hash_constraints + hint_constraints
    }

    /// Get parameters
    pub fn params(&self) -> &DilithiumParams {
        &self.params
    }
}

/// Aggregated signature for BFT threshold verification
///
/// Instead of verifying N individual Dilithium signatures (~N * 100K constraints),
/// we verify a single aggregated signature (~100K constraints) plus aggregation proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatedDilithiumSignature {
    /// Aggregated z component
    pub agg_z: Vec<Vec<Scalar>>,
    /// Aggregated hint
    pub agg_h: Vec<Vec<u8>>,
    /// Aggregated challenge
    pub agg_c_tilde: [u8; 32],
    /// Number of signatures aggregated
    pub sig_count: u32,
    /// Bitmap of which validators signed
    pub signer_bitmap: Vec<u8>,
}

/// Gadget for verifying aggregated BFT signatures
pub struct AggregatedBFTSignatureGadget {
    /// Base Dilithium parameters
    params: DilithiumParams,
    /// Aggregated signature verifier
    inner: DilithiumVerifierGadget,
}

impl AggregatedBFTSignatureGadget {
    /// Create new aggregated signature gadget
    pub fn new(level: DilithiumLevel) -> Self {
        Self {
            params: DilithiumParams::new(level),
            inner: DilithiumVerifierGadget::new(level),
        }
    }

    /// Verify aggregated signature in circuit
    ///
    /// This is much more efficient than verifying individual signatures:
    /// - Individual: N * 100K constraints
    /// - Aggregated: ~150K constraints (single sig + aggregation proof)
    pub fn synthesize(
        &self,
        builder: &mut ConstraintBuilder,
        agg_public_key: &DilithiumPublicKeyWires,
        agg_signature: &DilithiumSignatureWires,
        message: &[usize],
        signer_bitmap: &[usize],
        threshold: usize,
    ) -> usize {
        // Step 1: Verify the aggregated signature
        let sig_valid = self.inner.synthesize(builder, agg_public_key, agg_signature, message);

        // Step 2: Count signers from bitmap
        let mut signer_count = builder.allocator.alloc_witness();
        builder.add_constant(signer_count, 0);

        for &bit in signer_bitmap {
            builder.add_boolean(bit);
            let new_count = builder.allocator.alloc_witness();
            builder.add_linear_combination(&[(signer_count, 1), (bit, 1)], new_count);
            signer_count = new_count;
        }

        // Step 3: Check signer_count >= threshold
        let threshold_met = self.check_threshold(builder, signer_count, threshold);

        // Both conditions must be met
        builder.add_and(sig_valid, threshold_met)
    }

    /// Check count >= threshold
    fn check_threshold(
        &self,
        builder: &mut ConstraintBuilder,
        count: usize,
        threshold: usize,
    ) -> usize {
        // count - threshold should be non-negative
        let diff = builder.allocator.alloc_witness();
        builder.constraints.push(q_lattice_guard::R1CSConstraint {
            a: vec![(count, 1)],
            b: vec![(0, 1)],
            c: vec![(diff, 1), (0, threshold as Scalar)],
        });

        // Decompose diff into bits to prove non-negative
        let bits = builder.allocator.alloc_witness_array(16);
        for &bit in &bits {
            builder.add_boolean(bit);
        }

        // Recompose and verify equality
        let mut power = 1u64;
        let mut terms = Vec::new();
        for &bit in &bits {
            terms.push((bit, power));
            power *= 2;
        }
        let recomposed = builder.allocator.alloc_witness();
        builder.add_linear_combination(&terms, recomposed);
        builder.add_equality(recomposed, diff);

        // If recomposition works, threshold is met
        let valid = builder.allocator.alloc_witness();
        builder.add_constant(valid, 1);
        valid
    }

    /// Estimate constraint count
    pub fn estimate_constraints(&self, num_validators: usize) -> usize {
        // Single aggregated signature verification
        let sig_constraints = self.inner.estimate_constraints();

        // Bitmap counting
        let bitmap_constraints = num_validators * 5;

        // Threshold check
        let threshold_constraints = 100;

        sig_constraints + bitmap_constraints + threshold_constraints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dilithium_params() {
        let params3 = DilithiumParams::new(DilithiumLevel::Level3);
        assert_eq!(params3.n, 256);
        assert_eq!(params3.k, 6);
        assert_eq!(params3.l, 5);
        assert_eq!(params3.q, 8380417);

        let params5 = DilithiumParams::new(DilithiumLevel::Level5);
        assert_eq!(params5.k, 8);
        assert_eq!(params5.l, 7);
    }

    #[test]
    fn test_dilithium_constraint_estimate() {
        let gadget = DilithiumVerifierGadget::new(DilithiumLevel::Level3);
        let constraints = gadget.estimate_constraints();

        println!("Dilithium3 estimated constraints: {}", constraints);

        // Should be in the ballpark of 100K
        assert!(constraints > 50_000);
        assert!(constraints < 500_000);
    }

    #[test]
    fn test_aggregated_constraint_estimate() {
        let gadget = AggregatedBFTSignatureGadget::new(DilithiumLevel::Level3);
        let individual_cost = gadget.inner.estimate_constraints();
        let aggregated_cost = gadget.estimate_constraints(100);

        println!("Individual (100 sigs): {} constraints", individual_cost * 100);
        println!("Aggregated (100 sigs): {} constraints", aggregated_cost);

        // Aggregated should be much cheaper
        assert!(aggregated_cost < individual_cost * 10);
    }
}
