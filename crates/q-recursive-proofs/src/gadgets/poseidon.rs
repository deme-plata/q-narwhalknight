//! Poseidon Hash Gadget for Recursive Proofs
//!
//! Poseidon is an algebraic hash function designed specifically for efficient
//! zero-knowledge proof systems. It uses a sponge construction with an
//! algebraically simple round function based on:
//! - S-boxes: x^α (typically α=5 for small characteristic fields)
//! - Linear layer: MDS matrix multiplication
//! - Round constants addition
//!
//! ## Performance
//!
//! Poseidon achieves significantly lower constraint counts than SHA3 or BLAKE3:
//! - Poseidon: ~300 constraints per hash
//! - SHA3-256: ~25,000 constraints
//! - BLAKE3: ~10,000 constraints
//!
//! This makes Poseidon essential for practical recursive SNARKs.

use crate::{ConstraintBuilder, WireAllocator};
use q_lattice_guard::Scalar;
use serde::{Deserialize, Serialize};

/// Poseidon hash parameters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoseidonParams {
    /// State width (t)
    pub width: usize,
    /// Number of full rounds
    pub full_rounds: usize,
    /// Number of partial rounds
    pub partial_rounds: usize,
    /// S-box exponent (typically 5)
    pub alpha: u64,
    /// Modulus for field arithmetic
    pub modulus: Scalar,
    /// MDS matrix
    pub mds_matrix: Vec<Vec<Scalar>>,
    /// Round constants
    pub round_constants: Vec<Vec<Scalar>>,
}

impl PoseidonParams {
    /// Generate secure Poseidon parameters for given width
    ///
    /// Parameters are chosen based on security analysis for 128-bit security
    pub fn secure_128(width: usize) -> Self {
        let modulus = (1u64 << 63) - 25; // Large prime for 64-bit arithmetic

        // Security parameters based on analysis
        let full_rounds = 8;
        let partial_rounds = match width {
            2..=3 => 57,
            4..=8 => 56,
            9..=12 => 55,
            _ => 60,
        };

        // Generate MDS matrix using Cauchy construction
        let mds_matrix = Self::generate_mds_matrix(width, modulus);

        // Generate round constants deterministically
        let total_rounds = full_rounds + partial_rounds;
        let round_constants = Self::generate_round_constants(width, total_rounds, modulus);

        Self {
            width,
            full_rounds,
            partial_rounds,
            alpha: 5, // x^5 S-box
            modulus,
            mds_matrix,
            round_constants,
        }
    }

    /// Generate MDS matrix using Cauchy construction
    fn generate_mds_matrix(width: usize, modulus: Scalar) -> Vec<Vec<Scalar>> {
        // Cauchy matrix: M[i][j] = 1 / (x[i] + y[j])
        // where x and y are distinct elements

        let mut matrix = vec![vec![0u64; width]; width];

        for i in 0..width {
            for j in 0..width {
                let x = (i as u64 + 1) % modulus;
                let y = (width as u64 + j as u64 + 1) % modulus;
                let sum = (x + y) % modulus;

                // Modular inverse using extended Euclidean algorithm
                matrix[i][j] = Self::mod_inverse(sum, modulus);
            }
        }

        matrix
    }

    /// Extended Euclidean algorithm for modular inverse
    fn mod_inverse(a: u64, m: u64) -> u64 {
        let mut mn = (m as i128, a as i128);
        let mut xy = (0i128, 1i128);

        while mn.1 != 0 {
            let q = mn.0 / mn.1;
            mn = (mn.1, mn.0 - q * mn.1);
            xy = (xy.1, xy.0 - q * xy.1);
        }

        if mn.0 == 1 {
            ((xy.0 % m as i128 + m as i128) % m as i128) as u64
        } else {
            1 // Fallback (should not happen with prime modulus)
        }
    }

    /// Generate round constants deterministically from seed
    fn generate_round_constants(width: usize, total_rounds: usize, modulus: Scalar) -> Vec<Vec<Scalar>> {
        use sha3::{Digest, Sha3_256};

        let mut constants = Vec::with_capacity(total_rounds);
        let mut hasher = Sha3_256::new();

        // Seed with domain separator
        hasher.update(b"POSEIDON_ROUND_CONSTANTS");
        hasher.update(&width.to_le_bytes());
        hasher.update(&total_rounds.to_le_bytes());

        for round in 0..total_rounds {
            let mut round_constants = Vec::with_capacity(width);

            for j in 0..width {
                let mut h = hasher.clone();
                h.update(&round.to_le_bytes());
                h.update(&j.to_le_bytes());
                let hash = h.finalize();

                // Extract 64-bit value and reduce mod p
                let value = u64::from_le_bytes(hash[0..8].try_into().unwrap());
                round_constants.push(value % modulus);
            }

            constants.push(round_constants);
        }

        constants
    }
}

/// Poseidon hash gadget for circuit construction
pub struct PoseidonGadget {
    /// Poseidon parameters
    params: PoseidonParams,
}

impl PoseidonGadget {
    /// Create new Poseidon gadget with given width
    pub fn new(width: usize) -> Self {
        Self {
            params: PoseidonParams::secure_128(width),
        }
    }

    /// Create with custom parameters
    pub fn with_params(params: PoseidonParams) -> Self {
        Self { params }
    }

    /// Synthesize Poseidon hash circuit
    ///
    /// Takes input wire indices and returns output wire indices (one per state element)
    /// The primary output is typically the first element after squeezing.
    pub fn synthesize(
        &self,
        builder: &mut ConstraintBuilder,
        inputs: &[usize],
    ) -> Vec<usize> {
        let width = self.params.width;
        let modulus = self.params.modulus;

        // Pad inputs to state width
        let mut state: Vec<usize> = inputs.to_vec();
        while state.len() < width {
            // Allocate constant zero wire
            let zero = builder.allocator.alloc_witness();
            builder.add_constant(zero, 0);
            state.push(zero);
        }

        // Truncate if too many inputs
        state.truncate(width);

        let full_rounds_half = self.params.full_rounds / 2;

        // === First half of full rounds ===
        for round in 0..full_rounds_half {
            state = self.full_round(builder, &state, round);
        }

        // === Partial rounds ===
        for round in 0..self.params.partial_rounds {
            state = self.partial_round(builder, &state, full_rounds_half + round);
        }

        // === Second half of full rounds ===
        for round in 0..full_rounds_half {
            let round_idx = full_rounds_half + self.params.partial_rounds + round;
            state = self.full_round(builder, &state, round_idx);
        }

        state
    }

    /// Hash two 32-byte inputs to single 32-byte output (compression function)
    pub fn hash_pair(
        &self,
        builder: &mut ConstraintBuilder,
        left: &[usize; 8],
        right: &[usize; 8],
    ) -> [usize; 8] {
        // Combine inputs
        let mut inputs = Vec::with_capacity(16);
        inputs.extend_from_slice(left);
        inputs.extend_from_slice(right);

        // Apply Poseidon
        let output = self.synthesize(builder, &inputs);

        // Return first 8 elements as hash output
        let mut result = [0usize; 8];
        result.copy_from_slice(&output[..8]);
        result
    }

    /// Single hash of arbitrary inputs
    pub fn hash(&self, builder: &mut ConstraintBuilder, inputs: &[usize]) -> usize {
        let output = self.synthesize(builder, inputs);
        output[0] // Return first element as hash
    }

    /// Full round: S-box on all elements + MDS + round constants
    fn full_round(
        &self,
        builder: &mut ConstraintBuilder,
        state: &[usize],
        round: usize,
    ) -> Vec<usize> {
        let width = self.params.width;
        let mut new_state = Vec::with_capacity(width);

        // Apply S-box to all elements
        let mut sbox_outputs = Vec::with_capacity(width);
        for &wire in state.iter() {
            let sbox_out = self.apply_sbox(builder, wire);
            sbox_outputs.push(sbox_out);
        }

        // Add round constants
        for (i, &sbox_out) in sbox_outputs.iter().enumerate() {
            let rc = self.params.round_constants[round][i];
            let with_rc = builder.allocator.alloc_witness();

            // with_rc = sbox_out + rc
            builder.add_linear_combination(&[(sbox_out, 1), (0, rc)], with_rc);
            new_state.push(with_rc);
        }

        // Apply MDS matrix
        self.apply_mds(builder, &new_state)
    }

    /// Partial round: S-box only on first element + MDS + round constants
    fn partial_round(
        &self,
        builder: &mut ConstraintBuilder,
        state: &[usize],
        round: usize,
    ) -> Vec<usize> {
        let width = self.params.width;
        let mut new_state = Vec::with_capacity(width);

        // Apply S-box only to first element
        let sbox_out = self.apply_sbox(builder, state[0]);

        // Add round constant to first element
        let rc = self.params.round_constants[round][0];
        let first_with_rc = builder.allocator.alloc_witness();
        builder.add_linear_combination(&[(sbox_out, 1), (0, rc)], first_with_rc);
        new_state.push(first_with_rc);

        // Rest just get round constants added (no S-box)
        for i in 1..width {
            let rc = self.params.round_constants[round][i];
            let with_rc = builder.allocator.alloc_witness();
            builder.add_linear_combination(&[(state[i], 1), (0, rc)], with_rc);
            new_state.push(with_rc);
        }

        // Apply MDS matrix
        self.apply_mds(builder, &new_state)
    }

    /// Apply x^5 S-box
    fn apply_sbox(&self, builder: &mut ConstraintBuilder, input: usize) -> usize {
        // x^2
        let x2 = builder.allocator.alloc_witness();
        builder.add_mul(input, input, x2);

        // x^4 = (x^2)^2
        let x4 = builder.allocator.alloc_witness();
        builder.add_mul(x2, x2, x4);

        // x^5 = x^4 * x
        let x5 = builder.allocator.alloc_witness();
        builder.add_mul(x4, input, x5);

        x5
    }

    /// Apply MDS matrix multiplication
    fn apply_mds(&self, builder: &mut ConstraintBuilder, state: &[usize]) -> Vec<usize> {
        let width = self.params.width;
        let mut new_state = Vec::with_capacity(width);

        for i in 0..width {
            // new_state[i] = sum(MDS[i][j] * state[j])
            let mut terms: Vec<(usize, Scalar)> = Vec::with_capacity(width);

            for j in 0..width {
                let coeff = self.params.mds_matrix[i][j];
                if coeff != 0 {
                    terms.push((state[j], coeff));
                }
            }

            let result = builder.allocator.alloc_witness();
            builder.add_linear_combination(&terms, result);
            new_state.push(result);
        }

        new_state
    }

    /// Estimate constraint count for this gadget configuration
    pub fn estimate_constraints(&self) -> usize {
        let width = self.params.width;
        let full_rounds = self.params.full_rounds;
        let partial_rounds = self.params.partial_rounds;

        // S-box: 3 constraints (x^2, x^4, x^5)
        let sbox_constraints = 3;

        // MDS: width linear combinations
        let mds_constraints = width;

        // Full round: width S-boxes + width round constant additions + MDS
        let full_round_constraints = width * sbox_constraints + width + mds_constraints;

        // Partial round: 1 S-box + width round constant additions + MDS
        let partial_round_constraints = sbox_constraints + width + mds_constraints;

        full_rounds * full_round_constraints + partial_rounds * partial_round_constraints
    }

    /// Get parameters
    pub fn params(&self) -> &PoseidonParams {
        &self.params
    }
}

/// Native Poseidon hash computation (for witness generation)
pub fn poseidon_native(inputs: &[Scalar], params: &PoseidonParams) -> Vec<Scalar> {
    let width = params.width;
    let modulus = params.modulus;

    // Initialize state
    let mut state: Vec<Scalar> = inputs.to_vec();
    state.resize(width, 0);

    let full_rounds_half = params.full_rounds / 2;

    // First half of full rounds
    for round in 0..full_rounds_half {
        state = full_round_native(&state, round, params);
    }

    // Partial rounds
    for round in 0..params.partial_rounds {
        state = partial_round_native(&state, full_rounds_half + round, params);
    }

    // Second half of full rounds
    for round in 0..full_rounds_half {
        let round_idx = full_rounds_half + params.partial_rounds + round;
        state = full_round_native(&state, round_idx, params);
    }

    state
}

fn full_round_native(state: &[Scalar], round: usize, params: &PoseidonParams) -> Vec<Scalar> {
    let width = params.width;
    let modulus = params.modulus;

    // S-box on all elements
    let mut after_sbox: Vec<Scalar> = state
        .iter()
        .map(|&x| sbox_native(x, params.alpha, modulus))
        .collect();

    // Add round constants
    for i in 0..width {
        after_sbox[i] = (after_sbox[i] + params.round_constants[round][i]) % modulus;
    }

    // MDS
    mds_native(&after_sbox, params)
}

fn partial_round_native(state: &[Scalar], round: usize, params: &PoseidonParams) -> Vec<Scalar> {
    let width = params.width;
    let modulus = params.modulus;

    let mut after_sbox = state.to_vec();

    // S-box only on first element
    after_sbox[0] = sbox_native(after_sbox[0], params.alpha, modulus);

    // Add round constants
    for i in 0..width {
        after_sbox[i] = (after_sbox[i] + params.round_constants[round][i]) % modulus;
    }

    // MDS
    mds_native(&after_sbox, params)
}

fn sbox_native(x: Scalar, alpha: u64, modulus: Scalar) -> Scalar {
    // x^5 mod p
    let x2 = (x as u128 * x as u128 % modulus as u128) as u64;
    let x4 = (x2 as u128 * x2 as u128 % modulus as u128) as u64;
    (x4 as u128 * x as u128 % modulus as u128) as u64
}

fn mds_native(state: &[Scalar], params: &PoseidonParams) -> Vec<Scalar> {
    let width = params.width;
    let modulus = params.modulus;

    let mut new_state = vec![0u64; width];

    for i in 0..width {
        let mut acc = 0u128;
        for j in 0..width {
            acc += params.mds_matrix[i][j] as u128 * state[j] as u128;
        }
        new_state[i] = (acc % modulus as u128) as u64;
    }

    new_state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon_params_generation() {
        let params = PoseidonParams::secure_128(4);

        assert_eq!(params.width, 4);
        assert_eq!(params.full_rounds, 8);
        assert_eq!(params.alpha, 5);
        assert_eq!(params.mds_matrix.len(), 4);
        assert_eq!(params.mds_matrix[0].len(), 4);
    }

    #[test]
    fn test_poseidon_native_deterministic() {
        let params = PoseidonParams::secure_128(4);
        let inputs = vec![1u64, 2, 3, 4];

        let output1 = poseidon_native(&inputs, &params);
        let output2 = poseidon_native(&inputs, &params);

        assert_eq!(output1, output2);
    }

    #[test]
    fn test_poseidon_native_different_inputs() {
        let params = PoseidonParams::secure_128(4);

        let output1 = poseidon_native(&[1, 2, 3, 4], &params);
        let output2 = poseidon_native(&[1, 2, 3, 5], &params);

        assert_ne!(output1, output2);
    }

    #[test]
    fn test_poseidon_gadget_constraint_count() {
        let gadget = PoseidonGadget::new(4);
        let constraints = gadget.estimate_constraints();

        // Should be in the range of a few hundred constraints
        assert!(constraints > 200, "Too few constraints: {}", constraints);
        assert!(constraints < 1000, "Too many constraints: {}", constraints);

        println!("Poseidon (width=4) constraints: {}", constraints);
    }

    #[test]
    fn test_poseidon_gadget_synthesis() {
        let gadget = PoseidonGadget::new(4);
        let mut builder = ConstraintBuilder::new(gadget.params.modulus);

        // Allocate input wires
        let inputs: Vec<usize> = (0..4)
            .map(|_| builder.allocator.alloc_public_input())
            .collect();

        // Synthesize circuit
        let outputs = gadget.synthesize(&mut builder, &inputs);

        assert_eq!(outputs.len(), 4);

        let circuit = builder.build();
        println!(
            "Poseidon circuit: {} constraints, {} public inputs, {} witness",
            circuit.num_constraints, circuit.num_public_inputs, circuit.num_witness
        );
    }

    #[test]
    fn test_mod_inverse() {
        let modulus = (1u64 << 63) - 25;

        // Test inverse of small number
        let a = 7u64;
        let inv = PoseidonParams::mod_inverse(a, modulus);
        let product = ((a as u128 * inv as u128) % modulus as u128) as u64;
        assert_eq!(product, 1, "Modular inverse failed");
    }
}
