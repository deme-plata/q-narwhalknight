//! Poseidon hash gadget — real R1CS implementation over BLS12-381 Fr.
//!
//! Replaces the placeholder (addition fold) with the actual Poseidon permutation:
//!   AddRoundConstants → SubWords (x^5 S-box) → MixLayer (MDS matrix)
//!
//! Parameters: t=3, α=5, full_rounds=8, partial_rounds=57 (128-bit on BLS12-381).
//! Round constants derived deterministically from SHA3-256 with domain separator
//! "POSEIDON_BLS12381_RC_T3_V1". MDS matrix is a Cauchy construction.
//!
//! S-box (x → x^5): 3 multiplications per element.
//!   x² = x·x     (1 R1CS constraint)
//!   x⁴ = x²·x²   (1 R1CS constraint)
//!   x⁵ = x⁴·x    (1 R1CS constraint)
//!
//! Constraint estimate for t=3, 8+57 rounds:
//!   Full rounds  (8):  3 elements × 3 S-box = 9 mul constraints each → 72 total
//!   Partial rounds (57): 1 element × 3 S-box = 3 mul constraints each → 171 total
//!   MDS and AddRoundConstants: pure linear combinations → 0 mul constraints
//!   Total: ~243 constraints per permutation (negligible vs. 50K for BLAKE3)
//!
//! IMPORTANT: The LatticeGuard prover's Fiat-Shamir transcript MUST use the
//! same Poseidon parameters (same domain separator, same t, same rounds).
//! Any mismatch causes proof verification to always fail.

use ark_ff::PrimeField;
use ark_r1cs_std::{
    fields::fp::FpVar,
    prelude::*,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};
use sha3::{Digest, Sha3_256};

/// State width t=3: capacity=1, rate=2 (absorb 2 field elements per permutation).
pub const POSEIDON_T: usize = 3;
/// Full rounds (applied at start and end of permutation).
pub const POSEIDON_FULL_ROUNDS: usize = 8;
/// Partial rounds (middle section, S-box applied to first element only).
pub const POSEIDON_PARTIAL_ROUNDS: usize = 57;

/// In-circuit Poseidon hash gadget over BLS12-381 Fr.
///
/// All arithmetic is performed on `FpVar<F>` which automatically generates
/// R1CS constraints. Linear operations (MDS, AddRoundConstants) are free;
/// only the S-box multiplications add constraints.
pub struct PoseidonGadget;

impl PoseidonGadget {
    /// Hash 1–2 field elements to a single field element.
    ///
    /// Absorbs inputs at rate positions (state[1..]), runs permutation,
    /// returns state[1]. ~243 R1CS constraints.
    pub fn hash<F: PrimeField>(
        _cs: ConstraintSystemRef<F>,
        inputs: &[FpVar<F>],
    ) -> Result<FpVar<F>, SynthesisError> {
        assert!(!inputs.is_empty(), "Poseidon requires at least one input");
        assert!(
            inputs.len() <= POSEIDON_T - 1,
            "use hash_many for more than {} inputs",
            POSEIDON_T - 1
        );

        let mds = Self::mds_matrix::<F>();
        let rc = Self::round_constants::<F>();

        // Capacity at index 0 (domain separator), rate at indices 1..t.
        let mut state = vec![FpVar::Constant(F::zero()); POSEIDON_T];
        for (i, inp) in inputs.iter().enumerate() {
            state[i + 1] = inp.clone();
        }

        state = Self::permutation(state, &mds, &rc)?;
        Ok(state[1].clone())
    }

    /// Hash two field elements (common case: Merkle node, state root pair).
    pub fn hash2<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        a: &FpVar<F>,
        b: &FpVar<F>,
    ) -> Result<FpVar<F>, SynthesisError> {
        Self::hash::<F>(cs, &[a.clone(), b.clone()])
    }

    /// Hash a sequence of field elements using the sponge construction.
    ///
    /// Processes inputs in blocks of `rate = t-1 = 2`. Each block XOR-absorbs
    /// into the rate portion of the state, then runs the full permutation.
    /// Returns the first rate element after the final permutation.
    pub fn hash_many<F: PrimeField>(
        _cs: ConstraintSystemRef<F>,
        inputs: &[FpVar<F>],
    ) -> Result<FpVar<F>, SynthesisError> {
        if inputs.is_empty() {
            return Ok(FpVar::Constant(F::zero()));
        }

        let rate = POSEIDON_T - 1; // 2
        let mds = Self::mds_matrix::<F>();
        let rc = Self::round_constants::<F>();

        // Initial state: all zero (capacity = 0 acts as domain separator)
        let mut state = vec![FpVar::Constant(F::zero()); POSEIDON_T];

        for chunk in inputs.chunks(rate) {
            // XOR-absorb: state[1+i] += chunk[i]
            for (i, inp) in chunk.iter().enumerate() {
                state[i + 1] = state[i + 1].clone() + inp.clone();
            }
            state = Self::permutation(state, &mds, &rc)?;
        }

        Ok(state[1].clone())
    }

    // ── Internal permutation ────────────────────────────────────────────────

    fn permutation<F: PrimeField>(
        mut state: Vec<FpVar<F>>,
        mds: &[Vec<F>],
        rc: &[Vec<F>],
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let half_f = POSEIDON_FULL_ROUNDS / 2;
        let mut round = 0;

        for _ in 0..half_f {
            state = Self::full_round(state, &rc[round], mds)?;
            round += 1;
        }
        for _ in 0..POSEIDON_PARTIAL_ROUNDS {
            state = Self::partial_round(state, &rc[round], mds)?;
            round += 1;
        }
        for _ in 0..half_f {
            state = Self::full_round(state, &rc[round], mds)?;
            round += 1;
        }

        Ok(state)
    }

    /// Full round: AddRoundConstants → x^5 S-box on all t elements → MDS.
    fn full_round<F: PrimeField>(
        state: Vec<FpVar<F>>,
        constants: &[F],
        mds: &[Vec<F>],
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let mut after_sbox = Vec::with_capacity(state.len());
        for (x, &c) in state.iter().zip(constants.iter()) {
            let x_c = x.clone() + FpVar::Constant(c);
            after_sbox.push(Self::sbox_x5(&x_c));
        }
        Self::mds_mul(&after_sbox, mds)
    }

    /// Partial round: AddRoundConstants → x^5 S-box on state[0] only → MDS.
    fn partial_round<F: PrimeField>(
        state: Vec<FpVar<F>>,
        constants: &[F],
        mds: &[Vec<F>],
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let mut after_rc: Vec<FpVar<F>> = state
            .iter()
            .zip(constants.iter())
            .map(|(x, &c)| x.clone() + FpVar::Constant(c))
            .collect();
        // S-box only on first element
        let first = Self::sbox_x5(&after_rc[0].clone());
        after_rc[0] = first;
        Self::mds_mul(&after_rc, mds)
    }

    /// x^5 S-box: 3 multiplication gates (for witness FpVars; free for constants).
    #[inline]
    fn sbox_x5<F: PrimeField>(x: &FpVar<F>) -> FpVar<F> {
        let x2 = x.clone() * x.clone();       // constraint 1
        let x4 = x2.clone() * x2.clone();     // constraint 2
        x4 * x.clone()                         // constraint 3
    }

    /// MDS matrix multiply (linear combinations, zero mul constraints).
    fn mds_mul<F: PrimeField>(
        state: &[FpVar<F>],
        mds: &[Vec<F>],
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        let t = state.len();
        let mut new_state = Vec::with_capacity(t);

        for i in 0..t {
            let mut acc = FpVar::Constant(F::zero());
            for j in 0..t {
                // FpVar * FpVar::Constant is a linear scaling, not a multiplication gate
                let term = state[j].clone() * FpVar::Constant(mds[i][j]);
                acc = acc + term;
            }
            new_state.push(acc);
        }

        Ok(new_state)
    }

    // ── Parameter generation ───────────────────────────────────────────────

    /// Cauchy MDS matrix over field F: M[i][j] = 1 / (x_i + y_j).
    ///
    /// x_i = i+1, y_j = t+j+1 — all distinct, all nonzero sums guaranteed.
    /// Cauchy matrices are Maximum Distance Separable (MDS), meeting the
    /// security requirement for Poseidon's linear layer.
    fn mds_matrix<F: PrimeField>() -> Vec<Vec<F>> {
        let t = POSEIDON_T;
        let mut matrix = vec![vec![F::zero(); t]; t];

        for i in 0..t {
            for j in 0..t {
                let xi = F::from((i + 1) as u64);
                let yj = F::from((t + j + 1) as u64);
                let sum = xi + yj;
                matrix[i][j] = sum
                    .inverse()
                    .expect("Cauchy MDS: sum xi+yj is zero — parameter conflict");
            }
        }

        matrix
    }

    /// SHA3-derived round constants, reduced mod |F|.
    ///
    /// Constant[round][pos] = SHA3("POSEIDON_BLS12381_RC_T3_V1" || round || pos)
    /// The domain separator "V1" must be changed if t or round counts change,
    /// to avoid cross-parameter collisions.
    fn round_constants<F: PrimeField>() -> Vec<Vec<F>> {
        let total = POSEIDON_FULL_ROUNDS + POSEIDON_PARTIAL_ROUNDS;
        let t = POSEIDON_T;
        let mut out = Vec::with_capacity(total);

        for round in 0u32..total as u32 {
            let mut row = Vec::with_capacity(t);
            for pos in 0u32..t as u32 {
                let mut h = Sha3_256::new();
                h.update(b"POSEIDON_BLS12381_RC_T3_V1");
                h.update(round.to_le_bytes());
                h.update(pos.to_le_bytes());
                let digest = h.finalize();
                row.push(F::from_le_bytes_mod_order(&digest));
            }
            out.push(row);
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    #[test]
    fn test_poseidon_gadget_compiles() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let a = FpVar::new_witness(cs.clone(), || Ok(Fr::from(42u64))).unwrap();
        let b = FpVar::new_witness(cs.clone(), || Ok(Fr::from(13u64))).unwrap();
        let _hash = PoseidonGadget::hash2(cs.clone(), &a, &b).unwrap();
        let satisfied = cs.is_satisfied().unwrap();
        let num_constraints = cs.num_constraints();
        println!(
            "Poseidon t=3 hash2 constraints: {} (expected ~243)",
            num_constraints
        );
        assert!(satisfied, "Poseidon circuit is unsatisfied");
        // With real S-boxes: 8*9 + 57*3 = 72+171 = 243 mul constraints
        assert!(
            num_constraints >= 240,
            "Too few constraints ({}): S-box not implemented?",
            num_constraints
        );
    }

    #[test]
    fn test_poseidon_hash_many() {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let inputs: Vec<_> = (0..5)
            .map(|i| FpVar::new_witness(cs.clone(), || Ok(Fr::from(i as u64))).unwrap())
            .collect();
        let _hash = PoseidonGadget::hash_many(cs.clone(), &inputs).unwrap();
        assert!(cs.is_satisfied().unwrap(), "hash_many circuit unsatisfied");
    }

    #[test]
    fn test_poseidon_deterministic() {
        // Native check: same inputs → same output
        let cs1 = ConstraintSystem::<Fr>::new_ref();
        let cs2 = ConstraintSystem::<Fr>::new_ref();

        let a1 = FpVar::new_witness(cs1.clone(), || Ok(Fr::from(7u64))).unwrap();
        let b1 = FpVar::new_witness(cs1.clone(), || Ok(Fr::from(8u64))).unwrap();
        let h1 = PoseidonGadget::hash2(cs1, &a1, &b1).unwrap();

        let a2 = FpVar::new_witness(cs2.clone(), || Ok(Fr::from(7u64))).unwrap();
        let b2 = FpVar::new_witness(cs2.clone(), || Ok(Fr::from(8u64))).unwrap();
        let h2 = PoseidonGadget::hash2(cs2, &a2, &b2).unwrap();

        assert_eq!(
            h1.value().unwrap(),
            h2.value().unwrap(),
            "Poseidon not deterministic"
        );
    }

    #[test]
    fn test_poseidon_distinct_inputs_different_outputs() {
        let cs = ConstraintSystem::<Fr>::new_ref();

        let a = FpVar::new_witness(cs.clone(), || Ok(Fr::from(1u64))).unwrap();
        let b = FpVar::new_witness(cs.clone(), || Ok(Fr::from(2u64))).unwrap();
        let c = FpVar::new_witness(cs.clone(), || Ok(Fr::from(3u64))).unwrap();

        let h_ab = PoseidonGadget::hash2(cs.clone(), &a, &b).unwrap();
        let h_ac = PoseidonGadget::hash2(cs.clone(), &a, &c).unwrap();

        assert_ne!(
            h_ab.value().unwrap(),
            h_ac.value().unwrap(),
            "Poseidon collision: different inputs gave same output"
        );
    }
}
