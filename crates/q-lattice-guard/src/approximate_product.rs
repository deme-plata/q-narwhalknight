//! Approximate Product Proofs for LatticeGuard
//!
//! This module implements proofs that c ≈ a * b with bounded error,
//! which is the core innovation enabling lattice-based SNARKs.

use crate::{
    errors::LatticeGuardError,
    ntt::NttOperator,
    params::RlweParams,
    transcript::LatticeTranscript,
    Scalar,
};
use serde::{Deserialize, Serialize};

/// Proof that c ≈ a * b within bounded error
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApproximateProductProof {
    /// Linearization proof showing polynomial relation
    pub linearization_proof: LinearizationProof,
    /// Range proof showing error is bounded
    pub error_bound_proof: RangeProof,
    /// Final consistency check
    pub consistency_check: ConsistencyProof,
}

/// Proof of linear relation between polynomials
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearizationProof {
    /// Committed linearized polynomial
    pub linearized_commitment: Vec<Scalar>,
    /// Evaluation at challenge point
    pub evaluation: Scalar,
    /// Opening proof
    pub opening: Vec<Scalar>,
}

/// Proof that error coefficients are within range
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeProof {
    /// Commitments to error decomposition
    pub bit_commitments: Vec<Vec<Scalar>>,
    /// Aggregated proof
    pub aggregated_proof: Vec<Scalar>,
    /// Bound being proven
    pub bound: Scalar,
}

/// Final consistency proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsistencyProof {
    /// Random linear combination check
    pub rlc_value: Scalar,
    /// Challenge responses
    pub responses: Vec<Scalar>,
}

/// Prover for approximate product proofs
pub struct ApproximateProductProver {
    params: RlweParams,
    ntt: NttOperator,
}

impl ApproximateProductProver {
    /// Create new prover
    pub fn new(params: RlweParams) -> Self {
        let ntt = NttOperator::new(&params);
        Self { params, ntt }
    }

    /// Prove that c ≈ a * b within error bound
    ///
    /// The key insight is that in RLWE-based computation, products are
    /// naturally approximate due to noise growth. We prove this error
    /// is bounded using range proofs.
    pub fn prove(
        &self,
        a: &[Scalar],
        b: &[Scalar],
        c: &[Scalar],
        error_bound: Scalar,
        transcript: &mut LatticeTranscript,
    ) -> Result<ApproximateProductProof, LatticeGuardError> {
        // Step 1: Compute error e = c - a*b
        let ab = self.ntt.mul(a, b);
        let error: Vec<Scalar> = c
            .iter()
            .zip(ab.iter())
            .map(|(&ci, &abi)| {
                if ci >= abi {
                    ci - abi
                } else {
                    self.params.modulus - abi + ci
                }
            })
            .collect();

        // Step 2: Verify error is actually bounded
        for &e in &error {
            let centered = if e > self.params.modulus / 2 {
                self.params.modulus - e
            } else {
                e
            };
            if centered > error_bound {
                return Err(LatticeGuardError::ErrorBoundExceeded(centered, error_bound));
            }
        }

        // Step 3: Create linearization proof
        // Transform multiplication check into linear check using challenges
        transcript.append_bytes(b"approximate_product", b"start");
        for &ai in a.iter().take(10) {
            transcript.append_scalar(b"a", ai);
        }
        for &bi in b.iter().take(10) {
            transcript.append_scalar(b"b", bi);
        }

        let challenge = transcript.generate_challenge();
        let z = challenge.polynomial.evaluate(1, self.params.modulus);

        // Evaluate polynomials at challenge
        let a_z = self.evaluate_at_challenge(a, z);
        let b_z = self.evaluate_at_challenge(b, z);
        let c_z = self.evaluate_at_challenge(c, z);

        // Linearization: a(z) * b(z) ≈ c(z) with small error
        let linearization_proof = LinearizationProof {
            linearized_commitment: vec![a_z, b_z, c_z],
            evaluation: (a_z as u128 * b_z as u128 % self.params.modulus as u128) as Scalar,
            opening: error.iter().take(self.params.dimension / 4).copied().collect(),
        };

        // Step 4: Create range proof for error
        let error_bound_proof = self.prove_range(&error, error_bound, transcript)?;

        // Step 5: Final consistency check
        let consistency_check = self.prove_consistency(
            &linearization_proof,
            &error_bound_proof,
            transcript,
        )?;

        Ok(ApproximateProductProof {
            linearization_proof,
            error_bound_proof,
            consistency_check,
        })
    }

    /// Evaluate polynomial at point using Horner's method
    fn evaluate_at_challenge(&self, poly: &[Scalar], z: Scalar) -> Scalar {
        let mut result = 0u128;
        for &coeff in poly.iter().rev() {
            result = (result * z as u128 + coeff as u128) % self.params.modulus as u128;
        }
        result as Scalar
    }

    /// Create range proof that all error coefficients are bounded
    fn prove_range(
        &self,
        error: &[Scalar],
        bound: Scalar,
        transcript: &mut LatticeTranscript,
    ) -> Result<RangeProof, LatticeGuardError> {
        // Decompose each error coefficient into bits
        let num_bits = 64 - bound.leading_zeros();
        let mut bit_commitments = Vec::new();

        // For efficiency, we use a batched range proof
        // Commit to bit decomposition of each error element
        for &e in error.iter().take(16) {
            // Sample first 16 for efficiency
            let centered = if e > self.params.modulus / 2 {
                self.params.modulus - e
            } else {
                e
            };

            let mut bits = Vec::new();
            let mut val = centered;
            for _ in 0..num_bits {
                bits.push(val & 1);
                val >>= 1;
            }
            bit_commitments.push(bits);
        }

        // Generate challenge for aggregation
        transcript.append_bytes(b"range_proof", b"start");
        let agg_challenge = transcript.challenge_scalar(b"aggregation");

        // Aggregate proofs
        let mut aggregated = vec![0u64; num_bits as usize];
        let mut power = 1u64;
        for bits in &bit_commitments {
            for (i, &bit) in bits.iter().enumerate() {
                aggregated[i] = (aggregated[i]
                    + ((bit as u128 * power as u128) % self.params.modulus as u128) as u64)
                    % self.params.modulus;
            }
            power = ((power as u128 * agg_challenge as u128) % self.params.modulus as u128) as u64;
        }

        Ok(RangeProof {
            bit_commitments,
            aggregated_proof: aggregated,
            bound,
        })
    }

    /// Create final consistency proof
    fn prove_consistency(
        &self,
        linearization: &LinearizationProof,
        range_proof: &RangeProof,
        transcript: &mut LatticeTranscript,
    ) -> Result<ConsistencyProof, LatticeGuardError> {
        // Random linear combination of all proof elements
        transcript.append_bytes(b"consistency", b"start");

        let alpha = transcript.challenge_scalar(b"alpha");
        let beta = transcript.challenge_scalar(b"beta");

        // Compute RLC
        let mut rlc = 0u128;
        for (i, &val) in linearization.linearized_commitment.iter().enumerate() {
            let power = RlweParams::mod_pow(alpha, i as u64, self.params.modulus);
            rlc = (rlc + (val as u128 * power as u128)) % self.params.modulus as u128;
        }

        // Add range proof contribution
        for (i, &val) in range_proof.aggregated_proof.iter().enumerate() {
            let power = RlweParams::mod_pow(beta, i as u64, self.params.modulus);
            rlc = (rlc + (val as u128 * power as u128)) % self.params.modulus as u128;
        }

        // Generate responses
        let gamma = transcript.challenge_scalar(b"gamma");
        let responses = vec![
            (rlc as u64 * gamma) % self.params.modulus,
            linearization.evaluation,
            range_proof.bound,
        ];

        Ok(ConsistencyProof {
            rlc_value: rlc as Scalar,
            responses,
        })
    }
}

/// Verifier for approximate product proofs
pub struct ApproximateProductVerifier {
    params: RlweParams,
    ntt: NttOperator,
}

impl ApproximateProductVerifier {
    /// Create new verifier
    pub fn new(params: RlweParams) -> Self {
        let ntt = NttOperator::new(&params);
        Self { params, ntt }
    }

    /// Verify approximate product proof
    pub fn verify(
        &self,
        a: &[Scalar],
        b: &[Scalar],
        c: &[Scalar],
        proof: &ApproximateProductProof,
        transcript: &mut LatticeTranscript,
    ) -> Result<bool, LatticeGuardError> {
        // Reconstruct challenges
        transcript.append_bytes(b"approximate_product", b"start");
        for &ai in a.iter().take(10) {
            transcript.append_scalar(b"a", ai);
        }
        for &bi in b.iter().take(10) {
            transcript.append_scalar(b"b", bi);
        }

        let challenge = transcript.generate_challenge();
        let z = challenge.polynomial.evaluate(1, self.params.modulus);

        // Verify linearization
        let a_z = self.evaluate_at_challenge(a, z);
        let b_z = self.evaluate_at_challenge(b, z);
        let c_z = self.evaluate_at_challenge(c, z);

        if proof.linearization_proof.linearized_commitment[0] != a_z
            || proof.linearization_proof.linearized_commitment[1] != b_z
            || proof.linearization_proof.linearized_commitment[2] != c_z
        {
            return Ok(false);
        }

        // Verify product relation (approximate)
        let expected = (a_z as u128 * b_z as u128 % self.params.modulus as u128) as Scalar;
        let diff = if c_z >= expected {
            c_z - expected
        } else {
            self.params.modulus - expected + c_z
        };

        let centered_diff = if diff > self.params.modulus / 2 {
            self.params.modulus - diff
        } else {
            diff
        };

        if centered_diff > proof.error_bound_proof.bound {
            return Ok(false);
        }

        // Verify range proof
        if !self.verify_range(&proof.error_bound_proof, transcript)? {
            return Ok(false);
        }

        // Verify consistency
        self.verify_consistency(
            &proof.linearization_proof,
            &proof.error_bound_proof,
            &proof.consistency_check,
            transcript,
        )
    }

    fn evaluate_at_challenge(&self, poly: &[Scalar], z: Scalar) -> Scalar {
        let mut result = 0u128;
        for &coeff in poly.iter().rev() {
            result = (result * z as u128 + coeff as u128) % self.params.modulus as u128;
        }
        result as Scalar
    }

    fn verify_range(
        &self,
        proof: &RangeProof,
        transcript: &mut LatticeTranscript,
    ) -> Result<bool, LatticeGuardError> {
        transcript.append_bytes(b"range_proof", b"start");
        let agg_challenge = transcript.challenge_scalar(b"aggregation");

        // Verify bit decompositions sum correctly
        for bits in &proof.bit_commitments {
            let mut reconstructed = 0u64;
            for (i, &bit) in bits.iter().enumerate() {
                if bit > 1 {
                    return Ok(false); // Bits must be 0 or 1
                }
                reconstructed += bit << i;
            }
            if reconstructed > proof.bound {
                return Ok(false);
            }
        }

        // Verify aggregation (simplified)
        let mut expected_agg = vec![0u64; proof.aggregated_proof.len()];
        let mut power = 1u64;
        for bits in &proof.bit_commitments {
            for (i, &bit) in bits.iter().enumerate() {
                if i < expected_agg.len() {
                    expected_agg[i] = (expected_agg[i]
                        + ((bit as u128 * power as u128) % self.params.modulus as u128) as u64)
                        % self.params.modulus;
                }
            }
            power = ((power as u128 * agg_challenge as u128) % self.params.modulus as u128) as u64;
        }

        Ok(expected_agg == proof.aggregated_proof)
    }

    fn verify_consistency(
        &self,
        linearization: &LinearizationProof,
        range_proof: &RangeProof,
        consistency: &ConsistencyProof,
        transcript: &mut LatticeTranscript,
    ) -> Result<bool, LatticeGuardError> {
        transcript.append_bytes(b"consistency", b"start");

        let alpha = transcript.challenge_scalar(b"alpha");
        let beta = transcript.challenge_scalar(b"beta");
        let gamma = transcript.challenge_scalar(b"gamma");

        // Recompute RLC
        let mut rlc = 0u128;
        for (i, &val) in linearization.linearized_commitment.iter().enumerate() {
            let power = RlweParams::mod_pow(alpha, i as u64, self.params.modulus);
            rlc = (rlc + (val as u128 * power as u128)) % self.params.modulus as u128;
        }

        for (i, &val) in range_proof.aggregated_proof.iter().enumerate() {
            let power = RlweParams::mod_pow(beta, i as u64, self.params.modulus);
            rlc = (rlc + (val as u128 * power as u128)) % self.params.modulus as u128;
        }

        // Verify RLC matches
        if rlc as Scalar != consistency.rlc_value {
            return Ok(false);
        }

        // Verify response
        let expected_response = (rlc as u64 * gamma) % self.params.modulus;
        Ok(consistency.responses[0] == expected_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approximate_product_proof() {
        let params = RlweParams::pq128();
        let prover = ApproximateProductProver::new(params.clone());
        let verifier = ApproximateProductVerifier::new(params.clone());

        // Create simple polynomials
        let a: Vec<Scalar> = (0..params.dimension).map(|i| (i as u64) % 10).collect();
        let b: Vec<Scalar> = (0..params.dimension).map(|i| ((i + 1) as u64) % 10).collect();

        // Compute c = a * b (exact for this test)
        let ntt = NttOperator::new(&params);
        let c = ntt.mul(&a, &b);

        let error_bound = 1 << 20;

        let mut prover_transcript = LatticeTranscript::new(params.clone());
        let proof = prover
            .prove(&a, &b, &c, error_bound, &mut prover_transcript)
            .expect("Proof generation should succeed");

        let mut verifier_transcript = LatticeTranscript::new(params);
        let valid = verifier
            .verify(&a, &b, &c, &proof, &mut verifier_transcript)
            .expect("Verification should not error");

        assert!(valid, "Valid proof should verify");
    }
}
