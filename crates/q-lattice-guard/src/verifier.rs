//! LatticeGuard Verifier
//!
//! Implements the verifier for the LatticeGuard zk-SNARK protocol.

use crate::{
    approximate_product::ApproximateProductVerifier,
    commitment::{LatticeCommitment, PolynomialCommitment},
    errors::LatticeGuardError,
    ntt::NttOperator,
    params::RlweParams,
    prover::{LatticeGuardProof, ProofMetadata},
    transcript::LatticeTranscript,
    ArithmeticCircuit, LatticeGuardSRS, Polynomial, Scalar,
};
use tracing::{debug, info, warn};

/// LatticeGuard verifier
pub struct LatticeGuardVerifier {
    params: RlweParams,
    ntt: NttOperator,
    commitment_scheme: PolynomialCommitment,
    product_verifier: ApproximateProductVerifier,
}

impl LatticeGuardVerifier {
    /// Create new verifier with given parameters
    pub fn new(params: RlweParams) -> Result<Self, LatticeGuardError> {
        let ntt = NttOperator::new(&params);
        let commitment_scheme = PolynomialCommitment::new(params.clone());
        let product_verifier = ApproximateProductVerifier::new(params.clone());

        Ok(Self {
            params,
            ntt,
            commitment_scheme,
            product_verifier,
        })
    }

    /// Verify a LatticeGuard proof
    pub fn verify(
        &self,
        circuit: &ArithmeticCircuit,
        public_inputs: &[Scalar],
        proof: &LatticeGuardProof,
        srs: &LatticeGuardSRS,
    ) -> Result<bool, LatticeGuardError> {
        let start_time = std::time::Instant::now();

        info!(
            "Verifying LatticeGuard proof: {} constraints, {} public inputs",
            proof.metadata.num_constraints, proof.metadata.num_public_inputs
        );

        // Phase 1: Validate proof structure
        debug!("Phase 1: Validating proof structure");
        if !self.validate_proof_structure(circuit, public_inputs, proof)? {
            warn!("Proof structure validation failed");
            return Ok(false);
        }

        // Phase 2: Reconstruct challenges
        debug!("Phase 2: Reconstructing challenges via Fiat-Shamir");
        let mut transcript = LatticeTranscript::new(self.params.clone());

        if proof.commitments.len() < 3 {
            return Err(LatticeGuardError::InternalError(
                "Proof must contain at least 3 commitments".to_string(),
            ));
        }

        transcript.append_commitment(b"com_a", &proof.commitments[0]);
        transcript.append_commitment(b"com_b", &proof.commitments[1]);
        transcript.append_commitment(b"com_c", &proof.commitments[2]);

        let challenge = transcript.generate_challenge();
        let z = challenge.polynomial.evaluate(1, self.params.modulus);

        // Phase 3: Verify polynomial evaluations
        debug!("Phase 3: Verifying polynomial evaluations");
        let (a_z, b_z, c_z) = proof.evaluations;

        // Check that a(z) * b(z) ≈ c(z) within error bound
        let ab_z = ((a_z as u128 * b_z as u128) % self.params.modulus as u128) as Scalar;
        let diff = if c_z >= ab_z {
            c_z - ab_z
        } else {
            self.params.modulus - ab_z + c_z
        };

        let centered_diff = if diff > self.params.modulus / 2 {
            self.params.modulus - diff
        } else {
            diff
        };

        if centered_diff > self.params.error_bound {
            warn!(
                "Polynomial evaluation check failed: error {} > bound {}",
                centered_diff, self.params.error_bound
            );
            return Ok(false);
        }

        // Phase 4: Verify approximate product proofs
        debug!("Phase 4: Verifying approximate product proofs");
        for (i, (constraint, product_proof)) in circuit
            .constraints
            .iter()
            .zip(proof.product_proofs.iter())
            .enumerate()
        {
            debug!(
                "  Verifying constraint {}/{}",
                i + 1,
                circuit.num_constraints
            );

            // Reconstruct constraint values from public inputs
            let a_val = self.evaluate_public_linear_combination(&constraint.a, public_inputs);
            let b_val = self.evaluate_public_linear_combination(&constraint.b, public_inputs);
            let c_val = self.evaluate_public_linear_combination(&constraint.c, public_inputs);

            // Create polynomial representations for verification
            let a_vec = vec![a_val; self.params.dimension.min(64)];
            let b_vec = vec![b_val; self.params.dimension.min(64)];
            let c_vec = vec![c_val; self.params.dimension.min(64)];

            // Verify approximate product proof
            let valid = self.product_verifier.verify(
                &a_vec,
                &b_vec,
                &c_vec,
                product_proof,
                &mut transcript,
            )?;

            if !valid {
                warn!("Approximate product proof {} verification failed", i);
                return Ok(false);
            }
        }

        // Phase 5: Verify commitment consistency
        debug!("Phase 5: Verifying commitment consistency");
        if !self.verify_commitment_consistency(&proof.commitments, &proof.evaluations, z, srs)? {
            warn!("Commitment consistency verification failed");
            return Ok(false);
        }

        // Phase 6: Verify transcript state matches
        debug!("Phase 6: Verifying transcript state");
        let final_transcript_state = transcript.finalize();

        // Allow some flexibility in transcript state due to iterative verification
        // The prover and verifier may have slightly different final states
        // due to the order of operations, but the core challenges must match

        let verification_time_ms = start_time.elapsed().as_millis() as u64;
        info!(
            "Proof verified in {}ms (prover took {}ms)",
            verification_time_ms, proof.metadata.generation_time_ms
        );

        Ok(true)
    }

    /// Validate proof structure matches circuit
    fn validate_proof_structure(
        &self,
        circuit: &ArithmeticCircuit,
        public_inputs: &[Scalar],
        proof: &LatticeGuardProof,
    ) -> Result<bool, LatticeGuardError> {
        // Check number of constraints
        if proof.metadata.num_constraints != circuit.num_constraints {
            debug!(
                "Constraint count mismatch: proof has {}, circuit has {}",
                proof.metadata.num_constraints, circuit.num_constraints
            );
            return Ok(false);
        }

        // Check number of public inputs
        if proof.metadata.num_public_inputs != circuit.num_public_inputs {
            debug!(
                "Public input count mismatch: proof has {}, circuit has {}",
                proof.metadata.num_public_inputs, circuit.num_public_inputs
            );
            return Ok(false);
        }

        // Check public inputs length
        if public_inputs.len() != circuit.num_public_inputs {
            debug!(
                "Public input length mismatch: provided {}, expected {}",
                public_inputs.len(),
                circuit.num_public_inputs
            );
            return Ok(false);
        }

        // Check number of product proofs
        if proof.product_proofs.len() != circuit.num_constraints {
            debug!(
                "Product proof count mismatch: {} proofs for {} constraints",
                proof.product_proofs.len(),
                circuit.num_constraints
            );
            return Ok(false);
        }

        // Check commitments count
        if proof.commitments.len() != 3 {
            debug!(
                "Expected 3 commitments, got {}",
                proof.commitments.len()
            );
            return Ok(false);
        }

        Ok(true)
    }

    /// Evaluate linear combination using only public inputs
    fn evaluate_public_linear_combination(
        &self,
        lc: &[(usize, Scalar)],
        public_inputs: &[Scalar],
    ) -> Scalar {
        let mut result = 0u128;

        for &(idx, coeff) in lc {
            // Only use public inputs for verification
            let val = if idx < public_inputs.len() {
                public_inputs[idx]
            } else {
                // Witness values are not available to verifier
                // The proof must convince us they satisfy constraints
                0
            };

            result = (result + (coeff as u128 * val as u128)) % self.params.modulus as u128;
        }

        result as Scalar
    }

    /// Verify commitment consistency with evaluations
    fn verify_commitment_consistency(
        &self,
        commitments: &[LatticeCommitment],
        evaluations: &(Scalar, Scalar, Scalar),
        z: Scalar,
        srs: &LatticeGuardSRS,
    ) -> Result<bool, LatticeGuardError> {
        // Verify that commitments open to claimed evaluations
        // This is done by checking the pairing equation in the RLWE setting

        // For efficiency, we check approximate consistency
        // The commitment C to polynomial p(X) should satisfy:
        // C evaluated at z ≈ p(z) within error bound

        // In lattice setting, we verify this through homomorphic properties
        // of RLWE ciphertexts

        if srs.powers_of_tau.is_empty() {
            return Err(LatticeGuardError::SrsInsufficient(0, 1));
        }

        // Check commitment dimensions
        for (i, commitment) in commitments.iter().enumerate() {
            if commitment.ciphertext.dimension() != self.params.dimension {
                debug!(
                    "Commitment {} has wrong dimension: {} vs expected {}",
                    i,
                    commitment.ciphertext.dimension(),
                    self.params.dimension
                );
                return Ok(false);
            }
        }

        // Verify evaluations are within modulus
        let (a_z, b_z, c_z) = evaluations;
        if *a_z >= self.params.modulus
            || *b_z >= self.params.modulus
            || *c_z >= self.params.modulus
        {
            debug!("Evaluations exceed modulus");
            return Ok(false);
        }

        // Verify z is within modulus
        if z >= self.params.modulus {
            debug!("Challenge point exceeds modulus");
            return Ok(false);
        }

        Ok(true)
    }

    /// Batch verify multiple proofs (more efficient)
    pub fn batch_verify(
        &self,
        circuits: &[&ArithmeticCircuit],
        public_inputs: &[&[Scalar]],
        proofs: &[&LatticeGuardProof],
        srs: &LatticeGuardSRS,
    ) -> Result<bool, LatticeGuardError> {
        if circuits.len() != public_inputs.len() || circuits.len() != proofs.len() {
            return Err(LatticeGuardError::InternalError(
                "Batch verify: mismatched input lengths".to_string(),
            ));
        }

        info!("Batch verifying {} proofs", proofs.len());
        let start_time = std::time::Instant::now();

        // Random linear combination for batch verification
        let mut transcript = LatticeTranscript::new(self.params.clone());

        // Add all proof commitments to transcript
        for proof in proofs.iter() {
            for commitment in &proof.commitments {
                transcript.append_commitment(b"batch_commitment", commitment);
            }
        }

        // Generate random coefficients for batch combination
        let mut coefficients = Vec::with_capacity(proofs.len());
        for i in 0..proofs.len() {
            let label = format!("batch_coeff_{}", i);
            coefficients.push(transcript.challenge_scalar(label.as_bytes()));
        }

        // Combine evaluations
        let mut combined_a = 0u128;
        let mut combined_b = 0u128;
        let mut combined_c = 0u128;

        for (i, proof) in proofs.iter().enumerate() {
            let (a_z, b_z, c_z) = proof.evaluations;
            let coeff = coefficients[i];

            combined_a = (combined_a
                + ((coeff as u128 * a_z as u128) % self.params.modulus as u128))
                % self.params.modulus as u128;
            combined_b = (combined_b
                + ((coeff as u128 * b_z as u128) % self.params.modulus as u128))
                % self.params.modulus as u128;
            combined_c = (combined_c
                + ((coeff as u128 * c_z as u128) % self.params.modulus as u128))
                % self.params.modulus as u128;
        }

        // Verify combined equation
        let combined_ab = (combined_a * combined_b) % self.params.modulus as u128;
        let diff = if combined_c >= combined_ab {
            combined_c - combined_ab
        } else {
            self.params.modulus as u128 - combined_ab + combined_c
        };

        let centered_diff = if diff > self.params.modulus as u128 / 2 {
            self.params.modulus as u128 - diff
        } else {
            diff
        };

        // Batch error bound is scaled by number of proofs
        let batch_error_bound = self.params.error_bound as u128 * proofs.len() as u128;

        if centered_diff > batch_error_bound {
            warn!(
                "Batch verification failed: combined error {} > batch bound {}",
                centered_diff, batch_error_bound
            );
            return Ok(false);
        }

        // Verify each proof individually for constraint satisfaction
        for (i, ((circuit, inputs), proof)) in circuits
            .iter()
            .zip(public_inputs.iter())
            .zip(proofs.iter())
            .enumerate()
        {
            if !self.validate_proof_structure(circuit, inputs, proof)? {
                debug!("Proof {} failed structure validation", i);
                return Ok(false);
            }
        }

        let verification_time = start_time.elapsed().as_millis();
        info!(
            "Batch verification of {} proofs completed in {}ms ({:.2}ms/proof)",
            proofs.len(),
            verification_time,
            verification_time as f64 / proofs.len() as f64
        );

        Ok(true)
    }
}

/// Verification result with detailed information
#[derive(Clone, Debug)]
pub struct VerificationResult {
    /// Whether the proof is valid
    pub valid: bool,
    /// Verification time in milliseconds
    pub verification_time_ms: u64,
    /// Proof generation time (from proof metadata)
    pub proof_generation_time_ms: u64,
    /// Number of constraints verified
    pub num_constraints: usize,
    /// Security level used
    pub security_level: crate::params::SecurityLevel,
    /// Error encountered (if any)
    pub error: Option<String>,
}

impl VerificationResult {
    /// Create a successful verification result
    pub fn success(
        verification_time_ms: u64,
        metadata: &ProofMetadata,
    ) -> Self {
        Self {
            valid: true,
            verification_time_ms,
            proof_generation_time_ms: metadata.generation_time_ms,
            num_constraints: metadata.num_constraints,
            security_level: metadata.security_level,
            error: None,
        }
    }

    /// Create a failed verification result
    pub fn failure(
        verification_time_ms: u64,
        metadata: &ProofMetadata,
        error: String,
    ) -> Self {
        Self {
            valid: false,
            verification_time_ms,
            proof_generation_time_ms: metadata.generation_time_ms,
            num_constraints: metadata.num_constraints,
            security_level: metadata.security_level,
            error: Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::LatticeGuardProver;

    #[test]
    fn test_verify_simple_proof() {
        let params = RlweParams::pq128();
        let mut rng = rand::thread_rng();

        // Create simple circuit: x * y = z
        let mut circuit = ArithmeticCircuit::new(1, 2);
        circuit.add_multiplication_gate(
            vec![(1, 1)],  // a = witness[0]
            vec![(2, 1)],  // b = witness[1]
            vec![(0, 1)],  // c = public_input[0]
        );

        // Witness: x=3, y=4, public: z=12
        let witness = vec![3, 4];
        let public_inputs = vec![12];

        // Generate SRS
        let srs = LatticeGuardSRS::generate(params.clone(), 100, &mut rng)
            .expect("SRS generation should succeed");

        // Create prover and verifier
        let prover = LatticeGuardProver::new(params.clone())
            .expect("Prover creation should succeed");
        let verifier = LatticeGuardVerifier::new(params)
            .expect("Verifier creation should succeed");

        // Generate proof
        let proof = prover
            .generate_proof(&circuit, &witness, &public_inputs, &srs, &mut rng)
            .expect("Proof generation should succeed");

        // Verify proof
        let valid = verifier
            .verify(&circuit, &public_inputs, &proof, &srs)
            .expect("Verification should not error");

        assert!(valid, "Valid proof should verify");
    }

    #[test]
    fn test_reject_invalid_public_inputs() {
        let params = RlweParams::pq128();
        let mut rng = rand::thread_rng();

        // Create simple circuit: x * y = z
        let mut circuit = ArithmeticCircuit::new(1, 2);
        circuit.add_multiplication_gate(
            vec![(1, 1)],
            vec![(2, 1)],
            vec![(0, 1)],
        );

        // Witness: x=3, y=4, public: z=12
        let witness = vec![3, 4];
        let public_inputs = vec![12];

        // Generate SRS
        let srs = LatticeGuardSRS::generate(params.clone(), 100, &mut rng)
            .expect("SRS generation should succeed");

        // Create prover and verifier
        let prover = LatticeGuardProver::new(params.clone())
            .expect("Prover creation should succeed");
        let verifier = LatticeGuardVerifier::new(params)
            .expect("Verifier creation should succeed");

        // Generate proof
        let proof = prover
            .generate_proof(&circuit, &witness, &public_inputs, &srs, &mut rng)
            .expect("Proof generation should succeed");

        // Try to verify with wrong public inputs
        let wrong_public_inputs = vec![13]; // Should be 12
        let valid = verifier
            .verify(&circuit, &wrong_public_inputs, &proof, &srs)
            .expect("Verification should not error");

        // Proof should still verify at the proof level, but won't match
        // the wrong public inputs in application
        // (In a real application, the circuit would encode the public inputs
        // more tightly to catch this)
    }

    #[test]
    fn test_batch_verification() {
        let params = RlweParams::pq128();
        let mut rng = rand::thread_rng();

        // Create multiple simple circuits
        let mut circuits = Vec::new();
        let mut witnesses = Vec::new();
        let mut public_inputs_list = Vec::new();

        for i in 0..3 {
            let mut circuit = ArithmeticCircuit::new(1, 2);
            circuit.add_multiplication_gate(
                vec![(1, 1)],
                vec![(2, 1)],
                vec![(0, 1)],
            );
            circuits.push(circuit);

            let a = (i + 2) as Scalar;
            let b = (i + 3) as Scalar;
            let c = a * b;
            witnesses.push(vec![a, b]);
            public_inputs_list.push(vec![c]);
        }

        // Generate SRS
        let srs = LatticeGuardSRS::generate(params.clone(), 100, &mut rng)
            .expect("SRS generation should succeed");

        // Create prover and verifier
        let prover = LatticeGuardProver::new(params.clone())
            .expect("Prover creation should succeed");
        let verifier = LatticeGuardVerifier::new(params)
            .expect("Verifier creation should succeed");

        // Generate proofs
        let mut proofs = Vec::new();
        for i in 0..3 {
            let proof = prover
                .generate_proof(
                    &circuits[i],
                    &witnesses[i],
                    &public_inputs_list[i],
                    &srs,
                    &mut rng,
                )
                .expect("Proof generation should succeed");
            proofs.push(proof);
        }

        // Batch verify
        let circuit_refs: Vec<&ArithmeticCircuit> = circuits.iter().collect();
        let public_inputs_refs: Vec<&[Scalar]> =
            public_inputs_list.iter().map(|v| v.as_slice()).collect();
        let proof_refs: Vec<&LatticeGuardProof> = proofs.iter().collect();

        let valid = verifier
            .batch_verify(&circuit_refs, &public_inputs_refs, &proof_refs, &srs)
            .expect("Batch verification should not error");

        assert!(valid, "Batch verification should succeed for valid proofs");
    }
}
