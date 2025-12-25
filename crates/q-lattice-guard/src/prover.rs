//! LatticeGuard Prover
//!
//! Implements the prover for the LatticeGuard zk-SNARK protocol.

use crate::{
    approximate_product::{ApproximateProductProof, ApproximateProductProver},
    commitment::{LatticeCommitment, PolynomialCommitment},
    errors::LatticeGuardError,
    ntt::NttOperator,
    params::RlweParams,
    transcript::LatticeTranscript,
    ArithmeticCircuit, LatticeGuardSRS, Polynomial, Scalar,
};
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Complete LatticeGuard proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatticeGuardProof {
    /// Commitments to witness polynomials
    pub commitments: Vec<LatticeCommitment>,
    /// Evaluations at challenge points
    pub evaluations: (Scalar, Scalar, Scalar),
    /// Approximate product proofs for constraint satisfaction
    pub product_proofs: Vec<ApproximateProductProof>,
    /// Transcript state for verification
    pub transcript_state: [u8; 32],
    /// Proof metadata
    pub metadata: ProofMetadata,
}

/// Proof metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Number of constraints proven
    pub num_constraints: usize,
    /// Number of public inputs
    pub num_public_inputs: usize,
    /// Security level used
    pub security_level: crate::params::SecurityLevel,
    /// Proof generation time in milliseconds
    pub generation_time_ms: u64,
}

/// LatticeGuard prover
pub struct LatticeGuardProver {
    params: RlweParams,
    ntt: NttOperator,
    commitment_scheme: PolynomialCommitment,
    product_prover: ApproximateProductProver,
}

impl LatticeGuardProver {
    /// Create new prover with given parameters
    pub fn new(params: RlweParams) -> Result<Self, LatticeGuardError> {
        let ntt = NttOperator::new(&params);
        let commitment_scheme = PolynomialCommitment::new(params.clone());
        let product_prover = ApproximateProductProver::new(params.clone());

        Ok(Self {
            params,
            ntt,
            commitment_scheme,
            product_prover,
        })
    }

    /// Generate a complete proof for an arithmetic circuit
    pub fn generate_proof<R: Rng + CryptoRng>(
        &self,
        circuit: &ArithmeticCircuit,
        witness: &[Scalar],
        public_inputs: &[Scalar],
        srs: &LatticeGuardSRS,
        rng: &mut R,
    ) -> Result<LatticeGuardProof, LatticeGuardError> {
        let start_time = std::time::Instant::now();

        info!(
            "Generating LatticeGuard proof: {} constraints, {} public inputs",
            circuit.num_constraints, circuit.num_public_inputs
        );

        // Validate inputs
        self.validate_inputs(circuit, witness, public_inputs, srs)?;

        // Phase 1: Encode constraints as polynomials
        debug!("Phase 1: Encoding constraints as polynomials");
        let (a_poly, b_poly, c_poly) = self.encode_constraints(circuit, witness, public_inputs)?;

        // Phase 2: Commit to polynomials
        debug!("Phase 2: Committing to polynomials");
        let com_a = self.commitment_scheme.commit(&a_poly, srs, rng)?;
        let com_b = self.commitment_scheme.commit(&b_poly, srs, rng)?;
        let com_c = self.commitment_scheme.commit(&c_poly, srs, rng)?;

        // Phase 3: Fiat-Shamir challenge generation
        debug!("Phase 3: Generating challenges via Fiat-Shamir");
        let mut transcript = LatticeTranscript::new(self.params.clone());
        transcript.append_commitment(b"com_a", &com_a);
        transcript.append_commitment(b"com_b", &com_b);
        transcript.append_commitment(b"com_c", &com_c);

        let challenge = transcript.generate_challenge();
        let z = challenge.polynomial.evaluate(1, self.params.modulus);

        // Phase 4: Evaluate polynomials at challenge
        debug!("Phase 4: Evaluating at challenge point");
        let a_z = a_poly.evaluate(z, self.params.modulus);
        let b_z = b_poly.evaluate(z, self.params.modulus);
        let c_z = c_poly.evaluate(z, self.params.modulus);

        // Phase 5: Generate approximate product proofs
        debug!("Phase 5: Generating approximate product proofs");
        let mut product_proofs = Vec::new();

        for (i, constraint) in circuit.constraints.iter().enumerate() {
            debug!("  Proving constraint {}/{}", i + 1, circuit.num_constraints);

            // Evaluate constraint at witness
            let a_val = self.evaluate_linear_combination(&constraint.a, witness, public_inputs);
            let b_val = self.evaluate_linear_combination(&constraint.b, witness, public_inputs);
            let c_val = self.evaluate_linear_combination(&constraint.c, witness, public_inputs);

            // Create polynomial representations
            let a_vec = vec![a_val; self.params.dimension.min(64)];
            let b_vec = vec![b_val; self.params.dimension.min(64)];
            let c_vec = vec![c_val; self.params.dimension.min(64)];

            // Generate approximate product proof
            let proof = self.product_prover.prove(
                &a_vec,
                &b_vec,
                &c_vec,
                self.params.error_bound,
                &mut transcript,
            )?;

            product_proofs.push(proof);
        }

        // Phase 6: Finalize proof
        let transcript_state = transcript.finalize();
        let generation_time_ms = start_time.elapsed().as_millis() as u64;

        info!(
            "Proof generated in {}ms ({} constraints)",
            generation_time_ms, circuit.num_constraints
        );

        Ok(LatticeGuardProof {
            commitments: vec![com_a, com_b, com_c],
            evaluations: (a_z, b_z, c_z),
            product_proofs,
            transcript_state,
            metadata: ProofMetadata {
                num_constraints: circuit.num_constraints,
                num_public_inputs: circuit.num_public_inputs,
                security_level: self.params.security_level,
                generation_time_ms,
            },
        })
    }

    /// Validate prover inputs
    fn validate_inputs(
        &self,
        circuit: &ArithmeticCircuit,
        witness: &[Scalar],
        public_inputs: &[Scalar],
        srs: &LatticeGuardSRS,
    ) -> Result<(), LatticeGuardError> {
        // Check witness size
        if witness.len() != circuit.num_witness {
            return Err(LatticeGuardError::WitnessSizeMismatch(
                circuit.num_witness,
                witness.len(),
            ));
        }

        // Check public inputs size
        if public_inputs.len() != circuit.num_public_inputs {
            return Err(LatticeGuardError::WitnessSizeMismatch(
                circuit.num_public_inputs,
                public_inputs.len(),
            ));
        }

        // Check SRS size
        if srs.max_constraints < circuit.num_constraints {
            return Err(LatticeGuardError::SrsInsufficient(
                srs.max_constraints,
                circuit.num_constraints,
            ));
        }

        Ok(())
    }

    /// Encode circuit constraints as polynomials
    fn encode_constraints(
        &self,
        circuit: &ArithmeticCircuit,
        witness: &[Scalar],
        public_inputs: &[Scalar],
    ) -> Result<(Polynomial, Polynomial, Polynomial), LatticeGuardError> {
        let n = circuit.num_constraints.max(1);

        let mut a_coeffs = vec![0u64; n];
        let mut b_coeffs = vec![0u64; n];
        let mut c_coeffs = vec![0u64; n];

        for (i, constraint) in circuit.constraints.iter().enumerate() {
            a_coeffs[i] =
                self.evaluate_linear_combination(&constraint.a, witness, public_inputs);
            b_coeffs[i] =
                self.evaluate_linear_combination(&constraint.b, witness, public_inputs);
            c_coeffs[i] =
                self.evaluate_linear_combination(&constraint.c, witness, public_inputs);
        }

        Ok((
            Polynomial::new(a_coeffs),
            Polynomial::new(b_coeffs),
            Polynomial::new(c_coeffs),
        ))
    }

    /// Evaluate a linear combination at the witness/public inputs
    fn evaluate_linear_combination(
        &self,
        lc: &[(usize, Scalar)],
        witness: &[Scalar],
        public_inputs: &[Scalar],
    ) -> Scalar {
        let total_public = public_inputs.len();

        let mut result = 0u128;
        for &(idx, coeff) in lc {
            let val = if idx < total_public {
                public_inputs[idx]
            } else if idx - total_public < witness.len() {
                witness[idx - total_public]
            } else {
                0
            };

            result = (result + (coeff as u128 * val as u128)) % self.params.modulus as u128;
        }
        result as Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::R1CSConstraint;

    #[test]
    fn test_simple_proof_generation() {
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

        // Create prover
        let prover = LatticeGuardProver::new(params).expect("Prover creation should succeed");

        // Generate proof
        let proof = prover
            .generate_proof(&circuit, &witness, &public_inputs, &srs, &mut rng)
            .expect("Proof generation should succeed");

        assert_eq!(proof.metadata.num_constraints, 1);
        assert_eq!(proof.commitments.len(), 3);
    }
}
