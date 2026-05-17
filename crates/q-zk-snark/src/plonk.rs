//! PLONK zk-SNARK implementation for Q-NarwhalKnight
//!
//! PLONK provides universal trusted setup, meaning the same setup can be used
//! for any circuit up to a maximum size, making it ideal for smart contracts.

use anyhow::Result;
use ark_ec::pairing::Pairing;
use ark_ec::CurveGroup;
use ark_ff::{One, PrimeField, Zero};
use ark_poly::polynomial::univariate::DensePolynomial;
use ark_poly::{DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain, Polynomial};
use ark_poly_commit::kzg10::Powers;
use ark_poly_commit::kzg10::{
    Proof as KZGProof, UniversalParams, VerifierKey as KZGVerifierKey, KZG10,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_snark::SNARK;
use ark_std::{marker::PhantomData, rand::RngCore, vec::Vec};

use crate::SNARKError;

/// PLONK SNARK implementation
pub struct PLONKSNARK<E: Pairing> {
    _phantom: PhantomData<E>,
}

/// PLONK universal setup parameters (SRS - Structured Reference String)
#[derive(Clone)]
pub struct PLONKSrs<E: Pairing> {
    /// Universal parameters for KZG
    pub universal_params: UniversalParams<E>,
    /// Maximum degree supported
    pub max_degree: usize,
}

/// PLONK proving key for a specific circuit
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PLONKProvingKey<E: Pairing> {
    /// Selector polynomials
    pub q_l: DensePolynomial<E::ScalarField>, // Left selector
    pub q_r: DensePolynomial<E::ScalarField>, // Right selector
    pub q_o: DensePolynomial<E::ScalarField>, // Output selector
    pub q_m: DensePolynomial<E::ScalarField>, // Multiplication selector
    pub q_c: DensePolynomial<E::ScalarField>, // Constant selector

    /// Permutation polynomials
    pub sigma_1: DensePolynomial<E::ScalarField>,
    pub sigma_2: DensePolynomial<E::ScalarField>,
    pub sigma_3: DensePolynomial<E::ScalarField>,

    /// Domain for polynomial operations
    pub domain: GeneralEvaluationDomain<E::ScalarField>,

    /// Circuit size
    pub circuit_size: usize,
}

/// PLONK verifying key
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PLONKVerifyingKey<E: Pairing> {
    /// Commitments to selector polynomials
    pub q_l_comm: E::G1Affine,
    pub q_r_comm: E::G1Affine,
    pub q_o_comm: E::G1Affine,
    pub q_m_comm: E::G1Affine,
    pub q_c_comm: E::G1Affine,

    /// Commitments to permutation polynomials
    pub sigma_1_comm: E::G1Affine,
    pub sigma_2_comm: E::G1Affine,
    pub sigma_3_comm: E::G1Affine,

    /// Domain size
    pub domain_size: usize,

    /// Public input size
    pub public_input_size: usize,
}

/// PLONK proof
#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct PLONKProof<E: Pairing> {
    /// Wire commitments
    pub a_comm: E::G1Affine,
    pub b_comm: E::G1Affine,
    pub c_comm: E::G1Affine,

    /// Permutation commitment
    pub z_comm: E::G1Affine,

    /// Quotient polynomial commitment
    pub t_lo_comm: E::G1Affine,
    pub t_mid_comm: E::G1Affine,
    pub t_hi_comm: E::G1Affine,

    /// Opening proofs
    pub w_zeta_proof: KZGProof<E>,
    pub w_zeta_omega_proof: KZGProof<E>,

    /// Evaluations at challenge point
    pub a_eval: E::ScalarField,
    pub b_eval: E::ScalarField,
    pub c_eval: E::ScalarField,
    pub s1_eval: E::ScalarField,
    pub s2_eval: E::ScalarField,
    pub z_omega_eval: E::ScalarField,

    /// Public inputs
    pub public_inputs: Vec<E::ScalarField>,
}

/// PLONK circuit representation
pub struct PLONKCircuit<F: PrimeField> {
    /// Gate constraints in PLONK format
    pub gates: Vec<PLONKGate<F>>,
    /// Wire assignments
    pub wires: Vec<F>,
    /// Public inputs
    pub public_inputs: Vec<F>,
    /// Copy constraints (wire permutations)
    pub copy_constraints: Vec<(usize, usize)>,
}

/// PLONK gate constraint
#[derive(Debug, Clone)]
pub struct PLONKGate<F: PrimeField> {
    /// Left wire index
    pub left_wire: usize,
    /// Right wire index  
    pub right_wire: usize,
    /// Output wire index
    pub output_wire: usize,
    /// Gate selectors
    pub q_l: F, // Coefficient for left wire
    pub q_r: F, // Coefficient for right wire
    pub q_o: F, // Coefficient for output wire
    pub q_m: F, // Coefficient for left * right
    pub q_c: F, // Constant term
}

impl<E: Pairing> PLONKSNARK<E> {
    /// Create new PLONK system
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    /// Generate universal SRS (structured reference string)
    /// This is a one-time trusted setup that can be used for any circuit
    pub fn universal_setup<R>(max_degree: usize, rng: &mut R) -> Result<PLONKSrs<E>>
    where
        R: RngCore,
    {
        // Use the proper PolynomialCommitment trait setup
        let universal_params = KZG10::<E, DensePolynomial<E::ScalarField>>::setup(
            max_degree, false, rng,
        )
        .map_err(|e| {
            anyhow::anyhow!(SNARKError::SetupFailed(format!(
                "KZG setup failed: {:?}",
                e
            )))
        })?;

        Ok(PLONKSrs {
            universal_params,
            max_degree,
        })
    }

    /// Circuit-specific setup using universal SRS
    pub fn circuit_setup(
        srs: &PLONKSrs<E>,
        circuit: &PLONKCircuit<E::ScalarField>,
    ) -> Result<(PLONKProvingKey<E>, PLONKVerifyingKey<E>)> {
        let circuit_size = circuit.gates.len().next_power_of_two();

        if circuit_size > srs.max_degree {
            return Err(SNARKError::InvalidParameters(format!(
                "Circuit size {} exceeds SRS max degree {}",
                circuit_size, srs.max_degree
            ))
            .into());
        }

        // Create evaluation domain
        let domain =
            GeneralEvaluationDomain::<E::ScalarField>::new(circuit_size).ok_or_else(|| {
                SNARKError::SetupFailed("Failed to create evaluation domain".to_string())
            })?;

        // Build selector polynomials
        let (q_l, q_r, q_o, q_m, q_c) = Self::build_selector_polynomials(circuit, &domain)?;

        // Build permutation polynomials
        let (sigma_1, sigma_2, sigma_3) = Self::build_permutation_polynomials(circuit, &domain)?;

        let proving_key = PLONKProvingKey {
            q_l: q_l.clone(),
            q_r: q_r.clone(),
            q_o: q_o.clone(),
            q_m: q_m.clone(),
            q_c: q_c.clone(),
            sigma_1: sigma_1.clone(),
            sigma_2: sigma_2.clone(),
            sigma_3: sigma_3.clone(),
            domain,
            circuit_size,
        };

        // Commit to polynomials for verifying key
        let q_l_comm = Self::commit_polynomial(&srs.universal_params, &q_l)?;
        let q_r_comm = Self::commit_polynomial(&srs.universal_params, &q_r)?;
        let q_o_comm = Self::commit_polynomial(&srs.universal_params, &q_o)?;
        let q_m_comm = Self::commit_polynomial(&srs.universal_params, &q_m)?;
        let q_c_comm = Self::commit_polynomial(&srs.universal_params, &q_c)?;

        let sigma_1_comm = Self::commit_polynomial(&srs.universal_params, &sigma_1)?;
        let sigma_2_comm = Self::commit_polynomial(&srs.universal_params, &sigma_2)?;
        let sigma_3_comm = Self::commit_polynomial(&srs.universal_params, &sigma_3)?;

        let verifying_key = PLONKVerifyingKey {
            q_l_comm,
            q_r_comm,
            q_o_comm,
            q_m_comm,
            q_c_comm,
            sigma_1_comm,
            sigma_2_comm,
            sigma_3_comm,
            domain_size: circuit_size,
            public_input_size: circuit.public_inputs.len(),
        };

        Ok((proving_key, verifying_key))
    }

    /// Generate PLONK proof
    pub fn prove<R>(
        srs: &PLONKSrs<E>,
        proving_key: &PLONKProvingKey<E>,
        circuit: &PLONKCircuit<E::ScalarField>,
        rng: &mut R,
    ) -> Result<PLONKProof<E>>
    where
        R: RngCore,
    {
        // Build witness polynomials
        let (a_poly, b_poly, c_poly) =
            Self::build_witness_polynomials(circuit, &proving_key.domain)?;

        // Commit to witness polynomials
        let a_comm = Self::commit_polynomial(&srs.universal_params, &a_poly)?;
        let b_comm = Self::commit_polynomial(&srs.universal_params, &b_poly)?;
        let c_comm = Self::commit_polynomial(&srs.universal_params, &c_poly)?;

        // Generate challenge beta, gamma (using Fiat-Shamir)
        let (beta, gamma) = Self::generate_challenges(&[
            Self::serialize_commitment(&a_comm)?,
            Self::serialize_commitment(&b_comm)?,
            Self::serialize_commitment(&c_comm)?,
        ])?;

        // Build permutation polynomial
        let z_poly = Self::build_permutation_polynomial(
            circuit,
            &proving_key,
            &a_poly,
            &b_poly,
            &c_poly,
            beta,
            gamma,
        )?;

        let z_comm = Self::commit_polynomial(&srs.universal_params, &z_poly)?;

        // Generate challenge alpha
        let alpha = Self::generate_single_challenge(&Self::serialize_commitment(&z_comm)?)?;

        // Build quotient polynomial
        let (t_lo_poly, t_mid_poly, t_hi_poly) = Self::build_quotient_polynomial(
            &proving_key,
            &a_poly,
            &b_poly,
            &c_poly,
            &z_poly,
            alpha,
            beta,
            gamma,
        )?;

        let t_lo_comm = Self::commit_polynomial(&srs.universal_params, &t_lo_poly)?;
        let t_mid_comm = Self::commit_polynomial(&srs.universal_params, &t_mid_poly)?;
        let t_hi_comm = Self::commit_polynomial(&srs.universal_params, &t_hi_poly)?;

        // Generate challenge zeta
        let zeta = Self::generate_single_challenge(
            &[
                Self::serialize_commitment(&t_lo_comm)?,
                Self::serialize_commitment(&t_mid_comm)?,
                Self::serialize_commitment(&t_hi_comm)?,
            ]
            .concat(),
        )?;

        // Evaluate polynomials at zeta
        let a_eval = a_poly.evaluate(&zeta);
        let b_eval = b_poly.evaluate(&zeta);
        let c_eval = c_poly.evaluate(&zeta);
        let s1_eval = proving_key.sigma_1.evaluate(&zeta);
        let s2_eval = proving_key.sigma_2.evaluate(&zeta);
        let z_omega_eval = z_poly.evaluate(&(zeta * proving_key.domain.group_gen()));

        // Create opening proofs
        let w_zeta_proof = Self::create_opening_proof(
            &srs.universal_params,
            &[&a_poly, &b_poly, &c_poly],
            zeta,
            rng,
        )?;

        let w_zeta_omega_proof = Self::create_opening_proof(
            &srs.universal_params,
            &[&z_poly],
            zeta * proving_key.domain.group_gen(),
            rng,
        )?;

        Ok(PLONKProof {
            a_comm,
            b_comm,
            c_comm,
            z_comm,
            t_lo_comm,
            t_mid_comm,
            t_hi_comm,
            w_zeta_proof,
            w_zeta_omega_proof,
            a_eval,
            b_eval,
            c_eval,
            s1_eval,
            s2_eval,
            z_omega_eval,
            public_inputs: circuit.public_inputs.clone(),
        })
    }

    /// Verify PLONK proof
    pub fn verify(
        srs: &PLONKSrs<E>,
        verifying_key: &PLONKVerifyingKey<E>,
        proof: &PLONKProof<E>,
    ) -> Result<bool> {
        // Recreate challenges
        let (beta, gamma) = Self::generate_challenges(&[
            Self::serialize_commitment(&proof.a_comm)?,
            Self::serialize_commitment(&proof.b_comm)?,
            Self::serialize_commitment(&proof.c_comm)?,
        ])?;

        let alpha = Self::generate_single_challenge(&Self::serialize_commitment(&proof.z_comm)?)?;

        let zeta = Self::generate_single_challenge(
            &[
                Self::serialize_commitment(&proof.t_lo_comm)?,
                Self::serialize_commitment(&proof.t_mid_comm)?,
                Self::serialize_commitment(&proof.t_hi_comm)?,
            ]
            .concat(),
        )?;

        // Create verifier key from universal params - simplified placeholder
        let verifier_key = KZGVerifierKey {
            g: srs.universal_params.powers_of_g.first().unwrap().clone(),
            gamma_g: srs.universal_params.powers_of_g.first().unwrap().clone(), // Use g as placeholder
            h: srs.universal_params.h,
            beta_h: srs.universal_params.h, // Use h as placeholder for beta_h
            prepared_h: srs.universal_params.h.into(),
            prepared_beta_h: srs.universal_params.h.into(), // Use h as placeholder
        };

        // Verify opening proofs
        let zeta_valid = Self::verify_opening_proof(
            &verifier_key,
            &[proof.a_comm, proof.b_comm, proof.c_comm],
            &[proof.a_eval, proof.b_eval, proof.c_eval],
            zeta,
            &proof.w_zeta_proof,
        )?;

        if !zeta_valid {
            return Ok(false);
        }

        let domain = GeneralEvaluationDomain::<E::ScalarField>::new(verifying_key.domain_size)
            .ok_or_else(|| SNARKError::VerificationFailed("Failed to create domain".to_string()))?;

        let omega = domain.group_gen();
        let zeta_omega_valid = Self::verify_opening_proof(
            &verifier_key,
            &[proof.z_comm],
            &[proof.z_omega_eval],
            zeta * omega,
            &proof.w_zeta_omega_proof,
        )?;

        if !zeta_omega_valid {
            return Ok(false);
        }

        // Verify the PLONK equation
        Self::verify_plonk_equation(verifying_key, proof, alpha, beta, gamma, zeta)
    }

    // Helper methods (implementation details)

    fn build_selector_polynomials(
        circuit: &PLONKCircuit<E::ScalarField>,
        domain: &GeneralEvaluationDomain<E::ScalarField>,
    ) -> Result<(
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
    )> {
        let domain_size = domain.size();
        let mut q_l_evals = vec![E::ScalarField::zero(); domain_size];
        let mut q_r_evals = vec![E::ScalarField::zero(); domain_size];
        let mut q_o_evals = vec![E::ScalarField::zero(); domain_size];
        let mut q_m_evals = vec![E::ScalarField::zero(); domain_size];
        let mut q_c_evals = vec![E::ScalarField::zero(); domain_size];

        for (i, gate) in circuit.gates.iter().enumerate() {
            if i >= domain_size {
                break;
            }
            q_l_evals[i] = gate.q_l;
            q_r_evals[i] = gate.q_r;
            q_o_evals[i] = gate.q_o;
            q_m_evals[i] = gate.q_m;
            q_c_evals[i] = gate.q_c;
        }

        let q_l = domain.ifft(&q_l_evals);
        let q_r = domain.ifft(&q_r_evals);
        let q_o = domain.ifft(&q_o_evals);
        let q_m = domain.ifft(&q_m_evals);
        let q_c = domain.ifft(&q_c_evals);

        Ok((
            DensePolynomial::from_coefficients_vec(q_l),
            DensePolynomial::from_coefficients_vec(q_r),
            DensePolynomial::from_coefficients_vec(q_o),
            DensePolynomial::from_coefficients_vec(q_m),
            DensePolynomial::from_coefficients_vec(q_c),
        ))
    }

    fn build_permutation_polynomials(
        _circuit: &PLONKCircuit<E::ScalarField>,
        domain: &GeneralEvaluationDomain<E::ScalarField>,
    ) -> Result<(
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
    )> {
        // Simplified permutation polynomial construction
        // Real implementation would compute proper copy constraint permutations
        let domain_size = domain.size();
        let roots: Vec<_> = domain.elements().collect();

        let sigma_1_evals: Vec<_> = (0..domain_size).map(|i| roots[i]).collect();
        let sigma_2_evals: Vec<_> = (0..domain_size).map(|i| roots[i]).collect();
        let sigma_3_evals: Vec<_> = (0..domain_size).map(|i| roots[i]).collect();

        let sigma_1 = domain.ifft(&sigma_1_evals);
        let sigma_2 = domain.ifft(&sigma_2_evals);
        let sigma_3 = domain.ifft(&sigma_3_evals);

        Ok((
            DensePolynomial::from_coefficients_vec(sigma_1),
            DensePolynomial::from_coefficients_vec(sigma_2),
            DensePolynomial::from_coefficients_vec(sigma_3),
        ))
    }

    fn build_witness_polynomials(
        circuit: &PLONKCircuit<E::ScalarField>,
        domain: &GeneralEvaluationDomain<E::ScalarField>,
    ) -> Result<(
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
    )> {
        let domain_size = domain.size();
        let mut a_evals = vec![E::ScalarField::zero(); domain_size];
        let mut b_evals = vec![E::ScalarField::zero(); domain_size];
        let mut c_evals = vec![E::ScalarField::zero(); domain_size];

        for (i, gate) in circuit.gates.iter().enumerate() {
            if i >= domain_size {
                break;
            }
            a_evals[i] = circuit
                .wires
                .get(gate.left_wire)
                .copied()
                .unwrap_or_default();
            b_evals[i] = circuit
                .wires
                .get(gate.right_wire)
                .copied()
                .unwrap_or_default();
            c_evals[i] = circuit
                .wires
                .get(gate.output_wire)
                .copied()
                .unwrap_or_default();
        }

        let a_coeffs = domain.ifft(&a_evals);
        let b_coeffs = domain.ifft(&b_evals);
        let c_coeffs = domain.ifft(&c_evals);

        Ok((
            DensePolynomial::from_coefficients_vec(a_coeffs),
            DensePolynomial::from_coefficients_vec(b_coeffs),
            DensePolynomial::from_coefficients_vec(c_coeffs),
        ))
    }

    fn commit_polynomial(
        universal_params: &UniversalParams<E>,
        poly: &DensePolynomial<E::ScalarField>,
    ) -> Result<E::G1Affine> {
        // Create Powers struct from universal params - simplified
        let powers = Powers {
            powers_of_g: universal_params.powers_of_g.clone().into(),
            powers_of_gamma_g: universal_params.powers_of_g.clone().into(), // Use powers_of_g as placeholder
        };
        let commitment =
            KZG10::<E, DensePolynomial<E::ScalarField>>::commit(&powers, poly, None, None)
                .map_err(|e| {
                    SNARKError::ProvingFailed(format!("Polynomial commitment failed: {:?}", e))
                })?;
        Ok(commitment.0 .0)
    }

    // Placeholder implementations for remaining helper methods
    fn generate_challenges(data: &[Vec<u8>]) -> Result<(E::ScalarField, E::ScalarField)> {
        // Use Fiat-Shamir to generate challenges from transcript
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        for d in data {
            hasher.update(d);
        }
        let hash = hasher.finalize();

        // Convert hash to field elements (simplified)
        let beta = E::ScalarField::from_le_bytes_mod_order(&hash[..16]);
        let gamma = E::ScalarField::from_le_bytes_mod_order(&hash[16..]);
        Ok((beta, gamma))
    }

    fn generate_single_challenge(data: &[u8]) -> Result<E::ScalarField> {
        use sha3::{Digest, Sha3_256};
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        let hash = hasher.finalize();
        Ok(E::ScalarField::from_le_bytes_mod_order(&hash))
    }

    fn serialize_commitment(comm: &E::G1Affine) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        comm.serialize_uncompressed(&mut bytes).map_err(|e| {
            SNARKError::Serialization(format!("Commitment serialization failed: {:?}", e))
        })?;
        Ok(bytes)
    }

    // Placeholder for complex PLONK-specific methods
    fn build_permutation_polynomial(
        _circuit: &PLONKCircuit<E::ScalarField>,
        _proving_key: &PLONKProvingKey<E>,
        _a_poly: &DensePolynomial<E::ScalarField>,
        _b_poly: &DensePolynomial<E::ScalarField>,
        _c_poly: &DensePolynomial<E::ScalarField>,
        _beta: E::ScalarField,
        _gamma: E::ScalarField,
    ) -> Result<DensePolynomial<E::ScalarField>> {
        // Simplified - real implementation would build proper permutation argument
        Ok(DensePolynomial::from_coefficients_vec(vec![
            E::ScalarField::one(),
        ]))
    }

    fn build_quotient_polynomial(
        _proving_key: &PLONKProvingKey<E>,
        _a_poly: &DensePolynomial<E::ScalarField>,
        _b_poly: &DensePolynomial<E::ScalarField>,
        _c_poly: &DensePolynomial<E::ScalarField>,
        _z_poly: &DensePolynomial<E::ScalarField>,
        _alpha: E::ScalarField,
        _beta: E::ScalarField,
        _gamma: E::ScalarField,
    ) -> Result<(
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
        DensePolynomial<E::ScalarField>,
    )> {
        // Simplified - real implementation would compute quotient polynomial properly
        let t_lo = DensePolynomial::from_coefficients_vec(vec![E::ScalarField::one()]);
        let t_mid = DensePolynomial::from_coefficients_vec(vec![E::ScalarField::one()]);
        let t_hi = DensePolynomial::from_coefficients_vec(vec![E::ScalarField::one()]);
        Ok((t_lo, t_mid, t_hi))
    }

    fn create_opening_proof<R>(
        _universal_params: &UniversalParams<E>,
        _polys: &[&DensePolynomial<E::ScalarField>],
        _point: E::ScalarField,
        _rng: &mut R,
    ) -> Result<KZGProof<E>>
    where
        R: RngCore,
    {
        // Simplified - real implementation would create proper KZG opening proof
        Ok(KZGProof {
            w: E::G1::zero().into_affine(),
            random_v: None,
        })
    }

    fn verify_opening_proof(
        _vk: &KZGVerifierKey<E>,
        _commitments: &[E::G1Affine],
        _values: &[E::ScalarField],
        _point: E::ScalarField,
        _proof: &KZGProof<E>,
    ) -> Result<bool> {
        // Simplified - real implementation would verify KZG opening proof
        Ok(true)
    }

    fn verify_plonk_equation(
        _verifying_key: &PLONKVerifyingKey<E>,
        _proof: &PLONKProof<E>,
        _alpha: E::ScalarField,
        _beta: E::ScalarField,
        _gamma: E::ScalarField,
        _zeta: E::ScalarField,
    ) -> Result<bool> {
        // Simplified - real implementation would verify the main PLONK equation
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::{Bn254, Fr};
    use ark_std::test_rng;

    #[test]
    fn test_plonk_universal_setup() {
        let mut rng = test_rng();
        let max_degree = 1024;

        let srs = PLONKSNARK::<Bn254>::universal_setup(max_degree, &mut rng)
            .expect("Universal setup should succeed");

        assert_eq!(srs.max_degree, max_degree);
        assert!(!srs.universal_params.powers_of_g.powers_of_g.is_empty());
    }

    #[test]
    fn test_plonk_circuit_setup() {
        let mut rng = test_rng();
        let max_degree = 1024;

        let srs = PLONKSNARK::<Bn254>::universal_setup(max_degree, &mut rng)
            .expect("Universal setup should succeed");

        // Create simple test circuit
        let circuit = PLONKCircuit {
            gates: vec![PLONKGate {
                left_wire: 0,
                right_wire: 1,
                output_wire: 2,
                q_l: Fr::one(),
                q_r: Fr::one(),
                q_o: -Fr::one(),
                q_m: Fr::zero(),
                q_c: Fr::zero(),
            }],
            wires: vec![Fr::from(2u64), Fr::from(3u64), Fr::from(5u64)],
            public_inputs: vec![Fr::from(5u64)],
            copy_constraints: vec![],
        };

        let (pk, vk) = PLONKSNARK::<Bn254>::circuit_setup(&srs, &circuit)
            .expect("Circuit setup should succeed");

        assert_eq!(pk.circuit_size, 2); // Next power of 2 above 1
        assert_eq!(vk.public_input_size, 1);
    }
}
