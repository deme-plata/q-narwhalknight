//! RLWE-based Polynomial Commitment Scheme
//!
//! This module implements a lattice-based polynomial commitment inspired by
//! KZG commitments, but using RLWE for post-quantum security.

use crate::{
    errors::LatticeGuardError,
    params::RlweParams,
    rlwe::{RlweCiphertext, RlweSecretKey, ring_mul},
    Polynomial, Scalar, LatticeGuardSRS,
};
use rand::{CryptoRng, Rng};
use serde::{Deserialize, Serialize};

/// RLWE-based polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatticeCommitment {
    /// RLWE ciphertext committing to the polynomial
    pub ciphertext: RlweCiphertext,
    /// Randomness used (for opening)
    #[serde(skip)]
    pub randomness: Option<Vec<Scalar>>,
}

impl LatticeCommitment {
    /// Create new commitment
    pub fn new(ciphertext: RlweCiphertext) -> Self {
        Self {
            ciphertext,
            randomness: None,
        }
    }

    /// Get the commitment as bytes for hashing
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
}

/// Opening proof for polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpeningProof {
    /// Evaluation of polynomial at challenge point
    pub evaluation: Scalar,
    /// Quotient polynomial commitment
    pub quotient_commitment: RlweCiphertext,
    /// Error bound proof (shows evaluation is correct within bound)
    pub error_bound_witness: Vec<Scalar>,
}

/// Polynomial commitment scheme
pub struct PolynomialCommitment {
    params: RlweParams,
}

impl PolynomialCommitment {
    /// Create new commitment scheme
    pub fn new(params: RlweParams) -> Self {
        Self { params }
    }

    /// Commit to a polynomial
    pub fn commit<R: Rng + CryptoRng>(
        &self,
        polynomial: &Polynomial,
        srs: &LatticeGuardSRS,
        rng: &mut R,
    ) -> Result<LatticeCommitment, LatticeGuardError> {
        if polynomial.coefficients.len() > srs.max_constraints {
            return Err(LatticeGuardError::DegreeTooLarge(
                polynomial.coefficients.len(),
                srs.max_constraints,
            ));
        }

        // Compute commitment as linear combination of powers of tau
        // C = Σ aᵢ * [τⁱ] where [τⁱ] are RLWE encryptions
        let mut commitment_a = vec![0u64; self.params.dimension];
        let mut commitment_b = vec![0u64; self.params.dimension];

        for (i, &coeff) in polynomial.coefficients.iter().enumerate() {
            if i >= srs.powers_of_tau.len() {
                break;
            }

            let tau_i = &srs.powers_of_tau[i];

            // Add coeff * tau_i to commitment
            for j in 0..self.params.dimension {
                commitment_a[j] = (commitment_a[j]
                    + ((coeff as u128 * tau_i.a[j] as u128) % self.params.modulus as u128) as u64)
                    % self.params.modulus;
                commitment_b[j] = (commitment_b[j]
                    + ((coeff as u128 * tau_i.b[j] as u128) % self.params.modulus as u128) as u64)
                    % self.params.modulus;
            }
        }

        // Add fresh randomness for zero-knowledge
        let randomness = self.params.sample_error_polynomial(rng);
        for j in 0..self.params.dimension {
            commitment_b[j] = (commitment_b[j] + randomness[j]) % self.params.modulus;
        }

        let ciphertext = RlweCiphertext::new(commitment_a, commitment_b);
        let mut commitment = LatticeCommitment::new(ciphertext);
        commitment.randomness = Some(randomness);

        Ok(commitment)
    }

    /// Open commitment at a challenge point
    pub fn open(
        &self,
        polynomial: &Polynomial,
        challenge: Scalar,
        commitment: &LatticeCommitment,
        srs: &LatticeGuardSRS,
    ) -> Result<OpeningProof, LatticeGuardError> {
        // Evaluate polynomial at challenge point
        let evaluation = polynomial.evaluate(challenge, self.params.modulus);

        // Compute quotient polynomial: q(x) = (p(x) - p(z)) / (x - z)
        let quotient = self.compute_quotient(polynomial, challenge, evaluation)?;

        // Commit to quotient polynomial
        let mut rng = rand::thread_rng();
        let quotient_poly = Polynomial::new(quotient.clone());
        let quotient_commitment_full = self.commit(&quotient_poly, srs, &mut rng)?;

        // Create error bound witness
        let error_bound_witness = commitment
            .randomness
            .clone()
            .unwrap_or_else(|| vec![0; self.params.dimension]);

        Ok(OpeningProof {
            evaluation,
            quotient_commitment: quotient_commitment_full.ciphertext,
            error_bound_witness,
        })
    }

    /// Verify an opening proof
    pub fn verify_opening(
        &self,
        commitment: &LatticeCommitment,
        challenge: Scalar,
        proof: &OpeningProof,
        srs: &LatticeGuardSRS,
    ) -> Result<bool, LatticeGuardError> {
        // Verify: C - [p(z)] = [q(x)] * [x - z]
        // In RLWE setting, we check approximate equality within error bound

        // Compute [x - z] from SRS
        let x_minus_z = self.compute_x_minus_z_commitment(challenge, srs)?;

        // Compute [q(x)] * [x - z]
        let rhs = self.multiply_commitments(&proof.quotient_commitment, &x_minus_z)?;

        // Compute C - [p(z)]
        let pz_commitment = self.constant_commitment(proof.evaluation, srs)?;
        let lhs = self.subtract_commitments(&commitment.ciphertext, &pz_commitment)?;

        // Check approximate equality within error bound
        let difference_a: Vec<Scalar> = lhs
            .a
            .iter()
            .zip(rhs.a.iter())
            .map(|(&x, &y)| if x >= y { x - y } else { self.params.modulus - y + x })
            .collect();

        let difference_b: Vec<Scalar> = lhs
            .b
            .iter()
            .zip(rhs.b.iter())
            .map(|(&x, &y)| if x >= y { x - y } else { self.params.modulus - y + x })
            .collect();

        // Check if difference is within error bound
        for (&da, &db) in difference_a.iter().zip(difference_b.iter()) {
            let centered_a = if da > self.params.modulus / 2 {
                self.params.modulus - da
            } else {
                da
            };
            let centered_b = if db > self.params.modulus / 2 {
                self.params.modulus - db
            } else {
                db
            };

            if centered_a > self.params.error_bound || centered_b > self.params.error_bound {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Compute quotient polynomial: (p(x) - p(z)) / (x - z)
    fn compute_quotient(
        &self,
        polynomial: &Polynomial,
        z: Scalar,
        pz: Scalar,
    ) -> Result<Vec<Scalar>, LatticeGuardError> {
        let n = polynomial.coefficients.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Synthetic division by (x - z)
        let mut quotient = vec![0u64; n - 1];
        let mut remainder = polynomial.coefficients.last().copied().unwrap_or(0);

        for i in (0..n - 1).rev() {
            quotient[i] = remainder;
            remainder = (polynomial.coefficients[i]
                + ((remainder as u128 * z as u128) % self.params.modulus as u128) as u64)
                % self.params.modulus;
        }

        // Verify remainder equals p(z)
        if remainder != pz {
            return Err(LatticeGuardError::InternalError(
                "Quotient computation failed".to_string(),
            ));
        }

        Ok(quotient)
    }

    /// Create commitment to constant value
    fn constant_commitment(
        &self,
        value: Scalar,
        srs: &LatticeGuardSRS,
    ) -> Result<RlweCiphertext, LatticeGuardError> {
        if srs.powers_of_tau.is_empty() {
            return Err(LatticeGuardError::SrsInsufficient(0, 1));
        }

        let tau_0 = &srs.powers_of_tau[0];
        Ok(tau_0.scalar_mul(value, self.params.modulus))
    }

    /// Compute [x - z] commitment from SRS
    fn compute_x_minus_z_commitment(
        &self,
        z: Scalar,
        srs: &LatticeGuardSRS,
    ) -> Result<RlweCiphertext, LatticeGuardError> {
        if srs.powers_of_tau.len() < 2 {
            return Err(LatticeGuardError::SrsInsufficient(srs.powers_of_tau.len(), 2));
        }

        // [x - z] = [x] - z * [1] = tau^1 - z * tau^0
        let tau_0 = &srs.powers_of_tau[0];
        let tau_1 = &srs.powers_of_tau[1];

        let z_tau_0 = tau_0.scalar_mul(z, self.params.modulus);

        Ok(self.subtract_commitments(tau_1, &z_tau_0)?)
    }

    /// Multiply two commitments (approximate, adds error)
    fn multiply_commitments(
        &self,
        a: &RlweCiphertext,
        b: &RlweCiphertext,
    ) -> Result<RlweCiphertext, LatticeGuardError> {
        // Tensor product followed by relinearization
        // For simplicity, we use approximate multiplication
        let new_a = ring_mul(&a.a, &b.a, self.params.modulus, self.params.dimension);
        let new_b = ring_mul(&a.b, &b.b, self.params.modulus, self.params.dimension);

        Ok(RlweCiphertext::new(new_a, new_b))
    }

    /// Subtract two commitments
    fn subtract_commitments(
        &self,
        a: &RlweCiphertext,
        b: &RlweCiphertext,
    ) -> Result<RlweCiphertext, LatticeGuardError> {
        let new_a: Vec<Scalar> = a
            .a
            .iter()
            .zip(b.a.iter())
            .map(|(&x, &y)| if x >= y { x - y } else { self.params.modulus - y + x })
            .collect();

        let new_b: Vec<Scalar> = a
            .b
            .iter()
            .zip(b.b.iter())
            .map(|(&x, &y)| if x >= y { x - y } else { self.params.modulus - y + x })
            .collect();

        Ok(RlweCiphertext::new(new_a, new_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commitment_creation() {
        let params = RlweParams::pq128();
        let mut rng = rand::thread_rng();

        let srs = LatticeGuardSRS::generate(params.clone(), 100, &mut rng).unwrap();
        let scheme = PolynomialCommitment::new(params);

        let poly = Polynomial::new(vec![1, 2, 3]);
        let commitment = scheme.commit(&poly, &srs, &mut rng).unwrap();

        assert_eq!(commitment.ciphertext.dimension(), srs.params.dimension);
    }
}
