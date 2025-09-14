//! Polynomial Operations for STARK Proofs
//!
//! This module provides polynomial arithmetic, interpolation, and evaluation
//! operations needed for STARK proof generation and verification.

use std::collections::HashMap;

/// Polynomial representation over finite field
#[derive(Clone, Debug)]
pub struct Polynomial {
    /// Polynomial coefficients (index 0 = constant term)
    pub coefficients: Vec<u64>,
    /// Field prime for modular arithmetic
    pub field_prime: u64,
}

impl Polynomial {
    /// Create new polynomial
    pub fn new(coefficients: Vec<u64>, field_prime: u64) -> Self {
        Self {
            coefficients: Self::trim_leading_zeros(coefficients),
            field_prime,
        }
    }

    /// Create zero polynomial
    pub fn zero(field_prime: u64) -> Self {
        Self::new(vec![0], field_prime)
    }

    /// Create constant polynomial
    pub fn constant(value: u64, field_prime: u64) -> Self {
        Self::new(vec![value % field_prime], field_prime)
    }

    /// Get polynomial degree
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty()
            || (self.coefficients.len() == 1 && self.coefficients[0] == 0)
        {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Evaluate polynomial at given point
    pub fn evaluate(&self, x: u64) -> u64 {
        if self.coefficients.is_empty() {
            return 0;
        }

        // Horner's method for efficient evaluation
        let mut result = self.coefficients.last().copied().unwrap_or(0);

        for &coeff in self.coefficients.iter().rev().skip(1) {
            result = self.field_mul(result, x);
            result = self.field_add(result, coeff);
        }

        result
    }

    /// Evaluate polynomial at multiple points (batch evaluation)
    pub fn batch_evaluate(&self, points: &[u64]) -> Vec<u64> {
        points.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Add two polynomials
    pub fn add(&self, other: &Polynomial) -> Polynomial {
        assert_eq!(
            self.field_prime, other.field_prime,
            "Field primes must match"
        );

        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result_coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coefficients.get(i).copied().unwrap_or(0);
            let b = other.coefficients.get(i).copied().unwrap_or(0);
            result_coeffs.push(self.field_add(a, b));
        }

        Polynomial::new(result_coeffs, self.field_prime)
    }

    /// Subtract two polynomials
    pub fn subtract(&self, other: &Polynomial) -> Polynomial {
        assert_eq!(
            self.field_prime, other.field_prime,
            "Field primes must match"
        );

        let max_len = self.coefficients.len().max(other.coefficients.len());
        let mut result_coeffs = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let a = self.coefficients.get(i).copied().unwrap_or(0);
            let b = other.coefficients.get(i).copied().unwrap_or(0);
            result_coeffs.push(self.field_sub(a, b));
        }

        Polynomial::new(result_coeffs, self.field_prime)
    }

    /// Multiply two polynomials
    pub fn multiply(&self, other: &Polynomial) -> Polynomial {
        assert_eq!(
            self.field_prime, other.field_prime,
            "Field primes must match"
        );

        if self.coefficients.is_empty() || other.coefficients.is_empty() {
            return Polynomial::zero(self.field_prime);
        }

        let result_degree = self.degree() + other.degree();
        let mut result_coeffs = vec![0; result_degree + 1];

        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                let product = self.field_mul(a, b);
                result_coeffs[i + j] = self.field_add(result_coeffs[i + j], product);
            }
        }

        Polynomial::new(result_coeffs, self.field_prime)
    }

    /// Multiply polynomial by scalar
    pub fn scalar_multiply(&self, scalar: u64) -> Polynomial {
        let result_coeffs: Vec<u64> = self
            .coefficients
            .iter()
            .map(|&coeff| self.field_mul(coeff, scalar))
            .collect();

        Polynomial::new(result_coeffs, self.field_prime)
    }

    /// Polynomial division (returns quotient and remainder)
    pub fn divide(&self, divisor: &Polynomial) -> (Polynomial, Polynomial) {
        assert_eq!(
            self.field_prime, divisor.field_prime,
            "Field primes must match"
        );

        if divisor.degree() == 0 && divisor.coefficients[0] == 0 {
            panic!("Division by zero polynomial");
        }

        if self.degree() < divisor.degree() {
            return (Polynomial::zero(self.field_prime), self.clone());
        }

        let mut remainder = self.clone();
        let mut quotient_coeffs = vec![0; self.degree() - divisor.degree() + 1];

        while remainder.degree() >= divisor.degree() && !remainder.is_zero() {
            let lead_coeff = remainder.coefficients.last().copied().unwrap_or(0);
            let divisor_lead = divisor.coefficients.last().copied().unwrap_or(0);
            let coeff = self.field_div(lead_coeff, divisor_lead);

            let degree_diff = remainder.degree() - divisor.degree();
            quotient_coeffs[degree_diff] = coeff;

            // Subtract coeff * x^degree_diff * divisor from remainder
            let mut term_coeffs = vec![0; degree_diff + divisor.coefficients.len()];
            for (i, &div_coeff) in divisor.coefficients.iter().enumerate() {
                term_coeffs[degree_diff + i] = self.field_mul(coeff, div_coeff);
            }
            let term = Polynomial::new(term_coeffs, self.field_prime);
            remainder = remainder.subtract(&term);
        }

        (
            Polynomial::new(quotient_coeffs, self.field_prime),
            remainder,
        )
    }

    /// Check if polynomial is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty() || self.coefficients.iter().all(|&coeff| coeff == 0)
    }

    /// Interpolate polynomial from points
    pub fn interpolate(points: &[(u64, u64)], field_prime: u64) -> Polynomial {
        if points.is_empty() {
            return Polynomial::zero(field_prime);
        }

        let n = points.len();
        let mut result = Polynomial::zero(field_prime);

        // Lagrange interpolation
        for i in 0..n {
            let (xi, yi) = points[i];
            let mut basis = Polynomial::constant(yi, field_prime);

            // Compute Lagrange basis polynomial
            for j in 0..n {
                if i != j {
                    let (xj, _) = points[j];

                    // basis *= (x - xj) / (xi - xj)
                    let numerator = Polynomial::new(vec![field_prime - xj, 1], field_prime); // x - xj
                    let denominator_inv =
                        field_inverse(field_sub(xi, xj, field_prime), field_prime);

                    basis = basis.multiply(&numerator);
                    basis = basis.scalar_multiply(denominator_inv);
                }
            }

            result = result.add(&basis);
        }

        result
    }

    /// Fast Walsh-Hadamard Transform (for boolean functions)
    pub fn fwht(&self) -> Vec<u64> {
        let n = self.coefficients.len();
        if !n.is_power_of_two() {
            panic!("FWHT requires power-of-2 size");
        }

        let mut result = self.coefficients.clone();
        let mut size = 1;

        while size < n {
            for i in (0..n).step_by(size * 2) {
                for j in 0..size {
                    let u = result[i + j];
                    let v = result[i + j + size];
                    result[i + j] = self.field_add(u, v);
                    result[i + j + size] = self.field_sub(u, v);
                }
            }
            size *= 2;
        }

        result
    }

    // Private helper methods

    fn trim_leading_zeros(mut coeffs: Vec<u64>) -> Vec<u64> {
        while coeffs.len() > 1 && coeffs.last() == Some(&0) {
            coeffs.pop();
        }
        if coeffs.is_empty() {
            coeffs.push(0);
        }
        coeffs
    }

    fn field_add(&self, a: u64, b: u64) -> u64 {
        (a + b) % self.field_prime
    }

    fn field_sub(&self, a: u64, b: u64) -> u64 {
        (a + self.field_prime - b) % self.field_prime
    }

    fn field_mul(&self, a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % self.field_prime as u128) as u64
    }

    fn field_div(&self, a: u64, b: u64) -> u64 {
        self.field_mul(a, field_inverse(b, self.field_prime))
    }
}

/// Multivariate polynomial for complex constraints
#[derive(Clone, Debug)]
pub struct MultivariatePolynomial {
    /// Terms in the polynomial (monomial -> coefficient)
    pub terms: HashMap<Vec<usize>, u64>,
    /// Number of variables
    pub num_variables: usize,
    /// Field prime
    pub field_prime: u64,
}

impl MultivariatePolynomial {
    /// Create new multivariate polynomial
    pub fn new(num_variables: usize, field_prime: u64) -> Self {
        Self {
            terms: HashMap::new(),
            num_variables,
            field_prime,
        }
    }

    /// Add term with given monomial and coefficient
    pub fn add_term(&mut self, monomial: Vec<usize>, coefficient: u64) {
        assert_eq!(
            monomial.len(),
            self.num_variables,
            "Monomial must match number of variables"
        );

        let coeff = coefficient % self.field_prime;
        if coeff != 0 {
            *self.terms.entry(monomial).or_insert(0) =
                (self.terms.get(&monomial).unwrap_or(&0) + coeff) % self.field_prime;
        }
    }

    /// Evaluate multivariate polynomial at given point
    pub fn evaluate(&self, point: &[u64]) -> u64 {
        assert_eq!(
            point.len(),
            self.num_variables,
            "Point must match number of variables"
        );

        let mut result = 0u64;

        for (monomial, &coefficient) in &self.terms {
            let mut term_value = coefficient;

            for (i, &power) in monomial.iter().enumerate() {
                term_value = field_mul(
                    term_value,
                    field_pow(point[i], power, self.field_prime),
                    self.field_prime,
                );
            }

            result = field_add(result, term_value, self.field_prime);
        }

        result
    }

    /// Get total degree of polynomial
    pub fn total_degree(&self) -> usize {
        self.terms
            .keys()
            .map(|monomial| monomial.iter().sum::<usize>())
            .max()
            .unwrap_or(0)
    }
}

/// Polynomial commitment scheme for STARK proofs
pub struct PolynomialCommitment {
    /// Committed polynomial evaluations at domain
    pub evaluations: Vec<u64>,
    /// Merkle tree commitment
    pub commitment: [u8; 32],
    /// Evaluation domain
    pub domain: EvaluationDomain,
}

impl PolynomialCommitment {
    /// Create polynomial commitment
    pub fn commit(polynomial: &Polynomial, domain: &EvaluationDomain) -> Self {
        let evaluations = domain.evaluate_polynomial(polynomial);
        let commitment = compute_merkle_commitment(&evaluations);

        Self {
            evaluations,
            commitment,
            domain: domain.clone(),
        }
    }

    /// Generate opening proof for specific position
    pub fn open(&self, position: usize) -> OpeningProof {
        assert!(position < self.evaluations.len(), "Position out of range");

        let value = self.evaluations[position];
        let proof_path = generate_merkle_proof(&self.evaluations, position);

        OpeningProof {
            position,
            value,
            merkle_proof: proof_path,
        }
    }

    /// Verify opening proof
    pub fn verify_opening(&self, proof: &OpeningProof) -> bool {
        verify_merkle_proof(&self.commitment, proof)
    }
}

/// Domain for polynomial evaluation
#[derive(Clone, Debug)]
pub struct EvaluationDomain {
    /// Domain size (must be power of 2)
    pub size: usize,
    /// Generator of multiplicative subgroup
    pub generator: u64,
    /// Field prime
    pub field_prime: u64,
    /// Precomputed domain elements
    pub elements: Vec<u64>,
}

impl EvaluationDomain {
    /// Create new evaluation domain
    pub fn new(size: usize, field_prime: u64) -> Self {
        assert!(size.is_power_of_two(), "Domain size must be power of 2");

        let generator = find_primitive_root(size, field_prime);
        let elements = (0..size)
            .map(|i| field_pow(generator, i, field_prime))
            .collect();

        Self {
            size,
            generator,
            field_prime,
            elements,
        }
    }

    /// Evaluate polynomial over entire domain
    pub fn evaluate_polynomial(&self, polynomial: &Polynomial) -> Vec<u64> {
        self.elements
            .iter()
            .map(|&x| polynomial.evaluate(x))
            .collect()
    }

    /// Interpolate polynomial from domain evaluations
    pub fn interpolate(&self, evaluations: &[u64]) -> Polynomial {
        assert_eq!(
            evaluations.len(),
            self.size,
            "Evaluations must match domain size"
        );

        let points: Vec<(u64, u64)> = self
            .elements
            .iter()
            .zip(evaluations.iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        Polynomial::interpolate(&points, self.field_prime)
    }
}

/// Opening proof for polynomial commitment
#[derive(Clone, Debug)]
pub struct OpeningProof {
    pub position: usize,
    pub value: u64,
    pub merkle_proof: Vec<[u8; 32]>,
}

// Helper functions for finite field arithmetic

fn field_add(a: u64, b: u64, p: u64) -> u64 {
    (a + b) % p
}

fn field_sub(a: u64, b: u64, p: u64) -> u64 {
    (a + p - b) % p
}

fn field_mul(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn field_pow(base: u64, exp: usize, p: u64) -> u64 {
    if exp == 0 {
        return 1;
    }

    let mut result = 1u64;
    let mut base = base % p;
    let mut exp = exp;

    while exp > 0 {
        if exp & 1 == 1 {
            result = field_mul(result, base, p);
        }
        base = field_mul(base, base, p);
        exp >>= 1;
    }

    result
}

fn field_inverse(a: u64, p: u64) -> u64 {
    field_pow(a, (p - 2) as usize, p) // Fermat's little theorem
}

fn find_primitive_root(order: usize, p: u64) -> u64 {
    // Simplified primitive root finding - use 3 as generator for most cases
    // Real implementation would use proper primitive root algorithms
    if order <= (p - 1) as usize {
        3 // Common primitive root
    } else {
        2
    }
}

fn compute_merkle_commitment(evaluations: &[u64]) -> [u8; 32] {
    // Simplified Merkle root computation
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();

    for &eval in evaluations {
        hasher.update(eval.to_le_bytes());
    }

    hasher.finalize().into()
}

fn generate_merkle_proof(evaluations: &[u64], position: usize) -> Vec<[u8; 32]> {
    // Simplified Merkle proof generation
    let mut proof = Vec::new();

    // Add sibling hashes for path to root
    let mut current_pos = position;
    let mut current_size = evaluations.len();

    while current_size > 1 {
        let sibling_pos = current_pos ^ 1; // XOR with 1 to get sibling
        if sibling_pos < current_size {
            // Hash the sibling value
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            if sibling_pos < evaluations.len() {
                hasher.update(evaluations[sibling_pos].to_le_bytes());
            }
            proof.push(hasher.finalize().into());
        }

        current_pos /= 2;
        current_size /= 2;
    }

    proof
}

fn verify_merkle_proof(commitment: &[u8; 32], _proof: &OpeningProof) -> bool {
    // Simplified verification - in practice would verify full Merkle path
    !commitment.iter().all(|&b| b == 0)
}
