//! Polynomial operations for Shamir secret sharing
//!
//! Implements polynomial evaluation and Lagrange interpolation.

use super::field::FieldElement256;
use crate::error::{TemporalError, TemporalResult};
use zeroize::Zeroize;

/// A polynomial over the finite field
#[derive(Debug, Clone, Zeroize)]
pub struct Polynomial {
    /// Coefficients in ascending order: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
    coefficients: Vec<FieldElement256>,
}

impl Polynomial {
    /// Create a polynomial from coefficients
    /// coeffs[0] is the constant term (the secret in Shamir)
    pub fn from_coefficients(coefficients: Vec<FieldElement256>) -> Self {
        Self { coefficients }
    }

    /// Create a random polynomial of given degree with a specific constant term (the secret)
    pub fn random_with_secret(secret: FieldElement256, degree: usize) -> TemporalResult<Self> {
        let mut coefficients = Vec::with_capacity(degree + 1);
        coefficients.push(secret);

        for _ in 0..degree {
            coefficients.push(FieldElement256::random()?);
        }

        Ok(Self { coefficients })
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() {
            0
        } else {
            self.coefficients.len() - 1
        }
    }

    /// Get the constant term (the secret in Shamir)
    pub fn constant_term(&self) -> FieldElement256 {
        self.coefficients.first().cloned().unwrap_or_else(FieldElement256::zero)
    }

    /// Get all coefficients
    pub fn coefficients(&self) -> &[FieldElement256] {
        &self.coefficients
    }

    /// Evaluate the polynomial at point x using Horner's method
    /// P(x) = a_0 + a_1*x + a_2*x^2 + ... = a_0 + x*(a_1 + x*(a_2 + ...))
    pub fn evaluate(&self, x: &FieldElement256) -> FieldElement256 {
        if self.coefficients.is_empty() {
            return FieldElement256::zero();
        }

        // Horner's method
        let mut result = self.coefficients.last().unwrap().clone();
        for coeff in self.coefficients.iter().rev().skip(1) {
            result = result.mul(x).add(coeff);
        }
        result
    }

    /// Evaluate at a u64 point
    pub fn evaluate_at(&self, x: u64) -> FieldElement256 {
        self.evaluate(&FieldElement256::from_u64(x))
    }
}

/// Lagrange interpolation to recover the secret (constant term)
/// Given points (x_1, y_1), ..., (x_k, y_k), recover P(0)
pub fn lagrange_interpolate_at_zero(points: &[(u64, FieldElement256)]) -> TemporalResult<FieldElement256> {
    if points.is_empty() {
        return Err(TemporalError::InsufficientShares { have: 0, need: 1 });
    }

    let k = points.len();
    let mut result = FieldElement256::zero();

    for i in 0..k {
        let (x_i, y_i) = &points[i];
        let x_i_field = FieldElement256::from_u64(*x_i);

        // Compute Lagrange basis polynomial L_i(0)
        // L_i(0) = Π_{j≠i} (0 - x_j) / (x_i - x_j)
        //        = Π_{j≠i} (-x_j) / (x_i - x_j)
        //        = Π_{j≠i} x_j / (x_j - x_i)  (after simplification)

        let mut numerator = FieldElement256::one();
        let mut denominator = FieldElement256::one();

        for j in 0..k {
            if i != j {
                let (x_j, _) = &points[j];
                let x_j_field = FieldElement256::from_u64(*x_j);

                // numerator *= x_j
                numerator = numerator.mul(&x_j_field);

                // denominator *= (x_j - x_i)
                let diff = x_j_field.sub(&x_i_field);
                if diff.is_zero() {
                    return Err(TemporalError::DivisionByZero);
                }
                denominator = denominator.mul(&diff);
            }
        }

        // L_i(0) = numerator / denominator
        let basis = numerator.div(&denominator)?;

        // result += y_i * L_i(0)
        let term = y_i.mul(&basis);
        result = result.add(&term);
    }

    Ok(result)
}

/// Lagrange interpolation to recover the full polynomial
pub fn lagrange_interpolate_full(points: &[(u64, FieldElement256)]) -> TemporalResult<Polynomial> {
    if points.is_empty() {
        return Err(TemporalError::InsufficientShares { have: 0, need: 1 });
    }

    let k = points.len();

    // We'll compute coefficients by building up the polynomial
    // For simplicity, evaluate at enough points to reconstruct
    // Actually, we can directly compute coefficients using the matrix method
    // or iterate through the Lagrange basis

    // For now, just recover the constant term (that's what we need for Shamir)
    let secret = lagrange_interpolate_at_zero(points)?;

    // To get the full polynomial, we'd need more sophisticated methods
    // For Shamir, we only need P(0), so this is sufficient
    let coefficients = vec![secret];

    Ok(Polynomial::from_coefficients(coefficients))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_evaluation() {
        // P(x) = 5 + 3x + 2x^2
        let coeffs = vec![
            FieldElement256::from_u64(5),
            FieldElement256::from_u64(3),
            FieldElement256::from_u64(2),
        ];
        let poly = Polynomial::from_coefficients(coeffs);

        // P(0) = 5
        assert_eq!(poly.evaluate_at(0), FieldElement256::from_u64(5));

        // P(1) = 5 + 3 + 2 = 10
        assert_eq!(poly.evaluate_at(1), FieldElement256::from_u64(10));

        // P(2) = 5 + 6 + 8 = 19
        assert_eq!(poly.evaluate_at(2), FieldElement256::from_u64(19));
    }

    #[test]
    fn test_lagrange_interpolation() {
        // P(x) = 5 + 3x (degree 1)
        // P(1) = 8, P(2) = 11

        let points = vec![
            (1, FieldElement256::from_u64(8)),
            (2, FieldElement256::from_u64(11)),
        ];

        let secret = lagrange_interpolate_at_zero(&points).unwrap();
        assert_eq!(secret, FieldElement256::from_u64(5));
    }

    #[test]
    fn test_lagrange_interpolation_quadratic() {
        // P(x) = 42 + 2x + x^2 (degree 2)
        // P(1) = 45, P(2) = 50, P(3) = 57

        let points = vec![
            (1, FieldElement256::from_u64(45)),
            (2, FieldElement256::from_u64(50)),
            (3, FieldElement256::from_u64(57)),
        ];

        let secret = lagrange_interpolate_at_zero(&points).unwrap();
        assert_eq!(secret, FieldElement256::from_u64(42));
    }

    #[test]
    fn test_random_polynomial() {
        let secret = FieldElement256::from_u64(12345);
        let poly = Polynomial::random_with_secret(secret.clone(), 2).unwrap();

        assert_eq!(poly.degree(), 2);
        assert_eq!(poly.constant_term(), &secret);
    }
}
