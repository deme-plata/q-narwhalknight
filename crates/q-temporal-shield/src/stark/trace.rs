//! Execution trace generation for STARK proofs
//!
//! Builds the trace table that will be proven.

use winter_math::fields::f64::BaseElement;
use winter_math::FieldElement;
use winter_prover::matrix::ColMatrix;

use super::air::{TRACE_WIDTH, COEFF_START, ACC_START, SHARE_START, AUX_START, compute_trace_length};
use crate::shamir::FieldElement256;
use crate::error::{TemporalError, TemporalResult};

/// Build the execution trace for Shamir consistency proof
pub fn build_shamir_trace(
    secret_chunks: &[FieldElement256],
    polynomial_coefficients: &[Vec<FieldElement256>], // coeffs for each chunk
    shares: &[Vec<FieldElement256>], // shares for each trustee (each contains all chunks)
    threshold: usize,
    total_trustees: usize,
) -> TemporalResult<ColMatrix<BaseElement>> {
    let num_chunks = secret_chunks.len();
    let trace_length = compute_trace_length(threshold, total_trustees, num_chunks);

    // Initialize trace columns
    let mut trace = vec![vec![BaseElement::ZERO; trace_length]; TRACE_WIDTH];

    let mut row = 0;

    // For each chunk
    for chunk_idx in 0..num_chunks {
        let coeffs = &polynomial_coefficients[chunk_idx];

        // For each trustee (share)
        for trustee_idx in 0..total_trustees {
            let x = BaseElement::new((trustee_idx + 1) as u64);
            let share_value = &shares[trustee_idx][chunk_idx];

            // Horner's method: evaluate polynomial at x
            // P(x) = c_0 + x*(c_1 + x*(c_2 + ... ))
            let mut acc = BaseElement::ZERO;

            // Process coefficients from highest to lowest degree
            for (coeff_idx, coeff) in coeffs.iter().rev().enumerate() {
                if row >= trace_length {
                    break;
                }

                // Convert coefficient to field element
                let coeff_fe = field256_to_base(coeff);

                // Store coefficient
                trace[COEFF_START][row] = coeff_fe;

                // Store accumulator state
                trace[ACC_START][row] = acc;

                // Horner step: acc = acc * x + coeff
                acc = acc * x + coeff_fe;

                // Store auxiliary info
                trace[AUX_START][row] = x;
                trace[AUX_START + 1][row] = BaseElement::new(coeff_idx as u64);

                row += 1;
            }

            // Store final share value
            if row > 0 && row <= trace_length {
                let share_fe = field256_to_base(share_value);
                trace[SHARE_START][row - 1] = share_fe;

                // Verify: acc should equal share_value
                // (This is what the STARK proves)
            }
        }
    }

    // Fill remaining rows with zeros (padding to power of 2)
    // The constraints will be satisfied trivially for these rows

    Ok(ColMatrix::new(trace))
}

/// Convert FieldElement256 to BaseElement (truncate to 64 bits)
fn field256_to_base(fe: &FieldElement256) -> BaseElement {
    let bytes = fe.to_bytes();
    // Take the last 8 bytes (least significant)
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[24..32]);
    BaseElement::new(u64::from_be_bytes(arr))
}

/// Build a simplified trace for testing/demonstration
pub fn build_simple_trace(
    secret: &[u8],
    threshold: usize,
    total_trustees: usize,
) -> TemporalResult<ColMatrix<BaseElement>> {
    // Convert secret to field elements
    let secret_chunks: Vec<FieldElement256> = secret
        .chunks(31)
        .map(|chunk| {
            let mut bytes = [0u8; 32];
            bytes[32 - chunk.len()..].copy_from_slice(chunk);
            FieldElement256::from_bytes_32(bytes)
        })
        .collect();

    // Generate random polynomials
    let mut polynomial_coefficients = Vec::new();
    for chunk in &secret_chunks {
        let poly = crate::shamir::polynomial::Polynomial::random_with_secret(
            chunk.clone(),
            threshold - 1,
        )?;
        polynomial_coefficients.push(poly.coefficients().to_vec());
    }

    // Evaluate shares
    let mut shares = vec![Vec::new(); total_trustees];
    for (chunk_idx, coeffs) in polynomial_coefficients.iter().enumerate() {
        for trustee_idx in 0..total_trustees {
            let x = FieldElement256::from_u64((trustee_idx + 1) as u64);

            // Evaluate polynomial at x
            let poly = crate::shamir::polynomial::Polynomial::from_coefficients(coeffs.clone());
            let share_value = poly.evaluate(&x);

            shares[trustee_idx].push(share_value);
        }
    }

    build_shamir_trace(
        &secret_chunks,
        &polynomial_coefficients,
        &shares,
        threshold,
        total_trustees,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field256_to_base() {
        let fe = FieldElement256::from_u64(12345);
        let base = field256_to_base(&fe);
        assert_eq!(base, BaseElement::new(12345));
    }

    #[test]
    fn test_build_simple_trace() {
        let secret = b"test secret";
        let trace = build_simple_trace(secret, 2, 3).unwrap();

        // Verify trace dimensions
        assert_eq!(trace.num_cols(), TRACE_WIDTH);
        assert!(trace.num_rows().is_power_of_two());
    }
}
