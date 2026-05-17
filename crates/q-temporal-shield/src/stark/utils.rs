//! Utility functions for STARK proofs

use winter_math::fields::f64::BaseElement;
use winter_math::FieldElement;

/// Convert bytes to field elements (for public inputs)
pub fn bytes_to_field_elements(bytes: &[u8], num_elements: usize) -> Vec<BaseElement> {
    let mut elements = Vec::with_capacity(num_elements);

    for chunk in bytes.chunks(8) {
        if elements.len() >= num_elements {
            break;
        }

        let mut arr = [0u8; 8];
        let len = chunk.len().min(8);
        arr[..len].copy_from_slice(&chunk[..len]);

        elements.push(BaseElement::new(u64::from_le_bytes(arr)));
    }

    // Pad with zeros if necessary
    while elements.len() < num_elements {
        elements.push(BaseElement::ZERO);
    }

    elements
}

/// Convert field elements back to bytes
pub fn field_elements_to_bytes(elements: &[BaseElement]) -> Vec<u8> {
    elements
        .iter()
        .flat_map(|e| {
            let value: u64 = (*e).into();
            value.to_le_bytes().to_vec()
        })
        .collect()
}

/// Compute proof size estimate
pub fn estimate_proof_size(
    trace_length: usize,
    num_columns: usize,
    blowup_factor: usize,
    num_queries: usize,
) -> usize {
    // Rough estimate based on STARK proof components:
    // - Trace commitment: ~32 bytes
    // - Constraint commitment: ~32 bytes
    // - FRI commitments: ~32 * log2(trace_length * blowup_factor)
    // - Query responses: num_queries * num_columns * log2(trace_length)

    let extended_length = trace_length * blowup_factor;
    let log_extended = (extended_length as f64).log2().ceil() as usize;

    let commitment_size = 32; // BLAKE3 hash
    let trace_commitment = commitment_size;
    let constraint_commitment = commitment_size;
    let fri_commitments = commitment_size * log_extended;

    // Query response size (simplified)
    let query_size = num_columns * 8 + log_extended * 32; // field element + Merkle path
    let total_query_size = num_queries * query_size;

    trace_commitment + constraint_commitment + fri_commitments + total_query_size
}

/// Format proof for human-readable display
pub fn format_proof_info(proof_bytes: &[u8]) -> String {
    let size_kb = proof_bytes.len() as f64 / 1024.0;
    format!(
        "STARK Proof:\n  Size: {:.2} KB ({} bytes)\n  NO TRUSTED SETUP",
        size_kb,
        proof_bytes.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_field_elements() {
        let bytes = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let elements = bytes_to_field_elements(&bytes, 2);

        assert_eq!(elements.len(), 2);
        // First element should be from first 8 bytes
        let expected1 = u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(elements[0], BaseElement::new(expected1));
    }

    #[test]
    fn test_estimate_proof_size() {
        let size = estimate_proof_size(1024, 12, 8, 28);
        // Should be a reasonable size (tens of KB)
        assert!(size > 1000);
        assert!(size < 1_000_000);
    }
}
