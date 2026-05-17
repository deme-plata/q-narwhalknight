//! Shamir Secret Sharing implementation
//!
//! Split secrets into shares and reconstruct from threshold.

use super::field::FieldElement256;
use super::polynomial::{Polynomial, lagrange_interpolate_at_zero};
use crate::error::{TemporalError, TemporalResult};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

/// A single share in the Shamir scheme
#[derive(Debug, Clone, Serialize, Deserialize, Zeroize)]
pub struct ShamirShare {
    /// The x-coordinate (trustee index, 1-indexed)
    pub index: u64,
    /// The share data (concatenated field elements)
    #[zeroize(skip)] // Vec doesn't implement Zeroize by default
    pub data: Vec<u8>,
}

impl ShamirShare {
    /// Get the share as field elements
    pub fn to_field_elements(&self) -> Vec<FieldElement256> {
        self.data
            .chunks(32)
            .map(|chunk| {
                let mut bytes = [0u8; 32];
                let len = chunk.len().min(32);
                bytes[32 - len..].copy_from_slice(&chunk[..len]);
                FieldElement256::from_bytes_32(bytes)
            })
            .collect()
    }

    /// Create from field elements
    pub fn from_field_elements(index: u64, elements: &[FieldElement256]) -> Self {
        let data = elements
            .iter()
            .flat_map(|e| e.to_bytes().to_vec())
            .collect();
        Self { index, data }
    }

    /// Get the size in bytes
    pub fn size(&self) -> usize {
        8 + self.data.len() // index + data
    }
}

/// Split a key into n shares with threshold k
///
/// # Arguments
/// * `key` - The secret key to split
/// * `threshold` - Minimum shares needed to reconstruct (k)
/// * `total_shares` - Total number of shares to create (n)
///
/// # Returns
/// Vector of n shares, any k of which can reconstruct the key
pub fn shamir_split(
    key: &[u8],
    threshold: usize,
    total_shares: usize,
) -> TemporalResult<Vec<ShamirShare>> {
    // Validate inputs
    if threshold == 0 || threshold > total_shares {
        return Err(TemporalError::InvalidThreshold {
            threshold,
            total_trustees: total_shares,
        });
    }

    if key.is_empty() {
        return Err(TemporalError::KeyLengthMismatch {
            key_len: 0,
            ciphertext_len: 0,
        });
    }

    // Split key into 31-byte chunks (to fit in field with room for reduction)
    let chunk_size = 31;
    let chunks: Vec<FieldElement256> = key
        .chunks(chunk_size)
        .map(|chunk| {
            let mut bytes = [0u8; 32];
            // Put chunk at the end (big-endian)
            bytes[32 - chunk.len()..].copy_from_slice(chunk);
            FieldElement256::from_bytes_32(bytes)
        })
        .collect();

    // For each chunk, create a polynomial and evaluate at points 1..=n
    let num_chunks = chunks.len();
    let mut all_shares: Vec<Vec<FieldElement256>> = vec![Vec::with_capacity(num_chunks); total_shares];

    for chunk in chunks {
        // Create random polynomial with chunk as constant term
        // Degree is (threshold - 1) so we need k points to reconstruct
        let poly = Polynomial::random_with_secret(chunk, threshold - 1)?;

        // Evaluate at points 1, 2, ..., n (not 0, since that's the secret)
        for i in 0..total_shares {
            let x = (i + 1) as u64; // 1-indexed
            let share_value = poly.evaluate_at(x);
            all_shares[i].push(share_value);
        }
    }

    // Convert to ShamirShare format
    let shares: Vec<ShamirShare> = all_shares
        .into_iter()
        .enumerate()
        .map(|(i, elements)| ShamirShare::from_field_elements((i + 1) as u64, &elements))
        .collect();

    Ok(shares)
}

/// Reconstruct a key from k or more shares
///
/// # Arguments
/// * `shares` - The shares to reconstruct from (at least k)
/// * `threshold` - The threshold k used during splitting
/// * `original_len` - The original key length (to trim padding)
///
/// # Returns
/// The reconstructed key
pub fn shamir_reconstruct(
    shares: &[ShamirShare],
    threshold: usize,
    original_len: usize,
) -> TemporalResult<Vec<u8>> {
    if shares.len() < threshold {
        return Err(TemporalError::InsufficientShares {
            have: shares.len(),
            need: threshold,
        });
    }

    // Take exactly k shares (any k will do)
    let shares_to_use: Vec<&ShamirShare> = shares.iter().take(threshold).collect();

    // Convert to field elements
    let share_elements: Vec<Vec<FieldElement256>> = shares_to_use
        .iter()
        .map(|s| s.to_field_elements())
        .collect();

    // Verify all shares have the same number of chunks
    let num_chunks = share_elements.first().map(|s| s.len()).unwrap_or(0);
    for elements in &share_elements {
        if elements.len() != num_chunks {
            return Err(TemporalError::ShareCommitmentMismatch { index: 0 });
        }
    }

    // Reconstruct each chunk
    let mut reconstructed_chunks: Vec<FieldElement256> = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        // Gather points for this chunk
        let points: Vec<(u64, FieldElement256)> = shares_to_use
            .iter()
            .zip(share_elements.iter())
            .map(|(share, elements)| (share.index, elements[chunk_idx].clone()))
            .collect();

        // Lagrange interpolation to get the constant term (the secret chunk)
        let secret_chunk = lagrange_interpolate_at_zero(&points)?;
        reconstructed_chunks.push(secret_chunk);
    }

    // Convert back to bytes
    let chunk_size = 31;
    let mut key_bytes: Vec<u8> = Vec::with_capacity(num_chunks * 32);

    for (i, chunk) in reconstructed_chunks.iter().enumerate() {
        let bytes = chunk.to_bytes();
        // Extract the original chunk (last chunk_size bytes, or less for the final chunk)
        let is_last = i == num_chunks - 1;
        let remaining = if is_last {
            original_len - (i * chunk_size)
        } else {
            chunk_size
        };

        // The chunk is stored big-endian, so take from the end
        key_bytes.extend_from_slice(&bytes[32 - remaining..]);
    }

    // Trim to original length
    key_bytes.truncate(original_len);

    Ok(key_bytes)
}

/// Verify that a share is consistent with its commitment
pub fn verify_share_commitment(share: &ShamirShare, commitment: &[u8; 32]) -> bool {
    let hash = blake3::hash(&share.data);
    hash.as_bytes() == commitment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_and_reconstruct() {
        let secret = b"This is a secret message for testing Shamir!";
        let threshold = 3;
        let total = 5;

        let shares = shamir_split(secret, threshold, total).unwrap();
        assert_eq!(shares.len(), total);

        // Reconstruct with exactly k shares
        let selected: Vec<ShamirShare> = shares[0..threshold].to_vec();
        let reconstructed = shamir_reconstruct(&selected, threshold, secret.len()).unwrap();
        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_reconstruct_with_different_shares() {
        let secret = b"Another secret to split";
        let threshold = 2;
        let total = 5;

        let shares = shamir_split(secret, threshold, total).unwrap();

        // Use shares 1 and 2
        let selected1: Vec<ShamirShare> = vec![shares[0].clone(), shares[1].clone()];
        let reconstructed1 = shamir_reconstruct(&selected1, threshold, secret.len()).unwrap();
        assert_eq!(reconstructed1, secret);

        // Use shares 3 and 5
        let selected2: Vec<ShamirShare> = vec![shares[2].clone(), shares[4].clone()];
        let reconstructed2 = shamir_reconstruct(&selected2, threshold, secret.len()).unwrap();
        assert_eq!(reconstructed2, secret);
    }

    #[test]
    fn test_insufficient_shares() {
        let secret = b"Secret";
        let threshold = 3;
        let total = 5;

        let shares = shamir_split(secret, threshold, total).unwrap();

        // Try with only 2 shares (need 3)
        let selected: Vec<ShamirShare> = shares[0..2].to_vec();
        let result = shamir_reconstruct(&selected, threshold, secret.len());
        assert!(result.is_err());
    }

    #[test]
    fn test_large_secret() {
        // Test with a secret larger than one chunk
        let secret: Vec<u8> = (0..=255).cycle().take(500).collect();
        let threshold = 3;
        let total = 5;

        let shares = shamir_split(&secret, threshold, total).unwrap();
        let selected: Vec<ShamirShare> = shares[0..threshold].to_vec();
        let reconstructed = shamir_reconstruct(&selected, threshold, secret.len()).unwrap();

        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_single_byte_secret() {
        let secret = vec![42u8];
        let threshold = 2;
        let total = 3;

        let shares = shamir_split(&secret, threshold, total).unwrap();
        let selected: Vec<ShamirShare> = shares[0..threshold].to_vec();
        let reconstructed = shamir_reconstruct(&selected, threshold, secret.len()).unwrap();

        assert_eq!(reconstructed, secret);
    }

    #[test]
    fn test_share_commitment() {
        let secret = b"Test";
        let shares = shamir_split(secret, 2, 3).unwrap();

        let commitment = *blake3::hash(&shares[0].data).as_bytes();
        assert!(verify_share_commitment(&shares[0], &commitment));

        // Wrong commitment should fail
        let wrong = [0u8; 32];
        assert!(!verify_share_commitment(&shares[0], &wrong));
    }
}
