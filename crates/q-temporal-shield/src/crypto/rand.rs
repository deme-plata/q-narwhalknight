//! Secure random number generation for TemporalShield
//!
//! Uses OS-provided cryptographic randomness.

use crate::error::{TemporalError, TemporalResult};

/// Fill a buffer with cryptographically secure random bytes
pub fn fill_random(buffer: &mut [u8]) -> TemporalResult<()> {
    getrandom::getrandom(buffer)
        .map_err(|e| TemporalError::RandomnessFailed(e.to_string()))
}

/// Generate a random 32-byte value
pub fn random_32() -> TemporalResult<[u8; 32]> {
    let mut bytes = [0u8; 32];
    fill_random(&mut bytes)?;
    Ok(bytes)
}

/// Generate a random 24-byte nonce
pub fn random_nonce() -> TemporalResult<[u8; 24]> {
    let mut bytes = [0u8; 24];
    fill_random(&mut bytes)?;
    Ok(bytes)
}

/// Generate random bytes of specified length
pub fn random_bytes(len: usize) -> TemporalResult<Vec<u8>> {
    let mut bytes = vec![0u8; len];
    fill_random(&mut bytes)?;
    Ok(bytes)
}

/// Generate a random u64
pub fn random_u64() -> TemporalResult<u64> {
    let mut bytes = [0u8; 8];
    fill_random(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_32() {
        let r1 = random_32().unwrap();
        let r2 = random_32().unwrap();
        // Extremely unlikely to be equal
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_random_nonce() {
        let n1 = random_nonce().unwrap();
        let n2 = random_nonce().unwrap();
        assert_ne!(n1, n2);
    }

    #[test]
    fn test_random_bytes() {
        let b1 = random_bytes(100).unwrap();
        let b2 = random_bytes(100).unwrap();
        assert_eq!(b1.len(), 100);
        assert_ne!(b1, b2);
    }
}
