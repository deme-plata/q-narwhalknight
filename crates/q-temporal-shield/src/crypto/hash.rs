//! Hash functions for TemporalShield
//!
//! Uses BLAKE3 for all hashing needs (fast, secure, parallelizable).

use blake3::{Hasher, derive_key};

/// Hash data with BLAKE3
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Keyed hash (MAC) with BLAKE3
pub fn keyed_hash(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
    *blake3::keyed_hash(key, data).as_bytes()
}

/// Derive a key from context and input key material
pub fn derive_key_material(context: &str, ikm: &[u8]) -> [u8; 32] {
    derive_key(context, ikm)
}

/// Create a commitment to data with blinding
pub fn commit(data: &[u8], blinding: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.update(blinding);
    *hasher.finalize().as_bytes()
}

/// Hash multiple inputs together
pub fn hash_concat(inputs: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for input in inputs {
        hasher.update(input);
    }
    *hasher.finalize().as_bytes()
}

/// Domain-separated hash
pub fn domain_hash(domain: &str, data: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    // Length-prefixed domain
    hasher.update(&(domain.len() as u32).to_le_bytes());
    hasher.update(domain.as_bytes());
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

/// Extendable output for larger hash values
pub fn xof(data: &[u8], output_len: usize) -> Vec<u8> {
    let mut hasher = Hasher::new();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut output = vec![0u8; output_len];
    reader.fill(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_hash() {
        let data = b"test data";
        let hash1 = blake3_hash(data);
        let hash2 = blake3_hash(data);
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, [0u8; 32]);
    }

    #[test]
    fn test_keyed_hash() {
        let key = [1u8; 32];
        let data = b"test data";
        let hash1 = keyed_hash(&key, data);
        let hash2 = keyed_hash(&key, data);
        assert_eq!(hash1, hash2);

        let different_key = [2u8; 32];
        let hash3 = keyed_hash(&different_key, data);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_commitment() {
        let data = b"secret data";
        let blinding1 = [1u8; 32];
        let blinding2 = [2u8; 32];

        let commit1 = commit(data, &blinding1);
        let commit2 = commit(data, &blinding2);

        // Same data, different blinding = different commitment
        assert_ne!(commit1, commit2);

        // Same data and blinding = same commitment
        let commit3 = commit(data, &blinding1);
        assert_eq!(commit1, commit3);
    }

    #[test]
    fn test_domain_hash() {
        let data = b"test";
        let hash1 = domain_hash("domain1", data);
        let hash2 = domain_hash("domain2", data);
        assert_ne!(hash1, hash2);
    }
}
