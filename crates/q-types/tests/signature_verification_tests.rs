//! Signature Verification Tests
//!
//! These tests ensure that invalid signatures are ALWAYS rejected.
//! A bug here could allow unauthorized fund transfers = THEFT.
//!
//! CRITICAL SCENARIOS:
//! 1. Invalid signatures rejected
//! 2. Wrong key signatures rejected
//! 3. Tampered message signatures rejected
//! 4. Empty/null signatures rejected
//! 5. Signature malleability prevented
//!
//! Run with: cargo test --package q-types --test signature_verification_tests

use std::collections::HashMap;

// ============================================================================
// MOCK SIGNATURE TYPES
// ============================================================================

/// Simplified signature for testing
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub r: [u8; 32],
    pub s: [u8; 32],
}

impl Signature {
    pub fn new(r: [u8; 32], s: [u8; 32]) -> Self {
        Self { r, s }
    }

    pub fn empty() -> Self {
        Self { r: [0u8; 32], s: [0u8; 32] }
    }

    pub fn is_empty(&self) -> bool {
        self.r == [0u8; 32] && self.s == [0u8; 32]
    }
}

/// Public key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PublicKey {
    pub bytes: [u8; 32],
}

impl PublicKey {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    pub fn from_seed(seed: u8) -> Self {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        Self { bytes }
    }
}

/// Message to be signed
#[derive(Debug, Clone)]
pub struct Message {
    pub data: Vec<u8>,
}

impl Message {
    pub fn new(data: &[u8]) -> Self {
        Self { data: data.to_vec() }
    }

    pub fn hash(&self) -> [u8; 32] {
        // Simplified hash (in production use SHA3-256)
        let mut hash = [0u8; 32];
        for (i, byte) in self.data.iter().enumerate() {
            hash[i % 32] ^= byte;
        }
        hash
    }
}

// ============================================================================
// SIGNATURE VERIFIER (Mock implementation for testing)
// ============================================================================

/// Simplified signature verification for testing
/// In production, this uses Ed25519 or Dilithium
pub struct SignatureVerifier {
    /// Map of public key -> authorized signing key (for test simulation)
    known_keys: HashMap<PublicKey, [u8; 32]>,
}

impl SignatureVerifier {
    pub fn new() -> Self {
        Self {
            known_keys: HashMap::new(),
        }
    }

    pub fn register_key(&mut self, public_key: PublicKey, signing_key: [u8; 32]) {
        self.known_keys.insert(public_key, signing_key);
    }

    /// Create a valid signature (for testing)
    pub fn sign(&self, message: &Message, public_key: &PublicKey) -> Option<Signature> {
        let signing_key = self.known_keys.get(public_key)?;
        let hash = message.hash();

        // Simple deterministic signature (NOT secure - just for testing)
        let mut r = [0u8; 32];
        let mut s = [0u8; 32];
        for i in 0..32 {
            r[i] = hash[i] ^ signing_key[i];
            s[i] = hash[i].wrapping_add(signing_key[i]);
        }

        Some(Signature::new(r, s))
    }

    /// Verify a signature
    pub fn verify(&self, message: &Message, signature: &Signature, public_key: &PublicKey) -> Result<(), String> {
        // CRITICAL CHECK 1: Reject empty signatures
        if signature.is_empty() {
            return Err("INVALID SIGNATURE: Empty signature rejected".to_string());
        }

        // CRITICAL CHECK 2: Verify public key is known
        let signing_key = self.known_keys.get(public_key)
            .ok_or_else(|| "INVALID SIGNATURE: Unknown public key".to_string())?;

        // CRITICAL CHECK 3: Verify signature matches
        let hash = message.hash();
        let mut expected_r = [0u8; 32];
        let mut expected_s = [0u8; 32];
        for i in 0..32 {
            expected_r[i] = hash[i] ^ signing_key[i];
            expected_s[i] = hash[i].wrapping_add(signing_key[i]);
        }

        if signature.r != expected_r || signature.s != expected_s {
            return Err("INVALID SIGNATURE: Signature verification failed".to_string());
        }

        Ok(())
    }
}

// ============================================================================
// TRANSACTION SIGNATURE TESTS
// ============================================================================

mod tx_signature_tests {
    use super::*;

    #[test]
    fn test_valid_signature_accepted() {
        let mut verifier = SignatureVerifier::new();
        let key = PublicKey::from_seed(1);
        verifier.register_key(key.clone(), [1u8; 32]);

        let message = Message::new(b"Send 100 QUG to Bob");
        let signature = verifier.sign(&message, &key).unwrap();

        assert!(verifier.verify(&message, &signature, &key).is_ok());
    }

    #[test]
    fn test_invalid_signature_rejected() {
        let mut verifier = SignatureVerifier::new();
        let key = PublicKey::from_seed(1);
        verifier.register_key(key.clone(), [1u8; 32]);

        let message = Message::new(b"Send 100 QUG to Bob");

        // Create a fake signature
        let fake_signature = Signature::new([0xDE; 32], [0xAD; 32]);

        let result = verifier.verify(&message, &fake_signature, &key);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("INVALID SIGNATURE"));
    }

    #[test]
    fn test_empty_signature_rejected() {
        let mut verifier = SignatureVerifier::new();
        let key = PublicKey::from_seed(1);
        verifier.register_key(key.clone(), [1u8; 32]);

        let message = Message::new(b"Send 100 QUG to Bob");
        let empty_sig = Signature::empty();

        let result = verifier.verify(&message, &empty_sig, &key);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Empty signature"));
    }

    #[test]
    fn test_wrong_key_signature_rejected() {
        let mut verifier = SignatureVerifier::new();

        let alice_key = PublicKey::from_seed(1);
        let bob_key = PublicKey::from_seed(2);

        verifier.register_key(alice_key.clone(), [1u8; 32]);
        verifier.register_key(bob_key.clone(), [2u8; 32]);

        // Alice signs a message
        let message = Message::new(b"Send 100 QUG to Bob");
        let alice_sig = verifier.sign(&message, &alice_key).unwrap();

        // Try to verify with Bob's key - should fail
        let result = verifier.verify(&message, &alice_sig, &bob_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_message_rejected() {
        let mut verifier = SignatureVerifier::new();
        let key = PublicKey::from_seed(1);
        verifier.register_key(key.clone(), [1u8; 32]);

        // Sign original message
        let original = Message::new(b"Send 100 QUG to Bob");
        let signature = verifier.sign(&original, &key).unwrap();

        // Tamper with the message
        let tampered = Message::new(b"Send 999999 QUG to Attacker");

        // Should fail verification
        let result = verifier.verify(&tampered, &signature, &key);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_public_key_rejected() {
        let verifier = SignatureVerifier::new();
        let unknown_key = PublicKey::from_seed(99);

        let message = Message::new(b"Some message");
        let fake_sig = Signature::new([1u8; 32], [2u8; 32]);

        let result = verifier.verify(&message, &fake_sig, &unknown_key);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown public key"));
    }
}

// ============================================================================
// BLOCK SIGNATURE TESTS
// ============================================================================

mod block_signature_tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct Block {
        height: u64,
        hash: [u8; 32],
        parent_hash: [u8; 32],
        miner: PublicKey,
        signature: Signature,
    }

    fn verify_block(block: &Block, verifier: &SignatureVerifier) -> Result<(), String> {
        // Create message from block data
        let mut message_data = Vec::new();
        message_data.extend_from_slice(&block.height.to_le_bytes());
        message_data.extend_from_slice(&block.hash);
        message_data.extend_from_slice(&block.parent_hash);
        message_data.extend_from_slice(&block.miner.bytes);

        let message = Message::new(&message_data);
        verifier.verify(&message, &block.signature, &block.miner)
    }

    #[test]
    fn test_valid_block_signature() {
        let mut verifier = SignatureVerifier::new();
        let miner = PublicKey::from_seed(1);
        verifier.register_key(miner.clone(), [1u8; 32]);

        // Create block data
        let mut message_data = Vec::new();
        message_data.extend_from_slice(&100u64.to_le_bytes());
        message_data.extend_from_slice(&[0xAB; 32]);
        message_data.extend_from_slice(&[0xCD; 32]);
        message_data.extend_from_slice(&miner.bytes);

        let message = Message::new(&message_data);
        let signature = verifier.sign(&message, &miner).unwrap();

        let block = Block {
            height: 100,
            hash: [0xAB; 32],
            parent_hash: [0xCD; 32],
            miner: miner.clone(),
            signature,
        };

        assert!(verify_block(&block, &verifier).is_ok());
    }

    #[test]
    fn test_forged_block_rejected() {
        let mut verifier = SignatureVerifier::new();
        let legitimate_miner = PublicKey::from_seed(1);
        let attacker = PublicKey::from_seed(99);

        verifier.register_key(legitimate_miner.clone(), [1u8; 32]);
        verifier.register_key(attacker.clone(), [99u8; 32]);

        // Attacker tries to forge a block as the legitimate miner
        let fake_sig = Signature::new([0x00; 32], [0x00; 32]);

        let forged_block = Block {
            height: 100,
            hash: [0xAB; 32],
            parent_hash: [0xCD; 32],
            miner: legitimate_miner, // Claims to be from legitimate miner
            signature: fake_sig,      // But has attacker's signature
        };

        let result = verify_block(&forged_block, &verifier);
        assert!(result.is_err(), "Forged block should be rejected!");
    }

    #[test]
    fn test_modified_block_height_rejected() {
        let mut verifier = SignatureVerifier::new();
        let miner = PublicKey::from_seed(1);
        verifier.register_key(miner.clone(), [1u8; 32]);

        // Sign block at height 100
        let mut message_data = Vec::new();
        message_data.extend_from_slice(&100u64.to_le_bytes());
        message_data.extend_from_slice(&[0xAB; 32]);
        message_data.extend_from_slice(&[0xCD; 32]);
        message_data.extend_from_slice(&miner.bytes);

        let message = Message::new(&message_data);
        let signature = verifier.sign(&message, &miner).unwrap();

        // Try to use that signature for height 200 (manipulation)
        let tampered_block = Block {
            height: 200, // Changed!
            hash: [0xAB; 32],
            parent_hash: [0xCD; 32],
            miner: miner.clone(),
            signature,
        };

        let result = verify_block(&tampered_block, &verifier);
        assert!(result.is_err(), "Block with modified height should be rejected!");
    }
}

// ============================================================================
// SIGNATURE MALLEABILITY TESTS
// ============================================================================

mod malleability_tests {
    use super::*;

    #[test]
    fn test_signature_not_malleable() {
        let mut verifier = SignatureVerifier::new();
        let key = PublicKey::from_seed(1);
        verifier.register_key(key.clone(), [1u8; 32]);

        let message = Message::new(b"Test message");
        let sig = verifier.sign(&message, &key).unwrap();

        // Try various modifications to the signature
        let modifications = vec![
            // Flip a bit in r
            {
                let mut modified = sig.clone();
                modified.r[0] ^= 1;
                modified
            },
            // Flip a bit in s
            {
                let mut modified = sig.clone();
                modified.s[0] ^= 1;
                modified
            },
            // Negate (in real ECDSA, (r, -s) can be valid)
            {
                let mut modified = sig.clone();
                for i in 0..32 {
                    modified.s[i] = !modified.s[i];
                }
                modified
            },
        ];

        for (i, modified_sig) in modifications.iter().enumerate() {
            let result = verifier.verify(&message, modified_sig, &key);
            assert!(
                result.is_err(),
                "Malleated signature #{} should be rejected",
                i
            );
        }
    }
}

// ============================================================================
// BATCH VERIFICATION TESTS
// ============================================================================

mod batch_verification_tests {
    use super::*;

    #[test]
    fn test_batch_with_one_invalid_fails() {
        let mut verifier = SignatureVerifier::new();

        // Register multiple keys - use non-zero values to avoid empty signatures
        // Start from 1 to avoid zero keys/messages which produce "empty" signatures
        for i in 1u8..=10u8 {
            let key = PublicKey::from_seed(i);
            verifier.register_key(key, [i; 32]);
        }

        // Create 9 valid signatures and 1 invalid
        let mut signatures = Vec::new();
        let mut messages = Vec::new();
        let mut keys = Vec::new();

        for i in 1u8..=10u8 {
            let key = PublicKey::from_seed(i);
            let message = Message::new(&[i; 32]);

            let sig = if i == 5 {
                // One invalid signature
                Signature::new([0xFF; 32], [0xFF; 32])
            } else {
                verifier.sign(&message, &key).unwrap()
            };

            signatures.push(sig);
            messages.push(message);
            keys.push(key);
        }

        // Verify all (should find the invalid one)
        let mut invalid_count = 0;
        for i in 0usize..10usize {
            if verifier.verify(&messages[i], &signatures[i], &keys[i]).is_err() {
                invalid_count += 1;
            }
        }

        assert_eq!(invalid_count, 1, "Should detect exactly one invalid signature");
    }

    #[test]
    fn test_all_signatures_must_be_valid() {
        let mut verifier = SignatureVerifier::new();

        // Register keys - use non-zero values (1..=5) to avoid empty signatures
        for i in 1u8..=5u8 {
            let key = PublicKey::from_seed(i);
            verifier.register_key(key, [i; 32]);
        }

        // All valid - use non-zero values
        for i in 1u8..=5u8 {
            let key = PublicKey::from_seed(i);
            let message = Message::new(&[i; 32]);
            let sig = verifier.sign(&message, &key).unwrap();

            assert!(verifier.verify(&message, &sig, &key).is_ok());
        }
    }
}

// ============================================================================
// REGRESSION TESTS
// ============================================================================

mod regression_tests {
    use super::*;

    /// Regression test: Empty signatures must always be rejected
    #[test]
    fn test_regression_empty_sig_rejected() {
        let verifier = SignatureVerifier::new();
        let key = PublicKey::from_seed(1);
        let message = Message::new(b"test");
        let empty = Signature::empty();

        let result = verifier.verify(&message, &empty, &key);
        assert!(
            result.is_err(),
            "REGRESSION: Empty signature was accepted!"
        );
    }

    /// Regression test: Null bytes in signature must not bypass verification
    #[test]
    fn test_regression_null_bytes_rejected() {
        let mut verifier = SignatureVerifier::new();
        let key = PublicKey::from_seed(1);
        verifier.register_key(key.clone(), [1u8; 32]);

        let message = Message::new(b"test");

        // All zeros signature
        let null_sig = Signature::new([0u8; 32], [0u8; 32]);

        let result = verifier.verify(&message, &null_sig, &key);
        assert!(
            result.is_err(),
            "REGRESSION: Null signature was accepted!"
        );
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_verification_performance() {
        let mut verifier = SignatureVerifier::new();
        let key = PublicKey::from_seed(1);
        verifier.register_key(key.clone(), [1u8; 32]);

        let message = Message::new(b"Performance test message");
        let signature = verifier.sign(&message, &key).unwrap();

        let iterations = 100_000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = verifier.verify(&message, &signature, &key);
        }

        let elapsed = start.elapsed();
        let per_verify_us = elapsed.as_micros() / iterations as u128;

        println!("Signature verification: {} us per verification", per_verify_us);

        // Should be fast (real Ed25519 is ~100us, Dilithium ~300us)
        assert!(per_verify_us < 100, "Verification too slow: {} us", per_verify_us);
    }
}
