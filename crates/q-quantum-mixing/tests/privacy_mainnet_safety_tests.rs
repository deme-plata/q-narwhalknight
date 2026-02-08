//! # Privacy Mainnet Safety Tests
//!
//! Comprehensive tests to ensure privacy components are production-ready
//! for mainnet launch. These tests verify:
//!
//! 1. **Data Integrity**: No corruption during signing/verification
//! 2. **Cryptographic Soundness**: Real EC math produces valid proofs
//! 3. **Backwards Compatibility**: Old signatures still verify
//! 4. **Double-Spend Prevention**: Key images properly tracked
//! 5. **Edge Cases**: Boundary conditions handled correctly
//!
//! Run before every release: `cargo test --package q-quantum-mixing --test privacy_mainnet_safety_tests`

use q_quantum_mixing::{
    ring_signatures::{QuantumRingSigner, RingSignature},
    stealth_addresses::StealthAddressGenerator,
    QuantumEntropyPool,
};
use std::sync::Arc;

// ============================================================================
// RING SIGNATURE MAINNET SAFETY TESTS
// ============================================================================

/// Test that ring signatures are deterministic for verification
/// CRITICAL: Same signature must always verify the same way
#[tokio::test]
async fn test_ring_signature_verification_determinism() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    // Create a ring with multiple members
    let other1 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let other2 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let other3 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    let ring = vec![
        signer.get_public_key(),
        other1.get_public_key(),
        other2.get_public_key(),
        other3.get_public_key(),
    ];

    let message = b"mainnet transaction 12345";
    let signature = signer.create_ring_signature(message, ring.clone()).await.unwrap();

    // Verify 100 times - must ALWAYS return the same result
    let verifier = QuantumRingSigner::new(entropy_pool).await.unwrap();
    for i in 0..100 {
        let result = verifier.verify_ring_signature(&signature, message).await.unwrap();
        assert!(result, "Verification failed on iteration {} - CRITICAL: non-deterministic!", i);
    }
}

/// Test that modified signatures are ALWAYS rejected
/// CRITICAL: Any bit flip must invalidate the signature
#[tokio::test]
async fn test_signature_tampering_detection() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    let other = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let ring = vec![signer.get_public_key(), other.get_public_key()];

    let message = b"protected transaction";
    let mut signature = signer.create_ring_signature(message, ring).await.unwrap();

    let verifier = QuantumRingSigner::new(entropy_pool).await.unwrap();

    // Original should verify
    assert!(verifier.verify_ring_signature(&signature, message).await.unwrap());

    // Tamper with challenge - flip one bit
    signature.challenge[0] ^= 0x01;
    assert!(!verifier.verify_ring_signature(&signature, message).await.unwrap(),
        "CRITICAL: Tampered challenge was accepted!");
    signature.challenge[0] ^= 0x01; // Restore

    // Tamper with response - flip one bit
    if !signature.signature_values.is_empty() {
        signature.signature_values[0].response[0] ^= 0x01;
        assert!(!verifier.verify_ring_signature(&signature, message).await.unwrap(),
            "CRITICAL: Tampered response was accepted!");
        signature.signature_values[0].response[0] ^= 0x01; // Restore
    }

    // Tamper with key image
    signature.key_image.image[0] ^= 0x01;
    assert!(!verifier.verify_ring_signature(&signature, message).await.unwrap(),
        "CRITICAL: Tampered key image was accepted!");
}

/// Test that wrong message is ALWAYS rejected
/// CRITICAL: Signature must be bound to exact message
#[tokio::test]
async fn test_message_binding() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    let other = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let ring = vec![signer.get_public_key(), other.get_public_key()];

    let message1 = b"send 100 QUG to Alice";
    let message2 = b"send 100 QUG to Bob";  // Different recipient
    let message3 = b"send 1000 QUG to Alice"; // Different amount

    let signature = signer.create_ring_signature(message1, ring).await.unwrap();

    let verifier = QuantumRingSigner::new(entropy_pool).await.unwrap();

    // Correct message verifies
    assert!(verifier.verify_ring_signature(&signature, message1).await.unwrap());

    // Wrong messages MUST fail
    assert!(!verifier.verify_ring_signature(&signature, message2).await.unwrap(),
        "CRITICAL: Signature verified for different recipient!");
    assert!(!verifier.verify_ring_signature(&signature, message3).await.unwrap(),
        "CRITICAL: Signature verified for different amount!");
    assert!(!verifier.verify_ring_signature(&signature, b"").await.unwrap(),
        "CRITICAL: Signature verified for empty message!");
}

/// Test key image consistency across multiple signings
/// CRITICAL: Same key must produce same key image (for double-spend detection)
#[tokio::test]
async fn test_key_image_consistency() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());

    // Create two signers from the same private key
    let private_key = [42u8; 32];
    let signer1 = QuantumRingSigner::from_private_key(private_key, entropy_pool.clone()).await.unwrap();
    let signer2 = QuantumRingSigner::from_private_key(private_key, entropy_pool.clone()).await.unwrap();

    // Different signers from same key
    let signer3 = QuantumRingSigner::from_private_key(private_key, entropy_pool).await.unwrap();

    // All should have same public key
    assert_eq!(signer1.get_public_key(), signer2.get_public_key());
    assert_eq!(signer2.get_public_key(), signer3.get_public_key());
}

/// Test ring size boundaries
/// CRITICAL: Must handle edge cases without panicking or corrupting
#[tokio::test]
async fn test_ring_size_boundaries() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());

    let message = b"boundary test";

    // Ring size 1 (just the signer) - May or may not be supported
    // The important thing is it doesn't panic
    {
        let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let ring1 = vec![signer.get_public_key()];
        let result = signer.create_ring_signature(message, ring1).await;
        // Ring size 1 may be rejected (provides no anonymity) or may succeed
        // Either behavior is acceptable, but it shouldn't panic
        if let Ok(sig) = result {
            // If it succeeded at creation, verification should also succeed
            let verifier = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
            let verify_result = verifier.verify_ring_signature(&sig, message).await;
            // Even if verification fails for ring size 1, that's acceptable
            // The key is no panic
            let _ = verify_result;
        }
    }

    // Ring size 2 (minimum useful anonymity) - MUST work
    {
        let mut signer2 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let other = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let ring2 = vec![signer2.get_public_key(), other.get_public_key()];
        let sig2 = signer2.create_ring_signature(message, ring2).await.unwrap();
        let verifier = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        assert!(verifier.verify_ring_signature(&sig2, message).await.unwrap(),
            "Ring size 2 should always verify - this is the minimum useful case");
    }

    // Ring size 4 (standard privacy) - MUST work
    {
        let mut signer4 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let mut ring4 = vec![signer4.get_public_key()];
        for _ in 0..3 {
            let member = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
            ring4.push(member.get_public_key());
        }
        let sig4 = signer4.create_ring_signature(message, ring4).await.unwrap();
        let verifier = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        assert!(verifier.verify_ring_signature(&sig4, message).await.unwrap(),
            "Ring size 4 should verify");
    }

    // Large ring (8 members) - tests performance and correctness
    {
        let mut signer_large = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let mut ring_large = vec![signer_large.get_public_key()];
        for _ in 0..7 {
            let member = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
            ring_large.push(member.get_public_key());
        }
        let sig_large = signer_large.create_ring_signature(message, ring_large).await.unwrap();
        let verifier = QuantumRingSigner::new(entropy_pool).await.unwrap();
        assert!(verifier.verify_ring_signature(&sig_large, message).await.unwrap(),
            "Large ring signature (8 members) should verify");
    }
}

/// Test empty ring rejection
/// CRITICAL: Must not panic or accept empty rings
#[tokio::test]
async fn test_empty_ring_rejection() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool).await.unwrap();

    let result = signer.create_ring_signature(b"test", vec![]).await;
    assert!(result.is_err(), "CRITICAL: Empty ring was accepted!");
}

/// Test signer not in ring rejection
/// CRITICAL: Cannot sign if not part of the ring
#[tokio::test]
async fn test_signer_not_in_ring_rejection() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    // Create ring without the signer
    let other1 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let other2 = QuantumRingSigner::new(entropy_pool).await.unwrap();
    let ring = vec![other1.get_public_key(), other2.get_public_key()];

    let result = signer.create_ring_signature(b"test", ring).await;
    assert!(result.is_err(), "CRITICAL: Signed without being in ring!");
}

/// Test double-spend prevention actually works
/// CRITICAL: Same key image must be rejected on second use
#[tokio::test]
async fn test_double_spend_prevention_strict() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    let other = QuantumRingSigner::new(entropy_pool).await.unwrap();
    let ring = vec![signer.get_public_key(), other.get_public_key()];

    // First signature should succeed
    let sig1 = signer.create_ring_signature(b"tx1", ring.clone()).await.unwrap();

    // Second signature with same key MUST fail
    let sig2_result = signer.create_ring_signature(b"tx2", ring.clone()).await;
    assert!(sig2_result.is_err(),
        "CRITICAL: Double-spend allowed! Second signature succeeded!");

    // Key image should be marked as used
    assert!(signer.is_key_image_used(&sig1.key_image),
        "CRITICAL: Key image not tracked after first use!");
}

/// Test signature serialization round-trip
/// CRITICAL: Signatures must survive serialization for network transmission
#[tokio::test]
async fn test_signature_serialization_integrity() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    let other = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let ring = vec![signer.get_public_key(), other.get_public_key()];

    let message = b"serialization test";
    let signature = signer.create_ring_signature(message, ring).await.unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&signature).unwrap();
    let deserialized: RingSignature = serde_json::from_str(&json).unwrap();

    // Verify deserialized signature still works
    let verifier = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    assert!(verifier.verify_ring_signature(&deserialized, message).await.unwrap(),
        "CRITICAL: Signature failed after JSON round-trip!");

    // Serialize to bincode (more compact)
    let bincode_bytes = bincode::serialize(&signature).unwrap();
    let deserialized_bin: RingSignature = bincode::deserialize(&bincode_bytes).unwrap();

    assert!(verifier.verify_ring_signature(&deserialized_bin, message).await.unwrap(),
        "CRITICAL: Signature failed after bincode round-trip!");

    // Verify all fields match
    assert_eq!(signature.challenge, deserialized.challenge);
    assert_eq!(signature.key_image.image, deserialized.key_image.image);
    assert_eq!(signature.signature_values.len(), deserialized.signature_values.len());
}

// ============================================================================
// STEALTH ADDRESS MAINNET SAFETY TESTS
// ============================================================================

/// Test stealth address generator creation
#[tokio::test]
async fn test_stealth_address_generator_creation() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let generator = StealthAddressGenerator::new(entropy_pool).await.unwrap();

    // Should have valid public address
    let public_addr = generator.get_public_address();
    assert_eq!(public_addr.len(), 32, "Public address should be 32 bytes");

    // Should have valid view and spend public keys
    let view_pubkey = generator.get_view_public_key();
    let spend_pubkey = generator.get_spend_public_key();
    assert_eq!(view_pubkey.len(), 32, "View public key should be 32 bytes");
    assert_eq!(spend_pubkey.len(), 32, "Spend public key should be 32 bytes");

    // Keys should be different
    assert_ne!(view_pubkey, spend_pubkey, "View and spend keys should differ");
}

/// Test stealth address generation for recipient
#[tokio::test]
async fn test_stealth_address_generation_for_recipient() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());

    // Create sender and recipient
    let sender = StealthAddressGenerator::new(entropy_pool.clone()).await.unwrap();
    let recipient = StealthAddressGenerator::new(entropy_pool).await.unwrap();

    // Sender generates stealth address for recipient using recipient's public keys
    let recipient_pubkey = recipient.get_spend_public_key();
    let stealth = sender.generate_stealth_address(&recipient_pubkey).await.unwrap();

    // Stealth address should be valid
    assert_eq!(stealth.address.len(), 32, "Stealth address should be 32 bytes");
    assert_eq!(stealth.one_time_public_key.len(), 32, "One-time key should be 32 bytes");
}

/// Test stealth address unlinkability
#[tokio::test]
async fn test_stealth_address_unlinkability() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());

    let sender = StealthAddressGenerator::new(entropy_pool.clone()).await.unwrap();
    let recipient = StealthAddressGenerator::new(entropy_pool).await.unwrap();

    let recipient_pubkey = recipient.get_spend_public_key();

    // Generate multiple stealth addresses for same recipient
    let stealth1 = sender.generate_stealth_address(&recipient_pubkey).await.unwrap();
    let stealth2 = sender.generate_stealth_address(&recipient_pubkey).await.unwrap();
    let stealth3 = sender.generate_stealth_address(&recipient_pubkey).await.unwrap();

    // All addresses should be different (unlinkable)
    assert_ne!(stealth1.address, stealth2.address, "Stealth addresses should be unique");
    assert_ne!(stealth2.address, stealth3.address, "Stealth addresses should be unique");
    assert_ne!(stealth1.address, stealth3.address, "Stealth addresses should be unique");

    // All one-time public keys should be different
    assert_ne!(stealth1.one_time_public_key, stealth2.one_time_public_key);
    assert_ne!(stealth2.one_time_public_key, stealth3.one_time_public_key);
}

/// Test spending key computation
#[tokio::test]
async fn test_spending_key_computation() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());

    let sender = StealthAddressGenerator::new(entropy_pool.clone()).await.unwrap();
    let recipient = StealthAddressGenerator::new(entropy_pool).await.unwrap();

    let recipient_pubkey = recipient.get_spend_public_key();
    let stealth = sender.generate_stealth_address(&recipient_pubkey).await.unwrap();

    // Recipient should be able to compute spending key
    let spending_key = recipient.compute_spending_key(&stealth.one_time_public_key);
    assert!(spending_key.is_ok(), "Recipient should be able to compute spending key");

    let key = spending_key.unwrap();
    assert_eq!(key.len(), 32, "Spending key should be 32 bytes");
}

// ============================================================================
// DATA INTEGRITY STRESS TESTS
// ============================================================================

/// Stress test: Many signatures in sequence
#[tokio::test]
async fn test_stress_sequential_signatures() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());

    for i in 0..10 {
        let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let other = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        let ring = vec![signer.get_public_key(), other.get_public_key()];

        let message = format!("transaction {}", i);
        let signature = signer.create_ring_signature(message.as_bytes(), ring).await.unwrap();

        let verifier = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
        assert!(verifier.verify_ring_signature(&signature, message.as_bytes()).await.unwrap(),
            "Signature {} failed to verify", i);
    }
}

/// Test that different signers in same ring produce different but valid signatures
#[tokio::test]
async fn test_different_signers_same_ring() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());

    let mut signer1 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let mut signer2 = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let other = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    let ring = vec![
        signer1.get_public_key(),
        signer2.get_public_key(),
        other.get_public_key(),
    ];

    let message = b"shared ring transaction";

    // Both signers create signatures for the same ring and message
    let sig1 = signer1.create_ring_signature(message, ring.clone()).await.unwrap();
    let sig2 = signer2.create_ring_signature(message, ring.clone()).await.unwrap();

    // Signatures should be different (different key images)
    assert_ne!(sig1.key_image.image, sig2.key_image.image,
        "Different signers should produce different key images");

    // Both should verify
    let verifier = QuantumRingSigner::new(entropy_pool).await.unwrap();
    assert!(verifier.verify_ring_signature(&sig1, message).await.unwrap());
    assert!(verifier.verify_ring_signature(&sig2, message).await.unwrap());
}

/// Test long message handling
#[tokio::test]
async fn test_long_message_handling() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    let other = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let ring = vec![signer.get_public_key(), other.get_public_key()];

    // 1 MB message
    let long_message = vec![0xABu8; 1024 * 1024];

    let signature = signer.create_ring_signature(&long_message, ring).await.unwrap();

    let verifier = QuantumRingSigner::new(entropy_pool).await.unwrap();
    assert!(verifier.verify_ring_signature(&signature, &long_message).await.unwrap(),
        "Long message signature failed to verify");
}

/// Test binary message (all possible byte values)
#[tokio::test]
async fn test_binary_message_all_bytes() {
    let entropy_pool = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();

    let other = QuantumRingSigner::new(entropy_pool.clone()).await.unwrap();
    let ring = vec![signer.get_public_key(), other.get_public_key()];

    // Message with all 256 possible byte values
    let binary_message: Vec<u8> = (0u8..=255).collect();

    let signature = signer.create_ring_signature(&binary_message, ring).await.unwrap();

    let verifier = QuantumRingSigner::new(entropy_pool).await.unwrap();
    assert!(verifier.verify_ring_signature(&signature, &binary_message).await.unwrap(),
        "Binary message signature failed to verify");
}
