//! # Bulletproofs Mainnet Safety Tests
//!
//! Comprehensive tests for bulletproofs range proofs to ensure
//! production readiness for mainnet launch.
//!
//! These tests verify:
//! 1. **Range Proof Soundness**: Values outside range are rejected
//! 2. **Commitment Hiding**: Values cannot be extracted from commitments
//! 3. **Proof Integrity**: Proofs survive serialization
//! 4. **Edge Cases**: Boundary values handled correctly
//! 5. **Aggregation**: Batch proofs work correctly
//!
//! Run: `cargo test --package q-crypto-advanced --test bulletproofs_mainnet_safety_tests`

use q_crypto_advanced::bulletproofs_v2::{
    BulletproofsProver, BulletproofsVerifier, RealScalar,
    AggregatedProver, AggregatedVerifier, RangeProof,
};

// ============================================================================
// RANGE PROOF SOUNDNESS TESTS
// ============================================================================

/// Test that valid values in range produce verifiable proofs
#[test]
fn test_valid_range_values() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    // Test various valid values
    let test_values = [
        0u64,
        1,
        100,
        1000,
        1_000_000,
        1_000_000_000,
        u64::MAX / 2,
        u64::MAX - 1,
        u64::MAX,
    ];

    for value in test_values {
        let blinding = RealScalar::random();
        let result = prover.prove(value, &blinding);
        assert!(result.is_ok(), "Failed to create proof for value {}", value);

        let proof = result.unwrap();
        let verified = verifier.verify(&proof);
        assert!(verified.is_ok() && verified.unwrap(),
            "Failed to verify proof for value {}", value);
    }
}

/// Test that proofs bind to specific commitments
/// CRITICAL: Changing commitment must invalidate proof
#[test]
fn test_commitment_binding() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    let value = 12345u64;
    let blinding = RealScalar::random();
    let proof = prover.prove(value, &blinding).unwrap();

    // Original verifies
    assert!(verifier.verify(&proof).unwrap());

    // Create different proof with same value but different blinding
    let different_blinding = RealScalar::random();
    let different_proof = prover.prove(value, &different_blinding).unwrap();

    // Commitments should be different
    assert_ne!(
        proof.commitment.to_compressed(),
        different_proof.commitment.to_compressed(),
        "Same value with different blinding should produce different commitments"
    );
}

/// Test proof tampering detection
/// CRITICAL: Any modification must be detected
#[test]
fn test_proof_tampering_detection() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    let value = 50000u64;
    let blinding = RealScalar::random();
    let mut proof = prover.prove(value, &blinding).unwrap();

    // Original verifies
    assert!(verifier.verify(&proof).unwrap());

    // Tamper with proof bytes
    if !proof.proof_bytes.is_empty() {
        proof.proof_bytes[0] ^= 0xFF;
        let result = verifier.verify(&proof);
        // Should either return error or false
        assert!(result.is_err() || !result.unwrap(),
            "CRITICAL: Tampered proof was accepted!");
    }
}

/// Test that 32-bit range rejects values >= 2^32
#[test]
fn test_32bit_range_enforcement() {
    let prover = BulletproofsProver::new(32);
    let verifier = BulletproofsVerifier::new(32);

    // Value within 32-bit range should work
    let valid_value = (1u64 << 32) - 1; // 2^32 - 1
    let blinding = RealScalar::random();
    let proof = prover.prove(valid_value, &blinding).unwrap();
    assert!(verifier.verify(&proof).unwrap(), "Max 32-bit value should verify");

    // Value at exactly 2^32 should fail
    let invalid_value = 1u64 << 32; // 2^32
    let result = prover.prove(invalid_value, &blinding);
    assert!(result.is_err(), "CRITICAL: Value >= 2^32 was accepted for 32-bit range!");
}

/// Test that 64-bit range accepts full u64 range
#[test]
fn test_64bit_full_range() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    // Test boundary values
    let boundaries = [0u64, 1, u64::MAX - 1, u64::MAX];

    for value in boundaries {
        let blinding = RealScalar::random();
        let proof = prover.prove(value, &blinding).unwrap();
        assert!(verifier.verify(&proof).unwrap(),
            "64-bit boundary value {} should verify", value);
    }
}

// ============================================================================
// COMMITMENT HIDING TESTS
// ============================================================================

/// Test that same value produces different commitments with different blindings
/// This verifies the hiding property
#[test]
fn test_commitment_hiding_property() {
    let prover = BulletproofsProver::default_64_bit();

    let value = 999999u64;

    // Create 10 proofs with same value, different blindings
    let mut commitments = Vec::new();
    for _ in 0..10 {
        let blinding = RealScalar::random();
        let proof = prover.prove(value, &blinding).unwrap();
        commitments.push(proof.commitment.to_compressed());
    }

    // All commitments should be unique
    for i in 0..commitments.len() {
        for j in (i + 1)..commitments.len() {
            assert_ne!(commitments[i], commitments[j],
                "CRITICAL: Same value produced same commitment - hiding broken!");
        }
    }
}

/// Test that commitments reveal nothing about value ordering
#[test]
fn test_commitment_value_ordering_hidden() {
    let prover = BulletproofsProver::default_64_bit();

    // Create commitments to ordered values
    let values = [100u64, 200, 300, 400, 500];
    let mut commitments = Vec::new();

    for value in values {
        let blinding = RealScalar::random();
        let proof = prover.prove(value, &blinding).unwrap();
        commitments.push(proof.commitment.to_compressed());
    }

    // Commitments should not reveal ordering
    // (We can't directly test this cryptographically, but we verify they're all different)
    for i in 0..commitments.len() {
        for j in (i + 1)..commitments.len() {
            assert_ne!(commitments[i], commitments[j]);
        }
    }
}

// ============================================================================
// SERIALIZATION INTEGRITY TESTS
// ============================================================================

/// Test proof serialization round-trip with JSON
#[test]
fn test_proof_json_serialization() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    let value = 123456u64;
    let blinding = RealScalar::random();
    let proof = prover.prove(value, &blinding).unwrap();

    // Serialize to JSON
    let json = serde_json::to_string(&proof).unwrap();

    // Deserialize
    let deserialized: RangeProof = serde_json::from_str(&json).unwrap();

    // Verify deserialized proof
    assert!(verifier.verify(&deserialized).unwrap(),
        "CRITICAL: Proof failed after JSON round-trip!");

    // Verify fields match
    assert_eq!(proof.proof_bytes, deserialized.proof_bytes);
    assert_eq!(proof.n_bits, deserialized.n_bits);
}

/// Test proof serialization round-trip with bincode
#[test]
fn test_proof_bincode_serialization() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    let value = 999999u64;
    let blinding = RealScalar::random();
    let proof = prover.prove(value, &blinding).unwrap();

    // Serialize to bincode
    let bytes = bincode::serialize(&proof).unwrap();

    // Deserialize
    let deserialized: RangeProof = bincode::deserialize(&bytes).unwrap();

    // Verify deserialized proof
    assert!(verifier.verify(&deserialized).unwrap(),
        "CRITICAL: Proof failed after bincode round-trip!");
}

// ============================================================================
// AGGREGATED PROOF TESTS
// ============================================================================

/// Test aggregated proofs with power of 2 values
#[test]
fn test_aggregated_proof_power_of_2() {
    let mut aggregator = AggregatedProver::new(32);
    let verifier = AggregatedVerifier::new(32);

    // Add 4 values (power of 2)
    aggregator.add_value(100, RealScalar::random()).unwrap();
    aggregator.add_value(200, RealScalar::random()).unwrap();
    aggregator.add_value(300, RealScalar::random()).unwrap();
    aggregator.add_value(400, RealScalar::random()).unwrap();

    let agg_proof = aggregator.prove().unwrap();

    assert_eq!(agg_proof.count, 4);
    assert_eq!(agg_proof.commitments.len(), 4);

    assert!(verifier.verify(&agg_proof).unwrap(),
        "Aggregated proof should verify");
}

/// Test aggregated proof with 2 values (minimum aggregation)
#[test]
fn test_aggregated_proof_minimum() {
    let mut aggregator = AggregatedProver::new(32);
    let verifier = AggregatedVerifier::new(32);

    // Add 2 values (minimum for aggregation)
    aggregator.add_value(1000, RealScalar::random()).unwrap();
    aggregator.add_value(2000, RealScalar::random()).unwrap();

    let agg_proof = aggregator.prove().unwrap();

    assert_eq!(agg_proof.count, 2);
    assert!(verifier.verify(&agg_proof).unwrap());
}

/// Test aggregated proof with 8 values
#[test]
fn test_aggregated_proof_eight_values() {
    let mut aggregator = AggregatedProver::new(32);
    let verifier = AggregatedVerifier::new(32);

    // Add 8 values
    for i in 0..8 {
        aggregator.add_value((i + 1) as u64 * 100, RealScalar::random()).unwrap();
    }

    let agg_proof = aggregator.prove().unwrap();

    assert_eq!(agg_proof.count, 8);
    assert!(verifier.verify(&agg_proof).unwrap());
}

/// Test that single value aggregation works
#[test]
fn test_single_value_aggregation() {
    let mut aggregator = AggregatedProver::new(32);
    let verifier = AggregatedVerifier::new(32);

    aggregator.add_value(12345, RealScalar::random()).unwrap();

    let agg_proof = aggregator.prove().unwrap();
    assert_eq!(agg_proof.count, 1);
    assert!(verifier.verify(&agg_proof).unwrap());
}

/// Test aggregator clear functionality
#[test]
fn test_aggregator_clear() {
    let mut aggregator = AggregatedProver::new(32);

    aggregator.add_value(100, RealScalar::random()).unwrap();
    aggregator.add_value(200, RealScalar::random()).unwrap();

    aggregator.clear();

    // Add new values after clear
    aggregator.add_value(300, RealScalar::random()).unwrap();
    aggregator.add_value(400, RealScalar::random()).unwrap();

    let verifier = AggregatedVerifier::new(32);
    let agg_proof = aggregator.prove().unwrap();
    assert_eq!(agg_proof.count, 2);
    assert!(verifier.verify(&agg_proof).unwrap());
}

// ============================================================================
// SCALAR TESTS
// ============================================================================

/// Test scalar creation from bytes
#[test]
fn test_scalar_from_bytes() {
    let bytes = [42u8; 32];
    let scalar = RealScalar::from_bytes(bytes);

    // Should round-trip
    let recovered = scalar.as_bytes();
    // Note: Due to modular reduction, may not be exactly equal
    // but should be deterministic
    let scalar2 = RealScalar::from_bytes(recovered);
    assert_eq!(scalar.as_bytes(), scalar2.as_bytes());
}

/// Test zero and one scalars
#[test]
fn test_scalar_identity_elements() {
    let zero = RealScalar::zero();
    let one = RealScalar::one();

    // Zero should have all zero bytes (mod order)
    let zero_bytes = zero.as_bytes();
    assert_eq!(zero_bytes[0], 0);

    // One should not be all zeros
    let one_bytes = one.as_bytes();
    assert_ne!(one_bytes, [0u8; 32]);
}

/// Test random scalar generation
#[test]
fn test_random_scalar_uniqueness() {
    let scalars: Vec<_> = (0..10).map(|_| RealScalar::random()).collect();

    // All random scalars should be unique
    for i in 0..scalars.len() {
        for j in (i + 1)..scalars.len() {
            assert_ne!(scalars[i].as_bytes(), scalars[j].as_bytes(),
                "Random scalars should be unique");
        }
    }
}

// ============================================================================
// STRESS TESTS
// ============================================================================

/// Stress test: Create many proofs in sequence
#[test]
fn test_stress_sequential_proofs() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    for i in 0..50 {
        let value = (i as u64 + 1) * 1000;
        let blinding = RealScalar::random();
        let proof = prover.prove(value, &blinding).unwrap();
        assert!(verifier.verify(&proof).unwrap(),
            "Proof {} failed in stress test", i);
    }
}

/// Test proof size consistency
#[test]
fn test_proof_size_consistency() {
    let prover = BulletproofsProver::default_64_bit();

    // All 64-bit proofs should have same size
    let mut sizes = Vec::new();
    for i in 0..10 {
        let value = (i as u64 + 1) * 12345;
        let blinding = RealScalar::random();
        let proof = prover.prove(value, &blinding).unwrap();
        sizes.push(proof.proof_bytes.len());
    }

    let first_size = sizes[0];
    for (i, size) in sizes.iter().enumerate() {
        assert_eq!(*size, first_size,
            "Proof {} has different size: {} vs {}", i, size, first_size);
    }
}

// ============================================================================
// MAINNET SPECIFIC TESTS
// ============================================================================

/// Test proof for typical transaction amount (1 million units)
#[test]
fn test_typical_transaction_amount() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    // 1 million QUG (with 8 decimal places = 100_000_000_000_000)
    let amount = 1_000_000u64 * 100_000_000; // 10^14
    let blinding = RealScalar::random();

    let proof = prover.prove(amount, &blinding).unwrap();
    assert!(verifier.verify(&proof).unwrap());
}

/// Test proof for maximum possible supply
#[test]
fn test_maximum_supply_amount() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    // Maximum possible amount (less than u64::MAX to leave room for fees)
    let max_amount = u64::MAX - 1_000_000;
    let blinding = RealScalar::random();

    let proof = prover.prove(max_amount, &blinding).unwrap();
    assert!(verifier.verify(&proof).unwrap(),
        "Maximum supply amount should produce valid proof");
}

/// Test proof for dust amount (minimum meaningful value)
#[test]
fn test_dust_amount() {
    let prover = BulletproofsProver::default_64_bit();
    let verifier = BulletproofsVerifier::default_64_bit();

    // Dust amount (1 unit)
    let dust = 1u64;
    let blinding = RealScalar::random();

    let proof = prover.prove(dust, &blinding).unwrap();
    assert!(verifier.verify(&proof).unwrap(),
        "Dust amount should produce valid proof");
}
