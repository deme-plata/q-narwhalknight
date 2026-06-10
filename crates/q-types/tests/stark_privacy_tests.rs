//! Circle STARK Privacy Tests
//!
//! Tests for cryptographic security properties of Circle STARK transaction privacy:
//! - Proof generation completeness
//! - Proof verification soundness
//! - Commitment hiding
//! - Nullifier uniqueness for double-spend prevention

use q_types::privacy_layer::{
    PrivateTransactionBuilder, PrivateTransactionCommitments,
    NullifierSet, BalanceCommitment, PrivateTransaction,
    PrivacyError,
};

#[cfg(feature = "advanced-crypto")]
use q_types::privacy_layer::{verify_stark_proof, verify_private_transaction};

/// Test basic commitment generation
#[test]
fn test_commitment_generation() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    // All commitments should be non-zero
    assert_ne!(commitments.sender_commitment, [0u8; 32]);
    assert_ne!(commitments.receiver_commitment, [0u8; 32]);
    assert_ne!(commitments.amount_commitment, [0u8; 32]);
    assert_ne!(commitments.nullifier, [0u8; 32]);
}

/// Test commitment hiding - same values, different blinding produce different commitments
#[test]
fn test_commitment_hiding() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    // Generate two sets of commitments for same transaction
    let commitments1 = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    let commitments2 = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    // Same inputs should produce different commitments due to random blinding
    assert_ne!(
        commitments1.sender_commitment,
        commitments2.sender_commitment,
        "Random blinding should produce different sender commitments"
    );
    assert_ne!(
        commitments1.amount_commitment,
        commitments2.amount_commitment,
        "Random blinding should produce different amount commitments"
    );
}

/// Test nullifier uniqueness per transaction
#[test]
fn test_nullifier_uniqueness() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    // Same sender, same amount, but different commitments = different nullifiers
    let commitments1 = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    let commitments2 = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    // Nullifiers should be different (different blinding factors)
    assert_ne!(
        commitments1.nullifier,
        commitments2.nullifier,
        "Different blinding factors should produce different nullifiers"
    );
}

/// Test nullifier set tracks spent nullifiers
#[test]
fn test_nullifier_set_double_spend_detection() {
    let mut nullifier_set = NullifierSet::new();
    let nullifier = [42u8; 32];

    // First spend should succeed
    assert!(!nullifier_set.is_spent(&nullifier));
    assert!(nullifier_set.add_spent(nullifier, 100));
    assert!(nullifier_set.is_spent(&nullifier));

    // Second spend (double spend) should be detected
    assert!(!nullifier_set.add_spent(nullifier, 101));
    assert_eq!(nullifier_set.spent_at_height(&nullifier), Some(100));
}

/// Test balance commitment creation
#[test]
fn test_balance_commitment() {
    let address = [1u8; 32];
    let balance = 10000u128;
    let blinding = [2u8; 32];

    let commitment = BalanceCommitment::new(&address, balance, &blinding);

    assert_ne!(commitment.commitment, [0u8; 32]);
    assert_ne!(commitment.blinding_commitment, [0u8; 32]);
    assert_ne!(commitment.address_commitment, [0u8; 32]);
}

/// Test balance commitment hiding
#[test]
fn test_balance_commitment_hiding() {
    let address = [1u8; 32];
    let balance = 10000u128;
    let blinding1 = [2u8; 32];
    let blinding2 = [3u8; 32];

    let commitment1 = BalanceCommitment::new(&address, balance, &blinding1);
    let commitment2 = BalanceCommitment::new(&address, balance, &blinding2);

    // Same balance with different blinding should produce different commitments
    assert_ne!(
        commitment1.commitment,
        commitment2.commitment,
        "Different blinding should produce different commitments"
    );
}

/// Test execution trace generation
#[test]
fn test_execution_trace_structure() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    let trace = commitments.execution_trace();

    // Trace should have 5 rows
    assert_eq!(trace.len(), 5, "Execution trace should have 5 rows");

    // All rows should be non-empty
    for (i, row) in trace.iter().enumerate() {
        assert!(!row.is_empty(), "Row {} should not be empty", i);
    }
}

/// Test public inputs generation
#[test]
fn test_public_inputs() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    let public_inputs = commitments.public_inputs();

    // Should have 4 public inputs (sender, receiver, amount, nullifier commitments)
    assert_eq!(public_inputs.len(), 4, "Should have 4 public inputs");
}

/// Test finalize without proof (testing mode)
#[test]
fn test_finalize_without_proof() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    let tx = commitments.finalize_without_proof();

    assert_eq!(tx.tx_id, commitments.tx_id);
    assert_eq!(tx.nullifier, commitments.nullifier);
    assert!(tx.stark_proof.is_empty(), "No STARK proof in testing mode");
    assert_eq!(tx.fee, 10);
}

/// Test builder validation - missing sender
#[test]
fn test_builder_missing_sender() {
    let result = PrivateTransactionBuilder::new()
        .receiver([2u8; 32])
        .amount(1000)
        .build_commitments();

    assert!(result.is_err());
    match result {
        Err(PrivacyError::MissingSender) => {}
        _ => panic!("Expected MissingSender error"),
    }
}

/// Test builder validation - missing receiver
#[test]
fn test_builder_missing_receiver() {
    let result = PrivateTransactionBuilder::new()
        .sender([1u8; 32])
        .amount(1000)
        .build_commitments();

    assert!(result.is_err());
    match result {
        Err(PrivacyError::MissingReceiver) => {}
        _ => panic!("Expected MissingReceiver error"),
    }
}

/// Test builder validation - missing amount
#[test]
fn test_builder_missing_amount() {
    let result = PrivateTransactionBuilder::new()
        .sender([1u8; 32])
        .receiver([2u8; 32])
        .build_commitments();

    assert!(result.is_err());
    match result {
        Err(PrivacyError::MissingAmount) => {}
        _ => panic!("Expected MissingAmount error"),
    }
}

/// Test u128 amount handling in trace (v3.0.4 fix)
#[test]
fn test_u128_amount_handling() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    // Large amount that requires u128
    let large_amount: u128 = (1u128 << 100) + 12345;
    let large_balance: u128 = (1u128 << 100) + 50000;

    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(large_amount)
        .sender_balance(large_balance)
        .fee(100)
        .build_commitments()
        .unwrap();

    let trace = commitments.execution_trace();

    // Second row contains amount info: [amount_lo, amount_hi, balance_lo, balance_hi, ...]
    assert!(trace.len() >= 2, "Trace should have amount row");
    let amount_row = &trace[1];
    assert!(amount_row.len() >= 4, "Amount row should have split values");

    // Verify the amount is correctly split
    let amount_lo = amount_row[0];
    let amount_hi = amount_row[1];
    let reconstructed = (amount_hi as u128) << 64 | (amount_lo as u128);

    assert_eq!(
        reconstructed, large_amount,
        "u128 amount should be correctly split and reconstructable"
    );
}

/// Test memo field handling
#[test]
fn test_memo_handling() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];
    let memo = b"Test payment for services".to_vec();

    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .memo(memo.clone())
        .build_commitments()
        .unwrap();

    assert_eq!(commitments.memo(), memo.as_slice());
}

// ========================================================================
// Tests requiring advanced-crypto feature
// ========================================================================

/// Test STARK proof generation
/// Requires: cargo test --features advanced-crypto
#[cfg(feature = "advanced-crypto")]
#[test]
fn test_stark_proof_generation() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    let proof_result = commitments.generate_stark_proof();
    assert!(proof_result.is_ok(), "STARK proof should generate successfully");

    let proof = proof_result.unwrap();
    assert!(proof.len() > 100, "Proof should have substantial size");
}

/// Test STARK proof verification
/// Requires: cargo test --features advanced-crypto
#[cfg(feature = "advanced-crypto")]
#[test]
fn test_stark_proof_verification() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(1000)
        .sender_balance(5000)
        .fee(10)
        .build_commitments()
        .unwrap();

    // Finalize with STARK proof
    let tx = commitments.finalize_with_stark_proof().unwrap();

    // Verify the proof
    let result = verify_private_transaction(&tx);
    assert!(result.is_ok(), "Verification should not error");
    assert!(result.unwrap(), "Valid proof should verify");
}

/// Test invalid STARK proof is rejected
/// Requires: cargo test --features advanced-crypto
#[cfg(feature = "advanced-crypto")]
#[test]
fn test_invalid_stark_proof_rejected() {
    let tx = PrivateTransaction {
        tx_id: [1u8; 32],
        sender_commitment: [2u8; 32],
        receiver_commitment: [3u8; 32],
        amount_commitment: [4u8; 32],
        nullifier: [5u8; 32],
        stark_proof: vec![0xBA, 0xAD, 0xF0, 0x0D], // Invalid proof
        aegis_signature: Vec::new(),
        timestamp: chrono::Utc::now().timestamp(),
        fee: 100,
        encrypted_memo: Vec::new(),
    };

    let result = verify_private_transaction(&tx);
    // Should either error or return false
    let is_invalid = result.is_err() || !result.unwrap_or(true);
    assert!(is_invalid, "Invalid proof should not verify");
}

/// Test empty STARK proof is rejected
/// Requires: cargo test --features advanced-crypto
#[cfg(feature = "advanced-crypto")]
#[test]
fn test_empty_stark_proof_rejected() {
    let tx = PrivateTransaction {
        tx_id: [1u8; 32],
        sender_commitment: [2u8; 32],
        receiver_commitment: [3u8; 32],
        amount_commitment: [4u8; 32],
        nullifier: [5u8; 32],
        stark_proof: Vec::new(), // Empty proof
        aegis_signature: Vec::new(),
        timestamp: chrono::Utc::now().timestamp(),
        fee: 100,
        encrypted_memo: Vec::new(),
    };

    let result = verify_private_transaction(&tx);
    assert!(result.is_err(), "Empty proof should error");
}

/// Test full transaction roundtrip
/// Requires: cargo test --features advanced-crypto
#[cfg(feature = "advanced-crypto")]
#[test]
fn test_full_transaction_roundtrip() {
    let sender = [1u8; 32];
    let receiver = [2u8; 32];

    // Build commitments
    let commitments = PrivateTransactionBuilder::new()
        .sender(sender)
        .receiver(receiver)
        .amount(5000)
        .sender_balance(10000)
        .fee(100)
        .memo(b"Test roundtrip".to_vec())
        .build_commitments()
        .unwrap();

    // Finalize with STARK proof
    let tx = commitments.finalize_with_stark_proof().unwrap();

    // Verify all fields
    assert_eq!(tx.tx_id, commitments.tx_id);
    assert_eq!(tx.sender_commitment, commitments.sender_commitment);
    assert_eq!(tx.receiver_commitment, commitments.receiver_commitment);
    assert_eq!(tx.amount_commitment, commitments.amount_commitment);
    assert_eq!(tx.nullifier, commitments.nullifier);
    assert!(!tx.stark_proof.is_empty());
    assert_eq!(tx.fee, 100);
    assert_eq!(tx.encrypted_memo, b"Test roundtrip".to_vec());

    // Verify the proof
    assert!(verify_private_transaction(&tx).unwrap());
}
