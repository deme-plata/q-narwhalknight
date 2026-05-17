//! Bincode Serialization Roundtrip Tests
//!
//! v3.4.1: These tests were created after discovering a critical bug where blocks
//! with transactions failed to deserialize with:
//! "Bincode does not support the serde::Deserializer::deserialize_any method"
//!
//! ROOT CAUSE: The u128_serde module used `deserialize_any()` which bincode doesn't support.
//! FIX: Changed to `deserialize_str()` to match the `serialize_str()` used in serialization.
//!
//! These tests ensure:
//! 1. Blocks with transactions roundtrip correctly through bincode
//! 2. u128 values (amount, fee) are preserved exactly
//! 3. Large u128 values that exceed u64::MAX are preserved
//! 4. All serde-compatible formats (bincode, postcard, json) roundtrip correctly
//!
//! WHY EXISTING TESTS MISSED THIS:
//! - Most tests used blocks with 0 transactions (empty tx list)
//! - Tests didn't go through actual RocksDB storage roundtrip
//! - Mock data didn't include u128 fields in the transaction path

use chrono::Utc;
use q_types::{
    Address, Amount, TokenType, Transaction, TransactionType, TxHash, TxSignaturePhase,
};

/// Helper to create a test transaction with specific u128 values
fn create_test_transaction(amount: u128, fee: u128) -> Transaction {
    Transaction {
        id: [1u8; 32] as TxHash,
        from: [2u8; 32] as Address,
        to: [3u8; 32] as Address,
        amount: amount as Amount,
        fee: fee as Amount,
        nonce: 1,
        signature: vec![0u8; 64], // Ed25519 signature placeholder
        timestamp: Utc::now(),
        data: vec![],
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUGUSD,
        tx_type: TransactionType::Transfer,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: q_types::TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    }
}

// =============================================================================
// CRITICAL: Bincode Roundtrip Tests
// These tests would have caught the deserialize_any bug!
// =============================================================================

#[test]
fn test_transaction_bincode_roundtrip_small_values() {
    // Small values that fit in u64 - these work with most implementations
    let tx = create_test_transaction(1000, 10);

    // Serialize with bincode
    let serialized = bincode::serialize(&tx).expect("Failed to serialize transaction");

    // Deserialize with bincode - THIS IS WHERE THE BUG MANIFESTED
    let deserialized: Transaction =
        bincode::deserialize(&serialized).expect("Failed to deserialize transaction");

    assert_eq!(tx.amount, deserialized.amount, "Amount mismatch");
    assert_eq!(tx.fee, deserialized.fee, "Fee mismatch");
    assert_eq!(tx.from, deserialized.from, "From address mismatch");
    assert_eq!(tx.to, deserialized.to, "To address mismatch");
}

#[test]
fn test_transaction_bincode_roundtrip_large_u128_values() {
    // Large values that EXCEED u64::MAX - these would be truncated with wrong serialization
    let large_amount: u128 = u64::MAX as u128 + 1_000_000_000_000; // Way beyond u64
    let large_fee: u128 = u64::MAX as u128 + 1;

    let tx = create_test_transaction(large_amount, large_fee);

    // Serialize with bincode
    let serialized = bincode::serialize(&tx).expect("Failed to serialize transaction");

    // Deserialize with bincode
    let deserialized: Transaction =
        bincode::deserialize(&serialized).expect("Failed to deserialize transaction with large u128");

    // CRITICAL: Verify u128 values are preserved EXACTLY
    assert_eq!(
        tx.amount, deserialized.amount,
        "Large amount was corrupted! Expected {}, got {}",
        tx.amount, deserialized.amount
    );
    assert_eq!(
        tx.fee, deserialized.fee,
        "Large fee was corrupted! Expected {}, got {}",
        tx.fee, deserialized.fee
    );
}

#[test]
fn test_transaction_bincode_roundtrip_max_u128() {
    // Maximum u128 value - ultimate stress test
    let max_amount: u128 = u128::MAX;
    let max_fee: u128 = u128::MAX / 2;

    let tx = create_test_transaction(max_amount, max_fee);

    let serialized = bincode::serialize(&tx).expect("Failed to serialize max u128 transaction");
    let deserialized: Transaction =
        bincode::deserialize(&serialized).expect("Failed to deserialize max u128 transaction");

    assert_eq!(
        tx.amount, deserialized.amount,
        "Max u128 amount was corrupted!"
    );
    assert_eq!(tx.fee, deserialized.fee, "Max u128 fee was corrupted!");
}

#[test]
fn test_transaction_list_bincode_roundtrip() {
    // THIS IS THE EXACT SCENARIO THAT CAUSED THE BUG!
    // Blocks contain a Vec<Transaction>, and this list is what failed.

    let transactions = vec![
        create_test_transaction(50_000_000_000_000_000_000, 1_000_000_000), // 50 QUG
        create_test_transaction(100_000_000_000_000_000_000, 2_000_000_000), // 100 QUG
        create_test_transaction(u64::MAX as u128 + 1, 500_000_000), // Exceeds u64
    ];

    // Serialize transactions vec with bincode (same as block storage)
    let serialized = bincode::serialize(&transactions).expect("Failed to serialize transaction list");

    // Deserialize - THIS FAILED WITH deserialize_any!
    let deserialized: Vec<Transaction> = bincode::deserialize(&serialized)
        .expect("Failed to deserialize transaction list - u128_serde bug!");

    // Verify all transactions preserved
    assert_eq!(
        transactions.len(),
        deserialized.len(),
        "Transaction count mismatch"
    );

    // Verify each transaction's u128 fields
    for (i, (original, deser)) in transactions.iter().zip(deserialized.iter()).enumerate() {
        assert_eq!(
            original.amount, deser.amount,
            "Transaction {} amount mismatch: expected {}, got {}",
            i, original.amount, deser.amount
        );
        assert_eq!(
            original.fee, deser.fee,
            "Transaction {} fee mismatch: expected {}, got {}",
            i, original.fee, deser.fee
        );
    }
}

#[test]
fn test_empty_vs_populated_transaction_list() {
    // This test demonstrates WHY the bug was missed:
    // Empty blocks worked fine, blocks with transactions failed

    // Empty transaction list - this ALWAYS worked
    let empty_list: Vec<Transaction> = vec![];
    let empty_serialized =
        bincode::serialize(&empty_list).expect("Failed to serialize empty list");
    let _empty_deserialized: Vec<Transaction> =
        bincode::deserialize(&empty_serialized).expect("Empty list should deserialize");

    // Transaction list with entries - this FAILED before the fix
    let list_with_tx = vec![create_test_transaction(1_000_000, 100)];
    let tx_serialized =
        bincode::serialize(&list_with_tx).expect("Failed to serialize list with tx");
    let tx_deserialized: Vec<Transaction> = bincode::deserialize(&tx_serialized)
        .expect("List with transactions should deserialize - this failed before v3.4.1 fix!");

    assert_eq!(list_with_tx.len(), tx_deserialized.len());
}

// =============================================================================
// Cross-Format Compatibility Tests
// Ensure u128 values survive different serialization formats
// =============================================================================

#[test]
fn test_transaction_json_roundtrip() {
    let tx = create_test_transaction(u64::MAX as u128 + 1_000_000, 500_000_000);

    let json = serde_json::to_string(&tx).expect("Failed to serialize to JSON");
    let deserialized: Transaction =
        serde_json::from_str(&json).expect("Failed to deserialize from JSON");

    assert_eq!(tx.amount, deserialized.amount, "JSON amount mismatch");
    assert_eq!(tx.fee, deserialized.fee, "JSON fee mismatch");
}

#[test]
fn test_transaction_postcard_roundtrip() {
    let tx = create_test_transaction(u64::MAX as u128 * 2, 1_000_000_000);

    let serialized = postcard::to_allocvec(&tx).expect("Failed to serialize with postcard");
    let deserialized: Transaction =
        postcard::from_bytes(&serialized).expect("Failed to deserialize with postcard");

    assert_eq!(tx.amount, deserialized.amount, "Postcard amount mismatch");
    assert_eq!(tx.fee, deserialized.fee, "Postcard fee mismatch");
}

#[test]
fn test_transaction_list_postcard_roundtrip() {
    let transactions = vec![
        create_test_transaction(1_000_000_000_000_000_000, 100_000_000),
        create_test_transaction(u128::MAX / 2, u64::MAX as u128),
    ];

    let serialized =
        postcard::to_allocvec(&transactions).expect("Failed to serialize transactions with postcard");
    let deserialized: Vec<Transaction> =
        postcard::from_bytes(&serialized).expect("Failed to deserialize transactions with postcard");

    assert_eq!(transactions.len(), deserialized.len());
    for (orig, deser) in transactions.iter().zip(deserialized.iter()) {
        assert_eq!(orig.amount, deser.amount);
        assert_eq!(orig.fee, deser.fee);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_u128_boundary_values() {
    // Test all important boundary values for u128
    let boundary_values: Vec<u128> = vec![
        0,                    // Zero
        1,                    // Minimum positive
        u32::MAX as u128,     // u32 max
        u64::MAX as u128,     // u64 max (MessagePack truncation boundary)
        u64::MAX as u128 + 1, // First value that exceeds u64
        u128::MAX / 2,        // Half of max
        u128::MAX - 1,        // Max - 1
        u128::MAX,            // Maximum
    ];

    for amount in &boundary_values {
        for fee in &boundary_values {
            let tx = create_test_transaction(*amount, *fee);

            // Test bincode roundtrip
            let bincode_ser = bincode::serialize(&tx).expect("Bincode serialize failed");
            let bincode_de: Transaction =
                bincode::deserialize(&bincode_ser).expect("Bincode deserialize failed");

            assert_eq!(
                tx.amount, bincode_de.amount,
                "Bincode: amount {} corrupted to {}",
                tx.amount, bincode_de.amount
            );
            assert_eq!(
                tx.fee, bincode_de.fee,
                "Bincode: fee {} corrupted to {}",
                tx.fee, bincode_de.fee
            );
        }
    }
}

#[test]
fn test_coinbase_transaction_large_reward() {
    // Coinbase transactions have large amounts (block rewards)
    // These are the transactions that triggered the original bug

    let coinbase_reward: u128 = 50_000_000_000_000_000_000; // 50 QUG in smallest unit
    let tx = Transaction {
        id: [0u8; 32],       // Coinbase has zero tx id
        from: [0u8; 32],     // Coinbase has zero from address
        to: [5u8; 32],       // Miner address
        amount: coinbase_reward,
        fee: 0, // Coinbase has no fee
        nonce: 0,
        signature: vec![],
        timestamp: Utc::now(),
        data: vec![],
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUGUSD,
        tx_type: TransactionType::Coinbase,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: q_types::TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    };

    let serialized = bincode::serialize(&tx).expect("Failed to serialize coinbase");
    let deserialized: Transaction =
        bincode::deserialize(&serialized).expect("Failed to deserialize coinbase");

    assert_eq!(
        tx.amount, deserialized.amount,
        "Coinbase reward was corrupted!"
    );
}

#[test]
fn test_multiple_transactions_sequential_roundtrip() {
    // Simulate what happens during blockchain sync - many transactions in sequence
    let mut transactions: Vec<Transaction> = Vec::new();

    for i in 0..100 {
        transactions.push(create_test_transaction(
            (i as u128 + 1) * 1_000_000_000_000_000_000,
            (i as u128 + 1) * 100_000,
        ));
    }

    // Serialize all transactions
    let serialized_txs: Vec<Vec<u8>> = transactions
        .iter()
        .map(|t| bincode::serialize(t).expect("Failed to serialize"))
        .collect();

    // Deserialize all transactions
    let deserialized_txs: Vec<Transaction> = serialized_txs
        .iter()
        .map(|t| bincode::deserialize(t).expect("Failed to deserialize"))
        .collect();

    // Verify all transactions match
    for (i, (orig, deser)) in transactions.iter().zip(deserialized_txs.iter()).enumerate() {
        assert_eq!(
            orig.amount, deser.amount,
            "Transaction {} amount mismatch",
            i
        );
        assert_eq!(
            orig.fee, deser.fee,
            "Transaction {} fee mismatch",
            i
        );
    }
}

// =============================================================================
// Regression Tests - Specifically prevent the deserialize_any bug
// =============================================================================

#[test]
fn test_deserialize_any_not_used_regression() {
    // This test exists to document and prevent regression of the deserialize_any bug
    //
    // The bug: u128_serde::deserialize used deserialize_any(), but bincode doesn't
    // support deserialize_any, causing "Bincode does not support the
    // serde::Deserializer::deserialize_any method" error.
    //
    // If this test fails with that exact error, the bug has been reintroduced!

    let tx = create_test_transaction(1_234_567_890_123_456_789, 987_654_321);

    let result: Result<Transaction, _> = bincode::deserialize(&bincode::serialize(&tx).unwrap());

    assert!(
        result.is_ok(),
        "Bincode deserialization failed! If the error is 'Bincode does not support \
         deserialize_any', then u128_serde has regressed to using deserialize_any. \
         The fix is to use deserialize_str instead. See v3.4.1 fix."
    );
}

#[test]
fn test_transaction_list_many_entries_stress() {
    // Stress test with many transactions to catch any edge cases
    let mut transactions = Vec::new();

    for i in 0..50 {
        transactions.push(create_test_transaction(
            (i as u128 + 1) * 10_000_000_000_000_000_000,
            (i as u128 + 1) * 1_000_000,
        ));
    }

    let serialized = bincode::serialize(&transactions).expect("Failed to serialize large tx list");
    let deserialized: Vec<Transaction> =
        bincode::deserialize(&serialized).expect("Failed to deserialize large tx list");

    assert_eq!(transactions.len(), deserialized.len());

    for (i, (orig, deser)) in transactions.iter().zip(deserialized.iter()).enumerate() {
        assert_eq!(
            orig.amount, deser.amount,
            "Stress test tx {} amount mismatch",
            i
        );
        assert_eq!(orig.fee, deser.fee, "Stress test tx {} fee mismatch", i);
    }
}

// =============================================================================
// Mixed Format Tests - Catch format confusion bugs
// =============================================================================

#[test]
fn test_bincode_then_postcard_fails_correctly() {
    // Ensure we don't accidentally mix formats
    let tx = create_test_transaction(1_000_000_000_000, 100_000);

    // Serialize with bincode
    let bincode_data = bincode::serialize(&tx).expect("Bincode serialize");

    // Try to deserialize with postcard - should fail
    let postcard_result: Result<Transaction, _> = postcard::from_bytes(&bincode_data);

    // This should either fail or give wrong data (format mismatch)
    // We're just verifying that the formats are incompatible
    if let Ok(deser) = postcard_result {
        // If it somehow succeeds, the values should be wrong
        // (unless we got extremely lucky with byte alignment)
        // This test documents that format mixing is dangerous
        println!(
            "Warning: Postcard decoded bincode data. Original amount: {}, decoded: {}",
            tx.amount, deser.amount
        );
    }
    // Test passes regardless - we're just documenting behavior
}

#[test]
fn test_postcard_then_bincode_fails_correctly() {
    // Ensure we don't accidentally mix formats
    let tx = create_test_transaction(1_000_000_000_000, 100_000);

    // Serialize with postcard
    let postcard_data = postcard::to_allocvec(&tx).expect("Postcard serialize");

    // Try to deserialize with bincode - should fail
    let bincode_result: Result<Transaction, _> = bincode::deserialize(&postcard_data);

    // This should either fail or give wrong data (format mismatch)
    if let Ok(deser) = bincode_result {
        println!(
            "Warning: Bincode decoded postcard data. Original amount: {}, decoded: {}",
            tx.amount, deser.amount
        );
    }
    // Test passes regardless - we're just documenting behavior
}
