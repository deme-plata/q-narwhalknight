//! Fee validation tests for Q-NarwhalKnight
//!
//! v3.4.0-beta: Tests for height-gated fee reduction (10x cheaper after block 350,000)
//!
//! Run with: cargo test --package q-types --test fee_validation_tests

use chrono::Utc;
use q_types::{
    Transaction, TransactionType, Address, TokenType, TxSignaturePhase,
    BASE_GAS, MIN_FEE_PER_GAS, MIN_FEE_PER_GAS_LEGACY,
    MIN_TRANSACTION_FEE, MIN_TRANSACTION_FEE_LEGACY, MIN_TRANSACTION_FEE_V1,
    FEE_REDUCTION_DIVISOR, MAX_TRANSACTION_FEE,
    get_min_transaction_fee, get_fee_divisor, is_reduced_fees_active,
    upgrades::upgrades::REDUCED_FEES_V1,
};

// ============================================================================
// Height-Gated Fee Constants Tests
// ============================================================================

#[test]
fn test_fee_constants_are_correct() {
    // Legacy fees (before activation)
    assert_eq!(MIN_FEE_PER_GAS, 1);
    assert_eq!(MIN_FEE_PER_GAS_LEGACY, 1);
    assert_eq!(BASE_GAS, 21_000);
    assert_eq!(MIN_TRANSACTION_FEE_LEGACY, 21_000); // 21,000 * 1 = 0.00021 QUG
    assert_eq!(MIN_TRANSACTION_FEE, 21_000);

    // Reduced fees (after activation)
    assert_eq!(FEE_REDUCTION_DIVISOR, 10);
    assert_eq!(MIN_TRANSACTION_FEE_V1, 2_100); // 21,000 / 10 = 0.000021 QUG
}

#[test]
fn test_reduced_fees_activation_height() {
    assert_eq!(REDUCED_FEES_V1.activation_height, 350_000);
    assert_eq!(REDUCED_FEES_V1.name, "reduced_fees_v1");
}

// ============================================================================
// get_min_transaction_fee() Tests
// ============================================================================

#[test]
fn test_get_min_transaction_fee_before_activation() {
    // Before activation height (legacy fees)
    assert_eq!(get_min_transaction_fee(0), MIN_TRANSACTION_FEE_LEGACY);
    assert_eq!(get_min_transaction_fee(1), MIN_TRANSACTION_FEE_LEGACY);
    assert_eq!(get_min_transaction_fee(100_000), MIN_TRANSACTION_FEE_LEGACY);
    assert_eq!(get_min_transaction_fee(300_000), MIN_TRANSACTION_FEE_LEGACY);
    assert_eq!(get_min_transaction_fee(349_999), MIN_TRANSACTION_FEE_LEGACY);
}

#[test]
fn test_get_min_transaction_fee_at_activation() {
    // At activation height (reduced fees)
    assert_eq!(get_min_transaction_fee(350_000), MIN_TRANSACTION_FEE_V1);
}

#[test]
fn test_get_min_transaction_fee_after_activation() {
    // After activation height (reduced fees)
    assert_eq!(get_min_transaction_fee(350_001), MIN_TRANSACTION_FEE_V1);
    assert_eq!(get_min_transaction_fee(500_000), MIN_TRANSACTION_FEE_V1);
    assert_eq!(get_min_transaction_fee(1_000_000), MIN_TRANSACTION_FEE_V1);
}

#[test]
fn test_fee_reduction_is_10x() {
    // Verify 10x reduction
    let legacy_fee = get_min_transaction_fee(0);
    let reduced_fee = get_min_transaction_fee(350_000);

    assert_eq!(legacy_fee, reduced_fee * FEE_REDUCTION_DIVISOR);
    assert_eq!(legacy_fee / 10, reduced_fee);
}

// ============================================================================
// get_fee_divisor() Tests
// ============================================================================

#[test]
fn test_get_fee_divisor_before_activation() {
    assert_eq!(get_fee_divisor(0), 1);
    assert_eq!(get_fee_divisor(349_999), 1);
}

#[test]
fn test_get_fee_divisor_at_and_after_activation() {
    assert_eq!(get_fee_divisor(350_000), 10);
    assert_eq!(get_fee_divisor(500_000), 10);
}

// ============================================================================
// is_reduced_fees_active() Tests
// ============================================================================

#[test]
fn test_is_reduced_fees_active() {
    assert!(!is_reduced_fees_active(0));
    assert!(!is_reduced_fees_active(349_999));
    assert!(is_reduced_fees_active(350_000));
    assert!(is_reduced_fees_active(500_000));
}

// ============================================================================
// Transaction Fee Validation Tests
// ============================================================================

fn create_test_transaction(fee: u128, tx_type: TransactionType, amount: u128) -> Transaction {
    use sha3::{Sha3_256, Digest};

    let from: Address = [1u8; 32];
    let to: Address = [2u8; 32];

    // Generate a deterministic tx id
    let mut hasher = Sha3_256::new();
    hasher.update(&from);
    hasher.update(&to);
    hasher.update(&fee.to_le_bytes());
    let hash = hasher.finalize();
    let mut id = [0u8; 32];
    id.copy_from_slice(&hash);

    Transaction {
        id,
        from,
        to,
        amount,
        fee,
        nonce: 1,
        timestamp: Utc::now(),
        signature: vec![0u8; 64],
        tx_type,
        data: vec![],
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUG,
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

#[test]
fn test_validate_fee_legacy_transfer_success() {
    // Legacy fees: 21,000 satoshis for transfer
    let tx = create_test_transaction(21_000, TransactionType::Transfer, 100_000);
    assert!(tx.validate_fee().is_ok());
    assert!(tx.validate_fee_at_height(0).is_ok());
}

#[test]
fn test_validate_fee_legacy_transfer_failure() {
    // Legacy fees: < 21,000 should fail
    let tx = create_test_transaction(20_999, TransactionType::Transfer, 100_000);
    assert!(tx.validate_fee_at_height(0).is_err());
}

#[test]
fn test_validate_fee_reduced_transfer_success() {
    // Reduced fees: 2,100 satoshis for transfer after activation
    let tx = create_test_transaction(2_100, TransactionType::Transfer, 100_000);
    assert!(tx.validate_fee_at_height(350_000).is_ok());
    assert!(tx.validate_fee_at_height(500_000).is_ok());
}

#[test]
fn test_validate_fee_reduced_transfer_failure() {
    // Reduced fees: < 2,100 should fail after activation
    let tx = create_test_transaction(2_099, TransactionType::Transfer, 100_000);
    assert!(tx.validate_fee_at_height(350_000).is_err());
}

#[test]
fn test_validate_fee_at_boundary() {
    // At block 349,999 (legacy fees): 21,000 needed
    let tx_legacy = create_test_transaction(21_000, TransactionType::Transfer, 100_000);
    assert!(tx_legacy.validate_fee_at_height(349_999).is_ok());

    // Same fee at block 350,000 (reduced fees): 2,100 needed (21,000 is more than enough)
    assert!(tx_legacy.validate_fee_at_height(350_000).is_ok());

    // At block 349,999: 2,100 should fail (needs 21,000)
    let tx_reduced = create_test_transaction(2_100, TransactionType::Transfer, 100_000);
    assert!(tx_reduced.validate_fee_at_height(349_999).is_err());

    // At block 350,000: 2,100 should succeed
    assert!(tx_reduced.validate_fee_at_height(350_000).is_ok());
}

// ============================================================================
// Transaction Type Gas Multiplier Tests
// ============================================================================

#[test]
fn test_gas_multipliers_with_reduced_fees() {
    // Transfer: 1x gas = 21,000 / 10 = 2,100 min fee after reduction
    let transfer_fee = BASE_GAS * MIN_FEE_PER_GAS / FEE_REDUCTION_DIVISOR;
    assert_eq!(transfer_fee, 2_100);

    // Token transfer: 2x gas = 42,000 / 10 = 4,200 min fee after reduction
    let token_transfer_fee = BASE_GAS * 2 * MIN_FEE_PER_GAS / FEE_REDUCTION_DIVISOR;
    assert_eq!(token_transfer_fee, 4_200);

    // Swap: 3x gas = 63,000 / 10 = 6,300 min fee after reduction
    let swap_fee = BASE_GAS * 3 * MIN_FEE_PER_GAS / FEE_REDUCTION_DIVISOR;
    assert_eq!(swap_fee, 6_300);

    // Contract call: 5x gas = 105,000 / 10 = 10,500 min fee after reduction
    let contract_call_fee = BASE_GAS * 5 * MIN_FEE_PER_GAS / FEE_REDUCTION_DIVISOR;
    assert_eq!(contract_call_fee, 10_500);
}

#[test]
fn test_swap_transaction_reduced_fees() {
    // Swap uses 3x gas multiplier
    // Legacy: 21,000 * 3 = 63,000 min fee
    // Reduced: 63,000 / 10 = 6,300 min fee

    let tx = create_test_transaction(6_300, TransactionType::Swap, 1_000_000);

    // Should fail with legacy fees
    assert!(tx.validate_fee_at_height(0).is_err());

    // Should succeed with reduced fees
    assert!(tx.validate_fee_at_height(350_000).is_ok());
}

#[test]
fn test_contract_call_reduced_fees() {
    // ContractCall uses 5x gas multiplier
    // Legacy: 21,000 * 5 = 105,000 min fee
    // Reduced: 105,000 / 10 = 10,500 min fee

    let mut tx = create_test_transaction(10_500, TransactionType::ContractCall, 0);
    tx.data = vec![1, 2, 3, 4]; // Contract call needs data

    // Should fail with legacy fees (needs 105,000)
    assert!(tx.validate_fee_at_height(0).is_err());

    // Should succeed with reduced fees
    assert!(tx.validate_fee_at_height(350_000).is_ok());
}

// ============================================================================
// Maximum Fee Tests
// ============================================================================

#[test]
fn test_max_fee_unchanged() {
    // Max fee should be the same regardless of height
    assert_eq!(MAX_TRANSACTION_FEE, 1_000_000_000); // 10 QUG

    let tx = create_test_transaction(MAX_TRANSACTION_FEE + 1, TransactionType::Transfer, 100_000);
    assert!(tx.validate_fee_at_height(0).is_err());
    assert!(tx.validate_fee_at_height(350_000).is_err());
}

// ============================================================================
// Coinbase/System Transaction Tests
// ============================================================================

#[test]
fn test_coinbase_exempt_from_fee_validation() {
    // Coinbase is detected by from == [0u8; 32], not TransactionType
    let mut tx = create_test_transaction(0, TransactionType::Coinbase, 1_000_000);
    tx.from = [0u8; 32]; // This makes it a coinbase transaction
    tx.fee = 0; // Coinbase has no fee

    // Should succeed at any height
    assert!(tx.validate_fee_at_height(0).is_ok());
    assert!(tx.validate_fee_at_height(350_000).is_ok());
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_historical_blocks_validate_with_legacy_rules() {
    // Critical mainnet safety test:
    // Old blocks that were valid with legacy fees must still validate

    let tx = create_test_transaction(21_000, TransactionType::Transfer, 1_000_000);

    // Should validate at height 0 (genesis)
    assert!(tx.validate_fee_at_height(0).is_ok());

    // Should validate at any historical height
    assert!(tx.validate_fee_at_height(100_000).is_ok());
    assert!(tx.validate_fee_at_height(200_000).is_ok());

    // Should still validate after fee reduction (overpaying is allowed)
    assert!(tx.validate_fee_at_height(350_000).is_ok());
}

#[test]
fn test_new_lower_fees_rejected_on_old_blocks() {
    // Critical mainnet safety test:
    // Transactions with new lower fees must fail validation on old blocks

    let tx = create_test_transaction(2_100, TransactionType::Transfer, 1_000_000);

    // Should fail at historical heights (need 21,000 fee)
    assert!(tx.validate_fee_at_height(0).is_err());
    assert!(tx.validate_fee_at_height(100_000).is_err());
    assert!(tx.validate_fee_at_height(349_999).is_err());

    // Should succeed at activation height and after
    assert!(tx.validate_fee_at_height(350_000).is_ok());
    assert!(tx.validate_fee_at_height(500_000).is_ok());
}
