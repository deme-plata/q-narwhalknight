//! Root-cause probe for the "QUGUSD send → validation failed" report (2026-07-28).
//!
//! `send_transaction_inner` (q-api-server/src/handlers.rs) signs
//! `signable_payload()` and THEN mutates `data`:
//!   * QUG        : data []            -> pubkey(32)
//!   * TokenTransfer: data token_addr(32) -> token_addr(32) ‖ pubkey(32)
//!
//! `signable_payload()` is SHA3 over postcard(tx) with only `signature`/`id`
//! zeroed — `data` IS covered. So any post-signature edit of `data` changes
//! every verification target (`signable_payload`, `hash`, `p2p_signable_hash`)
//! and `verify_signature()` must fail.
//!
//! These tests pin that behaviour so the ordering can never silently regress.

use chrono::Utc;
use ed25519_dalek::{Signer, SigningKey};
use q_types::{
    TokenType, Transaction, TransactionPrivacyLevel, TransactionType, TxHash, TxSignaturePhase,
};

fn signing_key() -> SigningKey {
    // Deterministic key — no randomness, so the test is reproducible.
    SigningKey::from_bytes(&[7u8; 32])
}

fn base_tx(tx_type: TransactionType, token_type: TokenType, data: Vec<u8>) -> Transaction {
    let sk = signing_key();
    let pk: [u8; 32] = sk.verifying_key().to_bytes();
    Transaction {
        id: TxHash::default(),
        from: pk, // `from` == pubkey, the first candidate key the verifier tries
        to: [9u8; 32],
        amount: 1_000_000,
        fee: 21_000,
        nonce: 1,
        signature: vec![],
        timestamp: Utc::now(),
        data,
        token_type,
        fee_token_type: TokenType::QUGUSD,
        tx_type,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    }
}

fn sign_in_place(tx: &mut Transaction) {
    let sk = signing_key();
    let msg = tx.signable_payload();
    tx.signature = sk.sign(&msg).to_bytes().to_vec();
}

/// The live handler order for a QUGUSD send: sign, THEN append the pubkey.
#[test]
fn token_transfer_signed_before_data_mutation_fails_verification() {
    let pk: [u8; 32] = signing_key().verifying_key().to_bytes();
    let mut tx = base_tx(
        TransactionType::TokenTransfer,
        TokenType::QUGUSD,
        q_types::QUGUSD_TOKEN_ADDRESS.to_vec(),
    );

    sign_in_place(&mut tx);
    // …exactly what handlers.rs does after signing (v10.11.87 layout fix):
    tx.data.extend_from_slice(&pk);
    assert_eq!(tx.data.len(), 64);

    assert!(
        tx.verify_signature().is_err(),
        "expected the post-signature data mutation to invalidate the signature"
    );
}

/// The live handler order for a plain QUG send: sign, THEN overwrite data.
#[test]
fn plain_transfer_signed_before_data_mutation_fails_verification() {
    let pk: [u8; 32] = signing_key().verifying_key().to_bytes();
    let mut tx = base_tx(TransactionType::Transfer, TokenType::QUG, vec![]);

    sign_in_place(&mut tx);
    tx.data = pk.to_vec();

    assert!(
        tx.verify_signature().is_err(),
        "expected the post-signature data mutation to invalidate the signature"
    );
}

/// The fix: build the FINAL data layout first, then sign.
#[test]
fn token_transfer_data_finalised_before_signing_verifies() {
    let pk: [u8; 32] = signing_key().verifying_key().to_bytes();
    let mut data = q_types::QUGUSD_TOKEN_ADDRESS.to_vec();
    data.extend_from_slice(&pk);

    let mut tx = base_tx(TransactionType::TokenTransfer, TokenType::QUGUSD, data);
    sign_in_place(&mut tx);

    assert!(
        tx.verify_signature().is_ok(),
        "signing the final 64-byte token layout must verify: {:?}",
        tx.verify_signature()
    );
}

/// Same for native QUG: pubkey in data before signing.
#[test]
fn plain_transfer_data_finalised_before_signing_verifies() {
    let pk: [u8; 32] = signing_key().verifying_key().to_bytes();
    let mut tx = base_tx(TransactionType::Transfer, TokenType::QUG, pk.to_vec());
    sign_in_place(&mut tx);

    assert!(
        tx.verify_signature().is_ok(),
        "signing with the final data must verify: {:?}",
        tx.verify_signature()
    );
}

/// Privacy-level mutation after signing is the same class of bug — pinned so a
/// future `apply_privacy_proofs` change cannot silently break settlement.
#[test]
fn privacy_level_mutation_after_signing_fails_verification() {
    let pk: [u8; 32] = signing_key().verifying_key().to_bytes();
    let mut tx = base_tx(TransactionType::Transfer, TokenType::QUG, pk.to_vec());
    sign_in_place(&mut tx);
    assert!(tx.verify_signature().is_ok(), "control: must verify first");

    tx.privacy_level = TransactionPrivacyLevel::FullPrivacy;

    assert!(
        tx.verify_signature().is_err(),
        "expected a post-signature privacy_level change to invalidate the signature"
    );
}
