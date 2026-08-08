//! Integration tests for the real mixer settlement path.
//!
//! Kept as an integration test (its own compile target) so it is unaffected by
//! pre-existing breakage in the crate's inline `#[cfg(test)]` unit tests (e.g.
//! `bulletproofs_pp::test_proof_determinism`, which references a non-existent
//! `prove_with_rng`). Exercises only the public API.

use std::sync::Arc;

use q_quantum_mixing::clsag::{
    create_pedersen_commitment, generate_commitment_mask, CLSAGSigner,
};
use q_quantum_mixing::quantum_entropy::QuantumEntropyPool;
use q_quantum_mixing::settlement::{
    settlement_message, InMemoryLedger, Settler, SettlementError, SettlementLedger,
    SettlementRequest,
};

const FEE_ACCT: [u8; 32] = [0xFE; 32];
const MIX_FEE: u64 = 10;

/// Build a valid, ring-signed settlement request. Uses a fresh signer per call;
/// tests that need a stable key image (replay) reuse the same request object.
async fn signed_request(
    sender: [u8; 32],
    recipient: [u8; 32],
    amount: u64,
    nonce: [u8; 32],
) -> SettlementRequest {
    let entropy = Arc::new(QuantumEntropyPool::new().await.expect("entropy"));

    let mut signer = CLSAGSigner::new(entropy.clone()).await.expect("signer");
    let decoy1 = CLSAGSigner::new(entropy.clone()).await.expect("decoy1");
    let decoy2 = CLSAGSigner::new(entropy.clone()).await.expect("decoy2");
    let ring = vec![
        signer.get_public_key(),
        decoy1.get_public_key(),
        decoy2.get_public_key(),
    ];

    let mask = generate_commitment_mask(&entropy).await.expect("mask");
    let (commitment, _pt) = create_pedersen_commitment(amount, &mask);

    let message = settlement_message(&sender, &recipient, amount, &nonce);
    let signature = signer
        .sign(&message, &ring, &commitment, &mask)
        .await
        .expect("sign");

    SettlementRequest {
        sender,
        recipient,
        amount,
        nonce,
        mask: mask.to_bytes(),
        signature,
    }
}

fn ledger_with(sender: [u8; 32], bal: u64) -> InMemoryLedger {
    let l = InMemoryLedger::new();
    l.seed(sender, bal);
    l
}

#[tokio::test]
async fn valid_settlement_moves_funds_and_conserves() {
    let sender = [1u8; 32];
    let recipient = [2u8; 32];
    let req = signed_request(sender, recipient, 100, [7u8; 32]).await;

    let settler = Settler::new(ledger_with(sender, 1_000), FEE_ACCT, MIX_FEE);
    let receipt = settler.verify_and_settle(&req).expect("should settle");
    assert_eq!(receipt.amount, 100);
    assert_eq!(receipt.fee, MIX_FEE);

    let l = settler.ledger();
    assert_eq!(l.balance(&sender).unwrap(), 1_000 - 100 - MIX_FEE);
    assert_eq!(l.balance(&recipient).unwrap(), 100);
    assert_eq!(l.balance(&FEE_ACCT).unwrap(), MIX_FEE);
    let total = l.balance(&sender).unwrap()
        + l.balance(&recipient).unwrap()
        + l.balance(&FEE_ACCT).unwrap();
    assert_eq!(total, 1_000, "value conserved");
    assert_eq!(l.spent_count(), 1);
}

#[tokio::test]
async fn replay_of_identical_transfer_is_double_spend() {
    let sender = [1u8; 32];
    let recipient = [2u8; 32];
    let req = signed_request(sender, recipient, 100, [7u8; 32]).await;

    let settler = Settler::new(ledger_with(sender, 1_000), FEE_ACCT, MIX_FEE);
    settler.verify_and_settle(&req).expect("first settles");

    let err = settler.verify_and_settle(&req).unwrap_err();
    assert!(matches!(err, SettlementError::DoubleSpend), "got {err:?}");

    let l = settler.ledger();
    assert_eq!(l.balance(&recipient).unwrap(), 100, "no second credit");
    assert_eq!(l.balance(&sender).unwrap(), 1_000 - 100 - MIX_FEE);
    assert_eq!(l.spent_count(), 1);
}

#[tokio::test]
async fn tampered_amount_fails_and_changes_nothing() {
    let sender = [1u8; 32];
    let recipient = [2u8; 32];
    let mut req = signed_request(sender, recipient, 100, [7u8; 32]).await;

    // Bump the amount but keep the signature+mask for amount=100. Either the
    // signed message no longer matches, or the commitment no longer re-opens.
    req.amount = 1_000_000;

    let settler = Settler::new(ledger_with(sender, 10_000_000), FEE_ACCT, MIX_FEE);
    let err = settler.verify_and_settle(&req).unwrap_err();
    assert!(
        matches!(
            err,
            SettlementError::InvalidSignature | SettlementError::CommitmentMismatch
        ),
        "got {err:?}"
    );

    let l = settler.ledger();
    assert_eq!(l.balance(&recipient).unwrap(), 0);
    assert_eq!(l.balance(&sender).unwrap(), 10_000_000);
    assert_eq!(l.spent_count(), 0);
}

#[tokio::test]
async fn insufficient_funds_rejected() {
    let sender = [1u8; 32];
    let recipient = [2u8; 32];
    let req = signed_request(sender, recipient, 100, [7u8; 32]).await;

    let settler = Settler::new(ledger_with(sender, 50), FEE_ACCT, MIX_FEE);
    let err = settler.verify_and_settle(&req).unwrap_err();
    assert!(matches!(err, SettlementError::InsufficientFunds { .. }), "got {err:?}");
    assert_eq!(settler.ledger().spent_count(), 0);
}

#[tokio::test]
async fn signature_over_different_recipient_is_rejected() {
    let sender = [1u8; 32];
    let recipient = [2u8; 32];
    let mut req = signed_request(sender, recipient, 100, [7u8; 32]).await;

    req.recipient = [9u8; 32]; // re-aim without re-signing

    let settler = Settler::new(ledger_with(sender, 1_000), FEE_ACCT, MIX_FEE);
    let err = settler.verify_and_settle(&req).unwrap_err();
    assert!(matches!(err, SettlementError::InvalidSignature), "got {err:?}");
    assert_eq!(settler.ledger().balance(&[9u8; 32]).unwrap(), 0);
    assert_eq!(settler.ledger().spent_count(), 0);
}

#[test]
fn distinct_nonces_give_distinct_messages() {
    let s = [1u8; 32];
    let r = [2u8; 32];
    assert_ne!(
        settlement_message(&s, &r, 100, &[1u8; 32]),
        settlement_message(&s, &r, 100, &[2u8; 32]),
    );
}
