//! End-to-end smoke tests for q-multisig.
//!
//! Covers the three flows we care about for v0:
//!
//! 1. Create a 2-of-2 wallet with default_threshold = 2 (unanimous).
//! 2. Propose a Transfer with required_override = Some(1) (either signer
//!    can spend solo) and verify single-sig is enough.
//! 3. Propose a MintToken with no override (defaults to threshold = 2,
//!    "both must agree") and verify both halves of the hybrid sig (Ed25519
//!    + Dilithium5) are required.

use ed25519_dalek::{SigningKey, Signer};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as _, SignedMessage as _};
use rand::rngs::OsRng;
use rand::TryRngCore;

use q_multisig::{
    proposal::{MultisigAction, MultisigProposal, SignatureContribution},
    verify::{verify_member_signature, verify_proposal, VerifyError},
    wallet::{HybridPublicKey, Member, MultisigWallet},
};

struct TestSigner {
    label: String,
    ed25519_sk: SigningKey,
    dilithium5_sk: dilithium5::SecretKey,
    member: Member,
}

fn gen_signer(label: &str) -> TestSigner {
    let mut seed = [0u8; 32];
    OsRng.try_fill_bytes(&mut seed).unwrap();
    let ed25519_sk = SigningKey::from_bytes(&seed);
    let ed25519_pk = ed25519_sk.verifying_key();

    let (dilithium5_pk, dilithium5_sk) = dilithium5::keypair();
    let pubkey = HybridPublicKey {
        ed25519: ed25519_pk,
        dilithium5: dilithium5_pk.as_bytes().to_vec(),
    };
    let member = Member {
        label: label.to_string(),
        pubkey,
    };
    TestSigner {
        label: label.to_string(),
        ed25519_sk,
        dilithium5_sk,
        member,
    }
}

fn sign_proposal(signer: &TestSigner, proposal: &MultisigProposal) -> SignatureContribution {
    let payload = proposal.payload_hash();
    let ed25519_sig = signer.ed25519_sk.sign(&payload);
    let dilithium5_signed = dilithium5::sign(&payload, &signer.dilithium5_sk);
    SignatureContribution {
        member_addr: signer.member.pubkey.member_address(),
        ed25519_sig,
        dilithium5_signed_msg: dilithium5_signed.as_bytes().to_vec(),
        at_unix: chrono::Utc::now().timestamp(),
    }
}

#[test]
fn wallet_address_is_deterministic_under_member_reorder() {
    let user = gen_signer("user");
    let claude = gen_signer("claude");

    let w1 = MultisigWallet::new(vec![user.member.clone(), claude.member.clone()], 2, "us")
        .expect("valid wallet");
    let w2 = MultisigWallet::new(vec![claude.member.clone(), user.member.clone()], 2, "us")
        .expect("valid wallet");
    assert_eq!(w1.address, w2.address);
    assert!(w1.address_string().starts_with("qnk"));
}

#[test]
fn wallet_rejects_invalid_thresholds() {
    let a = gen_signer("a");
    let b = gen_signer("b");
    // threshold=0 → ThresholdZero
    assert!(MultisigWallet::new(vec![a.member.clone(), b.member.clone()], 0, "x").is_err());
    // threshold=3 with 2 members → ThresholdTooHigh
    assert!(MultisigWallet::new(vec![a.member.clone(), b.member.clone()], 3, "x").is_err());
    // single member → NotEnoughMembers
    assert!(MultisigWallet::new(vec![a.member.clone()], 1, "x").is_err());
}

#[test]
fn either_signer_alone_can_spend_when_required_override_is_one() {
    let user = gen_signer("user");
    let claude = gen_signer("claude");
    let wallet = MultisigWallet::new(vec![user.member.clone(), claude.member.clone()], 2, "us")
        .unwrap();

    let action = MultisigAction::Transfer {
        token: "QUG".to_string(),
        recipient: [0xAB; 32],
        amount_raw: 10_u128.pow(24), // 1 QUG
        memo: Some("solo-spend test".to_string()),
    };
    let mut proposal = MultisigProposal::new(
        wallet.address,
        action,
        Some(1), // either can spend
        wallet.default_threshold,
        wallet.members.len(),
    );
    assert_eq!(proposal.required, 1);

    // Only Claude signs.
    let contrib = sign_proposal(&claude, &proposal);
    verify_member_signature(&wallet, &proposal, &contrib).expect("claude contrib should verify");
    proposal.add_signature(contrib);

    // Threshold met.
    verify_proposal(&wallet, &proposal).expect("solo spend with required=1 should pass");
}

#[test]
fn unanimous_required_when_no_override_and_default_is_two() {
    let user = gen_signer("user");
    let claude = gen_signer("claude");
    let wallet = MultisigWallet::new(vec![user.member.clone(), claude.member.clone()], 2, "us")
        .unwrap();

    let action = MultisigAction::MintToken {
        symbol: "QCO".to_string(),
        name: "Quillon Claude Original".to_string(),
        decimals: 24,
        initial_supply_raw: 1_000_000_u128 * 10_u128.pow(24),
        initial_holders: vec![
            (user.member.pubkey.member_address(), 500_000_u128 * 10_u128.pow(24)),
            (claude.member.pubkey.member_address(), 500_000_u128 * 10_u128.pow(24)),
        ],
    };
    let mut proposal = MultisigProposal::new(
        wallet.address,
        action,
        None, // use default_threshold = 2
        wallet.default_threshold,
        wallet.members.len(),
    );
    assert_eq!(proposal.required, 2);

    // Only user signs → should NOT pass.
    proposal.add_signature(sign_proposal(&user, &proposal));
    let err = verify_proposal(&wallet, &proposal).expect_err("one sig of two should fail");
    assert!(matches!(err, VerifyError::BelowThreshold { valid: 1, required: 2 }));

    // Claude adds her sig → passes.
    proposal.add_signature(sign_proposal(&claude, &proposal));
    verify_proposal(&wallet, &proposal).expect("unanimous 2-of-2 should pass");
}

#[test]
fn forged_ed25519_with_valid_dilithium_still_fails() {
    // Hybrid verification REQUIRES both halves to pass. A signer who knows
    // only the Dilithium5 key can't bluff the Ed25519 half.
    let user = gen_signer("user");
    let claude = gen_signer("claude");
    let attacker_ed25519 = SigningKey::from_bytes(&[7u8; 32]);

    let wallet = MultisigWallet::new(vec![user.member.clone(), claude.member.clone()], 1, "us")
        .unwrap();

    let action = MultisigAction::Transfer {
        token: "QUG".to_string(),
        recipient: [0xAB; 32],
        amount_raw: 42,
        memo: None,
    };
    let mut proposal = MultisigProposal::new(
        wallet.address,
        action,
        Some(1),
        wallet.default_threshold,
        wallet.members.len(),
    );

    // Build a contribution with the WRONG Ed25519 key but Claude's
    // Dilithium5 key.
    let payload = proposal.payload_hash();
    let bad_ed25519_sig = attacker_ed25519.sign(&payload); // wrong key!
    let good_dilithium = dilithium5::sign(&payload, &claude.dilithium5_sk);
    let contrib = SignatureContribution {
        member_addr: claude.member.pubkey.member_address(),
        ed25519_sig: bad_ed25519_sig,
        dilithium5_signed_msg: good_dilithium.as_bytes().to_vec(),
        at_unix: chrono::Utc::now().timestamp(),
    };
    proposal.add_signature(contrib);

    let err = verify_proposal(&wallet, &proposal).expect_err("hybrid mismatch should fail");
    assert!(matches!(err, VerifyError::Ed25519Failed(_)));
}

#[test]
fn duplicate_member_signing_counts_once() {
    let user = gen_signer("user");
    let claude = gen_signer("claude");
    let wallet = MultisigWallet::new(vec![user.member.clone(), claude.member.clone()], 2, "us")
        .unwrap();

    let action = MultisigAction::Transfer {
        token: "QUG".to_string(),
        recipient: [0xAB; 32],
        amount_raw: 1,
        memo: None,
    };
    let mut proposal = MultisigProposal::new(
        wallet.address,
        action,
        Some(2),
        wallet.default_threshold,
        wallet.members.len(),
    );
    // Claude signs twice → still 1 unique signer.
    proposal.add_signature(sign_proposal(&claude, &proposal));
    proposal.add_signature(sign_proposal(&claude, &proposal));
    assert_eq!(proposal.signatures.len(), 1);
    let err = verify_proposal(&wallet, &proposal).expect_err("one signer can't be two");
    assert!(matches!(err, VerifyError::BelowThreshold { valid: 1, required: 2 }));
}
