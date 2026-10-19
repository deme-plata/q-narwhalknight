//! Threshold-aware hybrid verification.
//!
//! For each `SignatureContribution`:
//!   1. Resolve `member_addr` → `Member` via the wallet's member set.
//!   2. Verify the Ed25519 half over the canonical payload hash.
//!   3. Verify the Dilithium5 half (detached) with `q_types`' shared verifier:
//!      ML-DSA-87 as published (FIPS 204, what browsers and the MCP sign), then the
//!      2023 draft `pqcrypto-dilithium` implements. Until 2026-10-19 this checked the
//!      draft only, so no browser-made co-signature could ever count.
//!   4. Count this member toward the threshold if BOTH pass.
//!
//! A proposal passes only if the count of valid contributions ≥
//! `proposal.required`. Duplicate signatures from the same member count once.

use q_types::signature_verification::{verify_tx_pq_signature, MLDSA87_SIG_BYTES};
use std::collections::HashSet;
use thiserror::Error;

use crate::proposal::MultisigProposal;
use crate::wallet::{Address, MultisigWallet};

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("proposal references wallet {proposal_wallet:?} but checked against {wallet:?}")]
    WalletMismatch {
        proposal_wallet: Address,
        wallet: Address,
    },
    #[error("signature contribution from unknown member: {member_addr_hex}")]
    UnknownMember { member_addr_hex: String },
    #[error("ed25519 verification failed for member {0}")]
    Ed25519Failed(String),
    #[error("dilithium5 verification failed for member {0}")]
    Dilithium5Failed(String),
    #[error("invalid dilithium5 signature bytes for member {0}")]
    Dilithium5BadEncoding(String),
    #[error("threshold not met: {valid}/{required}")]
    BelowThreshold { valid: u8, required: u8 },
}

/// Verify a single member's contribution against the proposal payload.
/// Used in isolation by the `/multisig/sign` endpoint (so the server can
/// reject bad sigs as they arrive, not just at execution time).
pub fn verify_member_signature(
    wallet: &MultisigWallet,
    proposal: &MultisigProposal,
    contrib: &crate::proposal::SignatureContribution,
) -> Result<(), VerifyError> {
    let member = wallet
        .find_member(&contrib.member_addr)
        .ok_or_else(|| VerifyError::UnknownMember {
            member_addr_hex: hex::encode(contrib.member_addr),
        })?;

    let payload = proposal.payload_hash();

    // Ed25519 verify (classical, fast path).
    member
        .pubkey
        .ed25519
        .verify_strict(&payload, &contrib.ed25519_sig)
        .map_err(|_| VerifyError::Ed25519Failed(hex::encode(contrib.member_addr)))?;

    // Dilithium5 verify (post-quantum, detached — matches the P2P hybrid-send
    // convention; see the field doc on `dilithium5_signature`).
    if contrib.dilithium5_signature.len() != MLDSA87_SIG_BYTES {
        return Err(VerifyError::Dilithium5BadEncoding(hex::encode(contrib.member_addr)));
    }
    verify_tx_pq_signature(&contrib.dilithium5_signature, &member.pubkey.dilithium5, &payload)
        .map_err(|_| VerifyError::Dilithium5Failed(hex::encode(contrib.member_addr)))?;

    Ok(())
}

/// Verify the full proposal: every contribution passes hybrid checks AND
/// the count of unique valid signers ≥ required threshold.
pub fn verify_proposal(
    wallet: &MultisigWallet,
    proposal: &MultisigProposal,
) -> Result<(), VerifyError> {
    if proposal.wallet_addr != wallet.address {
        return Err(VerifyError::WalletMismatch {
            proposal_wallet: proposal.wallet_addr,
            wallet: wallet.address,
        });
    }

    let mut valid_signers: HashSet<Address> = HashSet::new();
    for contrib in &proposal.signatures {
        // Hybrid verify — both halves must pass.
        verify_member_signature(wallet, proposal, contrib)?;
        valid_signers.insert(contrib.member_addr);
    }

    let valid = valid_signers.len() as u8;
    if valid < proposal.required {
        return Err(VerifyError::BelowThreshold {
            valid,
            required: proposal.required,
        });
    }
    Ok(())
}
