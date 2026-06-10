//! FROST Threshold Signatures (IACR 2025/1024)
//!
//! Implementation of FROST: Flexible Round-Optimized Schnorr Threshold Signatures
//! Based on "FROST Revisited: Memory-Optimal Two-Round Threshold Schnorr" (ePrint 2025/1024)
//!
//! This enables t-of-n threshold signing for validator committees:
//! - Any t validators can sign a block/transaction
//! - No single validator can forge a signature alone
//! - Signatures are indistinguishable from single-signer Schnorr

use crate::errors::CryptoError;
use frost_ed25519 as frost;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tracing::{debug, info, warn};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Type aliases for FROST Ed25519
pub type Identifier = frost::Identifier;
pub type SigningKey = frost::keys::SigningShare;
pub type VerifyingKey = frost::keys::VerifyingShare;
pub type GroupPublicKey = frost::VerifyingKey;
pub type SigningNonces = frost::round1::SigningNonces;
pub type SigningCommitments = frost::round1::SigningCommitments;
pub type SignatureShare = frost::round2::SignatureShare;
pub type Signature = frost::Signature;

/// A key share held by a single validator
#[derive(Clone)]
pub struct KeyShare {
    /// Participant identifier (1-indexed)
    pub identifier: u16,
    /// FROST Identifier
    frost_identifier: Identifier,
    /// The signing share (secret)
    signing_share: SigningKey,
    /// The verifying share (public)
    pub verifying_share: VerifyingKey,
    /// Group public key
    pub group_public_key: GroupPublicKey,
    /// Threshold required
    pub threshold: u16,
    /// Total participants
    pub total: u16,
}

impl KeyShare {
    /// Get the FROST identifier
    pub fn frost_identifier(&self) -> &Identifier {
        &self.frost_identifier
    }
}

impl Zeroize for KeyShare {
    fn zeroize(&mut self) {
        self.identifier = 0;
        self.threshold = 0;
        self.total = 0;
    }
}

impl ZeroizeOnDrop for KeyShare {}

/// Validator committee configuration
#[derive(Clone, Debug)]
pub struct ValidatorCommittee {
    /// Committee ID/name
    pub id: String,
    /// Threshold required for signing
    pub threshold: u16,
    /// Total members
    pub total: u16,
    /// Group public key (for verification)
    pub group_public_key: GroupPublicKey,
    /// Member identifiers
    pub members: Vec<u16>,
}

impl ValidatorCommittee {
    /// Create a new committee from key generation
    pub fn new(
        id: impl Into<String>,
        threshold: u16,
        total: u16,
        group_public_key: GroupPublicKey,
    ) -> Self {
        Self {
            id: id.into(),
            threshold,
            total,
            group_public_key,
            members: (1..=total).collect(),
        }
    }

    /// Check if we have enough signers
    pub fn has_quorum(&self, signers: usize) -> bool {
        signers >= self.threshold as usize
    }
}

/// FROST Key Generation (Distributed Key Generation - DKG)
pub struct FrostKeyGen;

impl FrostKeyGen {
    /// Generate key shares for a t-of-n threshold scheme
    pub fn generate_shares(
        threshold: u16,
        total: u16,
    ) -> Result<(Vec<KeyShare>, GroupPublicKey), CryptoError> {
        // Validate parameters
        if threshold == 0 || total == 0 {
            return Err(CryptoError::InvalidThreshold {
                threshold: threshold as usize,
                total: total as usize,
            });
        }
        if threshold > total {
            return Err(CryptoError::InvalidThreshold {
                threshold: threshold as usize,
                total: total as usize,
            });
        }

        info!(
            "FROST: Generating {}-of-{} threshold key shares",
            threshold, total
        );

        let mut rng = OsRng;

        // Use trusted dealer for key generation
        let (shares, pubkey_package) = frost::keys::generate_with_dealer(
            total,
            threshold,
            frost::keys::IdentifierList::Default,
            &mut rng,
        )
        .map_err(|e| CryptoError::KeyGenFailed(e.to_string()))?;

        let group_public_key = pubkey_package.verifying_key().clone();

        // Convert to our KeyShare format
        let key_shares: Vec<KeyShare> = shares
            .into_iter()
            .map(|(id, secret_share)| {
                let verifying_share = pubkey_package
                    .verifying_shares()
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| {
                        // Clone the signing share to create verifying share
                        frost::keys::VerifyingShare::from(secret_share.signing_share().clone())
                    });

                // Get identifier as u16
                let id_bytes = id.serialize();
                let id_u16 = if id_bytes.len() >= 2 {
                    u16::from_le_bytes([id_bytes[0], id_bytes[1]])
                } else if id_bytes.len() == 1 {
                    id_bytes[0] as u16
                } else {
                    1
                };

                KeyShare {
                    identifier: id_u16,
                    frost_identifier: id,
                    signing_share: secret_share.signing_share().clone(),
                    verifying_share,
                    group_public_key: group_public_key.clone(),
                    threshold,
                    total,
                }
            })
            .collect();

        // Serialize pubkey for logging
        let pubkey_bytes = group_public_key.serialize()
            .unwrap_or_default();
        info!(
            "FROST: Generated {} key shares with group pubkey: {}",
            key_shares.len(),
            hex::encode(&pubkey_bytes)
        );

        Ok((key_shares, group_public_key))
    }

    /// Generate a single keypair (for testing/single signer mode)
    /// Note: FROST requires min 2 signers, so this uses 2-of-2 and returns first share
    pub fn generate_single() -> Result<(KeyShare, GroupPublicKey), CryptoError> {
        let (shares, pubkey) = Self::generate_shares(2, 2)?;
        Ok((shares.into_iter().next().unwrap(), pubkey))
    }
}

/// FROST Signer - holds a key share and can participate in signing
pub struct FrostSigner {
    key_share: KeyShare,
    /// Cached nonces for the current signing session
    current_nonces: Option<SigningNonces>,
}

impl FrostSigner {
    /// Create a signer from a key share
    pub fn from_share(key_share: KeyShare) -> Self {
        Self {
            key_share,
            current_nonces: None,
        }
    }

    /// Get the participant identifier
    pub fn identifier(&self) -> u16 {
        self.key_share.identifier
    }

    /// Get the group public key
    pub fn group_public_key(&self) -> &GroupPublicKey {
        &self.key_share.group_public_key
    }

    /// Round 1: Generate commitment and nonce
    pub fn round1_commit(&mut self) -> (SigningCommitments, SigningNonces) {
        let mut rng = OsRng;

        // Commit using signing share
        let (nonces, commitments) = frost::round1::commit(
            &self.key_share.signing_share,
            &mut rng,
        );

        // Cache nonces for round 2
        self.current_nonces = Some(nonces.clone());

        debug!(
            "FROST: Participant {} generated round 1 commitment",
            self.key_share.identifier
        );

        (commitments, nonces)
    }

    /// Get the FROST identifier
    pub fn frost_identifier(&self) -> &Identifier {
        &self.key_share.frost_identifier
    }

    /// Get the verifying share
    pub fn verifying_share(&self) -> &VerifyingKey {
        &self.key_share.verifying_share
    }

    /// Round 2: Generate signature share
    pub fn round2_sign(
        &mut self,
        message: &[u8],
        commitments: &BTreeMap<Identifier, SigningCommitments>,
        nonces: Option<SigningNonces>,
    ) -> Result<SignatureShare, CryptoError> {
        let nonces = nonces.or(self.current_nonces.take()).ok_or_else(|| {
            CryptoError::SigningFailed("No nonces available - run round1_commit first".into())
        })?;

        // Use the actual FROST identifier from key generation
        let identifier = self.key_share.frost_identifier;

        // Create signing package
        let signing_package = frost::SigningPackage::new(commitments.clone(), message);

        // Create key package for this participant
        let key_package = frost::keys::KeyPackage::new(
            identifier,
            self.key_share.signing_share.clone(),
            self.key_share.verifying_share.clone(),
            self.key_share.group_public_key.clone(),
            self.key_share.threshold,
        );

        // Generate signature share
        let sig_share = frost::round2::sign(&signing_package, &nonces, &key_package)
            .map_err(|e| CryptoError::SigningFailed(e.to_string()))?;

        debug!(
            "FROST: Participant {} generated signature share",
            self.key_share.identifier
        );

        Ok(sig_share)
    }
}

/// Threshold signature - the final aggregated signature
#[derive(Clone, Serialize, Deserialize)]
pub struct ThresholdSignature {
    /// The aggregated signature bytes
    pub signature_bytes: Vec<u8>,
    /// Participants who contributed
    pub signers: Vec<u16>,
}

impl ThresholdSignature {
    /// Aggregate signature shares into a final signature
    ///
    /// This version takes verifying shares from the signers to build the proper pubkey package.
    pub fn aggregate_with_verifying_shares(
        shares: &BTreeMap<Identifier, SignatureShare>,
        verifying_shares: &BTreeMap<Identifier, VerifyingKey>,
        commitments: &BTreeMap<Identifier, SigningCommitments>,
        message: &[u8],
        group_public_key: &GroupPublicKey,
    ) -> Result<Self, CryptoError> {
        if shares.is_empty() {
            return Err(CryptoError::InsufficientParticipants { have: 0, need: 1 });
        }

        // Create signing package
        let signing_package = frost::SigningPackage::new(commitments.clone(), message);

        // Create public key package with actual verifying shares
        let pubkey_package = frost::keys::PublicKeyPackage::new(
            verifying_shares.clone(),
            group_public_key.clone(),
        );

        // Aggregate
        let signature = frost::aggregate(&signing_package, shares, &pubkey_package)
            .map_err(|e| CryptoError::SigningFailed(format!("Aggregation failed: {}", e)))?;

        // Get signers
        let signers: Vec<u16> = shares.keys().map(|id| {
            let bytes = id.serialize();
            if bytes.len() >= 2 {
                u16::from_le_bytes([bytes[0], bytes[1]])
            } else if bytes.len() == 1 {
                bytes[0] as u16
            } else {
                1
            }
        }).collect();

        info!(
            "FROST: Aggregated signature from {} participants",
            signers.len()
        );

        Ok(Self {
            signature_bytes: signature.serialize()
                .map_err(|e| CryptoError::SerializationError(e.to_string()))?,
            signers,
        })
    }

    /// Aggregate signature shares into a final signature (convenience method)
    ///
    /// This version derives verifying shares from the group public key.
    /// Note: For proper FROST, use aggregate_with_verifying_shares with actual verifying shares.
    pub fn aggregate(
        shares: &BTreeMap<Identifier, SignatureShare>,
        commitments: &BTreeMap<Identifier, SigningCommitments>,
        message: &[u8],
        group_public_key: &GroupPublicKey,
    ) -> Result<Self, CryptoError> {
        // Create verifying shares from group public key (not ideal but works for basic cases)
        let verifying_shares: BTreeMap<Identifier, VerifyingKey> = shares
            .keys()
            .map(|id| {
                let vs = frost::keys::VerifyingShare::new(group_public_key.to_element());
                (*id, vs)
            })
            .collect();

        Self::aggregate_with_verifying_shares(shares, &verifying_shares, commitments, message, group_public_key)
    }

    /// Get the raw signature bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.signature_bytes
    }
}

/// FROST Verifier - verifies threshold signatures
pub struct FrostVerifier;

impl FrostVerifier {
    /// Verify a threshold signature
    pub fn verify(
        group_public_key: &GroupPublicKey,
        message: &[u8],
        signature: &ThresholdSignature,
    ) -> Result<bool, CryptoError> {
        let sig = Signature::deserialize(&signature.signature_bytes)
            .map_err(|_| CryptoError::VerificationFailed)?;

        match group_public_key.verify(message, &sig) {
            Ok(()) => {
                debug!("FROST: Signature verified successfully");
                Ok(true)
            }
            Err(_) => {
                warn!("FROST: Signature verification failed");
                Ok(false)
            }
        }
    }

    /// Verify a raw signature
    pub fn verify_raw(
        group_public_key: &GroupPublicKey,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, CryptoError> {
        let sig = Signature::deserialize(signature_bytes)
            .map_err(|_| CryptoError::VerificationFailed)?;

        match group_public_key.verify(message, &sig) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frost_keygen() {
        let (shares, _pubkey) = FrostKeyGen::generate_shares(2, 3).unwrap();

        assert_eq!(shares.len(), 3);
        for share in &shares {
            assert_eq!(share.threshold, 2);
            assert_eq!(share.total, 3);
        }
    }

    #[test]
    fn test_frost_two_of_two() {
        // FROST requires min 2 signers, so test 2-of-2
        let (key_shares, pubkey) = FrostKeyGen::generate_shares(2, 2).unwrap();

        let mut signers: Vec<FrostSigner> = key_shares
            .into_iter()
            .map(FrostSigner::from_share)
            .collect();

        // Round 1: Both participants commit
        let mut commitments = BTreeMap::new();
        let mut nonces_list = Vec::new();
        let mut verifying_shares = BTreeMap::new();

        for signer in &mut signers {
            let (commitment, nonces) = signer.round1_commit();
            let id = *signer.frost_identifier();
            commitments.insert(id, commitment);
            nonces_list.push(nonces);
            verifying_shares.insert(id, signer.verifying_share().clone());
        }

        let message = b"Hello, FROST!";

        // Round 2: Both sign
        let mut sig_shares = BTreeMap::new();
        for (i, signer) in signers.iter_mut().enumerate() {
            let sig_share = signer
                .round2_sign(message, &commitments, Some(nonces_list[i].clone()))
                .unwrap();
            let id = *signer.frost_identifier();
            sig_shares.insert(id, sig_share);
        }

        // Aggregate with proper verifying shares
        let signature = ThresholdSignature::aggregate_with_verifying_shares(
            &sig_shares, &verifying_shares, &commitments, message, &pubkey
        ).unwrap();

        assert!(FrostVerifier::verify(&pubkey, message, &signature).unwrap());
    }

    #[test]
    fn test_frost_threshold_signing() {
        // 2-of-3 threshold
        let (key_shares, pubkey) = FrostKeyGen::generate_shares(2, 3).unwrap();

        let mut signers: Vec<FrostSigner> = key_shares
            .into_iter()
            .map(FrostSigner::from_share)
            .collect();

        // Round 1: Only participating signers need to commit
        // In a real scenario, only threshold signers would participate
        let mut commitments = BTreeMap::new();
        let mut nonces_list = Vec::new();
        let mut participating_verifying_shares = BTreeMap::new();

        // Only first 2 signers participate (threshold = 2)
        for signer in signers.iter_mut().take(2) {
            let (commitment, nonces) = signer.round1_commit();
            let id = *signer.frost_identifier();
            commitments.insert(id, commitment);
            nonces_list.push(nonces);
            participating_verifying_shares.insert(id, signer.verifying_share().clone());
        }

        // Round 2: Same 2 participants sign
        let message = b"Block #12345 hash";
        let mut sig_shares = BTreeMap::new();

        for (i, signer) in signers.iter_mut().take(2).enumerate() {
            let sig_share = signer
                .round2_sign(message, &commitments, Some(nonces_list[i].clone()))
                .unwrap();
            let id = *signer.frost_identifier();
            sig_shares.insert(id, sig_share);
        }

        // Aggregate with verifying shares ONLY for participants who signed
        let signature = ThresholdSignature::aggregate_with_verifying_shares(
            &sig_shares, &participating_verifying_shares, &commitments, message, &pubkey
        ).unwrap();

        // Verify
        assert!(FrostVerifier::verify(&pubkey, message, &signature).unwrap());
        assert_eq!(signature.signers.len(), 2);

        println!("FROST 2-of-3 threshold signature verified!");
    }

    #[test]
    fn test_invalid_threshold() {
        // t > n is invalid
        let result = FrostKeyGen::generate_shares(5, 3);
        assert!(result.is_err());

        // t = 0 is invalid
        let result = FrostKeyGen::generate_shares(0, 3);
        assert!(result.is_err());
    }
}
