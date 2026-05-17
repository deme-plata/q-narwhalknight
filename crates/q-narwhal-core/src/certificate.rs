use anyhow::{anyhow, Result};
use q_types::*;
use std::collections::{BTreeMap, HashMap};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::validator_set::ValidatorSet;

/// Certificate management for vertex availability
/// Certificates prove that 2f+1 nodes acknowledged a vertex
pub struct CertificateStore {
    certificates: RwLock<HashMap<VertexId, Certificate>>,
    pending_acks: RwLock<HashMap<VertexId, BTreeMap<NodeId, Vec<u8>>>>,
    validator_set: ValidatorSet,
}

impl CertificateStore {
    pub fn new(validator_set: ValidatorSet) -> Self {
        Self {
            certificates: RwLock::new(HashMap::new()),
            pending_acks: RwLock::new(HashMap::new()),
            validator_set,
        }
    }

    /// Get the validator set
    pub fn validator_set(&self) -> &ValidatorSet {
        &self.validator_set
    }

    /// Store a certificate
    pub async fn store_certificate(&self, certificate: Certificate) -> Result<()> {
        info!(
            "Storing certificate for vertex {} in round {}",
            hex::encode(certificate.vertex_id),
            certificate.round
        );

        let mut certificates = self.certificates.write().await;
        certificates.insert(certificate.vertex_id, certificate);
        Ok(())
    }

    /// Get certificate for a vertex
    pub async fn get_certificate(&self, vertex_id: &VertexId) -> Option<Certificate> {
        let certificates = self.certificates.read().await;
        certificates.get(vertex_id).cloned()
    }

    /// Add acknowledgment for a vertex
    pub async fn add_acknowledgment(
        &self,
        vertex_id: VertexId,
        node_id: NodeId,
        signature: Vec<u8>,
    ) -> Result<Option<Certificate>> {
        debug!(
            "Adding acknowledgment for vertex {} from node {:?}",
            hex::encode(vertex_id),
            node_id
        );

        let mut pending_acks = self.pending_acks.write().await;
        let acks = pending_acks.entry(vertex_id).or_insert_with(BTreeMap::new);
        acks.insert(node_id, signature);

        // Check if we have Byzantine quorum (dynamic 2f+1 based on validator set)
        let acks_len = acks.len();
        if self.validator_set.has_quorum(acks_len) {
            let certificate = Certificate {
                vertex_id,
                round: 0, // TODO: Get from vertex
                signatures: acks.clone(),
                threshold_met: true,
            };

            // Store the certificate
            let mut certificates = self.certificates.write().await;
            certificates.insert(vertex_id, certificate.clone());

            // Remove from pending
            pending_acks.remove(&vertex_id);

            info!(
                "Certificate created for vertex {} with {} signatures",
                hex::encode(vertex_id),
                acks_len
            );
            return Ok(Some(certificate));
        }

        Ok(None)
    }

    /// List all certificates for a round
    pub async fn get_certificates_for_round(&self, round: Round) -> Vec<Certificate> {
        let certificates = self.certificates.read().await;
        certificates
            .values()
            .filter(|cert| cert.round == round)
            .cloned()
            .collect()
    }

    /// Check if vertex has certificate
    pub async fn has_certificate(&self, vertex_id: &VertexId) -> bool {
        let certificates = self.certificates.read().await;
        certificates.contains_key(vertex_id)
    }

    /// Get certificate statistics
    pub async fn get_stats(&self) -> CertificateStats {
        let certificates = self.certificates.read().await;
        let pending_acks = self.pending_acks.read().await;

        let mut round_counts = BTreeMap::new();
        for cert in certificates.values() {
            *round_counts.entry(cert.round).or_insert(0) += 1;
        }

        CertificateStats {
            total_certificates: certificates.len(),
            pending_acknowledgments: pending_acks.len(),
            certificates_by_round: round_counts,
        }
    }

    /// Clean up old certificates and pending acks
    pub async fn cleanup_old_data(&self, keep_rounds: u64) -> Result<()> {
        let current_round = 0u64; // TODO: Get current round from consensus
        let cutoff_round = current_round.saturating_sub(keep_rounds);

        let mut certificates = self.certificates.write().await;
        certificates.retain(|_, cert| cert.round >= cutoff_round);

        info!("Cleaned up certificates older than round {}", cutoff_round);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CertificateStats {
    pub total_certificates: usize,
    pub pending_acknowledgments: usize,
    pub certificates_by_round: BTreeMap<Round, usize>,
}

/// Certificate verification utilities
pub struct CertificateVerifier;

impl CertificateVerifier {
    /// Verify certificate signatures with Byzantine quorum
    ///
    /// # Security Requirements
    /// - Verifies ALL signatures cryptographically (no shortcuts!)
    /// - Checks dynamic Byzantine quorum (2f+1) based on validator set
    /// - Ensures all signers are in the validator set
    ///
    /// # Byzantine Attack Prevention
    /// - Prevents single Byzantine node from forging certificates
    /// - Prevents quorum bypass with fake signatures
    /// - Prevents sybil attacks (only registered validators count)
    pub fn verify_certificate(
        certificate: &Certificate,
        validator_set: &ValidatorSet,
        round: Round,
    ) -> Result<bool> {
        // 1. Verify we have Byzantine quorum
        let signature_count = certificate.signatures.len();
        if !validator_set.has_quorum(signature_count) {
            warn!(
                "Certificate has {} signatures, but quorum requires {}",
                signature_count,
                validator_set.quorum_threshold()
            );
            return Ok(false);
        }

        // 2. Verify all signers are valid validators
        let signer_ids: Vec<NodeId> = certificate.signatures.keys().copied().collect();
        validator_set.verify_signers(&signer_ids)?;

        // 3. Cryptographically verify EACH signature (CRITICAL - no shortcuts!)
        let mut valid_signatures = 0;
        for (node_id, signature) in &certificate.signatures {
            match Self::verify_acknowledgment(
                &certificate.vertex_id,
                node_id,
                signature,
                round,
                validator_set,
            ) {
                Ok(true) => {
                    valid_signatures += 1;
                }
                Ok(false) => {
                    warn!(
                        "Invalid signature from validator {} for vertex {}",
                        hex::encode(node_id),
                        hex::encode(certificate.vertex_id)
                    );
                    return Ok(false); // FAIL HARD - any invalid signature fails the certificate
                }
                Err(e) => {
                    warn!(
                        "Signature verification error for validator {}: {}",
                        hex::encode(node_id),
                        e
                    );
                    return Ok(false);
                }
            }
        }

        // 4. Final quorum check with verified signatures
        if !validator_set.has_quorum(valid_signatures) {
            warn!(
                "Only {} valid signatures (quorum requires {})",
                valid_signatures,
                validator_set.quorum_threshold()
            );
            return Ok(false);
        }

        info!(
            "✅ Certificate verified: {} valid signatures (quorum: {})",
            valid_signatures,
            validator_set.quorum_threshold()
        );
        Ok(true)
    }

    /// Verify individual acknowledgment with Ed25519 signature
    ///
    /// # Message Format
    /// The signed message is: NARWHAL_ACK || vertex_id || round
    /// This prevents signature replay attacks across different vertices/rounds
    ///
    /// # Security
    /// - Cryptographically verifies Ed25519 signature
    /// - Checks signer is in validator set
    /// - Prevents replay attacks with round binding
    pub fn verify_acknowledgment(
        vertex_id: &VertexId,
        node_id: &NodeId,
        signature: &[u8],
        round: Round,
        validator_set: &ValidatorSet,
    ) -> Result<bool> {
        // 1. Verify node_id is in validator set
        let public_key = validator_set
            .get_public_key(node_id)
            .ok_or_else(|| anyhow!("Node {} not in validator set", hex::encode(node_id)))?;

        // 2. Reconstruct the signed message
        // Format: "NARWHAL_ACK" || vertex_id || round
        let mut message = Vec::new();
        message.extend_from_slice(b"NARWHAL_ACK"); // Domain separator
        message.extend_from_slice(vertex_id);
        message.extend_from_slice(&round.to_be_bytes());

        // 3. Parse Ed25519 signature
        let signature_bytes: &[u8; 64] = signature
            .try_into()
            .map_err(|_| anyhow!("Signature must be exactly 64 bytes, got {}", signature.len()))?;

        let sig = Signature::from_bytes(signature_bytes);

        // 4. Verify Ed25519 signature
        match public_key.verify_strict(&message, &sig) {
            Ok(()) => {
                debug!(
                    "✅ Valid signature from {} for vertex {}",
                    hex::encode(node_id),
                    hex::encode(vertex_id)
                );
                Ok(true)
            }
            Err(e) => {
                warn!(
                    "❌ Invalid signature from {} for vertex {}: {}",
                    hex::encode(node_id),
                    hex::encode(vertex_id),
                    e
                );
                Ok(false)
            }
        }
    }

    /// Extract voting power from certificate (Phase 0: equal power)
    pub fn get_voting_power(certificate: &Certificate) -> u64 {
        // For Phase 0, assume equal voting power (1 per node)
        // Phase 1+: Can implement stake-weighted voting
        certificate.signatures.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validator_set::ValidatorInfo;
    use ed25519_dalek::SigningKey;
    use rand::{rngs::OsRng, RngCore};

    fn create_test_validator_set() -> ValidatorSet {
        use rand::TryRngCore as _;  // For try_fill_bytes (rand 0.9)
        let mut validators = Vec::new();
        for _ in 0..4 {
            let mut secret_bytes = [0u8; 32];
            OsRng.try_fill_bytes(&mut secret_bytes).unwrap();
            let signing_key = SigningKey::from_bytes(&secret_bytes);
            let public_key = signing_key.verifying_key();

            // NodeId = hash of public key
            let node_id = {
                use sha3::{Digest, Sha3_256};
                let mut hasher = Sha3_256::new();
                hasher.update(public_key.as_bytes());
                hasher.finalize().into()
            };

            validators.push(ValidatorInfo {
                node_id,
                public_key,
                stake: 1,
                active: true,
            });
        }
        ValidatorSet::new(validators).unwrap()
    }

    #[tokio::test]
    async fn test_certificate_store_creation() {
        let validator_set = create_test_validator_set();
        let store = CertificateStore::new(validator_set);
        let stats = store.get_stats().await;

        assert_eq!(stats.total_certificates, 0);
        assert_eq!(stats.pending_acknowledgments, 0);
    }

    #[tokio::test]
    async fn test_certificate_storage() {
        let validator_set = create_test_validator_set();
        let store = CertificateStore::new(validator_set);
        let vertex_id = [1u8; 32];

        let certificate = Certificate {
            vertex_id,
            round: 1,
            signatures: BTreeMap::new(),
            threshold_met: true,
        };

        store.store_certificate(certificate.clone()).await.unwrap();

        let retrieved = store.get_certificate(&vertex_id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().vertex_id, vertex_id);
    }

    #[tokio::test]
    async fn test_acknowledgment_accumulation() {
        let validator_set = create_test_validator_set();
        let store = CertificateStore::new(validator_set);
        let vertex_id = [1u8; 32];

        // Add first acknowledgment
        let result = store
            .add_acknowledgment(vertex_id, [1u8; 32], vec![1, 2, 3])
            .await
            .unwrap();
        assert!(result.is_none()); // Not enough acks yet

        // Add second acknowledgment
        let result = store
            .add_acknowledgment(vertex_id, [2u8; 32], vec![4, 5, 6])
            .await
            .unwrap();
        assert!(result.is_none()); // Still not enough

        // Add third acknowledgment - should create certificate
        let result = store
            .add_acknowledgment(vertex_id, [3u8; 32], vec![7, 8, 9])
            .await
            .unwrap();
        assert!(result.is_some()); // Certificate created!

        let certificate = result.unwrap();
        assert_eq!(certificate.signatures.len(), 3);
        assert!(certificate.threshold_met);
    }

    #[test]
    fn test_certificate_verification_placeholder() {
        // This test is a placeholder - real Ed25519 verification tests would require
        // proper signing keys and signatures. For now, we test the validator_set module
        // which has comprehensive tests for quorum thresholds.
        let validator_set = create_test_validator_set();
        assert_eq!(validator_set.total_validators(), 4);
        assert_eq!(validator_set.quorum_threshold(), 3); // 2f+1 for n=4
    }
}
