use q_types::*;
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Certificate management for vertex availability
/// Certificates prove that 2f+1 nodes acknowledged a vertex
pub struct CertificateStore {
    certificates: RwLock<HashMap<VertexId, Certificate>>,
    pending_acks: RwLock<HashMap<VertexId, BTreeMap<NodeId, Vec<u8>>>>,
}

impl CertificateStore {
    pub fn new() -> Self {
        Self {
            certificates: RwLock::new(HashMap::new()),
            pending_acks: RwLock::new(HashMap::new()),
        }
    }

    /// Store a certificate
    pub async fn store_certificate(&self, certificate: Certificate) -> Result<()> {
        info!("Storing certificate for vertex {} in round {}", 
              hex::encode(certificate.vertex_id), certificate.round);

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
        debug!("Adding acknowledgment for vertex {} from node {:?}", 
               hex::encode(vertex_id), node_id);

        let mut pending_acks = self.pending_acks.write().await;
        let acks = pending_acks.entry(vertex_id).or_insert_with(BTreeMap::new);
        acks.insert(node_id, signature);

        // Check if we have enough acknowledgments (2f+1 = 3 for f=1)
        if acks.len() >= 3 {
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

            info!("Certificate created for vertex {} with {} signatures", 
                  hex::encode(vertex_id), acks.len());
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
    /// Verify certificate signatures
    pub fn verify_certificate(certificate: &Certificate) -> Result<bool> {
        // TODO: Implement proper signature verification
        // For Phase 0, just check threshold
        Ok(certificate.signatures.len() >= 3 && certificate.threshold_met)
    }

    /// Verify individual acknowledgment
    pub fn verify_acknowledgment(
        vertex_id: &VertexId,
        node_id: &NodeId,
        signature: &[u8],
    ) -> Result<bool> {
        // TODO: Implement Ed25519 signature verification
        // For Phase 0, accept all signatures
        Ok(!signature.is_empty())
    }

    /// Extract voting power from certificate
    pub fn get_voting_power(certificate: &Certificate) -> u64 {
        // For Phase 0, assume equal voting power (1 per node)
        certificate.signatures.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_certificate_store_creation() {
        let store = CertificateStore::new();
        let stats = store.get_stats().await;
        
        assert_eq!(stats.total_certificates, 0);
        assert_eq!(stats.pending_acknowledgments, 0);
    }

    #[tokio::test]
    async fn test_certificate_storage() {
        let store = CertificateStore::new();
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
        let store = CertificateStore::new();
        let vertex_id = [1u8; 32];
        
        // Add first acknowledgment
        let result = store.add_acknowledgment(
            vertex_id,
            [1u8; 32],
            vec![1, 2, 3],
        ).await.unwrap();
        assert!(result.is_none()); // Not enough acks yet

        // Add second acknowledgment
        let result = store.add_acknowledgment(
            vertex_id,
            [2u8; 32],
            vec![4, 5, 6],
        ).await.unwrap();
        assert!(result.is_none()); // Still not enough

        // Add third acknowledgment - should create certificate
        let result = store.add_acknowledgment(
            vertex_id,
            [3u8; 32],
            vec![7, 8, 9],
        ).await.unwrap();
        assert!(result.is_some()); // Certificate created!

        let certificate = result.unwrap();
        assert_eq!(certificate.signatures.len(), 3);
        assert!(certificate.threshold_met);
    }

    #[test]
    fn test_certificate_verification() {
        let mut signatures = BTreeMap::new();
        signatures.insert([1u8; 32], vec![1, 2, 3]);
        signatures.insert([2u8; 32], vec![4, 5, 6]);
        signatures.insert([3u8; 32], vec![7, 8, 9]);

        let certificate = Certificate {
            vertex_id: [1u8; 32],
            round: 1,
            signatures,
            threshold_met: true,
        };

        let is_valid = CertificateVerifier::verify_certificate(&certificate).unwrap();
        assert!(is_valid);
    }
}