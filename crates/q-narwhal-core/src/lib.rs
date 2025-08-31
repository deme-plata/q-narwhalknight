/// Q-Narwhal: DAG-based mempool implementation
/// Phase 0: Classical Ed25519 implementation
/// Future phases will add post-quantum cryptography

use q_types::*;
use anyhow::Result;
use async_trait::async_trait;
use std::collections::{BTreeMap, HashMap, HashSet};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub mod reliable_broadcast;
pub mod certificate;
pub mod vertex_store;

pub use reliable_broadcast::ReliableBroadcast;
pub use certificate::{Certificate, CertificateStore};
pub use vertex_store::VertexStore;

/// Narwhal mempool core implementation
pub struct NarwhalCore {
    pub node_id: NodeId,
    pub vertex_store: VertexStore,
    pub certificate_store: CertificateStore,
    pub reliable_broadcast: ReliableBroadcast,
    pub current_round: RwLock<Round>,
}

impl NarwhalCore {
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            vertex_store: VertexStore::new(),
            certificate_store: CertificateStore::new(),
            reliable_broadcast: ReliableBroadcast::new(node_id),
            current_round: RwLock::new(0),
        }
    }

    /// Create a new vertex with transactions
    pub async fn create_vertex(
        &self,
        transactions: Vec<Transaction>,
        parents: Vec<VertexId>,
    ) -> Result<Vertex> {
        let round = *self.current_round.read().await;
        
        // Compute transaction root
        let tx_root = self.compute_tx_root(&transactions);
        
        let vertex = Vertex {
            id: [0u8; 32], // Will be computed after signing
            round,
            author: self.node_id,
            tx_root,
            parents,
            transactions,
            signature: vec![], // Will be added after signing
            timestamp: chrono::Utc::now(),
        };

        // TODO: Sign vertex
        // let signed_vertex = self.sign_vertex(vertex).await?;
        
        Ok(vertex)
    }

    /// Process received vertex
    pub async fn process_vertex(&self, vertex: Vertex) -> Result<Option<Certificate>> {
        info!("Processing vertex from author {:?} for round {}", 
              vertex.author, vertex.round);

        // Validate vertex
        self.validate_vertex(&vertex).await?;

        // Store vertex
        self.vertex_store.store_vertex(vertex.clone()).await?;

        // Trigger reliable broadcast
        self.reliable_broadcast.broadcast_vertex(vertex).await?;

        // Check if we can create a certificate
        if self.has_sufficient_acknowledgements(&vertex.id).await? {
            let certificate = self.create_certificate(&vertex.id).await?;
            self.certificate_store.store_certificate(certificate.clone()).await?;
            return Ok(Some(certificate));
        }

        Ok(None)
    }

    /// Validate vertex structure and signatures
    async fn validate_vertex(&self, vertex: &Vertex) -> Result<()> {
        // Check round validity
        let current_round = *self.current_round.read().await;
        if vertex.round > current_round + 1 {
            return Err(anyhow::anyhow!("Vertex from future round"));
        }

        // Validate transaction root
        let computed_root = self.compute_tx_root(&vertex.transactions);
        if computed_root != vertex.tx_root {
            return Err(anyhow::anyhow!("Invalid transaction root"));
        }

        // TODO: Validate signature
        // self.verify_vertex_signature(vertex)?;

        // TODO: Validate parent references
        // self.validate_parents(&vertex.parents).await?;

        Ok(())
    }

    /// Check if vertex has sufficient acknowledgements for certificate
    async fn has_sufficient_acknowledgements(&self, vertex_id: &VertexId) -> Result<bool> {
        // TODO: Implement threshold check (2f+1)
        // For now, return false to avoid certificate creation
        Ok(false)
    }

    /// Create certificate from acknowledgements
    async fn create_certificate(&self, vertex_id: &VertexId) -> Result<Certificate> {
        // TODO: Collect acknowledgements and create certificate
        let certificate = Certificate {
            vertex_id: *vertex_id,
            round: 0, // TODO: Get from vertex
            signatures: BTreeMap::new(),
            threshold_met: true,
        };
        Ok(certificate)
    }

    /// Compute Merkle root of transactions
    fn compute_tx_root(&self, transactions: &[Transaction]) -> TxHash {
        use sha3::{Digest, Sha3_256};
        
        if transactions.is_empty() {
            return [0u8; 32];
        }

        // Simple hash of all transaction IDs (Phase 0)
        // TODO: Implement proper Merkle tree
        let mut hasher = Sha3_256::new();
        for tx in transactions {
            hasher.update(tx.id);
        }
        hasher.finalize().into()
    }

    /// Advance to next round
    pub async fn advance_round(&self) -> Result<()> {
        let mut current_round = self.current_round.write().await;
        *current_round += 1;
        info!("Advanced to round {}", *current_round);
        Ok(())
    }

    /// Get current round
    pub async fn get_current_round(&self) -> Round {
        *self.current_round.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_vertex_creation() {
        let node_id = [1u8; 32];
        let narwhal = NarwhalCore::new(node_id);
        
        let tx = Transaction {
            id: [1u8; 32],
            from: [2u8; 32],
            to: [3u8; 32],
            amount: 1000,
            fee: 10,
            nonce: 1,
            signature: vec![],
            timestamp: chrono::Utc::now(),
        };

        let vertex = narwhal
            .create_vertex(vec![tx], vec![])
            .await
            .unwrap();

        assert_eq!(vertex.author, node_id);
        assert_eq!(vertex.round, 0);
        assert_eq!(vertex.transactions.len(), 1);
    }

    #[tokio::test]
    async fn test_round_advancement() {
        let node_id = [1u8; 32];
        let narwhal = NarwhalCore::new(node_id);
        
        assert_eq!(narwhal.get_current_round().await, 0);
        
        narwhal.advance_round().await.unwrap();
        assert_eq!(narwhal.get_current_round().await, 1);
    }
}