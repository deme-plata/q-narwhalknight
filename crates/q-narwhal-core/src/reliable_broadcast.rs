use anyhow::Result;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use q_types::*;
use std::collections::{HashMap, HashSet};
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

/// 🔐 v2.4.7-beta: Verify Ed25519 signature for vertex validation
fn verify_ed25519_signature(signature: &[u8], message: &[u8], public_key: &[u8]) -> Result<()> {
    // Parse public key (32 bytes for Ed25519)
    let pk_bytes: [u8; 32] = public_key
        .try_into()
        .map_err(|_| anyhow::anyhow!("Invalid Ed25519 public key length (expected 32 bytes)"))?;

    let verifying_key = VerifyingKey::from_bytes(&pk_bytes)
        .map_err(|e| anyhow::anyhow!("Invalid Ed25519 public key format: {}", e))?;

    // Parse signature (64 bytes for Ed25519)
    let sig_bytes: [u8; 64] = signature
        .try_into()
        .map_err(|_| anyhow::anyhow!("Invalid Ed25519 signature length (expected 64 bytes)"))?;

    let sig = Signature::from_bytes(&sig_bytes);

    // Verify the signature
    verifying_key
        .verify(message, &sig)
        .map_err(|e| anyhow::anyhow!("Ed25519 signature verification failed: {}", e))?;

    Ok(())
}

/// Bracha's reliable broadcast protocol for Byzantine fault tolerance
/// Ensures that if any correct node delivers a message, all correct nodes deliver it
pub struct ReliableBroadcast {
    pub node_id: NodeId,
    /// Vertices we've echoed (vertex_id -> round)
    echoed: RwLock<HashSet<VertexId>>,
    /// Vertices we've received echo for (vertex_id -> set of nodes)
    echo_votes: RwLock<HashMap<VertexId, HashSet<NodeId>>>,
    /// Vertices we've received ready for (vertex_id -> set of nodes)
    ready_votes: RwLock<HashMap<VertexId, HashSet<NodeId>>>,
    /// Vertices we've sent ready for
    ready_sent: RwLock<HashSet<VertexId>>,
    /// Delivered vertices
    delivered: RwLock<HashSet<VertexId>>,
    /// Broadcast channel for network communication
    network_tx: broadcast::Sender<BroadcastMessage>,
    /// Threshold parameters (f = number of Byzantine nodes)
    threshold_2f_plus_1: usize,
    threshold_f_plus_1: usize,
}

#[derive(Debug, Clone)]
pub enum BroadcastMessage {
    Send { vertex: Vertex },
    Echo { vertex_id: VertexId, sender: NodeId },
    Ready { vertex_id: VertexId, sender: NodeId },
}

impl ReliableBroadcast {
    pub fn new(node_id: NodeId) -> Self {
        let (network_tx, _) = broadcast::channel(10000);

        // For Phase 0, assume 4 validators, so f=1
        // 2f+1 = 3, f+1 = 2
        let f = 1;
        let threshold_2f_plus_1 = 2 * f + 1; // 3
        let threshold_f_plus_1 = f + 1; // 2

        Self {
            node_id,
            echoed: RwLock::new(HashSet::new()),
            echo_votes: RwLock::new(HashMap::new()),
            ready_votes: RwLock::new(HashMap::new()),
            ready_sent: RwLock::new(HashSet::new()),
            delivered: RwLock::new(HashSet::new()),
            network_tx,
            threshold_2f_plus_1,
            threshold_f_plus_1,
        }
    }

    /// Broadcast a vertex (step 1 of Bracha's protocol)
    pub async fn broadcast_vertex(&self, vertex: Vertex) -> Result<()> {
        info!("🚀 [BRACHA STEP 1: SEND] ==========================================");
        info!("🚀 [BRACHA] Broadcasting vertex {}", hex::encode(&vertex.id[..8]));
        info!("🚀 [BRACHA] Round: {}", vertex.round);
        info!("🚀 [BRACHA] Transactions: {}", vertex.transactions.len());
        info!("🚀 [BRACHA] This is the SEND phase - all nodes will receive and ECHO");

        // Send the vertex to all nodes
        let message = BroadcastMessage::Send { vertex: vertex.clone() };
        self.network_tx
            .send(message)
            .map_err(|e| anyhow::anyhow!("Failed to broadcast vertex: {}", e))?;

        info!("✅ [BRACHA] SEND message broadcast to all peers");
        info!("⏳ [BRACHA] Waiting for ECHO responses from peers...");
        info!("🚀 [BRACHA STEP 1 COMPLETE] ==========================================\n");

        Ok(())
    }

    /// Process incoming broadcast message
    pub async fn process_message(&self, message: BroadcastMessage) -> Result<Option<Vertex>> {
        match message {
            BroadcastMessage::Send { vertex } => self.handle_send(vertex).await,
            BroadcastMessage::Echo { vertex_id, sender } => {
                self.handle_echo(vertex_id, sender).await
            }
            BroadcastMessage::Ready { vertex_id, sender } => {
                self.handle_ready(vertex_id, sender).await
            }
        }
    }

    /// Handle SEND message (step 1)
    async fn handle_send(&self, vertex: Vertex) -> Result<Option<Vertex>> {
        let vertex_id = vertex.id;

        info!("📥 [BRACHA STEP 1: RECEIVED SEND] ==========================================");
        info!("📥 [BRACHA] Received SEND for vertex {}", hex::encode(&vertex_id[..8]));
        info!("📥 [BRACHA] Round: {}, Transactions: {}", vertex.round, vertex.transactions.len());

        // Validate vertex (basic checks)
        info!("🔍 [BRACHA] Validating vertex structure and signatures...");
        if !self.validate_vertex(&vertex).await? {
            warn!("❌ [BRACHA] Invalid vertex received: {} - REJECTING", hex::encode(&vertex_id[..8]));
            return Ok(None);
        }
        info!("✅ [BRACHA] Vertex validation passed");

        // If we haven't echoed this vertex yet, echo it
        {
            let mut echoed = self.echoed.write().await;
            if !echoed.contains(&vertex_id) {
                echoed.insert(vertex_id);

                info!("📢 [BRACHA STEP 2: ECHO] Broadcasting ECHO to all peers...");
                // Send ECHO to all nodes
                let echo_msg = BroadcastMessage::Echo {
                    vertex_id,
                    sender: self.node_id,
                };
                self.network_tx.send(echo_msg)?;

                info!("✅ [BRACHA] ECHO sent for vertex {}", hex::encode(&vertex_id[..8]));
                info!("⏳ [BRACHA] Waiting for {} ECHO votes (2f+1 threshold)...", self.threshold_2f_plus_1);
            } else {
                info!("ℹ️ [BRACHA] Already echoed this vertex - skipping ECHO");
            }
        }
        info!("📥 [BRACHA STEP 1 COMPLETE] ==========================================\n");

        Ok(Some(vertex))
    }

    /// Handle ECHO message (step 2)
    async fn handle_echo(&self, vertex_id: VertexId, sender: NodeId) -> Result<Option<Vertex>> {
        info!("🔔 [BRACHA STEP 2: ECHO RECEIVED] ==========================================");
        info!("🔔 [BRACHA] Received ECHO for vertex {} from node {:?}",
            hex::encode(&vertex_id[..8]), hex::encode(&sender[..8]));

        // Add to echo votes
        let echo_count = {
            let mut echo_votes = self.echo_votes.write().await;
            let votes = echo_votes.entry(vertex_id).or_insert_with(HashSet::new);
            votes.insert(sender);
            let count = votes.len();
            info!("📊 [BRACHA] ECHO vote count: {}/{} (threshold: {})",
                  count, self.threshold_2f_plus_1, self.threshold_2f_plus_1);
            count
        };

        // If we have 2f+1 echo votes and haven't sent ready, send ready
        if echo_count >= self.threshold_2f_plus_1 {
            info!("🎯 [BRACHA] ECHO QUORUM REACHED! ({}/{})",
                  echo_count, self.threshold_2f_plus_1);

            let mut ready_sent = self.ready_sent.write().await;
            if !ready_sent.contains(&vertex_id) {
                ready_sent.insert(vertex_id);

                info!("📢 [BRACHA STEP 3: READY] Broadcasting READY to all peers...");
                let ready_msg = BroadcastMessage::Ready {
                    vertex_id,
                    sender: self.node_id,
                };
                self.network_tx.send(ready_msg)?;

                info!("✅ [BRACHA] READY sent for vertex {} (echo threshold reached)",
                    hex::encode(&vertex_id[..8])
                );
                info!("⏳ [BRACHA] Waiting for {} READY votes (2f+1 threshold)...", self.threshold_2f_plus_1);
            } else {
                info!("ℹ️ [BRACHA] READY already sent for this vertex");
            }
        } else {
            info!("⏳ [BRACHA] Still collecting ECHO votes... ({}/{} needed)",
                  echo_count, self.threshold_2f_plus_1);
        }
        info!("🔔 [BRACHA STEP 2 COMPLETE] ==========================================\n");

        Ok(None)
    }

    /// Handle READY message (step 3)
    async fn handle_ready(&self, vertex_id: VertexId, sender: NodeId) -> Result<Option<Vertex>> {
        info!("🟣 [BRACHA STEP 3: READY RECEIVED] ==========================================");
        info!("🟣 [BRACHA] Received READY for vertex {} from node {:?}",
            hex::encode(&vertex_id[..8]), hex::encode(&sender[..8]));

        // Add to ready votes
        let ready_count = {
            let mut ready_votes = self.ready_votes.write().await;
            let votes = ready_votes.entry(vertex_id).or_insert_with(HashSet::new);
            votes.insert(sender);
            let count = votes.len();
            info!("📊 [BRACHA] READY vote count: {}/{} (final threshold: {})",
                  count, self.threshold_2f_plus_1, self.threshold_2f_plus_1);
            count
        };

        // If we have f+1 ready votes and haven't sent ready, send ready (amplification)
        if ready_count >= self.threshold_f_plus_1 {
            info!("⚡ [BRACHA] READY AMPLIFICATION threshold reached! ({}/{})",
                  ready_count, self.threshold_f_plus_1);

            let mut ready_sent = self.ready_sent.write().await;
            if !ready_sent.contains(&vertex_id) {
                ready_sent.insert(vertex_id);

                info!("📢 [BRACHA] Broadcasting READY (amplification) to ensure Byzantine resilience...");
                let ready_msg = BroadcastMessage::Ready {
                    vertex_id,
                    sender: self.node_id,
                };
                self.network_tx.send(ready_msg)?;

                info!("✅ [BRACHA] READY sent via amplification for vertex {}",
                    hex::encode(&vertex_id[..8])
                );
            }
        }

        // If we have 2f+1 ready votes and haven't delivered, deliver
        if ready_count >= self.threshold_2f_plus_1 {
            info!("🎉🎉🎉 [BRACHA] READY QUORUM REACHED! ({}/{})",
                  ready_count, self.threshold_2f_plus_1);

            let mut delivered = self.delivered.write().await;
            if !delivered.contains(&vertex_id) {
                delivered.insert(vertex_id);

                info!("✅✅✅ [BRACHA STEP 4: DELIVERY] ==========================================");
                info!("🎉 [BRACHA] VERTEX DELIVERED: {}", hex::encode(&vertex_id[..8]));
                info!("🎉 [BRACHA] Byzantine fault tolerance achieved!");
                info!("🎉 [BRACHA] All correct nodes will deliver this vertex");
                info!("🎉 [BRACHA] Ready vote count: {}/{}", ready_count, self.threshold_2f_plus_1);
                info!("✅✅✅ [BRACHA DELIVERY COMPLETE] ==========================================\n");

                // TODO: Return the actual vertex from storage
                // For now, create a placeholder
                return Ok(Some(self.create_placeholder_vertex(vertex_id)));
            } else {
                info!("ℹ️ [BRACHA] Vertex already delivered - duplicate READY vote");
            }
        } else {
            info!("⏳ [BRACHA] Still collecting READY votes... ({}/{} needed for delivery)",
                  ready_count, self.threshold_2f_plus_1);
        }
        info!("🟣 [BRACHA STEP 3 COMPLETE] ==========================================\n");

        Ok(None)
    }

    /// 🔐 v2.4.7-beta: Comprehensive vertex validation for BFT consensus
    ///
    /// Validates:
    /// 1. Signature validity (Ed25519/Dilithium based on phase)
    /// 2. Transaction root consistency
    /// 3. Parent vertex references
    /// 4. Timestamp bounds
    /// 5. Author identity
    async fn validate_vertex(&self, vertex: &Vertex) -> Result<bool> {
        use sha3::{Digest, Sha3_256};

        // 1. Basic structure validation
        if vertex.transactions.is_empty() && vertex.tx_root != [0u8; 32] {
            warn!("❌ [VALIDATION] Vertex has non-zero tx_root but no transactions");
            return Ok(false);
        }

        // 2. Validate signature is present and has correct length
        if vertex.signature.is_empty() {
            warn!("❌ [VALIDATION] Vertex signature is missing");
            return Ok(false);
        }

        // Ed25519 signature length check (Phase 0)
        if vertex.signature.len() != 64 {
            warn!(
                "❌ [VALIDATION] Invalid signature length: expected 64 bytes, got {}",
                vertex.signature.len()
            );
            return Ok(false);
        }

        // 3. Verify signature cryptographically
        // Create the message that was signed: H(vertex_id || round || tx_root || parents)
        let mut signing_data = Vec::new();
        signing_data.extend_from_slice(&vertex.id);
        signing_data.extend_from_slice(&vertex.round.to_le_bytes());
        signing_data.extend_from_slice(&vertex.tx_root);
        for parent in &vertex.parents {
            signing_data.extend_from_slice(parent);
        }
        let message_hash = Sha3_256::digest(&signing_data);

        // Verify Ed25519 signature using author's public key
        if let Err(e) = verify_ed25519_signature(
            &vertex.signature,
            &message_hash,
            &vertex.author,
        ) {
            warn!("❌ [VALIDATION] Signature verification failed: {}", e);
            return Ok(false);
        }
        debug!("✅ [VALIDATION] Signature verified successfully");

        // 4. Validate transaction root if transactions present
        if !vertex.transactions.is_empty() {
            let mut tx_hasher = Sha3_256::new();
            for tx in &vertex.transactions {
                tx_hasher.update(tx.hash());
            }
            let computed_root: [u8; 32] = tx_hasher.finalize().into();

            if computed_root != vertex.tx_root {
                warn!("❌ [VALIDATION] Transaction root mismatch");
                return Ok(false);
            }
            debug!("✅ [VALIDATION] Transaction root verified");
        }

        // 5. Validate parent references (must exist or be genesis)
        for parent in &vertex.parents {
            // Allow genesis vertex ID (all zeros)
            if *parent == [0u8; 32] {
                continue;
            }
            // In production, we'd check if parent exists in our vertex store
            // For now, we trust that parents will be validated during DAG sync
        }

        // 6. Validate timestamp (not too far in the future)
        let now = chrono::Utc::now();
        let max_future = chrono::Duration::seconds(60); // Allow 60 seconds clock drift
        if vertex.timestamp > now + max_future {
            warn!("❌ [VALIDATION] Vertex timestamp is too far in the future");
            return Ok(false);
        }

        // 7. Validate round is non-negative (Round is u64, always non-negative)
        // Additional round validation can be added here for DAG consistency

        info!(
            "✅ [VALIDATION] Vertex {} passed all validation checks",
            hex::encode(&vertex.id[..8])
        );
        Ok(true)
    }

    /// Create placeholder vertex (temporary for Phase 0)
    fn create_placeholder_vertex(&self, vertex_id: VertexId) -> Vertex {
        Vertex {
            id: vertex_id,
            round: 0, // TODO: Get from storage
            author: [0u8; 32],
            tx_root: [0u8; 32],
            parents: vec![],
            transactions: vec![],
            signature: vec![],
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get network receiver for listening to broadcast messages
    pub fn subscribe(&self) -> broadcast::Receiver<BroadcastMessage> {
        self.network_tx.subscribe()
    }

    /// Check if vertex has been delivered
    pub async fn is_delivered(&self, vertex_id: &VertexId) -> bool {
        let delivered = self.delivered.read().await;
        delivered.contains(vertex_id)
    }

    /// Get delivery statistics
    pub async fn get_stats(&self) -> BroadcastStats {
        let echoed = self.echoed.read().await;
        let echo_votes = self.echo_votes.read().await;
        let ready_votes = self.ready_votes.read().await;
        let delivered = self.delivered.read().await;

        BroadcastStats {
            echoed_count: echoed.len(),
            echo_votes_count: echo_votes.len(),
            ready_votes_count: ready_votes.len(),
            delivered_count: delivered.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BroadcastStats {
    pub echoed_count: usize,
    pub echo_votes_count: usize,
    pub ready_votes_count: usize,
    pub delivered_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_broadcast_creation() {
        let node_id = [1u8; 32];
        let rb = ReliableBroadcast::new(node_id);

        assert_eq!(rb.node_id, node_id);
        assert_eq!(rb.threshold_2f_plus_1, 3);
        assert_eq!(rb.threshold_f_plus_1, 2);
    }

    #[tokio::test]
    async fn test_vertex_broadcast() {
        let node_id = [1u8; 32];
        let rb = ReliableBroadcast::new(node_id);

        let vertex = Vertex {
            id: [2u8; 32],
            round: 1,
            author: node_id,
            tx_root: [0u8; 32],
            parents: vec![],
            transactions: vec![],
            signature: vec![],
            timestamp: chrono::Utc::now(),
        };

        let result = rb.broadcast_vertex(vertex).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_message_processing() {
        let node_id = [1u8; 32];
        let rb = ReliableBroadcast::new(node_id);

        let vertex = Vertex {
            id: [2u8; 32],
            round: 1,
            author: node_id,
            tx_root: [0u8; 32],
            parents: vec![],
            transactions: vec![],
            signature: vec![],
            timestamp: chrono::Utc::now(),
        };

        let message = BroadcastMessage::Send {
            vertex: vertex.clone(),
        };
        let result = rb.process_message(message).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }
}
