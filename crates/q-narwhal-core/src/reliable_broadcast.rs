use q_types::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

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
        info!("Broadcasting vertex {} from round {}", 
              hex::encode(vertex.id), vertex.round);

        // Send the vertex to all nodes
        let message = BroadcastMessage::Send { vertex };
        self.network_tx.send(message).map_err(|e| {
            anyhow::anyhow!("Failed to broadcast vertex: {}", e)
        })?;

        Ok(())
    }

    /// Process incoming broadcast message
    pub async fn process_message(&self, message: BroadcastMessage) -> Result<Option<Vertex>> {
        match message {
            BroadcastMessage::Send { vertex } => {
                self.handle_send(vertex).await
            }
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
        
        debug!("Received SEND for vertex {}", hex::encode(vertex_id));

        // Validate vertex (basic checks)
        if !self.validate_vertex(&vertex).await? {
            warn!("Invalid vertex received: {}", hex::encode(vertex_id));
            return Ok(None);
        }

        // If we haven't echoed this vertex yet, echo it
        {
            let mut echoed = self.echoed.write().await;
            if !echoed.contains(&vertex_id) {
                echoed.insert(vertex_id);
                
                // Send ECHO to all nodes
                let echo_msg = BroadcastMessage::Echo { 
                    vertex_id, 
                    sender: self.node_id 
                };
                self.network_tx.send(echo_msg)?;
                
                debug!("Sent ECHO for vertex {}", hex::encode(vertex_id));
            }
        }

        Ok(Some(vertex))
    }

    /// Handle ECHO message (step 2)
    async fn handle_echo(&self, vertex_id: VertexId, sender: NodeId) -> Result<Option<Vertex>> {
        debug!("Received ECHO for vertex {} from {:?}", 
               hex::encode(vertex_id), sender);

        // Add to echo votes
        let echo_count = {
            let mut echo_votes = self.echo_votes.write().await;
            let votes = echo_votes.entry(vertex_id).or_insert_with(HashSet::new);
            votes.insert(sender);
            votes.len()
        };

        // If we have 2f+1 echo votes and haven't sent ready, send ready
        if echo_count >= self.threshold_2f_plus_1 {
            let mut ready_sent = self.ready_sent.write().await;
            if !ready_sent.contains(&vertex_id) {
                ready_sent.insert(vertex_id);
                
                let ready_msg = BroadcastMessage::Ready { 
                    vertex_id, 
                    sender: self.node_id 
                };
                self.network_tx.send(ready_msg)?;
                
                debug!("Sent READY for vertex {} (echo threshold reached)", 
                       hex::encode(vertex_id));
            }
        }

        Ok(None)
    }

    /// Handle READY message (step 3)
    async fn handle_ready(&self, vertex_id: VertexId, sender: NodeId) -> Result<Option<Vertex>> {
        debug!("Received READY for vertex {} from {:?}", 
               hex::encode(vertex_id), sender);

        // Add to ready votes
        let ready_count = {
            let mut ready_votes = self.ready_votes.write().await;
            let votes = ready_votes.entry(vertex_id).or_insert_with(HashSet::new);
            votes.insert(sender);
            votes.len()
        };

        // If we have f+1 ready votes and haven't sent ready, send ready
        if ready_count >= self.threshold_f_plus_1 {
            let mut ready_sent = self.ready_sent.write().await;
            if !ready_sent.contains(&vertex_id) {
                ready_sent.insert(vertex_id);
                
                let ready_msg = BroadcastMessage::Ready { 
                    vertex_id, 
                    sender: self.node_id 
                };
                self.network_tx.send(ready_msg)?;
                
                debug!("Sent READY for vertex {} (ready threshold reached)", 
                       hex::encode(vertex_id));
            }
        }

        // If we have 2f+1 ready votes and haven't delivered, deliver
        if ready_count >= self.threshold_2f_plus_1 {
            let mut delivered = self.delivered.write().await;
            if !delivered.contains(&vertex_id) {
                delivered.insert(vertex_id);
                
                info!("DELIVERED vertex {} (ready delivery threshold reached)", 
                      hex::encode(vertex_id));
                
                // TODO: Return the actual vertex from storage
                // For now, create a placeholder
                return Ok(Some(self.create_placeholder_vertex(vertex_id)));
            }
        }

        Ok(None)
    }

    /// Basic vertex validation
    async fn validate_vertex(&self, vertex: &Vertex) -> Result<bool> {
        // TODO: Implement proper validation
        // - Check signature
        // - Check transaction validity
        // - Check causal dependencies
        
        // For Phase 0, just basic structure checks
        if vertex.transactions.is_empty() && vertex.tx_root != [0u8; 32] {
            return Ok(false);
        }

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

        let message = BroadcastMessage::Send { vertex: vertex.clone() };
        let result = rb.process_message(message).await;
        
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }
}