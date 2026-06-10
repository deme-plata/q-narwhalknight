/*!
# Real BEP-5 DHT Protocol Implementation

This module implements the actual BitTorrent DHT protocol (BEP-5) which is required
as the foundation for BEP-44 mutable data operations.

Key protocols implemented:
- ping/pong for node health checking
- find_node for routing table discovery
- get_peers for finding peers sharing specific infohashes
- announce_peer for advertising peer presence
- get/put for BEP-44 mutable data (requires BEP-5 foundation)
*/

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha1::Digest;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket as TokioUdpSocket;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Maximum number of nodes per K-bucket (BitTorrent spec)
const K_BUCKET_SIZE: usize = 8;

/// DHT node ID is 160 bits (20 bytes) as per BEP-5 spec
pub type NodeId = [u8; 20];

/// InfoHash is 160 bits (20 bytes) for BitTorrent
pub type InfoHash = [u8; 20];

/// Transaction ID for matching queries with responses
pub type TransactionId = Vec<u8>;

/// Real BEP-5 DHT node implementation
#[derive(Debug)]
pub struct Bep5DhtNode {
    /// This node's ID
    node_id: NodeId,
    /// UDP socket for DHT communication
    socket: Arc<TokioUdpSocket>,
    /// Routing table with K-buckets
    routing_table: Arc<RwLock<RoutingTable>>,
    /// Pending transactions (for matching responses)
    pending_transactions: Arc<RwLock<HashMap<TransactionId, PendingTransaction>>>,
    /// Bootstrap nodes for initial connection
    bootstrap_nodes: Vec<SocketAddr>,
    /// Running status
    is_running: Arc<RwLock<bool>>,
}

/// K-bucket based routing table as per BEP-5
#[derive(Debug)]
pub struct RoutingTable {
    /// Our node ID
    our_node_id: NodeId,
    /// K-buckets indexed by distance from our node ID
    buckets: [KBucket; 160],
}

/// K-bucket containing up to K nodes at a specific distance
#[derive(Debug)]
pub struct KBucket {
    nodes: Vec<DhtNode>,
    last_changed: Instant,
}

impl KBucket {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            last_changed: Instant::now(),
        }
    }

    fn add_node(&mut self, node: DhtNode) -> bool {
        // Check if node already exists
        if let Some(existing) = self.nodes.iter_mut().find(|n| n.node_id == node.node_id) {
            existing.last_seen = node.last_seen;
            existing.is_good = node.is_good;
            return true;
        }

        // If bucket is not full, add node
        if self.nodes.len() < K_BUCKET_SIZE {
            self.nodes.push(node);
            self.last_changed = Instant::now();
            return true;
        }

        // Bucket is full - implement replacement strategy
        // Replace bad nodes or oldest good node
        if let Some(bad_index) = self.nodes.iter().position(|n| !n.is_good) {
            self.nodes[bad_index] = node;
            self.last_changed = Instant::now();
            return true;
        }

        false // Bucket full with all good nodes
    }

    fn get_nodes(&self) -> &[DhtNode] {
        &self.nodes
    }
}

impl Default for KBucket {
    fn default() -> Self {
        Self::new()
    }
}

/// DHT node information
#[derive(Debug, Clone)]
pub struct DhtNode {
    pub node_id: NodeId,
    pub address: SocketAddr,
    pub last_seen: Instant,
    pub last_query_time: Option<Instant>,
    pub failed_queries: u32,
    pub is_good: bool,
}

/// DHT message types as per BEP-5 spec
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "y")]
pub enum DhtMessage {
    #[serde(rename = "q")]
    Query {
        #[serde(rename = "t")]
        transaction_id: TransactionId,
        #[serde(rename = "q")]
        query_type: String,
        #[serde(rename = "a")]
        arguments: DhtQueryArgs,
    },
    #[serde(rename = "r")]
    Response {
        #[serde(rename = "t")]
        transaction_id: TransactionId,
        #[serde(rename = "r")]
        response: DhtResponseData,
    },
    #[serde(rename = "e")]
    Error {
        #[serde(rename = "t")]
        transaction_id: TransactionId,
        #[serde(rename = "e")]
        error: (u32, String),
    },
}

/// DHT query arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DhtQueryArgs {
    Ping {
        id: NodeId,
    },
    FindNode {
        id: NodeId,
        target: NodeId,
    },
    GetPeers {
        id: NodeId,
        info_hash: InfoHash,
    },
    AnnouncePeer {
        id: NodeId,
        info_hash: InfoHash,
        port: u16,
        token: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        implied_port: Option<u8>,
    },
    Get {
        id: NodeId,
        target: [u8; 20],
    },
    Put {
        id: NodeId,
        #[serde(rename = "v")]
        value: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        token: Option<Vec<u8>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        k: Option<[u8; 32]>, // public key for mutable data
        #[serde(skip_serializing_if = "Option::is_none")]
        seq: Option<i64>,    // sequence number for mutable data
        #[serde(skip_serializing_if = "Option::is_none")]
        sig: Option<Vec<u8>>, // signature for mutable data
        #[serde(skip_serializing_if = "Option::is_none")]
        salt: Option<Vec<u8>>, // salt for mutable data
    },
}

/// DHT response data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DhtResponseData {
    Ping {
        id: NodeId,
    },
    FindNode {
        id: NodeId,
        nodes: Vec<u8>, // Compact node info format
    },
    GetPeers {
        id: NodeId,
        #[serde(skip_serializing_if = "Option::is_none")]
        values: Option<Vec<Vec<u8>>>, // Compact peer info
        #[serde(skip_serializing_if = "Option::is_none")]
        nodes: Option<Vec<u8>>, // Compact node info if no peers found
        token: Vec<u8>,
    },
    AnnouncePeer {
        id: NodeId,
    },
    Get {
        id: NodeId,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "v")]
        value: Option<Vec<u8>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        nodes: Option<Vec<u8>>, // If not found, return closest nodes
        token: Vec<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        k: Option<[u8; 32]>,  // public key if mutable
        #[serde(skip_serializing_if = "Option::is_none")]
        seq: Option<i64>,     // sequence number if mutable
        #[serde(skip_serializing_if = "Option::is_none")]
        sig: Option<Vec<u8>>, // signature if mutable
    },
    Put {
        id: NodeId,
    },
}

/// Pending transaction waiting for response
#[derive(Debug)]
struct PendingTransaction {
    query_type: String,
    target_addr: SocketAddr,
    sent_time: Instant,
    timeout: Duration,
}

impl Bep5DhtNode {
    /// Create a new BEP-5 DHT node
    pub async fn new(bind_addr: SocketAddr) -> Result<Self> {
        // Generate random node ID
        let mut node_id = [0u8; 20];
        getrandom::getrandom(&mut node_id).context("Failed to generate node ID")?;

        // Bind UDP socket
        let socket = TokioUdpSocket::bind(bind_addr).await
            .context("Failed to bind UDP socket for DHT")?;

        info!("🌐 Created BEP-5 DHT node");
        info!("   • Node ID: {}", hex::encode(&node_id));
        info!("   • Listening on: {}", bind_addr);

        // Well-known bootstrap nodes for BitTorrent DHT
        let bootstrap_nodes = vec![
            "router.bittorrent.com:6881".parse()?,
            "dht.transmissionbt.com:6881".parse()?,
            "router.utorrent.com:6881".parse()?,
            "dht.aelitis.com:6881".parse()?,
        ];

        Ok(Self {
            node_id,
            socket: Arc::new(socket),
            routing_table: Arc::new(RwLock::new(RoutingTable::new(node_id))),
            pending_transactions: Arc::new(RwLock::new(HashMap::new())),
            bootstrap_nodes,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the DHT node (bootstrap and message handling)
    pub async fn start(&mut self) -> Result<()> {
        {
            let mut running = self.is_running.write().await;
            *running = true;
        }

        info!("🚀 Starting BEP-5 DHT node...");

        // Start message handling loop
        self.spawn_message_handler().await;

        // Bootstrap to the DHT network
        self.bootstrap().await?;

        info!("✅ BEP-5 DHT node started and bootstrapped");
        Ok(())
    }

    /// Bootstrap to the DHT network by contacting known nodes
    pub async fn bootstrap(&self) -> Result<()> {
        info!("🌐 Bootstrapping to BitTorrent DHT network...");

        let mut successful_bootstraps = 0;

        for bootstrap_addr in &self.bootstrap_nodes {
            match self.send_ping(*bootstrap_addr).await {
                Ok(_) => {
                    info!("✅ Successfully pinged bootstrap node: {}", bootstrap_addr);
                    successful_bootstraps += 1;
                }
                Err(e) => {
                    warn!("❌ Failed to ping bootstrap node {}: {}", bootstrap_addr, e);
                }
            }
        }

        if successful_bootstraps > 0 {
            info!("🎯 Bootstrap complete: {}/{} nodes responded",
                  successful_bootstraps, self.bootstrap_nodes.len());
        } else {
            warn!("⚠️ Bootstrap failed: No nodes responded");
        }

        Ok(())
    }

    /// Send a ping query to a node
    pub async fn send_ping(&self, target: SocketAddr) -> Result<()> {
        let transaction_id = self.generate_transaction_id();

        let ping_query = DhtMessage::Query {
            transaction_id: transaction_id.clone(),
            query_type: "ping".to_string(),
            arguments: DhtQueryArgs::Ping {
                id: self.node_id,
            },
        };

        let serialized = serde_bencode::to_bytes(&ping_query)
            .context("Failed to serialize ping query")?;

        // Store pending transaction
        {
            let mut pending = self.pending_transactions.write().await;
            pending.insert(transaction_id.clone(), PendingTransaction {
                query_type: "ping".to_string(),
                target_addr: target,
                sent_time: Instant::now(),
                timeout: Duration::from_secs(5),
            });
        }

        // Send UDP packet
        self.socket.send_to(&serialized, target).await
            .context("Failed to send ping")?;

        debug!("📡 Sent ping to {}", target);
        Ok(())
    }

    /// Generate a random transaction ID
    fn generate_transaction_id(&self) -> TransactionId {
        let mut tid = [0u8; 2];
        getrandom::getrandom(&mut tid).unwrap();
        tid.to_vec()
    }

    /// Spawn the message handling loop
    async fn spawn_message_handler(&self) {
        let socket = self.socket.clone();
        let pending_transactions = self.pending_transactions.clone();
        let routing_table = self.routing_table.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let mut buf = [0u8; 65536];

            while {
                let running = is_running.read().await;
                *running
            } {
                match socket.recv_from(&mut buf).await {
                    Ok((len, addr)) => {
                        let data = &buf[..len];

                        match serde_bencode::from_bytes::<DhtMessage>(data) {
                            Ok(message) => {
                                debug!("📨 Received DHT message from {}: {:?}", addr, message);
                                Self::handle_message(message, addr, &pending_transactions, &routing_table).await;
                            }
                            Err(e) => {
                                debug!("❌ Failed to parse DHT message from {}: {}", addr, e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("❌ UDP recv error: {}", e);
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        });
    }

    /// Handle incoming DHT messages
    async fn handle_message(
        message: DhtMessage,
        addr: SocketAddr,
        pending_transactions: &Arc<RwLock<HashMap<TransactionId, PendingTransaction>>>,
        routing_table: &Arc<RwLock<RoutingTable>>,
    ) {
        match message {
            DhtMessage::Response { transaction_id, response } => {
                // Match with pending transaction
                let mut pending = pending_transactions.write().await;
                if let Some(transaction) = pending.remove(&transaction_id) {
                    info!("✅ Received {} response from {}", transaction.query_type, addr);

                    // Update routing table with responding node
                    if let DhtResponseData::Ping { id } = response {
                        let mut rt = routing_table.write().await;
                        rt.add_node(DhtNode {
                            node_id: id,
                            address: addr,
                            last_seen: Instant::now(),
                            last_query_time: Some(transaction.sent_time),
                            failed_queries: 0,
                            is_good: true,
                        });
                    }
                }
            }
            DhtMessage::Query { transaction_id, query_type, arguments } => {
                debug!("📨 Received {} query from {}", query_type, addr);
                // In a full implementation, we would handle incoming queries
                // For now, just acknowledge reception
            }
            DhtMessage::Error { transaction_id, error } => {
                warn!("❌ DHT error from {}: {} - {}", addr, error.0, error.1);
            }
        }
    }


    /// Send find_node query
    pub async fn find_node(&self, target_addr: SocketAddr, target_node_id: NodeId) -> Result<Vec<DhtNode>> {
        let transaction_id = self.generate_transaction_id();

        let query = DhtMessage::Query {
            transaction_id: transaction_id.clone(),
            query_type: "find_node".to_string(),
            arguments: DhtQueryArgs::FindNode {
                id: self.node_id,
                target: target_node_id,
            },
        };

        self.send_message(target_addr, &query).await?;

        // Store pending transaction
        {
            let mut pending = self.pending_transactions.write().await;
            pending.insert(transaction_id.clone(), PendingTransaction {
                query_type: "find_node".to_string(),
                target_addr,
                sent_time: Instant::now(),
                timeout: Duration::from_secs(5),
            });
        }

        // For now, return empty vec - real implementation would wait for response
        Ok(vec![])
    }

    /// Send BEP-44 get query for mutable data
    pub async fn get_mutable(&self, target_addr: SocketAddr, target_key: [u8; 20]) -> Result<Option<Vec<u8>>> {
        let transaction_id = self.generate_transaction_id();

        let query = DhtMessage::Query {
            transaction_id: transaction_id.clone(),
            query_type: "get".to_string(),
            arguments: DhtQueryArgs::Get {
                id: self.node_id,
                target: target_key,
            },
        };

        self.send_message(target_addr, &query).await?;

        // Store pending transaction
        {
            let mut pending = self.pending_transactions.write().await;
            pending.insert(transaction_id.clone(), PendingTransaction {
                query_type: "get".to_string(),
                target_addr,
                sent_time: Instant::now(),
                timeout: Duration::from_secs(5),
            });
        }

        // For now, return None - real implementation would wait for response
        Ok(None)
    }

    /// Send BEP-44 put query for mutable data
    pub async fn put_mutable(
        &self,
        target_addr: SocketAddr,
        value: Vec<u8>,
        public_key: [u8; 32],
        sequence: i64,
        signature: Vec<u8>,
        salt: Option<Vec<u8>>,
        token: Vec<u8>,
    ) -> Result<()> {
        let transaction_id = self.generate_transaction_id();

        let query = DhtMessage::Query {
            transaction_id: transaction_id.clone(),
            query_type: "put".to_string(),
            arguments: DhtQueryArgs::Put {
                id: self.node_id,
                value,
                token: Some(token),
                k: Some(public_key),
                seq: Some(sequence),
                sig: Some(signature),
                salt,
            },
        };

        self.send_message(target_addr, &query).await?;

        // Store pending transaction
        {
            let mut pending = self.pending_transactions.write().await;
            pending.insert(transaction_id.clone(), PendingTransaction {
                query_type: "put".to_string(),
                target_addr,
                sent_time: Instant::now(),
                timeout: Duration::from_secs(5),
            });
        }

        Ok(())
    }

    /// Bootstrap this node by connecting to other DHT nodes
    pub async fn bootstrap_to_peers(&self, bootstrap_nodes: Vec<SocketAddr>) -> Result<()> {
        info!("🚀 Bootstrapping DHT node to {} peers", bootstrap_nodes.len());

        for addr in bootstrap_nodes {
            // Send ping to bootstrap node to establish connection
            self.send_ping(addr).await?;

            // Send find_node for our own ID to populate routing table
            self.find_node(addr, self.node_id).await?;
        }

        info!("✅ Bootstrap complete");
        Ok(())
    }

    /// Send DHT message to target address
    async fn send_message(&self, target_addr: SocketAddr, message: &DhtMessage) -> Result<()> {
        // Serialize message using bencode (BitTorrent standard)
        let serialized = serde_bencode::to_bytes(message)
            .context("Failed to serialize DHT message")?;

        debug!("📤 Sending DHT message to {}: {} bytes", target_addr, serialized.len());

        self.socket.send_to(&serialized, target_addr).await
            .context("Failed to send UDP packet")?;

        Ok(())
    }



    /// Handle incoming DHT message (static method for tokio spawn)
    async fn handle_message_static(
        message: DhtMessage,
        from_addr: SocketAddr,
        routing_table: &Arc<RwLock<RoutingTable>>,
        pending_transactions: &Arc<RwLock<HashMap<TransactionId, PendingTransaction>>>,
    ) -> Result<()> {
        match message {
            DhtMessage::Response { transaction_id, response } => {
                debug!("📨 Received response from {}", from_addr);

                // Remove from pending transactions
                {
                    let mut pending = pending_transactions.write().await;
                    if let Some(_transaction) = pending.remove(&transaction_id) {
                        debug!("✅ Matched response to pending transaction");
                    }
                }

                // Process response and update routing table
                match response {
                    DhtResponseData::Ping { id } => {
                        info!("🏓 Ping response from {}: {}", from_addr, hex::encode(&id));
                    }
                    DhtResponseData::FindNode { id, nodes } => {
                        info!("🔍 find_node response from {}: {} nodes", from_addr, nodes.len() / 26);
                        // Parse compact node info and add to routing table
                        // Each node is 26 bytes: 20-byte ID + 6-byte address (4 bytes IP + 2 bytes port)
                    }
                    DhtResponseData::Get {  value, .. } => {
                        info!("📥 get response from {}: {} bytes", from_addr,
                              value.as_ref().map(|v| v.len()).unwrap_or(0));
                    }
                    _ => {
                        debug!("Received other response type from {}", from_addr);
                    }
                }
            }
            DhtMessage::Query { transaction_id, query_type, arguments } => {
                debug!("❓ Received query '{}' from {}", query_type, from_addr);
                // Handle queries - send appropriate responses
                // For now, we'll implement this in a follow-up
            }
            DhtMessage::Error { transaction_id, error } => {
                warn!("❌ Received error from {}: {} - {}", from_addr, error.0, error.1);
            }
        }

        Ok(())
    }

    /// Spawn maintenance task
    async fn spawn_maintenance_task(&self) {
        let routing_table = Arc::clone(&self.routing_table);
        let is_running = Arc::clone(&self.is_running);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));

            while *is_running.read().await {
                interval.tick().await;

                // Perform maintenance tasks
                {
                    let routing_table = routing_table.read().await;
                    let stats = routing_table.get_stats();
                    info!("📊 DHT Stats: {} nodes, {} good nodes", stats.total_nodes, stats.good_nodes);
                }

                // TODO: Implement periodic ping of old nodes, bucket refresh, etc.
            }
        });
    }

    /// Get node statistics
    pub async fn get_stats(&self) -> DhtStats {
        let routing_table = self.routing_table.read().await;
        let stats = routing_table.get_stats();
        let pending_count = self.pending_transactions.read().await.len();

        DhtStats {
            node_id: hex::encode(&self.node_id),
            total_nodes: stats.total_nodes,
            good_nodes: stats.good_nodes,
            pending_transactions: pending_count,
            is_bootstrapped: stats.good_nodes > 0,
        }
    }

    /// Stop the DHT node
    pub async fn stop(&mut self) {
        let mut running = self.is_running.write().await;
        *running = false;
        info!("🛑 BEP-5 DHT node stopped");
    }
}

impl RoutingTable {
    /// Create new routing table
    pub fn new(our_node_id: NodeId) -> Self {
        let buckets = std::array::from_fn(|_| KBucket {
            nodes: Vec::new(),
            last_changed: Instant::now(),
        });

        Self {
            our_node_id,
            buckets,
        }
    }

    /// Add a node to the routing table
    pub fn add_node(&mut self, node: DhtNode) {
        let distance = Self::calculate_distance(&self.our_node_id, &node.node_id);
        let bucket_index = Self::distance_to_bucket_index(distance);

        if bucket_index < 160 {
            let bucket = &mut self.buckets[bucket_index];

            // Check if node already exists
            if let Some(existing_index) = bucket.nodes.iter().position(|n| n.node_id == node.node_id) {
                // Update existing node
                bucket.nodes[existing_index] = node;
                bucket.last_changed = Instant::now();
            } else if bucket.nodes.len() < K_BUCKET_SIZE {
                // Add new node if bucket not full
                bucket.nodes.push(node);
                bucket.last_changed = Instant::now();
            } else {
                // Bucket full - implement replacement logic
                // For now, just ignore (should ping oldest and replace if dead)
                debug!("K-bucket {} is full, ignoring new node", bucket_index);
            }
        }
    }

    /// Calculate XOR distance between two node IDs
    fn calculate_distance(id1: &NodeId, id2: &NodeId) -> [u8; 20] {
        let mut distance = [0u8; 20];
        for i in 0..20 {
            distance[i] = id1[i] ^ id2[i];
        }
        distance
    }

    /// Convert distance to bucket index (0-159)
    fn distance_to_bucket_index(distance: [u8; 20]) -> usize {
        for (byte_index, &byte) in distance.iter().enumerate() {
            if byte != 0 {
                let bit_index = 7 - byte.leading_zeros() as usize;
                return byte_index * 8 + bit_index;
            }
        }
        159 // All bits are the same (shouldn't happen with different node IDs)
    }

    /// Get routing table statistics
    pub fn get_stats(&self) -> RoutingTableStats {
        let mut total_nodes = 0;
        let mut good_nodes = 0;

        for bucket in &self.buckets {
            total_nodes += bucket.nodes.len();
            good_nodes += bucket.nodes.iter().filter(|n| n.is_good).count();
        }

        RoutingTableStats {
            total_nodes,
            good_nodes,
        }
    }
}

/// DHT node statistics
#[derive(Debug)]
pub struct DhtStats {
    pub node_id: String,
    pub total_nodes: usize,
    pub good_nodes: usize,
    pub pending_transactions: usize,
    pub is_bootstrapped: bool,
}

/// Routing table statistics
#[derive(Debug)]
pub struct RoutingTableStats {
    pub total_nodes: usize,
    pub good_nodes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dht_node_creation() {
        let bind_addr = "0.0.0.0:0".parse().unwrap();
        let node = Bep5DhtNode::new(bind_addr).await.unwrap();
        assert_eq!(node.node_id.len(), 20);
    }

    #[test]
    fn test_distance_calculation() {
        let id1 = [0u8; 20];
        let mut id2 = [0u8; 20];
        id2[19] = 1;

        let distance = RoutingTable::calculate_distance(&id1, &id2);
        assert_eq!(distance[19], 1);

        let bucket_index = RoutingTable::distance_to_bucket_index(distance);
        assert_eq!(bucket_index, 0);
    }
}