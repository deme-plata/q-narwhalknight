use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use tokio::net::UdpSocket;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error, debug_span, Instrument};
use serde::{Serialize, Deserialize};
use serde_bencode::{to_bytes, from_bytes};
use sha1::{Sha1, Digest};
use dashmap::DashMap;
use ed25519_dalek::{Signer, Verifier, PublicKey, Signature, Keypair};

pub type NodeId = [u8; 20];
pub type TransactionId = [u8; 2];

/// Bootstrap strategy following your hybrid approach
#[derive(Clone, Debug)]
pub enum BootstrapStrategy {
    /// Local mDNS discovery for autonomous joins
    MdnsLocal,
    /// Private Q-NarwhalKnight validator nodes
    Private(Vec<SocketAddr>),
    /// Hybrid: try primary first, fall back to secondary
    Hybrid {
        primary: Box<BootstrapStrategy>,
        fallback: Vec<SocketAddr>,
    },
}

/// BEP-5 DHT message structures with proper binary serialization
#[derive(Serialize, Deserialize, Debug)]
struct PingQuery {
    #[serde(rename = "t")]
    tx_id: TransactionId,
    #[serde(rename = "y")]
    y: String, // "q" for query
    #[serde(rename = "q")]
    q: String, // "ping"
    #[serde(rename = "a")]
    a: PingArgs,
}

#[derive(Serialize, Deserialize, Debug)]
struct PingArgs {
    #[serde(rename = "id", with = "binary")]
    id: NodeId,
}

#[derive(Serialize, Deserialize, Debug)]
struct PingResponse {
    #[serde(rename = "t")]
    tx_id: TransactionId,
    #[serde(rename = "y")]
    y: String, // "r" for response
    #[serde(rename = "r")]
    r: PingResponseArgs,
}

#[derive(Serialize, Deserialize, Debug)]
struct PingResponseArgs {
    #[serde(rename = "id", with = "binary")]
    id: NodeId,
}

/// BEP-44 mutable data structure with Ed25519 signatures
#[derive(Serialize, Deserialize, Debug)]
pub struct MutablePut {
    #[serde(rename = "k")]
    pub pubkey: [u8; 32],
    #[serde(rename = "seq")]
    pub seq: u64,
    #[serde(rename = "sig")]
    pub sig: [u8; 64],
    #[serde(rename = "v")]
    pub value: Vec<u8>,
}

/// Node information for routing table
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: NodeId,
    pub addr: SocketAddr,
    pub last_seen: Instant,
}

/// Pending transaction for timeout management
#[derive(Debug)]
pub struct PendingTransaction {
    pub addr: SocketAddr,
    pub timeout: Instant,
    pub query_type: String,
}

/// Enhanced routing table with dashmap for concurrency
pub struct RoutingTable {
    node_id: NodeId,
    buckets: Vec<Arc<RwLock<Vec<NodeInfo>>>>,
    k: usize, // k=20 for quantum network resilience
}

impl RoutingTable {
    pub fn new(node_id: NodeId) -> Self {
        let buckets = (0..160)
            .map(|_| Arc::new(RwLock::new(Vec::new())))
            .collect();

        Self {
            node_id,
            buckets,
            k: 20, // Higher k for Q-NarwhalKnight churn tolerance
        }
    }

    /// Insert node using XOR distance metric
    pub async fn insert_node(&self, node: NodeInfo) -> Result<()> {
        let dist = xor_distance(&self.node_id, &node.id);
        let bucket_idx = leading_zeros(&dist);

        if bucket_idx < self.buckets.len() {
            let mut bucket = self.buckets[bucket_idx].write().await;

            // Remove existing entry if present
            bucket.retain(|n| n.id != node.id);

            // Add to front if under capacity
            if bucket.len() < self.k {
                bucket.insert(0, node);
                info!("Node inserted into bucket {}: {} nodes total", bucket_idx, bucket.len());
            } else {
                // Replace least recently seen node
                if let Some(oldest_idx) = bucket.iter().enumerate()
                    .min_by_key(|(_, n)| n.last_seen)
                    .map(|(idx, _)| idx) {
                    bucket[oldest_idx] = node;
                }
            }
        }

        Ok(())
    }

    pub async fn total_nodes(&self) -> usize {
        let mut total = 0;
        for bucket in &self.buckets {
            total += bucket.read().await.len();
        }
        total
    }

    pub async fn active_buckets(&self) -> usize {
        let mut active = 0;
        for bucket in &self.buckets {
            if !bucket.read().await.is_empty() {
                active += 1;
            }
        }
        active
    }
}

/// Main BEP-5 DHT node with improvements
pub struct Bep5DhtNode {
    node_id: NodeId,
    socket: UdpSocket,
    routing_table: Arc<RoutingTable>,
    pending_transactions: Arc<DashMap<TransactionId, PendingTransaction>>,
    is_running: Arc<RwLock<bool>>,
}

impl Bep5DhtNode {
    /// Create new DHT node with random ID
    pub async fn new(bind_addr: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind(bind_addr).await?;
        let node_id = generate_node_id();
        let routing_table = Arc::new(RoutingTable::new(node_id));
        let pending_transactions = Arc::new(DashMap::new());

        info!("Created BEP-5 DHT node with ID: {}", hex::encode(&node_id));
        info!("Listening on: {}", socket.local_addr()?);

        let node = Self {
            node_id,
            socket,
            routing_table,
            pending_transactions,
            is_running: Arc::new(RwLock::new(false)),
        };

        // Start transaction cleanup task
        node.spawn_cleanup().await;

        Ok(node)
    }

    /// Bootstrap using your hybrid strategy
    pub async fn bootstrap(&mut self, strategy: BootstrapStrategy) -> Result<()> {
        let span = debug_span!("bootstrap", strategy = ?strategy);

        async {
            *self.is_running.write().await = true;

            match strategy {
                BootstrapStrategy::MdnsLocal => {
                    info!("Starting mDNS local discovery");
                    // For MVP: scan local network IPs
                    self.scan_local_network().await?;
                }
                BootstrapStrategy::Private(addrs) => {
                    info!("Bootstrapping to {} private nodes", addrs.len());
                    for addr in addrs {
                        if let Err(e) = self.send_ping(addr).await {
                            warn!("Failed to ping {}: {}", addr, e);
                        }
                    }
                }
                BootstrapStrategy::Hybrid { primary, fallback } => {
                    info!("Hybrid bootstrap: trying primary strategy first");

                    // Try primary strategy with timeout
                    let primary_result = tokio::time::timeout(
                        Duration::from_secs(3),
                        self.bootstrap(*primary)
                    ).await;

                    if primary_result.is_err() || self.routing_table.total_nodes().await == 0 {
                        info!("Primary bootstrap failed, trying fallback nodes");
                        for addr in fallback {
                            if let Err(e) = self.send_ping(addr).await {
                                warn!("Fallback ping failed for {}: {}", addr, e);
                            }
                        }
                    }
                }
            }

            // Wait for responses with 5s timeout (your target)
            tokio::time::sleep(Duration::from_secs(5)).await;

            let total_nodes = self.routing_table.total_nodes().await;
            let active_buckets = self.routing_table.active_buckets().await;

            info!("Bootstrap complete: {} nodes in {} buckets", total_nodes, active_buckets);

            if total_nodes == 0 {
                warn!("Bootstrap failed: no nodes discovered");
            }

            Ok(())
        }.instrument(span).await
    }

    /// Send ping with proper bencode serialization
    pub async fn send_ping(&self, addr: SocketAddr) -> Result<()> {
        let tx_id = rand::random::<TransactionId>();

        let query = PingQuery {
            tx_id,
            y: "q".to_string(),
            q: "ping".to_string(),
            a: PingArgs { id: self.node_id },
        };

        let payload = to_bytes(&query)?;

        debug_span!("ping",
            target = %addr,
            tx_id = %hex::encode(&tx_id),
            payload_hex = %hex::encode(&payload)
        ).in_scope(|| {
            debug!("Sending ping to {}", addr);
        });

        self.socket.send_to(&payload, addr).await?;

        // Store pending transaction
        self.pending_transactions.insert(tx_id, PendingTransaction {
            addr,
            timeout: Instant::now() + Duration::from_secs(10),
            query_type: "ping".to_string(),
        });

        Ok(())
    }

    /// Process received messages
    pub async fn handle_message(&self, data: &[u8], from: SocketAddr) -> Result<()> {
        // Try to parse as ping response first
        if let Ok(response) = from_bytes::<PingResponse>(data) {
            self.handle_ping_response(response, from).await?;
        } else if let Ok(query) = from_bytes::<PingQuery>(data) {
            self.handle_ping_query(query, from).await?;
        } else {
            debug!("Unknown message format from {}: {}", from, hex::encode(data));
        }

        Ok(())
    }

    async fn handle_ping_response(&self, response: PingResponse, from: SocketAddr) -> Result<()> {
        debug!("Received ping response from {} with tx_id: {}",
               from, hex::encode(&response.tx_id));

        // Remove from pending transactions
        if let Some((_, pending)) = self.pending_transactions.remove(&response.tx_id) {
            debug!("Matched transaction from {}", pending.addr);

            // Add node to routing table
            let node_info = NodeInfo {
                id: response.r.id,
                addr: from,
                last_seen: Instant::now(),
            };

            self.routing_table.insert_node(node_info).await?;
        }

        Ok(())
    }

    async fn handle_ping_query(&self, query: PingQuery, from: SocketAddr) -> Result<()> {
        debug!("Received ping query from {}", from);

        let response = PingResponse {
            tx_id: query.tx_id,
            y: "r".to_string(),
            r: PingResponseArgs { id: self.node_id },
        };

        let payload = to_bytes(&response)?;
        self.socket.send_to(&payload, from).await?;

        // Add querying node to routing table
        let node_info = NodeInfo {
            id: query.a.id,
            addr: from,
            last_seen: Instant::now(),
        };

        self.routing_table.insert_node(node_info).await?;

        Ok(())
    }

    /// BEP-44 mutable data verification
    pub fn verify_mutable(&self, put: &MutablePut) -> bool {
        let pubkey = match PublicKey::from_bytes(&put.pubkey) {
            Ok(pk) => pk,
            Err(_) => return false,
        };

        // Create message according to BEP-44 spec: seq+value concatenation
        let seq_bytes = put.seq.to_be_bytes();
        let mut msg = Vec::new();
        msg.extend_from_slice(b"3:seqi");
        msg.extend_from_slice(&seq_bytes.to_string().as_bytes());
        msg.extend_from_slice(b"e1:v");
        msg.extend_from_slice(&put.value);

        let signature = match Signature::from_bytes(&put.sig) {
            Ok(sig) => sig,
            Err(_) => return false,
        };

        pubkey.verify(&msg, &signature).is_ok()
    }

    /// Sign and create BEP-44 mutable put
    pub fn create_mutable_put(&self, keypair: &Keypair, value: Vec<u8>, seq: u64) -> MutablePut {
        // Create message for signing
        let seq_bytes = seq.to_be_bytes();
        let mut msg = Vec::new();
        msg.extend_from_slice(b"3:seqi");
        msg.extend_from_slice(&seq_bytes.to_string().as_bytes());
        msg.extend_from_slice(b"e1:v");
        msg.extend_from_slice(&value);

        let signature = keypair.sign(&msg);

        MutablePut {
            pubkey: keypair.public.to_bytes(),
            seq,
            sig: signature.to_bytes(),
            value,
        }
    }

    /// Local network scanning for mDNS-like discovery
    async fn scan_local_network(&self) -> Result<()> {
        let local_addr = self.socket.local_addr()?;
        let base_port = 6881;

        // Scan common DHT ports on local network
        for port_offset in 0..10 {
            let port = base_port + port_offset;
            if port != local_addr.port() {
                let test_addr = SocketAddr::new(local_addr.ip(), port);

                tokio::spawn({
                    let node = self.clone();
                    async move {
                        if let Err(e) = node.send_ping(test_addr).await {
                            debug!("Local scan ping failed for {}: {}", test_addr, e);
                        }
                    }
                });
            }
        }

        Ok(())
    }

    /// Spawn transaction cleanup task
    async fn spawn_cleanup(&self) {
        let pending = Arc::clone(&self.pending_transactions);
        let is_running = Arc::clone(&self.is_running);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                if !*is_running.read().await {
                    break;
                }

                let now = Instant::now();
                pending.retain(|_, tx| tx.timeout > now);

                debug!("Cleanup: {} pending transactions", pending.len());
            }
        });
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }
}

// Clone implementation for testing
impl Clone for Bep5DhtNode {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id,
            socket: self.socket.try_clone().expect("Failed to clone socket"),
            routing_table: Arc::clone(&self.routing_table),
            pending_transactions: Arc::clone(&self.pending_transactions),
            is_running: Arc::clone(&self.is_running),
        }
    }
}

/// Binary serialization module for exact 20-byte node IDs
mod binary {
    use serde::{Serializer, Deserializer, Deserialize};

    pub fn serialize<S>(data: &[u8; 20], ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ser.serialize_bytes(&data[..])
    }

    pub fn deserialize<'de, D>(des: D) -> Result<[u8; 20], D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = Vec::deserialize(des)?;
        if bytes.len() != 20 {
            return Err(serde::de::Error::invalid_length(
                bytes.len(),
                &"exactly 20 bytes",
            ));
        }

        let mut array = [0u8; 20];
        array.copy_from_slice(&bytes);
        Ok(array)
    }
}

/// XOR distance calculation for DHT routing
pub fn xor_distance(a: &[u8; 20], b: &[u8; 20]) -> [u8; 20] {
    let mut dist = [0u8; 20];
    for i in 0..20 {
        dist[i] = a[i] ^ b[i];
    }
    dist
}

/// Count leading zeros in XOR distance for bucket selection
pub fn leading_zeros(data: &[u8; 20]) -> usize {
    for (byte_idx, byte) in data.iter().enumerate() {
        if *byte != 0 {
            return byte_idx * 8 + (byte.leading_zeros() as usize);
        }
    }
    160 // All zeros
}

/// Generate random 160-bit node ID
pub fn generate_node_id() -> NodeId {
    let mut hasher = Sha1::new();
    hasher.update(&rand::random::<[u8; 32]>());
    let result = hasher.finalize();

    let mut node_id = [0u8; 20];
    node_id.copy_from_slice(&result[..20]);
    node_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_node_creation() -> Result<()> {
        let addr = "127.0.0.1:0".parse()?;
        let node = Bep5DhtNode::new(addr).await?;
        assert!(node.routing_table.total_nodes().await == 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_ping_serialization() -> Result<()> {
        let node_id = generate_node_id();
        let tx_id = [0x12, 0x34];

        let query = PingQuery {
            tx_id,
            y: "q".to_string(),
            q: "ping".to_string(),
            a: PingArgs { id: node_id },
        };

        let encoded = to_bytes(&query)?;
        let decoded: PingQuery = from_bytes(&encoded)?;

        assert_eq!(decoded.tx_id, tx_id);
        assert_eq!(decoded.q, "ping");
        assert_eq!(decoded.a.id, node_id);

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_node_bootstrap() -> Result<()> {
        let mut nodes = Vec::new();

        // Create 3 test nodes
        for i in 0..3 {
            let addr = format!("127.0.0.1:{}", 6880 + i).parse()?;
            let node = Bep5DhtNode::new(addr).await?;
            nodes.push(node);
        }

        // Bootstrap node 1 to node 0
        let bootstrap_addr = nodes[0].local_addr()?;
        let strategy = BootstrapStrategy::Private(vec![bootstrap_addr]);
        nodes[1].bootstrap(strategy).await?;

        // Bootstrap node 2 to both nodes 0 and 1
        let strategy = BootstrapStrategy::Private(vec![
            nodes[0].local_addr()?,
            nodes[1].local_addr()?,
        ]);
        nodes[2].bootstrap(strategy).await?;

        // Verify nodes discovered each other
        tokio::time::sleep(Duration::from_secs(1)).await;

        // At least node 0 should have discovered node 1
        let node0_peers = nodes[0].routing_table.total_nodes().await;
        assert!(node0_peers > 0, "Node 0 should have discovered peers");

        info!("Test successful: {} nodes in network", node0_peers);
        Ok(())
    }

    #[test]
    fn test_xor_distance() {
        let a = [0x00; 20];
        let mut b = [0x00; 20];
        b[0] = 0xFF;

        let dist = xor_distance(&a, &b);
        assert_eq!(dist[0], 0xFF);
        assert_eq!(leading_zeros(&dist), 0);
    }

    #[test]
    fn test_bep44_signature() {
        use rand::rngs::OsRng;

        let keypair = Keypair::generate(&mut OsRng);
        let value = b"test data".to_vec();
        let seq = 1;

        let addr = "127.0.0.1:0".parse().unwrap();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let node = rt.block_on(Bep5DhtNode::new(addr)).unwrap();

        let put = node.create_mutable_put(&keypair, value.clone(), seq);
        assert!(node.verify_mutable(&put));

        // Verify tampering detection
        let mut tampered = put.clone();
        tampered.value = b"tampered".to_vec();
        assert!(!node.verify_mutable(&tampered));
    }
}