use anyhow::Result;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

// Import using relative paths
use crate::consensus::pbft::Block;
use crate::vm::VmError;

// Define types directly in this file to avoid circular dependencies
pub type NodeId = String;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: [u8; 32],
    pub data: Vec<u8>,
    pub sender: [u8; 32],
    pub nonce: u64,
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
    pub timestamp: u64,
}

// Define a type alias for clarity
pub type NarwhalTransaction = Transaction;

pub struct Narwhal {
    _node_id: NodeId,
    _peers: Vec<NodeId>,
    _vertices: DashMap<u64, Vec<u8>>,
    _latest_round: RwLock<u64>,
    pub tx_pool: Arc<Mutex<Vec<NarwhalTransaction>>>,
    _tx_network: mpsc::Sender<(NodeId, Vec<u8>)>,
}

impl Narwhal {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> (Self, mpsc::Receiver<(NodeId, Vec<u8>)>) {
        let (tx, rx) = mpsc::channel(1000);
        (
            Self {
                _node_id: node_id,
                _peers: peers,
                _vertices: DashMap::new(),
                _latest_round: RwLock::new(0),
                tx_pool: Arc::new(Mutex::new(Vec::new())),
                _tx_network: tx,
            },
            rx,
        )
    }
}

pub struct Bullshark {
    _node_id: NodeId,
    _peers: Vec<NodeId>,
    latest_round: RwLock<u64>,
    finalized_blocks: DashMap<u64, Block>,
}

impl Bullshark {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, _narwhal: Arc<Narwhal>) -> Self {
        Self {
            _node_id: node_id,
            _peers: peers,
            latest_round: RwLock::new(0),
            finalized_blocks: DashMap::new(),
        }
    }

    pub async fn get_latest_finalized(&self) -> u64 {
        *self.latest_round.read().await
    }

    pub async fn get_finalized_block(&self, seq_num: u64) -> Option<Block> {
        self.finalized_blocks.get(&seq_num).map(|b| b.clone())
    }
}

pub struct NarwhalBullshark {
    _node_id: NodeId,
    _peers: Vec<NodeId>,
    narwhal: Arc<Narwhal>,
    bullshark: Arc<Bullshark>,
    _finalized_blocks: DashMap<u64, Block>,
    _latest_height: RwLock<u64>,
    _tx_network: mpsc::Sender<(NodeId, Vec<u8>)>,
    _rx_narwhal: mpsc::Receiver<(NodeId, Vec<u8>)>,
    _tx_mempool: mpsc::Sender<NarwhalTransaction>,
    _rx_mempool: mpsc::Receiver<NarwhalTransaction>,
}

impl NarwhalBullshark {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> Self {
        let (narwhal, rx_narwhal) = Narwhal::new(node_id.clone(), peers.clone());
        let narwhal = Arc::new(narwhal);
        let bullshark = Arc::new(Bullshark::new(
            node_id.clone(),
            peers.clone(),
            narwhal.clone(),
        ));
        let (tx_mempool, rx_mempool) = mpsc::channel(1000);
        let (tx_network, _) = mpsc::channel(1000);
        Self {
            _node_id: node_id,
            _peers: peers,
            narwhal,
            bullshark,
            _finalized_blocks: DashMap::new(),
            _latest_height: RwLock::new(0),
            _tx_network: tx_network,
            _rx_narwhal: rx_narwhal,
            _tx_mempool: tx_mempool,
            _rx_mempool: rx_mempool,
        }
    }

    pub async fn start(&self) {
        println!("Starting NarwhalBullshark consensus...");
        // Implementation would go here
    }

    pub async fn get_latest_finalized(&self) -> u64 {
        (*self.bullshark).get_latest_finalized().await
    }

    pub async fn get_finalized_block(&self, seq_num: u64) -> Option<Block> {
        (*self.bullshark).get_finalized_block(seq_num).await
    }

    pub async fn add_transaction(&self, tx: NarwhalTransaction) -> Result<(), VmError> {
        let _tx_data =
            bincode::serialize(&tx).map_err(|e| VmError::SerializationError(e.to_string()))?;
        // Simplified implementation
        let mut pool = self.narwhal.tx_pool.lock();
        pool.push(tx);
        Ok(())
    }

    /// Propose a new block to the network
    pub async fn propose_block(&self, block_data: Vec<u8>) -> Result<String, anyhow::Error> {
        // Generate a block hash
        let block_hash = blake3::hash(&block_data);
        let block_hash_str = hex::encode(block_hash.as_bytes());

        // In a real implementation, this would broadcast the block to the network
        println!("Proposing block with hash: {}", block_hash_str);

        Ok(block_hash_str)
    }

    /// Validate a transaction
    pub async fn validate_transaction(&self, tx_data: &[u8]) -> Result<bool, anyhow::Error> {
        // Basic validation - check if transaction data is not empty
        if tx_data.is_empty() {
            return Ok(false);
        }

        // In a real implementation, this would validate:
        // - Transaction signature
        // - Sender balance
        // - Nonce
        // - Gas limits
        // For now, always return true for non-empty transactions
        Ok(true)
    }

    /// Get current network state
    pub async fn get_network_state(&self) -> Result<String, anyhow::Error> {
        let latest_round = self.get_latest_finalized().await;
        let state = serde_json::json!({
            "latest_finalized_round": latest_round,
            "node_id": self._node_id,
            "peer_count": self._peers.len(),
            "status": "active"
        });

        Ok(state.to_string())
    }
}
