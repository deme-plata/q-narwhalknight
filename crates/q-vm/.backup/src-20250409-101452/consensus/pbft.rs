use async_trait::async_trait;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::RwLock;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};
use crate::vm::VmError;
use crate::transaction::Transaction;
use crate::vm::{ConsensusEngine, VmError as VMError};

type BlockHash = [u8; 32];
type NodeId = String;

// PBFT message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PbftMessage {
    PrepareRequest(PrepareRequest),
    PrepareResponse(PrepareResponse),
    CommitRequest(CommitRequest),
    CommitResponse(CommitResponse),
    ViewChange(ViewChange),
    NewView(NewView),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareRequest {
    pub view: u64,
    pub seq_num: u64,
    pub block_hash: BlockHash,
    pub block_data: Vec<u8>,
    pub primary_id: NodeId,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareResponse {
    pub view: u64,
    pub seq_num: u64,
    pub block_hash: BlockHash,
    pub node_id: NodeId,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitRequest {
    pub view: u64,
    pub seq_num: u64,
    pub block_hash: BlockHash,
    pub node_id: NodeId,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitResponse {
    pub view: u64,
    pub seq_num: u64,
    pub block_hash: BlockHash,
    pub node_id: NodeId,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewChange {
    pub new_view: u64,
    pub seq_num: u64,
    pub node_id: NodeId,
    pub checkpoint: Option<(u64, BlockHash)>,
    pub prepared_proofs: Vec<(u64, BlockHash, Vec<PrepareResponse>)>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewView {
    pub view: u64,
    pub view_change_proofs: Vec<ViewChange>,
    pub prepare_requests: Vec<PrepareRequest>,
    pub node_id: NodeId,
    pub timestamp: u64,
}

// Block structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub hash: BlockHash,
    pub parent_hash: BlockHash,
    pub seq_num: u64,
    pub transactions: Vec<Transaction>,
    pub timestamp: u64,
    pub proposer: NodeId,
}

impl Block {
    pub fn new(parent_hash: BlockHash, seq_num: u64, transactions: Vec<Transaction>, proposer: NodeId) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        let mut block = Self {
            hash: [0; 32],
            parent_hash,
            seq_num,
            transactions,
            timestamp,
            proposer,
        };
        
        block.hash = block.compute_hash();
        block
    }
    
    pub fn compute_hash(&self) -> BlockHash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.parent_hash);
        hasher.update(&self.seq_num.to_le_bytes());
        
        for tx in &self.transactions {
            hasher.update(&tx.hash);
        }
        
        hasher.update(&self.timestamp.to_le_bytes());
        hasher.update(self.proposer.as_bytes());
        
        let mut hash = [0; 32];
        hash.copy_from_slice(hasher.finalize().as_bytes());
        hash
    }
}

// PBFT consensus engine
#[derive(Debug)]
pub struct PbftConsensus {
    node_id: NodeId,
    peers: Vec<NodeId>,
    view: Arc<RwLock<u64>>,
    seq_num: Arc<RwLock<u64>>,
    is_primary: Arc<RwLock<bool>>,
    
    // Message channels
    tx_network: Sender<(NodeId, PbftMessage)>,
    rx_network: Receiver<(NodeId, PbftMessage)>,
    
    // State
    prepare_requests: Arc<DashMap<(u64, u64), PrepareRequest>>,
    prepare_responses: Arc<DashMap<(u64, u64, BlockHash), HashSet<NodeId>>>,
    commit_requests: Arc<DashMap<(u64, u64, BlockHash), HashSet<NodeId>>>,
    commit_responses: Arc<DashMap<(u64, u64, BlockHash), HashSet<NodeId>>>,
    
    // View change state
    view_changes: Arc<DashMap<(u64, NodeId), ViewChange>>,
    
    // Blockchain state
    blockchain: Arc<RwLock<HashMap<BlockHash, Block>>>,
    finalized_blocks: Arc<RwLock<HashMap<u64, BlockHash>>>,
    latest_finalized: Arc<RwLock<u64>>,
    
    // Timer for view change
    view_change_timeout: Duration,
    last_activity: Arc<Mutex<Instant>>,
}

impl PbftConsensus {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> Self {
        let (tx_network, rx_network) = mpsc::channel(1000);
        
        let is_primary = node_id == Self::get_primary(0, &peers);
        
        Self {
            node_id,
            peers,
            view: Arc::new(RwLock::new(0)),
            seq_num: Arc::new(RwLock::new(0)),
            is_primary: Arc::new(RwLock::new(is_primary)),
            
            tx_network,
            rx_network,
            
            prepare_requests: Arc::new(DashMap::new()),
            prepare_responses: Arc::new(DashMap::new()),
            commit_requests: Arc::new(DashMap::new()),
            commit_responses: Arc::new(DashMap::new()),
            
            view_changes: Arc::new(DashMap::new()),
            
            blockchain: Arc::new(RwLock::new(HashMap::new())),
            finalized_blocks: Arc::new(RwLock::new(HashMap::new())),
            latest_finalized: Arc::new(RwLock::new(0)),
            
            view_change_timeout: Duration::from_secs(30),
            last_activity: Arc::new(Mutex::new(Instant::now())),
        }
    }
    
    // Get the primary node for a view
    fn get_primary(view: u64, peers: &[NodeId]) -> NodeId {
        let idx = (view as usize) % (peers.len() + 1);
        if idx < peers.len() {
            peers[idx].clone()
        } else {
            peers[0].clone() // Fallback
        }
    }
    
    // Start the consensus engine
    pub async fn start(&mut self) {
        // Start view change timer
        self.start_view_change_timer();
        
        // Start message processing loop
        self.process_messages().await;
    }
    
    // Start the view change timer
    fn start_view_change_timer(&self) {
        let view = self.view.clone();
        let last_activity = self.last_activity.clone();
        let node_id = self.node_id.clone();
        let peers = self.peers.clone();
        let tx_network = self.tx_network.clone();
        let seq_num = self.seq_num.clone();
        let view_change_timeout = self.view_change_timeout;
        
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                
                let current_view = *view.read().await;
                let elapsed = {
                    let last = last_activity.lock();
                    last.elapsed()
                };
                
                if elapsed > view_change_timeout {
                    // Trigger view change
                    let new_view = current_view + 1;
                    let current_seq = *seq_num.read().await;
                    
                    // Create view change message
                    let view_change = ViewChange {
                        new_view,
                        seq_num: current_seq,
                        node_id: node_id.clone(),
                        checkpoint: None, // For simplicity
                        prepared_proofs: Vec::new(), // For simplicity
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    };
                    
                    // Broadcast view change to all peers
                    for peer in &peers {
                        let _ = tx_network.send((peer.clone(), PbftMessage::ViewChange(view_change.clone()))).await;
                    }
                    
                    // Update view locally
                    *view.write().await = new_view;
                    
                    // Reset activity timer
                    {
                        let mut last = last_activity.lock();
                        *last = Instant::now();
                    }
                }
            }
        });
    }
    
    // Process incoming messages
    async fn process_messages(&mut self) {
        while let Some((_sender, message)) = self.rx_network.recv().await {
            // Update activity timer
            {
                let mut last = self.last_activity.lock();
                *last = Instant::now();
            }
            
            match message {
                PbftMessage::PrepareRequest(req) => {
                    self.handle_prepare_request(req).await;
                },
                PbftMessage::PrepareResponse(resp) => {
                    self.handle_prepare_response(resp).await;
                },
                PbftMessage::CommitRequest(req) => {
                    self.handle_commit_request(req).await;
                },
                PbftMessage::CommitResponse(resp) => {
                    self.handle_commit_response(resp).await;
                },
                PbftMessage::ViewChange(vc) => {
                    self.handle_view_change(vc).await;
                },
                PbftMessage::NewView(nv) => {
                    self.handle_new_view(nv).await;
                },
            }
        }
    }
    
    // Handle prepare request
    async fn handle_prepare_request(&self, req: PrepareRequest) {
        let current_view = *self.view.read().await;
        
        // Verify view and sequence number
        if req.view != current_view {
            return; // Ignore requests from different views
        }
        
        // Store the prepare request
        self.prepare_requests.insert((req.view, req.seq_num), req.clone());
        
        // Create prepare response
        let prepare_response = PrepareResponse {
            view: req.view,
            seq_num: req.seq_num,
            block_hash: req.block_hash,
            node_id: self.node_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Send prepare response to primary
        let _ = self.tx_network.send((req.primary_id.clone(), 
            PbftMessage::PrepareResponse(prepare_response))).await;
    }
    
    // Handle prepare response
    async fn handle_prepare_response(&self, resp: PrepareResponse) {
        let current_view = *self.view.read().await;
        
        // Verify view
        if resp.view != current_view {
            return; // Ignore responses from different views
        }
        
        // Add to prepare responses
        let key = (resp.view, resp.seq_num, resp.block_hash);
        let mut entry = self.prepare_responses.entry(key).or_insert_with(HashSet::new);
        entry.insert(resp.node_id.clone());
        
        // Check if we have 2f+1 prepare responses
        if entry.len() >= self.get_quorum_size() {
            // Create commit request
            let commit_request = CommitRequest {
                view: resp.view,
                seq_num: resp.seq_num,
                block_hash: resp.block_hash,
                node_id: self.node_id.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
            
            // Broadcast commit request to all peers
            for peer in &self.peers {
                let _ = self.tx_network.send((peer.clone(), 
                    PbftMessage::CommitRequest(commit_request.clone()))).await;
            }
        }
    }
    
    // Handle commit request
    async fn handle_commit_request(&self, req: CommitRequest) {
        let current_view = *self.view.read().await;
        
        // Verify view
        if req.view != current_view {
            return; // Ignore requests from different views
        }
        
        // Add to commit requests
        let key = (req.view, req.seq_num, req.block_hash);
        let mut entry = self.commit_requests.entry(key).or_insert_with(HashSet::new);
        entry.insert(req.node_id.clone());
        
        // Create commit response
        let commit_response = CommitResponse {
            view: req.view,
            seq_num: req.seq_num,
            block_hash: req.block_hash,
            node_id: self.node_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Broadcast commit response to all peers
        for peer in &self.peers {
            let _ = self.tx_network.send((peer.clone(), 
                PbftMessage::CommitResponse(commit_response.clone()))).await;
        }
    }
    
    // Handle commit response
    async fn handle_commit_response(&self, resp: CommitResponse) {
        let current_view = *self.view.read().await;
        
        // Verify view
        if resp.view != current_view {
            return; // Ignore responses from different views
        }
        
        // Add to commit responses
        let key = (resp.view, resp.seq_num, resp.block_hash);
        let mut entry = self.commit_responses.entry(key).or_insert_with(HashSet::new);
        entry.insert(resp.node_id.clone());
        
        // Check if we have 2f+1 commit responses
        if entry.len() >= self.get_quorum_size() {
            // Finalize the block
            self.finalize_block(resp.seq_num, resp.block_hash).await;
        }
    }
    
    // Handle view change
    async fn handle_view_change(&self, vc: ViewChange) {
        let current_view = *self.view.read().await;
        
        // Only consider view changes for views greater than current
        if vc.new_view <= current_view {
            return;
        }
        
        // Store view change
        self.view_changes.insert((vc.new_view, vc.node_id.clone()), vc.clone());
        
        // Check if we have enough view changes for the new view
        let mut view_changes_for_new_view = Vec::new();
        for item in self.view_changes.iter() {
            let ((view, _), vc) = item.pair();
            if *view == vc.new_view {
                view_changes_for_new_view.push(vc.clone());
            }
        }
        
        // Check if we have 2f+1 view changes
        if view_changes_for_new_view.len() >= self.get_quorum_size() {
            // Become primary if it's our turn
            let is_primary = self.node_id == Self::get_primary(vc.new_view, &self.peers);
            *self.is_primary.write().await = is_primary;
            
            if is_primary {
                // Create new view message
                let new_view = NewView {
                    view: vc.new_view,
                    view_change_proofs: view_changes_for_new_view.clone(),
                    prepare_requests: Vec::new(), // Simplified for now
                    node_id: self.node_id.clone(),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };
                
                // Broadcast new view to all peers
                for peer in &self.peers {
                    let _ = self.tx_network.send((peer.clone(), 
                        PbftMessage::NewView(new_view.clone()))).await;
                }
            }
            
            // Update view
            *self.view.write().await = vc.new_view;
        }
    }
    
    // Handle new view
    async fn handle_new_view(&self, nv: NewView) {
        let current_view = *self.view.read().await;
        
        // Verify new view
        if nv.view <= current_view {
            return; // Ignore outdated new views
        }
        
        // Verify new view has enough view change proofs
        if nv.view_change_proofs.len() < self.get_quorum_size() {
            return; // Not enough proofs
        }
        
        // Update view
        *self.view.write().await = nv.view;
        
        // Process prepare requests if any
        for prep_req in nv.prepare_requests {
            self.handle_prepare_request(prep_req).await;
        }
    }
    
    // Get quorum size (2f+1 where f is max faulty nodes)
    fn get_quorum_size(&self) -> usize {
        let n = self.peers.len() + 1; // Total nodes including self
        let f = (n - 1) / 3; // Max faulty nodes
        2 * f + 1
    }
    
    // Finalize a block
    async fn finalize_block(&self, seq_num: u64, block_hash: BlockHash) {
            let blockchain = self.blockchain.write().await;
        let mut finalized_blocks = self.finalized_blocks.write().await;
        let mut latest_finalized = self.latest_finalized.write().await;
        
        // Mark block as finalized
        finalized_blocks.insert(seq_num, block_hash);
        
        // Update latest finalized if this is newer
        if seq_num > *latest_finalized {
            *latest_finalized = seq_num;
        }
        
        // Update sequence number if needed
        if seq_num >= *self.seq_num.read().await {
            *self.seq_num.write().await = seq_num + 1;
        }
        
        println!("Block finalized: seq={}, hash={:?}", seq_num, block_hash);
    }
    
    // Propose a new block
    pub async fn propose_block(&self, parent_hash: BlockHash, transactions: Vec<Transaction>) -> Result<BlockHash, VmError> {
        let is_primary = *self.is_primary.read().await;
        if !is_primary {
            return Err(VmError::ConsensusFailure("Not the primary node".to_string()));
        }
        
        let view = *self.view.read().await;
        let seq_num = *self.seq_num.read().await;
        
        // Create new block
        let block = Block::new(parent_hash, seq_num, transactions, self.node_id.clone());
        
        // Store block in blockchain
        {
            let blockchain = self.blockchain.write().await;
            blockchain.insert(block.hash, block.clone());
        }
        
        // Create prepare request
        let prepare_request = PrepareRequest {
            view,
            seq_num,
            block_hash: block.hash,
            block_data: bincode::serialize(&block).map_err(|_| VmError::SerializationError)?,
            primary_id: self.node_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Broadcast prepare request to all peers
        for peer in &self.peers {
            let _ = self.tx_network.send((peer.clone(), 
                PbftMessage::PrepareRequest(prepare_request.clone()))).await;
        }
        
        // Handle prepare request locally
        self.handle_prepare_request(prepare_request).await;
        
        Ok(block.hash)
    }
    
    // Get network sender
    pub fn get_network_sender(&self) -> Sender<(NodeId, PbftMessage)> {
        self.tx_network.clone()
    }
    
    // Get the latest finalized block height
    pub async fn get_latest_finalized(&self) -> u64 {
        *self.latest_finalized.read().await
    }
    
    // Get a finalized block by sequence number
    pub async fn get_finalized_block(&self, seq_num: u64) -> Option<Block> {
        let finalized_blocks = self.finalized_blocks.read().await;
        let blockchain = self.blockchain.read().await;
        
        if let Some(hash) = finalized_blocks.get(&seq_num) {
            blockchain.get(hash).cloned()
        } else {
            None
        }
    }
    
    // For testing: Create a view change message
    pub async fn create_view_change(&self, new_view: u64) -> ViewChange {
        let current_seq = *self.seq_num.read().await;
        
        ViewChange {
            new_view,
            seq_num: current_seq,
            node_id: self.node_id.clone(),
            checkpoint: None, // For simplicity
            prepared_proofs: Vec::new(), // For simplicity
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    // For testing: Create a view change message for a specific node
    pub async fn create_view_change_for_test(&self, node_id: NodeId, new_view: u64) -> ViewChange {
        let current_seq = *self.seq_num.read().await;
        
        ViewChange {
            new_view,
            seq_num: current_seq,
            node_id,
            checkpoint: None, // For simplicity
            prepared_proofs: Vec::new(), // For simplicity
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    // For testing: Get current view
    pub async fn get_current_view(&self) -> u64 {
        *self.view.read().await
    }
}

#[async_trait]
impl ConsensusEngine for PbftConsensus {
    async fn validate_contract(&self, _hash: [u8; 32], _bytecode: &[u8]) -> Result<(), VmError> {
        // Implement contract validation logic for PBFT consensus
        // For simplicity, we'll just accept all contracts for now
        Ok(())
    }

    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError> {
        // Create a transaction for the contract
        let tx = Transaction {
            hash,
            data: bytecode,
            sender: [0; 32], // Placeholder
            nonce: 0,        // Placeholder
            signature: [0; 64], // Placeholder
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        // Get the latest finalized block's hash for parent
        let latest_seq = self.get_latest_finalized().await;
        let parent_hash = if latest_seq > 0 {
            if let Some(block) = self.get_finalized_block(latest_seq).await {
                block.hash
            } else {
                [0; 32] // Genesis block hash
            }
        } else {
            [0; 32] // Genesis block hash
        };
        
        // Propose a new block with this transaction
        match self.propose_block(parent_hash, vec![tx]).await {
            Ok(_) => Ok(()),
            Err(e) => Err(VMError::ConsensusFailure(format!("Failed to propose block: {:?}", e))),
        }
    }
}
