#!/bin/bash

# Master fix script for DagKnight VM
echo "Starting master fixes for DagKnight VM..."

PROJECT_DIR=$(pwd)

# First, let's back up all the files we're going to modify
echo "Creating backups..."
cp "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs.bak"
cp "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs.bak"
cp "${PROJECT_DIR}/src/state/mod.rs" "${PROJECT_DIR}/src/state/mod.rs.bak"
cp "${PROJECT_DIR}/src/vm/mod.rs" "${PROJECT_DIR}/src/vm/mod.rs.bak"

# 1. First fix the shared types - Create common types file
echo "Creating common types file..."
mkdir -p "${PROJECT_DIR}/src/types"
cat > "${PROJECT_DIR}/src/types/mod.rs" << 'EOL'
use serde::{Serialize, Deserialize};
use serde_big_array::BigArray;
use std::collections::HashMap;

pub type NodeId = String;
pub type Address = u64;

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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VmState {
    pub contracts: HashMap<Address, Vec<u8>>,
    pub storage: HashMap<Address, HashMap<Vec<u8>, Vec<u8>>>,
    pub balances: HashMap<Address, u64>,
    pub nonces: HashMap<Address, u64>,
}

// Local version of ExecutionResult for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub gas_used: u64,
    pub logs: Vec<String>,
    pub error: Option<String>,
}
EOL
echo "  Created common types file"

# 2. Update src/lib.rs to expose the types module
echo "Updating lib.rs..."
if [ -f "${PROJECT_DIR}/src/lib.rs" ]; then
  cat > "${PROJECT_DIR}/temp_lib.rs" << 'EOL'
// Add the new types module and config re-export
pub mod types; // Common types used across modules
pub mod consensus;
pub mod vm;
pub mod state;
pub mod network;
pub mod transaction;
pub use crate::vm::narwhal_bullshark_vm::config;

pub fn init(config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let _ = config::load_config(config_path);
    
    // Handle batch size argument if provided
    if let Some(size_str) = std::env::args().nth(2) {
        if let Ok(size) = size_str.parse::<usize>() {
            config::update_batch_size(size);
        }
    }
    
    Ok(())
}
EOL
  mv "${PROJECT_DIR}/temp_lib.rs" "${PROJECT_DIR}/src/lib.rs"
  echo "  Updated lib.rs"
fi

# 3. Fix the consensus/narwhal_bullshark.rs file
echo "Fixing narwhal_bullshark.rs..."
cat > "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" << 'EOL'
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tokio::sync::{mpsc, RwLock};
use parking_lot::Mutex;
use anyhow::Result;
use dashmap::DashMap;

// Use the common types
use crate::types::{NodeId, Transaction};
use crate::consensus::pbft::Block;
use crate::vm::VmError;

pub struct Narwhal {
    node_id: NodeId,
    peers: Vec<NodeId>,
    vertices: DashMap<u64, Vec<u8>>, // Modified to use u64 and Vec<u8> 
    latest_round: RwLock<u64>,
    tx_pool: Arc<Mutex<Vec<Transaction>>>,
    tx_network: mpsc::Sender<(NodeId, Vec<u8>)>,
}

impl Narwhal {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> (Self, mpsc::Receiver<(NodeId, Vec<u8>)>) {
        let (tx, rx) = mpsc::channel(1000);
        
        (Self {
            node_id,
            peers,
            vertices: DashMap::new(),
            latest_round: RwLock::new(0),
            tx_pool: Arc::new(Mutex::new(Vec::new())),
            tx_network: tx,
        }, rx)
    }
}

pub struct Bullshark {
    node_id: NodeId,
    peers: Vec<NodeId>,
    latest_round: RwLock<u64>,
    finalized_blocks: DashMap<u64, Block>,
}

impl Bullshark {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, narwhal: Arc<Narwhal>) -> Self {
        Self {
            node_id,
            peers,
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
    node_id: NodeId,
    peers: Vec<NodeId>,
    narwhal: Arc<Narwhal>,
    bullshark: Arc<Bullshark>,
    finalized_blocks: DashMap<u64, Block>,
    latest_height: RwLock<u64>,
    tx_network: mpsc::Sender<(NodeId, Vec<u8>)>,
    rx_narwhal: mpsc::Receiver<(NodeId, Vec<u8>)>,
    tx_mempool: mpsc::Sender<Transaction>,
    rx_mempool: mpsc::Receiver<Transaction>,
}

impl NarwhalBullshark {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> Self {
        let (narwhal, rx_narwhal) = Narwhal::new(node_id.clone(), peers.clone());
        let narwhal = Arc::new(narwhal);
        let bullshark = Arc::new(Bullshark::new(node_id.clone(), peers.clone(), narwhal.clone()));
        
        let (tx_mempool, rx_mempool) = mpsc::channel(1000);
        let (tx_network, _) = mpsc::channel(1000);
        
        Self {
            node_id,
            peers,
            narwhal,
            bullshark,
            finalized_blocks: DashMap::new(),
            latest_height: RwLock::new(0),
            tx_network,
            rx_narwhal,
            tx_mempool,
            rx_mempool,
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
    
    pub async fn add_transaction(&self, tx: Transaction) -> Result<(), VmError> {
        let tx_data = bincode::serialize(&tx)
            .map_err(|e| VmError::SerializationError(e.to_string()))?;
            
        // Simplified implementation
        let mut pool = self.narwhal.tx_pool.lock();
        pool.push(tx);
        
        Ok(())
    }
}
EOL
echo "  Fixed narwhal_bullshark.rs"

# 4. Fix the vm/mod.rs file
echo "Fixing vm/mod.rs..."
cat > "${PROJECT_DIR}/temp_vm_mod.rs" << 'EOL'
use async_trait::async_trait;
use anyhow::Result;
use std::sync::Arc;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Re-export from types module
pub use crate::types::{ExecutionResult, NodeId, Transaction, VmState, Address};

// Error handling
#[derive(Debug)]
pub enum VmError {
    SerializationError(String),
    InsufficientBalance,
    InvalidNonce,
    ContractNotFound(String),
}

impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
            Self::InsufficientBalance => write!(f, "Insufficient balance"),
            Self::InvalidNonce => write!(f, "Invalid nonce"),
            Self::ContractNotFound(addr) => write!(f, "Contract not found: {}", addr),
        }
    }
}

impl std::error::Error for VmError {}

#[derive(Clone)]
pub struct ContractState {
    pub code: Vec<u8>,
    pub storage: HashMap<Vec<u8>, Vec<u8>>,
}

pub struct CallData {
    pub contract_address: Address,
    pub function: String,
    pub arguments: Vec<u8>,
    pub sender: Address,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub value: u64,
}

#[async_trait]
pub trait StateAccess: Send + Sync {
    async fn get_contract(&self, address: Address) -> Result<Option<Vec<u8>>, VmError>;
    async fn get_storage(&self, address: Address, key: &[u8]) -> Result<Option<Vec<u8>>, VmError>;
    async fn set_storage(&self, address: Address, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError>;
    async fn get_balance(&self, address: Address) -> Result<u64, VmError>;
    async fn set_balance(&self, address: Address, amount: u64) -> Result<(), VmError>;
    async fn get_nonce(&self, address: Address) -> Result<u64, VmError>;
    async fn get_contract_state(&self, address: Address) -> Result<Option<ContractState>, VmError>;
}

pub struct VirtualMachine {
    state_db: Arc<crate::state::StateDB>,
}

impl VirtualMachine {
    pub fn new(state_db: Arc<crate::state::StateDB>) -> Self {
        Self { state_db }
    }

    pub async fn execute(&mut self, call_data: &CallData, state_access: &dyn StateAccess) -> Result<ExecutionResult, VmError> {
        // Simplified implementation
        let gas_used = call_data.gas_limit / 2;
        Ok(ExecutionResult {
            success: true,
            return_data: Vec::new(),
            gas_used,
            logs: vec![format!("Executed function: {}", call_data.function)],
            error: None,
        })
    }
}

// Include narwhal_bullshark_vm module
pub mod narwhal_bullshark_vm;
// Include AI executor module if present
pub mod ai;
EOL
mv "${PROJECT_DIR}/temp_vm_mod.rs" "${PROJECT_DIR}/src/vm/mod.rs"
echo "  Fixed vm/mod.rs"

# 5. Fix the state/mod.rs file
echo "Fixing state/mod.rs..."
cat > "${PROJECT_DIR}/temp_state_mod.rs" << 'EOL'
/// State management for DAGKnight
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::types::VmState;

pub struct StateDB {
    pub state: Arc<RwLock<VmState>>,
    pub resource_ledger: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl StateDB {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(VmState::default())),
            resource_ledger: None,
        }
    }

    pub fn with_state(state: Arc<RwLock<VmState>>) -> Self {
        Self {
            state,
            resource_ledger: None,
        }
    }
}
EOL
mv "${PROJECT_DIR}/temp_state_mod.rs" "${PROJECT_DIR}/src/state/mod.rs"
echo "  Fixed state/mod.rs"

# 6. Fix the vm/narwhal_bullshark_vm.rs file
echo "Fixing narwhal_bullshark_vm.rs..."
cat > "${PROJECT_DIR}/temp_narwhal_bullshark_vm.rs" << 'EOL'
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex as TokioMutex};
use parking_lot::Mutex;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use serde_big_array::BigArray;
use dashmap::DashMap;
use anyhow::Result;

// Use the common types
use crate::types::{NodeId, Transaction, VmState, Address, ExecutionResult as CommonExecutionResult};
use crate::consensus::narwhal_bullshark::NarwhalBullshark;
use crate::consensus::pbft::Block;
use crate::vm::{VirtualMachine, VmError, ContractState, CallData, StateAccess};
use crate::state::StateDB;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartContractTx {
    pub address: Address,
    pub function: String,
    pub arguments: Vec<u8>,
    pub sender: Address,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub nonce: u64,
    pub value: u64,
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmBlockResult {
    pub block_hash: [u8; 32],
    pub height: u64,
    pub transactions: Vec<SmartContractTx>,
    pub tx_results: Vec<CommonExecutionResult>,
    pub state_root: [u8; 32],
    pub timestamp: u64,
}

pub struct NarwhalBullsharkVm {
    consensus: Arc<NarwhalBullshark>,
    vm: Arc<VirtualMachine>,
    current_state: Arc<RwLock<VmState>>,
    state_history: Arc<DashMap<u64, [u8; 32]>>,
    pending_txs: Arc<Mutex<Vec<SmartContractTx>>>,
    tx_results: Arc<DashMap<[u8; 32], CommonExecutionResult>>,
    block_results: Arc<DashMap<u64, VmBlockResult>>,
    last_processed_height: Arc<RwLock<u64>>,
    tx_throughput: Arc<RwLock<(u64, Instant)>>,
    shutdown: Arc<TokioMutex<bool>>,
}

impl NarwhalBullsharkVm {
    pub fn new(node_id: NodeId, peers: Vec<NodeId>, vm: Arc<VirtualMachine>) -> Self {
        let consensus = Arc::new(NarwhalBullshark::new(node_id, peers));
        let current_state = Arc::new(RwLock::new(VmState::default()));

        Self {
            consensus,
            vm,
            current_state,
            state_history: Arc::new(DashMap::new()),
            pending_txs: Arc::new(Mutex::new(Vec::new())),
            tx_results: Arc::new(DashMap::new()),
            block_results: Arc::new(DashMap::new()),
            last_processed_height: Arc::new(RwLock::new(0)),
            tx_throughput: Arc::new(RwLock::new((0, Instant::now()))),
            shutdown: Arc::new(TokioMutex::new(false)),
        }
    }

    pub async fn start(&self) -> Result<(), VmError> {
        let consensus = Arc::clone(&self.consensus);
        tokio::spawn(async move {
            (*consensus).start().await;
        });

        self.start_execution_loop();
        self.start_block_processor();
        self.start_metrics_reporter();

        Ok(())
    }

    pub async fn submit_transaction(&self, tx: SmartContractTx) -> Result<[u8; 32], VmError> {
        self.validate_transaction(&tx).await?;
        let tx_hash = self.compute_tx_hash(&tx);
        let consensus_tx = Transaction {
            hash: tx_hash,
            data: bincode::serialize(&tx)
                .map_err(|e| VmError::SerializationError(e.to_string()))?,
            sender: self.address_to_bytes(&tx.sender),
            nonce: tx.nonce,
            signature: tx.signature,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        {
            let mut pending = self.pending_txs.lock();
            pending.push(tx);
        }

        self.consensus.add_transaction(consensus_tx).await
            .map_err(|_| VmError::SerializationError("Failed to add transaction to consensus".to_string()))?;
        Ok(tx_hash)
    }

    async fn validate_transaction(&self, tx: &SmartContractTx) -> Result<(), VmError> {
        let state = self.current_state.read().await;
        let balance = *state.balances.get(&tx.sender).unwrap_or(&0);
        let tx_cost = tx.gas_limit * tx.gas_price + tx.value;

        if balance < tx_cost {
            return Err(VmError::InsufficientBalance);
        }

        let current_nonce = *state.nonces.get(&tx.sender).unwrap_or(&0);
        if tx.nonce < current_nonce {
            return Err(VmError::InvalidNonce);
        }

        if !tx.function.is_empty() && !state.contracts.contains_key(&tx.address) {
            return Err(VmError::ContractNotFound(tx.address.to_string()));
        }

        Ok(())
    }

    fn start_execution_loop(&self) {
        let vm = Arc::clone(&self.vm);
        let current_state: Arc<RwLock<VmState>> = Arc::clone(&self.current_state);
        let pending_txs = Arc::clone(&self.pending_txs);
        let tx_results = Arc::clone(&self.tx_results);

        tokio::spawn(async move {
            let state_access = VmStateAccess::new(Arc::clone(&current_state));
            loop {
                let batch = {
                    let mut pending = pending_txs.lock();
                    let count = pending.len().min(100);
                    if count == 0 {
                        Vec::new()
                    } else {
                        pending.drain(0..count).collect::<Vec<_>>()
                    }
                };
                if batch.is_empty() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }

                for tx in batch {
                    let tx_hash = Self::compute_tx_hash_static(&tx);
                    let call_data = CallData {
                        contract_address: tx.address,
                        function: tx.function.clone(),
                        arguments: tx.arguments.clone(),
                        sender: tx.sender,
                        gas_limit: tx.gas_limit,
                        gas_price: tx.gas_price,
                        value: tx.value,
                    };

                    match vm.execute(&call_data, &state_access).await {
                        Ok(result) => {
                            tx_results.insert(tx_hash, result);
                        },
                        Err(e) => {
                            tx_results.insert(tx_hash, CommonExecutionResult {
                                success: false,
                                return_data: Vec::new(),
                                gas_used: 0,
                                logs: Vec::new(),
                                error: Some(e.to_string()),
                            });
                        }
                    }
                }
            }
        });
    }

    fn start_block_processor(&self) {
        let consensus = Arc::clone(&self.consensus);
        let current_state: Arc<RwLock<VmState>> = Arc::clone(&self.current_state);
        let state_history = Arc::clone(&self.state_history);
        let block_results = Arc::clone(&self.block_results);
        let tx_results = Arc::clone(&self.tx_results);
        let last_processed_height = Arc::clone(&self.last_processed_height);
        let tx_throughput = Arc::clone(&self.tx_throughput);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            loop {
                interval.tick().await;
                let latest_height = consensus.get_latest_finalized().await;
                let processed_height = *last_processed_height.read().await;

                if latest_height <= processed_height {
                    continue;
                }

                for height in (processed_height + 1)..=latest_height {
                    if let Some(block) = consensus.get_finalized_block(height).await {
                        let vm_block_result = process_block(&block, &tx_results).await;
                        update_state(&block, &vm_block_result, &current_state).await;
                        state_history.insert(height, vm_block_result.state_root);
                        block_results.insert(height, vm_block_result);
                        let mut throughput = tx_throughput.write().await;
                        throughput.0 += block.transactions.len() as u64;
                    }
                }

                let mut last_height = last_processed_height.write().await;
                *last_height = latest_height;
            }
        });

        async fn process_block(
            block: &Block,
            tx_results: &DashMap<[u8; 32], CommonExecutionResult>,
        ) -> VmBlockResult {
            let mut vm_txs = Vec::new();
            let mut results = Vec::new();

            for tx in &block.transactions {
                if let Ok(vm_tx) = bincode::deserialize::<SmartContractTx>(&tx.data) {
                    if let Some(result) = tx_results.get(&tx.hash) {
                        vm_txs.push(vm_tx);
                        results.push(result.clone());
                    }
                }
            }

            let mut hasher = blake3::Hasher::new();
            // Manually hash each field instead of trying to serialize
            for (i, tx) in vm_txs.iter().enumerate() {
                if let Ok(tx_bytes) = bincode::serialize(tx) {
                    hasher.update(&tx_bytes);
                    let result = &results[i];
                    hasher.update(&[result.success as u8]);
                    hasher.update(&result.return_data);
                    hasher.update(&result.gas_used.to_le_bytes());
                    for log in &result.logs {
                        hasher.update(log.as_bytes());
                    }
                    if let Some(error) = &result.error {
                        hasher.update(error.as_bytes());
                    }
                }
            }

            let mut state_root = [0u8; 32];
            state_root.copy_from_slice(hasher.finalize().as_bytes());

            VmBlockResult {
                block_hash: block.hash,
                height: block.seq_num,
                transactions: vm_txs,
                tx_results: results,
                state_root,
                timestamp: block.timestamp,
            }
        }

        async fn update_state(
            _block: &Block,
            vm_block_result: &VmBlockResult,
            current_state: &RwLock<VmState>,
        ) {
            let mut state = current_state.write().await;
            for (i, tx) in vm_block_result.transactions.iter().enumerate() {
                let result = &vm_block_result.tx_results[i];
                if result.success {
                    state.nonces.insert(tx.sender, tx.nonce + 1);
                    if let Some(balance) = state.balances.get_mut(&tx.sender) {
                        let gas_cost = result.gas_used * tx.gas_price;
                        *balance = balance.saturating_sub(gas_cost);
                    }
                    if tx.value > 0 {
                        if let Some(sender_balance) = state.balances.get_mut(&tx.sender) {
                            *sender_balance = sender_balance.saturating_sub(tx.value);
                        }
                        let recipient_balance = state.balances.entry(tx.address).or_insert(0);
                        *recipient_balance = recipient_balance.saturating_add(tx.value);
                    }
                    if tx.function.is_empty() && !result.return_data.is_empty() {
                        state.contracts.insert(tx.address, result.return_data.clone());
                    }
                }
            }
        }
    }

    fn start_metrics_reporter(&self) {
        let tx_throughput = Arc::clone(&self.tx_throughput);
        let shutdown = Arc::clone(&self.shutdown);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                if *shutdown.lock().await {
                    break;
                }
                let mut throughput = tx_throughput.write().await;
                let tx_count = throughput.0;
                let elapsed = throughput.1.elapsed();
                if tx_count > 0 && elapsed.as_secs() > 0 {
                    let tps = tx_count as f64 / elapsed.as_secs() as f64;
                    println!("VM TPS: {:.2} ({} transactions in {:?})", tps, tx_count, elapsed);
                    throughput.0 = 0;
                    throughput.1 = Instant::now();
                }
            }
        });
    }

    pub async fn stop(&self) -> Result<(), VmError> {
        let mut shutdown = self.shutdown.lock().await;
        *shutdown = true;
        Ok(())
    }

    pub async fn get_tps(&self) -> f64 {
        let throughput = self.tx_throughput.read().await;
        let tx_count = throughput.0;
        let elapsed = throughput.1.elapsed();
        if tx_count > 0 && elapsed.as_secs() > 0 {
            tx_count as f64 / elapsed.as_secs() as f64
        } else {
            0.0
        }
    }

    pub async fn get_block_result(&self, height: u64) -> Option<VmBlockResult> {
        self.block_results.get(&height).map(|r| r.clone())
    }

    pub async fn get_transaction_result(&self, tx_hash: [u8; 32]) -> Option<CommonExecutionResult> {
        self.tx_results.get(&tx_hash).map(|r| r.clone())
    }

    fn address_to_bytes(&self, address: &Address) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        let addr_bytes = address.to_be_bytes();
        bytes[32 - addr_bytes.len()..].copy_from_slice(&addr_bytes);
        bytes
    }

    fn compute_tx_hash(&self, tx: &SmartContractTx) -> [u8; 32] {
        Self::compute_tx_hash_static(tx)
    }

    fn compute_tx_hash_static(tx: &SmartContractTx) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&tx.address.to_be_bytes());
        hasher.update(tx.function.as_bytes());
        hasher.update(&tx.arguments);
        hasher.update(&tx.sender.to_be_bytes());
        hasher.update(&tx.gas_limit.to_le_bytes());
        hasher.update(&tx.gas_price.to_le_bytes());
        hasher.update(&tx.nonce.to_le_bytes());
        hasher.update(&tx.value.to_le_bytes());
        let mut hash = [0u8; 32];
        hash.copy_from_slice(hasher.finalize().as_bytes());
        hash
    }
}

struct VmStateAccess {
    state: Arc<RwLock<VmState>>,
}

impl VmStateAccess {
    fn new(state: Arc<RwLock<VmState>>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl StateAccess for VmStateAccess {
    async fn get_contract(&self, address: Address) -> Result<Option<Vec<u8>>, VmError> {
        let state = self.state.read().await;
        Ok(state.contracts.get(&address).cloned())
    }

    async fn get_storage(&self, address: Address, key: &[u8]) -> Result<Option<Vec<u8>>, VmError> {
        let state = self.state.read().await;
        Ok(state.storage.get(&address).and_then(|s| s.get(key).cloned()))
    }

    async fn set_storage(&self, address: Address, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError> {
    let mut state = self.state.write().await;
        state.storage.entry(address).or_insert_with(HashMap::new).insert(key, value);
        Ok(())
    }

    async fn get_balance(&self, address: Address) -> Result<u64, VmError> {
        let state = self.state.read().await;
        Ok(*state.balances.get(&address).unwrap_or(&0))
    }

    async fn set_balance(&self, address: Address, amount: u64) -> Result<(), VmError> {
        let mut state = self.state.write().await;
        state.balances.insert(address, amount);
        Ok(())
    }

    async fn get_nonce(&self, address: Address) -> Result<u64, VmError> {
        let state = self.state.read().await;
        Ok(*state.nonces.get(&address).unwrap_or(&0))
    }

    async fn get_contract_state(&self, address: Address) -> Result<Option<ContractState>, VmError> {
        let state = self.state.read().await;
        state.contracts.get(&address).map(|code| {
            let storage = state.storage.get(&address).cloned().unwrap_or_default();
            Ok(Some(ContractState { code: code.clone(), storage }))
        }).unwrap_or(Ok(None))
    }
}

pub struct TpsBenchmark {
    vm: Arc<NarwhalBullsharkVm>,
    transaction_count: usize,
    batch_size: usize,
}

impl TpsBenchmark {
    pub fn new(vm: Arc<NarwhalBullsharkVm>, transaction_count: usize, batch_size: usize) -> Self {
        Self { vm, transaction_count, batch_size }
    }

    pub async fn run(&self) -> Result<f64, VmError> {
        println!("Starting TPS benchmark with {} transactions in batches of {}", 
                 self.transaction_count, self.batch_size);

        let start_time = Instant::now();
        let mut submitted = 0;
        let sender_addr = 1001u64;
        {
            let mut state = self.vm.current_state.write().await;
            state.balances.insert(sender_addr, 10_000_000_000);
            state.nonces.insert(sender_addr, 0);
        }

        while submitted < self.transaction_count {
            let batch_size = self.batch_size.min(self.transaction_count - submitted);
            let mut batch = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let nonce = submitted as u64 + i as u64;
                let tx = SmartContractTx {
                    address: 1000u64,
                    function: "transfer".to_string(),
                    arguments: vec![0u8; 32],
                    sender: sender_addr,
                    gas_limit: 100_000,
                    gas_price: 1,
                    nonce,
                    value: 0,
                    signature: [0u8; 64],
                };
                batch.push(tx);
            }

            for tx in batch {
                if let Err(e) = self.vm.submit_transaction(tx).await {
                    println!("Error submitting transaction: {:?}", e);
                }
            }

            submitted += batch_size;
            if submitted % (self.transaction_count / 10) == 0 || submitted == self.transaction_count {
                println!("Submitted {}/{} transactions ({:.1}%)", 
                         submitted, self.transaction_count, 
                         (submitted as f64 / self.transaction_count as f64) * 100.0);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        println!("Waiting for transactions to be processed...");
        let mut processed = 0;
        let timeout = Duration::from_secs(60);
        let start_wait = Instant::now();

        while processed < self.transaction_count && start_wait.elapsed() < timeout {
            let height = *self.vm.last_processed_height.read().await;
            if let Some(block_result) = self.vm.get_block_result(height).await {
                processed += block_result.transactions.len();
                println!("Processed {}/{} transactions ({:.1}%) in block {}", 
                         processed, self.transaction_count, 
                         (processed as f64 / self.transaction_count as f64) * 100.0,
                         height);
                if processed >= self.transaction_count {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        let elapsed = start_time.elapsed();
        let tps = self.transaction_count as f64 / elapsed.as_secs_f64();
        println!("Benchmark completed:");
        println!("  Total transactions: {}", self.transaction_count);
        println!("  Elapsed time: {:.2} seconds", elapsed.as_secs_f64());
        println!("  Transactions per second: {:.2} TPS", tps);
        Ok(tps)
    }
}

pub async fn run_benchmark_suite(node_id: String, peers: Vec<String>, vm: Arc<VirtualMachine>) -> Result<(), VmError> {
    println!("Starting Narwhal-Bullshark VM benchmark suite");
    let nbs_vm = Arc::new(NarwhalBullsharkVm::new(node_id, peers, vm));
    nbs_vm.start().await?;
    tokio::time::sleep(Duration::from_secs(5)).await;

    let batch_sizes = [10, 50, 100, 500, 1000];
    let transaction_count = 10000;
    let mut results = Vec::new();

    for &batch_size in &batch_sizes {
        println!("\nRunning benchmark with batch size: {}", batch_size);
        let benchmark = TpsBenchmark::new(nbs_vm.clone(), transaction_count, batch_size);
        match benchmark.run().await {
            Ok(tps) => results.push((batch_size, tps)),
            Err(e) => println!("Benchmark failed: {:?}", e),
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    println!("\nBenchmark Results Summary:");
    println!("---------------------------");
    for (batch_size, tps) in &results {
        println!("Batch size {}: {:.2} TPS", batch_size, tps);
    }

    if let Some((best_batch, best_tps)) = results.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
        println!("\nBest performance: {:.2} TPS with batch size {}", best_tps, best_batch);
    }

    nbs_vm.stop().await?;
    Ok(())
}

// Config module
pub mod config {
    pub fn load_config(_config_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Placeholder implementation
        Ok(())
    }

    pub fn update_batch_size(_size: usize) {
        // Placeholder implementation
    }
}
EOL
mv "${PROJECT_DIR}/temp_narwhal_bullshark_vm.rs" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
echo "  Fixed narwhal_bullshark_vm.rs"

echo "All fixes applied! Try building your project again with 'cargo build'"
