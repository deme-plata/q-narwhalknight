use std::sync::Arc;
use std::collections::HashMap;

use crate::contracts::{Contract, ContractCall, ContractResult, ContractRegistry};
use crate::state::StateDB;
use self::executor::{WasmExecutor, VMEnvironment};
use self::cache::ContractCache;
use self::parallel_executor::ParallelExecutor;
use self::tiered_vm::TieredVM;

// Submodules of the VM
pub mod executor;
pub mod ai;
pub mod memory;
pub mod cache;
pub mod batch;
pub mod parallel_executor;
pub mod tiered_vm;

// Transaction structure
pub struct Transaction {
    pub from: [u8; 32],
    pub to: [u8; 32],
    pub nonce: u64,
    pub data: Vec<u8>,
    pub signature: Vec<u8>,
}

// VmError enum - centralized error type for the VM
#[derive(Clone, Debug, thiserror::Error)]
pub enum VmError {
    #[error("Consensus error: {0}")]
    ConsensusFailure(String),
    
    #[error("Storage error: {0}")]
    StorageError(#[from] rocksdb::Error),
    
    #[error("Serialization error")]
    SerializationError,
    
    #[error("Contract not found: {0}")]
    ContractNotFound(String),
    
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
    
    #[error("Compilation error: {0}")]
    CompilationError(String),
    
    #[error("Instantiation error: {0}")]
    InstantiationError(String),
    
    #[error("Execution error: {0}")]
    ExecutionError(String),
    
    #[error("Out of gas")]
    OutOfGas,
    
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),
}

// We must use trait objects with Box instead of Arc for async methods
// Or we can use type erasure with the async_trait macro
#[async_trait::async_trait]
pub trait NetworkInterface: Send + Sync {
    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError>;
}

#[async_trait::async_trait]
pub trait ConsensusEngine: Send + Sync {
    async fn validate_contract(&self, hash: [u8; 32], bytecode: &[u8]) -> Result<(), VmError>;
    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError>;
}

// The main DAGKnight VM implementation
pub struct DagkVm {
    pub state_db: Arc<StateDB>,
    pub contract_registry: Arc<ContractRegistry>,
    pub network: Arc<dyn NetworkInterface>,
    pub consensus_engine: Arc<dyn ConsensusEngine>,
    pub executor: WasmExecutor,
    pub contract_cache: ContractCache,
    pub parallel_executor: ParallelExecutor,
    pub tiered_vm: TieredVM,
}

impl DagkVm {
    pub fn new(
        state_db: Arc<StateDB>,
        contract_registry: Arc<ContractRegistry>,
        network: Arc<dyn NetworkInterface>,
        consensus_engine: Arc<dyn ConsensusEngine>,
    ) -> Self {
        // Initialize memory pools if available
        if let Ok(pool) = std::path::Path::new("src/vm/memory/pool.rs").try_exists() {
            if pool {
                // Just call a dummy init to ensure the code compiles
                // In real code this would initialize the pools
            }
        }
        
        // Create VM components
        let executor = WasmExecutor::new();
        let contract_cache = ContractCache::new();
        let parallel_executor = ParallelExecutor::new(Arc::clone(&state_db));
        let tiered_vm = TieredVM::new(Arc::clone(&state_db));
        
        Self {
            state_db,
            contract_registry,
            network,
            consensus_engine,
            executor,
            contract_cache,
            parallel_executor,
            tiered_vm,
        }
    }
    
    // Execute a contract call
    pub async fn call_contract(&self, contract_address: [u8; 32], function: &str, args: Vec<Vec<u8>>, 
                              _sender: [u8; 32], _nonce: u64) -> Result<[u8; 32], VmError> {
        // Validate the transaction with consensus
        self.consensus_engine.validate_contract(contract_address, &[])
            .await
            .map_err(|e| VmError::ConsensusFailure(e.to_string()))?;
            
        // Get the contract - using find_contract helper instead of direct get
        let contract = self.find_contract(&contract_address)
            .ok_or_else(|| VmError::ContractNotFound(hex::encode(contract_address)))?;
            
        // TieredVM is now cloneable and takes &self instead of &mut self
        let _result = self.tiered_vm.execute(&contract, function, &args)?;
        
        // Return a dummy hash for now (in real implementation, this would be the state root)
        Ok([0u8; 32])
    }
    
    // Helper method to find a contract in the registry
    fn find_contract(&self, address: &[u8; 32]) -> Option<Contract> {
        // Use the public getter method
        self.contract_registry.get(address).map(|arc_contract| {
            let contract_ref = arc_contract.as_ref();
            contract_ref.clone()
        })
    }

    
    // Execute a read-only contract view function
    pub fn view_contract(&self, contract_address: [u8; 32], function: &str, _args: Vec<Vec<u8>>, 
                         _sender: [u8; 32], _nonce: u64) -> Result<Vec<u8>, VmError> {
        // Get the contract - using find_contract helper instead of direct get
        let contract = self.find_contract(&contract_address)
            .ok_or_else(|| VmError::ContractNotFound(hex::encode(contract_address)))?;
            
        // Use the executor for read-only calls
        let env = VMEnvironment::new(Arc::clone(&self.state_db), 1_000_000); // 1M gas limit for view calls
        
        // Create wasmer values from args (simplified)
        let wasm_args = vec![];
        
        // Instead of cloning the executor (which now won't work due to Store not being cloneable),
        // we'll create a new one each time
        let mut new_executor = WasmExecutor::new();
        
        // Fixed: Using contract.code instead of contract.bytecode
        let _result = new_executor.execute(&contract.code, env, function, wasm_args)
            .map_err(|e| VmError::ExecutionError(e.to_string()))?;
            
        // Convert result to bytes (simplified)
        Ok(vec![0u8; 4])
    }
    
    // Submit a transaction to the network
    pub async fn submit_transaction(&self, tx: Transaction) -> Result<[u8; 32], VmError> {
        // Validate the transaction with consensus
        self.consensus_engine.validate_contract(tx.to, &tx.data)
            .await
            .map_err(|e| VmError::ConsensusFailure(e.to_string()))?;
            
        // Broadcast the transaction
        self.network.broadcast_contract(tx.to, tx.data.clone())
            .await
            .map_err(|e| VmError::ConsensusFailure(format!("Failed to broadcast transaction: {:?}", e)))?;
            
        // Return a dummy transaction hash for now
        Ok([0u8; 32])
    }
    
    // Execute a batch of contract calls
    pub async fn batch_call_contracts(&self, calls: Vec<ContractCall>) -> Vec<ContractResult> {
        
        // Prepare the calls with their contracts
        let mut call_with_contracts = Vec::new();
        
        for call in calls {
            if let Some(contract) = self.find_contract(&call.contract_address) {
                call_with_contracts.push((call, Arc::new(contract)));
            } else {
                // Contract not found, skip it
                continue;
            }
        }
        
        // Execute in parallel
        let batch_result = self.parallel_executor.execute_batch(call_with_contracts).await;
        let _total_gas_used = batch_result.gas_used;
        
        // Clone results to avoid ownership issues
        let results_clone = batch_result.results.clone();
        let _batch_len = results_clone.len() as u64;
        
        // Convert results to ContractResult objects
        let results: Vec<ContractResult> = results_clone.into_iter()
            .enumerate()
            .map(|(_idx, result)| {
                match result {
                    Ok(data) => ContractResult {
                        output: data,
                        success: true,
                        state_changes: HashMap::new(),
                    },
                    Err(_) => ContractResult {
                        output: Vec::new(),
                        success: false,
                        state_changes: HashMap::new(),
                    }
                }
            })
            .collect();
            
        results
    }
}