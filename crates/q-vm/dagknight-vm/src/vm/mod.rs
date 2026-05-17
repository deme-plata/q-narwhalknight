//! VM module for DAGKnight VM

// Submodules
pub mod ai;
pub mod cache;
pub mod narwhal_bullshark_vm;
pub mod ultra_performance_bridge;

// Consensus engine trait
#[async_trait::async_trait]
pub trait ConsensusEngine: Send + Sync {
    // Original methods
    async fn validate_contract(&self, hash: [u8; 32], bytecode: &[u8]) -> Result<(), VmError>;
    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError>;
    
    // Added methods needed by PBFT implementation with default implementations
    async fn validate_block(&self, _block: &[u8]) -> Result<bool, VmError> {
        // Default implementation
        Ok(true)
    }
    
    async fn finalize_block(&self, _block: &[u8]) -> Result<(), VmError> {
        // Default implementation
        Ok(())
    }
    
    async fn get_latest_block(&self) -> Result<Vec<u8>, VmError> {
        // Default implementation
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum VmError {
    #[error("Consensus error: {0}")]
    ConsensusFailure(String),
    
    #[error("Storage error")]
    StorageError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
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
    
    #[error("Insufficient balance")]
    InsufficientBalance,
    
    #[error("Invalid nonce")]
    InvalidNonce,
}

// Define necessary types for interaction with narwhal_bullshark_vm
pub struct VirtualMachine {
    pub state_db: Arc<crate::state::StateDB>,
}

impl VirtualMachine {
    pub fn new(state_db: Arc<crate::state::StateDB>) -> Self {
        Self { state_db }
    }
}

// Contract state for StateAccess
#[derive(Debug, Clone)]
pub struct ContractState {
    pub code: Vec<u8>,
    pub storage: std::collections::HashMap<Vec<u8>, Vec<u8>>,
}

// Call data for executing contracts
#[derive(Debug, Clone)]
pub struct CallData {
    pub contract_address: u64,
    pub function: String,
    pub arguments: Vec<u8>,
    pub sender: u64,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub value: u64,
}

// Result of execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub gas_used: u64,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

// State access trait
#[async_trait::async_trait]
pub trait StateAccess: Send + Sync {
    async fn get_contract(&self, address: u64) -> Result<Option<Vec<u8>>, VmError>;
    async fn get_storage(&self, address: u64, key: &[u8]) -> Result<Option<Vec<u8>>, VmError>;
    async fn set_storage(&self, address: u64, key: Vec<u8>, value: Vec<u8>) -> Result<(), VmError>;
    async fn get_balance(&self, address: u64) -> Result<u64, VmError>;
    async fn set_balance(&self, address: u64, amount: u64) -> Result<(), VmError>;
    async fn get_nonce(&self, address: u64) -> Result<u64, VmError>;
    async fn get_contract_state(&self, address: u64) -> Result<Option<ContractState>, VmError>;
}

use std::sync::Arc;
