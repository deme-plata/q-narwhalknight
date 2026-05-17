use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::vm::VmError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    pub address: [u8; 32],  // Contract address (hash of its bytecode)
    pub bytecode: Vec<u8>,  // WebAssembly bytecode
    pub owner: [u8; 32],    // Address of the contract creator
    pub created_at: u64,    // Block height when contract was created
}

impl Contract {
    pub fn new(bytecode: Vec<u8>, owner: [u8; 32], created_at: u64) -> Self {
        let address = Self::compute_address(&bytecode);
        Self {
            address,
            bytecode,
            owner,
            created_at,
        }
    }

    pub fn compute_address(bytecode: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(bytecode);
        let mut address = [0u8; 32];
        address.copy_from_slice(hasher.finalize().as_bytes());
        address
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractCall {
    pub contract_address: [u8; 32],      // Target contract address
    pub function: String,               // Function to call
    pub args: Vec<Vec<u8>>,             // Arguments for the function
    pub sender: [u8; 32],               // Caller's address
    pub value: u64,                     // Value transferred (if any)
    pub gas_limit: u64,                 // Gas limit for execution
    pub nonce: u64,                     // Caller's nonce
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractResult {
    pub success: bool,                  // Whether the call succeeded
    pub return_data: Vec<u8>,           // Return data if success
    pub error: Option<String>,          // Error message if failure
    pub gas_used: u64,                  // Amount of gas used
    pub state_changes: HashMap<Vec<u8>, Option<Vec<u8>>>, // Key-value pairs changed
    pub logs: Vec<ContractLog>,         // Events emitted during execution
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractLog {
    pub contract_address: [u8; 32],     // Contract that emitted the log
    pub topics: Vec<[u8; 32]>,          // Indexed topics
    pub data: Vec<u8>,                  // Log data
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractRegistry {
    contracts: HashMap<[u8; 32], Contract>,
}

impl ContractRegistry {
    pub fn new() -> Self {
        Self {
            contracts: HashMap::new(),
        }
    }

    pub fn register(&mut self, contract: Contract) {
        self.contracts.insert(contract.address, contract);
    }

    pub fn get(&self, address: &[u8; 32]) -> Option<&Contract> {
        self.contracts.get(address)
    }

    pub fn exists(&self, address: &[u8; 32]) -> bool {
        self.contracts.contains_key(address)
    }
}
