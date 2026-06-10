use serde::{Serialize, Deserialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contract {
    pub address: [u8; 32],
    pub code: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractCall {
    pub contract_address: [u8; 32],
    pub method: String,
    pub args: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ContractResult {
    pub success: bool,
    pub output: Vec<u8>,
    pub state_changes: std::collections::HashMap<Vec<u8>, Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct ContractRegistry {
    contracts: std::collections::HashMap<[u8; 32], Arc<Contract>>,
}

impl ContractRegistry {
    pub fn new() -> Self {
        Self {
            contracts: std::collections::HashMap::new(),
        }
    }
}
