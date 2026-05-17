//! Core types used throughout the DAGKnight VM system

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::collections::HashMap;

/// Identifier for a node in the network
pub type NodeId = String;

/// Smart contract or account address
pub type Address = u64;

/// Transaction structure representing operations on the blockchain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction hash - unique identifier
    pub hash: [u8; 32],
    /// Transaction payload data
    pub data: Vec<u8>,
    /// Address of the transaction sender
    pub sender: [u8; 32],
    /// Sender's transaction sequence number
    pub nonce: u64,
    /// Cryptographic signature
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
    /// Unix timestamp when the transaction was created
    pub timestamp: u64,
}

/// Virtual machine state containing contracts, storage, and account data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VmState {
    /// Map of contract addresses to bytecode
    pub contracts: HashMap<Address, Vec<u8>>,
    /// Map of contract addresses to their key-value storage
    pub storage: HashMap<Address, HashMap<Vec<u8>, Vec<u8>>>,
    /// Map of addresses to account balances
    pub balances: HashMap<Address, u64>,
    /// Map of addresses to their current nonce
    pub nonces: HashMap<Address, u64>,
}

/// Result of executing a transaction or contract call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution was successful
    pub success: bool,
    /// Data returned from execution
    pub return_data: Vec<u8>,
    /// Amount of gas consumed during execution
    pub gas_used: u64,
    /// Log messages emitted during execution
    pub logs: Vec<String>,
    /// Error message if execution failed
    pub error: Option<String>,
}
