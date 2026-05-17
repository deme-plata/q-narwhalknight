#!/bin/bash

# Script to fix Rust compilation errors and warnings for DAGKnight project

# Ensure we're in a Rust project directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Must be run from a Rust project directory containing Cargo.toml"
    exit 1
fi

# Create backup directory
mkdir -p .backup
cp -r src .backup/src-$(date +%Y%m%d-%H%M%S)

# Function to check if a file exists and create it if it doesn't
ensure_file() {
    local file=$1
    if [ ! -f "$file" ]; then
        mkdir -p "$(dirname "$file")"
        touch "$file"
    fi
}

# 1. Fix contract-related imports and add missing types
echo "Fixing contract-related imports and adding missing types..."

# Create contracts module
ensure_file "src/contracts/mod.rs"
cat << 'EOF' > src/contracts/mod.rs
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
EOF

# Fix vm/mod.rs
ensure_file "src/vm/mod.rs"
sed -i '/use crate::contracts::{Contract, ContractRegistry, ContractCall, ContractResult};/d' src/vm/mod.rs
sed -i '/mod.rs/a use crate::contracts::{Contract, ContractRegistry, ContractCall, ContractResult};' src/vm/mod.rs
sed -i '/use tokio::sync::RwLock;/d' src/vm/mod.rs
sed -i '/use std::collections::HashMap;/d' src/vm/mod.rs
sed -i '/mod.rs/a use std::collections::HashMap;' src/vm/mod.rs

# Fix vm/parallel_executor.rs
ensure_file "src/vm/parallel_executor.rs"
sed -i '/use crate::contracts::{ContractCall, Contract};/d' src/vm/parallel_executor.rs
sed -i '/parallel_executor.rs/a use crate::contracts::{Contract, ContractCall};' src/vm/parallel_executor.rs
sed -i '/parallel_executor.rs/a use std::sync::Arc;' src/vm/parallel_executor.rs

# Fix vm/tiered_vm.rs
ensure_file "src/vm/tiered_vm.rs"
sed -i '/use crate::contracts::Contract;/d' src/vm/tiered_vm.rs
sed -i '/tiered_vm.rs/a use crate::contracts::Contract;' src/vm/tiered_vm.rs
sed -i '/use wasmer::Store;/d' src/vm/tiered_vm.rs

# Fix transaction/mod.rs
ensure_file "src/transaction/mod.rs"
sed -i '/use crate::contracts::{ContractCall};/d' src/transaction/mod.rs
sed -i '/mod.rs/a use crate::contracts::ContractCall;' src/transaction/mod.rs

# 2. Fix P2P network issues
echo "Fixing P2P network issues..."

# Fix p2p.rs syntax errors and unused variables
ensure_file "src/network/p2p.rs"
# Reset p2p.rs to a clean state and apply correct fixes
cp .backup/src-$(ls -t .backup | head -1)/network/p2p.rs src/network/p2p.rs
sed -i 's/contract: Bytes32/_contract: Bytes32/g' src/network/p2p.rs
sed -i 's/timestamp: u64/_timestamp: u64/g' src/network/p2p.rs
sed -i 's/proof: Bytes64/_proof: Bytes64/g' src/network/p2p.rs
sed -i 's/resources: ResourceUsage/_resources: ResourceUsage/g' src/network/p2p.rs
sed -i 's/node_id: Bytes32/_node_id: Bytes32/g' src/network/p2p.rs
sed -i 's/status: NodeStatus/_status: NodeStatus/g' src/network/p2p.rs
sed -i 's/available_resources: AvailableResources/_available_resources: AvailableResources/g' src/network/p2p.rs
sed -i 's/action: ModelRegistryAction/_action: ModelRegistryAction/g' src/network/p2p.rs
sed -i 's/hash: Bytes32/_hash: Bytes32/g' src/network/p2p.rs
sed -i 's/height: u64/_height: u64/g' src/network/p2p.rs
sed -i 's/consensus_type: ConsensusType/_consensus_type: ConsensusType/g' src/network/p2p.rs
sed -i '/use std::time::Duration;/d' src/network/p2p.rs

# Enhance P2pNetwork implementation
sed -i '/pub struct P2pNetwork {/i #[derive(Debug)]' src/network/p2p.rs
if ! grep -q "get_local_peer_id" src/network/p2p.rs; then
    cat << 'EOF' >> src/network/p2p.rs
impl P2pNetwork {
    pub fn get_local_peer_id(&self) -> String {
        "peer_id_placeholder".to_string() // Replace with actual implementation
    }
    
    pub fn get_connected_peers_count(&self) -> usize {
        0 // Replace with actual implementation
    }
}
EOF
fi

# Fix p2p_debug.rs
ensure_file "src/network/p2p_debug.rs"
sed -i 's/use super::p2p::P2pNetwork;/use crate::network::p2p::P2pNetwork;/' src/network/p2p_debug.rs

# Fix network/stub.rs
ensure_file "src/network/stub.rs"
sed -i '/use std::sync::Arc;/d' src/network/stub.rs

# 3. Fix mempool imports
echo "Fixing mempool imports..."
ensure_file "src/mempool/mod.rs"
sed -i 's/use std::collections::{BTreeMap, HashMap, HashSet};/use std::collections::{HashMap, HashSet};/' src/mempool/mod.rs
sed -i '/use tokio::sync::RwLock;/d' src/mempool/mod.rs

# 4. Fix state and StateDB
echo "Fixing StateDB..."
ensure_file "src/state/mod.rs"
sed -i '/use std::sync::Arc;/d' src/state/mod.rs
if ! grep -q "pub struct StateDB" src/state/mod.rs; then
    cat << 'EOF' > src/state/mod.rs
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct StateDB {
    data: Vec<u8>,
}

impl StateDB {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    pub fn insert(&mut self, _key: [u8; 32], _value: Vec<u8>) {
        // Placeholder implementation
    }
}
EOF
else
    sed -i '/pub struct StateDB {/i #[derive(Debug)]' src/state/mod.rs
fi

# 5. Fix consensus
echo "Fixing consensus..."
ensure_file "src/consensus/pbft.rs"
sed -i 's/let blockchain = self.blockchain.write().await;/let mut blockchain = self.blockchain.write().await;/' src/consensus/pbft.rs

# 6. Verify and format
echo "Verifying compilation..."
cargo check
if [ $? -eq 0 ]; then
    echo "Successfully fixed compilation errors!"
else
    echo "Warning: Some issues may remain. Check cargo check output."
    echo "Restoring original p2p.rs for inspection..."
    cp .backup/src-$(ls -t .backup | head -1)/network/p2p.rs src/network/p2p.rs
fi

cargo fmt

echo "Done! Original files backed up in .backup directory."
