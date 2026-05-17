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

# 1. Fix unresolved imports from crate::contracts
echo "Fixing contract-related imports..."

# Modify src/vm/mod.rs
ensure_file "src/vm/mod.rs"
sed -i '/use crate::contracts::{Contract, ContractRegistry, ContractCall, ContractResult};/d' src/vm/mod.rs
sed -i '/mod.rs/a use crate::transaction::TransactionType::ContractExecution;' src/vm/mod.rs
sed -i '/use tokio::sync::RwLock;/d' src/vm/mod.rs
sed -i '/use std::collections::HashMap;/d' src/vm/mod.rs

# Modify src/vm/parallel_executor.rs
ensure_file "src/vm/parallel_executor.rs"
sed -i '/use crate::contracts::{ContractCall, Contract};/d' src/vm/parallel_executor.rs
sed -i '/parallel_executor.rs/a use crate::transaction::TransactionType::ContractExecution;' src/vm/parallel_executor.rs

# Modify src/vm/tiered_vm.rs
ensure_file "src/vm/tiered_vm.rs"
sed -i '/use crate::contracts::Contract;/d' src/vm/tiered_vm.rs
sed -i '/use wasmer::Store;/d' src/vm/tiered_vm.rs

# Modify src/transaction/mod.rs
ensure_file "src/transaction/mod.rs"
sed -i '/use crate::contracts::{ContractCall};/d' src/transaction/mod.rs
sed -i '/mod.rs/a use crate::transaction::TransactionType::ContractExecution;' src/transaction/mod.rs

# 2. Fix P2P network import
echo "Fixing P2P network imports..."

# Create basic P2pNetwork struct if it doesn't exist
ensure_file "src/network/p2p.rs"
if ! grep -q "pub struct P2pNetwork" src/network/p2p.rs; then
    cat << 'EOF' >> src/network/p2p.rs
#[derive(Debug)]
pub struct P2pNetwork {
    // Basic P2P network implementation
}

impl P2pNetwork {
    pub fn new() -> Self {
        Self {}
    }
}
EOF
fi

# Fix p2p_debug.rs import
ensure_file "src/network/p2p_debug.rs"
sed -i 's/use super::p2p::P2pNetwork;/use crate::network::p2p::P2pNetwork;/' src/network/p2p_debug.rs

# Remove unused Duration import
sed -i '/use std::time::Duration;/d' src/network/p2p.rs

# 3. Fix unused imports
echo "Fixing unused imports..."

# Remove unused Arc from network/stub.rs
ensure_file "src/network/stub.rs"
sed -i '/use std::sync::Arc;/d' src/network/stub.rs

# Fix mempool imports
ensure_file "src/mempool/mod.rs"
sed -i 's/use std::collections::{BTreeMap, HashMap, HashSet};/use std::collections::{HashMap, HashSet};/' src/mempool/mod.rs
sed -i '/use tokio::sync::RwLock;/d' src/mempool/mod.rs

# Remove Arc from state
ensure_file "src/state/mod.rs"
sed -i '/use std::sync::Arc;/d' src/state/mod.rs

# Remove HashMap from contracts
ensure_file "src/contracts/mod.rs"
sed -i '/use std::collections::HashMap;/d' src/contracts/mod.rs

# 4. Fix unused variables in p2p.rs
echo "Fixing unused variables in p2p.rs..."
sed -i 's/async fn handle_transaction(&self, data: Vec<u8>, hash: Bytes32, timestamp: u64)/async fn handle_transaction(&self, data: Vec<u8>, _hash: Bytes32, _timestamp: u64)/' src/network/p2p.rs
sed -i 's/async fn handle_block(&self, data: Vec<u8>, hash: Bytes32, height: u64, timestamp: u64)/async fn handle_block(&self, data: Vec<u8>, _hash: Bytes32, _height: u64, _timestamp: u64)/' src/network/p2p.rs
sed -i 's/async fn handle_consensus(&self, consensus_type: ConsensusType, data: Vec<u8>, timestamp: u64)/async fn handle_consensus(&self, _consensus_type: ConsensusType, data: Vec<u8>, _timestamp: u64)/' src/network/p2p.rs
sed -i 's/contract: Bytes32,/ _contract: Bytes32,/' src/network/p2p.rs
sed -i 's/timestamp: u64,/ _timestamp: u64,/' src/network/p2p.rs
sed -i 's/proof: Bytes64,/ _proof: Bytes64,/' src/network/p2p.rs
sed -i 's/resources: ResourceUsage,/ _resources: ResourceUsage,/' src/network/p2p.rs
sed -i 's/node_id: Bytes32,/ _node_id: Bytes32,/' src/network/p2p.rs
sed -i 's/status: NodeStatus,/ _status: NodeStatus,/' src/network/p2p.rs
sed -i 's/available_resources: AvailableResources,/ _available_resources: AvailableResources,/' src/network/p2p.rs
sed -i 's/action: ModelRegistryAction,/ _action: ModelRegistryAction,/' src/network/p2p.rs

# 5. Fix consensus unused variables
echo "Fixing consensus variables..."
ensure_file "src/consensus/pbft.rs"
sed -i 's/let mut blockchain = self.blockchain.write().await;/let blockchain = self.blockchain.write().await;/' src/consensus/pbft.rs

# 6. Fix StateDB Debug implementation
echo "Adding StateDB Debug implementation..."

# Create or modify state.rs with StateDB
ensure_file "src/state/mod.rs"
if ! grep -q "pub struct StateDB" src/state/mod.rs; then
    cat << 'EOF' >> src/state/mod.rs
use std::sync::Arc;

#[derive(Clone)]
pub struct StateDB {
    // Basic state database implementation
    data: Vec<u8>,
}

impl std::fmt::Debug for StateDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StateDB")
         .field("data_len", &self.data.len())
         .finish()
    }
}

impl StateDB {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}
EOF
else
    # Add Debug implementation if struct exists but lacks it
    if ! grep -q "impl std::fmt::Debug for StateDB" src/state/mod.rs; then
        sed -i '/pub struct StateDB {/a #[derive(Debug)]' src/state/mod.rs
    fi
fi

# 7. Verify changes
echo "Verifying compilation..."
cargo check
if [ $? -eq 0 ]; then
    echo "Successfully fixed compilation errors!"
else
    echo "Warning: Some issues may remain. Check cargo check output."
fi

# Format code
cargo fmt

echo "Done! Original files backed up in .backup directory."
