#!/bin/bash

# Script to fix issues in the DagKnight VM codebase

# Check if we're in the project root directory
if [[ ! -f "Cargo.toml" ]]; then
  echo "Error: Please run this script from the project root directory (where Cargo.toml is located)"
  exit 1
fi

echo "Creating backup of your codebase..."
# Create a backup of the project
BACKUP_DIR="../dagknight-vm-backup-$(date +%Y%m%d%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR"
echo "Backup created at $BACKUP_DIR"

echo "Adding required dependencies to Cargo.toml..."
# Add required dependencies
if ! grep -q "toml" Cargo.toml; then
  # Add toml dependency if it doesn't exist
  sed -i '/^\[dependencies\]/a toml = "0.5"' Cargo.toml
fi

if ! grep -q "clap" Cargo.toml; then
  # Add clap dependency if it doesn't exist
  sed -i '/^\[dependencies\]/a clap = { version = "4.0", features = ["derive"] }' Cargo.toml
fi

echo "Fixing AI Executor ShardingCapability pattern matching..."
# Fix ShardingCapability matching in AI executor
if [[ -f "src/vm/ai/executor.rs" ]]; then
  sed -i 's/ShardingCapability::Horizontal => ShardingStrategy::Horizontal,/ShardingCapability::Horizontal | ShardingCapability::DataParallel => ShardingStrategy::Horizontal,/' src/vm/ai/executor.rs
  sed -i 's/ShardingCapability::Vertical => ShardingStrategy::Vertical,/ShardingCapability::Vertical | ShardingCapability::ModelParallel => ShardingStrategy::Vertical,/' src/vm/ai/executor.rs
  
  # Remove any existing DataParallel and ModelParallel pattern matches if they exist
  sed -i '/ShardingCapability::DataParallel => ShardingStrategy::Horizontal,/d' src/vm/ai/executor.rs
  sed -i '/ShardingCapability::ModelParallel => ShardingStrategy::Vertical,/d' src/vm/ai/executor.rs
fi

echo "Fixing Narwhal Bullshark Bench Binary..."
# Fix the narwhal_bullshark_bench.rs file
if [[ -f "src/bin/narwhal_bullshark_bench.rs" ]]; then
  # Update use statements
  sed -i 's/use clap::Parser;/use clap::Parser;\nuse dagknight_vm::config;\nuse dagknight_vm::vm::narwhal_bullshark_vm::{NarwhalBullsharkVm, SmartContractTx};\nuse dagknight_vm::vm::VirtualMachine;/' src/bin/narwhal_bullshark_bench.rs
  
  # Update the Args struct to derive Parser
  sed -i 's/struct Args {/#[derive(Parser)]\nstruct Args {/' src/bin/narwhal_bullshark_bench.rs
  
  # Replace all dagknight_node references with dagknight_vm
  sed -i 's/dagknight_node::config/config/g' src/bin/narwhal_bullshark_bench.rs
  sed -i 's/dagknight_node::vm::VirtualMachine/VirtualMachine/g' src/bin/narwhal_bullshark_bench.rs
  sed -i 's/dagknight_node::vm::NarwhalBullsharkVm/NarwhalBullsharkVm/g' src/bin/narwhal_bullshark_bench.rs
  sed -i 's/dagknight_node::vm::narwhal_bullshark_vm::SmartContractTx/SmartContractTx/g' src/bin/narwhal_bullshark_bench.rs
fi

echo "Fixing types module issues..."
# Create a types module if it doesn't exist
mkdir -p "src/types"
if [[ ! -f "src/types/mod.rs" ]]; then
  cat > "src/types/mod.rs" << 'EOF'
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub type NodeId = String;
pub type Address = [u8; 32];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: [u8; 32],
    pub data: Vec<u8>,
    pub sender: Address,
    pub nonce: u64,
    pub signature: [u8; 64],
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub data: Vec<u8>,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmState {
    pub blocks: HashMap<u64, Vec<u8>>,
    pub transactions: HashMap<[u8; 32], Transaction>,
    pub results: HashMap<[u8; 32], ExecutionResult>,
    pub contracts: HashMap<Address, Vec<u8>>,
    pub storage: HashMap<Address, HashMap<Vec<u8>, Vec<u8>>>,
}
EOF

  # Update imports in affected files
  sed -i 's/use crate::types/use crate::types/' src/consensus/narwhal_bullshark.rs
  sed -i 's/use crate::types/use crate::types/' src/state/mod.rs
  sed -i 's/use crate::types/use crate::types/' src/vm/mod.rs
  sed -i 's/use crate::types/use crate::types/' src/vm/narwhal_bullshark_vm.rs
fi

echo "Fixing contract module issues..."
# Create or update the contracts module
if [[ -f "src/contracts/mod.rs" ]]; then
  # Add ModelRegistration and ResourceRequirements if they don't exist
  if ! grep -q "ModelRegistration" "src/contracts/mod.rs"; then
    cat >> "src/contracts/mod.rs" << 'EOF'

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_cpu_cores: u32,
    pub min_memory_mb: u64,
    pub min_gpu_memory_mb: u64,
    pub preferred_batch_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistration {
    pub model_id: String,
    pub version: String,
    pub owner: [u8; 32],
    pub description: String,
    pub capabilities: ShardingCapability,
    pub resources: ResourceRequirements,
    pub hash: [u8; 32],
    pub timestamp: u64,
}
EOF
  fi
fi

echo "Fixing Arc type annotation in narwhal_bullshark_vm.rs..."
if [[ -f "src/vm/narwhal_bullshark_vm.rs" ]]; then
  sed -i 's/let tx_results = Arc::clone(&self.tx_results);/let tx_results: Arc<DashMap<\[u8; 32\], ExecutionResult>> = Arc::clone(\&self.tx_results);/' src/vm/narwhal_bullshark_vm.rs
fi

echo "Fixing unused variables warnings..."
# Fix unused variables by prefixing them with underscores
sed -i 's/\bblockchain\b/_blockchain/g' src/consensus/pbft.rs
sed -i 's/\btx_data\b/_tx_data/g' src/consensus/narwhal_bullshark.rs
sed -i 's/\bhash: Bytes32\b/_hash: Bytes32/g' src/network/p2p.rs
sed -i 's/\btimestamp: u64\b/_timestamp: u64/g' src/network/p2p.rs
sed -i 's/\bheight: u64\b/_height: u64/g' src/network/p2p.rs
sed -i 's/\bconsensus_type: ConsensusType\b/_consensus_type: ConsensusType/g' src/network/p2p.rs
sed -i 's/\bcontract: Bytes32\b/_contract: Bytes32/g' src/network/p2p.rs
sed -i 's/\bproof: Bytes64\b/_proof: Bytes64/g' src/network/p2p.rs
sed -i 's/\bresources: ResourceUsage\b/_resources: ResourceUsage/g' src/network/p2p.rs
sed -i 's/\bnode_id: Bytes32\b/_node_id: Bytes32/g' src/network/p2p.rs
sed -i 's/\bstatus: NodeStatus\b/_status: NodeStatus/g' src/network/p2p.rs
sed -i 's/\bavailable_resources: AvailableResources\b/_available_resources: AvailableResources/g' src/network/p2p.rs
sed -i 's/\baction: ModelRegistryAction\b/_action: ModelRegistryAction/g' src/network/p2p.rs
sed -i 's/\bnarwhal: Arc<Narwhal>\b/_narwhal: Arc<Narwhal>/g' src/consensus/narwhal_bullshark.rs
sed -i 's/for (i, &idx) in/for (_i, &_idx) in/g' src/fault_tolerance/mod.rs
sed -i 's/let mut final_results/let final_results/g' src/fault_tolerance/mod.rs
sed -i 's/let mut succeeded_indices/let succeeded_indices/g' src/fault_tolerance/mod.rs
sed -i 's/\brecovery\b/_recovery/g' src/main.rs
sed -i 's/\bstate_db\b/_state_db/g' src/main.rs
sed -i 's/\bai_executor\b/_ai_executor/g' src/main.rs

echo "Removing unused imports..."
sed -i '/use std::sync::Arc;/d' src/config/mod.rs
sed -i '/use std::collections::HashMap;/d' src/config/mod.rs
sed -i '/use std::collections::{HashMap, HashSet};/c\use std::collections::HashMap;' src/consensus/narwhal_bullshark.rs
sed -i '/use std::time::Instant;/d' src/consensus/narwhal_bullshark.rs
sed -i '/use serde::{Serialize, Deserialize};/d' src/vm/mod.rs
sed -i '/use crate::state::StateDB;/d' src/vm/narwhal_bullshark_vm.rs
sed -i '/use crate::vm::VmError;/d' src/vm/ai/executor.rs

echo "All fixes have been applied!"
echo "Please run 'cargo build' to check if all issues have been resolved."
