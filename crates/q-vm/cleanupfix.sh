#!/bin/bash

# Cleanup script for remaining issues
echo "Cleaning up remaining issues..."

PROJECT_DIR=$(pwd)

# 1. Fix HashMap import in narwhal_bullshark_vm.rs
echo "Fixing HashMap import in narwhal_bullshark_vm.rs..."
if grep -q "use std::collections::HashMap;" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  echo "  HashMap already imported"
else
  sed -i '1s/^/use std::collections::HashMap;\n/' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Added HashMap import"
fi

# 2. Create stub modules for missing imports
echo "Creating stub modules for missing imports..."

# Create contracts module
mkdir -p "${PROJECT_DIR}/src/contracts"
cat > "${PROJECT_DIR}/src/contracts/mod.rs" << 'EOF'
// Stub contracts module to satisfy imports

#[derive(Debug, Clone)]
pub struct ContractCall {
    pub contract_id: String,
    pub function: String,
    pub args: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct AIModelCall {
    pub model_id: String,
    pub input: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum ShardingCapability {
    None,
    DataParallel,
    ModelParallel,
}
EOF
echo "  Created contracts module"

# Add ResourceUsage to state/mod.rs
echo "Adding ResourceUsage to state/mod.rs..."
cat >> "${PROJECT_DIR}/src/state/mod.rs" << 'EOF'

// Add ResourceUsage for AI executor
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub compute_units: u64,
    pub memory_bytes: u64,
    pub storage_bytes: u64,
}
EOF
echo "  Added ResourceUsage to state/mod.rs"

# Add cache module to vm
mkdir -p "${PROJECT_DIR}/src/vm/cache"
cat > "${PROJECT_DIR}/src/vm/cache/mod.rs" << 'EOF'
// Stub contract cache module

use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;

#[derive(Debug)]
pub struct ContractCache {
    contracts: RwLock<HashMap<String, Vec<u8>>>,
}

impl ContractCache {
    pub fn new() -> Self {
        Self {
            contracts: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.contracts.read().get(key).cloned()
    }

    pub fn insert(&self, key: String, value: Vec<u8>) {
        self.contracts.write().insert(key, value);
    }
}
EOF
echo "  Created vm/cache module"

# Update vm/mod.rs to include ConsensusEngine and cache module
echo "Updating vm/mod.rs with ConsensusEngine and cache module..."
cat >> "${PROJECT_DIR}/src/vm/mod.rs" << 'EOF'

// ConsensusEngine trait for PBFT
#[async_trait]
pub trait ConsensusEngine: Send + Sync {
    async fn validate_block(&self, block: &[u8]) -> Result<bool, VmError>;
    async fn finalize_block(&self, block: &[u8]) -> Result<(), VmError>;
    async fn get_latest_block(&self) -> Result<Vec<u8>, VmError>;
}

// Include the cache module
pub mod cache;
EOF
echo "  Updated vm/mod.rs"

# Update lib.rs to expose contracts module
echo "Updating lib.rs to expose contracts module..."
if grep -q "pub mod contracts" "${PROJECT_DIR}/src/lib.rs"; then
  echo "  contracts module already exposed in lib.rs"
else
  # Insert after the first pub mod line
  sed -i '0,/pub mod/s//pub mod contracts;\npub mod/' "${PROJECT_DIR}/src/lib.rs"
  echo "  Exposed contracts module in lib.rs"
fi

# Make sure the DashMap is imported in VmStateAccess
if grep -q "std::collections::DashMap" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  echo "  DashMap already imported"
else
  sed -i 's/use dashmap::DashMap;/use dashmap::DashMap;\nuse std::collections::HashMap;/' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Added HashMap import along with DashMap"
fi

echo "All cleanup complete! Try building your project again with 'cargo build'"
