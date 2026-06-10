#!/bin/bash

# Final import and model fixes
echo "Applying final import and model fixes..."

PROJECT_DIR=$(pwd)

# 1. Fix the types module imports
echo "Fixing types module imports..."
# In vm/mod.rs
sed -i 's/pub use crate::types/pub use super::types/g' "${PROJECT_DIR}/src/vm/mod.rs"

# In state/mod.rs
sed -i 's/use crate::types::VmState;/use crate::vm::VmState;/g' "${PROJECT_DIR}/src/state/mod.rs"

# In narwhal_bullshark_vm.rs
sed -i 's/use crate::types::{NodeId, Transaction, VmState, Address, ExecutionResult as CommonExecutionResult};/use crate::vm::{NodeId, Transaction, VmState, Address, ExecutionResult as CommonExecutionResult};/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"

echo "  Fixed types module imports"

# 2. Fix models/mod.rs issues
echo "Fixing models/mod.rs issues..."
MODELS_FILE="${PROJECT_DIR}/src/models/mod.rs"

if [ -f "$MODELS_FILE" ]; then
  # Fix ModelInfo.performance.avg_gpu_usage_mb (Option<u64> vs u64)
  sed -i 's/avg_gpu_usage_mb: registration.resources.min_gpu_memory_mb,/avg_gpu_usage_mb: registration.resources.min_gpu_memory_mb,/g' "$MODELS_FILE"
  
  # Fix is_some() on u64
  sed -i 's/m.registration.resources.min_gpu_memory_mb.is_some()/m.registration.resources.min_gpu_memory_mb > 0/g' "$MODELS_FILE"
  
  # Fix capability.clone() in match statement
  sed -i 's/match (capability, /match (capability.clone(), /g' "$MODELS_FILE"
  
  # Fix all the ModelRegistration missing fields
  # Find all ModelRegistration { without hash,owner,timestamp
  grep -n "ModelRegistration {" "$MODELS_FILE" | while read -r line; do
    LINE_NUM=$(echo "$line" | cut -d':' -f1)
    # Check if it already has hash, owner, timestamp
    NEXT_LINE=$((LINE_NUM + 1))
    if ! grep -A 3 -n "hash:" "$MODELS_FILE" | grep -q "^$NEXT_LINE:"; then
      # Insert the missing fields
      sed -i "${NEXT_LINE}i\\                hash: [0; 32],\\n                owner: \"system\".to_string(),\\n                timestamp: 0," "$MODELS_FILE"
    fi
  done
  
  # Fix ResourceRequirements missing fields
  grep -n "ResourceRequirements {" "$MODELS_FILE" | while read -r line; do
    LINE_NUM=$(echo "$line" | cut -d':' -f1)
    # Check if it already has preferred_batch_size
    LAST_LINE=$(grep -A 5 -n "min_gpu_memory_mb:" "$MODELS_FILE" | grep -n "}" | head -1 | cut -d':' -f2 | cut -d':' -f1)
    if [ -n "$LAST_LINE" ]; then
      LAST_LINE=$((LAST_LINE - 1))
      # Insert the missing field
      sed -i "${LAST_LINE}a\\                    preferred_batch_size: 16," "$MODELS_FILE"
    fi
  done
  
  # Fix Option<u64> to u64 for min_gpu_memory_mb
  sed -i 's/min_gpu_memory_mb: Some(/min_gpu_memory_mb: /g' "$MODELS_FILE"
  sed -i 's/),/,/g' "$MODELS_FILE"
  
  echo "  Fixed models/mod.rs issues"
else
  echo "  Could not find models/mod.rs file"
fi

# 3. Fix dyn std::any::Any in fault_tolerance/mod.rs
echo "Fixing fault_tolerance/mod.rs issues..."
FAULT_FILE="${PROJECT_DIR}/src/fault_tolerance/mod.rs"

if [ -f "$FAULT_FILE" ]; then
  sed -i 's/std::any::Any::downcast_ref/(&dyn std::any::Any)::downcast_ref/g' "$FAULT_FILE"
  echo "  Fixed dyn std::any::Any issue"
else
  echo "  Could not find fault_tolerance/mod.rs file"
fi

# 4. Fix ModelRegistration struct in contracts/mod.rs
echo "Fixing contracts/mod.rs for ModelRegistration..."
CONTRACTS_FILE="${PROJECT_DIR}/src/contracts/mod.rs"

if [ -f "$CONTRACTS_FILE" ]; then
  # Add ModelRegistration and ResourceRequirements structs if not already present
  if ! grep -q "pub struct ModelRegistration" "$CONTRACTS_FILE"; then
    cat >> "$CONTRACTS_FILE" << 'EOF'

#[derive(Debug, Clone)]
pub struct ModelRegistration {
    pub model_id: String,
    pub description: String,
    pub version: String,
    pub capabilities: ShardingCapability,
    pub resources: ResourceRequirements,
    pub hash: [u8; 32],
    pub owner: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub min_cpu_cores: u32,
    pub min_memory_mb: u64,
    pub min_gpu_memory_mb: u64,
    pub preferred_batch_size: u32,
}
EOF
    echo "  Added ModelRegistration and ResourceRequirements to contracts/mod.rs"
  else
    echo "  ModelRegistration already exists in contracts/mod.rs"
  fi
else
  echo "  Could not find contracts/mod.rs file"
fi

echo "All final import and model fixes applied! Try building your project again with 'cargo build'"
