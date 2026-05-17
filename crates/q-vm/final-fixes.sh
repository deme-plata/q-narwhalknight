#!/bin/bash

# Final fixes for DagKnight VM
echo "Applying final fixes for DagKnight VM..."

PROJECT_DIR=$(pwd)

# 1. Fix outer doc comment in state/mod.rs
echo "Fixing doc comment in state/mod.rs..."
if [ -f "${PROJECT_DIR}/src/state/mod.rs" ]; then
  sed -i 's/\/\/! State management for DAGKnight/\/\/\/ State management for DAGKnight/g' "${PROJECT_DIR}/src/state/mod.rs"
  echo "  Fixed doc comment style"
fi

# 2. Fix duplicate RwLock import in state/mod.rs
echo "Fixing duplicate RwLock import..."
if [ -f "${PROJECT_DIR}/src/state/mod.rs" ]; then
  LINE_NUM=$(grep -n "use tokio::sync::RwLock;" "${PROJECT_DIR}/src/state/mod.rs" | tail -1 | cut -d':' -f1)
  if [ -n "$LINE_NUM" ]; then
    sed -i "${LINE_NUM}d" "${PROJECT_DIR}/src/state/mod.rs"
    echo "  Removed duplicate RwLock import"
  fi
fi

# 3. Define Transaction and NodeId in narwhal_bullshark module
echo "Defining Transaction and NodeId in narwhal_bullshark module..."
if [ -f "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" ]; then
  # Remove the problematic imports
  sed -i '/use crate::consensus::narwhal_bullshark::Transaction;/d' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  sed -i '/use crate::consensus::narwhal_bullshark::NodeId;/d' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  
  # Add proper type definitions at the top of the file
  TYPE_DEFS="
// Type definitions
pub type NodeId = String;

#[derive(Clone, Debug, Default)]
pub struct Transaction {
    pub hash: [u8; 32],
    pub data: Vec<u8>,
    pub sender: [u8; 32],
    pub nonce: u64,
    pub signature: [u8; 64],
    pub timestamp: u64,
}
"
  # Insert after the imports section
  LAST_IMPORT=$(grep -n "^use " "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" | tail -1 | cut -d':' -f1)
  if [ -n "$LAST_IMPORT" ]; then
    INSERT_LINE=$((LAST_IMPORT + 1))
    sed -i "${INSERT_LINE}i${TYPE_DEFS}" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
    echo "  Added Transaction and NodeId definitions"
  fi
fi

# 4. Fix imports in narwhal_bullshark_vm.rs
echo "Fixing imports in narwhal_bullshark_vm.rs..."
if [ -f "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" ]; then
  # Fix imports
  sed -i 's/use crate::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction, NodeId};/use crate::consensus::narwhal_bullshark::NarwhalBullshark;\nuse std::sync::Arc;/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  
  # Add Arc import if needed
  if ! grep -q "use std::sync::Arc;" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
    sed -i '1i use std::sync::Arc;' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  fi
  
  # Define VmState type at the top if not already there
  if ! grep -q "struct VmState" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
    VMSTATE_DEF="
// Define our own types to avoid circular dependency issues
type NodeId = String;
type Transaction = consensus::narwhal_bullshark::Transaction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmState {
    pub contracts: HashMap<Address, Vec<u8>>,
    pub storage: HashMap<Address, HashMap<Vec<u8>, Vec<u8>>>,
    pub balances: HashMap<Address, u64>,
    pub nonces: HashMap<Address, u64>,
}
"
    # Find a good place to insert
    IMPORTS_END=$(grep -n "^use " "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" | tail -1 | cut -d':' -f1)
    if [ -n "$IMPORTS_END" ]; then
      INSERT_LINE=$((IMPORTS_END + 1))
      sed -i "${INSERT_LINE}i${VMSTATE_DEF}" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
      echo "  Added VmState definition"
    fi
  fi
  
  # Fix ExecutionResult import 
  sed -i 's/use crate::vm::{VirtualMachine, VmError, ExecutionResult, ContractState, CallData, StateAccess, Address, StateDB, VmState};/use crate::vm::{VirtualMachine, VmError, ExecutionResult, ContractState, CallData, StateAccess, Address, StateDB};/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Fixed imports"
fi

# 5. Add start method to NarwhalBullshark
echo "Adding start method to NarwhalBullshark..."
if [ -f "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" ]; then
  if ! grep -q "pub async fn start(&self)" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
    # Find the impl NarwhalBullshark block
    if grep -q "impl NarwhalBullshark" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
      # Create temporary file with the start method
      cat > "${PROJECT_DIR}/start_method.txt" << 'EOL'
    pub async fn start(&self) {
        println!("Starting NarwhalBullshark consensus...");
        // Placeholder implementation
    }
EOL
      # Find where to insert it
      IMPL_LINE=$(grep -n "impl NarwhalBullshark" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" | head -1 | cut -d':' -f1)
      if [ -n "$IMPL_LINE" ]; then
        # Insert after the opening brace of the impl block
        BRACE_LINE=$((IMPL_LINE + 1))
        sed -i "${BRACE_LINE}r ${PROJECT_DIR}/start_method.txt" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
        echo "  Added start method to NarwhalBullshark"
      fi
      # Clean up
      rm "${PROJECT_DIR}/start_method.txt"
    fi
  fi
fi

# 6. Add get_latest_finalized and get_finalized_block methods to Bullshark
echo "Adding necessary methods to Bullshark..."
if [ -f "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" ]; then
  # Check if Bullshark struct exists
  if grep -q "pub struct Bullshark" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
    if ! grep -q "pub async fn get_latest_finalized(&self)" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
      # Create temporary file with the methods
      cat > "${PROJECT_DIR}/bullshark_methods.txt" << 'EOL'

impl Bullshark {
    pub async fn get_latest_finalized(&self) -> u64 {
        // Placeholder implementation
        0
    }
    
    pub async fn get_finalized_block(&self, _seq_num: u64) -> Option<crate::consensus::pbft::Block> {
        // Placeholder implementation
        None
    }
}
EOL
      # Append to the file
      cat "${PROJECT_DIR}/bullshark_methods.txt" >> "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
      echo "  Added methods to Bullshark"
      # Clean up
      rm "${PROJECT_DIR}/bullshark_methods.txt"
    fi
  else
    # Create Bullshark struct and implementation
    cat > "${PROJECT_DIR}/bullshark_struct.txt" << 'EOL'

// Bullshark consensus implementation
pub struct Bullshark {
    node_id: NodeId,
}

impl Bullshark {
    pub async fn get_latest_finalized(&self) -> u64 {
        // Placeholder implementation
        0
    }
    
    pub async fn get_finalized_block(&self, _seq_num: u64) -> Option<crate::consensus::pbft::Block> {
        // Placeholder implementation
        None
    }
}
EOL
    # Append to the file
    cat "${PROJECT_DIR}/bullshark_struct.txt" >> "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
    echo "  Added Bullshark struct and implementation"
    # Clean up
    rm "${PROJECT_DIR}/bullshark_struct.txt"
  fi
fi

# 7. Fix VmState not found in mod.rs and RwLock issues
echo "Fixing VmState and StateDB in state/mod.rs..."
if [ -f "${PROJECT_DIR}/src/state/mod.rs" ]; then
  # Remove problematic import
  sed -i '/use crate::vm::VmState;/d' "${PROJECT_DIR}/src/state/mod.rs"
  
  # Define VmState directly in state/mod.rs
  VMSTATE_DEF="
// Define VmState locally to avoid circular dependencies
type Address = u64;

#[derive(Debug, Clone, Default)]
pub struct VmState {
    pub contracts: std::collections::HashMap<Address, Vec<u8>>,
    pub storage: std::collections::HashMap<Address, std::collections::HashMap<Vec<u8>, Vec<u8>>>,
    pub balances: std::collections::HashMap<Address, u64>,
    pub nonces: std::collections::HashMap<Address, u64>,
}
"
  # Find where to insert
  sed -i '5i\'"${VMSTATE_DEF}" "${PROJECT_DIR}/src/state/mod.rs"
  
  # Fix StateDB struct to include state field
  if grep -q "pub struct StateDB" "${PROJECT_DIR}/src/state/mod.rs"; then
    # Create a backup
    cp "${PROJECT_DIR}/src/state/mod.rs" "${PROJECT_DIR}/src/state/mod.rs.bak"
    
    # Update the struct definition
    awk '{
      if ($0 ~ /pub struct StateDB {/) {
        print $0;
        print "    pub state: Arc<RwLock<VmState>>,";
      } else {
        print $0;
      }
    }' "${PROJECT_DIR}/src/state/mod.rs.bak" > "${PROJECT_DIR}/src/state/mod.rs"
    
    # Fix the with_state method
    sed -i 's/Self { state, resource_ledger: None }/Self { state, resource_ledger: None }/g' "${PROJECT_DIR}/src/state/mod.rs"
    echo "  Updated StateDB struct and implementation"
  fi
fi

# 8. Fix ExecutionResult serialization issue
echo "Fixing ExecutionResult serialization..."
if [ -f "${PROJECT_DIR}/src/vm/mod.rs" ]; then
  if grep -q "pub struct ExecutionResult" "${PROJECT_DIR}/src/vm/mod.rs"; then
    # Make sure it has Serialize and Deserialize derives
    sed -i 's/#\[derive(Clone)\]/#\[derive(Clone, Serialize, Deserialize)\]/g' "${PROJECT_DIR}/src/vm/mod.rs"
    echo "  Added Serialize/Deserialize to ExecutionResult"
  fi
fi

# 9. Fix VmBlockResult to use VmState's ExecutionResult
echo "Fixing VmBlockResult to use local ExecutionResult..."
if [ -f "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" ]; then
  # Define local ExecutionResult type
  LOCAL_EXEC_RESULT="
// Local version of ExecutionResult for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalExecutionResult {
    pub success: bool,
    pub return_data: Vec<u8>,
    pub gas_used: u64,
    pub logs: Vec<String>,
    pub error: Option<String>,
}

// Helper function to convert from vm::ExecutionResult
fn to_local_result(result: &ExecutionResult) -> LocalExecutionResult {
    LocalExecutionResult {
        success: result.success,
        return_data: result.return_data.clone(),
        gas_used: result.gas_used,
        logs: result.logs.clone(),
        error: result.error.clone(),
    }
}
"
  # Insert before VmBlockResult
  if grep -q "pub struct VmBlockResult" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
    VBR_LINE=$(grep -n "pub struct VmBlockResult" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" | head -1 | cut -d':' -f1)
    if [ -n "$VBR_LINE" ]; then
      VBR_LINE=$((VBR_LINE - 1))
      sed -i "${VBR_LINE}i${LOCAL_EXEC_RESULT}" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
      
      # Update VmBlockResult to use LocalExecutionResult
      sed -i 's/pub tx_results: Vec<ExecutionResult>,/pub tx_results: Vec<LocalExecutionResult>,/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
      
      # Fix process_block function to convert ExecutionResult to LocalExecutionResult
      if grep -q "results.push(result.clone());" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
        sed -i 's/results.push(result.clone());/results.push(to_local_result(\&result));/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
      fi
      
      echo "  Updated VmBlockResult to use LocalExecutionResult"
    fi
  fi
fi

echo "All final fixes applied! Try building your project again with 'cargo build'"
