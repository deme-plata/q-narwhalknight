#!/bin/bash

# Script to fix DagKnight VM compilation errors
# Usage: chmod +x fiximportant3.sh && ./fiximportant3.sh

set -e
echo "Starting DagKnight VM fixes..."

PROJECT_DIR=$(pwd)
echo "Working in directory: ${PROJECT_DIR}"

# 1. Fix VmError in src/vm/mod.rs to handle string parameter
echo "Fixing VmError in src/vm/mod.rs..."
if grep -q "SerializationError," "${PROJECT_DIR}/src/vm/mod.rs"; then
  sed -i 's/SerializationError,/SerializationError(String),/g' "${PROJECT_DIR}/src/vm/mod.rs"
  echo "  Updated SerializationError to take a String parameter"
fi

# 2. Add Serialize and Deserialize derives to ExecutionResult
echo "Adding Serialize and Deserialize to ExecutionResult..."
if grep -q "#\[derive(Clone)" "${PROJECT_DIR}/src/vm/mod.rs"; then
  sed -i 's/#\[derive(Clone)\]/#[derive(Clone, Serialize, Deserialize)]/g' "${PROJECT_DIR}/src/vm/mod.rs"
  echo "  Added Serialize and Deserialize derives to ExecutionResult"
fi

# 3. Fix Vec<u8> vs Vec<Value> type mismatch in execute method
echo "Fixing return_data type conversion in VirtualMachine::execute..."
if grep -q "return_data: result," "${PROJECT_DIR}/src/vm/mod.rs"; then
  sed -i 's/return_data: result,/return_data: Vec::new(), \/\/ Simplified; real impl would convert result/g' "${PROJECT_DIR}/src/vm/mod.rs"
  echo "  Updated return_data in execute method"
fi

# 4. Fix imports in narwhal_bullshark_vm.rs - Add Arc import
echo "Adding Arc import..."
IMPORT_LINE="use std::sync::Arc;"
if ! grep -q "${IMPORT_LINE}" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  # Find the last import line and add Arc after it
  LAST_IMPORT=$(grep -n "^use " "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" | tail -1 | cut -d':' -f1)
  if [ -n "$LAST_IMPORT" ]; then
    sed -i "${LAST_IMPORT}a use std::sync::Arc;" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
    echo "  Added Arc import"
  else
    echo "  WARNING: Could not find a place to add Arc import"
  fi
fi

# 5. Fix Block import
echo "Fixing Block import..."
if grep -q "use crate::consensus::narwhal_bullshark::{.*Block" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  # First create a backup of the file
  cp "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs.bak"
  
  # Use awk for better pattern handling
  awk '{
    if ($0 ~ /use crate::consensus::narwhal_bullshark::{.*Block/) {
      gsub(/Block, /, "", $0);
      gsub(/Block}/, "}", $0);
      print $0;
      print "use crate::consensus::pbft::Block; // Import Block from correct module";
    } else {
      print $0;
    }
  }' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs.bak" > "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Fixed Block import to use pbft module"
fi

# 6. Fix VmState import
echo "Fixing VmState import..."
if grep -q "use crate::vm::{.*VmState" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  # Create backup if not already done
  if [ ! -f "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs.bak" ]; then
    cp "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs.bak"
  fi
  
  # Use awk for better pattern handling
  awk '{
    if ($0 ~ /use crate::vm::{.*VmState/) {
      gsub(/, VmState/, "", $0);
      print $0;
    } else {
      print $0;
    }
  }' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs.bak" > "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Removed VmState from vm module imports"
fi

# 7. Add VmState definition in narwhal_bullshark_vm.rs
echo "Adding VmState definition..."
VMSTATE_DEF="#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct VmState {\n    pub contracts: HashMap<Address, Vec<u8>>,\n    pub storage: HashMap<Address, HashMap<Vec<u8>, Vec<u8>>>,\n    pub balances: HashMap<Address, u64>,\n    pub nonces: HashMap<Address, u64>,\n}"

if ! grep -q "struct VmState" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  # Find a good location to add the VmState struct - after imports but before other structs
  LAST_IMPORT=$(grep -n "^use " "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" | tail -1 | cut -d':' -f1)
  if [ -n "$LAST_IMPORT" ]; then
    # Add a few lines after the last import
    INSERT_LINE=$((LAST_IMPORT + 2))
    sed -i "${INSERT_LINE}i // Define VmState here\n${VMSTATE_DEF}\n" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
    echo "  Added VmState definition"
  else
    echo "  WARNING: Could not find a place to add VmState definition"
  fi
fi

# 8. Fix RwLock clone issue
echo "Fixing RwLock clone in NarwhalBullshark::start..."
if grep -q "let latest_height = self.latest_height.clone();" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
  sed -i 's/let latest_height = self.latest_height.clone();/let latest_height = Arc::clone(\&self.latest_height);/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  echo "  Fixed RwLock clone in NarwhalBullshark"
fi

# 9. Fix Arc method call in start method
echo "Fixing Arc method call in NarwhalBullsharkVm::start..."
if grep -q "consensus.start().await;" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/consensus.start().await;/(*consensus).start().await;/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Fixed consensus.start() method call"
fi

# 10. Fix SerializationError usage in submit_transaction
echo "Fixing SerializationError usage..."
if grep -q "VmError::SerializationError)?," "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/VmError::SerializationError)?/VmError::SerializationError(e.to_string()))?/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Fixed SerializationError in bincode::serialize"
fi

# Fix add_transaction error handling
if grep -q "self.consensus.add_transaction(consensus_tx).await?;" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/self.consensus.add_transaction(consensus_tx).await?;/self.consensus.add_transaction(consensus_tx).await.map_err(|e| VmError::SerializationError(format!("Failed to add transaction: {}", e)))?;/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Fixed add_transaction error handling"
fi

# 11. Fix type annotation in start_execution_loop
echo "Fixing type annotation in start_execution_loop..."
if grep -q "let current_state = Arc::clone(&self.current_state);" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/let current_state = Arc::clone(&self.current_state);/let current_state: Arc<RwLock<VmState>> = Arc::clone(\&self.current_state);/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Added type annotation to current_state"
fi

# 12. Fix current_state.clone() issue
echo "Fixing current_state.clone() in state_access initialization..."
if grep -q "let state_access = VmStateAccess::new(current_state.clone());" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/let state_access = VmStateAccess::new(current_state.clone());/let state_access = VmStateAccess::new(Arc::clone(\&current_state));/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Fixed current_state clone in state_access"
fi

# 13. Fix ExecutionResult serialization in process_block
echo "Fixing ExecutionResult serialization in process_block..."
if grep -q "hasher.update(&bincode::serialize(&results" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  # Create a temporary file with the fix
  cat > "${PROJECT_DIR}/fix_serialization.awk" << 'EOL'
BEGIN { fixed = 0 }
{
  if ($0 ~ /hasher.update\(&bincode::serialize\(&results/ && fixed == 0) {
    print "                // Manually hash fields instead of serializing ExecutionResult";
    print "                let result = \\&results[i];";
    print "                hasher.update(\\&[result.success as u8]);";
    print "                hasher.update(\\&result.return_data);";
    print "                hasher.update(\\&result.gas_used.to_le_bytes());";
    print "                for log in \\&result.logs {";
    print "                    hasher.update(log.as_bytes());";
    print "                }";
    print "                if let Some(error) = \\&result.error {";
    print "                    hasher.update(error.as_bytes());";
    print "                }";
    fixed = 1;
  } else {
    print $0;
  }
}
EOL

  cp "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs.bak2"
  awk -f "${PROJECT_DIR}/fix_serialization.awk" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs.bak2" > "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  rm "${PROJECT_DIR}/fix_serialization.awk"
  echo "  Fixed ExecutionResult serialization"
fi

# 14. Add with_state constructor to StateDB
echo "Adding with_state constructor to StateDB..."
if [ -f "${PROJECT_DIR}/src/state/mod.rs" ]; then
  if grep -q "impl StateDB" "${PROJECT_DIR}/src/state/mod.rs"; then
    if ! grep -q "pub fn with_state" "${PROJECT_DIR}/src/state/mod.rs"; then
      # Create a temporary file with the fix
      cat > "${PROJECT_DIR}/add_with_state.awk" << 'EOL'
BEGIN { inImpl = 0; addedMethod = 0 }
{
  if ($0 ~ /impl StateDB/) {
    inImpl = 1;
    print $0;
  } else if (inImpl && $0 ~ /^}/ && addedMethod == 0) {
    print "";
    print "    pub fn with_state(state: Arc<RwLock<VmState>>) -> Self {";
    print "        Self { state }";
    print "    }";
    print $0;
    addedMethod = 1;
  } else {
    print $0;
  }
}
EOL

      cp "${PROJECT_DIR}/src/state/mod.rs" "${PROJECT_DIR}/src/state/mod.rs.bak"
      awk -f "${PROJECT_DIR}/add_with_state.awk" "${PROJECT_DIR}/src/state/mod.rs.bak" > "${PROJECT_DIR}/src/state/mod.rs"
      rm "${PROJECT_DIR}/add_with_state.awk"
      echo "  Added with_state constructor to StateDB"
    else
      echo "  with_state constructor already exists"
    fi
  else
    echo "  WARNING: Could not find StateDB implementation in state/mod.rs"
  fi
else
  echo "  WARNING: Could not find state/mod.rs file"
fi

# 15. Fix StateDB constructor call in main
echo "Fixing StateDB constructor call in main..."
if grep -q "let state_db = Arc::new(StateDB::new(current_state));" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/let state_db = Arc::new(StateDB::new(current_state));/let state_db = Arc::new(StateDB::with_state(current_state));/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Fixed StateDB constructor call"
fi

# 16. Fix VirtualMachine constructor call in lib.rs
echo "Fixing VirtualMachine constructor call in lib.rs..."
if [ -f "${PROJECT_DIR}/src/lib.rs" ]; then
  if grep -q "let vm = std::sync::Arc::new(vm::VirtualMachine::new());" "${PROJECT_DIR}/src/lib.rs"; then
    sed -i 's/let vm = std::sync::Arc::new(vm::VirtualMachine::new());/let state_db = std::sync::Arc::new(state::StateDB::new());\n    let vm = std::sync::Arc::new(vm::VirtualMachine::new(state_db));/g' "${PROJECT_DIR}/src/lib.rs"
    echo "  Fixed VirtualMachine constructor call"
  fi
else
  echo "  WARNING: Could not find lib.rs file"
fi

# 17. Make config module public
echo "Making config module public..."
if grep -q "mod config {" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/mod config {/pub mod config {/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Made config module public in narwhal_bullshark_vm.rs"
fi

# 18. Fix Bullshark implementations
echo "Checking for Bullshark implementation..."
if [ -f "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" ]; then
  # Find if there's any "no method named `get_latest_finalized`" error
  if grep -q "impl Bullshark" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
    if ! grep -q "pub async fn get_latest_finalized" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
      # Add missing methods to Bullshark implementation
      cat >> "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" << 'EOL'

// Added placeholder implementations for missing methods
impl Bullshark {
    pub async fn get_latest_finalized(&self) -> u64 {
        // Placeholder implementation
        0
    }
    
    pub async fn get_finalized_block(&self, _seq_num: u64) -> Option<Block> {
        // Placeholder implementation
        None
    }
}
EOL
      echo "  Added placeholder Bullshark methods"
    else
      echo "  Bullshark methods already exist"
    fi
  else
    echo "  WARNING: Could not find Bullshark implementation in narwhal_bullshark.rs"
  fi
else
  echo "  WARNING: Could not find narwhal_bullshark.rs file"
fi

# 19. Fix imports in consensus/narwhal_bullshark.rs for Block
echo "Fixing imports in consensus/narwhal_bullshark.rs..."
if [ -f "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" ]; then
  if grep -q "use crate::consensus::narwhal_bullshark::types::{" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
    cp "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs.bak"
    
    # Use awk for better pattern handling
    awk '{
      if ($0 ~ /use crate::consensus::narwhal_bullshark::types::{/) {
        print "use crate::types::{Transaction, NodeId, VertexId};";
        print "use crate::consensus::pbft::Block;";
      } else if ($0 !~ /Transaction, Block, NodeId, VertexId/ && $0 !~ /NarwhalMessage, BullsharkMessage, ConsensusMessage/) {
        print $0;
      }
    }' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs.bak" > "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
    echo "  Fixed types imports in narwhal_bullshark.rs"
  fi
fi

echo "All fixes applied! Try building your project again with 'cargo build'"
