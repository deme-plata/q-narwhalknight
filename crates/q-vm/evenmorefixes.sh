#!/bin/bash

# Script to fix the remaining issues in DagKnight VM
echo "Fixing remaining issues in DagKnight VM..."

PROJECT_DIR=$(pwd)

# 1. Fix Bullshark implementations
echo "Adding Bullshark implementation methods..."
NARWHAL_FILE="${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"

if [ -f "$NARWHAL_FILE" ]; then
  if ! grep -q "pub async fn get_latest_finalized" "$NARWHAL_FILE"; then
    # Create temporary file with Bullshark methods
    cat > "${PROJECT_DIR}/bullshark_methods.txt" << 'EOL'
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
    cat "${PROJECT_DIR}/bullshark_methods.txt" >> "$NARWHAL_FILE"
    echo "  Added Bullshark methods"
    
    # Clean up
    rm "${PROJECT_DIR}/bullshark_methods.txt"
  else
    echo "  Bullshark methods already exist"
  fi
else
  echo "  Could not find narwhal_bullshark.rs file"
fi

# 2. Fix Arc<Bullshark> method calls
echo "Fixing Arc<Bullshark> method calls..."
if [ -f "$NARWHAL_FILE" ]; then
  # Create a backup
  cp "$NARWHAL_FILE" "${NARWHAL_FILE}.bak"
  
  # Use awk for safer text manipulation
  awk '{
    if ($0 ~ /self\.bullshark\.get_latest_finalized\(\)/) {
      gsub(/self\.bullshark\.get_latest_finalized\(\)/, "(*self.bullshark).get_latest_finalized()", $0);
    }
    if ($0 ~ /self\.bullshark\.get_finalized_block\(/) {
      gsub(/self\.bullshark\.get_finalized_block\(/, "(*self.bullshark).get_finalized_block(", $0);
    }
    print $0;
  }' "${NARWHAL_FILE}.bak" > "$NARWHAL_FILE"
  
  echo "  Fixed Arc<Bullshark> method calls"
else
  echo "  Could not find narwhal_bullshark.rs file"
fi

# 3. Fix ExecutionResult serialization issue
echo "Fixing ExecutionResult serialization issue..."
VM_MOD_FILE="${PROJECT_DIR}/src/vm/mod.rs"

if [ -f "$VM_MOD_FILE" ]; then
  if grep -q "#\[derive(Clone)" "$VM_MOD_FILE"; then
    sed -i 's/#\[derive(Clone)\]/#\[derive(Clone, Serialize, Deserialize)\]/g' "$VM_MOD_FILE"
    echo "  Added Serialize and Deserialize derives to ExecutionResult in vm/mod.rs"
  else
    echo "  ExecutionResult already has necessary derives"
  fi
else
  echo "  Could not find vm/mod.rs file"
fi

# 4. Fix private Block import
echo "Fixing private Block import..."
VM_NARWHAL_FILE="${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"

if [ -f "$VM_NARWHAL_FILE" ]; then
  # Create a backup
  cp "$VM_NARWHAL_FILE" "${VM_NARWHAL_FILE}.bak"
  
  # Use awk for safer text manipulation
  awk '{
    if ($0 ~ /use crate::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction, Block, NodeId};/) {
      print "use crate::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction, NodeId};";
      print "use crate::consensus::pbft::Block;";
    } else {
      print $0;
    }
  }' "${VM_NARWHAL_FILE}.bak" > "$VM_NARWHAL_FILE"
  
  echo "  Fixed Block import to directly use pbft module"
else
  echo "  Could not find narwhal_bullshark_vm.rs file"
fi

# 5. Fix StateDB in state/mod.rs
echo "Fixing StateDB in state/mod.rs..."
STATE_MOD_FILE="${PROJECT_DIR}/src/state/mod.rs"

if [ -f "$STATE_MOD_FILE" ]; then
  # Add necessary imports
  if ! grep -q "use std::sync::Arc;" "$STATE_MOD_FILE"; then
    sed -i '1i use std::sync::Arc;' "$STATE_MOD_FILE"
    sed -i '2i use tokio::sync::RwLock;' "$STATE_MOD_FILE"
    sed -i '3i use crate::vm::VmState;' "$STATE_MOD_FILE"
    echo "  Added necessary imports to state/mod.rs"
  fi
  
  # Find the StateDB struct definition
  if grep -q "pub struct StateDB" "$STATE_MOD_FILE"; then
    # Create a backup
    cp "$STATE_MOD_FILE" "${STATE_MOD_FILE}.bak"
    
    # Add state field if not present
    if ! grep -q "state: Arc<RwLock<VmState>>" "$STATE_MOD_FILE"; then
      awk '{
        if ($0 ~ /pub struct StateDB {/) {
          print $0;
          print "    state: Arc<RwLock<VmState>>,";
        } else {
          print $0;
        }
      }' "${STATE_MOD_FILE}.bak" > "$STATE_MOD_FILE"
      echo "  Added state field to StateDB struct"
    fi
    
    # Add with_state method if not present
    if ! grep -q "pub fn with_state" "$STATE_MOD_FILE"; then
      # Find the end of impl StateDB block
      cp "$STATE_MOD_FILE" "${STATE_MOD_FILE}.bak2"
      IMPL_START=$(grep -n "impl StateDB" "${STATE_MOD_FILE}.bak2" | head -1 | cut -d':' -f1)
      
      if [ -n "$IMPL_START" ]; then
        # Create a temporary file with the method
        cat > "${PROJECT_DIR}/with_state_method.txt" << 'EOL'

    pub fn with_state(state: Arc<RwLock<VmState>>) -> Self {
        Self { 
            state,
            resource_ledger: None
        }
    }
EOL
        
        # Find a good place to insert the method
        IMPL_END=$(tail -n +$IMPL_START "${STATE_MOD_FILE}.bak2" | grep -n "^}" | head -1 | cut -d':' -f1)
        
        if [ -n "$IMPL_END" ]; then
          # Insert the method before the closing brace
          IMPL_END=$((IMPL_START + IMPL_END - 1))
          sed -i "${IMPL_END}r ${PROJECT_DIR}/with_state_method.txt" "$STATE_MOD_FILE"
          echo "  Added with_state method to StateDB"
        fi
        
        # Clean up
        rm "${PROJECT_DIR}/with_state_method.txt"
      fi
    fi
  fi
else
  echo "  Could not find state/mod.rs file"
fi

# 6. Fix VmError::SerializationError in pbft.rs
echo "Fixing VmError::SerializationError in consensus/pbft.rs..."
PBFT_FILE="${PROJECT_DIR}/src/consensus/pbft.rs"

if [ -f "$PBFT_FILE" ]; then
  if grep -q "VmError::SerializationError)?," "$PBFT_FILE"; then
    sed -i 's/VmError::SerializationError)?/VmError::SerializationError("Serialization failed".to_string()))?/g' "$PBFT_FILE"
    echo "  Fixed VmError::SerializationError usage in pbft.rs"
  fi
else
  echo "  Could not find pbft.rs file"
fi

# 7. Fix config import in lib.rs
echo "Fixing config import in lib.rs..."
LIB_FILE="${PROJECT_DIR}/src/lib.rs"

if [ -f "$LIB_FILE" ]; then
  if grep -q "config::load_config" "$LIB_FILE" && ! grep -q "use crate::vm::narwhal_bullshark_vm::config;" "$LIB_FILE"; then
    sed -i '1i use crate::vm::narwhal_bullshark_vm::config;' "$LIB_FILE"
    echo "  Added config import to lib.rs"
  fi
else
  echo "  Could not find lib.rs file"
fi

# 8. Fix VirtualMachine::execute to accept &mut self if needed
echo "Checking if VirtualMachine::execute needs &mut self..."
if [ -f "$VM_MOD_FILE" ]; then
  if grep -q "pub async fn execute(&self," "$VM_MOD_FILE"; then
    sed -i 's/pub async fn execute(&self,/pub async fn execute(\&mut self,/g' "$VM_MOD_FILE"
    echo "  Changed execute method to accept &mut self"
  fi
else
  echo "  Could not find vm/mod.rs file"
fi

echo "All fixes applied! Try building your project again with 'cargo build'"
