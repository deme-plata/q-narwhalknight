#!/bin/bash

# Comprehensive script to fix DagKnight VM compilation errors
# Usage: chmod +x fix-dagknight-complete.sh && ./fix-dagknight-complete.sh

set -e
echo "Starting comprehensive DagKnight VM fixes..."

PROJECT_DIR=$(pwd)
echo "Working in directory: ${PROJECT_DIR}"

# 1. Fix VmState visibility and imports
echo "Fixing VmState visibility and imports..."

# Add VmState import to narwhal_bullshark_vm.rs
if grep -q "use crate::consensus::narwhal_bullshark::{NarwhalBullshark" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/use crate::vm::{VirtualMachine, VmError, ExecutionResult, ContractState, CallData, StateAccess, Address, StateDB};/use crate::vm::{VirtualMachine, VmError, ExecutionResult, ContractState, CallData, StateAccess, Address, StateDB, VmState};/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Added VmState import in narwhal_bullshark_vm.rs"
fi

# 2. Fix Arc import in narwhal_bullshark.rs (consensus module)
echo "Fixing Arc import in consensus/narwhal_bullshark.rs..."
if grep -q "use std::sync::Arc;" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
  echo "  Arc import already exists in narwhal_bullshark.rs"
else
  # Add Arc import after the first use statement
  sed -i '1s/^/use std::sync::Arc;\n/' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  echo "  Added Arc import to narwhal_bullshark.rs"
fi

# 3. Fix crate::types import
echo "Fixing crate::types import in consensus/narwhal_bullshark.rs..."
if grep -q "use crate::types" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
  # Replace with appropriate imports
  sed -i 's/use crate::types::{Transaction, NodeId, VertexId};/use crate::consensus::narwhal_bullshark::Transaction;\nuse crate::consensus::narwhal_bullshark::NodeId;/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  echo "  Fixed types import"
fi

# 4. Fix missing struct types in Narwhal and NarwhalBullshark
echo "Fixing missing types in Narwhal and NarwhalBullshark..."
if grep -q "vertices: DashMap<VertexId, Vertex>," "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
  # Create the missing types
  cat > "${PROJECT_DIR}/src/consensus/narwhal_bullshark/types.rs" << 'EOL'
#[derive(Clone, Debug)]
pub struct VertexId(pub [u8; 32]);

#[derive(Clone, Debug)]
pub struct Vertex {
    pub id: VertexId,
    pub round: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum NarwhalMessage {
    Vertex(Vertex),
    Sync(u64),
}

#[derive(Clone, Debug)]
pub enum ConsensusMessage {
    Propose(u64),
    Vote(u64, [u8; 32]),
}
EOL
  
  echo "  Created missing type definitions"
  
  # Fix struct definitions
  sed -i 's/pub struct Narwhal {/pub struct Narwhal {\n    \/* Added placeholder for missing Vertex type *\//g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  sed -i 's/vertices: DashMap<VertexId, Vertex>,/vertices: DashMap<u64, Vec<u8>>, \/\/ Modified to use available types/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  sed -i 's/tx_network: mpsc::Sender<(NodeId, NarwhalMessage)>,/tx_network: mpsc::Sender<(NodeId, Vec<u8>)>, \/\/ Modified to use Vec<u8> instead of NarwhalMessage/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  sed -i 's/pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> (Self, mpsc::Receiver<(NodeId, NarwhalMessage)>) {/pub fn new(node_id: NodeId, peers: Vec<NodeId>) -> (Self, mpsc::Receiver<(NodeId, Vec<u8>)>) {/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  
  sed -i 's/tx_network: mpsc::Sender<(NodeId, ConsensusMessage)>,/tx_network: mpsc::Sender<(NodeId, Vec<u8>)>, \/\/ Modified to use Vec<u8> instead of ConsensusMessage/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  sed -i 's/rx_narwhal: mpsc::Receiver<(NodeId, NarwhalMessage)>,/rx_narwhal: mpsc::Receiver<(NodeId, Vec<u8>)>, \/\/ Modified to use Vec<u8> instead of NarwhalMessage/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  
  echo "  Modified struct definitions to use available types"
fi

# 5. Add start() method to NarwhalBullshark
echo "Adding start() method to NarwhalBullshark..."
if ! grep -q "pub async fn start(&self)" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
  # Find the end of NarwhalBullshark implementation
  START_METHOD="
    pub async fn start(&self) {
        // Placeholder implementation to allow compilation
        println!(\"Starting NarwhalBullshark consensus on node {}\", self.node_id);
        
        // Actual implementation would spawn tasks for processing
        // and start the consensus algorithm
    }
"
  # Append the method to the NarwhalBullshark impl
  sed -i "/impl NarwhalBullshark {/a\\${START_METHOD}" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  echo "  Added start() method to NarwhalBullshark"
fi

# 6. Add Bullshark implementations
echo "Adding Bullshark implementation methods..."
if grep -q "no method named \`get_latest_finalized\` found for struct" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs" || ! grep -q "pub async fn get_latest_finalized(&self)" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
  # Find appropriate place to add Bullshark implementation
  if grep -q "impl Bullshark {" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
    # Add methods to existing impl block
    BULLSHARK_METHODS="
    pub async fn get_latest_finalized(&self) -> u64 {
        // Placeholder implementation
        0
    }
    
    pub async fn get_finalized_block(&self, _seq_num: u64) -> Option<crate::consensus::pbft::Block> {
        // Placeholder implementation
        None
    }
"
    sed -i "/impl Bullshark {/a\\${BULLSHARK_METHODS}" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  else
    # Create new impl block
    BULLSHARK_IMPL="
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
"
    echo "${BULLSHARK_IMPL}" >> "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  fi
  echo "  Added Bullshark methods"
fi

# 7. Fix Arc<Bullshark> methods
echo "Fixing Arc<Bullshark> method calls..."
if grep -q "self.bullshark.get_latest_finalized()" "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"; then
  sed -i 's/self.bullshark.get_latest_finalized()/(\*self.bullshark).get_latest_finalized()/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  sed -i 's/self.bullshark.get_finalized_block(/(\*self.bullshark).get_finalized_block(/g' "${PROJECT_DIR}/src/consensus/narwhal_bullshark.rs"
  echo "  Fixed Arc<Bullshark> method calls"
fi

# 8. Fix ExecutionResult serialization issue
echo "Fixing ExecutionResult serialization issue..."
if grep -q "#\[derive(Clone)" "${PROJECT_DIR}/src/vm/mod.rs"; then
  sed -i 's/#\[derive(Clone)\]/#\[derive(Clone, Serialize, Deserialize)\]/g' "${PROJECT_DIR}/src/vm/mod.rs"
  echo "  Added Serialize and Deserialize derives to ExecutionResult in vm/mod.rs"
fi

# 9. Fix private Block import
echo "Fixing private Block import..."
if grep -q "use crate::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction, Block, NodeId};" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  sed -i 's/use crate::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction, Block, NodeId};/use crate::consensus::narwhal_bullshark::{NarwhalBullshark, Transaction, NodeId};\nuse crate::consensus::pbft::Block;/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
  echo "  Fixed Block import to directly use pbft module"
fi

# 10. Fix StateDB in state/mod.rs
echo "Fixing StateDB in state/mod.rs..."
if [ -f "${PROJECT_DIR}/src/state/mod.rs" ]; then
  # Add Arc and RwLock import if needed
  if ! grep -q "use std::sync::Arc;" "${PROJECT_DIR}/src/state/mod.rs"; then
    sed -i '1s/^/use std::sync::Arc;\nuse tokio::sync::RwLock;\n/' "${PROJECT_DIR}/src/state/mod.rs"
    echo "  Added Arc and RwLock imports to state/mod.rs"
  fi
  
  # Add VmState import if needed
  if ! grep -q "use crate::vm::VmState;" "${PROJECT_DIR}/src/state/mod.rs"; then
    sed -i '3s/^/use crate::vm::VmState;\n/' "${PROJECT_DIR}/src/state/mod.rs"
    echo "  Added VmState import to state/mod.rs"
  fi
  
  # Fix StateDB struct and with_state implementation
  if grep -q "pub fn with_state" "${PROJECT_DIR}/src/state/mod.rs"; then
    # Check if StateDB has state field
    if ! grep -q "state: Arc<RwLock<VmState>>" "${PROJECT_DIR}/src/state/mod.rs"; then
      # Add state field to StateDB struct
      sed -i '/pub struct StateDB {/a\\    state: Arc<RwLock<VmState>>,\\n    // Original fields below' "${PROJECT_DIR}/src/state/mod.rs"
      echo "  Added state field to StateDB struct"
    fi
    
    # Fix with_state implementation
    sed -i 's/Self { state }/Self { state, resource_ledger: None }/g' "${PROJECT_DIR}/src/state/mod.rs"
    echo "  Fixed with_state implementation in StateDB"
  else
    # Add with_state method
    WITH_STATE_METHOD="
    pub fn with_state(state: Arc<RwLock<VmState>>) -> Self {
        Self { 
            state,
            resource_ledger: None
        }
    }
"
    # Find end of impl StateDB block
    IMPL_LINE=$(grep -n "impl StateDB" "${PROJECT_DIR}/src/state/mod.rs" | head -1 | cut -d':' -f1)
    if [ -n "$IMPL_LINE" ]; then
      # Find the closing brace of the impl block
      END_LINE=$(tail -n +$IMPL_LINE "${PROJECT_DIR}/src/state/mod.rs" | grep -n "^}" | head -1 | cut -d':' -f1)
      if [ -n "$END_LINE" ]; then
        END_LINE=$((IMPL_LINE + END_LINE - 1))
        # Insert with_state method before the closing brace
        sed -i "${END_LINE}i\\${WITH_STATE_METHOD}" "${PROJECT_DIR}/src/state/mod.rs"
        echo "  Added with_state method to StateDB"
      fi
    fi
  fi
fi

# 11. Fix VmError::SerializationError in pbft.rs
echo "Fixing VmError::SerializationError in consensus/pbft.rs..."
if grep -q "VmError::SerializationError" "${PROJECT_DIR}/src/consensus/pbft.rs"; then
  sed -i 's/VmError::SerializationError/VmError::SerializationError(e.to_string())/g' "${PROJECT_DIR}/src/consensus/pbft.rs"
  echo "  Fixed VmError::SerializationError usage in pbft.rs"
fi

# 12. Fix config import in lib.rs
echo "Fixing config import in lib.rs..."
if [ -f "${PROJECT_DIR}/src/lib.rs" ]; then
  if grep -q "config::load_config" "${PROJECT_DIR}/src/lib.rs"; then
    # Add import at the top of lib.rs
    sed -i '1s/^/use crate::vm::narwhal_bullshark_vm::config;\n/' "${PROJECT_DIR}/src/lib.rs"
    echo "  Added config import to lib.rs"
  fi
fi

# 13. Fix VirtualMachine::execute to accept &mut self if needed
echo "Checking if VirtualMachine::execute needs &mut self..."
if grep -q "cannot borrow \`self.executor\` as mutable" "${PROJECT_DIR}/src/vm/mod.rs"; then
  sed -i 's/pub async fn execute(&self, call_data: &CallData, state_access: &dyn StateAccess)/pub async fn execute(\&mut self, call_data: \&CallData, state_access: \&dyn StateAccess)/g' "${PROJECT_DIR}/src/vm/mod.rs"
  echo "  Changed execute method to accept &mut self"
fi

echo "All fixes applied! Try building your project again with 'cargo build'"
