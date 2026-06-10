#!/bin/bash

# Final fixes for DagKnight VM
echo "Applying final fixes for DagKnight VM..."

PROJECT_DIR=$(pwd)

# 1. Fix duplicate HashMap import
echo "Fixing duplicate HashMap import..."
if grep -q "use std::collections::HashMap;" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"; then
  # Keep only the first occurrence
  FIRST_LINE=$(grep -n "use std::collections::HashMap;" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" | head -1 | cut -d':' -f1)
  SECOND_LINE=$(grep -n "use std::collections::HashMap;" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" | tail -1 | cut -d':' -f1)
  if [ "$FIRST_LINE" != "$SECOND_LINE" ]; then
    sed -i "${SECOND_LINE}d" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
    echo "  Removed duplicate HashMap import"
  fi
fi

# 2. Fix the ConsensusEngine trait
echo "Fixing ConsensusEngine trait..."
cat > "${PROJECT_DIR}/temp_ConsensusEngine.txt" << 'EOF'
// ConsensusEngine trait for PBFT
#[async_trait]
pub trait ConsensusEngine: Send + Sync {
    async fn validate_block(&self, block: &[u8]) -> Result<bool, VmError>;
    async fn finalize_block(&self, block: &[u8]) -> Result<(), VmError>;
    async fn get_latest_block(&self) -> Result<Vec<u8>, VmError>;
    
    // Additional methods needed by PBFT
    async fn validate_contract(&self, hash: [u8; 32], bytecode: &[u8]) -> Result<(), VmError>;
    async fn broadcast_contract(&self, hash: [u8; 32], bytecode: Vec<u8>) -> Result<(), VmError>;
}
EOF

# Update vm/mod.rs
sed -i '/pub trait ConsensusEngine/,/}$/d' "${PROJECT_DIR}/src/vm/mod.rs"
CACHEMOD_LINE=$(grep -n "pub mod cache;" "${PROJECT_DIR}/src/vm/mod.rs" | cut -d':' -f1)
if [ -n "$CACHEMOD_LINE" ]; then
  # Insert before the cache module line
  sed -i "${CACHEMOD_LINE}i$(cat ${PROJECT_DIR}/temp_ConsensusEngine.txt)" "${PROJECT_DIR}/src/vm/mod.rs"
else
  # Append at the end of the file
  cat "${PROJECT_DIR}/temp_ConsensusEngine.txt" >> "${PROJECT_DIR}/src/vm/mod.rs"
fi
echo "  Updated ConsensusEngine trait"

# 3. Add ConsensusFailure to VmError
echo "Adding ConsensusFailure to VmError..."
sed -i 's/pub enum VmError {/pub enum VmError {\n    ConsensusFailure(String),/g' "${PROJECT_DIR}/src/vm/mod.rs"
# Also update the Display implementation
sed -i '/Self::SerializationError(e)/,/},/s/},/},\n            Self::ConsensusFailure(e) => write!(f, "Consensus failure: {}", e),/g' "${PROJECT_DIR}/src/vm/mod.rs"
echo "  Added ConsensusFailure to VmError"

# 4. Fix ShardingCapability enum
echo "Fixing ShardingCapability enum..."
cat > "${PROJECT_DIR}/temp_ShardingCapability.txt" << 'EOF'
#[derive(Debug, Clone)]
pub enum ShardingCapability {
    None,
    DataParallel,
    ModelParallel,
    Horizontal,  // Added for AI executor
    Vertical,    // Added for AI executor
    Full,        // Added for AI executor
}
EOF

sed -i '/pub enum ShardingCapability/,/}/d' "${PROJECT_DIR}/src/contracts/mod.rs"
ENUM_LINE=$(grep -n "pub struct AIModelCall" "${PROJECT_DIR}/src/contracts/mod.rs" | cut -d':' -f1)
if [ -n "$ENUM_LINE" ]; then
  # Insert before AIModelCall
  sed -i "${ENUM_LINE}i$(cat ${PROJECT_DIR}/temp_ShardingCapability.txt)\n" "${PROJECT_DIR}/src/contracts/mod.rs"
else
  # Append at the end of the file
  cat "${PROJECT_DIR}/temp_ShardingCapability.txt" >> "${PROJECT_DIR}/src/contracts/mod.rs"
fi
echo "  Updated ShardingCapability enum"

# 5. Fix AIModelCall struct
echo "Fixing AIModelCall struct..."
cat > "${PROJECT_DIR}/temp_AIModelCall.txt" << 'EOF'
#[derive(Debug, Clone)]
pub struct AIModelCall {
    pub model_id: String,
    pub input: Vec<u8>,
    pub model: String,  // Added for compatibility with AI executor
    pub shard_count: u64, // Added for compatibility with AI executor
}
EOF

sed -i '/pub struct AIModelCall/,/}/d' "${PROJECT_DIR}/src/contracts/mod.rs"
STRUCT_LINE=$(grep -n "pub enum ShardingCapability" "${PROJECT_DIR}/src/contracts/mod.rs" | cut -d':' -f1)
if [ -n "$STRUCT_LINE" ]; then
  # Find end of enum
  END_LINE=$(tail -n +$STRUCT_LINE "${PROJECT_DIR}/src/contracts/mod.rs" | grep -n "}" | head -1 | cut -d':' -f1)
  END_LINE=$((STRUCT_LINE + END_LINE))
  # Insert after ShardingCapability enum
  sed -i "${END_LINE}a\\$(cat ${PROJECT_DIR}/temp_AIModelCall.txt)" "${PROJECT_DIR}/src/contracts/mod.rs"
else
  # Append at the end of the file
  cat "${PROJECT_DIR}/temp_AIModelCall.txt" >> "${PROJECT_DIR}/src/contracts/mod.rs"
fi
echo "  Updated AIModelCall struct"

# 6. Update ResourceUsage struct
echo "Updating ResourceUsage struct..."
cat > "${PROJECT_DIR}/temp_ResourceUsage.txt" << 'EOF'
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub compute_units: u64,
    pub memory_bytes: u64,
    pub storage_bytes: u64,
    pub cpu_time: u64,     // Added for compatibility with AI executor
    pub memory_used: u64,  // Added for compatibility with AI executor
    pub gpu_time: u64,     // Added for compatibility with AI executor
}
EOF

sed -i '/pub struct ResourceUsage/,/}/d' "${PROJECT_DIR}/src/state/mod.rs"
echo "$(cat ${PROJECT_DIR}/temp_ResourceUsage.txt)" >> "${PROJECT_DIR}/src/state/mod.rs"
echo "  Updated ResourceUsage struct"

# 7. Add Debug derive to StateDB
echo "Adding Debug derive to StateDB..."
sed -i 's/pub struct StateDB {/#[derive(Debug)]\npub struct StateDB {/g' "${PROJECT_DIR}/src/state/mod.rs"
echo "  Added Debug derive to StateDB"

# 8. Add Serialize and Deserialize to ContractCall
echo "Adding Serialize and Deserialize to ContractCall..."
sed -i 's/pub struct ContractCall {/#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct ContractCall {/g' "${PROJECT_DIR}/src/contracts/mod.rs"
echo "  Added Serialize and Deserialize to ContractCall"

# 9. Fix the Arc<VirtualMachine> mutable borrow issue
echo "Fixing Arc<VirtualMachine> mutable borrow issue..."
# Create a VmClone struct that can be used to fix the issue
cat > "${PROJECT_DIR}/temp_vm_fix.txt" << 'EOF'
// Add this to allow calling the execute method on Arc<VirtualMachine>
struct VmClone {
    inner: Arc<VirtualMachine>,
}

impl VmClone {
    fn new(vm: Arc<VirtualMachine>) -> Self {
        Self { inner: vm }
    }

    async fn execute(&self, call_data: &CallData, state_access: &dyn StateAccess) -> Result<CommonExecutionResult, VmError> {
        // We clone the VirtualMachine here to avoid the mutable borrow issue
        let mut vm_clone = VirtualMachine::new(self.inner.state_db.clone());
        vm_clone.execute(call_data, state_access).await
    }
}
EOF

# Insert before NarwhalBullsharkVm struct
NARWHAL_LINE=$(grep -n "pub struct NarwhalBullsharkVm" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs" | cut -d':' -f1)
if [ -n "$NARWHAL_LINE" ]; then
  NARWHAL_LINE=$((NARWHAL_LINE - 1))
  sed -i "${NARWHAL_LINE}i$(cat ${PROJECT_DIR}/temp_vm_fix.txt)\n" "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
fi

# Update the start_execution_loop method to use VmClone
sed -i '/fn start_execution_loop(&self) {/,/tokio::spawn/s/let vm = Arc::clone(&self.vm);/let vm = VmClone::new(Arc::clone(\&self.vm));/' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
sed -i 's/match vm.execute(&call_data, &state_access).await {/match vm.execute(\&call_data, \&state_access).await {/g' "${PROJECT_DIR}/src/vm/narwhal_bullshark_vm.rs"
echo "  Fixed Arc<VirtualMachine> mutable borrow issue"

# Cleanup temp files
rm "${PROJECT_DIR}/temp_ConsensusEngine.txt"
rm "${PROJECT_DIR}/temp_ShardingCapability.txt"
rm "${PROJECT_DIR}/temp_AIModelCall.txt"
rm "${PROJECT_DIR}/temp_ResourceUsage.txt"
rm "${PROJECT_DIR}/temp_vm_fix.txt"

echo "All final fixes applied! Try building your project again with 'cargo build'"
