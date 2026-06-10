#!/bin/bash

# Script to fix the DagKnight VM compilation errors

echo "Starting error fixing script..."
echo "Creating backup of codebase..."

# Create a backup directory
BACKUP_DIR="../dagknight-vm-backup-$(date +%Y%m%d%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR"
echo "Backup created at $BACKUP_DIR"

# Fix 1: Fix the duplicate #[derive(Parser)] in narwhal_bullshark_bench.rs
echo "Fixing duplicate Parser derive in narwhal_bullshark_bench.rs..."
if [ -f "src/bin/narwhal_bullshark_bench.rs" ]; then
  # Remove the second #[derive(Parser)]
  sed -i '13s/#\[derive(Parser)\]//' src/bin/narwhal_bullshark_bench.rs
fi

# Fix 2: Fix for loop syntax error in fault_tolerance/mod.rs
echo "Fixing for loop syntax in fault_tolerance/mod.rs..."
if [ -f "src/fault_tolerance/mod.rs" ]; then
  # Replace the problematic line
  sed -i '200s/for (_i, for (i, &idx) in_idx) in failed_indices.iter().enumerate() {/for (_i, &_idx) in failed_indices.iter().enumerate() {/' src/fault_tolerance/mod.rs
fi

# Fix 3: Fix imports in files using crate::types
echo "Fixing types imports..."
FILES=("src/consensus/narwhal_bullshark.rs" "src/state/mod.rs" "src/vm/mod.rs" "src/vm/narwhal_bullshark_vm.rs")
for file in "${FILES[@]}"; do
  if [ -f "$file" ]; then
    # Replace crate::types with dagknight_vm::types
    sed -i 's/use crate::types/use dagknight_vm::types/' "$file"
    sed -i 's/pub use crate::types/pub use dagknight_vm::types/' "$file"
  fi
done

# Fix 4: Fix VirtualMachine::new() missing argument
echo "Fixing VirtualMachine::new() call..."
if [ -f "src/bin/narwhal_bullshark_bench.rs" ]; then
  # Add the required StateDB argument
  sed -i '64s/VirtualMachine::new()/VirtualMachine::new(Arc::new(crate::state::StateDB::new()))/' src/bin/narwhal_bullshark_bench.rs
fi

# Fix 5: Fix ExecutionResult type in narwhal_bullshark_vm.rs
echo "Fixing ExecutionResult type usage..."
if [ -f "src/vm/narwhal_bullshark_vm.rs" ]; then
  # Replace ExecutionResult with CommonExecutionResult
  sed -i '138s/ExecutionResult/CommonExecutionResult/' src/vm/narwhal_bullshark_vm.rs
  sed -i '197s/ExecutionResult/CommonExecutionResult/' src/vm/narwhal_bullshark_vm.rs
fi

# Fix 6: Fix ModelRegistration field access
echo "Fixing ModelRegistration field access in models/mod.rs..."
if [ -f "src/models/mod.rs" ]; then
  # Fix resource_requirements field accesses
  sed -i 's/resource_requirements\.min_memory_mb/resources.min_memory_mb/g' src/models/mod.rs
  sed -i 's/resource_requirements\.gpu_memory_mb/resources.min_gpu_memory_mb/g' src/models/mod.rs
  sed -i 's/resource_requirements\.min_memory_mb/resources.min_memory_mb/g' src/models/mod.rs
  
  # Fix sharding_capability field access
  sed -i 's/sharding_capability/capabilities/g' src/models/mod.rs
  
  # Fix ModelRegistration struct instantiation
  sed -i 's/memory_required:/\/\/ memory_required:/g' src/models/mod.rs
  sed -i 's/resource_requirements:/resources:/g' src/models/mod.rs
  
  # Fix ResourceRequirements field names
  sed -i 's/gpu_memory_mb:/min_gpu_memory_mb:/g' src/models/mod.rs
  sed -i 's/disk_space_mb:/\/\/ disk_space_mb:/g' src/models/mod.rs
  sed -i 's/avg_exec_time_per_token_ms:/\/\/ avg_exec_time_per_token_ms:/g' src/models/mod.rs
fi

# Fix 7: Fix the ShardingCapability pattern matching in the AI executor
echo "Fixing ShardingCapability pattern matching in AI executor..."
if [ -f "src/vm/ai/executor.rs" ]; then
  # Create a temporary file with the correct pattern matching
  cat > temp_match.txt << 'EOF'
        let sharding_strategy = match model_info.capabilities {
            ShardingCapability::None => {
                if model_call.shard_count > 1 {
                    warn!("Model {} doesn't support sharding but {} shards requested", 
                           model_call.model, model_call.shard_count);
                }
                ShardingStrategy::None
            },
            ShardingCapability::Horizontal | ShardingCapability::DataParallel => ShardingStrategy::Horizontal,
            ShardingCapability::Vertical | ShardingCapability::ModelParallel => ShardingStrategy::Vertical,
            ShardingCapability::Full => {
                // Choose best strategy based on historical performance
                self.determine_best_strategy(&model_call.model, model_call.input.len()).await
            }
        };
EOF

  # Replace the pattern matching section
  # First, find the line number where the pattern matching starts
  MATCH_START=$(grep -n "let sharding_strategy = match model_info.capabilities" src/vm/ai/executor.rs | cut -d':' -f1)
  if [ -n "$MATCH_START" ]; then
    # Find the line where it ends (the line with the closing '};')
    MATCH_END=$(tail -n +$MATCH_START src/vm/ai/executor.rs | grep -n "^[[:space:]]*};" | head -1)
    if [ -n "$MATCH_END" ]; then
      MATCH_END=$(($MATCH_START + $(echo $MATCH_END | cut -d':' -f1) - 1))
      
      # Create a new file with the substitution
      sed -i "${MATCH_START},${MATCH_END}d" src/vm/ai/executor.rs
      sed -i "${MATCH_START}i\\$(cat temp_match.txt)" src/vm/ai/executor.rs
    fi
  fi
  
  # Clean up the temporary file
  rm temp_match.txt
fi

# Fix 8: Add derive(Serialize, Deserialize) to ShardingCapability enum
echo "Adding Serialize, Deserialize to ShardingCapability enum..."
if [ -f "src/contracts/mod.rs" ]; then
  # Add the derive attributes
  sed -i '/pub enum ShardingCapability/i #[derive(Debug, Clone, Serialize, Deserialize)]' src/contracts/mod.rs
fi

echo "All fixes applied! Try running 'cargo build' again."
