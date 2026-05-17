#!/bin/bash

# Script to fix the remaining DagKnight VM compilation errors

echo "Starting error fixing script for remaining issues..."
echo "Creating backup of codebase..."

# Create a backup directory
BACKUP_DIR="../dagknight-vm-backup-$(date +%Y%m%d%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR"
echo "Backup created at $BACKUP_DIR"

# Fix 1: Revert the dagknight_vm::types imports back to crate::types
echo "Fixing imports from dagknight_vm::types back to crate::types..."
FILES=("src/consensus/narwhal_bullshark.rs" "src/state/mod.rs" "src/vm/mod.rs" "src/vm/narwhal_bullshark_vm.rs")
for file in "${FILES[@]}"; do
  if [ -f "$file" ]; then
    # Replace dagknight_vm::types with crate::types
    sed -i 's/use dagknight_vm::types/use crate::types/' "$file"
    sed -i 's/pub use dagknight_vm::types/pub use crate::types/' "$file"
  fi
done

# Fix 2: Fix CommonCommonExecutionResult to CommonExecutionResult
echo "Fixing CommonCommonExecutionResult typo..."
if [ -f "src/vm/narwhal_bullshark_vm.rs" ]; then
  sed -i 's/CommonCommonExecutionResult/CommonExecutionResult/g' src/vm/narwhal_bullshark_vm.rs
fi

# Fix 3: Fix 'sharding_strategy' not found error
echo "Fixing 'sharding_strategy' not found error..."
if [ -f "src/vm/ai/executor.rs" ]; then
  cat > temp_ai_executor.txt << 'EOF'
        // Check if model supports sharding
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
        
        // Prepare execution
        let start_time = Instant::now();
        let result: Result<Vec<u8>>;
        let nodes_used: u64;
        
        // Execute based on sharding strategy
        match sharding_strategy {
EOF

  # Find the section to replace
  MATCH_START=$(grep -n "// Check if model supports sharding" src/vm/ai/executor.rs | cut -d':' -f1)
  if [ -n "$MATCH_START" ]; then
    # Find the start of the second match statement
    MATCH_END=$(tail -n +$MATCH_START src/vm/ai/executor.rs | grep -n "match sharding_strategy {" | head -1)
    if [ -n "$MATCH_END" ]; then
      MATCH_END=$(($MATCH_START + $(echo $MATCH_END | cut -d':' -f1) - 1))
      
      # Replace the section
      sed -i "${MATCH_START},${MATCH_END}c\\$(cat temp_ai_executor.txt)" src/vm/ai/executor.rs
    fi
  fi
  
  # Clean up
  rm temp_ai_executor.txt
fi

# Fix 4: Remove duplicate derive for ShardingCapability
echo "Fixing duplicate derive for ShardingCapability..."
if [ -f "src/contracts/mod.rs" ]; then
  # Find the line with the first derive
  FIRST_DERIVE=$(grep -n "#\[derive(Debug, Clone, Serialize, Deserialize)\]" src/contracts/mod.rs | head -1 | cut -d':' -f1)
  
  if [ -n "$FIRST_DERIVE" ]; then
    # Get the next line which should be another derive or the enum definition
    NEXT_LINE=$((FIRST_DERIVE + 1))
    
    # Check if the next line is another derive macro
    SECOND_DERIVE=$(sed -n "${NEXT_LINE}p" src/contracts/mod.rs)
    if [[ "$SECOND_DERIVE" == *"#[derive"* ]]; then
      # Remove the second derive line
      sed -i "${NEXT_LINE}d" src/contracts/mod.rs
    fi
  fi
fi

# Add missing serde imports if needed
echo "Adding missing serde imports..."
if grep -q "Serialize" src/contracts/mod.rs && ! grep -q "serde::{Serialize, Deserialize}" src/contracts/mod.rs; then
  # Add serde import at the top of the file
  sed -i '1i use serde::{Serialize, Deserialize};' src/contracts/mod.rs
fi

echo "All remaining fixes applied! Try running 'cargo build' again."
