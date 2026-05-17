#!/bin/bash

# Simple script to fix issues in the DagKnight VM codebase

echo "Creating backup of your codebase..."
# Create a backup of the project
BACKUP_DIR="../dagknight-vm-backup-$(date +%Y%m%d%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR"
echo "Backup created at $BACKUP_DIR"

echo "Fixing AI Executor ShardingCapability pattern matching..."
# Fix ShardingCapability matching in AI executor
if [[ -f "src/vm/ai/executor.rs" ]]; then
  echo "Modifying executor.rs..."
  sed -i 's/ShardingCapability::Horizontal => ShardingStrategy::Horizontal,/ShardingCapability::Horizontal | ShardingCapability::DataParallel => ShardingStrategy::Horizontal,/' src/vm/ai/executor.rs
  sed -i 's/ShardingCapability::Vertical => ShardingStrategy::Vertical,/ShardingCapability::Vertical | ShardingCapability::ModelParallel => ShardingStrategy::Vertical,/' src/vm/ai/executor.rs
  
  # Remove any existing DataParallel and ModelParallel pattern matches if they exist
  sed -i '/ShardingCapability::DataParallel => ShardingStrategy::Horizontal,/d' src/vm/ai/executor.rs
  sed -i '/ShardingCapability::ModelParallel => ShardingStrategy::Vertical,/d' src/vm/ai/executor.rs
  echo "executor.rs modified"
fi

echo "All fixes have been applied!"
echo "Please run 'cargo build' to check if all issues have been resolved."
