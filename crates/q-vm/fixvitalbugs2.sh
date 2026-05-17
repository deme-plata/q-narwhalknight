#!/bin/bash

# fixp2p.sh - Script to fix DagKnight VM syntax errors
# Focuses on fixing the p2p.rs syntax errors and other compile issues

set -e  # Exit on any error

# Create backup directory
BACKUP_DIR=".backup/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Creating backup in $BACKUP_DIR"

# Function to backup a file before modifying it
backup_file() {
  local file_path="$1"
  local relative_path="${file_path#"$(pwd)/"}"
  local backup_path="$BACKUP_DIR/$relative_path"
  
  mkdir -p "$(dirname "$backup_path")"
  cp "$file_path" "$backup_path"
  echo "Backed up $file_path"
}

echo "Fixing P2P network module..."

# Fix the p2p.rs file
P2P_FILE="src/network/p2p.rs"
if [ -f "$P2P_FILE" ]; then
  backup_file "$P2P_FILE"
  
  # Fix the field name issues (remove leading underscore and triple underscores)
  sed -i 's/_ ___\([a-zA-Z_]*\)/\1/g' "$P2P_FILE"
  
  # Fix duplicated method declarations
  sed -i 's/async fn handle_transaction(async fn handle_transaction(async fn handle_transaction/async fn handle_transaction/g' "$P2P_FILE"
  sed -i 's/async fn handle_block(async fn handle_block(async fn handle_block/async fn handle_block/g' "$P2P_FILE"
  sed -i 's/async fn handle_consensus(async fn handle_consensus(async fn handle_consensus/async fn handle_consensus/g' "$P2P_FILE"
  
  # Fix the method parameter duplications
  sed -i 's/&self, data: Vec<u8>, __hash: Bytes32, __timestamp: u64)self, data: Vec<u8>, ___hash: Bytes32, ___timestamp: u64)self, data: Vec<u8>, ___hash: Bytes32, ___timestamp: u64/&self, data: Vec<u8>, hash: Bytes32, timestamp: u64/g' "$P2P_FILE"
  sed -i 's/&self, data: Vec<u8>, __hash: Bytes32, __height: u64, __timestamp: u64)self, data: Vec<u8>, ___hash: Bytes32, ___height: u64, ___timestamp: u64)self, data: Vec<u8>, ___hash: Bytes32, ___height: u64, ___timestamp: u64/&self, data: Vec<u8>, hash: Bytes32, height: u64, timestamp: u64/g' "$P2P_FILE"
  sed -i 's/&self, __consensus_type: ConsensusType, data: Vec<u8>, __timestamp: u64)self, ___consensus_type: ConsensusType, data: Vec<u8>, ___timestamp: u64)self, ___consensus_type: ConsensusType, data: Vec<u8>, ___timestamp: u64/&self, consensus_type: ConsensusType, data: Vec<u8>, timestamp: u64/g' "$P2P_FILE"
  
  # Fix parameter names in other handle_* methods
  sed -i 's/_ ___\([a-zA-Z_]*\)/\1/g' "$P2P_FILE"
  
  echo "Fixed P2P network module syntax issues"
else
  echo "ERROR: $P2P_FILE not found!"
  exit 1
fi

echo "Fixing conflicting Debug implementations..."

# Fix multiple Debug derive macros for P2pNetwork
sed -i '/^}#\[derive(Debug)\]/s/}#\[derive(Debug)\]/}/' "$P2P_FILE"
sed -i '/^#\[derive(Debug)\]/d' "$P2P_FILE"

# Fix StateDB Debug implementation
STATE_MOD_FILE="src/state/mod.rs"
if [ -f "$STATE_MOD_FILE" ]; then
  backup_file "$STATE_MOD_FILE"
  
  # Remove duplicate Debug derives
  sed -i '/#\[derive(Debug)\]/{N;/#\[derive(Debug)\]/d;}' "$STATE_MOD_FILE"
  
  # Add Debug derive to ResourceLedger
  RESOURCE_LEDGER_LINE=$(grep -n "pub struct ResourceLedger" "$STATE_MOD_FILE" | cut -d: -f1)
  if [ -n "$RESOURCE_LEDGER_LINE" ]; then
    RESOURCE_LEDGER_LINE=$((RESOURCE_LEDGER_LINE - 1))
    sed -i "${RESOURCE_LEDGER_LINE}i#[derive(Debug)]" "$STATE_MOD_FILE"
    echo "Added Debug derive to ResourceLedger"
  fi
  
  echo "Fixed StateDB Debug implementation"
else
  echo "WARNING: $STATE_MOD_FILE not found, skipping StateDB fixes"
fi

echo "Fixing missing imports..."

# Fix missing Contract imports in VM modules
VM_FILES=("src/vm/parallel_executor.rs" "src/vm/tiered_vm.rs" "src/vm/mod.rs")
for VM_FILE in "${VM_FILES[@]}"; do
  if [ -f "$VM_FILE" ]; then
    backup_file "$VM_FILE"
    
    # Add imports at the top of the file
    sed -i '1s/^/use crate::contracts::{Contract, ContractCall, ContractResult, ContractRegistry};\nuse std::collections::HashMap;\n\n/' "$VM_FILE"
    
    echo "Fixed imports in $VM_FILE"
  else
    echo "WARNING: $VM_FILE not found, skipping"
  fi
done

# Fix ContractCall import in transaction module
TRANSACTION_FILE="src/transaction/mod.rs"
if [ -f "$TRANSACTION_FILE" ]; then
  backup_file "$TRANSACTION_FILE"
  
  # Add import at the top of the file
  sed -i '1s/^/use crate::contracts::ContractCall;\n\n/' "$TRANSACTION_FILE"
  
  echo "Fixed imports in $TRANSACTION_FILE"
else
  echo "WARNING: $TRANSACTION_FILE not found, skipping"
fi

echo "Running cargo check to verify fixes..."
cargo check

echo "Done! Your code should now compile successfully or have fewer errors."
echo "If there are still errors, please check the cargo check output for details."
