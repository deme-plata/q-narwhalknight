#!/bin/bash
# Final fix for the pbft.rs blockchain variable issue

set -e

VM_DIR="${1:-/home/myuser/viper/dagknight-vm}"
echo "====== DAGKnight VM PBFT Fix ======"
echo "Target directory: $VM_DIR"

# Create backup of current state
BACKUP_DIR="$VM_DIR/backup-pbft-$(date +%Y%m%d%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r "$VM_DIR/src" "$BACKUP_DIR/"
echo "✅ Created backup at $BACKUP_DIR"

# Fix the pbft.rs file for blockchain variable issue
if [ -f "$VM_DIR/src/consensus/pbft.rs" ]; then
    # Create a temporary file with the fix
    awk '{
        if ($0 ~ /let _blockchain = self.blockchain.write\(\).await;/) {
            print "            let mut blockchain = self.blockchain.write().await;";
        } else {
            print $0;
        }
    }' "$VM_DIR/src/consensus/pbft.rs" > "$VM_DIR/src/consensus/pbft.rs.fix"
    
    # Replace the original file
    mv "$VM_DIR/src/consensus/pbft.rs.fix" "$VM_DIR/src/consensus/pbft.rs"
    echo "✅ Fixed blockchain variable in pbft.rs"
fi

# Final message
echo ""
echo "====== PBFT Fix Complete ======"
echo "The blockchain variable issue in pbft.rs has been fixed."
echo "You can now run 'cargo build' to build the VM."
