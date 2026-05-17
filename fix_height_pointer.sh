#!/bin/bash
# Direct RocksDB fix for height pointer desync v0.9.99-beta
# Updates qblock:latest from 703 to 710

set -e

DB_PATH="./data-mine10/hot"

echo "🔧 Q-NarwhalKnight Height Pointer Fix"
echo "====================================="
echo ""
echo "Database: $DB_PATH"
echo "Target: Update qblock:latest from 703 to 710"
echo ""

# Create Python script to update RocksDB
cat > /tmp/fix_pointer.py << 'EOF'
#!/usr/bin/env python3
import rocksdb
import struct
import sys

db_path = "./data-mine10/hot"
print(f"Opening database: {db_path}")

# Open database
opts = rocksdb.Options()
opts.create_if_missing = False
db = rocksdb.DB(db_path, opts)

# Read current pointer
current_bytes = db.get(b"qblock:latest")
if current_bytes:
    current_height = struct.unpack('>Q', current_bytes)[0]
    print(f"Current pointer: {current_height}")
else:
    print("No pointer found!")
    sys.exit(1)

# Update to 710
new_height = 710
new_bytes = struct.pack('>Q', new_height)
db.put(b"qblock:latest", new_bytes)

# Verify
verify_bytes = db.get(b"qblock:latest")
verify_height = struct.unpack('>Q', verify_bytes)[0]

if verify_height == new_height:
    print(f"✅ Success! Pointer updated: {current_height} → {verify_height}")
else:
    print(f"❌ Failed! Expected {new_height}, got {verify_height}")
    sys.exit(1)

print("")
print("✅ Height pointer repair complete")
print("   You can now restart: systemctl start q-api-server")
EOF

# Try Python approach first
if command -v python3 &> /dev/null; then
    echo "Using Python3 with python-rocksdb..."
    python3 /tmp/fix_pointer.py
else
    echo "❌ Python3 not found"
    echo "❌ Cannot proceed with repair"
    exit 1
fi
