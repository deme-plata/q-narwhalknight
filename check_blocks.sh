#!/bin/bash
DB_PATH="./data-mine6/hot"

# Use rocksdb-tools to check blocks column family
echo "Checking blocks in $DB_PATH..."

# Try to list keys in blocks column family
echo "Attempting to scan blocks column family..."
./target/release/repair-database "$DB_PATH" <<< "2" 2>&1 | grep -A20 "Scan Results"
