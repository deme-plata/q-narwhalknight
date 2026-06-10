#!/bin/bash

# Fix the serde-big-array dependency name in Cargo.toml
echo "Fixing serde-big-array dependency..."
sed -i 's/serde_big_array = "0.4.0"/serde-big-array = "0.4.0"/' Cargo.toml

# Also update the import in the transaction module
echo "Updating import in transaction/mod.rs..."
sed -i 's/use serde_big_array::BigArray;/use serde_big_array::big_array;/' src/transaction/mod.rs

# Update the big_array macro usage
echo "Fixing big_array macro usage..."
sed -i '1a\
big_array! { BigArray; 64 }' src/transaction/mod.rs

echo "Cargo dependency fixed. Try building with 'cargo build' now."
