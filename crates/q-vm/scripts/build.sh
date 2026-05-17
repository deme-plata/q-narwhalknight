#!/bin/bash
# Build script for DAGKnight VM

set -e

echo "Building DAGKnight VM..."
cargo build --release

echo "Running tests..."
cargo test

echo "Build completed successfully"
