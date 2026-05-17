#!/bin/bash
# Debug build script for DAGKnight VM

set -e

echo "Running minimal build with verbose output..."
cargo clean
cargo build -v

if [ $? -eq 0 ]; then
    echo "Basic build successful!"
else
    echo "Build failed. Check the error messages above."
    exit 1
fi

echo "Running tests with single thread and increased verbosity..."
cargo test -- --test-threads=1 -v

if [ $? -eq 0 ]; then
    echo "Tests passed!"
else
    echo "Tests failed. Check the error messages above."
    exit 1
fi

echo "All checks passed!"
