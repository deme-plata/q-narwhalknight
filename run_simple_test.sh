#!/bin/bash

echo "🚀 Running Simple Four Node Test"
echo "================================="

# Build q-api-server first
echo "🔨 Building q-api-server..."
timeout 300 cargo build --package q-api-server --bin q-api-server --quiet

if [ $? -eq 0 ]; then
    echo "✅ Build successful"
    
    # Run the simple test
    echo "🧪 Running test..."
    timeout 120 cargo test --test simple_four_node_test -- --nocapture
    
    if [ $? -eq 0 ]; then
        echo "🎉 TEST PASSED!"
    else
        echo "❌ TEST FAILED"
    fi
else
    echo "❌ Build failed"
fi