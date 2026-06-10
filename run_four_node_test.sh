#!/bin/bash

# Run Four Node Real Connection Test
# This script runs the 4-node real IP connection test

echo "🌟 Q-NarwhalKnight Four Node Real Connection Test"
echo "================================================"
echo ""

# Set environment variables
export RUST_LOG=info
export RUST_BACKTRACE=1

# Build the q-api-server binary first (with timeout)
echo "🔨 Building q-api-server binary..."
timeout 600 cargo build --package q-api-server --bin q-api-server
if [ $? -ne 0 ]; then
    echo "❌ Failed to build q-api-server binary"
    exit 1
fi

echo "✅ q-api-server binary built successfully"
echo ""

# Run the test
echo "🚀 Running four node real connection test..."
echo "This test will:"
echo "  1. Start 4 real q-api-server processes"
echo "  2. Connect them via direct IP addresses"
echo "  3. Verify network connectivity" 
echo "  4. Test transaction propagation"
echo "  5. Clean up all processes"
echo ""

# Run just the basic connectivity test (fastest one)
timeout 300 cargo test --test four_node_real_connection_test test_four_node_real_ip_connections -- --nocapture

TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "✅ Four Node Real Connection Test PASSED!"
    echo "🎉 Successfully connected 4 real Q-NarwhalKnight nodes via IP"
else
    echo ""
    echo "❌ Four Node Real Connection Test FAILED!"
    echo "⚠️  Check logs above for details"
fi

exit $TEST_RESULT