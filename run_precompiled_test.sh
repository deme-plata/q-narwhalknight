#!/bin/bash

echo "🚀 Four Node Test Using Pre-compiled Binary"
echo "==========================================="
echo ""

# Check if q-api-server binary exists
BINARY_FOUND=false

# Check common locations
if [ -f "./target/release/q-api-server" ]; then
    echo "✅ Found release binary: ./target/release/q-api-server"
    BINARY_FOUND=true
elif [ -f "./target/debug/q-api-server" ]; then
    echo "✅ Found debug binary: ./target/debug/q-api-server"
    BINARY_FOUND=true
elif command -v q-api-server &> /dev/null; then
    echo "✅ Found q-api-server in PATH"
    BINARY_FOUND=true
else
    echo "🔍 Searching for binary..."
    # Try to find with cargo metadata
    TARGET_DIR=$(cargo metadata --format-version 1 2>/dev/null | grep -o '"target_directory":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$TARGET_DIR" ]; then
        if [ -f "$TARGET_DIR/release/q-api-server" ]; then
            echo "✅ Found binary: $TARGET_DIR/release/q-api-server"
            BINARY_FOUND=true
        elif [ -f "$TARGET_DIR/debug/q-api-server" ]; then
            echo "✅ Found binary: $TARGET_DIR/debug/q-api-server"
            BINARY_FOUND=true
        fi
    fi
fi

if [ "$BINARY_FOUND" = false ]; then
    echo "❌ No compiled q-api-server binary found!"
    echo "   Please run one of these first:"
    echo "   - cargo build --bin q-api-server (debug)"
    echo "   - cargo build --release --bin q-api-server (release)"
    exit 1
fi

echo ""
echo "🧪 Running test with pre-compiled binary..."
echo "This test will:"
echo "  1. Use existing q-api-server binary (no recompilation)"
echo "  2. Start 4 nodes on ports 18021-18024"  
echo "  3. Test connectivity and functionality"
echo "  4. Clean shutdown"
echo ""

# Set minimal environment for speed
export RUST_LOG=error
export SKIP_BITCOIN=1
export SKIP_DNS=1

# Run the test
timeout 180 cargo test --test four_node_precompiled_test test_four_nodes_precompiled -- --nocapture

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "🎉 PRE-COMPILED FOUR NODE TEST PASSED!"
    echo "✅ Successfully used existing binary to test 4-node network"
else
    echo "❌ Pre-compiled four node test failed"
    echo "⚠️  Check the output above for details"
fi

echo ""
echo "🚀 Running rapid startup/shutdown test..."
timeout 60 cargo test --test four_node_precompiled_test test_rapid_startup_shutdown -- --nocapture

if [ $? -eq 0 ]; then
    echo "✅ Rapid test also passed!"
else
    echo "⚠️  Rapid test had issues"
fi

exit $TEST_RESULT