#!/bin/bash

# 🔥 Q-NarwhalKnight Tor DHT Discovery Test Script
# Tests actual Tor DHT connectivity between your nodes

set -e

echo "🔥 Q-NarwhalKnight Tor DHT Connection Test"
echo "=============================================="

# Check if Tor is running
if ! pgrep -x "tor" > /dev/null; then
    echo "⚠️  Tor daemon not running. Starting Tor..."
    # Try to start Tor if available
    if command -v tor &> /dev/null; then
        tor --RunAsDaemon 1 --DataDirectory /tmp/tor_test_data &
        sleep 5
        echo "✅ Tor daemon started"
    else
        echo "❌ Tor not found. Please install Tor:"
        echo "   Ubuntu/Debian: sudo apt install tor"
        echo "   macOS: brew install tor"
        echo "   Fedora/RHEL: sudo dnf install tor"
        exit 1
    fi
else
    echo "✅ Tor daemon is running"
fi

# Build the test if needed
echo "🔨 Building Tor DHT connection test..."
if ! cargo build --example tor_dht_connection_test --quiet; then
    echo "❌ Failed to build test. Check dependencies:"
    echo "   Make sure arti-client crate is properly configured"
    exit 1
fi
echo "✅ Test built successfully"

# Test mode selection
echo ""
echo "Select test mode:"
echo "1) 🚀 Start as PUBLISHER node (run this first)"
echo "2) 🔍 Start as SEARCHER node (run this second from another terminal)"
echo "3) 🤖 Automated two-node test"
echo "4) 📊 Check current implementation status"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Starting as PUBLISHER node..."
        NODE_ID="${NODE_ID:-ALPHA_$(date +%s)}"
        PORT="${PORT:-8333}"
        
        echo "   Node ID: $NODE_ID"
        echo "   Port: $PORT"
        echo ""
        echo "ℹ️  This node will advertise itself on Tor DHT"
        echo "ℹ️  Leave this running and start searcher from another terminal"
        echo ""
        
        cargo run --example tor_dht_connection_test -- \
            --mode publisher \
            --node-id "$NODE_ID" \
            --port "$PORT" \
            --verbose
        ;;
        
    2)
        echo "🔍 Starting as SEARCHER node..."
        NODE_ID="${NODE_ID:-BETA_$(date +%s)}"
        PORT="${PORT:-8334}"
        TARGET="${TARGET:-ALPHA}"
        
        read -p "Enter target node ID to search for (default: $TARGET): " input_target
        TARGET="${input_target:-$TARGET}"
        
        echo "   Node ID: $NODE_ID"
        echo "   Port: $PORT"
        echo "   Searching for: $TARGET"
        echo ""
        echo "ℹ️  This will search Tor DHT for the target node"
        echo ""
        
        cargo run --example tor_dht_connection_test -- \
            --mode searcher \
            --node-id "$NODE_ID" \
            --port "$PORT" \
            --target-node "$TARGET" \
            --timeout 120 \
            --verbose
        ;;
        
    3)
        echo "🤖 Running automated two-node test..."
        echo "   This will start both publisher and searcher automatically"
        echo ""
        
        # Start publisher in background
        NODE_ID_1="AUTO_ALPHA_$(date +%s)"
        NODE_ID_2="AUTO_BETA_$(date +%s)"
        
        echo "🚀 Starting publisher node: $NODE_ID_1"
        cargo run --example tor_dht_connection_test -- \
            --mode publisher \
            --node-id "$NODE_ID_1" \
            --port 8333 &
        
        PUBLISHER_PID=$!
        echo "   Publisher PID: $PUBLISHER_PID"
        
        # Wait for publisher to initialize
        echo "⏰ Waiting 10 seconds for publisher to initialize..."
        sleep 10
        
        # Start searcher
        echo "🔍 Starting searcher node: $NODE_ID_2"
        cargo run --example tor_dht_connection_test -- \
            --mode searcher \
            --node-id "$NODE_ID_2" \
            --port 8334 \
            --target-node "$NODE_ID_1" \
            --timeout 60 \
            --verbose &
        
        SEARCHER_PID=$!
        echo "   Searcher PID: $SEARCHER_PID"
        
        # Wait for test to complete
        wait $SEARCHER_PID
        SEARCHER_EXIT=$?
        
        # Clean up publisher
        echo "🧹 Cleaning up publisher..."
        kill $PUBLISHER_PID 2>/dev/null || true
        
        # Report results
        echo ""
        echo "📊 Automated Test Results:"
        if [ $SEARCHER_EXIT -eq 0 ]; then
            echo "✅ SUCCESS: Nodes successfully discovered each other!"
            echo "🎉 Your Tor DHT implementation is working!"
        else
            echo "❌ FAILED: Nodes could not discover each other"
            echo "🔧 Check your Tor DHT implementation"
        fi
        ;;
        
    4)
        echo "📊 Checking current implementation status..."
        echo ""
        
        # Check if implementation files exist
        echo "🔍 Implementation Status:"
        
        if [ -f "crates/q-tor-client/src/tor_dht_discovery.rs" ]; then
            echo "✅ Tor DHT discovery module found"
            
            # Check if it's using real implementation
            if grep -q "simulate.*DHT" crates/q-tor-client/src/tor_dht_discovery.rs; then
                echo "⚠️  Implementation contains simulation code"
                echo "   Need to replace with real Tor DHT calls"
            else
                echo "✅ Implementation appears to use real Tor calls"
            fi
        else
            echo "❌ Tor DHT discovery module not found"
        fi
        
        if [ -f "crates/q-tor-client/src/unified_free_discovery.rs" ]; then
            echo "✅ Unified discovery coordinator found"
        else
            echo "❌ Unified discovery coordinator not found"
        fi
        
        echo ""
        echo "🔧 To fix simulation issues:"
        echo "1. Update tor_dht_discovery.rs publish_to_dht() function"
        echo "2. Update tor_dht_discovery.rs query_dht() function"
        echo "3. Use real arti-client Tor directory operations"
        echo "4. Test with this script again"
        ;;
        
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🔥 Test completed!"