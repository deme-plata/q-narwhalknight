#!/bin/bash

echo "🎯 Direct Binary Four Node Test"
echo "==============================="

# Find the binary
BINARY=""
if [ -f "./target/x86_64-unknown-linux-gnu/release/q-api-server" ]; then
    BINARY="./target/x86_64-unknown-linux-gnu/release/q-api-server"
elif [ -f "./target/release/q-api-server" ]; then
    BINARY="./target/release/q-api-server"
elif [ -f "./target/debug/q-api-server" ]; then
    BINARY="./target/debug/q-api-server"
elif [ -f "./target/x86_64-unknown-linux-gnu/debug/q-api-server" ]; then
    BINARY="./target/x86_64-unknown-linux-gnu/debug/q-api-server"
else
    echo "❌ No q-api-server binary found!"
    echo "Checked locations:"
    echo "  - ./target/x86_64-unknown-linux-gnu/release/q-api-server"
    echo "  - ./target/release/q-api-server"
    echo "  - ./target/debug/q-api-server"
    echo "  - ./target/x86_64-unknown-linux-gnu/debug/q-api-server"
    exit 1
fi

echo "📁 Using binary: $BINARY"

# Start 4 nodes in background
PIDS=()
PORTS=(18031 18032 18033 18034)

echo "🚀 Starting 4 nodes..."

for i in "${!PORTS[@]}"; do
    NODE_ID="direct-node-$((i+1))"
    PORT="${PORTS[$i]}"
    DATA_DIR="/tmp/q-test-node-$((i+1))"
    
    echo "🔧 Starting $NODE_ID on port $PORT"
    
    # Clean up any existing data directory
    rm -rf "$DATA_DIR" 2>/dev/null
    mkdir -p "$DATA_DIR"
    
    RUST_LOG=error SKIP_BITCOIN=1 SKIP_DNS=1 \
    Q_DB_PATH="$DATA_DIR/db" Q_HOT_DB_PATH="$DATA_DIR/hot" \
    $BINARY --node-id "$NODE_ID" --port "$PORT" \
    > /dev/null 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "  📍 PID: $PID (data: $DATA_DIR)"
done

echo "⏳ Waiting for nodes to start..."
sleep 10

# Test connectivity
echo "🔍 Testing connectivity..."
ALL_GOOD=true

for PORT in "${PORTS[@]}"; do
    URL="http://127.0.0.1:$PORT/api/v1/health"
    
    if curl -s --max-time 3 "$URL" > /dev/null 2>&1; then
        echo "✅ Node on port $PORT is responding"
    else
        echo "❌ Node on port $PORT is not responding"
        ALL_GOOD=false
    fi
done

# Check status of each node
echo ""
echo "📊 Node Status:"
for PORT in "${PORTS[@]}"; do
    URL="http://127.0.0.1:$PORT/api/v1/status"
    
    STATUS=$(curl -s --max-time 3 "$URL" 2>/dev/null | grep -o '"node_id":"[^"]*"' | cut -d'"' -f4)
    if [ -n "$STATUS" ]; then
        echo "✅ Port $PORT: Node ID = $STATUS"
    else
        echo "⚠️  Port $PORT: No response"
    fi
done

# Send test transaction
echo ""
echo "🧪 Testing transaction submission..."
TX_URL="http://127.0.0.1:${PORTS[0]}/api/v1/transactions"
TX_DATA='{"from":"test_sender","to":"test_receiver","amount":1000,"nonce":1}'

if curl -s --max-time 5 -X POST -H "Content-Type: application/json" -d "$TX_DATA" "$TX_URL" > /dev/null 2>&1; then
    echo "✅ Test transaction submitted successfully"
else
    echo "⚠️  Test transaction failed"
fi

# Results
echo ""
if [ "$ALL_GOOD" = true ]; then
    echo "🎉 DIRECT BINARY TEST PASSED!"
    echo "✅ All 4 nodes started and responded correctly"
else
    echo "⚠️  Some issues detected in the test"
fi

# Cleanup
echo ""
echo "🛑 Shutting down nodes..."
for PID in "${PIDS[@]}"; do
    if kill "$PID" 2>/dev/null; then
        echo "✅ Stopped PID $PID"
    else
        echo "⚠️  PID $PID already stopped"
    fi
done

# Wait a moment for cleanup
sleep 2

# Clean up data directories
echo "🧹 Cleaning up data directories..."
for i in {1..4}; do
    rm -rf "/tmp/q-test-node-$i" 2>/dev/null
done

echo "✅ Direct binary test completed!"