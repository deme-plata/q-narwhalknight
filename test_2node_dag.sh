#!/bin/bash
# Q-NarwhalKnight 2-Node DAG Consensus Test
# Tests multi-node DAG propagation without disrupting main node on port 8080

set -e

echo "🚀 Q-NarwhalKnight 2-Node DAG Consensus Test"
echo "============================================="
echo ""
echo "Main node (port 8080): PRESERVED for user mining"
echo "Test nodes (ports 8091, 8093): Fresh instances for DAG testing"
echo ""

# Clean up any existing test nodes
echo "🧹 Cleaning up existing test nodes..."
pkill -f "q-api-server --port 809" || true
sleep 2

# Clean test data directories
echo "🗑️  Cleaning test data directories..."
rm -rf /opt/orobit/shared/q-narwhalknight/testdag-node1
rm -rf /opt/orobit/shared/q-narwhalknight/testdag-node3

# Create fresh directories
mkdir -p /opt/orobit/shared/q-narwhalknight/testdag-node1
mkdir -p /opt/orobit/shared/q-narwhalknight/testdag-node3

echo ""
echo "🔧 Starting 2-node DAG testnet..."
echo ""

# Start Node 1 (port 8091, P2P 9091)
echo "📡 Starting Node 1 (HTTP: 8091, P2P: 9091)..."
Q_DB_PATH=./testdag-node1 \
Q_P2P_PORT=9091 \
nohup timeout 3600 ./target/release/q-api-server \
  --port 8091 \
  --node-id testdag-node1 \
  > testdag-node1.log 2>&1 &
NODE1_PID=$!
echo "   Node 1 PID: $NODE1_PID"
sleep 5

# Start Node 3 (port 8093, P2P 9093)
echo "📡 Starting Node 3 (HTTP: 8093, P2P: 9093)..."
Q_DB_PATH=./testdag-node3 \
Q_P2P_PORT=9093 \
nohup timeout 3600 ./target/release/q-api-server \
  --port 8093 \
  --node-id testdag-node3 \
  > testdag-node3.log 2>&1 &
NODE3_PID=$!
echo "   Node 3 PID: $NODE3_PID"
sleep 5

echo ""
echo "⏳ Waiting for nodes to initialize (10 seconds)..."
sleep 10

echo ""
echo "🔗 Fetching node peer IDs..."
echo ""

# Get peer IDs
PEER1=$(curl -s http://localhost:8091/api/v1/peer-id | jq -r '.data.peer_id' 2>/dev/null || echo "unknown")
PEER3=$(curl -s http://localhost:8093/api/v1/peer-id | jq -r '.data.peer_id' 2>/dev/null || echo "unknown")

echo "   Node 1 Peer ID: $PEER1"
echo "   Node 3 Peer ID: $PEER3"

echo ""
echo "🌐 Connecting nodes (Node 3 → Node 1)..."
echo ""

# Connect Node 3 to Node 1
if [ "$PEER1" != "unknown" ]; then
    echo "   Connecting Node 3 → Node 1..."
    curl -s -X POST http://localhost:8093/api/v1/connect \
      -H "Content-Type: application/json" \
      -d "{\"multiaddr\": \"/ip4/127.0.0.1/tcp/9091/p2p/$PEER1\"}" | jq '.'
    sleep 2
fi

echo ""
echo "✅ 2-Node DAG Testnet Running!"
echo ""
echo "📊 Node Information:"
echo "   Node 1: http://localhost:8091 (Leader/Bootstrap)"
echo "   Node 3: http://localhost:8093 (Connected to Node 1)"
echo ""
echo "📝 Log Files:"
echo "   Node 1: testdag-node1.log"
echo "   Node 3: testdag-node3.log"
echo ""
echo "🔍 Monitor Commands:"
echo "   tail -f testdag-node1.log | grep -E '📦|📡|🎯|FINALIZED'"
echo "   tail -f testdag-node3.log | grep -E '📦|📡|🎯|FINALIZED'"
echo ""
echo "🧪 Test Block Propagation:"
echo "   Watch Node 1 produce blocks (time-based every ~15s)"
echo "   Verify Node 3 receives blocks via '📦 Received block' messages"
echo "   Verify both nodes finalize blocks via '🎯 INCOMING BLOCK FINALIZED' messages"
echo ""
echo "⏹️  Stop Test:"
echo "   pkill -f 'q-api-server --port 809'"
echo ""
echo "Press Ctrl+C to stop monitoring, or run the monitor command above."
echo ""

# Monitor both nodes
tail -f testdag-node1.log testdag-node3.log | grep --line-buffered -E '📦|📡|🎯|FINALIZED|height|ERROR'
