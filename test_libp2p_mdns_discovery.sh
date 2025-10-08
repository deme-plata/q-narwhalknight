#!/bin/bash
# libp2p mDNS Discovery Test
# Tests 2-node zero-config peer discovery via mDNS

set -e

echo "================================================================================"
echo "🔍 Q-NarwhalKnight libp2p mDNS Discovery Test"
echo "================================================================================"
echo ""

# Configuration
NODE1_DB="./data-mdns-node1"
NODE2_DB="./data-mdns-node2"
NODE1_HTTP=9110
NODE2_HTTP=9120
NODE1_P2P=9211
NODE2_P2P=9212

# Cleanup
echo "📋 Pre-flight Cleanup:"
echo "  1. Stopping existing nodes..."
pkill -9 q-api-server 2>/dev/null || true
sleep 2

echo "  2. Cleaning data directories..."
rm -rf $NODE1_DB $NODE2_DB mdns-node1.log mdns-node2.log
mkdir -p $NODE1_DB $NODE2_DB

echo ""
echo "================================================================================"
echo "🚀 Launching 2-Node Network with libp2p mDNS Discovery"
echo "================================================================================"
echo ""

# Launch Node 1
echo "Node 1 Configuration:"
echo "  Database: $NODE1_DB"
echo "  HTTP Port: $NODE1_HTTP"
echo "  P2P Port: $NODE1_P2P"
echo "  Discovery: libp2p mDNS (zero-config)"
echo ""

Q_DB_PATH=$NODE1_DB \
Q_P2P_PORT=$NODE1_P2P \
RUST_LOG=info,q_network::unified_network_manager=debug,libp2p_mdns=trace,libp2p_swarm=debug \
./target/x86_64-unknown-linux-gnu/release/q-api-server \
  --port $NODE1_HTTP \
  > mdns-node1.log 2>&1 &

NODE1_PID=$!
echo "✅ Node 1 started (PID: $NODE1_PID)"
echo ""

# Wait for Node 1 to initialize
echo "⏳ Waiting 5 seconds for Node 1 to initialize libp2p..."
sleep 5

# Get Node 1 validator ID
echo "📝 Fetching Node 1 info..."
NODE1_ID=$(curl -s http://localhost:$NODE1_HTTP/node_id 2>/dev/null | jq -r '.node_id' || echo "unknown")
echo "   Node 1 ID: $NODE1_ID"
echo ""

# Launch Node 2
echo "Node 2 Configuration:"
echo "  Database: $NODE2_DB"
echo "  HTTP Port: $NODE2_HTTP"
echo "  P2P Port: $NODE2_P2P"
echo "  Discovery: libp2p mDNS (zero-config)"
echo ""

Q_DB_PATH=$NODE2_DB \
Q_P2P_PORT=$NODE2_P2P \
RUST_LOG=info,q_network::unified_network_manager=debug,libp2p_mdns=trace,libp2p_swarm=debug \
./target/x86_64-unknown-linux-gnu/release/q-api-server \
  --port $NODE2_HTTP \
  > mdns-node2.log 2>&1 &

NODE2_PID=$!
echo "✅ Node 2 started (PID: $NODE2_PID)"
echo ""

# Wait for mDNS discovery
echo "⏳ Waiting 10 seconds for mDNS peer discovery..."
for i in {1..10}; do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Get Node 2 validator ID
echo "📝 Fetching Node 2 info..."
NODE2_ID=$(curl -s http://localhost:$NODE2_HTTP/node_id 2>/dev/null | jq -r '.node_id' || echo "unknown")
echo "   Node 2 ID: $NODE2_ID"
echo ""

echo "================================================================================"
echo "📊 libp2p mDNS Discovery Results"
echo "================================================================================"
echo ""

# Check Node 1 logs for discovery
echo "🔍 Node 1 mDNS Events:"
if grep -q "Zero-Knowledge Discovery initialized" mdns-node1.log; then
    echo "   ✅ libp2p initialized"
else
    echo "   ❌ libp2p NOT initialized"
fi

if grep -q "Listening on" mdns-node1.log; then
    LISTEN=$(grep "Listening on" mdns-node1.log | head -1)
    echo "   $LISTEN"
else
    echo "   ⚠️  No listen address found"
fi

if grep -q "mDNS discovered" mdns-node1.log; then
    echo "   ✅ Peer discovered via mDNS!"
    grep "mDNS discovered" mdns-node1.log | sed 's/^/   /'
else
    echo "   ❌ No mDNS peer discovery"
fi

if grep -q "Connected to peer" mdns-node1.log; then
    echo "   ✅ Peer connection established!"
    grep "Connected to peer" mdns-node1.log | sed 's/^/   /'
else
    echo "   ⚠️  No peer connection logged"
fi

echo ""

# Check Node 2 logs for discovery
echo "🔍 Node 2 mDNS Events:"
if grep -q "Zero-Knowledge Discovery initialized" mdns-node2.log; then
    echo "   ✅ libp2p initialized"
else
    echo "   ❌ libp2p NOT initialized"
fi

if grep -q "Listening on" mdns-node2.log; then
    LISTEN=$(grep "Listening on" mdns-node2.log | head -1)
    echo "   $LISTEN"
else
    echo "   ⚠️  No listen address found"
fi

if grep -q "mDNS discovered" mdns-node2.log; then
    echo "   ✅ Peer discovered via mDNS!"
    grep "mDNS discovered" mdns-node2.log | sed 's/^/   /'
else
    echo "   ❌ No mDNS peer discovery"
fi

if grep -q "Connected to peer" mdns-node2.log; then
    echo "   ✅ Peer connection established!"
    grep "Connected to peer" mdns-node2.log | sed 's/^/   /'
else
    echo "   ⚠️  No peer connection logged"
fi

echo ""
echo "================================================================================"
echo "📈 Discovery Performance Metrics"
echo "================================================================================"
echo ""

# Calculate discovery time
if grep -q "mDNS discovered" mdns-node1.log && grep -q "mDNS discovered" mdns-node2.log; then
    echo "Discovery Status: ✅ SUCCESS"
    echo "Discovery Method: Zero-config mDNS (no bootstrap nodes required)"
    echo "Expected Discovery Time: <2 seconds"
    echo ""
fi

# Count total discovered peers
PEERS_NODE1=$(grep -c "mDNS discovered" mdns-node1.log 2>/dev/null || echo "0")
PEERS_NODE2=$(grep -c "mDNS discovered" mdns-node2.log 2>/dev/null || echo "0")
echo "Peers Discovered:"
echo "  Node 1: $PEERS_NODE1 peer(s)"
echo "  Node 2: $PEERS_NODE2 peer(s)"

echo ""
echo "================================================================================"
echo "🧪 Test Summary"
echo "================================================================================"
echo ""

# Node status
echo "Node Status:"
if ps -p $NODE1_PID > /dev/null 2>&1; then
    echo "  ✅ Node 1 running (PID: $NODE1_PID)"
else
    echo "  ❌ Node 1 crashed"
fi

if ps -p $NODE2_PID > /dev/null 2>&1; then
    echo "  ✅ Node 2 running (PID: $NODE2_PID)"
else
    echo "  ❌ Node 2 crashed"
fi

echo ""

# Overall result
DISCOVERY_SUCCESS=0
if grep -q "mDNS discovered" mdns-node1.log && grep -q "mDNS discovered" mdns-node2.log; then
    DISCOVERY_SUCCESS=1
fi

if [ "$DISCOVERY_SUCCESS" -eq 1 ]; then
    echo "🎉 TEST RESULT: ✅ SUCCESS"
    echo ""
    echo "libp2p mDNS peer discovery is working!"
    echo "Nodes discovered each other with ZERO configuration!"
else
    echo "❌ TEST RESULT: FAILED"
    echo ""
    echo "Peer discovery did not complete. Check logs for details:"
    echo "  Node 1: mdns-node1.log"
    echo "  Node 2: mdns-node2.log"
fi

echo ""
echo "================================================================================"
echo "📁 Log Files and Cleanup"
echo "================================================================================"
echo ""
echo "View logs:"
echo "  Node 1: tail -f mdns-node1.log"
echo "  Node 2: tail -f mdns-node2.log"
echo ""
echo "Search for key events:"
echo "  grep 'Zero-Knowledge Discovery' mdns-node*.log"
echo "  grep 'Listening on' mdns-node*.log"
echo "  grep 'mDNS discovered' mdns-node*.log"
echo "  grep 'Connected to peer' mdns-node*.log"
echo ""
echo "Stop nodes:"
echo "  kill $NODE1_PID $NODE2_PID"
echo ""
echo "API Endpoints:"
echo "  Node 1: http://localhost:$NODE1_HTTP"
echo "  Node 2: http://localhost:$NODE2_HTTP"
echo ""
echo "================================================================================"
echo "✅ libp2p mDNS Discovery Test Complete"
echo "================================================================================"
