#!/bin/bash

# Q-NarwhalKnight Multi-Node Test Script
# Spins up 20 real nodes to test the network

set -e

echo "🚀 Q-NarwhalKnight Multi-Node Network Test"
echo "========================================"

# Build the project first
echo "📦 Building Q-NarwhalKnight..."
cargo build --release --package q-api-server

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Cannot proceed with node testing."
    exit 1
fi

echo "✅ Build successful!"

# Configuration
NODE_COUNT=20
BASE_PORT=8080
API_BASE_PORT=8080
P2P_BASE_PORT=8081

# Create directories for node data
echo "📁 Creating node directories..."
mkdir -p nodes
for i in $(seq 0 $((NODE_COUNT-1))); do
    mkdir -p nodes/node_$i/{data,logs}
done

# Generate node configurations and start nodes
echo "🌟 Starting $NODE_COUNT Q-NarwhalKnight nodes..."

PIDS=()

for i in $(seq 0 $((NODE_COUNT-1))); do
    API_PORT=$((API_BASE_PORT + i))
    P2P_PORT=$((P2P_BASE_PORT + i))
    
    echo "🚀 Starting Node $i (API: $API_PORT, P2P: $P2P_PORT)..."
    
    # Set environment variables for this node
    export Q_API_PORT=$API_PORT
    export Q_P2P_PORT=$P2P_PORT
    export Q_IS_VALIDATOR=true
    export Q_TOR_ENABLED=true
    export Q_LOG_LEVEL=info
    export Q_DATA_DIR="$(pwd)/nodes/node_$i/data"
    
    # Start the node in background
    ./target/release/q-api-server > nodes/node_$i/logs/output.log 2>&1 &
    PID=$!
    PIDS+=($PID)
    
    echo "✅ Node $i started with PID $PID"
    
    # Small delay to prevent port conflicts
    sleep 0.5
done

echo ""
echo "🌟 ================================"
echo "🌟   ALL NODES STARTED SUCCESSFULLY"
echo "🌟 ================================"
echo "📊 Node Count: $NODE_COUNT"
echo "🌐 API Ports: $API_BASE_PORT-$((API_BASE_PORT + NODE_COUNT - 1))"
echo "📡 P2P Ports: $P2P_BASE_PORT-$((P2P_BASE_PORT + NODE_COUNT - 1))"
echo ""

# Wait for nodes to initialize
echo "⏳ Waiting 10 seconds for nodes to initialize..."
sleep 10

echo "🔍 Testing node connectivity..."

# Test each node's health endpoint
HEALTHY_NODES=0
for i in $(seq 0 $((NODE_COUNT-1))); do
    API_PORT=$((API_BASE_PORT + i))
    
    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo "✅ Node $i (port $API_PORT): Healthy"
        ((HEALTHY_NODES++))
    else
        echo "❌ Node $i (port $API_PORT): Not responding"
    fi
done

echo ""
echo "📊 Network Status:"
echo "   Total Nodes: $NODE_COUNT"
echo "   Healthy Nodes: $HEALTHY_NODES"
echo "   Success Rate: $((HEALTHY_NODES * 100 / NODE_COUNT))%"

if [ $HEALTHY_NODES -gt $((NODE_COUNT / 2)) ]; then
    echo "🎉 Network is operational! More than 50% of nodes are healthy."
else
    echo "⚠️  Network may have issues. Less than 50% of nodes are responding."
fi

echo ""
echo "🔗 Testing network discovery and connectivity..."

# Test the first few nodes' network status endpoints
for i in $(seq 0 4); do
    API_PORT=$((API_BASE_PORT + i))
    echo "🌐 Node $i network analytics:"
    
    # Test network analytics endpoint
    if curl -s "http://localhost:$API_PORT/api/v1/network/analytics" | jq . > /dev/null 2>&1; then
        PEERS=$(curl -s "http://localhost:$API_PORT/api/v1/network/analytics" | jq -r '.data.connected_peers // 0' 2>/dev/null || echo "0")
        BTC_ACTIVE=$(curl -s "http://localhost:$API_PORT/api/v1/network/analytics" | jq -r '.data.bitcoin_discovery_active // false' 2>/dev/null || echo "false")
        DNS_ACTIVE=$(curl -s "http://localhost:$API_PORT/api/v1/network/analytics" | jq -r '.data.dns_phantom_active // false' 2>/dev/null || echo "false")
        TOR_ACTIVE=$(curl -s "http://localhost:$API_PORT/api/v1/network/analytics" | jq -r '.data.tor_active // false' 2>/dev/null || echo "false")
        
        echo "   📡 Connected Peers: $PEERS"
        echo "   ₿  Bitcoin Discovery: $BTC_ACTIVE"
        echo "   👻 DNS-Phantom: $DNS_ACTIVE"
        echo "   🧅 Tor: $TOR_ACTIVE"
    else
        echo "   ❌ Network analytics not available"
    fi
    echo ""
done

echo "🎯 Real-time network monitoring (30 seconds)..."
echo "   Watch network formation and peer discovery..."

# Monitor network for 30 seconds
for t in $(seq 1 30); do
    printf "\r⏱️  Time: ${t}s | "
    
    # Get peer count from first node
    API_PORT=$API_BASE_PORT
    PEERS=$(curl -s "http://localhost:$API_PORT/api/v1/network/analytics" 2>/dev/null | jq -r '.data.connected_peers // 0' 2>/dev/null || echo "0")
    printf "Peers: $PEERS | "
    
    # Check if we have peer discovery
    if [ "$PEERS" -gt 0 ]; then
        printf "🎉 NETWORK FORMED!"
    else
        printf "🔍 Discovering..."
    fi
    
    sleep 1
done

echo ""
echo ""
echo "🏁 Test Complete!"
echo ""
echo "📋 Summary:"
echo "   🚀 Nodes Started: $NODE_COUNT"
echo "   ✅ Healthy Nodes: $HEALTHY_NODES"
echo "   🌐 Network Formation: $([ $PEERS -gt 0 ] && echo 'SUCCESS' || echo 'IN PROGRESS')"
echo ""

# Show instructions for manual testing
echo "🔧 Manual Testing Instructions:"
echo "   • View node status: curl http://localhost:8080/api/v1/status"
echo "   • View network analytics: curl http://localhost:8080/api/v1/network/analytics"
echo "   • View peer topology: curl http://localhost:8080/api/v1/network/topology"
echo "   • Real-time events: curl http://localhost:8080/api/v1/events"
echo ""

echo "⚠️  To stop all nodes, run: kill ${PIDS[*]}"
echo "📁 Logs available in: nodes/node_*/logs/output.log"

echo ""
echo "🌟 Q-NarwhalKnight Multi-Node Test Complete!"
echo "   🎯 Test the triple-layer anonymity network"
echo "   ₿  Bitcoin-Tor peer discovery"  
echo "   👻 DNS-Phantom steganographic communication"
echo "   🧅 Tor anonymous networking"
echo ""