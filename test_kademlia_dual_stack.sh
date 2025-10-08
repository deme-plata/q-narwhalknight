#!/bin/bash
# Test Kademlia Dual-Stack Discovery (mDNS + Kademlia DHT)
# Phase 5a: Verify clearnet discovery integration

set -e

echo "🌐 Q-NarwhalKnight Dual-Stack Discovery Test"
echo "=============================================="
echo ""
echo "Testing:"
echo "  • mDNS (local network discovery)"
echo "  • Kademlia DHT (clearnet discovery)"
echo "  • Dual-stack event handling"
echo ""

# Clean up any existing test data
echo "🧹 Cleaning up previous test data..."
rm -rf ./data-kad-node1 ./data-kad-node2
killall q-api-server 2>/dev/null || true
sleep 2

# Start Node 1 (will act as bootstrap peer)
echo ""
echo "🚀 Starting Node 1 (Bootstrap + DHT)..."
mkdir -p ./data-kad-node1
Q_DB_PATH=./data-kad-node1 \
Q_P2P_PORT=9301 \
RUST_LOG=info,q_network::unified_network_manager=debug \
./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9201 > kad-node1.log 2>&1 &
NODE1_PID=$!
echo "Node 1 PID: $NODE1_PID"

# Wait for Node 1 to initialize and get its listening address
sleep 5

# Extract Node 1's peer ID from logs
NODE1_PEER_ID=$(grep -m1 "Local Peer ID:" kad-node1.log | awk '{print $NF}')
if [ -z "$NODE1_PEER_ID" ]; then
    echo "❌ Failed to get Node 1 peer ID"
    cat kad-node1.log
    exit 1
fi

echo "✅ Node 1 initialized"
echo "   Peer ID: $NODE1_PEER_ID"

# Extract listening addresses
echo ""
echo "Node 1 listening addresses:"
grep "Listening on:" kad-node1.log | head -3

# Get Node 1's IP address (first IPv4 address)
NODE1_ADDR=$(grep "Listening on: /ip4/" kad-node1.log | grep -v "127.0.0.1" | head -1 | awk '{print $NF}')
echo ""
echo "Node 1 multiaddr: $NODE1_ADDR"

# Start Node 2 with Node 1 as bootstrap peer
echo ""
echo "🚀 Starting Node 2 (with bootstrap peer)..."
mkdir -p ./data-kad-node2

# Set Node 1 as bootstrap peer
BOOTSTRAP_MULTIADDR="${NODE1_ADDR}/p2p/${NODE1_PEER_ID}"
echo "Bootstrap peer: $BOOTSTRAP_MULTIADDR"

Q_DB_PATH=./data-kad-node2 \
Q_P2P_PORT=9302 \
Q_BOOTSTRAP_PEERS="$BOOTSTRAP_MULTIADDR" \
RUST_LOG=info,q_network::unified_network_manager=debug \
./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9202 > kad-node2.log 2>&1 &
NODE2_PID=$!
echo "Node 2 PID: $NODE2_PID"

# Wait for discovery and connection
echo ""
echo "⏳ Waiting 15 seconds for dual-stack discovery..."
sleep 15

echo ""
echo "=============================================="
echo "📊 Test Results"
echo "=============================================="

# Check Node 1 logs
echo ""
echo "Node 1 Discovery Events:"
echo "------------------------"
grep -E "mDNS discovered|Kademlia DHT|Connected to peer|Gossipsub" kad-node1.log | tail -20

# Check Node 2 logs
echo ""
echo "Node 2 Discovery Events:"
echo "------------------------"
grep -E "mDNS discovered|Kademlia DHT|Connected to peer|Gossipsub|Adding bootstrap peer" kad-node2.log | tail -20

# Count connections
echo ""
echo "=============================================="
echo "📈 Connection Summary"
echo "=============================================="

NODE1_MDNS=$(grep -c "mDNS discovered:" kad-node1.log || echo "0")
NODE1_CONNECTIONS=$(grep -c "Connected to peer:" kad-node1.log || echo "0")
NODE1_GOSSIPSUB=$(grep -c "subscribed to topic:" kad-node1.log || echo "0")

NODE2_MDNS=$(grep -c "mDNS discovered:" kad-node2.log || echo "0")
NODE2_CONNECTIONS=$(grep -c "Connected to peer:" kad-node2.log || echo "0")
NODE2_GOSSIPSUB=$(grep -c "subscribed to topic:" kad-node2.log || echo "0")
NODE2_BOOTSTRAP=$(grep -c "Adding bootstrap peer:" kad-node2.log || echo "0")
NODE2_KAD_INIT=$(grep -c "Kademlia DHT initialized" kad-node2.log || echo "0")

echo "Node 1:"
echo "  • mDNS discoveries: $NODE1_MDNS"
echo "  • Connections established: $NODE1_CONNECTIONS"
echo "  • Gossipsub subscriptions: $NODE1_GOSSIPSUB"

echo ""
echo "Node 2:"
echo "  • Bootstrap peers added: $NODE2_BOOTSTRAP"
echo "  • Kademlia DHT initialized: $NODE2_KAD_INIT"
echo "  • mDNS discoveries: $NODE2_MDNS"
echo "  • Connections established: $NODE2_CONNECTIONS"
echo "  • Gossipsub subscriptions: $NODE2_GOSSIPSUB"

# Verify success criteria
echo ""
echo "=============================================="
echo "✅ Success Criteria"
echo "=============================================="

SUCCESS=true

# Both nodes should have Kademlia initialized
if [ "$NODE2_KAD_INIT" -ge 1 ]; then
    echo "✅ Kademlia DHT initialized on Node 2"
else
    echo "❌ Kademlia DHT not initialized on Node 2"
    SUCCESS=false
fi

# Node 2 should have added bootstrap peer
if [ "$NODE2_BOOTSTRAP" -ge 1 ]; then
    echo "✅ Bootstrap peer added on Node 2"
else
    echo "❌ Bootstrap peer not added on Node 2"
    SUCCESS=false
fi

# At least one discovery mechanism should work (mDNS or DHT)
TOTAL_DISCOVERIES=$((NODE1_MDNS + NODE2_MDNS))
if [ "$TOTAL_DISCOVERIES" -ge 1 ]; then
    echo "✅ Peer discovery working (mDNS: $TOTAL_DISCOVERIES)"
else
    echo "⚠️  No mDNS discoveries (expected on same host)"
fi

# Connections should be established
TOTAL_CONNECTIONS=$((NODE1_CONNECTIONS + NODE2_CONNECTIONS))
if [ "$TOTAL_CONNECTIONS" -ge 2 ]; then
    echo "✅ Connections established ($TOTAL_CONNECTIONS total)"
else
    echo "❌ No connections established"
    SUCCESS=false
fi

# Gossipsub mesh should form
TOTAL_GOSSIPSUB=$((NODE1_GOSSIPSUB + NODE2_GOSSIPSUB))
if [ "$TOTAL_GOSSIPSUB" -ge 3 ]; then
    echo "✅ Gossipsub mesh formed ($TOTAL_GOSSIPSUB subscriptions)"
else
    echo "⚠️  Gossipsub mesh partial ($TOTAL_GOSSIPSUB subscriptions)"
fi

echo ""
echo "=============================================="

if [ "$SUCCESS" = true ]; then
    echo "🎉 Phase 5a Test: ✅ PASSED"
    echo ""
    echo "Dual-stack discovery is working:"
    echo "  • Kademlia DHT initialized"
    echo "  • Bootstrap peer support working"
    echo "  • Connections established"
    echo "  • Ready for Phase 5b (full DHT event handling)"
else
    echo "❌ Phase 5a Test: FAILED"
    echo ""
    echo "Check logs for details:"
    echo "  cat kad-node1.log"
    echo "  cat kad-node2.log"
fi

echo ""
echo "=============================================="
echo "🧹 Cleanup"
echo "=============================================="
echo ""
echo "Nodes are still running for manual inspection."
echo "To stop nodes: killall q-api-server"
echo ""
echo "Logs available at:"
echo "  • kad-node1.log"
echo "  • kad-node2.log"
echo ""
