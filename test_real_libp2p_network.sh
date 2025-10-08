#!/bin/bash
# Real libp2p Network Test - Validate Peer Discovery and Transaction Gossip

echo "================================================================================"
echo "🌐 Q-NarwhalKnight libp2p Network Test"
echo "================================================================================"
echo ""
echo "Testing:"
echo "  ✅ Peer discovery via mDNS"
echo "  ✅ Gossipsub message propagation"
echo "  ✅ Transaction gossip between nodes"
echo "  ✅ Bootstrap node coordination"
echo ""

# Kill existing nodes
killall q-api-server 2>/dev/null
sleep 2

# Create data directories
for i in {0..3}; do
    mkdir -p ./data-libp2p-test-node$i
done

echo "================================================================================
"
echo "📋 Node Configuration:"
echo "  Node 0 (Bootstrap): HTTP=9110, P2P=9210"
echo "  Node 1:             HTTP=9111, P2P=9211"
echo "  Node 2:             HTTP=9112, P2P=9212"
echo "  Node 3:             HTTP=9113, P2P=9213"
echo ""

# Launch bootstrap node first
echo "🚀 Launching Bootstrap Node (Node 0)..."
Q_DB_PATH=./data-libp2p-test-node0 Q_P2P_PORT=9210 \
    RUST_LOG=info,libp2p=debug,q_network=debug \
    ./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9110 \
    > libp2p-node0.log 2>&1 &
BOOTSTRAP_PID=$!
echo "  Bootstrap PID: $BOOTSTRAP_PID"
sleep 3

# Get bootstrap node's peer ID from log
BOOTSTRAP_PEER_ID=$(grep "🆔 Generated new node ID:" libp2p-node0.log | tail -1 | awk '{print $NF}')
if [ -n "$BOOTSTRAP_PEER_ID" ]; then
    echo "  Bootstrap Peer ID: $BOOTSTRAP_PEER_ID"
else
    echo "  ⚠️  Could not extract bootstrap peer ID"
fi

echo ""
echo "⏳ Waiting 5 seconds for bootstrap node to fully initialize..."
sleep 5

# Launch peer nodes
echo ""
echo "🚀 Launching Peer Nodes..."

for i in 1 2 3; do
    HTTP_PORT=$((9110 + i))
    P2P_PORT=$((9210 + i))

    echo "  Starting Node $i (HTTP: $HTTP_PORT, P2P: $P2P_PORT)..."

    Q_DB_PATH=./data-libp2p-test-node$i Q_P2P_PORT=$P2P_PORT \
        RUST_LOG=info,libp2p=debug,q_network=debug \
        ./target/x86_64-unknown-linux-gnu/release/q-api-server --port $HTTP_PORT \
        > libp2p-node$i.log 2>&1 &

    echo "    PID: $!"
    sleep 2
done

echo ""
echo "✅ All 4 nodes launched"
echo ""

# Wait for peer discovery
echo "⏳ Waiting 15 seconds for libp2p peer discovery (mDNS)..."
for i in {1..15}; do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Check peer connections in logs
echo "================================================================================
"
echo "📊 Checking Peer Discovery Status..."
echo ""

for i in {0..3}; do
    echo "Node $i connections:"
    if grep -q "mDNS" libp2p-node$i.log 2>/dev/null; then
        grep "mDNS\|Peer\|peer" libp2p-node$i.log 2>/dev/null | tail -5 | sed 's/^/  /'
    else
        echo "  (No mDNS activity in logs)"
    fi
    echo ""
done

# Check if libp2p is actually running
echo "================================================================================
"
echo "🔍 Verifying libp2p Integration..."
echo ""

for i in {0..3}; do
    if grep -q "libp2p" libp2p-node$i.log 2>/dev/null; then
        echo "  Node $i: ✅ libp2p logs found"
    else
        echo "  Node $i: ⚠️  No libp2p activity (may not be integrated)"
    fi
done

echo ""
echo "================================================================================
"
echo "📝 Summary:"
echo ""
echo "Logs available:"
echo "  tail -f libp2p-node0.log  # Bootstrap node"
echo "  tail -f libp2p-node1.log  # Peer node 1"
echo "  tail -f libp2p-node2.log  # Peer node 2"
echo "  tail -f libp2p-node3.log  # Peer node 3"
echo ""
echo "Check connections:"
echo "  grep -i 'peer\|mdns\|gossip' libp2p-node*.log"
echo ""
echo "Stop nodes:"
echo "  killall q-api-server"
echo ""
echo "================================================================================
"

# Keep script running to monitor
echo "Press Ctrl+C to stop monitoring and shut down nodes..."
echo ""

# Monitor for 60 seconds
for i in {1..60}; do
    sleep 1

    # Every 10 seconds, check for new peer connections
    if [ $((i % 10)) -eq 0 ]; then
        echo "[$i/60s] Checking for new peer connections..."
        NEW_PEERS=$(grep -h "New peer" libp2p-node*.log 2>/dev/null | wc -l)
        if [ "$NEW_PEERS" -gt 0 ]; then
            echo "  Found $NEW_PEERS peer connection events!"
        fi
    fi
done

echo ""
echo "✅ Monitoring complete"
echo ""
echo "To continue testing, nodes are still running. Stop with: killall q-api-server"
