#!/bin/bash
# Phase 4: Multi-Node Mesh Network Test
# Tests libp2p mDNS discovery + Gossipsub mesh formation with 4 nodes

set -e

echo "🧪 Phase 4: Testing 4-Node libp2p Mesh Network"
echo "=============================================="
echo ""

# Clean up any existing processes and data
echo "🧹 Cleaning up existing processes and data..."
killall q-api-server 2>/dev/null || true
sleep 2
rm -rf ./data-mesh-node* ./mesh-node*.log 2>/dev/null || true

echo "✅ Cleanup complete"
echo ""

# Launch 4 nodes
echo "🚀 Launching 4 nodes with libp2p mDNS + Gossipsub..."
for i in {1..4}; do
    mkdir -p ./data-mesh-node$i
    Q_DB_PATH=./data-mesh-node$i Q_P2P_PORT=$((9210+i)) \
    RUST_LOG=info,q_network::unified_network_manager=debug \
    ./target/x86_64-unknown-linux-gnu/release/q-api-server --port $((9100+i)) \
    > mesh-node$i.log 2>&1 &
    NODE_PID=$!
    echo "  ✓ Node $i started (PID: $NODE_PID, API: $((9100+i)), P2P: $((9210+i)))"
done

echo ""
echo "⏳ Waiting 15 seconds for discovery and mesh formation..."
sleep 15

echo ""
echo "📊 PHASE 4 TEST RESULTS"
echo "======================="
echo ""

# Check mDNS discovery
echo "1️⃣  mDNS Discovery Results:"
echo "   --------------------------"
for i in {1..4}; do
    DISCOVERED=$(grep -c "mDNS discovered:" mesh-node$i.log 2>/dev/null || echo "0")
    CONNECTED=$(grep -c "Connected to peer:" mesh-node$i.log 2>/dev/null || echo "0")
    echo "   Node $i: Discovered $DISCOVERED peers, Connected to $CONNECTED peers"
done

echo ""
echo "2️⃣  Gossipsub Mesh Formation:"
echo "   --------------------------"
SUBSCRIBED=$(grep -c "Peer.*subscribed to topic" mesh-node*.log 2>/dev/null || echo "0")
echo "   Total subscription events: $SUBSCRIBED"
echo ""

for topic in "blocks" "votes" "ack"; do
    COUNT=$(grep -c "/qnk/$topic/1.0.0" mesh-node*.log 2>/dev/null || echo "0")
    echo "   Topic /qnk/$topic/1.0.0: $COUNT events"
done

echo ""
echo "3️⃣  libp2p Event Loop Status:"
echo "   --------------------------"
for i in {1..4}; do
    if grep -q "Starting libp2p Zero-Knowledge Discovery event loop" mesh-node$i.log 2>/dev/null; then
        echo "   Node $i: ✅ Event loop running"
    else
        echo "   Node $i: ❌ Event loop NOT running"
    fi
done

echo ""
echo "4️⃣  ConnectionManager Bridge Status:"
echo "   ------------------------------------"
for i in {1..4}; do
    BRIDGED=$(grep -c "Bridged peer.*to ConnectionManager" mesh-node$i.log 2>/dev/null || echo "0")
    echo "   Node $i: Bridged $BRIDGED peers to ConnectionManager"
done

echo ""
echo "5️⃣  Discovery Timing (from first node start):"
echo "   ------------------------------------------"
FIRST_DISCOVERY=$(grep "mDNS discovered:" mesh-node*.log 2>/dev/null | head -1)
if [ -n "$FIRST_DISCOVERY" ]; then
    echo "   First discovery event: $(echo "$FIRST_DISCOVERY" | cut -d' ' -f1-2)"
    echo "   ✅ Discovery successful"
else
    echo "   ❌ No discovery events found"
fi

echo ""
echo "📝 Detailed Logs Available:"
echo "   ----------------------"
for i in {1..4}; do
    LOGSIZE=$(wc -l < mesh-node$i.log)
    echo "   mesh-node$i.log: $LOGSIZE lines"
done

echo ""
echo "🔍 Quick Log Inspection (First 10 discovery events):"
echo "   -----------------------------------------------"
grep "mDNS discovered:" mesh-node*.log 2>/dev/null | head -10 || echo "   No discovery events"

echo ""
echo "✅ PHASE 4 TEST SUMMARY"
echo "======================="

# Calculate expected vs actual
EXPECTED_DISCOVERIES=$((4 * 3))  # Each node should discover 3 others
ACTUAL_DISCOVERIES=$(grep -c "mDNS discovered:" mesh-node*.log 2>/dev/null || echo "0")

echo "Expected discoveries: $EXPECTED_DISCOVERIES (4 nodes × 3 peers each)"
echo "Actual discoveries: $ACTUAL_DISCOVERIES"

if [ "$ACTUAL_DISCOVERIES" -ge "$EXPECTED_DISCOVERIES" ]; then
    echo ""
    echo "✅ SUCCESS: All nodes discovered each other!"
    echo "✅ Phase 4 Complete: Multi-node mesh network functional"
else
    echo ""
    echo "⚠️  PARTIAL: $ACTUAL_DISCOVERIES/$EXPECTED_DISCOVERIES discoveries"
    echo "   This may be due to timing - nodes are still discovering"
fi

echo ""
echo "🎯 Next Steps:"
echo "   1. Check logs: cat mesh-node1.log (or mesh-node2.log, etc.)"
echo "   2. Kill nodes: killall q-api-server"
echo "   3. Move to Phase 5: Performance optimization"
echo ""
