#!/bin/bash
#
# Launch Distributed libp2p Validator Nodes for 1M TPS Testing
#

echo "================================================================================"
echo "🌟 DISTRIBUTED LIBP2P VALIDATOR NODE LAUNCHER"
echo "================================================================================"
echo "Launching 4 Q-NarwhalKnight validator nodes with libp2p gossipsub"
echo ""

# Kill any existing instances
killall q-api-server 2>/dev/null
sleep 2

# Create data directories
for i in {0..3}; do
    mkdir -p ./data-libp2p-node$i
done

# Launch validator nodes in background
echo "🚀 Launching validator nodes..."
echo ""

# Node 0
echo "  Starting Node 0 (HTTP: 9100, P2P: 9200)"
Q_DB_PATH=./data-libp2p-node0 Q_P2P_PORT=9200 \
    ./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9100 \
    > node0.log 2>&1 &
NODE0_PID=$!
sleep 1

# Node 1
echo "  Starting Node 1 (HTTP: 9101, P2P: 9201)"
Q_DB_PATH=./data-libp2p-node1 Q_P2P_PORT=9201 \
    ./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9101 \
    > node1.log 2>&1 &
NODE1_PID=$!
sleep 1

# Node 2
echo "  Starting Node 2 (HTTP: 9102, P2P: 9202)"
Q_DB_PATH=./data-libp2p-node2 Q_P2P_PORT=9202 \
    ./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9102 \
    > node2.log 2>&1 &
NODE2_PID=$!
sleep 1

# Node 3
echo "  Starting Node 3 (HTTP: 9103, P2P: 9203)"
Q_DB_PATH=./data-libp2p-node3 Q_P2P_PORT=9203 \
    ./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9103 \
    > node3.log 2>&1 &
NODE3_PID=$!

echo ""
echo "✅ All 4 nodes launched successfully"
echo ""
echo "Node PIDs:"
echo "  Node 0: $NODE0_PID"
echo "  Node 1: $NODE1_PID"
echo "  Node 2: $NODE2_PID"
echo "  Node 3: $NODE3_PID"
echo ""
echo "⏳ Waiting 10 seconds for libp2p peer discovery (mDNS)..."
sleep 10

echo ""
echo "✅ Peer discovery window complete"
echo ""
echo "📊 Check node logs:"
echo "  tail -f node0.log"
echo "  tail -f node1.log"
echo "  tail -f node2.log"
echo "  tail -f node3.log"
echo ""
echo "🧪 Run distributed TPS benchmark:"
echo "  cargo run --release --bin tps-benchmark -- --distributed"
echo ""
echo "🛑 To stop all nodes:"
echo "  kill $NODE0_PID $NODE1_PID $NODE2_PID $NODE3_PID"
echo "  # or"
echo "  killall q-api-server"
echo ""
echo "================================================================================"
echo "🌐 Distributed libp2p network ready for testing!"
echo "================================================================================"
