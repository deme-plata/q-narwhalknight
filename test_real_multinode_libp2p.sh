#!/bin/bash
# Real multi-node libp2p bootstrap test using actual q-api-server instances
# Tests improved libp2p implementation with REAL production binaries

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Q-NarwhalKnight REAL Multi-Node libp2p Bootstrap Test     ║"
echo "║  Using PRODUCTION q-api-server binary with Kademlia + Gossipsub ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
NUM_NODES=${1:-5}
TEST_DURATION=${2:-60}
BOOTSTRAP_ADDR=${3:-"185.182.185.227:6881"}
BASE_PORT=9000
BASE_API_PORT=9100

echo "🎯 Test Configuration:"
echo "   Number of nodes: $NUM_NODES"
echo "   Test duration: ${TEST_DURATION}s"
echo "   Bootstrap server (libp2p): $BOOTSTRAP_ADDR"
echo "   Base P2P port: $BASE_PORT"
echo "   Base API port: $BASE_API_PORT"
echo ""

# Clean up any previous test data
echo "🧹 Cleaning up previous test data..."
rm -rf ./data-libp2p-node-*
rm -f real_multinode_test_*.log
rm -f real_multinode_test_pids.txt

# Check if q-api-server binary exists
if [ ! -f "./target/x86_64-unknown-linux-gnu/release/q-api-server" ]; then
    echo "❌ q-api-server binary not found!"
    echo "   Building it now with 10-hour timeout..."
    timeout 36000 cargo build --release --package q-api-server --bin q-api-server
    echo "✅ Build complete"
fi

echo ""

# Function to start a single node
start_node() {
    local node_id=$1
    local p2p_port=$((BASE_PORT + node_id - 1))
    local api_port=$((BASE_API_PORT + node_id - 1))
    local log_file="real_multinode_test_node_${node_id}.log"
    local data_dir="./data-libp2p-node-${node_id}"

    echo "🔵 Starting Node $node_id:"
    echo "   P2P Port: $p2p_port"
    echo "   API Port: $api_port"
    echo "   Data Dir: $data_dir"

    # Create data directory
    mkdir -p "$data_dir"

    # Set environment variables for this node
    export Q_DB_PATH="$data_dir"
    export Q_NODE_ID="libp2p-test-node-$node_id"
    export Q_P2P_PORT=$p2p_port
    export Q_BOOTSTRAP_PEERS="$BOOTSTRAP_ADDR"
    export Q_NARWHAL_BOOTSTRAP_NODE="$BOOTSTRAP_ADDR"
    export RUST_LOG="info,q_bep44_discovery=debug,libp2p=debug,q_network=debug"
    export RUST_BACKTRACE=1
    export SKIP_BITCOIN=1
    export SKIP_DNS=1

    # Run the REAL q-api-server binary
    ./target/x86_64-unknown-linux-gnu/release/q-api-server \
        --node-id "libp2p-test-node-$node_id" \
        --port "$api_port" \
        --production \
        > "$log_file" 2>&1 &

    local pid=$!

    echo "   PID: $pid"
    echo "   Log: $log_file"

    # Store PID for later cleanup
    echo $pid >> real_multinode_test_pids.txt

    # Wait a bit before starting next node to avoid overwhelming the system
    sleep 3
}

# Start all nodes
echo ""
echo "🚀 Launching $NUM_NODES REAL q-api-server instances..."
echo ""

rm -f real_multinode_test_pids.txt
for i in $(seq 1 $NUM_NODES); do
    start_node $i
done

echo ""
echo "✅ All $NUM_NODES nodes launched successfully"
echo ""
echo "⏱️  Running test for ${TEST_DURATION}s..."
echo "   Monitor logs with: tail -f real_multinode_test_node_*.log"
echo ""
echo "   Check node status:"
for i in $(seq 1 $NUM_NODES); do
    api_port=$((BASE_API_PORT + i - 1))
    echo "   curl http://localhost:$api_port/health"
done
echo ""

# Wait for nodes to initialize
echo "⏳ Waiting 10s for nodes to initialize..."
sleep 10

# Check node health
echo ""
echo "🏥 Checking node health status..."
healthy_nodes=0
for i in $(seq 1 $NUM_NODES); do
    api_port=$((BASE_API_PORT + i - 1))
    if curl -s http://localhost:$api_port/health | jq -e '.success == true' > /dev/null 2>&1; then
        echo "   Node $i (port $api_port): ✅ Healthy"
        healthy_nodes=$((healthy_nodes + 1))
    else
        echo "   Node $i (port $api_port): ❌ Not responding"
    fi
done

echo ""
echo "📊 Health Check: $healthy_nodes/$NUM_NODES nodes healthy"
echo ""

# Continue running for test duration
remaining_time=$((TEST_DURATION - 10))
for i in $(seq 1 $remaining_time); do
    printf "\r   Progress: [%3d/%3ds] " $((i + 10)) $TEST_DURATION
    sleep 1
done

echo ""
echo ""
echo "⏰ Test duration elapsed, analyzing results..."
echo ""

# Query peer discovery stats from each node
echo "╔═══════════════════════════════════════════════════════╗"
echo "║         Peer Discovery Statistics (Real-Time)         ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

total_peers_discovered=0
nodes_with_peers=0

for i in $(seq 1 $NUM_NODES); do
    api_port=$((BASE_API_PORT + i - 1))
    echo "Node $i (port $api_port):"

    # Try to get peer stats from API (if endpoint exists)
    peer_count=$(curl -s "http://localhost:$api_port/network/peers" 2>/dev/null | jq -r '.data.discovered_peers | length' 2>/dev/null || echo "0")

    # Fallback to log analysis
    log_file="real_multinode_test_node_${i}.log"
    if [ -f "$log_file" ]; then
        libp2p_connections=$(grep -c "CONNECTION ESTABLISHED\|Connection established with peer" "$log_file" 2>/dev/null || echo "0")
        peer_discoveries=$(grep -c "Identified peer\|PeerDiscovered" "$log_file" 2>/dev/null || echo "0")
        bootstrap_success=$(grep -c "Bootstrap successful\|✅.*Bootstrap" "$log_file" 2>/dev/null || echo "0")
        gossip_messages=$(grep -c "Received gossip message\|GossipsubEvent::Message" "$log_file" 2>/dev/null || echo "0")

        echo "  ├─ libp2p connections: $libp2p_connections"
        echo "  ├─ Peer discoveries: $peer_discoveries"
        echo "  ├─ Bootstrap: $([ "$bootstrap_success" -gt 0 ] && echo "✅ Success" || echo "❌ Failed")"
        echo "  └─ Gossip messages: $gossip_messages"

        if [ "$peer_discoveries" -gt 0 ] || [ "$libp2p_connections" -gt 0 ]; then
            nodes_with_peers=$((nodes_with_peers + 1))
            total_peers_discovered=$((total_peers_discovered + peer_discoveries + libp2p_connections))
        fi
    else
        echo "  └─ ⚠️  Log file not found"
    fi
    echo ""
done

# Stop all nodes
echo "🛑 Stopping all nodes..."
if [ -f real_multinode_test_pids.txt ]; then
    while read pid; do
        if kill -0 $pid 2>/dev/null; then
            echo "   Stopping PID $pid..."
            kill $pid 2>/dev/null || true
        fi
    done < real_multinode_test_pids.txt
    rm -f real_multinode_test_pids.txt
fi

sleep 2

# Final analysis from logs
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║              Final Test Results Analysis              ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║ Total Nodes:            $NUM_NODES                            ║"
echo "║ Nodes with Discoveries: $nodes_with_peers                            ║"
echo "║ Total Peer Discoveries: $total_peers_discovered                            ║"

success_rate=0
if [ "$NUM_NODES" -gt 0 ]; then
    success_rate=$((nodes_with_peers * 100 / NUM_NODES))
fi

echo "║ Success Rate:           ${success_rate}%                           ║"
echo "╠═══════════════════════════════════════════════════════╣"

# Evaluate test success
if [ "$success_rate" -ge 60 ]; then
    echo "║ Status: ✅ TEST PASSED                                ║"
    echo "║ Nodes successfully connected via libp2p bootstrap     ║"
    test_passed=true
else
    echo "║ Status: ⚠️  TEST NEEDS REVIEW                         ║"
    echo "║ Check logs for connection issues                      ║"
    test_passed=false
fi

echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Error analysis
echo "🔍 Key Events Analysis:"
echo ""

for i in $(seq 1 $NUM_NODES); do
    log_file="real_multinode_test_node_${i}.log"

    if [ -f "$log_file" ]; then
        echo "Node $i highlights:"

        # Show successful events
        grep -i "✅\|successfully\|established" "$log_file" 2>/dev/null | head -3 | sed 's/^/  ✅ /' || true

        # Show errors
        grep -i "error\|failed\|❌" "$log_file" 2>/dev/null | head -2 | sed 's/^/  ❌ /' || true

        echo ""
    fi
done

echo "📁 Log Files: real_multinode_test_node_*.log"
echo "📁 Data Directories: data-libp2p-node-*"
echo ""
echo "To inspect logs:"
echo "   tail -n 100 real_multinode_test_node_1.log"
echo "   grep -i 'libp2p\|bootstrap\|peer' real_multinode_test_node_*.log"
echo ""

# Final status
if [ "$test_passed" = true ]; then
    echo "✅ Real multi-node libp2p bootstrap test PASSED"
    exit 0
else
    echo "⚠️  Real multi-node libp2p bootstrap test completed with issues"
    echo "   Review logs for connection problems"
    exit 1
fi