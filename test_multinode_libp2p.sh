#!/bin/bash
# Multi-node libp2p bootstrap test orchestration script
# Tests improved libp2p implementation with multiple nodes

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Q-NarwhalKnight Multi-Node libp2p Bootstrap Test           ║"
echo "║  Testing Kademlia DHT + Gossipsub peer discovery            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
NUM_NODES=${1:-5}
TEST_DURATION=${2:-60}
BOOTSTRAP_ADDR=${3:-"185.182.185.227:6881"}
BASE_PORT=7000

echo "🎯 Test Configuration:"
echo "   Number of nodes: $NUM_NODES"
echo "   Test duration: ${TEST_DURATION}s"
echo "   Bootstrap server: $BOOTSTRAP_ADDR"
echo "   Base port: $BASE_PORT"
echo ""

# Clean up any previous test data
echo "🧹 Cleaning up previous test data..."
rm -rf ./test-libp2p-node-*
rm -f multinode_test_*.log

# Build the test binary
echo "🔨 Building libp2p test binary..."
timeout 60 cargo build --release --package q-bep44-discovery --bin test_libp2p_bootstrap 2>&1 | grep -E "(Compiling|Finished|error)" || true
echo ""

# Function to start a single node
start_node() {
    local node_id=$1
    local port=$((BASE_PORT + node_id - 1))
    local log_file="multinode_test_node_${node_id}.log"

    echo "🔵 Starting Node $node_id on port $port..."

    # Set environment variables for this node
    export Q_NODE_ID=$node_id
    export Q_LISTEN_PORT=$port
    export Q_BOOTSTRAP_ADDR=$BOOTSTRAP_ADDR
    export RUST_LOG="info,libp2p=debug,q_bep44_discovery=debug"

    # Run the test binary in background
    ./target/release/test_libp2p_bootstrap > "$log_file" 2>&1 &
    local pid=$!

    echo "   PID: $pid"
    echo "   Log: $log_file"

    # Store PID for later cleanup
    echo $pid >> multinode_test_pids.txt

    # Wait a bit before starting next node
    sleep 2
}

# Start all nodes
echo ""
echo "🚀 Launching $NUM_NODES nodes..."
echo ""

rm -f multinode_test_pids.txt
for i in $(seq 1 $NUM_NODES); do
    start_node $i
done

echo ""
echo "✅ All $NUM_NODES nodes launched successfully"
echo ""
echo "⏱️  Running test for ${TEST_DURATION}s..."
echo "   Monitor logs with: tail -f multinode_test_node_*.log"
echo ""

# Wait for test duration
for i in $(seq 1 $TEST_DURATION); do
    printf "\r   Progress: [%3d/%3ds] " $i $TEST_DURATION
    sleep 1
done

echo ""
echo ""
echo "⏰ Test duration elapsed, analyzing results..."
echo ""

# Stop all nodes
echo "🛑 Stopping all nodes..."
if [ -f multinode_test_pids.txt ]; then
    while read pid; do
        if kill -0 $pid 2>/dev/null; then
            echo "   Stopping PID $pid..."
            kill $pid 2>/dev/null || true
        fi
    done < multinode_test_pids.txt
    rm -f multinode_test_pids.txt
fi

sleep 2

# Analyze logs and generate report
echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║          Multi-Node Test Results Analysis             ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

total_connections=0
total_peers_discovered=0
nodes_with_connections=0

for i in $(seq 1 $NUM_NODES); do
    log_file="multinode_test_node_${i}.log"

    if [ -f "$log_file" ]; then
        # Count connection established messages
        connections=$(grep -c "CONNECTION ESTABLISHED" "$log_file" 2>/dev/null || echo "0")

        # Count peer discoveries
        peers_discovered=$(grep -c "Identified peer" "$log_file" 2>/dev/null || echo "0")

        # Count bootstrap successes
        bootstrap_success=$(grep -c "Bootstrap successful" "$log_file" 2>/dev/null || echo "0")

        # Count gossip messages
        gossip_messages=$(grep -c "Received gossip message" "$log_file" 2>/dev/null || echo "0")

        echo "Node $i:"
        echo "  ├─ Connections: $connections"
        echo "  ├─ Peers discovered: $peers_discovered"
        echo "  ├─ Bootstrap: $([ "$bootstrap_success" -gt 0 ] && echo "✅ Success" || echo "❌ Failed")"
        echo "  └─ Gossip messages: $gossip_messages"
        echo ""

        total_connections=$((total_connections + connections))
        total_peers_discovered=$((total_peers_discovered + peers_discovered))

        if [ "$connections" -gt 0 ]; then
            nodes_with_connections=$((nodes_with_connections + 1))
        fi
    else
        echo "Node $i: ⚠️  Log file not found"
        echo ""
    fi
done

echo "╔═══════════════════════════════════════════════════════╗"
echo "║                    Summary Statistics                 ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║ Total Nodes:            $NUM_NODES                            ║"
echo "║ Nodes with Connections: $nodes_with_connections                            ║"
echo "║ Total Connections:      $total_connections                            ║"
echo "║ Total Peers Discovered: $total_peers_discovered                            ║"

avg_connections_per_node=0
if [ "$NUM_NODES" -gt 0 ]; then
    avg_connections_per_node=$((total_connections / NUM_NODES))
fi

echo "║ Avg Connections/Node:   $avg_connections_per_node                            ║"
echo "╠═══════════════════════════════════════════════════════╣"

# Evaluate test success
success_rate=$((nodes_with_connections * 100 / NUM_NODES))

if [ "$success_rate" -ge 80 ] && [ "$avg_connections_per_node" -ge 1 ]; then
    echo "║ Status: ✅ TEST PASSED                                ║"
    echo "║ $(printf '%.0f%%' $success_rate) of nodes successfully connected                 ║"
    test_passed=true
else
    echo "║ Status: ⚠️  TEST ISSUES                               ║"
    echo "║ Only $(printf '%.0f%%' $success_rate) of nodes connected successfully          ║"
    test_passed=false
fi

echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Detailed error analysis
echo "🔍 Error Analysis:"
echo ""

for i in $(seq 1 $NUM_NODES); do
    log_file="multinode_test_node_${i}.log"

    if [ -f "$log_file" ]; then
        errors=$(grep -i "error\|failed\|timeout" "$log_file" 2>/dev/null | head -3)

        if [ -n "$errors" ]; then
            echo "Node $i errors:"
            echo "$errors" | sed 's/^/  /'
            echo ""
        fi
    fi
done

# Offer log inspection
echo "📁 Log Files:"
echo "   multinode_test_node_*.log"
echo ""
echo "To inspect detailed logs:"
echo "   less multinode_test_node_1.log"
echo "   tail -n 100 multinode_test_node_*.log"
echo ""

# Final status
if [ "$test_passed" = true ]; then
    echo "✅ Multi-node libp2p bootstrap test PASSED"
    exit 0
else
    echo "⚠️  Multi-node libp2p bootstrap test FAILED"
    echo "   Check logs for details"
    exit 1
fi