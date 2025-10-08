#!/bin/bash

# Test script for dynamic bootstrap peer discovery
# This demonstrates how nodes on different servers can discover each other

set -e

echo "==================================================================="
echo "Q-NarwhalKnight Dynamic Bootstrap Discovery Test"
echo "==================================================================="
echo ""
echo "This test demonstrates how to run nodes that can discover each other"
echo "across different servers without hardcoded bootstrap IPs."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to start a node
start_node() {
    local NODE_NAME=$1
    local PORT=$2
    local P2P_PORT=$3
    local BOOTSTRAP_PEERS=$4
    local DB_PATH=$5

    echo -e "${YELLOW}Starting node: ${NODE_NAME}${NC}"
    echo "  API Port: $PORT"
    echo "  P2P Port: $P2P_PORT"
    if [ -n "$BOOTSTRAP_PEERS" ]; then
        echo "  Bootstrap Peers: $BOOTSTRAP_PEERS"
    else
        echo "  Bootstrap Peers: None (this is the first bootstrap node)"
    fi
    echo ""

    # Set environment variables and run the node
    Q_DB_PATH="$DB_PATH" \
    Q_P2P_PORT="$P2P_PORT" \
    Q_BOOTSTRAP_PEERS="$BOOTSTRAP_PEERS" \
    cargo run --release --bin q-api-server -- \
        --node-id "$NODE_NAME" \
        --port "$PORT" \
        2>&1 | sed "s/^/[$NODE_NAME] /" &

    echo -e "${GREEN}Node $NODE_NAME started with PID $!${NC}"
    echo ""
}

echo "==================================================================="
echo "SCENARIO 1: Single Bootstrap Node (Running on Server A)"
echo "==================================================================="
echo ""
echo "On your first server (e.g., Server A), run:"
echo ""
echo -e "${GREEN}# Start the bootstrap node (no peers configured)${NC}"
echo 'export Q_BOOTSTRAP_PEERS=""'
echo 'export Q_DB_PATH="./data-bootstrap"'
echo 'export Q_P2P_PORT=8081'
echo './target/release/q-api-server --node-id bootstrap-node --port 8080'
echo ""
echo "This node will start without trying to connect to any hardcoded IP."
echo "It will listen on port 8081 for P2P connections."
echo ""

echo "==================================================================="
echo "SCENARIO 2: Additional Nodes (Running on Servers B, C, etc.)"
echo "==================================================================="
echo ""
echo "On your other servers, run nodes that connect to the bootstrap:"
echo ""
echo -e "${GREEN}# On Server B${NC}"
echo 'export Q_BOOTSTRAP_PEERS="<server-a-ip>:8081"'
echo 'export Q_DB_PATH="./data-node2"'
echo 'export Q_P2P_PORT=8082'
echo './target/release/q-api-server --node-id node2 --port 8080'
echo ""
echo -e "${GREEN}# On Server C${NC}"
echo 'export Q_BOOTSTRAP_PEERS="<server-a-ip>:8081,<server-b-ip>:8082"'
echo 'export Q_DB_PATH="./data-node3"'
echo 'export Q_P2P_PORT=8083'
echo './target/release/q-api-server --node-id node3 --port 8080'
echo ""

echo "==================================================================="
echo "SCENARIO 3: Mesh Network (Multiple Bootstrap Peers)"
echo "==================================================================="
echo ""
echo "For a resilient mesh network, configure multiple bootstrap peers:"
echo ""
echo 'export Q_BOOTSTRAP_PEERS="<server-a>:8081,<server-b>:8082,<server-c>:8083"'
echo ""
echo "Each node will try to connect to all bootstrap peers for redundancy."
echo ""

echo "==================================================================="
echo "SCENARIO 4: Using Public DHT (No Private Bootstrap)"
echo "==================================================================="
echo ""
echo "If you don't have any bootstrap nodes, nodes will use public DHT:"
echo ""
echo 'export Q_BOOTSTRAP_PEERS=""  # Empty = use public BitTorrent DHT'
echo './target/release/q-api-server --node-id public-dht-node --port 8080'
echo ""
echo "This will use router.bittorrent.com, dht.transmissionbt.com, etc."
echo ""

echo "==================================================================="
echo "LOCAL TEST EXAMPLE (All on one machine)"
echo "==================================================================="
echo ""
echo "Let's test locally with 3 nodes to verify the fix works:"
echo ""

# Check if binary exists
if [ ! -f "./target/release/q-api-server" ]; then
    echo -e "${YELLOW}Building q-api-server...${NC}"
    cargo build --release --bin q-api-server
fi

# Kill any existing nodes
echo "Cleaning up any existing nodes..."
pkill -f "q-api-server" 2>/dev/null || true
sleep 2

# Start bootstrap node (no peers)
echo -e "${GREEN}1. Starting bootstrap node (no configured peers)...${NC}"
start_node "bootstrap" 8080 8081 "" "./data-bootstrap-test"
sleep 5

# Start second node connecting to bootstrap
echo -e "${GREEN}2. Starting node2 (connecting to bootstrap at localhost:8081)...${NC}"
start_node "node2" 8082 8083 "127.0.0.1:8081" "./data-node2-test"
sleep 5

# Start third node connecting to both
echo -e "${GREEN}3. Starting node3 (connecting to bootstrap and node2)...${NC}"
start_node "node3" 8084 8085 "127.0.0.1:8081,127.0.0.1:8083" "./data-node3-test"

echo ""
echo -e "${GREEN}All nodes started!${NC}"
echo ""
echo "Watch the logs to see peer discovery in action:"
echo "- Bootstrap node should see connections from node2 and node3"
echo "- Node2 should connect to bootstrap and discover node3"
echo "- Node3 should connect to both bootstrap and node2"
echo ""
echo "Press Ctrl+C to stop all nodes"
echo ""

# Wait for user to stop
trap "echo 'Stopping all nodes...'; pkill -f 'q-api-server'; exit" INT
wait