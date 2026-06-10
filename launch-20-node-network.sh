#!/bin/bash

# Q-NarwhalKnight 20-Node Network Launcher
# This script launches 20 interconnected quantum consensus nodes

set -e

echo "🌟 ================================"
echo "🌟  LAUNCHING 20-NODE Q-NETWORK   "
echo "🌟 ================================"

# Build the project first
echo "🔨 Building Q-NarwhalKnight..."
cargo build --release -p q-api-server --quiet

# Configuration
BITCOIN_RPC_URL="http://161.35.219.10:8332"
BITCOIN_RPC_USER="rpcuser"
BITCOIN_RPC_PASSWORD="rpcpass"
BASE_API_PORT=8080
BASE_P2P_PORT=8181
BOOTSTRAP_PEERS=""

# Create logs directory
mkdir -p logs

echo "🚀 Starting 20 Q-NarwhalKnight nodes..."

# Function to generate node configuration
generate_node_config() {
    local node_id=$1
    local api_port=$((BASE_API_PORT + node_id))
    local p2p_port=$((BASE_P2P_PORT + node_id))
    
    # Build bootstrap peers list (exclude self)
    local bootstrap_list=""
    for i in {1..20}; do
        if [ $i -ne $node_id ]; then
            if [ -z "$bootstrap_list" ]; then
                bootstrap_list="127.0.0.1:$((BASE_P2P_PORT + i))"
            else
                bootstrap_list="$bootstrap_list,127.0.0.1:$((BASE_P2P_PORT + i))"
            fi
        fi
    done
    
    echo "Node-$node_id: API=$api_port P2P=$p2p_port"
    
    # Launch node in background
    Q_API_PORT=$api_port \
    Q_P2P_PORT=$p2p_port \
    Q_IS_VALIDATOR=true \
    Q_BOOTSTRAP_PEERS="$bootstrap_list" \
    Q_LOG_LEVEL=info \
    BITCOIN_RPC_URL="$BITCOIN_RPC_URL" \
    BITCOIN_RPC_USER="$BITCOIN_RPC_USER" \
    BITCOIN_RPC_PASSWORD="$BITCOIN_RPC_PASSWORD" \
    ./target/release/q-api-server > logs/node-$node_id.log 2>&1 &
    
    local pid=$!
    echo "$pid" > logs/node-$node_id.pid
    echo "✅ Node-$node_id started (PID: $pid, API: $api_port, P2P: $p2p_port)"
    
    # Brief startup delay
    sleep 2
}

# Launch all 20 nodes
for i in {1..20}; do
    generate_node_config $i
done

echo ""
echo "🌟 All 20 nodes launched!"
echo "🌟 API endpoints: http://localhost:8081-8100"
echo "🌟 P2P network: 127.0.0.1:8182-8201"
echo ""

# Wait for nodes to initialize
echo "⏳ Waiting for nodes to initialize..."
sleep 10

echo "🔍 Checking node status..."

# Check if nodes are responding
active_nodes=0
for i in {1..20}; do
    api_port=$((BASE_API_PORT + i))
    if curl -s -m 2 http://localhost:$api_port/health > /dev/null 2>&1; then
        active_nodes=$((active_nodes + 1))
        echo "✅ Node-$i (port $api_port) is healthy"
    else
        echo "❌ Node-$i (port $api_port) is not responding"
    fi
done

echo ""
echo "🌟 Network Status: $active_nodes/20 nodes active"
echo "🌟 Logs available in: logs/node-*.log"
echo "🌟 PIDs stored in: logs/node-*.pid"
echo ""
echo "🚀 20-Node Q-NarwhalKnight Network is LIVE!"
echo ""

# Show network topology
echo "📊 NETWORK TOPOLOGY:"
echo "===================="
for i in {1..20}; do
    api_port=$((BASE_API_PORT + i))
    p2p_port=$((BASE_P2P_PORT + i))
    printf "Node-%02d: API=http://localhost:%d P2P=127.0.0.1:%d\n" $i $api_port $p2p_port
done

echo ""
echo "🌟 Use 'bash shutdown-network.sh' to stop all nodes"