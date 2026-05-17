#!/bin/bash

echo "🔬 Enhanced Post-Connection Network Test"
echo "======================================="

# Find the binary
BINARY=""
if [ -f "./target/x86_64-unknown-linux-gnu/release/q-api-server" ]; then
    BINARY="./target/x86_64-unknown-linux-gnu/release/q-api-server"
elif [ -f "./target/release/q-api-server" ]; then
    BINARY="./target/release/q-api-server"
elif [ -f "./target/debug/q-api-server" ]; then
    BINARY="./target/debug/q-api-server"
else
    echo "❌ No q-api-server binary found!"
    exit 1
fi

echo "📁 Using binary: $BINARY"

# Start 4 nodes with better spacing for stability
PIDS=()
PORTS=(18041 18042 18043 18044)
NODE_IDS=()

echo "🚀 Starting 4 nodes with enhanced monitoring..."

for i in "${!PORTS[@]}"; do
    NODE_ID="enhanced-node-$((i+1))"
    PORT="${PORTS[$i]}"
    DATA_DIR="/tmp/q-enhanced-node-$((i+1))"
    
    echo "🔧 Starting $NODE_ID on port $PORT"
    
    # Clean up any existing data directory
    rm -rf "$DATA_DIR" 2>/dev/null
    mkdir -p "$DATA_DIR"
    
    # Start with minimal but informative logging
    RUST_LOG=warn SKIP_BITCOIN=1 SKIP_DNS=1 \
    Q_DB_PATH="$DATA_DIR/db" Q_HOT_DB_PATH="$DATA_DIR/hot" \
    $BINARY --node-id "$NODE_ID" --port "$PORT" \
    > "$DATA_DIR/node.log" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    NODE_IDS+=("$NODE_ID")
    echo "  📍 PID: $PID (log: $DATA_DIR/node.log)"
    
    # Stagger startup to avoid resource conflicts
    sleep 2
done

echo "⏳ Waiting for all nodes to fully initialize..."
sleep 15

# Function to test node health
test_node_health() {
    local port=$1
    local node_name=$2
    
    local url="http://127.0.0.1:$port/api/v1/health"
    if curl -s --max-time 3 "$url" > /dev/null 2>&1; then
        echo "✅ $node_name (port $port) is healthy"
        return 0
    else
        echo "❌ $node_name (port $port) is not responding"
        return 1
    fi
}

# Function to get detailed node status
get_node_status() {
    local port=$1
    local node_name=$2
    
    local url="http://127.0.0.1:$port/api/v1/status"
    local response=$(curl -s --max-time 5 "$url" 2>/dev/null)
    
    if [ -n "$response" ]; then
        echo "📊 $node_name Status:"
        echo "$response" | python3 -m json.tool 2>/dev/null | head -20 || echo "$response"
        echo ""
    else
        echo "⚠️  $node_name: No status response"
    fi
}

# Function to test network analytics
get_network_analytics() {
    local port=$1
    local node_name=$2
    
    local url="http://127.0.0.1:$port/api/v1/network/analytics"
    local response=$(curl -s --max-time 5 "$url" 2>/dev/null)
    
    if [ -n "$response" ]; then
        echo "📈 $node_name Network Analytics:"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
        echo ""
    fi
}

# Function to get active peers
get_active_peers() {
    local port=$1
    local node_name=$2
    
    local url="http://127.0.0.1:$port/api/v1/network/active-peers"
    local response=$(curl -s --max-time 5 "$url" 2>/dev/null)
    
    if [ -n "$response" ]; then
        echo "🌐 $node_name Active Peers:"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
        echo ""
    fi
}

# Phase 1: Initial Health Check
echo ""
echo "🔍 Phase 1: Initial Health Assessment"
echo "===================================="

HEALTHY_NODES=()
for i in "${!PORTS[@]}"; do
    if test_node_health "${PORTS[$i]}" "${NODE_IDS[$i]}"; then
        HEALTHY_NODES+=($i)
    fi
done

echo "📊 Healthy nodes: ${#HEALTHY_NODES[@]}/4"

if [ ${#HEALTHY_NODES[@]} -eq 0 ]; then
    echo "❌ No healthy nodes found. Exiting."
    exit 1
fi

# Phase 2: Detailed Status Analysis
echo ""
echo "📋 Phase 2: Detailed Node Status Analysis"
echo "========================================="

for idx in "${HEALTHY_NODES[@]}"; do
    get_node_status "${PORTS[$idx]}" "${NODE_IDS[$idx]}"
done

# Phase 3: Network Analytics Before Connection
echo ""
echo "📈 Phase 3: Pre-Connection Network Analytics"
echo "==========================================="

for idx in "${HEALTHY_NODES[@]}"; do
    get_network_analytics "${PORTS[$idx]}" "${NODE_IDS[$idx]}"
    get_active_peers "${PORTS[$idx]}" "${NODE_IDS[$idx]}"
done

# Phase 4: Attempt Manual Node Connections
echo ""
echo "🔗 Phase 4: Manual Node Connection Attempts"
echo "==========================================="

if [ ${#HEALTHY_NODES[@]} -ge 2 ]; then
    # Try to connect first healthy node to second healthy node
    idx1=${HEALTHY_NODES[0]}
    idx2=${HEALTHY_NODES[1]}
    port1=${PORTS[$idx1]}
    port2=${PORTS[$idx2]}
    
    echo "🔄 Attempting to connect ${NODE_IDS[$idx1]} -> ${NODE_IDS[$idx2]}"
    
    # Try mesh connection API
    connection_data="{\"target\":\"127.0.0.1:$port2\",\"node_id\":\"${NODE_IDS[$idx2]}\"}"
    connect_url="http://127.0.0.1:$port1/api/mesh/connect"
    
    echo "📡 Mesh connection attempt..."
    mesh_result=$(curl -s --max-time 10 -X POST -H "Content-Type: application/json" -d "$connection_data" "$connect_url" 2>/dev/null)
    if [ -n "$mesh_result" ]; then
        echo "✅ Mesh connection response: $mesh_result"
    else
        echo "⚠️  Mesh connection no response"
    fi
    
    # Try bridge connection API
    bridge_url="http://127.0.0.1:$port1/api/v1/bitcoin/bridge/connect/${NODE_IDS[$idx2]}"
    bridge_data="{\"target_address\":\"127.0.0.1:$port2\",\"node_id\":\"${NODE_IDS[$idx2]}\"}"
    
    echo "🌉 Bridge connection attempt..."
    bridge_result=$(curl -s --max-time 10 -X POST -H "Content-Type: application/json" -d "$bridge_data" "$bridge_url" 2>/dev/null)
    if [ -n "$bridge_result" ]; then
        echo "✅ Bridge connection response: $bridge_result"
    else
        echo "⚠️  Bridge connection no response"
    fi
    
    # Wait for connection to establish
    echo "⏳ Waiting for connection to stabilize..."
    sleep 5
fi

# Phase 5: Post-Connection Analysis
echo ""
echo "🔬 Phase 5: Post-Connection Network Analysis"
echo "==========================================="

for idx in "${HEALTHY_NODES[@]}"; do
    echo "📊 Post-connection analysis for ${NODE_IDS[$idx]}:"
    get_active_peers "${PORTS[$idx]}" "${NODE_IDS[$idx]}"
    get_network_analytics "${PORTS[$idx]}" "${NODE_IDS[$idx]}"
done

# Phase 6: Transaction Propagation Test
echo ""
echo "🧪 Phase 6: Transaction Propagation Testing"
echo "=========================================="

if [ ${#HEALTHY_NODES[@]} -ge 1 ]; then
    # Send transactions from different nodes
    for i in $(seq 1 3); do
        idx=${HEALTHY_NODES[0]}
        port=${PORTS[$idx]}
        
        echo "💸 Sending transaction #$i from ${NODE_IDS[$idx]}"
        
        tx_data="{\"from\":\"test_sender_$i\",\"to\":\"test_receiver_$i\",\"amount\":$((1000 + i)),\"nonce\":$i}"
        tx_url="http://127.0.0.1:$port/api/v1/transactions"
        
        tx_result=$(curl -s --max-time 10 -X POST -H "Content-Type: application/json" -d "$tx_data" "$tx_url" 2>/dev/null)
        if [ -n "$tx_result" ]; then
            echo "✅ Transaction #$i response: $tx_result"
        else
            echo "⚠️  Transaction #$i failed"
        fi
        
        sleep 2
    done
    
    # Wait for propagation
    echo "⏳ Waiting for transaction propagation..."
    sleep 5
    
    # Check recent transactions on all nodes
    echo ""
    echo "📨 Recent Transactions Check:"
    for idx in "${HEALTHY_NODES[@]}"; do
        port=${PORTS[$idx]}
        tx_url="http://127.0.0.1:$port/api/v1/transactions/recent"
        
        echo "📋 Recent transactions on ${NODE_IDS[$idx]}:"
        recent_txs=$(curl -s --max-time 5 "$tx_url" 2>/dev/null)
        if [ -n "$recent_txs" ]; then
            echo "$recent_txs" | python3 -m json.tool 2>/dev/null | head -30 || echo "$recent_txs"
        else
            echo "⚠️  No recent transactions data"
        fi
        echo ""
    done
fi

# Phase 7: Consensus and Mesh Status
echo ""
echo "🎯 Phase 7: Consensus and Mesh Network Status"
echo "============================================="

for idx in "${HEALTHY_NODES[@]}"; do
    port=${PORTS[$idx]}
    node_id=${NODE_IDS[$idx]}
    
    echo "🔍 Deep dive for $node_id:"
    
    # Mesh status
    mesh_url="http://127.0.0.1:$port/api/mesh/status"
    mesh_status=$(curl -s --max-time 5 "$mesh_url" 2>/dev/null)
    if [ -n "$mesh_status" ]; then
        echo "🌐 Mesh Status:"
        echo "$mesh_status" | python3 -m json.tool 2>/dev/null || echo "$mesh_status"
    fi
    
    # Network topology
    topology_url="http://127.0.0.1:$port/api/v1/network/topology"
    topology=$(curl -s --max-time 5 "$topology_url" 2>/dev/null)
    if [ -n "$topology" ]; then
        echo "🗺️  Network Topology:"
        echo "$topology" | python3 -m json.tool 2>/dev/null | head -20 || echo "$topology"
    fi
    
    # Discovery stats
    discovery_url="http://127.0.0.1:$port/api/v1/network/discovery/stats"
    discovery=$(curl -s --max-time 5 "$discovery_url" 2>/dev/null)
    if [ -n "$discovery" ]; then
        echo "🔍 Discovery Stats:"
        echo "$discovery" | python3 -m json.tool 2>/dev/null || echo "$discovery"
    fi
    
    echo "----------------------------------------"
done

# Phase 8: Log Analysis
echo ""
echo "📝 Phase 8: Log Analysis Summary"
echo "==============================="

for i in "${!PORTS[@]}"; do
    data_dir="/tmp/q-enhanced-node-$((i+1))"
    if [ -f "$data_dir/node.log" ]; then
        echo "📄 Last 10 lines of ${NODE_IDS[$i]} log:"
        tail -10 "$data_dir/node.log" | sed 's/^/  /'
        echo ""
    fi
done

# Results Summary
echo ""
echo "📊 Final Results Summary"
echo "======================="
echo "🔢 Total nodes started: ${#PORTS[@]}"
echo "✅ Healthy nodes: ${#HEALTHY_NODES[@]}"
echo "🌐 Connection attempts made"
echo "💸 Transaction tests completed"
echo "📈 Network analytics gathered"

# Cleanup
echo ""
echo "🛑 Shutting down all nodes..."
for PID in "${PIDS[@]}"; do
    if kill "$PID" 2>/dev/null; then
        echo "✅ Stopped PID $PID"
    else
        echo "⚠️  PID $PID already stopped"
    fi
done

sleep 3

echo "🧹 Cleaning up data directories..."
for i in {1..4}; do
    rm -rf "/tmp/q-enhanced-node-$i" 2>/dev/null
done

echo ""
echo "🎉 Enhanced Post-Connection Test Completed!"
echo "Full network behavior analysis complete."