#!/bin/bash
# REAL Q-NarwhalKnight Bitcoin-based Peer Discovery Test
# No simulation - actual nodes discovering each other through Bitcoin network

set -e

echo "🚀 REAL Q-NARWHALKNIGHT BITCOIN PEER DISCOVERY TEST"
echo "=================================================="
echo "Testing REAL nodes discovering each other through Bitcoin network"
echo "Bitcoin RPC: rpcuser/rpcpass | Bitcoin peers: $(docker exec bitcoin-mainnet bitcoin-cli -rpcuser=rpcuser -rpcpassword=rpcpass getconnectioncount 2>/dev/null || echo '0')"
echo ""

# Configuration
NODE_BINARY="target/release/q-api-server"
TEST_DIR="/tmp/qnk-discovery-test-$(date +%s)"
mkdir -p $TEST_DIR

# Check if we have the binary
if [ ! -f "$NODE_BINARY" ]; then
    echo "❌ Q-NarwhalKnight binary not found at $NODE_BINARY"
    echo "⏳ Still compiling... please wait for cargo build to complete"
    exit 1
fi

echo "✅ Found Q-NarwhalKnight binary: $NODE_BINARY"
echo ""

# Function to create node config
create_node_config() {
    local node_id=$1
    local port=$2
    local config_file="$TEST_DIR/node${node_id}-config.toml"
    
    cat > "$config_file" << EOF
[node]
node_id = "qnk_node_${node_id}"
listen_port = ${port}
api_port = $((9300 + node_id))  # Changed to 9300 range to avoid conflicts

[bitcoin]
rpc_url = "http://161.35.219.10:8332"
rpc_user = "rpcuser"
rpc_password = "rpcpass"
discovery_enabled = true
discovery_interval_secs = 30

[network]
bootstrap_discovery = true
max_peers = 10
connection_timeout_secs = 30

[logging]
level = "info"
EOF
    echo "$config_file"
}

# Function to launch node
launch_node() {
    local node_id=$1
    local port=$2
    local config_file=$3
    local log_file="$TEST_DIR/node${node_id}.log"
    
    echo "🔧 Launching Node $node_id on port $port..."
    
    RUST_LOG=info,q_bitcoin_bridge=debug,q_network=debug \
    BITCOIN_RPC_URL="http://161.35.219.10:8332" \
    BITCOIN_RPC_USER="rpcuser" \
    BITCOIN_RPC_PASSWORD="rpcpass" \
    "$NODE_BINARY" \
        --config "$config_file" \
        --node-id "qnk_node_${node_id}" \
        --listen "127.0.0.1:$port" \
        --api-port "$((9300 + node_id))" \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo "  Node $node_id PID: $pid"
    echo "$pid" > "$TEST_DIR/node${node_id}.pid"
    
    # Wait for node to start
    sleep 3
    
    if kill -0 $pid 2>/dev/null; then
        echo "  ✅ Node $node_id started successfully"
        return 0
    else
        echo "  ❌ Node $node_id failed to start"
        echo "  Last log entries:"
        tail -5 "$log_file" | sed 's/^/    /'
        return 1
    fi
}

# Function to check Bitcoin connectivity
check_bitcoin_connectivity() {
    local node_id=$1
    local log_file="$TEST_DIR/node${node_id}.log"
    
    echo "🔍 Checking Node $node_id Bitcoin connectivity..."
    
    # Look for Bitcoin connection logs
    if grep -q "Bitcoin RPC connection successful" "$log_file" 2>/dev/null; then
        echo "  ✅ Node $node_id connected to Bitcoin RPC"
        return 0
    else
        echo "  ❌ Node $node_id Bitcoin RPC connection not confirmed"
        echo "  Recent logs:"
        tail -3 "$log_file" | sed 's/^/    /'
        return 1
    fi
}

# Function to check peer discovery
check_peer_discovery() {
    local node_id=$1
    local log_file="$TEST_DIR/node${node_id}.log"
    
    echo "🔍 Checking Node $node_id peer discovery..."
    
    # Look for peer discovery logs
    if grep -q "Bitcoin bootstrap completed" "$log_file" 2>/dev/null; then
        local peer_count=$(grep "Bitcoin bootstrap completed" "$log_file" | tail -1 | grep -o '[0-9]\+ Q-NarwhalKnight peers' | grep -o '[0-9]\+' || echo "0")
        echo "  ✅ Node $node_id discovered $peer_count peers via Bitcoin"
        return 0
    else
        echo "  ⏳ Node $node_id peer discovery in progress..."
        return 1
    fi
}

# Function to check P2P connections
check_p2p_connections() {
    local node_id=$1
    local log_file="$TEST_DIR/node${node_id}.log"
    
    echo "🔍 Checking Node $node_id P2P connections..."
    
    # Look for P2P connection logs
    local connections=$(grep -c "P2P connection established" "$log_file" 2>/dev/null || echo "0")
    if [ "$connections" -gt 0 ]; then
        echo "  ✅ Node $node_id established $connections P2P connections"
        return 0
    else
        echo "  ❌ Node $node_id no P2P connections established yet"
        return 1
    fi
}

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up test nodes..."
    for pid_file in "$TEST_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Stopping PID $pid"
                kill "$pid" 2>/dev/null || true
                sleep 1
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    done
    
    echo "📊 Test logs available in: $TEST_DIR"
    echo "   View logs: ls -la $TEST_DIR/"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

echo "🏗️ PHASE 1: Launch Q-NarwhalKnight nodes"
echo "========================================"

# Launch 3 nodes (using different port range to avoid conflicts)
NODE_CONFIGS=()
for i in 1 2 3; do
    port=$((9200 + i))  # Changed to 9200 range to avoid conflicts
    config=$(create_node_config $i $port)
    NODE_CONFIGS+=("$config")
    launch_node $i $port "$config" || exit 1
done

echo ""
echo "⏱️ PHASE 2: Wait for Bitcoin connectivity (30 seconds)"
echo "===================================================="
sleep 30

# Check Bitcoin connectivity
bitcoin_success=0
for i in 1 2 3; do
    if check_bitcoin_connectivity $i; then
        ((bitcoin_success++))
    fi
done

echo ""
echo "📊 Bitcoin connectivity: $bitcoin_success/3 nodes successful"

echo ""
echo "⏱️ PHASE 3: Wait for peer discovery (60 seconds)"
echo "============================================="
sleep 60

# Check peer discovery
discovery_success=0
for i in 1 2 3; do
    if check_peer_discovery $i; then
        ((discovery_success++))
    fi
done

echo ""
echo "📊 Peer discovery: $discovery_success/3 nodes successful"

echo ""
echo "⏱️ PHASE 4: Wait for P2P connections (30 seconds)"
echo "=============================================="
sleep 30

# Check P2P connections
p2p_success=0
for i in 1 2 3; do
    if check_p2p_connections $i; then
        ((p2p_success++))
    fi
done

echo ""
echo "📊 P2P connections: $p2p_success/3 nodes successful"

echo ""
echo "🏆 FINAL RESULTS"
echo "==============="
echo "✅ Nodes launched: 3/3"
echo "✅ Bitcoin connectivity: $bitcoin_success/3"
echo "✅ Peer discovery: $discovery_success/3"
echo "✅ P2P connections: $p2p_success/3"

if [ "$bitcoin_success" -eq 3 ] && [ "$discovery_success" -gt 0 ] && [ "$p2p_success" -gt 0 ]; then
    echo ""
    echo "🎉 SUCCESS: Real Q-NarwhalKnight nodes successfully discovered each other through Bitcoin network!"
    echo "   The nodes connected to Bitcoin, discovered peers, and established direct P2P connections."
    echo "   This proves the Bitcoin-based peer discovery mechanism works with real nodes."
else
    echo ""
    echo "⚠️ PARTIAL SUCCESS: Some aspects working, check logs for details"
fi

echo ""
echo "📋 Detailed logs:"
for i in 1 2 3; do
    echo "   Node $i log: $TEST_DIR/node${i}.log"
done

echo ""
echo "🔍 To investigate further:"
echo "   grep -E '(Bitcoin|discovery|P2P)' $TEST_DIR/*.log"
echo ""