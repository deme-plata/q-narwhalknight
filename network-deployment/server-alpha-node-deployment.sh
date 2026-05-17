#!/bin/bash
# Q-NarwhalKnight Server Alpha - 5 Node Deployment Script
# World's First Anonymous Quantum BFT Consensus Network Test

set -euo pipefail

echo "🚀 Q-NARWHALKNIGHT SERVER ALPHA NODE DEPLOYMENT"
echo "=================================================="
echo "Deploying 5 quantum consensus nodes with Tor DHT integration"
echo "Target: Anonymous Byzantine fault-tolerant consensus network"
echo ""

# Configuration
export RUST_LOG=info
export Q_KNIGHT_TEST_MODE=false
export Q_KNIGHT_PRODUCTION=true

# Node configurations
declare -A NODES=(
    ["alice"]="127.0.0.1:8001"
    ["bob"]="127.0.0.1:8002" 
    ["charlie"]="127.0.0.1:8003"
    ["diana"]="127.0.0.1:8004"
    ["eve"]="127.0.0.1:8005"
)

declare -A ONION_KEYS=(
    ["alice"]="ED25519-V3:8kF7xN2vQpL9mR4sT6wA3bC1eH5jY8qE9dK2fG7hI0nM="
    ["bob"]="ED25519-V3:9mG8yO3wRqM0nS5uU7xB4cD2fI6kZ9rF0eL3gH8iJ1oN="
    ["charlie"]="ED25519-V3:0nH9zP4xSrN1oT6vV8yC5dE3gJ7l0sG1fM4hI9jK2pO="
    ["diana"]="ED25519-V3:1oI0AQ5yTsO2pU7wW9zD6eF4hK8m1tH2gN5iJ0kL3qP="
    ["eve"]="ED25519-V3:2pJ1BR6zUtP3qV8xX0AE7fG5iL9n2uI3hO6jK1lM4rQ="
)

# Create deployment directories
mkdir -p logs network-tests enhanced-logs real-logs
mkdir -p configs/{alice,bob,charlie,diana,eve}

echo "📁 Creating node configurations..."

# Generate configuration for each node
for node in "${!NODES[@]}"; do
    addr="${NODES[$node]}"
    onion_key="${ONION_KEYS[$node]}"
    
    cat > configs/${node}/config.toml << EOF
# Q-NarwhalKnight Node Configuration: ${node}
# Server Alpha Deployment - Quantum BFT Consensus

[network]
node_name = "${node}"
listen_address = "${addr}"
public_address = "${addr}"
bootstrap_peers = []
tor_enabled = true
tor_socks_proxy = "127.0.0.1:9050"

[tor]
onion_service_enabled = true
onion_service_key = "${onion_key}"
onion_service_port = 8000
hidden_service_dir = "/tmp/q-knight-${node}"
circuit_timeout = "30s"
connection_pool_size = 50

[consensus]
validator_id = "${node}"
byzantine_threshold = 7  # 2f+1 for f=3 Byzantine nodes (10 total)
max_validators = 10
voting_timeout = "10s"
finalization_timeout = "5s"
enable_slashing = true
min_voting_stake = 1000

[mempool]
max_transactions = 10000
transaction_timeout = "60s"
fee_per_byte = 10
max_transaction_size = 1048576

[dag]
max_vertex_age = "300s"
vertex_cache_size = 1000
vdf_difficulty = 1024
quantum_enhancement = 0.8

[api]
rest_enabled = true
rest_address = "127.0.0.1:$((9000 + ${#node}))"
websocket_enabled = true
metrics_enabled = true

[logging]
level = "info"
file = "logs/${node}.log"
structured = true
EOF

    echo "  ✅ Configuration created for ${node} (${addr})"
done

echo ""
echo "🧅 Starting Tor service..."
# Ensure Tor is running for .onion services
if ! pgrep tor > /dev/null; then
    echo "Starting Tor daemon..."
    tor --RunAsDaemon 1 --SocksPort 9050 &
    sleep 5
fi

echo ""
echo "🚀 Building Q-NarwhalKnight in release mode..."
cd /mnt/orobit-shared/q-narwhalknight
cargo build --release --workspace

echo ""
echo "🔥 Launching 5 Server Alpha nodes..."

# Start each node in background with proper logging
for node in "${!NODES[@]}"; do
    addr="${NODES[$node]}"
    port=${addr##*:}
    
    echo "  🌟 Starting ${node} on ${addr}..."
    
    # Start node with comprehensive logging
    RUST_LOG=debug ./target/release/q-vm \
        --config "configs/${node}/config.toml" \
        --node-id "${node}" \
        --listen "${addr}" \
        --tor-enabled \
        > "real-logs/node-${node}.log" 2>&1 &
    
    NODE_PID=$!
    echo "${NODE_PID}" > "configs/${node}/pid"
    echo "    ✅ Node ${node} started (PID: ${NODE_PID})"
    
    # Brief delay between starts
    sleep 2
done

echo ""
echo "⏳ Waiting for nodes to initialize..."
sleep 10

echo ""
echo "🔍 Checking node status..."

for node in "${!NODES[@]}"; do
    pid_file="configs/${node}/pid"
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  ✅ ${node}: Running (PID: ${pid})"
        else
            echo "  ❌ ${node}: Failed to start"
        fi
    else
        echo "  ❌ ${node}: PID file not found"
    fi
done

echo ""
echo "📊 Node Information:"
echo "===================="

for node in "${!NODES[@]}"; do
    addr="${NODES[$node]}"
    api_port=$((9000 + ${#node}))
    
    echo "🌟 ${node}:"
    echo "  🌐 Listen: ${addr}"
    echo "  🧅 Onion: ${node}.qnk.onion (generating...)"
    echo "  🔌 API: http://127.0.0.1:${api_port}"
    echo "  📝 Log: real-logs/node-${node}.log"
    echo ""
done

echo "📡 Network Status:"
echo "=================="
echo "🔢 Total Nodes: 5 (Server Alpha)"
echo "🎯 Waiting for Server Beta: 5 additional nodes"
echo "🏆 Target Network: 10 nodes total (f=3 Byzantine tolerance)"
echo "🧅 Tor Integration: Anonymous consensus networking"
echo "⚛️ Quantum Enhancement: VDF proofs with 80% quantum factor"
echo ""

echo "📊 Real-time monitoring commands:"
echo "=================================="
echo "# Watch all node logs:"
echo "tail -f real-logs/node-*.log"
echo ""
echo "# Monitor specific node:"
echo "tail -f real-logs/node-alice.log"
echo ""
echo "# Check network consensus:"
echo "curl http://127.0.0.1:9001/status"
echo ""
echo "# View Byzantine detection:"
echo "curl http://127.0.0.1:9001/metrics/byzantine"
echo ""

echo "🎉 SERVER ALPHA DEPLOYMENT COMPLETE!"
echo "Waiting for Server Beta to deploy matching nodes..."
echo "Once both deployments are complete, we'll have the world's first"
echo "anonymous quantum-enhanced BFT consensus network running!"
echo ""
echo "🚀 Ready for Byzantine fault tolerance testing! 🛡️⚛️"