#!/bin/bash

# Q-NarwhalKnight Validator Node Deployment Script
# Deploys a full validator node with all Phase 3+4 optimizations

set -euo pipefail

# Configuration
NODE_ID=${1:-"alpha-1"}
PORT=${2:-8006}
API_PORT=${3:-9006}
PHASE=${4:-"Phase4"}

echo "🚀 Deploying Q-NarwhalKnight Validator Node: $NODE_ID"
echo "📊 Configuration:"
echo "  Node ID: $NODE_ID"
echo "  P2P Port: $PORT"
echo "  API Port: $API_PORT"
echo "  Optimization Phase: $PHASE"

# Environment variables for optimization
export RUST_LOG="q_api_server=info,q_bitcoin_bridge=info,q_tor_client=info"
export Q_NARWHAL_NODE_ID="$NODE_ID"
export Q_NARWHAL_PORT="$PORT"
export Q_NARWHAL_API_PORT="$API_PORT"
export Q_NARWHAL_PHASE="$PHASE"

# Phase 3+4 optimizations
export Q_ENABLE_SIMD="true"
export Q_ENABLE_KERNEL_IO="true"
export Q_SIMD_MODE="avx512"
export Q_ENABLE_IO_URING="true"
export Q_ENABLE_NUMA="true"
export Q_ENABLE_ZERO_COPY="true"

# Cache settings for Phase 2 integration
export Q_CACHE_SIZE="2GB"
export Q_L1_CACHE="1MB"
export Q_L2_CACHE="100MB"
export Q_L3_CACHE="1GB"

# Network configuration
export Q_BOOTSTRAP_PEERS="bootstrap.q-narwhalknight.network"
export Q_ENABLE_BITCOIN_DISCOVERY="true"
export Q_ENABLE_TOR="true"

# Create node directory
NODE_DIR="/mnt/orobit-shared/q-narwhalknight/nodes/$NODE_ID"
mkdir -p "$NODE_DIR"
cd "$NODE_DIR"

echo "📂 Node directory: $NODE_DIR"

# Generate node configuration
cat > node_config.toml << EOF
[node]
id = "$NODE_ID"
port = $PORT
api_port = $API_PORT
phase = "$PHASE"

[optimization]
enable_simd = true
enable_kernel_io = true
simd_mode = "avx512"
enable_io_uring = true
enable_numa = true
enable_zero_copy = true

[cache]
total_size = "2GB"
l1_size = "1MB"
l2_size = "100MB"
l3_size = "1GB"
enable_ml_prefetch = true

[network]
bootstrap_peers = ["bootstrap.q-narwhalknight.network"]
enable_bitcoin_discovery = true
enable_tor = true
max_peers = 50

[consensus]
timeout_ms = 5000
batch_size = 1000
enable_sharding = true
shard_count = 8

[logging]
level = "info"
enable_metrics = true
metrics_port = $((API_PORT + 100))
EOF

echo "⚙️  Node configuration created"

# Start the validator node
echo "🔥 Starting Q-NarwhalKnight Validator Node..."

# Use the compiled binary when ready
BINARY_PATH="/mnt/orobit-shared/q-narwhalknight/target/release/q-api-server"

if [[ -f "$BINARY_PATH" ]]; then
    echo "✅ Using compiled binary: $BINARY_PATH"
    exec "$BINARY_PATH" --config node_config.toml
else
    echo "🏗️  Binary not ready, starting with cargo run..."
    cd /mnt/orobit-shared/q-narwhalknight
    exec cargo run --release --bin q-api-server -- --config "$NODE_DIR/node_config.toml"
fi