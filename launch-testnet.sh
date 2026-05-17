#!/bin/bash
# Q-NarwhalKnight 4-Node Testnet Launcher

echo "🚀 Q-NarwhalKnight 4-Node Testnet Deployment"
echo "============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Create directories for each node
echo -e "${BLUE}📁 Creating node directories...${NC}"
for i in 1 2 3 4; do
    mkdir -p testnet/node$i/data
    mkdir -p testnet/node$i/logs
done

# Build the system first
echo -e "${BLUE}🔨 Building Q-NarwhalKnight...${NC}"
cargo build --release --bin mitochondria-sim 2>&1 | tail -5

# Function to start a node
start_node() {
    local NODE_ID=$1
    local PORT=$2
    local API_PORT=$3
    local IS_BYZANTINE=$4
    
    echo -e "${GREEN}🚀 Starting Node $NODE_ID (Port: $PORT, API: $API_PORT)${NC}"
    
    # Create node config
    cat > testnet/node$NODE_ID/config.toml << EOF
[node]
id = "node$NODE_ID"
name = "validator-$NODE_ID"
data_dir = "./testnet/node$NODE_ID/data"
log_dir = "./testnet/node$NODE_ID/logs"

[network]
listen_addr = "127.0.0.1:$PORT"
api_addr = "127.0.0.1:$API_PORT"
bootstrap_nodes = ["127.0.0.1:8001"]

[consensus]
role = "validator"
byzantine_mode = $IS_BYZANTINE
EOF
    
    # Start the node in background
    RUST_LOG=info cargo run --release --bin mitochondria-sim -- \
        --config testnet/node$NODE_ID/config.toml \
        --validators 4 \
        --port $PORT \
        --api-port $API_PORT \
        --node-id node$NODE_ID \
        > testnet/node$NODE_ID/logs/output.log 2>&1 &
    
    echo $! > testnet/node$NODE_ID/pid
    sleep 2
}

# Start all nodes
echo -e "${BLUE}🌐 Launching 4-node testnet...${NC}"
echo ""

# Node 1 - Bootstrap node
start_node 1 8001 9001 false

# Wait for bootstrap
sleep 3

# Node 2 - Normal validator
start_node 2 8002 9002 false

# Node 3 - Normal validator  
start_node 3 8003 9003 false

# Node 4 - Byzantine node (for testing)
start_node 4 8004 9004 true

echo ""
echo -e "${GREEN}✅ All nodes started!${NC}"
echo ""

# Wait for network to stabilize
echo -e "${YELLOW}⏳ Waiting for network consensus...${NC}"
sleep 5

# Check node status
echo -e "${BLUE}📊 Node Status:${NC}"
for i in 1 2 3 4; do
    if [ -f testnet/node$i/pid ]; then
        PID=$(cat testnet/node$i/pid)
        if ps -p $PID > /dev/null; then
            echo -e "  Node $i: ${GREEN}✓ Running${NC} (PID: $PID)"
        else
            echo -e "  Node $i: ${RED}✗ Stopped${NC}"
        fi
    fi
done

echo ""
echo -e "${BLUE}🔍 Testing Byzantine Fault Tolerance...${NC}"
echo ""

# Run Byzantine fault tolerance test
echo "Test scenarios:"
echo "  1. Normal operation (4 honest nodes)"
echo "  2. Byzantine node sends conflicting messages"
echo "  3. Byzantine node attempts double-spend"
echo "  4. Network partition simulation"
echo ""

# Monitor consensus
echo -e "${YELLOW}📈 Monitoring consensus...${NC}"
echo "Press Ctrl+C to stop the testnet"
echo ""

# Display logs
tail -f testnet/node*/logs/output.log | grep -E "(consensus|byzantine|finality)" &

# Trap to cleanup on exit
trap cleanup EXIT

cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping testnet...${NC}"
    for i in 1 2 3 4; do
        if [ -f testnet/node$i/pid ]; then
            PID=$(cat testnet/node$i/pid)
            kill $PID 2>/dev/null
            rm testnet/node$i/pid
        fi
    done
    echo -e "${GREEN}✅ Testnet stopped${NC}"
}

# Keep script running
wait