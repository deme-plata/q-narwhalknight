#!/bin/bash

# Q-NarwhalKnight 4-Node Local Testnet
# Creates 4 nodes with separate data directories for proper testing

set -e

BINARY="./target/release/q-api-server"
BASE_DIR="./testnet-data"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Q-NarwhalKnight 4-Node Local Testnet Launcher      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo -e "${YELLOW}⚠ Binary not found at $BINARY${NC}"
    echo "Building q-api-server..."
    cargo build --release --package q-api-server
fi

# Kill existing nodes
echo -e "${YELLOW}🧹 Cleaning up existing nodes...${NC}"
killall -9 q-api-server 2>/dev/null || true
sleep 2

# Create data directories
echo -e "${BLUE}📁 Creating data directories...${NC}"
mkdir -p "$BASE_DIR"/{node1,node2,node3,node4}

# Start Node 1 (Port 8080)
echo -e "${GREEN}🚀 Starting Node 1 (Port 8080)...${NC}"
Q_DB_PATH="$BASE_DIR/node1" nohup timeout 36000 $BINARY \
    --port 8080 \
    --node-id node1 \
    > "$BASE_DIR/node1/console.log" 2>&1 &
NODE1_PID=$!
echo "   PID: $NODE1_PID"

# Start Node 2 (Port 8084)
echo -e "${GREEN}🚀 Starting Node 2 (Port 8084)...${NC}"
Q_DB_PATH="$BASE_DIR/node2" nohup timeout 36000 $BINARY \
    --port 8084 \
    --node-id node2 \
    > "$BASE_DIR/node2/console.log" 2>&1 &
NODE2_PID=$!
echo "   PID: $NODE2_PID"

# Start Node 3 (Port 9060)
echo -e "${GREEN}🚀 Starting Node 3 (Port 9060)...${NC}"
Q_DB_PATH="$BASE_DIR/node3" nohup timeout 36000 $BINARY \
    --port 9060 \
    --node-id node3 \
    > "$BASE_DIR/node3/console.log" 2>&1 &
NODE3_PID=$!
echo "   PID: $NODE3_PID"

# Start Node 4 (Port 9666)
echo -e "${GREEN}🚀 Starting Node 4 (Port 9666)...${NC}"
Q_DB_PATH="$BASE_DIR/node4" nohup timeout 36000 $BINARY \
    --port 9666 \
    --node-id node4 \
    > "$BASE_DIR/node4/console.log" 2>&1 &
NODE4_PID=$!
echo "   PID: $NODE4_PID"

echo ""
echo -e "${BLUE}⏳ Waiting 10 seconds for nodes to initialize...${NC}"
sleep 10

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ All 4 nodes started successfully!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Node 1: http://localhost:8080 (PID: $NODE1_PID)"
echo "Node 2: http://localhost:8084 (PID: $NODE2_PID)"
echo "Node 3: http://localhost:9060 (PID: $NODE3_PID)"
echo "Node 4: http://localhost:9666 (PID: $NODE4_PID)"
echo ""
echo "Data directories:"
echo "  $BASE_DIR/node1/"
echo "  $BASE_DIR/node2/"
echo "  $BASE_DIR/node3/"
echo "  $BASE_DIR/node4/"
echo ""
echo "Logs:"
echo "  tail -f $BASE_DIR/node1/console.log"
echo "  tail -f $BASE_DIR/node2/console.log"
echo "  tail -f $BASE_DIR/node3/console.log"
echo "  tail -f $BASE_DIR/node4/console.log"
echo ""
echo -e "${YELLOW}📊 Checking node status...${NC}"
echo ""

# Check status of each node
for port in 8080 8084 9060 9666; do
    echo -e "${BLUE}Node on port $port:${NC}"
    curl -s "http://localhost:$port/api/v1/status" | jq '{node_id, connected_peers, blockchain_height}' 2>/dev/null || echo "  ⚠ Not responding yet"
    echo ""
done

echo ""
echo -e "${GREEN}🎉 Testnet ready for testing!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run transaction test: cd test_tx_propagation && ./target/release/test_tx_propagation"
echo "  2. Monitor logs: tail -f $BASE_DIR/node*/console.log"
echo "  3. Stop testnet: killall q-api-server"
echo ""
