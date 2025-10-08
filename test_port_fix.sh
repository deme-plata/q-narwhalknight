#!/bin/bash

# Test P2P port configuration fix

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 TESTING P2P PORT FIX${NC}"
echo "=========================="

BINARY="./target/x86_64-unknown-linux-gnu/release/q-api-server"

cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    killall q-api-server 2>/dev/null || true
    rm -rf ./test-port-* 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM EXIT

echo -e "${YELLOW}Step 1: Testing Bootstrap Node (P2P Port 9001)${NC}"

# Start bootstrap node on port 9001
Q_BOOTSTRAP_PEERS="" Q_DB_PATH="./test-port-bootstrap" Q_P2P_PORT=9001 \
$BINARY --node-id bootstrap-port-test --port 8080 &
BOOTSTRAP_PID=$!

echo "Bootstrap node started (PID: $BOOTSTRAP_PID)"
echo "Waiting 20 seconds for bootstrap to initialize..."
sleep 20

if ! kill -0 $BOOTSTRAP_PID 2>/dev/null; then
    echo -e "${RED}❌ Bootstrap node crashed!${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 2: Testing Client Node (P2P Port 9002)${NC}"

# Start client node on port 9002
Q_BOOTSTRAP_PEERS="127.0.0.1:9001" Q_DB_PATH="./test-port-client" Q_P2P_PORT=9002 \
$BINARY --node-id client-port-test --port 8081 &
CLIENT_PID=$!

echo "Client node started (PID: $CLIENT_PID)"
echo "Waiting 30 seconds for discovery..."
sleep 30

if ! kill -0 $CLIENT_PID 2>/dev/null; then
    echo -e "${RED}❌ Client node crashed!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=== PORT CONFIGURATION TEST ===${NC}"

# Check that nodes are using different ports
echo -e "${YELLOW}Checking listening ports:${NC}"
PORTS=$(netstat -tulpn 2>/dev/null | grep -E ":(9001|9002)" | grep q-api-server | wc -l)
echo "Q-NarwhalKnight nodes listening on separate P2P ports: $PORTS"

if [ "$PORTS" -eq "2" ]; then
    echo -e "${GREEN}✅ SUCCESS: Nodes using separate P2P ports (9001, 9002)${NC}"
else
    echo -e "${RED}❌ FAILED: Port configuration issue${NC}"
    echo "Port details:"
    netstat -tulpn 2>/dev/null | grep -E ":(9001|9002|8080|8081)" | head -10
fi

echo ""
echo -e "${YELLOW}Waiting 10 more seconds for observation...${NC}"
sleep 10