#!/bin/bash

# Test ACTUAL peer-to-peer connection between two nodes

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔗 REAL PEER CONNECTION TEST${NC}"
echo "=============================="

BINARY="./target/x86_64-unknown-linux-gnu/release/q-api-server"

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    killall q-api-server 2>/dev/null || true
    rm -rf ./test-conn-* 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM EXIT

echo -e "${YELLOW}Step 1: Starting BOOTSTRAP node (Server A simulation)${NC}"

# Start bootstrap node (no bootstrap peers - acts as the bootstrap server)
Q_BOOTSTRAP_PEERS="" Q_DB_PATH="./test-conn-bootstrap" Q_P2P_PORT=9001 \
$BINARY --node-id bootstrap --port 8080 &
BOOTSTRAP_PID=$!

echo "Bootstrap node started (PID: $BOOTSTRAP_PID)"
echo "Waiting 45 seconds for bootstrap to fully initialize..."
sleep 45

# Check if bootstrap node is running and responding
if ! kill -0 $BOOTSTRAP_PID 2>/dev/null; then
    echo -e "${RED}❌ Bootstrap node crashed!${NC}"
    exit 1
fi

if curl -s -f "http://localhost:8080/api/v1/health" > /dev/null; then
    echo -e "${GREEN}✅ Bootstrap node API responding${NC}"
else
    echo -e "${RED}❌ Bootstrap node API not responding${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 2: Starting CLIENT node (Server B simulation)${NC}"

# Start client node that should connect to bootstrap
Q_BOOTSTRAP_PEERS="127.0.0.1:9001" Q_DB_PATH="./test-conn-client" Q_P2P_PORT=9002 \
$BINARY --node-id client --port 8081 &
CLIENT_PID=$!

echo "Client node started (PID: $CLIENT_PID)"
echo "Waiting 60 seconds for peer discovery and connection..."
sleep 60

# Check if client node is running
if ! kill -0 $CLIENT_PID 2>/dev/null; then
    echo -e "${RED}❌ Client node crashed!${NC}"
    exit 1
fi

if curl -s -f "http://localhost:8081/api/v1/health" > /dev/null; then
    echo -e "${GREEN}✅ Client node API responding${NC}"
else
    echo -e "${RED}❌ Client node API not responding${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=== CONNECTIVITY ANALYSIS ===${NC}"

# Check bootstrap node peers
echo -e "${YELLOW}Bootstrap node peer status:${NC}"
BOOTSTRAP_PEERS=$(curl -s "http://localhost:8080/api/v1/network/peers" 2>/dev/null | jq '.peers | length' 2>/dev/null || echo "0")
echo "Connected peers: $BOOTSTRAP_PEERS"

# Check client node peers
echo -e "${YELLOW}Client node peer status:${NC}"
CLIENT_PEERS=$(curl -s "http://localhost:8081/api/v1/network/peers" 2>/dev/null | jq '.peers | length' 2>/dev/null || echo "0")
echo "Connected peers: $CLIENT_PEERS"

# Get discovery status from both nodes
echo ""
echo -e "${YELLOW}Bootstrap discovery status:${NC}"
curl -s "http://localhost:8080/api/v1/network/discovery/status" 2>/dev/null | jq '.' 2>/dev/null | head -20

echo ""
echo -e "${YELLOW}Client discovery status:${NC}"
curl -s "http://localhost:8081/api/v1/network/discovery/status" 2>/dev/null | jq '.' 2>/dev/null | head -20

echo ""
echo -e "${BLUE}=== TEST RESULTS ===${NC}"

if [ "$BOOTSTRAP_PEERS" -gt "0" ] || [ "$CLIENT_PEERS" -gt "0" ]; then
    echo -e "${GREEN}🎉 SUCCESS: Peer connections detected!${NC}"
    echo "Bootstrap peers: $BOOTSTRAP_PEERS"
    echo "Client peers: $CLIENT_PEERS"
else
    echo -e "${RED}❌ FAILED: No peer connections established${NC}"
    echo ""
    echo -e "${YELLOW}Possible issues:${NC}"
    echo "1. Discovery services running but not finding each other"
    echo "2. Peer connection logic not working"
    echo "3. Network ports/protocols mismatch"
    echo "4. Missing connection establishment code"

    # Show some diagnostic info
    echo ""
    echo -e "${YELLOW}Network diagnostic:${NC}"
    echo "Processes listening on ports:"
    netstat -tulpn 2>/dev/null | grep -E ":(9001|9002|8080|8081)" | head -10
fi

echo ""
echo -e "${YELLOW}Keeping nodes running for 30 more seconds for observation...${NC}"
sleep 30