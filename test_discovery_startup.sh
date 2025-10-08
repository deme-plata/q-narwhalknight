#!/bin/bash

# Test if discovery services actually start now

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔍 Testing Discovery Service Startup${NC}"
echo "======================================"

BINARY="./target/x86_64-unknown-linux-gnu/release/q-api-server"

# Start a node and check if discovery actually starts
echo -e "${YELLOW}Starting node with discovery fix...${NC}"

Q_BOOTSTRAP_PEERS="127.0.0.1:9001" Q_DB_PATH="./test-discovery-startup" Q_P2P_PORT=9005 \
$BINARY --node-id discovery-test --port 8095 &
PID=$!

echo "Node started with PID: $PID"
echo "Waiting 60 seconds to check discovery activity..."

sleep 60

if kill -0 $PID 2>/dev/null; then
    echo -e "${GREEN}✅ Node is still running${NC}"

    # Check if API is responding
    if curl -s -f "http://localhost:8095/api/v1/health" > /dev/null; then
        echo -e "${GREEN}✅ API is responding${NC}"

        # Get discovery status
        echo -e "${YELLOW}Checking discovery status...${NC}"
        DISCOVERY_STATUS=$(curl -s "http://localhost:8095/api/v1/network/discovery/status" 2>/dev/null || echo '{}')
        echo "$DISCOVERY_STATUS" | jq '.' 2>/dev/null || echo "$DISCOVERY_STATUS"

        # Check production discovery status
        echo -e "${YELLOW}Checking production discovery status...${NC}"
        PROD_STATUS=$(curl -s "http://localhost:8095/api/v1/network/production/discovery/status" 2>/dev/null || echo '{}')
        echo "$PROD_STATUS" | jq '.' 2>/dev/null || echo "$PROD_STATUS"

    else
        echo -e "${RED}❌ API not responding${NC}"
    fi

    kill $PID
else
    echo -e "${RED}❌ Node crashed${NC}"
fi

# Cleanup
rm -rf ./test-discovery-startup 2>/dev/null || true

echo -e "${GREEN}Test completed${NC}"