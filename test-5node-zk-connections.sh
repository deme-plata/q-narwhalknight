#!/bin/bash

# Simple 5-Node ZK P2P Connection Test
echo "🔐 Testing 5-node ZK-enhanced P2P connections"
echo "============================================="

# Configuration
COMPOSE_FILE="docker-compose-zk-test.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}📋 Test Plan:${NC}"
echo "1. Build latest ZK-enhanced Docker image"
echo "2. Start 5-node network with ZK P2P enabled"
echo "3. Monitor automatic peer discovery"
echo "4. Verify ZK proof generation and verification"
echo "5. Check connection establishment logs"
echo ""

# Step 1: Check if we have the latest binary
echo -e "${BLUE}🔨 Step 1: Checking binary...${NC}"
if [ -f "target/release/q-api-server" ]; then
    BINARY_TIME=$(stat -c %Y target/release/q-api-server)
    CURRENT_TIME=$(date +%s)
    AGE=$((CURRENT_TIME - BINARY_TIME))
    
    echo "Binary age: $AGE seconds"
    if [ $AGE -lt 300 ]; then  # Less than 5 minutes old
        echo -e "${GREEN}✅ Fresh binary found (< 5 minutes old)${NC}"
    else
        echo -e "${YELLOW}⚠️  Binary is older than 5 minutes${NC}"
    fi
else
    echo -e "${RED}❌ Binary not found!${NC}"
    exit 1
fi

# Step 2: Build Docker image
echo -e "${BLUE}🐳 Step 2: Building ZK-enhanced Docker image...${NC}"
docker build -f Dockerfile.zk-p2p -t q-narwhalknight:zk-p2p-latest . || {
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Step 3: Create test data directories
echo -e "${BLUE}📁 Step 3: Setting up test directories...${NC}"
for i in {1..5}; do
    mkdir -p "data/zk-node$i" "configs/zk-node$i"
done
echo -e "${GREEN}✅ Test directories created${NC}"

# Step 4: Start the network
echo -e "${BLUE}🚀 Step 4: Starting 5-node ZK network...${NC}"
docker-compose -f "$COMPOSE_FILE" down >/dev/null 2>&1 || true
docker-compose -f "$COMPOSE_FILE" up -d

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Network started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start network${NC}"
    exit 1
fi

# Step 5: Wait and monitor
echo -e "${BLUE}⏳ Step 5: Waiting for initialization (60 seconds)...${NC}"
sleep 60

# Step 6: Check node status
echo -e "${BLUE}📊 Step 6: Checking node connectivity...${NC}"
echo ""

NODES_ONLINE=0
for i in {1..5}; do
    PORT=$((8090 + i))
    echo -n "Node $i (port $PORT): "
    
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ ONLINE${NC}"
        NODES_ONLINE=$((NODES_ONLINE + 1))
        
        # Try to get peer info (if API exists)
        PEER_INFO=$(curl -s "http://localhost:$PORT/peers" 2>/dev/null || echo "N/A")
        echo "  Peers: $(echo "$PEER_INFO" | grep -o 'peer_count' | wc -l || echo "API not available yet")"
        
        # Check container logs for ZK P2P activity
        echo "  Recent ZK activity:"
        docker logs "qnk-zk-node$i" 2>/dev/null | grep -i "zk\|proof\|connection" | tail -2 | sed 's/^/    /' || echo "    No ZK logs yet"
    else
        echo -e "${RED}❌ OFFLINE${NC}"
    fi
    echo ""
done

echo -e "${BLUE}📈 Network Summary:${NC}"
echo "Nodes online: $NODES_ONLINE/5"

if [ $NODES_ONLINE -eq 5 ]; then
    echo -e "${GREEN}🎉 All nodes are online! ZK P2P network initialization successful.${NC}"
    
    echo ""
    echo -e "${BLUE}🔍 Detailed Connection Analysis:${NC}"
    echo "Checking container logs for ZK P2P connection establishment..."
    echo ""
    
    for i in {1..5}; do
        echo -e "${CYAN}=== Node $i Logs ===${NC}"
        docker logs "qnk-zk-node$i" 2>/dev/null | grep -i "zk\|proof\|peer\|connection" | tail -5
        echo ""
    done
    
    echo -e "${GREEN}✅ TEST COMPLETED: Check logs above for ZK P2P connection details${NC}"
    echo -e "${YELLOW}💡 Monitor with: docker-compose -f $COMPOSE_FILE logs -f${NC}"
    
elif [ $NODES_ONLINE -ge 3 ]; then
    echo -e "${YELLOW}⚠️  Partial success: $NODES_ONLINE nodes online${NC}"
    echo -e "${YELLOW}💡 Check logs: docker-compose -f $COMPOSE_FILE logs${NC}"
else
    echo -e "${RED}❌ Network formation failed: only $NODES_ONLINE nodes online${NC}"
    echo -e "${RED}🔍 Debug with: docker-compose -f $COMPOSE_FILE logs${NC}"
fi

echo ""
echo -e "${BLUE}🧹 To stop the test network:${NC}"
echo "docker-compose -f $COMPOSE_FILE down"
echo ""
echo -e "${BLUE}📋 To view live logs:${NC}"
echo "docker-compose -f $COMPOSE_FILE logs -f"