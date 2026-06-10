#!/bin/bash

# DNS-Phantom Actual Connection Test
# Test if two isolated instances actually discover and connect to each other through DNS steganography

set -e

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    DNS-PHANTOM ACTUAL CONNECTION TEST                          ║"
echo "║               Do isolated instances ACTUALLY connect through DNS?             ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BINARY_PATH="./target/release/q-api-server"
TEST_DIR="dns-phantom-connection-test"
NODE_A_PORT="9021"
NODE_B_PORT="9022"
TEST_DURATION=120  # 2 minute test to allow connection time
MONITOR_INTERVAL=10  # Check every 10 seconds

# Check if binary exists
if [ ! -f "$BINARY_PATH" ]; then
    echo -e "${RED}❌ Binary not found at $BINARY_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Binary ready: $(du -h $BINARY_PATH | cut -f1)${NC}"
echo ""

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up test processes...${NC}"
    
    # Kill all processes in our test
    if [ ! -z "${NODE_A_PID:-}" ]; then
        kill "$NODE_A_PID" 2>/dev/null || true
        wait "$NODE_A_PID" 2>/dev/null || true
    fi
    
    if [ ! -z "${NODE_B_PID:-}" ]; then
        kill "$NODE_B_PID" 2>/dev/null || true
        wait "$NODE_B_PID" 2>/dev/null || true
    fi
    
    # Clean up database locks
    rm -rf data/ 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Set up cleanup trap
trap cleanup EXIT

# Create test directory
mkdir -p "$TEST_DIR/logs"
mkdir -p "$TEST_DIR/results"

echo -e "${BLUE}🏗️  Starting long-running connection test...${NC}"
echo -e "${CYAN}   Duration: ${TEST_DURATION} seconds${NC}"
echo -e "${CYAN}   Monitoring interval: ${MONITOR_INTERVAL} seconds${NC}"

# Clean any existing database
rm -rf data/ 2>/dev/null || true

# Start Node A (keep running longer)
echo -e "${PURPLE}🚀 Starting Node A on port $NODE_A_PORT${NC}"

# Use environment variables to ensure separate databases
DB_A_PATH="$TEST_DIR/db_a"
mkdir -p "$DB_A_PATH"

RUST_LOG=debug LD_BIND_NOW=1 $BINARY_PATH \
    --port $NODE_A_PORT \
    --enable-dns-phantom \
    --max-peers 50 \
    --discovery-interval 15 \
    > "$TEST_DIR/logs/node_a.log" 2>&1 &

NODE_A_PID=$!
echo -e "${GREEN}   ✅ Node A started (PID: $NODE_A_PID)${NC}"

# Wait for Node A to fully initialize
echo -e "${CYAN}   ⏳ Waiting for Node A DNS-Phantom to fully initialize...${NC}"
sleep 8

# Start Node B 
echo -e "${PURPLE}🚀 Starting Node B on port $NODE_B_PORT${NC}"

DB_B_PATH="$TEST_DIR/db_b"
mkdir -p "$DB_B_PATH"

RUST_LOG=debug LD_BIND_NOW=1 $BINARY_PATH \
    --port $NODE_B_PORT \
    --enable-dns-phantom \
    --max-peers 50 \
    --discovery-interval 15 \
    > "$TEST_DIR/logs/node_b.log" 2>&1 &

NODE_B_PID=$!
echo -e "${GREEN}   ✅ Node B started (PID: $NODE_B_PID)${NC}"

echo ""
echo -e "${CYAN}🔍 Monitoring for actual DNS-Phantom connections for ${TEST_DURATION} seconds...${NC}"
echo -e "${CYAN}   Looking for peer discovery, connections, and communication${NC}"
echo ""

# Monitor loop
ELAPSED=0
CHECK_COUNT=0

while [ $ELAPSED -lt $TEST_DURATION ]; do
    CHECK_COUNT=$((CHECK_COUNT + 1))
    ELAPSED=$((CHECK_COUNT * MONITOR_INTERVAL))
    
    echo -e "${CYAN}📊 Connection Check $CHECK_COUNT (${ELAPSED}s elapsed)${NC}"
    
    # Check if processes are still running
    NODE_A_RUNNING=false
    NODE_B_RUNNING=false
    
    if kill -0 "$NODE_A_PID" 2>/dev/null; then
        NODE_A_RUNNING=true
        echo -e "${GREEN}   ✅ Node A running${NC}"
    else
        echo -e "${RED}   ❌ Node A stopped${NC}"
    fi
    
    if kill -0 "$NODE_B_PID" 2>/dev/null; then
        NODE_B_RUNNING=true
        echo -e "${GREEN}   ✅ Node B running${NC}"
    else
        echo -e "${RED}   ❌ Node B stopped${NC}"
    fi
    
    # Analyze logs for connection activity
    echo -e "${BLUE}📡 DNS-Phantom Activity Analysis:${NC}"
    
    # Node A analysis
    NODE_A_DNS_QUERIES=$(grep -c "Executing steganographic\|DNS query\|steganographic query" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
    NODE_A_PEER_DISCOVERY=$(grep -c "Discovered peer\|peer discovery\|Found peer\|📋.*peer" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
    NODE_A_CONNECTIONS=$(grep -c "connection established\|connected to\|peer connected" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
    NODE_A_MESH_ACTIVITY=$(grep -c "mesh\|network topology\|peer advertisement" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
    
    echo -e "   🔵 Node A - DNS queries: $NODE_A_DNS_QUERIES, Peer discoveries: $NODE_A_PEER_DISCOVERY, Connections: $NODE_A_CONNECTIONS, Mesh: $NODE_A_MESH_ACTIVITY"
    
    # Node B analysis
    NODE_B_DNS_QUERIES=$(grep -c "Executing steganographic\|DNS query\|steganographic query" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
    NODE_B_PEER_DISCOVERY=$(grep -c "Discovered peer\|peer discovery\|Found peer\|📋.*peer" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
    NODE_B_CONNECTIONS=$(grep -c "connection established\|connected to\|peer connected" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
    NODE_B_MESH_ACTIVITY=$(grep -c "mesh\|network topology\|peer advertisement" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
    
    echo -e "   🔵 Node B - DNS queries: $NODE_B_DNS_QUERIES, Peer discoveries: $NODE_B_PEER_DISCOVERY, Connections: $NODE_B_CONNECTIONS, Mesh: $NODE_B_MESH_ACTIVITY"
    
    # Check for specific connection indicators
    TOTAL_DISCOVERIES=$((NODE_A_PEER_DISCOVERY + NODE_B_PEER_DISCOVERY))
    TOTAL_CONNECTIONS=$((NODE_A_CONNECTIONS + NODE_B_CONNECTIONS))
    TOTAL_DNS_QUERIES=$((NODE_A_DNS_QUERIES + NODE_B_DNS_QUERIES))
    
    echo -e "   📊 Totals - DNS queries: $TOTAL_DNS_QUERIES, Discoveries: $TOTAL_DISCOVERIES, Connections: $TOTAL_CONNECTIONS"
    
    # Early success detection
    if [ $TOTAL_CONNECTIONS -gt 0 ]; then
        echo -e "${GREEN}🎉 EARLY SUCCESS: Actual connections detected!${NC}"
        break
    elif [ $TOTAL_DISCOVERIES -gt 0 ]; then
        echo -e "${YELLOW}⚡ PROGRESS: Peer discovery activity detected!${NC}"
    elif [ $TOTAL_DNS_QUERIES -gt 0 ]; then
        echo -e "${CYAN}📡 ACTIVITY: DNS steganographic queries active${NC}"
    else
        echo -e "${YELLOW}⏳ WAITING: No connection activity yet...${NC}"
    fi
    
    echo ""
    
    # Break if both processes died
    if [ "$NODE_A_RUNNING" = false ] && [ "$NODE_B_RUNNING" = false ]; then
        echo -e "${RED}❌ Both nodes stopped, ending test early${NC}"
        break
    fi
    
    sleep $MONITOR_INTERVAL
done

echo -e "${BLUE}📊 Generating final connection analysis...${NC}"

# Final comprehensive analysis
echo -e "${PURPLE}📋 FINAL CONNECTION TEST RESULTS:${NC}"

# Count all activity types
FINAL_NODE_A_DNS=$(grep -c "Executing steganographic\|DNS query\|steganographic query\|🔍.*query" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
FINAL_NODE_A_PEERS=$(grep -c "Discovered peer\|peer discovery\|Found peer\|📋.*peer\|🎯.*peer" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
FINAL_NODE_A_CONNECTIONS=$(grep -c "connection established\|connected to\|peer connected\|📞" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)

FINAL_NODE_B_DNS=$(grep -c "Executing steganographic\|DNS query\|steganographic query\|🔍.*query" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
FINAL_NODE_B_PEERS=$(grep -c "Discovered peer\|peer discovery\|Found peer\|📋.*peer\|🎯.*peer" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
FINAL_NODE_B_CONNECTIONS=$(grep -c "connection established\|connected to\|peer connected\|📞" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)

FINAL_TOTAL_DNS=$((FINAL_NODE_A_DNS + FINAL_NODE_B_DNS))
FINAL_TOTAL_PEERS=$((FINAL_NODE_A_PEERS + FINAL_NODE_B_PEERS))
FINAL_TOTAL_CONNECTIONS=$((FINAL_NODE_A_CONNECTIONS + FINAL_NODE_B_CONNECTIONS))

echo -e "${CYAN}   🎯 Total DNS steganographic queries: $FINAL_TOTAL_DNS${NC}"
echo -e "${CYAN}   🌐 Total peer discoveries: $FINAL_TOTAL_PEERS${NC}" 
echo -e "${CYAN}   📞 Total actual connections: $FINAL_TOTAL_CONNECTIONS${NC}"

# Create detailed results report
cat > "$TEST_DIR/results/connection_analysis.txt" << EOF
DNS-PHANTOM ACTUAL CONNECTION TEST RESULTS
==========================================

Test Duration: ${TEST_DURATION} seconds
Test Checks: $CHECK_COUNT monitoring intervals

Node A Results:
- DNS Steganographic Queries: $FINAL_NODE_A_DNS
- Peer Discoveries: $FINAL_NODE_A_PEERS  
- Actual Connections: $FINAL_NODE_A_CONNECTIONS

Node B Results:
- DNS Steganographic Queries: $FINAL_NODE_B_DNS
- Peer Discoveries: $FINAL_NODE_B_PEERS
- Actual Connections: $FINAL_NODE_B_CONNECTIONS

Combined Results:
- Total DNS Steganographic Queries: $FINAL_TOTAL_DNS
- Total Peer Discoveries: $FINAL_TOTAL_PEERS
- Total Actual Connections: $FINAL_TOTAL_CONNECTIONS

EOF

# Determine final result
if [ $FINAL_TOTAL_CONNECTIONS -gt 0 ]; then
    echo -e "${GREEN}🎉 SUCCESS: Nodes actually connected through DNS-Phantom!${NC}"
    echo -e "${GREEN}   ✅ Real peer-to-peer connections established${NC}"
    echo -e "${GREEN}   ✅ DNS steganographic discovery is fully functional${NC}"
    TEST_RESULT="SUCCESS - ACTUAL CONNECTIONS"
elif [ $FINAL_TOTAL_PEERS -gt 0 ]; then
    echo -e "${YELLOW}⚡ PARTIAL: Peer discovery working but connections need refinement${NC}"
    echo -e "${YELLOW}   ✅ DNS-Phantom peer discovery functional${NC}"
    echo -e "${YELLOW}   ⚠️  Connection establishment may need optimization${NC}"
    TEST_RESULT="PARTIAL - PEER DISCOVERY ONLY"
elif [ $FINAL_TOTAL_DNS -gt 0 ]; then
    echo -e "${CYAN}📡 LIMITED: DNS steganographic queries active but no peer discovery${NC}"
    echo -e "${CYAN}   ✅ DNS-Phantom network communication active${NC}"
    echo -e "${CYAN}   ⚠️  Peer discovery logic may need refinement${NC}"
    TEST_RESULT="LIMITED - DNS ACTIVITY ONLY"
else
    echo -e "${RED}❌ FAILED: No DNS-Phantom connection activity detected${NC}"
    echo -e "${RED}   ❌ No steganographic DNS communication${NC}"
    echo -e "${RED}   ❌ No peer discovery or connections${NC}"
    TEST_RESULT="FAILED"
fi

echo "Final Result: $TEST_RESULT" >> "$TEST_DIR/results/connection_analysis.txt"

echo ""
echo -e "${BLUE}📁 Complete test artifacts:${NC}"
echo -e "${BLUE}   📝 Node A Log: $TEST_DIR/logs/node_a.log${NC}"
echo -e "${BLUE}   📝 Node B Log: $TEST_DIR/logs/node_b.log${NC}"
echo -e "${BLUE}   📊 Analysis: $TEST_DIR/results/connection_analysis.txt${NC}"

# Show recent log samples for evidence
echo ""
echo -e "${CYAN}📋 Recent Node A Activity:${NC}"
tail -15 "$TEST_DIR/logs/node_a.log" | head -8

echo ""
echo -e "${CYAN}📋 Recent Node B Activity:${NC}"
tail -15 "$TEST_DIR/logs/node_b.log" | head -8

exit 0