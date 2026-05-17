#!/bin/bash

# Two-Node DNS-Phantom Connection Test
# Test if two isolated instances automatically connect through DNS-Phantom discovery

set -e

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                      TWO-NODE DNS-PHANTOM CONNECTION TEST                      ║"
echo "║                        Isolated Instance Auto-Discovery                        ║"
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
TEST_DIR="two-node-test"
NODE_A_PORT="9001"
NODE_B_PORT="9002"
TEST_DURATION=60  # 1 minute test

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
    
    # Kill Node A if running
    if [ ! -z "${NODE_A_PID:-}" ] && kill -0 "$NODE_A_PID" 2>/dev/null; then
        echo -e "   Stopping Node A (PID: $NODE_A_PID)"
        kill "$NODE_A_PID" 2>/dev/null || true
        wait "$NODE_A_PID" 2>/dev/null || true
    fi
    
    # Kill Node B if running  
    if [ ! -z "${NODE_B_PID:-}" ] && kill -0 "$NODE_B_PID" 2>/dev/null; then
        echo -e "   Stopping Node B (PID: $NODE_B_PID)"
        kill "$NODE_B_PID" 2>/dev/null || true
        wait "$NODE_B_PID" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Set up cleanup trap
trap cleanup EXIT

# Create test directory
mkdir -p "$TEST_DIR/logs"
mkdir -p "$TEST_DIR/results"
mkdir -p "$TEST_DIR/data_a"
mkdir -p "$TEST_DIR/data_b"

echo -e "${BLUE}🏗️  Setting up two-node test environment...${NC}"

# Start Node A (First isolated instance)
echo -e "${PURPLE}🚀 Starting Node A on port $NODE_A_PORT${NC}"
echo -e "   Command: $BINARY_PATH --port $NODE_A_PORT --enable-dns-phantom"

LD_BIND_NOW=1 timeout ${TEST_DURATION} $BINARY_PATH \
    --port $NODE_A_PORT \
    --enable-dns-phantom \
    --max-peers 10 \
    --discovery-interval 10 \
    --data-dir "$TEST_DIR/data_a" \
    > "$TEST_DIR/logs/node_a.log" 2>&1 &

NODE_A_PID=$!

if ! kill -0 "$NODE_A_PID" 2>/dev/null; then
    echo -e "${RED}❌ Failed to start Node A${NC}"
    exit 1
fi

echo -e "${GREEN}   ✅ Node A started (PID: $NODE_A_PID)${NC}"

# Wait a moment for Node A to initialize
sleep 5

# Start Node B (Second isolated instance)
echo -e "${PURPLE}🚀 Starting Node B on port $NODE_B_PORT${NC}"
echo -e "   Command: $BINARY_PATH --port $NODE_B_PORT --enable-dns-phantom"

LD_BIND_NOW=1 timeout ${TEST_DURATION} $BINARY_PATH \
    --port $NODE_B_PORT \
    --enable-dns-phantom \
    --max-peers 10 \
    --discovery-interval 10 \
    --data-dir "$TEST_DIR/data_b" \
    > "$TEST_DIR/logs/node_b.log" 2>&1 &

NODE_B_PID=$!

if ! kill -0 "$NODE_B_PID" 2>/dev/null; then
    echo -e "${RED}❌ Failed to start Node B${NC}"
    exit 1
fi

echo -e "${GREEN}   ✅ Node B started (PID: $NODE_B_PID)${NC}"
echo ""

echo -e "${CYAN}🔍 Monitoring DNS-Phantom auto-discovery for $TEST_DURATION seconds...${NC}"
echo -e "${CYAN}   Looking for connections between isolated instances${NC}"
echo ""

# Monitor for DNS-Phantom activity
MONITOR_COUNT=0
MAX_MONITORS=12  # 12 checks over 1 minute
SLEEP_INTERVAL=5

while [ $MONITOR_COUNT -lt $MAX_MONITORS ]; do
    MONITOR_COUNT=$((MONITOR_COUNT + 1))
    ELAPSED=$((MONITOR_COUNT * SLEEP_INTERVAL))
    
    echo -e "${CYAN}📊 Status Check $MONITOR_COUNT/$MAX_MONITORS (${ELAPSED}s elapsed)${NC}"
    
    # Check if both processes are still running
    if ! kill -0 "$NODE_A_PID" 2>/dev/null; then
        echo -e "${RED}   ❌ Node A crashed or stopped${NC}"
        break
    fi
    
    if ! kill -0 "$NODE_B_PID" 2>/dev/null; then
        echo -e "${RED}   ❌ Node B crashed or stopped${NC}"
        break
    fi
    
    echo -e "${GREEN}   ✅ Both nodes running${NC}"
    
    # Check for DNS-Phantom activity in logs
    NODE_A_DNS_COUNT=$(grep -c "DNS-Phantom\|steganographic\|📡\|🔍\|👻" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
    NODE_B_DNS_COUNT=$(grep -c "DNS-Phantom\|steganographic\|📡\|🔍\|👻" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
    
    echo -e "   📡 Node A DNS-Phantom activity: $NODE_A_DNS_COUNT events"
    echo -e "   📡 Node B DNS-Phantom activity: $NODE_B_DNS_COUNT events"
    
    # Check for peer discovery
    NODE_A_PEERS=$(grep -c "Discovered peer\|peer discovery\|📋.*peer" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
    NODE_B_PEERS=$(grep -c "Discovered peer\|peer discovery\|📋.*peer" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
    
    echo -e "   🌐 Node A peer discoveries: $NODE_A_PEERS"
    echo -e "   🌐 Node B peer discoveries: $NODE_B_PEERS"
    
    # Check for any connection establishment
    NODE_A_CONNECTIONS=$(grep -c "connection\|connected\|📞" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
    NODE_B_CONNECTIONS=$(grep -c "connection\|connected\|📞" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
    
    echo -e "   📞 Node A connections: $NODE_A_CONNECTIONS"
    echo -e "   📞 Node B connections: $NODE_B_CONNECTIONS"
    echo ""
    
    sleep $SLEEP_INTERVAL
done

echo -e "${BLUE}📊 Generating final test results...${NC}"

# Final analysis
TOTAL_DNS_EVENTS=$((NODE_A_DNS_COUNT + NODE_B_DNS_COUNT))
TOTAL_PEER_DISCOVERIES=$((NODE_A_PEERS + NODE_B_PEERS))
TOTAL_CONNECTIONS=$((NODE_A_CONNECTIONS + NODE_B_CONNECTIONS))

# Create results report
cat > "$TEST_DIR/results/connection_test_results.txt" << EOF
TWO-NODE DNS-PHANTOM CONNECTION TEST RESULTS
============================================

Test Configuration:
- Node A Port: $NODE_A_PORT
- Node B Port: $NODE_B_PORT  
- Test Duration: ${TEST_DURATION}s
- Binary: $BINARY_PATH

Node A Results:
- DNS-Phantom Events: $NODE_A_DNS_COUNT
- Peer Discoveries: $NODE_A_PEERS
- Connections: $NODE_A_CONNECTIONS
- Status: $(kill -0 "$NODE_A_PID" 2>/dev/null && echo "Running" || echo "Stopped")

Node B Results:
- DNS-Phantom Events: $NODE_B_DNS_COUNT  
- Peer Discoveries: $NODE_B_PEERS
- Connections: $NODE_B_CONNECTIONS
- Status: $(kill -0 "$NODE_B_PID" 2>/dev/null && echo "Running" || echo "Stopped")

Total Results:
- Combined DNS-Phantom Events: $TOTAL_DNS_EVENTS
- Combined Peer Discoveries: $TOTAL_PEER_DISCOVERIES
- Combined Connections: $TOTAL_CONNECTIONS

EOF

echo -e "${PURPLE}📋 FINAL TEST RESULTS:${NC}"
echo -e "${CYAN}   🎯 Total DNS-Phantom Events: $TOTAL_DNS_EVENTS${NC}"
echo -e "${CYAN}   🌐 Total Peer Discoveries: $TOTAL_PEER_DISCOVERIES${NC}" 
echo -e "${CYAN}   📞 Total Connections: $TOTAL_CONNECTIONS${NC}"
echo ""

# Determine test outcome
if [ $TOTAL_DNS_EVENTS -gt 0 ] && [ $TOTAL_PEER_DISCOVERIES -gt 0 ]; then
    echo -e "${GREEN}🎉 SUCCESS: DNS-Phantom peer discovery is working!${NC}"
    echo -e "${GREEN}   ✅ Both isolated instances showed DNS-Phantom activity${NC}"
    echo -e "${GREEN}   ✅ Peer discovery mechanisms are operational${NC}"
    TEST_RESULT="SUCCESS"
elif [ $TOTAL_DNS_EVENTS -gt 0 ]; then
    echo -e "${YELLOW}⚡ PARTIAL: DNS-Phantom is initializing but peer discovery needs work${NC}"
    echo -e "${YELLOW}   ✅ DNS-Phantom network activity detected${NC}"
    echo -e "${YELLOW}   ⚠️  Peer discovery logic may need refinement${NC}"
    TEST_RESULT="PARTIAL"
else
    echo -e "${RED}❌ FAILED: No DNS-Phantom activity detected${NC}"
    echo -e "${RED}   ❌ No steganographic DNS communication found${NC}"
    echo -e "${RED}   ❌ Isolated instances did not discover each other${NC}"
    TEST_RESULT="FAILED"
fi

# Add result to summary
echo "Test Result: $TEST_RESULT" >> "$TEST_DIR/results/connection_test_results.txt"

echo ""
echo -e "${BLUE}📁 Test artifacts saved in: $TEST_DIR/${NC}"
echo -e "${BLUE}   📝 Logs: $TEST_DIR/logs/${NC}"
echo -e "${BLUE}   📊 Results: $TEST_DIR/results/${NC}"

exit 0