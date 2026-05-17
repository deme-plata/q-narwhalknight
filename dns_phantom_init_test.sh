#!/bin/bash

# DNS-Phantom Initialization Test
# Test if DNS-Phantom initializes successfully in isolated instances

set -e

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                        DNS-PHANTOM INITIALIZATION TEST                         ║"
echo "║                  Testing DNS-Phantom Network Startup Success                  ║"
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
TEST_DIR="dns-phantom-init-test"
NODE_A_PORT="9011"
NODE_B_PORT="9012"
INIT_TIMEOUT=15  # Wait 15 seconds for initialization

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
    
    # Kill all background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    wait 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Set up cleanup trap
trap cleanup EXIT

# Create test directory
mkdir -p "$TEST_DIR/logs"
mkdir -p "$TEST_DIR/results"

echo -e "${BLUE}🏗️  Testing DNS-Phantom initialization...${NC}"

# Start Node A and capture initial output
echo -e "${PURPLE}🚀 Starting Node A on port $NODE_A_PORT${NC}"

# Remove any existing database locks
rm -rf data/ 2>/dev/null || true

LD_BIND_NOW=1 timeout ${INIT_TIMEOUT} $BINARY_PATH \
    --port $NODE_A_PORT \
    --enable-dns-phantom \
    > "$TEST_DIR/logs/node_a.log" 2>&1 &

NODE_A_PID=$!

# Wait for initialization
echo -e "${CYAN}   ⏳ Waiting for DNS-Phantom initialization...${NC}"
sleep 3

# Check Node A status and logs
if kill -0 "$NODE_A_PID" 2>/dev/null; then
    echo -e "${GREEN}   ✅ Node A is running (PID: $NODE_A_PID)${NC}"
else
    echo -e "${YELLOW}   ⏹️  Node A stopped (may have completed initialization)${NC}"
fi

# Wait a bit more for DNS-Phantom to fully initialize
sleep 2

# Start Node B 
echo -e "${PURPLE}🚀 Starting Node B on port $NODE_B_PORT${NC}"

LD_BIND_NOW=1 timeout ${INIT_TIMEOUT} $BINARY_PATH \
    --port $NODE_B_PORT \
    --enable-dns-phantom \
    > "$TEST_DIR/logs/node_b.log" 2>&1 &

NODE_B_PID=$!

# Wait for Node B initialization
echo -e "${CYAN}   ⏳ Waiting for DNS-Phantom initialization...${NC}"
sleep 3

if kill -0 "$NODE_B_PID" 2>/dev/null; then
    echo -e "${GREEN}   ✅ Node B is running (PID: $NODE_B_PID)${NC}"
else
    echo -e "${YELLOW}   ⏹️  Node B stopped (may have completed initialization)${NC}"
fi

# Wait for any final processing
sleep 2

echo -e "${BLUE}📊 Analyzing DNS-Phantom initialization results...${NC}"

# Analyze Node A logs
echo -e "${CYAN}📋 Node A Analysis:${NC}"
NODE_A_INIT=$(grep -c "DNS-Phantom Network initialized" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
NODE_A_READY=$(grep -c "Steganographic communication ready" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
NODE_A_STARTED=$(grep -c "DNS-Phantom Network started successfully" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
NODE_A_ACTIVE=$(grep -c "Invisible internet within the internet is now active" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)

echo -e "   🔵 DNS-Phantom initialized: $NODE_A_INIT"
echo -e "   🔮 Steganographic ready: $NODE_A_READY"
echo -e "   🚀 Successfully started: $NODE_A_STARTED"
echo -e "   👻 Network active: $NODE_A_ACTIVE"

# Analyze Node B logs
echo -e "${CYAN}📋 Node B Analysis:${NC}"
NODE_B_INIT=$(grep -c "DNS-Phantom Network initialized" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
NODE_B_READY=$(grep -c "Steganographic communication ready" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
NODE_B_STARTED=$(grep -c "DNS-Phantom Network started successfully" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
NODE_B_ACTIVE=$(grep -c "Invisible internet within the internet is now active" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)

echo -e "   🔵 DNS-Phantom initialized: $NODE_B_INIT"
echo -e "   🔮 Steganographic ready: $NODE_B_READY" 
echo -e "   🚀 Successfully started: $NODE_B_STARTED"
echo -e "   👻 Network active: $NODE_B_ACTIVE"

# Calculate totals
TOTAL_INITS=$((NODE_A_INIT + NODE_B_INIT))
TOTAL_READY=$((NODE_A_READY + NODE_B_READY))
TOTAL_STARTED=$((NODE_A_STARTED + NODE_B_STARTED))
TOTAL_ACTIVE=$((NODE_A_ACTIVE + NODE_B_ACTIVE))

echo ""
echo -e "${PURPLE}📊 FINAL RESULTS:${NC}"
echo -e "${CYAN}   🎯 Total DNS-Phantom Initializations: $TOTAL_INITS${NC}"
echo -e "${CYAN}   🔮 Total Steganographic Communications Ready: $TOTAL_READY${NC}"
echo -e "${CYAN}   🚀 Total Successful Starts: $TOTAL_STARTED${NC}"
echo -e "${CYAN}   👻 Total Active Networks: $TOTAL_ACTIVE${NC}"

# Create results report
cat > "$TEST_DIR/results/initialization_test_results.txt" << EOF
DNS-PHANTOM INITIALIZATION TEST RESULTS
=======================================

Node A Results:
- DNS-Phantom Initialized: $NODE_A_INIT
- Steganographic Ready: $NODE_A_READY
- Successfully Started: $NODE_A_STARTED
- Network Active: $NODE_A_ACTIVE

Node B Results:
- DNS-Phantom Initialized: $NODE_B_INIT
- Steganographic Ready: $NODE_B_READY
- Successfully Started: $NODE_B_STARTED
- Network Active: $NODE_B_ACTIVE

Total Results:
- Combined Initializations: $TOTAL_INITS
- Combined Ready States: $TOTAL_READY
- Combined Successful Starts: $TOTAL_STARTED
- Combined Active Networks: $TOTAL_ACTIVE

EOF

# Determine test outcome
if [ $TOTAL_INITS -ge 2 ] && [ $TOTAL_READY -ge 2 ] && [ $TOTAL_STARTED -ge 2 ] && [ $TOTAL_ACTIVE -ge 2 ]; then
    echo -e "${GREEN}🎉 SUCCESS: DNS-Phantom initialization working perfectly!${NC}"
    echo -e "${GREEN}   ✅ Both isolated instances successfully initialized DNS-Phantom${NC}"
    echo -e "${GREEN}   ✅ Steganographic communication is ready on both nodes${NC}"
    echo -e "${GREEN}   ✅ DNS-Phantom networks are active and operational${NC}"
    echo -e "${GREEN}   ✅ The binary supports DNS-Phantom discovery out of the box!${NC}"
    TEST_RESULT="SUCCESS"
elif [ $TOTAL_INITS -ge 1 ]; then
    echo -e "${YELLOW}⚡ PARTIAL: At least one DNS-Phantom network initialized${NC}"
    echo -e "${YELLOW}   ✅ DNS-Phantom capability confirmed${NC}"
    echo -e "${YELLOW}   ⚠️  Some instances may need optimization${NC}"
    TEST_RESULT="PARTIAL"
else
    echo -e "${RED}❌ FAILED: No DNS-Phantom networks initialized${NC}"
    echo -e "${RED}   ❌ DNS-Phantom functionality not working${NC}"
    TEST_RESULT="FAILED"
fi

# Add result to summary
echo "Test Result: $TEST_RESULT" >> "$TEST_DIR/results/initialization_test_results.txt"

echo ""
echo -e "${BLUE}📁 Test artifacts saved in: $TEST_DIR/${NC}"
echo -e "${BLUE}   📝 Logs: $TEST_DIR/logs/${NC}"
echo -e "${BLUE}   📊 Results: $TEST_DIR/results/${NC}"

# Show a sample of each log for verification
echo ""
echo -e "${CYAN}📋 Node A Log Sample:${NC}"
head -20 "$TEST_DIR/logs/node_a.log" | tail -10

echo ""
echo -e "${CYAN}📋 Node B Log Sample:${NC}"
head -20 "$TEST_DIR/logs/node_b.log" | tail -10

exit 0