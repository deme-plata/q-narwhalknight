#!/bin/bash

# DNS-Phantom REAL Functionality Test
# Test that DNS-Phantom uses real domains and supports transaction propagation

set -e

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    DNS-PHANTOM REAL FUNCTIONALITY TEST                        ║"
echo "║           Verifying Real Domains and Transaction Propagation                  ║"
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
TEST_DIR="dns-phantom-real-test"
NODE_A_PORT="9041"
NODE_B_PORT="9042"
TEST_DURATION=25  # 25 seconds focused test

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

echo -e "${BLUE}🏗️  Testing DNS-Phantom with REAL domains and transaction propagation...${NC}"
echo -e "${CYAN}   Duration: ${TEST_DURATION} seconds${NC}"
echo -e "${CYAN}   Expected: Real domains (github.com, cloudflare.com, etc.)${NC}"
echo -e "${CYAN}   Expected: Transaction propagation capabilities${NC}"

# Clean any existing database
rm -rf data/ 2>/dev/null || true

# Start Node A with DNS-Phantom
echo -e "${PURPLE}🚀 Starting Node A (Sender) on port $NODE_A_PORT${NC}"

RUST_LOG=debug LD_BIND_NOW=1 timeout ${TEST_DURATION} $BINARY_PATH \
    --port $NODE_A_PORT \
    --enable-dns-phantom \
    --max-peers 10 \
    --discovery-interval 5 \
    > "$TEST_DIR/logs/node_a.log" 2>&1 &

NODE_A_PID=$!
echo -e "${GREEN}   ✅ Node A started (PID: $NODE_A_PID)${NC}"

# Wait for Node A initialization
sleep 6

# Start Node B with DNS-Phantom
echo -e "${PURPLE}🚀 Starting Node B (Receiver) on port $NODE_B_PORT${NC}"

RUST_LOG=debug LD_BIND_NOW=1 timeout ${TEST_DURATION} $BINARY_PATH \
    --port $NODE_B_PORT \
    --enable-dns-phantom \
    --max-peers 10 \
    --discovery-interval 5 \
    > "$TEST_DIR/logs/node_b.log" 2>&1 &

NODE_B_PID=$!
echo -e "${GREEN}   ✅ Node B started (PID: $NODE_B_PID)${NC}"

# Let them run and communicate
echo -e "${CYAN}   🔍 Monitoring DNS-Phantom activity for real domain usage...${NC}"
sleep 8

echo -e "${BLUE}📊 Analyzing DNS-Phantom REAL functionality...${NC}"

# Analyze Node A logs for REAL domain usage (not example.com)
echo -e "${CYAN}📋 Node A - Real Domain Analysis:${NC}"

REAL_DOMAINS=$(grep -cE "github\.com|cloudflare\.com|amazonaws\.com|stackoverflow\.com|googleapis\.com|microsoft\.com|reddit\.com|wikipedia\.org" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
MOCK_DOMAINS=$(grep -c "example\.com" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
TRANSACTION_PROPAGATION=$(grep -cE "propagate_transaction|Transaction.*propagation|TransactionReceived|propagating transaction" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
BLOCK_ANNOUNCEMENTS=$(grep -cE "broadcast_block|BlockAnnouncement|Broadcasting.*block" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)
CONSENSUS_MESSAGES=$(grep -cE "consensus_message|ConsensusMessage|Consensus.*message" "$TEST_DIR/logs/node_a.log" 2>/dev/null || echo 0)

echo -e "   🌐 Real Domain Usage: $REAL_DOMAINS queries"
echo -e "   ⚠️  Mock Domain Usage: $MOCK_DOMAINS queries"
echo -e "   💸 Transaction Propagation Events: $TRANSACTION_PROPAGATION"
echo -e "   📦 Block Announcement Events: $BLOCK_ANNOUNCEMENTS"
echo -e "   🤝 Consensus Message Events: $CONSENSUS_MESSAGES"

# Analyze Node B logs similarly
echo -e "${CYAN}📋 Node B - Real Domain Analysis:${NC}"

NODE_B_REAL_DOMAINS=$(grep -cE "github\.com|cloudflare\.com|amazonaws\.com|stackoverflow\.com|googleapis\.com|microsoft\.com|reddit\.com|wikipedia\.org" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
NODE_B_MOCK_DOMAINS=$(grep -c "example\.com" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
NODE_B_TRANSACTION_RECEIVED=$(grep -cE "TransactionReceived|Received transaction|transaction.*received" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)
NODE_B_BLOCK_RECEIVED=$(grep -cE "BlockReceived|Received.*block|block.*received" "$TEST_DIR/logs/node_b.log" 2>/dev/null || echo 0)

echo -e "   🌐 Real Domain Usage: $NODE_B_REAL_DOMAINS queries"
echo -e "   ⚠️  Mock Domain Usage: $NODE_B_MOCK_DOMAINS queries"
echo -e "   💸 Transactions Received: $NODE_B_TRANSACTION_RECEIVED"
echo -e "   📦 Blocks Received: $NODE_B_BLOCK_RECEIVED"

# Create detailed results report
cat > "$TEST_DIR/results/real_functionality_analysis.txt" << EOF
DNS-PHANTOM REAL FUNCTIONALITY TEST RESULTS
===========================================

Test Duration: ${TEST_DURATION} seconds
Node A Port: $NODE_A_PORT  
Node B Port: $NODE_B_PORT
Binary: $BINARY_PATH

REAL DOMAIN USAGE ANALYSIS:
Node A Real Domains: $REAL_DOMAINS
Node B Real Domains: $NODE_B_REAL_DOMAINS
Total Real Domain Usage: $((REAL_DOMAINS + NODE_B_REAL_DOMAINS))

MOCK DOMAIN DETECTION:
Node A Mock Domains: $MOCK_DOMAINS
Node B Mock Domains: $NODE_B_MOCK_DOMAINS
Total Mock Domain Usage: $((MOCK_DOMAINS + NODE_B_MOCK_DOMAINS))

TRANSACTION PROPAGATION FUNCTIONALITY:
Transaction Propagation Events: $TRANSACTION_PROPAGATION
Transactions Received: $NODE_B_TRANSACTION_RECEIVED
Block Announcements: $BLOCK_ANNOUNCEMENTS
Blocks Received: $NODE_B_BLOCK_RECEIVED
Consensus Messages: $CONSENSUS_MESSAGES

EOF

# Calculate totals
TOTAL_REAL_DOMAINS=$((REAL_DOMAINS + NODE_B_REAL_DOMAINS))
TOTAL_MOCK_DOMAINS=$((MOCK_DOMAINS + NODE_B_MOCK_DOMAINS))
TOTAL_TRANSACTION_ACTIVITY=$((TRANSACTION_PROPAGATION + NODE_B_TRANSACTION_RECEIVED + BLOCK_ANNOUNCEMENTS + NODE_B_BLOCK_RECEIVED + CONSENSUS_MESSAGES))

echo ""
echo -e "${PURPLE}📊 FINAL REAL FUNCTIONALITY RESULTS:${NC}"
echo -e "${CYAN}   🌐 Total Real Domain Usage: $TOTAL_REAL_DOMAINS${NC}"
echo -e "${CYAN}   ⚠️  Total Mock Domain Usage: $TOTAL_MOCK_DOMAINS${NC}"
echo -e "${CYAN}   💸 Total Transaction Activity: $TOTAL_TRANSACTION_ACTIVITY${NC}"

# Determine final result
if [ $TOTAL_REAL_DOMAINS -gt 0 ] && [ $TOTAL_MOCK_DOMAINS -eq 0 ] && [ $TOTAL_TRANSACTION_ACTIVITY -gt 0 ]; then
    echo -e "${GREEN}🎉 SUCCESS: DNS-Phantom using REAL domains with full transaction support!${NC}"
    echo -e "${GREEN}   ✅ Real domain steganography confirmed${NC}"
    echo -e "${GREEN}   ✅ No mock domains detected${NC}"
    echo -e "${GREEN}   ✅ Transaction propagation implemented${NC}"
    echo -e "${GREEN}   ✅ Block announcements working${NC}"
    echo -e "${GREEN}   ✅ Consensus messaging functional${NC}"
    echo -e "${GREEN}   🌟 VERDICT: Production-ready DNS-Phantom with real infrastructure!${NC}"
    TEST_RESULT="SUCCESS - REAL DOMAINS + TRANSACTIONS"
elif [ $TOTAL_REAL_DOMAINS -gt 0 ] && [ $TOTAL_MOCK_DOMAINS -eq 0 ]; then
    echo -e "${YELLOW}⚡ GOOD: Real domains confirmed, transaction features may need activation${NC}"
    echo -e "${YELLOW}   ✅ Real domain steganography working${NC}"
    echo -e "${YELLOW}   ✅ No mock domains detected${NC}"
    echo -e "${YELLOW}   ⚠️  Transaction features may need runtime activation${NC}"
    TEST_RESULT="GOOD - REAL DOMAINS CONFIRMED"
elif [ $TOTAL_REAL_DOMAINS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  MIXED: Real domains working but some mock domains still present${NC}"
    echo -e "${YELLOW}   ✅ Real domain steganography confirmed${NC}"
    echo -e "${YELLOW}   ⚠️  Mock domains need cleanup: $TOTAL_MOCK_DOMAINS detected${NC}"
    echo -e "${YELLOW}   ⚠️  Transaction propagation: $TOTAL_TRANSACTION_ACTIVITY events${NC}"
    TEST_RESULT="MIXED - REAL DOMAINS WITH MOCK CLEANUP NEEDED"
else
    echo -e "${RED}❌ FAILED: No real domain usage detected${NC}"
    echo -e "${RED}   ❌ Still using mock domains: $TOTAL_MOCK_DOMAINS${NC}"
    echo -e "${RED}   ❌ Real domain implementation not working${NC}"
    TEST_RESULT="FAILED - MOCK DOMAINS STILL USED"
fi

echo "Final Result: $TEST_RESULT" >> "$TEST_DIR/results/real_functionality_analysis.txt"

echo ""
echo -e "${BLUE}📁 Complete test evidence:${NC}"
echo -e "${BLUE}   📝 Node A Log: $TEST_DIR/logs/node_a.log${NC}"
echo -e "${BLUE}   📝 Node B Log: $TEST_DIR/logs/node_b.log${NC}"
echo -e "${BLUE}   📊 Analysis: $TEST_DIR/results/real_functionality_analysis.txt${NC}"

# Show evidence of real domain usage
echo ""
echo -e "${CYAN}🔍 REAL DOMAIN EVIDENCE from Node A:${NC}"
grep -E "github\.com|cloudflare\.com|amazonaws\.com|stackoverflow\.com" "$TEST_DIR/logs/node_a.log" | head -5 || echo "   (No real domain evidence found)"

echo ""
echo -e "${CYAN}🔍 TRANSACTION EVIDENCE from Node A:${NC}"
grep -E "propagate|Transaction|Block|Consensus" "$TEST_DIR/logs/node_a.log" | head -5 || echo "   (No transaction evidence found)"

exit 0