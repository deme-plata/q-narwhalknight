#!/bin/bash
# Byzantine Fault Tolerance Testing Script

echo "🔒 Q-NarwhalKnight Byzantine Fault Tolerance Test"
echo "================================================"
echo ""

# Test configuration
TOTAL_NODES=4
BYZANTINE_NODES=1
TEST_DURATION=30
TPS_TARGET=1000

echo "📊 Test Configuration:"
echo "  • Total Nodes: $TOTAL_NODES"
echo "  • Byzantine Nodes: $BYZANTINE_NODES (25%)"
echo "  • Test Duration: ${TEST_DURATION}s"
echo "  • Target TPS: $TPS_TARGET"
echo "  • Expected: System should maintain consensus with f=1"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local TEST_NAME=$1
    local COMMAND=$2
    local EXPECTED=$3
    
    echo -e "${YELLOW}Running: $TEST_NAME${NC}"
    
    # Execute test
    RESULT=$(eval $COMMAND 2>&1)
    
    if [[ $RESULT == *"$EXPECTED"* ]]; then
        echo -e "  ${GREEN}✅ PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "  ${RED}❌ FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test 1: Consensus with all honest nodes
echo "🧪 Test 1: Consensus with All Honest Nodes"
run_test "4 honest nodes reach consensus" \
    "cargo run --release --bin mitochondria-sim -- --validators 4 --rounds 10 --byzantine 0 2>&1 | grep -c 'consensus achieved'" \
    "10"

echo ""

# Test 2: Consensus with 1 Byzantine node
echo "🧪 Test 2: Consensus with 1 Byzantine Node (25%)"
run_test "3 honest + 1 Byzantine maintain consensus" \
    "cargo run --release --bin mitochondria-sim -- --validators 4 --rounds 10 --byzantine 1 2>&1 | grep -c 'consensus achieved'" \
    "10"

echo ""

# Test 3: No consensus with 2 Byzantine nodes
echo "🧪 Test 3: No Consensus with 2 Byzantine Nodes (50%)"
run_test "2 honest + 2 Byzantine cannot reach consensus" \
    "cargo run --release --bin mitochondria-sim -- --validators 4 --rounds 10 --byzantine 2 2>&1 | grep -c 'consensus failed'" \
    "consensus failed"

echo ""

# Test 4: Finality time measurement
echo "🧪 Test 4: Consensus Finality Time"
echo -e "${YELLOW}Measuring finality time...${NC}"

START_TIME=$(date +%s%N)
cargo run --release --bin mitochondria-sim -- --validators 4 --rounds 1 --measure-finality 2>&1 | grep "finality" > /tmp/finality.log
END_TIME=$(date +%s%N)

FINALITY_TIME=$(( ($END_TIME - $START_TIME) / 1000000 ))
echo "  Finality achieved in: ${FINALITY_TIME}ms"

if [ $FINALITY_TIME -lt 3000 ]; then
    echo -e "  ${GREEN}✅ PASSED${NC} (Target: <3000ms)"
    ((TESTS_PASSED++))
else
    echo -e "  ${RED}❌ FAILED${NC} (Target: <3000ms)"
    ((TESTS_FAILED++))
fi

echo ""

# Test 5: Double-spend prevention
echo "🧪 Test 5: Double-Spend Prevention"
run_test "Byzantine node double-spend rejected" \
    "cargo run --release --bin mitochondria-sim -- --validators 4 --test-double-spend 2>&1 | grep -c 'double-spend rejected'" \
    "rejected"

echo ""

# Test 6: Network partition recovery
echo "🧪 Test 6: Network Partition Recovery"
echo -e "${YELLOW}Simulating network partition...${NC}"

# This would normally involve more complex network simulation
run_test "Network recovers after partition" \
    "cargo run --release --bin mitochondria-sim -- --validators 4 --simulate-partition 2>&1 | grep -c 'partition recovered'" \
    "recovered"

echo ""

# Summary
echo "════════════════════════════════════════"
echo "📊 Byzantine Fault Tolerance Test Results"
echo "════════════════════════════════════════"
echo -e "  Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "  Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All Byzantine fault tolerance tests PASSED!${NC}"
    echo "The system correctly tolerates f=1 Byzantine nodes out of n=4"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Review the results above.${NC}"
    exit 1
fi