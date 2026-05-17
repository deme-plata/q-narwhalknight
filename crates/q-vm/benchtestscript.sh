#!/bin/bash
# DAGKnight VM Narwhal-Bullshark Consensus Test Script

# Configuration
NODES=4
WALLETS=10
TRANSACTIONS=1000
BATCH_SIZE=100
DURATION=60
OUTPUT_DIR="test_results"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p $OUTPUT_DIR

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}   DAGKnight VM Narwhal-Bullshark Consensus Test      ${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

# Check if the test binary is available
if [ ! -f "./target/release/narwhal-bullshark-test" ]; then
    echo -e "${RED}Test binary not found. Compiling...${NC}"
    
    # Build the test binary
    cargo build --release --bin narwhal-bullshark-test
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build test binary. Please check error messages.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Executing Narwhal-Bullshark consensus test with the following parameters:${NC}"
echo "  Nodes: $NODES"
echo "  Wallets: $WALLETS"
echo "  Transactions per wallet: $TRANSACTIONS"
echo "  Batch size: $BATCH_SIZE"
echo "  Test duration: $DURATION seconds"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run the test
echo -e "${GREEN}Starting test...${NC}"
./target/release/narwhal-bullshark-test \
    --nodes $NODES \
    --wallets $WALLETS \
    --transactions $TRANSACTIONS \
    --batch-size $BATCH_SIZE \
    --duration $DURATION \
    --output-dir $OUTPUT_DIR

if [ $? -ne 0 ]; then
    echo -e "${RED}Test failed. Please check error messages.${NC}"
    exit 1
fi

# Check if results file exists
RESULTS_FILE="$OUTPUT_DIR/narwhal_bullshark_results.txt"
if [ -f "$RESULTS_FILE" ]; then
    echo -e "${GREEN}Test completed successfully. Results:${NC}"
    echo ""
    cat $RESULTS_FILE
    
    # Extract TPS value
    TPS=$(grep "Overall TPS:" $RESULTS_FILE | awk '{print $3}')
    
    echo ""
    if (( $(echo "$TPS >= 5000" | bc -l) )); then
        echo -e "${GREEN}SUCCESS: Target TPS of 5000 was met or exceeded! (Got: $TPS)${NC}"
    else
        echo -e "${RED}WARNING: Target TPS of 5000 was not reached. (Got: $TPS)${NC}"
        echo "Consider investigating optimizations for the Narwhal-Bullshark consensus."
    fi
else
    echo -e "${RED}Test completed but no results file was found.${NC}"
fi

echo ""
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}                 Test Completed                        ${NC}"
echo -e "${BLUE}=======================================================${NC}"
