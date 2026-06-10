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
if [ ! -f "./target/release/narwhal_bullshark_bench" ]; then
    echo -e "${RED}Test binary not found. Compiling...${NC}"
    
    # Build the test binary
    cargo build --release --bin narwhal_bullshark_bench
    
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
./target/release/narwhal_bullshark_bench \
    --nodes $NODES \
    --transactions $TRANSACTIONS \
    --batch-size $BATCH_SIZE \
    --mode stress \
    --duration $DURATION

if [ $? -ne 0 ]; then
    echo -e "${RED}Test failed. Please check error messages.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Test completed. Check the console output for results.${NC}"
echo -e "${GREEN}The benchmark natively reports TPS metrics, so we'll use those.${NC}"

echo ""
echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}                 Test Completed                        ${NC}"
echo -e "${BLUE}=======================================================${NC}"
