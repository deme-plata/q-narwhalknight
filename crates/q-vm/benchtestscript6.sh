#!/bin/bash
# Multi-Node DAGKnight VM Narwhal-Bullshark Consensus Test Script
# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
NODE_COUNTS=(4 8 16)  # Test with 4, 8, and 16 nodes
TRANSACTIONS=1000
BATCH_SIZE=100
DURATION=60
OUTPUT_DIR="multi_node_results"
TIMEOUT=300  # Set a reasonable timeout limit in seconds (5 minutes)

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}   Multi-Node DAGKnight VM Consensus Test Suite        ${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

# Check if narwhal_bullshark_bench exists
if [ ! -f "./target/release/narwhal_bullshark_bench" ]; then
    echo -e "${GREEN}Building narwhal_bullshark_bench...${NC}"
    cargo build --release --bin narwhal_bullshark_bench
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build benchmark tool. Please check error messages.${NC}"
        exit 1
    fi
fi

# Function to run a single test
run_test() {
    local nodes=$1
    local output_file="${OUTPUT_DIR}/nodes_${nodes}_results.txt"
    
    echo -e "${GREEN}Running test with ${nodes} nodes...${NC}"
    echo -e "${BLUE}-----------------------------------------------------${NC}"
    echo -e "Nodes: ${nodes}"
    echo -e "Transactions: ${TRANSACTIONS}"
    echo -e "Batch size: ${BATCH_SIZE}"
    echo -e "Duration: ${DURATION} seconds"
    echo -e "${BLUE}-----------------------------------------------------${NC}"
    
    # Create node-specific output directory
    mkdir -p "${OUTPUT_DIR}/test_${nodes}"
    
    # Run the test with timeout and capture output
    echo -e "Starting test execution (timeout: ${TIMEOUT}s)..."
    timeout ${TIMEOUT} ./target/release/narwhal_bullshark_bench \
        --nodes ${nodes} \
        --transactions ${TRANSACTIONS} \
        --batch-size ${BATCH_SIZE} \
        --coins 200 \
        --wallets 100 \
        --mode multi-coin \
        --duration ${DURATION} \
        --output-dir "${OUTPUT_DIR}/test_${nodes}" > ${output_file} 2>&1
    
    # Check the result of the test
    if [ $? -eq 124 ]; then
        echo -e "${RED}Test timed out after ${TIMEOUT} seconds.${NC}"
        echo "Test timed out after ${TIMEOUT} seconds." >> ${output_file}
        echo "${nodes},TIMEOUT,TIMEOUT" >> ${OUTPUT_DIR}/summary.csv
        return
    elif [ $? -ne 0 ]; then
        echo -e "${RED}Test failed with error code $?.${NC}"
        echo "See ${output_file} for details."
        echo "${nodes},ERROR,ERROR" >> ${OUTPUT_DIR}/summary.csv
        return
    fi
    
    # Print test progress in real-time
    echo -e "${GREEN}Test output (last 20 lines):${NC}"
    tail -n 20 ${output_file}
    
    # Extract the overall TPS from the results
    local tps=$(grep "Overall TPS:" ${output_file} | tail -1 | awk '{print $3}')
    
    if [ -z "$tps" ]; then
        echo -e "${RED}Could not extract TPS data from output.${NC}"
        echo "${nodes},NO_DATA,NO_DATA" >> ${OUTPUT_DIR}/summary.csv
        return
    fi
    
    # Extract node TPS values and calculate average
    local node_tps_sum=0
    local node_tps_count=0
    local node_tps_values=""
    
    for i in $(seq 0 $((nodes-1))); do
        local node_tps=$(grep -A 3 "node_${i}:" ${output_file} | grep "TPS:" | awk '{print $2}')
        if [ ! -z "$node_tps" ]; then
            node_tps_sum=$(echo "$node_tps_sum + $node_tps" | bc -l)
            node_tps_count=$((node_tps_count + 1))
            node_tps_values="${node_tps_values},${node_tps}"
        fi
    done
    
    # Calculate average node TPS if we have data
    local avg_node_tps="N/A"
    if [ $node_tps_count -gt 0 ]; then
        avg_node_tps=$(echo "scale=2; $node_tps_sum / $node_tps_count" | bc -l)
    fi
    
    echo -e "${GREEN}Test with ${nodes} nodes completed.${NC}"
    echo -e "Overall TPS: ${tps}"
    echo -e "Average Node TPS: ${avg_node_tps}"
    echo ""
    
    # Save TPS values to summary file
    echo "${nodes},${tps},${avg_node_tps}${node_tps_values}" >> ${OUTPUT_DIR}/summary.csv
    
    # Copy the detailed results file if it exists
    if [ -f "${OUTPUT_DIR}/test_${nodes}/multi_coin_results.txt" ]; then
        cp "${OUTPUT_DIR}/test_${nodes}/multi_coin_results.txt" "${OUTPUT_DIR}/detailed_${nodes}_nodes.txt"
    fi
}

# Create summary CSV header
echo "Nodes,Overall TPS,Avg Node TPS,Node0,Node1,Node2..." > ${OUTPUT_DIR}/summary.csv

# Run tests for each node count
for nodes in "${NODE_COUNTS[@]}"; do
    run_test ${nodes}
    
    # Short pause between tests
    sleep 5
done

echo -e "${GREEN}All tests completed.${NC}"
echo -e "Results have been saved to ${OUTPUT_DIR}/summary.csv"

# Generate a simple plot if gnuplot is available
if command -v gnuplot &> /dev/null; then
    echo -e "${GREEN}Generating plot with gnuplot...${NC}"
    
    # Create gnuplot script
    cat > ${OUTPUT_DIR}/plot.gp << EOF
set terminal png size 800,600
set output "${OUTPUT_DIR}/scaling_results.png"
set title "DAGKnight VM Narwhal-Bullshark Multi-Coin Scaling Performance"
set xlabel "Number of Nodes"
set ylabel "Transactions Per Second (TPS)"
set grid
set datafile separator ","
set key outside
plot "${OUTPUT_DIR}/summary.csv" using 1:2 with linespoints lw 2 title "Overall TPS", \
     "${OUTPUT_DIR}/summary.csv" using 1:3 with linespoints lw 2 title "Avg Node TPS"
EOF
    # Run gnuplot
    gnuplot ${OUTPUT_DIR}/plot.gp
    
    echo -e "${GREEN}Plot generated: ${OUTPUT_DIR}/scaling_results.png${NC}"
else
    echo -e "${RED}Gnuplot not found. Skipping plot generation.${NC}"
    echo -e "You can install gnuplot with: sudo apt-get install gnuplot (Debian/Ubuntu) or brew install gnuplot (macOS)"
fi

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}                 Test Suite Completed                  ${NC}"
echo -e "${BLUE}=======================================================${NC}"
