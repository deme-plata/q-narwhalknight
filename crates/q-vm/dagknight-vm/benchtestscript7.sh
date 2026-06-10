#!/bin/bash
# Simplified Multi-Node DAGKnight VM Test Script

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration - test only 4 and 8 nodes since you mentioned those work
NODE_COUNTS=(4 8)
TRANSACTIONS=1000
BATCH_SIZE=100
DURATION=60
OUTPUT_DIR="multi_node_results"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}   Multi-Node DAGKnight VM Consensus Test Suite        ${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo ""

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

    # Run the test and capture output
    ./target/release/narwhal_bullshark_bench \
        --nodes ${nodes} \
        --transactions ${TRANSACTIONS} \
        --batch-size ${BATCH_SIZE} \
        --mode stress \
        --duration ${DURATION} > ${output_file} 2>&1

    # Display the last few lines to see what happened
    echo -e "${BLUE}Test output (last 10 lines):${NC}"
    tail -n 10 ${output_file}

    # Extract the overall TPS from the results
    local tps=$(grep "Overall throughput:" ${output_file} | awk '{print $3}')
    local vm_tps=$(grep "VM reported TPS:" ${output_file} | awk '{print $4}')

    if [ -z "$tps" ] || [ -z "$vm_tps" ]; then
        echo -e "${RED}Could not extract TPS data. Check the output file for errors.${NC}"
        echo "${nodes},ERROR,ERROR" >> ${OUTPUT_DIR}/summary.csv
        return
    fi

    echo -e "${GREEN}Test with ${nodes} nodes completed.${NC}"
    echo -e "Overall TPS: ${tps}"
    echo -e "VM reported TPS: ${vm_tps}"
    echo ""

    # Save TPS values to summary file
    echo "${nodes},${tps},${vm_tps}" >> ${OUTPUT_DIR}/summary.csv
}

# Create summary CSV header
echo "Nodes,Overall TPS,VM TPS" > ${OUTPUT_DIR}/summary.csv

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
set title "DAGKnight VM Narwhal-Bullshark Scaling Performance"
set xlabel "Number of Nodes"
set ylabel "Transactions Per Second (TPS)"
set grid
set datafile separator ","
set key outside
plot "${OUTPUT_DIR}/summary.csv" using 1:2 with linespoints title "Overall TPS", \
     "${OUTPUT_DIR}/summary.csv" using 1:3 with linespoints title "VM TPS"
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
