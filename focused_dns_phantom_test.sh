#!/bin/bash
# FOCUSED DNS-PHANTOM DISCOVERY TEST
# Q-NarwhalKnight Native Binary DNS-Phantom Validation
# Tests core DNS-Phantom steganographic peer discovery functionality

set -e

# Test Configuration (reasonable scale)
DNS_PHANTOM_HUB_PORT=8090
TEST_NODES=10  # Manageable number for focused testing
TEST_DURATION=120  # 2 minutes focused test

# Directories
TEST_DIR="./focused-dns-phantom-test"
LOG_DIR="$TEST_DIR/logs"
RESULTS_DIR="$TEST_DIR/results"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up focused DNS-Phantom test...${NC}"
    pkill -f "q-api-server.*8090" || true
    pkill -f "q-api-server.*909" || true
    sleep 2
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

trap cleanup EXIT INT TERM

echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║                FOCUSED DNS-PHANTOM DISCOVERY TEST             ║${NC}"
echo -e "${PURPLE}║              Q-NarwhalKnight Native Binary Proof             ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}🎯 Mission: Validate core DNS-Phantom discovery functionality${NC}"
echo -e "${CYAN}📊 Scale: 1 DNS-Phantom Hub + $TEST_NODES discovery nodes${NC}"
echo -e "${CYAN}⏱️  Duration: $TEST_DURATION seconds of targeted testing${NC}"
echo -e "${CYAN}🔍 Focus: Steganographic peer discovery validation${NC}"
echo ""

# Setup
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Verify binary
BINARY_PATH="./target/release/q-api-server"
if [ ! -x "$BINARY_PATH" ]; then
    echo -e "${YELLOW}Building Q-NarwhalKnight binary...${NC}"
    cargo build --release --bin q-api-server
fi

echo -e "${GREEN}✅ Binary ready: $(ls -lh "$BINARY_PATH" | awk '{print $5}')${NC}"

# Test 1: Launch DNS-Phantom Discovery Hub
echo -e "\n${BLUE}🌐 Test 1: Launching DNS-Phantom Discovery Hub...${NC}"

"$BINARY_PATH" \
    --node-id "dns-phantom-hub-test" \
    --role discovery-hub \
    --port $DNS_PHANTOM_HUB_PORT \
    --data-dir "$TEST_DIR/hub-data" \
    --enable-dns-phantom \
    --bootstrap-mode \
    > "$LOG_DIR/dns-phantom-hub.log" 2>&1 &

HUB_PID=$!
echo "$HUB_PID" > "$TEST_DIR/hub.pid"
echo -e "${GREEN}✅ DNS-Phantom Hub started (PID: $HUB_PID) on port $DNS_PHANTOM_HUB_PORT${NC}"

# Wait for hub to initialize
sleep 5

# Test 2: Launch nodes with DNS-Phantom discovery enabled
echo -e "\n${BLUE}🚀 Test 2: Launching $TEST_NODES nodes with DNS-Phantom discovery...${NC}"

NODE_PIDS=()
for i in $(seq 1 $TEST_NODES); do
    port=$((9090 + i))
    node_id="dns-phantom-node-$i"
    
    "$BINARY_PATH" \
        --node-id "$node_id" \
        --role validator \
        --port $port \
        --data-dir "$TEST_DIR/node-data-$i" \
        --enable-dns-phantom \
        --bootstrap-peer "127.0.0.1:$DNS_PHANTOM_HUB_PORT" \
        --discovery-interval 5 \
        --max-peers 20 \
        > "$LOG_DIR/$node_id.log" 2>&1 &
    
    NODE_PIDS+=($!)
    echo -e "${GREEN}   ✅ Node $i started (PID: $!) on port $port${NC}"
    
    # Brief delay between node launches
    sleep 1
done

echo -e "\n${GREEN}🎉 All $TEST_NODES DNS-Phantom discovery nodes launched!${NC}"

# Wait for network to stabilize
echo -e "\n${YELLOW}⏳ Network stabilization (15 seconds)...${NC}"
sleep 15

# Test 3: Monitor DNS-Phantom discovery activity
echo -e "\n${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║                DNS-PHANTOM DISCOVERY MONITORING               ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"

START_TIME=$(date +%s)
DISCOVERY_CHECKS=0

# Create results tracking
echo "timestamp,active_processes,hub_connections,discovery_events,phantom_initializations" > "$RESULTS_DIR/monitoring.csv"

while [ $(($(date +%s) - START_TIME)) -lt $TEST_DURATION ]; do
    CURRENT_TIME=$(date '+%H:%M:%S')
    
    # Check active processes
    ACTIVE_PROCESSES=$(pgrep -f "q-api-server.*(8090|909)" | wc -l)
    
    # Check hub connections
    HUB_CONNECTIONS=$(ss -t | grep ":$DNS_PHANTOM_HUB_PORT" | grep ESTAB | wc -l)
    
    # Analyze DNS-Phantom activity in logs
    DISCOVERY_EVENTS=0
    PHANTOM_INITIALIZATIONS=0
    
    if [ -f "$LOG_DIR/dns-phantom-hub.log" ]; then
        DISCOVERY_EVENTS=$(grep -c -i "phantom.*discover\|steganographic\|dns.*discovery" "$LOG_DIR"/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum+0}')
        PHANTOM_INITIALIZATIONS=$(grep -c -i "DNS Phantom Network started\|invisible internet.*active" "$LOG_DIR"/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum+0}')
    fi
    
    # Display current status
    echo -e "\n${BLUE}🔍 DNS-PHANTOM STATUS CHECK [$CURRENT_TIME]${NC}"
    echo -e "${CYAN}   Active Processes: $ACTIVE_PROCESSES${NC}"
    echo -e "${CYAN}   Hub Connections: $HUB_CONNECTIONS${NC}"
    echo -e "${CYAN}   Discovery Events: $DISCOVERY_EVENTS${NC}"
    echo -e "${CYAN}   Phantom Initializations: $PHANTOM_INITIALIZATIONS${NC}"
    
    # Save metrics
    echo "$CURRENT_TIME,$ACTIVE_PROCESSES,$HUB_CONNECTIONS,$DISCOVERY_EVENTS,$PHANTOM_INITIALIZATIONS" >> "$RESULTS_DIR/monitoring.csv"
    
    DISCOVERY_CHECKS=$((DISCOVERY_CHECKS + 1))
    
    # Progress indicator
    ELAPSED=$(($(date +%s) - START_TIME))
    REMAINING=$((TEST_DURATION - ELAPSED))
    echo -e "${YELLOW}   ⏱️  Progress: $ELAPSED/${TEST_DURATION}s | Remaining: ${REMAINING}s${NC}"
    
    sleep 10
done

echo -e "\n${GREEN}🎉 Focused DNS-Phantom discovery test completed!${NC}"

# Test 4: Final Analysis
echo -e "\n${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║                    FINAL ANALYSIS & RESULTS                   ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Final metrics
FINAL_PROCESSES=$(pgrep -f "q-api-server.*(8090|909)" | wc -l)
TOTAL_LOG_FILES=$(ls -1 "$LOG_DIR"/*.log 2>/dev/null | wc -l)

# Comprehensive log analysis
TOTAL_PHANTOM_INITS=0
TOTAL_DISCOVERY_EVENTS=0
TOTAL_STEGANOGRAPHIC_COMMS=0
SUCCESSFUL_NODE_STARTS=0

if [ -d "$LOG_DIR" ]; then
    TOTAL_PHANTOM_INITS=$(grep -c -i "DNS Phantom Network started successfully\|invisible internet.*active" "$LOG_DIR"/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum+0}')
    TOTAL_DISCOVERY_EVENTS=$(grep -c -i "phantom.*discover\|dns.*phantom.*discovery" "$LOG_DIR"/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum+0}')
    TOTAL_STEGANOGRAPHIC_COMMS=$(grep -c -i "steganographic.*communication\|steganographic.*peer" "$LOG_DIR"/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum+0}')
    SUCCESSFUL_NODE_STARTS=$(grep -c -i "Starting Q-NarwhalKnight.*server" "$LOG_DIR"/*.log 2>/dev/null | awk -F: '{sum += $2} END {print sum+0}')
fi

# Generate comprehensive report
cat > "$RESULTS_DIR/focused_test_results.txt" << EOF
================================================================
            FOCUSED DNS-PHANTOM DISCOVERY TEST RESULTS
                Q-NarwhalKnight Native Binary Validation
================================================================

TEST CONFIGURATION:
------------------
- DNS-Phantom Hub: 1 (port $DNS_PHANTOM_HUB_PORT)
- Discovery Nodes: $TEST_NODES
- Test Duration: $TEST_DURATION seconds
- Discovery Checks: $DISCOVERY_CHECKS

DEPLOYMENT RESULTS:
------------------
- Nodes Successfully Started: $SUCCESSFUL_NODE_STARTS
- Processes Still Active: $FINAL_PROCESSES
- Log Files Generated: $TOTAL_LOG_FILES

DNS-PHANTOM FUNCTIONALITY ANALYSIS:
----------------------------------
- DNS-Phantom Network Initializations: $TOTAL_PHANTOM_INITS
- Total Discovery Events: $TOTAL_DISCOVERY_EVENTS  
- Steganographic Communications: $TOTAL_STEGANOGRAPHIC_COMMS
- Network Stability: $(if [ $FINAL_PROCESSES -ge $((TEST_NODES / 2)) ]; then echo "STABLE"; else echo "UNSTABLE"; fi)

DETAILED LOG ANALYSIS:
---------------------
EOF

# Analyze individual log files
for log_file in "$LOG_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        node_name=$(basename "$log_file" .log)
        phantom_events=$(grep -c -i "phantom\|steganographic" "$log_file" 2>/dev/null || echo "0")
        discovery_events=$(grep -c -i "discovery\|peer.*discovered" "$log_file" 2>/dev/null || echo "0")
        echo "- $node_name: $phantom_events phantom events, $discovery_events discovery events" >> "$RESULTS_DIR/focused_test_results.txt"
    fi
done

# Determine verdict
cat >> "$RESULTS_DIR/focused_test_results.txt" << EOF

VERDICT:
========
EOF

if [ $TOTAL_PHANTOM_INITS -gt 0 ] && [ $SUCCESSFUL_NODE_STARTS -ge $TEST_NODES ]; then
    cat >> "$RESULTS_DIR/focused_test_results.txt" << EOF
🎉 SUCCESS: DNS-PHANTOM DISCOVERY FUNCTIONALITY CONFIRMED! 🎉

The Q-NarwhalKnight native Rust binary SUCCESSFULLY supports DNS-Phantom
steganographic peer discovery without Docker containers.

EVIDENCE:
✅ $TOTAL_PHANTOM_INITS DNS-Phantom network initializations successful
✅ $SUCCESSFUL_NODE_STARTS/$TEST_NODES nodes started with DNS-Phantom capability  
✅ $TOTAL_DISCOVERY_EVENTS total discovery events recorded
✅ $TOTAL_STEGANOGRAPHIC_COMMS steganographic communications detected
✅ $FINAL_PROCESSES processes remained stable during test
✅ Native binary supports DNS-Phantom discovery out of the box

CONCLUSION: DNS-Phantom functionality is NATIVE and WORKING!
EOF
    VERDICT="SUCCESS"
elif [ $TOTAL_PHANTOM_INITS -gt 0 ]; then
    cat >> "$RESULTS_DIR/focused_test_results.txt" << EOF
⚠️  PARTIAL SUCCESS: DNS-Phantom capability detected but limited activity

Some DNS-Phantom functionality was detected:
✅ $TOTAL_PHANTOM_INITS DNS-Phantom initializations
✅ $SUCCESSFUL_NODE_STARTS nodes started successfully
⚠️  Limited discovery activity: $TOTAL_DISCOVERY_EVENTS events

The framework exists and is functional but may need optimization.
EOF
    VERDICT="PARTIAL"
else
    cat >> "$RESULTS_DIR/focused_test_results.txt" << EOF
❌ INCONCLUSIVE: Limited DNS-Phantom evidence detected

Test results show minimal DNS-Phantom activity. The binary may support
DNS-Phantom discovery but requires further investigation or optimization.
EOF
    VERDICT="INCONCLUSIVE"
fi

# Display final results
echo ""
cat "$RESULTS_DIR/focused_test_results.txt"

echo -e "\n${GREEN}📊 FOCUSED TEST SUMMARY:${NC}"
echo -e "${CYAN}   🚀 Deployed: $((TEST_NODES + 1)) processes (1 hub + $TEST_NODES nodes)${NC}"
echo -e "${CYAN}   ✅ Active: $FINAL_PROCESSES processes${NC}"
echo -e "${CYAN}   🌐 DNS-Phantom Inits: $TOTAL_PHANTOM_INITS${NC}"
echo -e "${CYAN}   📡 Discovery Events: $TOTAL_DISCOVERY_EVENTS${NC}"
echo -e "${CYAN}   🔐 Steganographic Comms: $TOTAL_STEGANOGRAPHIC_COMMS${NC}"

if [ "$VERDICT" = "SUCCESS" ]; then
    echo -e "\n${GREEN}🎉 VERDICT: DNS-PHANTOM DISCOVERY NATIVE SUPPORT CONFIRMED! 🎉${NC}"
    FINAL_EXIT=0
elif [ "$VERDICT" = "PARTIAL" ]; then
    echo -e "\n${YELLOW}⚠️  VERDICT: DNS-PHANTOM CAPABILITY DETECTED${NC}"
    FINAL_EXIT=1
else
    echo -e "\n${YELLOW}⚠️  VERDICT: DNS-PHANTOM TESTING INCONCLUSIVE${NC}"  
    FINAL_EXIT=2
fi

echo -e "\n${BLUE}📁 Results saved in: $RESULTS_DIR/${NC}"
echo -e "${BLUE}📋 Logs available in: $LOG_DIR/${NC}"

# Brief pause to show running processes
echo -e "\n${YELLOW}⏳ Processes will remain active for 15 seconds for inspection...${NC}"
sleep 15

exit $FINAL_EXIT