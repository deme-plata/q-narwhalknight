#!/bin/bash
# Q-NarwhalKnight Network Monitor
# Real-time monitoring of 10-node anonymous BFT consensus network

set -euo pipefail

echo "📊 Q-NARWHALKNIGHT NETWORK MONITOR"
echo "=================================="
echo "Monitoring anonymous quantum BFT consensus network"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Server Alpha Nodes
ALPHA_NODES=("alice:8001" "bob:8002" "charlie:8003" "diana:8004" "eve:8005")
ALPHA_APIS=(9001 9002 9003 9004 9005)

# Server Beta Nodes (when deployed)
BETA_NODES=("frank:8006" "grace:8007" "henry:8008" "iris:8009" "jack:8010")
BETA_APIS=(9006 9007 9008 9009 9010)

check_node_status() {
    local name=$1
    local port=$2
    
    if curl -s -f "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ ${name}${NC} - Running"
        return 0
    else
        echo -e "${RED}❌ ${name}${NC} - Down"
        return 1
    fi
}

get_consensus_status() {
    local name=$1
    local port=$2
    
    local status=$(curl -s "http://127.0.0.1:${port}/status" 2>/dev/null)
    if [[ $? -eq 0 && -n "$status" ]]; then
        local round=$(echo "$status" | grep -o '"current_round":[0-9]*' | cut -d':' -f2)
        local finalized=$(echo "$status" | grep -o '"vertices_finalized":[0-9]*' | cut -d':' -f2)
        echo -e "${BLUE}🔄 ${name}${NC} - Round: ${round:-0}, Finalized: ${finalized:-0}"
    else
        echo -e "${YELLOW}⏳ ${name}${NC} - Status unavailable"
    fi
}

monitor_network() {
    clear
    echo -e "${PURPLE}🧅 Q-NARWHALKNIGHT ANONYMOUS BFT NETWORK${NC}"
    echo "========================================"
    echo "$(date)"
    echo ""
    
    echo -e "${CYAN}🌟 SERVER ALPHA NODES (5/5):${NC}"
    echo "----------------------------"
    local alpha_running=0
    for i in "${!ALPHA_NODES[@]}"; do
        local node_info="${ALPHA_NODES[$i]}"
        local name="${node_info%%:*}"
        local api_port="${ALPHA_APIS[$i]}"
        
        if check_node_status "$name" "$api_port"; then
            ((alpha_running++))
            get_consensus_status "$name" "$api_port"
        fi
    done
    echo ""
    
    echo -e "${CYAN}🌟 SERVER BETA NODES (0/5):${NC}"
    echo "----------------------------"
    local beta_running=0
    echo -e "${YELLOW}⏳ Waiting for Server Beta deployment...${NC}"
    echo ""
    
    # Network summary
    local total_running=$((alpha_running + beta_running))
    echo -e "${PURPLE}📊 NETWORK SUMMARY:${NC}"
    echo "==================="
    echo -e "🔢 Total Nodes: ${total_running}/10"
    echo -e "🛡️ Byzantine Tolerance: $(( (total_running - 1) / 3 )) malicious nodes"
    echo -e "🎯 Consensus Threshold: $(( (total_running * 2) / 3 + 1 )) votes needed"
    echo ""
    
    if [[ $total_running -ge 7 ]]; then
        echo -e "${GREEN}✅ BFT CONSENSUS READY${NC}"
        echo -e "Network can tolerate Byzantine faults"
    elif [[ $total_running -ge 4 ]]; then
        echo -e "${YELLOW}⚠️ PARTIAL NETWORK${NC}"
        echo -e "Need $(( 7 - total_running )) more nodes for full BFT"
    else
        echo -e "${RED}❌ INSUFFICIENT NODES${NC}"
        echo -e "Need $(( 7 - total_running )) more nodes for consensus"
    fi
    echo ""
    
    # Tor status
    if pgrep tor > /dev/null; then
        echo -e "${GREEN}🧅 Tor Service: Running${NC}"
    else
        echo -e "${RED}🧅 Tor Service: Down${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}📈 MONITORING COMMANDS:${NC}"
    echo "======================="
    echo "# Watch this monitor: ./network-deployment/network-monitor.sh"
    echo "# View all logs: tail -f real-logs/node-*.log"
    echo "# Test consensus: curl http://127.0.0.1:9001/consensus/status"
    echo "# Byzantine metrics: curl http://127.0.0.1:9001/metrics/byzantine"
    echo ""
}

# Main monitoring loop
if [[ "${1:-}" == "--loop" ]]; then
    while true; do
        monitor_network
        echo "Refreshing in 10 seconds... (Ctrl+C to exit)"
        sleep 10
    done
else
    monitor_network
fi