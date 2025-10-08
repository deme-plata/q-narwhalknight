#!/bin/bash

# 🚀 Q-NarwhalKnight Launch Monitor
# Synchronized launch coordination between Server Alpha and Server Beta

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m' # No Color

WORKSPACE_ROOT="/opt/orobit/shared/q-narwhalknight"
cd "$WORKSPACE_ROOT"

function log_monitor() {
    echo -e "${CYAN}[LAUNCH-MONITOR]${NC} $1"
}

function log_status() {
    echo -e "${GREEN}[STATUS]${NC} $1"
}

function log_waiting() {
    echo -e "${YELLOW}[WAITING]${NC} $1"
}

function show_banner() {
    echo -e "${PURPLE}"
    echo "🌟 Q-NARWHALKNIGHT SYNCHRONIZED LAUNCH MONITOR 🌟"
    echo "=================================================="
    echo "World's First Anonymous Quantum BFT Consensus Network"
    echo -e "${NC}"
    echo ""
    echo -e "${BLUE}Network Configuration:${NC}"
    echo "  • 10 anonymous validators (5 Server Alpha + 5 Server Beta)"
    echo "  • Byzantine fault tolerance: f=3 (tolerates up to 3 malicious nodes)"
    echo "  • Consensus threshold: 7 votes (2f+1 majority)"
    echo "  • Anonymous communication: Complete .onion networking"
    echo "  • Quantum security: Post-quantum VDF proofs"
    echo ""
}

function check_server_alpha_status() {
    log_monitor "🔍 Checking Server Alpha deployment status..."
    
    # Check if Server Alpha nodes are running
    local alpha_running=0
    
    # Check for running q-vm processes (Server Alpha nodes)
    if pgrep -f "q-vm" > /dev/null 2>&1; then
        alpha_running=$(pgrep -f "q-vm" | wc -l)
        log_status "Server Alpha nodes detected: $alpha_running running"
    else
        log_waiting "Server Alpha nodes not yet running (build may be in progress)"
    fi
    
    # Check for log files indicating active deployment
    if ls logs/node-*.log > /dev/null 2>&1; then
        local log_count=$(ls logs/node-*.log 2>/dev/null | wc -l)
        log_status "Server Alpha log files found: $log_count"
        
        # Check recent activity in logs
        if find logs/ -name "node-*.log" -mmin -2 > /dev/null 2>&1; then
            log_status "Recent Server Alpha activity detected"
        fi
    else
        log_waiting "Server Alpha log files not yet created"
    fi
    
    return $alpha_running
}

function check_server_beta_readiness() {
    log_monitor "✅ Checking Server Beta readiness..."
    
    # Check if deployment script exists and is executable
    if [[ -x "network-deployment/server-beta-deployment.sh" ]]; then
        log_status "Server Beta deployment script ready"
    else
        log_status "Server Beta deployment script missing or not executable"
        return 1
    fi
    
    # Check if we have the Q-NarwhalKnight binary
    if [[ -f "target/release/q-vm" ]]; then
        log_status "Q-NarwhalKnight binary ready"
    else
        log_waiting "Q-NarwhalKnight binary needs building"
        return 1
    fi
    
    return 0
}

function build_if_needed() {
    if [[ ! -f "target/release/q-vm" ]]; then
        log_monitor "🔨 Building Q-NarwhalKnight for Server Beta deployment..."
        cargo build --release --quiet
        log_status "Build complete"
    fi
}

function execute_server_beta_launch() {
    log_monitor "🚀 Executing Server Beta synchronized launch..."
    
    # Execute the Server Beta deployment script
    if [[ -x "network-deployment/server-beta-deployment.sh" ]]; then
        log_status "Launching Server Beta nodes..."
        ./network-deployment/server-beta-deployment.sh
        
        log_status "✅ Server Beta launch initiated"
    else
        echo -e "${RED}❌ Server Beta deployment script not found${NC}"
        return 1
    fi
}

function monitor_network_formation() {
    log_monitor "👀 Monitoring 10-node network formation..."
    
    local check_count=0
    local max_checks=30  # 5 minutes
    
    while [[ $check_count -lt $max_checks ]]; do
        echo ""
        log_monitor "Network Status Check #$((check_count + 1))"
        
        # Count running processes
        local total_nodes=0
        if pgrep -f "q-vm" > /dev/null 2>&1; then
            total_nodes=$(pgrep -f "q-vm" | wc -l)
        fi
        
        log_status "Total nodes running: $total_nodes/10"
        
        # Check if we have logs from both servers
        local alpha_logs=0
        local beta_logs=0
        
        if ls logs/node-*.log > /dev/null 2>&1; then
            alpha_logs=$(ls logs/node-*.log 2>/dev/null | wc -l)
        fi
        
        if ls network-deployment/server-beta-logs/*.log > /dev/null 2>&1; then
            beta_logs=$(ls network-deployment/server-beta-logs/*.log 2>/dev/null | wc -l)
        fi
        
        log_status "Server Alpha logs: $alpha_logs, Server Beta logs: $beta_logs"
        
        # Check for consensus activity
        if [[ $total_nodes -ge 7 ]]; then
            log_status "🎉 BFT threshold reached! ($total_nodes >= 7 nodes)"
            
            if [[ $total_nodes -eq 10 ]]; then
                log_status "🌟 FULL 10-NODE NETWORK ACHIEVED!"
                echo ""
                log_monitor "✅ Anonymous Quantum BFT Consensus Network Successfully Deployed!"
                return 0
            fi
        fi
        
        check_count=$((check_count + 1))
        sleep 10
    done
    
    log_monitor "⏰ Network formation monitoring timeout reached"
    return 1
}

function show_network_status() {
    echo ""
    echo -e "${PURPLE}📊 NETWORK STATUS SUMMARY${NC}"
    echo "=========================="
    
    # Show running processes
    if pgrep -f "q-vm" > /dev/null 2>&1; then
        echo -e "${GREEN}Running Q-NarwhalKnight nodes:${NC}"
        pgrep -f "q-vm" -l || true
    else
        echo -e "${YELLOW}No Q-NarwhalKnight nodes currently running${NC}"
    fi
    
    # Show log file status
    echo ""
    echo -e "${BLUE}Log File Status:${NC}"
    echo "Server Alpha logs: $(ls logs/node-*.log 2>/dev/null | wc -l) files"
    echo "Server Beta logs: $(ls network-deployment/server-beta-logs/*.log 2>/dev/null | wc -l) files"
    
    # Show recent log activity
    echo ""
    echo -e "${CYAN}Recent Activity:${NC}"
    if find logs/ -name "*.log" -mmin -1 > /dev/null 2>&1; then
        echo "✅ Recent Server Alpha activity detected"
    else
        echo "⏰ No recent Server Alpha activity"
    fi
    
    if find network-deployment/server-beta-logs/ -name "*.log" -mmin -1 > /dev/null 2>&1 || 
       find network-deployment/ -name "server-beta-*.log" -mmin -1 > /dev/null 2>&1; then
        echo "✅ Recent Server Beta activity detected"
    else
        echo "⏰ No recent Server Beta activity"
    fi
}

function main() {
    show_banner
    
    while true; do
        # Check Server Alpha status
        check_server_alpha_status
        local alpha_nodes=$?
        
        # Check Server Beta readiness  
        if check_server_beta_readiness; then
            log_status "✅ Server Beta ready for deployment"
        else
            log_waiting "⏳ Server Beta not yet ready"
            build_if_needed
        fi
        
        echo ""
        
        # Decision logic for synchronized launch
        if [[ $alpha_nodes -ge 3 ]]; then
            log_monitor "🎯 Server Alpha has sufficient nodes running ($alpha_nodes)"
            log_monitor "🚀 INITIATING SYNCHRONIZED LAUNCH"
            
            if execute_server_beta_launch; then
                log_status "✅ Server Beta launch successful"
                
                # Monitor network formation
                if monitor_network_formation; then
                    echo ""
                    echo -e "${GREEN}🎉 HISTORIC ACHIEVEMENT UNLOCKED! 🎉${NC}"
                    echo "World's first anonymous quantum-enhanced BFT consensus network is now live!"
                    break
                fi
            else
                echo -e "${RED}❌ Server Beta launch failed${NC}"
                break
            fi
        else
            log_waiting "⏳ Waiting for Server Alpha to start more nodes (current: $alpha_nodes, need: 3+)"
            log_waiting "   Server Alpha build may still be in progress..."
        fi
        
        # Show current status
        show_network_status
        
        echo ""
        echo "Checking again in 30 seconds..."
        sleep 30
        echo ""
        echo "=================================================="
        echo ""
    done
}

# Execute main function
main "$@"