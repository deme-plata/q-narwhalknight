#!/bin/bash

# 🔥 Q-NarwhalKnight Live Battle Test Status Checker
# Real-time coordination status between Server Alpha and Server Beta

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Status check function
check_status() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}🔥 Q-NARWHALKNIGHT LIVE BATTLE TEST STATUS${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}⏰ Timestamp: $(date)${NC}"
    echo ""

    # Check Server Alpha status
    echo -e "${PURPLE}🅰️  SERVER ALPHA STATUS:${NC}"
    
    if [ -f "/mnt/shared/alpha_onion_info.env" ]; then
        source /mnt/shared/alpha_onion_info.env
        echo -e "   ${GREEN}✅ Alpha Environment Ready${NC}"
        echo -e "   🧅 Onion: ${ALPHA_ONION_ADDRESS:-"Not set"}"
        echo -e "   🚪 Port: ${ALPHA_PORT:-"Not set"}"
        echo -e "   🆔 Node ID: ${ALPHA_NODE_ID:-"Not set"}"
        echo -e "   🎯 Battle Test ID: ${BATTLE_TEST_ID:-"Not set"}"
        
        # Check if Alpha node is running
        if pgrep -f "q-narwhal-validator.*alpha" > /dev/null; then
            echo -e "   ${GREEN}✅ Alpha Node Process Running${NC}"
            
            # Try to query Alpha node
            if curl -s --max-time 5 http://localhost:8333/health > /dev/null 2>&1; then
                ALPHA_PEERS=$(curl -s --max-time 3 http://localhost:8333/peers/count 2>/dev/null || echo "0")
                ALPHA_COST=$(curl -s --max-time 3 http://localhost:8333/discovery/cost 2>/dev/null || echo "0.00")
                echo -e "   ${GREEN}✅ Alpha API Responding${NC}"
                echo -e "   👥 Peers Discovered: $ALPHA_PEERS"
                echo -e "   💰 Daily Cost: \$${ALPHA_COST} ${GREEN}(FREE!)${NC}"
            else
                echo -e "   ${YELLOW}⚠️  Alpha API Not Responding${NC}"
            fi
        else
            echo -e "   ${RED}❌ Alpha Node Process Not Running${NC}"
        fi
    else
        echo -e "   ${RED}❌ Alpha Environment Not Ready${NC}"
        echo -e "   ⏳ Waiting for Server Alpha to start..."
    fi
    
    echo ""
    
    # Check Server Beta status
    echo -e "${PURPLE}🅱️  SERVER BETA STATUS:${NC}"
    
    if [ -f "/mnt/shared/beta_onion_info.env" ]; then
        source /mnt/shared/beta_onion_info.env
        echo -e "   ${GREEN}✅ Beta Environment Ready${NC}"
        echo -e "   🧅 Onion: ${BETA_ONION_ADDRESS:-"Not set"}"
        echo -e "   🚪 Port: ${BETA_PORT:-"Not set"}"
        echo -e "   🆔 Node ID: ${BETA_NODE_ID:-"Not set"}"
        
        # Check if Beta node is running
        if pgrep -f "q-narwhal-validator.*beta" > /dev/null; then
            echo -e "   ${GREEN}✅ Beta Node Process Running${NC}"
            
            # Try to query Beta node
            if curl -s --max-time 5 http://localhost:8334/health > /dev/null 2>&1; then
                BETA_PEERS=$(curl -s --max-time 3 http://localhost:8334/peers/count 2>/dev/null || echo "0")
                BETA_COST=$(curl -s --max-time 3 http://localhost:8334/discovery/cost 2>/dev/null || echo "0.00")
                echo -e "   ${GREEN}✅ Beta API Responding${NC}"
                echo -e "   👥 Peers Discovered: $BETA_PEERS"
                echo -e "   💰 Daily Cost: \$${BETA_COST} ${GREEN}(FREE!)${NC}"
                
                # Check if Beta found Alpha
                if [ -n "${ALPHA_ONION_ADDRESS:-}" ]; then
                    if curl -s --max-time 3 http://localhost:8334/peers/list 2>/dev/null | grep -q "$ALPHA_ONION_ADDRESS"; then
                        echo -e "   ${GREEN}🎯 ✅ FOUND ALPHA! Cross-discovery SUCCESS!${NC}"
                    else
                        echo -e "   ${YELLOW}🎯 ⏳ Searching for Alpha...${NC}"
                    fi
                fi
            else
                echo -e "   ${YELLOW}⚠️  Beta API Not Responding${NC}"
            fi
        else
            echo -e "   ${RED}❌ Beta Node Process Not Running${NC}"
        fi
    else
        echo -e "   ${YELLOW}⚠️  Beta Environment Not Ready${NC}"
        echo -e "   ℹ️  Beta will start after Alpha is ready"
    fi
    
    echo ""
    
    # Cross-Server Discovery Status
    echo -e "${PURPLE}🔄 CROSS-SERVER DISCOVERY STATUS:${NC}"
    
    if [ -f "/mnt/shared/alpha_onion_info.env" ] && [ -f "/mnt/shared/beta_onion_info.env" ]; then
        source /mnt/shared/alpha_onion_info.env
        source /mnt/shared/beta_onion_info.env
        
        # Check mutual discovery
        ALPHA_FOUND_BETA=false
        BETA_FOUND_ALPHA=false
        
        if [ -n "${BETA_ONION_ADDRESS:-}" ] && curl -s --max-time 3 http://localhost:8333/peers/list 2>/dev/null | grep -q "$BETA_ONION_ADDRESS"; then
            ALPHA_FOUND_BETA=true
        fi
        
        if [ -n "${ALPHA_ONION_ADDRESS:-}" ] && curl -s --max-time 3 http://localhost:8334/peers/list 2>/dev/null | grep -q "$ALPHA_ONION_ADDRESS"; then
            BETA_FOUND_ALPHA=true
        fi
        
        if [ "$ALPHA_FOUND_BETA" = true ] && [ "$BETA_FOUND_ALPHA" = true ]; then
            echo -e "   ${GREEN}🏆 BATTLE TEST SUCCESS! Mutual discovery achieved!${NC}"
            echo -e "   ${GREEN}✅ Alpha ↔ Beta connection established via FREE methods${NC}"
        elif [ "$BETA_FOUND_ALPHA" = true ]; then
            echo -e "   ${GREEN}✅ Beta discovered Alpha${NC}"
            echo -e "   ${YELLOW}⏳ Waiting for Alpha to discover Beta${NC}"
        elif [ "$ALPHA_FOUND_BETA" = true ]; then
            echo -e "   ${GREEN}✅ Alpha discovered Beta${NC}"
            echo -e "   ${YELLOW}⏳ Waiting for Beta to discover Alpha${NC}"
        else
            echo -e "   ${YELLOW}⏳ Cross-discovery in progress...${NC}"
        fi
    else
        echo -e "   ${YELLOW}⏳ Waiting for both servers to be ready${NC}"
    fi
    
    echo ""
    
    # System Health
    echo -e "${PURPLE}🛡️  SYSTEM HEALTH:${NC}"
    
    # Tor status
    if pgrep -x "tor" > /dev/null; then
        echo -e "   ${GREEN}✅ Tor daemon running${NC}"
        
        # Test Tor connectivity
        if timeout 5 curl -s --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip > /dev/null; then
            echo -e "   ${GREEN}✅ Tor connectivity working${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Tor connectivity issues${NC}"
        fi
    else
        echo -e "   ${RED}❌ Tor daemon not running${NC}"
    fi
    
    # Cost verification
    TOTAL_COST=0.00
    if [ -n "${ALPHA_COST:-}" ] && [ "${ALPHA_COST}" != "0.00" ]; then
        TOTAL_COST=$(echo "$TOTAL_COST + $ALPHA_COST" | bc -l 2>/dev/null || echo "$TOTAL_COST")
    fi
    if [ -n "${BETA_COST:-}" ] && [ "${BETA_COST}" != "0.00" ]; then
        TOTAL_COST=$(echo "$TOTAL_COST + $BETA_COST" | bc -l 2>/dev/null || echo "$TOTAL_COST")
    fi
    
    if [ "$TOTAL_COST" = "0.00" ] || [ -z "$TOTAL_COST" ]; then
        echo -e "   ${GREEN}💰 ✅ Perfect! $0.00 total daily cost (FREE!)${NC}"
    else
        echo -e "   ${RED}💰 ⚠️  Non-zero cost detected: \$$TOTAL_COST${NC}"
    fi
    
    echo ""
    
    # Next Steps
    echo -e "${PURPLE}📋 NEXT STEPS:${NC}"
    
    if [ ! -f "/mnt/shared/alpha_onion_info.env" ]; then
        echo -e "   ${YELLOW}1. Start Server Alpha: ./scripts/battle_test_alpha.sh${NC}"
    elif [ ! -f "/mnt/shared/beta_onion_info.env" ]; then
        echo -e "   ${YELLOW}1. Start Server Beta: ./scripts/battle_test_beta.sh${NC}"
    elif [ "$BETA_FOUND_ALPHA" != true ]; then
        echo -e "   ${YELLOW}1. Wait for Beta to discover Alpha (should happen within 5 minutes)${NC}"
    elif [ "$ALPHA_FOUND_BETA" != true ]; then
        echo -e "   ${YELLOW}1. Wait for Alpha to discover Beta via gossip protocol${NC}"
    else
        echo -e "   ${GREEN}1. Battle test COMPLETE! Generate reports${NC}"
        echo -e "   ${GREEN}2. Run: cargo run --example battle_test_report -- --combined --output results.json${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}🔄 Run './scripts/battle_test_status.sh' again to refresh${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}"
}

# Main execution
if [ "${1:-}" = "watch" ]; then
    # Continuous monitoring mode
    while true; do
        clear
        check_status
        echo -e "\n${BLUE}⏳ Refreshing in 10 seconds... (Press Ctrl+C to stop)${NC}"
        sleep 10
    done
else
    # Single check
    check_status
fi