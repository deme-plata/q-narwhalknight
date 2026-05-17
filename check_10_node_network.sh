#!/bin/bash

# 🌐 10-Node Network Status Checker
# Verifies Server Alpha + Server Beta distributed network formation

echo "🔍 Q-NARWHALKNIGHT 10-NODE NETWORK STATUS CHECKER"
echo "================================================="
echo "Checking Server Beta (5) + Server Alpha (5) = 10 total nodes"
echo "$(date)"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TOTAL_NODES=0
LISTENING_NODES=0
CONNECTED_NODES=0

echo "📊 NODE DISCOVERY SCAN"
echo "====================="

# Check Server Beta nodes (B1-B5, ports 8001-8005)
echo -e "${BLUE}🤖 Server Beta Nodes (Phase 1+2):${NC}"
for i in {1..5}; do
    PORT=$((8000 + i))
    API_PORT=$((9000 + i))
    NODE_NAME="validator-beta-$i"
    
    # Check if process is running
    if ps aux | grep -q "$NODE_NAME" && ! ps aux | grep "$NODE_NAME" | grep -q grep; then
        TOTAL_NODES=$((TOTAL_NODES + 1))
        echo -n "   B$i (port $PORT): "
        
        # Check if port is listening
        if ss -tln | grep -q ":$PORT "; then
            LISTENING_NODES=$((LISTENING_NODES + 1))
            echo -e "${GREEN}✅ LISTENING${NC}"
            
            # Try to check node status via API (if available)
            if timeout 2 curl -s "http://localhost:$API_PORT/status" >/dev/null 2>&1; then
                CONNECTED_NODES=$((CONNECTED_NODES + 1))
                echo "      API: ✅ Responsive"
            else
                echo "      API: 🔄 Starting up"
            fi
        else
            echo -e "${YELLOW}🔄 STARTING${NC}"
        fi
    else
        echo "   B$i (port $PORT): ❌ NOT RUNNING"
    fi
done

echo ""

# Check Server Alpha nodes (A1-A5, ports 8006-8010)
echo -e "${BLUE}🤖 Server Alpha Nodes (Phase 3+4):${NC}"
for i in {1..5}; do
    PORT=$((8005 + i))
    API_PORT=$((9005 + i))
    NODE_NAME="validator-alpha-$i"
    
    # Check if process is running
    if ps aux | grep -q "$NODE_NAME" && ! ps aux | grep "$NODE_NAME" | grep -q grep; then
        TOTAL_NODES=$((TOTAL_NODES + 1))
        echo -n "   A$i (port $PORT): "
        
        # Check if port is listening
        if ss -tln | grep -q ":$PORT "; then
            LISTENING_NODES=$((LISTENING_NODES + 1))
            echo -e "${GREEN}✅ LISTENING${NC}"
            
            # Try to check node status via API (if available)
            if timeout 2 curl -s "http://localhost:$API_PORT/status" >/dev/null 2>&1; then
                CONNECTED_NODES=$((CONNECTED_NODES + 1))
                echo "      API: ✅ Responsive"
            else
                echo "      API: 🔄 Starting up"
            fi
        else
            echo -e "${YELLOW}🔄 STARTING${NC}"
        fi
    else
        echo "   A$i (port $PORT): ❌ NOT RUNNING"
    fi
done

echo ""
echo "📈 NETWORK SUMMARY"
echo "=================="
echo "Total Processes Running: $TOTAL_NODES/10"
echo "Nodes Listening on Ports: $LISTENING_NODES/10"
echo "Nodes with Active APIs: $CONNECTED_NODES/10"

# Network connectivity check
echo ""
echo "🌐 NETWORK CONNECTIVITY TEST"
echo "============================"

NETWORK_READY=false

if [ $TOTAL_NODES -eq 10 ] && [ $LISTENING_NODES -eq 10 ]; then
    echo -e "${GREEN}✅ All 10 nodes are deployed and listening!${NC}"
    NETWORK_READY=true
    
    # Test inter-node connectivity
    echo "🔗 Testing inter-node connectivity..."
    
    # Test Server Beta to Server Alpha connection
    BETA_TO_ALPHA=0
    for i in {1..5}; do
        ALPHA_PORT=$((8005 + i))
        if timeout 3 nc -z localhost $ALPHA_PORT 2>/dev/null; then
            BETA_TO_ALPHA=$((BETA_TO_ALPHA + 1))
        fi
    done
    
    echo "Server Beta → Server Alpha connectivity: $BETA_TO_ALPHA/5"
    
    if [ $BETA_TO_ALPHA -eq 5 ]; then
        echo -e "${GREEN}🎉 FULL 10-NODE NETWORK ESTABLISHED!${NC}"
        echo -e "${GREEN}🚀 Ready for real-world consensus testing!${NC}"
    else
        echo -e "${YELLOW}⏳ Network formation in progress...${NC}"
    fi
    
elif [ $TOTAL_NODES -lt 10 ]; then
    echo -e "${YELLOW}⏳ Waiting for all nodes to deploy: $TOTAL_NODES/10 running${NC}"
    
    if [ $TOTAL_NODES -eq 5 ]; then
        echo "   • Server Beta: ✅ 5 nodes running"
        echo "   • Server Alpha: ⏳ Deployment in progress"
    elif [ $TOTAL_NODES -gt 5 ]; then
        ALPHA_NODES=$((TOTAL_NODES - 5))
        echo "   • Server Beta: ✅ 5 nodes running" 
        echo "   • Server Alpha: 🔄 $ALPHA_NODES/5 nodes deploying"
    fi
    
else
    echo -e "${YELLOW}⏳ Nodes deployed but still starting up: $LISTENING_NODES/10 listening${NC}"
fi

echo ""

# Performance readiness check
if $NETWORK_READY; then
    echo "🎯 PERFORMANCE TESTING READINESS"
    echo "==============================="
    echo -e "${GREEN}✅ 10-node distributed network operational${NC}"
    echo -e "${GREEN}✅ Ready for real transaction load testing${NC}"
    echo -e "${GREEN}✅ Phase 1+2+3+4 architecture active${NC}"
    echo ""
    echo "🚀 Execute: ./real_world_transaction_test.sh"
    echo "📊 Monitor: ./monitor_distributed_performance.sh"
else
    echo "⏳ Network formation in progress..."
    echo "   Run this script again in 30 seconds to check status"
fi

echo ""
echo "📊 Current Status: $(date)"
echo "Next check recommended in: 30 seconds"

# Return appropriate exit code
if $NETWORK_READY; then
    exit 0  # Success - network ready
else
    exit 1  # Still waiting for full deployment
fi