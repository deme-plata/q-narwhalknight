#!/bin/bash

# 🔄 Continuous 10-Node Network Formation Monitor
echo "🚀 CONTINUOUS Q-NARWHALKNIGHT 10-NODE NETWORK MONITOR"
echo "====================================================="
echo "Monitoring Server Beta + Server Alpha deployment progress..."
echo "$(date)"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PREVIOUS_TOTAL=0
NETWORK_READY=false

while true; do
    clear
    echo -e "${BLUE}🔄 Q-NARWHALKNIGHT NETWORK FORMATION MONITOR${NC}"
    echo "=============================================="
    echo "$(date)"
    echo ""
    
    TOTAL_NODES=0
    LISTENING_NODES=0
    COMPILATION_PROCESSES=0
    
    # Count compilation processes
    CARGO_PROCESSES=$(ps aux | grep "cargo run --bin dagknight" | grep -v grep | wc -l)
    
    echo "📊 DEPLOYMENT STATUS:"
    echo "===================="
    echo "Active cargo compilation processes: $CARGO_PROCESSES"
    
    # Check Server Beta nodes
    echo ""
    echo -e "${BLUE}🤖 Server Beta Nodes (Phase 1+2):${NC}"
    for i in {1..5}; do
        PORT=$((8000 + i))
        NODE_NAME="validator-beta-$i"
        
        if ps aux | grep -q "$NODE_NAME" && ! ps aux | grep "$NODE_NAME" | grep -q grep; then
            TOTAL_NODES=$((TOTAL_NODES + 1))
            echo -n "   B$i: "
            
            if ss -tln | grep -q ":$PORT "; then
                LISTENING_NODES=$((LISTENING_NODES + 1))
                echo -e "${GREEN}✅ OPERATIONAL${NC}"
            else
                echo -e "${YELLOW}🔄 COMPILING${NC}"
            fi
        else
            echo "   B$i: ❌ NOT STARTED"
        fi
    done
    
    # Check Server Alpha nodes
    echo ""
    echo -e "${BLUE}🤖 Server Alpha Nodes (Phase 3+4):${NC}"
    for i in {1..5}; do
        PORT=$((8005 + i))
        NODE_NAME="validator-alpha-$i"
        
        if ps aux | grep -q "$NODE_NAME" && ! ps aux | grep "$NODE_NAME" | grep -q grep; then
            TOTAL_NODES=$((TOTAL_NODES + 1))
            echo -n "   A$i: "
            
            if ss -tln | grep -q ":$PORT "; then
                LISTENING_NODES=$((LISTENING_NODES + 1))
                echo -e "${GREEN}✅ OPERATIONAL${NC}"
            else
                echo -e "${YELLOW}🔄 COMPILING${NC}"
            fi
        else
            echo "   A$i: ❌ NOT STARTED"
        fi
    done
    
    echo ""
    echo "📈 NETWORK SUMMARY:"
    echo "==================="
    echo "Total Processes: $TOTAL_NODES/10"
    echo "Listening Nodes: $LISTENING_NODES/10"
    echo "Cargo Processes: $CARGO_PROCESSES"
    
    # Progress indicator
    if [ $TOTAL_NODES -gt $PREVIOUS_TOTAL ]; then
        echo -e "${GREEN}📈 PROGRESS: Network formation advancing!${NC}"
    fi
    
    # Check if network is ready
    if [ $TOTAL_NODES -eq 10 ] && [ $LISTENING_NODES -eq 10 ]; then
        echo ""
        echo -e "${GREEN}🎉 NETWORK FORMATION COMPLETE!${NC}"
        echo -e "${GREEN}✅ All 10 nodes operational and listening${NC}"
        echo -e "${GREEN}🚀 Ready for real-world transaction testing!${NC}"
        NETWORK_READY=true
        break
    elif [ $TOTAL_NODES -eq 10 ]; then
        echo -e "${YELLOW}⏳ All nodes deployed, waiting for compilation to complete...${NC}"
        PROGRESS_PCT=$(( LISTENING_NODES * 100 / 10 ))
        echo "Compilation progress: $PROGRESS_PCT% ($LISTENING_NODES/10 ready)"
    elif [ $TOTAL_NODES -gt 0 ]; then
        echo -e "${YELLOW}🔄 Partial deployment: $TOTAL_NODES/10 nodes started${NC}"
    else
        echo -e "${YELLOW}⏳ Waiting for node deployment to begin...${NC}"
    fi
    
    PREVIOUS_TOTAL=$TOTAL_NODES
    
    echo ""
    echo "🔄 Next update in 15 seconds... (Ctrl+C to stop)"
    sleep 15
done

if $NETWORK_READY; then
    echo ""
    echo "🚀 EXECUTE REAL-WORLD TESTING:"
    echo "=============================="
    echo "./real_world_transaction_test.sh"
    echo "./monitor_distributed_performance.sh"
fi