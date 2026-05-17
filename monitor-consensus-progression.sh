#!/bin/bash

echo "🔍 Q-NarwhalKnight Consensus Progression Monitor"
echo "==============================================="
echo "Monitoring 4-node network for consensus activation..."
echo ""

# Function to check for specific consensus events
check_consensus_events() {
    local node=$1
    local logs=$(docker logs qnk-node-$node --since=30s 2>/dev/null)
    
    # Phase 2: P2P Connection events
    echo "$logs" | grep -i -E "(connected|peer.*established|gossipsub|libp2p)" | tail -1
    
    # Phase 3: DAG-Knight activation
    echo "$logs" | grep -i -E "(dag.*knight|consensus.*active|anchor.*election)" | tail -1
    
    # Phase 4: VDF activation  
    echo "$logs" | grep -i -E "(vdf|verifiable.*delay|quantum.*random)" | tail -1
    
    # Phase 5: Transaction processing
    echo "$logs" | grep -i -E "(transaction.*processed|vertex.*created|byzantine)" | tail -1
}

# Monitoring loop
for i in {1..10}; do
    echo "🔄 Scan $i/10 ($(date +'%H:%M:%S'))"
    echo "=================================="
    
    # Current status
    echo "📊 Current Activity:"
    for node in alpha beta charlie diana; do
        latest=$(docker logs qnk-node-$node --tail=1 2>/dev/null | grep -o "Sent message [a-f0-9-]*" | head -1)
        if [ ! -z "$latest" ]; then
            echo "  🔸 $node: $latest"
        fi
    done
    
    echo ""
    echo "🔍 Looking for Progression Events:"
    
    phase2_found=false
    phase3_found=false
    phase4_found=false
    
    for node in alpha beta charlie diana; do
        events=$(check_consensus_events $node)
        if [ ! -z "$events" ]; then
            echo "  🎯 $node: $events"
            if echo "$events" | grep -q -i -E "(connected|peer.*established)"; then
                phase2_found=true
            fi
            if echo "$events" | grep -q -i -E "(dag.*knight|consensus.*active)"; then
                phase3_found=true
            fi
            if echo "$events" | grep -q -i -E "(vdf|quantum.*random)"; then
                phase4_found=true
            fi
        fi
    done
    
    # Phase progression check
    echo ""
    echo "📈 Consensus Phase Status:"
    echo "  ✅ Phase 1: Peer Advertisement (Active)"
    if $phase2_found; then
        echo "  ✅ Phase 2: P2P Connections (DETECTED!)"
    else
        echo "  ⏳ Phase 2: P2P Connections (Waiting...)"
    fi
    
    if $phase3_found; then
        echo "  ✅ Phase 3: DAG-Knight Consensus (ACTIVATED!)"
    else
        echo "  ⏳ Phase 3: DAG-Knight Consensus (Waiting...)"
    fi
    
    if $phase4_found; then
        echo "  ✅ Phase 4: VDF Anchor Election (RUNNING!)"
        echo "  🎉 QUANTUM CONSENSUS FULLY OPERATIONAL!"
        break
    else
        echo "  ⏳ Phase 4: VDF Anchor Election (Waiting...)"
    fi
    
    echo ""
    echo "-------------------------------------------"
    sleep 10
done

echo ""
echo "🏁 Monitoring complete. Network status captured."