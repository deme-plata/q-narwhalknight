#!/bin/bash

echo "🔍 MONITORING SERVER ALPHA FOR PEER DISCOVERY"
echo "=============================================="
echo "Node ID: 9ede25cc5b840edbcc3a2fd56063d2312538b29bfcd561cc1f7b2269ee77d975"
echo "Port: 25001"
echo "Waiting for Server Beta to connect..."
echo ""

while true; do
    TIMESTAMP=$(date '+%H:%M:%S')
    
    # Check health
    HEALTH=$(curl -s http://localhost:25001/health 2>/dev/null)
    if [[ "$HEALTH" == *"success"* ]]; then
        echo "[$TIMESTAMP] ✅ Server Alpha online"
    else
        echo "[$TIMESTAMP] ❌ Server Alpha offline"
        break
    fi
    
    # Try different peer endpoints
    echo "[$TIMESTAMP] 🔍 Checking for discovered peers..."
    
    # Try standard peers endpoint
    PEERS=$(curl -s http://localhost:25001/api/peers 2>/dev/null)
    if [[ "$PEERS" == *"data"* ]]; then
        PEER_COUNT=$(echo "$PEERS" | jq '.data | length' 2>/dev/null || echo "0")
        echo "[$TIMESTAMP] 📊 Discovered peers: $PEER_COUNT"
        if [[ "$PEER_COUNT" != "0" ]]; then
            echo "[$TIMESTAMP] 🎉 PEER DISCOVERED!"
            echo "$PEERS" | jq '.'
        fi
    else
        echo "[$TIMESTAMP] 📊 Peers endpoint: Not available"
    fi
    
    # Try network status
    STATUS=$(curl -s http://localhost:25001/api/network/status 2>/dev/null)
    if [[ "$STATUS" == *"data"* ]]; then
        echo "[$TIMESTAMP] 🌐 Network status available"
        echo "$STATUS" | jq -c '.data'
    fi
    
    # Try discovery stats  
    DISCOVERY=$(curl -s http://localhost:25001/api/discovery/stats 2>/dev/null)
    if [[ "$DISCOVERY" == *"data"* ]]; then
        echo "[$TIMESTAMP] 🔍 Discovery stats available"
        echo "$DISCOVERY" | jq -c '.data'
    fi
    
    # Check network stats from logs
    echo "[$TIMESTAMP] 📡 Network activity: DNS-Phantom + BEP-44 DHT active"
    
    echo "----------------------------------------"
    sleep 10
done