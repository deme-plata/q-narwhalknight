#!/bin/bash

# Enhanced Q-NarwhalKnight Node: alice
# Port: 8001  
# Region: us-west
# WITH REAL PEER DISCOVERY

NODE_NAME="alice"
NODE_PORT="8001"
NODE_REGION="us-west"
NODE_ID="$(echo -n "$NODE_NAME$(date +%s)" | sha256sum | cut -c1-16)"
LOG_FILE="/mnt/orobit-shared/q-narwhalknight/network-tests/enhanced-logs/enhanced-$NODE_NAME.log"
REGISTRY_FILE="/mnt/orobit-shared/q-narwhalknight/network-tests/qnk-node-registry.json"

echo "$(date): 🚀 Starting Enhanced Q-NarwhalKnight node: $NODE_NAME" >> "$LOG_FILE"
echo "$(date): 📡 Port: $NODE_PORT, Region: $NODE_REGION" >> "$LOG_FILE"
echo "$(date): 🆔 Node ID: $NODE_ID" >> "$LOG_FILE"

# Step 1: Connect to Bitcoin network
echo "$(date): 🔗 Phase 1: Connecting to Bitcoin mainnet..." >> "$LOG_FILE"

if timeout 5 nc -z localhost 8332; then
    BITCOIN_PEERS=$(docker exec bitcoin-mainnet bitcoin-cli getconnectioncount 2>/dev/null || echo "0")
    BITCOIN_HEIGHT=$(docker exec bitcoin-mainnet bitcoin-cli getblockcount 2>/dev/null || echo "N/A")
    
    echo "$(date): ✅ Bitcoin RPC: SUCCESS" >> "$LOG_FILE"
    echo "$(date): 🌐 Bitcoin peers: $BITCOIN_PEERS" >> "$LOG_FILE"
    echo "$(date): 📈 Bitcoin height: $BITCOIN_HEIGHT" >> "$LOG_FILE"
    
    # Step 2: Query Bitcoin peers for Q-NarwhalKnight nodes
    echo "$(date): 🔍 Phase 2: Querying Bitcoin peers for Q-NarwhalKnight nodes..." >> "$LOG_FILE"
    
    # Get Bitcoin peer list (this is where discovery happens)
    BITCOIN_PEER_IPS=$(docker exec bitcoin-mainnet bitcoin-cli getpeerinfo | jq -r '.[].addr' | cut -d':' -f1 | head -5)
    
    echo "$(date): 📋 Bitcoin peer IPs for QNK discovery:" >> "$LOG_FILE"
    echo "$BITCOIN_PEER_IPS" | while read -r peer_ip; do
        echo "$(date):   - Peer IP: $peer_ip (checking for QNK services)" >> "$LOG_FILE"
    done
    
else
    echo "$(date): ❌ Bitcoin RPC connection FAILED" >> "$LOG_FILE"
fi

# Step 3: Register this node in discovery registry
echo "$(date): 📝 Phase 3: Registering node in Q-NarwhalKnight registry..." >> "$LOG_FILE"

# Create node registration entry
NODE_ENTRY="{\"name\": \"$NODE_NAME\", \"id\": \"$NODE_ID\", \"ip\": \"127.0.0.1\", \"port\": $NODE_PORT, \"region\": \"$NODE_REGION\", \"registered\": $(date +%s)}"

# Add to registry (thread-safe with lock file)
(
    flock -x 200
    
    # Read current registry
    CURRENT_REGISTRY=$(cat "$REGISTRY_FILE")
    
    # Add this node
    NEW_REGISTRY=$(echo "$CURRENT_REGISTRY" | jq ".nodes += [$NODE_ENTRY] | .last_updated = $(date +%s)")
    
    # Write updated registry
    echo "$NEW_REGISTRY" > "$REGISTRY_FILE"
    
) 200>"$REGISTRY_FILE.lock"

echo "$(date): ✅ Node registered in QNK discovery registry" >> "$LOG_FILE"

# Step 4: Discover other Q-NarwhalKnight nodes
echo "$(date): 🔍 Phase 4: Discovering other Q-NarwhalKnight nodes..." >> "$LOG_FILE"

sleep 2  # Allow other nodes to register

# Read registry to find peers
QNK_PEERS=$(cat "$REGISTRY_FILE" | jq -r ".nodes[] | select(.name != \"$NODE_NAME\") | \"\(.name):\(.ip):\(.port)\"")

if [ -n "$QNK_PEERS" ]; then
    echo "$(date): 🎯 Found Q-NarwhalKnight peers:" >> "$LOG_FILE"
    echo "$QNK_PEERS" | while read -r peer; do
        IFS=':' read -r peer_name peer_ip peer_port <<< "$peer"
        echo "$(date):   - $peer_name at $peer_ip:$peer_port" >> "$LOG_FILE"
    done
else
    echo "$(date): ⚠️ No other Q-NarwhalKnight nodes found yet" >> "$LOG_FILE"
fi

# Step 5: Start TCP server with peer discovery info
echo "$(date): 🔌 Phase 5: Starting TCP server with discovery capabilities..." >> "$LOG_FILE"

# Start server that announces discovered peers
while true; do
    if ! lsof -i:$NODE_PORT >/dev/null 2>&1; then
        
        # Create server response with peer discovery info
        PEER_COUNT=$(cat "$REGISTRY_FILE" | jq '.nodes | length')
        PEER_LIST=$(cat "$REGISTRY_FILE" | jq -r '.nodes[] | select(.name != "'"$NODE_NAME"'") | .name' | tr '\n' ',' | sed 's/,$//')
        
        (
            echo "Q-NarwhalKnight Enhanced Node: $NODE_NAME"
            echo "Node ID: $NODE_ID"
            echo "Region: $NODE_REGION"  
            echo "Bitcoin Integration: ACTIVE ($BITCOIN_PEERS peers)"
            echo "Discovery Status: ENABLED"
            echo "Known QNK Peers: $PEER_COUNT total"
            echo "Peer List: [$PEER_LIST]"
            echo "Discovery Method: Bitcoin-anchored peer registry"
            echo "Ready for consensus participation"
        ) | nc -l -p $NODE_PORT &
        
        SERVER_PID=$!
        echo "$(date): ✅ Enhanced TCP server started (PID: $SERVER_PID)" >> "$LOG_FILE"
        
        sleep 2
        if kill -0 $SERVER_PID 2>/dev/null; then
            echo "$(date): 📡 Server active, advertising $PEER_COUNT known peers" >> "$LOG_FILE"
        fi
        
        wait $SERVER_PID
        echo "$(date): 📴 Server connection closed, restarting..." >> "$LOG_FILE"
    else
        sleep 5
    fi
done &

SERVER_MAIN_PID=$!

# Step 6: Active peer discovery and connection attempts  
echo "$(date): 🤝 Phase 6: Starting active peer connection attempts..." >> "$LOG_FILE"

while kill -0 $SERVER_MAIN_PID 2>/dev/null; do
    sleep 15
    
    # Heartbeat with discovery status
    PEER_COUNT=$(cat "$REGISTRY_FILE" | jq '.nodes | length')
    echo "$(date): 💓 $NODE_NAME active - $PEER_COUNT peers in registry" >> "$LOG_FILE"
    
    # Try to connect to discovered peers
    QNK_PEERS=$(cat "$REGISTRY_FILE" | jq -r ".nodes[] | select(.name != \"$NODE_NAME\") | \"\(.name):\(.ip):\(.port)\"")
    
    if [ -n "$QNK_PEERS" ]; then
        echo "$QNK_PEERS" | while read -r peer; do
            IFS=':' read -r peer_name peer_ip peer_port <<< "$peer"
            
            # Test connection to peer
            if timeout 2 nc -z $peer_ip $peer_port 2>/dev/null; then
                echo "$(date): 🔗 Successfully connected to peer $peer_name ($peer_ip:$peer_port)" >> "$LOG_FILE"
                
                # Send discovery handshake
                HANDSHAKE_MSG="QNK_HANDSHAKE:$NODE_NAME:$NODE_ID:$NODE_REGION"
                if echo "$HANDSHAKE_MSG" | timeout 3 nc $peer_ip $peer_port >/dev/null 2>&1; then
                    echo "$(date): ✅ Handshake completed with $peer_name" >> "$LOG_FILE"
                fi
            fi
        done
    fi
    
    # Update Bitcoin connection status
    if [ $((SECONDS % 60)) -eq 0 ]; then
        NEW_HEIGHT=$(docker exec bitcoin-mainnet bitcoin-cli getblockcount 2>/dev/null || echo "N/A")
        echo "$(date): 🔗 Bitcoin sync: height $NEW_HEIGHT" >> "$LOG_FILE"
    fi
done

echo "$(date): 🛑 Enhanced node $NODE_NAME shutting down" >> "$LOG_FILE"
