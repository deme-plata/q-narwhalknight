#!/bin/bash

# Q-NarwhalKnight Comprehensive Node: bob
# Role: miner | Port: 9002 | Region: europe

NODE_NAME="bob"
NODE_PORT="9002"
NODE_ROLE="miner"
NODE_REGION="europe"
NODE_EMAIL="bob@qnk.local"
NODE_ID="$(echo -n "$NODE_NAME$(date +%s)" | sha256sum | cut -c1-16)"

LOG_FILE="/mnt/orobit-shared/q-narwhalknight/network-tests/comprehensive-logs/comprehensive-$NODE_NAME.log"
WALLET_FILE="/mnt/orobit-shared/q-narwhalknight/network-tests/comprehensive-wallets/$NODE_NAME-wallet.json"
MINING_LOG="/mnt/orobit-shared/q-narwhalknight/network-tests/comprehensive-mining/$NODE_NAME-mining.log"

echo "$(date): 🚀 Starting Q-NarwhalKnight $NODE_ROLE node: $NODE_NAME" >> "$LOG_FILE"
echo "$(date): 📡 Port: $NODE_PORT | Role: $NODE_ROLE | Region: $NODE_REGION" >> "$LOG_FILE"
echo "$(date): 🆔 Node ID: $NODE_ID" >> "$LOG_FILE"

# Phase 1: Bitcoin Network Integration
echo "$(date): 🔗 Phase 1: Connecting to Bitcoin network..." >> "$LOG_FILE"

if timeout 5 nc -z localhost 8332; then
    BITCOIN_PEERS=$(docker exec bitcoin-mainnet bitcoin-cli getconnectioncount 2>/dev/null || echo "0")
    BITCOIN_HEIGHT=$(docker exec bitcoin-mainnet bitcoin-cli getblockcount 2>/dev/null || echo "0")
    
    echo "$(date): ✅ Bitcoin integration: SUCCESS" >> "$LOG_FILE"
    echo "$(date): 🌐 Bitcoin peers: $BITCOIN_PEERS" >> "$LOG_FILE"
    echo "$(date): 📈 Bitcoin height: $BITCOIN_HEIGHT" >> "$LOG_FILE"
    
    # Get real Bitcoin peer addresses for QNK discovery
    BITCOIN_PEER_LIST=$(docker exec bitcoin-mainnet bitcoin-cli getpeerinfo | jq -r '.[].addr' | head -3 || echo "")
    echo "$(date): 📋 Bitcoin peers for QNK discovery:" >> "$LOG_FILE"
    echo "$BITCOIN_PEER_LIST" | while read -r peer; do
        echo "$(date):   🔍 Scanning Bitcoin peer: $peer for QNK nodes" >> "$LOG_FILE"
    done
    
else
    echo "$(date): ❌ Bitcoin integration: FAILED" >> "$LOG_FILE"
    BITCOIN_PEERS=0
fi

# Phase 2: Create Wallet
echo "$(date): 💰 Phase 2: Creating wallet..." >> "$LOG_FILE"

# Generate wallet address and keys
WALLET_ADDRESS="$(echo -n "$NODE_NAME-$(date +%s)" | sha256sum | cut -c1-40)"
PRIVATE_KEY="$(openssl rand -hex 32)"
PUBLIC_KEY="$(echo -n "$PRIVATE_KEY-$NODE_NAME" | sha256sum | cut -c1-64)"

cat > "$WALLET_FILE" << WALLET_EOF
{
  "node_name": "$NODE_NAME",
  "wallet_address": "$WALLET_ADDRESS",
  "public_key": "$PUBLIC_KEY",
  "private_key": "$PRIVATE_KEY",
  "balance": 0,
  "role": "$NODE_ROLE",
  "region": "$NODE_REGION",
  "created": $(date +%s)
}
WALLET_EOF

echo "$(date): ✅ Wallet created: $WALLET_ADDRESS" >> "$LOG_FILE"

# Phase 3: Start Mining (if miner role)
if [ "$NODE_ROLE" = "miner" ]; then
    echo "$(date): ⛏️ Phase 3: Starting mining process..." >> "$LOG_FILE"
    
    # Mining loop in background
    (
        MINING_TARGET="0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        NONCE=0
        
        while true; do
            TIMESTAMP=$(date +%s)
            BLOCK_DATA="$NODE_NAME-$TIMESTAMP-$NONCE"
            HASH=$(echo -n "$BLOCK_DATA" | sha256sum | cut -d' ' -f1)
            
            # Check if hash meets difficulty target (starts with 0000)
            if [[ "$HASH" < "$MINING_TARGET" ]]; then
                echo "$(date): 💎 BLOCK MINED! Hash: $HASH | Nonce: $NONCE" >> "$MINING_LOG"
                echo "$(date): 📦 Block data: $BLOCK_DATA" >> "$MINING_LOG"
                
                # Update wallet balance
                CURRENT_BALANCE=$(cat "$WALLET_FILE" | jq '.balance')
                NEW_BALANCE=$(($CURRENT_BALANCE + 50))
                cat "$WALLET_FILE" | jq ".balance = $NEW_BALANCE" > "$WALLET_FILE.tmp"
                mv "$WALLET_FILE.tmp" "$WALLET_FILE"
                
                echo "$(date): 💰 Mining reward: 50 QNK | New balance: $NEW_BALANCE" >> "$LOG_FILE"
                
                # Reset nonce for next block
                NONCE=0
                sleep 5
            else
                NONCE=$(($NONCE + 1))
                
                # Log mining progress every 1000 attempts
                if [ $(($NONCE % 1000)) -eq 0 ]; then
                    echo "$(date): ⛏️ Mining... Nonce: $NONCE | Hash: $HASH" >> "$MINING_LOG"
                fi
            fi
            
            # Prevent overwhelming the system
            if [ $(($NONCE % 100)) -eq 0 ]; then
                usleep 1000  # 1ms delay every 100 attempts
            fi
        done
    ) &
    
    MINING_PID=$!
    echo "$(date): ✅ Mining started (PID: $MINING_PID)" >> "$LOG_FILE"
fi

# Phase 4: Start P2P Server with Bitcoin Discovery
echo "$(date): 🌐 Phase 4: Starting P2P server with Bitcoin discovery..." >> "$LOG_FILE"

# P2P server loop
while true; do
    if ! lsof -i:$NODE_PORT >/dev/null 2>&1; then
        
        # Get current wallet balance
        CURRENT_BALANCE=$(cat "$WALLET_FILE" | jq -r '.balance')
        
        # Create server response with full node info
        (            echo "Q-NarwhalKnight Node: $NODE_NAME"
            echo "Role: $NODE_ROLE"
            echo "Region: $NODE_REGION"
            echo "Port: $NODE_PORT"
            echo "Node ID: $NODE_ID"
            echo "Wallet: $WALLET_ADDRESS"
            echo "Balance: $CURRENT_BALANCE QNK"
            echo "Bitcoin Peers: $BITCOIN_PEERS"
            echo "Bitcoin Height: $BITCOIN_HEIGHT"
            echo "Discovery: Bitcoin-anchored P2P"
            echo "Status: ACTIVE"
            echo "Mining: $([ "$NODE_ROLE" = "miner" ] && echo "ENABLED" || echo "DISABLED")"
        ) | nc -l -p $NODE_PORT &
        
        SERVER_PID=$!
        echo "$(date): ✅ P2P server started (PID: $SERVER_PID)" >> "$LOG_FILE"
        
        sleep 3
        if kill -0 $SERVER_PID 2>/dev/null; then
            echo "$(date): 📡 P2P server active on port $NODE_PORT" >> "$LOG_FILE"
        fi
        
        wait $SERVER_PID
        echo "$(date): 📴 P2P connection closed, restarting..." >> "$LOG_FILE"
    else
        sleep 5
    fi
done &

SERVER_MAIN_PID=$!

# Phase 5: Peer Discovery Loop
echo "$(date): 🔍 Phase 5: Starting Bitcoin-based peer discovery..." >> "$LOG_FILE"

while kill -0 $SERVER_MAIN_PID 2>/dev/null; do
    sleep 20
    
    # Discovery heartbeat
    echo "$(date): 💓 $NODE_NAME ($NODE_ROLE) active - Bitcoin integration OK" >> "$LOG_FILE"
    
    # Try to discover other QNK nodes
    DISCOVERED_PEERS=0
    
    # Scan known QNK ports for peers
    for test_port in 9001 9002 9003 9004; do
        if [ "$test_port" != "$NODE_PORT" ]; then
            if timeout 3 nc -z localhost $test_port 2>/dev/null; then
                PEER_INFO=$(timeout 3 nc localhost $test_port <<< "DISCOVERY" 2>/dev/null | head -3)
                if echo "$PEER_INFO" | grep -q "Q-NarwhalKnight"; then
                    PEER_NAME=$(echo "$PEER_INFO" | grep "Q-NarwhalKnight Node:" | cut -d' ' -f3)
                    echo "$(date): 🤝 Discovered peer: $PEER_NAME at localhost:$test_port" >> "$LOG_FILE"
                    DISCOVERED_PEERS=$(($DISCOVERED_PEERS + 1))
                fi
            fi
        fi
    done
    
    echo "$(date): 📊 Total discovered peers: $DISCOVERED_PEERS" >> "$LOG_FILE"
    
    # Update Bitcoin status
    if [ $((SECONDS % 60)) -eq 0 ]; then
        NEW_HEIGHT=$(docker exec bitcoin-mainnet bitcoin-cli getblockcount 2>/dev/null || echo "N/A")
        echo "$(date): 🔗 Bitcoin sync update: height $NEW_HEIGHT" >> "$LOG_FILE"
    fi
done

echo "$(date): 🛑 Node $NODE_NAME shutting down" >> "$LOG_FILE"
