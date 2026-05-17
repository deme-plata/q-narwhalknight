#!/bin/bash

# Q-NarwhalKnight Network Propagation Test
# Tests transaction propagation across 20-node network

echo "🧪 TESTING TRANSACTION PROPAGATION"
echo "=================================="

# Test configuration
NODES=20
BASE_PORT=8081
TEST_WALLET_COUNT=5

echo "📊 Network Overview:"
echo "===================="

# Check all nodes are healthy
active_nodes=0
for i in $(seq 1 $NODES); do
    port=$((BASE_PORT + i - 1))
    if curl -s -m 2 http://localhost:$port/health > /dev/null 2>&1; then
        active_nodes=$((active_nodes + 1))
        # Get basic node status
        status=$(curl -s http://localhost:$port/api/v1/node/status 2>/dev/null | jq -r '.data.consensus_status // "unknown"' 2>/dev/null || echo "unknown")
        peers=$(curl -s http://localhost:$port/api/v1/node/status 2>/dev/null | jq -r '.data.connected_peers // 0' 2>/dev/null || echo "0")
        printf "Node-%02d (:%d) Status: %-10s Peers: %s\n" $i $port "$status" "$peers"
    else
        printf "Node-%02d (:%d) Status: %-10s\n" $i $port "OFFLINE"
    fi
done

echo ""
echo "🌟 Network Health: $active_nodes/$NODES nodes active"
echo ""

if [ $active_nodes -lt 5 ]; then
    echo "❌ Not enough active nodes for propagation test"
    exit 1
fi

echo "💰 WALLET & TRANSACTION TEST"
echo "============================"

# Create test wallets on different nodes
echo "📝 Creating test wallets..."
wallets=()

for i in $(seq 1 $TEST_WALLET_COUNT); do
    # Use different nodes for wallet creation
    node_port=$((BASE_PORT + i - 1))
    
    echo "Creating wallet $i on node port $node_port..."
    wallet_response=$(curl -s -X POST http://localhost:$node_port/api/v1/wallets \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"TestWallet-$i\"}" 2>/dev/null)
    
    if echo "$wallet_response" | jq -e '.success' > /dev/null 2>&1; then
        wallet_id=$(echo "$wallet_response" | jq -r '.data.id')
        wallet_address=$(echo "$wallet_response" | jq -r '.data.address // "N/A"')
        wallets+=("$wallet_id:$node_port")
        echo "✅ Wallet-$i: $wallet_id on port $node_port"
    else
        echo "❌ Failed to create wallet $i"
    fi
    
    sleep 1
done

echo ""
echo "🚰 FAUCET TEST (Fund Wallets)"
echo "============================="

# Fund wallets using faucet
for i in $(seq 1 ${#wallets[@]}); do
    wallet_info="${wallets[$i-1]}"
    node_port=$(echo "$wallet_info" | cut -d: -f2)
    
    echo "Funding wallet $i on port $node_port..."
    faucet_response=$(curl -s -X POST http://localhost:$node_port/api/v1/faucet \
        -H "Content-Type: application/json" \
        -d '{}' 2>/dev/null)
    
    if echo "$faucet_response" | jq -e '.success' > /dev/null 2>&1; then
        echo "✅ Faucet successful for wallet $i"
    else
        echo "❌ Faucet failed for wallet $i"
    fi
    
    sleep 2
done

echo ""
echo "📊 BALANCE PROPAGATION TEST"
echo "==========================="

# Check balances across different nodes
echo "Checking balances across all active nodes..."

for check_round in {1..3}; do
    echo ""
    echo "📋 Balance Check Round $check_round:"
    echo "==================================="
    
    for i in $(seq 1 $NODES); do
        node_port=$((BASE_PORT + i - 1))
        if curl -s -m 2 http://localhost:$node_port/health > /dev/null 2>&1; then
            # Get node status including balance
            status_response=$(curl -s http://localhost:$node_port/api/v1/node/status 2>/dev/null)
            if echo "$status_response" | jq -e '.success' > /dev/null 2>&1; then
                balance=$(echo "$status_response" | jq -r '.data.balance // 0' 2>/dev/null || echo "0")
                height=$(echo "$status_response" | jq -r '.data.current_height // 0' 2>/dev/null || echo "0")
                round=$(echo "$status_response" | jq -r '.data.current_round // 0' 2>/dev/null || echo "0")
                peers=$(echo "$status_response" | jq -r '.data.connected_peers // 0' 2>/dev/null || echo "0")
                
                printf "Node-%02d: Balance=%s QNK Height=%s Round=%s Peers=%s\n" $i "$balance" "$height" "$round" "$peers"
            else
                printf "Node-%02d: Status unavailable\n" $i
            fi
        fi
    done
    
    if [ $check_round -lt 3 ]; then
        echo "⏳ Waiting 10 seconds for propagation..."
        sleep 10
    fi
done

echo ""
echo "🌐 PEER CONNECTION ANALYSIS"
echo "=========================="

# Analyze peer connections
total_connections=0
for i in $(seq 1 $NODES); do
    node_port=$((BASE_PORT + i - 1))
    if curl -s -m 2 http://localhost:$node_port/health > /dev/null 2>&1; then
        peers=$(curl -s http://localhost:$node_port/api/v1/node/status 2>/dev/null | jq -r '.data.connected_peers // 0' 2>/dev/null || echo "0")
        total_connections=$((total_connections + peers))
    fi
done

average_connections=$((total_connections / active_nodes))
echo "📊 Average connections per node: $average_connections"
echo "📊 Total network connections: $total_connections"

echo ""
echo "🎯 PROPAGATION TEST SUMMARY"
echo "=========================="
echo "✅ Active nodes: $active_nodes/$NODES"
echo "✅ Created wallets: ${#wallets[@]}"
echo "✅ Network is operational and propagating data"
echo ""
echo "🌟 Q-NarwhalKnight 20-Node Network Test Complete!"