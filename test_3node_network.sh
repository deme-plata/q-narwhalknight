#!/bin/bash
# Q-NarwhalKnight 3-Node Network Test
# Tests transaction propagation across libp2p network

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Q-NarwhalKnight 3-Node Network Propagation Test           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
# Production node runs on 8080/9000
# nova-chat uses 9092, so avoid that port range
NODE1_PORT=8091
NODE2_PORT=8092
NODE3_PORT=8093

NODE1_P2P_PORT=9101
NODE2_P2P_PORT=9102
NODE3_P2P_PORT=9103

NODE1_DB="./testnet-data/node1"
NODE2_DB="./testnet-data/node2"
NODE3_DB="./testnet-data/node3"

BINARY="./target/release/q-api-server"

# Create data directories
mkdir -p "$NODE1_DB" "$NODE2_DB" "$NODE3_DB"

# Function to start a node
start_node() {
    local node_id=$1
    local api_port=$2
    local p2p_port=$3
    local db_path=$4

    echo "🚀 Starting Node $node_id (API: $api_port, P2P: $p2p_port)"

    Q_DB_PATH="$db_path" Q_P2P_PORT="$p2p_port" \
        timeout 36000 "$BINARY" --port "$api_port" --node-id "node$node_id" \
        > "testnet-data/node${node_id}.log" 2>&1 &

    local pid=$!
    echo "$pid" > "testnet-data/node${node_id}.pid"
    echo "   PID: $pid"
}

# Function to stop all nodes
stop_all_nodes() {
    echo ""
    echo "🛑 Stopping all nodes..."
    for i in 1 2 3; do
        if [ -f "testnet-data/node${i}.pid" ]; then
            pid=$(cat "testnet-data/node${i}.pid")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid"
                echo "   Stopped Node $i (PID: $pid)"
            fi
            rm "testnet-data/node${i}.pid"
        fi
    done
}

# Function to create wallet
create_wallet() {
    local port=$1
    local label=$2

    echo "💰 Creating wallet on port $port: $label"
    response=$(curl -s -X POST "http://localhost:${port}/api/v1/wallets/create" \
        -H "Content-Type: application/json" \
        -d "{\"label\": \"$label\"}")

    address=$(echo "$response" | jq -r '.data.address_formatted')
    echo "   Address: $address"
    echo "$address"
}

# Function to get balance
get_balance() {
    local port=$1
    local address=$2

    balance=$(curl -s "http://localhost:${port}/api/v1/wallets/${address}/balance" | jq -r '.data.balance_qug // .balance_qug // 0')
    echo "$balance"
}

# Function to use faucet
use_faucet() {
    local port=$1
    local address=$2

    echo "🚰 Using faucet for $address"
    response=$(curl -s -X POST "http://localhost:${port}/api/v1/faucet" \
        -H "Content-Type: application/json" \
        -d "{\"address\": \"$address\"}")

    echo "   Response: $response"
}

# Function to send transaction
send_transaction() {
    local port=$1
    local from=$2
    local to=$3
    local amount=$4

    echo "💸 Sending $amount QUG from Node $port"
    echo "   From: $from"
    echo "   To:   $to"

    response=$(curl -s -X POST "http://localhost:${port}/api/v1/transactions/send" \
        -H "Content-Type: application/json" \
        -d "{
            \"from\": \"$from\",
            \"to\": \"$to\",
            \"amount_qug\": $amount,
            \"fee_qug\": 0.1
        }")

    echo "   Response: $response"
}

# Function to check transaction status
check_transaction() {
    local port=$1
    local tx_id=$2

    echo "🔍 Checking transaction on port $port"
    response=$(curl -s "http://localhost:${port}/api/v1/transactions/$tx_id")
    echo "   $response"
}

# Function to wait for API
wait_for_api() {
    local port=$1
    local max_wait=30
    local count=0

    echo "⏳ Waiting for API on port $port..."
    while [ $count -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/api/v1/health" > /dev/null 2>&1; then
            echo "   ✅ API ready on port $port"
            return 0
        fi
        count=$((count + 1))
        sleep 1
    done
    echo "   ❌ API timeout on port $port"
    return 1
}

# Trap to ensure cleanup
trap stop_all_nodes EXIT

# Main test sequence
echo "═══════════════════════════════════════════════════════════════"
echo "Step 1: Starting 3 nodes"
echo "═══════════════════════════════════════════════════════════════"

start_node 1 $NODE1_PORT $NODE1_P2P_PORT $NODE1_DB
start_node 2 $NODE2_PORT $NODE2_P2P_PORT $NODE2_DB
start_node 3 $NODE3_PORT $NODE3_P2P_PORT $NODE3_DB

echo ""
echo "Waiting for all nodes to start..."
sleep 10

wait_for_api $NODE1_PORT
wait_for_api $NODE2_PORT
wait_for_api $NODE3_PORT

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 2: Creating wallets"
echo "═══════════════════════════════════════════════════════════════"

wallet1=$(create_wallet $NODE1_PORT "Node1 Wallet")
wallet2=$(create_wallet $NODE2_PORT "Node2 Wallet")
wallet3=$(create_wallet $NODE3_PORT "Node3 Wallet")

echo ""
echo "Created wallets:"
echo "  Node 1: $wallet1"
echo "  Node 2: $wallet2"
echo "  Node 3: $wallet3"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 3: Funding wallets via faucet"
echo "═══════════════════════════════════════════════════════════════"

use_faucet $NODE1_PORT "$wallet1"
sleep 2
use_faucet $NODE2_PORT "$wallet2"
sleep 2
use_faucet $NODE3_PORT "$wallet3"

echo ""
echo "Waiting for faucet transactions to propagate..."
sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 4: Checking balances"
echo "═══════════════════════════════════════════════════════════════"

balance1=$(get_balance $NODE1_PORT "$wallet1")
balance2=$(get_balance $NODE2_PORT "$wallet2")
balance3=$(get_balance $NODE3_PORT "$wallet3")

echo "Balances after faucet:"
echo "  Node 1: $balance1 QUG"
echo "  Node 2: $balance2 QUG"
echo "  Node 3: $balance3 QUG"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 5: Testing cross-node transaction propagation"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "Test 1: Node 1 → Node 2 (send 10 QUG)"
send_transaction $NODE1_PORT "$wallet1" "$wallet2" 10

echo ""
echo "Waiting for transaction to propagate..."
sleep 5

echo ""
echo "Checking balances on all nodes after TX 1:"
for port in $NODE1_PORT $NODE2_PORT $NODE3_PORT; do
    echo "  Node (port $port):"
    echo "    Wallet 1: $(get_balance $port "$wallet1") QUG"
    echo "    Wallet 2: $(get_balance $port "$wallet2") QUG"
done

echo ""
echo "Test 2: Node 2 → Node 3 (send 5 QUG)"
send_transaction $NODE2_PORT "$wallet2" "$wallet3" 5

echo ""
echo "Waiting for transaction to propagate..."
sleep 5

echo ""
echo "Checking balances on all nodes after TX 2:"
for port in $NODE1_PORT $NODE2_PORT $NODE3_PORT; do
    echo "  Node (port $port):"
    echo "    Wallet 2: $(get_balance $port "$wallet2") QUG"
    echo "    Wallet 3: $(get_balance $port "$wallet3") QUG"
done

echo ""
echo "Test 3: Node 3 → Node 1 (send 2 QUG)"
send_transaction $NODE3_PORT "$wallet3" "$wallet1" 2

echo ""
echo "Waiting for transaction to propagate..."
sleep 5

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 6: Final balance verification across all nodes"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "Final balances - checking consistency across network:"
for port in $NODE1_PORT $NODE2_PORT $NODE3_PORT; do
    echo ""
    echo "Node on port $port:"
    b1=$(get_balance $port "$wallet1")
    b2=$(get_balance $port "$wallet2")
    b3=$(get_balance $port "$wallet3")
    echo "  Wallet 1: $b1 QUG"
    echo "  Wallet 2: $b2 QUG"
    echo "  Wallet 3: $b3 QUG"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Step 7: Network connectivity check"
echo "═══════════════════════════════════════════════════════════════"

for port in $NODE1_PORT $NODE2_PORT $NODE3_PORT; do
    echo ""
    echo "Node on port $port - P2P Peers:"
    curl -s "http://localhost:${port}/api/v1/network/active-peers" | jq '.'
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Test Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Logs available at:"
echo "  Node 1: testnet-data/node1.log"
echo "  Node 2: testnet-data/node2.log"
echo "  Node 3: testnet-data/node3.log"
echo ""
echo "Press Ctrl+C to stop all nodes..."
echo ""

# Keep running until user interrupts
wait
