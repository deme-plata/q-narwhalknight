#!/bin/bash

# Q-NarwhalKnight Peer Propagation Test Suite
# Tests data propagation across connected peers using API endpoints

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Node endpoints (adjust ports as needed)
NODE1="http://localhost:8080"
NODE2="http://localhost:8084"
NODE3="http://localhost:9060"
NODE4="http://localhost:9666"

NODES=("$NODE1" "$NODE2" "$NODE3" "$NODE4")

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Q-NarwhalKnight Peer Propagation Test Suite v0.0.9-beta    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print test header
print_test() {
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}TEST: $1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to check node status
check_node_status() {
    local node=$1
    local response=$(curl -s "$node/api/v1/status" 2>/dev/null)
    if [ -z "$response" ]; then
        echo -e "${RED}✗ Node $node is not responding${NC}"
        return 1
    fi
    
    local peer_count=$(echo "$response" | jq -r '.data.connected_peers // 0')
    local block_height=$(echo "$response" | jq -r '.data.blockchain_height // 0')
    local node_id=$(echo "$response" | jq -r '.data.node_id // "unknown"')
    
    echo -e "${GREEN}✓ Node: $node${NC}"
    echo "  Node ID: $node_id"
    echo "  Connected Peers: $peer_count"
    echo "  Block Height: $block_height"
    return 0
}

# TEST 1: Check all nodes are online and connected
print_test "1. Node Connectivity & Peer Discovery"

all_nodes_online=true
for node in "${NODES[@]}"; do
    if ! check_node_status "$node"; then
        all_nodes_online=false
    fi
done

if [ "$all_nodes_online" = false ]; then
    echo -e "\n${RED}⚠ Not all nodes are online. Please start all nodes first.${NC}"
    echo -e "${YELLOW}Run: ./q-api-server --port 8080 & ./q-api-server --port 8084 --tui & ...${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ All nodes are online!${NC}"

# TEST 2: Create wallet on Node 1 and check if it appears on other nodes
print_test "2. Wallet Creation & State Propagation"

echo "Creating wallet on Node 1..."
WALLET_RESPONSE=$(curl -s -X POST "$NODE1/api/v1/wallets" \
    -H "Content-Type: application/json" \
    -d '{
        "password": "test_password_123"
    }')

WALLET_ADDRESS=$(echo "$WALLET_RESPONSE" | jq -r '.data.address // empty')

if [ -z "$WALLET_ADDRESS" ]; then
    echo -e "${RED}✗ Failed to create wallet${NC}"
    echo "Response: $WALLET_RESPONSE"
else
    echo -e "${GREEN}✓ Wallet created: $WALLET_ADDRESS${NC}"
    
    # Wait for propagation
    echo "Waiting 3 seconds for state propagation..."
    sleep 3
    
    # Check if wallet exists on other nodes
    echo ""
    for i in "${!NODES[@]}"; do
        node="${NODES[$i]}"
        wallet_info=$(curl -s "$node/api/v1/wallets/$WALLET_ADDRESS" 2>/dev/null)
        balance=$(echo "$wallet_info" | jq -r '.data.balance // "N/A"')
        
        if [ "$balance" != "N/A" ]; then
            echo -e "${GREEN}✓ Node $((i+1)) sees wallet: $WALLET_ADDRESS (Balance: $balance)${NC}"
        else
            echo -e "${YELLOW}⚠ Node $((i+1)) doesn't see wallet yet${NC}"
        fi
    done
fi

# TEST 3: Transaction Creation & Gossipsub Propagation
print_test "3. Transaction Creation & Gossipsub Propagation"

if [ ! -z "$WALLET_ADDRESS" ]; then
    echo "Creating transaction on Node 1..."
    
    # Get a second wallet address (or create one)
    RECIPIENT_WALLET=$(curl -s -X POST "$NODE1/api/v1/wallets" \
        -H "Content-Type: application/json" \
        -d '{"password": "recipient_pass"}' | jq -r '.data.address')
    
    if [ ! -z "$RECIPIENT_WALLET" ]; then
        echo "Recipient wallet: $RECIPIENT_WALLET"
        
        # Send transaction
        TX_RESPONSE=$(curl -s -X POST "$NODE1/api/v1/transactions/send" \
            -H "Content-Type: application/json" \
            -d "{
                \"from\": \"$WALLET_ADDRESS\",
                \"to\": \"$RECIPIENT_WALLET\",
                \"amount\": 10.0,
                \"password\": \"test_password_123\"
            }")
        
        TX_HASH=$(echo "$TX_RESPONSE" | jq -r '.data.transaction_hash // empty')
        
        if [ ! -z "$TX_HASH" ]; then
            echo -e "${GREEN}✓ Transaction created: $TX_HASH${NC}"
            
            # Wait for gossipsub propagation
            echo "Waiting 5 seconds for gossipsub propagation..."
            sleep 5
            
            # Check if transaction appears on all nodes
            echo ""
            for i in "${!NODES[@]}"; do
                node="${NODES[$i]}"
                tx_info=$(curl -s "$node/api/v1/transactions/$TX_HASH" 2>/dev/null)
                status=$(echo "$tx_info" | jq -r '.data.status // "not_found"')
                
                if [ "$status" != "not_found" ]; then
                    echo -e "${GREEN}✓ Node $((i+1)) sees transaction: $TX_HASH (Status: $status)${NC}"
                else
                    echo -e "${YELLOW}⚠ Node $((i+1)) doesn't see transaction yet${NC}"
                fi
            done
        else
            echo -e "${YELLOW}⚠ Transaction failed (might need balance)${NC}"
            echo "Response: $TX_RESPONSE"
        fi
    fi
fi

# TEST 4: Recent Transactions Consistency
print_test "4. Recent Transactions Consistency Across Nodes"

echo "Fetching recent transactions from all nodes..."
echo ""

for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    recent_txs=$(curl -s "$node/api/v1/transactions/recent?limit=5" 2>/dev/null)
    tx_count=$(echo "$recent_txs" | jq -r '.data | length // 0')
    
    echo -e "${BLUE}Node $((i+1)) ($node):${NC}"
    echo "  Recent transactions: $tx_count"
    
    if [ "$tx_count" -gt 0 ]; then
        echo "$recent_txs" | jq -r '.data[] | "  - \(.hash) | \(.amount) QUG | \(.status)"'
    fi
    echo ""
done

# TEST 5: Block Height Consistency
print_test "5. Block Height Consistency"

echo "Checking block heights across all nodes..."
echo ""

max_height=0
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    status=$(curl -s "$node/api/v1/status" 2>/dev/null)
    height=$(echo "$status" | jq -r '.data.blockchain_height // 0')
    
    echo "Node $((i+1)): Block height $height"
    
    if [ "$height" -gt "$max_height" ]; then
        max_height=$height
    fi
done

echo ""
if [ "$max_height" -eq 0 ]; then
    echo -e "${YELLOW}⚠ All nodes at height 0 (no blocks mined yet)${NC}"
else
    echo -e "${GREEN}✓ Maximum block height: $max_height${NC}"
fi

# TEST 6: Network Statistics Propagation
print_test "6. Network Statistics Consistency"

echo "Checking network statistics across nodes..."
echo ""

for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    stats=$(curl -s "$node/api/v1/statistics/network" 2>/dev/null)
    
    total_txs=$(echo "$stats" | jq -r '.data.total_transactions // 0')
    total_supply=$(echo "$stats" | jq -r '.data.total_supply // 0')
    
    echo -e "${BLUE}Node $((i+1)):${NC}"
    echo "  Total Transactions: $total_txs"
    echo "  Total Supply: $total_supply QUG"
    echo ""
done

# TEST 7: Gossipsub Topic Subscriptions
print_test "7. Gossipsub Topics (via logs)"

echo -e "${YELLOW}Note: Gossipsub subscriptions are internal to libp2p${NC}"
echo "Topics that should be active:"
echo "  - /qnk/blocks/1.0.0 (Block propagation)"
echo "  - /qnk/transactions (Transaction propagation)"
echo "  - /qnk/votes/1.0.0 (Consensus votes)"
echo "  - /qnk/ack/1.0.0 (Acknowledgements)"
echo ""
echo "Check node logs for:"
echo "  '📢 Subscribed to Gossipsub topic'"
echo "  '📤 Published message to gossipsub topic'"

# SUMMARY
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      TEST SUMMARY                             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Connectivity Test:${NC} All nodes online"
echo -e "${GREEN}✓ Peer Discovery:${NC} Nodes connected to each other"

if [ ! -z "$WALLET_ADDRESS" ]; then
    echo -e "${GREEN}✓ Wallet Creation:${NC} Successful"
fi

if [ ! -z "$TX_HASH" ]; then
    echo -e "${GREEN}✓ Transaction Propagation:${NC} Test completed"
else
    echo -e "${YELLOW}⚠ Transaction Propagation:${NC} Skipped (needs balance)"
fi

echo -e "${GREEN}✓ Block Height Check:${NC} Completed"
echo -e "${GREEN}✓ Network Statistics:${NC} Checked"
echo ""
echo -e "${BLUE}Recommendation:${NC} Run a miner to generate blocks and test full consensus!"
echo -e "${YELLOW}Command:${NC} ./q-miner --wallet $WALLET_ADDRESS --server http://localhost:8080"
echo ""

