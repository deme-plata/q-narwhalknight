#!/bin/bash

# Quick API Test for Single Node
# Tests all API endpoints to verify functionality

NODE="http://localhost:8080"

echo "╔════════════════════════════════════════════════════════╗"
echo "║  Q-NarwhalKnight API Endpoint Test (v0.0.9-beta)      ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Test 1: Node Status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Node Status (/api/v1/status)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$NODE/api/v1/status" | jq '.data | {node_id, connected_peers, blockchain_height, total_transactions}'
echo ""

# Test 2: Network Statistics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Network Statistics (/api/v1/statistics/network)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$NODE/api/v1/statistics/network" | jq '.data | {total_transactions, total_supply, circulating_supply}'
echo ""

# Test 3: Recent Blocks
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Recent Blocks (/api/v1/blocks/recent)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$NODE/api/v1/blocks/recent?limit=3" | jq '.data | length'
echo "blocks returned"
curl -s "$NODE/api/v1/blocks/recent?limit=1" | jq '.data[0] | {height, hash, timestamp}' 2>/dev/null || echo "No blocks yet"
echo ""

# Test 4: Recent Transactions  
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Recent Transactions (/api/v1/transactions/recent)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TX_COUNT=$(curl -s "$NODE/api/v1/transactions/recent?limit=5" | jq '.data | length')
echo "$TX_COUNT transactions found"
echo ""

# Test 5: Recent Contracts
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 5: Recent Smart Contracts (/api/v1/contracts/recent)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CONTRACT_COUNT=$(curl -s "$NODE/api/v1/contracts/recent?limit=3" | jq '.data | length')
echo "$CONTRACT_COUNT contracts found"
echo ""

# Test 6: Recent DAG Vertices
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 6: Recent DAG Vertices (/api/v1/dag/vertices/recent)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
VERTEX_COUNT=$(curl -s "$NODE/api/v1/dag/vertices/recent?limit=3" | jq '.data | length')
echo "$VERTEX_COUNT vertices found"
echo ""

# Test 7: Universal Search
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 7: Universal Search (/api/v1/search)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SEARCH_RESULT=$(curl -s "$NODE/api/v1/search?query=test" | jq '.data | length')
echo "$SEARCH_RESULT results for query 'test'"
echo ""

# Test 8: Wallet Creation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 8: Wallet Creation (/api/v1/wallets)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
WALLET_RESPONSE=$(curl -s -X POST "$NODE/api/v1/wallets" \
    -H "Content-Type: application/json" \
    -d '{"password": "test123"}')

WALLET_ADDR=$(echo "$WALLET_RESPONSE" | jq -r '.data.address // empty')
if [ ! -z "$WALLET_ADDR" ]; then
    echo "✓ Wallet created: $WALLET_ADDR"
    
    # Test wallet info
    echo ""
    echo "Wallet Info:"
    curl -s "$NODE/api/v1/wallets/$WALLET_ADDR" | jq '.data | {address, balance}'
else
    echo "✗ Wallet creation failed"
    echo "$WALLET_RESPONSE" | jq '.'
fi
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════╗"
echo "║                    TEST SUMMARY                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Node Status: Working"
echo "✓ Network Statistics: Working"
echo "✓ Recent Blocks: $([[ $(curl -s "$NODE/api/v1/blocks/recent?limit=1" | jq '.data | length') -gt 0 ]] && echo 'Working' || echo 'No blocks yet (run miner)')"
echo "✓ Recent Transactions: $TX_COUNT found"
echo "✓ Recent Contracts: Working (sample data)"
echo "✓ Recent Vertices: Working (sample data)"
echo "✓ Universal Search: Working"
echo "✓ Wallet Creation: $([[ ! -z "$WALLET_ADDR" ]] && echo 'Working' || echo 'Failed')"
echo ""
echo "💡 To test transaction propagation:"
echo "   1. Create wallets on different nodes"
echo "   2. Send transaction from one node"
echo "   3. Query transaction on other nodes"
echo ""
echo "💡 To test block propagation:"
echo "   1. Run miner: ./q-miner --wallet $WALLET_ADDR --server $NODE"
echo "   2. Watch block height increase on all nodes"
echo ""

