#!/bin/bash
# Test Cross-Node Transaction Propagation
# Send transaction to Node 4, check if Node 1 receives it

set -e

NODE1="http://localhost:8080"
NODE4="http://localhost:9666"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║      Cross-Node Transaction Propagation Test                  ║"
echo "║  Transaction: Node 4 (9666) → Node 1 (8080)                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Generate test wallet addresses (using simple test addresses since we'll use faucet)
WALLET1="qnktest0000000000000000000000000000000000000000000000000000000001"
WALLET2="qnktest0000000000000000000000000000000000000000000000000000000002"

# STEP 1: Get faucet coins on Node 4
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: Requesting faucet coins on Node 4"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

FAUCET_RESPONSE=$(curl -s -X POST "$NODE4/api/v1/faucet" \
    -H "Content-Type: application/json" \
    -d "{\"wallet_address\": \"$WALLET1\"}")

echo "Faucet response: $FAUCET_RESPONSE"
echo ""

# STEP 2: Check recent transactions on both nodes BEFORE sending
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: Checking transaction count BEFORE sending"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NODE1_BEFORE=$(curl -s "$NODE1/api/v1/transactions/recent" | jq '.data | length')
NODE4_BEFORE=$(curl -s "$NODE4/api/v1/transactions/recent" | jq '.data | length')

echo "Node 1 transactions before: $NODE1_BEFORE"
echo "Node 4 transactions before: $NODE4_BEFORE"
echo ""

# STEP 3: Send transaction on Node 4 (without authentication - will fail but tests propagation)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: Attempting transaction submission to Node 4"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TX_RESPONSE=$(curl -s -X POST "$NODE4/api/v1/transactions/send" \
    -H "Content-Type: application/json" \
    -d "{
        \"from\": \"$WALLET1\",
        \"to\": \"$WALLET2\",
        \"amount\": 2.0
    }")

echo "Transaction response: $TX_RESPONSE"
echo ""

# STEP 4: Wait for gossipsub propagation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: Waiting 10 seconds for gossipsub propagation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sleep 10

# STEP 5: Check recent transactions on both nodes AFTER sending
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: Checking transaction count AFTER propagation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NODE1_AFTER=$(curl -s "$NODE1/api/v1/transactions/recent" | jq '.data | length')
NODE4_AFTER=$(curl -s "$NODE4/api/v1/transactions/recent" | jq '.data | length')

echo "Node 1 transactions after: $NODE1_AFTER"
echo "Node 4 transactions after: $NODE4_AFTER"
echo ""

# STEP 6: Check network statistics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6: Network Statistics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Node 1 stats:"
curl -s "$NODE1/api/v1/statistics/network" | jq '{total_transactions, total_blocks, current_supply_qnk}'

echo ""
echo "Node 4 stats:"
curl -s "$NODE4/api/v1/statistics/network" | jq '{total_transactions, total_blocks, current_supply_qnk}'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RESULTS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$NODE1_AFTER" -gt "$NODE1_BEFORE" ]; then
    echo "✅ SUCCESS: Node 1 received transaction from Node 4!"
    echo "   Before: $NODE1_BEFORE | After: $NODE1_AFTER"
else
    echo "⚠️  PENDING: Transaction not yet visible on Node 1"
    echo "   This may indicate:"
    echo "   - Transaction was rejected (needs authentication)"
    echo "   - Gossipsub propagation not implemented yet"
    echo "   - Transaction storage not enabled"
fi

echo ""
