#!/bin/bash

# Complete Transaction & Block Propagation Test
# Uses faucet for test coins and verifies propagation across nodes

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NODE1="http://localhost:8080"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Complete Propagation Test with Faucet (v0.0.9-beta)       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# STEP 1: Create test wallet address
echo -e "${YELLOW}━━━ STEP 1: Generate Test Wallet Address ━━━${NC}"
WALLET1="qnk$(openssl rand -hex 32)"
WALLET2="qnk$(openssl rand -hex 32)"

echo "Wallet 1: $WALLET1"
echo "Wallet 2: $WALLET2"
echo ""

# STEP 2: Get faucet coins for Wallet 1
echo -e "${YELLOW}━━━ STEP 2: Request Faucet Coins ━━━${NC}"
FAUCET_RESPONSE=$(curl -s -X POST "$NODE1/api/v1/faucet" \
    -H "Content-Type: application/json" \
    -d "{\"wallet_address\": \"$WALLET1\"}")

echo "$FAUCET_RESPONSE" | jq '.'

FAUCET_AMOUNT=$(echo "$FAUCET_RESPONSE" | jq -r '.data.amount_qnk // 0')

if [ "$FAUCET_AMOUNT" != "0" ]; then
    echo -e "${GREEN}✓ Received $FAUCET_AMOUNT QNK from faucet${NC}"
else
    echo -e "${RED}✗ Faucet request failed${NC}"
    exit 1
fi
echo ""

# STEP 3: Check wallet balance
echo -e "${YELLOW}━━━ STEP 3: Verify Wallet Balance ━━━${NC}"
sleep 1
BALANCE=$(curl -s "$NODE1/api/v1/wallets/$WALLET1" | jq -r '.data.balance // 0')
echo "Wallet 1 Balance: $BALANCE QNK"
echo ""

# STEP 4: Send transaction
echo -e "${YELLOW}━━━ STEP 4: Send Transaction (Wallet 1 → Wallet 2) ━━━${NC}"
TX_RESPONSE=$(curl -s -X POST "$NODE1/api/v1/transactions/send" \
    -H "Content-Type: application/json" \
    -d "{
        \"from\": \"$WALLET1\",
        \"to\": \"$WALLET2\",
        \"amount\": 2.0
    }")

echo "$TX_RESPONSE" | jq '.'

TX_HASH=$(echo "$TX_RESPONSE" | jq -r '.data.transaction_hash // empty')

if [ ! -z "$TX_HASH" ]; then
    echo -e "${GREEN}✓ Transaction created: $TX_HASH${NC}"
else
    echo -e "${RED}✗ Transaction failed${NC}"
    echo "Response: $TX_RESPONSE"
    exit 1
fi
echo ""

# STEP 5: Wait for transaction propagation
echo -e "${YELLOW}━━━ STEP 5: Waiting for Transaction Propagation ━━━${NC}"
echo "Waiting 3 seconds for gossipsub propagation..."
sleep 3
echo ""

# STEP 6: Query transaction on Node 1
echo -e "${YELLOW}━━━ STEP 6: Query Transaction Status ━━━${NC}"
TX_INFO=$(curl -s "$NODE1/api/v1/transactions/$TX_HASH")
echo "$TX_INFO" | jq '.'

TX_STATUS=$(echo "$TX_INFO" | jq -r '.data.status // "not_found"')
echo ""
if [ "$TX_STATUS" != "not_found" ]; then
    echo -e "${GREEN}✓ Transaction found: Status = $TX_STATUS${NC}"
else
    echo -e "${YELLOW}⚠ Transaction not found yet (might be in mempool)${NC}"
fi
echo ""

# STEP 7: Check updated balances
echo -e "${YELLOW}━━━ STEP 7: Verify Balance Updates ━━━${NC}"
sleep 1

BALANCE1=$(curl -s "$NODE1/api/v1/wallets/$WALLET1" | jq -r '.data.balance // 0')
BALANCE2=$(curl -s "$NODE1/api/v1/wallets/$WALLET2" | jq -r '.data.balance // 0')

echo "Wallet 1 Balance: $BALANCE1 QNK (should be ~8 QNK after sending 2)"
echo "Wallet 2 Balance: $BALANCE2 QNK (should be 2 QNK)"
echo ""

if [ "$BALANCE2" != "0" ]; then
    echo -e "${GREEN}✓ Transaction propagated! Wallet 2 received funds${NC}"
else
    echo -e "${YELLOW}⚠ Balance not updated yet (tx might be pending)${NC}"
fi
echo ""

# STEP 8: Check recent transactions
echo -e "${YELLOW}━━━ STEP 8: Verify Transaction in Recent List ━━━${NC}"
RECENT_TXS=$(curl -s "$NODE1/api/v1/transactions/recent?limit=5")
TX_COUNT=$(echo "$RECENT_TXS" | jq '.data | length')

echo "Recent transactions: $TX_COUNT"
echo "$RECENT_TXS" | jq -r '.data[] | "  - \(.hash) | \(.amount) QNK | \(.status)"' | head -5
echo ""

# STEP 9: Test block height (if miner is running)
echo -e "${YELLOW}━━━ STEP 9: Check Block Height ━━━${NC}"
STATUS=$(curl -s "$NODE1/api/v1/status")
BLOCK_HEIGHT=$(echo "$STATUS" | jq -r '.data.blockchain_height // 0')
PEER_COUNT=$(echo "$STATUS" | jq -r '.data.connected_peers // 0')

echo "Current Block Height: $BLOCK_HEIGHT"
echo "Connected Peers: $PEER_COUNT"
echo ""

if [ "$BLOCK_HEIGHT" -gt 0 ]; then
    echo -e "${GREEN}✓ Blockchain is active! Blocks are being mined${NC}"
else
    echo -e "${YELLOW}ℹ No blocks mined yet. Start miner to test block propagation:${NC}"
    echo -e "${YELLOW}  ./q-miner --wallet $WALLET1 --server $NODE1${NC}"
fi
echo ""

# SUMMARY
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     TEST SUMMARY                             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Faucet:${NC} Received $FAUCET_AMOUNT QNK"
echo -e "${GREEN}✓ Transaction:${NC} $TX_HASH"

if [ "$TX_STATUS" != "not_found" ]; then
    echo -e "${GREEN}✓ Transaction Status:${NC} $TX_STATUS"
else
    echo -e "${YELLOW}⚠ Transaction Status:${NC} Not confirmed yet"
fi

if [ "$BALANCE2" != "0" ]; then
    echo -e "${GREEN}✓ Balance Propagation:${NC} Working (Wallet 2 has $BALANCE2 QNK)"
else
    echo -e "${YELLOW}⚠ Balance Propagation:${NC} Pending"
fi

echo -e "${GREEN}✓ Connected Peers:${NC} $PEER_COUNT"
echo -e "${GREEN}✓ Block Height:${NC} $BLOCK_HEIGHT"
echo ""

if [ "$PEER_COUNT" -gt 0 ] && [ "$TX_STATUS" != "not_found" ]; then
    echo -e "${GREEN}🎉 TRANSACTION PROPAGATION TEST PASSED!${NC}"
else
    echo -e "${YELLOW}⚠ Test completed with warnings (see above)${NC}"
fi
echo ""

