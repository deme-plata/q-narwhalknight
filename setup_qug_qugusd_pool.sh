#!/bin/bash
# Setup QUG/QUGUSD Liquidity Pool
# This script helps you create the initial liquidity pool for QUG/QUGUSD trading

API_URL="${API_URL:-http://localhost:8090}"
WALLET_ADDRESS="${1:-}"

if [ -z "$WALLET_ADDRESS" ]; then
    echo "❌ Error: Wallet address required"
    echo "Usage: $0 <wallet-address>"
    echo "Example: $0 qnk1234567890abcdef..."
    exit 1
fi

echo "🌊 Setting up QUG/QUGUSD Liquidity Pool"
echo "================================================"
echo "API Server: $API_URL"
echo "Wallet: $WALLET_ADDRESS"
echo ""

# Default amounts: 10 QUG and 10 QUGUSD
AMOUNT_QUG="${AMOUNT_QUG:-1000000000}"      # 10 QUG (8 decimals)
AMOUNT_QUGUSD="${AMOUNT_QUGUSD:-1000000000}" # 10 QUGUSD (8 decimals)

echo "📊 Pool Parameters:"
echo "  Token 0: QUG"
echo "  Token 1: QUGUSD"
echo "  Amount 0: $AMOUNT_QUG ($(echo "scale=2; $AMOUNT_QUG / 100000000" | bc) QUG)"
echo "  Amount 1: $AMOUNT_QUGUSD ($(echo "scale=2; $AMOUNT_QUGUSD / 100000000" | bc) QUGUSD)"
echo ""

# Check if API server is running
echo "🔍 Checking API server..."
if ! curl -s "$API_URL/api/v1/liquidity/pools" > /dev/null 2>&1; then
    echo "❌ Error: API server not reachable at $API_URL"
    echo "Please start the API server first:"
    echo "  cargo run --bin q-api-server --release"
    exit 1
fi
echo "✅ API server is running"
echo ""

# Check existing pools
echo "📋 Checking existing pools..."
EXISTING_POOLS=$(curl -s "$API_URL/api/v1/liquidity/pools" | jq -r '.data // [] | length')
echo "Found $EXISTING_POOLS existing pool(s)"
echo ""

# Check if QUG/QUGUSD pool already exists
QUG_QUGUSD_EXISTS=$(curl -s "$API_URL/api/v1/liquidity/pools" | \
    jq -r '.data // [] | map(select(.token0 == "QUG" and .token1 == "QUGUSD" or .token0 == "QUGUSD" and .token1 == "QUG")) | length')

if [ "$QUG_QUGUSD_EXISTS" -gt 0 ]; then
    echo "⚠️  QUG/QUGUSD pool already exists!"
    echo "Pool details:"
    curl -s "$API_URL/api/v1/liquidity/pools" | \
        jq -r '.data // [] | map(select(.token0 == "QUG" and .token1 == "QUGUSD" or .token0 == "QUGUSD" and .token1 == "QUG"))'
    echo ""
    read -p "Add more liquidity to existing pool? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Add liquidity
echo "💰 Adding liquidity to QUG/QUGUSD pool..."
RESPONSE=$(curl -s -X POST "$API_URL/api/v1/liquidity/add" \
    -H "Content-Type: application/json" \
    -d "{
        \"token0\": \"QUG\",
        \"token1\": \"QUGUSD\",
        \"amount0\": $AMOUNT_QUG,
        \"amount1\": $AMOUNT_QUGUSD,
        \"provider\": \"$WALLET_ADDRESS\"
    }")

# Check response
SUCCESS=$(echo "$RESPONSE" | jq -r '.success // false')

if [ "$SUCCESS" = "true" ]; then
    echo "✅ Liquidity pool created successfully!"
    echo ""
    echo "📊 Pool Details:"
    echo "$RESPONSE" | jq -r '.data'
    echo ""
    echo "🎉 You can now perform QUG ↔ QUGUSD swaps!"
    echo ""
    echo "Test swap with:"
    echo "  curl -X POST $API_URL/api/v1/swap \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"from_token\":\"QUG\",\"to_token\":\"QUGUSD\",\"amount\":100000000,\"wallet\":\"$WALLET_ADDRESS\"}'"
else
    echo "❌ Failed to create liquidity pool"
    echo "Error: $(echo "$RESPONSE" | jq -r '.error // "Unknown error"')"
    echo ""
    echo "Full response:"
    echo "$RESPONSE" | jq '.'
    exit 1
fi
