#!/bin/bash

# Simple script to set up QUG/QUGUSD liquidity pool for swapping
# This creates a pool with oracle-based pricing (1 QUG = $42.50)

set -e  # Exit on error

WALLET_ADDR=${1:-"qnk7d87d4734b9e021ebd3da9b16dbcf1b37d4fbcfee315c3dfd0e94e327e145d7c"}
API_URL="http://localhost:8080"

echo "=================================================="
echo "Q-NarwhalKnight: QUG/QUGUSD Pool Setup"
echo "=================================================="
echo ""
echo "Wallet: $WALLET_ADDR"
echo "API: $API_URL"
echo ""

# Step 1: Check existing pools
echo "📋 Step 1: Checking existing pools..."
POOLS=$(curl -s "$API_URL/api/v1/liquidity/pools" | jq -r '.data | length')
echo "   Found $POOLS existing pools"
echo ""

# Step 2: Mint QUGUSD (needed for liquidity)
echo "💵 Step 2: Minting QUGUSD for liquidity..."
echo "   Amount: 85 QUGUSD"
echo "   Collateral: 2 QUG"
echo "   Ratio: 160%"

MINT_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/quillon-bank/stablecoin/mint" \
  -H "Content-Type: application/json" \
  -d "{
    \"amount\": 8500000000,
    \"collateral_type\": \"QUG\",
    \"collateral_amount\": 2.0,
    \"wallet_address\": \"$WALLET_ADDR\",
    \"reason\": \"Liquidity pool creation\"
  }")

MINT_SUCCESS=$(echo "$MINT_RESPONSE" | jq -r '.success // false')
if [ "$MINT_SUCCESS" = "true" ]; then
  echo "   ✅ Minted 85 QUGUSD successfully"
else
  MINT_ERROR=$(echo "$MINT_RESPONSE" | jq -r '.error // "Unknown error"')
  echo "   ⚠️  Minting failed: $MINT_ERROR"
  echo "   Continuing anyway (you may already have QUGUSD)..."
fi
echo ""

# Step 3: Create liquidity pool
echo "💧 Step 3: Creating QUG/QUGUSD liquidity pool..."
echo "   Pool ratio: 1 QUG = 42.5 QUGUSD"
echo "   Adding: 1 QUG + 42.5 QUGUSD"

# Use the actual QUGUSD token address
QUGUSD_ADDR="0x0000000000000000000000000000000000000000000000000000000000000002"

POOL_RESPONSE=$(curl -s -X POST "$API_URL/api/v1/liquidity/add" \
  -H "Content-Type: application/json" \
  -d "{
    \"token0\": \"QUG\",
    \"token1\": \"$QUGUSD_ADDR\",
    \"amount0\": 100000000,
    \"amount1\": 4250000000,
    \"provider\": \"$WALLET_ADDR\"
  }")

POOL_SUCCESS=$(echo "$POOL_RESPONSE" | jq -r '.success // false')
if [ "$POOL_SUCCESS" = "true" ]; then
  POOL_ID=$(echo "$POOL_RESPONSE" | jq -r '.data.pool_id')
  echo "   ✅ Liquidity pool created: $POOL_ID"
else
  POOL_ERROR=$(echo "$POOL_RESPONSE" | jq -r '.error // "Unknown error"')
  echo "   ❌ Pool creation failed: $POOL_ERROR"
  echo ""
  echo "Troubleshooting:"
  echo "  1. Make sure you have enough QUG balance (need at least 1 QUG)"
  echo "  2. Make sure you have QUGUSD balance (minted in step 2)"
  echo "  3. Check wallet address format is correct"
  exit 1
fi
echo ""

# Step 4: Verify pool
echo "✅ Step 4: Verifying pool creation..."
POOLS_AFTER=$(curl -s "$API_URL/api/v1/liquidity/pools")
QUG_POOLS=$(echo "$POOLS_AFTER" | jq -r '.data[] | select(.token0 == "QUG" or .token1 == "QUG")')

if [ -z "$QUG_POOLS" ]; then
  echo "   ⚠️  Warning: No QUG pools found after creation"
else
  echo "   ✅ Pool verified!"
  echo ""
  echo "$QUG_POOLS" | jq -r '
    "   Pool ID: " + .pool_id + "\n" +
    "   Tokens: " + .token0 + " / " + .token1 + "\n" +
    "   Reserves: " + (.reserve0|tostring) + " / " + (.reserve1|tostring) + "\n" +
    "   Provider: " + .provider
  '
fi
echo ""

# Step 5: Final status
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "You can now swap QUG ↔ QUGUSD in the wallet GUI"
echo "Navigate to: DEX → Swap"
echo ""
echo "Pool Details:"
echo "  - 1 QUG ≈ 42.5 QUGUSD"
echo "  - 0.3% swap fee"
echo "  - Instant execution"
echo ""
echo "Happy swapping! 💱✨"
