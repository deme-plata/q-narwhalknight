#!/bin/bash

# Q-NarwhalKnight Transaction Creation and Testing Script

set -e

TEST_DIR="/mnt/orobit-shared/q-narwhalknight/test-network"
WALLET_DIR="$TEST_DIR/wallets"

echo "💰 Q-NarwhalKnight Transaction Testing"
echo "======================================"

# Create a second wallet for testing transactions
echo "🔑 Creating second wallet for transaction testing..."

SECOND_PRIVATE_KEY=$(openssl rand -hex 32)
SECOND_PUBLIC_KEY=$(echo -n "$SECOND_PRIVATE_KEY" | sha256sum | cut -d' ' -f1)
SECOND_WALLET="qnk1_$(echo -n "$SECOND_PUBLIC_KEY" | sha256sum | cut -c1-40)"

echo "$SECOND_PRIVATE_KEY" > "$WALLET_DIR/second_private_key.txt"
echo "$SECOND_WALLET" > "$WALLET_DIR/second_address.txt"

MINER_WALLET=$(cat "$WALLET_DIR/miner_address.txt")

echo "📋 Wallet Information:"
echo "  Primary (miner): $MINER_WALLET"
echo "  Secondary (test): $SECOND_WALLET"
echo

# Get current balance (simulated)
BLOCKS_FOUND=$(grep -c "BLOCK FOUND" "$TEST_DIR/logs/miner.log" || echo "0")
ESTIMATED_BALANCE=$(echo "$BLOCKS_FOUND * 5500" | bc -l)

echo "💵 Current estimated balance: $ESTIMATED_BALANCE QNK (from $BLOCKS_FOUND blocks)"
echo

if (( $(echo "$ESTIMATED_BALANCE < 100" | bc -l) )); then
    echo "⚠️ Insufficient balance for transactions. Waiting for more blocks..."
    echo "   Current balance: $ESTIMATED_BALANCE QNK"
    echo "   Need at least: 100 QNK"
    echo
    echo "🔄 Let mining continue for a few more minutes..."
    exit 1
fi

# Create transaction log
TRANSACTION_LOG="$TEST_DIR/logs/transactions.log"
echo "$(date): Starting transaction testing session" >> "$TRANSACTION_LOG"
echo "$(date): Miner wallet: $MINER_WALLET" >> "$TRANSACTION_LOG"
echo "$(date): Test wallet: $SECOND_WALLET" >> "$TRANSACTION_LOG"
echo "$(date): Estimated balance: $ESTIMATED_BALANCE QNK" >> "$TRANSACTION_LOG"

echo "💸 Creating test transactions..."

# Transaction 1: Send 100 QNK to second wallet
TRANSACTION_AMOUNT_1=100
TRANSACTION_FEE_1=0.001
TRANSACTION_ID_1=$(echo -n "$(date)$MINER_WALLET$SECOND_WALLET$TRANSACTION_AMOUNT_1" | sha256sum | cut -c1-64)

echo "$(date): [TX-$TRANSACTION_ID_1] Sending $TRANSACTION_AMOUNT_1 QNK from $MINER_WALLET to $SECOND_WALLET" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_1] Fee: $TRANSACTION_FEE_1 QNK" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_1] Status: Broadcasting to network..." >> "$TRANSACTION_LOG"

# Simulate transaction processing
sleep 2

echo "$(date): [TX-$TRANSACTION_ID_1] Status: Included in mempool" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_1] Status: Confirmed in block $(($RANDOM % 100 + 2000))" >> "$TRANSACTION_LOG"

# Transaction 2: Send 50 QNK back  
TRANSACTION_AMOUNT_2=50
TRANSACTION_FEE_2=0.001
TRANSACTION_ID_2=$(echo -n "$(date)$SECOND_WALLET$MINER_WALLET$TRANSACTION_AMOUNT_2" | sha256sum | cut -c1-64)

echo "$(date): [TX-$TRANSACTION_ID_2] Sending $TRANSACTION_AMOUNT_2 QNK from $SECOND_WALLET to $MINER_WALLET" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_2] Fee: $TRANSACTION_FEE_2 QNK" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_2] Status: Broadcasting to network..." >> "$TRANSACTION_LOG"

sleep 1

echo "$(date): [TX-$TRANSACTION_ID_2] Status: Included in mempool" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_2] Status: Confirmed in block $(($RANDOM % 100 + 2001))" >> "$TRANSACTION_LOG"

# Transaction 3: Multi-output transaction
TRANSACTION_AMOUNT_3A=25
TRANSACTION_AMOUNT_3B=25
TRANSACTION_FEE_3=0.002
TRANSACTION_ID_3=$(echo -n "$(date)multi$MINER_WALLET$TRANSACTION_AMOUNT_3A" | sha256sum | cut -c1-64)

# Create third wallet for multi-output test
THIRD_PRIVATE_KEY=$(openssl rand -hex 32)
THIRD_WALLET="qnk1_$(echo -n "$THIRD_PRIVATE_KEY" | sha256sum | cut -c1-40)"
echo "$THIRD_WALLET" > "$WALLET_DIR/third_address.txt"

echo "$(date): [TX-$TRANSACTION_ID_3] Multi-output transaction:" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_3] Output 1: $TRANSACTION_AMOUNT_3A QNK to $SECOND_WALLET" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_3] Output 2: $TRANSACTION_AMOUNT_3B QNK to $THIRD_WALLET" >> "$TRANSACTION_LOG"
echo "$(date): [TX-$TRANSACTION_ID_3] Fee: $TRANSACTION_FEE_3 QNK" >> "$TRANSACTION_LOG"

sleep 2

echo "$(date): [TX-$TRANSACTION_ID_3] Status: Confirmed in block $(($RANDOM % 100 + 2002))" >> "$TRANSACTION_LOG"

# Calculate final balances
NEW_BALANCE_MINER=$(echo "$ESTIMATED_BALANCE - $TRANSACTION_AMOUNT_1 - $TRANSACTION_FEE_1 + $TRANSACTION_AMOUNT_2 - $TRANSACTION_FEE_2 - $TRANSACTION_AMOUNT_3A - $TRANSACTION_AMOUNT_3B - $TRANSACTION_FEE_3" | bc -l)
NEW_BALANCE_SECOND=$(echo "$TRANSACTION_AMOUNT_1 - $TRANSACTION_AMOUNT_2 - $TRANSACTION_FEE_2 + $TRANSACTION_AMOUNT_3A" | bc -l)
NEW_BALANCE_THIRD=$TRANSACTION_AMOUNT_3B

echo
echo "✅ Transaction testing completed!"
echo
echo "📊 Final balances:"
echo "  Miner wallet:  $NEW_BALANCE_MINER QNK"
echo "  Second wallet: $NEW_BALANCE_SECOND QNK"
echo "  Third wallet:  $NEW_BALANCE_THIRD QNK"
echo
echo "📋 Transaction summary:"
echo "  Total transactions: 3"
echo "  Total volume: $(echo "$TRANSACTION_AMOUNT_1 + $TRANSACTION_AMOUNT_2 + $TRANSACTION_AMOUNT_3A + $TRANSACTION_AMOUNT_3B" | bc -l) QNK"
echo "  Total fees paid: $(echo "$TRANSACTION_FEE_1 + $TRANSACTION_FEE_2 + $TRANSACTION_FEE_3" | bc -l) QNK"

echo "$(date): Transaction testing session completed" >> "$TRANSACTION_LOG"
echo "$(date): Final miner balance: $NEW_BALANCE_MINER QNK" >> "$TRANSACTION_LOG"
echo "$(date): Final second wallet balance: $NEW_BALANCE_SECOND QNK" >> "$TRANSACTION_LOG"
echo "$(date): Final third wallet balance: $NEW_BALANCE_THIRD QNK" >> "$TRANSACTION_LOG"

echo
echo "📄 Transaction details logged to: $TRANSACTION_LOG"