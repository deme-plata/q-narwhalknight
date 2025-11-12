#!/usr/bin/env bash

# Q-NarwhalKnight Miner Configuration Script for HiveOS

# Get configuration from HiveOS
[[ -z $CUSTOM_TEMPLATE ]] && echo -e "${YELLOW}CUSTOM_TEMPLATE is empty${NOCOLOR}" && return 1
[[ -z $CUSTOM_URL ]] && echo -e "${YELLOW}CUSTOM_URL is empty${NOCOLOR}" && return 1

# Parse HiveOS configuration
conf="-t $CUSTOM_TEMPLATE"

# Extract wallet address from CUSTOM_TEMPLATE (should be qnk address)
WALLET=$CUSTOM_TEMPLATE

# Extract node URL from CUSTOM_URL (bootstrap node endpoint)
NODE_URL=$CUSTOM_URL

# Number of threads (default to CPU count)
if [[ ! -z $CUSTOM_USER_CONFIG ]]; then
    THREADS=$(echo $CUSTOM_USER_CONFIG | jq -r '.threads // empty')
fi

# Default to all CPU cores if not specified
if [[ -z $THREADS ]]; then
    THREADS=$(nproc)
fi

# Generate miner configuration
cat > $MINER_DIR/$MINER_VER/miner.conf <<MINER_CONF
{
  "wallet_address": "$WALLET",
  "node_url": "$NODE_URL",
  "threads": $THREADS,
  "log_level": "info"
}
MINER_CONF

echo "Q-NarwhalKnight miner configured:"
echo "  Wallet: $WALLET"
echo "  Node: $NODE_URL"
echo "  Threads: $THREADS"
