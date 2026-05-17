#!/usr/bin/env bash

# Q-NarwhalKnight Miner Run Script for HiveOS

cd $(dirname $0)

# Load configuration
[[ ! -f miner.conf ]] && echo "No miner.conf found" && exit 1

# Parse configuration
WALLET=$(jq -r '.wallet_address' miner.conf)
NODE_URL=$(jq -r '.node_url' miner.conf)
THREADS=$(jq -r '.threads' miner.conf)

# Launch miner
./q-miner \
  --wallet "$WALLET" \
  --node "$NODE_URL" \
  --threads "$THREADS" \
  2>&1 | tee --append $MINER_LOG_BASENAME.log
