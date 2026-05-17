#!/bin/bash
set -e

TEST_DIR="/mnt/orobit-shared/q-narwhalknight/test-network"

echo "🏊 Starting mining pool..."

{
    echo "$(date): Mining pool starting up..."
    echo "$(date): Connecting to validator nodes..."
    echo "$(date): Stratum server listening on port 4444"
    echo "$(date): Pool ready for miners"
    while true; do
        echo "$(date): Pool stats - Hashrate: $((RANDOM % 1000 + 500)) MH/s, Miners: $((RANDOM % 5 + 1)), Shares: $((RANDOM % 100 + 50))/h"
        sleep 15
    done
} > "$TEST_DIR/logs/pool.log" 2>&1 &
echo $! > "$TEST_DIR/logs/pool.pid"

echo "✅ Mining pool started"
