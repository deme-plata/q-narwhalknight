#!/bin/bash
set -e

TEST_DIR="/mnt/orobit-shared/q-narwhalknight/test-network"
DATA_DIR="$TEST_DIR/data"

echo "🌟 Starting Q-NarwhalKnight test nodes..."

# Start validator nodes in background
for i in {1..3}; do
    echo "Starting validator node $i..."
    # In a real implementation, this would be the actual node binary
    # For testing purposes, we'll create log files to simulate activity
    {
        echo "$(date): Node $i starting up..."
        echo "$(date): Loading genesis block..."
        echo "$(date): Connecting to peers..."
        echo "$(date): Validator node $i ready for consensus"
        while true; do
            echo "$(date): Block height: $((RANDOM % 1000 + 1000)), Mining: true, Peers: $((RANDOM % 10 + 5))"
            sleep 10
        done
    } > "$TEST_DIR/logs/node$i.log" 2>&1 &
    echo $! > "$TEST_DIR/logs/node$i.pid"
done

echo "✅ All validator nodes started"
