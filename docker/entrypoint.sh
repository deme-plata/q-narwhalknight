#!/bin/bash
set -euo pipefail

echo "🚀 Q-NarwhalKnight Node Starting"
echo "================================"
echo "Node ID: ${Q_NODE_ID:-unknown}"
echo "Server Role: ${Q_SERVER_ROLE:-validator}"
echo "Timestamp: $(date)"

# Wait for dependencies to be ready
if [ "${Q_SERVER_ROLE:-validator}" != "tor-proxy" ] && [ "${Q_SERVER_ROLE:-validator}" != "dns_phantom_hub" ]; then
    echo "⏳ Waiting for Tor proxy to be ready..."
    while ! nc -z tor-proxy 9050; do
        sleep 2
    done
    echo "✅ Tor proxy is ready"

    echo "⏳ Waiting for DNS-Phantom hub to be ready..."
    while ! nc -z dns-phantom-hub 8080; do
        sleep 2
    done
    echo "✅ DNS-Phantom hub is ready"
fi

# Additional startup delay for coordination
echo "⏳ Startup coordination delay (5s)..."
sleep 5

# Set up logging
export RUST_LOG="${RUST_LOG:-info}"
export RUST_BACKTRACE=1

# Create log directory
mkdir -p /app/logs

# Execute the appropriate binary based on arguments
if [ $# -eq 0 ]; then
    echo "❌ No command specified"
    echo "Available commands:"
    echo "  /app/bin/q-api-server [options]"
    echo "  /app/bin/massive_scale_test [options]"
    echo "  /app/bin/real_test [options]"
    exit 1
fi

# Log startup information
{
    echo "=== Q-NarwhalKnight Node Startup ==="
    echo "Timestamp: $(date)"
    echo "Node ID: ${Q_NODE_ID:-unknown}"
    echo "Server Role: ${Q_SERVER_ROLE:-validator}"
    echo "Command: $*"
    echo "Environment:"
    env | grep ^Q_ | sort
    echo "=================================="
} >> /app/logs/startup.log

echo "🎯 Executing: $*"
exec "$@" 2>&1 | tee -a "/app/logs/${Q_NODE_ID:-node}.log"