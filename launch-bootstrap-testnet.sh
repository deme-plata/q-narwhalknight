#!/bin/bash
# Launch Q-NarwhalKnight Bootstrap Node for Testnet
# This node will run on the hardcoded bootstrap port (9001) for libp2p P2P

set -e

echo "🚀 Launching Q-NarwhalKnight Testnet Bootstrap Node"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  • HTTP API Port: 18080"
echo "  • libp2p P2P Port: 9001 (fixed, for bootstrap)"
echo "  • Database: ./data-bootstrap-testnet"
echo "  • Node ID: bootstrap-testnet"
echo "  • Network: Q-NarwhalKnight Testnet"
echo ""
echo "This bootstrap node will be accessible at:"
echo "  • API: http://185.182.185.227:18080"
echo "  • P2P: /ip4/185.182.185.227/tcp/9001/p2p/<PEER_ID>"
echo ""
echo "Other nodes should add this to bootstrap_peers in their config"
echo ""
echo "=================================================="
echo ""

# Set environment variables for fixed libp2p port
export Q_LIBP2P_PORT=9001
export Q_DB_PATH=./data-bootstrap-testnet

# Kill any existing bootstrap node on this port
pkill -f "q-api-server.*18080.*bootstrap" || true
sleep 2

# Launch bootstrap node
exec /opt/orobit/shared/q-narwhalknight/target/release/q-api-server \
  --port 18080 \
  --node-id bootstrap-testnet 2>&1 | tee bootstrap-testnet.log
