#!/bin/bash
# Launch remote node with proper environment variables

export Q_DB_PATH=./data-remote-node
export Q_P2P_PORT=9002
export Q_NARWHAL_BOOTSTRAP_NODE=185.182.185.227:6881
export Q_BOOTSTRAP_PEERS=185.182.185.227:6881

echo "🚀 Launching Q-NarwhalKnight Remote Node"
echo "Environment variables set:"
echo "  Q_P2P_PORT=$Q_P2P_PORT"
echo "  Q_NARWHAL_BOOTSTRAP_NODE=$Q_NARWHAL_BOOTSTRAP_NODE"
echo "  Q_BOOTSTRAP_PEERS=$Q_BOOTSTRAP_PEERS"
echo "  Q_DB_PATH=$Q_DB_PATH"
echo ""
echo "Expected ports:"
echo "  API: 8090"
echo "  librqbit DHT: 9002"
echo "  libp2p: 9102 (Q_P2P_PORT + 100)"
echo "  P2P listener: 8091"
echo ""
echo "Starting server..."
echo ""

exec /mnt/orobit-shared/q-narwhalknight/target/x86_64-unknown-linux-gnu/release/q-api-server \
  --port 8090 \
  --node-id remote-node \
  --production