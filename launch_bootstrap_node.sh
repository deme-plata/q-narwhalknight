#!/bin/bash
# Launch bootstrap node with proper libp2p configuration

export Q_DB_PATH=./data-bootstrap
export Q_P2P_PORT=6881
export Q_NARWHAL_BOOTSTRAP_NODE=185.182.185.227:6881

echo "🚀 Launching Q-NarwhalKnight Bootstrap Node"
echo "Environment variables set:"
echo "  Q_P2P_PORT=$Q_P2P_PORT"
echo "  Q_DB_PATH=$Q_DB_PATH"
echo ""
echo "This node will listen on:"
echo "  API: 8080"
echo "  librqbit DHT: 6881"
echo "  libp2p: 6981 (Q_P2P_PORT + 100)"
echo "  P2P listener: 8081"
echo ""
echo "Other nodes should set:"
echo "  Q_NARWHAL_BOOTSTRAP_NODE=185.182.185.227:6881"
echo "  Q_BOOTSTRAP_PEERS=185.182.185.227:6881"
echo ""
echo "Starting bootstrap node..."
echo ""

exec /opt/orobit/shared/q-narwhalknight/target/x86_64-unknown-linux-gnu/release/q-api-server \
  --port 8080 \
  --node-id bootstrap-node \
  --production