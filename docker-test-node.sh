#!/bin/bash
# Launch Q-NarwhalKnight test node in Docker for distributed AI testing

set -e

echo "🐳 Q-NarwhalKnight Test Node Launcher"
echo "======================================"
echo ""

# Create persistent data directory
mkdir -p docker-test-data

# Remove old container if it exists
docker rm -f q-test-node 2>/dev/null || true

echo "📦 Starting test node in Docker..."
echo "  API Port: 8090"
echo "  P2P Port: 9002"
echo "  Node ID: test-docker-node"
echo ""

# Launch the test node
docker run -d \
  --name q-test-node \
  --network host \
  -v "$(pwd)/docker-test-data:/app/data" \
  -e Q_DB_PATH=/app/data \
  -e Q_P2P_PORT=9002 \
  -e RUST_LOG=info,q_api_server=debug,q_network=debug,q_network::distributed_ai=debug \
  q-narwhalknight-test:v0.2.0

echo "✅ Test node started!"
echo ""
echo "📊 Node Status:"
docker ps | grep q-test-node

echo ""
echo "📋 View logs with: docker logs -f q-test-node"
echo "🛑 Stop node with: docker stop q-test-node"
echo "🗑️  Remove node with: docker rm q-test-node"
echo ""
echo "🌐 Test API endpoint: curl http://localhost:8090/api/stats"
