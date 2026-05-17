#!/bin/bash

# Q-NarwhalKnight Testnet Node Connection Script
# Manually connects 4 local testnet nodes via libp2p

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   Q-NarwhalKnight Testnet Node P2P Connection Script        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Node Information
# Node 1: Peer ID 12D3KooWQb2gzpDFN4VtMY9FY9ShdsrWwYSDtrRFJdhLUdFBPrpC, P2P: 127.0.0.1:33305
# Node 2: Peer ID 12D3KooWLGbDRyWaRmQyfU9pMxeDAbd4kKnCsvkrh2yWKhZuvrwv, P2P: 127.0.0.1:35999
# Node 3: Peer ID 12D3KooWJqZBydAwrwDNHkzNSmxj3DpTyQWUy5DXgZ9sbkV5tCVm, P2P: 127.0.0.1:43681
# Node 4: Peer ID 12D3KooWCSAfm4whSBWaYB8VkctaRudxKspX4NnmhwXsPZEGYobn, P2P: 127.0.0.1:38879

echo "🔗 Connecting Node 1 to Node 2..."
curl -s -X POST http://localhost:8080/api/v1/network/peers/connect \
  -H "Content-Type: application/json" \
  -d '{"multiaddr": "/ip4/127.0.0.1/tcp/35999/p2p/12D3KooWLGbDRyWaRmQyfU9pMxeDAbd4kKnCsvkrh2yWKhZuvrwv"}' | jq '.success, .message // .error'

echo ""
echo "🔗 Connecting Node 1 to Node 3..."
curl -s -X POST http://localhost:8080/api/v1/network/peers/connect \
  -H "Content-Type: application/json" \
  -d '{"multiaddr": "/ip4/127.0.0.1/tcp/43681/p2p/12D3KooWJqZBydAwrwDNHkzNSmxj3DpTyQWUy5DXgZ9sbkV5tCVm"}' | jq '.success, .message // .error'

echo ""
echo "🔗 Connecting Node 1 to Node 4..."
curl -s -X POST http://localhost:8080/api/v1/network/peers/connect \
  -H "Content-Type: application/json" \
  -d '{"multiaddr": "/ip4/127.0.0.1/tcp/38879/p2p/12D3KooWCSAfm4whSBWaYB8VkctaRudxKspX4NnmhwXsPZEGYobn"}' | jq '.success, .message // .error'

echo ""
echo "⏳ Waiting 3 seconds for connections to establish..."
sleep 3

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 Checking Peer Counts"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for port in 8080 8084 9060 9666; do
    echo "Node on port $port:"
    curl -s "http://localhost:$port/api/v1/status" | jq '{node_id: .data.node_id, connected_peers: .data.connected_peers}' 2>/dev/null || echo "  ⚠ Not responding"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo "✅ Connection script complete!"
echo ""
echo "Expected Result:"
echo "  Node 1: connected_peers: 3"
echo "  Node 2: connected_peers: 1 (or more via gossip)"
echo "  Node 3: connected_peers: 1 (or more via gossip)"
echo "  Node 4: connected_peers: 1 (or more via gossip)"
echo ""
echo "Next: Run transaction propagation test"
echo "  cd test_tx_propagation && ./target/release/test_tx_propagation"
echo "═══════════════════════════════════════════════════════════════"
