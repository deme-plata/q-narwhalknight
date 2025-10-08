#!/bin/bash

echo "🔍 Enhanced Debug Test - DNS Phantom Bridge Verification"
echo "======================================================"

# Clean up any existing containers
docker ps -a | grep q-narwhal | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true

# Create isolated network
docker network rm qnk-debug-net 2>/dev/null || true
docker network create qnk-debug-net

echo "🚀 Starting 3 nodes with enhanced debugging..."

# Node 1
docker run -d --name qnk-debug-node1 \
  --network qnk-debug-net \
  -e QNK_NODE_NAME="debug-node-1" \
  -e QNK_API_PORT="8080" \
  -e RUST_LOG="debug,q_dns_phantom=trace,q_network=debug,q_tor_client=debug,q_narwhal_core=debug" \
  q-narwhalknight-debug:latest

# Node 2  
docker run -d --name qnk-debug-node2 \
  --network qnk-debug-net \
  -e QNK_NODE_NAME="debug-node-2" \
  -e QNK_API_PORT="8080" \
  -e RUST_LOG="debug,q_dns_phantom=trace,q_network=debug,q_tor_client=debug,q_narwhal_core=debug" \
  q-narwhalknight-debug:latest

# Node 3
docker run -d --name qnk-debug-node3 \
  --network qnk-debug-net \
  -e QNK_NODE_NAME="debug-node-3" \
  -e QNK_API_PORT="8080" \
  -e RUST_LOG="debug,q_dns_phantom=trace,q_network=debug,q_tor_client=debug,q_narwhal_core=debug" \
  q-narwhalknight-debug:latest

echo "⏳ Waiting 15 seconds for nodes to initialize..."
sleep 15

echo ""
echo "📋 NODE STATUS:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🔍 ENHANCED DEBUG LOGS - Node 1:"
echo "================================="
docker logs qnk-debug-node1 2>&1 | tail -30

echo ""
echo "🔍 ENHANCED DEBUG LOGS - Node 2:" 
echo "================================="
docker logs qnk-debug-node2 2>&1 | tail -30

echo ""
echo "🔍 ENHANCED DEBUG LOGS - Node 3:"
echo "================================="
docker logs qnk-debug-node3 2>&1 | tail -30

echo ""
echo "🔬 ANALYZING DNS-PHANTOM BRIDGE FUNCTIONALITY:"
echo "=============================================="

# Check for specific debug patterns that should indicate working bridge
echo "🔍 Searching for DNS-phantom discovery events..."
docker logs qnk-debug-node1 2>&1 | grep -i "phantom.*peer\|dns.*phantom\|steganographic\|bridge.*phantom" | head -10 || echo "❌ No DNS-phantom discovery events found"

echo ""
echo "🔍 Searching for P2P connection attempts..."
docker logs qnk-debug-node1 2>&1 | grep -i "p2p.*connection\|libp2p\|network.*manager\|peer.*connect" | head -10 || echo "❌ No P2P connection attempts found"

echo ""
echo "🔍 Searching for Tor connectivity details..."
docker logs qnk-debug-node1 2>&1 | grep -i "tor.*connect\|socks.*proxy\|onion.*address\|hidden.*service" | head -10 || echo "❌ No Tor connectivity details found"

echo ""
echo "✅ Test completed. Check above for enhanced debugging evidence."