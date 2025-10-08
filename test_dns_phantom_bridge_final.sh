#!/bin/bash

echo "🔧 FINAL DNS-PHANTOM TO P2P BRIDGE TEST"
echo "======================================="
echo "Testing complete end-to-end functionality with fixes applied"
echo ""

# Clean up any existing containers
docker ps -a | grep qnk-bridge-test | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
docker network rm qnk-bridge-test-net 2>/dev/null || true

# Create test network
docker network create qnk-bridge-test-net

echo "🚀 Starting 3 test nodes with fixed Tor configuration..."

# Node 1 - Primary test node
docker run -d --name qnk-bridge-test1 \
  --network qnk-bridge-test-net \
  -e QNK_NODE_NAME="bridge-test-1" \
  -e QNK_API_PORT="8080" \
  -e RUST_LOG="debug,q_dns_phantom=trace,q_network=debug,q_tor_client=debug" \
  q-narwhalknight-fixed:latest

# Node 2 - Discovery target
docker run -d --name qnk-bridge-test2 \
  --network qnk-bridge-test-net \
  -e QNK_NODE_NAME="bridge-test-2" \
  -e QNK_API_PORT="8080" \
  -e RUST_LOG="debug,q_dns_phantom=trace,q_network=debug,q_tor_client=debug" \
  q-narwhalknight-fixed:latest

# Node 3 - Additional peer
docker run -d --name qnk-bridge-test3 \
  --network qnk-bridge-test-net \
  -e QNK_NODE_NAME="bridge-test-3" \
  -e QNK_API_PORT="8080" \
  -e RUST_LOG="debug,q_dns_phantom=trace,q_network=debug,q_tor_client=debug" \
  q-narwhalknight-fixed:latest

echo "⏳ Waiting 60 seconds for full initialization and Tor bootstrap..."
sleep 60

echo ""
echo "🔍 COMPREHENSIVE BRIDGE FUNCTIONALITY ANALYSIS"
echo "=============================================="

echo ""
echo "📊 NODE STATUSES:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep qnk-bridge-test

echo ""
echo "🧅 TOR CONNECTIVITY TEST (Node 1):"
echo "=================================="
echo "🔌 SOCKS proxy status:"
docker exec qnk-bridge-test1 netstat -tuln | grep 9050 && echo "✅ SOCKS port 9050 is listening" || echo "❌ SOCKS port 9050 not listening"

echo ""
echo "🧪 Manual SOCKS connectivity test:"
timeout 10 docker exec qnk-bridge-test1 bash -c 'echo -e "GET / HTTP/1.0\r\n\r\n" | nc -X 5 -x 127.0.0.1:9050 check.torproject.org 80 2>/dev/null' | head -3 && echo "✅ Tor SOCKS proxy working" || echo "❌ Tor SOCKS proxy failed"

echo ""
echo "🧅 Hidden service address:"
docker exec qnk-bridge-test1 cat /app/data/tor/hidden_service/hostname 2>/dev/null || echo "❌ No hidden service generated"

echo ""
echo "🔍 DNS-PHANTOM BRIDGE INITIALIZATION STATUS:"
echo "==========================================="

# Check all 3 nodes for bridge initialization
for i in 1 2 3; do
    echo ""
    echo "📡 NODE $i BRIDGE STATUS:"
    
    BRIDGE_INIT=$(docker exec qnk-bridge-test$i grep -c "NetworkManager initialized - DNS-phantom bridge ready" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    TOR_SUCCESS=$(docker exec qnk-bridge-test$i grep -c "Tor client initialized successfully" /app/logs/qnk-stdout.log 2>/dev/null || echo "0") 
    PHANTOM_BROADCAST=$(docker exec qnk-bridge-test$i grep -c "Broadcasted peer advertisement through DNS phantom network" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    
    echo "  🔗 NetworkManager Bridge: $BRIDGE_INIT"
    echo "  🧅 Tor Client Success: $TOR_SUCCESS" 
    echo "  👻 Phantom Broadcasts: $PHANTOM_BROADCAST"
    
    if [ "$BRIDGE_INIT" -gt "0" ]; then
        echo "  ✅ Bridge initialized successfully"
    else
        echo "  ❌ Bridge initialization failed"
    fi
done

echo ""
echo "🌐 PEER DISCOVERY & CONNECTION ANALYSIS:"
echo "======================================="

# Check for cross-node discovery
TOTAL_DISCOVERIES=0
TOTAL_CONNECTIONS=0

for i in 1 2 3; do
    DISCOVERIES=$(docker exec qnk-bridge-test$i grep -c "Phantom peer discovered" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    CONNECTIONS=$(docker exec qnk-bridge-test$i grep -c "Attempting P2P connection to phantom peer" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    ESTABLISHED=$(docker exec qnk-bridge-test$i grep -c "P2P connection established to phantom peer" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
    
    echo "📡 Node $i:"
    echo "  👻 Peer Discoveries: $DISCOVERIES"
    echo "  🔗 Connection Attempts: $CONNECTIONS"
    echo "  ✅ Established Connections: $ESTABLISHED"
    
    TOTAL_DISCOVERIES=$((TOTAL_DISCOVERIES + DISCOVERIES))
    TOTAL_CONNECTIONS=$((TOTAL_CONNECTIONS + CONNECTIONS))
done

echo ""
echo "🎯 FINAL RESULTS SUMMARY:"
echo "========================"

if [ "$TOTAL_DISCOVERIES" -gt "0" ]; then
    echo "✅ DNS-PHANTOM PEER DISCOVERY: WORKING ($TOTAL_DISCOVERIES total discoveries)"
else
    echo "❌ DNS-PHANTOM PEER DISCOVERY: NOT WORKING (0 discoveries)"
fi

if [ "$TOTAL_CONNECTIONS" -gt "0" ]; then
    echo "✅ DNS-PHANTOM TO P2P BRIDGE: WORKING ($TOTAL_CONNECTIONS connection attempts)"
else
    echo "❌ DNS-PHANTOM TO P2P BRIDGE: NOT WORKING (0 connection attempts)"
fi

echo ""
echo "🔬 DETAILED ERROR ANALYSIS (Node 1):"
echo "===================================="
echo "Recent error logs:"
docker exec qnk-bridge-test1 grep -i "error\|failed\|warn" /app/logs/qnk-stdout.log | tail -10 || echo "No errors found"

echo ""
echo "🧹 Cleaning up test nodes..."
docker rm -f qnk-bridge-test1 qnk-bridge-test2 qnk-bridge-test3
docker network rm qnk-bridge-test-net

echo ""
echo "🏁 TEST COMPLETED - Check results above for bridge functionality status"