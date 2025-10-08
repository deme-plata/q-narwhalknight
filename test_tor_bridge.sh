#!/bin/bash

echo "🔧 Q-NarwhalKnight DNS-Phantom Bridge Functionality Test"
echo "========================================================="

# Clean up
docker ps -a | grep qnk-tor-test | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true
docker network rm qnk-tor-test-net 2>/dev/null || true

# Create test network
docker network create qnk-tor-test-net

echo "📦 Starting test container with Tor..."
docker run -d --name qnk-tor-test \
  --network qnk-tor-test-net \
  -e RUST_LOG="debug" \
  q-narwhalknight-debug:latest

echo "⏳ Waiting 45 seconds for full Tor bootstrap..."
sleep 45

echo ""
echo "🔍 ANALYZING TOR BOOTSTRAP STATUS:"
echo "=================================="

# Check Tor process
echo "📊 Tor process status:"
docker exec qnk-tor-test ps aux | grep tor | head -3

echo ""
echo "🌐 Tor bootstrap logs:"
docker exec qnk-tor-test tail -20 /app/logs/tor.log 2>/dev/null || echo "❌ No Tor logs found"

echo ""
echo "🧅 Hidden service status:"
docker exec qnk-tor-test ls -la /app/data/tor/hidden_service/ 2>/dev/null || echo "❌ No hidden service directory"
docker exec qnk-tor-test cat /app/data/tor/hidden_service/hostname 2>/dev/null || echo "❌ No hostname file"

echo ""
echo "🔌 SOCKS proxy connectivity test:"
docker exec qnk-tor-test netstat -tuln | grep 9050 || echo "❌ SOCKS port 9050 not listening"

echo ""
echo "🧪 Manual SOCKS connection test:"
docker exec qnk-tor-test timeout 10 bash -c 'echo -e "GET / HTTP/1.0\r\n\r\n" | nc -X 5 -x 127.0.0.1:9050 check.torproject.org 80' 2>/dev/null | head -5 || echo "❌ SOCKS proxy connection failed"

echo ""
echo "📜 Q-NarwhalKnight application logs (last 30 lines):"
echo "==================================================="
docker exec qnk-tor-test tail -30 /app/logs/qnk-stdout.log | grep -E "(Tor|phantom|NetworkManager|bridge)" || echo "❌ No relevant logs found"

echo ""
echo "❓ DNS-PHANTOM BRIDGE STATUS ANALYSIS:"
echo "======================================"

# Search for specific bridge initialization messages
BRIDGE_INIT=$(docker exec qnk-tor-test grep -c "NetworkManager initialized - DNS-phantom bridge ready" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
PHANTOM_DISCOVERY=$(docker exec qnk-tor-test grep -c "Phantom peer discovered" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")
P2P_ATTEMPTS=$(docker exec qnk-tor-test grep -c "Attempting P2P connection to phantom peer" /app/logs/qnk-stdout.log 2>/dev/null || echo "0")

echo "🔗 NetworkManager Bridge Initialized: $BRIDGE_INIT times"
echo "👻 Phantom Peers Discovered: $PHANTOM_DISCOVERY times"
echo "🌐 P2P Connection Attempts: $P2P_ATTEMPTS times"

if [ "$BRIDGE_INIT" -gt "0" ]; then
    echo "✅ DNS-phantom bridge is initialized"
else
    echo "❌ DNS-phantom bridge failed to initialize"
fi

if [ "$PHANTOM_DISCOVERY" -gt "0" ] || [ "$P2P_ATTEMPTS" -gt "0" ]; then
    echo "✅ DNS-phantom to P2P bridge is attempting connections"
else
    echo "❌ DNS-phantom to P2P bridge is NOT working"
fi

echo ""
echo "🧹 Cleaning up..."
docker rm -f qnk-tor-test
docker network rm qnk-tor-test-net

echo "✅ Test completed!"