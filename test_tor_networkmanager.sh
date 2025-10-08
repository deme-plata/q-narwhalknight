#!/bin/bash
# Test NetworkManager with Tor Integration

echo "================================================================================"
echo "🧅 Q-NarwhalKnight Tor + NetworkManager Integration Test"
echo "================================================================================"
echo ""

# Verify Tor is running
echo "📋 Pre-flight Checks:"
echo "  1. Checking Tor daemon..."
if ss -tlnp | grep -q 9150; then
    echo "     ✅ Tor SOCKS running on port 9150"
else
    echo "     ❌ Tor SOCKS not found on port 9150"
    echo "     Starting Tor..."
    systemctl start tor@default
    sleep 3
fi

# Kill existing nodes
echo ""
echo "  2. Stopping existing nodes..."
pkill -9 q-api-server 2>/dev/null
sleep 2

# Create test data directory
echo ""
echo "  3. Creating test directory..."
rm -rf ./data-tor-networkmanager
mkdir -p ./data-tor-networkmanager

echo ""
echo "================================================================================"
echo "🚀 Launching Node with Tor-Enabled NetworkManager"
echo "================================================================================"
echo ""

# Launch node with Tor debug logging
export Q_DB_PATH=./data-tor-networkmanager
export Q_P2P_PORT=9210
export RUST_LOG=info,q_tor_client=debug,q_network=debug,q_api_server=info

echo "Configuration:"
echo "  Database: $Q_DB_PATH"
echo "  P2P Port: $Q_P2P_PORT"
echo "  HTTP Port: 9110"
echo "  Log Level: $RUST_LOG"
echo ""

# Run in background and capture output
./target/x86_64-unknown-linux-gnu/release/q-api-server --port 9110 \
  > tor-networkmanager-test.log 2>&1 &

NODE_PID=$!
echo "Node PID: $NODE_PID"
echo ""

# Wait for initialization
echo "⏳ Waiting 10 seconds for initialization..."
for i in {1..10}; do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Check logs for Tor initialization
echo "================================================================================"
echo "📊 Checking Tor Integration Status..."
echo "================================================================================"
echo ""

# Check for Tor client initialization
if grep -q "🧅 Initializing Q-Tor-Client" tor-networkmanager-test.log; then
    echo "✅ Tor client initialization started"

    # Check for successful connection
    if grep -q "✅.*Tor.*successful\|Tor.*connected\|SOCKS.*connection.*established" tor-networkmanager-test.log; then
        echo "✅ Tor SOCKS connection established"
    elif grep -q "Failed to connect to Tor\|Tor connection.*failed\|Invalid response version" tor-networkmanager-test.log; then
        echo "❌ Tor connection failed"
        echo ""
        echo "Error details:"
        grep -i "tor.*error\|failed.*tor\|invalid response" tor-networkmanager-test.log | head -5 | sed 's/^/  /'
    else
        echo "⚠️  Tor connection status unclear"
    fi
else
    echo "❌ Tor client not initialized"
fi

echo ""

# Check NetworkManager status
if grep -q "✅ NetworkManager initialized" tor-networkmanager-test.log; then
    echo "✅ NetworkManager successfully initialized"
elif grep -q "⚠️ NetworkManager initialization failed" tor-networkmanager-test.log; then
    echo "❌ NetworkManager initialization failed"
    echo ""
    echo "Error details:"
    grep -A 2 "NetworkManager initialization failed" tor-networkmanager-test.log | sed 's/^/  /'
else
    echo "⚠️  NetworkManager status unknown"
fi

echo ""

# Check Tor Integration flag in status
echo "Checking Tor Integration status display:"
if grep -q "🧅 Tor Integration: ✅" tor-networkmanager-test.log; then
    echo "  ✅ Tor Integration: ENABLED"
elif grep -q "🧅 Tor Integration: ❌" tor-networkmanager-test.log; then
    echo "  ❌ Tor Integration: DISABLED"
else
    echo "  ⚠️  Status not found in logs"
fi

echo ""
echo "================================================================================"
echo "📝 Full Initialization Log:"
echo "================================================================================"
echo ""
head -100 tor-networkmanager-test.log | grep -E "Tor|NetworkManager|🧅|🌐" | sed 's/^/  /'

echo ""
echo "================================================================================"
echo "📊 Test Summary"
echo "================================================================================"
echo ""
echo "Node Status:"
if ps -p $NODE_PID > /dev/null 2>&1; then
    echo "  ✅ Node running (PID: $NODE_PID)"
else
    echo "  ❌ Node crashed"
fi

echo ""
echo "Logs available at:"
echo "  tail -f tor-networkmanager-test.log"
echo ""
echo "Stop node:"
echo "  kill $NODE_PID"
echo ""
echo "================================================================================"
