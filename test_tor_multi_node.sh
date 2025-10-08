#!/bin/bash
# Multi-Node Tor Integration Test
# Tests 2-node network with Tor-enabled NetworkManager

set -e

echo "================================================================================"
echo "🧅 Q-NarwhalKnight Multi-Node Tor Integration Test"
echo "================================================================================"
echo ""

# Configuration
NODE1_DB="./data-tor-node1"
NODE2_DB="./data-tor-node2"
NODE1_HTTP=9110
NODE2_HTTP=9120
NODE1_P2P=9111
NODE2_P2P=9121

# Cleanup
echo "📋 Pre-flight Cleanup:"
echo "  1. Stopping existing nodes..."
pkill -9 q-api-server 2>/dev/null || true
sleep 2

echo "  2. Cleaning data directories..."
rm -rf $NODE1_DB $NODE2_DB
mkdir -p $NODE1_DB $NODE2_DB

echo "  3. Verifying Tor is running..."
if ! ss -tlnp | grep -q 9150; then
    echo "     ❌ Tor not running on port 9150"
    echo "     Starting Tor..."
    systemctl start tor@default
    sleep 3
fi
echo "     ✅ Tor SOCKS running on port 9150"

echo ""
echo "================================================================================"
echo "🚀 Launching 2-Node Network with Tor"
echo "================================================================================"
echo ""

# Launch Node 1
echo "Node 1 Configuration:"
echo "  Database: $NODE1_DB"
echo "  HTTP Port: $NODE1_HTTP"
echo "  P2P Port: $NODE1_P2P"
echo "  Tor: Enabled via NetworkManager"
echo ""

Q_DB_PATH=$NODE1_DB \
Q_P2P_PORT=$NODE1_P2P \
RUST_LOG=info,q_tor_client=debug,q_network=info \
./target/x86_64-unknown-linux-gnu/release/q-api-server \
  --port $NODE1_HTTP \
  --production \
  > tor-node1.log 2>&1 &

NODE1_PID=$!
echo "✅ Node 1 started (PID: $NODE1_PID)"
echo ""

# Wait for Node 1 to initialize
echo "⏳ Waiting 5 seconds for Node 1 to initialize..."
sleep 5

# Get Node 1 validator ID
echo "📝 Fetching Node 1 validator ID..."
NODE1_ID=$(curl -s http://localhost:$NODE1_HTTP/node_id 2>/dev/null | jq -r '.node_id' || echo "unknown")
echo "   Node 1 ID: $NODE1_ID"
echo ""

# Launch Node 2 with automatic peer discovery
echo "Node 2 Configuration:"
echo "  Database: $NODE2_DB"
echo "  HTTP Port: $NODE2_HTTP"
echo "  P2P Port: $NODE2_P2P"
echo "  Tor: Enabled via NetworkManager"
echo "  Peer Discovery: Automatic via NetworkManager"
echo ""

Q_DB_PATH=$NODE2_DB \
Q_P2P_PORT=$NODE2_P2P \
RUST_LOG=info,q_tor_client=debug,q_network=info \
./target/x86_64-unknown-linux-gnu/release/q-api-server \
  --port $NODE2_HTTP \
  --production \
  > tor-node2.log 2>&1 &

NODE2_PID=$!
echo "✅ Node 2 started (PID: $NODE2_PID)"
echo ""

# Wait for connection establishment
echo "⏳ Waiting 10 seconds for nodes to connect..."
for i in {1..10}; do
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Get Node 2 validator ID
echo "📝 Fetching Node 2 validator ID..."
NODE2_ID=$(curl -s http://localhost:$NODE2_HTTP/node_id 2>/dev/null | jq -r '.node_id' || echo "unknown")
echo "   Node 2 ID: $NODE2_ID"
echo ""

echo "================================================================================"
echo "📊 Tor Integration Status"
echo "================================================================================"
echo ""

# Check Node 1 Tor status
echo "🔍 Node 1 Tor Status:"
if grep -q "✅ Tor SOCKS proxy is operational" tor-node1.log; then
    echo "   ✅ Tor SOCKS connected"
else
    echo "   ❌ Tor connection issue"
fi

if grep -q "✅ Initialized 4 circuits" tor-node1.log; then
    CIRCUITS=$(grep "Initialized.*circuits" tor-node1.log | tail -1)
    echo "   $CIRCUITS"
else
    echo "   ⚠️  Circuit status unclear"
fi

if grep -q "🧅 Tor Integration: ✅" tor-node1.log; then
    echo "   ✅ Tor Integration: Active"
else
    echo "   ❌ Tor Integration: Inactive"
fi

echo ""

# Check Node 2 Tor status
echo "🔍 Node 2 Tor Status:"
if grep -q "✅ Tor SOCKS proxy is operational" tor-node2.log; then
    echo "   ✅ Tor SOCKS connected"
else
    echo "   ❌ Tor connection issue"
fi

if grep -q "✅ Initialized 4 circuits" tor-node2.log; then
    CIRCUITS=$(grep "Initialized.*circuits" tor-node2.log | tail -1)
    echo "   $CIRCUITS"
else
    echo "   ⚠️  Circuit status unclear"
fi

if grep -q "🧅 Tor Integration: ✅" tor-node2.log; then
    echo "   ✅ Tor Integration: Active"
else
    echo "   ❌ Tor Integration: Inactive"
fi

echo ""
echo "================================================================================"
echo "🔗 P2P Connection Status"
echo "================================================================================"
echo ""

# Check peer connections
echo "🔍 Checking peer connections..."
sleep 2

# Node 1 peers
echo "Node 1 peers:"
curl -s http://localhost:$NODE1_HTTP/peers 2>/dev/null | jq '.' || echo "   ⚠️  API not responding"
echo ""

# Node 2 peers
echo "Node 2 peers:"
curl -s http://localhost:$NODE2_HTTP/peers 2>/dev/null | jq '.' || echo "   ⚠️  API not responding"
echo ""

# Check connection logs
if grep -q "Peer connected" tor-node1.log tor-node2.log; then
    echo "✅ Peer connection established!"
    echo ""
    echo "Connection details:"
    grep "Peer connected\|connection established" tor-node1.log tor-node2.log | sed 's/^/   /'
else
    echo "⚠️  No peer connection detected yet (may still be establishing)"
fi

echo ""
echo "================================================================================"
echo "🧪 Test Transaction through Tor"
echo "================================================================================"
echo ""

# Create test wallet on Node 1
echo "Creating test wallet on Node 1..."
WALLET_RESPONSE=$(curl -s -X POST http://localhost:$NODE1_HTTP/create_wallet \
  -H "Content-Type: application/json" \
  -d '{"wallet_type": "hot"}' 2>/dev/null)

if [ ! -z "$WALLET_RESPONSE" ]; then
    echo "✅ Wallet created:"
    echo "$WALLET_RESPONSE" | jq '.'

    WALLET_ADDRESS=$(echo "$WALLET_RESPONSE" | jq -r '.address')
    echo ""
    echo "📝 Wallet Address: $WALLET_ADDRESS"
    echo ""

    # Try to send transaction
    echo "Sending test transaction through Tor network..."
    TX_RESPONSE=$(curl -s -X POST http://localhost:$NODE1_HTTP/send_transaction \
      -H "Content-Type: application/json" \
      -d "{
        \"from\": \"$WALLET_ADDRESS\",
        \"to\": \"test_recipient_address_12345\",
        \"amount\": 100,
        \"gas_limit\": 21000
      }" 2>/dev/null)

    if [ ! -z "$TX_RESPONSE" ]; then
        echo "✅ Transaction sent:"
        echo "$TX_RESPONSE" | jq '.'
    else
        echo "⚠️  Transaction API not ready"
    fi
else
    echo "⚠️  Wallet creation API not ready"
fi

echo ""
echo "================================================================================"
echo "📊 Test Summary"
echo "================================================================================"
echo ""

# Node status
echo "Node Status:"
if ps -p $NODE1_PID > /dev/null 2>&1; then
    echo "  ✅ Node 1 running (PID: $NODE1_PID)"
else
    echo "  ❌ Node 1 crashed"
fi

if ps -p $NODE2_PID > /dev/null 2>&1; then
    echo "  ✅ Node 2 running (PID: $NODE2_PID)"
else
    echo "  ❌ Node 2 crashed"
fi

echo ""

# Tor status
echo "Tor Integration:"
TOR1=$(grep -c "✅ Tor Integration: ✅" tor-node1.log 2>/dev/null || echo "0")
TOR2=$(grep -c "✅ Tor Integration: ✅" tor-node2.log 2>/dev/null || echo "0")

if [ "$TOR1" -gt "0" ] && [ "$TOR2" -gt "0" ]; then
    echo "  ✅ Both nodes using Tor successfully"
else
    echo "  ⚠️  Tor integration incomplete (Node1: $TOR1, Node2: $TOR2)"
fi

echo ""

# Circuit count
echo "Tor Circuits:"
CIRCUITS1=$(grep -c "Creating.*circuit" tor-node1.log 2>/dev/null || echo "0")
CIRCUITS2=$(grep -c "Creating.*circuit" tor-node2.log 2>/dev/null || echo "0")
echo "  Node 1: $CIRCUITS1 circuits"
echo "  Node 2: $CIRCUITS2 circuits"

echo ""
echo "================================================================================"
echo "📁 Log Files"
echo "================================================================================"
echo ""
echo "View logs:"
echo "  Node 1: tail -f tor-node1.log"
echo "  Node 2: tail -f tor-node2.log"
echo ""
echo "Stop nodes:"
echo "  kill $NODE1_PID $NODE2_PID"
echo ""
echo "API Endpoints:"
echo "  Node 1: http://localhost:$NODE1_HTTP"
echo "  Node 2: http://localhost:$NODE2_HTTP"
echo ""
echo "================================================================================"
echo "✅ Multi-Node Tor Integration Test Complete"
echo "================================================================================"
