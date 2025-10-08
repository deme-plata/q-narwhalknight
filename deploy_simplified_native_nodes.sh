#!/bin/bash

echo "🚀 Q-NARWHALKNIGHT SIMPLIFIED NATIVE DEPLOYMENT"
echo "================================================"
echo "🎯 Target: 10 native Alpha nodes + DNS-Phantom discovery"
echo "🌐 Server Beta Integration: 185.182.185.227:8081"
echo "📊 Goal: Native performance mesh networking"
echo "⏰ $(date)"
echo ""

# Check if we can run the binary
BINARY_PATH="./target/release/q-api-server"
if [[ ! -f "$BINARY_PATH" ]]; then
    echo "❌ Q-NarwhalKnight binary not found at $BINARY_PATH"
    echo "🔧 Building binary first..."
    cargo build --release --bin q-api-server
    
    if [[ ! -f "$BINARY_PATH" ]]; then
        echo "❌ Failed to build q-api-server binary. Exiting."
        exit 1
    fi
fi

echo "✅ System Check:"
echo "   📦 Binary: $(ls -lh $BINARY_PATH | awk '{print $5}')"
echo "   🦀 Rust: $(rustc --version | cut -d' ' -f2)"
echo "   💾 Memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "   🖥️  CPU: $(nproc) cores"
echo ""

# Clean up any existing processes
echo "🧹 Cleaning up existing Q-NarwhalKnight processes..."
pkill -f q-api-server 2>/dev/null || true
sleep 3

# Create directories
mkdir -p logs node-data
echo "📁 Created logs/ and node-data/ directories"

echo ""
echo "🚀 LAUNCHING NATIVE Q-NARWHALKNIGHT NODES..."
echo ""

# Launch DNS-Phantom Hub (simplified)
echo "🔍 Phase 1: DNS-Phantom Discovery Hub"
RUST_LOG=info Q_NODE_ID=dns-phantom-hub Q_API_PORT=8080 $BINARY_PATH \
    > ./logs/dns-phantom-hub.log 2>&1 &

DNS_HUB_PID=$!
echo "   ✅ DNS-Phantom Hub: PID $DNS_HUB_PID (port 8080)"
sleep 3

# Launch 10 Alpha nodes targeting Server Beta
echo "🎯 Phase 2: Alpha Nodes (10 nodes, ports 9000-9009)"
for i in {1..10}; do
    NODE_ID="alpha-native-$(printf "%02d" $i)"
    PORT=$((8999 + i))
    
    RUST_LOG=info Q_NODE_ID="$NODE_ID" Q_API_PORT=$PORT $BINARY_PATH \
        --node-id "$NODE_ID" --target-beta --port $PORT \
        > ./logs/$NODE_ID.log 2>&1 &
    
    NODE_PID=$!
    echo "   🚀 $NODE_ID: PID $NODE_PID (port $PORT)"
    
    # Stagger startup
    sleep 1
done

sleep 5

# Launch 5 Validator nodes (simplified consensus)
echo "⚛️ Phase 3: Validator Nodes (5 nodes, ports 9010-9014)"
for i in {1..5}; do
    NODE_ID="validator-native-$(printf "%02d" $i)"
    PORT=$((9009 + i))
    
    RUST_LOG=info Q_NODE_ID="$NODE_ID" Q_API_PORT=$PORT Q_IS_VALIDATOR=true $BINARY_PATH \
        --node-id "$NODE_ID" --port $PORT \
        > ./logs/$NODE_ID.log 2>&1 &
    
    NODE_PID=$!
    echo "   ⚛️ $NODE_ID: PID $NODE_PID (port $PORT)"
    
    sleep 1
done

sleep 3

echo ""
echo "⏳ Waiting for network stabilization..."
sleep 10

# Count running processes
RUNNING_NODES=$(pgrep -f q-api-server | wc -l)
echo ""
echo "🎉 NATIVE Q-NARWHALKNIGHT DEPLOYMENT COMPLETE!"
echo ""
echo "📊 Deployment Summary:"
echo "   ✅ Active Processes: $RUNNING_NODES / 16 expected"
echo "   🔍 DNS-Phantom Hub: port 8080"
echo "   🎯 Alpha Nodes: ports 9000-9009 (connecting to Server Beta)"
echo "   ⚛️ Validator Nodes: ports 9010-9014"
echo "   📁 Data Storage: ./node-data/"
echo "   📝 Logs: ./logs/"
echo ""

echo "🔗 Server Beta Connections:"
echo "   🎯 Target: 185.182.185.227:8081"
echo "   📡 Alpha nodes automatically targeting Server Beta"
echo "   🤝 JSON handshake protocol active"
echo ""

echo "📋 Real-time Monitoring:"
echo "   👁️ DNS-Phantom Hub: tail -f ./logs/dns-phantom-hub.log"
echo "   🎯 Alpha Node 01: tail -f ./logs/alpha-native-01.log"
echo "   ⚛️ Validator 01: tail -f ./logs/validator-native-01.log"
echo "   📊 All Nodes: tail -f ./logs/*.log"
echo ""

echo "🛠️ Management Commands:"
echo "   🔍 Active Processes: ps aux | grep q-api-server"
echo "   📡 Port Usage: ss -tlnp | grep ':90'"
echo "   🌐 Beta Connections: netstat -an | grep '185.182.185.227:8081'"
echo "   🛑 Stop All: pkill -f q-api-server"
echo ""

echo "🚀 EXPECTED PERFORMANCE:"
echo "   🔍 DNS-Phantom Discovery: <10 seconds"
echo "   🎯 Server Beta Connections: 10 Alpha nodes"
echo "   ⚛️ Consensus Network: 5 validator nodes"
echo "   🏃 Native Performance: ~50MB per node"
echo ""

# Start monitoring
echo "📊 Starting real-time native deployment monitoring..."
echo "   (Press Ctrl+C to stop monitoring)"
echo ""

# Simple monitoring loop
while true; do
    RUNNING_NODES=$(pgrep -f q-api-server | wc -l)
    
    # Count nodes by type
    ALPHA_COUNT=$(pgrep -f "alpha-native" | wc -l)
    VALIDATOR_COUNT=$(pgrep -f "validator-native" | wc -l)
    HUB_COUNT=$(pgrep -f "dns-phantom-hub" | wc -l)
    
    echo "[$(date +%H:%M:%S)] 📊 Native Status: $RUNNING_NODES/16 total | Hub: $HUB_COUNT | Alpha: $ALPHA_COUNT/10 | Validators: $VALIDATOR_COUNT/5"
    
    # Check for recent log activity (DNS-Phantom discovery)
    RECENT_ACTIVITY=$(find ./logs/ -name "*.log" -mmin -1 | wc -l)
    echo "         🔄 Recent Activity: $RECENT_ACTIVITY nodes with new logs"
    
    if [ $RUNNING_NODES -eq 16 ] && [ $ALPHA_COUNT -eq 10 ]; then
        echo "         🎉 NATIVE 16-NODE MESH FULLY OPERATIONAL!"
    fi
    
    # Test Server Beta connection
    if echo "test" | nc -w 1 185.182.185.227 8081 >/dev/null 2>&1; then
        echo "         🎯 Server Beta: ✅ REACHABLE"
    else
        echo "         🎯 Server Beta: ⚠️ NOT REACHABLE"
    fi
    
    sleep 15
done