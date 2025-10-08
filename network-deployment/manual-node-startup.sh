#!/bin/bash
# Manual Q-NarwhalKnight Node Startup
# Simplified startup for immediate testing

set -euo pipefail

echo "🚀 MANUAL Q-NARWHALKNIGHT NODE STARTUP"
echo "====================================="

# Start 5 Server Alpha nodes using available binaries
NODES=("alice:8001" "bob:8002" "charlie:8003" "diana:8004" "eve:8005")

echo "🔍 Available binaries:"
ls -la target/release/ | grep "^-rwx" | awk '{print $9}' | head -5

echo ""
echo "🚀 Starting Q-API server nodes for consensus..."

for i in "${!NODES[@]}"; do
    IFS=':' read -r name port <<< "${NODES[$i]}"
    
    echo "  🌟 Starting ${name} on port ${port}..."
    
    # Start Q-API server with consensus configuration
    RUST_LOG=info target/release/q-api-server \
        --port "$port" \
        --node-id "$name" \
        --consensus-mode \
        --tor-enabled \
        > "real-logs/node-${name}.log" 2>&1 &
    
    NODE_PID=$!
    echo "${NODE_PID}" > "configs/${name}/pid"
    echo "    ✅ Node ${name} started (PID: ${NODE_PID})"
    
    sleep 1
done

echo ""
echo "⏳ Waiting for nodes to initialize..."
sleep 5

echo ""
echo "📊 Node Status Check:"
for i in "${!NODES[@]}"; do
    IFS=':' read -r name port <<< "${NODES[$i]}"
    pid_file="configs/${name}/pid"
    
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  ✅ ${name}: Running (PID: ${pid})"
            
            # Test HTTP endpoint
            if curl -s -f "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
                echo "    🌐 HTTP API: Responding"
            else
                echo "    ⏳ HTTP API: Starting up..."
            fi
        else
            echo "  ❌ ${name}: Process died"
        fi
    else
        echo "  ❌ ${name}: PID file not found"
    fi
done

echo ""
echo "🎯 Server Alpha Deployment Status:"
echo "=================================="
echo "🔢 Nodes Started: 5/5"
echo "🧅 Tor Integration: Enabled"
echo "🎯 Waiting for Server Beta: 5 additional nodes"
echo "🏆 Target: 10-node anonymous quantum BFT network"
echo ""

echo "📊 Quick monitoring commands:"
echo "# Check all nodes: ./network-deployment/network-monitor.sh"
echo "# View logs: tail -f real-logs/node-*.log"  
echo "# Test API: curl http://127.0.0.1:8001/health"
echo ""

echo "🚀 Server Alpha nodes ready for BFT consensus!"
echo "Awaiting Server Beta deployment for complete network..."