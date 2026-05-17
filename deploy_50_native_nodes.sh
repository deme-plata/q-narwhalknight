#!/bin/bash

echo "=�=% Q-NARWHALKNIGHT NATIVE 50-NODE DEPLOYMENT"
echo "=============================================="
echo "<� Target: 50 native Q-NarwhalKnight binaries with DNS-Phantom discovery"
echo "< Network: Native mesh formation through steganographic DNS discovery"
echo "= Discovery: Built-in DNS-Phantom peer detection (no Docker containers)"
echo "� Goal: 30,000+ TPS with native Rust binary mesh networking"
echo "� $(date)"
echo ""

# Set working directory
WORK_DIR="/opt/orobit/shared/q-narwhalknight"
cd "$WORK_DIR"

# Check if binary exists
BINARY_PATH="$WORK_DIR/target/release/q-api-server"
if [[ ! -f "$BINARY_PATH" ]]; then
    echo "L Q-NarwhalKnight binary not found at $BINARY_PATH"
    echo "=( Building binary first..."
    cargo build --release --package q-api-server
    
    if [[ ! -f "$BINARY_PATH" ]]; then
        echo "L Failed to build q-api-server binary. Exiting."
        exit 1
    fi
fi

echo "=� System Check:"
echo "   Binary: $(ls -lh $BINARY_PATH | awk '{print $5}')"
echo "   Rust Version: $(rustc --version | cut -d' ' -f2)"
echo "   Available Memory: $(free -h | grep '^Mem:' | awk '{print $7}')"
echo "   CPU Cores: $(nproc)"
echo "   Port Range: 9000-9049 (50 nodes)"
echo ""

# Create logs directory
mkdir -p logs node-data
echo "=� Created logs/ and node-data/ directories"

# Clean up any existing processes
echo ">� Cleaning up existing Q-NarwhalKnight processes..."
pkill -f q-api-server 2>/dev/null || true
sleep 2

echo ""
echo "�  DEPLOYMENT CONFIGURATION:"
echo "   Node Count: 50 native processes"
echo "   Port Range: 9000-9049" 
echo "   Discovery: DNS-Phantom steganographic mesh"
echo "   Data Directory: ./node-data/"
echo "   Log Directory: ./logs/"
echo ""
read -p "Continue with native 50-node deployment? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "L Deployment cancelled."
    exit 1
fi

echo ""
echo "=� LAUNCHING 50 NATIVE Q-NARWHALKNIGHT NODES..."
echo ""

# Create a discovery hub node first (acts as initial bootstrap)
echo "= Phase 1: DNS-Phantom Discovery Hub (Bootstrap Node)"
RUST_LOG=info $BINARY_PATH \
    --node-id "dns-phantom-hub" \
    --role discovery-hub \
    --port 8080 \
    --data-dir "./node-data/hub" \
    --enable-dns-phantom \
    --bootstrap-mode \
    > ./logs/dns-phantom-hub.log 2>&1 &

DNS_HUB_PID=$!
echo "   <� DNS-Phantom Hub started: PID $DNS_HUB_PID (port 8080)"
sleep 3

# Launch 10 Alpha nodes (simulating Server Alpha connections)
echo "=� Phase 2: Alpha Nodes (10 nodes, ports 9000-9009)"
for i in {0..9}; do
    NODE_ID="alpha-node-$(printf "%02d" $((i+1)))"
    PORT=$((9000 + i))
    
    RUST_LOG=info $BINARY_PATH \
        --node-id "$NODE_ID" \
        --role alpha \
        --port $PORT \
        --data-dir "./node-data/$NODE_ID" \
        --enable-dns-phantom \
        --discovery-hub "127.0.0.1:8080" \
        --beta-target "185.182.185.227:8081" \
        > ./logs/$NODE_ID.log 2>&1 &
    
    NODE_PID=$!
    echo "   =� $NODE_ID started: PID $NODE_PID (port $PORT)"
    
    # Stagger startup to prevent resource overload
    sleep 0.5
done

sleep 5

# Launch 39 Validator nodes (consensus participants)
echo "� Phase 3: Validator Nodes (39 nodes, ports 9010-9048)"
for i in {10..48}; do
    NODE_ID="validator-$(printf "%02d" $((i-9)))"
    PORT=$((9000 + i))
    
    RUST_LOG=info $BINARY_PATH \
        --node-id "$NODE_ID" \
        --role validator \
        --port $PORT \
        --data-dir "./node-data/$NODE_ID" \
        --enable-dns-phantom \
        --discovery-hub "127.0.0.1:8080" \
        --consensus dag-knight \
        > ./logs/$NODE_ID.log 2>&1 &
    
    NODE_PID=$!
    echo "   � $NODE_ID started: PID $NODE_PID (port $PORT)"
    
    # Stagger startup
    sleep 0.3
done

sleep 2

# Launch monitoring node
echo "=� Phase 4: Network Monitor (port 9049)"
RUST_LOG=info $BINARY_PATH \
    --node-id "network-monitor" \
    --role monitor \
    --port 9049 \
    --data-dir "./node-data/monitor" \
    --enable-dns-phantom \
    --discovery-hub "127.0.0.1:8080" \
    --monitoring-mode \
    > ./logs/network-monitor.log 2>&1 &

MONITOR_PID=$!
echo "   =� Network Monitor started: PID $MONITOR_PID (port 9049)"

echo ""
echo "� Waiting for network stabilization..."
sleep 15

echo ""
echo " NATIVE 50-NODE Q-NARWHALKNIGHT MESH DEPLOYMENT COMPLETE!"
echo ""

# Count running processes
RUNNING_NODES=$(pgrep -f q-api-server | wc -l)
echo "=� Deployment Summary:"
echo "   " Active Processes: $RUNNING_NODES / 51 expected (50 nodes + 1 hub)"
echo "   " Port Range: 8080 (hub), 9000-9049 (50 nodes)"
echo "   " Discovery: DNS-Phantom steganographic mesh networking"
echo "   " Data Storage: ./node-data/ (51 node directories)"
echo "   " Logs: ./logs/ (51 log files)"
echo ""

echo "< Node Types:"
echo "   " DNS-Phantom Hub: 1 (bootstrap discovery)"
echo "   " Alpha Nodes: 10 (Server Alpha simulation)"  
echo "   " Validator Nodes: 39 (consensus participants)"
echo "   " Network Monitor: 1 (real-time status)"
echo ""

echo "=� Real-time Monitoring:"
echo "   " Network Monitor Log: tail -f ./logs/network-monitor.log"
echo "   " DNS-Phantom Hub: tail -f ./logs/dns-phantom-hub.log"
echo "   " Alpha Node 01: tail -f ./logs/alpha-node-01.log"
echo "   " Validator 01: tail -f ./logs/validator-01.log"
echo "   " All Nodes: tail -f ./logs/*.log"
echo ""

echo "= System Commands:"
echo "   " Active Processes: ps aux | grep q-api-server"
echo "   " Port Usage: ss -tlnp | grep :90"
echo "   " Network Connections: netstat -an | grep :90 | grep ESTABLISHED"
echo "   " Process Tree: pstree -p \$(pgrep -f q-api-server | head -1)"
echo "   " Resource Usage: top -p \$(pgrep -f q-api-server | tr '\\n' ',' | sed 's/,\$//')"
echo ""

echo "<� EXPECTED NATIVE PERFORMANCE:"
echo "   " Node Discovery: <5 seconds via DNS-Phantom"
echo "   " Mesh Formation: 50-node interconnected consensus"
echo "   " Target TPS: 30,000+ transactions per second"
echo "   " Consensus Latency: <3 seconds (native DAG-Knight)"
echo "   " Memory Usage: ~50-100MB per node (efficient native binaries)"
echo "   " Network Overhead: Minimal (no container networking)"
echo ""

echo "=� TESTING WORKFLOW:"
echo "   1. Monitor DNS-Phantom peer discovery (first 30 seconds)"
echo "   2. Verify Alpha nodes connect to Server Beta (185.182.185.227:8081)"
echo "   3. Watch consensus mesh formation among 39 validators"
echo "   4. Observe native binary performance and resource usage"
echo "   5. Test massive scale transaction throughput"
echo ""

echo "=� Management Commands:"
echo "   " Stop All Nodes: pkill -f q-api-server"
echo "   " Restart Single Node: kill <PID> && ./deploy_50_native_nodes.sh"
echo "   " View Specific Log: tail -f ./logs/<node-id>.log"
echo "   " Check Discovery: grep -i 'phantom\\|discovery\\|peer' ./logs/*.log"
echo ""

echo "<� NATIVE Q-NARWHALKNIGHT 50-NODE MESH IS LIVE!"
echo "   Ready for distributed quantum consensus at native performance!"

# Start real-time monitoring
echo ""
echo "=� Starting real-time native deployment monitoring..."
echo "   (Press Ctrl+C to stop monitoring and return to shell)"
echo ""

# Monitor native deployment
while true; do
    RUNNING_NODES=$(pgrep -f q-api-server | wc -l)
    ACTIVE_CONNECTIONS=$(ss -an | grep :90 | grep ESTABLISHED | wc -l)
    
    echo "[$(date +%H:%M:%S)] = Native Status: $RUNNING_NODES/51 processes | $ACTIVE_CONNECTIONS active connections"
    
    # Check key services
    if pgrep -f "dns-phantom-hub" > /dev/null; then
        HUB_STATUS=""
    else
        HUB_STATUS="L"
    fi
    
    ALPHA_COUNT=$(pgrep -f "alpha-node" | wc -l)
    VALIDATOR_COUNT=$(pgrep -f "validator-" | wc -l)
    
    echo "         Services: Hub $HUB_STATUS | Alpha: $ALPHA_COUNT/10 | Validators: $VALIDATOR_COUNT/39"
    
    # Check for DNS-Phantom discovery activity
    DISCOVERY_EVENTS=$(find ./logs/ -name "*.log" -exec grep -l -i "phantom\|discovery\|peer" {} \; 2>/dev/null | wc -l)
    echo "         Discovery: $DISCOVERY_EVENTS nodes show DNS-Phantom activity"
    
    if [ $RUNNING_NODES -eq 51 ] && [ $DISCOVERY_EVENTS -gt 20 ]; then
        echo "         <� NATIVE 50-NODE MESH FULLY OPERATIONAL WITH DNS-PHANTOM DISCOVERY!"
    fi
    
    sleep 10
done