#!/bin/bash

# 🔥 Q-NarwhalKnight Battle Test - Server Alpha (Bootstrap Node)
# This script sets up Server Alpha as the primary bootstrap node for battle testing
# FREE discovery methods across distributed servers.

set -euo pipefail

# Battle test configuration
BATTLE_TEST_ID="alpha_beta_$(date +%s)"
LOG_FILE="/mnt/shared/Q-NarwhalKnight/logs/battle_test_alpha.log"
RESULTS_DIR="/mnt/shared/Q-NarwhalKnight/battle_test_results"
ONION_FILE="/tmp/q_narwhal_alpha_onion.txt"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$RESULTS_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ALPHA] $*" | tee -a "$LOG_FILE"
}

log "🔥 Starting Q-NarwhalKnight Battle Test - Server Alpha"
log "Battle Test ID: $BATTLE_TEST_ID"

# Set Server Alpha environment
export Q_NARWHAL_NODE_ROLE="bootstrap_primary"
export Q_NARWHAL_NODE_ID="SERVER_ALPHA_BOOTSTRAP_$BATTLE_TEST_ID"
export Q_NARWHAL_PORT="8333"
export Q_NARWHAL_FREE_ONLY="true"
export Q_NARWHAL_BATTLE_TEST="true"
export Q_NARWHAL_BATTLE_TEST_ID="$BATTLE_TEST_ID"

# Enable ALL FREE discovery methods
export Q_NARWHAL_TOR_DHT="true"
export Q_NARWHAL_BOOTSTRAP="true"
export Q_NARWHAL_GOSSIP="true"
export Q_NARWHAL_BITCOIN_FREE="true"
export Q_NARWHAL_DNS_DISCOVERY="true"
export Q_NARWHAL_MAX_DAILY_COST="0.00"

log "✅ Environment configured - Server Alpha (Bootstrap Node)"
log "   Node Role: bootstrap_primary"
log "   Node ID: $Q_NARWHAL_NODE_ID"
log "   Port: $Q_NARWHAL_PORT"
log "   FREE Mode: $Q_NARWHAL_FREE_ONLY"
log "   Max Daily Cost: $Q_NARWHAL_MAX_DAILY_COST"

# Check prerequisites
log "🔍 Checking battle test prerequisites..."

# Check if Tor is running
if ! pgrep -x "tor" > /dev/null; then
    log "⚠️  Tor daemon not running - attempting to start..."
    if command -v tor > /dev/null; then
        tor --runasdaemon 1 --socksport 9050 --controlport 9051 || true
        sleep 3
    else
        log "❌ Tor not installed - install with: sudo apt install tor"
        exit 1
    fi
fi

if pgrep -x "tor" > /dev/null; then
    log "✅ Tor daemon is running"
else
    log "❌ Tor daemon failed to start"
    exit 1
fi

# Check if we can build the project
log "🔨 Building Q-NarwhalKnight for battle test..."
cd /mnt/orobit-shared/q-narwhalknight

if cargo build --release --bin q-narwhal-validator; then
    log "✅ Build successful"
else
    log "❌ Build failed - cannot proceed with battle test"
    exit 1
fi

# Test Tor connectivity
log "🧅 Testing Tor connectivity..."
if timeout 10 curl -s --socks5-hostname 127.0.0.1:9050 https://check.torproject.org/api/ip > /dev/null; then
    log "✅ Tor connectivity verified"
else
    log "⚠️  Tor connectivity test failed - proceeding anyway"
fi

# Start the battle test node
log "🚀 Starting Server Alpha battle test node..."

# Create battle test configuration
cat > /tmp/alpha_battle_config.toml << EOF
[node]
role = "bootstrap_primary"
node_id = "$Q_NARWHAL_NODE_ID"
port = $Q_NARWHAL_PORT
battle_test = true
battle_test_id = "$BATTLE_TEST_ID"

[discovery]
free_methods_only = true
max_cost_per_day = 0.0
tor_dht_enabled = true
bootstrap_enabled = true
gossip_enabled = true
bitcoin_discovery_enabled = true
dns_discovery_enabled = true

[tor]
enable_onion_service = true
socks_port = 9050
control_port = 9051

[battle_test]
server_role = "alpha"
peer_discovery_timeout = 300
report_interval = 30
log_level = "info"
EOF

log "⚙️  Configuration created - starting node..."

# Start the node in background with comprehensive logging
cargo run --release --bin q-narwhal-validator -- \
    --config /tmp/alpha_battle_config.toml \
    --log-level info \
    --battle-test \
    > "$LOG_FILE.node" 2>&1 &

NODE_PID=$!
echo $NODE_PID > /tmp/alpha_node_pid

log "🎯 Server Alpha node started with PID: $NODE_PID"

# Wait for onion service to be established
log "🧅 Waiting for Tor onion service to be established..."
ONION_ADDRESS=""
WAIT_COUNT=0
MAX_WAIT=60

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    # Check if node is still running
    if ! kill -0 $NODE_PID 2>/dev/null; then
        log "❌ Node process died - check logs"
        exit 1
    fi

    # Try to extract onion address from logs
    if [ -f "$LOG_FILE.node" ]; then
        ONION_ADDRESS=$(grep -o '[a-z2-7]\{56\}\.onion' "$LOG_FILE.node" | head -1 || true)
    fi

    if [ -n "$ONION_ADDRESS" ]; then
        log "🎉 Onion service established: $ONION_ADDRESS"
        echo "$ONION_ADDRESS" > "$ONION_FILE"
        break
    fi

    sleep 3
    ((WAIT_COUNT++))
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        log "⏳ Still waiting for onion address... ($WAIT_COUNT/${MAX_WAIT})"
    fi
done

if [ -z "$ONION_ADDRESS" ]; then
    log "❌ Failed to establish onion service within $MAX_WAIT seconds"
    kill $NODE_PID 2>/dev/null || true
    exit 1
fi

# Start discovery monitoring
log "📊 Starting discovery monitoring..."
cargo run --release --example battle_test_monitor -- \
    --role alpha \
    --port $Q_NARWHAL_PORT \
    --onion-address "$ONION_ADDRESS" \
    --output "$RESULTS_DIR/alpha_monitor.json" \
    > "$LOG_FILE.monitor" 2>&1 &

MONITOR_PID=$!
echo $MONITOR_PID > /tmp/alpha_monitor_pid
log "✅ Discovery monitoring started with PID: $MONITOR_PID"

# Display battle test status
log "═══════════════════════════════════════════"
log "🏆 SERVER ALPHA BATTLE TEST READY"
log "═══════════════════════════════════════════"
log "🏷️  Node ID: $Q_NARWHAL_NODE_ID"
log "🧅 Onion Address: $ONION_ADDRESS"
log "🚪 Port: $Q_NARWHAL_PORT"
log "📊 Monitor PID: $MONITOR_PID"
log "🔍 Node PID: $NODE_PID"
log "📁 Results Directory: $RESULTS_DIR"
log "═══════════════════════════════════════════"

# Share onion address with Server Beta
log "📢 Broadcasting Server Alpha onion address for Server Beta..."
echo "ALPHA_ONION_ADDRESS=$ONION_ADDRESS" > /mnt/shared/alpha_onion_info.env
echo "ALPHA_PORT=$Q_NARWHAL_PORT" >> /mnt/shared/alpha_onion_info.env
echo "ALPHA_NODE_ID=$Q_NARWHAL_NODE_ID" >> /mnt/shared/alpha_onion_info.env
echo "BATTLE_TEST_ID=$BATTLE_TEST_ID" >> /mnt/shared/alpha_onion_info.env

log "✅ Onion address broadcasted to Server Beta"
log "📄 Server Beta can source /mnt/shared/alpha_onion_info.env"

# Start real-time status display
log "📊 Starting real-time battle test status..."

display_status() {
    while true; do
        if ! kill -0 $NODE_PID 2>/dev/null; then
            log "❌ Node process died"
            break
        fi

        # Get current stats
        PEER_COUNT=$(curl -s http://localhost:$Q_NARWHAL_PORT/peers/count 2>/dev/null || echo "0")
        DISCOVERY_COST=$(curl -s http://localhost:$Q_NARWHAL_PORT/discovery/cost 2>/dev/null || echo "0.00")
        UPTIME=$(ps -o etime= -p $NODE_PID 2>/dev/null | tr -d ' ' || echo "unknown")

        log "📊 STATUS: Peers: $PEER_COUNT | Cost: \$$DISCOVERY_COST | Uptime: $UPTIME"
        
        sleep 30
    done
}

display_status &
STATUS_PID=$!
echo $STATUS_PID > /tmp/alpha_status_pid

# Wait for battle test completion or manual termination
log "⏳ Battle test running... Press Ctrl+C to stop or run stop_battle_test_alpha.sh"
log "🔗 Server Beta should connect to: $ONION_ADDRESS:$Q_NARWHAL_PORT"

# Set up signal handlers for clean shutdown
cleanup() {
    log "🛑 Shutting down Server Alpha battle test..."
    
    # Kill all spawned processes
    for pid_file in /tmp/alpha_*_pid; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            if kill -0 "$PID" 2>/dev/null; then
                log "Stopping process $PID..."
                kill "$PID" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    
    log "✅ Server Alpha battle test stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep running until interrupted
wait