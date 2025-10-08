#!/bin/bash

# 🔥 Q-NarwhalKnight Battle Test - Server Beta (Validator Node)
# This script sets up Server Beta as a validator node that discovers and connects
# to Server Alpha using FREE discovery methods.

set -euo pipefail

# Battle test configuration
LOG_FILE="/mnt/shared/Q-NarwhalKnight-Beta/logs/battle_test_beta.log"
RESULTS_DIR="/mnt/shared/Q-NarwhalKnight-Beta/battle_test_results"
ONION_FILE="/tmp/q_narwhal_beta_onion.txt"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$RESULTS_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [BETA] $*" | tee -a "$LOG_FILE"
}

log "🔥 Starting Q-NarwhalKnight Battle Test - Server Beta"

# Wait for Server Alpha to be ready
log "⏳ Waiting for Server Alpha to be ready..."
ALPHA_INFO_FILE="/mnt/shared/alpha_onion_info.env"
WAIT_COUNT=0
MAX_WAIT=120

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if [ -f "$ALPHA_INFO_FILE" ]; then
        source "$ALPHA_INFO_FILE"
        if [ -n "${ALPHA_ONION_ADDRESS:-}" ] && [ -n "${BATTLE_TEST_ID:-}" ]; then
            log "✅ Server Alpha ready! Onion: $ALPHA_ONION_ADDRESS"
            break
        fi
    fi
    sleep 2
    ((WAIT_COUNT++))
    if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
        log "⏳ Still waiting for Server Alpha... ($WAIT_COUNT/${MAX_WAIT})"
    fi
done

if [ -z "${ALPHA_ONION_ADDRESS:-}" ]; then
    log "❌ Server Alpha not ready within $MAX_WAIT seconds"
    exit 1
fi

# Set Server Beta environment
export Q_NARWHAL_NODE_ROLE="validator_secondary"
export Q_NARWHAL_NODE_ID="SERVER_BETA_VALIDATOR_$BATTLE_TEST_ID"
export Q_NARWHAL_PORT="8334"
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

# Use Server Alpha as bootstrap node
export Q_NARWHAL_BOOTSTRAP_NODE="$ALPHA_ONION_ADDRESS:$ALPHA_PORT"

log "✅ Environment configured - Server Beta (Validator Node)"
log "   Node Role: validator_secondary"
log "   Node ID: $Q_NARWHAL_NODE_ID"
log "   Port: $Q_NARWHAL_PORT"
log "   Bootstrap Node: $Q_NARWHAL_BOOTSTRAP_NODE"
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

# Build the project
log "🔨 Building Q-NarwhalKnight for battle test..."
cd /mnt/orobit-shared/q-narwhalknight

if cargo build --release --bin q-narwhal-validator; then
    log "✅ Build successful"
else
    log "❌ Build failed - cannot proceed with battle test"
    exit 1
fi

# Test connection to Server Alpha
log "🧅 Testing connection to Server Alpha..."
if timeout 30 curl -s --socks5-hostname 127.0.0.1:9050 "http://$ALPHA_ONION_ADDRESS:$ALPHA_PORT/health" > /dev/null; then
    log "✅ Successfully connected to Server Alpha"
else
    log "⚠️  Cannot connect to Server Alpha - proceeding with battle test anyway"
fi

# Create battle test configuration
cat > /tmp/beta_battle_config.toml << EOF
[node]
role = "validator_secondary"
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

[bootstrap_nodes]
nodes = ["$ALPHA_ONION_ADDRESS:$ALPHA_PORT"]

[tor]
enable_onion_service = true
socks_port = 9050
control_port = 9051

[battle_test]
server_role = "beta"
target_peer = "$ALPHA_ONION_ADDRESS:$ALPHA_PORT"
discovery_target_time = 60
peer_discovery_timeout = 300
report_interval = 30
log_level = "info"
EOF

log "⚙️  Configuration created - starting Server Beta node..."

# Start the node in background
cargo run --release --bin q-narwhal-validator -- \
    --config /tmp/beta_battle_config.toml \
    --log-level info \
    --battle-test \
    > "$LOG_FILE.node" 2>&1 &

NODE_PID=$!
echo $NODE_PID > /tmp/beta_node_pid

log "🎯 Server Beta node started with PID: $NODE_PID"

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
    --role beta \
    --port $Q_NARWHAL_PORT \
    --onion-address "$ONION_ADDRESS" \
    --target-peer "$ALPHA_ONION_ADDRESS:$ALPHA_PORT" \
    --output "$RESULTS_DIR/beta_monitor.json" \
    > "$LOG_FILE.monitor" 2>&1 &

MONITOR_PID=$!
echo $MONITOR_PID > /tmp/beta_monitor_pid
log "✅ Discovery monitoring started with PID: $MONITOR_PID"

# Start peer discovery test
log "🔍 Starting peer discovery battle test..."

test_discovery() {
    local discovery_start=$(date +%s)
    local discovered_alpha=false
    local test_timeout=300 # 5 minutes

    log "🎯 Target: Discover Server Alpha at $ALPHA_ONION_ADDRESS:$ALPHA_PORT"
    
    while [ $(($(date +%s) - discovery_start)) -lt $test_timeout ]; do
        # Check if we discovered Server Alpha
        if curl -s "http://localhost:$Q_NARWHAL_PORT/peers/list" 2>/dev/null | grep -q "$ALPHA_ONION_ADDRESS"; then
            local discovery_time=$(($(date +%s) - discovery_start))
            log "🎉 SUCCESS: Discovered Server Alpha in ${discovery_time}s via FREE methods!"
            discovered_alpha=true
            break
        fi
        
        # Log current peer count
        local peer_count=$(curl -s "http://localhost:$Q_NARWHAL_PORT/peers/count" 2>/dev/null || echo "0")
        local discovery_cost=$(curl -s "http://localhost:$Q_NARWHAL_PORT/discovery/cost" 2>/dev/null || echo "0.00")
        
        if [ $(($(date +%s) - discovery_start)) -gt 0 ] && [ $((($(date +%s) - discovery_start) % 30)) -eq 0 ]; then
            log "🔍 Discovery progress: ${peer_count} peers found, cost: \$${discovery_cost} ($(( $(date +%s) - discovery_start ))s elapsed)"
        fi
        
        sleep 5
    done

    if [ "$discovered_alpha" = true ]; then
        log "✅ BATTLE TEST SUCCESS: Cross-server discovery working!"
        return 0
    else
        log "❌ BATTLE TEST FAILED: Could not discover Server Alpha within ${test_timeout}s"
        return 1
    fi
}

# Run discovery test in background
test_discovery &
TEST_PID=$!
echo $TEST_PID > /tmp/beta_test_pid

# Display battle test status
log "═══════════════════════════════════════════"
log "🏆 SERVER BETA BATTLE TEST READY"  
log "═══════════════════════════════════════════"
log "🏷️  Node ID: $Q_NARWHAL_NODE_ID"
log "🧅 Onion Address: $ONION_ADDRESS"
log "🚪 Port: $Q_NARWHAL_PORT"
log "🎯 Target: $ALPHA_ONION_ADDRESS:$ALPHA_PORT"
log "📊 Monitor PID: $MONITOR_PID"
log "🔍 Node PID: $NODE_PID"
log "🧪 Test PID: $TEST_PID"
log "═══════════════════════════════════════════"

# Share our info back to Server Alpha
echo "BETA_ONION_ADDRESS=$ONION_ADDRESS" > /mnt/shared/beta_onion_info.env
echo "BETA_PORT=$Q_NARWHAL_PORT" >> /mnt/shared/beta_onion_info.env
echo "BETA_NODE_ID=$Q_NARWHAL_NODE_ID" >> /mnt/shared/beta_onion_info.env
echo "DISCOVERY_TEST_PID=$TEST_PID" >> /mnt/shared/beta_onion_info.env

log "✅ Onion address shared back to Server Alpha"

# Start real-time status display
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
        
        # Check if we've found Alpha
        FOUND_ALPHA="❌"
        if curl -s "http://localhost:$Q_NARWHAL_PORT/peers/list" 2>/dev/null | grep -q "$ALPHA_ONION_ADDRESS"; then
            FOUND_ALPHA="✅"
        fi

        log "📊 STATUS: Peers: $PEER_COUNT | Found Alpha: $FOUND_ALPHA | Cost: \$$DISCOVERY_COST | Uptime: $UPTIME"
        
        sleep 30
    done
}

display_status &
STATUS_PID=$!
echo $STATUS_PID > /tmp/beta_status_pid

# Set up signal handlers for clean shutdown
cleanup() {
    log "🛑 Shutting down Server Beta battle test..."
    
    # Kill all spawned processes
    for pid_file in /tmp/beta_*_pid; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            if kill -0 "$PID" 2>/dev/null; then
                log "Stopping process $PID..."
                kill "$PID" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    
    # Generate final battle test report
    log "📊 Generating final battle test report..."
    if command -v cargo > /dev/null; then
        cargo run --release --example battle_test_report -- \
            --node beta \
            --battle-test-id "$BATTLE_TEST_ID" \
            --output "$RESULTS_DIR/beta_final_report.json" 2>/dev/null || true
    fi
    
    log "✅ Server Beta battle test stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for discovery test to complete
log "⏳ Running discovery battle test... Press Ctrl+C to stop"
wait $TEST_PID
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    log "🏆 BATTLE TEST VICTORY: FREE discovery methods successfully connected Server Alpha ↔ Server Beta!"
else
    log "💥 BATTLE TEST DEFEAT: Discovery methods failed to connect servers within timeout"
fi

# Keep monitoring running
log "📊 Continuing monitoring... Press Ctrl+C to stop"
wait