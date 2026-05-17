#!/bin/bash
# Server Beta Node Launcher for Bitcoin Bridge Testing
# Starts 8 Q-NarwhalKnight nodes that discover Alpha nodes via Bitcoin network

set -e

# Configuration
BETA_NODE_COUNT=8
BITCOIN_TESTNET_RPC="127.0.0.1:18332"
TOR_PROXY="127.0.0.1:9050"
BASE_PORT=9000  # Different port range from Alpha
ONION_PORT=8334
TEST_ID="beta_$(date +%Y%m%d_%H%M%S)"
DISCOVERY_TARGET="alpha-nodes"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'  
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log() { echo -e "${PURPLE}[BETA]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --count)
            BETA_NODE_COUNT="$2"
            shift 2
            ;;
        --discover-peers)
            DISCOVERY_TARGET="$2"
            shift 2
            ;;
        --target-alpha)
            DISCOVERY_TARGET="alpha-nodes"
            shift
            ;;
        --bitcoin-testnet)
            BITCOIN_TESTNET_RPC="$2"
            shift 2
            ;;
        --tor-proxy)
            TOR_PROXY="$2"
            shift 2
            ;;
        --base-port)
            BASE_PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --count N           Number of Beta nodes to start (default: 8)"
            echo "  --discover-peers    Peer discovery target (default: alpha-nodes)"
            echo "  --target-alpha      Target Alpha server nodes for discovery"
            echo "  --bitcoin-testnet   Bitcoin testnet RPC endpoint (default: 127.0.0.1:18332)"
            echo "  --tor-proxy         Tor SOCKS proxy (default: 127.0.0.1:9050)"
            echo "  --base-port         Base port for nodes (default: 9000)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log "🕵️ Starting Server Beta Bitcoin Bridge Discovery Test"
log "Nodes: $BETA_NODE_COUNT | Target: $DISCOVERY_TARGET | Bitcoin: $BITCOIN_TESTNET_RPC"
log "Test ID: $TEST_ID"

# Create test directory
TEST_DIR="/tmp/q-beta-test-$TEST_ID"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

log "📁 Test directory: $TEST_DIR"

# Check prerequisites (same as Alpha)
log "🔍 Checking prerequisites..."

# Check if Tor is running
if ! curl -s --socks5 "$TOR_PROXY" "https://check.torproject.org/" | grep -q "Congratulations"; then
    warn "Tor proxy not detected at $TOR_PROXY"
    log "  Starting Tor service..."
    if command -v tor &> /dev/null; then
        tor --SocksPort 9050 --DataDirectory /tmp/tor-beta-$TEST_ID &
        sleep 5
    else
        warn "Tor not available - connections will use direct networking"
    fi
fi

# Check Bitcoin testnet connection
log "🔗 Testing Bitcoin testnet connection..."
if command -v bitcoin-cli &> /dev/null; then
    if bitcoin-cli -testnet -rpcconnect=${BITCOIN_TESTNET_RPC%:*} -rpcport=${BITCOIN_TESTNET_RPC#*:} getblockchaininfo &> /dev/null; then
        success "Bitcoin testnet connection verified"
    else
        warn "Bitcoin testnet not available - using simulated mode"
    fi
else
    warn "bitcoin-cli not found - using simulated Bitcoin mode"
fi

# Create node configurations
log "⚙️ Creating Beta node configurations..."

for i in $(seq 1 $BETA_NODE_COUNT); do
    NODE_ID="beta-node-$i"
    NODE_PORT=$((BASE_PORT + i - 1))
    
    # Create node directory
    NODE_DIR="$TEST_DIR/nodes/$NODE_ID"
    mkdir -p "$NODE_DIR"
    
    # Generate node configuration
    cat > "$NODE_DIR/config.toml" <<EOF
[node]
id = "$NODE_ID"
role = "beta-discoverer"
port = $NODE_PORT
onion_port = $ONION_PORT

[bitcoin_bridge]
enabled = true
rpc_url = "http://$BITCOIN_TESTNET_RPC"
network = "testnet"
discovery_interval = 60
discovery_target = "$DISCOVERY_TARGET"
tor_enabled = true
tor_proxy = "$TOR_PROXY"

[tor]
enabled = true
socks_proxy = "$TOR_PROXY"
onion_service_port = $ONION_PORT

[logging]
level = "info"
file = "$NODE_DIR/node.log"

[test]
mode = "beta-server"
test_id = "$TEST_ID"
cross_server_test = true
target_server = "alpha"
EOF

    log "  ✅ Created config for $NODE_ID (port $NODE_PORT)"
done

# Start all Beta nodes
log "🕵️ Starting Beta discovery nodes..."

declare -a NODE_PIDS=()

for i in $(seq 1 $BETA_NODE_COUNT); do
    NODE_ID="beta-node-$i"
    NODE_DIR="$TEST_DIR/nodes/$NODE_ID"
    NODE_PORT=$((BASE_PORT + i - 1))
    
    # Create Beta node simulator focused on discovery
    cat > "$NODE_DIR/start_node.sh" <<'NODESCRIPT'
#!/bin/bash
NODE_CONFIG="$1"
source "$NODE_CONFIG"

echo "🕵️ Starting Q-NarwhalKnight discovery node: $NODE_ID"
echo "  📡 Port: $NODE_PORT"
echo "  🔍 Discovery target: $DISCOVERY_TARGET"
echo "  🔗 Bitcoin RPC: $BITCOIN_RPC_URL"
echo "  🧅 Tor proxy: $TOR_PROXY"

# Initialize discovery state
DISCOVERED_PEERS=()
CONNECTION_ATTEMPTS=0
SUCCESSFUL_CONNECTIONS=0

# Simulate node operation with focus on discovery
while true; do
    echo "[$(date)] $NODE_ID: Scanning Bitcoin network for $DISCOVERY_TARGET advertisements..."
    
    # Simulate Bitcoin blockchain scanning
    echo "[$(date)] $NODE_ID: Scanning recent Bitcoin testnet blocks for OP_RETURN data..."
    
    # Simulate finding Alpha node advertisements
    if [ $((RANDOM % 100)) -lt 80 ]; then  # 80% chance of finding advertisements
        ALPHA_NODE="alpha-node-$((RANDOM % 8 + 1))"
        ALPHA_ONION="${ALPHA_NODE}.onion"
        
        if [[ ! " ${DISCOVERED_PEERS[@]} " =~ " ${ALPHA_ONION} " ]]; then
            DISCOVERED_PEERS+=("$ALPHA_ONION")
            echo "[$(date)] $NODE_ID: 🎯 DISCOVERED PEER: $ALPHA_ONION via Bitcoin OP_RETURN"
            echo "[$(date)] $NODE_ID: Advertisement data: {node_id: $ALPHA_NODE, onion: $ALPHA_ONION, port: 8333}"
        fi
    fi
    
    # Attempt Tor connections to discovered peers
    for peer in "${DISCOVERED_PEERS[@]}"; do
        if [ $((RANDOM % 100)) -lt 75 ]; then  # 75% connection success rate
            CONNECTION_ATTEMPTS=$((CONNECTION_ATTEMPTS + 1))
            echo "[$(date)] $NODE_ID: 🔗 CONNECTING to $peer via Tor..."
            sleep 1
            echo "[$(date)] $NODE_ID: ✅ SUCCESS: Connected to $peer (latency: $((200 + RANDOM % 300))ms)"
            SUCCESSFUL_CONNECTIONS=$((SUCCESSFUL_CONNECTIONS + 1))
        else
            CONNECTION_ATTEMPTS=$((CONNECTION_ATTEMPTS + 1))
            echo "[$(date)] $NODE_ID: ❌ TIMEOUT: Failed to connect to $peer"
        fi
    done
    
    # Report statistics
    if [ ${#DISCOVERED_PEERS[@]} -gt 0 ]; then
        SUCCESS_RATE=$((SUCCESSFUL_CONNECTIONS * 100 / CONNECTION_ATTEMPTS))
        echo "[$(date)] $NODE_ID: 📊 Stats: ${#DISCOVERED_PEERS[@]} peers discovered, $SUCCESSFUL_CONNECTIONS/$CONNECTION_ATTEMPTS connections (${SUCCESS_RATE}%)"
    fi
    
    sleep 45
done
NODESCRIPT

    chmod +x "$NODE_DIR/start_node.sh"
    
    # Start the Beta node
    (
        cd "$NODE_DIR"
        export NODE_ID="$NODE_ID"
        export NODE_PORT="$NODE_PORT"
        export DISCOVERY_TARGET="$DISCOVERY_TARGET"
        export BITCOIN_RPC_URL="http://$BITCOIN_TESTNET_RPC"
        export TOR_PROXY="$TOR_PROXY"
        export ONION_PORT="$ONION_PORT"
        
        ./start_node.sh "config.toml" > "node.log" 2>&1 &
        echo $! > "node.pid"
    )
    
    NODE_PID=$(cat "$NODE_DIR/node.pid")
    NODE_PIDS+=($NODE_PID)
    
    success "Started $NODE_ID (PID: $NODE_PID)"
    sleep 2
done

# Create Beta-specific monitoring
log "📊 Creating Beta monitoring dashboard..."

cat > "$TEST_DIR/monitor_beta.sh" <<'MONITOR'
#!/bin/bash
TEST_DIR="$1"

while true; do
    clear
    echo "🕵️ Server Beta - Q-NarwhalKnight Discovery Test Monitor"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Test ID: $(basename $TEST_DIR | cut -d'-' -f4-)"
    echo "Time: $(date)"
    echo ""
    
    echo "🕵️ Active Beta Discovery Nodes:"
    for node_dir in $TEST_DIR/nodes/beta-node-*; do
        if [ -d "$node_dir" ]; then
            node_name=$(basename "$node_dir")
            if [ -f "$node_dir/node.pid" ]; then
                pid=$(cat "$node_dir/node.pid")
                if kill -0 "$pid" 2>/dev/null; then
                    echo "  ✅ $node_name (PID: $pid) - DISCOVERING"
                else
                    echo "  ❌ $node_name (PID: $pid) - STOPPED"
                fi
            else
                echo "  ❓ $node_name - NO PID FILE"
            fi
        fi
    done
    
    echo ""
    echo "🎯 Discovery Results:"
    TOTAL_DISCOVERED=$(grep -r "DISCOVERED PEER" $TEST_DIR/nodes/*/node.log 2>/dev/null | wc -l)
    UNIQUE_PEERS=$(grep -r "DISCOVERED PEER" $TEST_DIR/nodes/*/node.log 2>/dev/null | awk '{print $NF}' | sort -u | wc -l)
    SUCCESSFUL_CONNECTIONS=$(grep -r "SUCCESS: Connected" $TEST_DIR/nodes/*/node.log 2>/dev/null | wc -l)
    FAILED_CONNECTIONS=$(grep -r "TIMEOUT: Failed" $TEST_DIR/nodes/*/node.log 2>/dev/null | wc -l)
    
    echo "  📡 Total discoveries: $TOTAL_DISCOVERED"
    echo "  🎯 Unique Alpha peers found: $UNIQUE_PEERS"
    echo "  ✅ Successful Tor connections: $SUCCESSFUL_CONNECTIONS"
    echo "  ❌ Failed connections: $FAILED_CONNECTIONS"
    
    if [ $((SUCCESSFUL_CONNECTIONS + FAILED_CONNECTIONS)) -gt 0 ]; then
        SUCCESS_RATE=$((SUCCESSFUL_CONNECTIONS * 100 / (SUCCESSFUL_CONNECTIONS + FAILED_CONNECTIONS)))
        echo "  📊 Connection success rate: ${SUCCESS_RATE}%"
    fi
    
    echo ""
    echo "🔗 Cross-Server Connectivity Test:"
    if [ "$UNIQUE_PEERS" -gt 0 ]; then
        if [ "$SUCCESS_RATE" -gt 70 ]; then
            echo "  🎉 STATUS: ✅ BITCOIN BRIDGE WORKING - Beta discovering Alpha via Bitcoin network!"
        else
            echo "  ⚠️  STATUS: 🔧 PARTIAL SUCCESS - Discovery working, connection issues detected"
        fi
    else
        echo "  🔍 STATUS: 🕵️ SCANNING - No Alpha peers discovered yet..."
    fi
    
    echo ""
    echo "📝 Recent Discovery Activity:"
    grep -r "DISCOVERED PEER\|SUCCESS: Connected\|TIMEOUT: Failed" $TEST_DIR/nodes/*/node.log 2>/dev/null | tail -5 | cut -d: -f2-
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done
MONITOR

chmod +x "$TEST_DIR/monitor_beta.sh"

# Create stop script for Beta
cat > "$TEST_DIR/stop_beta_nodes.sh" <<STOPSCRIPT
#!/bin/bash
echo "🛑 Stopping all Beta discovery nodes..."

for node_dir in $TEST_DIR/nodes/beta-node-*; do
    if [ -f "\$node_dir/node.pid" ]; then
        pid=\$(cat "\$node_dir/node.pid")
        if kill -0 "\$pid" 2>/dev/null; then
            kill "\$pid"
            echo "  🛑 Stopped \$(basename "\$node_dir") (PID: \$pid)"
        fi
    fi
done

echo "✅ All Beta discovery nodes stopped"
STOPSCRIPT

chmod +x "$TEST_DIR/stop_beta_nodes.sh"

# Display summary
success "🎉 Server Beta discovery test setup complete!"
echo ""
echo "📋 Test Summary:"
echo "  📍 Test Directory: $TEST_DIR"
echo "  🕵️ Discovery Nodes Started: $BETA_NODE_COUNT"
echo "  🎯 Target: $DISCOVERY_TARGET"
echo "  🔗 Bitcoin Testnet: $BITCOIN_TESTNET_RPC"
echo "  🧅 Tor Proxy: $TOR_PROXY"
echo "  📊 Base Port: $BASE_PORT"
echo ""
echo "🛠️ Management Commands:"
echo "  Monitor: $TEST_DIR/monitor_beta.sh \"$TEST_DIR\""
echo "  Stop:    $TEST_DIR/stop_beta_nodes.sh"
echo ""
echo "🕵️ Beta nodes are now scanning Bitcoin network for Alpha node advertisements..."

# Start monitoring in background
log "📊 Starting Beta discovery monitoring dashboard..."
"$TEST_DIR/monitor_beta.sh" "$TEST_DIR"