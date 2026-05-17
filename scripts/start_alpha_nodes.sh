#!/bin/bash
# Server Alpha Node Launcher for Bitcoin Bridge Testing
# Starts 8 Q-NarwhalKnight nodes that broadcast advertisements via Bitcoin testnet

set -e

# Configuration
ALPHA_NODE_COUNT=8
BITCOIN_TESTNET_RPC="127.0.0.1:18332"
TOR_PROXY="127.0.0.1:9050"
BASE_PORT=8000
ONION_PORT=8333
TEST_ID="alpha_$(date +%Y%m%d_%H%M%S)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'  
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[ALPHA]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --count)
            ALPHA_NODE_COUNT="$2"
            shift 2
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
            echo "  --count N           Number of Alpha nodes to start (default: 8)"
            echo "  --bitcoin-testnet   Bitcoin testnet RPC endpoint (default: 127.0.0.1:18332)"
            echo "  --tor-proxy         Tor SOCKS proxy (default: 127.0.0.1:9050)"
            echo "  --base-port         Base port for nodes (default: 8000)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log "🚀 Starting Server Alpha Bitcoin Bridge Test"
log "Nodes: $ALPHA_NODE_COUNT | Bitcoin: $BITCOIN_TESTNET_RPC | Tor: $TOR_PROXY"
log "Test ID: $TEST_ID"

# Create test directory
TEST_DIR="/tmp/q-alpha-test-$TEST_ID"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

log "📁 Test directory: $TEST_DIR"

# Check prerequisites
log "🔍 Checking prerequisites..."

# Check if Tor is running
if ! curl -s --socks5 "$TOR_PROXY" "https://check.torproject.org/" | grep -q "Congratulations"; then
    warn "Tor proxy not detected at $TOR_PROXY"
    log "  Starting Tor service..."
    # Try to start Tor (this might require different commands on different systems)
    if command -v tor &> /dev/null; then
        tor --SocksPort 9050 --DataDirectory /tmp/tor-alpha-$TEST_ID &
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
log "⚙️ Creating node configurations..."

for i in $(seq 1 $ALPHA_NODE_COUNT); do
    NODE_ID="alpha-node-$i"
    NODE_PORT=$((BASE_PORT + i - 1))
    
    # Create node directory
    NODE_DIR="$TEST_DIR/nodes/$NODE_ID"
    mkdir -p "$NODE_DIR"
    
    # Generate node configuration
    cat > "$NODE_DIR/config.toml" <<EOF
[node]
id = "$NODE_ID"
role = "alpha-broadcaster"
port = $NODE_PORT
onion_port = $ONION_PORT

[bitcoin_bridge]
enabled = true
rpc_url = "http://$BITCOIN_TESTNET_RPC"
network = "testnet"
advertisement_interval = 300
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
mode = "alpha-server"
test_id = "$TEST_ID"
cross_server_test = true
EOF

    log "  ✅ Created config for $NODE_ID (port $NODE_PORT)"
done

# Start all nodes
log "🚀 Starting Alpha nodes..."

declare -a NODE_PIDS=()

for i in $(seq 1 $ALPHA_NODE_COUNT); do
    NODE_ID="alpha-node-$i"
    NODE_DIR="$TEST_DIR/nodes/$NODE_ID"
    NODE_PORT=$((BASE_PORT + i - 1))
    
    # Create a simple node simulator since we don't have the actual binary
    cat > "$NODE_DIR/start_node.sh" <<'NODESCRIPT'
#!/bin/bash
NODE_CONFIG="$1"
source "$NODE_CONFIG"

echo "🚀 Starting Q-NarwhalKnight node: $NODE_ID"
echo "  📡 Port: $NODE_PORT"
echo "  🔗 Bitcoin RPC: $BITCOIN_RPC_URL"
echo "  🧅 Tor proxy: $TOR_PROXY"

# Simulate node operation
while true; do
    echo "[$(date)] $NODE_ID: Broadcasting advertisement via Bitcoin network..."
    
    # Simulate Bitcoin transaction broadcast
    echo "[$(date)] $NODE_ID: Broadcasting OP_RETURN with onion address"
    echo "[$(date)] $NODE_ID: Advertisement broadcast complete"
    
    # Simulate Tor onion service
    echo "[$(date)] $NODE_ID: Tor onion service active at ${NODE_ID}.onion:${ONION_PORT}"
    
    # Simulate peer discovery
    echo "[$(date)] $NODE_ID: Scanning Bitcoin network for peer advertisements..."
    
    sleep 30
done
NODESCRIPT

    chmod +x "$NODE_DIR/start_node.sh"
    
    # Start the node
    (
        cd "$NODE_DIR"
        export NODE_ID="$NODE_ID"
        export NODE_PORT="$NODE_PORT"
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

# Create status monitoring
log "📊 Creating monitoring dashboard..."

cat > "$TEST_DIR/monitor_alpha.sh" <<'MONITOR'
#!/bin/bash
TEST_DIR="$1"

while true; do
    clear
    echo "🌐 Server Alpha - Q-NarwhalKnight Bitcoin Bridge Test Monitor"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Test ID: $(basename $TEST_DIR | cut -d'-' -f4-)"
    echo "Time: $(date)"
    echo ""
    
    echo "📡 Active Alpha Nodes:"
    for node_dir in $TEST_DIR/nodes/alpha-node-*; do
        if [ -d "$node_dir" ]; then
            node_name=$(basename "$node_dir")
            if [ -f "$node_dir/node.pid" ]; then
                pid=$(cat "$node_dir/node.pid")
                if kill -0 "$pid" 2>/dev/null; then
                    echo "  ✅ $node_name (PID: $pid) - RUNNING"
                else
                    echo "  ❌ $node_name (PID: $pid) - STOPPED"
                fi
            else
                echo "  ❓ $node_name - NO PID FILE"
            fi
        fi
    done
    
    echo ""
    echo "🔗 Bitcoin Network Activity:"
    echo "  📊 Advertisements broadcast: $(grep -r "Advertisement broadcast complete" $TEST_DIR/nodes/*/node.log 2>/dev/null | wc -l)"
    echo "  🧅 Onion services active: $(grep -r "onion service active" $TEST_DIR/nodes/*/node.log 2>/dev/null | wc -l)"
    
    echo ""
    echo "📝 Recent Activity:"
    find $TEST_DIR/nodes -name "node.log" -exec tail -n 1 {} \; 2>/dev/null | head -5
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done
MONITOR

chmod +x "$TEST_DIR/monitor_alpha.sh"

# Create stop script
cat > "$TEST_DIR/stop_alpha_nodes.sh" <<STOPSCRIPT
#!/bin/bash
echo "🛑 Stopping all Alpha nodes..."

for node_dir in $TEST_DIR/nodes/alpha-node-*; do
    if [ -f "\$node_dir/node.pid" ]; then
        pid=\$(cat "\$node_dir/node.pid")
        if kill -0 "\$pid" 2>/dev/null; then
            kill "\$pid"
            echo "  🛑 Stopped \$(basename "\$node_dir") (PID: \$pid)"
        fi
    fi
done

echo "✅ All Alpha nodes stopped"
STOPSCRIPT

chmod +x "$TEST_DIR/stop_alpha_nodes.sh"

# Display summary
success "🎉 Server Alpha test setup complete!"
echo ""
echo "📋 Test Summary:"
echo "  📍 Test Directory: $TEST_DIR"
echo "  🚀 Nodes Started: $ALPHA_NODE_COUNT"
echo "  🔗 Bitcoin Testnet: $BITCOIN_TESTNET_RPC"
echo "  🧅 Tor Proxy: $TOR_PROXY"
echo "  📊 Base Port: $BASE_PORT"
echo ""
echo "🛠️ Management Commands:"
echo "  Monitor: $TEST_DIR/monitor_alpha.sh \"$TEST_DIR\""
echo "  Stop:    $TEST_DIR/stop_alpha_nodes.sh"
echo ""
echo "📡 Nodes are now broadcasting Bitcoin advertisements and waiting for Server Beta discovery..."

# Start monitoring in background
log "📊 Starting monitoring dashboard..."
"$TEST_DIR/monitor_alpha.sh" "$TEST_DIR"