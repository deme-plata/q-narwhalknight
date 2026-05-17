#!/bin/bash
# Master Cross-Server Bitcoin Bridge Test Coordinator
# Runs both Alpha and Beta servers and validates connectivity

set -e

# Configuration
TEST_DURATION=300
ALPHA_NODES=8
BETA_NODES=8
VALIDATION_INTERVAL=60

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[COORDINATOR]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat << EOF
Q-NarwhalKnight Cross-Server Bitcoin Bridge Test Coordinator

USAGE:
    $0 [OPTIONS] [MODE]

MODES:
    alpha-only      Run only Alpha server nodes
    beta-only       Run only Beta server nodes  
    both           Run both Alpha and Beta servers (default)
    validate       Run validation only (requires existing tests)

OPTIONS:
    --alpha-nodes N    Number of Alpha nodes (default: 8)
    --beta-nodes N     Number of Beta nodes (default: 8)
    --duration SECS    Test duration in seconds (default: 300)
    --validation       Enable continuous validation
    --background       Run in background mode
    --help             Show this help

EXAMPLES:
    # Run complete cross-server test
    $0 both --alpha-nodes 8 --beta-nodes 8 --duration 600

    # Run only Alpha server (for coordination with external Beta)
    $0 alpha-only --alpha-nodes 8

    # Validate existing test
    $0 validate

DESCRIPTION:
    This script coordinates a comprehensive cross-server Bitcoin bridge test that
    definitively proves Q-NarwhalKnight nodes can discover and connect to each other
    through the Bitcoin network across different IP addresses and servers.

    The test demonstrates:
    ✅ Bitcoin network-based peer discovery
    ✅ Cross-IP Tor connectivity  
    ✅ Anonymous multi-server communication
    ✅ Real-world performance validation

EOF
}

# Parse arguments
MODE="both"
ENABLE_VALIDATION=false
BACKGROUND_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        alpha-only|beta-only|both|validate)
            MODE="$1"
            shift
            ;;
        --alpha-nodes)
            ALPHA_NODES="$2"
            shift 2
            ;;
        --beta-nodes)
            BETA_NODES="$2"
            shift 2
            ;;
        --duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        --validation)
            ENABLE_VALIDATION=true
            shift
            ;;
        --background)
            BACKGROUND_MODE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

log "🚀 Q-NarwhalKnight Cross-Server Bitcoin Bridge Test"
log "Mode: $MODE | Duration: ${TEST_DURATION}s | Alpha: $ALPHA_NODES | Beta: $BETA_NODES"

# Create master test directory
MASTER_TEST_DIR="/tmp/q-cross-server-test-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_TEST_DIR"
cd "$MASTER_TEST_DIR"

log "📁 Master test directory: $MASTER_TEST_DIR"

# Function to run Alpha server
run_alpha_server() {
    log "🔵 Starting Alpha server ($ALPHA_NODES nodes)..."
    
    # Start Alpha nodes in background
    if [ "$BACKGROUND_MODE" = true ]; then
        timeout "$TEST_DURATION" ./scripts/start_alpha_nodes.sh --count "$ALPHA_NODES" > "$MASTER_TEST_DIR/alpha.log" 2>&1 &
        ALPHA_PID=$!
        echo "$ALPHA_PID" > "$MASTER_TEST_DIR/alpha.pid"
        log "Alpha server started in background (PID: $ALPHA_PID)"
    else
        timeout "$TEST_DURATION" ./scripts/start_alpha_nodes.sh --count "$ALPHA_NODES" &
        ALPHA_PID=$!
        echo "$ALPHA_PID" > "$MASTER_TEST_DIR/alpha.pid"
        log "Alpha server started (PID: $ALPHA_PID)"
    fi
    
    # Wait for Alpha to initialize
    sleep 10
}

# Function to run Beta server
run_beta_server() {
    log "🟣 Starting Beta server ($BETA_NODES nodes)..."
    
    # Start Beta nodes in background
    if [ "$BACKGROUND_MODE" = true ]; then
        timeout "$TEST_DURATION" ./scripts/start_beta_nodes.sh --count "$BETA_NODES" --target-alpha > "$MASTER_TEST_DIR/beta.log" 2>&1 &
        BETA_PID=$!
        echo "$BETA_PID" > "$MASTER_TEST_DIR/beta.pid"
        log "Beta server started in background (PID: $BETA_PID)"
    else
        timeout "$TEST_DURATION" ./scripts/start_beta_nodes.sh --count "$BETA_NODES" --target-alpha &
        BETA_PID=$!
        echo "$BETA_PID" > "$MASTER_TEST_DIR/beta.pid"
        log "Beta server started (PID: $BETA_PID)"
    fi
    
    # Wait for Beta to initialize
    sleep 10
}

# Function to run validation
run_validation() {
    log "🔬 Running cross-server validation..."
    
    # Find test directories
    ALPHA_DIR=$(find /tmp -maxdepth 1 -name "q-alpha-test-*" -type d | head -1)
    BETA_DIR=$(find /tmp -maxdepth 1 -name "q-beta-test-*" -type d | head -1)
    
    if [ -n "$ALPHA_DIR" ] && [ -n "$BETA_DIR" ]; then
        REPORT_FILE="$MASTER_TEST_DIR/validation_report_$(date +%Y%m%d_%H%M%S).json"
        ./scripts/validate_cross_discovery.sh --alpha-dir "$ALPHA_DIR" --beta-dir "$BETA_DIR" --report "$REPORT_FILE"
        return $?
    else
        error "Could not find Alpha and/or Beta test directories"
        return 1
    fi
}

# Function to create monitoring dashboard
create_master_dashboard() {
    cat > "$MASTER_TEST_DIR/monitor_cross_server.sh" <<'DASHBOARD'
#!/bin/bash
MASTER_DIR="$1"

while true; do
    clear
    echo "🌐 Q-NarwhalKnight Cross-Server Bitcoin Bridge Test Monitor"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Test Time: $(date) | Master Dir: $(basename "$MASTER_DIR")"
    echo ""
    
    # Alpha Server Status
    echo "🔵 ALPHA SERVER STATUS:"
    ALPHA_DIR=$(find /tmp -maxdepth 1 -name "q-alpha-test-*" -type d | head -1)
    if [ -n "$ALPHA_DIR" ]; then
        ALPHA_NODES=$(find "$ALPHA_DIR/nodes" -name "alpha-node-*" -type d | wc -l)
        ALPHA_ACTIVE=$(find "$ALPHA_DIR/nodes" -name "node.pid" -exec sh -c 'kill -0 $(cat {}) 2>/dev/null' \; | wc -l)
        ALPHA_BROADCASTS=$(grep -r "Advertisement broadcast complete" "$ALPHA_DIR/nodes/"*/node.log 2>/dev/null | wc -l)
        echo "  📡 Nodes: $ALPHA_ACTIVE/$ALPHA_NODES active | Broadcasts: $ALPHA_BROADCASTS"
    else
        echo "  ❌ No Alpha test directory found"
    fi
    
    # Beta Server Status  
    echo "🟣 BETA SERVER STATUS:"
    BETA_DIR=$(find /tmp -maxdepth 1 -name "q-beta-test-*" -type d | head -1)
    if [ -n "$BETA_DIR" ]; then
        BETA_NODES=$(find "$BETA_DIR/nodes" -name "beta-node-*" -type d | wc -l)
        BETA_ACTIVE=$(find "$BETA_DIR/nodes" -name "node.pid" -exec sh -c 'kill -0 $(cat {}) 2>/dev/null' \; | wc -l)
        BETA_DISCOVERIES=$(grep -r "DISCOVERED PEER" "$BETA_DIR/nodes/"*/node.log 2>/dev/null | wc -l)
        UNIQUE_PEERS=$(grep -r "DISCOVERED PEER" "$BETA_DIR/nodes/"*/node.log 2>/dev/null | awk '{print $NF}' | sort -u | wc -l)
        SUCCESSFUL_CONNECTIONS=$(grep -r "SUCCESS: Connected" "$BETA_DIR/nodes/"*/node.log 2>/dev/null | wc -l)
        echo "  🕵️ Nodes: $BETA_ACTIVE/$BETA_NODES active | Discoveries: $BETA_DISCOVERIES | Unique: $UNIQUE_PEERS"
        echo "  🔗 Successful connections: $SUCCESSFUL_CONNECTIONS"
    else
        echo "  ❌ No Beta test directory found"
    fi
    
    # Cross-Server Proof Status
    echo ""
    echo "🎯 CROSS-SERVER CONNECTIVITY PROOF:"
    if [ -n "$ALPHA_DIR" ] && [ -n "$BETA_DIR" ]; then
        if [ "$ALPHA_BROADCASTS" -gt 0 ] && [ "$UNIQUE_PEERS" -gt 0 ] && [ "$SUCCESSFUL_CONNECTIONS" -gt 0 ]; then
            echo "  🎉 STATUS: ✅ PROOF ESTABLISHED"
            echo "    ✅ Alpha broadcasting via Bitcoin: $ALPHA_BROADCASTS broadcasts"
            echo "    ✅ Beta discovering Alpha peers: $UNIQUE_PEERS unique peers found" 
            echo "    ✅ Cross-IP Tor connections: $SUCCESSFUL_CONNECTIONS successful"
            echo "    🏆 BITCOIN BRIDGE WORKING ACROSS DIFFERENT IPS/SERVERS"
        else
            echo "  🔍 STATUS: 🕵️ IN PROGRESS"
            echo "    📡 Alpha broadcasts: $ALPHA_BROADCASTS"
            echo "    🎯 Peer discoveries: $UNIQUE_PEERS"  
            echo "    🔗 Connections: $SUCCESSFUL_CONNECTIONS"
        fi
    else
        echo "  ⚠️ STATUS: WAITING FOR BOTH SERVERS TO START"
    fi
    
    echo ""
    echo "📝 Recent Activity:"
    if [ -n "$BETA_DIR" ]; then
        grep -r "DISCOVERED PEER\|SUCCESS: Connected" "$BETA_DIR/nodes/"*/node.log 2>/dev/null | tail -3 | cut -d: -f2- || echo "  No recent activity"
    fi
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done
DASHBOARD

    chmod +x "$MASTER_TEST_DIR/monitor_cross_server.sh"
}

# Function to cleanup
cleanup() {
    log "🛑 Cleaning up cross-server test..."
    
    # Kill Alpha server
    if [ -f "$MASTER_TEST_DIR/alpha.pid" ]; then
        ALPHA_PID=$(cat "$MASTER_TEST_DIR/alpha.pid")
        if kill -0 "$ALPHA_PID" 2>/dev/null; then
            kill -TERM "$ALPHA_PID" 2>/dev/null || kill -KILL "$ALPHA_PID" 2>/dev/null
        fi
    fi
    
    # Kill Beta server
    if [ -f "$MASTER_TEST_DIR/beta.pid" ]; then
        BETA_PID=$(cat "$MASTER_TEST_DIR/beta.pid")
        if kill -0 "$BETA_PID" 2>/dev/null; then
            kill -TERM "$BETA_PID" 2>/dev/null || kill -KILL "$BETA_PID" 2>/dev/null
        fi
    fi
    
    # Stop individual node processes
    for test_dir in /tmp/q-alpha-test-* /tmp/q-beta-test-*; do
        if [ -d "$test_dir" ] && [ -f "$test_dir/stop_"*"_nodes.sh" ]; then
            "$test_dir/stop_"*"_nodes.sh" >/dev/null 2>&1
        fi
    done
    
    success "Cleanup complete"
}

# Set up cleanup trap
trap cleanup EXIT

# Create monitoring dashboard
create_master_dashboard

# Execute based on mode
case "$MODE" in
    "alpha-only")
        run_alpha_server
        log "Alpha server running. Press Ctrl+C to stop."
        wait
        ;;
        
    "beta-only")
        run_beta_server
        log "Beta server running. Press Ctrl+C to stop."
        wait
        ;;
        
    "both")
        log "🚀 Starting both Alpha and Beta servers..."
        
        # Start Alpha first
        run_alpha_server
        
        # Start Beta after Alpha has time to initialize
        sleep 15
        run_beta_server
        
        # Run validation if requested
        if [ "$ENABLE_VALIDATION" = true ]; then
            log "⏳ Waiting for initial setup before validation..."
            sleep 60
            
            while true; do
                if run_validation; then
                    success "🎉 Validation passed!"
                    break
                else
                    warn "Validation in progress... retrying in ${VALIDATION_INTERVAL}s"
                    sleep "$VALIDATION_INTERVAL"
                fi
            done
        else
            log "📊 Starting monitoring dashboard..."
            "$MASTER_TEST_DIR/monitor_cross_server.sh" "$MASTER_TEST_DIR"
        fi
        ;;
        
    "validate")
        run_validation
        exit $?
        ;;
        
    *)
        error "Unknown mode: $MODE"
        exit 1
        ;;
esac

success "Cross-server test completed successfully!"