#!/bin/bash
# Bitcoin Network Multi-Node Test Runner
# 
# This script sets up and runs comprehensive Bitcoin network connectivity tests
# for Q-NarwhalKnight nodes to validate peer discovery and consensus operation.

set -e

# Configuration
DEFAULT_NODE_COUNT=8
DEFAULT_TEST_TIMEOUT=300
DEFAULT_LOG_LEVEL="info"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_RESULTS_DIR="$PROJECT_ROOT/test-results/bitcoin-network"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Bitcoin Network Multi-Node Test Runner

USAGE:
    $0 [OPTIONS] [TEST_TYPE]

TEST_TYPES:
    basic       Run basic 6-node connectivity test (default)
    standard    Run standard 8-node full test
    stress      Run stress test with 12+ nodes
    quick       Run quick 4-node validation test
    custom      Run with custom configuration

OPTIONS:
    -n, --nodes COUNT       Number of nodes to test (default: $DEFAULT_NODE_COUNT)
    -t, --timeout SECONDS   Test timeout in seconds (default: $DEFAULT_TEST_TIMEOUT)
    -l, --log-level LEVEL   Log level: trace, debug, info, warn, error (default: $DEFAULT_LOG_LEVEL)
    -r, --results-dir DIR   Directory to store test results (default: $TEST_RESULTS_DIR)
    -b, --bitcoin-rpc HOST:PORT  Bitcoin RPC endpoint (default: 127.0.0.1:18332)
    -k, --keep-nodes        Keep nodes running after test completion
    -v, --verbose           Enable verbose output
    -h, --help             Show this help message

EXAMPLES:
    # Run standard 8-node test
    $0 standard

    # Run stress test with verbose logging
    $0 stress -v -l debug

    # Run custom test with 12 nodes and 10 minute timeout
    $0 custom -n 12 -t 600

    # Run quick validation test
    $0 quick

ENVIRONMENT VARIABLES:
    BITCOIN_RPC_HOST        Bitcoin RPC host (default: 127.0.0.1)
    BITCOIN_RPC_PORT        Bitcoin RPC port (default: 18332)
    BITCOIN_RPC_USER        Bitcoin RPC username
    BITCOIN_RPC_PASS        Bitcoin RPC password
    Q_TEST_TIMEOUT          Override test timeout
    RUST_LOG               Override Rust log level

REQUIREMENTS:
    - Bitcoin Core testnet node running (or access to testnet RPC)
    - Cargo and Rust toolchain
    - Network ports 8000-9000 available
    - At least 4GB RAM for stress tests

EOF
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Cargo is available
    if ! command -v cargo &> /dev/null; then
        error "Cargo not found. Please install Rust toolchain."
        exit 1
    fi

    # Check if project builds
    log "Verifying project builds..."
    if ! cargo check --workspace --quiet; then
        error "Project does not compile. Please fix build errors first."
        exit 1
    fi

    # Check Bitcoin RPC connection (if configured)
    if [[ -n "$BITCOIN_RPC_HOST" && -n "$BITCOIN_RPC_PORT" ]]; then
        log "Testing Bitcoin RPC connection to $BITCOIN_RPC_HOST:$BITCOIN_RPC_PORT..."
        if command -v bitcoin-cli &> /dev/null; then
            if bitcoin-cli -testnet getblockchaininfo &> /dev/null; then
                success "Bitcoin testnet RPC connection verified"
            else
                warning "Bitcoin RPC connection failed - tests will use mock mode"
            fi
        else
            warning "bitcoin-cli not found - cannot verify RPC connection"
        fi
    fi

    # Check available ports
    log "Checking port availability (8000-8100)..."
    local ports_in_use=0
    for port in {8000..8100}; do
        if netstat -ln 2>/dev/null | grep -q ":$port "; then
            ((ports_in_use++))
        fi
    done

    if [ $ports_in_use -gt 10 ]; then
        warning "$ports_in_use ports in test range are in use - tests may fail"
    fi

    success "Prerequisites check complete"
}

# Function to setup test environment
setup_test_environment() {
    log "Setting up test environment..."

    # Create results directory
    mkdir -p "$TEST_RESULTS_DIR"

    # Set environment variables
    export RUST_LOG="${RUST_LOG:-$DEFAULT_LOG_LEVEL}"
    export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
    
    # Bitcoin RPC configuration
    export BITCOIN_RPC_HOST="${BITCOIN_RPC_HOST:-127.0.0.1}"
    export BITCOIN_RPC_PORT="${BITCOIN_RPC_PORT:-18332}"
    
    # Test configuration
    export Q_TEST_RESULTS_DIR="$TEST_RESULTS_DIR"
    export Q_TEST_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

    success "Test environment configured"
}

# Function to run test with specified parameters
run_network_test() {
    local test_name="$1"
    local node_count="$2"
    local timeout="$3"
    local extra_args="$4"

    log "Starting $test_name test with $node_count nodes (${timeout}s timeout)"

    local test_start=$(date +%s)
    local results_file="$TEST_RESULTS_DIR/network_test_${test_name}_${Q_TEST_TIMESTAMP}.json"

    # Build test command
    local test_cmd="cargo test --test bitcoin_network_test"
    
    if [ "$test_name" = "stress" ]; then
        test_cmd="$test_cmd test_bitcoin_network_stress"
    else
        test_cmd="$test_cmd test_bitcoin_network_multi_node"
    fi

    test_cmd="$test_cmd --"
    
    # Add test-specific environment variables
    export Q_TEST_NODE_COUNT="$node_count"
    export Q_TEST_TIMEOUT="$timeout"
    export Q_TEST_RESULTS_FILE="$results_file"

    log "Executing: $test_cmd"
    log "Results will be saved to: $results_file"

    # Run the test
    if eval "$test_cmd" 2>&1 | tee "$TEST_RESULTS_DIR/test_${test_name}_${Q_TEST_TIMESTAMP}.log"; then
        local test_duration=$(($(date +%s) - test_start))
        success "$test_name test completed successfully in ${test_duration}s"
        
        # Extract and display key metrics
        display_test_results "$results_file"
        return 0
    else
        local test_duration=$(($(date +%s) - test_start))
        error "$test_name test failed after ${test_duration}s"
        
        # Still try to display partial results
        if [ -f "$results_file" ]; then
            warning "Displaying partial test results:"
            display_test_results "$results_file"
        fi
        return 1
    fi
}

# Function to display test results
display_test_results() {
    local results_file="$1"
    
    if [ ! -f "$results_file" ]; then
        warning "Results file not found: $results_file"
        return
    fi

    log "Test Results Summary:"
    echo "─────────────────────────────────────────────"
    
    # Parse JSON results (simplified - would use jq in production)
    if command -v jq &> /dev/null; then
        local overall_success=$(jq -r '.overall_success' "$results_file" 2>/dev/null || echo "unknown")
        local node_success=$(jq -r '.node_startup_success_rate' "$results_file" 2>/dev/null || echo "unknown")
        local connectivity=$(jq -r '.network_connectivity_score' "$results_file" 2>/dev/null || echo "unknown")
        local consensus=$(jq -r '.consensus_achieved' "$results_file" 2>/dev/null || echo "unknown")
        local performance=$(jq -r '.performance_score' "$results_file" 2>/dev/null || echo "unknown")

        if [ "$overall_success" = "true" ]; then
            success "Overall Result: PASSED"
        else
            error "Overall Result: FAILED"
        fi

        echo "Node Startup Success: ${node_success}%"
        echo "Network Connectivity: ${connectivity}%"
        echo "Consensus Achieved: $consensus"
        echo "Performance Score: ${performance}/100"
        
        # Show recommendations if available
        local recommendations=$(jq -r '.recommendations[]?' "$results_file" 2>/dev/null)
        if [ -n "$recommendations" ]; then
            echo ""
            log "Recommendations:"
            echo "$recommendations" | sed 's/^/  /'
        fi
    else
        warning "jq not available - displaying raw results file:"
        cat "$results_file"
    fi
    
    echo "─────────────────────────────────────────────"
    log "Full results saved to: $results_file"
}

# Function to cleanup test processes
cleanup_test_environment() {
    log "Cleaning up test environment..."
    
    # Kill any remaining test processes
    pkill -f "q-narwhal" 2>/dev/null || true
    pkill -f "bitcoin_network_test" 2>/dev/null || true
    
    # Wait a moment for cleanup
    sleep 2
    
    success "Cleanup complete"
}

# Function to run predefined test configurations
run_predefined_test() {
    local test_type="$1"
    
    case "$test_type" in
        "quick")
            run_network_test "quick" 4 120
            ;;
        "basic")
            run_network_test "basic" 6 240
            ;;
        "standard")
            run_network_test "standard" 8 300
            ;;
        "stress")
            run_network_test "stress" 12 600
            ;;
        *)
            error "Unknown test type: $test_type"
            show_usage
            exit 1
            ;;
    esac
}

# Parse command line arguments
NODE_COUNT="$DEFAULT_NODE_COUNT"
TIMEOUT="$DEFAULT_TEST_TIMEOUT"
LOG_LEVEL="$DEFAULT_LOG_LEVEL"
RESULTS_DIR="$TEST_RESULTS_DIR"
VERBOSE=false
KEEP_NODES=false
TEST_TYPE="standard"

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nodes)
            NODE_COUNT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -b|--bitcoin-rpc)
            IFS=':' read -r BITCOIN_RPC_HOST BITCOIN_RPC_PORT <<< "$2"
            shift 2
            ;;
        -k|--keep-nodes)
            KEEP_NODES=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            LOG_LEVEL="debug"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        quick|basic|standard|stress|custom)
            TEST_TYPE="$1"
            shift
            ;;
        *)
            error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set verbose mode
if [ "$VERBOSE" = true ]; then
    set -x
fi

# Main execution
main() {
    log "Q-NarwhalKnight Bitcoin Network Test Runner"
    log "Test Type: $TEST_TYPE | Nodes: $NODE_COUNT | Timeout: ${TIMEOUT}s"

    # Setup cleanup trap
    trap cleanup_test_environment EXIT

    # Run prerequisite checks
    check_prerequisites

    # Setup test environment
    setup_test_environment

    # Run the specified test
    case "$TEST_TYPE" in
        "custom")
            run_network_test "custom" "$NODE_COUNT" "$TIMEOUT"
            ;;
        *)
            run_predefined_test "$TEST_TYPE"
            ;;
    esac

    local exit_code=$?

    if [ "$KEEP_NODES" = false ]; then
        cleanup_test_environment
    else
        log "Keeping nodes running as requested (use --cleanup to stop them later)"
    fi

    if [ $exit_code -eq 0 ]; then
        success "All tests completed successfully!"
        log "Test results available in: $RESULTS_DIR"
    else
        error "Some tests failed. Check logs for details."
        log "Test results available in: $RESULTS_DIR"
        exit $exit_code
    fi
}

# Run main function
main "$@"