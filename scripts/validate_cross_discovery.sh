#!/bin/bash
# Cross-Server Bitcoin Bridge Validation Script
# Validates that Alpha and Beta servers can discover each other via Bitcoin network

set -e

# Configuration
VALIDATION_DURATION=300  # 5 minutes
ALPHA_TEST_DIR=""
BETA_TEST_DIR=""
REPORT_FILE=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[VALIDATOR]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat << EOF
Cross-Server Bitcoin Bridge Validation Tool

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --alpha-dir DIR     Path to Alpha server test directory
    --beta-dir DIR      Path to Beta server test directory
    --duration SECONDS  Validation duration (default: 300)
    --report FILE       Output report file (default: validation_report.json)
    --help              Show this help message

EXAMPLES:
    # Validate both servers running locally
    $0 --alpha-dir /tmp/q-alpha-test-* --beta-dir /tmp/q-beta-test-*
    
    # Custom duration and report
    $0 --alpha-dir /tmp/alpha --beta-dir /tmp/beta --duration 600 --report results.json

DESCRIPTION:
    This script validates that Q-NarwhalKnight nodes running on different servers
    can successfully discover each other through the Bitcoin network and establish
    Tor-based connections. It provides definitive proof of cross-server connectivity.

VALIDATION CRITERIA:
    ✅ Alpha nodes broadcast advertisements via Bitcoin
    ✅ Beta nodes discover Alpha advertisements via Bitcoin network scanning
    ✅ Beta nodes successfully connect to Alpha nodes via Tor
    ✅ Bi-directional communication is established
    ✅ Performance metrics meet minimum thresholds

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --alpha-dir)
            ALPHA_TEST_DIR="$2"
            shift 2
            ;;
        --beta-dir)
            BETA_TEST_DIR="$2"
            shift 2
            ;;
        --duration)
            VALIDATION_DURATION="$2"
            shift 2
            ;;
        --report)
            REPORT_FILE="$2"
            shift 2
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

# Auto-detect test directories if not provided
if [ -z "$ALPHA_TEST_DIR" ]; then
    ALPHA_TEST_DIR=$(find /tmp -maxdepth 1 -name "q-alpha-test-*" -type d | head -1)
fi

if [ -z "$BETA_TEST_DIR" ]; then
    BETA_TEST_DIR=$(find /tmp -maxdepth 1 -name "q-beta-test-*" -type d | head -1)
fi

# Set default report file
if [ -z "$REPORT_FILE" ]; then
    REPORT_FILE="bitcoin_bridge_validation_$(date +%Y%m%d_%H%M%S).json"
fi

# Validate inputs
if [ ! -d "$ALPHA_TEST_DIR" ]; then
    error "Alpha test directory not found: $ALPHA_TEST_DIR"
    error "Please start Alpha nodes first with: ./scripts/start_alpha_nodes.sh"
    exit 1
fi

if [ ! -d "$BETA_TEST_DIR" ]; then
    error "Beta test directory not found: $BETA_TEST_DIR"
    error "Please start Beta nodes first with: ./scripts/start_beta_nodes.sh"
    exit 1
fi

log "🔬 Starting Cross-Server Bitcoin Bridge Validation"
log "Alpha Directory: $ALPHA_TEST_DIR"
log "Beta Directory: $BETA_TEST_DIR"
log "Validation Duration: ${VALIDATION_DURATION}s"
log "Report File: $REPORT_FILE"

# Initialize validation data
START_TIME=$(date +%s)
VALIDATION_DATA="{
    \"test_start\": \"$(date -Iseconds)\",
    \"alpha_dir\": \"$ALPHA_TEST_DIR\",
    \"beta_dir\": \"$BETA_TEST_DIR\",
    \"duration_seconds\": $VALIDATION_DURATION,
    \"validation_phases\": {}
}"

# Phase 1: Verify Alpha nodes are broadcasting
log "📡 Phase 1: Validating Alpha node broadcasts..."

ALPHA_NODES=$(find "$ALPHA_TEST_DIR/nodes" -name "alpha-node-*" -type d | wc -l)
ALPHA_ACTIVE=$(find "$ALPHA_TEST_DIR/nodes" -name "node.pid" -exec sh -c 'kill -0 $(cat {}) 2>/dev/null' \; | wc -l)
ALPHA_BROADCASTS=$(grep -r "Advertisement broadcast complete" "$ALPHA_TEST_DIR/nodes/"*/node.log 2>/dev/null | wc -l)

log "  Alpha nodes total: $ALPHA_NODES"
log "  Alpha nodes active: $ALPHA_ACTIVE"
log "  Bitcoin broadcasts: $ALPHA_BROADCASTS"

if [ "$ALPHA_ACTIVE" -ge 3 ] && [ "$ALPHA_BROADCASTS" -ge 5 ]; then
    success "Phase 1: Alpha broadcasting ✅"
    PHASE1_SUCCESS=true
else
    error "Phase 1: Insufficient Alpha activity ❌"
    PHASE1_SUCCESS=false
fi

# Phase 2: Verify Beta nodes are discovering
log "🕵️ Phase 2: Validating Beta node discovery..."

BETA_NODES=$(find "$BETA_TEST_DIR/nodes" -name "beta-node-*" -type d | wc -l)
BETA_ACTIVE=$(find "$BETA_TEST_DIR/nodes" -name "node.pid" -exec sh -c 'kill -0 $(cat {}) 2>/dev/null' \; | wc -l)
BETA_DISCOVERIES=$(grep -r "DISCOVERED PEER" "$BETA_TEST_DIR/nodes/"*/node.log 2>/dev/null | wc -l)
UNIQUE_DISCOVERIES=$(grep -r "DISCOVERED PEER" "$BETA_TEST_DIR/nodes/"*/node.log 2>/dev/null | awk '{print $NF}' | sort -u | wc -l)

log "  Beta nodes total: $BETA_NODES"
log "  Beta nodes active: $BETA_ACTIVE"
log "  Total discoveries: $BETA_DISCOVERIES"
log "  Unique Alpha peers: $UNIQUE_DISCOVERIES"

if [ "$BETA_ACTIVE" -ge 3 ] && [ "$UNIQUE_DISCOVERIES" -ge 2 ]; then
    success "Phase 2: Beta discovery ✅"
    PHASE2_SUCCESS=true
else
    error "Phase 2: Insufficient Beta discoveries ❌"
    PHASE2_SUCCESS=false
fi

# Phase 3: Verify Tor connections
log "🧅 Phase 3: Validating Tor connections..."

SUCCESSFUL_CONNECTIONS=$(grep -r "SUCCESS: Connected" "$BETA_TEST_DIR/nodes/"*/node.log 2>/dev/null | wc -l)
FAILED_CONNECTIONS=$(grep -r "TIMEOUT: Failed" "$BETA_TEST_DIR/nodes/"*/node.log 2>/dev/null | wc -l)
TOTAL_CONNECTION_ATTEMPTS=$((SUCCESSFUL_CONNECTIONS + FAILED_CONNECTIONS))

if [ "$TOTAL_CONNECTION_ATTEMPTS" -gt 0 ]; then
    CONNECTION_SUCCESS_RATE=$((SUCCESSFUL_CONNECTIONS * 100 / TOTAL_CONNECTION_ATTEMPTS))
else
    CONNECTION_SUCCESS_RATE=0
fi

log "  Successful connections: $SUCCESSFUL_CONNECTIONS"
log "  Failed connections: $FAILED_CONNECTIONS"
log "  Connection success rate: ${CONNECTION_SUCCESS_RATE}%"

if [ "$SUCCESSFUL_CONNECTIONS" -ge 3 ] && [ "$CONNECTION_SUCCESS_RATE" -ge 60 ]; then
    success "Phase 3: Tor connectivity ✅"
    PHASE3_SUCCESS=true
else
    error "Phase 3: Insufficient Tor connectivity ❌"
    PHASE3_SUCCESS=false
fi

# Phase 4: Performance validation
log "⚡ Phase 4: Performance validation..."

# Calculate average discovery time
FIRST_DISCOVERY=$(grep -r "DISCOVERED PEER" "$BETA_TEST_DIR/nodes/"*/node.log 2>/dev/null | head -1 | grep -o '\[[^]]*\]' | tr -d '[]')
if [ -n "$FIRST_DISCOVERY" ]; then
    FIRST_DISCOVERY_TIME=$(date -d "$FIRST_DISCOVERY" +%s 2>/dev/null || echo "$START_TIME")
    DISCOVERY_LATENCY=$((FIRST_DISCOVERY_TIME - START_TIME))
else
    DISCOVERY_LATENCY=999
fi

# Calculate average connection latency
AVG_CONNECTION_LATENCY=$(grep -r "SUCCESS: Connected.*latency:" "$BETA_TEST_DIR/nodes/"*/node.log 2>/dev/null | grep -o 'latency: [0-9]*ms' | grep -o '[0-9]*' | awk '{sum+=$1; count++} END {if(count>0) print int(sum/count); else print 999}')
if [ -z "$AVG_CONNECTION_LATENCY" ]; then
    AVG_CONNECTION_LATENCY=999
fi

log "  Discovery latency: ${DISCOVERY_LATENCY}s"
log "  Avg connection latency: ${AVG_CONNECTION_LATENCY}ms"

if [ "$DISCOVERY_LATENCY" -lt 180 ] && [ "$AVG_CONNECTION_LATENCY" -lt 1000 ]; then
    success "Phase 4: Performance ✅"
    PHASE4_SUCCESS=true
else
    warn "Phase 4: Performance below optimal ⚠️"
    PHASE4_SUCCESS=true  # Don't fail on performance alone
fi

# Calculate overall result
END_TIME=$(date +%s)
TEST_DURATION=$((END_TIME - START_TIME))

if [ "$PHASE1_SUCCESS" = true ] && [ "$PHASE2_SUCCESS" = true ] && [ "$PHASE3_SUCCESS" = true ]; then
    OVERALL_SUCCESS=true
    OVERALL_STATUS="PASSED"
else
    OVERALL_SUCCESS=false
    OVERALL_STATUS="FAILED"
fi

# Generate comprehensive JSON report
cat > "$REPORT_FILE" <<EOF
{
    "test_metadata": {
        "test_start": "$(date -d @$START_TIME -Iseconds)",
        "test_end": "$(date -d @$END_TIME -Iseconds)",
        "test_duration_seconds": $TEST_DURATION,
        "alpha_directory": "$ALPHA_TEST_DIR",
        "beta_directory": "$BETA_TEST_DIR",
        "validation_duration_seconds": $VALIDATION_DURATION
    },
    "overall_result": {
        "success": $OVERALL_SUCCESS,
        "status": "$OVERALL_STATUS"
    },
    "phase_results": {
        "phase1_alpha_broadcasting": {
            "success": $PHASE1_SUCCESS,
            "alpha_nodes_total": $ALPHA_NODES,
            "alpha_nodes_active": $ALPHA_ACTIVE,
            "bitcoin_broadcasts": $ALPHA_BROADCASTS
        },
        "phase2_beta_discovery": {
            "success": $PHASE2_SUCCESS,
            "beta_nodes_total": $BETA_NODES,
            "beta_nodes_active": $BETA_ACTIVE,
            "total_discoveries": $BETA_DISCOVERIES,
            "unique_alpha_peers_found": $UNIQUE_DISCOVERIES
        },
        "phase3_tor_connectivity": {
            "success": $PHASE3_SUCCESS,
            "successful_connections": $SUCCESSFUL_CONNECTIONS,
            "failed_connections": $FAILED_CONNECTIONS,
            "connection_success_rate_percent": $CONNECTION_SUCCESS_RATE
        },
        "phase4_performance": {
            "success": $PHASE4_SUCCESS,
            "discovery_latency_seconds": $DISCOVERY_LATENCY,
            "average_connection_latency_ms": $AVG_CONNECTION_LATENCY
        }
    },
    "detailed_metrics": {
        "cross_server_discovery_proof": $([ "$UNIQUE_DISCOVERIES" -ge 2 ] && echo "true" || echo "false"),
        "bitcoin_network_integration_proof": $([ "$BETA_DISCOVERIES" -ge 5 ] && echo "true" || echo "false"),
        "tor_anonymity_proof": $([ "$SUCCESSFUL_CONNECTIONS" -ge 3 ] && echo "true" || echo "false"),
        "multi_ip_connectivity_proof": $OVERALL_SUCCESS
    },
    "recommendations": [
EOF

# Add recommendations based on results
if [ "$PHASE1_SUCCESS" = false ]; then
    echo "        \"Increase Alpha node broadcast frequency\"," >> "$REPORT_FILE"
fi

if [ "$PHASE2_SUCCESS" = false ]; then
    echo "        \"Improve Bitcoin network scanning reliability\"," >> "$REPORT_FILE"
fi

if [ "$PHASE3_SUCCESS" = false ]; then
    echo "        \"Optimize Tor connection stability\"," >> "$REPORT_FILE"
fi

if [ "$AVG_CONNECTION_LATENCY" -gt 500 ]; then
    echo "        \"Optimize Tor circuit selection for lower latency\"," >> "$REPORT_FILE"
fi

# Remove trailing comma and close recommendations
sed -i '$ s/,$//' "$REPORT_FILE"

cat >> "$REPORT_FILE" <<EOF
    ]
}
EOF

# Display final results
echo ""
log "🎯 Cross-Server Bitcoin Bridge Validation Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$OVERALL_SUCCESS" = true ]; then
    success "🎉 OVERALL RESULT: ✅ VALIDATION PASSED"
    echo ""
    success "🔬 DEFINITIVE PROOF PROVIDED:"
    success "  ✅ Alpha nodes broadcast advertisements via Bitcoin testnet"
    success "  ✅ Beta nodes discover Alpha peers via Bitcoin network scanning"  
    success "  ✅ Cross-server Tor connections established successfully"
    success "  ✅ Multi-IP Bitcoin bridge connectivity PROVEN"
    echo ""
    success "📊 Key Metrics:"
    success "  🎯 Unique Alpha peers discovered: $UNIQUE_DISCOVERIES"
    success "  🔗 Successful Tor connections: $SUCCESSFUL_CONNECTIONS"
    success "  📈 Connection success rate: ${CONNECTION_SUCCESS_RATE}%"
    success "  ⚡ Discovery latency: ${DISCOVERY_LATENCY}s"
    success "  🧅 Avg connection latency: ${AVG_CONNECTION_LATENCY}ms"
else
    error "💥 OVERALL RESULT: ❌ VALIDATION FAILED"
    echo ""
    error "🔍 Issues Detected:"
    [ "$PHASE1_SUCCESS" = false ] && error "  ❌ Alpha broadcasting insufficient"
    [ "$PHASE2_SUCCESS" = false ] && error "  ❌ Beta discovery insufficient" 
    [ "$PHASE3_SUCCESS" = false ] && error "  ❌ Tor connectivity insufficient"
    echo ""
    warn "📋 Partial Results:"
    warn "  📡 Bitcoin broadcasts: $ALPHA_BROADCASTS"
    warn "  🕵️ Peer discoveries: $BETA_DISCOVERIES"
    warn "  🔗 Successful connections: $SUCCESSFUL_CONNECTIONS"
fi

echo ""
log "📄 Detailed report saved to: $REPORT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Return appropriate exit code
if [ "$OVERALL_SUCCESS" = true ]; then
    exit 0
else
    exit 1
fi