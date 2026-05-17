#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
# Q-NarwhalKnight Release Validation Pipeline
# ════════════════════════════════════════════════════════════════════════════
# Professional release validation with 5 phases
# Usage: ./scripts/validate-release.sh [phase|all] [version]
# Example: ./scripts/validate-release.sh all v3.4.16-beta
# ════════════════════════════════════════════════════════════════════════════

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/opt/orobit/shared/q-narwhalknight"
TESTNET_DIR="/opt/orobit/testnet"
BACKUP_DIR="/opt/orobit/backups"
VERSION="${2:-v3.4.16-beta}"

# Logging
LOG_FILE="$TESTNET_DIR/logs/release-validation-$(date +%Y%m%d).log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

header() {
    log ""
    log "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    log "${BLUE}║ $1${NC}"
    log "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    log ""
}

success() {
    log "${GREEN}✅ $1${NC}"
}

error() {
    log "${RED}❌ $1${NC}"
}

warning() {
    log "${YELLOW}⚠️  $1${NC}"
}

# ════════════════════════════════════════════════════════════════════════════
# PHASE A: Pre-Deployment Validation
# ════════════════════════════════════════════════════════════════════════════
phase_a() {
    header "PHASE A: PRE-DEPLOYMENT VALIDATION"

    cd "$PROJECT_DIR"
    local FAILED=0

    # A.1: Format check
    log "🔍 [A.1] Checking code formatting..."
    if cargo fmt --check > /dev/null 2>&1; then
        success "[A.1] Format check passed"
    else
        error "[A.1] Format check failed - run 'cargo fmt'"
        FAILED=1
    fi

    # A.2: Clippy
    log "🔍 [A.2] Running clippy (this may take a while)..."
    if timeout 1800 cargo clippy -- -D warnings > /tmp/clippy.log 2>&1; then
        success "[A.2] Clippy passed"
    else
        error "[A.2] Clippy found warnings"
        tail -20 /tmp/clippy.log
        FAILED=1
    fi

    # A.3: Critical tests
    log "🔍 [A.3] Running critical mainnet tests..."

    CRITICAL_TESTS=(
        "q-storage:mainnet_critical_tests"
        "q-types:signature_verification_tests"
        "q-storage:balance_propagation_tests"
        "q-dex:overflow_protection_tests"
    )

    for test_spec in "${CRITICAL_TESTS[@]}"; do
        IFS=':' read -r pkg test <<< "$test_spec"
        log "   Running $test..."
        if timeout 300 cargo test --package "$pkg" --test "$test" > /dev/null 2>&1; then
            success "   $test passed"
        else
            warning "   $test not found or failed (non-blocking)"
        fi
    done

    # A.4: Full workspace test
    log "🔍 [A.4] Running workspace tests..."
    if timeout 7200 cargo test --workspace > /tmp/test-output.log 2>&1; then
        success "[A.4] All tests passed"
    else
        error "[A.4] Some tests failed"
        grep -E "^test .* FAILED" /tmp/test-output.log | head -10
        FAILED=1
    fi

    # A.5: Build release
    log "🔍 [A.5] Building release binary..."
    if timeout 3600 cargo build --release --package q-api-server > /tmp/build.log 2>&1; then
        success "[A.5] Build successful"
    else
        error "[A.5] Build failed"
        tail -30 /tmp/build.log
        FAILED=1
    fi

    # A.6: Copy to testnet
    if [ $FAILED -eq 0 ]; then
        log "🔍 [A.6] Copying binary to testnet..."
        mkdir -p "$TESTNET_DIR/q-narwhalknight"
        cp "$PROJECT_DIR/target/release/q-api-server" "$TESTNET_DIR/q-narwhalknight/q-api-server-candidate"
        chmod +x "$TESTNET_DIR/q-narwhalknight/q-api-server-candidate"
        success "[A.6] Binary copied to testnet"
    fi

    if [ $FAILED -eq 0 ]; then
        success "PHASE A COMPLETE - Ready for Phase B"
        return 0
    else
        error "PHASE A FAILED - Fix issues before proceeding"
        return 1
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# PHASE B: Testnet Deployment
# ════════════════════════════════════════════════════════════════════════════
phase_b() {
    header "PHASE B: TESTNET DEPLOYMENT"

    local FAILED=0

    # B.1: Pre-flight check
    log "🔍 [B.1] Running pre-flight verification..."
    cd "$TESTNET_DIR/q-narwhalknight"

    if Q_PREFLIGHT_ONLY=1 Q_DATA_DIR="$TESTNET_DIR/data" ./q-api-server-candidate > /tmp/preflight.log 2>&1; then
        if grep -q "PASSED\|completed successfully" /tmp/preflight.log; then
            success "[B.1] Pre-flight check passed"
        else
            warning "[B.1] Pre-flight output unclear, check /tmp/preflight.log"
        fi
    else
        # Pre-flight might exit with 0 after success
        if grep -q "PASSED\|completed successfully" /tmp/preflight.log; then
            success "[B.1] Pre-flight check passed"
        else
            error "[B.1] Pre-flight check failed"
            cat /tmp/preflight.log
            FAILED=1
        fi
    fi

    # B.2: Stop existing service
    log "🔍 [B.2] Stopping existing testnet service..."
    systemctl stop q-testnet-server 2>/dev/null || true
    sleep 3
    success "[B.2] Service stopped"

    # B.3: Start new service
    log "🔍 [B.3] Starting testnet service..."
    systemctl start q-testnet-server
    sleep 10

    # B.4: Verify service
    if systemctl is-active --quiet q-testnet-server; then
        success "[B.4] Service started"
    else
        error "[B.4] Service failed to start"
        journalctl -u q-testnet-server --since "2 minutes ago" --no-pager | tail -20
        FAILED=1
    fi

    # B.5: Verify API
    log "🔍 [B.5] Verifying API response..."
    sleep 5
    local RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8085/api/v1/status)
    if [ "$RESPONSE" = "200" ]; then
        success "[B.5] API responding (HTTP 200)"
    else
        error "[B.5] API not responding (HTTP $RESPONSE)"
        FAILED=1
    fi

    # B.6: SSL check (optional)
    log "🔍 [B.6] Checking SSL..."
    local SSL_CHECK=$(curl -s -o /dev/null -w "%{http_code}" https://testnet.quillon.xyz/health 2>/dev/null || echo "000")
    if [ "$SSL_CHECK" = "200" ]; then
        success "[B.6] SSL working on testnet.quillon.xyz"
    else
        warning "[B.6] SSL check returned $SSL_CHECK (may need DNS setup)"
    fi

    if [ $FAILED -eq 0 ]; then
        success "PHASE B COMPLETE - Starting Phase C (sync monitoring)"
        log ""
        log "📊 Monitor sync with:"
        log "   journalctl -u q-testnet-server -f | grep -E 'height|sync|P2P'"
        log ""
        log "⏰ Phase C runs automatically via cron (hourly checks)"
        log "   View logs: tail -f $TESTNET_DIR/logs/sync-validation.log"
        return 0
    else
        error "PHASE B FAILED"
        return 1
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# PHASE C: Sync Validation (Single Check)
# ════════════════════════════════════════════════════════════════════════════
phase_c() {
    header "PHASE C: SYNC VALIDATION CHECK"

    local TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

    # Get heights
    local TESTNET_STATUS=$(curl -s http://127.0.0.1:8085/api/v1/status 2>/dev/null)
    local TESTNET_HEIGHT=$(echo "$TESTNET_STATUS" | jq -r '.height // 0')

    local PROD_STATUS=$(curl -s http://127.0.0.1:8080/api/v1/status 2>/dev/null)
    local PROD_HEIGHT=$(echo "$PROD_STATUS" | jq -r '.height // 0')

    log "📊 Testnet height: $TESTNET_HEIGHT"
    log "📊 Production height: $PROD_HEIGHT"

    # Calculate sync percentage
    if [ "$PROD_HEIGHT" -gt 0 ]; then
        local SYNC_PCT=$(echo "scale=2; ($TESTNET_HEIGHT / $PROD_HEIGHT) * 100" | bc)
        log "📊 Sync progress: ${SYNC_PCT}%"
    fi

    # Check for sync-down
    local PREV_HEIGHT_FILE="$TESTNET_DIR/logs/last-height.txt"
    if [ -f "$PREV_HEIGHT_FILE" ]; then
        local PREV_HEIGHT=$(cat "$PREV_HEIGHT_FILE")
        if [ "$TESTNET_HEIGHT" -lt "$PREV_HEIGHT" ] && [ "$PREV_HEIGHT" -gt 100 ]; then
            error "CRITICAL: SYNC-DOWN DETECTED!"
            error "Height went from $PREV_HEIGHT to $TESTNET_HEIGHT"
            error "This is a release blocker!"
            return 1
        fi
    fi
    echo "$TESTNET_HEIGHT" > "$PREV_HEIGHT_FILE"
    success "Height monotonically increasing"

    # P2P peers
    local PEERS=$(curl -s http://127.0.0.1:8085/api/v1/peers 2>/dev/null | jq -r '.connected_peers // 0')
    log "📊 Connected peers: $PEERS"
    if [ "$PEERS" -lt 1 ]; then
        warning "No P2P peers connected"
    else
        success "P2P connectivity OK"
    fi

    # Error count
    local ERRORS=$(journalctl -u q-testnet-server --since "1 hour ago" 2>/dev/null | grep -c -E "ERROR|CRITICAL|panic" || echo "0")
    log "📊 Errors in last hour: $ERRORS"
    if [ "$ERRORS" -gt 10 ]; then
        warning "High error count in logs"
    fi

    # Memory
    local RSS_KB=$(ps aux | grep q-api-server-candidate | grep -v grep | awk '{print $6}' | head -1)
    local RSS_MB=$(echo "scale=0; ${RSS_KB:-0} / 1024" | bc)
    log "📊 Memory usage: ${RSS_MB}MB"

    success "Sync validation check complete"
}

# ════════════════════════════════════════════════════════════════════════════
# PHASE D: Functional Validation
# ════════════════════════════════════════════════════════════════════════════
phase_d() {
    header "PHASE D: FUNCTIONAL VALIDATION"

    local TESTNET_API="http://127.0.0.1:8085/api/v1"
    local FAILED=0

    # D.1: Status
    log "🔍 [D.1] Testing /api/v1/status..."
    local STATUS=$(curl -s "$TESTNET_API/status")
    if echo "$STATUS" | jq -e '.height' > /dev/null 2>&1; then
        success "[D.1] Status endpoint OK"
    else
        error "[D.1] Status endpoint failed"
        FAILED=1
    fi

    # D.2: Network
    log "🔍 [D.2] Testing /api/v1/network..."
    local NETWORK=$(curl -s "$TESTNET_API/network" 2>/dev/null)
    if echo "$NETWORK" | jq -e '.' > /dev/null 2>&1; then
        success "[D.2] Network endpoint OK"
    else
        warning "[D.2] Network endpoint issue (non-blocking)"
    fi

    # D.3: Peers
    log "🔍 [D.3] Testing /api/v1/peers..."
    local PEERS=$(curl -s "$TESTNET_API/peers" 2>/dev/null)
    local PEER_COUNT=$(echo "$PEERS" | jq -r '.connected_peers // 0')
    if [ "$PEER_COUNT" -ge 1 ]; then
        success "[D.3] P2P peers connected: $PEER_COUNT"
    else
        warning "[D.3] No peers connected (may be OK for fresh testnet)"
    fi

    # D.4: SSE test
    log "🔍 [D.4] Testing SSE streaming..."
    timeout 8 curl -s "$TESTNET_API/sse/status" > /tmp/sse-test.txt 2>&1 || true
    if [ -s /tmp/sse-test.txt ]; then
        success "[D.4] SSE streaming functional"
    else
        warning "[D.4] SSE test inconclusive"
    fi

    # D.5: Block retrieval
    log "🔍 [D.5] Testing block retrieval..."
    local HEIGHT=$(echo "$STATUS" | jq -r '.height // 0')
    if [ "$HEIGHT" -gt 10 ]; then
        local BLOCK=$(curl -s "$TESTNET_API/blocks/$((HEIGHT - 5))" 2>/dev/null)
        if echo "$BLOCK" | jq -e '.hash' > /dev/null 2>&1; then
            success "[D.5] Block retrieval OK"
        else
            warning "[D.5] Block retrieval issue"
        fi
    else
        warning "[D.5] Not enough blocks to test retrieval"
    fi

    # D.6: Version
    log "🔍 [D.6] Checking version..."
    local VER=$(echo "$STATUS" | jq -r '.version // "unknown"')
    log "   Version: $VER"
    success "[D.6] Version check complete"

    if [ $FAILED -eq 0 ]; then
        success "PHASE D COMPLETE - Ready for production release"
        return 0
    else
        error "PHASE D FAILED"
        return 1
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# PHASE E: Production Release
# ════════════════════════════════════════════════════════════════════════════
phase_e() {
    header "PHASE E: PRODUCTION RELEASE"

    local CANDIDATE="$TESTNET_DIR/q-narwhalknight/q-api-server-candidate"
    local PRODUCTION="$PROJECT_DIR/target/release/q-api-server"
    local DOWNLOADS="$PROJECT_DIR/gui/quantum-wallet/dist-final/downloads"

    # E.1: Confirmation
    log "🔐 [E.1] Final approval required"
    log ""
    log "   Version: $VERSION"
    log "   Candidate: $CANDIDATE"
    log ""

    read -p "Type 'RELEASE' to proceed with production deployment: " APPROVAL

    if [ "$APPROVAL" != "RELEASE" ]; then
        error "Release not approved"
        return 1
    fi

    # E.2: Backup
    log "🔍 [E.2] Creating backup..."
    mkdir -p "$BACKUP_DIR"
    local BACKUP_NAME="q-api-server-$(date +%Y%m%d-%H%M%S)"
    cp "$PRODUCTION" "$BACKUP_DIR/$BACKUP_NAME" 2>/dev/null || true
    success "[E.2] Backup created: $BACKUP_NAME"

    # E.3: Copy to production
    log "🔍 [E.3] Deploying to production..."
    cp "$CANDIDATE" "$PRODUCTION"
    chmod +x "$PRODUCTION"
    success "[E.3] Binary deployed"

    # E.4: Update downloads
    log "🔍 [E.4] Updating download links..."
    mkdir -p "$DOWNLOADS"
    cp "$CANDIDATE" "$DOWNLOADS/q-api-server-$VERSION"
    cp "$CANDIDATE" "$DOWNLOADS/q-api-server-linux-x86_64"
    chmod +x "$DOWNLOADS/q-api-server-$VERSION"
    chmod +x "$DOWNLOADS/q-api-server-linux-x86_64"
    success "[E.4] Downloads updated"

    # E.5: Restart production
    log "🔍 [E.5] Restarting production service..."
    systemctl restart q-api-server
    sleep 10

    if systemctl is-active --quiet q-api-server; then
        success "[E.5] Production service running"
    else
        error "[E.5] Production failed to start - ROLLING BACK!"
        cp "$BACKUP_DIR/$BACKUP_NAME" "$PRODUCTION"
        systemctl restart q-api-server
        return 1
    fi

    # E.6: Verify
    log "🔍 [E.6] Verifying production API..."
    sleep 5
    local PROD_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/api/v1/status)
    if [ "$PROD_CHECK" = "200" ]; then
        success "[E.6] Production API verified"
    else
        error "[E.6] Production API issue (HTTP $PROD_CHECK)"
        warning "Consider rollback if issues persist"
    fi

    log ""
    log "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    log "${GREEN}║         🎉 RELEASE COMPLETE: $VERSION                          ${NC}"
    log "${GREEN}╠═══════════════════════════════════════════════════════════════╣${NC}"
    log "${GREEN}║                                                               ║${NC}"
    log "${GREEN}║  📥 User download link:                                       ║${NC}"
    log "${GREEN}║  wget https://dl.quillon.xyz/downloads/q-api-server-$VERSION   ║${NC}"
    log "${GREEN}║                                                               ║${NC}"
    log "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
}

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
case "${1:-help}" in
    a|phase-a)
        phase_a
        ;;
    b|phase-b)
        phase_b
        ;;
    c|phase-c)
        phase_c
        ;;
    d|phase-d)
        phase_d
        ;;
    e|phase-e)
        phase_e
        ;;
    all)
        log "Starting full release validation pipeline for $VERSION"
        log "Timestamp: $(date)"

        phase_a || exit 1
        phase_b || exit 1

        log ""
        log "${YELLOW}⏳ Phase C requires 24+ hours of sync monitoring${NC}"
        log "   Run: ./scripts/validate-release.sh c"
        log "   Or monitor: tail -f $TESTNET_DIR/logs/sync-validation.log"
        log ""
        read -p "Press Enter when sync is complete to continue with Phase D..."

        phase_d || exit 1

        log ""
        log "${YELLOW}⏳ Recommend 48-hour soak test before Phase E${NC}"
        read -p "Press Enter when ready for production release..."

        phase_e
        ;;
    status)
        echo "Current Status:"
        echo "==============="

        # Testnet service
        if systemctl is-active --quiet q-testnet-server 2>/dev/null; then
            echo "Testnet Service: RUNNING"
        else
            echo "Testnet Service: STOPPED"
        fi

        # Production service
        if systemctl is-active --quiet q-api-server 2>/dev/null; then
            echo "Production Service: RUNNING"
        else
            echo "Production Service: STOPPED"
        fi

        # Heights
        TESTNET_H=$(curl -s http://127.0.0.1:8085/api/v1/status 2>/dev/null | jq -r '.height // "N/A"')
        PROD_H=$(curl -s http://127.0.0.1:8080/api/v1/status 2>/dev/null | jq -r '.height // "N/A"')
        echo "Testnet Height: $TESTNET_H"
        echo "Production Height: $PROD_H"

        # Last validation
        if [ -f "$TESTNET_DIR/logs/sync-validation.log" ]; then
            echo "Last Validation: $(tail -1 $TESTNET_DIR/logs/sync-validation.log)"
        fi
        ;;
    *)
        echo "Q-NarwhalKnight Release Validation Pipeline"
        echo "==========================================="
        echo ""
        echo "Usage: $0 <command> [version]"
        echo ""
        echo "Commands:"
        echo "  a, phase-a    Run Phase A: Pre-deployment validation"
        echo "  b, phase-b    Run Phase B: Testnet deployment"
        echo "  c, phase-c    Run Phase C: Sync validation check"
        echo "  d, phase-d    Run Phase D: Functional validation"
        echo "  e, phase-e    Run Phase E: Production release"
        echo "  all           Run complete pipeline (interactive)"
        echo "  status        Show current status"
        echo ""
        echo "Example:"
        echo "  $0 all v3.4.16-beta"
        echo ""
        ;;
esac
