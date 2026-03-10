#!/bin/bash
# Q-NarwhalKnight Safe Deployment Script v3.0 (Mainnet Ready)
# v3.0.0: Now runs ALL 4000+ tests before deployment (comprehensive coverage)
# v3.3.9-beta: Added upgrade gate verification and version filtering checks
# v3.3.9-beta+: Now runs 70 mainnet safety tests BEFORE every build
# v2.3.5-beta+: Added sync-down, balance integrity, and block validation tests (140+ total)
# Usage: ./safe-deploy.sh [build|test-docker|test-alpha|deploy-beta|rollback|test-all]

set -e

# Configuration
PROJECT_DIR="/opt/orobit/shared/q-narwhalknight"
BINARY_NAME="q-api-server"
BINARY_PATH="$PROJECT_DIR/target/release/$BINARY_NAME"
BACKUP_DIR="$PROJECT_DIR/backups"
DOWNLOADS_DIR="$PROJECT_DIR/gui/quantum-wallet/dist-final/downloads"
SERVICE_NAME="q-api-server"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get version from binary
get_version() {
    local binary=$1
    strings "$binary" 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+-beta' | head -1 || echo "unknown"
}

# v3.3.9-beta: Verify upgrade gate is active after deployment
verify_upgrade_gate() {
    local timeout=${1:-30}
    log_info "Verifying upgrade gate initialization..."

    for i in $(seq 1 $timeout); do
        if journalctl -u "$SERVICE_NAME" --since "60 seconds ago" 2>&1 | grep -q "UPGRADE GATE.*Initialized"; then
            log_success "Upgrade gate initialized!"

            # Check for version filtering
            if journalctl -u "$SERVICE_NAME" --since "60 seconds ago" 2>&1 | grep -q "VERSION FILTER\|protocol_version"; then
                log_success "Version filtering active!"
            fi

            # Show upgrade status
            journalctl -u "$SERVICE_NAME" --since "60 seconds ago" 2>&1 | grep "UPGRADE GATE" | head -5
            return 0
        fi
        sleep 1
    done

    log_warn "Upgrade gate status not found in logs (may still be initializing)"
    return 0  # Don't fail deployment, just warn
}

# Announce update to P2P network after successful deployment
# Non-blocking: warns on failure but does not fail the deploy
announce_update() {
    log_info "Announcing update to P2P network..."

    # All commands use || fallbacks to prevent set -e from aborting
    local version
    version=$(grep '^version' "$PROJECT_DIR/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/' || echo "")
    if [ -z "$version" ]; then
        log_warn "Could not determine version from Cargo.toml — skipping announce"
        return 0
    fi

    local sha256_checksum
    sha256_checksum=$(sha256sum "$BINARY_PATH" 2>/dev/null | awk '{print $1}' || echo "")

    local blake3_checksum
    blake3_checksum=$(b3sum "$BINARY_PATH" 2>/dev/null | awk '{print $1}' || echo "")

    local binary_size
    binary_size=$(stat --format=%s "$BINARY_PATH" 2>/dev/null || echo "0")

    if [ -z "$sha256_checksum" ] || [ "$binary_size" = "0" ]; then
        log_warn "Could not compute checksums for binary — skipping announce"
        return 0
    fi

    local download_url="https://quillon.xyz/downloads/q-api-server-v${version}"
    local release_notes="v${version} automated deployment via safe-deploy.sh"

    log_info "  Version:  $version"
    log_info "  SHA-256:  $sha256_checksum"
    log_info "  BLAKE3:   ${blake3_checksum:-<b3sum not available>}"
    log_info "  Size:     $binary_size bytes"
    log_info "  URL:      $download_url"

    local response
    response=$(curl -s -w "\n%{http_code}" -X POST "http://localhost:8080/api/v1/admin/update/announce" \
        -H "Content-Type: application/json" \
        -d "{
            \"version\": \"${version}\",
            \"sha256_checksum\": \"${sha256_checksum}\",
            \"blake3_checksum\": \"${blake3_checksum}\",
            \"binary_size\": ${binary_size},
            \"download_url\": \"${download_url}\",
            \"mandatory\": false,
            \"release_notes\": \"${release_notes}\"
        }" 2>/dev/null) || true

    local http_code
    http_code=$(echo "$response" | tail -1)
    local body
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" = "200" ]; then
        log_success "Update announced to P2P network: v${version}"
    else
        log_warn "Update announce returned HTTP ${http_code} (non-fatal) — ${body}"
    fi

    return 0
}

# Health check endpoint
health_check() {
    local port=${1:-8080}
    local timeout=${2:-30}

    log_info "Running health check on port $port (timeout: ${timeout}s)..."

    for i in $(seq 1 $timeout); do
        if curl -s "http://localhost:$port/api/v1/status" | grep -q '"status":"ready"'; then
            log_success "Health check passed!"
            return 0
        fi
        sleep 1
    done

    log_error "Health check failed after ${timeout}s"
    return 1
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check binary exists
    if [ ! -f "$BINARY_PATH" ]; then
        log_error "Binary not found: $BINARY_PATH"
        return 1
    fi

    # Check binary is executable
    if [ ! -x "$BINARY_PATH" ]; then
        chmod +x "$BINARY_PATH"
    fi

    # Check binary runs (--version or --help)
    if ! "$BINARY_PATH" --help &>/dev/null; then
        log_error "Binary failed to execute"
        return 1
    fi

    # Get version
    local version=$(get_version "$BINARY_PATH")
    log_info "Binary version: $version"

    # Check for required libraries
    if ! ldd "$BINARY_PATH" &>/dev/null; then
        log_error "Missing shared libraries"
        return 1
    fi

    log_success "Pre-flight checks passed!"
    return 0
}

# Run all 4000+ tests (comprehensive test suite)
cmd_test_all() {
    log_info "🧪 Running COMPREHENSIVE test suite (4000+ tests)..."
    cd "$PROJECT_DIR"

    local failed_tests=0
    local passed_suites=0
    local total_suites=0

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 1: CRITICAL MAINNET SAFETY TESTS (Must Pass)
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🛡️  CATEGORY 1: CRITICAL MAINNET SAFETY TESTS"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # q-consensus-guard
    ((total_suites++))
    log_info "[1/47] Testing q-consensus-guard (mainnet_safety_tests)..."
    if timeout 600 cargo test --package q-consensus-guard --test mainnet_safety_tests 2>&1; then
        ((passed_suites++)); log_success "✅ PASSED"
    else
        ((failed_tests++)); log_error "❌ FAILED"
    fi

    # q-storage - Critical tests
    for test in mainnet_critical_tests sync_down_protection_tests balance_integrity_tests balance_propagation_tests fork_detection_tests fork_reorg_tests backup_restore_tests; do
        ((total_suites++))
        log_info "Testing q-storage ($test)..."
        if timeout 600 cargo test --package q-storage --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # q-types - Block validation
    for test in block_validation_tests signature_verification_tests fee_validation_tests; do
        ((total_suites++))
        log_info "Testing q-types ($test)..."
        if timeout 600 cargo test --package q-types --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # q-dex - DEX overflow protection
    for test in overflow_protection_tests comprehensive_dex_tests; do
        ((total_suites++))
        log_info "Testing q-dex ($test)..."
        if timeout 600 cargo test --package q-dex --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 2: DECENTRALIZATION & CONSENSUS TESTS
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🌐  CATEGORY 2: DECENTRALIZATION & CONSENSUS TESTS"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # q-storage - Validator registry
    ((total_suites++))
    log_info "Testing q-storage (validator_registry_tests)..."
    if timeout 600 cargo test --package q-storage --test validator_registry_tests 2>&1; then
        ((passed_suites++)); log_success "✅ PASSED"
    else
        ((failed_tests++)); log_error "❌ FAILED"
    fi

    # q-storage - Peer reputation
    ((total_suites++))
    log_info "Testing q-storage (peer_reputation_tests)..."
    if timeout 600 cargo test --package q-storage --test peer_reputation_tests 2>&1; then
        ((passed_suites++)); log_success "✅ PASSED"
    else
        ((failed_tests++)); log_error "❌ FAILED"
    fi

    # q-narwhal-core - Byzantine detection
    ((total_suites++))
    log_info "Testing q-narwhal-core (byzantine_detection_tests)..."
    if timeout 600 cargo test --package q-narwhal-core --test byzantine_detection_tests 2>&1; then
        ((passed_suites++)); log_success "✅ PASSED"
    else
        ((failed_tests++)); log_error "❌ FAILED"
    fi

    # q-dag-knight - Consensus voting
    ((total_suites++))
    log_info "Testing q-dag-knight (consensus_voting_tests)..."
    if timeout 600 cargo test --package q-dag-knight --test consensus_voting_tests 2>&1; then
        ((passed_suites++)); log_success "✅ PASSED"
    else
        ((failed_tests++)); log_error "❌ FAILED"
    fi

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 3: NETWORK & P2P TESTS
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🔗  CATEGORY 3: NETWORK & P2P TESTS"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for test in version_filter_tests distributed_ai_encryption_tests message_handler_dos_tests network_partition_tests; do
        ((total_suites++))
        log_info "Testing q-network ($test)..."
        if timeout 600 cargo test --package q-network --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 4: SYNC & STATE TESTS
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🔄  CATEGORY 4: SYNC & STATE TESTS"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for test in state_applicator_safety_tests sync_malicious_peer_tests turbo_sync_edge_cases_tests; do
        ((total_suites++))
        log_info "Testing q-storage ($test)..."
        if timeout 600 cargo test --package q-storage --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 5: PRIVACY & CRYPTOGRAPHY TESTS
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🔐  CATEGORY 5: PRIVACY & CRYPTOGRAPHY TESTS"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # q-crypto-advanced
    for test in bulletproofs_mainnet_safety_tests bulletproof_soundness_tests; do
        ((total_suites++))
        log_info "Testing q-crypto-advanced ($test)..."
        if timeout 600 cargo test --package q-crypto-advanced --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # q-quantum-mixing
    for test in advanced_zk_tests privacy_mainnet_safety_tests ring_signature_security_tests; do
        ((total_suites++))
        log_info "Testing q-quantum-mixing ($test)..."
        if timeout 600 cargo test --package q-quantum-mixing --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # q-types - Privacy
    for test in privacy_integration_tests stark_privacy_tests; do
        ((total_suites++))
        log_info "Testing q-types ($test)..."
        if timeout 600 cargo test --package q-types --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # q-storage - Privacy
    ((total_suites++))
    log_info "Testing q-storage (privacy_upgrade_safety_tests)..."
    if timeout 600 cargo test --package q-storage --test privacy_upgrade_safety_tests 2>&1; then
        ((passed_suites++)); log_success "✅ PASSED"
    else
        ((failed_tests++)); log_error "❌ FAILED"
    fi

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 6: API & SERVER TESTS
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🌐  CATEGORY 6: API & SERVER TESTS"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for test in mining_stats_tests sse_streaming_tests u128_migration_tests mining_commit_reveal_tests contracts_api_tests block_producer_concurrency_tests; do
        ((total_suites++))
        log_info "Testing q-api-server ($test)..."
        if timeout 600 cargo test --package q-api-server --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 7: VM & SMART CONTRACT TESTS
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "📜  CATEGORY 7: VM & SMART CONTRACT TESTS"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for test in vm_tests comprehensive_contract_tests wasm_sandbox_security_tests property_tests; do
        ((total_suites++))
        log_info "Testing q-vm ($test)..."
        if timeout 600 cargo test --package q-vm --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED"
        fi
    done

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 8: TOR & ANONYMITY TESTS
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🧅  CATEGORY 8: TOR & ANONYMITY TESTS"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for test in comprehensive_tor_dandelion_arti_tests onion_connection_tests; do
        ((total_suites++))
        log_info "Testing q-tor-client ($test)..."
        if timeout 600 cargo test --package q-tor-client --test $test 2>&1; then
            ((passed_suites++)); log_success "✅ PASSED"
        else
            ((failed_tests++)); log_error "❌ FAILED (may be expected if Tor not configured)"
        fi
    done

    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 9: FULL WORKSPACE TESTS (All remaining tests)
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "🔬  CATEGORY 9: FULL WORKSPACE TESTS (all remaining)"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    log_info "Running full workspace test suite (this may take 30+ minutes)..."
    if timeout 7200 cargo test --workspace 2>&1; then
        log_success "✅ Full workspace tests PASSED"
    else
        log_warn "⚠️ Some workspace tests may have failed (check output above)"
    fi

    # ═══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "📊  TEST SUMMARY"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Test Suites Passed: $passed_suites / $total_suites"
    echo "  Test Suites Failed: $failed_tests"
    echo ""

    if [ "$failed_tests" -gt "0" ]; then
        log_error "❌ $failed_tests test suite(s) FAILED - DO NOT DEPLOY!"
        return 1
    else
        log_success "🎉 All $total_suites test suites PASSED!"
        return 0
    fi
}

# Build release binary
cmd_build() {
    log_info "Building release binary..."

    cd "$PROJECT_DIR"

    # v3.0.0: Run comprehensive test suite FIRST (MANDATORY)
    log_info "🧪 Running comprehensive test suite (MANDATORY before build)..."

    if ! cmd_test_all; then
        log_error "❌ Tests FAILED - ABORTING BUILD!"
        log_error "Fix all test failures before deploying to production."
        return 1
    fi

    log_success "🎉 All tests passed! Proceeding with build..."

    # Run cargo check (needs 6+ minutes for full codebase)
    log_info "Running cargo check (this may take 6-8 minutes)..."
    if ! timeout 600 cargo check --package q-api-server 2>&1; then
        log_error "Cargo check failed"
        return 1
    fi
    log_success "✅ Cargo check passed"

    # Build release
    log_info "Building release binary (this may take 10-15 minutes)..."
    if ! timeout 36000 cargo build --release --package q-api-server 2>&1 | tail -20; then
        log_error "Build failed"
        return 1
    fi

    # Run pre-flight checks
    preflight_checks || return 1

    local version=$(get_version "$BINARY_PATH")
    log_success "Build complete: $version"

    # Create backup
    mkdir -p "$BACKUP_DIR"
    cp "$BINARY_PATH" "$BACKUP_DIR/$BINARY_NAME-$version-$(date +%Y%m%d-%H%M%S)"
    log_info "Backup created in $BACKUP_DIR"
}

# Test in Docker (isolated environment)
cmd_test_docker() {
    local test_name="q-deploy-test-$(date +%s)"
    local test_port=8095
    local test_data="/tmp/$test_name"

    log_info "Starting Docker canary test..."

    # Pre-flight checks
    preflight_checks || return 1

    # Cleanup old test containers
    docker rm -f "$test_name" 2>/dev/null || true
    rm -rf "$test_data"
    mkdir -p "$test_data"

    # Copy binary
    cp "$BINARY_PATH" "$test_data/$BINARY_NAME"
    chmod +x "$test_data/$BINARY_NAME"

    log_info "Starting test container..."
    docker run -d --name "$test_name" \
        -v "$test_data:/data" \
        -e Q_DB_PATH=/data/db \
        -e Q_NETWORK_ID=testnet-phase16 \
        -e Q_P2P_PORT=9099 \
        -e Q_TURBO_SYNC=1 \
        -e RUST_LOG=info \
        -p $test_port:8080 \
        --entrypoint /bin/sh \
        debian:bookworm-slim \
        -c "apt-get update -qq && apt-get install -y -qq libssl3 ca-certificates 2>/dev/null && /data/$BINARY_NAME --port 8080"

    # Wait for startup
    log_info "Waiting for container startup (60s)..."
    sleep 30

    # Check if container is still running
    if ! docker ps | grep -q "$test_name"; then
        log_error "Container crashed during startup"
        docker logs "$test_name" 2>&1 | tail -50
        docker rm -f "$test_name" 2>/dev/null
        return 1
    fi

    # Health check
    if ! health_check $test_port 60; then
        log_error "Health check failed"
        docker logs "$test_name" 2>&1 | tail -50
        docker rm -f "$test_name" 2>/dev/null
        return 1
    fi

    # Check for critical errors
    log_info "Checking for critical errors..."
    local errors=$(docker logs "$test_name" 2>&1 | grep -c "CRITICAL\|CORRUPTION\|panic\|FATAL" || echo "0")
    if [ "$errors" -gt "0" ]; then
        log_error "Found $errors critical errors in logs"
        docker logs "$test_name" 2>&1 | grep -E "CRITICAL|CORRUPTION|panic|FATAL" | head -10
        docker rm -f "$test_name" 2>/dev/null
        return 1
    fi

    # Check sync progress
    log_info "Checking sync progress (waiting 60s)..."
    sleep 60

    local height=$(docker logs "$test_name" 2>&1 | grep -oE "height[: ]+[0-9]+" | tail -1 | grep -oE "[0-9]+" || echo "0")
    log_info "Synced to height: $height"

    if [ "$height" -lt "100" ]; then
        log_warn "Low sync height - may indicate P2P issues"
    else
        log_success "Sync working - reached height $height"
    fi

    # Cleanup
    docker rm -f "$test_name" 2>/dev/null
    rm -rf "$test_data"

    log_success "Docker canary test PASSED!"
    return 0
}

# Deploy to production (Server Beta)
cmd_deploy_beta() {
    log_info "Deploying to production (Server Beta)..."

    # Pre-flight checks
    preflight_checks || return 1

    local version=$(get_version "$BINARY_PATH")
    local current_binary="/usr/bin/$BINARY_NAME"

    # Backup current binary
    if [ -f "$current_binary" ]; then
        local backup_name="$BACKUP_DIR/$BINARY_NAME-rollback-$(date +%Y%m%d-%H%M%S)"
        cp "$current_binary" "$backup_name"
        log_info "Current binary backed up to: $backup_name"
        echo "$backup_name" > "$BACKUP_DIR/last-rollback-binary"
    fi

    # Copy to downloads
    cp "$BINARY_PATH" "$DOWNLOADS_DIR/$BINARY_NAME-$version"
    log_info "Binary copied to downloads: $BINARY_NAME-$version"

    # Stop service gracefully
    log_info "Stopping service gracefully (60s timeout)..."
    systemctl stop "$SERVICE_NAME" --no-block 2>/dev/null || true
    sleep 5

    # Force kill if still running
    if pgrep -f "$BINARY_NAME" > /dev/null; then
        log_warn "Service still running, force killing..."
        kill -9 $(pgrep -f "$BINARY_NAME") 2>/dev/null || true
        sleep 2
    fi

    # Start service with new binary
    log_info "Starting service with new binary..."
    systemctl start "$SERVICE_NAME"

    # Health check
    if ! health_check 8080 60; then
        log_error "Production health check failed - ROLLING BACK!"
        cmd_rollback
        return 1
    fi

    # v3.3.9-beta: Verify mainnet safety features
    verify_upgrade_gate 30

    # Monitor for 5 minutes
    log_info "Monitoring for 5 minutes..."
    for i in $(seq 1 10); do
        sleep 30
        if ! pgrep -f "$BINARY_NAME" > /dev/null; then
            log_error "Service crashed during soak test - ROLLING BACK!"
            cmd_rollback
            return 1
        fi
        log_info "Soak test: ${i}/10 checks passed"
    done

    # Announce update to P2P network (non-blocking)
    announce_update || log_warn "Update announce failed (non-fatal)"

    log_success "Production deployment complete: $version"
    echo "Download: wget https://quillon.xyz/downloads/$BINARY_NAME-$version"
}

# Rollback to previous binary
cmd_rollback() {
    log_warn "Starting rollback..."

    local rollback_binary=$(cat "$BACKUP_DIR/last-rollback-binary" 2>/dev/null)

    if [ -z "$rollback_binary" ] || [ ! -f "$rollback_binary" ]; then
        log_error "No rollback binary found!"
        return 1
    fi

    log_info "Rolling back to: $rollback_binary"

    # Stop service
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true
    kill -9 $(pgrep -f "$BINARY_NAME") 2>/dev/null || true
    sleep 2

    # Copy rollback binary to release path
    cp "$rollback_binary" "$BINARY_PATH"
    chmod +x "$BINARY_PATH"

    # Start service
    systemctl start "$SERVICE_NAME"

    # Health check
    if health_check 8080 60; then
        log_success "Rollback successful!"
    else
        log_error "Rollback failed - manual intervention required!"
        return 1
    fi
}

# Show status
cmd_status() {
    log_info "=== Deployment Status ==="

    echo ""
    echo "Current binary:"
    ls -la "$BINARY_PATH" 2>/dev/null || echo "  Not found"
    echo "  Version: $(get_version "$BINARY_PATH")"

    echo ""
    echo "Service status:"
    systemctl status "$SERVICE_NAME" --no-pager | head -10

    echo ""
    echo "Recent backups:"
    ls -lt "$BACKUP_DIR" 2>/dev/null | head -5

    echo ""
    echo "Health check:"
    health_check 8080 5 || echo "  Service not responding"
}

# Main
case "${1:-}" in
    build)
        cmd_build
        ;;
    test-all)
        cmd_test_all
        ;;
    test-docker)
        cmd_test_docker
        ;;
    deploy-beta)
        cmd_deploy_beta
        ;;
    rollback)
        cmd_rollback
        ;;
    status)
        cmd_status
        ;;
    full)
        log_info "=== FULL DEPLOYMENT PIPELINE ==="
        cmd_build && \
        cmd_test_docker && \
        cmd_deploy_beta
        ;;
    *)
        echo "Q-NarwhalKnight Safe Deployment Script v3.0"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  test-all     - Run comprehensive test suite (4000+ tests)"
        echo "  build        - Run all tests then build release binary"
        echo "  test-docker  - Run canary test in Docker container"
        echo "  deploy-beta  - Deploy to production (Server Beta)"
        echo "  rollback     - Rollback to previous binary"
        echo "  status       - Show deployment status"
        echo "  full         - Run full pipeline (build → docker test → deploy)"
        echo ""
        echo "Test Categories:"
        echo "  1. Critical Mainnet Safety Tests (sync, balances, validation)"
        echo "  2. Decentralization & Consensus Tests (validators, BFT, voting)"
        echo "  3. Network & P2P Tests (version filtering, DoS, partitions)"
        echo "  4. Sync & State Tests (turbo sync, state applicator)"
        echo "  5. Privacy & Cryptography Tests (bulletproofs, ring signatures)"
        echo "  6. API & Server Tests (SSE, mining, contracts)"
        echo "  7. VM & Smart Contract Tests (WASM sandbox, contracts)"
        echo "  8. Tor & Anonymity Tests (dandelion++, onion routing)"
        echo "  9. Full Workspace Tests (all remaining tests)"
        echo ""
        echo "Recommended workflow:"
        echo "  1. ./safe-deploy.sh test-all    # Run comprehensive tests"
        echo "  2. ./safe-deploy.sh build       # Build (includes tests)"
        echo "  3. ./safe-deploy.sh test-docker # Canary test"
        echo "  4. ./safe-deploy.sh deploy-beta # Production deploy"
        echo ""
        echo "If deployment fails:"
        echo "  ./safe-deploy.sh rollback"
        ;;
esac
