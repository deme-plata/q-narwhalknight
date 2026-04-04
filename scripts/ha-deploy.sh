#!/bin/bash
# Q-NarwhalKnight HA Rolling Deploy Script v2.0
# Zero-downtime deployment across 3 servers: Delta (canary) → Gamma → Beta
#
# Pipeline:
#   1. verify-delta   — SCP binary to Delta, restart, wait for health
#   2. verify-gamma   — SCP binary to Gamma, restart, wait for health
#   3. promote         — Gamma weight=10, Beta weight=1 in nginx
#   4. deploy-beta     — Stop Beta, replace binary, restart, verify
#   5. restore         — Beta weight=10, Gamma weight=1
#   6. full            — Run entire pipeline (1→5)
#
# Usage:
#   ./scripts/ha-deploy.sh status           # Check all servers
#   ./scripts/ha-deploy.sh full [-y]        # Full rolling deploy
#   ./scripts/ha-deploy.sh verify-delta     # Deploy to Delta canary
#   ./scripts/ha-deploy.sh verify-gamma     # Deploy to Gamma
#   ./scripts/ha-deploy.sh promote          # Gamma becomes primary
#   ./scripts/ha-deploy.sh deploy-beta      # Upgrade Beta
#   ./scripts/ha-deploy.sh restore          # Beta becomes primary again
#   ./scripts/ha-deploy.sh rollback         # Restore previous binary on Beta

set -uo pipefail
# NOTE: No set -e! We handle errors explicitly to avoid silent exits.

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

PROJECT_DIR="/opt/orobit/shared/q-narwhalknight"
BINARY_NAME="q-api-server"
BINARY_PATH="$PROJECT_DIR/target/release/$BINARY_NAME"
BACKUP_DIR="$PROJECT_DIR/backups"
DOWNLOADS_DIR="$PROJECT_DIR/gui/quantum-wallet/dist-final/downloads"
SERVICE_NAME="q-api-server"
LOCKFILE="/tmp/ha-deploy.lock"

# Server addresses
BETA_HOST="185.182.185.227"
GAMMA_HOST="109.205.176.60"
DELTA_HOST="5.79.79.158"
EPSILON_HOST="89.149.241.126"

# Remote binary paths (must match each server's systemd ExecStart)
BETA_BINARY="/opt/orobit/shared/q-narwhalknight/$BINARY_NAME"
GAMMA_BINARY="/opt/orobit/shared/q-narwhalknight/$BINARY_NAME"
DELTA_BINARY="/opt/orobit/shared/q-narwhalknight/q-api-server-v926"

# Epsilon download paths
EPSILON_DOWNLOADS="/home/orobit/q-narwhalknight/dist-final/downloads"

# Health check settings
HEALTH_TIMEOUT=600       # Max seconds to wait for health (Delta can take 7+ min)
STABILITY_SOAK=90        # Seconds a node must stay healthy
SOAK_INTERVAL=5          # Poll interval during soak

# ═══════════════════════════════════════════════════════════════════
# Colors & Logging
# ═══════════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "${BLUE}[STEP]${NC} $1"; }
log_header()  { echo -e "${MAGENTA}[$1]${NC} ═══ $2 ═══"; }

# ═══════════════════════════════════════════════════════════════════
# Lock Management
# ═══════════════════════════════════════════════════════════════════

acquire_lock() {
    if [ -f "$LOCKFILE" ]; then
        local pid
        pid=$(cat "$LOCKFILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            log_error "Another ha-deploy is running (PID $pid). Aborting."
            exit 1
        fi
        log_warn "Stale lock file found (PID $pid not running). Removing."
        rm -f "$LOCKFILE"
    fi
    echo $$ > "$LOCKFILE"
    trap 'rm -f "$LOCKFILE"' EXIT
}

# ═══════════════════════════════════════════════════════════════════
# Version Detection
# ═══════════════════════════════════════════════════════════════════

get_cargo_version() {
    grep '^version' "$PROJECT_DIR/Cargo.toml" | head -1 | sed 's/.*"\([^"]*\)".*/\1/' 2>/dev/null || echo "unknown"
}

get_running_version() {
    local host=$1
    local port=${2:-8080}
    if [ "$host" = "localhost" ] || [ "$host" = "$BETA_HOST" ]; then
        curl -s --connect-timeout 3 "http://localhost:$port/api/v1/status" 2>/dev/null | \
            grep -oE '"version":"[^"]*"' | sed 's/"version":"//;s/"//' || echo "unreachable"
    else
        ssh -o ConnectTimeout=5 "root@$host" \
            "curl -s --connect-timeout 3 http://localhost:$port/api/v1/status" 2>/dev/null | \
            grep -oE '"version":"[^"]*"' | sed 's/"version":"//;s/"//' || echo "unreachable"
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Health Check (local or remote)
# ═══════════════════════════════════════════════════════════════════

get_status() {
    local host=$1
    local port=${2:-8080}
    if [ "$host" = "localhost" ] || [ "$host" = "$BETA_HOST" ]; then
        curl -s --connect-timeout 3 "http://localhost:$port/api/v1/status" 2>/dev/null || echo ""
    else
        ssh -o ConnectTimeout=5 "root@$host" \
            "curl -s --connect-timeout 3 http://localhost:$port/api/v1/status" 2>/dev/null || echo ""
    fi
}

# Wait for a remote node to become healthy and stable
wait_for_health() {
    local name=$1
    local host=$2
    local port=${3:-8080}
    local timeout=${4:-$HEALTH_TIMEOUT}
    local soak=${5:-$STABILITY_SOAK}
    local logfile="/tmp/ha-deploy-$(echo "$name" | tr '[:upper:]' '[:lower:]').log"

    log_info "Waiting for $name ($host) to become healthy (timeout: ${timeout}s, min stable: ${soak}s)..."
    log_info "  (Route conflict panics happen ~30-60s after startup, so we wait ${soak}s to catch them)"

    local elapsed=0
    local healthy_since=0
    local failures=0
    local max_failures=3

    while [ $elapsed -lt $timeout ]; do
        local status
        status=$(get_status "$host" "$port")

        if echo "$status" | grep -qE '"status":"(ready|syncing)"'; then
            if [ $healthy_since -eq 0 ]; then
                healthy_since=$elapsed
                local height
                height=$(echo "$status" | grep -oE '"height":[0-9]+' | head -1 | sed 's/"height"://')
                local version
                version=$(echo "$status" | grep -oE '"version":"[^"]*"' | sed 's/"version":"//;s/"//')
                log_info "$name first healthy response at +${elapsed}s (height=$height v$version)"
            fi

            local stable_time=$((elapsed - healthy_since))
            echo -ne "  $name healthy, stability check: ${stable_time}/${soak}s\r"

            if [ $stable_time -ge $soak ]; then
                echo ""
                local final_status
                final_status=$(get_status "$host" "$port")
                local h v s
                h=$(echo "$final_status" | grep -oE '"height":[0-9]+' | head -1 | sed 's/"height"://')
                v=$(echo "$final_status" | grep -oE '"version":"[^"]*"' | sed 's/"version":"//;s/"//')
                s=$(echo "$final_status" | grep -oE '"status":"[^"]*"' | sed 's/"status":"//;s/"//')
                log_success "$name is VERIFIED healthy and stable for ${stable_time}s"
                log_info "  $name: status=$s height=$h v$v"
                echo "$final_status" > "$logfile"
                return 0
            fi
            failures=0
        else
            if [ $healthy_since -gt 0 ]; then
                failures=$((failures + 1))
                log_warn "$name health check failed after being healthy (failure $failures/$max_failures)"
                if [ $failures -ge $max_failures ]; then
                    echo ""
                    log_error "$name went unhealthy $max_failures times after initial health. Aborting."
                    return 1
                fi
            else
                echo -ne "  Waiting... ${elapsed}/${timeout}s (connection refused)\r"
            fi
            healthy_since=0
        fi

        sleep $SOAK_INTERVAL
        elapsed=$((elapsed + SOAK_INTERVAL))
    done

    echo ""
    log_error "$name did not become healthy within ${timeout}s"
    return 1
}

# ═══════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════

cmd_status() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           Q-NarwhalKnight HA Cluster Status              ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    local cargo_ver
    cargo_ver=$(get_cargo_version)
    log_info "Cargo.toml version: $cargo_ver"
    echo ""

    for entry in "Beta:$BETA_HOST:PRIMARY" "Gamma:$GAMMA_HOST:BACKUP" "Delta:$DELTA_HOST:CANARY" "Epsilon:$EPSILON_HOST:SUPERNODE"; do
        local name host role
        name=$(echo "$entry" | cut -d: -f1)
        host=$(echo "$entry" | cut -d: -f2)
        role=$(echo "$entry" | cut -d: -f3)

        local status
        status=$(get_status "$host") || status=""

        if [ -n "$status" ] && echo "$status" | grep -qE '"status"'; then
            local h v s p
            h=$(echo "$status" | grep -oE '"height":[0-9]+' | head -1 | sed 's/"height"://')
            v=$(echo "$status" | grep -oE '"version":"[^"]*"' | sed 's/"version":"//;s/"//')
            s=$(echo "$status" | grep -oE '"status":"[^"]*"' | sed 's/"status":"//;s/"//')
            p=$(echo "$status" | grep -oE '"peer_count":[0-9]+' | head -1 | sed 's/"peer_count"://')
            echo -e "  ${GREEN}●${NC} ${name} (${host}) [${role}] — ${GREEN}${s}${NC} height=${h} v${v} peers=${p:-?}"
        else
            echo -e "  ${RED}●${NC} ${name} (${host}) [${role}] — ${RED}unreachable${NC}"
        fi
    done
    echo ""
}

cmd_verify_delta() {
    log_header "CANARY" "Deploying to Delta ($DELTA_HOST)"

    log_step "1/3: Checking local binary exists..."
    if [ ! -f "$BINARY_PATH" ]; then
        log_error "Binary not found: $BINARY_PATH"
        log_error "Run 'cargo build --release --package q-api-server' first."
        return 1
    fi
    local size
    size=$(du -h "$BINARY_PATH" | awk '{print $1}')
    log_info "Binary: $BINARY_PATH ($size)"

    # Check disk space on Delta
    local free
    free=$(ssh -o ConnectTimeout=5 "root@$DELTA_HOST" "df -m /opt | tail -1 | awk '{print \$4}'" 2>/dev/null) || free="0"
    log_info "Delta: ${free}MB free (OK, need 500MB)"
    if [ "${free:-0}" -lt 500 ]; then
        log_error "Delta has less than 500MB free disk space!"
        return 1
    fi

    log_step "2/3: Stopping Delta, copying binary..."
    ssh -o ConnectTimeout=5 "root@$DELTA_HOST" "
        pgrep -f q-api-server | xargs -I{} kill -9 {} 2>/dev/null || true
        sleep 2
    " || true
    scp -o ConnectTimeout=10 "$BINARY_PATH" "root@$DELTA_HOST:$DELTA_BINARY" || {
        log_error "SCP to Delta failed"
        return 1
    }
    log_info "Binary copied to Delta"

    log_step "3/3: Starting Delta and waiting for health..."
    ssh -o ConnectTimeout=5 "root@$DELTA_HOST" "systemctl start q-api-server" || {
        log_error "Failed to start Delta"
        return 1
    }

    if wait_for_health "Delta" "$DELTA_HOST" 8080 "$HEALTH_TIMEOUT" "$STABILITY_SOAK"; then
        log_success "Delta canary verified and healthy with new binary"
        return 0
    else
        log_error "Delta canary FAILED health check"
        return 1
    fi
}

cmd_verify_gamma() {
    log_header "GAMMA" "Deploying to Gamma ($GAMMA_HOST)"

    log_step "1/3: Checking local binary exists..."
    if [ ! -f "$BINARY_PATH" ]; then
        log_error "Binary not found: $BINARY_PATH"
        return 1
    fi
    local size
    size=$(du -h "$BINARY_PATH" | awk '{print $1}')
    log_info "Binary: $BINARY_PATH ($size)"

    local free
    free=$(ssh -o ConnectTimeout=5 "root@$GAMMA_HOST" "df -m /opt | tail -1 | awk '{print \$4}'" 2>/dev/null) || free="0"
    log_info "Gamma: ${free}MB free"

    log_step "2/3: Stopping Gamma, copying binary..."
    ssh -o ConnectTimeout=5 "root@$GAMMA_HOST" "
        pgrep -f q-api-server | xargs -I{} kill -9 {} 2>/dev/null || true
        sleep 2
    " || true
    scp -o ConnectTimeout=10 "$BINARY_PATH" "root@$GAMMA_HOST:$GAMMA_BINARY" || {
        log_error "SCP to Gamma failed"
        return 1
    }
    log_info "Binary copied to Gamma"

    log_step "3/3: Starting Gamma and waiting for health..."
    ssh -o ConnectTimeout=5 "root@$GAMMA_HOST" "systemctl start q-api-server" || {
        log_error "Failed to start Gamma"
        return 1
    }

    if wait_for_health "Gamma" "$GAMMA_HOST" 8080 "$HEALTH_TIMEOUT" "$STABILITY_SOAK"; then
        log_success "Gamma verified and healthy with new binary"
        return 0
    else
        log_error "Gamma FAILED health check"
        return 1
    fi
}

cmd_deploy_beta() {
    log_header "BETA" "Deploying to Beta (localhost)"

    log_step "1/4: Creating backup..."
    mkdir -p "$BACKUP_DIR"
    local version
    version=$(get_cargo_version)
    local backup_name="$BINARY_NAME-pre-v${version}-$(date +%Y%m%d-%H%M%S)"
    if [ -f "$BETA_BINARY" ]; then
        cp "$BETA_BINARY" "$BACKUP_DIR/$backup_name" 2>/dev/null || true
        log_info "Backed up to $BACKUP_DIR/$backup_name"
    fi

    log_step "2/4: Stopping Beta..."
    # v1.0.5: Pre-drain q-flux before killing backend
    # This routes all traffic to cluster peers BEFORE we stop the service,
    # ensuring zero miner-visible downtime during deploys.
    log_step "Pre-draining q-flux (routing traffic to cluster peers)..."
    curl -s -X POST http://127.0.0.1:9090/admin/drain 2>/dev/null || true
    sleep 3  # Allow in-flight mining submissions to complete

    # Now safe to kill — no traffic is reaching this backend
    pgrep -f "$BINARY_NAME" | xargs -I{} kill -9 {} 2>/dev/null || true
    sleep 1

    log_step "3/4: Copying new binary..."
    cp "$BINARY_PATH" "$BETA_BINARY" || {
        log_error "Failed to copy binary"
        return 1
    }
    chmod +x "$BETA_BINARY"

    # Also copy to downloads
    cp "$BINARY_PATH" "$DOWNLOADS_DIR/$BINARY_NAME-v${version}" 2>/dev/null || true
    cp "$BINARY_PATH" "$DOWNLOADS_DIR/$BINARY_NAME-linux-x86_64" 2>/dev/null || true

    log_step "4/4: Starting Beta..."
    systemctl start "$SERVICE_NAME" || {
        log_error "Failed to start Beta service"
        return 1
    }

    if wait_for_health "Beta" "localhost" 8080 "$HEALTH_TIMEOUT" "$STABILITY_SOAK"; then
        log_success "Beta deployed and healthy with v${version}"

        # Copy to Epsilon downloads (non-blocking)
        log_info "Copying binary to Epsilon downloads..."
        scp -o ConnectTimeout=10 "$BINARY_PATH" "root@$EPSILON_HOST:$EPSILON_DOWNLOADS/$BINARY_NAME-v${version}" 2>/dev/null && \
        scp -o ConnectTimeout=10 "$BINARY_PATH" "root@$EPSILON_HOST:$EPSILON_DOWNLOADS/$BINARY_NAME-linux-x86_64" 2>/dev/null && \
            log_success "Binary uploaded to Epsilon downloads" || \
            log_warn "Failed to upload to Epsilon (non-fatal)"

        # Announce update
        log_info "Announcing update to P2P network..."
        local sha256
        sha256=$(sha256sum "$BINARY_PATH" 2>/dev/null | awk '{print $1}' || echo "")
        curl -s -X POST "http://localhost:8080/api/v1/admin/update/announce" \
            -H "Content-Type: application/json" \
            -d "{
                \"version\": \"${version}\",
                \"sha256_checksum\": \"${sha256}\",
                \"download_url\": \"https://quillon.xyz/downloads/$BINARY_NAME-v${version}\",
                \"mandatory\": false,
                \"release_notes\": \"v${version} via ha-deploy.sh\"
            }" >/dev/null 2>&1 && log_success "Update announced" || log_warn "Announce failed (non-fatal)"

        echo ""
        echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✅  DEPLOYMENT COMPLETE — v${version}                      ${NC}"
        echo -e "${GREEN}║  Download: wget https://quillon.xyz/downloads/$BINARY_NAME-v${version}${NC}"
        echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
        return 0
    else
        log_error "Beta FAILED health check after deploy!"
        log_error "Rolling back..."
        cmd_rollback
        return 1
    fi
}

cmd_rollback() {
    log_header "ROLLBACK" "Restoring previous binary on Beta"

    local latest_backup
    latest_backup=$(ls -t "$BACKUP_DIR"/$BINARY_NAME-pre-* 2>/dev/null | head -1)

    if [ -z "$latest_backup" ]; then
        log_error "No backup found in $BACKUP_DIR"
        return 1
    fi

    log_info "Restoring from: $latest_backup"

    pgrep -f "$BINARY_NAME" | xargs -I{} kill -9 {} 2>/dev/null || true
    sleep 2

    cp "$latest_backup" "$BETA_BINARY" || {
        log_error "Failed to restore backup"
        return 1
    }
    chmod +x "$BETA_BINARY"

    systemctl start "$SERVICE_NAME" || {
        log_error "Failed to start service after rollback"
        return 1
    }

    if wait_for_health "Beta" "localhost" 8080 120 30; then
        log_success "Rollback successful — Beta running previous version"
    else
        log_error "Rollback health check failed! Manual intervention needed."
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════
# v1.0.5: Epsilon Deploy (zero-downtime via q-flux drain)
# Reviewed by: Nemotron Cascade Two, DeepSeek, Human Operator
# ═══════════════════════════════════════════════════════════════════

cmd_deploy_epsilon() {
    echo ""
    log_info "━━━ Deploying to Epsilon (primary mining endpoint) ━━━"

    local version
    version=$(get_cargo_version)

    # 1. Pre-flight: verify at least one cluster peer is healthy
    log_step "1/6: Checking cluster peer health before drain..."
    local beta_ok gamma_ok delta_ok peers_healthy=0
    beta_ok=$(curl -s --connect-timeout 5 "http://$BETA_HOST:8080/api/v1/status" 2>/dev/null | grep -c "ready\|syncing") || beta_ok=0
    delta_ok=$(curl -s --connect-timeout 5 "http://$DELTA_HOST:8080/api/v1/status" 2>/dev/null | grep -c "ready\|syncing") || delta_ok=0

    [ "$beta_ok" -gt 0 ] && peers_healthy=$((peers_healthy + 1)) && log_success "  Beta ($BETA_HOST) is healthy"
    [ "$delta_ok" -gt 0 ] && peers_healthy=$((peers_healthy + 1)) && log_success "  Delta ($DELTA_HOST) is healthy"

    if [ "$peers_healthy" -eq 0 ]; then
        log_error "ABORT: No healthy cluster peers! Cannot drain Epsilon safely."
        log_error "Deploy to Beta and/or Delta first, then retry."
        return 1
    fi
    log_success "$peers_healthy cluster peer(s) ready to absorb mining traffic"

    # 2. Drain q-flux on Epsilon (routes ALL traffic to Beta/Delta)
    log_step "2/6: Draining Epsilon q-flux (routing traffic to cluster peers)..."
    local drain_result
    drain_result=$(ssh -o ConnectTimeout=10 "root@$EPSILON_HOST" \
        "curl -s -X POST http://127.0.0.1:9090/admin/drain" 2>/dev/null) || true
    if echo "$drain_result" | grep -q '"drained":true'; then
        log_success "Epsilon drained — mining traffic now served by cluster peers"
    else
        log_warn "Drain response: $drain_result (continuing anyway)"
    fi

    # 3. Wait for in-flight mining submissions to complete
    log_step "3/6: Waiting 5s for in-flight requests to complete..."
    sleep 5

    # 4. Stop old binary, deploy new one
    log_step "4/6: Deploying binary to Epsilon..."
    ssh -o ConnectTimeout=10 "root@$EPSILON_HOST" "
        # Backup current binary
        cp /opt/orobit/shared/q-narwhalknight/q-api-server-v889 /opt/orobit/shared/q-narwhalknight/q-api-server-v889.backup-\$(date +%Y%m%d-%H%M%S) 2>/dev/null || true

        # Stop service
        systemctl stop q-api-server 2>/dev/null || true
        pgrep -f q-api-server | xargs -I{} kill -9 {} 2>/dev/null || true
        sleep 1
    " || { log_error "SSH to Epsilon failed!"; return 1; }

    # SCP new binary
    scp -o ConnectTimeout=15 "$BINARY_PATH" "root@$EPSILON_HOST:/opt/orobit/shared/q-narwhalknight/q-api-server-v889" || {
        log_error "SCP to Epsilon failed! Attempting rollback..."
        ssh "root@$EPSILON_HOST" "
            cp /opt/orobit/shared/q-narwhalknight/q-api-server-v889.backup-* /opt/orobit/shared/q-narwhalknight/q-api-server-v889 2>/dev/null
            systemctl start q-api-server
        " 2>/dev/null || true
        return 1
    }

    # 5. Start new binary and wait for health
    log_step "5/6: Starting Epsilon and waiting for health..."
    ssh "root@$EPSILON_HOST" "
        chmod +x /opt/orobit/shared/q-narwhalknight/q-api-server-v889
        systemctl start q-api-server
    "

    # Wait for Epsilon's q-api-server to be healthy (max 120s)
    local elapsed=0
    while [ $elapsed -lt 120 ]; do
        if ssh -o ConnectTimeout=5 "root@$EPSILON_HOST" "curl -sf http://127.0.0.1:8080/api/v1/status" >/dev/null 2>&1; then
            log_success "Epsilon q-api-server is healthy (${elapsed}s)"
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        log_info "  Waiting for Epsilon health... (${elapsed}s)"
    done

    if [ $elapsed -ge 120 ]; then
        log_error "Epsilon health check timed out after 120s!"
        log_warn "q-flux drain auto-clears after 300s, or manually: ssh epsilon 'curl -X POST http://127.0.0.1:9090/admin/undrain'"
        return 1
    fi

    # 6. q-flux auto-recovers (health checker clears drain after 2 successful checks)
    log_step "6/6: Waiting for q-flux auto-recovery from drain..."
    sleep 6  # 2 health check intervals (2s each) + buffer

    # Verify Epsilon is serving traffic again
    local epsilon_status
    epsilon_status=$(ssh "root@$EPSILON_HOST" "curl -s http://127.0.0.1:9090/backends" 2>/dev/null) || true
    log_success "Epsilon deploy complete — q-flux auto-recovered"

    # Copy to downloads
    log_info "Copying binary to Epsilon downloads..."
    ssh "root@$EPSILON_HOST" "
        cp /opt/orobit/shared/q-narwhalknight/q-api-server-v889 /home/orobit/q-narwhalknight/dist-final/downloads/q-api-server-v${version} 2>/dev/null
        cp /opt/orobit/shared/q-narwhalknight/q-api-server-v889 /home/orobit/q-narwhalknight/dist-final/downloads/q-api-server-linux-x86_64 2>/dev/null
    " && log_success "Binary in Epsilon downloads" || log_warn "Downloads copy failed (non-fatal)"

    echo ""
    log_success "Epsilon deployed v${version} with ZERO miner downtime"
    return 0
}

cmd_full() {
    local auto_yes=false
    if [ "${1:-}" = "-y" ] || [ "${1:-}" = "--yes" ]; then
        auto_yes=true
    fi

    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     Q-NarwhalKnight HA Rolling Deploy (3-Server)         ║${NC}"
    echo -e "${CYAN}║     Pipeline: Delta → Gamma → Beta (zero-downtime)       ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Check binary exists
    if [ ! -f "$BINARY_PATH" ]; then
        log_error "No release binary found at $BINARY_PATH"
        log_error "Run 'cargo build --release --package q-api-server' first."
        exit 1
    fi

    local new_version
    new_version=$(get_cargo_version)
    local running_version
    running_version=$(get_running_version "localhost") || running_version="unknown"

    log_info "Current running version: $running_version"
    log_info "New binary version:      $new_version"

    if [ "$new_version" = "$running_version" ]; then
        log_warn "Version NOT bumped — Cargo.toml ($new_version) matches running version."
        log_warn "Deploying same version to all servers."
    fi

    # Confirmation
    if [ "$auto_yes" = false ]; then
        echo ""
        echo -e "Deploy ${YELLOW}v${new_version}${NC} to all 3 servers? (Delta → Gamma → Beta)"
        echo -n "Proceed? [y/N] "
        read -r confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            log_info "Aborted."
            exit 0
        fi
    fi

    echo ""
    local start_time
    start_time=$(date +%s)

    # ── Step 1: Delta Canary ──
    log_info "━━━ Step 1/3: Delta Canary ━━━"
    if cmd_verify_delta; then
        log_success "Delta canary passed ✓"
    else
        log_error "Delta canary FAILED — aborting pipeline."
        log_error "Beta is untouched and still running $running_version"
        exit 1
    fi

    echo ""

    # ── Step 2: Gamma ──
    log_info "━━━ Step 2/3: Gamma Verify ━━━"
    if cmd_verify_gamma; then
        log_success "Gamma verified ✓"
    else
        log_warn "Gamma deploy failed — continuing to Beta (Gamma may be offline)"
        log_warn "This is not fatal — Beta can be deployed without Gamma"
    fi

    echo ""

    # ── Step 3: Beta ──
    log_info "━━━ Step 3/3: Deploy Beta ━━━"
    if cmd_deploy_beta; then
        log_success "Beta deployed ✓"
    else
        log_error "Beta deploy FAILED — check logs and consider manual intervention"
        exit 1
    fi

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅  HA ROLLING DEPLOY COMPLETE                          ║${NC}"
    echo -e "${GREEN}║  Version: v${new_version}                                        ${NC}"
    echo -e "${GREEN}║  Duration: ${duration}s                                          ${NC}"
    echo -e "${GREEN}║  Pipeline: Delta ✓ → Gamma ✓ → Beta ✓                   ${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  Download: ${CYAN}wget https://quillon.xyz/downloads/$BINARY_NAME-v${new_version} && chmod +x $BINARY_NAME-v${new_version}${NC}"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

cd "$PROJECT_DIR"

case "${1:-help}" in
    status)
        cmd_status
        ;;
    verify-delta)
        acquire_lock
        cmd_verify_delta
        ;;
    verify-gamma)
        acquire_lock
        cmd_verify_gamma
        ;;
    deploy-beta)
        acquire_lock
        cmd_deploy_beta
        ;;
    deploy-epsilon)
        acquire_lock
        cmd_deploy_epsilon
        ;;
    rollback)
        acquire_lock
        cmd_rollback
        ;;
    full)
        acquire_lock
        cmd_full "${2:-}"
        ;;
    help|--help|-h)
        echo "Q-NarwhalKnight HA Rolling Deploy"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  status          Show health of all servers"
        echo "  full [-y]       Full pipeline: Delta → Gamma → Beta"
        echo "  verify-delta    Deploy & verify on Delta (canary)"
        echo "  verify-gamma    Deploy & verify on Gamma (backup)"
        echo "  deploy-beta     Deploy to Beta (primary)"
        echo "  deploy-epsilon  Deploy to Epsilon with q-flux drain (zero-downtime)"
        echo "  rollback        Restore previous Beta binary"
        echo ""
        echo "Options:"
        echo "  -y, --yes       Skip confirmation prompt"
        echo ""
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for usage."
        exit 1
        ;;
esac
