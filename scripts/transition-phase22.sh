#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════
# Q-NarwhalKnight Phase 22 Transition Script
# This script transitions servers from testnet-phase21 to testnet-phase22
# ═══════════════════════════════════════════════════════════════════
#
# PREREQUISITES:
#   1. Binary already built: cargo build --release --package q-api-server
#   2. All code changes committed (NetworkId enum, GENESIS_TIMESTAMP, etc.)
#   3. Frontend rebuilt: cd gui/quantum-wallet && npm run build
#
# Usage: ./scripts/transition-phase22.sh [beta|gamma|both]
#   beta  - Transition Server Beta only
#   gamma - Transition Server Gamma only
#   both  - Transition both servers (recommended)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="/opt/orobit/shared/q-narwhalknight"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              Q-NarwhalKnight Phase 22 Transition              ║"
echo "║                                                               ║"
echo "║  Genesis: Feb 16, 2026 00:00:00 UTC (1771200000)             ║"
echo "║  Version: v6.5.0-beta                                        ║"
echo "║  Database: data-mine22                                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

TARGET=${1:-both}
echo "Target: $TARGET"
echo ""

# ═══════════════════════════════════════════════════════════════════
# BETA TRANSITION
# ═══════════════════════════════════════════════════════════════════
if [ "$TARGET" = "beta" ] || [ "$TARGET" = "both" ]; then
    echo "🔧 [BETA] Starting Phase 22 transition..."

    # Kill any running miner processes
    echo "  Stopping miner processes..."
    pgrep -f "q-miner" | xargs -I{} kill -9 {} 2>/dev/null || true

    # Stop the current service
    echo "  Stopping q-api-server service..."
    systemctl stop q-api-server 2>/dev/null || true
    sleep 2

    # Remove old encryption keys (Bug #15: NEVER pre-generate! Let server auto-create)
    echo "  Removing old encryption keys (server will auto-generate proper format)..."
    rm -f /opt/encryption-phase22.keys
    rm -f /opt/encryption-phase21.keys

    # Install Phase 22 service file
    echo "  Installing Phase 22 service file..."
    cp "$SCRIPT_DIR/phase22-service-beta.service" /etc/systemd/system/q-api-server.service
    systemctl daemon-reload
    echo "  ✅ Installed Phase 22 service (data-mine22, testnet-phase22)"

    # Start the service
    echo "  Starting q-api-server on Phase 22..."
    systemctl start q-api-server
    sleep 5

    # Verify it started
    if systemctl is-active --quiet q-api-server; then
        echo "  ✅ [BETA] q-api-server started successfully on Phase 22!"
    else
        echo "  ❌ [BETA] q-api-server FAILED to start! Check: journalctl -u q-api-server -n 50"
        if [ "$TARGET" = "beta" ]; then exit 1; fi
    fi

    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# GAMMA TRANSITION
# ═══════════════════════════════════════════════════════════════════
if [ "$TARGET" = "gamma" ] || [ "$TARGET" = "both" ]; then
    echo "🔧 [GAMMA] Starting Phase 22 transition..."

    # Stop Gamma
    echo "  Stopping Gamma q-api-server..."
    ssh root@109.205.176.60 "systemctl stop q-api-server 2>/dev/null || true" 2>/dev/null

    # Remove old encryption keys on Gamma (Bug #15 prevention)
    echo "  Removing old encryption keys on Gamma..."
    ssh root@109.205.176.60 "rm -f /opt/encryption-phase22.keys /opt/encryption-phase21.keys" 2>/dev/null

    # Copy binary to Gamma
    echo "  Copying binary to Gamma..."
    scp "$WORK_DIR/target/release/q-api-server" root@109.205.176.60:/opt/orobit/shared/q-narwhalknight/q-api-server

    # Copy and install service file
    echo "  Installing Phase 22 service on Gamma..."
    scp "$SCRIPT_DIR/phase22-service-gamma.service" root@109.205.176.60:/etc/systemd/system/q-api-server.service
    ssh root@109.205.176.60 "systemctl daemon-reload"

    # Start Gamma
    echo "  Starting Gamma on Phase 22..."
    ssh root@109.205.176.60 "systemctl start q-api-server"
    sleep 5

    # Verify Gamma started
    GAMMA_STATUS=$(ssh root@109.205.176.60 "systemctl is-active q-api-server" 2>/dev/null || echo "failed")
    if [ "$GAMMA_STATUS" = "active" ]; then
        echo "  ✅ [GAMMA] q-api-server started successfully on Phase 22!"
    else
        echo "  ❌ [GAMMA] q-api-server FAILED to start! Check: ssh root@109.205.176.60 journalctl -u q-api-server -n 50"
    fi

    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# POST-TRANSITION VERIFICATION
# ═══════════════════════════════════════════════════════════════════
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              Phase 22 Transition Complete for: $TARGET"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Post-transition checklist:                                   ║"
echo "║  [ ] Check: journalctl -u q-api-server | grep 'testnet-phase22'"
echo "║  [ ] Check: curl -s localhost:8080/api/v1/status | grep phase  ║"
echo "║  [ ] Verify no stale tokens: curl -s localhost:8080/api/v1/dex/supported-tokens"
echo "║  [ ] Call full purge if stale data: POST /api/v1/admin/purge-phase-data?full=true"
echo "║  [ ] Rebuild frontend: cd gui/quantum-wallet && npm run build  ║"
echo "║  [ ] Verify frontend phase: grep testnet-phase dist-final/assets/index-*.js"
echo "║  [ ] Update bootstrap peer IDs after servers generate new keys ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
