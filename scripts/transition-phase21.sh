#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Phase 21 Transition Script
# ═══════════════════════════════════════════════════════════════════
# This script transitions Server Beta from testnet-phase20 to testnet-phase21
#
# What it does:
#   1. Creates fresh data-mine21 directory
#   2. Preserves libp2p identity key
#   3. Generates fresh encryption keys
#   4. Installs Phase 21 systemd service
#   5. Rebuilds and deploys the binary
#
# Usage: ./scripts/transition-phase21.sh [beta|gamma|both]
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET=${1:-beta}

echo "═══════════════════════════════════════════════════════════════"
echo "  Q-NarwhalKnight Phase 21 Transition"
echo "  Target: $TARGET"
echo "  Genesis: Feb 15, 2026 00:00:00 UTC (1771113600)"
echo "═══════════════════════════════════════════════════════════════"

transition_beta() {
    echo ""
    echo "━━━ Step 1: Stop Beta service ━━━"
    systemctl stop q-api-server || true
    sleep 3

    echo ""
    echo "━━━ Step 2: Create fresh data-mine21 directory ━━━"
    if [ -d "$PROJECT_DIR/data-mine21" ]; then
        echo "⚠️  data-mine21 already exists! Skipping creation."
    else
        mkdir -p "$PROJECT_DIR/data-mine21"
        echo "✅ Created data-mine21"
    fi

    echo ""
    echo "━━━ Step 3: Copy libp2p identity key ━━━"
    if [ -f "$PROJECT_DIR/data-mine20/libp2p_identity.key" ]; then
        cp "$PROJECT_DIR/data-mine20/libp2p_identity.key" "$PROJECT_DIR/data-mine21/"
        echo "✅ Copied libp2p_identity.key from data-mine20"
    else
        echo "⚠️  No identity key in data-mine20. New identity will be generated."
    fi

    echo ""
    echo "━━━ Step 4: Encryption keys ━━━"
    if [ -f "/opt/encryption-phase21.keys" ]; then
        echo "⚠️  /opt/encryption-phase21.keys already exists! Removing to let server auto-create."
        rm -f /opt/encryption-phase21.keys
    fi
    echo "✅ Server will auto-generate encryption keys on first start"

    echo ""
    echo "━━━ Step 5: Install Phase 21 systemd service ━━━"
    cp "$SCRIPT_DIR/phase21-service-beta.service" /etc/systemd/system/q-api-server.service
    systemctl daemon-reload
    echo "✅ Installed Phase 21 service (Q_DB_PATH=./data-mine21, Q_NETWORK_ID=testnet-phase21)"

    echo ""
    echo "━━━ Step 6: Start Beta with Phase 21 ━━━"
    systemctl start q-api-server
    echo "✅ Started q-api-server"

    echo ""
    echo "━━━ Step 7: Verify ━━━"
    sleep 5
    if systemctl is-active --quiet q-api-server; then
        echo "✅ q-api-server is running"
        journalctl -u q-api-server --since "10 seconds ago" --no-pager | head -20
    else
        echo "❌ q-api-server failed to start!"
        journalctl -u q-api-server --since "30 seconds ago" --no-pager | tail -30
        exit 1
    fi
}

transition_gamma() {
    echo ""
    echo "━━━ Transitioning Gamma (109.205.176.60) ━━━"

    # Stop Gamma
    ssh root@109.205.176.60 "systemctl stop q-api-server" || true
    sleep 2

    # Create data-mine21 on Gamma
    ssh root@109.205.176.60 "mkdir -p /opt/orobit/shared/q-narwhalknight/data-mine21"

    # Copy identity key on Gamma
    ssh root@109.205.176.60 "
        if [ -f /opt/orobit/shared/q-narwhalknight/data-mine20/libp2p_identity.key ]; then
            cp /opt/orobit/shared/q-narwhalknight/data-mine20/libp2p_identity.key /opt/orobit/shared/q-narwhalknight/data-mine21/
            echo '✅ Copied identity key'
        else
            echo '⚠️  No identity key found'
        fi
    "

    # Remove any bad encryption keys on Gamma (server auto-creates proper format)
    ssh root@109.205.176.60 "rm -f /opt/encryption-phase21.keys && echo '✅ Server will auto-generate encryption keys'"

    # Copy binary to Gamma
    echo "Copying binary to Gamma..."
    scp "$PROJECT_DIR/target/release/q-api-server" root@109.205.176.60:/opt/orobit/shared/q-narwhalknight/q-api-server

    # Copy and install service file
    scp "$SCRIPT_DIR/phase21-service-gamma.service" root@109.205.176.60:/etc/systemd/system/q-api-server.service
    ssh root@109.205.176.60 "systemctl daemon-reload && systemctl start q-api-server"

    echo ""
    echo "━━━ Verifying Gamma ━━━"
    sleep 5
    ssh root@109.205.176.60 "systemctl is-active q-api-server && echo '✅ Gamma running' || echo '❌ Gamma failed'"
}

case "$TARGET" in
    beta)
        transition_beta
        ;;
    gamma)
        transition_gamma
        ;;
    both)
        transition_gamma
        transition_beta
        ;;
    *)
        echo "Usage: $0 [beta|gamma|both]"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 21 Transition Complete for: $TARGET"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Post-transition checklist:"
echo "  [ ] Verify Beta and Gamma are producing blocks on Phase 21"
echo "  [ ] Check: journalctl -u q-api-server | grep 'testnet-phase21'"
echo "  [ ] Update frontend: cd gui/quantum-wallet && npm run build"
echo "  [ ] Copy dist to dist-final"
echo "  [ ] Stop any Alpha Phase 20 containers"
echo "  [ ] Update CLAUDE.md with new peer IDs if changed"
echo "  [ ] Copy binary to downloads/"
