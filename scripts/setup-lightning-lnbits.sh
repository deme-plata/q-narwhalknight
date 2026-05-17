#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Lightning Network + LNbits Setup for Q-NarwhalKnight Bridge
# Server Delta (5.79.79.158)
# ═══════════════════════════════════════════════════════════════════
#
# Stack Architecture:
#   Bitcoin Knots v28.1 (L1) → Core Lightning (CLN) (L2) → LNbits (L3)
#                                                          └→ Boltz Extension (submarine swaps)
#
# Prerequisites:
#   - Bitcoin Knots FULLY SYNCED (check: getblockcount == headers)
#   - Docker installed
#   - Ports 8332, 8333, 28332-28334 accessible
#
# Usage:
#   ./scripts/setup-lightning-lnbits.sh status    # Check Bitcoin sync progress
#   ./scripts/setup-lightning-lnbits.sh btc-zmq   # Recreate BTC container with ZMQ ports
#   ./scripts/setup-lightning-lnbits.sh cln        # Install Core Lightning
#   ./scripts/setup-lightning-lnbits.sh lnbits     # Install LNbits
#   ./scripts/setup-lightning-lnbits.sh full       # Full setup (all steps)
#
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

DELTA_HOST="5.79.79.158"
BTC_DATA="/home/bitcoin/data"
BTC_BIN="/home/bitcoin/knots/bitcoin-28.1.knots20250305/bin"
CLN_DATA="/home/lightning/cln-data"
LNBITS_DIR="/home/lightning/lnbits"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ─── Check Bitcoin Knots sync status ───
cmd_status() {
    log_info "Checking Bitcoin Knots sync status on Delta ($DELTA_HOST)..."

    local result
    result=$(ssh root@${DELTA_HOST} "curl -s --connect-timeout 10 --max-time 15 \
        --user qnk:QnkBtcBridge2026 \
        -d '{\"jsonrpc\":\"1.0\",\"id\":1,\"method\":\"getblockchaininfo\",\"params\":[]}' \
        -H 'Content-Type: application/json' http://127.0.0.1:8332/" 2>/dev/null) || true

    if [ -z "$result" ]; then
        log_error "Bitcoin Knots RPC not responding (node may be starting up)"
        return 1
    fi

    local blocks headers progress chain
    blocks=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['blocks'])" 2>/dev/null || echo "0")
    headers=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['headers'])" 2>/dev/null || echo "0")
    progress=$(echo "$result" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)['result']['verificationprogress']*100:.2f}\")" 2>/dev/null || echo "0")
    chain=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['chain'])" 2>/dev/null || echo "unknown")

    echo ""
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║   Bitcoin Knots v28.1 — Delta Node       ║"
    echo "  ╠══════════════════════════════════════════╣"
    echo "  ║  Chain:    $chain"
    echo "  ║  Blocks:   $blocks / $headers"
    echo "  ║  Progress: ${progress}%"

    if [ "$blocks" = "$headers" ] || [ "$(echo "$progress > 99.9" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        echo -e "  ║  Status:   ${GREEN}✅ FULLY SYNCED${NC}"
        echo "  ║  Lightning Ready: YES"
    else
        local remaining=$((headers - blocks))
        echo -e "  ║  Status:   ${YELLOW}⏳ SYNCING ($remaining blocks remaining)${NC}"
        echo "  ║  Lightning Ready: NO (must be fully synced first)"
    fi
    echo "  ╚══════════════════════════════════════════╝"
    echo ""

    # Check ZMQ config
    local zmq_config
    zmq_config=$(ssh root@${DELTA_HOST} "docker exec bitcoin-knots grep zmq /bitcoin/data/bitcoin.conf" 2>/dev/null || echo "")
    if [ -n "$zmq_config" ]; then
        log_info "ZMQ configured in bitcoin.conf ✅"
    else
        log_warn "ZMQ NOT configured — run: $0 btc-zmq"
    fi

    # Check ZMQ ports
    local zmq_ports
    zmq_ports=$(ssh root@${DELTA_HOST} "docker port bitcoin-knots 2>/dev/null | grep 28332" 2>/dev/null || echo "")
    if [ -n "$zmq_ports" ]; then
        log_info "ZMQ ports exposed ✅"
    else
        log_warn "ZMQ ports NOT exposed — need to recreate container with: $0 btc-zmq"
    fi
}

# ─── Recreate Bitcoin Knots container with ZMQ ports ───
cmd_btc_zmq() {
    log_info "Recreating Bitcoin Knots container with ZMQ ports..."
    log_warn "This will briefly stop Bitcoin Knots. Data is preserved on volume."

    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted."
        return 0
    fi

    ssh root@${DELTA_HOST} bash <<'REMOTE_SCRIPT'
set -e

echo "Stopping old container..."
docker stop bitcoin-knots 2>/dev/null || true
docker rm bitcoin-knots 2>/dev/null || true

echo "Creating new container with ZMQ ports..."
docker run -d \
    --name bitcoin-knots \
    --restart unless-stopped \
    -p 8332:8332 \
    -p 8333:8333 \
    -p 28332:28332 \
    -p 28333:28333 \
    -p 28334:28334 \
    -v /home/bitcoin/data:/bitcoin/data \
    -v /home/bitcoin/knots/bitcoin-28.1.knots20250305/bin:/bitcoin/bin \
    debian:12-slim \
    /bitcoin/bin/bitcoind -conf=/bitcoin/data/bitcoin.conf -datadir=/bitcoin/data

echo "Waiting for startup..."
sleep 5

# Verify
if docker ps | grep -q bitcoin-knots; then
    echo "✅ Bitcoin Knots restarted with ZMQ ports (28332-28334)"
else
    echo "❌ Container failed to start!"
    docker logs bitcoin-knots --tail 20
    exit 1
fi
REMOTE_SCRIPT

    log_info "Bitcoin Knots restarted with ZMQ support"
}

# ─── Install Core Lightning (CLN) ───
cmd_cln() {
    log_info "Installing Core Lightning on Delta..."

    # Check Bitcoin sync first
    cmd_status 2>/dev/null || true

    ssh root@${DELTA_HOST} bash <<'REMOTE_SCRIPT'
set -e

CLN_VERSION="24.11.1"
CLN_DATA="/home/lightning/cln-data"

# Create directories
mkdir -p /home/lightning/cln-data
mkdir -p /home/lightning/cln-plugins

# Check if CLN already installed
if docker ps -a | grep -q cln-node; then
    echo "CLN container already exists"
    docker start cln-node 2>/dev/null || true
    exit 0
fi

echo "Pulling CLN Docker image..."
docker pull elementsproject/lightningd:v${CLN_VERSION} 2>/dev/null || \
    docker pull elementsproject/lightningd:latest

# Create CLN config
cat > ${CLN_DATA}/config <<EOF
# Core Lightning Configuration for Q-NarwhalKnight Bridge
network=bitcoin
alias=QNK-Lightning-Bridge
rgb=FF6600

# Bitcoin Knots connection
bitcoin-rpcconnect=172.17.0.1
bitcoin-rpcport=8332
bitcoin-rpcuser=qnk
bitcoin-rpcpassword=QnkBtcBridge2026

# Network
bind-addr=0.0.0.0:9735
announce-addr=$(curl -s ifconfig.me):9735

# REST API (for LNbits)
clnrest-port=3010
clnrest-protocol=https

# Fee settings
fee-base=1000
fee-per-satoshi=1

# Channel settings
min-capacity-sat=50000
large-channels

# Logging
log-level=info
log-file=/home/lightning/cln-data/cln.log

# Auto liquidity
autoclean-cycle=3600
EOF

echo "Starting Core Lightning..."
docker run -d \
    --name cln-node \
    --restart unless-stopped \
    -p 9735:9735 \
    -p 3010:3010 \
    -v /home/lightning/cln-data:/home/lightning/cln-data \
    -v /home/lightning/cln-plugins:/home/lightning/cln-plugins \
    --network host \
    elementsproject/lightningd:latest \
    --conf=/home/lightning/cln-data/config

echo "Waiting for CLN startup..."
sleep 10

if docker ps | grep -q cln-node; then
    echo "✅ Core Lightning installed and running"
    echo "   Port 9735 (P2P), Port 3010 (REST API)"
    # Get node info
    docker exec cln-node lightning-cli getinfo 2>/dev/null | head -20 || echo "(node still starting up)"
else
    echo "❌ CLN failed to start"
    docker logs cln-node --tail 20
fi
REMOTE_SCRIPT

    log_info "Core Lightning setup complete"
}

# ─── Install LNbits ───
cmd_lnbits() {
    log_info "Installing LNbits on Delta..."

    ssh root@${DELTA_HOST} bash <<'REMOTE_SCRIPT'
set -e

LNBITS_DIR="/home/lightning/lnbits"

# Check if already installed
if docker ps -a | grep -q lnbits; then
    echo "LNbits container already exists"
    docker start lnbits 2>/dev/null || true
    exit 0
fi

mkdir -p ${LNBITS_DIR}/data

# Create LNbits .env config
cat > ${LNBITS_DIR}/.env <<EOF
# LNbits Configuration for Q-NarwhalKnight Bridge

# Core settings
HOST=0.0.0.0
PORT=5000
DEBUG=false
LNBITS_ADMIN_UI=true
LNBITS_SITE_TITLE=QNK Lightning Bridge
LNBITS_SITE_TAGLINE=Bitcoin Lightning Swaps powered by Q-NarwhalKnight

# Funding source: Core Lightning (CLNRest)
LNBITS_BACKEND_WALLET_CLASS=CLNRestWallet
CLNREST_URL=https://127.0.0.1:3010
CLNREST_CA=/home/lightning/cln-data/ca.pem

# Database (SQLite for now, Postgres for production)
LNBITS_DATA_FOLDER=/data

# Extensions to auto-enable
LNBITS_EXTENSIONS_DEFAULT_INSTALL=boltz

# Security
LNBITS_ALLOWED_IPS=*
EOF

echo "Starting LNbits..."
docker run -d \
    --name lnbits \
    --restart unless-stopped \
    -p 5000:5000 \
    -v ${LNBITS_DIR}/data:/data \
    -v ${LNBITS_DIR}/.env:/app/.env \
    -v /home/lightning/cln-data:/home/lightning/cln-data:ro \
    --network host \
    lnbitsdocker/lnbits:latest

echo "Waiting for LNbits startup..."
sleep 10

if docker ps | grep -q lnbits; then
    echo "✅ LNbits installed and running on port 5000"
    echo "   Access: http://5.79.79.158:5000"
    echo "   Admin UI: http://5.79.79.158:5000/admin"
    echo ""
    echo "   Next steps:"
    echo "   1. Open http://5.79.79.158:5000 in browser"
    echo "   2. Create a wallet"
    echo "   3. Enable the Boltz Swaps extension for submarine swaps"
    echo "   4. Fund channels via CLN for liquidity"
else
    echo "❌ LNbits failed to start"
    docker logs lnbits --tail 20
fi
REMOTE_SCRIPT

    log_info "LNbits setup complete"
}

# ─── Full setup ───
cmd_full() {
    log_info "Full Lightning + LNbits setup"
    echo ""

    cmd_status

    echo ""
    log_warn "Bitcoin Knots must be FULLY SYNCED before Lightning can work."
    log_warn "Current sync status shown above."
    echo ""
    read -p "Continue with setup? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted. Run individual steps when ready."
        return 0
    fi

    cmd_btc_zmq
    echo ""
    cmd_cln
    echo ""
    cmd_lnbits

    echo ""
    log_info "═══════════════════════════════════════════"
    log_info "  Lightning + LNbits Setup Complete!"
    log_info "═══════════════════════════════════════════"
    echo ""
    echo "  Stack:"
    echo "    Bitcoin Knots v28.1  →  :8332 (RPC), :28332-28334 (ZMQ)"
    echo "    Core Lightning       →  :9735 (P2P), :3010 (REST)"
    echo "    LNbits               →  :5000 (Web UI)"
    echo ""
    echo "  Next steps:"
    echo "    1. Wait for Bitcoin to fully sync"
    echo "    2. Fund CLN wallet: lightning-cli newaddr"
    echo "    3. Open channels for liquidity"
    echo "    4. Access LNbits: http://$DELTA_HOST:5000"
    echo "    5. Enable Boltz extension for submarine swaps"
    echo ""
}

# ─── Main ───
case "${1:-status}" in
    status)    cmd_status ;;
    btc-zmq)   cmd_btc_zmq ;;
    cln)       cmd_cln ;;
    lnbits)    cmd_lnbits ;;
    full)      cmd_full ;;
    *)
        echo "Usage: $0 {status|btc-zmq|cln|lnbits|full}"
        echo ""
        echo "  status   — Check Bitcoin Knots sync & readiness"
        echo "  btc-zmq  — Recreate BTC container with ZMQ ports"
        echo "  cln      — Install Core Lightning"
        echo "  lnbits   — Install LNbits"
        echo "  full     — Full setup (all steps)"
        exit 1
        ;;
esac
