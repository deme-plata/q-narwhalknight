#!/usr/bin/env bash
# gamma-tui.sh — Stop Gamma's systemd service, SCP latest binary, run in foreground with TUI.
# Usage: ./scripts/gamma-tui.sh
#
# This script runs FROM Beta. It:
#   1. Builds the latest release binary (if needed)
#   2. SCPs it to Gamma
#   3. Stops the systemd service on Gamma
#   4. SSHs into Gamma and runs the node in foreground so you see the TUI

set -euo pipefail

GAMMA="root@109.205.176.60"
GAMMA_DIR="/opt/orobit/shared/q-narwhalknight"
BINARY_NAME="q-api-server-v925"
LOCAL_BINARY="target/release/q-api-server"

echo "=== Gamma TUI Launcher ==="
echo ""

# --- Step 1: Check if we have a release binary ---
if [[ ! -f "$LOCAL_BINARY" ]]; then
    echo "[1/4] No release binary found. Building..."
    cargo build --release --package q-api-server
else
    echo "[1/4] Release binary found: $(ls -lh $LOCAL_BINARY | awk '{print $5, $6, $7, $8}')"
    read -p "       Rebuild? (y/N) " rebuild
    if [[ "${rebuild:-n}" =~ ^[Yy] ]]; then
        cargo build --release --package q-api-server
    fi
fi

# --- Step 2: Stop Gamma's systemd service ---
echo "[2/4] Stopping q-api-server on Gamma..."
ssh "$GAMMA" "systemctl stop q-api-server 2>/dev/null || true"
sleep 1
# Make sure it's really dead
ssh "$GAMMA" "pgrep -f q-api-server | xargs -I{} kill -9 {} 2>/dev/null || true"
echo "       Stopped."

# --- Step 3: SCP binary to Gamma ---
echo "[3/4] Copying binary to Gamma..."
scp -q "$LOCAL_BINARY" "${GAMMA}:${GAMMA_DIR}/${BINARY_NAME}"
ssh "$GAMMA" "chmod +x ${GAMMA_DIR}/${BINARY_NAME}"
echo "       Done."

# --- Step 4: SSH into Gamma and run in foreground ---
echo "[4/4] Launching node on Gamma in foreground (TUI mode)..."
echo "       Press Ctrl+C to stop the node."
echo "       To go back to systemd later: ssh $GAMMA 'systemctl start q-api-server'"
echo ""
echo "========================================"

ssh -t "$GAMMA" "cd ${GAMMA_DIR} && \
    DUNE_API_KEY=QuQpt5Hk9XdPTJcMszm6IYQ8Nn2ChXWJ \
    DUNE_NAMESPACE=demetri \
    Q_DB_PATH=./data-mainnet-genesis \
    Q_NETWORK_ID=mainnet-genesis \
    Q_IS_VALIDATOR=true \
    Q_P2P_PORT=9001 \
    RUST_LOG=info \
    Q_ALLOW_SOLO_MINING=true \
    Q_TURBO_SYNC=1 \
    Q_BATCHED_WRITES=1 \
    Q_STATE_SYNC=1 \
    Q_PREFLIGHT_CHECK=1 \
    Q_ENABLE_AI=0 \
    Q_EXTERNAL_ADDRESS=/ip4/109.205.176.60/tcp/9001 \
    Q_GOSSIPSUB_HEARTBEAT_MS=300 \
    Q_TOR_BOOTSTRAP_TIMEOUT=5 \
    ROCKSDB_BLOCK_CACHE_MB=512 \
    Q_SYNC_MAX_CONCURRENCY=8 \
    Q_ROCKSDB_WRITE_RATE_MB=200 \
    Q_P2P_ONLY=1 \
    Q_TURBO_CHUNK_SIZE=500 \
    Q_ENABLE_MINING_POOL=1 \
    Q_ENABLE_DISTRIBUTED_POOL=1 \
    ./${BINARY_NAME} --port 8080"

# If we get here, the user hit Ctrl+C
echo ""
echo "Node stopped. To restart via systemd:"
echo "  ssh $GAMMA 'systemctl start q-api-server'"
