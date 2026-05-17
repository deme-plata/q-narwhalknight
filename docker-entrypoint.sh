#!/bin/bash
set -euo pipefail

BINARY=/data/q-api-server
DATA_DIR=/data/chain
DOWNLOAD_URL="https://quillon.xyz/downloads/q-api-server-linux-x86_64"

# Download the latest approved binary on first run (or if missing)
if [ ! -f "$BINARY" ]; then
    echo "[qnk] First run — downloading latest approved node binary from quillon.xyz..."
    mkdir -p /data
    curl -fL --retry 5 --retry-delay 3 "$DOWNLOAD_URL" -o "$BINARY"
    chmod +x "$BINARY"
    echo "[qnk] Download complete: $(ls -lh $BINARY | awk '{print $5}')"
fi

mkdir -p "$DATA_DIR"

# Default env — can be overridden by docker run -e
export Q_NETWORK_ID="${Q_NETWORK_ID:-mainnet-genesis}"
export Q_DB_PATH="${Q_DB_PATH:-/data/chain}"
export Q_P2P_PORT="${Q_P2P_PORT:-9001}"
export Q_WS_PORT="${Q_WS_PORT:-9002}"
export RUST_LOG="${RUST_LOG:-info}"
export ROCKSDB_BLOCK_CACHE_MB="${ROCKSDB_BLOCK_CACHE_MB:-512}"
export Q_TURBO_SYNC="${Q_TURBO_SYNC:-1}"
export Q_BATCHED_WRITES="${Q_BATCHED_WRITES:-1}"
export Q_STATE_SYNC="${Q_STATE_SYNC:-1}"
export Q_TURBO_CHUNK_SIZE="${Q_TURBO_CHUNK_SIZE:-500}"
export Q_TOR_BOOTSTRAP_TIMEOUT="${Q_TOR_BOOTSTRAP_TIMEOUT:-5}"

# Auto-update: node replaces its own binary when bootstrap nodes co-sign an update
export Q_AUTO_UPDATE="${Q_AUTO_UPDATE:-1}"
export Q_AUTO_UPDATE_CHECK_INTERVAL="${Q_AUTO_UPDATE_CHECK_INTERVAL:-300}"
export Q_AUTO_UPDATE_RESTART_DELAY="${Q_AUTO_UPDATE_RESTART_DELAY:-30}"
export Q_AUTO_UPDATE_ROLLBACK_TIMEOUT="${Q_AUTO_UPDATE_ROLLBACK_TIMEOUT:-60}"
export Q_AUTO_UPDATE_MIN_PEERS="${Q_AUTO_UPDATE_MIN_PEERS:-2}"

echo "[qnk] Starting Q-NarwhalKnight node (network: $Q_NETWORK_ID)"
exec "$BINARY" --port "${PORT:-8080}"
