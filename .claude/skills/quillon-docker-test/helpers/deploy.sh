#!/usr/bin/env bash
# Phase 2: Deploy fresh-DB sync containers on Epsilon (port 8085) + Delta (port 8086).
#
# Prereq: helpers/build.sh has completed successfully. The built binary lives
# at /home/orobit/target-debian12/release/q-api-server on Epsilon.
#
# Container env (matches CLAUDE.md "DOCKER SYNC TESTING ON EPSILON" section):
#   Q_NETWORK_ID=mainnet-genesis
#   Q_DB_PATH=/data/db
#   Q_P2P_PORT=9001   (CRITICAL: env var, not CLI flag — CLI flag does not exist)
#   RUST_LOG=info
#   ROCKSDB_BLOCK_CACHE_MB=2048
#   Q_TOR_BOOTSTRAP_TIMEOUT=5   (else startup blocks 120s)
#
# Usage: ./deploy.sh <version>
set -euo pipefail

VERSION="${1:?usage: deploy.sh <version>}"
VERSION="${VERSION#v}"  # strip leading 'v' (else container becomes q-sync-test-vv...)
EPSILON="root@89.149.241.126"
DELTA="root@5.79.79.158"
BINARY="/home/orobit/target-debian12/release/q-api-server"

# Port-collision pre-check: a stale container from a previous run can hold 8085/8086.
# docker run will fail with "port already allocated"; surface it early with context.
EPSILON_PORT_HOLDER=$(ssh -o ConnectTimeout=5 "$EPSILON" \
  "docker ps --filter 'publish=8085' --format '{{.Names}}'" 2>/dev/null | head -1)
if [ -n "$EPSILON_PORT_HOLDER" ] && [ "$EPSILON_PORT_HOLDER" != "q-sync-test-v${VERSION}-epsilon" ]; then
  echo "[deploy.sh] ABORT: port 8085 on Epsilon held by $EPSILON_PORT_HOLDER" >&2
  echo "[deploy.sh] Remove it first: ssh $EPSILON 'docker rm -f $EPSILON_PORT_HOLDER'" >&2
  exit 3
fi
DELTA_PORT_HOLDER=$(ssh -o ConnectTimeout=5 "$DELTA" \
  "docker ps --filter 'publish=8086' --format '{{.Names}}'" 2>/dev/null | head -1)
if [ -n "$DELTA_PORT_HOLDER" ] && [ "$DELTA_PORT_HOLDER" != "q-sync-test-v${VERSION}-delta" ]; then
  echo "[deploy.sh] ABORT: port 8086 on Delta held by $DELTA_PORT_HOLDER" >&2
  echo "[deploy.sh] Remove it first: ssh $DELTA 'docker rm -f $DELTA_PORT_HOLDER'" >&2
  exit 3
fi

# --- Epsilon ---
echo "[deploy.sh] deploying to Epsilon container q-sync-test-v${VERSION}-epsilon"
ssh "$EPSILON" "
  set -e
  test -f $BINARY || { echo 'binary missing: $BINARY (build first)'; exit 1; }
  mkdir -p /home/orobit/docker-sync-test-v${VERSION}-epsilon
  docker rm -f q-sync-test-v${VERSION}-epsilon 2>/dev/null || true

  docker run -d \
    --name q-sync-test-v${VERSION}-epsilon \
    --memory=8g \
    -p 8085:8080 -p 9005:9001 \
    -v $BINARY:/opt/q-api-server:ro \
    -v /home/orobit/docker-sync-test-v${VERSION}-epsilon:/data \
    -e Q_NETWORK_ID=mainnet-genesis \
    -e Q_DB_PATH=/data/db \
    -e Q_P2P_PORT=9001 \
    -e RUST_LOG=info \
    -e ROCKSDB_BLOCK_CACHE_MB=2048 \
    -e Q_TOR_BOOTSTRAP_TIMEOUT=5 \
    debian:12 \
    bash -c '
      apt-get update -qq && apt-get install -y -qq libssl3 ca-certificates curl >/dev/null 2>&1 && \
      cp /opt/q-api-server /usr/local/bin/q-api-server && \
      chmod +x /usr/local/bin/q-api-server && \
      echo \"\$(date +%s)\" > /data/sync_start_epoch.txt && \
      exec /usr/local/bin/q-api-server --port 8080 --admin-wallet qnk7154929a6aa0c118791373ea21004aca6e494e6e031c36f780cd5acedf031ccb 2>&1
    '
  echo '[deploy.sh][epsilon] container started'
"

# --- Delta ---
# Epsilon cannot SSH Delta directly (per CLAUDE.md). SCP via Beta is the route:
# We're already running on Beta (where this script lives), so Beta SCPs from
# Epsilon to itself, then Beta SCPs to Delta.
echo "[deploy.sh] copying binary Epsilon → Beta → Delta"
TMPDIR=$(mktemp -d)
scp "$EPSILON:$BINARY" "$TMPDIR/q-api-server" >/dev/null
scp "$TMPDIR/q-api-server" "$DELTA:/tmp/q-api-server-v${VERSION}" >/dev/null
rm -rf "$TMPDIR"

echo "[deploy.sh] deploying to Delta container q-sync-test-v${VERSION}-delta"
ssh "$DELTA" "
  set -e
  mkdir -p /home/orobit/docker-sync-test-v${VERSION}-delta
  docker rm -f q-sync-test-v${VERSION}-delta 2>/dev/null || true

  docker run -d \
    --name q-sync-test-v${VERSION}-delta \
    --memory=8g \
    -p 8086:8080 -p 9006:9001 \
    -v /tmp/q-api-server-v${VERSION}:/opt/q-api-server:ro \
    -v /home/orobit/docker-sync-test-v${VERSION}-delta:/data \
    -e Q_NETWORK_ID=mainnet-genesis \
    -e Q_DB_PATH=/data/db \
    -e Q_P2P_PORT=9001 \
    -e RUST_LOG=info \
    -e ROCKSDB_BLOCK_CACHE_MB=2048 \
    -e Q_TOR_BOOTSTRAP_TIMEOUT=5 \
    debian:12 \
    bash -c '
      apt-get update -qq && apt-get install -y -qq libssl3 ca-certificates curl >/dev/null 2>&1 && \
      cp /opt/q-api-server /usr/local/bin/q-api-server && \
      chmod +x /usr/local/bin/q-api-server && \
      echo \"\$(date +%s)\" > /data/sync_start_epoch.txt && \
      exec /usr/local/bin/q-api-server --port 8080 --admin-wallet qnk7154929a6aa0c118791373ea21004aca6e494e6e031c36f780cd5acedf031ccb 2>&1
    '
  echo '[deploy.sh][delta] container started'
"

echo "[deploy.sh] both containers launched. Status API:"
echo "  Epsilon: curl -s http://89.149.241.126:8085/api/v1/status | jq"
echo "  Delta  : curl -s http://5.79.79.158:8086/api/v1/status | jq"
