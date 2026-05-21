#!/usr/bin/env bash
# Phase 1: Build q-api-server in Epsilon Docker (rust:bookworm, target-debian12 cache).
#
# Streams compile output directly to the log file — NEVER pipes through `tail`,
# which buffers until EOF and hides 30+ minutes of progress.
#
# Returns 0 on "Finished `release` profile" line, non-zero otherwise.
#
# Usage: ./build.sh <version>
#   <version>  Any version tag for naming the log file. Pass "latest" if you
#              just want the cwd source built without a stable identifier.
set -euo pipefail

VERSION="${1:-latest}"
EPSILON="root@89.149.241.126"
SRC="/home/orobit/q-narwhalknight-src"
LOG="/home/orobit/tmp/build-v${VERSION}.log"
CACHE="/home/orobit/target-debian12"

echo "[build.sh] target version: v${VERSION}"
echo "[build.sh] log: ${LOG}"

# Verify Epsilon is reachable + source tree is at the right branch
ssh -o ConnectTimeout=5 "$EPSILON" "
  set -e
  cd $SRC
  git fetch origin >/dev/null 2>&1
  echo '[build.sh][epsilon] HEAD: '\$(git rev-parse --short HEAD)
  echo '[build.sh][epsilon] branch: '\$(git rev-parse --abbrev-ref HEAD)
  mkdir -p /home/orobit/tmp
" || { echo "[build.sh] ERROR: cannot reach Epsilon or source tree"; exit 2; }

# Kick off compile in the background (rust:bookworm + cached target dir).
# nohup + detached so the SSH command returns immediately.
ssh "$EPSILON" "cd $SRC && nohup docker run --rm \
  --name qnk-build-v${VERSION} \
  -v \$(pwd):/src \
  -v ${CACHE}:/src/target \
  -w /src --cpus=16 rust:bookworm \
  bash -c '
    set -e
    apt-get update -qq && \
    apt-get install -y -qq libssl-dev pkg-config cmake clang libudev-dev libclang-dev >/dev/null 2>&1 && \
    echo \"[build] starting at \$(date)\" && \
    cargo build --release --package q-api-server 2>&1
  ' > $LOG 2>&1 &
  echo \"docker PID: \$!\""

echo "[build.sh] build container launched. Watch progress with:"
echo "  ssh $EPSILON 'tail -f $LOG'"
echo "  ssh $EPSILON 'docker stats qnk-build-v${VERSION} --no-stream'"
echo "[build.sh] expected duration: 5-25 min (incremental cache) | 30-60 min (cold)"
