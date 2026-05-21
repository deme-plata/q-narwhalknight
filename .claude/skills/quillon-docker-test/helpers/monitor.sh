#!/usr/bin/env bash
# Phase 3: Poll both containers for sync progress, report wedge / divergence / tip-reach.
#
# This is a polling script intended to be wrapped by the Monitor tool, NOT to
# be run in a terminal and watched. Each interesting state change emits exactly
# one line to stdout (so the Monitor tool relays it as an event).
#
# Lines emitted:
#   PROGRESS epsilon h=<height> peers=<n> bps=<rate> uptime=<s>
#   PROGRESS delta   h=<height> peers=<n> bps=<rate> uptime=<s>
#   MILESTONE epsilon reached h=<height>           (every 10K blocks)
#   MILESTONE delta   reached h=<height>
#   WEDGE epsilon stuck at h=<height> for <s>s
#   DIVERGED epsilon vs delta: <h_e> vs <h_d> (delta_e=<d>)
#   TIP_REACHED epsilon h=<height> production=<h_p>
#   CRASHED <which>
#   DONE all containers reached tip or stopped
#
# Usage: ./monitor.sh <version> <duration_sec>
set -euo pipefail

VERSION="${1:?usage: monitor.sh <version> <duration_sec>}"
DURATION="${2:-3600}"
EPSILON_API="http://89.149.241.126:8085/api/v1/status"
DELTA_API="http://5.79.79.158:8086/api/v1/status"
PROD_API="http://89.149.241.126/api/v1/status"

START=$(date +%s)
LAST_E=0
LAST_E_AT=$START
LAST_D=0
LAST_D_AT=$START
LAST_MILESTONE_E=0
LAST_MILESTONE_D=0

while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  [ $ELAPSED -ge $DURATION ] && { echo "DONE timeout after ${DURATION}s"; break; }

  # Container existence — exits monitor early if a container vanished.
  E_RUNNING=$(ssh -o ConnectTimeout=5 root@89.149.241.126 \
    "docker ps --filter name=q-sync-test-v${VERSION}-epsilon --format '{{.Status}}'" 2>/dev/null | head -1 || echo "")
  D_RUNNING=$(ssh -o ConnectTimeout=5 root@5.79.79.158 \
    "docker ps --filter name=q-sync-test-v${VERSION}-delta --format '{{.Status}}'" 2>/dev/null | head -1 || echo "")

  [ -z "$E_RUNNING" ] && { echo "CRASHED epsilon"; break; }
  [ -z "$D_RUNNING" ] && { echo "CRASHED delta"; break; }

  # API status (tolerate transient failures — curl can flap during sync churn)
  E_JSON=$(curl -s -m 3 "$EPSILON_API" 2>/dev/null || echo "{}")
  D_JSON=$(curl -s -m 3 "$DELTA_API" 2>/dev/null || echo "{}")
  P_JSON=$(curl -s -m 3 "$PROD_API" 2>/dev/null || echo "{}")

  H_E=$(echo "$E_JSON" | grep -oE '"current_height":[0-9]+' | head -1 | grep -oE '[0-9]+' || echo "0")
  H_D=$(echo "$D_JSON" | grep -oE '"current_height":[0-9]+' | head -1 | grep -oE '[0-9]+' || echo "0")
  H_P=$(echo "$P_JSON" | grep -oE '"current_height":[0-9]+' | head -1 | grep -oE '[0-9]+' || echo "0")
  P_E=$(echo "$E_JSON" | grep -oE '"peer_count":[0-9]+' | head -1 | grep -oE '[0-9]+' || echo "0")
  P_D=$(echo "$D_JSON" | grep -oE '"peer_count":[0-9]+' | head -1 | grep -oE '[0-9]+' || echo "0")

  # Per-instance progress (every poll, so caller sees liveness)
  BPS_E=0
  [ $ELAPSED -gt 0 ] && BPS_E=$((H_E / ELAPSED))
  BPS_D=0
  [ $ELAPSED -gt 0 ] && BPS_D=$((H_D / ELAPSED))
  echo "PROGRESS epsilon h=$H_E peers=$P_E bps=$BPS_E uptime=${ELAPSED}s"
  echo "PROGRESS delta   h=$H_D peers=$P_D bps=$BPS_D uptime=${ELAPSED}s"

  # Milestone every 10K blocks (per container)
  MILE_E=$((H_E / 10000))
  if [ "$MILE_E" -gt "$LAST_MILESTONE_E" ]; then
    echo "MILESTONE epsilon reached h=$H_E"
    LAST_MILESTONE_E=$MILE_E
  fi
  MILE_D=$((H_D / 10000))
  if [ "$MILE_D" -gt "$LAST_MILESTONE_D" ]; then
    echo "MILESTONE delta reached h=$H_D"
    LAST_MILESTONE_D=$MILE_D
  fi

  # Wedge detection — 5 min at same height
  if [ "$H_E" -eq "$LAST_E" ]; then
    STALLED=$((NOW - LAST_E_AT))
    if [ $STALLED -ge 300 ]; then
      echo "WEDGE epsilon stuck at h=$H_E for ${STALLED}s"
    fi
  else
    LAST_E=$H_E
    LAST_E_AT=$NOW
  fi
  if [ "$H_D" -eq "$LAST_D" ]; then
    STALLED=$((NOW - LAST_D_AT))
    if [ $STALLED -ge 300 ]; then
      echo "WEDGE delta stuck at h=$H_D for ${STALLED}s"
    fi
  else
    LAST_D=$H_D
    LAST_D_AT=$NOW
  fi

  # Divergence between Epsilon and Delta — if both >0 and differ by >50 blocks
  if [ "$H_E" -gt 1000 ] && [ "$H_D" -gt 1000 ]; then
    DIFF=$((H_E > H_D ? H_E - H_D : H_D - H_E))
    if [ $DIFF -gt 50 ]; then
      echo "DIVERGED epsilon vs delta: $H_E vs $H_D (diff=$DIFF)"
    fi
  fi

  # Tip-reach
  if [ "$H_P" -gt 0 ]; then
    [ "$H_E" -ge $((H_P - 50)) ] && [ "$H_E" -gt 0 ] && \
      echo "TIP_REACHED epsilon h=$H_E production=$H_P" && break
    [ "$H_D" -ge $((H_P - 50)) ] && [ "$H_D" -gt 0 ] && \
      echo "TIP_REACHED delta h=$H_D production=$H_P" && break
  fi

  sleep 60
done
