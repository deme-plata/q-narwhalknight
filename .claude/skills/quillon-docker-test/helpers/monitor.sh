#!/usr/bin/env bash
# Phase 3 (v2 — sparse-chain-aware): Poll containers for sync progress.
#
# CHANGED 2026-05-21 vs v1:
#   v1 read current_height from GET /api/v1/status (data.upgrades.current_height).
#   That field is UNRELIABLE for fresh-syncing nodes — it tracks the upgrade-gate's
#   contiguously-applied tip, which stays at 0 until the chain is contiguous from
#   genesis. On Q-NarwhalKnight's sparse chain (~3% of pre-7M heights present, ~50%
#   of 7M-14M, dense post-14M), a fresh node may apply millions of discrete blocks
#   without ever advancing data.upgrades.current_height. Result: v1 falsely
#   reported WEDGE on a node that was actually syncing.
#
#   v2 reads height from `docker logs` in priority order:
#     1. Progress-bar line:  ┃ ⏳ Waiting for blocks... | <H>/<TIP> | <BPS> b/s | ETA
#        — emitted every ~6s once sync starts, gives current + tip + BPS in one line
#     2. HTTP BOOTSTRAP:     [HTTP BOOTSTRAP] ... our_height=<H>, ...
#        — fallback when no progress bar yet (early startup)
#     3. synced_through:     synced_through=<H>  (v10.9.55+)
#        — sparse-chain-aware pointer; advances by *requested window* not contiguity
#
# See docs/technical-review-sparse-chain-truth-v1.md for why.
#
# Lines emitted (each = one Monitor-tool event):
#   PROD_TIP h=<n>                                            (once at start)
#   PROGRESS {host} h=<H> synced_through=<S> bps=<BPS> uptime=<S>s    (each poll)
#   MILESTONE {host} reached h=<H>                            (each 100K, on dense chain)
#   WEDGE {host} stuck at h=<H> for <S>s                      (≥300s same h)
#   DIVERGED epsilon vs delta: <h_e> vs <h_d> (diff=<D>)      (>100 block delta sustained)
#   TIP_REACHED {host} h=<H> production=<H_P>                 (H >= H_P - 50)
#   CRASHED {host}
#   PRECHECK_FAIL <which> container not running
#   DONE timeout after <D>s | DONE all containers reached tip
#
# Usage: ./monitor.sh <version> [duration_sec]
#   <version>: tag of the form "10.11.3-latest" — leading 'v' is stripped automatically
#
# v4 (2026-05-21): dropped `set -e` from the loop. set -euo pipefail was too strict
# for a long-running observation script — a single transient SSH error,
# `[ test ]` returning non-zero, or `grep` finding nothing made the whole monitor
# exit silently. Symptom: after first PROD_TIP line, the loop never produced
# PROGRESS lines. The right contract for a monitor is keep-running-and-report,
# not abort-on-first-noise. set -u + pipefail stay (catch real bugs in our own
# variable references and pipe chains).
set -uo pipefail

VERSION="${1:?usage: monitor.sh <version> [duration_sec]}"
VERSION="${VERSION#v}"  # strip leading 'v' to avoid q-sync-test-vv... collisions
DURATION="${2:-3600}"

EPSILON_HOST="root@89.149.241.126"
DELTA_HOST="root@5.79.79.158"
EPSILON_CONTAINER="q-sync-test-v${VERSION}-epsilon"
DELTA_CONTAINER="q-sync-test-v${VERSION}-delta"
# Direct-IP to Epsilon :8080 — NOT https://quillon.xyz, because q-flux on Epsilon
# was observed routing /api/v1/status to a *different* backend (height 1.21M with
# IPv6 multiaddr) on 2026-05-21. The direct path always hits the authoritative
# Epsilon node serving the real tip.
PROD_API="http://89.149.241.126:8080/api/v1/status"

# get_height_from_logs <host> <container>
# Prints: "<height> <bps> <synced_through>"  (all integers, 0 = unknown)
# Uses pure grep pipelines — no gawk extensions (Beta has mawk; gawk's 3-arg
# match($0, re, m) doesn't work there).
get_height_from_logs() {
  local host="$1" container="$2"
  local logs height bps synced
  # Pull last 2000 lines — covers progress-bar at 6s cadence + bootstrap noise.
  # v4: bumped from 400 → 2000 because during rapid block ingestion (e.g.
  # post-bootstrap-override re-sync from 1.5M → 18M) the [BALANCE TX] firehose
  # crowds out the progress-bar lines from the last few hundred entries.
  logs=$(ssh -o ConnectTimeout=5 "$host" "docker logs --tail 2000 $container 2>&1" 2>/dev/null || echo "")
  # Primary: progress bar — "| <H>/<TIP> | <BPS> b/s"
  local pb
  pb=$(printf '%s\n' "$logs" | grep -oE 'Waiting for blocks.*\| [0-9]+/[0-9]+ \| [0-9]+ b/s' | tail -1)
  if [ -n "$pb" ]; then
    height=$(printf '%s\n' "$pb" | grep -oE '\| [0-9]+/' | head -1 | grep -oE '[0-9]+')
    bps=$(printf '%s\n' "$pb" | grep -oE '\| [0-9]+ b/s' | head -1 | grep -oE '[0-9]+')
  fi
  # Secondary: HTTP BOOTSTRAP our_height=N (use highest)
  if [ -z "${height:-}" ] || [ "${height:-0}" -eq 0 ]; then
    height=$(printf '%s\n' "$logs" | grep -oE 'our_height=[0-9]+' | grep -oE '[0-9]+' | sort -un | tail -1)
  fi
  # Tertiary: v10.9.55+ synced_through pointer (use highest)
  synced=$(printf '%s\n' "$logs" | grep -oE 'synced_through[=:][[:space:]]*[0-9]+' | grep -oE '[0-9]+' | sort -un | tail -1)
  printf '%s %s %s\n' "${height:-0}" "${bps:-0}" "${synced:-0}"
}

# get_prod_tip — production current_height from direct Epsilon endpoint
# Note: quillon.xyz behind q-flux LB was observed returning a different backend's
# height (1.21M) instead of Epsilon's true tip (18.17M). Use direct IP always.
get_prod_tip() {
  curl -s -m 5 "$PROD_API" 2>/dev/null | \
    grep -oE '"current_height":[0-9]+' | head -1 | grep -oE '[0-9]+' || echo 0
}

# Pre-flight: confirm both containers exist + are Up
ssh -o ConnectTimeout=5 "$EPSILON_HOST" \
  "docker ps --filter name=$EPSILON_CONTAINER --format '{{.Status}}'" 2>/dev/null | \
  head -1 | grep -q "^Up" || \
  { echo "PRECHECK_FAIL epsilon container $EPSILON_CONTAINER not running"; exit 1; }
ssh -o ConnectTimeout=5 "$DELTA_HOST" \
  "docker ps --filter name=$DELTA_CONTAINER --format '{{.Status}}'" 2>/dev/null | \
  head -1 | grep -q "^Up" || \
  { echo "PRECHECK_FAIL delta container $DELTA_CONTAINER not running"; exit 1; }

START=$(date +%s)
LAST_E=0; LAST_E_AT=$START
LAST_D=0; LAST_D_AT=$START
LAST_MILESTONE_E=0
LAST_MILESTONE_D=0

H_TIP=$(get_prod_tip)
if [ "$H_TIP" -gt 0 ]; then
  echo "PROD_TIP h=$H_TIP"
else
  echo "PROD_TIP unknown — production status endpoint unreachable; TIP_REACHED detection disabled"
fi

while true; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  [ $ELAPSED -ge $DURATION ] && { echo "DONE timeout after ${DURATION}s"; break; }

  # Container existence — exits early if a container died
  E_UP=$(ssh -o ConnectTimeout=5 "$EPSILON_HOST" \
    "docker ps --filter name=$EPSILON_CONTAINER --format '{{.Status}}'" 2>/dev/null | head -1)
  D_UP=$(ssh -o ConnectTimeout=5 "$DELTA_HOST" \
    "docker ps --filter name=$DELTA_CONTAINER --format '{{.Status}}'" 2>/dev/null | head -1)
  [[ "$E_UP" != Up* ]] && { echo "CRASHED epsilon (status=$E_UP)"; break; }
  [[ "$D_UP" != Up* ]] && { echo "CRASHED delta (status=$D_UP)"; break; }

  # Heights (log-based, sparse-chain-aware)
  read H_E BPS_E SYNCED_E < <(get_height_from_logs "$EPSILON_HOST" "$EPSILON_CONTAINER")
  read H_D BPS_D SYNCED_D < <(get_height_from_logs "$DELTA_HOST" "$DELTA_CONTAINER")

  echo "PROGRESS epsilon h=$H_E synced_through=$SYNCED_E bps=$BPS_E uptime=${ELAPSED}s"
  echo "PROGRESS delta   h=$H_D synced_through=$SYNCED_D bps=$BPS_D uptime=${ELAPSED}s"

  # Milestone every 100K blocks per container (1M would be too sparse for a 1h test)
  MILE_E=$((H_E / 100000))
  if [ "$MILE_E" -gt "$LAST_MILESTONE_E" ]; then
    echo "MILESTONE epsilon reached h=$H_E"
    LAST_MILESTONE_E=$MILE_E
  fi
  MILE_D=$((H_D / 100000))
  if [ "$MILE_D" -gt "$LAST_MILESTONE_D" ]; then
    echo "MILESTONE delta reached h=$H_D"
    LAST_MILESTONE_D=$MILE_D
  fi

  # Wedge detection — 5 min at same height
  # Notes: 30 b/s is the floor for the sparse pre-7M range; a TRULY-stuck node
  # holds the same integer height >5 min. We still emit but treat as warning.
  # v3 fix: `[ test ] && cmd` chains were tripping set -e when the test failed
  # (the whole AND-list returns non-zero as a top-level statement → set -e
  # exits the loop silently). Replaced with explicit if/then to keep set -e
  # honest. Symptom this fixed: monitor.sh exiting after the first PROD_TIP
  # line on a healthy run — the divergence check `[ $DIFF -gt 100 ] && ...`
  # short-circuited on convergent containers and triggered set -e, killing
  # the loop on iteration #1.
  if [ "$H_E" -eq "$LAST_E" ] && [ "$H_E" -gt 0 ]; then
    STALLED=$((NOW - LAST_E_AT))
    if [ $STALLED -ge 300 ]; then echo "WEDGE epsilon stuck at h=$H_E for ${STALLED}s"; fi
  else
    LAST_E=$H_E; LAST_E_AT=$NOW
  fi
  if [ "$H_D" -eq "$LAST_D" ] && [ "$H_D" -gt 0 ]; then
    STALLED=$((NOW - LAST_D_AT))
    if [ $STALLED -ge 300 ]; then echo "WEDGE delta stuck at h=$H_D for ${STALLED}s"; fi
  else
    LAST_D=$H_D; LAST_D_AT=$NOW
  fi

  # Divergence — if both >100K and differ by >100 blocks, surface
  if [ "$H_E" -gt 100000 ] && [ "$H_D" -gt 100000 ]; then
    DIFF=$((H_E > H_D ? H_E - H_D : H_D - H_E))
    if [ $DIFF -gt 100 ]; then echo "DIVERGED epsilon vs delta: $H_E vs $H_D (diff=$DIFF)"; fi
  fi

  # Tip-reach — works if H_TIP was retrieved AND container height is non-zero
  if [ "$H_TIP" -gt 0 ]; then
    E_AT_TIP=0; D_AT_TIP=0
    if [ "$H_E" -gt 0 ] && [ "$H_E" -ge $((H_TIP - 50)) ]; then
      echo "TIP_REACHED epsilon h=$H_E production=$H_TIP"
      E_AT_TIP=1
    fi
    if [ "$H_D" -gt 0 ] && [ "$H_D" -ge $((H_TIP - 50)) ]; then
      echo "TIP_REACHED delta h=$H_D production=$H_TIP"
      D_AT_TIP=1
    fi
    if [ "$E_AT_TIP" -eq 1 ] && [ "$D_AT_TIP" -eq 1 ]; then
      echo "DONE all containers reached tip"
      break
    fi
  fi

  sleep 60
done
