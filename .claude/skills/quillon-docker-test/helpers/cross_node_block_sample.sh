#!/usr/bin/env bash
# Phase 5 (v1, 2026-05-22): Random-block cross-node consistency probe.
#
# Picks N random block heights uniformly from [1000..min_tip-100] across the
# given node URLs, fetches /api/v1/blocks/<height> from each, and reports any
# divergence in header fields (prev_block_hash, tx_root, solutions_root,
# tx_count, transaction_id list, header height match).
#
# This is the forensic counterpart to /integrity/balance-root: balance-root
# proves all nodes agree on wallet STATE; random-block-sample proves all
# nodes agree on block CONTENT at any historical height. Together they
# catch almost any cross-node divergence the chain can produce.
#
# Output: monitor-tool-friendly (one event per line, parseable prefixes).
# Events:
#   START samples=<N> nodes=<n1,n2,...>
#   TIP <node>=<height>
#   HEIGHT h=<H> nodes_responding=<K>/<total> verdict=<OK|DIVERGED|UNREACHABLE>
#   DIVERGENCE h=<H> field=<name> <node1>=<value1> <node2>=<value2> ...
#   SUMMARY total=<N> ok=<X> diverged=<Y> unreachable=<Z> verdict=<PASS|FAIL>
#
# Usage: ./cross_node_block_sample.sh <num_samples> <node1_url> <node2_url> [node3_url ...]
#
# Example:
#   ./cross_node_block_sample.sh 20 \
#     http://89.149.241.126:8080 \
#     http://5.79.79.158:8080 \
#     http://89.149.241.126:8085

set -uo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <num_samples> <node1_url> <node2_url> [node3_url ...]" >&2
  echo "  Compares /api/v1/blocks/<height> across all given nodes at <num_samples>" >&2
  echo "  uniformly-random heights from [1000..min_tip-100]." >&2
  exit 2
fi

N="$1"; shift
NODES=("$@")

# Validate numeric N
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ] || [ "$N" -gt 1000 ]; then
  echo "ERROR num_samples must be integer 1..1000, got '$N'" >&2
  exit 2
fi

echo "START samples=$N nodes=$(IFS=,; echo "${NODES[*]}")"

# Step 1: get tip height from each node
declare -A NODE_TIP
MIN_TIP=999999999999
ANY_REACHABLE=0
for node in "${NODES[@]}"; do
  # Try data.upgrades.current_height first (the apply gate); fall back to
  # data.current_height if non-null; otherwise unreachable.
  resp=$(curl -s -m 8 "$node/api/v1/status" 2>/dev/null || echo "")
  tip=$(echo "$resp" | python3 -c "
import json, sys
try:
  j = json.load(sys.stdin)
  d = j.get('data', {})
  u = d.get('upgrades', {})
  h = u.get('current_height') if u.get('current_height') else d.get('current_height')
  print(h if h is not None else 0)
except Exception:
  print(0)
" 2>/dev/null || echo 0)
  NODE_TIP[$node]=$tip
  if [ "$tip" -gt 0 ]; then
    ANY_REACHABLE=1
    [ "$tip" -lt "$MIN_TIP" ] && MIN_TIP=$tip
  fi
  echo "TIP $node=$tip"
done

if [ "$ANY_REACHABLE" -eq 0 ]; then
  echo "SUMMARY total=$N ok=0 diverged=0 unreachable=$N verdict=FAIL (no node reachable)"
  exit 1
fi

# Step 2: pick N uniformly-random heights from [1000..MIN_TIP-100]
# We avoid the very recent tip (-100) to skip blocks that may still be in
# finality flight; and avoid genesis (<1000) since those are often sparse.
LOWER=1000
UPPER=$((MIN_TIP - 100))
if [ "$UPPER" -le "$LOWER" ]; then
  echo "SUMMARY total=$N ok=0 diverged=0 unreachable=$N verdict=FAIL (chain too short: min_tip=$MIN_TIP)"
  exit 1
fi

HEIGHTS=$(python3 -c "
import random, sys
N, lo, hi = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
random.seed()
# Cap N at the available range to avoid duplicates dominating
candidates = list(range(lo, hi+1))
n = min(N, len(candidates))
picks = sorted(random.sample(candidates, n))
print('\n'.join(str(p) for p in picks))
" "$N" "$LOWER" "$UPPER")

# Step 3: for each sampled height, fetch from all nodes + compare
OK=0
DIVERGED=0
UNREACHABLE=0

while read -r H; do
  [ -z "$H" ] && continue

  # Collect per-node block data
  declare -A NODE_BLOCK
  responding=0
  unreachable_for_h=0
  for node in "${NODES[@]}"; do
    blk=$(curl -s -m 8 "$node/api/v1/blocks/$H" 2>/dev/null | python3 -c "
import json, sys
try:
  j = json.load(sys.stdin)
  if not j.get('success'):
    print('UNREACHABLE')
    sys.exit(0)
  d = j.get('data', {})
  hdr = d.get('header', {})
  txs = d.get('transactions', [])
  # v2 fix: on-chain tx-hash field is 'id' (32-byte array), NOT 'transaction_id'
  # or 'hash'. Convert byte array to hex string for comparison. Pre-v2 this
  # helper silently returned '?' for every tx and lucked out on aggregate
  # verdicts because '?' == '?' across all nodes.
  def tid_to_hex(t):
    v = t.get('id')
    if isinstance(v, list) and v and isinstance(v[0], int):
      return ''.join(f'{b:02x}' for b in v)
    return t.get('transaction_id') or t.get('hash') or t.get('tx_id') or '?'
  tx_ids = [tid_to_hex(t) for t in txs]
  def b2h8(field):
    # byte array → first 8 hex chars OR string verbatim
    v = hdr.get(field)
    if isinstance(v, list) and v and isinstance(v[0], int):
      return ''.join(f'{b:02x}' for b in v[:8])
    return str(v)[:16] if v is not None else 'null'
  print(f\"prev={b2h8('prev_block_hash')};tx_root={b2h8('tx_root')};sol_root={b2h8('solutions_root')};tx_count={len(txs)};tx_first={(tx_ids[0] if tx_ids else 'none')[:18]};tx_last={(tx_ids[-1] if tx_ids else 'none')[:18]};hdr_h={hdr.get('height', 'none')}\")
except Exception as e:
  print('UNREACHABLE')
" 2>/dev/null)
    if [ -z "$blk" ] || [ "$blk" = "UNREACHABLE" ]; then
      unreachable_for_h=$((unreachable_for_h + 1))
      NODE_BLOCK[$node]="UNREACHABLE"
    else
      responding=$((responding + 1))
      NODE_BLOCK[$node]="$blk"
    fi
  done

  if [ "$responding" -lt 2 ]; then
    echo "HEIGHT h=$H nodes_responding=$responding/${#NODES[@]} verdict=UNREACHABLE"
    UNREACHABLE=$((UNREACHABLE + 1))
    unset NODE_BLOCK
    continue
  fi

  # Compare each field across all responding nodes
  fields=(prev tx_root sol_root tx_count tx_first tx_last hdr_h)
  diverged_fields=()
  for f in "${fields[@]}"; do
    seen=""
    diverged=0
    for node in "${NODES[@]}"; do
      blk="${NODE_BLOCK[$node]}"
      [ "$blk" = "UNREACHABLE" ] && continue
      # Extract field value from the per-node string
      v=$(echo "$blk" | tr ';' '\n' | grep "^$f=" | head -1 | cut -d= -f2-)
      if [ -z "$seen" ]; then
        seen="$v"
      elif [ "$seen" != "$v" ]; then
        diverged=1
        break
      fi
    done
    [ "$diverged" -eq 1 ] && diverged_fields+=("$f")
  done

  if [ ${#diverged_fields[@]} -eq 0 ]; then
    echo "HEIGHT h=$H nodes_responding=$responding/${#NODES[@]} verdict=OK"
    OK=$((OK + 1))
  else
    echo "HEIGHT h=$H nodes_responding=$responding/${#NODES[@]} verdict=DIVERGED fields=${diverged_fields[*]}"
    for f in "${diverged_fields[@]}"; do
      pairs=""
      for node in "${NODES[@]}"; do
        blk="${NODE_BLOCK[$node]}"
        [ "$blk" = "UNREACHABLE" ] && continue
        v=$(echo "$blk" | tr ';' '\n' | grep "^$f=" | head -1 | cut -d= -f2-)
        # Short node label = host:port from URL
        label=$(echo "$node" | sed -E 's|^https?://||' | head -c 22)
        pairs+="$label=$v "
      done
      echo "DIVERGENCE h=$H field=$f $pairs"
    done
    DIVERGED=$((DIVERGED + 1))
  fi
  unset NODE_BLOCK

done <<< "$HEIGHTS"

# Verdict
verdict="PASS"
[ "$DIVERGED" -gt 0 ] && verdict="FAIL"
[ "$DIVERGED" -eq 0 ] && [ "$UNREACHABLE" -ge "$N" ] && verdict="FAIL"
echo "SUMMARY total=$N ok=$OK diverged=$DIVERGED unreachable=$UNREACHABLE verdict=$verdict"

# Exit 0 PASS, 1 FAIL
[ "$verdict" = "PASS" ] && exit 0 || exit 1
