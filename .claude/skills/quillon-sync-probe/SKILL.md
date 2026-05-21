---
name: quillon-sync-probe
description: Diagnose why a Quillon node is sync-stalled, slow, or wedged. Probes a target node (production or test container on Epsilon/Delta) via four parallel data sources — API status + Prometheus metrics + filtered docker logs + tshark packet capture — and synthesizes a leading-hypothesis report. Invoke when the user says "diagnose sync", "why is this node slow", "10.10.x sync is broken", "probe the test container", "tshark this thing", "is the node wedged", or any close paraphrase. Pass an optional target (container name OR API URL OR server short-name) as the argument; if absent, ask the user which target.
---

# Quillon Sync Probe — diagnostic playbook

## What this skill does

Given a sync-suspect node, run four parallel diagnostics and return a
single-page report with a **leading hypothesis** plus the evidence.

Used heavily through the v10.10.10 → v10.10.13 sync-break investigation
(May 2026). Codifies that playbook so the next time a node looks stuck
you don't have to re-derive the probe set.

## When to invoke

- "diagnose sync on [container/server]"
- "why is this node slow"
- "10.10.x sync is broken"
- "probe the test container"
- "tshark this thing"
- "is the node wedged"
- "compare sync rate vs production"

Do NOT invoke for:
- Production deploys (use `./scripts/ha-deploy.sh full -y`)
- Building/deploying test containers (use `quillon-docker-test` skill)
- Application-layer bugs unrelated to consensus/sync (regular debug)

## Inputs

One positional argument — the target. Accepted forms:

| Form | Example | Notes |
|---|---|---|
| Container name | `q-test-v10.10.13-eps` | Auto-derive host (eps→Epsilon, delta→Delta) |
| Hostname | `epsilon` or `delta` or `89.149.241.126` | Probe the production systemd service |
| Full API URL | `http://localhost:8087/api/v1` | Use as-is |
| Empty | — | Ask user which target |

## Procedure

### Phase 0 — Target resolution

1. Resolve target to `(host, api_url, container_name|null, p2p_port|null)`.
2. Quick aliveness ping: `curl -s --max-time 5 ${api_url}/status | head` — if no response, skip Phase 1 API probes and rely on logs.
3. Establish baseline reference — production tip via `mcp__quillon-wallet__chain_overview` or `https://quillon.xyz/api/v1/engine/pulse`.

### Phase 1 — API state (read-only, fast)

Three calls, in parallel where possible:

```bash
curl -s ${api_url}/status                    # height, peers, syncing, network_id
curl -s ${api_url}/network/supply             # total_mined, current_height
curl -s ${api_url}/peers                      # peer list, their heights
```

If the wallet MCP tools are loaded, prefer:
- `mcp__quillon-wallet__verify_node_consistency` — primary vs secondary direct comparison.
- `mcp__quillon-wallet__chain_overview` — high-level.
- `mcp__quillon-wallet__engine_pulse` — detailed sync gauges.

Report: target height, peers connected, height delta vs production, total_supply consistency.

### Phase 2 — Prometheus metrics

```bash
curl -s ${api_url%/api/v1}/metrics | grep -E '^qnk_(peers_connected|height|sync_|turbo_|gap_|height_pointer|synced_through|gossipsub_|block_pack_)'
```

Trim to ~20 meaningful lines and report verbatim. Key gauges to look for:

| Gauge | Healthy value | Failure signature |
|---|---|---|
| `qnk_peers_connected` | ≥1 (test) / ≥3 (prod) | 0 = no mesh |
| `qnk_gap_to_tip` | matches real gap | 0 when real gap > 0 = sync-pointer code path missing |
| `qnk_block_pack_response_*` | non-zero counters | all zero = node has served zero block-packs |
| `qnk_gossipsub_forward_failed_total{reason="closed"}` | rare | high count on every topic = receiver task died, supervisor not catching |
| `qnk_synced_through_*` | present + advancing | absent entirely = pre-v10.9.55 sparse-chain code missing |
| `qnk_height_pointer_*` | present | absent = same as above |
| `qnk_gap_detected` / `qnk_known_gap_advances_total` | corresponds to logs | mismatch = sparse-chain code broken |

### Phase 3 — Docker logs

```bash
docker logs --since 5m ${container_name} 2>&1 | grep -E 'ERROR|WARN|sync_to_height|synced_through|turbo_sync|height_pointer|peer_height|gossipsub|stuck|stall|gap_detected|BalanceRoot|DialFailure|BLACKLIST' | tail -40
docker logs --tail=30 ${container_name} 2>&1   # general state
```

For production (not a container) use `journalctl`:
```bash
journalctl -u q-api-server --since "5 minutes ago" | grep -E '...' | tail -40
```

Look for:
- **Sync-loop signatures**: repeated `sync_to_height` with same target, no advance.
- **Dial-failure storms**: `DialFailure` + `BLACKLISTED` cycles (v10.10.11 regression).
- **Gossipsub task death**: `forward FAILED ... reason=closed` on multiple topics.
- **Balance-replay loops**: pre-v10.9.55 issue, should not recur post-cherry-pick.
- **Height-decay self-poisoning**: `[HEIGHT DECAY] Peer data stale` followed by network_height decreasing.
- **Sparse-chain gap handling**: absence of `synced_through` / `h_present` / `sync_to_height` log lines indicates the v10.9.55 code is NOT in this binary.

### Phase 4 — tshark on docker0 (or eth0 for prod)

```bash
# 60s capture, summarize TCP conversations
timeout 60 tshark -i docker0 -f "tcp port ${p2p_port} or tcp port ${api_port}" -q -z conv,tcp 2>&1 | head -30

# 60s capture, peek at packet flags (look for RST storms, half-open connections)
timeout 60 tshark -i docker0 -f "tcp port ${p2p_port}" -c 200 -T fields -e frame.time -e ip.src -e ip.dst -e tcp.flags 2>&1 | head -20
```

If `docker0` returns 0 frames, the container is doing NAT and traffic
flows on the `any` interface — re-run with `-i any` and broader filter.

Report:
- Bidirectional conversation count (peers actually exchanging data)
- Total frames in 60s, byte ratios
- RST/SYN-without-ACK patterns (broken connections)
- Whether containers are talking to expected peers (IP match)

### Phase 5 — Resource snapshot

```bash
docker stats --no-stream ${container_name}
docker exec ${container_name} ps -ef 2>/dev/null | head -5    # may fail on busybox-less images
```

CPU pegged + zero block-progress is the classic dial-loop signature.
RAM growth without bound = leak (gossipsub buffering, block-pack queue, etc).

### Phase 6 — Cross-check vs production

```bash
curl -s https://quillon.xyz/api/v1/status   # canonical tip reference
mcp__quillon-wallet__verify_node_consistency --secondary_url=${api_url}
```

Confirms: is the target diverging from prod (real consensus problem),
or merely lagging (slow sync, not a fork)?

## Synthesis — leading hypothesis output

Concentrate findings into a 6-section report:

```
== ${TARGET} SYNC PROBE ==

[1] API STATE
  height: N    peers: M    syncing: T/F    network_id: ...
  delta-from-prod: ΔH blocks   sync rate (last 60s): X blocks/sec

[2] PROMETHEUS — key gauges
  [verbatim grep output, trimmed to 20 most relevant]

[3] LOG SIGNATURES (last 5min)
  Top 5–10 damning lines with context.
  Categorize: dial-loop / gossipsub-dead / balance-replay / sparse-chain-missing / height-decay.

[4] TSHARK SUMMARY
  Frames in 60s, conv count, anomalies.

[5] RESOURCE
  CPU/RAM/PID count.

[6] LEADING HYPOTHESIS
  ONE sentence: "Sync is broken because <root cause>." with cited evidence.
```

## Sparse-chain expertise (mandatory mental model)

**This skill is an expert on `docs/technical-review-sparse-chain-truth-v1.md`**
— the 2026-05-18 read-only investigation that closed three previous wrong
diagnoses. Internalize before probing:

### The chain is sparse by BOTH design AND damage

| Height range | % present | Cause |
|---|---|---|
| 0 – 7M | ~3% | Compaction loss (kill -9, Mar 2026) + v10.2.8 cleanup |
| 7M – 14M | ~50% | v10.2.8 cleanup dominates |
| 14M – 15M | 78% | Cleanup tail + protocol sparsity |
| **15M – 18.1M (tip)** | **93-96%** | **Pure DAG-Knight design — no anchor every round** |

Decisive fact: 15M-16M has 584,366 gaps but only 73,835 missing heights —
**average gap is 0.13**. That is not damage; that is DAG-Knight legitimately
not finalizing an anchor at every integer round number.

### What this means for sync diagnosis

1. **"Height N+1 missing after N" is NOT a bug** in the post-14M range.
   It's protocol behavior. A sync layer that demands contiguous heights
   stalls forever. The v10.9.55 fix shipped `synced_through` pointer +
   `h_present` marker exactly to fix this.

2. **Pre-7M historical loss is permanent and irrelevant.** Beta + Gamma
   have the same loss. No peer has the data. Blocks are coinbase-only
   and balances are intact via P2P state sync.

3. **A node "stuck at 26K"** is almost always one of two causes:
   - Pre-v10.9.55 binary missing the sparse-chain pointers
   - v10.10.11/.12 dial regression preventing block-pack retrieval

4. **`Q_KNOWN_PERMANENT_GAPS=25988:100440`** is the canonical first
   gap-bypass for fresh nodes. v10.9.50 made it default-on; pre-v10.9.50
   nodes need it explicitly.

### Three previous wrong diagnoses to avoid repeating

The TR documents three TRs that reached different wrong conclusions:

| Date | TR | Wrong conclusion | Why it was wrong |
|---|---|---|---|
| 2026-04-16 | `epsilon-block-gap-forensics-v1.md` | "8.4M blocks gone, 1.6M-10M empty" | Used `ldb scan` with `prefix_iterator_cf` → bloom-filter false negatives on CF_BLOCKS overstated the gap |
| 2026-04-17 | `http-block-endpoint-fix-v1.md` | "HTTP endpoint returns 404 — Axum interception bug" | Symptom real, cause inverted: the underlying `scan_prefix` hit the bloom-filter bug. Indirectly fixed by v10.3.7 `scan_prefix_seek` |
| 2026-05-17 | "1M → 13M jump in fresh-node sync" | Needs permanent_gap framework | Half-right: the "jump" wasn't a jump — lex order on decimal heights is non-numeric |

**The shared mistake**: treating missing-height-N as a bug when it's largely
protocol behavior in the recent chain, plus localized historical damage
in pre-14M.

### Rules to never break (from §7 of the TR)

1. **Never delete from CF_BLOCKS / CF_TRANSACTIONS / CF_QUANTUM_METADATA
   in any hot read/sync path.** Deserialization failure → log + skip +
   return None. The v10.2.8 cleanup destroyed ~5.8M blocks via this.

2. **Never `kill -9` a running node.** Use SIGTERM with 60s timeout. The
   Mar 2026 compaction loss came from `kill -9` interrupting in-flight
   SST writes.

3. **Never assume `ldb scan` finds all keys without verifying.** Use
   `iterator_cf(IteratorMode::From)` or `scan_prefix_seek`. Bloom-filter
   false negatives on CFs without prefix extractors will silently miss
   keys.

4. **Never sort decimal-string heights lexicographically.** "1000000" <
   "10000031" but heights 1,000,001..9,999,999 sort between them. Always
   parse to integer first.

### Reproducible measurement command (read-only, no DB lock)

If you need to verify the per-million-bucket coverage yourself:

```bash
DB=/home/orobit/data-mainnet-genesis/hot
SEC=/home/orobit/tmp/ldb_sec_$$
mkdir -p $SEC

# Format 1: canonical chain pointer (qblock:height:N)
ldb --db=$DB --column_family=blocks --try_load_options --secondary_path=$SEC \
    scan --from='qblock:height:' --max_keys=20000000 2>/dev/null \
  | awk -F':' '/^qblock:height:[0-9]+/ {gsub(/ ==>.*/, "", $0); print $3}' \
  | sort -un > /home/orobit/tmp/heights_canonical.txt

# Format 2: DAG layer (qblock:dag:N:proposer_hex)
ldb --db=$DB --column_family=blocks --try_load_options --secondary_path=$SEC \
    scan --from='qblock:dag:' --max_keys=20000000 2>/dev/null \
  | awk -F':' '/^qblock:dag:[0-9]+:/ {print $3}' \
  | sort -un > /home/orobit/tmp/heights_dag.txt

# Per-million-bucket coverage
sort -un /home/orobit/tmp/heights_*.txt | awk '
  NR==1 { prev=$1; next }
  $1 == prev + 1 { prev=$1; next }
  $1 > prev + 1 {
    bucket = int(prev / 1000000); g = $1 - prev - 1
    missing[bucket] += g; gaps[bucket]++
    prev=$1
  }
  END {
    for (b=0; b<=18; b++) {
      m = missing[b]+0; gc = gaps[b]+0
      avg = (gc > 0) ? m/gc : 0
      printf "%2dM-%dM: %8d missing %7d gaps avg_gap=%.2f\n", b, b+1, m, gc, avg
    }
  }'

rm -rf $SEC
```

Expected output matches the §1 table of the TR. If pre-7M starts to fill
in over time → evidence of P2P refill. If post-15M density drops → that's
a regression to investigate.

## Hard-won lessons encoded in this skill

- **tshark on `docker0` may show 0 frames when the container is on a
  bridge network with NAT.** Fall through to `-i any` with a broader
  IP filter rather than concluding the network is dead.
- **`qnk_gap_to_tip = 0` while real gap > 0** is the smoking gun that
  the v10.9.55 sparse-chain pointer code is absent. The gauge is
  WRITTEN by `synced_through` advancement; if the writer is missing,
  the reader stays at 0.
- **DialFailure + Identify success** means the dial fix at
  `unified_network_manager.rs:~6669` is missing — gossipsub mesh
  works (Identify completes) but request-response dies because the
  PEER CHECK path returns `Some(pid)` without calling `swarm.dial`.
  This was the v10.10.11/.12 regression (fixed v10.10.13).
- **Last-solution timestamp display bug**: the MCP
  `mining_network` tool reports "Xs ago" but the value is a unix
  timestamp, not a delta. Sanity-check against current time before
  alarming about a 56-year-old solution.
- **Production Beta has been offline since 2026-05-17**. If you see
  `qnk_peers_connected=1` on prod Epsilon, that's likely Delta or
  Gamma — Beta isn't peering. Don't conclude consensus is broken
  from peer-count alone.
- **K-parameter ≈ 0 on the live chain since at least 2026-05-20**
  means effectively one validator producing anchors. Single-operator
  risk is a separate concern from individual-node sync — don't
  conflate them in the report.
- **Per CLAUDE.md balance-integrity rules**: NEVER run replay
  (`replay_post_checkpoint_balances`) as part of diagnosis. If you
  see balance writes happening during probe, that's the node's own
  reconciliation — do not trigger it manually.

## Auto-recovery suggestions (after the leading hypothesis)

| Hypothesis | Recommended next step |
|---|---|
| Dial-loop (DialFailure × N) | Verify v10.10.13+ binary running; if v10.10.11/.12, deploy v10.10.13 |
| Sparse-chain pointer absent | Verify v10.9.55+ cherry-pick is in the source tree |
| Gossipsub task died | Check supervisor wrap at `main.rs:10024`; deploy v10.10.11+ |
| Balance replay loop | Check `is_checkpoint_applied()` returns false; per CLAUDE.md Rule 2 |
| Height decay self-poisoning | Wait for peer-height gossip to resume after dial fix; this auto-heals |
| OOM under load | Check `ROCKSDB_BLOCK_CACHE_MB` env, recall v9.0.7 memory tuning |
| Fork (height divergent from prod) | STOP — do not deploy; investigate the divergent state separately |

## See also

- `quillon-docker-test` skill — builds + deploys fresh test containers
- `CLAUDE.md` — "How to check mining rewards and balances (CORRECT WAY)"
- Memory `mining_rejection_storm.md` — pattern for high-volume endpoint storms
- Memory `sync_starvation_root_cause.md` — v10.9.40 tokio::select! starvation
- `/tmp/qnk-sync-probe-reference.md` — longer-form command catalogue
