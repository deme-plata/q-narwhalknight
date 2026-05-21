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
