# P2P chunk stability — request-response streams keep closing mid-transfer

**Date**: 2026-05-18
**Status**: Brief for Codex investigation (or DeepSeek, or whoever picks it up)
**Targets**: v10.9.57 candidate
**Severity**: HIGH — blocking deploy of v10.9.56 to Gamma/Beta/Epsilon
**Companion**: `docs/v10.9.56-libp2p-mesh-failure-investigation.md`, `docs/peer-registry-self-heal.md`

---

## TL;DR

v10.9.56 fixed the stale-peer-ID class of mesh failures. It did **not** fix the underlying P2P chunk-transfer stability problem. Delta on v10.9.56 is currently wedged at height 18,143,624 with the network producing past 18,144,800 — a growing gap of 1,200+ blocks. The symptom is `Request-response channel closed` errors during turbo-sync chunk fetches, with the HTTP-fallback heuristic refusing to engage because `peers > 0`.

We need someone to investigate (a) **why** the libp2p request-response streams are closing mid-transfer, and (b) tighten the HTTP-fallback heuristic so a node with peers-but-zero-chunk-throughput actually falls back instead of starving forever.

---

## Observed symptoms

### Delta (5.79.79.158), running v10.9.56 since 2026-05-18 14:30 UTC

```
17:46:40  needs_http=false, p2p_only=true, net_h=18144632, peers=3, cur_h=18143624, gap=1008
17:49:51  needs_http=false, p2p_only=true, net_h=18144760, peers=3, cur_h=18143624, gap=1136
17:52:11  needs_http=false, p2p_only=true, net_h=18144859, peers=3, cur_h=18143624, gap=1235
```

`cur_h` is the node's local height. It has not advanced in 6+ minutes. `net_h` is the highest network height announced by peers — it advances normally. `peers=3` is stable. The gap grows ~200 blocks per 6 minutes (which matches network block production rate).

Concurrent turbo-sync errors:

```
ERROR q_storage::turbo_sync: Range: 18144651..=18144652
ERROR q_storage::turbo_sync: Peer: 12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM (score decreased)
ERROR q_storage::turbo_sync: ❌ Failed chunk 18144651-18144652 after 3 retries with 1 different peers:
                              P2P direct request failed for chunk 18144651..=18144652
                              (local height: 18143624): Request-response channel closed. Will retry.
ERROR q_storage::turbo_sync: 🚨 [TURBO SYNC] DOWNLOAD FAILED!
   Completed chunks: 0/1
   Failed chunks: 1
   Success rate: 0.0%
   This prevents phantom success - refusing to claim sync complete!
   Will fall back to HTTP sync for missing blocks.
ERROR q_api_server: ❌ [EMERGENCY SYNC] TurboSync failed: TURBO SYNC incomplete:
                     1/1 chunks failed (0.0% success rate). This prevents phantom success.
                     Falling back to HTTP sync.
```

The turbo_sync layer **says** it will fall back to HTTP. But the periodic HTTP-bootstrap-check heuristic disagrees, because it sees `peers=3` and decides P2P is sufficient. Result: Delta is stuck in a no-op loop where turbo_sync fails, says "fall back to HTTP", and the HTTP layer says "no need, you have peers".

### Beta canary (port 62017), running v10.9.56 since ~16:00 UTC

Same symptom on a different test node. Wedged at height ~25,787 for hours. Reason there is documented as "single-peer dependency on Epsilon and Epsilon's storage has a forward-seek gap at that height" — i.e. the peer it's dialling can't serve the requested chunk and there's no fallback.

The Delta and Beta-canary symptoms are the same shape: **node has peers, P2P request-response keeps closing, HTTP fallback never engages.**

---

## Hypotheses worth investigating (ranked by likelihood)

### H1 — Server-side block-pack semaphore is too restrictive (v9.1.9 inheritance)

The 2026-03-06 OOM-fix in `crates/q-network/src/unified_network_manager.rs` added a `block_pack_semaphore: Arc<Semaphore::new(4)>` — max 4 concurrent block-pack responses per node. Code path:

```rust
match block_pack_semaphore.clone().try_acquire_owned() {
    Ok(permit) => { /* serve, drop permit on completion */ }
    Err(_) => { /* drop request, peer retries */ }
}
```

`try_acquire_owned` returns `Err` immediately when all 4 permits are held. The current code (per the v9.1.9 commit) returns an *empty response* on permit-exhaustion. **An empty response from a request-response channel is indistinguishable from a closed channel on the client side.** The client sees "channel closed" and decreases the peer score.

If Epsilon is serving multiple sync clients (Delta, Beta canary, and others) concurrently, the 4-permit ceiling is exhausted and *all* simultaneous clients see "channel closed" intermittently. Each retry doesn't actually wait for a permit to free — it just rolls the dice again.

**What to check:**
- Is the semaphore ceiling of 4 still appropriate in 2026? At 50MB per response, 4 × 50MB = 200MB is the OOM-safe ceiling. But Epsilon has 64GB RAM. Could safely be 16 or 32.
- Does the empty-response path log a warning? It should (`debug!` is insufficient — this is a real protocol violation that the client mistakes for a closed channel).
- Could we switch from `try_acquire_owned` to `acquire_owned().await` with a short timeout? That gives clients backpressure-without-failure instead of pseudo-error.

### H2 — Request-response timeout is shorter than the actual transfer time

libp2p's `request-response` behaviour has a default timeout (10s in our config? Check `crates/q-network/src/protocols.rs` or wherever the behaviour is configured). For large chunks (200 blocks × 250KB average = 50MB) over a marginal connection, 10s might be insufficient on a slow link or under load.

**What to check:**
- The configured timeout for the block-pack `request-response::Behaviour`. Look for `Config::set_request_timeout` calls.
- Does it scale with chunk size? It probably shouldn't be a flat 10s for both a 1-block and a 200-block request.
- Are there `inflight_limit` constraints we're hitting?

### H3 — Per-substream backpressure inside libp2p Yamux

If the underlying Yamux substream has full receive buffers (because the client is busy validating blocks and not draining the stream fast enough), the server-side write may stall, eventually triggering an idle-timeout that closes the substream. From the client's perspective: "channel closed mid-transfer." From the server's: "client wasn't reading, I gave up."

**What to check:**
- Yamux configuration: `max_buffer_size`, `receive_window`. Defaults are tight. If we have CPU-bound block validation downstream, the receive window will fill and the upstream stalls.
- Block validation latency on a freshly-synced node — is it taking ~100ms per block? That's enough to cause Yamux receive-buffer pressure during a 200-block chunk transfer.

### H4 — Forward-seek gap in serving peer's storage

This was the diagnosed cause of the Beta-canary stall. If the peer has the blocks but they're in a corrupted or gapped state in RocksDB, the read fails, the response is short or empty, and the channel appears to close. **This is fundamentally a server-side data problem masquerading as a network problem.**

**What to check:**
- Verify Epsilon's storage contiguity at the heights Delta is asking for (18,144,651 onward). Use the existing health-check binary, or simply `q-storage::scan_highest_contiguous`.
- If there *are* gaps, fixing the data fixes the symptom. The harder question: how did the gaps appear, and why is the server returning a confusing response instead of an explicit "I don't have that block"?

### H5 — The HTTP-fallback heuristic in main.rs is broken

Separately from why P2P is failing: when P2P *is* failing, the fallback should engage. Current heuristic (look in `crates/q-api-server/src/main.rs` near the periodic `[HTTP BOOTSTRAP CHECK v10.2.8]` log line):

```rust
let needs_http = peers == 0 || /* some other condition */;
```

If `peers > 0` always means `needs_http = false`, then a peers-but-zero-throughput node will starve forever. The heuristic needs a **throughput signal**: if the local height has not advanced in N seconds despite peers > 0, fall back to HTTP regardless.

**Suggested change:**
```rust
let height_stalled = since_last_height_advance > Duration::from_secs(60);
let needs_http = peers == 0 || (height_stalled && local_height < network_height - 100);
```

---

## Acceptance criteria for a v10.9.57 fix

1. **Symptom test**: Delta on v10.9.57 placed in the same network conditions as 2026-05-18 17:46 UTC catches up to network tip within 5 minutes. Verified by `cur_h` reaching `net_h - 20` or better.

2. **Stress test**: 4 fresh canary nodes started simultaneously, all dialling Epsilon. None of them stalls. The Epsilon side does not OOM. The block-pack semaphore (if kept) provides backpressure-without-failure, not silent drops.

3. **Heuristic test**: A canary node configured with one good peer and one peer that intentionally drops 100% of block-pack requests must fall back to HTTP within 60 seconds and resume sync. The current heuristic would never fall back here.

4. **Logging**: Permit-exhaustion and timeout-induced channel closes must log at `WARN` level with the requesting peer ID and the chunk range, so operators can see throughput problems in the journal without enabling debug.

5. **No regression**: The OOM-safety property of the v9.1.9 fix must be preserved. The chain on Epsilon must not crash under the stress test.

---

## What this is NOT

- **Not a libp2p-version upgrade**: we're at libp2p 0.55 (or whatever the current pinned version is). The fix should work with the current version.
- **Not a switch away from request-response**: gossipsub for chunk transfer is a separate research question. v10.9.57 fixes request-response stability.
- **Not a replacement for task #34 (self-healing peer registry)**: that fixes peer *discovery*; this fixes peer *throughput*. Both are needed; neither subsumes the other.
- **Not a Tor-related fix**: the failures observed are direct-TCP, not Tor circuits. If we later find Tor circuits make this worse, that's a separate task.

---

## Suggested PR title

`fix(network): P2P chunk stability — semaphore tuning + timeout scaling + HTTP-fallback throughput heuristic`

## Files to read first

- `crates/q-network/src/unified_network_manager.rs` — search for `block_pack_semaphore`, the request-response handler
- `crates/q-network/src/protocols.rs` (or wherever request-response is configured)
- `crates/q-storage/src/turbo_sync.rs` — the client-side chunk-fetch loop that's seeing "channel closed"
- `crates/q-api-server/src/main.rs` — search for `[HTTP BOOTSTRAP CHECK v10.2.8]` for the heuristic
- `docs/v10.9.56-libp2p-mesh-failure-investigation.md` — the prior peer-ID-staleness investigation; useful for context
- `docs/peer-registry-self-heal.md` — companion task that fixes peer *discovery*; this task fixes peer *throughput*

## Observability suggestions while debugging

- `tcpdump -i any -nn port 9001 and host <peer-ip>` on Epsilon during a stress test to see whether streams are closing TCP-cleanly or RST'd
- Prometheus: histogram of block-pack response sizes and durations; gauge of `block_pack_semaphore.available_permits()`
- Add `tracing::info_span!("block_pack_serve", peer=%peer_id, range=?range)` around the response code so the journal shows each transfer

---

*Drafted by Claude Opus 4.7 from a Delta-soak diagnostic session on 2026-05-18 17:50 UTC. The button is yours to push.*
