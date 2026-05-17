# Technical Review: HTTP Block Endpoint — Why It Returns 404
## Investigating the high_performance_server Request Interception
### Date: 2026-04-17 | Status: OPEN — needs code audit

---

## 1. The Problem

`GET /api/v1/blocks/{height}` returns HTTP 404 with empty body for ALL heights — including blocks at chain tip that definitely exist in RocksDB.

The Axum handler (`handlers::get_block_by_height`) never executes — its `info!()` log message never appears. Something intercepts the request before it reaches the Axum router.

This blocks the checkpoint sync probe from finding blocks at lower heights, forcing new nodes to start syncing from ~14M instead of ~100K where DAG blocks exist.

## 2. Evidence

```
# All return 404 with empty body:
curl http://epsilon:8080/api/v1/blocks/100441   → 404  (DAG block exists)
curl http://epsilon:8080/api/v1/blocks/15000000  → 404  (qblock:height: exists)
curl http://epsilon:8080/api/v1/blocks/15730000  → 404  (current tip)

# But this works:
curl http://epsilon:8080/api/v1/sync/blocks?from_height=15730000&limit=1  → 200 (returns block data)
curl http://epsilon:8080/api/v1/status  → 200 (ready)

# And P2P block-pack serving works:
[BATCH FETCH] Got 121/121 blocks (via P2P, same get_qblocks_range function)
```

The `sync/blocks` endpoint at tip height works, but `blocks/{height}` at the same height doesn't. Both should call storage functions on the same DB.

## 3. Where to Investigate

### 3.1 `crates/q-api-server/src/high_performance_server.rs`

This custom HTTP server module may handle some routes before passing to Axum. Check:
- Does it have its own route table that matches `/api/v1/blocks/` patterns?
- Does it return 404 for unrecognized patterns before forwarding to Axum?
- Is there a request filter that blocks numeric path segments?

### 3.2 q-flux Reverse Proxy

Epsilon uses q-flux (NOT nginx) on ports 80/443. When curling `localhost:8080`, q-flux should NOT be involved. But verify:
- Is there a q-flux rule that intercepts `/api/v1/blocks/` on port 8080?
- Check `crates/q-flux/src/config.rs` and `upstream.rs` for route matching

### 3.3 Axum Middleware / Tower Layers

The Axum app may have middleware that rejects certain URL patterns:
- CORS layer (unlikely — other endpoints work)
- Rate limiter that blocks `/blocks/` path
- Authentication middleware that rejects unauthenticated block requests
- Request body size limiter that triggers on large responses

### 3.4 Route Ordering in main.rs

Line 23207: `.route("/api/v1/blocks/:height", get(handlers::get_block_by_height))`

Check if another route registered BEFORE this one matches the same pattern:
- Is there a `.route("/api/v1/blocks", ...)` without the `/:height` that catches everything?
- Is there a wildcard route like `.fallback(...)` that matches first?
- Does the `high_performance_server` module register its own routes that shadow this one?

### 3.5 The `sync_blocks` Handler Mystery

`/api/v1/sync/blocks?from_height=15730000&limit=1` returns block data at the same height where `/api/v1/blocks/15730000` returns 404. The `sync_blocks` handler's `info!()` log also doesn't fire (we checked). This means:

**Hypothesis: NEITHER handler is running.** Something else is generating the responses:
- For `/api/v1/sync/blocks`: returns `{"success":true,"data":{"blocks":[...]}}` — this JSON format matches our handler, so maybe it IS running but the log is filtered
- For `/api/v1/blocks/15730000`: returns empty 404 — this might be the default Axum 404 (route not found)

**Alternative hypothesis:** The high_performance_server has its OWN implementation of `/api/v1/sync/blocks` that works, but does NOT implement `/api/v1/blocks/:height`.

## 4. Proposed Investigation Steps

### Step 1: Read `high_performance_server.rs` (10 minutes)

```bash
grep -n "blocks\|route\|handler\|path\|api/v1" \
  crates/q-api-server/src/high_performance_server.rs | head -40
```

Look for:
- Route registration that shadows Axum routes
- Request matching/dispatching logic
- Any `/blocks/` pattern handling

### Step 2: Add info-level logging to the block handler (5 minutes)

Change the handler's debug to info so we can confirm it's being called or not:
```rust
pub async fn get_block_by_height(...) {
    info!("📥 [BLOCKS API] Request for height {}", height);  // Was debug
    ...
}
```

Rebuild and test. If the info log appears, the handler IS running but `get_qblock_any_format` returns None (different bug). If it doesn't appear, the route isn't being reached.

### Step 3: Check if q-flux intercepts the request (2 minutes)

```bash
# Test via q-flux (port 80) vs direct (port 8080)
curl http://localhost:80/api/v1/blocks/15730000 -w "%{http_code}\n"
curl http://localhost:8080/api/v1/blocks/15730000 -w "%{http_code}\n"
# If both return 404, q-flux is not the issue
```

### Step 4: Try a test route to confirm Axum routing works (5 minutes)

Add a temporary test route NEXT to the blocks route:
```rust
.route("/api/v1/blocks-test/:height", get(|Path(h): Path<u64>| async move {
    format!("Test: height {}", h)
}))
```

If `/api/v1/blocks-test/123` returns "Test: height 123" but `/api/v1/blocks/123` returns 404, the issue is specific to the `/blocks/:height` pattern.

## 5. Alternative Fix (If HTTP Endpoint Can't Be Fixed Quickly)

Change the checkpoint probe to use P2P block-pack requests instead of HTTP:

```rust
// In turbo_sync.rs, replace probe_network_gap_blocking (HTTP) with P2P probe
async fn probe_network_gap_p2p(&self, target_height: u64, peers: &[PeerId]) -> u64 {
    let heights = vec![1000, 10000, 50000, 100000, 250000, 500000, 1_000_000, ...];
    
    for &h in &heights {
        // Request 1 block at height h from any connected peer
        let blocks = self.request_blocks_from_peer(h, 1, &peers[0]).await;
        if !blocks.is_empty() {
            info!("Blocks found at height {} via P2P", h);
            // Binary search for exact start
            return binary_search_first_block(h, peers).await;
        }
    }
    0
}
```

This bypasses the broken HTTP endpoint entirely. The P2P block-pack handler (which has our DAG fallback) correctly serves blocks at all heights.

**Trade-off:** The P2P probe requires connected peers (takes 30-60 seconds to establish connections). The HTTP probe can run immediately on startup (but doesn't work). So the P2P probe would slightly delay the sync start.

## 6. Risk Assessment

| Fix | Risk | Time | Impact |
|-----|------|------|--------|
| Fix HTTP endpoint (find interception) | LOW | 1-4 hours | Full fix — checkpoint + API both work |
| P2P checkpoint probe (bypass HTTP) | LOW | 1-2 hours | Checkpoint works, API still broken |
| Both | LOW | 2-5 hours | Complete solution |

## 7. Questions for DeepSeek

### Q1: Could the high_performance_server be handling HTTP on a separate TCP listener that shares port 8080?

If the custom server binds port 8080 and forwards SOME requests to Axum but handles others directly, that would explain why some routes work (forwarded) and others don't (handled by the custom server with a default 404).

### Q2: Is there a known pattern in Axum where Path<u64> extraction fails silently for certain numeric values?

If `Path<u64>` can't parse the height, Axum returns 404 by default (before the handler runs). But we tested with various valid u64 values (100441, 15000000, 15730000) — all fail. So this is unlikely unless there's a route conflict.

### Q3: Should we add a `/health` endpoint while we're fixing the HTTP layer?

The watchdog bug (probing non-existent `/health`) could be fixed by adding a simple health endpoint. This is independent of the blocks endpoint but could be done in the same code change.

---

*Generated 2026-04-17 — Quillon Foundation*
*Next action: Read high_performance_server.rs and audit the request dispatch chain*
