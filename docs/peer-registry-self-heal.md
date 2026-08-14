# Self-healing peer registry (task #34)

**Date**: 2026-05-18
**Status**: Brief for Codex (or Claude, or DeepSeek) to implement
**Targets**: v10.9.57 candidate
**Companion**: `docs/v10.9.56-libp2p-mesh-failure-investigation.md` (PR #78), `papers/state-of-the-art-2026-05-18.pdf` §"The cost of speed"

---

## The problem this solves

Every time a Quillon Graph node's peer ID rotates (DB resync, identity regen, fresh install), the *hardcoded* peer-ID list in `crates/q-network/src/unified_network_manager.rs::HARDCODED_BOOTSTRAP_PEERS` and `crates/q-miner/src/p2p_network.rs` becomes wrong. We've hit this exact bug class three times this month:

- Gamma rotated 2026-05-16 (legacy `WFfZKfKbBn…` → `WHNhCWYmUi…`); production lists stayed stale until v10.9.54
- Delta rotated in v10.9.54 (`WLJJRvqo6m…` → `WPg1GsUh…`); two halves of the fix landed on different branches and `release/v10.9.55` snapshot was cut between them, leaving `q-network` stale even while `q-api-server` was correct — root cause documented in PR #78
- Gamma rotated AGAIN today 2026-05-18 12:55 CEST after fresh-DB resync (`WHNhCWYmUi…` → `WEZKN13gs…`); fixed in v10.9.56 manually-rewritten constants

The bug recurs because the source of truth lives **inside a compiled binary**. The data is correct when the binary is built; it goes stale the moment anything rotates. Operators discover the breakage when production peers drift to zero connections.

It also blocks legitimate test/canary setups: a v10.9.56 sync-test node on Beta running on port 62017 (not the canonical 9001) can never get admitted to the mesh because no production peer's allowlist knows about it. Observed today during the Delta soak — that test node has been stalled at height ~25,787 for hours because it has exactly *one* outbound peer (Epsilon-prod) and any single-peer dependency dies when that peer's storage has a forward-seek gap.

The fundamental fix is to move peer discovery out of the binary and into a runtime registry.

---

## The proposed design (three small parts)

### Part A — `GET /api/v1/p2p/known-peers` on every node

Returns, as JSON, the set of peers this node currently has admitted to its swarm:

```json
{
  "as_of": "2026-05-18T15:30:00Z",
  "node_peer_id": "12D3KooWFpbXxxZJ…",
  "stale_after_secs": 60,
  "peers": [
    {
      "peer_id": "12D3KooWPg1GsUh…",
      "multiaddrs": [
        "/ip4/5.79.79.158/tcp/9001",
        "/ip4/5.79.79.158/tcp/9001/p2p/12D3KooWPg1GsUh…"
      ],
      "last_seen_unix": 1779118450,
      "connected_now": true,
      "network_id": "mainnet-genesis"
    },
    ...
  ]
}
```

Source: the node's *current* `UnifiedNetworkManager` swarm state — `discovered_peers` + `peer_addresses` HashMaps, filtered to currently-connected and recently-seen. **No auth required**: peer addresses are public by definition.

The endpoint MUST NOT include any private identity, balance, or admin data. Pure peer discovery only.

Cache hint: `Cache-Control: max-age=60`. Result is computed cheaply (in-memory state), so 60s edge cache prevents thundering-herd from misbehaving clients without hurting freshness for legitimate use.

Suggested impl location: a new handler in `crates/q-api-server/src/handlers.rs` paired with route registration in `crates/q-api-server/src/main.rs` near the existing `/api/v1/status` route (which is similarly unauthenticated). Reuse the existing `UnifiedNetworkManager` accessor.

### Part B — `q-flux` registry aggregator

`q-flux` already maintains a `cluster_peers` list (see `q-flux.toml` on each server). Add a periodic background job that:

1. Every 30 seconds, GETs `/api/v1/p2p/known-peers` from each `cluster_peer` (currently Beta/Gamma/Delta/Epsilon).
2. Merges results: dedupe by `peer_id`, keep the multiaddrs union, keep the latest `last_seen_unix`, mark `connected_anywhere: true` if at least one cluster_peer reports it as connected.
3. Caches the merged blob in memory with a 60s TTL.
4. Serves the cached blob at `GET /peers.json` on the public domain (e.g. `https://quillon.xyz/peers.json`) with `Cache-Control: max-age=60`.

Result: one HTTPS GET against `quillon.xyz/peers.json` answers *who is alive on Quillon right now*, derived from the live state of every operator-trusted node, with no libp2p handshake required for *discovery*. Libp2p is still used for actual data exchange — but the cold-start dependency on hardcoded constants disappears.

Suggested impl location: a new module in `crates/q-flux/src/peer_registry.rs` + a route handler that serves the cached blob. Wire the periodic job from `main.rs` next to the existing health-check loop.

### Part C — Startup self-heal in `q-api-server`

After the existing `HARDCODED_BOOTSTRAP_PEERS` dial loop runs, if peer count is below threshold:

```rust
// After 30s of normal bootstrap attempts:
if peer_count < BOOTSTRAP_MIN_PEERS {
    if let Ok(registry) = fetch_registry("https://quillon.xyz/peers.json").await {
        for peer in registry.peers {
            // Insert with HIGHER priority than hardcoded — the registry is
            // by definition newer than the binary's compiled-in constants.
            kademlia.add_address(&peer.peer_id, peer.multiaddrs[0].clone());
            kademlia.bootstrap()?;
        }
    }
}
```

Constants:
- `BOOTSTRAP_MIN_PEERS` = 3
- Registry fetch timeout = 10 seconds
- Registry URL configurable via env (`Q_BOOTSTRAP_REGISTRY_URL`, default `https://quillon.xyz/peers.json`)
- TLS verification ON; this is the trust anchor for the network

Falling back to `https://` (rather than libp2p) for this *discovery* call is intentional: it lets a fresh node bootstrap even when its libp2p layer has zero peers to talk to. Once at least one peer is found via the registry, normal libp2p block-pack + gossipsub take over.

**Critical safety property**: a fresh node trusting `quillon.xyz/peers.json` is structurally identical to a fresh node trusting `HARDCODED_BOOTSTRAP_PEERS` — both are operator-signed lists. Replacing one with the other doesn't change the trust model, just the rotation frequency.

Suggested impl location: a new function `fetch_registry_and_seed_kademlia` in `crates/q-network/src/unified_network_manager.rs`, called from the existing bootstrap loop after the hardcoded-peer dial phase.

---

## Acceptance criteria

For the PR to be mergeable, the implementer should:

1. Add the three components (A, B, C) with at least minimal tests.
2. Confirm against the canary container that a fresh node on port `62017` (non-canonical) can announce itself by being admitted to one peer's `/known-peers`, and other nodes can then dial back. Specifically:
   - Start a fresh canary on `--port 8181 --p2p-port 9101 --bootstrap-registry https://quillon.xyz/peers.json`.
   - Within 60 seconds it should reach `qnk_peers_connected ≥ 2` from the registry seed alone.
   - It should appear in Epsilon's `/api/v1/p2p/known-peers` once admitted.
3. Document the env vars + the threat model:
   - Who can post to the registry? (No one — it's read-only, aggregated from operator-trusted cluster peers.)
   - What happens if the registry is wrong? (Same fallback as today: stale data, retry, log warn.)
   - What happens if the registry is hostile / poisoned? (The registry is hosted at `quillon.xyz` which is operator-controlled; same trust anchor as the binary's compiled-in `HARDCODED_BOOTSTRAP_PEERS`.)

---

## What this is NOT

- **Not a replacement for libp2p**: discovery is moved off-binary; actual data exchange (block-packs, gossipsub) still uses libp2p as today.
- **Not a federated/decentralised registry**: in v1, the registry is hosted at one URL (`quillon.xyz`). A future iteration could federate via DNS TXT records, multiple operator-run registries, or signed DHT records — but those are follow-ups, not v1.
- **Not a peer-ID rotation tool**: rotation itself is still operator-initiated. The registry just makes rotations *propagate* automatically once they occur.
- **Not a substitute for the v10.9.56 mesh fix**: v10.9.56 fixes today's stale-peer-ID symptoms via a one-shot constant refresh. This task prevents the *next* recurrence.

---

## Suggested PR title

`feat(network): self-healing peer registry — /known-peers + q-flux aggregator + startup fallback`

## Files Codex (or whoever implements this) might want to read first

- `crates/q-network/src/unified_network_manager.rs:249-280` — current `HARDCODED_BOOTSTRAP_PEERS`
- `crates/q-network/src/unified_network_manager.rs:282-310` — `bootstrap_http_port_for_ip` (v10.9.56 helper, similar pattern)
- `crates/q-network/src/unified_network_manager.rs:1027-1100` — `fetch_peer_id_from_http` (existing HTTP discovery for one peer; this task generalises it)
- `crates/q-api-server/src/main.rs:512-518` — `bootstrap_peer_id_for_url` (URL → peer ID map)
- `crates/q-flux/src/main.rs` — periodic background tasks pattern (health checks, OCSP)
- `crates/q-api-server/src/handlers.rs::status_handler` — example unauthenticated /api/v1 handler
- `docs/v10.9.56-libp2p-mesh-failure-investigation.md` — full context on why this matters

---

*Drafted by Claude Opus 4.7. The button is yours to push, Codex — same pattern as PR #80 and PR #82.*
