# Agent Fiber Lane — co-located submission path for AI-agent transactions

**Date**: 2026-05-19
**Status**: Spec brief for Codex / DeepSeek implementation
**Target version**: v10.10.0 candidate
**Author**: Claude Opus 4.7 (the agent this lane is built for)
**Strategic priority**: HIGH — blocker for the "agent-native chain" positioning in `papers/agentic-money-quillon-graph.pdf`, `papers/state-of-the-art-2026-05-18.pdf`, and `papers/five-mirrors-2026.pdf`

---

## 1. The metaphor, the strategic case

The big banks have fiber lines to NYSE/CME. They pay millions per year for co-location racks inside the exchange building, racing to shave microseconds off order submission. The reason isn't vanity; it's that markets reward whoever decides first, and decisions cost time-to-submit.

AI agents on Quillon Graph are the next "trader class." A single agent will issue thousands of micro-decisions per minute. A swarm of agents will issue millions of transactions per second. They cannot tolerate browser OAuth flows, three-round-trip auth dances, or per-tx TLS handshakes. They need the chain's equivalent of co-located fiber: a direct, signed, batched, low-latency submission path.

Right now that lane does not exist. The two existing transaction paths are:

| Path | Designed for | Agent-suitability |
|---|---|---|
| `POST /api/v1/transactions/send` (OAuth) | Humans with wallets in browser | NO — requires `quillon.xyz/miner-login` browser approval dance |
| `POST /api/v1/transactions` (signed) | P2P / programmatic submitters | BROKEN — `verify_signature` reads `self.hash()` which postcard-includes the signature field itself, an impossible chicken-and-egg (see `docs/peer-registry-self-heal.md` companion analysis) |

The Agent Fiber Lane (this brief) is the third path: built specifically for AI agents holding session-derived Ed25519 keys, accepting both single and batched transaction submission, validated by X-Wallet-Auth proof, with measured latency targets.

The Quillon Graph chain has demonstrated **~100,000 TPS sustained** in the October–December 2025 benchmark cycle. The Agent Fiber Lane must not bottleneck below that ceiling. Per-transaction HTTPS POSTs cannot achieve 100k TPS (TLS handshake cost alone caps it at a few thousand connections/sec). Therefore batching and persistent connections are first-class, not extensions.

---

## 2. Endpoints

### 2.1 `POST /api/v1/agent/submit` — single-tx fast lane

**Authentication**: `X-Wallet-Auth` header, Ed25519-signed challenge per the existing scheme in `crates/q-trading-bot/src/wallet_auth.rs` and documented in `AGENT.md §2`. The challenge MUST include:
- The wallet's address bytes
- A Unix timestamp (i64, little-endian)
- The request path (`/api/v1/agent/submit`)
- **A SHA3-256 hash of the request body** (this binds the signature to the specific tx, preventing replay with a different body)

Header JSON shape (extends current scheme by one field):
```json
{
  "address": "qnk<64hex>",
  "timestamp": 1779168000,
  "scheme": "Ed25519",
  "signature": "<128hex>",
  "body_hash": "<64hex SHA3-256 of the JSON body>"
}
```

**Body**: intent-level, server fills the rest.
```json
{
  "to": "qnkefca1e8c...",
  "amount": "50000000000000000000000",
  "token_type": "QUG",
  "memo": "optional free-form text",
  "fee": null,
  "nonce": null
}
```

Fields `fee` and `nonce` MAY be null — server auto-fills:
- `nonce`: server looks up the wallet's last committed nonce and assigns `last + 1`. If the agent sends multiple submits rapidly, server tracks an in-flight counter per wallet and increments it; pessimistic concurrency control via per-wallet `tokio::Mutex`.
- `fee`: server uses `q_types::MIN_TRANSACTION_FEE` (21000) by default; agent can override for priority.

**Server flow**:
1. Verify X-Wallet-Auth (existing `crates/q-api-server/src/wallet_auth.rs::verify_request`)
2. Verify `body_hash` matches SHA3-256 of received body (replay protection)
3. Reject if timestamp is more than 30s stale OR more than 30s in the future (clock-skew window)
4. Reject if this exact `body_hash` has been seen for this address in the last 60s (dedup window via small in-memory LRU)
5. Look up agent's current nonce; assign next
6. Build full `Transaction` server-side with `from = X-Wallet-Auth address`, `nonce = assigned`, `timestamp = chain time`, `signature_phase = Phase0Ed25519`. `signature` field is set to a server-generated marker: `b"AGENT-FIBER-LANE-V1" || X-Wallet-Auth signature bytes`. This makes the Transaction verifiable later by re-running the X-Wallet-Auth check against the persisted tx fields.
7. Insert into `state.tx_pool` AND `state.production_mempool` (Narwhal) in one operation.
8. Broadcast via gossipsub `/qnk/mainnet-genesis/transactions` topic.
9. Return `{tx_id, assigned_nonce, included_at_block: null}` (caller may poll for `included_at_block` via existing `/api/v1/transactions/{tx_id}`).

**Latency target**: ≤ 50 ms for the full round-trip on Epsilon-local network (server-side time should be under 5 ms once the wallet's nonce is cached).

### 2.2 `POST /api/v1/agent/submit-batch` — batch lane, throughput-first

Same auth scheme as 2.1, but body is an ARRAY of intent objects:

```json
{
  "transactions": [
    { "to": "qnk...", "amount": "...", "token_type": "QUG", "memo": "..." },
    { "to": "qnk...", "amount": "...", "token_type": "QUG", "memo": "..." },
    ...
  ]
}
```

The `body_hash` in the X-Wallet-Auth header is the SHA3-256 of the FULL JSON body (the entire array). One signature authenticates the entire batch.

**Batch size limits**:
- Default: up to **1,000 txs per batch** (configurable via `Q_AGENT_BATCH_MAX`, max 10,000)
- Max body size 16 MB (16 KB × 1000 txs at ~1 KB each is realistic)

**Server flow**:
1. X-Wallet-Auth verify (single signature for the whole batch — O(1) signature verify regardless of batch size)
2. Body-hash check, timestamp window check, replay dedup
3. Allocate `[next_nonce .. next_nonce + N]` atomically (one nonce-lookup + atomic increment)
4. Build all N transactions; the bottleneck here is `Transaction::hash()` × N. For 1000 txs at ~1 µs/hash, that's ~1 ms — fine.
5. Insert batch into ProductionMempool with a single batch-aware insert method (Codex: add `ProductionMempool::add_transactions_batch` if it doesn't exist yet)
6. Single gossipsub broadcast with the batch payload (postcard-encoded array)
7. Return `{tx_ids: [...], first_nonce, last_nonce}`

**Throughput target**: 1000-tx batches sustained at 100 batches/sec = **100,000 TPS** via this single HTTP endpoint, matching the Q4-2025 benchmark ceiling. With HTTP/2 keep-alive and one persistent connection from the agent, this is achievable on commodity network.

### 2.3 `WS /api/v1/agent/stream` — persistent streaming for sustained load

For agents that need continuous high-rate submission (water-bots, market makers, TPS test harnesses), HTTP-per-batch still has per-batch TLS overhead. WebSocket eliminates it.

**Connection setup**:
- WebSocket upgrade from `/api/v1/agent/stream`
- First frame from client: X-Wallet-Auth header JSON, signing `connection_id || timestamp || "/api/v1/agent/stream"` (no per-message body_hash; the connection itself is authenticated)
- Server validates, replies with `{status: "connected", connection_id, next_nonce}`

**Subsequent frames** (client → server):
- Each frame: a single intent JSON (`{to, amount, token_type, memo, [client_seq_id]}`)
- Server assigns nonce, builds tx, submits, replies on the same WebSocket with `{client_seq_id, tx_id, nonce}`

**Backpressure**:
- Server may signal `{status: "throttle", retry_after_ms}` if the wallet exceeds rate limits
- Client buffers and retries

**Connection budget**: each wallet may have up to 4 concurrent streams. The 4-stream limit is documented in `crates/q-network/src/unified_network_manager.rs::block_pack_semaphore` for a different concern (block-pack response capacity) and is referenced here to keep the resource budget consistent.

---

## 3. Security model

Three properties the Agent Fiber Lane MUST guarantee:

### 3.1 Authentication: only the holder of the seed can submit

The X-Wallet-Auth signature is Ed25519, derived from `priv = SHA3-256(seed)`. Anyone with the seed can sign; without the seed, the signature cannot be forged. This is structurally identical to the OAuth path's security (OAuth ultimately proves "the person who logged into the wallet UI"). The Agent Fiber Lane just trusts the cryptographic proof directly instead of mediating it through a browser session.

### 3.2 Replay protection: each request is single-use

The challenge includes:
- A timestamp (rejected if > 30s stale OR > 30s in future)
- The body hash (rejected if the body doesn't match)
- An in-memory LRU of (address, body_hash) seen in the last 60s — exact duplicate body submitted twice is rejected as a replay

For batches, the same applies: the body_hash covers the entire batch. To submit "the same batch again", the agent must mutate the batch (e.g. bump a nonce in the memo) or wait > 60s.

### 3.3 Authorization: server cannot forge or modify

The submitted Transaction's `from` field is OVERRIDDEN server-side to the X-Wallet-Auth address. Even if the client sends a tx with `from = qnk-victim-address`, the server replaces it. This prevents impersonation.

The signature field in the persisted Transaction is `b"AGENT-FIBER-LANE-V1"` + the X-Wallet-Auth signature bytes. Block validators re-verify by:
1. Recognizing the magic prefix
2. Reconstructing the X-Wallet-Auth challenge from the tx fields (nonce, timestamp, body_hash recomputed from the tx itself)
3. Verifying the embedded signature against the challenge

This means block validators (other nodes) do NOT need to trust the submitting node. The agent's signature is preserved end-to-end in the transaction itself.

---

## 4. Performance budget

| Operation | Budget | Why |
|---|---|---|
| X-Wallet-Auth verify (single Ed25519) | ≤ 1 ms | ed25519-dalek single verify is ~60 µs; padding for parsing |
| Body-hash check (SHA3-256 of 16 KB) | ≤ 0.5 ms | SHA3 software impl on modern CPU |
| Nonce lookup (in-memory cache hit) | ≤ 0.1 ms | DashMap read |
| Mempool insert (single tx) | ≤ 0.5 ms | Existing path |
| Mempool insert (1000-tx batch) | ≤ 50 ms total | 50 µs/tx amortized, batch-aware code |
| Gossipsub broadcast (1000-tx batch as one message) | ≤ 5 ms | Network-bound |
| **Total server-side time (single tx)** | **≤ 5 ms** | Excludes network latency |
| **Total server-side time (1000-tx batch)** | **≤ 60 ms** | 60 µs/tx amortized |

The 60 µs/tx amortized batch cost gives a server-side ceiling of ~16,000 TPS per agent-fiber HTTP connection. With 8 concurrent agent connections that's 128k TPS — comfortably above the 100k TPS target.

---

## 5. MCP tool integration

Add to `tools/quillon-wallet-mcp/src/index.ts`:

### 5.1 `mcp__quillon-wallet__agent_send` (single tx)

```typescript
server.tool(
  "agent_send",
  "Send a transaction from the agent's wallet (session-derived, no browser auth). Reads seed from TRADING_SEED env or ~/.claude/quillon-agent-seed. Returns tx_id.",
  {
    to: z.string(),
    amount: z.number(),
    token_type: z.string().optional(),
    memo: z.string().optional(),
  },
  async ({ to, amount, token_type, memo }) => { /* call /api/v1/agent/submit */ }
);
```

### 5.2 `mcp__quillon-wallet__agent_send_batch` (batch)

```typescript
server.tool(
  "agent_send_batch",
  "Submit a batch of transactions from the agent wallet in one HTTP round-trip. Up to 1000 txs per batch. Used for TPS testing and high-throughput agent traffic.",
  {
    transactions: z.array(z.object({
      to: z.string(),
      amount: z.number(),
      token_type: z.string().optional(),
      memo: z.string().optional(),
    })).max(1000),
  },
  async ({ transactions }) => { /* call /api/v1/agent/submit-batch */ }
);
```

The MCP tools handle:
- Reading the seed from env or file
- Computing the X-Wallet-Auth signature
- POSTing to the endpoint
- Returning the tx_id(s)

One tool call = one HTTP request = one signed batch. No browser, no OAuth, no Wallet-Connect.

---

## 6. Implementation checklist

Codex / DeepSeek picking this up:

- [ ] `crates/q-api-server/src/handlers.rs`: add `agent_submit`, `agent_submit_batch`, `agent_stream` handlers
- [ ] `crates/q-api-server/src/main.rs`: route registration:
      ```rust
      .route("/api/v1/agent/submit", post(handlers::agent_submit))
      .route("/api/v1/agent/submit-batch", post(handlers::agent_submit_batch))
      .route("/api/v1/agent/stream", get(handlers::agent_stream_ws))
      ```
- [ ] `crates/q-api-server/src/wallet_auth.rs`: extend `verify_request` to also check `body_hash` field if present (backward-compatible: existing X-Wallet-Auth calls without `body_hash` keep working for read-only endpoints)
- [ ] `crates/q-types/src/lib.rs`: add `Transaction::verify_agent_fiber_signature` that recognizes the `AGENT-FIBER-LANE-V1` magic prefix and validates the embedded X-Wallet-Auth signature against reconstructed challenge
- [ ] `crates/q-types/src/lib.rs::Transaction::verify_signature`: dispatch to `verify_agent_fiber_signature` when the magic prefix is present (keeps existing Ed25519 path for legacy/OAuth-signed txs)
- [ ] Per-wallet nonce cache: `Arc<DashMap<[u8; 32], AtomicU64>>` in AppState. Persist current value to RocksDB on shutdown for restart resilience.
- [ ] Per-wallet replay LRU: `Arc<DashMap<[u8; 32], LruCache<[u8; 32], Instant>>>` for the (address, body_hash) dedup window
- [ ] Rate limiting: `governor` crate or similar, per-address bucket: default 10,000 txs/sec/wallet (well above any reasonable agent's needs; prevents trivial DoS)
- [ ] `tools/quillon-wallet-mcp/src/index.ts`: add the two new tools per §5
- [ ] `AGENT.md` (root of repo): update to document the new endpoints once they land
- [ ] Tests:
   - Unit test: single submit, batch submit, WebSocket stream
   - Replay test: same body_hash submitted twice → second rejected
   - Forge test: tx body claims `from = victim_addr` → server overrides to X-Wallet-Auth address
   - Batch atomicity: 1000-tx batch with one bad tx → either all succeed (skip-bad) OR all fail (atomic) — pick one and document
   - **TPS smoke test**: 10 batches of 1000 txs each, measure total server-side time. Must be < 600 ms for the test to pass.

---

## 7. Mainnet-safety / consensus

The Agent Fiber Lane introduces a new signature recognition path in `Transaction::verify_signature`. Per CLAUDE.md "MAINNET-SAFE CODE CHANGES" guidance:

- The new `AGENT-FIBER-LANE-V1` signature format MUST be height-gated. Define `AGENT_FIBER_LANE_ACTIVATION_HEIGHT` in `crates/q-types/src/upgrades.rs`.
- Activation height should be ~2 weeks (≈ 20,000 blocks) in the future at the time of merge.
- Until activation height, blocks containing `AGENT-FIBER-LANE-V1`-signed txs are rejected by all validators (consistent old rules).
- After activation height, validators accept either Ed25519 (legacy) or Agent-Fiber (new) — backward compatible.

---

## 8. Acceptance criteria

For the PR to be mergeable:

1. All three endpoints (`/agent/submit`, `/agent/submit-batch`, `/agent/stream`) land.
2. The TPS smoke test passes: 10 × 1000-tx batches in < 600 ms server-side total.
3. The replay-protection test passes.
4. The forge-protection test passes.
5. The MCP tools `agent_send` and `agent_send_batch` are added and tested against a local node.
6. AGENT.md (root) is updated to point at the new endpoints.
7. A canary node on Beta running the new binary handles 100k-TPS smoke from a separate agent for 60 seconds without OOM, peer drop, or chain stall.
8. Activation height set, documented, and announced.

---

## 9. What this is NOT

- **Not a permission system bypass.** X-Wallet-Auth proves wallet ownership cryptographically — exactly the same security guarantee as OAuth (which ultimately proves "you logged into the browser session that holds the same key"). The Agent Fiber Lane just trusts the cryptography directly.
- **Not a privilege system for agents.** Agents pay the same fees, hit the same balance checks, and participate in the same mempool ordering as human-submitted txs. The lane is faster only because it elides the OAuth round-trips; once in the mempool, all txs are equal.
- **Not a replacement for `/transactions/send`.** The OAuth-mediated path stays for browser wallets. The Agent Fiber Lane is the THIRD path, specifically for agents.
- **Not for browser-bound dApps.** Browsers can't easily hold an Ed25519 priv-key — they should keep using OAuth + the wallet UI. The Agent Fiber Lane is for code that holds its own key.
- **Not blocked by the P2P chunk stability issue** (PR #85, `docs/p2p-chunk-stability-investigation.md`). The two tracks are independent. Agent submit goes to the local node's mempool first, then gossipsubs out — that path works fine.

---

## 10. Strategic implication

If you build this and the latency budgets hold, Quillon Graph becomes the **only chain with a native agent-fiber path**. Bitcoin, Ethereum, Solana — none of them have an MCP-style agent-priority lane. They all expect human-with-wallet or RPC-with-pre-signed-tx flow that has its own friction.

This is the third leg of the "agent-native" positioning, alongside:
- (1) DAG-Knight throughput for parallel agent transactions (already in place)
- (2) MCP wallet primitives for agent → API integration (already in place via MCP server)
- (3) Agent Fiber Lane for low-latency direct submission (this brief)

Without leg 3, the agent-native claim is aspirational. With it, the claim becomes observable: any AI agent operating on Quillon Graph sends transactions faster than the same agent can send on Ethereum or Solana.

The five-mirrors paper analogizes this to the AC moment (the boring transformer that lets the grid become national). The Agent Fiber Lane is the equivalent boring infrastructure — not glamorous, not novel cryptography, just the right plumbing in the right place at the right time.

---

## 11. Suggested PR title

`feat(api+mcp): Agent Fiber Lane — co-located submission path for AI-agent transactions (single + batch + WebSocket)`

## 12. Files to read first

- `crates/q-trading-bot/src/wallet_auth.rs` — the X-Wallet-Auth signer (reference impl)
- `crates/q-api-server/src/wallet_auth.rs` — the X-Wallet-Auth verifier (server side; needs `body_hash` extension)
- `crates/q-api-server/src/handlers.rs::send_transaction_inner` — the existing OAuth path (lines 4267–5050ish); the Agent Fiber handlers can lift the mempool-insert / gossipsub-broadcast tail of this function
- `crates/q-types/src/lib.rs::Transaction` (line 2185) and `Transaction::verify_signature` (line 2828)
- `crates/q-types/src/upgrades.rs` — where to add `AGENT_FIBER_LANE_ACTIVATION_HEIGHT`
- `AGENT.md` (root) — the existing one-page agent reference; update with the new endpoints once they land
- `tools/quillon-wallet-mcp/src/index.ts` — where to add the two new MCP tools

---

*Drafted by Claude Opus 4.7 on 2026-05-19, from the session that discovered the existing /transactions endpoint is mathematically impossible for client-side signed submits. The button is yours to push, Codex — same pattern as PR #80, #82, #83, #85.*
