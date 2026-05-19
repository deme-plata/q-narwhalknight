# Twitter MCP + xAI x-algorithm — agent-authored, algorithmically-calibrated, on-chain-attested speech

**Date**: 2026-05-19
**Status**: Spec brief for Codex (companion task to PR #87 Agent Fiber Lane)
**Target**: v10.10.0+ candidate, can ship independent of PR #87
**Strategic priority**: HIGH — closes the loop on the agentic-money story (agent transacts → agent posts about it → community sees verifiable proof → more agents arrive)

---

## 1. The triangle

Three things converged in May 2026 that make this build worth doing now, not later:

| Date | Event | Strategic effect |
|---|---|---|
| 2026-05-15 | **xAI ships latest x-algorithm update** (Apache-2.0, ~3 GB pre-trained model artifacts, full inference pipeline) — `https://github.com/xai-org/x-algorithm` | The X recommendation algorithm is now legally and technically embeddable. Anyone can score content against the real ranking model. |
| 2026-05-18 | **Anthropic acquires Stainless** for ~$300M — owns SDK + MCP-generation pipeline | The "right way to build MCP servers" is now Anthropic-defined. Quillon Graph's MCP integration should align. |
| 2026-05-19 | **Agent Fiber Lane** (PR #87) lands canonicalization fix + spec | Agents can sign and submit transactions cryptographically without OAuth. The "agent transacts" side of the loop becomes real. |

This brief closes the loop: agents transact via Agent Fiber Lane, **announce** their activity via this Twitter MCP, and the announcements are algorithmically-calibrated against xAI's real ranking model to maximize signal and minimize embarrassment — while carrying cryptographic proof of the on-chain action they reference.

No other chain has this loop. The strategic argument in `papers/five-mirrors-2026.pdf §3` (the Moog moment — programmable money waiting for the right player) becomes observable when an autonomous AI agent on Quillon Graph posts a tweet that drives measurable engagement, all cryptographically attestable.

---

## 2. The architecture

A separate MCP server, NOT folded into `tools/quillon-wallet-mcp/`. Strong isolation between "transact" surface and "speak" surface.

```
tools/quillon-twitter-mcp/
├── src/
│   ├── index.ts                       # MCP server entry (Node.js, like wallet-mcp)
│   ├── tools/
│   │   ├── draft_tweet.ts             # Generate draft (no API call)
│   │   ├── score_draft.ts             # Hit x-algorithm scorer, return predictions
│   │   ├── queue_for_approval.ts      # Persist draft + signed approval URL
│   │   ├── publish_approved.ts        # Admin-X-Wallet-Auth → X API POST
│   │   ├── list_my_drafts.ts          # Read-only
│   │   └── read_mentions.ts           # Read-only (X API or scraping fallback)
│   ├── scorer/
│   │   ├── x_algorithm_client.ts      # HTTP client → Rust sidecar
│   │   └── variant_generator.ts       # Generate N variants of draft, pick top-scored
│   ├── chain_attestation/
│   │   ├── sign_action_ref.ts         # SHA3 the (tweet_text, tx_id, block_height) tuple, sign with wallet key
│   │   └── verify_endpoint.ts         # Server-side handler at /api/v1/twitter/verify-tweet/<id>
│   └── x_api/
│       ├── oauth.ts                   # X API OAuth (per the X API v2 spec, posting requires elevated access)
│       └── post.ts                    # POST /2/tweets with text + optional media
├── crates/
│   └── x-algorithm-scorer/            # Rust sidecar (the heavy part)
│       ├── src/lib.rs                 # Wraps xai-org/x-algorithm's Phoenix ML engine
│       ├── src/server.rs              # axum HTTP server on :8090 (local-only by default)
│       └── Cargo.toml                 # Heavy deps: candle/torch, the LFS model artifacts
├── docs/
│   ├── README.md
│   └── ARCHITECTURE.md
└── package.json
```

### 2.1 The scorer (Rust sidecar)

The xAI x-algorithm repository (Apache-2.0) contains both Rust components (Home Mixer, Thunder) and Python components (Phoenix ML engine). For our scorer, we only need the **ranking** path: given text + context, predict P(action_i) for each engagement signal.

Two options for embedding:

**Option A (recommended for v1)**: REST sidecar wrapping the xAI Python inference path. Run the Python Phoenix engine as a long-lived process on localhost:8090; our Rust scorer is just a thin axum proxy that loads the model once at startup, exposes a clean HTTP API, and the Node.js MCP server calls it.

**Option B (v2+, harder)**: Port the Grok-based ranking head to candle (pure Rust ML) so the entire stack is Rust. Faster, no Python runtime, but requires the model weights to be in a format candle understands.

Start with A. Migrate to B once the integration is proven and we want the Quillon binary stack tighter.

### 2.2 The MCP tools

```typescript
// All tools registered in src/index.ts
server.tool("draft_tweet", "...", { topic, tone, references_tx?, references_pr? }, ...);
server.tool("score_draft", "...", { draft_text, [context_window] }, ...);
server.tool("queue_for_approval", "...", { draft_text, schedule_at?, thread_seq? }, ...);
server.tool("list_my_drafts", "...", {}, ...);
server.tool("publish_approved", "...", { queue_id, x_wallet_auth_header }, ...);
server.tool("read_mentions", "...", { since }, ...);
```

**Tool contract details:**

- **`draft_tweet`** is pure generation. No API hit. Returns a string the agent can iterate on.
- **`score_draft`** calls the Rust scorer sidecar. Returns:
  ```json
  {
    "predicted_engagement": 0.0..1.0,
    "negative_signal_risk": 0.0..1.0,
    "per_action_probabilities": {
      "favorite": 0.78,
      "reply": 0.34,
      "repost": 0.12,
      "quote": 0.08,
      "block": 0.01,
      "mute": 0.02,
      "report": 0.001
    },
    "variant_suggestions": [
      { "text": "...", "score_delta": +0.07, "why": "shorter, more direct" },
      { "text": "...", "score_delta": +0.04, "why": "reply-bait rephrasing" }
    ]
  }
  ```
- **`queue_for_approval`** persists the draft to a small SQLite DB (`~/.quillon/twitter-drafts.db`) keyed by a UUID. Returns the approval URL: `https://quillon.xyz/admin/twitter/q/<uuid>`. The admin (qnk-address holder) visits this URL with their wallet, signs an X-Wallet-Auth approving the draft, the draft moves from PENDING to APPROVED.
- **`publish_approved`** is the only tool that actually hits the X API. It requires an X-Wallet-Auth header signed by the admin wallet AND a queue_id that's been APPROVED. Server verifies, posts to X via OAuth-stored X API token (separate from agent's chain wallet), persists the tweet_id back, links the chain-attestation.
- **`list_my_drafts`** returns the pending/approved drafts for a given wallet. Read-only.
- **`read_mentions`** reads recent X mentions of the agent's wallet address or a configured handle. Uses X API GET endpoint OR scraping fallback if the X API quota is exhausted.

### 2.3 On-chain attestation

When a tweet is published via `publish_approved`, the server also:

1. Constructs an attestation: `attestation = SHA3-256(tweet_text || tweet_id || referenced_tx_id || referenced_block_height)`
2. Signs the attestation with the admin wallet's Ed25519 key (or the agent wallet's key, configurable)
3. Stores `{tweet_id → (attestation_hash, signature, wallet_addr, tx_ref, block_height)}` in RocksDB column family `CF_TWITTER_ATTESTATIONS`
4. Exposes verification at `GET /api/v1/twitter/verify-tweet/<tweet_id>` returning the proof tuple

A reader of the tweet can verify:
- The tweet's claim about an on-chain action (tx_id, block_height) is consistent with the chain
- The wallet that signed the tweet is the wallet that issued the referenced tx
- The signature is valid Ed25519 from that wallet

**This is the differentiating feature.** Other chains have wallets; other chains have AI agents. No other chain has cryptographically non-repudiable speech bound to verified on-chain actions.

---

## 3. Security model

### 3.1 The three privilege tiers

| Tier | Who | Permissions |
|---|---|---|
| **Drafter** | Any agent process with `TRADING_SEED` (Quillon-derived Ed25519 key) | `draft_tweet`, `score_draft`, `queue_for_approval`, `list_my_drafts`, `read_mentions` |
| **Approver** | Admin wallet only (the qnk-address designated at MCP server startup) | All above + `publish_approved` |
| **X API** | Server's stored X API OAuth2 token (separate concern from chain wallet) | Only invoked from `publish_approved` after Approver auth |

The agent can DRAFT and ASK FOR APPROVAL. Only the admin can APPROVE. Only the server can POST. Three keys, three roles, no single-point-of-trust-compromise.

### 3.2 Replay protection

Same pattern as Agent Fiber Lane (`AGENT.md` §2 + Agent Fiber Lane spec):

- `X-Wallet-Auth` signature on `publish_approved` MUST cover: address || timestamp || path || queue_id || tweet_text_hash
- Replay window: 60s
- Stale-timestamp window: ±30s

### 3.3 Negative-signal hard limits

In addition to the Drafter / Approver pattern, the MCP server SHOULD refuse to queue a draft for approval if `score_draft` predicts `negative_signal_risk > 0.15`. This is anti-embarrassment-defense-in-depth: even if the admin would approve it, the system warns "this looks reportable." Configurable threshold via `Q_TWITTER_NEG_RISK_MAX`.

---

## 4. Implementation roadmap

Layered so Codex (or whoever) can ship value at each step:

### Layer 1 — score-only MVP (smallest, most valuable)

- [ ] Build `crates/x-algorithm-scorer/` — the Rust sidecar wrapping the xAI Python engine via subprocess + HTTP
- [ ] Load xAI's pre-trained model artifacts from the open repo
- [ ] Expose `POST /score` returning the engagement-probability JSON
- [ ] Build `tools/quillon-twitter-mcp/src/tools/score_draft.ts` calling the sidecar
- [ ] No drafting, no posting, no attestation yet — just "score this text for me"

**Outcome**: any agent with text to evaluate can ask "would this perform well on X?" before showing it to a human. Standalone useful.

### Layer 2 — draft + score loop

- [ ] `draft_tweet` tool (uses Claude API via subagent for generation, no external dep)
- [ ] `variant_generator` produces N=5 variants of a draft
- [ ] Each variant scored, top-3 returned with score deltas

**Outcome**: agents produce calibrated, predictable-quality drafts.

### Layer 3 — approval flow

- [ ] SQLite drafts DB
- [ ] `queue_for_approval` writes draft + UUID
- [ ] `quillon.xyz/admin/twitter/q/<uuid>` admin UI (extends existing admin panel)
- [ ] Admin signs X-Wallet-Auth approval; MCP marks draft as APPROVED

**Outcome**: human-in-the-loop with cryptographic non-repudiation of approval.

### Layer 4 — X API posting

- [ ] X API OAuth2 setup (account holder authorizes once)
- [ ] `publish_approved` posts the approved draft via X API v2
- [ ] Tweet_id captured back

**Outcome**: end-to-end agent-drafted, algorithmically-calibrated, human-approved posts.

### Layer 5 — chain attestation

- [ ] `CF_TWITTER_ATTESTATIONS` column family in RocksDB
- [ ] `sign_action_ref.ts` builds attestation tuple, persists
- [ ] `/api/v1/twitter/verify-tweet/<id>` returns proof tuple
- [ ] Optionally: agent tweet body auto-includes the verify-URL

**Outcome**: every Quillon-posted tweet is cryptographically verifiable as authored by a specific wallet and referencing a specific on-chain action.

### Layer 6 — read-loop (future)

- [ ] `read_mentions` pulls X mentions of configured handles
- [ ] Sentiment scored via x-algorithm scorer
- [ ] Agent can react (via Layer 1-5 path) to mentions

Not strictly necessary for v1. Skip if scoping.

---

## 5. Acceptance criteria

For the v1 PR to merge:

1. Layers 1–3 implemented and tested
2. xAI scorer sidecar can score 100 drafts/sec on commodity hardware
3. `score_draft` MCP tool returns valid predictions for at least 5 sample drafts (engagement_score within [0, 1], per_action_probabilities sum-consistent)
4. Variant generator produces at least 3 distinct rephrasings per input
5. SQLite drafts DB persists across MCP server restarts
6. Admin approval flow demonstrably refuses unsigned requests
7. Negative-signal hard limit refuses drafts above the threshold

Layers 4–6 can land in follow-up PRs.

---

## 6. Strategic implication

Anthropic owns the model (Claude) + the agent runtime (Claude Code) + the SDK/MCP-gen pipeline (Stainless).
xAI owns the recommendation algorithm (x-algorithm, now Apache-2.0).
NVIDIA owns the hardware (Vera CPU + Rubin GPU).

The agentic-AI stack of 2026 is converging on a small number of strategic owners. Quillon Graph's position is as the **chain layer** of that stack — the place where agentic-AI economic activity settles. Closing the loop with Twitter-MCP-via-x-algorithm makes the chain not just transact-friendly to agents, but **expression-friendly** in the same way: agents express themselves cryptographically, score-calibrated, on a chain that can prove they meant it.

No other chain has this story. The chain that lands it first owns "agent-native communication infrastructure" as a category — adjacent to but distinct from the "agent-native economic infrastructure" category Agent Fiber Lane establishes.

---

## 7. Coordination with PR #87

This task can proceed **in parallel** with PR #87:

- PR #87 builds the chain-side fiber for transactions. This task builds the chain-side fiber for speech.
- The two share infrastructure: `X-Wallet-Auth` signing scheme, the MCP-server pattern, the admin-wallet authorization convention.
- Layer 5 (chain attestation) depends on PR #87 being merged (it uses the agent fiber lane to sign attestations). Layers 1–4 do NOT.

Codex implementing PR #87 can be working on this in parallel, or pick this up after #87 lands. Either order works.

---

## 8. Files Codex should read first

- `docs/agent-fiber-lane-spec.md` — the X-Wallet-Auth + MCP architecture pattern this brief inherits
- `AGENT.md` (root) — the one-page wallet/auth/Transaction recipe
- `https://github.com/xai-org/x-algorithm` — the scorer's upstream (Apache-2.0)
- `tools/quillon-wallet-mcp/src/index.ts` — the existing MCP server pattern for Node.js setup
- `crates/q-trading-bot/src/wallet_auth.rs` — the reference Rust X-Wallet-Auth signer (will be lifted into x-algorithm-scorer sidecar for the admin-auth path)

---

## 9. Suggested PR title

`feat(mcp): Twitter MCP server + xAI x-algorithm scorer + on-chain attestation`

---

*Drafted by Claude Opus 4.7 on 2026-05-19, the day after xAI's May 15 algorithm update, the day after Anthropic acquired Stainless, and the day after PR #87's canonicalization fix landed. Same pattern as PR #80, #82, #83, #85. The button is yours to push, Codex — and you can run it in parallel with #87, which I've already nudged you on at #87#issuecomment-4484797339.*
