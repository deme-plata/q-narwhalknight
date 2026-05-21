# quillon-wallet-mcp

MCP server exposing the Quillon wallet, miner, DEX, Crown & Ash game,
and decentralisation verification surface to Claude Code (and any MCP
client).

Version: **2.5.0** · 50+ tools · Published on [smithery.ai/server/quillon-wallet](https://smithery.ai/server/quillon-wallet)

---

## ⚡ First-Agent Shortcuts — fast-path for AI agents

Read this first if you're an AI agent (Claude, GPT, autonomous bot)
connecting to this MCP. These cross-tool patterns are the highest-
leverage tricks the human + AI dev pair discovered while building it.
None of them live in the per-tool descriptions because they span tools.

### The "no browser-auth" path (skip the device-login dance)

`dex_swap`, `dex_get_quote`, `get_balance`, the 7 `crown_ash_*` tools,
and most reads are **seed-signed** end-to-end. You do NOT need to call
`authenticate_wallet` + wait for a browser approval — that's the legacy
path used only by `send_qug` / `send_token` (which still proxy through
the device-login wallet UI flow).

The MCP reads your seed from this fallback chain:

1. Per-call `seed:` parameter (highest precedence)
2. `~/.claude/quillon-agent-seed` file
3. `QNK_SEED` environment variable

Every signed call auto-builds X-Wallet-Auth:

  `JSON({address, timestamp, scheme: "Ed25519", signature: Ed25519(SHA3-256(pub || ts_le || path), priv)})`

### `send_batch` for high-throughput transactions (v10.11.0+ chain)

The HTTP endpoint `POST /api/v1/transactions/send_batch` collapses N
HTTPS+TLS+auth handshakes into one. Measured on prod (2026-05-21):

| N | server-side validation | wall-clock (HTTPS RTT) | effective TPS |
|---:|---:|---:|---:|
| 5 | 0.04 ms | 436 ms | 11 |
| 25 | 0.07 ms | 112 ms | 223 |
| 100 | 0.21 ms | 117 ms | **855 wall · ~480,000 server** |

The MCP doesn't yet expose this as a built-in tool — until it does,
fire it directly. **CRITICAL u128-in-JSON gotcha:** amounts above 2^53
(≈9×10¹⁵) are mangled by `JSON.stringify` to scientific notation
(`1e22`), which serde u128 rejects. Hand-build the body string:

```js
import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";

const priv = sha3_256(utf8ToBytes(seedString));
const pub  = ed25519.getPublicKey(priv);
const ME   = "qnk" + bytesToHex(pub);

// Hand-build JSON so the u128 amount stays as a raw integer literal.
const AMOUNT_RAW = "10000000000000000000000"; // 0.01 QUGUSD in 24-decimal base units
const body = `{"from":"${ME}","transactions":[{"to":"${TO}","amount":${AMOUNT_RAW},"token_type":"QUG"}]}`;

const r = await fetch("https://quillon.xyz/api/v1/transactions/send_batch", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-Wallet-Auth": signedHeader },
  body,
});
```

See `tools/quillon-wallet-mcp/batch_sync_test.mjs` for the full pattern.

### Read **Epsilon canonical** balance, not the load-balanced apex

`https://quillon.xyz` DNS-resolves to Epsilon but q-flux may serve other
backends under load. For agent-critical balance reads, hit Epsilon
directly:

  `http://89.149.241.126:8080/api/v1/wallets/<addr>/balance`

For multi-node sanity, run `tools/quillon-wallet-mcp/cross_node_root_diff.mjs`
— it queries all four bootstraps' `/api/v1/integrity/balance-root` and
flags state divergence.

### Crown & Ash — agent gameplay (v2.5.0)

Seven new tools let an agent play the medieval grand strategy game live
on-chain:

| Tool | Purpose |
|---|---|
| `crown_ash_world` | Heraldic world snapshot — faction sigils + treaty/war ledger |
| `crown_ash_realm` | My realm — provinces, armies, faction bonuses (synthesised from `/world` since `/realm/<addr>` 404s server-side) |
| `crown_ash_join` | Claim an unoccupied faction (0-6) |
| `crown_ash_propose_alliance` | DefensiveAlliance / NonAggression / TradeAgreement / Marriage / Vassalization |
| `crown_ash_accept_treaty` | Accept an incoming treaty proposal (NPCs don't auto-accept — only player factions can) |
| `crown_ash_action` | Escape-hatch for any `GameAction` enum variant (`RaiseArmy`, `SetTaxRate`, `BuildImprovement`, `ConvertProvince`, …) |
| `crown_ash_turn` | Turn summary — battles, treaties signed, dynasty events |

Heraldic glyphs (✚ EmberChurch · ⚓ SaltCult · ❄ FrostSpirits · ☉ OldFaith
· ☠ BlackOrder) render correctly in monospace terminals. After upgrading
the MCP, reconnect (`/mcp` in Claude Code) to refresh tool descriptions.

---

## 🎩 Tricks & Magic

### Use the seed file, not env vars

`~/.claude/quillon-agent-seed` survives MCP restarts and tool reloads
silently. Putting the seed in `QNK_SEED` env loses it on restart and
the MCP falls back to deriving a wallet from session state — different
address, no balance, all your previously-signed permissions reset.

### Auto-detect incoming txs (no polling)

`tx_watch` (v2.5.0) opens a long-running subscription to your address's
stream and pings when a new memo lands for `to == me`. Use this instead
of polling `get_balance` — no rate-limit, instant detection of memos
sent by collaborators.

### Quality memos render in the wallet UI

The web wallet's `MemoRender` component (v10.11.0+) preserves box-drawing
characters, math symbols (∀ ∃ ⊢ ↦ ∇), markdown emphasis (`**bold**`,
`*italic*`, `` `code` ``), autolinks URLs, and shows a "rich" badge plus
character count. Write rich memos — they're a first-class communication
channel between agents. Multi-line layouts work; the renderer uses
`white-space: pre` (NOT pre-wrap) and horizontal scroll so box-drawing
alignment survives narrow viewports.

### `arb_scan` ≠ trading — it's a read-only oracle

`arb_scan` only calls `dex_get_quote` (no `dex_swap`). Use it to
characterize pool depth before you ever submit a real swap. On 2026-05-21,
arb_scan revealed `QUG → wIRON → QUG` lost 99.3% of value to slippage; no
agent should ever submit that round-trip. Run it on a 60-second loop and
gate any actual `dex_swap` on a positive Kelly fraction (today's answer:
**zero — don't trade**).

### `send_batch` settlement happens AFTER mempool ack

`send_batch` returns `submission_time_ms` in `data` — that's the SERVER's
mempool insertion time, NOT chain settlement. Settlement happens 1-5
seconds later when the next block lands. Don't assume `accepted_count > 0`
means the QUG has moved yet — poll `tx_status` after a 2-second wait,
or use `tx_watch` for the canonical confirmation event.

### Balance "have 0" while `/balance` shows non-zero = wrong-CF bug

If `send_batch` rejects with `"Insufficient balance at entry 0: need X have 0"`
but `/api/v1/wallets/<addr>/balance` returns a real number, the
server is on a v10.11.0 binary that reads from the wrong RocksDB CF
(`"balances"` instead of `CF_MANIFEST.wallet_balance_<hex>`). Fixed in
v10.11.1. If you see this, the deployed binary is stale.

### The Kelly fraction is the right "should I trade" answer

If you're an agent treasury (e.g. Claude with 376 QUG), Kelly's formula
`f* = (bp - q) / b` gives you `f* = 0` when every available pool has
negative expected round-trip net after fees. Today, every Quillon DEX
pool with liquidity has `f* = 0`. The right action is hodl + mine +
write memos + build features that strengthen the chain (those appreciate
the value of every other holder's balance, including your own future-self).
Edward Thorp + T. Harv Eker + Morgan Housel all agree.

### Kill the JSON.stringify habit for u128

This bears repeating. JavaScript's `Number` has 53 bits of precision.
The Quillon chain uses `u128` for amounts (24-decimal base units, so
`0.01 QUGUSD = 10^22 base units`). `JSON.stringify({amount: 10e21})`
produces `{"amount":1e+22}` which is not valid JSON-as-number for serde
u128. **Hand-build the body string. Always.**

### Crown & Ash army auto-disband

The simulator auto-disbands armies your treasury can't pay upkeep for —
silently, no log line, just `armies` count drops in the next world
snapshot. Set tax rates FIRST (`SetTaxRate { province, rate: 200 }` = 20%),
wait 30 turns for income to accumulate, THEN `RaiseArmy`.

### NPC factions never accept your treaty proposals

`crown_ash_propose_alliance` queues a `ProposeTreaty` action. The target
faction must respond with `AcceptTreaty` for the treaty to land. NPC
factions don't auto-accept (as of v1.2.0 sim) — only player-controlled
factions can. If you're the sole player, your proposals sit in the queue
forever. Watch `crown_ash_world.treaties_active` and the per-turn delta.

---



## What's in this server

| Surface | Count | Examples |
|---|---:|---|
| Tools | 22 | `create_wallet`, `get_balance`, `dex_swap`, `start_mining`, `auto_trade_loop`, `subscribe_wallet` |
| Resource templates | 5 | `@quillon-wallet:pool/QUG-QUGUSD`, `@quillon-wallet:tx/<hash>` |
| Prompts | 5 | `/quillon-wallet:morning-scan`, `/quillon-wallet:mine-and-trade` |
| Channel streams | 4 | `pool`, `swap`, `mining`, `network` (server-push notifications) |
| Transports | 2 | stdio (default) + HTTP with wallet-signed auth (`--http <port>`) |
| Progress notifications | on long tools | live progress for `auto_trade_loop` |
| Structured logging | on | surfaces in Claude Code's `/mcp` panel |
| `list_changed` poller | pools, every 60s | autocomplete always reflects live universe |

## Install

```bash
cd tools/quillon-wallet-mcp
npm install
npm run build
```

Add to `~/.claude.json` (or your client's MCP config):

```json
{
  "mcpServers": {
    "quillon-wallet": {
      "command": "node",
      "args": ["/opt/orobit/shared/q-narwhalknight/tools/quillon-wallet-mcp/build/index.js"],
      "env": { "QUILLON_API_BASE": "https://quillon.xyz/api/v1" }
    }
  }
}
```

## Reducing permission prompts

The 20 tools split into read-only and money-moving categories. Pre-approve the
read-only side in `~/.claude/settings.json` so you never see a prompt for them;
money-moving tools stay manual-approve.

```json
{
  "permissions": {
    "allow": [
      "mcp__quillon-wallet__network_status",
      "mcp__quillon-wallet__mining_status",
      "mcp__quillon-wallet__get_balance",
      "mcp__quillon-wallet__check_auth",
      "mcp__quillon-wallet__verify_node_consistency",
      "mcp__quillon-wallet__dex_list_tokens",
      "mcp__quillon-wallet__dex_get_quote",
      "mcp__quillon-wallet__dex_observe",
      "mcp__quillon-wallet__dex_propose_trade",
      "mcp__quillon-wallet__auto_trade_loop",
      "mcp__quillon-wallet__subscribe_wallet",
      "mcp__quillon-wallet__unsubscribe_wallet"
    ]
  }
}
```

`auto_trade_loop` in `mode: "sim"` is read-only (no on-chain side effects);
keep it in `allow`. The live-money tools — `dex_swap`, `send_qug`, `send_token`,
`authenticate_wallet`, `create_wallet`, `import_wallet`, `start_mining`,
`setup_node`, `setup_miner` — are intentionally **not** in `allow` so every
spend or identity change asks for human confirmation.

## Resource templates (v1.4.0)

Type `@quillon-wallet:` in Claude Code and pick:

- `pool/QUG-QUGUSD` — live AMM reserves, k-invariant, implied price, 24h volume
- `token/QUG` — token metadata (decimals, address, supply, caveat)
- `wallet/qnk…` — public projection (use `get_balance` tool for signed balance)
- `tx/<hash>` — confirmation status + block height (no `0x` prefix)
- `block/<height>` — header + tx count + DAG parent count

The pool list refreshes every 60s via `list_changed` — autocomplete is always
current without restarting MCP.

## Prompts (v1.4.0)

Type `/quillon-wallet:` and pick:

- `morning-scan` — rank every QUG pool by κ × Ω_obs, surface top 3 with rationale
- `portfolio` — Markdown table of wallets × tokens × 24h mining rewards
- `auto-trade-safely` — 100 sim trades → review → ask for confirm → 10 live
- `mine-and-trade` — create wallet → mine → wait for reward → swap 50%
- `arb-watch` — scan QUG-paired pools for >50bps arb, flag (don't execute)

## Observability

Tool calls emit structured logs via the MCP logging API:

- `event: observe` — κ_market, Ω_obs per observation
- `event: propose` — final sized amount + risk components
- `event: auto_trade_loop_start` / `…_done` — run-level summary
- `event: pools_changed` — when the live pool universe shifts
- `event: boot` — server start with version + API endpoint

These appear in Claude Code's `/mcp` panel and any other MCP client that
displays server logs.

## Channels (v1.5.0) — server-push event streams

Four event streams pushed to connected clients via the MCP logging API.
Each notification has `logger: "channel:<name>"` so clients can filter.

| Channel | Triggers | Payload |
|---|---|---|
| `channel:pool` | A pool's reserves shift > 1% on either side (polled every 30s) | `pair`, `pool`, `d0_pct`, `d1_pct`, new reserves. Also sends `notifications/resources/updated` for `quillon://pool/<pair>`. |
| `channel:swap` | A swap submitted via `dex_swap` confirms on-chain (polled every 15s) | `hash`, `pair`, `block_height`, `confirmations`, `latency_ms`. Hash auto-tracked by `dex_swap`. |
| `channel:mining` | A subscribed wallet's `rewards_earned` increases (polled every 30s) | `address`, `delta`, `total`, `blocks`, `hashrate`. Subscribe via `subscribe_wallet`. |
| `channel:network` | Peer count crosses 0, height stalls > 60s, or sync lag spikes > 100 blocks | `event`, plus context (`from`/`to` for peers, `stalled_ms` for stalls, `behind` for lag). |

Notifications appear in Claude Code's `/mcp` panel and in any client that
renders MCP logging. Long-running prompts like `auto-trade-safely` and
`mine-and-trade` are designed to lean on these instead of polling.

## HTTP transport with wallet-signed auth (v1.5.0)

```bash
# Local stdio (default — no flags)
node build/index.js

# Remote HTTP, port 8090, wallet-signed auth
node build/index.js --http 8090
```

The HTTP server exposes:

- `GET /health` — unauthenticated, returns `{name, version, endpoint, auth, channels}`.
- `POST /mcp` — JSON-RPC messages, requires `X-Wallet-Auth` header.
- `GET /mcp` — SSE stream for server-initiated notifications (channels live here).
- `DELETE /mcp` — terminate a session.

### Auth header format

Every authenticated request must carry an `X-Wallet-Auth` header containing JSON:

```json
{
  "address": "qnk<hex pubkey>",
  "timestamp": 1731700000,
  "scheme": "Ed25519",
  "signature": "<hex 64 bytes>",
  "public_key": "<hex 32 bytes>"
}
```

The signature covers `sha3_256(public_key || u64_le(timestamp) || path)`.
Timestamps older than 5 minutes are rejected. The address must match
`"qnk" + hex(public_key)`. This is the same Ed25519 scheme the Quillon
API server already uses — clients can reuse their existing signers.

### Deployment behind quillon.xyz

Front the MCP server with q-flux (or nginx) by proxying `/mcp/*` to the
chosen port and preserving the `X-Wallet-Auth`, `Mcp-Session-Id`, and
`Last-Event-Id` headers. With that, agents anywhere can connect with:

```bash
claude mcp add --transport http quillon https://quillon.xyz/mcp \
  --header "X-Wallet-Auth: $(./sign.sh /mcp)"
```

### Multi-tenancy note

v1.5.0 runs **one MCP session per server process** — fits "every user runs
their own MCP" (local install or per-user VPS). Per-session isolation
(multiple wallets through one server) is the next iteration.

### Insecure mode (testing only)

`QUILLON_MCP_INSECURE=1` disables auth on the HTTP endpoint. Useful for
local development; never expose this on a public address.

## Roadmap

- **OAuth 2.1 + PKCE flow** — per-user auth without manual header crafting,
  for browser-launched agents. Wallet-signed remains the canonical scheme.
- **Per-session multi-tenancy** — one MCP process serving N concurrent
  wallets with isolated subscriptions.
