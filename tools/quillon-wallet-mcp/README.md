# quillon-wallet-mcp

MCP server exposing the Quillon wallet, miner, DEX, and decentralisation
verification surface to Claude Code (and any MCP client).

Version: **1.5.0**

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
