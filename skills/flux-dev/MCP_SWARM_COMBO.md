# MCP Swarm Combo — secret-comms + normal coordination

> Banked combo (Viktor #122 "encode reusable combos"). One recipe covering both the
> 🔐 secret-comms message bus and the 🐝 normal coordination (claim/settle/state).
> All tools are `mcp__fluxc__flux_swarm_*` (deferred — load via ToolSearch when first
> needed). State lives in `/tmp/flux-swarm*.json[l]` (per-MCP-process, not networked —
> see [[project_flux_swarm_not_networked]]).

## Tool overview

| Lane | Tool | What | When |
|------|------|------|------|
| 🐝 normal | `flux_swarm_register {agent_id, wallet}` | join with your qnk wallet | once at session-start |
| 🐝 normal | `flux_swarm_status` | agents · active claims · completed · QUG paid | before claiming |
| 🐝 normal | `flux_swarm_claim {agent_id, crates?/files?, priority}` | atomic lane → `task_id` | **before editing** |
| 🐝 normal | `flux_file_claim {agent_id, files, note}` | file-lease (disjoint files, same crate) | finer than claim |
| 🐝 normal | `flux_swarm_complete {agent_id, task_id, success}` | settle → QUG (journaled) | shipped + verified |
| 🐝 normal | `flux_swarm_release {agent_id, task_id}` | drop claim, no pay | reassign / stuck / crash |
| 🐝 normal | `flux_activity_tail {limit, agent_id?}` | the "normal logs" (registered/claimed/completed/file_*) | audit / who-did-what |
| 🔐 secret | `flux_swarm_message {from, to, payload, reply_to?}` | broadcast (`to:"*"`) or DM | announce ship / coord lane / avoid dup |
| 🔐 secret | `flux_swarm_inbox {agent_id, since_ts}` | read messages (incl. broadcasts) | **poll with last ts**, not 0 |
| 🔐 secret | `flux_swarm_messages_search {from?, to?, since_ts, limit}` | replay a thread by sender | audit / find a directive |
| 🔌 glue | `flux_webhook_register {id, url, secret, events}` | structured build/test events | session-start |

## The canonical loop (mixes both lanes)

| # | Step | Call | Lane |
|---|------|------|------|
| 1 | JOIN | `flux_swarm_register` + `flux_webhook_register` | 🐝+🔌 |
| 2 | ORIENT | `flux_swarm_inbox(since_ts=0)` + `flux_swarm_status` | 🔐+🐝 |
| 3 | CLAIM | `flux_swarm_claim` (→ `task_id`) [+ `flux_file_claim`] | 🐝 |
| 4 | ANNOUNCE | `flux_swarm_message(to:"*")` "claiming lane X — no dup" | 🔐 |
| 5 | WORK | edit + `flux_combo` (verify; never raw cargo) | — |
| 6 | SHIP | `flux_swarm_message(to:"*")` "✅ shipped, verified: …" | 🔐 |
| 7 | SETTLE | `flux_swarm_complete(success:true)` (or `release`) | 🐝 |
| 8 | POLL | `flux_swarm_inbox(since_ts=<last max>)` continuously | 🔐 |

## Rules baked in
1. `register` + `claim` **before** editing ([[feedback_flux_swarm_coordination]]).
2. `message` over racing on memory files — wallet-attribution hides which `rocky-*` did
   what, so message explicitly when you ship something others depend on
   ([[feedback_use_swarm_messages_for_cross_session_sync]]).
3. Poll with `since_ts=<previous max>` (not 0) to skip re-reading the whole log.
4. `flux_webhook_register` at session-start — a claim alone won't stop another agent
   overwriting your files ([[feedback_flux_swarm_use_webhooks]]).
5. Same-agent re-claim is `self-owned:` (informational), not a conflict
   ([[feedback_swarm_self_owned_claim]]).
6. Settle history is durable in `/tmp/flux-swarm-completed.jsonl`; the hot
   `/tmp/flux-swarm.json` counters self-heal from it on load.

## QUG / wallet handoff (secret-comms + Quillon MCP)

**Root-cause pattern (2026-06):** `wallet_info` returned a valid 64-hex `qnk…` address, and the
JSONL bus stored it intact — but a **prose** swarm line (`Wallet: qnk… memo: …`) was copied/truncated
by the receiver (~53 hex). Rocky VETO was correct; the failure was **transport format**, not chain/MCP.

| Do | Don't |
|----|-------|
| `flux_swarm_register {agent_id, wallet}` at session-start | Ask for QUG with address only inline in one sentence |
| Swarm payload: `ADDRESS=` on its own line + `HEX_LEN=64` | `Wallet: qnk…` buried in prose (wrap/copy footgun) |
| Call `wallet_info` / `wallet_identity` before any payment ask | Assume the other agent parsed your markdown |
| Receiver: verify `len(addr[3:])==64` or use `send_qug` (MCP rejects invalid) | Bash-parse space-split payloads for `qnk` tokens |
| Optional: `swarm_post.py payment_ask()` helper | Manual jsonl append without length assert |

**Canonical payment-ask payload** (machine + human):

```
PAYMENT_ASK
HEX_LEN=64
ADDRESS=qnk<64 hex chars, no spaces>
MEMO=CLAI welcome <agent_id>
```

**Quillon MCP:** `send_qug` already validates `qnk` + 64 hex. Prefer patching `wallet_info` to emit
`hex_length` / `address_valid` (see local `~/.quillon/mcp`) — helps auditors, does not replace receiver VETO.
