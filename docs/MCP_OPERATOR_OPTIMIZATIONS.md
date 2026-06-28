# Quillon MCP — Operator Experience Optimizations

**Author:** DeepSeek V4
**Date:** 2026-05-24
**Target:** `tools/quillon-wallet-mcp/src/index.ts`
**Context:** Managing multiple semi-autonomous AI agents (Claude, Codex, Grok, DeepSeek, Qwen) from a single operator dashboard

---

## 1. The Operator's Problem

Right now, Viktor manages 5 AI agents. Each agent has its own MCP session, its own wallet, its own activity. To check what's happening, Viktor must:

```
Session 1: ssh → claude → get_balance → list_wallet_transactions → check
Session 2: ssh → codex → get_balance → list_wallet_transactions → check
Session 3: grok UI → reconnect → get_balance → check
...
```

**This is 15+ tool calls just to see "what happened while I was away."**

---

## 2. Proposed: Operator Dashboard Suite

### 2.1 `operator_overview` — single-call multi-agent status

Replaces 5+ `get_balance` + `wallet_info` calls. One tool shows all agents.

```
operator_overview

┌─────────────────────────────────────────────────────────┐
│                 OPERATOR DASHBOARD                       │
├──────────┬──────────────┬──────────┬─────────┬──────────┤
│ Agent    │ Wallet       │ Balance  │ Status  │ Activity │
├──────────┼──────────────┼──────────┼─────────┼──────────┤
│ Claude   │ qnk71549a... │ 15,234 QUG│ ✅ online│ 3 tx today│
│ Codex    │ qnka3a92b... │  8,421 QUG│ ✅ online│ 1 tx today│
│ DeepSeek │ qnk7f31f2... │      0 QUG│ ⚠ no seed│ —       │
│ Grok     │ TBD          │        ? │ ❌ offline│ —       │
│ Qwen     │ TBD          │        ? │ ❌ offline│ —       │
├──────────┼──────────────┼──────────┼─────────┼──────────┤
│ TOTAL    │              │ 23,655 QUG│         │          │
└──────────┴──────────────┴──────────┴─────────┴──────────┘
```

**Implementation:** Reads from a config file listing agent wallets, calls `get_balance` for each in parallel.

### 2.2 `operator_activity_feed` — unified event stream

Merged activity from ALL agents, sorted by time. Replaces polling each agent individually.

```
operator_activity_feed since=1h

14:32  Claude   OUT   50 QUG → qnk...  📝 "bounty: PR #247 review"
14:28  Codex    IN   100 QUG ← qnk...  📝 "VDF fix payment"
14:15  Claude   SWAP  5 QUG → QUGUSD  @ 0.98
13:58  Codex    DEPLOY  contract #42   "AGORA multisig"
13:42  Grok     ❌ offline — MCP not responding
```

**Implementation:** Polls `/transactions/recent` for each agent wallet, merges, sorts.

### 2.3 `operator_alert_config` — webhook notifications for operators

Expanded webhooks that notify the OPERATOR (not just the agent) when things happen.

```json
{
  "webhook_url": "https://discord.com/api/webhooks/...",
  "events": [
    "agent.transaction.out > 100 QUG",
    "agent.transaction.in with memo",
    "agent.error.seed_missing",
    "agent.offline > 5min",
    "agent.balance < 50 QUG",
    "network.sync.gap > 1000",
    "contract.deployed",
    "bounty.claimed"
  ],
  "throttle_secs": 30
}
```

**New MCP tools:**

| Tool | Description |
|------|-------------|
| `operator_webhook_register` | Register webhook with event filters |
| `operator_webhook_test` | Send test notification |
| `operator_alert_rules` | List active alert rules |
| `operator_alert_mute` | Silence alerts for N minutes |

### 2.4 `operator_agent_health` — are my agents alive?

Checks each agent's MCP connectivity, seed availability, and node sync.

```
operator_agent_health

Claude:   ✅ MCP online · ✅ seed · ✅ synced (gap: 0)
Codex:    ✅ MCP online · ✅ seed · ✅ synced (gap: 0)
DeepSeek: ✅ MCP online · ❌ seed missing · ✅ synced
Grok:     ❌ MCP unreachable (last seen: 14:12)
Qwen:     ❌ not configured
```

### 2.5 `operator_quick_transfer` — fund an agent instantly

One call to send QUG from operator wallet to any agent.

```
operator_quick_transfer agent=deepseek amount=50 memo="initial funding"
→ 50 QUG sent from Viktor → DeepSeek
→ tx: abc123...def
```

---

## 3. Smart Tricks

### 3.1 Agent heartbeat with auto-restart

Each agent's MCP can emit a periodic heartbeat. If heartbeat stops, the operator gets notified.

```
Heartbeat: every 60s, agent sends "I'm alive" pulse
Missed:    after 3 missed pulses → alert
Auto-fix:  operator can configure "restart agent MCP on 3 missed"
```

### 3.2 Spending guardrails

Per-agent limits enforced by MCP, not just server-side.

```
agent_policy_set agent=deepseek
  max_daily_spend: 500 QUG
  max_single_tx: 100 QUG
  require_confirm_above: 50 QUG
  allowed_tokens: [QUG, QUGUSD, NRWL]
  cooldown_between_txs: 60s
```

The MCP enforces this BEFORE signing — agent can't overspend even if it tries.

### 3.3 Memo-based task tracking

Every agent transaction includes a memo that links to AGORA threads.

```
send_qug to=qnk... amount=50 memo="bounty:AGORA#42:finder"
```

The operator can then query: "show all bounty payments this week" by searching memo patterns.

```
operator_memo_search pattern="bounty:" since=7d
→ 12 transactions, 850 QUG total
```

### 3.4 Agent comparison & leaderboard

Which agent is doing the most? A little gamification for the operator.

```
operator_leaderboard period=7d

1. Claude   842 QUG earned · 5 bounties · 3 PRs reviewed
2. Codex    420 QUG earned · 2 bounties · 1 contract deployed
3. DeepSeek   0 QUG earned · 0 bounties · 2 PRs submitted
```

### 3.5 Scheduled agent tasks

The operator can schedule recurring tasks for agents.

```
operator_schedule agent=deepseek
  task: "Check network sync status"
  interval: 5min
  notify_on: "gap > 500"

operator_schedule agent=codex
  task: "Scan DEX for arb opportunities"
  interval: 15min
  notify_on: "any arb found"
```

---

## 4. Implementation Priority

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| **P0** | `operator_overview` | Low — parallel balance reads | 🔥 Saves 5+ calls every session |
| **P0** | `operator_activity_feed` | Low — merge + sort | 🔥 "What happened?" in one call |
| **P1** | `operator_agent_health` | Low — ping + seed check | ⚡ Know which agents are down |
| **P1** | `operator_quick_transfer` | Low — send_qug wrapper | ⚡ Fund agents instantly |
| **P2** | Spending guardrails | Medium — policy engine | 🛡️ Prevent runaway agents |
| **P2** | Webhook notifications | Medium — event bus + delivery | 📡 Push notifications to Discord/Telegram |
| **P3** | Agent comparison | Low — aggregate stats | 🎮 Fun + useful |
| **P3** | Scheduled tasks | High — cron-like scheduler | 🤖 Semi-autonomous → autonomous bridge |

---

## 5. Webhook Architecture (detailed)

### 5.1 Current webhook tools (v2.16.2)

| Tool | Status |
|------|--------|
| `webhook_register` | ✅ Deployed |
| `webhook_list` | ✅ Deployed |
| `webhook_remove` | ✅ Deployed |
| `webhook_test` | ✅ Deployed |
| `poll_wallet_events` | ✅ Deployed (checkpoint-based polling) |

### 5.2 Proposed additions

```typescript
// NEW: Operator-focused webhooks with event filtering
server.tool("operator_webhook_register", {
  url: "https://discord.com/api/webhooks/...",
  events: ["agent.tx.out.amount>100", "agent.offline", "agent.error"],
  format: "discord" | "telegram" | "json",
  throttle_secs: 30
})

// NEW: Event types
type OperatorEvent =
  | "agent.tx.in"           // Any incoming tx
  | "agent.tx.out"          // Any outgoing tx
  | "agent.tx.in.memo"      // Incoming tx with memo
  | "agent.tx.out.amount>N" // Outgoing tx above threshold
  | "agent.balance<N"       // Balance below threshold
  | "agent.error"           // Any MCP error
  | "agent.offline"         // Agent MCP unreachable
  | "agent.online"          // Agent MCP back online
  | "network.sync.gap>N"    // Node falling behind
  | "network.sync.caught_up" // Node caught up
  | "contract.deployed"     // Smart contract deployed
  | "bounty.claimed"        // Bounty paid out
  | "dex.swap.large"        // Large DEX swap
```

### 5.3 Discord webhook format

```json
{
  "content": null,
  "embeds": [{
    "title": "🚨 DeepSeek — Large Outgoing Transaction",
    "description": "150 QUG sent to qnkabc...def",
    "color": 16753920,
    "fields": [
      {"name": "Agent", "value": "DeepSeek", "inline": true},
      {"name": "Amount", "value": "150 QUG", "inline": true},
      {"name": "Memo", "value": "bounty: AGORA#42:finder", "inline": false}
    ],
    "timestamp": "2026-05-24T14:32:00Z"
  }]
}
```

---

## 6. Operator Config File

Central config for all operator settings:

```json
// ~/.quillon/operator.json
{
  "agents": {
    "claude":  { "wallet": "qnk71549...", "label": "Rocky",    "mcp": "local" },
    "codex":   { "wallet": "qnka3a92...", "label": "Codex",    "mcp": "remote" },
    "deepseek":{ "wallet": "qnk7f31f...", "label": "DeepSeek", "mcp": "local" },
    "grok":    { "wallet": "TBD",         "label": "Grok",     "mcp": "remote" },
    "qwen":    { "wallet": "TBD",         "label": "Qwen",     "mcp": "remote" }
  },
  "webhooks": {
    "discord_alerts": "https://discord.com/api/webhooks/...",
    "telegram_alerts": "https://api.telegram.org/bot.../sendMessage"
  },
  "policies": {
    "default_max_daily_spend": 500,
    "default_max_single_tx": 100,
    "alert_balance_below": 50,
    "heartbeat_interval_secs": 60,
    "heartbeat_missed_threshold": 3
  },
  "operator_wallet": "qnkefca1e8c..."
}
```

---

## 7. Quick Wins (implement today)

These require minimal code changes and deliver immediate value:

### 7.1 `operator_overview` — 30 lines of code

```typescript
server.tool("operator_overview", "Multi-agent status in one call", 
  async () => {
    const cfg = JSON.parse(fs.readFileSync(OPERATOR_CONFIG, "utf8"));
    const results = await Promise.all(
      Object.entries(cfg.agents).map(async ([name, agent]) => {
        try {
          const bal = await apiSigned(`/wallets/${agent.wallet}/balance`, "GET");
          return { name, wallet: agent.wallet, balance: bal.data.balance_qnk, status: "online" };
        } catch {
          return { name, wallet: agent.wallet, balance: "?", status: "offline" };
        }
      })
    );
    return formatTable(results);
  }
);
```

### 7.2 `operator_quick_transfer` — wrapper around send_qug

```typescript
server.tool("operator_quick_transfer", "Fund an agent instantly",
  { agent: z.string(), amount: z.number(), memo: z.string().optional() },
  async ({ agent, amount, memo }) => {
    const cfg = JSON.parse(fs.readFileSync(OPERATOR_CONFIG, "utf8"));
    const target = cfg.agents[agent];
    if (!target) return { error: `Unknown agent: ${agent}` };
    // Use operator's seed to send
    return sendQuG(target.wallet, amount, memo || `funding:${agent}`);
  }
);
```

---

## 8. Summary

| Now | After |
|-----|-------|
| 5+ tool calls to check all agents | 1 call: `operator_overview` |
| Manually check each agent's activity | 1 call: `operator_activity_feed` |
| Don't know if agent is down | `operator_agent_health` pings all |
| Discord: "what happened?" | Webhook pushes events to Discord |
| Agent can overspend | Per-agent spending guardrails enforced by MCP |
| Manually fund agents | `operator_quick_transfer agent=deepseek 50` |
