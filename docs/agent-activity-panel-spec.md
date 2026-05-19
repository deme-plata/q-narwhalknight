# Agent Activity Panel — Codex/X-style task surface for agentic money on quillon.xyz

**Date**: 2026-05-19
**Status**: Spec brief for Codex — third companion to PR #87 (Agent Fiber Lane) and PR #88 (Twitter MCP + x-algorithm)
**Target**: v10.10.0+ candidate, ships standalone (frontend + read-side API extensions only)
**Strategic priority**: HIGH — this is the observable surface of the "agent-native" story. Without it, agents act invisibly; with it, the activity becomes evidence.

---

## 1. Why now, why this pattern

Three influences converge:

1. **Codex's task-panel UI** is the right mental model for "what is the agent doing right now" — a vertical list of active / queued / done tasks, with parallel execution visible, with dependency arrows when relevant, with inline approve/edit/discard affordances. Far better than the traditional dashboard-with-widgets pattern that every chain explorer ships.

2. **xAI's Home Mixer pipeline pattern** (from `github.com/xai-org/x-algorithm`, Apache-2.0) is the right *backend* architecture: 6 composable trait-types (`Source → Hydrator → Filter → Scorer → Selector → SideEffect`) that any decision-streaming workload fits. Quillon Graph's "what to show in the agent panel" is exactly this pattern — sources are SSE streams + RocksDB queries, hydrators add tx-detail / token-metadata / fee-token-context, filters drop noise, scorers rank by relevance, selectors choose top-K per zone (NOW / QUEUED / DONE).

3. **The agent-native chain story is now observable**. PR #87 (Agent Fiber Lane) puts agents into the mempool; PR #88 (Twitter MCP + x-algorithm) puts them onto X. Without a third leg — a place where humans WATCH the agents work — the story is invisible. This panel is the third leg.

The panel becomes the canonical "what's my agent up to right now" surface. Embeddable in dashboards, shareable as an iframe, anchored in cryptographic proof.

---

## 2. The visual layout

Restraint is the design principle. Codex's UI shows three datapoints per task, not ten. We follow that:

```
┌──────────────────────────────────────────────────────────────────┐
│ Agent Activity — qnk7154929a...                          🟢 alive │
│ chain height 18,145,821 · last block 0.9s ago · 3 peers           │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  NOW                                                              │
│  ▶ DCA water-bot                   pool QUG/QUGUSD · 32s ago      │
│    Started 14:02 · 47 swaps · avg slippage 0.31%                  │
│                                                                    │
│  ▶ Mining                          block #18145821 · 8s ago       │
│    3 solutions in last 5min · 0.249 QUG earned                    │
│                                                                    │
│  ▶ Twitter draft scoring           draft #a3f9 · 0.4s ago         │
│    Predicted engagement 0.84 · neg-signal-risk 0.03               │
│                                                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  QUEUED                                                           │
│  ⏸ Tweet awaiting approval        draft #a3f9 · 4min                │
│    "Just sent 0.05 QUG to viktor via Agent Fiber Lane..."         │
│    [Approve via wallet]  [Edit]  [Discard]                        │
│                                                                    │
│  ⏸ Bridge LP intent               wBTC/QUG · 12min                 │
│    Awaiting Bitcoin deposit confirmation (2/3 conf)               │
│                                                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  DONE (last 24h)                                                  │
│  ✓ Sent 0.05 QUG   tx 6d9265...  block 18113553  21min ago        │
│  ✓ Swap QUG→MOON   tx 7af2bb...  block 18113441  34min ago        │
│  ✓ Tweet posted    id 1834...    attestation verified ✓           │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 Per-task display contract

Three lines max in the collapsed view:
1. **Title** + **status icon** (▶ running / ⏸ queued / ✓ done / ⚠ failed) + **right-anchored age**
2. **Most important context** (the one number a human cares about — TPS, balance delta, score)
3. **Action affordance** (approve / edit / discard / retry / inspect) — only when actionable

Click anywhere on the row to expand: shows full detail, related tx hashes, signed proof links, attestation status, raw event payload.

### 2.2 Zones

Three zones, fixed order, vertical layout:

| Zone | Contents | Sort | Auto-collapse |
|---|---|---|---|
| **NOW** | tasks currently executing | most-recently-active first | never |
| **QUEUED** | awaiting human action OR external trigger | oldest-first (FIFO awareness) | after 5 |
| **DONE** | last 24h, completed | most-recent first | after 5, with "show more" |

Older history is paginated via a "View archive" link → separate page.

### 2.3 Visual constraints

- **Monospace font** for tx hashes and block heights (alignment matters)
- **Sans-serif** for everything else
- **Three colors max** (status green / queued amber / done gray) — same minimalism as Codex UI
- **No charts in the main panel** (graphs go to dedicated dashboards if needed)
- **No notification badges** — the task list IS the notification
- **No "loading..." skeletons** — empty zones are explicit ("No tasks running. Your agent is idle.")

---

## 3. Backend architecture — Home Mixer pattern applied

Steal the pattern from xAI x-algorithm directly. Define six trait-types in Rust under a new module `crates/q-api-server/src/agent_panel/`:

```rust
// crates/q-api-server/src/agent_panel/pipeline.rs

#[async_trait]
pub trait Source {
    type Candidate;
    async fn fetch(&self, ctx: &PanelContext) -> Vec<Self::Candidate>;
}

#[async_trait]
pub trait Hydrator<C> {
    async fn enrich(&self, candidate: &mut C, ctx: &PanelContext);
}

pub trait Filter<C> {
    fn keep(&self, candidate: &C, ctx: &PanelContext) -> bool;
}

#[async_trait]
pub trait Scorer<C> {
    async fn score(&self, candidate: &C, ctx: &PanelContext) -> f64;
}

pub trait Selector<C> {
    fn select(&self, candidates: Vec<(C, f64)>) -> Vec<C>;
}

#[async_trait]
pub trait SideEffect {
    async fn run(&self, ctx: &PanelContext);
}
```

### 3.1 Wiring for the panel

The agent-panel-pipeline assembles in `crates/q-api-server/src/handlers/agent_panel.rs`:

```rust
let now_pipeline = Pipeline::new()
    .source(MempoolTxSource::for_wallet(addr))
    .source(MinerSolutionSource::for_wallet(addr))
    .source(DexExecutionSource::for_wallet(addr))
    .source(TwitterDraftSource::for_wallet(addr))
    .hydrator(TokenMetadataHydrator)
    .hydrator(BlockReferenceHydrator)
    .filter(AgeFilter::max(Duration::from_secs(300))) // NOW = last 5 min
    .scorer(RecencyScorer::new())
    .selector(TopK::new(10));

let queued_pipeline = Pipeline::new()
    .source(PendingApprovalSource::for_wallet(addr))   // tweets, large sends, etc.
    .source(BridgeIntentSource::for_wallet(addr))      // wBTC LP intents pending Bitcoin confirms
    .hydrator(ApprovalUrlHydrator)
    .filter(NotExpiredFilter)
    .selector(FifoSelector);                            // FIFO so oldest-pending shows first

let done_pipeline = Pipeline::new()
    .source(ConfirmedTxSource::for_wallet(addr).since(Duration::from_hours(24)))
    .source(TwitterPostSource::for_wallet(addr).since(Duration::from_hours(24)))
    .hydrator(AttestationHydrator)                      // tag tweets that have verified attestation
    .scorer(TimestampScorer::desc())                    // most-recent first
    .selector(TopK::new(20));
```

Each pipeline-stage is testable in isolation. New activity sources land as new `Source` impls without touching the pipeline orchestration.

### 3.2 Sources currently identifiable

| Source | Reads from | Refresh trigger |
|---|---|---|
| `MempoolTxSource` | `state.tx_pool` (DashMap) | SSE: `TransactionSubmitted` event |
| `MinerSolutionSource` | mining handler in-memory state | SSE: `BlockProduced` event |
| `DexExecutionSource` | DEX swap log | SSE: `SwapExecuted` event |
| `TwitterDraftSource` | quillon-twitter-mcp SQLite drafts DB | poll every 5s (low rate) |
| `PendingApprovalSource` | drafts DB filtered `status=PENDING` | SSE: `DraftQueued` event |
| `BridgeIntentSource` | `CF_BTC_LP_INTENT` column family | poll every 30s + SSE |
| `ConfirmedTxSource` | recent_txs RocksDB query | SSE: `BlockProduced` event |
| `TwitterPostSource` | `CF_TWITTER_ATTESTATIONS` column family | SSE: `TweetPosted` event |

Each source is at most ~50 LOC. Composable, testable.

---

## 4. SSE event protocol

Frontend connects to `GET /api/v1/events/wallet/{addr}?token=<X-Wallet-Auth-base64url>` (existing endpoint; the panel becomes the canonical consumer).

New event types this panel introduces (to be added to `StreamEvent` enum in `crates/q-api-server/src/sse.rs`):

```rust
pub enum StreamEvent {
    // ... existing variants ...

    /// v10.10.0: agent activity panel events
    AgentTaskStarted { task_id: Uuid, kind: TaskKind, started_at: DateTime<Utc>, context: serde_json::Value },
    AgentTaskProgress { task_id: Uuid, last_event: serde_json::Value, age_secs: u64 },
    AgentTaskCompleted { task_id: Uuid, outcome: TaskOutcome, finished_at: DateTime<Utc> },
    AgentTaskFailed { task_id: Uuid, reason: String, finished_at: DateTime<Utc> },
    AgentTaskQueued { task_id: Uuid, awaiting: AwaitingReason },
    AgentTaskApproved { task_id: Uuid, approved_by: Address, approval_signature: Vec<u8> },
}

pub enum TaskKind {
    DcaSwap, Mining, TwitterDraftScoring, TweetPostPending,
    BridgeLpIntent, ChainTransfer, DexSwap, /* ... */
}

pub enum TaskOutcome {
    Success { tx_id: Option<TxHash>, attestation: Option<AttestationId> },
    PartialSuccess { detail: String },
}

pub enum AwaitingReason {
    HumanApproval { approval_url: String, expires_at: DateTime<Utc> },
    ExternalConfirmation { confirms_needed: u32, confirms_seen: u32 },
    Timer { fires_at: DateTime<Utc> },
}
```

Each event carries enough context that the frontend can render without follow-up queries. This is the same "candidate isolation" pattern from xAI x-algorithm: don't make the frontend correlate events across requests.

---

## 5. Frontend implementation

New React component tree under `gui/quantum-wallet/src/components/AgentActivityPanel/`:

```
AgentActivityPanel/
├── index.tsx                  # Main component, opens SSE, renders zones
├── Zone.tsx                   # Generic NOW/QUEUED/DONE zone container
├── TaskRow.tsx                # Single task row, click-to-expand
├── ApprovalButton.tsx         # X-Wallet-Auth signed approval flow
├── tasks/
│   ├── DcaSwapTask.tsx        # Specialized renderer for DCA tasks
│   ├── MiningTask.tsx
│   ├── TwitterDraftTask.tsx
│   ├── TwitterPostPendingTask.tsx
│   ├── BridgeLpIntentTask.tsx
│   └── ChainTransferTask.tsx
├── state/
│   ├── taskStore.ts           # Zustand store: zoneOf(task) → array
│   ├── sseClient.ts           # Event dispatch into store
│   └── selectors.ts           # Derived: counts, age sorting, etc.
└── styles/
    └── panel.css              # Three-color palette, monospace zones for hashes
```

### 5.1 State store

```typescript
interface TaskStoreState {
  now: Task[];        // active
  queued: Task[];     // pending action
  done: Task[];       // last 24h
  health: { height, blockAge, peerCount, alive };
  applyEvent: (event: StreamEvent) => void;
  approveTask: (taskId: string, walletSig: WalletAuth) => Promise<void>;
}
```

Single store, derived selectors, no cross-component prop drilling.

### 5.2 Approval flow

When a task is in QUEUED with `AwaitingReason::HumanApproval`:

1. Row shows `[Approve via wallet]` button
2. Click triggers `walletAuth.signRequest(approval_url, body_hash)`
3. POST signed approval to backend
4. Backend verifies X-Wallet-Auth, marks task `AgentTaskApproved`, executes the queued action (post tweet / submit large tx / etc.)
5. SSE event flows back through `applyEvent` → task moves from QUEUED to NOW or DONE

Same pattern as PR #87 (Agent Fiber Lane) for transaction submission and PR #88 (Twitter MCP) for tweet publishing — single auth scheme across the whole agent surface.

### 5.3 Embeddable iframe mode

`<AgentActivityPanel mode="embed" wallet={addr} />` renders without the chrome (no chain-health bar, no archive link). Use case: any wallet displays any other wallet's agent activity. *Social agent observation.* Bitcoin/Ethereum don't have this concept — they lack the task model and the SSE infrastructure.

---

## 6. Implementation roadmap

Layered. Codex can ship value at each layer:

### Layer 1 — read-only NOW + DONE

- [ ] Backend: `Source`, `Hydrator`, `Filter`, `Scorer`, `Selector` traits + `MempoolTxSource`, `ConfirmedTxSource`, `MinerSolutionSource`
- [ ] Backend: `/api/v1/agent/panel/{addr}` REST endpoint, X-Wallet-Auth-gated, returns the three-zone JSON snapshot
- [ ] Frontend: `AgentActivityPanel/index.tsx`, `Zone.tsx`, `TaskRow.tsx` rendering snapshot
- [ ] Frontend: poll every 5s; SSE wiring comes in Layer 2

**Outcome**: panel is visible, shows what's happening, refreshes on a timer. Useful immediately.

### Layer 2 — SSE wiring

- [ ] New `StreamEvent` variants emitted by tx/swap/mining handlers
- [ ] Frontend `sseClient.ts` dispatches events into the store
- [ ] Real-time updates, no polling overhead

### Layer 3 — QUEUED zone + approval flow

- [ ] `PendingApprovalSource`, `BridgeIntentSource`
- [ ] `[Approve via wallet]` button + X-Wallet-Auth approval submission to `/api/v1/agent/task/{id}/approve`
- [ ] Server-side handler verifies signature, marks task approved, triggers downstream action

### Layer 4 — Twitter integration

- [ ] `TwitterDraftSource`, `TwitterPostSource` once PR #88 lands
- [ ] Attestation badges (✓ attestation verified)

### Layer 5 — embed mode + cross-wallet view

- [ ] `<AgentActivityPanel mode="embed" />` props
- [ ] Public read endpoint (X-Wallet-Auth NOT required for embed; just reads public-visible tasks)
- [ ] Iframe-safe rendering

### Layer 6 — task graph view (future)

- [ ] Render dependency arrows between tasks (e.g. "DCA depends on balance > X")
- [ ] Pipeline visualization for advanced users
- [ ] Optional — skip if not requested

---

## 7. Acceptance criteria

For the v1 PR to be mergeable:

1. Backend `/api/v1/agent/panel/{addr}` endpoint returns valid three-zone JSON for any qnk-address
2. Frontend `AgentActivityPanel` mounts and renders the three zones with sample data
3. At least 4 task-kind renderers implemented (DcaSwap, Mining, TwitterDraftScoring, ChainTransfer)
4. SSE wiring updates the panel in real-time (Layer 2 complete)
5. Approval flow works end-to-end for one task type (Layer 3 minimal)
6. Empty-state copy ("Your agent is idle.") is friendly, not bug-looking
7. Embeddable iframe mode renders cleanly without chain-health chrome

Layers 4+ can ship in follow-up PRs.

---

## 8. Strategic implication

Three pieces close the agentic-money loop:

| Layer | PR | What it does |
|---|---|---|
| **Action** | #87 | Agents can transact via Agent Fiber Lane (chain-side fiber for tx) |
| **Voice** | #88 | Agents can speak (algorithmically calibrated, on-chain attested) via Twitter MCP |
| **Witness** | **#89 (this)** | Humans can see what their agents are doing, in real-time, in one canonical surface |

The first chain to ship all three owns "agent-native infrastructure" as an observable category, not just a marketing claim. Bitcoin, Ethereum, Solana lack the foundation to ship this combination — they don't have agent-priority chain endpoints, don't have integrated MCP scorers, don't have SSE infrastructure for per-wallet event streams.

The Codex/X-style task panel as the UI is critical: it positions Quillon Graph alongside the agentic-AI developer experience users already know from Codex. Same mental model, transferred to chain activity. Familiar surface for the right reason: agents are the new "tasks running in parallel" workload, regardless of whether they're writing code or transacting QUG.

---

## 9. Files Codex should read first

- `docs/agent-fiber-lane-spec.md` — the X-Wallet-Auth + MCP architecture pattern this panel inherits
- `docs/twitter-mcp-with-x-algorithm-spec.md` — the Twitter side this panel surfaces
- `AGENT.md` (root) — the wallet/auth recipe
- `crates/q-api-server/src/sse.rs` (or wherever `StreamEvent` lives) — for the new event variants
- `gui/quantum-wallet/src/components/RecentActivityPanel.tsx` — existing similar component; this new panel is an evolution
- `gui/quantum-wallet/src/services/walletAuth.ts` — X-Wallet-Auth signer for approval flow
- `https://github.com/xai-org/x-algorithm` (especially Home Mixer source) — for the pipeline trait architecture

---

## 10. Suggested PR title

`feat(frontend+api): Agent Activity Panel — Codex/X-style task surface with Home Mixer pipeline backend`

---

*Drafted by Claude Opus 4.7 on 2026-05-19. The third leg of the agentic-money trilogy alongside PR #87 (action) and PR #88 (voice). Same Apache-2.0 license stack as Quillon Graph + xAI x-algorithm. The button is yours, Codex — and you can run this in parallel with #87 and #88 since this is mostly read-side API + frontend with no consensus impact.*
