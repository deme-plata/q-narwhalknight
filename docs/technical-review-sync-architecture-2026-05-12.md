# Technical Review: Sync Architecture & Long-Term Viability
**Date**: 2026-05-12  
**Scope**: Balance sync, turbo sync, node integrity, 256-year operational horizon  
**Status**: Honest assessment — not a PR description

---

## Executive Summary

The immediate crisis (Beta at 1347 wallets, Gamma at 1340 vs. Epsilon's authoritative 1348) will be resolved within hours by the work in progress. The **foundation** — BLAKE3 balance integrity hashing, BAL-001 enforcement, max-wins write guard, P2P state sync — is architecturally correct.

But the **execution** has a systemic problem: every component designed to maintain balance integrity was added as a reactive fix to a corruption event, not as a designed invariant. The result is a stack of guards protecting a database whose writes were never made safe in the first place. That is not a 256-year design.

**Short verdict**: the foundation is right, the execution needs restructuring at two specific points. Fix those two things and the architecture becomes genuinely durable. Don't fix them and each new node joining the network silently corrupts itself.

---

## What We Found in This Session

### Root Cause Chain

Everything traces to a single decision in `turbo_sync.rs`:

```rust
// ~line 4367: "optimization" that skips transfers during bulk sync
if blocks_behind > 5_000 {
    process_block_coinbase_only_tx(block)  // silently drops every transfer
} else {
    process_block_full(block)
}
```

Any node that bootstraps when it is more than 5,000 blocks behind the tip — which is **every new node ever** — processes 17+ million blocks without crediting a single transfer transaction. Coinbase-only wallets (miners) appear. Transfer-only wallets (receivers) are silently omitted.

This is not a subtle edge case. It fires unconditionally on every fresh node.

### The Compounding Failure Sequence

The coinbase-only bug would be recoverable if the balance replay worked. It doesn't, because of a second independent bug:

`apply_balance_checkpoint()` refuses to wipe and re-import the embedded snapshot if the node already has **more wallets than `CHECKPOINT_WALLET_COUNT` (1,326)**. A node that ran turbo sync and accumulated, say, 1,340 wallets writes `"skipped-authoritative"` into its checkpoint marker. From that point forward, `is_genesis_node()` returns `true`, and `replay_post_checkpoint_balances()` skips permanently with the message:

> `Genesis/archive node detected — balances already correct from genesis, skipping replay.`

The node's 1,340 wallets are wrong. The system thinks they are authoritative. The contradiction is undetectable from inside the node.

### The Admin Reset Gap

The `POST /api/v1/admin/reset-balance-replay` endpoint (added in v10.9.2) clears the replay done-flag so SYNC-006 reruns. This works — but only if SYNC-006 is still alive. SYNC-006 is a `tokio::spawn` one-shot task. Once it exits (immediately at startup if the done-flag is set), clearing the flag is a no-op. Nothing re-spawns it.

This session's v10.9.5 fix extends the admin reset to also fix the checkpoint marker. That's the right immediate fix. But the underlying problem — SYNC-006 should be re-triggerable without a restart — remains.

### Current Network Wallet Count Distribution

From STATE SYNC P2P responses as of 2026-05-12 07:18 UTC:

| Node | Wallets | Status |
|------|---------|--------|
| Epsilon (ca6e7ea) | **1,348** | Authoritative — running since genesis |
| Beta (b00c9be) | 1,347 | Replay running, ~3h remaining |
| Gamma (807745) | 1,340 | Stuck — checkpoint marker fix deploying |
| bf35a38, 50a3452 | 1,340 | Same bug, no fix deployed |
| a41978a | 1,298 | Partial replay or early sync |
| fd732f5 | 1,192 | Heavy divergence |

**5 of 7 visible nodes have wrong wallet counts.** BAL-001 enforcement activates at h=20,000,000 (~237K blocks from h=17.8M, approximately 2.5 days). Nodes with wrong balance roots will begin rejecting valid blocks at that height.

---

## What Is Architecturally Sound

These components are correct and should not change:

**BLAKE3 balance integrity hash** — computing a deterministic hash of the sorted wallet→balance map and comparing it across peers is exactly the right approach. It detects corruption immediately. The hash at a given height is reproducible and verifiable.

**Max-wins write guard (CLAUDE.md Rule 1)** — `save_wallet_balances` checking `existing >= new` before overwriting is the only safe way to batch-write balances. This rule exists because violating it destroyed a real user's balance (3,200 → 1,484 QUG, May 2026 incident). Never relax it.

**Embedded checkpoint concept** — bootstrapping new nodes from a known-good snapshot rather than replaying 17M blocks from genesis is the right scalability decision. The problem is not the concept, it's the implementation details.

**BAL-001 quorum enforcement** — gating block acceptance on matching state root is the correct long-term mechanism. This turns balance integrity from a diagnostic into a consensus property.

**P2P state sync (opportunistic merge)** — importing wallet and token balances from peers on first sync is a practical bootstrap mechanism, as long as the source-of-truth is chain-derived (not peer-derived) by the time enforcement activates.

---

## What Will Fail Before 256 Years

Ranked by how soon they become critical.

### 1. The coinbase-only turbo sync threshold (critical — fails today)

The `blocks_behind > 5_000` threshold for coinbase-only processing is not an optimization, it is a corruption source. Every node that syncs from scratch is currently mis-configured by design.

**Fix**: Remove the threshold entirely. Process all transactions during turbo sync regardless of distance from tip. The speed cost is real but acceptable — current turbo sync rates (1,300–11,700 blocks/sec) are fast enough that processing transfers does not materially change sync time.

An alternative formulation: add an explicit `is_bootstrap_sync` flag set only during the initial genesis-to-tip sync, and enforce full transaction processing unconditionally during that phase. This makes the intent clearer and easier to test than a raw threshold removal. Either approach achieves the same result; removal is faster for the sprint.

**File**: `crates/q-storage/src/turbo_sync.rs` ~line 4367.

**Deadline**: Before any new node is deployed to production. This is the single highest-leverage fix in the codebase.

### 2. The embedded checkpoint becomes stale (critical — fails within ~1 year)

`CHECKPOINT_WALLET_COUNT = 1,326` at `CHECKPOINT_HEIGHT = 16,538,868` is hardcoded in the binary. As of today (h=17.8M) this checkpoint is 1.27M blocks old. In one year it will be ~30M blocks old. In 256 years it will be 8 billion blocks old.

When a new node joins and the checkpoint is, say, 5M blocks old, the replay phase processes 5M blocks at 1,300 blocks/sec = 1 hour. At 30M blocks old: 6.4 hours. At 500M blocks old: 4.5 days. At 8B blocks old: 71 days of replay before the node is usable.

This is not a 256-year architecture. It is barely a 2-year architecture.

**Fix**: Implement rolling checkpoint generation. Every N blocks (e.g., every 1M or 5M blocks), the network produces a signed checkpoint snapshot that is gossipped and verified by peers. New nodes bootstrap from the most recent valid checkpoint, not from a binary constant. The checkpoint is updated continuously, not hardcoded in a build.

This requires:
- A checkpoint generation protocol (who signs, how many signatures required)
- Checkpoint discovery via DHT or gossip
- Checkpoint verification (signature quorum + chain anchor)
- Fallback to genesis if no checkpoint is available

This is significant engineering but unavoidable for a chain meant to run past 2027.

A reasonable initial design:
- Generate checkpoint every 1M blocks (~8–10 days at current rates)
- Require 2/3 of active validator stake to co-sign the snapshot
- New nodes accept the most recent checkpoint endorsed by >50% of stake they see in the DHT
- Minimum viable checkpoint distance (too frequent = gossip overhead; too infrequent = long replay) is around 500K–2M blocks depending on network size

**This is P2, not Q3.** Every month this is deferred, the sync time for new nodes grows. At current block production rate (~1 block/sec), checkpoint staleness grows by ~2.6M blocks/month. By Q3 this is a 6–8 hour replay; by next year, >24 hours. That's a user-experience death before any 256-year argument is relevant.

### 3. Block history is unserveable on new nodes (high — fails today for historical queries)

Turbo sync downloads blocks from `effective_start_height` (~16.75M) forward. Blocks 1–16,749,999 are never stored on a new node. `current_height_atomic` is set to the MAX block height ever received, not the contiguous stored height. So a new node announces height 17.8M but has no data for 95% of the chain.

Any block explorer query for height < 16.75M returns `None` from a non-Epsilon node. This is silent data loss, not a graceful error.

**Fix A (1 day)**: Add `Q_ARCHIVE_NODE_URL` env var. `get_qblock_any_format()` and `get_qblocks_range()` HTTP-proxy to the archive node when local lookup fails. Block explorer queries are transparently served from Epsilon.

**Fix B (1 week)**: Add a `NODE_TYPE=light|full|archive` declaration. Light nodes announce they cannot serve pre-sync history. `current_height_atomic` reflects contiguous stored height, not MAX received. Nodes requesting historical blocks know to ask archive nodes.

### 4. SYNC-006 is a one-shot task with no external trigger (medium)

Once SYNC-006 exits, the only way to re-run the balance replay is a service restart. The admin endpoint's `"SYNC-006 will re-run on next 30s poll"` message is a lie if the task already exited — which it will in the common case (done-flag set on startup → task exits before any admin reset is called).

**Fix**: Convert SYNC-006 from a `tokio::spawn` that returns on the done-flag to a `tokio::spawn` loop that sleeps 30 seconds and re-checks on every cycle. The done-flag prevents unnecessary work; clearing the flag causes re-execution on the next tick without a restart.

```rust
loop {
    tokio::time::sleep(30s).await;
    if is_balance_replay_done() { continue; }  // done flag set, skip
    if !is_checkpoint_applied() { continue; }  // not a checkpoint node
    // run replay
}
```

This is a 20-line change that makes the admin endpoint work as documented.

### 5. State root is not a block header field (medium — necessary for BAL-001 to work as consensus)

BAL-001 currently detects divergence by comparing balance hashes out-of-band via STATE SYNC P2P responses. This means:
- Enforcement only fires when a peer happens to respond with a different hash
- The enforcement window is probabilistic, not deterministic
- A node can produce incorrect blocks for arbitrarily long if it doesn't happen to compare with an honest peer at the right moment

For a 256-year chain, balance state correctness must be a **first-class consensus property**, not a monitoring check. The state root (BLAKE3 of sorted wallet balances) should be included in every block header and validated by every peer that receives the block. A block with wrong state root is an invalid block, full stop.

This is how Ethereum (world state root), Solana (accounts hash), and every serious L1 handles it. It's the correct long-term design.

**Note**: This is a consensus rule change that requires a height-gated activation (cannot retroactively validate old blocks). Plan ~2–4 weeks of work.

**Critical deployment consideration**: adding the state root field splits the network the moment it ships. Old nodes reject blocks with the new field; new nodes reject blocks without it. The activation height must be announced conservatively — at least 6 months out — and the chain should require ~90% of validator stake to signal readiness before the cutover, with an automatic delay if signaling is insufficient. This is different from a simple height-gated change: it requires coordination across the entire node population.

### 6. Max-wins guard blocks replay from correcting spenders (medium — silent correctness gap)

`replay_post_checkpoint_balances()` builds the complete correct balance map in memory, then persists it via `save_wallet_balances()`. The max-wins guard in `save_wallet_balances` skips any write where `existing >= new`. For wallets that made QUG sends after the checkpoint height, the replay-derived balance is lower than the checkpoint value still on disk. The guard fires, the disk write is blocked, and the wallet's disk balance remains inflated at its checkpoint value.

The in-memory map (`wallet_balances`) IS unconditionally replaced with the correct replay result (line 6343: `*wb = replay_map`), so the node operates correctly while running. But on the next restart, RocksDB is reloaded and those wallets revert to their inflated checkpoint balances.

**Current impact**: Limited for nodes deployed recently against a fresh checkpoint (only ~1.27M post-checkpoint blocks, few spends). Grows over time as the checkpoint ages.

**Fix**: Add `save_wallet_balances_replay()` that writes the replay map directly to RocksDB without the max-wins check. Only `replay_post_checkpoint_balances` should call this path — all other callers continue to use the guarded version. The guard is correct for incremental live writes; it is wrong for the replay's authoritative full-state overwrite.

---

## The Replay Speed Problem

The current replay (reindex + balance scan) takes ~3 hours on Beta with modern hardware. This is because:

1. The reindex phase (`reindex_dag_blocks_to_height_keys`) scans all 17.8M blocks using the **old-DAG parser** for early blocks — a slow path added for legacy format compatibility. Observed rate: ~1,300 blocks/sec.

2. At this rate, as the chain grows: 1 year → 31.5M blocks → 6.7 hours replay. 5 years → 158M blocks → 33 hours. 10 years → 315M blocks → 67 hours. **Not 256-year viable.**

The old-DAG parser path needs to either be eliminated (migrate all stored blocks to the new format once, at upgrade time) or handled with a persistent index (build the key-conversion index once at startup and never recompute it). The per-boot reindex is the bottleneck.

---

## Priority Order for the Next 30 Days

| Priority | Fix | Deadline | Risk |
|----------|-----|----------|------|
| P0 | Deploy v10.9.5 to Gamma, verify wallets=1348 | Before h=20M (~2.5 days) | Low |
| P0 | Verify Beta reaches wallets=1348 after replay | Before h=20M | Low |
| P1 | **Remove coinbase-only turbo sync threshold** | This sprint | Medium — test on Alpha Docker first |
| P1 | Make SYNC-006 a persistent polling loop | This sprint | Low |
| P1 | `save_wallet_balances_replay()` — bypass max-wins guard in replay | This sprint | Low |
| **P2** | **Rolling checkpoint generation protocol** | **Next sprint** | High — protocol design needed |
| P3 | Archive node proxy (Q_ARCHIVE_NODE_URL) | Following sprint | Low |
| P3 | Fix current_height_atomic to contiguous stored height | Following sprint | Medium |
| P4 | State root in block headers | Q3 2026 | High — consensus change, 6-month rollout |
| P4 | Persistent DAG block reindex (eliminate per-boot cost) | Q4 2026 | Medium |

---

## The ZK-SNARK Endgame

The network has an existing whitepaper (`papers/RECURSIVE_SNARK_WEAK_SUBJECTIVITY_ELIMINATION.md`) describing an IVC (Incrementally Verifiable Computation) approach that supersedes rolling checkpoints entirely.

**The core idea**: each epoch produces a recursive post-quantum ZK-SNARK proof πₙ that includes verification of πₙ₋₁. The final proof cryptographically attests to the validity of all blocks from genesis. A new node downloads the current state + πₙ and verifies in ~10ms with no trust in any checkpoint provider.

This is the correct 256-year bootstrap mechanism. It eliminates:
- Rolling checkpoint gossip protocol (P2 in the 30-day table)
- Weak subjectivity (cannot distinguish honest from fake chains without a checkpoint)
- All replay time (constant verification, not linear in chain length)

**Prerequisites before IVC can deploy:**
1. **State root in block headers** (P4): the ZK circuit needs to prove state transitions, which requires committed state roots. Without this, you cannot prove "block N transitions state from A to B."
2. **Recursive circuit implementation**: LatticeGuard and ZK-STARK infrastructure exists, but recursive composition for BFT signature aggregation + state transitions is not yet implemented. This is 3–6 months of circuit engineering.
3. **GPU proving infrastructure**: Epoch proof generation takes ~2h on GPU per million blocks. The network needs distributed proving so no single node bears the full cost.

**Practical timeline**: Start circuit design in parallel with rolling checkpoints (Q3). Target IVC deployment in Q1 2027 at the earliest. Rolling checkpoints serve as the bridge — they solve the linear replay problem for the 1–2 years before IVC is ready.

---

## Verdict: 256-Year Viability

**The answer is no, not as currently implemented — but yes, with two structural changes.**

The two changes that determine whether this chain lasts 10 years or 256:

1. **Remove the coinbase-only turbo sync threshold.** Every node must process every transaction during sync. No shortcuts. This fix is one function call removal in `turbo_sync.rs`. Without it, every new node that ever joins has wrong balances, and the repair cost grows as the chain grows.

2. **Implement rolling checkpoint generation.** The embedded snapshot cannot be a binary constant. It must update continuously as the chain grows. Without this, sync time for new nodes grows without bound: linear in chain length, currently doubling roughly every 18 months.

Everything else — SYNC-006 improvements, archive node proxy, state root in headers — is important but iterative. These two are structural. The chain's integrity invariant is only as strong as the mechanism that enforces it on new joiners.

The good news: the integrity detection infrastructure (BLAKE3 hashes, BAL-001, max-wins guard, STATE SYNC) is correct. The network can detect corruption. The gap is that it cannot yet prevent it at the point of entry — during turbo sync — and cannot efficiently heal it on nodes that are months or years behind. Fix the prevention, fix the efficiency, and the architecture holds.

---

*This review reflects the state of the codebase as of commit `23af629e` on branch `feature/safe-batched-sync-v1.0.2`, 2026-05-12.*
