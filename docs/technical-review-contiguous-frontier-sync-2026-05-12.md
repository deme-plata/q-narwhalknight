# Technical Review: Contiguous Frontier Sync Performance

**Date:** 2026-05-12
**Branch:** `feature/safe-batched-sync-v1.0.2`
**Triggered by:** v10.9.9 soak observation — block-pack reception at ~28K bps but contiguous frontier advancing at only ~12.8 bps
**Scope:** Architecture of the initial-sync path for fresh checkpoint-bootstrapped nodes

---

## Executive Summary

After landing Option A (qblock:latest reflects contiguous storage only) and Option B (large gap ranges routed through `download_chunks_parallel`), the v10.9.9 soak revealed a real workload pattern: **the data-layer download is fast but the contiguous frontier crawls**. At observed rates a fresh node would need ~16 days to reach tip from genesis. The fix is not in the download engine — it works. The fix is in **what we ask it to download**.

This document analyses why the disparity exists, why it doesn't actually matter for security or consensus, and what to change so that *time-to-usable* matches *time-to-archive-complete* for the median user.

---

## Observed Performance (v10.9.9 Soak, 33 min uptime)

```
Block-pack reception (P2P)          ~28,600 blocks/sec
Batch commits to storage             ~6,300 blocks/sec
Balance updates applied                ~180 ops/sec
Contiguous frontier advance              12.8 blocks/sec
```

The first three numbers say the network and storage are doing their job. The fourth is the user-visible sync rate — "am I ready yet?" — and it's two orders of magnitude below the underlying throughput.

Heights observed at 33 min:
- **Contiguous chain**: 1 → 25,400 (the frontier the API and consensus rely on)
- **Stored chunks elsewhere**: clusters around 1.3M, 1.5M, 13.5M, 13.6M
- **Live tip seen via gossipsub**: 17.88M (queued, not applied — "need to sync 17,854,643 missing blocks first")

At 12.8 bps, reaching 17.88M from 25K is ~16 days.

---

## Root Cause

After v10.9.9 the sync pipeline is roughly:

```
                ┌──────────────────────────────────────────────┐
                │ download_chunks_parallel  (turbo machinery)   │
                │   - Apollo Kalman concurrency control          │
                │   - Warp Sync prefetch                         │
                │   - Gravity-assist peer ordering               │
                │   - 8 in-flight chunks of 1000 blocks each     │
                └─────────┬──────────────────────────┬──────────┘
                          │                          │
                          ▼                          ▼
                  Block-pack server          Block-pack server
                  (any peer at the           (any peer at the
                   right height)             right height)
                          │                          │
                          ▼                          ▼
                      Storage                    Storage
                  (height key)              (height key)
                          │                          │
                          ▼                          ▼
                ┌──────────────────────────────────────────────┐
                │ qblock:latest pointer (post-Option-A)         │
                │   advances ONLY when block current_pointer+1  │
                │   is in storage; otherwise stays put          │
                └──────────────────────────────────────────────┘
```

Because `download_chunks_parallel` was designed for tip-sync (where the chain end is always small distance from contiguous), it treats the request queue as **unordered** — any chunk can complete next. When we hand it 16,539 chunks covering heights 1–16,538,868, it parallelises across the whole space, so completed chunks land at random heights.

Per Option A's contiguity check, **only the chunk that contains height `current_pointer+1` advances the frontier**. Statistically that is 1 chunk out of ~16,500, so ~99.99% of completed downloads do nothing for the frontier in the short term.

This is the inversion of locality: maximum parallelism for the data layer, minimum locality for the contiguous tip.

---

## What Doesn't Need Fixing

A few things to rule out before designing changes:

**Option A is not the problem.** The pointer-cap is correct behavior. Before A, the pointer would have jumped to 17.88M instantly while the data underneath was full of holes. The "fast" appearance was a lie. We don't want to undo A.

**Option B is not the problem.** Delegating to `download_chunks_parallel` is the right machinery to use — it gives us bandwidth-weighted peer selection, prefetch, adaptive timeouts. The download throughput numbers prove it works.

**`download_chunks_parallel` is not buggy.** It is doing exactly what it was designed for. The issue is that it was designed for a smaller, tip-local workload (a few hundred chunks max during normal sync). Stretching it to 16,500 chunks exposes a missing scheduling layer above it.

**Per-balance-update latency is not the bottleneck.** 180 balance updates/sec is plenty — at 1.3 balance updates per block (typical), that's 138 blocks/sec of consensus work. The frontier could move at that rate if the right blocks were available.

---

## What Should Change

Three optimisation options, ordered by effort and impact.

### Option C-1: Skip pre-checkpoint backfill by default (lowest effort, biggest single win)

The 16.5M-block range from 1 to checkpoint exists *only* for archive purposes. Consensus correctness comes from the checkpoint snapshot — every wallet balance at h=16,538,868 is verified by the embedded SHA-256 hash. A node that has checkpoint balances + every block from checkpoint forward has full consensus validity and can validate every block, mine, and participate in BAL-001 at h=20M.

The pre-checkpoint blocks are needed for one thing only: serving historical block-explorer queries. That is a node-operator opt-in feature, not a default.

**Change**:

In `main.rs:7896-7912`, the v10.8.4 auto-detect of "pre-checkpoint history missing" should require an explicit `Q_ARCHIVE_NODE=1` (or equivalent). Without that flag the 1-to-checkpoint range is not queued for gap-fill. The forward sync then runs unimpeded from `CHECKPOINT_HEIGHT+1` to tip.

The contiguous pointer should also be initialised to `CHECKPOINT_HEIGHT` right after the checkpoint applies, not stay at 0. The chain from 1 to checkpoint is logically "covered" by the checkpoint snapshot; the contiguous reporting should reflect that.

**Outcome**: a fresh checkpoint-synced node has ~1.34M blocks to fetch (CHECKPOINT_HEIGHT=16.54M to tip=17.88M). At the *current* download_chunks_parallel rate (~6,300 commits/sec) this finishes in **~3.5 minutes**. The archive backfill, if requested, runs in the background after the node reaches tip.

This is the single highest-leverage change in this document. ~80 LOC.

### Option C-2: Forward-priority scheduling inside `download_chunks_parallel`

If we *do* want to keep archive backfill enabled by default, the scheduling problem must be solved directly. The fix is locality: prefer chunks near `current_pointer+1` over chunks far ahead.

**Change**:

`download_chunks_parallel` currently treats the chunk queue as FIFO. Replace the inner loop's chunk picker with a priority-based one:

- Maintain a "frontier window" of size N (e.g. 100K blocks) starting at `current_pointer+1`
- The scheduler may dispatch chunks within the window in any order (preserving parallelism)
- Chunks outside the window go to a deferred queue; they only run when in-flight count is below a low-water mark *and* the window can't be filled

The pointer advances naturally as window-front chunks complete. When it advances by W blocks, the window slides forward by W. The deferred queue is drained opportunistically when the network has spare capacity.

**Outcome**: the contiguous frontier advances at roughly the *commit rate inside the window*, which is the same ~6,300 bps the network is already delivering. Reaching tip from 25K = 17.85M / 6,300 ≈ **47 minutes** for the post-checkpoint range.

Cost: ~300-500 LOC inside the existing turbo_sync module; design care required to not regress tip-sync behaviour where N=tip-current is small.

### Option C-3: Headers-first sync (the big-L answer, longer engineering)

Bitcoin Core, Geth, and every serious L1 client uses *headers-first sync*: download the entire chain of block headers (small) before any bodies. Headers establish the canonical chain and the validator order. Bodies (transactions, payloads) are then fetched in parallel by hash with no ordering constraint.

For QNK this would mean:

1. Sequentially fetch ~16.5M headers (~80 bytes each → ~1.3 GB total). At 28K headers/sec this is ~10 minutes.
2. Use header chain to validate, advance pointer to tip rapidly. Mining, peer announcements, BAL-001 participation possible from this point.
3. Bodies (block transactions, coinbase records) fetched in parallel with no ordering constraint. Balance computation happens as bodies arrive.

**Outcome**: full chain ordering reached in ~10 minutes; balance state converges to tip as transactions arrive, typically within a further 30-60 minutes.

Cost: ~2-3 weeks of work. Block structure needs a clean header/body split; current `QBlock` is one struct. Per-CF storage layout would change. Block-pack codec needs a new request type (`HeaderPack`) that excludes transactions. Validator code needs to validate header-only chains. Worth doing eventually, not urgent.

---

## Recommended Path

**This sprint**: implement **Option C-1** (skip pre-checkpoint backfill by default). 80 LOC, contained to `main.rs` and the checkpoint code path. Bring sync-time for a fresh node from ~16 days to ~3.5 minutes.

**Next sprint**: implement **Option C-2** (forward-priority scheduling inside `download_chunks_parallel`). Even with C-1, for archive-enabled nodes this matters — the 16.5M archive backfill would still run for weeks otherwise. With C-2 the archive completes in a few hours rather than weeks.

**Quarterly horizon**: design **Option C-3** (headers-first). This is the long-term architectural correctness. It also unlocks light-client mode (header-only) which is the prerequisite for mobile wallets that don't store the full chain.

C-1 and C-2 are complementary, not substitutes. C-3 supersedes both for archive-mode but doesn't change the basic logic.

---

## Implementation Sketch — C-1 (recommended for v10.9.10 follow-up)

`main.rs:7894-7912` currently has:

```rust
// v10.8.4: Auto-detect pre-checkpoint history gap.
// Any checkpoint-synced node is missing blocks 1..=CP_HEIGHT (skipped during warp-sync).
// Add this range so the background gap-fill downloads full chain history from peers (Epsilon archive).
if state.storage_engine.is_checkpoint_applied().await {
    let has_genesis = state.storage_engine.get_qblock_by_height(1).await
        .ok().flatten().is_some();
    if !has_genesis {
        warn!("...");
        effective_gap_ranges.push((1, CP_HEIGHT));
    }
}
```

Change to:

```rust
// v1.0.2: Pre-checkpoint backfill is opt-in via Q_ARCHIVE_NODE=1.
// Consensus correctness comes from the checkpoint snapshot; pre-checkpoint blocks
// are needed only for historical block-explorer queries (archive mode).
// Without this flag, a fresh node syncs only from CHECKPOINT_HEIGHT+1 to tip,
// which takes minutes rather than days.
let archive_mode = std::env::var("Q_ARCHIVE_NODE")
    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    .unwrap_or(false);

if state.storage_engine.is_checkpoint_applied().await {
    let has_genesis = state.storage_engine.get_qblock_by_height(1).await
        .ok().flatten().is_some();
    if !has_genesis && archive_mode {
        warn!(
            "🔍 [ARCHIVE MODE] Pre-checkpoint history missing (blocks 1-{}) — scheduling background download.",
            CP_HEIGHT
        );
        effective_gap_ranges.push((1, CP_HEIGHT));
    } else if !has_genesis && !archive_mode {
        info!(
            "ℹ️ [LIGHT NODE] Pre-checkpoint history not fetched. Set Q_ARCHIVE_NODE=1 to enable historical block-explorer support."
        );
    }
}
```

Also in the checkpoint-apply path: when `apply_balance_checkpoint` completes, write `qblock:latest = CHECKPOINT_HEIGHT` (under Option A this is a legitimate advance because the checkpoint atomically establishes the state at that height). This bootstraps the contiguous frontier to the checkpoint, so forward sync starts from there.

Test: spin a fresh container with `Q_ARCHIVE_NODE` unset, verify contiguous reaches tip in minutes, verify all `/api/v1/integrity/*` endpoints return correct values, verify a block-explorer query for h<CP_HEIGHT returns a graceful "archive node required" message rather than `None`.

---

## Test Wallet Convergence — Real Cost Today

The soak test currently running on Epsilon expects the test wallet `qnkf07316...` to receive its balance on the soak container once that container reaches h=17,880,044. With current sync behaviour that's ~16 days. With C-1 deployed, the soak would reach tip within minutes and the convergence would be observable in real time — turning the soak into a feedback loop that runs in tens of seconds rather than days.

This is why C-1 matters beyond raw sync speed: it makes the whole class of "does the binary produce correct end-state?" tests practical to run.

---

## What This Is *Not*

This is not a consensus change. It does not affect block validation, BAL-001 enforcement, balance computation, or any chain-of-trust property. The checkpoint snapshot is already a load-bearing trust anchor; making pre-checkpoint backfill opt-in does not move that bar. Archive-mode nodes operate identically to current default; non-archive nodes simply skip a backfill that was never required for correctness.

---

## What This Is

A scheduling fix. The data layer is fast; the scheduling layer above it is missing locality. Adding locality unblocks the user-visible sync experience and turns a 16-day archive-rebuild into a 3-minute participation-ready bootstrap.

— Server Beta, 2026-05-12
