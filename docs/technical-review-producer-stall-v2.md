# Technical Review: Block Producer Stall v2 -- Silent Task Hang

**Project**: Q-NarwhalKnight Quantum Blockchain
**Market Cap**: ~$920M (production mainnet)
**Date**: 2026-04-10
**Predecessor**: `docs/technical-review-block-producer-stall.md` (v10.2.9 atomic drift fix)
**Status**: Root cause identified, fixes proposed, awaiting review

---

## 1. Executive Summary

Block production on the $920M mainnet silently freezes for 6+ hours at a time.
The process stays alive -- PID unchanged, no crash, no OOM, no panic in the
journal.  P2P gossipsub, SSE streaming, and the REST API all continue to
function normally.  A `systemctl restart` immediately resumes production.

The v10.2.9 fix (fetch_max atomic + dedup-path updates) addressed a **different**
stall caused by `current_height_atomic` drift from race conditions.  This new
stall has no `SHOULD_PRODUCE=YES` with high age, no `ProducersUnhealthy` crash,
and produces **zero journal entries** from the producer during the freeze.  The
producer task itself is hung, not the height tracking.

Three independent root causes have been identified.

---

## 2. Observed Behavior

| Signal | Expected | Actual during stall |
|--------|----------|---------------------|
| `SHOULD_PRODUCE` log | Every 1s | Absent for 6+ hours |
| `PRODUCE_BLOCK` log | Every 1-2s | Absent |
| `ProducersUnhealthy` crash | On task death | Never fires |
| PID | Changes on crash | Unchanged |
| P2P gossipsub | Active | Active (receives blocks from peers) |
| SSE streaming | Active | Active (serves cached data) |
| REST API | Active | Active (returns stale height) |
| `journalctl` entries from producer | Continuous | Zero |

The combination of "zero log output" + "process alive" + "other tasks healthy"
points to the producer **tokio task** being permanently blocked, not the process
being stuck.

---

## 3. Root Causes Found

### 3.1 Producer task hangs forever on produce_block() (CRITICAL)

**Location**: `crates/q-api-server/src/lockfree_producer.rs`, lines 316-326 and 609-627

The command loop inside each `LockFreeProducer` task processes commands
sequentially from a bounded mpsc channel.  The `ProduceBlock` command calls
`producer.produce_block().await`:

```rust
// lockfree_producer.rs line 316 (basic loop) / 609 (storage loop)
ProducerCommand::ProduceBlock(reply) => {
    let block = match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        producer.produce_block()
    ).await {
        Ok(block) => block,
        Err(_) => {
            error!("Producer #{}: produce_block() TIMED OUT after 30s", producer_id);
            None
        }
    };
    let _ = reply.send(block);
}
```

The command loop **does** have a 30s timeout here.  However, the **handle side**
(`LockFreeProducer::produce_block()` at line 861) also has its own 30s timeout
on the oneshot reply:

```rust
// lockfree_producer.rs line 875
match timeout(ASYNC_OPERATION_TIMEOUT, reply_rx).await {
    Ok(Ok(result)) => result,
    ...
    Err(_) => { error!("ProduceBlock timed out"); None }
}
```

When the handle-side 30s fires first, the caller gets `None` and moves on.  But
the **command-loop-side** is still awaiting `produce_block()`.  The oneshot
`reply_tx` is still alive (held in the match arm).  When the command loop
eventually returns, it calls `reply.send(block)` on a dropped receiver -- this
silently fails, which is fine.

**The real problem is at line 1474** in `produce_blocks()` on the pool:

```rust
// lockfree_producer.rs line 1474
if let Some(block) = producer.produce_block().await {
```

This call goes through the handle path (line 861) which has the 30s timeout.
After timeout, it returns `None`, but the **command loop task** is still stuck
on the inner `produce_block()`.  While stuck, the command loop cannot process
ANY other commands -- `ShouldProduce`, `GetHeight`, `SetLatestBlock`, etc. all
queue up in the bounded channel.

If the internal `produce_block()` hangs indefinitely (see hang points in
Section 5), the command loop is **permanently dead**.  The `ShouldProduce`
queries from the production loops will timeout after 30s, returning
`Err(TimedOut)`.  With both producers' command loops hung, the pool's
`should_produce()` sees all producers unhealthy -- but the threshold for the
`ProducersUnhealthy` crash may not trigger if only one of two producers is
stuck.

What can cause `produce_block()` to hang indefinitely inside the command loop:
- RocksDB read contention (get_qblock_by_height during parent lookup)
- DAG-Knight consensus calculation on a deep graph
- VDF verification stalling on CPU
- Memory pressure causing page faults in the block builder

**Fix**: Add a 30s `tokio::time::timeout` around the `produce_block()` call
**inside the command loop** (already present -- see lines 317-326 and 610-627).
The timeout IS there.  The actual gap is that after the command-loop timeout
fires and returns `None`, the **underlying `BlockProducer::produce_block()`
future is dropped but may not be cancel-safe**.  If it holds a lock or is
blocked on a synchronous operation wrapped in an async fn, the tokio task
itself is stuck on the `.await` point and the timeout never fires because
the future never yields.

The true fix: wrap the synchronous portions of `BlockProducer::produce_block()`
(in `block_producer.rs` line 534) in `tokio::task::spawn_blocking()` so they
can be cancelled by the timeout.  Alternatively, ensure all RocksDB calls in
the production path go through `spawn_blocking`.

### 3.2 production_in_progress zombie flag (HIGH)

**Location**: `crates/q-api-server/src/lockfree_producer.rs`, lines 1283-1284, 1439-1453

The pool serializes concurrent `produce_blocks()` calls with an `AtomicBool`:

```rust
// line 1439
if self.production_in_progress.compare_exchange(
    false, true, Ordering::SeqCst, Ordering::SeqCst
).is_err() {
    return Vec::new();  // Another call in progress
}
let _guard = ProductionGuard(&self.production_in_progress);
```

A `ProductionGuard` RAII struct clears the flag on drop (line 1448-1451).  This
is correct **if the task runs to completion or panics** (Drop runs on unwind).
However, if the tokio task is **cancelled** (e.g., by `tokio::select!` or
`JoinHandle::abort()`), Drop for the guard is guaranteed to run -- but only
if the future is at an `.await` point.  If the future is blocked in a
synchronous section, cancellation is deferred.

More critically, there is a `production_in_progress_since` timestamp (line
1284) that is set at line 1460 but **never checked**.  No code reads this
timestamp to detect staleness.  If the flag gets stuck `true` (e.g., due to
a panic in an unsafe block or a synchronous hang), all future
`produce_blocks()` calls return empty forever.

**Fix**: Add a staleness check before the CAS:
```rust
let since = self.production_in_progress_since.load(Ordering::SeqCst);
let now_ms = /* current epoch ms */;
if self.production_in_progress.load(Ordering::SeqCst)
    && since > 0
    && now_ms.saturating_sub(since) > 120_000
{
    warn!("production_in_progress stuck for >120s, force-clearing");
    self.production_in_progress.store(false, Ordering::SeqCst);
}
```

### 3.3 Crown-Ash game tick blocks tokio worker thread (HIGH)

**Location**: `crates/q-api-server/src/main.rs`, lines 21887-21910

Every 10 blocks (`BLOCKS_PER_TURN = 10` in `crown-ash-types/src/world.rs`),
the game tick scheduler acquires a write lock on `crown_ash_state` and runs:

```rust
// main.rs line 21897
let mut summary = tokio::task::block_in_place(|| {
    crown_ash_sim::tick(&mut game.world, &block_hash)
});
```

`block_in_place()` tells tokio "this closure will block -- move other tasks
off this worker thread."  This is correct for preventing worker starvation
in **multi-threaded** runtimes.  However:

1. The **write lock** (`crown_ash_state.write().await` at line 21889) is held
   across the entire `block_in_place` call.  Any other task that reads
   `crown_ash_state` (API handlers serving game state) will be blocked.

2. `crown_ash_sim::tick()` runs combat resolution, economy simulation, and
   population updates.  At scale (many provinces, factions, characters), this
   can take 100ms+ of CPU time.

3. While `block_in_place` prevents tokio worker starvation, it still occupies
   one worker thread for the duration.  On a system with few workers (e.g.,
   `worker_threads = 2`), this reduces available parallelism by 50%.

4. The RwLock on `crown_ash_state` is a **tokio RwLock**.  If the producer
   task or any handler holds a read lock and `block_in_place` holds the write
   lock, we get priority inversion.

**Why this contributes to the stall**: If the producer task's
`produce_block()` path reads `crown_ash_state` (unlikely but possible through
transaction processing), it will wait on the write lock held by `block_in_place`.
More likely, the general tokio worker starvation during a long tick can delay
the producer's `.await` resumption enough to cascade with issues 3.1 and 3.2.

**Fix**: Use `tokio::task::spawn_blocking()` instead of `block_in_place()`.
This moves the simulation to a dedicated blocking thread pool rather than
occupying a worker thread:

```rust
let world_snapshot = std::mem::take(&mut game.world);
let summary = tokio::task::spawn_blocking(move || {
    let mut w = world_snapshot;
    let s = crown_ash_sim::tick(&mut w, &block_hash);
    (w, s)
}).await.expect("spawn_blocking panicked");
game.world = summary.0;
let summary = summary.1;
```

This requires `World` to be `Send`, which it should be.

---

## 4. Why Previous Fixes (v10.2.9) Didn't Help

The v10.2.9 fix addressed `current_height_atomic` drift caused by dedup race
conditions in 4 block-save paths plus a reconciliation off-by-one.  That was
a **different failure mode** with a different symptom profile:

| Attribute | v10.2.9 stall (atomic drift) | This stall (task hang) |
|-----------|------------------------------|------------------------|
| `SHOULD_PRODUCE` logs | Present (YES with high age) | Absent |
| Duration | 20-30 minutes | 6+ hours |
| Automatic recovery | Reconciliation (if not off-by-one) | Never |
| Manual recovery | Not needed (self-heals) | Restart required |
| Root cause | Stale atomic value | Hung tokio task |
| Error logs | DEDUP-FIX messages | None |

Both bugs produce the same user-visible symptom (no new blocks), which is why
they were initially conflated.  They are independent and can occur separately.

---

## 5. Complete Hang Point Inventory

All `.await` calls in the block production critical path that could hang
indefinitely:

| Call site | File:Line | Timeout? | Hang mechanism |
|-----------|-----------|----------|----------------|
| `producer.produce_block()` inner | `block_producer.rs:534` | No (async fn, but may contain sync code) | RocksDB reads, DAG-Knight, VDF |
| `command_rx.recv()` | `lockfree_producer.rs:296` | No (waits for next command) | Normal -- wakes on send |
| `producer.produce_block()` in command loop | `lockfree_producer.rs:319/612` | 30s timeout | Timeout fires but future may not be cancel-safe |
| `crown_ash_state.write().await` | `main.rs:21889` | No | Priority inversion with block_in_place |
| `storage_engine.get_qblock_by_height()` | various | No explicit timeout | RocksDB stall under compaction |
| `wallet_balances.write().await` | `main.rs:3272+` | No | Contention from many writers |
| `block_writer_tx.send()` | main.rs (block save) | Bounded channel backpressure | Channel full if writer is slow |

---

## 6. The Fixes (3 Changes)

### Fix 1: Cancel-safe produce_block via spawn_blocking

**Before** (`block_producer.rs:534`): `produce_block()` is an async fn that
contains synchronous RocksDB calls and CPU-intensive DAG-Knight computation.
These do not yield to the tokio runtime, making the 30s timeout in the command
loop ineffective (timeout requires yield points).

**After**: Wrap the synchronous core of `produce_block()` in
`spawn_blocking()`.  This makes the future cancel-safe because `spawn_blocking`
returns a `JoinHandle` that yields immediately, allowing the timeout to fire
and drop the handle.

**Safety**: The spawned closure runs to completion on the blocking thread pool
even after the handle is dropped.  This is safe because `produce_block()` does
not hold any locks that would be orphaned -- it operates on owned data.

### Fix 2: Staleness check for production_in_progress

**Before** (`lockfree_producer.rs:1439`): CAS fails silently if flag is stuck.
`production_in_progress_since` is written but never read.

**After**: Before the CAS, check if the flag has been true for >120s.  If so,
log a warning and force-clear it.  120s is conservative (normal production
takes <1s).

**Safety**: The force-clear could theoretically race with a legitimate
production that is simply slow.  At 120s, this is extremely unlikely -- any
production taking >120s is itself stuck.  The guard pattern ensures the flag
is re-cleared when the stuck task eventually unblocks.

### Fix 3: spawn_blocking for Crown-Ash tick

**Before** (`main.rs:21897`): `block_in_place()` occupies a tokio worker
thread during game simulation.

**After**: `spawn_blocking()` moves the work to the dedicated blocking pool.
The write lock is released before spawning (world is moved out), and
reacquired after the spawn completes.

**Safety**: Requires `World` to be `Send`.  The simulation is deterministic
and operates on owned data, so there are no shared-state concerns.

---

## 7. Questions for Reviewers

1. **Timeout duration**: The command loop already has a 30s timeout on
   `produce_block()`.  The issue is that the timeout may not fire if the
   future does not yield.  Is `spawn_blocking` the right approach, or should
   we restructure `BlockProducer::produce_block()` to insert explicit yield
   points (`tokio::task::yield_now()`) between synchronous sections?

2. **Cancellation side effects**: If `spawn_blocking` runs the block builder
   to completion after the handle is dropped, we produce a block that is never
   saved.  Is this acceptable (wasted CPU but no correctness issue), or should
   we add an `AbortHandle` to kill the blocking task?

3. **block_in_place vs spawn_blocking for Crown-Ash**: `block_in_place` has
   lower overhead (no thread spawn, no data transfer).  `spawn_blocking`
   requires `Send` and has ~5us scheduling overhead.  Given the tick runs every
   10 blocks (~10s), is the overhead acceptable?  Does `World` implement
   `Send`?

4. **RocksDB in spawn_blocking**: Should ALL RocksDB calls in `kv.rs` be
   wrapped in `spawn_blocking`?  RocksDB can stall for 100ms+ during L0
   compaction.  This would be a large refactor but would make all async code
   truly cancel-safe.

5. **Tokio runtime health monitoring**: Is there a way to add a watchdog
   that detects when all tokio worker threads are blocked?  For example, a
   dedicated thread that periodically spawns a trivial task and measures
   latency.  If latency exceeds 500ms, emit a warning.

6. **production_in_progress_since threshold**: Is 120s the right staleness
   threshold?  Normal production takes <1s.  VDF verification on slow hardware
   could take 5-10s.  120s has a wide margin.  Should it be shorter (30s)?

---

## 8. Verification Plan

### Fix 1 (spawn_blocking produce_block)

1. Add structured logging at entry/exit of `produce_block()` with elapsed time.
2. Deploy to Alpha Docker canary.
3. Simulate RocksDB contention: run `db_bench --benchmarks=fillrandom` in
   parallel with block production.
4. Verify that the 30s command-loop timeout fires and the command loop
   recovers (processes subsequent `ShouldProduce` commands).
5. Monitor for 48 hours on canary.  The stall should not recur.

### Fix 2 (staleness check)

1. Unit test: set `production_in_progress` to true, set `since` to 130s ago,
   call `produce_blocks()`, verify flag is cleared and blocks are produced.
2. Integration test: hang a producer (inject sleep), verify the pool recovers
   after 120s.
3. Monitor production logs for "force-clearing" messages -- these indicate
   the fix is activating.

### Fix 3 (spawn_blocking Crown-Ash)

1. Verify `World` implements `Send` (`cargo check` after the change).
2. Add timing instrumentation to the tick: log wall-clock duration.
3. Run load test with many provinces/factions to stress the simulation.
4. Verify tokio worker utilization does not drop during ticks (use
   `tokio-console` or custom metrics).
5. Compare block production latency histograms before/after the change.

### Combined verification

After all three fixes are deployed to canary:
1. Let the node run for 72 hours without restart.
2. Compare block production gap histogram with pre-fix baseline.
3. Verify zero gaps >60s in the block height timeline.
4. Check journal for any "TIMED OUT" or "force-clearing" messages that
   indicate the fixes are actively preventing stalls.
