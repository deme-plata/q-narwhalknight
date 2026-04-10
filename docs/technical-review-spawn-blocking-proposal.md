# Technical Review: Production Loop Stall — spawn_blocking Proposal

**Date:** 2026-04-10
**Status:** PROPOSAL — NOT YET IMPLEMENTED. Seeking external AI review before any code changes.
**Severity:** Liveness issue (not safety). No funds at risk. Restart fixes it in 90 seconds.
**Market cap:** ~$920M

---

## 1. The Problem

The Epsilon production node (48-core, 64GB RAM, 10Gbit) silently stops producing blocks every few hours. The process stays alive, P2P and SSE continue, but the block production loop stops executing entirely — zero log output from the producer for 6+ hours.

`systemctl restart` immediately fixes it. systemd's ProducersUnhealthy watchdog also auto-restarts after detecting 30s of no production.

**This is annoying, not catastrophic.** No funds lost, no balances corrupted, no chain forks. When it stalls, mining solutions queue up and are processed after restart.

---

## 2. What We Know

### Confirmed facts:
- Process stays alive (PID unchanged, P2P/SSE/API all working)
- The block production loop (main.rs ~line 15650) stops executing entirely
- No `SHOULD_PRODUCE`, `PRODUCE_BLOCK`, or `BLOCK PRODUCED` log entries during stall
- No `ProducersUnhealthy` crash — the production loop just goes completely silent
- Not a connection pileup (connections stable at ~200-600)
- Not the height atomic bug (fetch_max fix deployed, no DEDUP-FIX events)
- Not the production_in_progress zombie flag (120s force-clear deployed, no zombie events)
- Not the Crown-Ash game tick alone (block_in_place deployed, stall still occurs)
- Journal timestamps lag by 2+ hours during stall, suggesting severe log buffer pressure

### What we suspect but haven't proven:
- Synchronous RocksDB calls inside async code block tokio worker threads
- When enough workers are blocked (compaction + block-pack serving + balance writes), the production loop's tokio task never gets scheduled
- The `sync_from_storage()` function called before every production cycle makes 3+ RocksDB calls without individual timeouts
- The `global_write_lock` (tokio::sync::Mutex) in the storage engine can be held across slow DB operations

### What we DON'T know:
- Whether it's RocksDB specifically, or some other blocking operation
- Whether 48 worker threads can truly ALL be starved simultaneously
- Whether the fix should be in the RocksDB wrapper (kv.rs), the storage engine (lib.rs), or the production loop (main.rs/lockfree_producer.rs)
- Whether changing execution context for storage calls introduces lock ordering or lifetime issues

---

## 3. Current Mitigations (Already Deployed)

| Mitigation | Version | Effect |
|------------|---------|--------|
| ProducersUnhealthy watchdog | v10.2.8 | Auto-restarts process after 30s timeout on producer health check |
| Producer task 30s timeout | v10.2.9 | Prevents individual produce_block() from hanging the command loop |
| Zombie flag detection (120s) | v10.2.9 | Force-clears production_in_progress if stuck |
| Crown-Ash block_in_place | v10.2.9 | Prevents game simulation from blocking tokio workers |
| Height atomic fetch_max | v10.2.9 | Prevents height drift from dedup races |
| q-flux connection limit (512) | v10.2.9 | Prevents connection pileup overwhelming the API server |

**Net result:** The stall still occurs, but recovery is automatic (systemd restart within 30-60s). Downtime per incident: ~90 seconds. Frequency: every 2-8 hours.

---

## 4. Proposed Fix Options (For AI Review)

### Option A: Production Loop Heartbeat Watchdog (LOWEST RISK)

Add a dedicated monitoring task that detects when the production loop stops executing.

**How it works:**
1. The production loop updates an `AtomicU64` timestamp at the top of every iteration
2. A separate tokio task checks this timestamp every 10 seconds
3. If the timestamp is >60s stale AND blocks are not being produced:
   - Log a critical warning with full diagnostic info
   - Force-clear production_in_progress flag
   - Attempt to re-trigger production by sending a wake signal
4. If >120s stale: trigger process exit for systemd restart (same as ProducersUnhealthy)

**Pros:** Zero risk to existing code. Purely additive. Reduces stall window from hours to 60 seconds.
**Cons:** Doesn't fix root cause. Just detects and recovers faster.

**Implementation:**
```rust
// In main.rs, near the production loop
let production_heartbeat = Arc::new(AtomicU64::new(0));
let heartbeat_clone = production_heartbeat.clone();

// Inside the production loop, at the top of every iteration:
production_heartbeat.store(now_epoch_ms(), Ordering::SeqCst);

// Separate watchdog task:
tokio::spawn(async move {
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        let last = heartbeat_clone.load(Ordering::SeqCst);
        let now = now_epoch_ms();
        let age = now.saturating_sub(last);
        if age > 60_000 {
            error!("🚨 [WATCHDOG] Production loop heartbeat stale for {}s!", age / 1000);
            // Force-clear production flag, log diagnostics
        }
        if age > 120_000 {
            error!("🚨 [WATCHDOG] Production loop dead for >120s — triggering restart");
            std::process::exit(1); // systemd restarts
        }
    }
});
```

### Option B: Timeouts on sync_from_storage() Internal Calls (MEDIUM RISK)

Wrap each RocksDB call inside `sync_from_storage()` with individual 5-second timeouts.

**How it works:**
```rust
// Current (no timeout):
let height = storage.get_highest_contiguous_block().await?;

// Proposed:
let height = match tokio::time::timeout(
    Duration::from_secs(5),
    storage.get_highest_contiguous_block()
).await {
    Ok(Ok(h)) => h,
    Ok(Err(e)) => return Err(e.into()),
    Err(_) => {
        error!("🚨 sync_from_storage: RocksDB call timed out after 5s");
        return Err(anyhow!("RocksDB timeout"));
    }
};
```

**Pros:** Prevents the production loop from blocking indefinitely on a single slow RocksDB call.
**Cons:** 
- `tokio::time::timeout` only works if the tokio timer can fire — if ALL worker threads are blocked, the timeout itself won't fire either
- May mask real storage errors
- Doesn't fix the underlying blocking-on-async issue

### Option C: spawn_blocking for RocksDB Calls (HIGHEST IMPACT, HIGHEST RISK)

Move synchronous RocksDB operations to tokio's blocking thread pool.

**How it works:**
```rust
// Current (blocks tokio worker):
pub async fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
    let cf_handle = self.db.cf_handle(cf).ok_or(...)?;
    Ok(self.db.get_cf(&cf_handle, key)?) // ← synchronous RocksDB call on tokio thread
}

// Proposed:
pub async fn get(&self, cf: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
    let db = self.db.clone();
    let cf_name = cf.to_string();
    let key_owned = key.to_vec();
    tokio::task::spawn_blocking(move || {
        let cf_handle = db.cf_handle(&cf_name).ok_or(...)?;
        Ok(db.get_cf(&cf_handle, &key_owned)?)
    }).await?
}
```

**Pros:** Properly isolates blocking I/O from the async runtime. Industry-standard fix. Tokio documentation explicitly recommends this.
**Cons:**
- Requires changing every RocksDB call in kv.rs (30+ methods)
- Adds overhead: thread switch + allocation per call (~5-10µs)
- May introduce lifetime issues (owned vs borrowed data)
- Lock ordering may change (tokio::sync::Mutex held across spawn_blocking boundary)
- **Risk of introducing new bugs in a 20,000+ line codebase**
- Needs extensive testing before production deployment

---

## 5. Recommended Approach

**Deploy in order of risk, not impact:**

1. **Option A first (watchdog)** — Deploy immediately. Zero risk, reduces stall window to 60s.
2. **Option B second (timeouts)** — Deploy after 1 week of watchdog data to understand stall patterns.
3. **Option C last (spawn_blocking)** — Deploy only after thorough review by multiple AI reviewers and testing on Delta/Beta for 1+ week.

---

## 6. Questions for External AI Reviewers

### For DeepSeek:

**Question 1 — Is RocksDB actually the cause?**

> We have a tokio multi-threaded runtime with 48 worker threads on a 48-core server. The production loop (a tokio::spawn'd async task) silently stops executing for 6+ hours. Other tasks (P2P networking, SSE streaming, HTTP API) continue working. The production loop calls `sync_from_storage()` which makes 3 RocksDB calls (get, get, get_by_height) that are implemented as synchronous `db.get_cf()` calls inside async functions.
>
> Can synchronous RocksDB calls on 48 worker threads cause the ENTIRE runtime to stall? Or would some threads always be free to run other tasks? What else could cause a single tokio task to stop being scheduled for 6+ hours while the runtime remains alive?

**Question 2 — sync_from_storage code review**

> Here is the `sync_from_storage` function from our lock-free producer pool. It's called before every block production cycle. It reads from RocksDB (via async functions that internally do synchronous DB calls) and updates producer state.
>
> [Paste sync_from_storage function here]
>
> What could cause this function to hang indefinitely? Consider: RocksDB compaction blocking reads, tokio mutex contention, the `get_height_consensus()` call that awaits responses from multiple producer tasks.

**Question 3 — Safest fix for a $1B chain**

> For a production blockchain with ~$1B market cap, we need to convert synchronous RocksDB calls (inside async functions) to properly isolated blocking operations. The codebase is 20,000+ lines of Rust with heavy use of `tokio::sync::Mutex`, `RwLock`, and `Arc<>` shared state.
>
> What is the safest deployment strategy? Should we:
> (a) Wrap individual RocksDB calls in spawn_blocking
> (b) Wrap entire functions like sync_from_storage in block_in_place
> (c) Create a dedicated RocksDB thread pool separate from tokio
> (d) Add a watchdog that detects stalls and auto-recovers
>
> What rollback plan should we have?

### For ChatGPT:

**Question 4 — Production loop analysis**

> Our block production loop runs as a tokio::spawn'd task. Every iteration it: (1) reads mining solutions from a sharded channel, (2) calls sync_from_storage() to update producer heights from RocksDB, (3) calls produce_blocks() which sends commands to producer tasks via channels, (4) saves produced blocks to RocksDB, (5) processes balance updates.
>
> The loop stops executing entirely — zero log output — for 6+ hours. The process stays alive. Other tokio tasks run fine.
>
> [Paste production loop code here — main.rs lines 15650-15900]
>
> What could cause this specific task to stop being scheduled? Consider:
> - All `.await` points that could hang
> - Channel operations that could block
> - Lock acquisitions that could deadlock
> - Interactions between this loop and other tasks

**Question 5 — Watchdog design**

> We want to add a heartbeat watchdog for our production loop. The loop updates an AtomicU64 timestamp every iteration. A separate task checks it every 10s. If stale >60s, it logs a warning. If >120s, it exits the process (systemd restarts).
>
> Is this design sound? Potential issues:
> - Could the watchdog task ALSO get starved? (It only does atomic loads and sleeps)
> - Is `std::process::exit(1)` safe from an async context?
> - Should we use a dedicated OS thread (std::thread) for the watchdog instead of a tokio task?

---

## 7. Current Workaround

The existing mitigation stack is solid:
- `ProducersUnhealthy` watchdog auto-exits after 30s of no production
- systemd `RestartSec=3` restarts the process
- Block production resumes in ~90 seconds (bootstrap + P2P reconnect)
- No funds at risk during stall — mining solutions queue and are processed after restart

**This is ugly but it works.** The chain has been running at $920M market cap with this workaround. The proper fix should be engineered carefully, not rushed.

---

## 8. What We're NOT Doing

- **NOT deploying spawn_blocking changes without external review** — too risky for production
- **NOT changing the RocksDB wrapper (kv.rs) without extensive testing** — 30+ methods, used everywhere
- **NOT panicking** — the chain is running, blocks are being produced, funds are safe
- **NOT rushing** — better to have a 60-second automatic recovery than a broken production deployment

---

*This document is for external AI review. Please share with DeepSeek and ChatGPT along with the relevant code sections. Their feedback will inform the implementation plan.*
