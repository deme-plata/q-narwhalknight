# BlockWriter Deadlock Technical Review for AI Analysis

**Document Version:** 1.0
**Date:** 2025-11-11
**Severity:** CRITICAL - Production Blocker
**Status:** ACTIVE BUG - Occurs after ~20-30 minutes of operation

---

## Executive Summary

The Q-NarwhalKnight blockchain experiences a **recurring BlockWriter deadlock** that causes block production to stall after approximately 20-30 minutes of operation. While blocks are created by lock-free producers, they fail to be persisted to the RocksDB database, resulting in the blockchain height freezing despite continued mining activity.

**Key Observation:** This bug is **NOT a database corruption issue**. The Phase 10 database durability hardening (sync=true enforcement) is working correctly - no phantom writes or data loss occurs. The problem is a **concurrency deadlock** in the block writing code path.

---

## System Architecture Overview

### Technology Stack
- **Language:** Rust 1.70+
- **Database:** RocksDB 0.22
- **Consensus:** Lock-free multi-producer block creation (8 parallel producers)
- **Storage:** Async block writer with queue-based persistence

### Key Components

```
┌─────────────────────────────────────────────────────────┐
│                  Lock-Free Producers                     │
│              (8 parallel producers)                      │
└────────────┬────────────────────────────────────────────┘
             │ Creates QBlock structs
             ▼
┌─────────────────────────────────────────────────────────┐
│              BlockWriter Queue                           │
│         (async channel-based)                            │
└────────────┬────────────────────────────────────────────┘
             │ Sends SaveBlockRequest
             ▼
┌─────────────────────────────────────────────────────────┐
│           BlockWriter Worker Thread                      │
│    (receives requests, writes to RocksDB)                │
└────────────┬────────────────────────────────────────────┘
             │ RocksDB put() operations
             ▼
┌─────────────────────────────────────────────────────────┐
│                  RocksDB Database                        │
│         (sync=true, hardened durability)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Symptom Description

### Observable Behavior

**Phase 1 - Normal Operation (0-23 minutes):**
```
2025-11-11T07:08:54.634121Z  INFO q_api_server::lockfree_producer: 🔍 [LOCK-FREE SYNC] Found highest block at height 2305 in storage
2025-11-11T07:08:54.706539Z  INFO q_api_server::lockfree_producer: ✅ Producer #6: Created block at height 2306
2025-11-11T07:08:54.706551Z  INFO q_api_server::lockfree_producer: 🎉 Lock-free producer #6 created block at height 2306
2025-11-11T07:08:54.707059Z  INFO q_storage::block_writer: 💾 Saving QBlock at height 2306 with hash 21309b56d70a7fa3
```

**Phase 2 - Deadlock (after ~23 minutes):**
```
2025-11-11T07:16:00.512130Z  INFO q_api_server::block_producer: 🏗️  Producing block: height=2848, solutions=100
2025-11-11T07:16:00.512540Z  INFO q_api_server::lockfree_producer: 🎉 Lock-free producer #6 created block at height 2848
2025-11-11T07:16:00.894943Z  INFO q_api_server::block_producer: 🏗️  Producing block: height=2848, solutions=100
2025-11-11T07:16:00.895172Z  INFO q_api_server::lockfree_producer: 🎉 Lock-free producer #7 created block at height 2848
[... blocks continue to be created at height 2848 but NEVER saved ...]
```

**Key Indicator:**
- ✅ Blocks are **created** repeatedly (log shows "Created block at height 2848")
- ❌ NO "💾 Saving QBlock" messages appear
- Last successful save: `2025-11-11T07:15:57.710673Z  INFO q_storage::block_writer: 💾 Saving QBlock at height 2849`

### Timeline
1. **T+0min:** Service starts, blocks produced normally
2. **T+23min:** Last successful block save (height 2849)
3. **T+23min:** BlockWriter stops processing queue
4. **T+24min+:** Producers create blocks but queue is never drained

---

## Code Locations

### Primary Files Involved

#### 1. Block Writer Implementation
**File:** `crates/q-storage/src/block_writer.rs`

**Key Structure:**
```rust
pub struct BlockWriter {
    db: Arc<DB>,
    request_rx: mpsc::UnboundedReceiver<SaveBlockRequest>,
    // ... other fields
}

pub struct SaveBlockRequest {
    pub block: QBlock,
    pub response_tx: oneshot::Sender<Result<(), String>>,
}

impl BlockWriter {
    pub async fn run(mut self) {
        while let Some(request) = self.request_rx.recv().await {
            // THIS IS WHERE THE DEADLOCK OCCURS
            // The receiver stops receiving after ~23 minutes
        }
    }
}
```

**Location:** Line ~50-150 (approximate)

#### 2. Lock-Free Producer
**File:** `crates/q-api-server/src/lockfree_producer.rs`

**Sends blocks to writer:**
```rust
// Creates block successfully
let block = create_block(...)?;

// Sends to BlockWriter queue
self.block_writer_tx.send(SaveBlockRequest {
    block: block.clone(),
    response_tx,
})?;

// Waits for response (MAY BE DEADLOCKING HERE)
response_rx.await??;
```

**Location:** Line ~200-300 (approximate)

#### 3. Block Producer
**File:** `crates/q-api-server/src/block_producer.rs`

**Creates blocks:**
```rust
pub async fn produce_block(&mut self) -> Result<QBlock> {
    let block = QBlock {
        header: BlockHeader {
            height: self.current_height + 1,
            // ... other fields
        },
        // ... block data
    };

    // Returns block to lock-free producer
    Ok(block)
}
```

**Location:** Line ~300-400 (approximate)

---

## Hypotheses for Root Cause

### Hypothesis 1: Channel Backpressure Deadlock ⭐ MOST LIKELY

**Theory:** The `mpsc::UnboundedReceiver` in BlockWriter may have a hidden capacity limit or the receiving loop is blocked on a synchronous operation.

**Evidence:**
- Blocks stop being saved exactly
- No errors in logs (silent failure)
- Channel appears to stop receiving messages

**Potential Causes:**
```rust
// BlockWriter::run() loop may be blocked on:
while let Some(request) = self.request_rx.recv().await {
    // If this RocksDB operation blocks forever...
    self.db.put(...)? // <-- POTENTIAL DEADLOCK HERE

    // Or if response channel is full/blocked...
    request.response_tx.send(Ok(()))?; // <-- OR HERE
}
```

**Fix Direction:**
- Add timeout to `recv().await`
- Use `tokio::select!` with timeout
- Log when entering/exiting critical sections

### Hypothesis 2: RocksDB Write Stall

**Theory:** RocksDB internal write buffer fills up and blocks indefinitely.

**Evidence:**
- Happens after consistent time period (~23 minutes)
- Approximately 2849 blocks = ~142MB of data (if 50KB/block)
- RocksDB default memtable size: 64MB

**Potential Causes:**
```rust
// RocksDB WriteOptions with sync=true may cause:
let mut write_opts = WriteOptions::default();
write_opts.set_sync(true); // Forces fsync on every write
write_opts.disable_wal(false);

// If OS disk write cache is full, this blocks FOREVER
db.put_opt(&key, &value, &write_opts)?;
```

**Fix Direction:**
- Check RocksDB write stall stats
- Increase memtable size
- Add background flush monitoring
- Use non-blocking writes with manual sync

### Hypothesis 3: Async Runtime Starvation

**Theory:** Tokio runtime has too few threads or the BlockWriter worker is blocking the executor.

**Evidence:**
- Multiple producers (8) competing for runtime resources
- Heavy mining activity (3520 KH/s on one miner)

**Potential Causes:**
```rust
// If BlockWriter is spawned on single-threaded runtime:
tokio::spawn(async move {
    block_writer.run().await; // Blocking here affects all tasks
});

// Or if using blocking operations in async context:
async fn save_block(&self, block: QBlock) {
    // This blocks the executor!
    std::thread::sleep(Duration::from_secs(1));
}
```

**Fix Direction:**
- Use `tokio::task::spawn_blocking` for RocksDB operations
- Increase Tokio worker threads
- Separate runtime for BlockWriter

### Hypothesis 4: Memory Leak/Exhaustion

**Theory:** Accumulated state in BlockWriter causes OOM or swap thrashing.

**Evidence:**
- Process RSS: 6GB (seems high for ~2849 blocks)
- May be accumulating unprocessed responses

**Monitoring Needed:**
```bash
# Check memory growth over time
ps -p 2450171 -o rss,vsz,%mem
```

---

## Diagnostic Questions for AI

### About the BlockWriter Queue

1. **Is `mpsc::UnboundedReceiver` truly unbounded in Tokio?**
   - Can it fill up and block senders?
   - What happens when receiver is slow?

2. **What happens if `response_tx.send()` fails?**
   - Does it panic the BlockWriter loop?
   - Is there error handling?

3. **How does Tokio schedule the BlockWriter::run() future?**
   - Can it be starved by other tasks?
   - Does it need dedicated thread?

### About RocksDB Behavior

4. **Can `db.put_opt()` with `sync=true` block indefinitely?**
   - What causes write stalls?
   - How to detect and recover?

5. **What are RocksDB compaction settings?**
   - Could compaction block writes?
   - Are there stall triggers we should configure?

6. **Does RocksDB have internal queue limits?**
   - Write buffer size limits?
   - Pending compaction limits?

### About Concurrency

7. **Are there hidden locks in this code path?**
   - Arc<DB> internal locks?
   - Tokio runtime locks?

8. **What is the exact execution order when deadlock occurs?**
   - Which task is waiting on which?
   - Can we detect with async-backtrace?

9. **Is there a race condition between:**
   - Block creation (lockfree_producer)
   - Block sending (channel send)
   - Block saving (BlockWriter worker)
   - Block confirmation (response channel)

---

## Attempted Fixes (Historical Context)

### What Has Been Tried

1. **✅ Database Durability Hardening (Phase 10)**
   - Result: No phantom writes, but deadlock persists
   - Conclusion: Not a data corruption issue

2. **✅ Lock-Free Producer Architecture**
   - Result: Producers work fine, writer is the bottleneck
   - Conclusion: Producer-side is healthy

3. **⚠️ Restart Service (Temporary Workaround)**
   - Result: Works for another 20-30 minutes
   - Conclusion: Bug is deterministic, time/load-based

### What Has NOT Been Tried

1. **Add Timeouts to BlockWriter Loop**
   ```rust
   use tokio::time::{timeout, Duration};

   while let Ok(Some(request)) = timeout(Duration::from_secs(30), self.request_rx.recv()).await {
       // Process request with timeout
   }
   ```

2. **Spawn BlockWriter on Dedicated Thread**
   ```rust
   std::thread::spawn(move || {
       let rt = tokio::runtime::Runtime::new().unwrap();
       rt.block_on(async move {
           block_writer.run().await
       });
   });
   ```

3. **Add Detailed Logging**
   ```rust
   while let Some(request) = self.request_rx.recv().await {
       info!("📥 BlockWriter received request for height {}", request.block.header.height);

       let start = Instant::now();
       self.db.put_opt(...)?;
       info!("✅ RocksDB write took {:?}", start.elapsed());

       request.response_tx.send(Ok(()))?;
       info!("📤 Response sent successfully");
   }
   ```

4. **Check RocksDB Statistics**
   ```rust
   let stats = self.db.property_value("rocksdb.stats")?;
   info!("RocksDB stats: {}", stats);
   ```

5. **Use Non-Blocking RocksDB Writes**
   ```rust
   // Write to RocksDB without waiting for fsync
   let mut write_opts = WriteOptions::default();
   write_opts.set_sync(false); // Don't wait for disk
   write_opts.disable_wal(false);

   // Manual sync every N blocks
   if block.header.height % 100 == 0 {
       self.db.flush()?;
   }
   ```

---

## Request to AI Assistants

### Primary Questions

**Q1: What is causing the BlockWriter channel to stop receiving messages?**
- Is it a Tokio runtime issue?
- Is it RocksDB blocking?
- Is it a memory/resource exhaustion?

**Q2: How can we diagnose the exact blocking point?**
- What logging should we add?
- What Rust profiling tools work for async code?
- Can we use tokio-console?

**Q3: What is the recommended pattern for async database writes in Rust?**
- Should BlockWriter use `spawn_blocking`?
- Should we use a dedicated runtime?
- Is there a better channel type?

### Specific Code Review Requests

**Please review these code patterns:**

1. **Channel Pattern (is this correct?):**
```rust
// Sender side (lockfree_producer.rs)
let (response_tx, response_rx) = oneshot::channel();
self.block_writer_tx.send(SaveBlockRequest { block, response_tx })?;
let result = response_rx.await?; // DOES THIS DEADLOCK?

// Receiver side (block_writer.rs)
while let Some(request) = self.request_rx.recv().await {
    // Process...
    request.response_tx.send(Ok(()))?; // DOES THIS BLOCK?
}
```

2. **RocksDB Write Pattern (is this safe?):**
```rust
let mut write_opts = WriteOptions::default();
write_opts.set_sync(true); // CAN THIS BLOCK FOREVER?

// In async context - is this wrong?
self.db.put_opt(&key, &value, &write_opts)?;
```

3. **Async Spawning (is this correct?):**
```rust
// main.rs
tokio::spawn(async move {
    block_writer.run().await
});

// Should this be instead?
tokio::task::spawn_blocking(move || {
    // Run BlockWriter on dedicated thread?
});
```

### Debugging Assistance Needed

**Please suggest:**

1. **Logging Strategy**
   - What to log at each stage?
   - How to detect the blocking point?
   - Structured logging format?

2. **Monitoring Metrics**
   - What RocksDB metrics to expose?
   - What Tokio metrics to track?
   - What system metrics are relevant?

3. **Testing Approach**
   - How to reproduce in <23 minutes?
   - How to stress test the channel?
   - How to simulate slow RocksDB?

---

## System Constraints

### Must Maintain

1. **Lock-Free Producers:** Cannot add locks/mutexes
2. **Database Durability:** sync=true must stay for data safety
3. **Performance:** Must handle 48k+ TPS
4. **Async Runtime:** Using Tokio (cannot switch to blocking)

### Can Change

1. Channel type (currently `mpsc::unbounded`)
2. BlockWriter spawning strategy
3. RocksDB write options (within durability constraints)
4. Runtime configuration

---

## Success Criteria

**The fix is successful when:**

1. ✅ Blockchain runs for **>24 hours** without height stall
2. ✅ All blocks are persisted to RocksDB
3. ✅ No phantom writes or data corruption
4. ✅ Performance maintained (>1000 blocks/hour)
5. ✅ Resource usage stable (no memory leaks)

---

## Additional Context

### Environment
- **OS:** Debian Linux 6.1.0-37-amd64
- **CPU:** 60 cores available
- **RAM:** 96GB total, ~6GB used by process
- **Disk:** SSD, ext4 filesystem
- **RocksDB Version:** 0.22
- **Tokio Version:** 1.40+

### Recent Logs (Last Successful Save)
```
Nov 11 08:15:57 vmi2628966.contaboserver.net q-api-server[2450171]: 2025-11-11T07:15:57.710673Z  INFO q_storage::block_writer: 💾 Saving QBlock at height 2849 with hash 82b212a3e65ba5e4
```

### Monitoring Command
```bash
journalctl -u q-api-server -f | grep -E "💾 Saving|Created block|BlockWriter"
```

---

## Files to Review

### Priority 1 (Critical)
1. `crates/q-storage/src/block_writer.rs` - BlockWriter implementation
2. `crates/q-api-server/src/lockfree_producer.rs` - Block sending logic
3. `crates/q-api-server/src/main.rs` - Runtime setup

### Priority 2 (Important)
4. `crates/q-storage/src/kv.rs` - RocksDB configuration
5. `crates/q-api-server/src/block_producer.rs` - Block creation

### Priority 3 (Context)
6. `Cargo.toml` - Dependency versions
7. `crates/q-types/src/block.rs` - Block structure

---

## Contact Information

**For questions or clarification:**
- This document is self-contained for AI analysis
- All code is available in the repository
- Logs can be provided on request

**Expected AI Response Format:**
1. Root cause diagnosis with confidence level
2. Specific code fixes with line-by-line changes
3. Testing strategy to validate fix
4. Monitoring recommendations for future detection

---

**Document End**

**Last Updated:** 2025-11-11 08:18 CET
**Status:** ACTIVE INVESTIGATION
**Severity:** CRITICAL
