# AI Consensus Action Plan - Mining Stall Fix

**Date**: 2025-11-13 13:00 CET
**Contributors**: Kimi AI, ChatGPT, DeepSeek AI
**Consensus Level**: ✅ **STRONG AGREEMENT** on root cause and fixes

---

## 🎯 **CONSENSUS ROOT CAUSE**

All three AI systems agree on the primary issue:

### **RwLock Deadlock in Storage Layer (85-90% confidence)**

**Mechanism:**
```rust
// DEADLOCK SCENARIO:
// Block Producer Task              // API/Mining Endpoint Task
let db = storage.db.write().await;  let db = storage.db.read().await;
// Blocks on blocking RocksDB I/O   // Frequent height queries
db.put(key, value)?; // ← BLOCKS!   // Can't proceed, waiting for write
// Cannot release write lock         // Accumulating read waiters
// ↓ DEADLOCK ↓                      // ↓ DEADLOCK ↓
```

**Why This Matches Symptoms:**
1. ✅ Block producer freezes (stuck waiting for lock)
2. ✅ API/network remain functional (can still get read locks during writer wait)
3. ✅ Mining submissions queue (miners unaware of producer freeze)
4. ✅ Challenge becomes stale (producer can't update it)
5. ✅ Binary search storm during shutdown (trying to query height while write lock held)
6. ✅ Force-kill works (confirms deadlock, not infinite loop)
7. ✅ 4-8 hour periodicity (RocksDB compaction triggers long write operations)

---

## 🚀 **UNANIMOUS RECOMMENDATIONS**

### **Priority 1: IMMEDIATE (Next 24 Hours)**

#### **1.1 Deploy HeightState Cache** ✅ (ALREADY CODED)

**Status**: Code complete in v1.0.2-beta, needs deployment

**Impact**:
- Eliminates read lock contention on height queries (most frequent operation)
- Fixes binary search storm during shutdown
- Reduces stall frequency but **MAY NOT eliminate completely**

**Action**:
```bash
# Deploy v1.0.2-beta with HeightState
sudo systemctl stop q-api-server
sudo cp /opt/orobit/shared/q-narwhalknight/target/release/q-api-server \
    /opt/orobit/shared/q-narwhalknight/target/release/q-api-server.backup
# Wait for compilation if needed
sudo systemctl start q-api-server
```

#### **1.2 Implement Watchdog (Emergency Recovery)**

**Consensus**: All three AIs recommend task-level watchdog (not process panic)

```rust
// crates/q-api-server/src/main.rs - Add near startup
tokio::spawn(async move {
    let mut last_height = 0u64;
    let mut stuck_count = 0u32;

    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;

        let current = height_state.cached(); // Using HeightState cache!

        if current == last_height {
            stuck_count += 1;
            warn!("⚠️ Block producer may be stalled: height {} unchanged for {}s",
                  current, stuck_count * 30);

            if stuck_count >= 4 { // 2 minutes stuck
                error!("🚨 BLOCK PRODUCER STALLED - Triggering recovery");

                // Option A: Graceful - abort block producer task and restart
                block_producer_abort_handle.abort();
                tokio::time::sleep(Duration::from_secs(5)).await;
                // Restart block producer here...

                // Option B: Last resort - force process exit (systemd will restart)
                // std::process::exit(1);

                stuck_count = 0;
            }
        } else {
            stuck_count = 0;
            last_height = current;
        }
    }
});
```

**Expected Outcome**: Automatic recovery within 2 minutes instead of indefinite stall

---

### **Priority 2: SHORT-TERM (1 Week)**

#### **2.1 Move RocksDB Operations to Dedicated Thread**

**Consensus**: All three AIs strongly recommend isolating blocking I/O from async runtime

**Implementation Strategy:**

```rust
// crates/q-storage/src/kv.rs - New async-safe storage wrapper

use tokio::sync::mpsc;
use tokio::sync::oneshot;

pub struct AsyncStorageEngine {
    command_tx: mpsc::Sender<StorageCommand>,
}

enum StorageCommand {
    SaveBlock {
        block: QBlock,
        response: oneshot::Sender<Result<()>>,
    },
    GetBlock {
        height: u64,
        response: oneshot::Sender<Option<QBlock>>,
    },
    GetHeight {
        response: oneshot::Sender<u64>,
    },
}

impl AsyncStorageEngine {
    pub fn new(db_path: &str) -> Self {
        let (command_tx, mut command_rx) = mpsc::channel::<StorageCommand>(1000);

        // Dedicated blocking storage thread
        std::thread::spawn(move || {
            let db = RocksDBKV::new(db_path).unwrap();

            while let Some(cmd) = command_rx.blocking_recv() {
                match cmd {
                    StorageCommand::SaveBlock { block, response } => {
                        let result = db.put(
                            &format!("qblock:{}", block.header.height),
                            &bincode::serialize(&block).unwrap()
                        );
                        let _ = response.send(result.map_err(Into::into));
                    }
                    StorageCommand::GetBlock { height, response } => {
                        let result = db.get(&format!("qblock:{}", height))
                            .ok()
                            .and_then(|v| bincode::deserialize(&v).ok());
                        let _ = response.send(result);
                    }
                    StorageCommand::GetHeight { response } => {
                        let height = db.get(b"qblock:latest")
                            .ok()
                            .and_then(|v| bincode::deserialize(&v).ok())
                            .unwrap_or(0);
                        let _ = response.send(height);
                    }
                }
            }
        });

        Self { command_tx }
    }

    pub async fn save_qblock(&self, block: &QBlock) -> Result<()> {
        let (response_tx, response_rx) = oneshot::channel();

        self.command_tx.send(StorageCommand::SaveBlock {
            block: block.clone(),
            response: response_tx,
        }).await?;

        response_rx.await?
    }

    pub async fn get_qblock(&self, height: u64) -> Option<QBlock> {
        let (response_tx, response_rx) = oneshot::channel();

        self.command_tx.send(StorageCommand::GetBlock {
            height,
            response: response_tx,
        }).await.ok()?;

        response_rx.await.ok()?
    }
}
```

**Benefits**:
- ✅ **Zero async locks** - eliminates deadlock potential
- ✅ **Predictable latency** - blocking I/O isolated to dedicated thread
- ✅ **Better observability** - channel depth metrics show backlog
- ✅ **Graceful degradation** - bounded channel prevents memory explosion

**Migration Plan**:
1. Create `AsyncStorageEngine` wrapper (above code)
2. Replace `StorageEngine` usage in block producer only (test in isolation)
3. Gradually migrate other components
4. Remove old `RwLock`-based storage once migration complete

---

#### **2.2 Add Comprehensive Lock Instrumentation**

**Consensus**: All three AIs recommend detailed lock timing metrics

```rust
// crates/q-storage/src/lib.rs - Add lock timing wrapper

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

pub struct InstrumentedRwLock<T> {
    inner: Arc<RwLock<T>>,
    name: &'static str,
}

impl<T> InstrumentedRwLock<T> {
    pub fn new(value: T, name: &'static str) -> Self {
        Self {
            inner: Arc::new(RwLock::new(value)),
            name,
        }
    }

    pub async fn write(&self) -> InstrumentedWriteGuard<'_, T> {
        let start = Instant::now();
        let guard = self.inner.write().await;
        let wait_time = start.elapsed();

        if wait_time > Duration::from_millis(200) {
            warn!("⚠️ Lock '{}' wait time: {:?}", self.name, wait_time);
        }

        metrics::histogram!(
            "storage.lock.wait_time_ms",
            wait_time.as_millis() as f64,
            "lock_name" => self.name,
            "lock_type" => "write"
        );

        InstrumentedWriteGuard {
            inner: guard,
            name: self.name,
            acquired_at: Instant::now(),
        }
    }
}

pub struct InstrumentedWriteGuard<'a, T> {
    inner: tokio::sync::RwLockWriteGuard<'a, T>,
    name: &'static str,
    acquired_at: Instant,
}

impl<'a, T> Drop for InstrumentedWriteGuard<'a, T> {
    fn drop(&mut self) {
        let hold_time = self.acquired_at.elapsed();

        if hold_time > Duration::from_millis(200) {
            warn!("⚠️ Lock '{}' held for: {:?}", self.name, hold_time);
        }

        metrics::histogram!(
            "storage.lock.hold_time_ms",
            hold_time.as_millis() as f64,
            "lock_name" => self.name,
            "lock_type" => "write"
        );
    }
}

impl<'a, T> std::ops::Deref for InstrumentedWriteGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, T> std::ops::DerefMut for InstrumentedWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
```

**Usage**:
```rust
// Replace: pub db: Arc<RwLock<RocksDBKV>>
// With:    pub db: InstrumentedRwLock<RocksDBKV>

let storage = StorageEngine {
    db: InstrumentedRwLock::new(RocksDBKV::new(path)?, "main_db"),
};
```

---

#### **2.3 Enable tokio-console for Production Debugging**

**Consensus**: All three AIs recommend tokio-console for real-time task monitoring

```bash
# Build with console support
RUSTFLAGS="--cfg tokio_unstable" cargo build --release --package q-api-server

# Add to Cargo.toml
[dependencies]
console-subscriber = "0.2"

# In main.rs startup:
console_subscriber::init();

# Run in production:
TOKIO_CONSOLE_BIND=127.0.0.1:6669 ./q-api-server

# Monitor from separate terminal:
tokio-console http://127.0.0.1:6669
```

**When Stall Occurs**:
- View all tasks in real-time
- See which task is stuck in `POLLING` state (waiting on lock)
- Identify resource contention (locks, channels, semaphores)
- Capture exact line number where task is blocked

---

### **Priority 3: MEDIUM-TERM (1 Month)**

#### **3.1 Architectural Improvements**

**Consensus Recommendations:**

1. **Add Task Supervision** (Kimi AI, ChatGPT)
```rust
// Supervisor loop that restarts crashed tasks
loop {
    let handle = tokio::spawn(block_producer_task());

    match handle.await {
        Ok(Ok(())) => info!("Producer exited normally"),
        Ok(Err(e)) => error!("Producer failed: {}", e),
        Err(e) => error!("Producer panicked: {}", e),
    }

    tokio::time::sleep(Duration::from_secs(5)).await;
}
```

2. **Narrow Lock Scopes** (All three AIs)
```rust
// ❌ BAD: Lock held across entire operation
let db = storage.db.write().await;
let block = create_block(solution);
db.put(key, value)?;
drop(db);

// ✅ GOOD: Lock only for minimal critical section
let block = create_block(solution);
{
    let mut db = storage.db.write().await;
    db.put(key, value)?;
} // Lock released immediately
```

3. **Add Backpressure Signaling** (Kimi AI)
```rust
// Notify miners when producer is stalled
let (health_tx, health_rx) = watch::channel(ProducerHealth::Healthy);

// In mining submission handler:
if *health_rx.borrow() == ProducerHealth::Stalled {
    return Err("Block producer stalled, please retry".into());
}
```

---

## 🔬 **DEBUGGING TOOLS CONSENSUS**

All three AIs recommend these specific tools:

### **1. Lock Contention Profiling**
```bash
# perf with lock contention
perf lock record -p $(pgrep q-api-server) sleep 30
perf lock report

# Look for high contention time on specific locks
```

### **2. Async Stack Traces**
```bash
# gdb stack trace capture during stall
gdb -p $(pgrep q-api-server) -batch -ex "thread apply all bt" > stall_stacktrace.txt

# Look for tasks stuck in:
# - tokio::sync::RwLock::write
# - rocksdb::DB::put
# - futex syscalls
```

### **3. Deadlock Detection**
```rust
// Add parking_lot with deadlock detection
[dependencies]
parking_lot = { version = "0.12", features = ["deadlock_detection"] }

// Startup thread:
std::thread::spawn(move || {
    loop {
        std::thread::sleep(Duration::from_secs(10));
        let deadlocks = parking_lot::deadlock::check_deadlock();
        if !deadlocks.is_empty() {
            eprintln!("🚨 DEADLOCK: {:#?}", deadlocks);
            std::process::exit(1);
        }
    }
});
```

---

## 📊 **EXPECTED OUTCOMES**

### **After Priority 1 (Immediate - 24 hours)**:
- ✅ HeightState deployed → Binary search storm eliminated
- ✅ Watchdog deployed → Automatic recovery within 2 minutes
- ✅ Stalls reduced from "indefinite" to "<2 minutes"
- ⚠️ Stalls MAY still occur (root cause not fixed yet)

### **After Priority 2 (Short-term - 1 week)**:
- ✅ AsyncStorageEngine deployed → Deadlock eliminated
- ✅ Lock instrumentation → Clear visibility into any remaining issues
- ✅ tokio-console enabled → Real-time task monitoring
- ✅ **LIKELY: Zero stalls for 7+ days continuous operation**

### **After Priority 3 (Medium-term - 1 month)**:
- ✅ Task supervision → Automatic recovery from any task crash
- ✅ Architectural improvements → Reduced complexity, better maintainability
- ✅ **TARGET: 99.9% uptime (zero manual interventions)**

---

## ⚠️ **CRITICAL WARNINGS FROM ALL AIs**

### **1. Never `.await` While Holding a Lock**
```rust
// ❌ WRONG - Deadlock risk!
let db = storage.db.write().await;
some_async_function().await; // ← NEVER DO THIS!
db.put(key, value)?;

// ✅ RIGHT - Drop lock before await
let data = {
    let db = storage.db.read().await;
    db.get(key)?
};
some_async_function(data).await;
```

### **2. Use `spawn_blocking` for All Blocking I/O**
```rust
// ❌ WRONG - Blocks async runtime
async fn save(&self) {
    db.put(key, value)?; // ← RocksDB is blocking!
}

// ✅ RIGHT - Isolate blocking work
async fn save(&self) {
    let db = self.db.clone();
    tokio::task::spawn_blocking(move || {
        db.put(key, value)
    }).await??
}
```

### **3. Consider `panic = "abort"` for Release**
```toml
# Cargo.toml
[profile.release]
panic = "abort" # Prevents poisoned locks

# Trade-off: No unwinding, but cleaner failure mode
```

---

## 🎯 **ACTION ITEMS SUMMARY**

### **IMMEDIATE (Today - Nov 13)**:
- [ ] Build release binary with HeightState (if not already done)
- [ ] Deploy v1.0.2-beta to production
- [ ] Implement watchdog monitoring script
- [ ] Monitor for 24 hours

### **SHORT-TERM (This Week)**:
- [ ] Implement `AsyncStorageEngine` with dedicated thread
- [ ] Add lock instrumentation (wait/hold time metrics)
- [ ] Enable tokio-console in production
- [ ] Create lock contention stress test
- [ ] Deploy to canary node for testing

### **MEDIUM-TERM (This Month)**:
- [ ] Migrate all storage operations to `AsyncStorageEngine`
- [ ] Add task supervision/restart logic
- [ ] Implement producer health backpressure
- [ ] Run 7-day continuous stability test
- [ ] Document architecture changes

---

## 💡 **KEY INSIGHTS FROM AI CONSENSUS**

### **What All Three AIs Agreed On:**

1. **Root Cause**: RwLock deadlock due to blocking I/O under async locks (90% consensus)
2. **Immediate Fix**: Deploy HeightState + Watchdog (100% consensus)
3. **Permanent Fix**: Dedicated storage thread, no async locks (100% consensus)
4. **Architecture Flaw**: Mixing blocking I/O with async runtime (100% consensus)
5. **Testing Strategy**: Lock contention stress tests + tokio-console (100% consensus)

### **Slight Disagreements:**

- **Kimi AI**: 90% confidence on RwLock panic poisoning
- **ChatGPT**: Emphasizes lock-order inversion and writer starvation
- **DeepSeek**: 85% confidence, also considers async runtime starvation

**Interpretation**: All are describing the **same underlying issue** (blocking I/O under locks) from slightly different angles. The consensus solution addresses all scenarios.

---

## 🚀 **RECOMMENDED DEPLOYMENT SEQUENCE**

### **Phase 1: Emergency Stabilization (Today)**
```bash
# 1. Deploy HeightState
sudo systemctl stop q-api-server
sudo cp target/release/q-api-server $(which q-api-server)
sudo systemctl start q-api-server

# 2. Deploy watchdog
sudo tee /usr/local/bin/mining_watchdog.sh << 'EOF'
#!/bin/bash
LAST_HEIGHT=0
STUCK_COUNT=0
while true; do
    CURRENT=$(curl -s http://localhost:8080/api/v1/node/status | jq -r .data.current_height)
    if [ "$CURRENT" = "$LAST_HEIGHT" ]; then
        STUCK_COUNT=$((STUCK_COUNT + 1))
        if [ $STUCK_COUNT -ge 4 ]; then
            echo "🚨 STALL at height $CURRENT - Restarting..."
            systemctl restart q-api-server
            STUCK_COUNT=0
        fi
    else
        STUCK_COUNT=0
    fi
    LAST_HEIGHT=$CURRENT
    sleep 30
done
EOF
sudo chmod +x /usr/local/bin/mining_watchdog.sh

# 3. Run watchdog in screen/tmux
screen -dmS watchdog /usr/local/bin/mining_watchdog.sh
```

### **Phase 2: Instrumentation (Tomorrow)**
- Add lock timing wrappers
- Enable tokio-console
- Create Grafana dashboard for lock metrics

### **Phase 3: Architecture Migration (This Week)**
- Implement `AsyncStorageEngine`
- Test on canary node
- Gradual rollout to production

---

## 📞 **ESCALATION MATRIX**

| Scenario | Action | Timeframe |
|----------|--------|-----------|
| Stall occurs with HeightState + Watchdog | Automatic restart (watchdog) | <2 minutes |
| Watchdog fails to restart | Manual force-kill + restart | <5 minutes |
| Repeated stalls (>5/day) | Emergency deploy AsyncStorageEngine | <24 hours |
| AsyncStorageEngine doesn't fix | Deep dive with tokio-console + external help | <1 week |

---

## ✅ **SUCCESS CRITERIA**

### **Week 1 (HeightState + Watchdog)**:
- [ ] Zero manual restarts (watchdog handles all stalls)
- [ ] Stall frequency reduced to <1/day
- [ ] Recovery time <2 minutes (automated)

### **Week 2 (AsyncStorageEngine)**:
- [ ] Zero stalls for 168 hours continuous
- [ ] Lock wait time <10ms (p99)
- [ ] Block producer iteration <100ms (p99)

### **Week 4 (Full Architecture)**:
- [ ] 99.9% uptime (zero downtime)
- [ ] All metrics green (locks, tasks, storage)
- [ ] Stress tests pass (1000+ TPS for 24 hours)

---

**Prepared By**: Server Beta (Claude Code) synthesizing input from:
- **Kimi AI**: Detailed deadlock analysis, panic poisoning hypothesis
- **ChatGPT**: Practical debugging steps, production-ready code patterns
- **DeepSeek**: Comprehensive testing strategy, architectural review

**Status**: ✅ **CONSENSUS REACHED** - Clear action plan with strong agreement from all AI systems
**Next Action**: Deploy Phase 1 (HeightState + Watchdog) immediately
**Expected Resolution**: 90%+ probability of complete fix after Phase 2 deployment
