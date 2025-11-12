# Recurring Data Loss - Root Cause Analysis v0.9.76-beta

## Critical Problem: Data Loss on Every Restart (10+ times)

### Symptoms
- Database loses ALL blocks on restart
- Happens **10+ times** (recurring pattern)
- `qblock:latest` pointer remains (shows 9606)
- Actual blocks: **0** (complete loss)

### Root Cause: SIGKILL During Restart

**Evidence from journalctl:**
```
Nov 09 14:48:43: systemd[1]: q-api-server.service: Killing process with SIGKILL
Nov 09 15:23:17: systemd[1]: q-api-server.service: Killing process with SIGKILL
Nov 09 15:23:18: systemd[1]: q-api-server.service: Failed with result 'timeout'
```

**The kill chain:**
1. `systemctl restart q-api-server` issued
2. Systemd sends SIGTERM (graceful shutdown)
3. Node doesn't respond within TimeoutStopSec (probably 90s)
4. **Systemd sends SIGKILL** (forceful kill, no cleanup possible)
5. RocksDB buffers not flushed
6. WAL not synced
7. **ALL recent blocks lost**

### Why RocksDB Loses Data

Even with `set_sync(true)` and `flush_cf_opt()`:

**The timing issue:**
```
T+0s:  systemctl restart issued
T+0s:  SIGTERM sent
T+1-90s: Node tries to shutdown gracefully
         - Spawned tasks still running
         - Mistral.rs AI model in memory
         - Network connections closing
         - RocksDB preparing final flush
T+90s: SIGKILL sent (timeout!)
       ❌ RocksDB flush() interrupted mid-operation
       ❌ SST files partially written
       ❌ MANIFEST not updated
       ❌ WAL truncated
T+91s: Process terminated immediately
       💥 ALL unflushed data LOST
```

**RocksDB guarantees with SIGKILL:**
- ❌ `set_sync(true)` - **USELESS** if killed before write completes
- ❌ `flush_cf_opt(wait=true)` - **USELESS** if killed during flush
- ❌ WAL - **USELESS** if MANIFEST doesn't reference it

**What gets lost:**
1. Blocks written in last 1-2 seconds (buffered)
2. Blocks being flushed when SIGKILL arrives
3. Entire SST files if flush incomplete
4. MANIFEST entries if update interrupted

### Why This Happens 10+ Times

**The vicious cycle:**
1. Node syncs blocks (writes to RocksDB)
2. Admin restarts service (testing/debugging)
3. Systemd timeout → SIGKILL
4. Data lost
5. Node starts with corrupted DB
6. Admin restarts again to fix
7. **Repeat 10+ times**

### The Flaw in Current Protection

**Line 668-670 in kv.rs:**
```rust
let mut write_opts = rocksdb::WriteOptions::default();
write_opts.set_sync(true); // ← Should protect against crashes
write_opts.disable_wal(false); // ← Should enable crash recovery
```

**Line 681-697 in kv.rs:**
```rust
let mut flush_opts = rocksdb::FlushOptions::default();
flush_opts.set_wait(true); // ← Should block until flush completes
self.db.flush_cf_opt(&cf_handle, &flush_opts)?;
```

**Why it still fails:**
- `set_sync(true)` only applies to individual `write()` calls
- SIGKILL can arrive BETWEEN `write()` and `flush()`
- SIGKILL can arrive DURING `flush()` operation
- No protection against mid-operation interruption

## The Complete Fix

### Fix 1: Increase Systemd Timeout (Immediate)

Edit `/etc/systemd/system/q-api-server.service`:
```ini
[Service]
TimeoutStopSec=300  # 5 minutes instead of 90 seconds
KillMode=mixed      # Try SIGTERM first, then SIGKILL only main process
SendSIGKILL=yes     # But still allow SIGKILL if needed
```

**Reload:**
```bash
systemctl daemon-reload
systemctl restart q-api-server
```

### Fix 2: Graceful Shutdown Handler (Code)

Add to `main.rs`:
```rust
use tokio::signal;

async fn setup_shutdown_handler(storage: Arc<QStorage>) {
    tokio::spawn(async move {
        signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");

        warn!("🛑 Shutdown signal received - flushing RocksDB...");

        // Force flush ALL column families
        if let Err(e) = storage.force_flush_all().await {
            error!("❌ Flush failed during shutdown: {}", e);
        } else {
            info!("✅ RocksDB flushed successfully");
        }

        // Give RocksDB time to complete
        tokio::time::sleep(Duration::from_secs(5)).await;

        std::process::exit(0);
    });
}
```

### Fix 3: Periodic Background Flush (Safety Net)

Add background flush task:
```rust
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(30));

    loop {
        interval.tick().await;

        // Flush critical column families every 30 seconds
        if let Err(e) = storage.flush_blocks_cf().await {
            warn!("⚠️ Periodic flush failed: {}", e);
        } else {
            debug!("✅ Periodic flush completed");
        }
    }
});
```

### Fix 4: Atomic Write-Ahead Log (RocksDB)

Update RocksDB options at database open:
```rust
let mut db_opts = Options::default();
db_opts.set_wal_ttl_seconds(3600); // Keep WAL for 1 hour
db_opts.set_wal_size_limit_mb(1024); // 1GB WAL limit
db_opts.set_manual_wal_flush(false); // Auto-flush WAL
db_opts.set_wal_recovery_mode(WALRecoveryMode::AbsoluteConsistency);
```

### Fix 5: Database Repair on Startup

Add to node startup:
```rust
// Check for corruption on startup
match storage.verify_integrity().await {
    Ok(true) => info!("✅ Database integrity verified"),
    Ok(false) => {
        error!("🚨 Database corruption detected!");
        error!("   Running automatic repair...");
        storage.repair_pointers().await?;
    }
    Err(e) => error!("⚠️ Integrity check failed: {}", e),
}
```

## Implementation Priority

### Phase 1: Immediate (Stop the bleeding)
1. ✅ Increase systemd timeout to 300s
2. ✅ Add graceful shutdown handler
3. ✅ Run database repair tool before next start

### Phase 2: This Week (Prevent recurrence)
1. Add periodic background flush (30s interval)
2. Update RocksDB WAL settings
3. Add integrity check on startup

### Phase 3: Long-term (Complete solution)
1. Implement proper shutdown coordination
2. Add database snapshots (hourly)
3. Implement RAFT-style replication
4. Add corruption detection and auto-recovery

## Testing Plan

### Test 1: Graceful Shutdown
```bash
# Start node
systemctl start q-api-server

# Wait for some blocks to sync
sleep 60

# Graceful restart (should NOT lose data)
systemctl restart q-api-server

# Verify blocks survived
./target/release/repair-database ./data-mine6/hot
```

### Test 2: Force Kill
```bash
# Start node
systemctl start q-api-server
sleep 60

# Get PID
PID=$(systemctl show -p MainPID q-api-server | cut -d= -f2)

# Send SIGKILL
kill -9 $PID

# Restart
systemctl start q-api-server

# Check if blocks survived (should work with periodic flush)
./target/release/repair-database ./data-mine6/hot
```

### Test 3: Crash Recovery
```bash
# Simulate crash during write
# (manually corrupt database while node running)

# Restart
systemctl start q-api-server

# Should detect corruption and auto-repair
journalctl -u q-api-server -f | grep "corruption\|repair"
```

## Deployment Steps

### Step 1: Fix Systemd Service
```bash
# Edit service file
nano /etc/systemd/system/q-api-server.service

# Add under [Service]:
TimeoutStopSec=300
KillMode=mixed

# Reload
systemctl daemon-reload
```

### Step 2: Repair Current Database
```bash
# Stop service
systemctl stop q-api-server

# Run repair
./target/release/repair-database ./data-mine6/hot
# Choose option 1: Fix pointer to 0

# Start with v0.9.76 (has P2P gap fill)
systemctl start q-api-server
```

### Step 3: Monitor Recovery
```bash
# Watch gap fill progress
journalctl -u q-api-server -f | grep "GAP FILL"

# Check height advancement
watch -n 5 'journalctl -u q-api-server --since "5 minutes ago" | grep "Height:"'
```

## Prevention Checklist

Before ANY restart:
- [ ] Check current height: `journalctl -u q-api-server | grep "Height:" | tail -1`
- [ ] Manually flush: (add API endpoint `/admin/flush`)
- [ ] Wait 30 seconds after last block write
- [ ] Use `systemctl restart` NOT `systemctl kill`
- [ ] Monitor logs for "Flushed successfully" before proceeding

## Metrics to Monitor

Add Prometheus metrics:
```rust
rocksdb_unflushed_blocks_count
rocksdb_wal_size_bytes
rocksdb_last_flush_timestamp
shutdown_graceful_count
shutdown_forced_count
data_loss_events_count
```

## Recovery SOP (Standard Operating Procedure)

If data loss occurs:
1. **STOP** - Don't restart again immediately
2. **DIAGNOSE** - Run repair tool, check logs
3. **FIX ROOT CAUSE** - Check why SIGKILL happened
4. **REPAIR** - Fix database pointer
5. **RESTART WITH MONITORING** - Watch gap fill closely
6. **VERIFY** - Confirm blocks syncing properly
7. **DOCUMENT** - Log what happened for analysis

## Success Criteria

✅ Node survives 10 consecutive restarts without data loss
✅ Graceful shutdown completes in <30 seconds
✅ SIGKILL (if needed) happens after flush completes
✅ Database integrity check passes on every startup
✅ No more "0 blocks found" events

---

**Status**: Root cause identified
**Severity**: CRITICAL - Recurring data loss
**Next**: Implement systemd timeout + graceful shutdown
**ETA**: Fixes deployed within 1 hour
