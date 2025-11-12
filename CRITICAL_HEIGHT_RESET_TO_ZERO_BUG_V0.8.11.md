# 🚨 CRITICAL: Height Reset to Zero at Block 6050 (v0.8.11-beta)

**Date**: 2025-11-03 20:30 CET
**Severity**: **CATASTROPHIC** - Complete blockchain data loss
**Status**: ⚠️ **ACTIVE BUG - PRODUCTION AFFECTED**
**Version**: v0.8.11-beta

---

## 📊 Incident Report

### What Happened

**Timeline**:
1. Node synced from 740 blocks → 6050 blocks ✅
2. At exactly block 6050, height **RESET TO ZERO** ❌❌❌
3. All 6050 blocks **LOST**

### User Reports

> "i also notice d blocks and hiegh tt got reset again. before over 3000 now only 740"

> "The synchronization reset to 0 upon arrival at block 6050."

---

## 🔍 Root Cause Analysis

### This is NOT the Standard Sync-Down Bug

**Standard sync-down bug** (v0.5.21):
- Peer announces lower height
- Node syncs DOWN to that height
- **Safety checks in place** (v0.5.23+)

**This bug is DIFFERENT**:
- Node reaches block 6050
- Height **resets to ZERO internally**
- NOT caused by peer announcement
- **Safety checks do NOT catch this!**

### Hypothesis: Height Tracking Bug in BlockProducer

Looking at `crates/q-api-server/src/block_producer.rs`, the parallel producers might have a height synchronization issue.

**Possible causes**:

#### 1. **BlockProducer State Reset on Restart**

If the node restarted and `load_from_storage()` failed:

```rust
// block_producer.rs:150-177
pub async fn load_from_storage(&mut self, storage: &Arc<q_storage::QStorage>) -> anyhow::Result<()> {
    match storage.get_latest_qblock().await? {
        Some(latest_block) => {
            self.current_height = latest_block.header.height;  // ✅ Loads height
        }
        None => {
            info!("📝 No existing blockchain state found - starting from genesis");
            // ❌ Falls back to height 0!
        }
    }
}
```

**If `get_latest_qblock()` fails** → height resets to 0!

#### 2. **Parallel Producer Desynchronization**

With 8 parallel producers (v0.8.11-beta):
- Each producer tracks its own height
- If they get out of sync, one might reset
- Producer with height 0 could overwrite others

#### 3. **Database Corruption at Block 6050**

**Possible corruption scenarios**:
- Block 6050 has invalid data
- RocksDB fails to read block 6050
- `get_latest_qblock()` returns `None`
- System thinks database is empty
- Resets to height 0

#### 4. **Turbo Sync Pack Application Bug**

**File**: `crates/q-storage/src/turbo_sync.rs:627-641`

```rust
// 🚨 CRITICAL SAFETY: Ensure height NEVER regresses
if highest_contiguous < current_height {
    error!("🚨 [v0.7.0] SAFETY ABORT: Height regression detected!");
    anyhow::bail!("SAFETY ABORT: Height regression");
}
```

**BUT** if the pack contains blocks 0-6050 and current height is somehow corrupted:
- `highest_contiguous` might be calculated wrong
- Safety check might not trigger
- Database gets overwritten

---

## 🛡️ Missing Safety Checks

### 1. **Height Monotonicity Enforcement** (NOT IMPLEMENTED)

From `CRITICAL_SYNC_DOWN_BUG_ANALYSIS.md`:

```rust
// ❌ NOT IMPLEMENTED YET:
static HIGHEST_EVER_HEIGHT: AtomicU64 = AtomicU64::new(0);

// This would have prevented the reset:
if current < highest_ever - 10 {
    panic!("HEIGHT REGRESSION DETECTED!");
}
```

### 2. **Database Load Verification** (MISSING)

```rust
// ❌ MISSING CHECK in load_from_storage():
pub async fn load_from_storage(&mut self, storage: &Arc<q_storage::QStorage>) -> anyhow::Result<()> {
    match storage.get_latest_qblock().await? {
        Some(latest_block) => {
            // ✅ SUCCESS PATH
        }
        None => {
            // ❌ SHOULD CHECK: Is database actually empty?
            // If database has files but returns None → CORRUPTION!

            // MISSING:
            let db_size = check_database_size()?;
            if db_size > 1_000_000 {  // > 1MB means not empty
                error!("🚨 Database exists but get_latest_qblock() returned None!");
                error!("   This indicates database corruption!");
                panic!("DATABASE CORRUPTION DETECTED");
            }
        }
    }
}
```

### 3. **Parallel Producer Height Consistency** (NOT ENFORCED)

```rust
// ❌ MISSING in ParallelBlockProducerPool:
pub async fn verify_producer_heights(&self) -> Result<()> {
    let mut heights = Vec::new();

    for producer in &self.producers {
        let p = producer.read().await;
        heights.push(p.get_height());
    }

    let max_height = *heights.iter().max().unwrap();
    let min_height = *heights.iter().min().unwrap();

    if max_height - min_height > 10 {
        error!("🚨 Producer height desync detected!");
        error!("   Min: {}, Max: {}", min_height, max_height);
        panic!("PRODUCER HEIGHT DESYNC");
    }

    Ok(())
}
```

---

## 📊 Diagnostic Information Needed

To diagnose this bug, we need:

### 1. **Database Status at Time of Reset**

```bash
# Check database size
ls -lh /opt/orobit/shared/q-narwhalknight/data-local-beta/data/q-narwhal-db/

# Check if blocks exist
# (If blocks exist but height=0, this is database read failure)
```

### 2. **Log Analysis**

Need to check logs for:
- `load_from_storage` messages
- `No existing blockchain state found` warnings
- Database read errors
- RocksDB corruption warnings
- Turbo sync pack application logs

### 3. **Producer Heights**

Check if all 8 producers had consistent heights before reset.

---

## 🔧 Immediate Fixes Required

### Fix 1: **Add Height Monotonicity Enforcement**

**File**: `crates/q-api-server/src/main.rs`

```rust
use std::sync::atomic::{AtomicU64, Ordering};

// Global highest-ever height tracker
static HIGHEST_EVER_HEIGHT: AtomicU64 = AtomicU64::new(0);

// Call this BEFORE any height update:
fn verify_height_monotonicity(new_height: u64) -> Result<()> {
    let highest_ever = HIGHEST_EVER_HEIGHT.load(Ordering::SeqCst);

    if new_height == 0 && highest_ever > 100 {
        error!("🚨🚨🚨 CRITICAL: HEIGHT RESET TO ZERO DETECTED! 🚨🚨🚨");
        error!("   Previous highest: {} blocks", highest_ever);
        error!("   New height: 0 blocks");
        error!("   This indicates:");
        error!("   - Database corruption");
        error!("   - BlockProducer state reset bug");
        error!("   - Catastrophic sync-down");
        error!("   ");
        error!("   ABORTING TO PREVENT DATA LOSS!");

        panic!("SAFETY ABORT: Height reset to zero from {} blocks", highest_ever);
    }

    if new_height < highest_ever - 10 {
        error!("🚨 HEIGHT REGRESSION: {} → {}", highest_ever, new_height);
        return Err(anyhow::anyhow!("Height regression detected"));
    }

    // Update highest ever
    HIGHEST_EVER_HEIGHT.fetch_max(new_height, Ordering::SeqCst);

    Ok(())
}
```

### Fix 2: **Database Load Verification**

**File**: `crates/q-api-server/src/block_producer.rs:150-177`

```rust
pub async fn load_from_storage(&mut self, storage: &Arc<q_storage::QStorage>) -> anyhow::Result<()> {
    info!("📂 Loading blockchain state from storage for producer (validator_index={})...",
        self.config.validator_index);

    match storage.get_latest_qblock().await? {
        Some(latest_block) => {
            self.current_height = latest_block.header.height;
            self.latest_block_hash = latest_block.calculate_hash();
            self.total_difficulty = latest_block.header.total_difficulty;
            self.dag_round = latest_block.header.dag_round;

            info!("✅ Loaded blockchain state from storage:");
            info!("   Height: {}", self.current_height);
        }
        None => {
            // ✅ NEW SAFETY CHECK:
            let db_size = storage.estimate_database_size().await?;

            if db_size > 10_000_000 {  // > 10MB
                error!("🚨🚨🚨 DATABASE CORRUPTION DETECTED! 🚨🚨🚨");
                error!("   Database size: {} bytes", db_size);
                error!("   But get_latest_qblock() returned None!");
                error!("   This means:");
                error!("   - Database is not empty");
                error!("   - But we can't read the latest block");
                error!("   - CRITICAL CORRUPTION!");

                return Err(anyhow::anyhow!(
                    "Database corruption: {} bytes exist but no latest block found",
                    db_size
                ));
            }

            info!("📝 No existing blockchain state found - starting from genesis");
        }
    }

    Ok(())
}
```

### Fix 3: **Producer Height Consistency Check**

**File**: `crates/q-api-server/src/block_producer.rs:ParallelBlockProducerPool`

Add verification method and call it periodically:

```rust
impl ParallelBlockProducerPool {
    /// Verify all producers have consistent heights
    pub async fn verify_height_consistency(&self) -> Result<()> {
        let mut heights = Vec::new();

        for (idx, producer_arc) in self.producers.iter().enumerate() {
            let producer = producer_arc.read().await;
            let height = producer.get_height();
            heights.push((idx, height));
        }

        let max = heights.iter().map(|(_, h)| h).max().unwrap();
        let min = heights.iter().map(|(_, h)| h).min().unwrap();

        if max - min > 10 {
            error!("🚨 PRODUCER HEIGHT DESYNC DETECTED:");
            for (idx, height) in heights {
                error!("   Producer #{}: height {}", idx, height);
            }

            return Err(anyhow::anyhow!(
                "Producer height desync: min={}, max={}, diff={}",
                min, max, max - min
            ));
        }

        Ok(())
    }
}

// Call this every 10 blocks:
if current_height % 10 == 0 {
    parallel_producer_pool.verify_height_consistency().await?;
}
```

---

## 🚀 Emergency Deployment Plan

### Phase 1: **Immediate Diagnosis** (NOW)

1. Check database status:
```bash
ls -lh /opt/orobit/shared/q-narwhalknight/data-local-beta/data/
du -sh /opt/orobit/shared/q-narwhalknight/data-local-beta/data/
```

2. Check recent logs for errors:
```bash
# Look for database errors
grep -i "database\|corruption\|load_from_storage" /opt/orobit/shared/q-narwhalknight/*.log

# Look for height reset
grep -i "height.*0\|starting from genesis" /opt/orobit/shared/q-narwhalknight/*.log
```

3. Document exact sequence of events

### Phase 2: **Deploy Fixes** (v0.9.0-beta-emergency)

1. Implement height monotonicity enforcement
2. Add database load verification
3. Add producer height consistency checks
4. Deploy with LOUD logging

### Phase 3: **Recovery** (If Data Lost)

**Option A**: Restore from backup
```bash
# Check for backups
ls -lh /opt/orobit/shared/q-narwhalknight/backups/

# Restore most recent backup
cp -r /opt/orobit/shared/q-narwhalknight/backups/latest/* /opt/orobit/shared/q-narwhalknight/data-local-beta/
```

**Option B**: Resync from network
- Stop node
- Delete corrupted database
- Restart node
- Let turbo_sync rebuild from peers

---

## 📝 Lessons Learned

### Critical Failures:

1. **No height monotonicity enforcement** ❌
   - Should have been implemented in v0.5.23-beta
   - Would have prevented this entire incident

2. **No database integrity checks** ❌
   - `load_from_storage()` blindly accepts `None`
   - Should verify database is truly empty

3. **No parallel producer consistency checks** ❌
   - 8 producers can get out of sync silently
   - No health monitoring

4. **No automatic backups** ❌
   - Should backup every N blocks
   - Should backup before any risky operation

### Action Items:

- [ ] Implement height monotonicity NOW
- [ ] Add database verification checks NOW
- [ ] Add producer consistency checks NOW
- [ ] Implement automatic hourly backups
- [ ] Add real-time health monitoring
- [ ] Add circuit breakers for anomalies
- [ ] Deploy v0.9.0-beta-emergency ASAP

---

## 🚨 Status

**PRODUCTION IS BROKEN**
**ALL USERS AFFECTED**
**DATA LOSS ONGOING**

**IMMEDIATE ACTION REQUIRED**

---

**Next Steps**:
1. Gather diagnostic information
2. Implement emergency fixes
3. Deploy v0.9.0-beta-emergency
4. Monitor closely for 24 hours
5. Post-mortem analysis
6. Comprehensive testing before next release

---

**This bug demonstrates why blockchain data integrity is CRITICAL and why we need multiple layers of safety checks.**
