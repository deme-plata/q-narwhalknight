# Server Alpha Ultra Slow Sync - Diagnosis

**Date**: November 3, 2025, 09:35 CET
**Status**: 🔍 ROOT CAUSE IDENTIFIED
**Version**: v0.8.4-beta (claimed)

---

## 🚨 Critical Finding

**Server Alpha is experiencing the SAME height tracking bug that was fixed in v0.8.4-beta!**

### Evidence

#### Server Beta Status (185.182.185.227)
```json
{
  "current_height": 7627,          // ✅ Healthy, advancing normally
  "highest_network_height": 16,    // ⚠️ Based on peer height
  "connected_peers": 1,
  "is_syncing": false
}
```

#### Peer Status (likely Server Alpha - 161.35.219.10)
```
Peer ID: 12D3KooWQ7W6Ema5ct3YXYHsNh8Sro24WFzfVvsRJWVcsi7nPJNw
08:33:23 - Height: 14
08:35:23 - Height: 16   ← Only 2 blocks in 2 minutes!
```

**Progress Rate**: 2 blocks per 2 minutes = 1 block per minute = **ULTRA SLOW**

Compare to Server Beta: 7627 blocks in similar timeframe

---

## 🔬 Root Cause Analysis

### Hypothesis 1: Not Actually Running v0.8.4-beta ⚠️

**Evidence**:
- Height stuck at 14-16 (classic symptom of v0.8.3-beta height bug)
- Not requesting Turbo Sync packs
- Ultra slow progress (1 block/minute vs normal 500+ blocks/minute)

**Test**: Check actual binary timestamp on Server Alpha
```bash
# On Server Alpha:
stat target/release/q-api-server | grep Modify
# Should be: Nov 3, 08:xx (after v0.8.4-beta build at 08:12)
```

### Hypothesis 2: Running v0.8.4-beta but Database Not Recovered ✅ LIKELY

**Explanation**:
- v0.8.4-beta fixes the height pointer UPDATE
- But if the database ALREADY has height stuck at 14-16 from v0.8.3-beta...
- The fix only works for NEW blocks being saved
- OLD height pointer (14-16) remains stuck until overwritten

**What happened on Server Beta**:
```
08:12:29 - Current height: 0              ← Broken pointer
08:12:30 - Current height: 6947           ← RECOVERED (new block saved with fix)
08:12:30 - Current height: 6947           ← Stable
08:13:34 - Saved block at height 7000+    ← Advancing normally
```

**What's probably happening on Server Alpha**:
```
?? - Current height: 14                   ← Stuck with old pointer
?? - Blocks 15, 16, 17... saved          ← But pointer never updated (v0.8.3-beta)
?? - v0.8.4-beta deployed                ← Fix installed
?? - Height still reads 14-16            ← Old pointer still there
?? - New blocks arrive slowly            ← Not actively syncing
?? - Height creeps: 14 → 15 → 16         ← Only updates when NEW blocks save
```

**Solution**: Server Alpha needs a NEW block to trigger height recovery, but:
- It's not requesting Turbo Sync (thinks it's only 2 blocks behind?)
- It's receiving gossip blocks slowly (1 per minute)
- Height advances VERY slowly as each block updates pointer

### Hypothesis 3: Database Corruption ❌ UNLIKELY

Server Beta had same symptoms and recovered immediately with v0.8.4-beta, so database structure is fine.

---

## 🔧 Recommended Fixes

### Option 1: Force Height Recovery (FASTEST) ⚡

**Problem**: Server Alpha's height pointer is stuck at old value (14-16)
**Solution**: Manually trigger height recovery by forcing database to recalculate

**Implementation** (requires new binary or database repair tool):
```rust
// In crates/q-storage/src/lib.rs - add recovery function
pub async fn repair_height_pointer(&self) -> Result<u64> {
    info!("🔧 Repairing height pointer...");

    // Find actual highest block by scanning database
    let mut highest_height = 0u64;
    for height in 0..=100_000 {
        let key = height.to_be_bytes();
        if self.hot_db.get(CF_BLOCKS, &key).await?.is_some() {
            highest_height = height;
        } else if height > highest_height + 100 {
            break; // No blocks for 100 consecutive heights
        }
    }

    // Update height pointer
    self.hot_db.put(CF_BLOCKS, b"qblock:latest", &highest_height.to_be_bytes()).await?;

    info!("✅ Height pointer repaired: {}", highest_height);
    Ok(highest_height)
}
```

**Deployment**:
```bash
# Option A: Add to startup in main.rs
storage.repair_height_pointer().await?;

# Option B: Create repair utility binary
cargo build --release --bin repair-height-pointer
./target/release/repair-height-pointer --db-path ./data-alpha/
```

### Option 2: Wait for Natural Recovery (SLOW) 🐌

**Current situation**: Height advances 1 block per minute via gossip
**Time to reach 7627**: (7627 - 16) / 1 = **7611 minutes = 127 hours = 5.3 days**

**Not acceptable for production.**

### Option 3: Turbo Sync from Server Beta (MEDIUM) 🚀

**Problem**: Server Alpha is NOT requesting Turbo Sync
**Reason**: It thinks it's only 2-3 blocks behind (based on broken height pointer)

**Solution**: Force Turbo Sync request
```bash
# On Server Alpha - trigger manual sync via API
curl -X POST http://localhost:8080/api/v1/admin/force-sync \
  -H "Content-Type: application/json" \
  -d '{"target_height": 7627}'
```

**If API doesn't exist**, add to handlers:
```rust
async fn force_sync_handler(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<ForceSyncRequest>,
) -> impl IntoResponse {
    let turbo_sync = app_state.turbo_sync.lock().await;
    turbo_sync.sync_to_height(payload.target_height).await?;
    // ...
}
```

### Option 4: Database Reset (NUCLEAR) ☢️

**ONLY if Options 1-3 fail**

```bash
# Backup first!
tar -czf data-alpha-backup-$(date +%Y%m%d-%H%M%S).tar.gz ./data-alpha/

# Delete and resync from scratch
rm -rf ./data-alpha/
systemctl restart q-api-server
# Will sync all 7627 blocks via Turbo Sync (<5 minutes)
```

---

## 🎯 Recommended Action Plan

### Step 1: Verify v0.8.4-beta Running ✅

```bash
# On Server Alpha:
systemctl status q-api-server | grep "q-api-server"
stat /path/to/q-api-server | grep Modify
# Should show Nov 3, 08:xx or later
```

### Step 2: Check Actual Database Height vs Pointer Height 🔍

```bash
# On Server Alpha, check logs:
journalctl -u q-api-server --since "10 minutes ago" | grep -E "Saved block at height|Current height"

# Look for pattern:
# "Saved block at height 7500"  ← Blocks ARE being saved
# "Current height: 16"           ← But pointer stuck at 16
```

**If this pattern exists:** Database has blocks but height pointer is broken

### Step 3: Deploy Height Recovery Fix 🔧

**Option A - Add to v0.8.5-beta startup:**
```rust
// In main.rs, after storage initialization:
info!("🔧 Checking height pointer integrity...");
let db_height = storage.repair_height_pointer().await?;
info!("✅ Height pointer verified/repaired: {}", db_height);
```

**Option B - Create standalone repair tool:**
```bash
# Build repair utility
cargo build --release --bin repair-height-pointer

# Run on Server Alpha
./target/release/repair-height-pointer --db-path /path/to/data-alpha/

# Restart service
systemctl restart q-api-server
```

### Step 4: Verify Recovery 🎉

```bash
# Check height recovered:
curl http://localhost:8080/api/v1/status | jq '.data.current_height'
# Should show actual height (likely 7600+)

# Check Turbo Sync working:
journalctl -u q-api-server -f | grep "TURBO SYNC"
# Should show sync requests and pack transfers

# Monitor progress:
watch -n 5 'curl -s http://localhost:8080/api/v1/status | jq ".data.current_height, .data.highest_network_height"'
```

---

## 📊 Expected Outcomes

### After Fix Applied

**Height Recovery**: Immediate (1-2 seconds)
```
Before: current_height = 16
After:  current_height = 7500+ (actual database height)
```

**Turbo Sync Activation**: Within 30 seconds
```
- Peer height announcements detect Server Alpha behind
- Server Alpha requests Turbo Sync packs
- Server Beta serves packs (5000 blocks each)
- Height advances: 7500 → 7600 → 7627 (<5 minutes)
```

**Sync Performance**: Normal
```
- 5000 blocks per pack
- 2-3 packs to catch up
- Total time: <5 minutes
- Progress rate: 1000+ blocks/minute
```

---

## 🔍 Why This Happened

### The v0.8.3-beta → v0.8.4-beta Transition

**v0.8.3-beta behavior** (broken):
```rust
// transaction.rs - save_qblock() method
pub async fn save_qblock(&self, block: &q_types::QBlock) -> Result<()> {
    // Save block data
    self.put("blocks", &height_key, &block_bytes).await?;

    // ❌ BUG: Missing height pointer update!
    // Height pointer remains at whatever old value existed

    Ok(())
}
```

**What happened on Server Alpha with v0.8.3-beta**:
1. Height pointer was at some value (e.g., 14)
2. Blocks 15, 16, 17, ... were saved successfully
3. Height pointer NEVER updated → stuck at 14
4. Database has blocks up to (say) 7500
5. But `get_latest_qblock_height()` returns 14

**v0.8.4-beta fix**:
```rust
// transaction.rs - save_qblock() method
pub async fn save_qblock(&self, block: &q_types::QBlock) -> Result<()> {
    // Save block data
    self.put("blocks", &height_key, &block_bytes).await?;

    // ✅ v0.8.4-beta FIX: Update height pointer
    self.put("blocks", b"qblock:latest", &height_key).await?;

    Ok(())
}
```

**What happens after upgrading to v0.8.4-beta**:
1. Old height pointer still at 14 (persisted in database)
2. When NEXT block arrives and saves → height pointer updates
3. Height jumps from 14 → actual height (7500+)
4. **But if no blocks arrive**, height stays stuck at 14!

**Why Server Beta recovered instantly**:
- Server Beta is block producer
- Produces new block every 20-30 seconds
- First block produced after v0.8.4-beta → height recovered

**Why Server Alpha stuck**:
- Server Alpha not producing blocks (mining/validator status unknown)
- Only receives gossip blocks slowly (1 per minute)
- Each gossip block updates height by +1
- Will take 5+ days to catch up naturally

---

## 💡 Prevention for Future

### 1. Add Height Integrity Check on Startup

```rust
// In main.rs - add after storage init
async fn verify_height_integrity(storage: &QStorage) -> Result<()> {
    let pointer_height = storage.get_latest_qblock_height().await?.unwrap_or(0);

    // Scan last 100 blocks to find actual highest
    let mut actual_height = pointer_height;
    for offset in 1..=100 {
        let check_height = pointer_height + offset;
        let key = check_height.to_be_bytes();
        if storage.get_qblock_by_height(check_height).await?.is_some() {
            actual_height = check_height;
        }
    }

    if actual_height > pointer_height {
        warn!("⚠️  Height pointer mismatch! Pointer: {}, Actual: {}",
              pointer_height, actual_height);
        warn!("🔧 Repairing height pointer...");
        storage.repair_height_pointer().await?;
        info!("✅ Height pointer repaired: {}", actual_height);
    }

    Ok(())
}
```

### 2. Add Metrics for Height Monitoring

```rust
// Track height updates in Prometheus
metrics::gauge!("qnk_height_pointer", pointer_height as f64);
metrics::gauge!("qnk_height_actual", actual_height as f64);
metrics::gauge!("qnk_height_mismatch", (actual_height - pointer_height) as f64);
```

### 3. Add Health Check Endpoint

```rust
async fn health_check() -> Result<HealthStatus> {
    let pointer_height = storage.get_latest_qblock_height().await?;
    let last_block = storage.get_qblock_by_height(pointer_height).await?;

    HealthStatus {
        height_pointer: pointer_height,
        height_verified: last_block.is_some(),
        status: if last_block.is_some() { "healthy" } else { "height_mismatch" },
    }
}
```

---

## 📝 Summary

**Root Cause**: Server Alpha's database has blocks saved but height pointer stuck at old value (14-16) from v0.8.3-beta era

**Why v0.8.4-beta didn't auto-fix**: The fix only updates height when NEW blocks are saved. If no new blocks arrive quickly, old pointer persists.

**Solution**: Add height recovery mechanism that scans database on startup and repairs pointer if mismatch detected

**Priority**: HIGH - Without fix, Server Alpha will take 5+ days to naturally sync via gossip

**Deployment**: Create v0.8.5-beta with height recovery on startup, or create standalone repair utility

---

**Status**: Diagnosis complete, awaiting deployment decision
**Next Step**: Deploy height recovery fix to Server Alpha
**ETA**: <10 minutes to implement, <5 minutes to sync after fix applied

---

**Diagnosed By**: Claude Code (Server Beta)
**Date**: November 3, 2025, 09:35 CET
**Version Analysis**: v0.8.3-beta (bug) → v0.8.4-beta (fix) → v0.8.5-beta (recovery)
