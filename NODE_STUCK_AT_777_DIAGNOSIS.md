# Node Stuck at Height 777 - Critical Sync Failure

**Date:** 2025-11-08
**Version:** v0.9.61-beta (Server Beta)
**Status:** 🚨 **CRITICAL - NODE CANNOT SYNC PAST HEIGHT 771**

---

## 🎯 PROBLEM SUMMARY

**Node is permanently stuck at height 771-777 and cannot sync further.**

**Current State:**
- Local height: 771 (highest contiguous block)
- Network height: 4364+ (from peers)
- Gap: 3,593+ blocks missing
- **Duration stuck**: 3+ hours
- **Impact**: Node unusable, mining ineffective, balances stale

---

## 🔍 ROOT CAUSE ANALYSIS

### Issue 1: Gossipsub Turbo Sync Sent Request, Never Received Response

**What happened:**
```
15:22:08 🚀 [TURBO SYNC] AUTO-TRIGGER: Local=777, Network=2974, Gap=2197 blocks
15:22:08 📤 [TURBO SYNC] Sent gossipsub request for blocks 777-2974 (chunk 1/1)
15:22:08 ✅ [TURBO SYNC] All 1 block-pack requests sent via gossipsub!
15:22:08 ⏳ [TURBO SYNC] Responses will be handled by gossipsub handler...
```

**What didn't happen:**
- NO block-pack-response ever received
- NO fallback to HTTP sync triggered
- NO retry mechanism activated

**Why:**
- `InsufficientPeers` error on `/qnk/testnet-phase6/block-pack-responses`
- Only 1 peer (Server Beta itself) subscribed to Phase 6 topics
- Gossipsub correctly refused to publish response (would be lost)

---

### Issue 2: HTTP Gap-Fill NEVER Activated

**Expected behavior:**
1. Turbo sync request sent via gossipsub
2. Wait 60-90 seconds for response
3. If no response: **FALLBACK to HTTP gap-fill**
4. HTTP fetch missing blocks from peer's API

**Actual behavior:**
- Turbo sync request sent ✅
- Wait for response... ⏳
- **NO response received** ❌
- **HTTP gap-fill NEVER activated** ❌❌❌

**Evidence:**
```bash
journalctl -u q-api-server --since "30 minutes ago" | grep -E "gap fill|HTTP.*772|batch sync"
# NO RESULTS - HTTP gap-fill never ran!
```

---

### Issue 3: Continuous Gap Detection, No Resolution

**What's happening:**
```
16:22:13 🔍 [HEIGHT DEBUG] FINAL RESULT: Returning height 771
16:22:13 🔍 Gap detected: Missing block at height 772
16:22:13 ⚠️ [GOSSIPSUB] Gap detected at height 772 (received block 4364)
```

**Repeated every few seconds for 3+ hours!**

The node:
1. Receives live blocks at height 4364 via gossipsub ✅
2. Detects gap at height 772 ✅
3. Logs warning ✅
4. **Does nothing to fill the gap** ❌

---

## 🐛 BUG IDENTIFICATION

### Bug 1: Missing HTTP Gap-Fill Fallback

**File:** `crates/q-api-server/src/main.rs` (gossipsub handler)

**Problem:** When gossipsub turbo sync fails (InsufficientPeers), there's NO automatic fallback to HTTP sync.

**Expected code flow:**
```rust
// After sending turbo sync request
tokio::spawn(async move {
    tokio::time::sleep(Duration::from_secs(90)).await;
    if still_have_gap() {
        // FALLBACK: Use HTTP to fill gap
        http_gap_fill(start_height, end_height).await;
    }
});
```

**Actual code:** This fallback doesn't exist! Turbo sync fails silently.

---

### Bug 2: No Retry Logic for Turbo Sync

**Problem:** Turbo sync sends ONE request, then gives up forever.

**Expected:**
- Try 1: Gossipsub turbo sync
- Try 2: Retry after 60s if no response
- Try 3: HTTP fallback after 120s

**Actual:**
- Try 1: Gossipsub turbo sync
- **Give up forever** ❌

---

### Bug 3: Gap Detection Without Resolution

**Problem:** Code detects gaps and logs warnings, but doesn't trigger sync.

**Location:** `crates/q-api-server/src/main.rs` (gossipsub block handler)

```rust
if network_height > current_height + 5 {
    warn!("⚠️ [GOSSIPSUB] Gap detected at height {} (received block {})",
          current_height + 1, network_height);
    // BUG: Should trigger gap-fill here!
    // MISSING: trigger_gap_fill(current_height + 1, network_height).await;
}
```

**The warning is logged, but no action is taken!**

---

## 📊 IMPACT ASSESSMENT

### User Impact:
- **Balances stale**: 3+ hours old
- **Transactions not processing**: Stuck in gap
- **Mining rewards not received**: Block height frozen
- **Explorer broken**: Shows height 777 (actual network: 4364+)
- **Node unusable**: Cannot sync, cannot participate in consensus

### Network Impact:
- **Server Beta stuck** (bootstrap node!)
- Other nodes may sync FROM Server Beta and also get stuck at 777
- **Network fragmentation risk**: Nodes at different heights can't reach consensus

---

## 🔧 IMMEDIATE FIXES NEEDED

### Fix 1: Add HTTP Gap-Fill Fallback (CRITICAL)

**Priority:** P0 (blocks all sync)

**Location:** `crates/q-api-server/src/main.rs`

**Implementation:**
```rust
// After turbo sync request sent
let storage_clone = storage.clone();
let start = current_height + 1;
let end = network_height;

tokio::spawn(async move {
    // Wait for gossipsub response
    tokio::time::sleep(Duration::from_secs(90)).await;

    // Check if gap still exists
    let current = storage_clone.get_highest_contiguous_block().await.unwrap_or(0);
    if current < end {
        warn!("🔄 [HTTP FALLBACK] Gossipsub turbo sync timed out, using HTTP gap-fill");

        // HTTP gap-fill
        for height in start..=end {
            match fetch_block_via_http(peer_address, height).await {
                Ok(block) => {
                    storage_clone.save_block(&block).await?;
                    info!("✅ [HTTP GAP-FILL] Fetched block {}", height);
                }
                Err(e) => {
                    error!("❌ [HTTP GAP-FILL] Failed to fetch block {}: {}", height, e);
                    break;
                }
            }
        }
    }
});
```

---

### Fix 2: Add Automatic Gap Resolution (CRITICAL)

**Priority:** P0 (prevents stuck nodes)

**Location:** `crates/q-api-server/src/main.rs` (gossipsub gap detection)

**Implementation:**
```rust
if network_height > current_height + 5 {
    warn!("⚠️ [GOSSIPSUB] Gap detected at height {} (received block {})",
          current_height + 1, network_height);

    // TRIGGER GAP-FILL IMMEDIATELY
    let storage_clone = storage.clone();
    let start = current_height + 1;
    let end = network_height;

    tokio::spawn(async move {
        info!("🔄 [AUTO GAP-FILL] Filling gap from {} to {}", start, end);
        trigger_turbo_sync(start, end).await;

        // Fallback to HTTP after 90s
        tokio::time::sleep(Duration::from_secs(90)).await;
        if gap_still_exists() {
            http_gap_fill(start, end).await;
        }
    });
}
```

---

### Fix 3: Add Turbo Sync Retry Logic (HIGH)

**Priority:** P1 (improves reliability)

**Implementation:**
```rust
async fn turbo_sync_with_retry(start: u64, end: u64, max_retries: u32) {
    for attempt in 1..=max_retries {
        info!("📤 [TURBO SYNC] Attempt {}/{}: Requesting blocks {}-{}",
              attempt, max_retries, start, end);

        send_gossipsub_request(start, end).await;

        // Wait for response
        tokio::time::sleep(Duration::from_secs(60)).await;

        // Check if successful
        let current = get_highest_contiguous_block().await.unwrap_or(0);
        if current >= end {
            info!("✅ [TURBO SYNC] Success on attempt {}", attempt);
            return;
        }

        warn!("⚠️ [TURBO SYNC] Attempt {} failed, retrying...", attempt);
    }

    warn!("❌ [TURBO SYNC] All {} attempts failed, falling back to HTTP", max_retries);
    http_gap_fill(start, end).await;
}
```

---

## 🚑 EMERGENCY WORKAROUND

### Option 1: Restart with Fresh Database (WORKS, but loses data)

```bash
# Stop node
systemctl stop q-api-server

# Backup corrupted database
mv ./data-mine6 ./data-mine6-stuck-777-backup

# Start fresh
systemctl start q-api-server
# Node will sync from genesis via HTTP (slow but functional)
```

---

### Option 2: Manual HTTP Gap-Fill (Advanced)

```bash
# Fetch missing blocks via HTTP API
for h in {772..4364}; do
    curl "http://185.182.185.227:8080/api/v1/blocks/$h" -o "block-$h.json"
    # Import to database (requires manual RocksDB interaction)
done
```

---

### Option 3: Deploy v0.9.62-beta with Fixes (RECOMMENDED)

1. **Fix the code** (add HTTP gap-fill fallback)
2. **Compile v0.9.62-beta** with fixes
3. **Deploy to Server Beta**
4. Node will automatically fill gap via HTTP

---

## 📝 TESTING CHECKLIST FOR FIX

### Scenario 1: Gossipsub Turbo Sync Fails (InsufficientPeers)
- [ ] Turbo sync request sent
- [ ] Wait 90 seconds
- [ ] HTTP gap-fill automatically activates
- [ ] Gap fills successfully via HTTP
- [ ] Node catches up to network height

### Scenario 2: Gap Detected via Live Blocks
- [ ] Node receives block at height 4364
- [ ] Current height: 771
- [ ] Gap detected and logged
- [ ] Automatic gap-fill triggers immediately
- [ ] Node syncs from 772 to 4364

### Scenario 3: Turbo Sync Succeeds
- [ ] Turbo sync request sent
- [ ] Response received within 60s
- [ ] Blocks applied successfully
- [ ] HTTP fallback NOT triggered (not needed)
- [ ] Node fully synced

---

## 🎯 PHASE 7 IMPLICATIONS

**This bug WILL occur in Phase 7 if not fixed:**

**Timeline:**
- Day 1 (Nov 15): First few miners join Phase 7
- **InsufficientPeers on turbo sync** (same as Phase 6!)
- Nodes get stuck at various heights
- **HTTP gap-fill is the ONLY way to sync**

**If unfixed:**
- Phase 7 launch will have same stuck-node issue
- Users frustrated, network unusable
- Emergency hotfix required (v0.9.63-beta)

**If fixed:**
- HTTP gap-fill automatically activates
- Nodes sync reliably even with few peers
- As peer count increases, turbo sync takes over
- Smooth Phase 7 launch

---

## ✅ RECOMMENDED ACTION

**BEFORE Phase 7 launch (November 15):**

1. **Implement Fix 1** (HTTP gap-fill fallback) - CRITICAL
2. **Implement Fix 2** (automatic gap resolution) - CRITICAL
3. **Test thoroughly** with stuck node scenario
4. **Compile v0.9.63-beta** with fixes
5. **Deploy to Server Beta** before Phase 7
6. **Monitor for 24 hours** to ensure gap-fill works

**DO NOT launch Phase 7 without fixing this bug!**

---

**Status:** 🚨 **CRITICAL BUG - BLOCKS ALL SYNC**
**Priority:** P0 (must fix before Phase 7)
**Next Step:** Implement HTTP gap-fill fallback
**ETA for fix:** 2-4 hours (code + compile + test)
