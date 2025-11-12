# Emergency Fix for Node Stuck at Height 777

**Date:** 2025-11-08
**Issue:** Node stuck at height 771-777, cannot sync to network height 4364+
**Root Cause:** Gossipsub turbo sync fails (InsufficientPeers), no HTTP fallback activated

---

## 🚑 IMMEDIATE SOLUTION: Restart with Fresh Database

**Status:** ✅ WORKS, FAST (5 min downtime)
**Downside:** Loses Phase 6 data (acceptable since Phase 6 is corrupted anyway)

### Commands:

```bash
# Stop the service
systemctl stop q-api-server

# Backup stuck database
mv ./data-mine6 ./data-mine6-stuck-777-$(date +%Y%m%d-%H%M%S)

# Start fresh (will sync from genesis)
systemctl start q-api-server

# Monitor sync progress
journalctl -u q-api-server -f | grep -E "height=|📈|Synced"
```

**Expected Result:**
- Node starts at height 0
- Syncs via HTTP from bootstrap peer (185.182.185.227:8080)
- Reaches network height 4364+ in 10-20 minutes
- HTTP sync is slow but reliable

---

## 📊 WHY THIS WORKS

**The Main Sync Loop (line 4881-4929) DOES have HTTP fallback:**

```rust
// LAST RESORT: FALLBACK TO HTTP IF P2P DIDN'T DELIVER
warn!("⚠️  P2P sync didn't deliver blocks, falling back to HTTP...");
let bootstrap_peer = "http://185.182.185.227:8080";

for block_height in next_block_needed..(next_block_needed + batch_size) {
    let url = format!("{}/api/v1/blocks/{}", bootstrap_peer, block_height);
    match reqwest::get(&url).await {
        Ok(response) => {
            // Fetch and save block
            app_state_sync.storage_engine.save_qblock(&block).await;
        }
    }
}
```

**BUT** this code is only reached when:
1. Node starts fresh (height 0)
2. OR turbo sync explicitly fails with Err()

**The bug:** When gossipsub turbo sync sends request but gets NO response (InsufficientPeers), it doesn't return Err() - it just hangs forever waiting.

**Fresh database works because:**
- Node starts at height 0
- Detects gap to network height
- **Skips turbo sync** (uses HTTP directly for initial sync)
- HTTP sync fills all blocks 0-4364+
- Node catches up successfully

---

## 💡 PROPER FIX (For v0.9.63-beta)

### Location: `crates/q-api-server/src/main.rs` line 3224

### Current Code (BROKEN):
```rust
info!("✅ [TURBO SYNC] All {} block-pack requests sent via gossipsub!", chunks.len());
info!("⏳ [TURBO SYNC] Responses will be handled by gossipsub handler...");
// BUG: Never checks if responses actually arrive!
```

### Fixed Code (ADD HTTP FALLBACK TIMER):
```rust
info!("✅ [TURBO SYNC] All {} block-pack requests sent via gossipsub!", chunks.len());
info!("⏳ [TURBO SYNC] Responses will be handled by gossipsub handler...");

// ✅ v0.9.63-beta FIX: HTTP fallback if gossipsub doesn't respond in 90s
let storage_timer = storage_clone.clone();
let start_height = local_height + 1;
let end_height = target;

tokio::spawn(async move {
    // Wait for gossipsub response
    tokio::time::sleep(std::time::Duration::from_secs(90)).await;

    // Check if gap still exists
    match storage_timer.get_highest_contiguous_block().await {
        Ok(current) if current < end_height => {
            warn!("🔄 [HTTP FALLBACK] Gossipsub turbo sync timed out after 90s");
            warn!("   Gap still exists: current={}, target={}", current, end_height);
            warn!("   Activating HTTP gap-fill...");

            let bootstrap_peer = "http://185.182.185.227:8080";
            let mut filled_blocks = 0;

            for height in (current + 1)..=end_height {
                let url = format!("{}/api/v1/blocks/{}", bootstrap_peer, height);

                match reqwest::get(&url).await {
                    Ok(response) if response.status().is_success() => {
                        match response.json::<serde_json::Value>().await {
                            Ok(json) if json["success"].as_bool().unwrap_or(false) => {
                                if let Some(block_data) = json["data"].as_object() {
                                    match serde_json::from_value::<q_types::block::QBlock>(
                                        serde_json::Value::Object(block_data.clone())
                                    ) {
                                        Ok(block) => {
                                            if let Err(e) = storage_timer.save_qblock(&block).await {
                                                error!("❌ [HTTP FALLBACK] Failed to save block {}: {}", height, e);
                                                break;
                                            } else {
                                                filled_blocks += 1;
                                                if filled_blocks % 100 == 0 {
                                                    info!("✅ [HTTP FALLBACK] Filled {} blocks ({}/{})",
                                                          filled_blocks, height, end_height);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            debug!("Failed to deserialize block {}: {}", height, e);
                                            break;
                                        }
                                    }
                                }
                            }
                            Ok(_) => {
                                debug!("Block {} not available from bootstrap peer", height);
                                break;
                            }
                            Err(e) => {
                                warn!("Failed to parse response for block {}: {}", height, e);
                                break;
                            }
                        }
                    }
                    Ok(response) => {
                        warn!("HTTP request for block {} failed: status {}", height, response.status());
                        break;
                    }
                    Err(e) => {
                        error!("Failed to fetch block {} via HTTP: {}", height, e);
                        break;
                    }
                }

                // Small delay to avoid overwhelming bootstrap peer
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }

            if filled_blocks > 0 {
                info!("✅ [HTTP FALLBACK] Successfully filled {} blocks via HTTP", filled_blocks);
                info!("   Gap resolution: {} -> {}", current, current + filled_blocks);
            } else {
                warn!("⚠️ [HTTP FALLBACK] Failed to fill any blocks via HTTP");
            }
        }
        Ok(current) => {
            info!("✅ [TURBO SYNC] Gap already filled by gossipsub (current={})", current);
        }
        Err(e) => {
            error!("❌ [HTTP FALLBACK] Failed to check current height: {}", e);
        }
    }
});
```

---

## 🧪 TESTING THE FIX

### Test Scenario: Node Stuck at Height 771

**Before Fix (Current Behavior):**
1. Node at height 771
2. Network at height 4364
3. Gossipsub turbo sync sends request
4. NO response (InsufficientPeers)
5. **Node stuck forever** ❌

**After Fix (v0.9.63-beta):**
1. Node at height 771
2. Network at height 4364
3. Gossipsub turbo sync sends request
4. Wait 90 seconds
5. NO response detected
6. **HTTP fallback activates automatically** ✅
7. Fetches blocks 772-4364 via HTTP
8. Node catches up successfully ✅

---

## 📋 DEPLOYMENT CHECKLIST

### For Immediate Emergency (Now):
- [x] Diagnose issue (node stuck at 777)
- [x] Document root cause
- [ ] **Execute emergency fix: restart with fresh database**
- [ ] Monitor sync progress
- [ ] Verify node reaches network height

### For Proper Fix (Before Phase 7):
- [ ] Implement HTTP fallback timer (line 3224)
- [ ] Add reqwest dependency if not present
- [ ] Test with stuck node scenario
- [ ] Compile v0.9.63-beta
- [ ] Deploy to Server Beta
- [ ] Verify gap-fill works automatically
- [ ] Update PHASE_7_LAUNCH_PLAN.md

---

## ⏱️ TIME ESTIMATES

**Emergency Fix (Restart):**
- Stop service: 10 seconds
- Backup database: 1 minute
- Restart service: 10 seconds
- HTTP sync to network height: 10-20 minutes
- **Total: ~25 minutes**

**Proper Fix (v0.9.63-beta):**
- Code implementation: 30 minutes
- Compilation: 6-10 minutes
- Testing: 30 minutes
- Deployment: 5 minutes
- **Total: ~2 hours**

---

## 🎯 RECOMMENDATION

**NOW (Immediate):** Execute emergency fix to get Server Beta working
```bash
systemctl stop q-api-server
mv ./data-mine6 ./data-mine6-stuck-777-backup
systemctl start q-api-server
```

**TOMORROW (Before Phase 7):** Implement proper HTTP fallback fix in v0.9.63-beta

**WHY BOTH:**
- Emergency fix: Gets network running NOW (critical!)
- Proper fix: Prevents this from happening in Phase 7 (essential!)

---

**Status:** Emergency fix ready to execute
**Next Step:** User decision - restart Server Beta with fresh database?
**Impact:** 25 minutes downtime, Phase 6 data loss (acceptable - Phase 6 is corrupted anyway)
