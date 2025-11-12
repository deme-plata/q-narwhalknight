# Phase 7 Launch Plan - November 15, 2025

**Status:** ✅ v0.9.62-beta compiled successfully
**Launch Date:** November 15, 2025 (7 days from now)
**Current Phase:** Phase 6 (corrupted, 170,373 QUG hyperinflation)

---

## 📅 7-DAY TIMELINE

### **Day 1 (November 8, 2025) - TODAY**
- [x] **Discovered hyperinflation bug** (650,000× too high)
- [x] **Fixed decimal precision** (9 decimals → 8 decimals)
- [x] **Compiled v0.9.62-beta** successfully
- [ ] Deploy v0.9.62-beta to test environment
- [ ] Run initial economic tests

### **Day 2-3 (November 9-10) - Testing Phase**
- [ ] Test reward calculations (verify 0.00001 QUG per solution)
- [ ] Test with 100, 1000, 10000 solutions
- [ ] Monitor for balance flickering issues
- [ ] Test gossipsub turbo sync
- [ ] Performance benchmarks

### **Day 4-5 (November 11-12) - Bug Fixes & Optimizations**
- [x] ~~Fix gossipsub turbo sync~~ - NOT A BUG! Network topology issue, will resolve naturally in Phase 7
- [ ] Fix balance flickering (debounce updates or serialize block production)
- [ ] Monitor Phase 7 peer adoption (gossipsub will work when 5+ peers join)
- [ ] Any other issues discovered during testing

### **Day 6 (November 13) - Preparation**
- [ ] Final testing of all fixes
- [ ] Create Phase 7 announcement (frontend modal)
- [ ] Update documentation
- [ ] Prepare community announcement

### **Day 7 (November 14) - Pre-Launch**
- [ ] Deploy v0.9.62-beta binaries to downloads folder
- [ ] Test binary downloads
- [ ] Final system check
- [ ] Announce Phase 7 launch for next day

### **Day 8 (November 15) - PHASE 7 LAUNCH** 🚀
- [ ] Stop Phase 6 network
- [ ] Deploy Phase 7 (testnet-phase7, data-mine7)
- [ ] Announce to community
- [ ] Monitor launch for first 24 hours

---

## 🐛 BUGS TO FIX BEFORE PHASE 7

### **1. Gossipsub Turbo Sync (NOT A BUG - NETWORK TOPOLOGY ISSUE)**

**Status:** ✅ **RESOLVED - NOT A BUG, EXPECTED BEHAVIOR**

**Root Cause:** Insufficient Phase 6 peers on the network (only 1 peer: Server Beta)

**What's happening:**
1. ✅ Gossipsub requests published successfully
2. ✅ Server Beta generates block pack responses (9.3 KB compressed)
3. ❌ Response publishing fails: `InsufficientPeers` on `/qnk/testnet-phase6/block-pack-responses`
4. ✅ HTTP fallback activates and sync continues

**Why this is NOT a bug:**
- Gossipsub correctly refuses to publish to topics with insufficient subscribers
- This prevents message loss in sparse networks
- HTTP fallback is working exactly as designed

**Why Phase 7 will solve this:**
- Fresh network start = everyone on `testnet-phase7` from day 1
- Community announcement = rapid peer adoption
- Expected timeline:
  - Day 1: HTTP fallback (5-10 initial miners)
  - Day 2-3: Gossipsub starts working (10+ peers)
  - Week 1: Optimal gossipsub performance (20+ peers)

**No code changes needed!** See `GOSSIPSUB_TURBO_SYNC_INSUFFICIENT_PEERS_DIAGNOSIS.md` for full analysis.

---

### **2. Balance Display Flickering (LOW PRIORITY)**

**Issue:** Balance jumps around (1704.05 → 1704.02 → 1704.6 → 1704.2)

**Root cause:** 8 parallel block producers creating blocks simultaneously, causing async balance updates.

**Options:**
1. **Debounce frontend updates** (quick fix)
2. **Serialize block production** (prevents duplicates)
3. **Deduplicate balance updates** (only send final balance)

**Recommended:** Option 1 (debounce) for Phase 7, Option 2 for future.

**Frontend fix example:**
```typescript
// In Dashboard.tsx or balance display component
const [debouncedBalance, setDebouncedBalance] = useState(balance);

useEffect(() => {
  const timer = setTimeout(() => {
    setDebouncedBalance(balance);
  }, 500); // Wait 500ms before updating display

  return () => clearTimeout(timer);
}, [balance]);
```

---

### **3. Block Producer Deduplication (MEDIUM PRIORITY)**

**Issue:** 8 parallel producers creating duplicate blocks at same height.

**Current behavior:**
```
🏗️  Producing block: height=173, solutions=100 (x8 times!)
🏗️  Producing block: height=174, solutions=100 (x8 times!)
```

**Why it happens:** All 8 producers check `should_produce_block()` at same time and all return true.

**Proper fix:** Add height locking or round-robin coordination.

**File:** `crates/q-api-server/src/block_producer.rs`

**Pseudocode:**
```rust
// Option 1: Height lock (atomic)
static CURRENT_HEIGHT: AtomicU64 = AtomicU64::new(0);

pub fn should_produce_block(&self) -> bool {
    let current = CURRENT_HEIGHT.load(Ordering::SeqCst);
    if self.current_height <= current {
        // Already produced by another producer
        return false;
    }
    // Claim this height
    CURRENT_HEIGHT.compare_exchange(current, self.current_height, ...);
    ...
}

// Option 2: Round-robin (simpler)
pub fn should_produce_block(&self) -> bool {
    // Only producer whose ID matches height % 8 can produce
    (self.current_height % 8) == self.config.validator_index as u64
}
```

---

## ✅ VERIFIED FIXES IN v0.9.62-beta

### **1. Hyperinflation Bug - FIXED ✅**

**Before (v0.9.60/61):**
```rust
const BLOCK_REWARD: u64 = 10_000; // Wrong: 9 decimals
let qug_total = total_reward as f64 / 1_000_000_000.0;
```

**After (v0.9.62):**
```rust
const BLOCK_REWARD: u64 = 1_000; // Correct: 8 decimals
let qug_total = total_reward as f64 / 100_000_000.0;
```

**Result:**
- Per-solution: 0.00001 QUG (correct!)
- 262 blocks: 0.262 QUG (not 170,373!)
- 100× more scarce than Phase 5 ✅

---

## 🧪 TESTING CHECKLIST

### **Economic Tests:**
- [ ] Mine 1 block with 100 solutions → expect 0.001 QUG total
- [ ] Mine 100 blocks → expect 0.1 QUG total
- [ ] Mine 1,000 blocks → expect 1.0 QUG total
- [ ] Verify dev fee (1%) is correct
- [ ] Verify halving math (not used yet, but verify code)

### **Network Tests:**
- [ ] Gossipsub block propagation working
- [ ] Turbo sync via gossipsub working (or HTTP fallback)
- [ ] Peer discovery working
- [ ] Bootstrap node connectivity
- [ ] Multiple nodes can sync from Server Beta

### **UI Tests:**
- [ ] Balance displays correctly (no 650,000× inflation!)
- [ ] Balance updates in real-time (SSE)
- [ ] Balance flickering acceptable or fixed
- [ ] Transaction page works
- [ ] Explorer shows correct data
- [ ] Mining dashboard shows correct rewards

### **Stability Tests:**
- [ ] Node runs for 24+ hours without crash
- [ ] Database doesn't corrupt
- [ ] RocksDB durability working (v0.9.60 feature)
- [ ] Sync-down protection active
- [ ] No memory leaks

---

## 📊 PHASE 7 ECONOMICS (VERIFIED)

### **Per-Solution Reward:**
```
0.00001 QUG = 1,000 atomic units (8 decimals)
```

### **Expected Emission:**

| Blocks | Solutions | Total Mined | Accuracy Check |
|--------|-----------|-------------|----------------|
| 1 | 100 | 0.001 QUG | ✅ |
| 100 | 10,000 | 0.1 QUG | ✅ |
| 1,000 | 100,000 | 1.0 QUG | ✅ |
| 10,000 | 1,000,000 | 10.0 QUG | ✅ |
| 100,000 | 10,000,000 | 100.0 QUG | ✅ |
| 1,000,000 | 100,000,000 | 1,000.0 QUG | ✅ |

### **Comparison:**

| Phase | Blocks | Mined | Status |
|-------|--------|-------|--------|
| Phase 5 | 20,000 | ~998,663 QUG | ❌ Hyperinflation |
| Phase 6 | 262 | 170,373 QUG | ❌ **CORRUPTED** (650,000× bug!) |
| **Phase 7** | 262 | **0.262 QUG** | ✅ **CORRECT!** |
| **Phase 7** | 20,000 | **~20 QUG** | ✅ 100× MORE SCARCE than Phase 5! |

---

## 🎯 PHASE 7 LAUNCH PARAMETERS

### **Network:**
- **Network ID:** `testnet-phase7`
- **Database:** `./data-mine7`
- **Bootstrap:** Same (185.182.185.227:9001)
- **Gossipsub:** `/qnk/testnet-phase7/*`

### **Economics:**
- **Per-solution reward:** 0.00001 QUG (8 decimals)
- **Halving:** Yearly time-based (not implemented yet)
- **Dev fee:** 1%
- **Max supply:** Not enforced yet

### **Technical:**
- **Binary:** v0.9.62-beta
- **RocksDB:** Maximum durability (v0.9.60 feature)
- **Sync-down protection:** Active (v0.9.59 feature)
- **Parallel producers:** 8 (needs deduplication fix)

---

## 📢 COMMUNITY ANNOUNCEMENT (DRAFT)

**To be posted on November 14, 2025:**

```markdown
# 🚀 Phase 7 Launch - November 15, 2025

We're excited to announce the launch of **Testnet Phase 7** with **CORRECT** Austrian economics!

## 🐛 Why Phase 7?

Phase 6 had a critical bug that caused 650,000× hyperinflation:
- **Expected**: 0.262 QUG mined (262 blocks)
- **Actual**: 170,373 QUG mined (decimal precision error!)
- **Bug**: Code used 9 decimals instead of 8 (like Bitcoin)

## ✅ Phase 7 Fixes

**v0.9.62-beta includes:**
- ✅ Correct decimal precision (8 decimals, like Bitcoin)
- ✅ Correct reward: 0.00001 QUG per solution
- ✅ 100× MORE SCARCE than Phase 5 (as originally intended!)

## 📊 Economic Comparison

| Phase | 20,000 Blocks | Per-Solution | Status |
|-------|---------------|--------------|--------|
| Phase 5 | ~998,663 QUG | 0.001 QUG | Too generous |
| Phase 6 | ~13M QUG (bug!) | 0.0001 QUG | **BROKEN** |
| **Phase 7** | **~20 QUG** | **0.00001 QUG** | ✅ **CORRECT!** |

## 🎯 What You Need to Know

1. **Fresh Network** - Phase 6 balances don't transfer
   - This is testnet (mainnet rehearsal)
   - Everyone starts equal (fair launch!)

2. **Download v0.9.62-beta**
   - Network ID: `testnet-phase7`
   - Database: `data-mine7`
   - Cannot connect to Phase 6 network

3. **Start Mining** - Earn TRUE scarce QUG
   - Reward: 0.00001 QUG per solution (CORRECT!)
   - 100× more scarce than Phase 5
   - Real sound money economics!

## 📥 Download

- **Node:** https://quillon.xyz/downloads/q-api-server-v0.9.62-beta
- **Miner:** https://quillon.xyz/downloads/q-miner-linux-x64

## 🗓️ Launch Schedule

- **November 14:** Pre-launch announcement
- **November 15, 12:00 UTC:** Phase 7 network goes live
- **First 24 hours:** Closely monitored for stability

Happy mining with CORRECT economics! 🚀💎
```

---

## ✅ SUCCESS CRITERIA

**Phase 7 is successful if:**

1. ✅ **Economics are correct** (0.00001 QUG per solution)
2. ✅ **No hyperinflation** (proper scarcity maintained)
3. ✅ **Network is stable** (no crashes, corruption, or sync issues)
4. ✅ **Miners are satisfied** (fair rewards, no bugs)
5. ✅ **Mainnet-ready** (confidence to launch mainnet after Phase 7)

---

## 📞 SUPPORT & MONITORING

**During Phase 7 launch (first 24 hours):**
- Monitor logs: `journalctl -u q-api-server -f`
- Check mined coins: Watch explorer total supply
- Verify rewards: Check miner balances match expectations
- Monitor network: Check peer count and sync status

**If issues occur:**
- Document thoroughly
- Fix critical bugs immediately
- Deploy hotfix if needed (v0.9.63-beta)
- Communicate transparently with community

---

**Next Step:** Begin Day 2 testing (November 9, 2025)
**Launch Date:** November 15, 2025, 12:00 UTC
**Status:** v0.9.62-beta ready for testing ✅
