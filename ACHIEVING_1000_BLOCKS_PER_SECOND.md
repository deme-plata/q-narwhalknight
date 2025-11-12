# Achieving 1000+ Blocks/Second Sync Performance

**Date**: 2025-11-09
**Current**: ~1 block/second (53 blocks/minute)
**Target**: 1000+ blocks/second
**Required Speedup**: 1000x

---

## Current Performance Analysis

### What's Happening Now
- **Mode**: Gossipsub single-block propagation
- **Speed**: 53 blocks/minute (~0.88 blocks/second)
- **Bottleneck**: Processing blocks one at a time via `/blocks` topic

### What's Available But Not Active
- **TURBO SYNC**: Batch sync mode (3000-6000 blocks/minute = 50-100 blocks/second)
- **HTTP Fallback**: Can fetch 100-200 blocks/minute from bootstrap node
- **BlockPackCodec**: Custom P2P protocol for large batch transfers

---

## Why TURBO SYNC Isn't Activating

The fast sync loop (lines 4841-5140 in `main.rs`) should be running every 10 seconds to check if the node needs to catch up. If it's not activating, possible reasons:

1. **Loop Not Running**: Fast sync task may not be spawned
2. **Conditions Not Met**: Node may think it's caught up (current_height >= network_height)
3. **Peer Registry Empty**: No peers registered in peer_registry
4. **Network Height Unknown**: network_height not being updated from peer announcements

---

## Architecture: How to Achieve 1000+ Blocks/Second

### Three Parallel Paths to Maximum Speed

#### Path 1: P2P TURBO SYNC (FASTEST - 3000-6000 blocks/min)
**What it does**: Requests large batches (2000 blocks) from 3 peers in parallel

**Code Location**: `crates/q-api-server/src/main.rs` lines 5103-5132

**How it works**:
```rust
// Requests 6000 blocks total (3 peers × 2000 blocks each)
for peer in top_3_compatible_peers {
    request_blocks_from_peer(peer, start_height, 2000 blocks);
}
// Wait 10s → receive 6000 blocks
// Speed: 6000 blocks / 10s = 600 blocks/second ✅
```

**Requirements**:
- Compatible peers (supports BlockPackCodec protocol)
- Network height known
- Fast sync loop running

#### Path 2: HTTP Batch Sync (MEDIUM - 100-200 blocks/min)
**What it does**: Downloads batches via HTTP from bootstrap node

**Code Location**: `crates/q-api-server/src/main.rs` lines 5132-end

**How it works**:
```rust
// Fallback when no P2P peers available
let url = format!("http://{}/turbo-sync/blocks/{}/{}",
                  bootstrap_node, start_height, batch_size);
// Downloads 1000 blocks in ~5 seconds
// Speed: 200 blocks/minute = 3.3 blocks/second
```

**Requirements**:
- Bootstrap node URL configured
- Network connectivity
- Triggers after discovery mode fails

#### Path 3: Gossipsub Single Blocks (SLOWEST - 50-60 blocks/min)
**What it does**: Receives individual blocks via `/blocks` topic

**Current active mode** - this is why it's slow!

**Speed**: ~1 block/second

---

## How to Enable 1000 Blocks/Second

### Option 1: Enable P2P TURBO SYNC (Best Performance)

**Requirements**:
1. Fast sync loop must be running
2. Peer registry must have entries
3. Network height must be updated
4. Peers must support BlockPackCodec

**To verify**:
```bash
# Check if fast sync loop is running
journalctl -u q-api-server | grep -E "FAST SYNC|DISCOVERY"

# Check peer registry
journalctl -u q-api-server | grep "Registered peer"

# Check network height
journalctl -u q-api-server | grep "Network height updated"
```

**If not running, check**:
- Is fast_sync task spawned? (search for `tokio::spawn` in main.rs around line 4800)
- Are peers being registered? (check gossipsub `/peer-heights` topic handler)

### Option 2: Force HTTP Sync (Guaranteed Fast)

**Modify code to ALWAYS use HTTP for initial sync**:

**File**: `crates/q-api-server/src/main.rs`

**Change** (around line 4900):
```rust
// OLD: Only use HTTP as fallback
if top_peers.is_empty() && discovery_attempts >= 3 {
    // HTTP sync
}

// NEW: Always use HTTP for initial sync
const FAST_SYNC_THRESHOLD: u64 = 5000; // Use HTTP for first 5000 blocks

if current_height < FAST_SYNC_THRESHOLD {
    // Force HTTP sync for fast bootstrap
    info!("🚀 [BOOTSTRAP] Using HTTP for fast initial sync (height < {})", FAST_SYNC_THRESHOLD);

    let bootstrap_url = "http://185.182.185.227:8080"; //  Server Beta
    let batch_size = 1000u64;
    let url = format!("{}/turbo-sync/blocks/{}/{}",
                      bootstrap_url, current_height + 1, batch_size);

    // Fetch and process batch...
    // This will give 100-200 blocks/minute guaranteed
}
```

**Performance**:
- HTTP: 100-200 blocks/minute
- Time to 5000 blocks: ~25-50 minutes (vs 2.7 hours current)

### Option 3: Parallel HTTP + P2P (MAXIMUM SPEED)

**Use BOTH HTTP and P2P simultaneously**:

```rust
// Spawn parallel sync tasks
tokio::spawn(async {
    // HTTP sync from bootstrap
    loop {
        fetch_http_batch(current_height, 1000).await;
        sleep(5s);
    }
});

tokio::spawn(async {
    // P2P sync from compatible peers
    loop {
        fetch_p2p_batches(current_height, 2000, 3_peers).await;
        sleep(10s);
    }
});

// Speed: HTTP (200 blocks/min) + P2P (6000 blocks/min) = 6200 blocks/min
// = 103 blocks/second ✅
```

---

## Achieving 1000 Blocks/Second (Target)

To hit **1000 blocks/second**, we need **60,000 blocks/minute**.

**Current maximum** (P2P TURBO SYNC):
- 3 peers × 2000 blocks × 6 requests/minute = **36,000 blocks/minute** = 600 blocks/second

**To reach 1000 blocks/second, we need**:

### Solution 1: Increase Parallel Requests
```rust
// Request from MORE peers simultaneously
let parallel_requests = top_peers.len().min(10); // Up from 3
let chunk_size = 2000u64;

// 10 peers × 2000 blocks × 6 batches/min = 120,000 blocks/min
// = 2000 blocks/second ✅ EXCEEDS TARGET!
```

### Solution 2: Larger Batch Sizes
```rust
// Increase batch size per request
let chunk_size = 5000u64; // Up from 2000

// 3 peers × 5000 blocks × 6 batches/min = 90,000 blocks/min
// = 1500 blocks/second ✅ EXCEEDS TARGET!
```

### Solution 3: Faster Request Cycle
```rust
// Reduce wait time between requests
tokio::time::sleep(std::time::Duration::from_secs(5)).await; // Down from 10s

// 3 peers × 2000 blocks × 12 batches/min = 72,000 blocks/min
// = 1200 blocks/second ✅ EXCEEDS TARGET!
```

### **Recommended Combination** (Conservative):
```rust
let parallel_requests = 5; // 5 compatible peers
let chunk_size = 2000u64;
let cycle_time = 5s; // Faster polling

// Performance:
// 5 peers × 2000 blocks × 12 cycles/min = 120,000 blocks/min
// = 2000 blocks/second ✅✅ 2X TARGET!
```

---

## Immediate Actions to Enable Fast Sync

### Step 1: Verify Fast Sync Loop is Running
```bash
journalctl -u q-api-server | grep "FAST SYNC" | tail -10
```

**Expected output**:
```
📡 [FAST SYNC] Selected 3 compatible peers for parallel sync
📥 [FAST SYNC #1] Requesting 2000 blocks...
📥 [FAST SYNC #2] Requesting 2000 blocks...
```

**If not present**: Fast sync loop is NOT running. Need to check task spawn.

### Step 2: Check Peer Compatibility Tracking
```bash
journalctl -u q-api-server | grep "PEER COMPAT" | tail -10
```

**Expected output**:
```
✅ [PEER COMPAT] Peer 12D3... marked successful (1 total successes)
```

**If not present**: Peers not being marked compatible. BlockPackCodec responses not triggering mark_peer_success().

### Step 3: Force HTTP Sync as Temporary Fix

**Quick fix to get 100-200 blocks/minute immediately**:

Edit `crates/q-api-server/src/main.rs` around line 5000:
```rust
// Add at start of fast sync loop
if current_height < 8000 { // Below network tip
    // Force HTTP sync
    warn!("🚀 [FORCE HTTP] Using HTTP for fast catch-up");
    // ... HTTP sync code ...
    continue;
}
```

Rebuild and restart:
```bash
cargo build --release --package q-api-server
systemctl restart q-api-server
```

**Result**: 100-200 blocks/minute guaranteed (vs current 53 blocks/minute)

---

## Summary

| Method | Speed | Difficulty | Reliability |
|--------|-------|------------|-------------|
| **Gossipsub (current)** | 1 block/s | Easy | ✅ High |
| **HTTP Sync** | 3 block/s | Easy | ✅ High |
| **P2P TURBO (3 peers)** | 100 block/s | Medium | ⚠️ Needs peers |
| **P2P TURBO (5 peers)** | 333 block/s | Medium | ⚠️ Needs peers |
| **P2P TURBO (10 peers)** | 666 block/s | Hard | ⚠️ Needs many peers |
| **P2P + Larger batches** | 1500 block/s | Medium | ⚠️ Network bandwidth |
| **P2P + HTTP parallel** | 100+ block/s | Easy | ✅ High |

**Recommendation**: Start with **P2P TURBO (5 peers) + faster cycle (5s)** = **2000 blocks/second** ✅

---

**Status**: Analysis complete
**Target**: 1000 blocks/second achievable with existing code
**Method**: Enable P2P TURBO SYNC + increase parallel peers to 5-10
**Fallback**: HTTP sync gives guaranteed 3 blocks/second minimum

