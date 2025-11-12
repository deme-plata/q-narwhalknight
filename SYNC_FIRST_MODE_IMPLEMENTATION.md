# Sync-First Mode Implementation - Block Production While Syncing Fix

## Problem Description

When a node starts and connects to the P2P network, it faces a conflict:
1. **Receives blocks from gossipsub** (syncing from the network)
2. **Produces its own blocks locally** (mining/validation)

This creates a race condition where the node:
- Receives block #756 from network → saves it → updates height to 756
- Time-based producer immediately produces block #757 locally
- BUT the network is actually at block 107,498+!

Result: **The node never catches up** because it's producing blocks as fast as it's receiving them.

## Solution: Sync-First Mode

Implemented a **sync-first mode** that:
1. **Tracks the highest block height** seen from the P2P network
2. **Pauses block production** until the node is synced (within 10 blocks of network height)
3. **Resumes mining** once caught up

### Implementation Details

#### 1. Network Height Tracking (`AppState`)

Added `highest_network_height` field to track maximum block height seen from peers:

```rust
// crates/q-api-server/src/lib.rs:485
pub highest_network_height: Arc<std::sync::atomic::AtomicU64>,
```

Initialized in `AppState::new()` and `AppState::new_with_networks()`:
```rust
highest_network_height: Arc::new(std::sync::atomic::AtomicU64::new(0)),
```

#### 2. Gossipsub Block Handler Update

When receiving blocks from the P2P network, update the highest network height:

```rust
// crates/q-api-server/src/main.rs:1890-1895
// SYNC MODE: Update highest network height seen
let current_highest = app_state_gossip.highest_network_height.load(std::sync::atomic::Ordering::Relaxed);
if block_height > current_highest {
    app_state_gossip.highest_network_height.store(block_height, std::sync::atomic::Ordering::Relaxed);
    info!("📈 Network height updated: {} -> {} (from received block)", current_highest, block_height);
}
```

#### 3. Time-Based Block Production Loop

Added sync check before producing blocks:

```rust
// crates/q-api-server/src/main.rs:1593-1608
// SYNC MODE CHECK: Don't produce blocks if we're catching up to the network
let current_height = app_state_block_producer.node_status.read().await.current_height;
let network_height = app_state_block_producer.highest_network_height.load(std::sync::atomic::Ordering::Relaxed);

// Allow mining if:
// 1. We're within 10 blocks of network height (synced), OR
// 2. Network height is 0 (no peers or we're bootstrap node)
let sync_threshold = 10;
let is_synced = network_height == 0 || (network_height > 0 && current_height + sync_threshold >= network_height);

if !is_synced {
    // We're behind - skip block production and let sync catch up
    debug!("⏸️  Block production paused: syncing {} blocks behind (current: {}, network: {})",
          network_height.saturating_sub(current_height), current_height, network_height);
    continue;
}
```

#### 4. Mining Submission Handler

Applied the same sync check to the mining submission handler:

```rust
// crates/q-api-server/src/main.rs:1407-1417
// SYNC MODE CHECK: Don't produce blocks if we're catching up
let current_height = app_state_mining.node_status.read().await.current_height;
let network_height = app_state_mining.highest_network_height.load(std::sync::atomic::Ordering::Relaxed);
let sync_threshold = 10;
let is_synced = network_height == 0 || (network_height > 0 && current_height + sync_threshold >= network_height);

if !is_synced {
    debug!("⏸️  Block production paused (mining handler): syncing {} blocks behind",
          network_height.saturating_sub(current_height));
    continue; // Skip block production, continue processing submissions
}
```

## Behavior

### Bootstrap Node (No Peers)
- `network_height` = 0
- `is_synced` = true (always)
- **Produces blocks normally**

### Syncing Node (Behind Network)
- Current height: 756
- Network height: 107,498
- Behind by: 106,742 blocks
- **Block production PAUSED** ⏸️
- Logs: `⏸️  Block production paused: syncing 106742 blocks behind`

### Synced Node (Caught Up)
- Current height: 107,489
- Network height: 107,498
- Behind by: 9 blocks (within threshold)
- **Block production ACTIVE** ✅
- Produces blocks normally, can mine

### Mining While Syncing
- Mining submissions are still **accepted** and queued
- Solutions accumulate in the block producer pool
- Once synced, queued solutions are included in blocks
- No mining work is lost

## Testing

To test the sync-first mode:

```bash
# Start a fresh node (will be behind the network)
./q-api-server --port 9080

# Check logs for sync messages:
# 📦 Received block 756 (height=756) from network
# 📈 Network height updated: 0 -> 756 (from received block)
# ⏸️  Block production paused: syncing 106742 blocks behind (current: 756, network: 107498)
#
# ... (receives more blocks) ...
#
# 📦 Received block 107488 (height=107488) from network
# 📈 Network height updated: 107487 -> 107488
# ⏰ PHASE 2: TIME-BASED PARALLEL BLOCK PRODUCED by Producer #7: Height 107489
# (Block production resumed - node is synced!)
```

## Configuration

The sync threshold is configurable:
```rust
let sync_threshold = 10; // Allow mining within 10 blocks of network
```

Adjust this value based on:
- **Smaller threshold (e.g., 5)**: More aggressive, requires closer sync
- **Larger threshold (e.g., 20)**: More lenient, starts mining sooner

## Benefits

1. ✅ **Prevents local block production during initial sync**
2. ✅ **Node catches up to network height quickly**
3. ✅ **Resumes mining automatically once synced**
4. ✅ **Bootstrap nodes unaffected** (network_height = 0)
5. ✅ **Mining solutions preserved** (queued during sync)
6. ✅ **No configuration required** (works automatically)

## Version

Implemented in: **v0.3.0-beta**
Related Fix: Miner real-time block synchronization via SSE

## Files Modified

- `crates/q-api-server/src/lib.rs` - Added `highest_network_height` field
- `crates/q-api-server/src/main.rs` - Added sync-first checks and network height tracking
- `crates/q-miner/src/main.rs` - Added real-time block notification via SSE (separate fix)
