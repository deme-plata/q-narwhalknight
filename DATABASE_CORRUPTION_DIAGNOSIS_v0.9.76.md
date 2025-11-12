# Database Corruption Diagnosis - v0.9.76-beta

## Critical Finding: Complete Data Loss

### Database State
```
📊 Scan Results (data-mine6/hot):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Total blocks found: 0
   Highest block: 0
   Highest contiguous: 0

   qblock:latest pointer: 9606 (WRONG!)
   Actual blocks stored: 0
```

### Root Cause
**Catastrophic data loss** - All blocks deleted but pointer not updated:
- **Expected**: 9606 blocks based on `qblock:latest` pointer
- **Actual**: 0 blocks in database
- **Lost**: All 9606 blocks

### How This Happened
Likely causes (in order of probability):
1. **Sync-down bug** (pre v0.5.23) - Node synced backwards and deleted all blocks
2. **Database corruption** - RocksDB corruption event
3. **Manual deletion** - Accidental `rm` or database reset
4. **Disk full event** - Database writes failed, blocks lost

### Why Node is Stuck
```
Node thinks: "I have 9606 blocks"
Reality: "I have 0 blocks"

Sync logic:
- current_height = 9606 (from pointer)
- network_height = 9606 (from peers)
- blocks_behind = 0
- Conclusion: "I'm synced! No need to sync"

Result: Node never requests blocks, stays stuck forever
```

### The Fix (v0.9.76-beta)

My P2P gap fill fix will automatically recover:

**Phase 1: Gap Detection**
```rust
// On startup, repair tool fixes pointer
qblock:latest = 0  // Reset to actual height

// Node starts
current_height = 0
gap_detected_at = 1  // First missing block
```

**Phase 2: P2P Recovery**
```rust
// Gap fill kicks in (line 4895-4990 in main.rs)
gap_detected_at_height_1:
  request_blocks(1..100) from 3 peers via BlockPackCodec
  wait 15s
  check if blocks arrived

gap_detected_at_height_101:
  request_blocks(101..200) from 3 peers
  ...continues until fully synced
```

**Phase 3: Full Sync**
```
Blocks 1-100 arrive → Gap moves to 101
Blocks 101-200 arrive → Gap moves to 201
...
Blocks 9501-9606 arrive → No more gaps!
Node fully synced: height 9606
```

## Recovery Steps

### Step 1: Fix the Pointer (REQUIRED)
```bash
# Run repair tool and choose option 1
./target/release/repair-database ./data-mine6/hot

# When prompted:
# Choose option 1: Fix qblock:latest pointer to 0
# This resets the pointer to match reality (0 blocks)
```

### Step 2: Start Node with P2P Gap Fill
```bash
# Build with v0.9.76-beta gap fill fix
timeout 36000 cargo build --release --package q-api-server

# Start node
systemctl start q-api-server

# Monitor gap fill progress
journalctl -u q-api-server -f | grep "GAP FILL"
```

### Expected Output
```
🚨 CRITICAL GAP DETECTED IN BLOCKCHAIN!
   Missing block at height: 1
   Current contiguous height: 0
   Node is stuck - cannot advance past gap!

📡 [GAP FILL] Found 5 peers with height >= 1
📥 [GAP FILL] Requesting 100 blocks (1-100) from capable peers
📤 [GAP FILL] Requesting from peer 12D3KooW... (height: 9606)
✅ [GAP FILL] Gap fill request sent to peer 12D3KooW...
⏳ [GAP FILL] Waiting 15s for gap fill responses...
✅ [GAP FILL] SUCCESS! Gap filled completely!

[Repeats for blocks 101-200, 201-300, etc. until 9606]
```

### Recovery Timeline
- **Pointer fix**: Instant (< 1 second)
- **Per batch (100 blocks)**: ~15-20 seconds
- **Total batches**: ~96 (9606 / 100)
- **Estimated recovery time**: ~25-30 minutes

## Prevention (For Future)

### Safety Checks Already in Place (v0.9.0+)
```rust
// Line 97-162 in main.rs
fn verify_height_monotonicity(new_height: u64) {
    if new_height == 0 && highest_ever > 100 {
        panic!("Height reset to zero detected!");
    }
}
```

### Additional Protection Needed
1. **Periodic pointer validation** - Check pointer matches actual blocks
2. **Backup automation** - Hourly RocksDB snapshots
3. **Corruption detection** - Check block continuity on startup
4. **Sync-down alerts** - Loud warnings if height decreases

## Technical Details

### Database Structure
```
Column Families in data-mine6/hot:
✅ default, blocks, dag_vertices, bullshark_cert
✅ manifest, transactions, balances
✅ block_hash_to_height, ai_chats, ai_credits
✅ ai_transactions, ai_treasury, ai_attachments
✅ payment_proposals, payment_votes, payment_locks
✅ banned_peers, sync_certificates, peer_trust

Total: 19 column families (all healthy)
```

### Pointer Format
```rust
// qblock:latest key in blocks CF
Key: b"qblock:latest"
Value: [u8; 8]  // u64 big-endian
Current: [0, 0, 0, 0, 0, 0, 37, 134]  // 9606 in big-endian
Should be: [0, 0, 0, 0, 0, 0, 0, 0]   // 0 in big-endian
```

### Block Storage Format
```rust
// Individual block keys
Key: format!("qblock:height:{}", height)
Example: "qblock:height:1", "qblock:height:2", etc.

Current state:
- "qblock:height:0" → NOT FOUND
- "qblock:height:1" → NOT FOUND
- ...
- "qblock:height:9606" → NOT FOUND

All blocks are MISSING!
```

## Comparison: Other Nodes

### Healthy Node Example
```
Total blocks found: 9606
Highest block: 9606
Highest contiguous: 9606
qblock:latest pointer: 9606 ✅ CORRECT!
```

### This Node (data-mine6)
```
Total blocks found: 0 ❌
Highest block: 0 ❌
Highest contiguous: 0 ❌
qblock:latest pointer: 9606 ❌ WRONG!
```

## Lessons Learned

1. **Never trust pointers** - Always validate against actual data
2. **Atomic operations** - Block deletion must update pointer atomically
3. **Fail-safe defaults** - If pointer corrupt, scan for truth
4. **P2P recovery** - Don't rely on centralized bootstrap
5. **Continuous validation** - Check database integrity regularly

## Status

- ✅ **Diagnosis Complete**: Total data loss confirmed
- ✅ **Root Cause Identified**: Sync-down or corruption event
- ✅ **Fix Implemented**: P2P gap fill (v0.9.76-beta)
- ✅ **Recovery Plan**: Fix pointer + restart with gap fill
- ⏳ **Pending**: Pointer repair + node restart
- ⏳ **ETA to Full Sync**: ~30 minutes

## Next Steps

1. Run repair tool to fix pointer
2. Rebuild with v0.9.76-beta
3. Restart node
4. Monitor gap fill logs
5. Verify full sync to height 9606+

---

**Version**: v0.9.76-beta
**Date**: 2025-11-09
**Severity**: CRITICAL - Complete data loss
**Recovery**: Automated via P2P gap fill
