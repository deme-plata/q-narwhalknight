# Technical Review: DAG Read-Path Fix
## Teaching the Block Server to Check Both Key Formats
### Date: 2026-04-17 | Risk: ZERO (read-only change, no DB writes)

---

## 1. The Problem (One Sentence)

The block-serving function searches for `qblock:height:{N}` but 545,710 blocks are stored as `qblock:dag:{N}:{proposer}` — it only checks one label when the file has a different label.

---

## 2. What We Change

### ONE function: `get_qblocks_range()` in `crates/q-storage/src/lib.rs`

**Before:**
```rust
pub async fn get_qblocks_range(&self, start_height: u64, limit: usize) -> Result<Vec<QBlock>> {
    // Build keys: "qblock:height:2000000", "qblock:height:2000001", ...
    let keys = (start..=end).map(|h| format!("qblock:height:{}", h)).collect();
    let results = self.hot_db.multi_get(CF_BLOCKS, &keys).await?;
    // Returns empty for heights 1.6M-10M because those use qblock:dag: format
}
```

**After:**
```rust
pub async fn get_qblocks_range(&self, start_height: u64, limit: usize) -> Result<Vec<QBlock>> {
    // Step 1: Try fast path (existing behavior, unchanged)
    let keys = (start..=end).map(|h| format!("qblock:height:{}", h)).collect();
    let results = self.hot_db.multi_get(CF_BLOCKS, &keys).await?;
    
    // Step 2: If we got everything, return (fast path succeeded)
    if blocks.len() == limit { return Ok(blocks); }
    
    // Step 3: For missing heights, check qblock:dag:{N}: prefix
    for height in missing_heights {
        let prefix = format!("qblock:dag:{}:", height);
        if let Some(entry) = self.hot_db.scan_prefix(CF_BLOCKS, prefix.as_bytes()).await?.first() {
            // Found it under DAG key — deserialize and add
            blocks.push(deserialize(entry));
        }
    }
    
    blocks.sort_by_key(|b| b.header.height);
    Ok(blocks)
}
```

### What this does NOT change:
- Does NOT write anything to the database
- Does NOT modify or delete any existing keys
- Does NOT change block validation or consensus
- Does NOT change P2P protocol or message format
- Does NOT affect nodes that are already synced (they never request blocks below their height)
- Does NOT change `save_dag_layer_block()` or any write path

### What this fixes:
- A syncing peer asks Epsilon for blocks at height 2M → Epsilon now finds them
- Checkpoint sync discovers blocks start at ~100K (not 10M)
- New nodes sync the full chain history instead of skipping 8.4M blocks

---

## 3. Files Modified

| File | Change | Lines |
|------|--------|-------|
| `crates/q-storage/src/lib.rs` | Add DAG fallback to `get_qblocks_range()` | ~15 lines added |
| `crates/q-storage/tests/dag_read_path_tests.rs` | New test file | ~200 lines |

That's it. One function enhanced, one test file added.

---

## 4. Test Plan (Docker on Gamma ONLY)

**We do NOT deploy to any production node until ALL tests pass on a disposable Docker container on Gamma (109.205.176.60).**

### Test 1: Unit Test — DAG Fallback Logic

```rust
#[tokio::test]
async fn test_get_qblocks_range_finds_dag_entries() {
    // Setup: create a test DB with:
    //   qblock:dag:2000941:abc123 → serialized block at height 2000941
    //   qblock:dag:2000942:abc123 → serialized block at height 2000942
    //   qblock:height:10000031    → serialized block at height 10000031
    //   (NO qblock:height:2000941 or qblock:height:2000942)
    
    // Act: call get_qblocks_range(2000940, 5)
    
    // Assert:
    //   - Returns 2 blocks (at 2000941 and 2000942)
    //   - Does NOT return a block at 2000940 (doesn't exist in either format)
    //   - Block data deserializes correctly
    //   - Heights are sorted ascending
}

#[tokio::test]
async fn test_get_qblocks_range_prefers_height_over_dag() {
    // Setup: create a test DB with:
    //   qblock:height:5000 → block A
    //   qblock:dag:5000:abc → block B (same height, different entry)
    
    // Act: call get_qblocks_range(5000, 1)
    
    // Assert: Returns block A (height format takes priority)
    //   The DAG entry is not used when height entry exists
}

#[tokio::test]
async fn test_get_qblocks_range_empty_range() {
    // Setup: DB with no blocks in range 3000-4000
    
    // Act: call get_qblocks_range(3000, 1000)
    
    // Assert: Returns empty vec, no errors, no panics
}

#[tokio::test]
async fn test_get_qblocks_range_mixed_formats() {
    // Setup: create a test DB with:
    //   qblock:dag:1000:abc → block at 1000 (DAG only)
    //   qblock:height:1005  → block at 1005 (height only)
    //   qblock:dag:1010:abc → block at 1010 (DAG only)
    //   qblock:height:1010  → block at 1010 (height AND DAG — both exist)
    
    // Act: call get_qblocks_range(999, 15)
    
    // Assert: Returns blocks at 1000, 1005, 1010 (deduplicated, sorted)
    //   Height 1010 uses height-format (priority), not DAG
}

#[tokio::test]
async fn test_get_qblocks_range_multiple_proposers_same_height() {
    // Setup: create a test DB with:
    //   qblock:dag:5000:proposer_A → block from miner A
    //   qblock:dag:5000:proposer_B → block from miner B
    
    // Act: call get_qblocks_range(5000, 1)
    
    // Assert: Returns exactly 1 block (first proposer wins, deterministic)
    //   Does NOT return 2 blocks for the same height
}

#[tokio::test]
async fn test_fast_path_still_works() {
    // Setup: DB with ONLY qblock:height: keys (no DAG entries)
    //   qblock:height:100, 101, 102, 103, 104
    
    // Act: call get_qblocks_range(100, 5)
    
    // Assert: Returns 5 blocks
    //   Fast path handles it, DAG fallback is never reached
    //   Performance is identical to before the change
}
```

### Test 2: Integration Test — Docker Sync on Gamma

```bash
# Step 1: Build new binary with the fix
cargo build --release --package q-api-server

# Step 2: SCP to Gamma
scp target/release/q-api-server root@109.205.176.60:/opt/orobit/shared/q-narwhalknight/node5/q-api-server-v10.3.6-dagfix

# Step 3: Start Docker container on Gamma with the FIXED binary
#   Uses Gamma's port 8089 (not conflicting with production on 8808)
#   Points at Epsilon (which has 545K DAG blocks) as bootstrap peer
docker run -d \
  --name q-dagfix-test \
  --network host \
  --memory=4g \
  -e Q_NETWORK_ID=mainnet-genesis \
  -e Q_DB_PATH=/data/db \
  -e Q_P2P_PORT=9009 \
  -e Q_TURBO_SYNC=1 \
  -e Q_BATCHED_WRITES=1 \
  -e Q_PREFLIGHT_CHECK=0 \
  -e Q_TOR_BOOTSTRAP_TIMEOUT=5 \
  -e "Q_BOOTSTRAP_PEERS=/ip4/89.149.241.126/tcp/9001/p2p/12D3KooWFpbXxxZJQ4FX9FGXrE5vaeNTCnZmLn6bqToRCMuiMpxM" \
  -v /path/to/binary:/opt/q-api-server:ro \
  -v /tmp/dagfix-test-data:/data \
  debian:12 \
  bash -c "apt-get update -qq && apt-get install -y -qq libssl3 >/dev/null 2>&1 && \
           cp /opt/q-api-server /usr/local/bin/q-api-server && \
           chmod +x /usr/local/bin/q-api-server && \
           /usr/local/bin/q-api-server --port 8089 2>&1"

# Step 4: Monitor sync — should now find blocks starting at ~100K, not 10M
docker logs -f q-dagfix-test 2>&1 | grep -E "CHECKPOINT SYNC|DIRECT APPLY|height"
```

**Expected result:**
```
BEFORE fix: "Checkpoint height: 14012628" → syncs 1.5M blocks (14M → 15.6M)
AFTER fix:  "Checkpoint height: ~100441"  → syncs ~15.5M blocks (100K → 15.6M)
```

**PASS criteria:**
1. Container finds blocks at height ~100K (not 10M or 14M)
2. Sync downloads blocks from the 2M-10M range (previously invisible)
3. Downloaded blocks deserialize correctly (no corruption)
4. Container reaches chain tip
5. Wallet balances on the container match Epsilon production
6. No errors, no panics, no OOM

**FAIL criteria (stop immediately):**
1. Any panic or crash
2. Block deserialization errors
3. Container height goes backwards
4. Memory exceeds 4GB (possible if DAG blocks are larger than expected)
5. Sync is slower than before the fix (regression)

### Test 3: A/B Comparison

Run TWO Docker containers on Gamma simultaneously:
- Container A: **old binary** (v10.3.5, no fix) on port 8089
- Container B: **new binary** (v10.3.6-dagfix) on port 8090

Both sync from Epsilon. Compare:

| Metric | Container A (old) | Container B (new) | Expected |
|--------|-------------------|-------------------|----------|
| First block height | ~10M or 14M | ~100K | B starts much lower |
| Blocks synced | ~1.5-5.6M | ~15.5M | B syncs 3-10x more blocks |
| Sync speed | Same | Same or slightly slower | DAG scan adds minor overhead |
| Final height | Same (chain tip) | Same (chain tip) | Both reach tip |
| Wallet balances | Same | Same | Must match exactly |
| Memory usage | ~2-3 GB | ~2-4 GB | B may use slightly more |
| Errors | None | None | Neither should error |

### Test 4: Verify Epsilon Is NOT Affected

After deploying the fix to the Docker container, check that Epsilon production (which is NOT running the new code) continues to operate normally:

```bash
# Epsilon should still be running v10.3.3
curl -s http://89.149.241.126:8080/api/v1/status | grep version
# Expected: "10.3.0" or "10.3.3"

# Epsilon should still be at chain tip
curl -s http://89.149.241.126:8080/api/v1/status | grep current_height
# Expected: ~15.6M+

# Epsilon peer count should not have decreased
# (our test container connecting to it should not cause issues)
```

This confirms that the read-path change in the NEW binary does not affect the SERVING behavior of the OLD binary on Epsilon. Epsilon still serves blocks the same way — the fix is on the RECEIVING end (the syncing node).

---

## 5. What We Are NOT Doing

| Action | Why Not |
|--------|---------|
| Writing new keys to any production DB | No DB writes needed — this is a read-only fix |
| Rebuilding or re-indexing anything | The blocks are already in the right place |
| Changing block validation | Blocks are validated identically regardless of key format |
| Changing the P2P protocol | Same block-pack request/response — just smarter DB lookup |
| Deploying to Epsilon or Beta | Docker test on Gamma only until ALL tests pass |
| Running any migration | No migrations — the fix is in the read path |
| Touching the MANIFEST or SST files | Read-only operations only |
| Modifying `save_dag_layer_block()` | Write path is unchanged |

---

## 6. Rollback Plan

If anything goes wrong during Docker testing:

```bash
# Stop the test container
docker stop q-dagfix-test && docker rm q-dagfix-test

# Clean up test data
rm -rf /tmp/dagfix-test-data

# That's it. Nothing was written to production.
# Gamma production continues running v10.3.5 on port 8808 unaffected.
```

---

## 7. Deployment Plan (ONLY After All Tests Pass)

**Phase 1: Docker test on Gamma** (this document)
- Build → test → verify on disposable container
- If FAIL: stop, investigate, fix, re-test
- If PASS: proceed to Phase 2

**Phase 2: Deploy to Gamma production** (separate review required)
- Stop Gamma v10.3.5 gracefully (systemctl stop)
- Swap binary to v10.3.6-dagfix
- Start Gamma
- Monitor for 24 hours
- Verify: serves blocks from 100K onwards to syncing peers

**Phase 3: Deploy to Beta and Epsilon** (separate review required)
- Only after Gamma runs 24+ hours with zero issues
- Standard ha-deploy.sh rolling deployment
- Beta first, then Epsilon

**Each phase requires explicit user approval before proceeding.**

---

## 8. Questions for DeepSeek

### Q1: Is the per-height scan_prefix approach safe for performance?

For each missing height in the range, we call:
```rust
self.hot_db.scan_prefix(CF_BLOCKS, format!("qblock:dag:{}:", height).as_bytes())
```

If a peer requests 200 blocks and none have `qblock:height:` keys, we do 200 prefix scans. Each scan is O(log N) seek + O(1) read (just the first entry). With 67K SST files, the seek involves checking bloom filters on each level.

Is this acceptable for a block-pack response that needs to complete in <10 seconds? Or should we batch the DAG lookups differently?

### Q2: Should the DAG fallback use scan_prefix or scan_prefix_from?

`scan_prefix("qblock:dag:2000941:")` finds entries starting at exactly height 2000941. But what if the peer requested height 2000000 and the first DAG entry is at 2000941? We'd need to do 941 individual prefix scans before finding it.

Alternative: use a broader scan from `"qblock:dag:2"` to find the first available entry in the 2M range, then iterate forward. This is O(1) seek instead of O(941) seeks. But it returns entries we may not need. Trade-off?

### Q3: Is there a correctness risk from serving DAG blocks to a peer expecting height-format blocks?

DAG blocks are stored via `save_dag_layer_block()` which serializes the full `QBlock` struct using bincode + optional LZ4 compression. Height-format blocks are stored via the turbo sync batch writer which also serializes `QBlock`.

Are the serialization formats identical? Could there be version differences (e.g., DAG blocks written by v7.x may use an older QBlock struct layout than height blocks written by v10.x)? The `deserialize_qblock_with_fallback()` function handles legacy formats, but does the block-pack RESPONSE serializer also handle this correctly?

### Q4: Multiple proposers at the same height — which block wins?

`scan_prefix("qblock:dag:3000344:")` may return entries from 2+ proposers:
```
qblock:dag:3000344:e14075fb581971d2 → block from miner A
qblock:dag:3000344:e4034c57046136a4 → block from miner B
```

The current code takes `.first()` (first in byte order). Is this deterministic across nodes? Will every node serve the same block for height 3000344? If not, could this cause consensus divergence where different peers provide different blocks for the same height?

For DAG-Knight consensus, multiple blocks per height is normal. But the sync protocol assumes one block per height (the "canonical" chain). How does this interact?

### Q5: Should we log the DAG fallback activations?

We want to monitor how often the fallback is used in production. Options:
a) Log every DAG fallback hit (noisy — 545K log lines during full sync)
b) Log summary every 1000 blocks ("DAG fallback served 847/1000 blocks in this batch")
c) Counter metric only (no log, just expose via /metrics)

What's the right level for a $1.1B mainnet?

---

*This document covers a read-only code change. No database modifications. No migrations. No rebuilding. The blocks are already there — we're just teaching the code to find them.*
