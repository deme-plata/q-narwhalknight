# Q-NarwhalKnight Fresh-Sync Bug — Technical Review and Fix Plan
**Date**: 2026-04-28  
**Author**: Server Beta (Claude Code)  
**Version target**: v10.5.0  
**Status**: Pending implementation  

---

## Executive Summary

New nodes (fresh DB, height=0) cannot sync all available blocks despite the v10.3.8 DAG-key fix being in the codebase. After the checkpoint replay at block 16,538,868, a fresh node is left with persistent gaps from blocks ~1–100,200 (permanently lost) and a height pointer stuck around 7,200 rather than advancing to the earliest available block (~100,201 on Epsilon).

This review documents five root causes with exact code locations, designs fixes for each, and defines the implementation order and tests.

---

## Background: What the v10.3.8 Fix Accomplished

In April 2026, two technical reviews (`technical-review-epsilon-block-gap-forensics-v1.md`, `technical-review-dag-key-discovery-sync-fix-v1.md`) identified that 545,710 blocks on Epsilon were stored in `qblock:dag:{height}:{proposer}` RocksDB key format, not the turbo-sync `qblock:height:{N}` format. RocksDB string-sort ordering made these blocks invisible to the range scanner (bloom filter false negatives).

Fix applied in `crates/q-storage/src/lib.rs` at `get_qblocks_range()`: added `scan_prefix_seek` fallback that reads `qblock:dag:` keys. This fix IS committed and present.

**However**, the v10.3.8 fix only solved block discovery. It did not solve the height pointer advancement problem that prevents a fresh node from ever requesting the DAG-format blocks in the first place.

---

## Network Data Availability — Ground Truth

| Height range | Status | Location |
|---|---|---|
| 1 – 100,200 | **Permanently lost** | No node has these blocks |
| 100,201 – ~9,000,000 | Sparse DAG format (~545,710 blocks) | Epsilon only (`qblock:dag:` keys) |
| ~9,000,000 – 15,600,000 | Dense turbo-sync format | Beta + Epsilon (`qblock:height:` keys) |
| 15,600,001 – tip | Live production blocks | All peers |

Epsilon's earliest block: `qblock:dag:100441:{proposer_hash}` (verified via RocksDB scan, April 2026).

The `probe_network_gap_blocking` binary search in `turbo_sync.rs:3244–3295` finds `first_found=100,000` (because the 1000-height window `from_height=100000` returns 472 blocks from Epsilon). After ±5,000-precision binary search between `last_empty=50,000` and `first_found=100,000`, the probe floor is approximately **75,000–100,000**.

This means a correctly functioning sync would:
1. Detect fresh start (height < 100)
2. Probe → floor ≈ 75,000
3. Load checkpoint at 16,538,868 (skipping unrecoverable blocks)
4. Replay whatever blocks exist above the checkpoint floor
5. Record the 1–75,000 range as confirmed-empty

Instead, the height pointer gets stuck at ~7,200 and never advances past the gap.

---

## Root Cause 1 (RC-1): Concurrent Invocation Race

### Location
`crates/q-storage/src/turbo_sync.rs` — `sync_to_height()` entry point

### What happens
Every incoming gossipsub `peer_height` event independently calls `sync_to_height()`. At startup, a fresh node receives peer announcements from Beta (~16.5M) and Epsilon (~16.4M) within milliseconds of each other, spawning two concurrent invocations:

- **Invocation A**: `local_height=0`, detects fresh start, sets `effective_start_height=0`, calls `probe_network_gap` (takes ~15s for 15 HTTP probes), eventually discovers floor ≈ 75,000
- **Invocation B**: `local_height=0` (read before A writes anything), also detects fresh start, sets `effective_start_height=0`, also calls `probe_network_gap` concurrently

Both invocations independently compute `effective_start_height` and proceed to request blocks from the same height ranges simultaneously. The shared `qblock:latest` RocksDB key gets written by both invocations in a race. Whichever invocation finishes last wins, potentially leaving the pointer at 7,200 (the contiguous height after applying a small batch) while the other invocation believes it advanced to 75,000+.

### Fix: Single-flight gate

Add three fields to `TurboSyncManager` (or equivalent state struct):

```rust
// In TurboSyncManager struct
fresh_sync_gate: Arc<tokio::sync::Mutex<()>>,   // One fresh-start sync at a time
fresh_sync_target: Arc<AtomicU64>,               // Latch: latest requested target
fresh_sync_done: Arc<AtomicBool>,                // Set when first full sync completes
```

Modify `sync_to_height()` preamble (before effective_start_height calculation):

```rust
pub async fn sync_to_height(&self, target_height: u64) -> anyhow::Result<()> {
    let local_height = self.storage.get_latest_qblock_height().await?.unwrap_or(0);
    
    // Fresh start gate: only one invocation runs probe + bootstrap sync
    if local_height < 100 && !self.fresh_sync_done.load(Ordering::Acquire) {
        // Latch the highest target seen
        self.fresh_sync_target.fetch_max(target_height, Ordering::AcqRel);
        
        // Try to become the single fresh-sync runner
        let _guard = match self.fresh_sync_gate.try_lock() {
            Ok(g) => g,
            Err(_) => {
                // Another invocation is running the probe — latch our target and exit
                info!("🔒 [FRESH-SYNC GATE] Deferred: invocation already in progress (target={})", target_height);
                return Ok(());
            }
        };
        
        // Use the highest target seen so far (latched by any concurrent caller)
        let target_height = self.fresh_sync_target.load(Ordering::Acquire);
        // ... continue with probe + sync using latched target_height
    }
    // ... rest of sync_to_height
}
```

After the fresh sync completes (pointer advances past 100), set `fresh_sync_done.store(true, Ordering::Release)` so subsequent calls skip the gate entirely.

---

## Root Cause 2 (RC-2): Range-Unaware GAP SKIP

### Location
`crates/q-storage/src/turbo_sync.rs:4862–4873` — `direct_apply_blocks_v2()`

### What happens

The v6.1.0 GAP SKIP logic:

```rust
// turbo_sync.rs lines ~4862-4873
} else if block.header.height > highest_contiguous + 1 {
    let gap_size = block.header.height - (highest_contiguous + 1);
    if blocks_forward == 0 && gap_size > 0 {
        warn!(
            "🚨 [v6.1.0 GAP SKIP] Gap at START! Missing blocks {}-{}",
            highest_contiguous + 1, block.header.height - 1
        );
        highest_contiguous = block.header.height;  // ← UNCONDITIONAL ADVANCE
        blocks_forward += 1;
    } else {
        break;
    }
}
```

The GAP SKIP fires when `blocks_forward == 0 && gap_size > 0`. This means: "if the first block in this response has a gap from our current pointer, jump the pointer forward to wherever the peer put its first block."

**The critical flaw**: `direct_apply_blocks_v2` is called per-batch without knowing what height range was requested. When Invocation A requests blocks 1–1000 and Epsilon returns blocks starting at height 16,429,228 (the peer's own tip blocks from gossipsub), the GAP SKIP jumps `highest_contiguous` from 0 to 16,429,228 — a 16-million height corruption.

This also fires legitimately in the DAG-block range: requesting heights 7,000–8,000, getting a response starting at height 7,201 (first DAG block in range), results in GAP SKIP advancing pointer from 7,200 to 7,201 correctly. But requesting heights 1–1000 and getting a response starting at 100,441 results in GAP SKIP advancing pointer to 100,441, skipping the empty range 1–100,440 without recording that those heights are unavailable.

### Fix: Range-bounded GAP SKIP

`direct_apply_blocks_v2` needs to know the requested range. Add `range_start` and `range_end` parameters:

```rust
async fn direct_apply_blocks_v2(
    &self,
    blocks: Vec<QBlock>,
    range_start: u64,   // NEW: what height we asked for
    range_end: u64,     // NEW: upper bound of request
) -> anyhow::Result<u64> {
```

Modify the GAP SKIP condition:

```rust
} else if block.header.height > highest_contiguous + 1 {
    let gap_size = block.header.height - (highest_contiguous + 1);
    
    // Out-of-range block: store it but do NOT advance the contiguous pointer
    if block.header.height < range_start || block.header.height > range_end {
        warn!(
            "⚠️ [RANGE FILTER] Block h={} outside requested range [{},{}] — storing, not advancing pointer",
            block.header.height, range_start, range_end
        );
        self.storage.put_block_raw(&block).await?;
        continue;
    }
    
    // In-range gap: only skip if gap is within configured tolerance
    let max_skip = self.config.max_intra_range_gap_skip.unwrap_or(5_000);
    let safe_to_skip = blocks_forward == 0 
        && gap_size > 0 
        && gap_size <= max_skip
        && block.header.height <= range_end;
    
    if safe_to_skip {
        warn!(
            "🔍 [GAP SKIP] In-range gap {}-{} (size {}), advancing pointer",
            highest_contiguous + 1, block.header.height - 1, gap_size
        );
        highest_contiguous = block.header.height;
        blocks_forward += 1;
    } else if gap_size > max_skip {
        warn!(
            "🚫 [GAP SKIP REFUSED] Gap {} too large (max {}), stopping batch",
            gap_size, max_skip
        );
        break;
    } else {
        break;
    }
}
```

Add `max_intra_range_gap_skip` to `TurboSyncConfig` (default: 5,000 heights).

---

## Root Cause 3 (RC-3): Probe Window Too Narrow for Sparse DAG Blocks

### Location
`crates/q-storage/src/turbo_sync.rs:3244` — `probe_network_gap_blocking()`

### What happens

The probe uses exponential checkpoints: `[1_000, 10_000, 50_000, 100_000, ...]`. At each checkpoint `h`, it queries:
```
GET /api/v1/sync/blocks?from_height={h}&limit=1000
```

The probe at h=50,000 returns 0 blocks (no blocks in range 50,000–51,000 on Epsilon). The probe at h=100,000 returns 472 blocks. Binary search produces floor ≈ 75,000.

**The problem**: The binary search returns a floor of 75,000 but Epsilon's actual earliest block is at 100,441. The sync code sets `effective_start_height = 75,000` and begins requesting heights 75,000–76,000, 76,000–77,000, etc. All return 0 blocks. After ~25 empty responses, the code concludes "network has no blocks in this range" and stalls.

The probe needs to also classify density: is the found range **sparse** (a few blocks per 500 heights) or **dense** (consecutive blocks)? Sparse ranges must be handled differently — the sync should not request 1000-height windows sequentially but should instead advance to the first available block.

### Fix: Density-aware probe result + forward-seek endpoint

#### Part A — Probe returns `ChainTopology` struct

Replace `probe_network_gap` returning `u64` with returning `ChainTopology`:

```rust
pub struct ChainTopology {
    pub confirmed_empty_floor: u64,  // No block exists on any peer before this height
    pub first_available: u64,        // Earliest confirmed block height
    pub dense_floor: u64,            // Height where blocks become dense (>0.5 block/height)
    pub is_sparse_region: bool,      // True if first_available..dense_floor is sparse
}
```

The probe at h=100,000 (count=472 in 1000-height window) → density = 0.472 blocks/height → **sparse**.  
A probe at h=14,000,000 (count=~950 in 1000-height window) → density = 0.95 → **dense**.

Binary search for `dense_floor` similarly to current `first_found` binary search.

#### Part B — New `/api/v1/sync/forward` endpoint on Bootstrap Peers

The existing `/api/v1/sync/blocks?from_height=X&limit=N` endpoint returns blocks in range `[X, X+N]`. For sparse DAG blocks, we need: "give me the next N blocks after height X regardless of gaps."

New endpoint (`crates/q-api-server/src/handlers.rs`):

```rust
// GET /api/v1/sync/forward?after_height={H}&limit={N}
// Returns next N blocks after height H, skipping empty heights
// Response: {"blocks": [...], "next_cursor": {height_of_last_block}, "total_available": {count}}
async fn sync_forward(
    State(state): State<AppState>,
    Query(params): Query<SyncForwardParams>,
) -> impl IntoResponse {
    let after = params.after_height.unwrap_or(0);
    let limit = params.limit.unwrap_or(500).min(1000);
    
    // Scan forward from `after` height, collecting limit blocks regardless of gaps
    let blocks = state.storage.get_qblocks_forward(after, limit).await?;
    // ...
}
```

Storage method `get_qblocks_forward(after: u64, limit: usize)` scans RocksDB with both key formats (dense `qblock:height:*` and sparse `qblock:dag:*`) and returns the first `limit` blocks found after `after` height, regardless of height gaps.

The fresh-sync path uses `/api/v1/sync/forward` exclusively when `is_sparse_region == true`, advancing the cursor block-by-block through the DAG range rather than window-by-window.

---

## Root Cause 4 (RC-4): No Confirmed-Empty Range Mechanism

### Location
`crates/q-storage/src/lib.rs` — RocksDB schema / height pointer advancement

### What happens

The height pointer (`qblock:latest`) represents the highest **contiguous** block height. It can never advance past a gap without a block filling that gap. Heights 1–100,200 are permanently lost — no peer has them, and they will never be filled.

Currently there is no way to tell the storage engine "heights 1–100,200 do not exist and will never exist — advance the pointer to 100,201 anyway." Without this, the pointer is permanently stuck at 0 on a fresh node that doesn't have the checkpoint replay applied.

Even with the checkpoint at 16,538,868 applied, the height pointer after checkpoint replay is 0 (the checkpoint sets balances but not the block height pointer). The missing-block warning added in `cc611be0` fires but there is no action taken.

### Fix: `chain_voids` RocksDB column family

Add a new CF `cf_chain_voids` that stores `VoidRecord` entries:

```rust
// crates/q-storage/src/lib.rs — new types
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VoidRecord {
    pub start: u64,           // First height in void (inclusive)
    pub end: u64,             // Last height in void (inclusive)
    pub kind: VoidKind,
    pub registered_at_height: u64,  // Current tip when registered
    pub witness_count: u8,    // How many peers confirmed this range is empty
    pub source: VoidSource,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum VoidKind {
    ConfirmedEmpty,   // Probed all known peers, none have blocks here
    ConfirmedSparse,  // Probed all known peers, <5% density
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum VoidSource {
    Probe,       // Discovered by probe_network_gap
    AutoRepair,  // Registered after N failed fetch attempts
    Manual,      // Registered by operator
}
```

Key: `chain_void:{start:020}` (lexicographic order for range scans).

New storage methods:

```rust
impl StorageEngine {
    /// Register a void range (heights permanently/likely unavailable)
    pub async fn register_chain_void(&self, record: VoidRecord) -> anyhow::Result<()>;
    
    /// Get all voids covering [from, to]
    pub async fn get_chain_voids(&self, from: u64, to: u64) -> Vec<VoidRecord>;
    
    /// Advance the height pointer past a void if it's registered
    /// Returns the new contiguous height after advancement, or current if not advanced
    pub async fn advance_past_void(&self, current: u64) -> anyhow::Result<u64> {
        let voids = self.get_chain_voids(current + 1, current + 200_000).await;
        for void in &voids {
            if void.start == current + 1 {
                // This void starts right after our current height — safe to advance to void.end + 1
                info!("⏭️ [CHAIN VOID] Advancing pointer past known-empty range {}-{}", 
                      void.start, void.end);
                self.set_contiguous_height(void.end).await?;
                return Ok(void.end);
            }
        }
        Ok(current)
    }
    
    /// Check if advancing from `from` to `to` is safe (all gaps are registered voids)
    pub async fn is_advance_safe(&self, from: u64, to: u64) -> bool;
}
```

**Integration into fresh-start sync**:

After `probe_network_gap` returns `ChainTopology { confirmed_empty_floor: 0, first_available: 100_441, .. }`:

```rust
// Register the confirmed-empty range before loading checkpoint
if topology.first_available > 0 {
    self.storage.register_chain_void(VoidRecord {
        start: 1,
        end: topology.first_available - 1,
        kind: VoidKind::ConfirmedEmpty,
        registered_at_height: 0,
        witness_count: 2,  // Both Beta and Epsilon confirmed empty
        source: VoidSource::Probe,
    }).await?;
    
    // Now advance the pointer past the void
    self.storage.advance_past_void(0).await?;
    // Pointer is now at first_available - 1 = 100,440
}
```

**Integration into checkpoint replay warning** (`lib.rs` — the missing-block warning from cc611be0):

```rust
// After checkpoint replay, register any remaining voids
let missing_pct = missing_blocks as f64 / expected_count as f64;
if missing_pct >= 0.05 {
    error!("⚠️ POST-CHECKPOINT: {}/{} blocks missing ({:.1}%)", 
           missing_blocks, expected_count, missing_pct * 100.0);
    // Auto-register voids for contiguous missing ranges
    for gap in &detected_gaps {
        self.register_chain_void(VoidRecord {
            start: gap.start, end: gap.end,
            kind: VoidKind::ConfirmedEmpty,
            source: VoidSource::AutoRepair,
            ..
        }).await?;
    }
}
```

---

## Root Cause 5 (RC-5): Auto-Repair Targets Wrong Height

### Location
`crates/q-storage/src/turbo_sync.rs` — auto-repair / `get_first_missing_height` integration

### What happens

After sync stalls, the auto-repair logic calls `get_first_missing_height()` which scans from the contiguous height pointer upward looking for the first gap. In a fresh-sync scenario where the pointer is at 7,200 and the next available block is at 100,441, `get_first_missing_height` returns 7,201 (the first height that isn't filled). Auto-repair then requests blocks 7,201–8,201 from peers. Peers return 0 blocks. Repair marks this range "attempted" and moves on to the next window.

This produces an infinite loop: repair attempts every 1000-height window from 7,201 to 100,441 — that's ~93 iterations — each taking 8 seconds of probe timeout, for a total of ~12 minutes of wasted time before the repair ever reaches a height that has blocks.

Worse: the repair invocations also race with the normal sync invocations (RC-1), each writing conflicting height pointers.

### Fix: Gap-floor targeting

`get_first_missing_height()` should return not just the first missing height, but also whether that gap has a registered void:

```rust
pub async fn get_first_missing_height(&self, from: u64) -> (u64, bool) {
    // Returns (missing_height, is_registered_void)
    let missing = /* existing scan logic */;
    let is_void = self.storage.get_chain_voids(missing, missing + 1).await.len() > 0;
    (missing, is_void)
}
```

If `is_registered_void == true`, auto-repair should skip to the void's end + 1 instead of iterating through the void range:

```rust
let (missing_height, is_void) = self.get_first_missing_height(current).await;
if is_void {
    let voids = self.storage.get_chain_voids(missing_height, missing_height + 1).await;
    if let Some(void) = voids.first() {
        info!("⏭️ [AUTO-REPAIR] Gap {} is registered void, skipping to {}", 
              missing_height, void.end + 1);
        self.storage.set_contiguous_height(void.end).await?;
        continue;  // Retry with new pointer
    }
}
// Only attempt fetch if gap is NOT a registered void
self.fetch_and_apply_range(missing_height, missing_height + 1000).await?;
```

---

## Implementation Order

These fixes have dependencies. Implement in this order:

### Phase 1: RC-4 — Chain Voids CF (Foundation)
**Estimated effort**: 2–3 days  
**Why first**: All other fixes need somewhere to record and query void ranges. RC-3 needs to register the probe result. RC-5 needs to query voids before spending time on fetch attempts.

Steps:
1. Add `CF_CHAIN_VOIDS = "cf_chain_voids"` constant to `crates/q-storage/src/lib.rs`
2. Register CF in `StorageEngine::open()` 
3. Implement `VoidRecord`, `VoidKind`, `VoidSource` types
4. Implement `register_chain_void`, `get_chain_voids`, `advance_past_void`, `is_advance_safe`
5. Write unit tests (pure RocksDB, no network)

### Phase 2: RC-3 — Density-Aware Probe (Topology)
**Estimated effort**: 2 days  
**Why second**: Produces the `ChainTopology` result that RC-1 and RC-4 use.

Steps:
1. Define `ChainTopology` struct
2. Refactor `probe_network_gap` to return `ChainTopology`
3. Add density classification to probe result
4. Implement `/api/v1/sync/forward` endpoint in `handlers.rs`
5. Implement `get_qblocks_forward()` in `lib.rs` (scans both key formats)
6. Deploy endpoint update to Beta and Epsilon before testing fresh sync

### Phase 3: RC-1 — Single-Flight Gate
**Estimated effort**: 1 day  
**Why third**: Needs `ChainTopology` probe result to pass to void registration.

Steps:
1. Add gate fields to `TurboSyncManager`
2. Wrap fresh-start path with `try_lock()` gate
3. Register probe result as `VoidRecord` (RC-4) after acquiring gate
4. Call `advance_past_void` after registration

### Phase 4: RC-2 — Range-Bounded GAP SKIP
**Estimated effort**: 1.5 days  
**Why fourth**: Needs `range_start`/`range_end` plumbed through all `direct_apply_blocks_v2` callers; simpler once other fixes are in.

Steps:
1. Add `range_start`, `range_end` to `direct_apply_blocks_v2` signature
2. Update all callers to pass actual request range
3. Replace unconditional GAP SKIP with range-bounded version
4. Add `max_intra_range_gap_skip` config (default 5,000)

### Phase 5: RC-5 — Auto-Repair Gap-Floor Targeting
**Estimated effort**: 0.5 days  
**Why last**: Simplest change; RC-4 voids must exist before this can skip them.

Steps:
1. Modify `get_first_missing_height` to return `(u64, bool)`
2. Add void-aware skip in the repair loop
3. Write test that verifies repair skips a registered void

---

## Test Plan

### Unit Tests (no network, pure in-process)

```rust
// crates/q-storage/tests/chain_voids_tests.rs
#[test]
fn test_register_and_query_void() { /* register [1, 100200], query it back */ }

#[test]
fn test_advance_past_void() { /* set pointer=0, register void [1, 100200], advance → 100200 */ }

#[test]
fn test_no_advance_without_void() { /* gap without void: pointer stays at 0 */ }

#[test]
fn test_range_bounded_gap_skip_in_range() { /* gap within requested range, size < 5000: advance */ }

#[test]
fn test_range_bounded_gap_skip_out_of_range() { /* block outside requested range: store, no advance */ }

#[test]
fn test_range_bounded_gap_skip_too_large() { /* gap > 5000: stop, no advance */ }

#[test]
fn test_auto_repair_skips_void() { /* repair loop with registered void: skips to void.end+1 */ }

#[test]
fn test_single_flight_gate() { /* two concurrent sync_to_height calls: second defers immediately */ }
```

### Integration Test (fresh-sync simulation)

```rust
// crates/q-storage/tests/fresh_sync_e2e_test.rs
#[tokio::test]
async fn test_fresh_sync_registers_void_and_advances() {
    // 1. Create empty StorageEngine
    // 2. Call probe_network_gap (mock: returns first_available=100_441, confirmed_empty=[1..100_440])
    // 3. Verify void [1..100_440] registered in cf_chain_voids
    // 4. Verify contiguous pointer advanced to 100_440
    // 5. Apply 10 blocks starting at 100_441
    // 6. Verify contiguous pointer = 100_450
}
```

### Docker Sync Test (end-to-end, real network)

Run a fresh container against Epsilon after deploying `/api/v1/sync/forward`:

```bash
ssh root@5.79.79.158 "docker run -d \
  --name q-fresh-v10500 \
  -e Q_NETWORK_ID=mainnet-genesis \
  -e Q_DB_PATH=/data/db \
  -e RUST_LOG=info \
  debian:12 bash -c '...' "

# After 10 minutes, check:
ssh root@5.79.79.158 "docker logs q-fresh-v10500 2>&1 | grep -E 'CHAIN VOID|GAP SKIP|FRESH SYNC|contiguous height|INTEGRITY' | head -30"
```

**Success criteria**:
- `[CHAIN VOID] Advancing pointer past known-empty range 1–100440` visible in logs within first 60 seconds
- Pointer reaches 100,441+ within 5 minutes
- Sync continues through DAG range (100K–9M) without stalling
- No `[v6.1.0 GAP SKIP] Gap at START! Missing blocks 1-16429227` log lines
- Missing-block warning from cc611be0 shows ≤1% gap after checkpoint replay

---

## Files Changed

| File | Section | Change |
|---|---|---|
| `crates/q-storage/src/lib.rs` | CF registration (~line 150) | Add `CF_CHAIN_VOIDS` |
| `crates/q-storage/src/lib.rs` | `StorageEngine::open()` | Register new CF |
| `crates/q-storage/src/lib.rs` | New `VoidRecord` types | Add types + impl |
| `crates/q-storage/src/lib.rs` | `StorageEngine` impl | Add 4 void methods |
| `crates/q-storage/src/lib.rs` | `get_qblocks_range()` | Add forward-seek variant |
| `crates/q-storage/src/turbo_sync.rs` | `TurboSyncManager` struct | Add gate fields |
| `crates/q-storage/src/turbo_sync.rs` | `sync_to_height()` | Add fresh-sync gate |
| `crates/q-storage/src/turbo_sync.rs` | `probe_network_gap` | Return `ChainTopology` |
| `crates/q-storage/src/turbo_sync.rs` | `direct_apply_blocks_v2` | Add range params + bounded skip |
| `crates/q-storage/src/turbo_sync.rs` | auto-repair loop | Void-aware skip |
| `crates/q-storage/src/turbo_sync.rs` | `get_first_missing_height` | Return `(u64, bool)` |
| `crates/q-api-server/src/handlers.rs` | new route | `/api/v1/sync/forward` handler |
| `crates/q-api-server/src/main.rs` | router | Register new route |
| `crates/q-storage/tests/` | new file | `chain_voids_tests.rs` |
| `crates/q-storage/tests/` | new file | `fresh_sync_e2e_test.rs` |

---

## What NOT to Change

- **`/api/v1/sync/blocks` endpoint** — correct as-is; returns 0 blocks for genuinely empty windows
- **`scan_prefix_seek` DAG-key fallback** — already fixed in v10.3.8; do not regress
- **Checkpoint data** — correct; leave it as hardcoded at 16,538,868
- **Probe exponential heights** — the existing list is fine; only the return type changes

---

## Risk Assessment

| RC | Risk | Mitigation |
|---|---|---|
| RC-1 (gate) | Gate deadlock if single invocation panics | Gate uses `tokio::Mutex`, dropped on panic via `_guard` RAII |
| RC-2 (range skip) | Out-of-range blocks silently dropped | Log warning + store in RocksDB before `continue`; blocks not lost |
| RC-3 (forward endpoint) | Endpoint not deployed before fresh-sync test | Gate fresh-sync to wait for endpoint availability check |
| RC-4 (chain_voids) | CF migration fails on existing nodes | CF only created if absent; existing nodes get empty CF, no data loss |
| RC-5 (repair target) | Repair skips legitimate gaps | Only skips if `is_registered_void == true`; manual registration gated by witness count ≥ 2 |

---

## Version and Deployment

1. Bump `Cargo.toml` workspace version to `10.5.0`
2. Build: `cargo build --release --package q-api-server`
3. **Deploy `/api/v1/sync/forward` endpoint to Beta and Epsilon FIRST** (before testing fresh sync)
4. Run full test suite: `cargo test --workspace`
5. Deploy via `./scripts/ha-deploy.sh full -y`
6. Start fresh Docker sync test on Delta
7. Monitor for `[CHAIN VOID]` log lines confirming void registration and pointer advancement
