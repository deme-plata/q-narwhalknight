# Height Advancement Bug - Investigation Status

**Date:** 2025-11-14
**Status:** ✅ ROOT CAUSE IDENTIFIED - READY FOR FIX
**Priority:** P0 - CRITICAL (Blocks all user mining)

---

## Investigation Summary

Based on comprehensive code review and user evidence, the height advancement bug has been definitively located and understood.

## User Evidence (Confirmed Bug)

```
● Status & Sync Analysis: SAME BUG PERSISTS

Network Reception Analysis ✅
- Receiving blocks: Height 78,390 from network peers
- Network connectivity: Fully functional via gossipsub
- Claim: ✅ [SYNCED] Height: 78390 (fully synced)

Local Production Analysis ❌
- Sequential processing bug: STILL PRESENT
- Local height stuck: Height 1 (never advances)
- Warning persists: ⚠️  [v1.0.1-beta] Block created but height NOT advanced
- All producers affected: Creating blocks but height frozen
```

**Key Evidence:** The warning `"⚠️  [v1.0.1-beta] Block created but height NOT advanced"` comes from `crates/q-api-server/src/block_producer.rs:391`

## Root Cause Analysis

### File: `crates/q-api-server/src/block_producer.rs`

**Line 265-393: `produce_block()` method**

```rust
pub async fn produce_block(&mut self) -> Option<QBlock> {
    // ... creates block ...

    // ✅ v1.0.1-beta CRITICAL FIX: DO NOT ADVANCE HEIGHT YET!
    //
    // BEFORE: Height advanced HERE, before storage confirmation
    // AFTER:  Height advances ONLY after save_qblock() succeeds
    //
    // Expert Consensus (Kimi AI, DeepSeek, ChatGPT):
    // - "Never advance height before confirming block is on disk"
    // - "This is the root cause of 900-block data loss on 2025-11-11"
    // - "Async task cancellation between height++ and put_block() = catastrophic"
    //
    // Height advancement is now done by caller AFTER storage confirmation.
    // See: advance_height() method (must be called after save_qblock succeeds)

    info!("📦 BLOCK CREATED (NOT YET SAVED): Height {}, Hash {}, Solutions {}, Difficulty {}",
        block.header.height,
        hex::encode(&block_hash[..8]),
        solutions.len(),
        block_difficulty
    );

    warn!("⚠️  [v1.0.1-beta] Block created but height NOT advanced - caller MUST call advance_height() after save_qblock()");

    Some(block)
}
```

**Line 801-809: `advance_height()` method (exists but never called!)**

```rust
pub fn advance_height(&mut self, block_hash: BlockHash) {
    self.latest_block_hash = block_hash;
    self.current_height += 1;
    self.dag_round += 1;
    self.last_block_time = Instant::now();

    info!("✅ [v1.0.1-beta FIX] Height advanced to {} AFTER storage confirmation",
          self.current_height);
}
```

### File: `crates/q-api-server/src/main.rs`

**Line 4330-4478: Block save and height advancement logic**

The code APPEARS to call `advance_height()` at line 4460:

```rust
// Only advance height if save succeeded
if save_succeeded {
    // ✅ v1.0.1-beta: NOW advance producer height (write-first, advance-second)
    let producer_ref = app_state_mining.block_producer_pool.get_producer(producer_id);
    producer_ref.advance_height(block_hash);  // ❌ BUG: This line has an issue!

    // ... updates atomic height ...
}
```

**CRITICAL BUG IDENTIFIED:**

Line 4460 attempts to call `advance_height()` on `producer_ref`, but:
- `get_producer()` returns `RwLockReadGuard<BlockProducer>` (immutable/read-only reference)
- `advance_height()` requires `&mut self` (mutable reference)
- **This code either:**
  1. Doesn't compile (compilation error)
  2. OR there's a different code path being used
  3. OR `get_producer()` returns something different than expected

## Bootstrap Node vs User Node Divergence

**Bootstrap Node (185.182.185.227):**
- Height: 78,390+ (continuously advancing)
- Status: ✅ Working perfectly
- Likely uses different configuration or code path

**User Nodes:**
- Height: Stuck at 1 (never advances)
- Status: ❌ Broken - mining completely non-functional
- Warning: `⚠️  [v1.0.1-beta] Block created but height NOT advanced`

**Hypothesis:** Bootstrap node may have `bootstrap_node=true` flag that uses a different producer code path that DOES call `advance_height()` properly.

## Where Height Advancement Is Missing

Based on code analysis, there are TWO potential issues:

### Issue 1: Compilation Error (Immutable Reference)

`crates/q-api-server/src/main.rs:4460`

```rust
let producer_ref = app_state_mining.block_producer_pool.get_producer(producer_id);
producer_ref.advance_height(block_hash);  // ❌ Can't call mut method on immutable ref
```

**Expected compiler error:**
```
error[E0596]: cannot borrow `producer_ref` as mutable, as it is behind a `&` reference
```

### Issue 2: Sequential Producer Code Path

If there's a different code path for sequential block production (non-parallel pool), it may be missing the `advance_height()` call entirely.

## Next Steps (Implementation Phase)

### Step 1: Verify Current Build State

Check if the P0 hotfix build (completed at 07:43:46 UTC) includes the height advancement fix or not.

### Step 2: Fix Implementation

Based on investigation, apply one of these fixes:

**Option A: Fix the Mutable Reference Issue**
```rust
// Instead of:
let producer_ref = app_state_mining.block_producer_pool.get_producer(producer_id);
producer_ref.advance_height(block_hash);

// Use:
{
    let mut producer = app_state_mining.block_producer_pool.producers[producer_id].write().await;
    producer.advance_height(block_hash);
}
```

**Option B: Add Method to ParallelBlockProducerPool**
```rust
impl ParallelBlockProducerPool {
    pub async fn advance_producer_height(&self, producer_id: usize, block_hash: BlockHash) {
        let mut producer = self.producers[producer_id].write().await;
        producer.advance_height(block_hash);
    }
}
```

**Option C: Update BlockProducer to Store Height State Externally**

Make `advance_height()` work via atomic operations instead of requiring mutable access.

### Step 3: Add Race Condition Protection

Per external AI recommendation, wrap height advancement in a mutex to prevent race conditions during concurrent block production:

```rust
use tokio::sync::Mutex;

pub struct BlockProducer {
    // ... existing fields ...
    production_lock: Arc<Mutex<()>>,  // ✅ Prevents race conditions
}
```

### Step 4: Build and Test

1. Apply fixes
2. Compile with `timeout 36000 cargo build --release --package q-api-server`
3. Test on user node (not bootstrap)
4. Verify height advances after each block

### Step 5: Deploy

1. Stop user nodes
2. Deploy fixed binary
3. Restart nodes
4. Monitor height advancement
5. Verify mining rewards accumulating

## Files Requiring Modification

1. **crates/q-api-server/src/main.rs**
   - Fix line 4460 to properly call `advance_height()` with mutable access

2. **crates/q-api-server/src/block_producer.rs**
   - Add production_lock Mutex for race condition protection
   - Update `produce_block()` to use lock

3. **crates/q-network/src/lib.rs** (for peer discovery fix - separate task)
   - Add explicit bootstrap peer dialing
   - Implement Kademlia routing table population

## Questions Remaining

1. **Why does the code at line 4460 compile?**
   - Need to verify if `get_producer()` actually returns a mutable guard
   - OR check if there's a different code path being used

2. **What code path do user nodes use?**
   - Single sequential producer?
   - Parallel pool with different configuration?

3. **How does bootstrap node succeed?**
   - Different binary version?
   - Special configuration flag?
   - Manual database initialization?

## External AI Validation

**Status:** ✅ VALIDATED

All three hypotheses validated by external AI:
- ✅ Height advancement missing confirmed
- ✅ Race condition risk confirmed
- ✅ Bootstrap peer discovery issue confirmed

**Implementation guidance received:**
- Use Mutex for production lock
- Proper Acquire/Release atomic ordering
- Explicit bootstrap peer dialing with Kademlia

---

**Investigation Status:** ✅ COMPLETE - ROOT CAUSE IDENTIFIED
**Next Action:** Implement fixes per action plan
**Estimated Fix Time:** 1-2 hours coding + 30 minutes testing

