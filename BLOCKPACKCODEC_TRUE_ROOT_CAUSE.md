# BlockPackCodec TRUE Root Cause - Event Loop Not Processing

**Date**: 2025-11-09 13:05 CET
**Status**: 🔍 **INVESTIGATION UPDATED**

---

## Previous Hypothesis: WRONG

**Previous theory**: Network ID mismatch (Phase5 vs Phase6)
**Reality**: Both nodes use **TestnetPhase6** for P2P (hardcoded at line 4963)

**The API showing "testnet-phase5" is just a display bug** - doesn't affect P2P.

---

## TRUE Root Cause

### Server Beta Event Loop Not Processing Block-Pack Requests

**Evidence**:
1. ✅ Server Alpha SENDS requests (logs confirm)
2. ❌ Server Beta has ZERO [BLOCK-PACK] logs (requests never reach handler)
3. ✅ Both use TestnetPhase6 for P2P (line 4963 in main.rs)
4. ❌ BlockPackCodec events not being processed on Server Beta

---

## Where The Problem Is

**File**: `crates/q-network/src/unified_network_manager.rs`

The event handler at lines 1160-1230 should process:
- `Message::Request` → Server responds with blocks
- `Message::Response` → Client receives blocks
- `Event::OutboundFailure` → Timeout handling

**But Server Beta shows NO logs from any of these handlers!**

This means **one of these is true**:

### Option 1: Poll Loop Not Running

The swarm event loop that calls `swarm.poll()` may not be running on Server Beta.

**Check**: Search for where `poll()` or `poll_next()` is called on the swarm.

### Option 2: BlockPackCodec Not Added to Swarm

The `block_sync` behaviour may not be registered in the swarm behaviour.

**Check**: Look at swarm initialization to see if BlockPackCodec is added.

### Option 3: Events Being Dropped

The poll loop might be running but not processing RequestResponse events.

**Check**: Look for event matching in the poll loop.

---

## Next Investigation Steps

### Step 1: Find The Poll Loop

```bash
grep -rn "swarm.poll\|poll_next\|StreamExt" crates/q-network/src/unified_network_manager.rs
```

### Step 2: Check Swarm Behaviour Registration

```bash
grep -rn "NetworkBehaviour\|block_sync.*RequestResponse" crates/q-network/src/
```

### Step 3: Add Debug Logging

If poll loop exists, add logging to see ALL events:

```rust
// In poll loop
debug!("🔍 [SWARM EVENT] {:?}", event);
```

This will show us if BlockPackCodec events are arriving but not being handled.

---

## Confirmed Facts

✅ **Both nodes use TestnetPhase6** (line 4963 in main.rs is hardcoded)
✅ **Requests are sent** from Server Alpha
✅ **Response handler code exists** (lines 1192-1217)
✅ **Request handler code exists** (lines 1160-1190)
❌ **No logs from any handler** on Server Beta
❌ **Responses never arrive** at Server Alpha

**Conclusion**: The event loop on Server Beta is either not running or not dispatching events to the BlockPackCodec handler.

---

**Status**: Need to find and verify the swarm poll loop

**Next**: Locate where `swarm.poll()` is called and verify it's running on Server Beta

---

*Analysis updated: 2025-11-09 13:05 CET*
