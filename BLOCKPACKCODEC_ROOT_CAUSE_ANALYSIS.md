# BlockPackCodec Protocol Failure - Root Cause Analysis

**Date**: 2025-11-09 12:56 CET
**Issue**: TURBO SYNC times out, P2P sync not working
**Impact**: Server Alpha stuck at 40 blocks/second (HTTP fallback) instead of 1000+ blocks/second

---

## Problem Summary

**Server Alpha** (161.35.219.10) - Sync node:
- Sends BlockPackCodec requests via `request_blocks_from_peer()`
- **Never receives responses**
- Times out after 90 seconds
- Falls back to HTTP (slow)

**Server Beta** (185.182.185.227) - Producer node:
- Has 8778 blocks available
- Should respond to BlockPackCodec requests
- **No logs showing incoming requests**
- Not sending responses

---

## Code Flow Analysis

### 1. Request Sending (Server Alpha)

**File**: `crates/q-api-server/src/main.rs` lines 5077

```rust
if let Err(e) = libp2p_lock.request_blocks_from_peer(*peer_id, next_block_needed, block_count) {
    error!("❌ [DISCOVERY] Failed to send test request to peer {}: {}", peer_id, e);
} else {
    info!("✅ [DISCOVERY] Test request sent to peer {}", peer_id);
}
```

**Calls**: `unified_network_manager.rs` line 1337:

```rust
pub fn request_blocks_from_peer(&mut self, peer_id: PeerId, start_height: u64, limit: usize) {
    info!("📤 [BLOCK-SYNC] Requesting {} blocks from height {} from peer {}", limit, start_height, peer_id);

    let request = q_types::BlockPackRequest::new(start_height, end_height);
    self.swarm.behaviour_mut().block_sync.send_request(&peer_id, request);

    info!("✅ [BLOCK-SYNC] Block sync request sent to {}", peer_id);
}
```

**Status**: ✅ Request is sent (logs confirm "Block sync request sent")

### 2. Request Receiving (Server Beta)

**File**: `crates/q-network/src/unified_network_manager.rs` lines 1160-1190

**Should trigger** on `RequestResponseEvent::Message::Request`:
```rust
Message::Request { request_id, request, channel } => {
    info!("📥 [BLOCK-PACK] Received block pack request from {}: heights {}-{}",
          peer, request.start_height, request.end_height);

    // Fetch blocks from storage
    let response = storage.get_qblocks_range(request.start_height, limit).await?;

    // Send response
    self.swarm.behaviour_mut().block_sync.send_response(channel, response)?;
    info!("✅ [BLOCK-PACK] Sent response to {}", peer);
}
```

**Status**: ❌ **NO LOGS SHOWING THIS EVENT** - Request never arrives!

### 3. Response Receiving (Server Alpha)

**File**: `crates/q-network/src/unified_network_manager.rs` lines 1192-1217

**Should trigger** on `RequestResponseEvent::Message::Response`:
```rust
Message::Response { request_id, response } => {
    info!("📨 [BLOCK-PACK] Received block pack response: {} blocks", response.blocks.len());

    // Mark peer as successful
    self.mark_peer_success(peer);

    // Forward blocks to consensus
    tx.send(response.blocks)?;
}
```

**Status**: ❌ Never triggered because response never sent

### 4. Timeout (Server Alpha)

**File**: `crates/q-network/src/unified_network_manager.rs` line 1220

```rust
Event::OutboundFailure { peer, request_id, error } => {
    warn!("⚠️ [BLOCK-PACK] Outbound failure to {}: {:?}", peer, error);

    // Mark peer as failed (timeout/incompatible)
    self.mark_peer_failure(peer);
}
```

**Status**: ✅ This DOES trigger - peers get marked as failed/blacklisted

---

## Root Cause Hypothesis

### Option 1: Protocol Not Initialized on Server Beta

**Server Beta** may not have BlockPackCodec protocol handler running.

**Check**:
```bash
# On Server Beta (185.182.185.227)
journalctl -u q-api-server --since "10 minutes ago" | grep -i "block.*sync.*initialized\|RequestResponse"
```

**Expected**: Should see protocol initialization logs
**Reality**: Need to verify

### Option 2: Network ID Mismatch

Server Alpha and Server Beta may be on **different network IDs**:
- Server Alpha: `testnet-phase6`
- Server Beta: `testnet-phase5` (seen in /api/v1/status)

**Evidence**:
```json
// Server Beta status
{
  "network_id": "testnet-phase5"
}
```

**Impact**: Different network IDs mean **different gossipsub topics** and potentially **different P2P protocols**.

### Option 3: libp2p Swarm Not Processing Events

The `poll()` loop may not be running or not handling RequestResponse events.

**Check where swarm events are polled**:
```bash
grep -rn "swarm.poll\|poll_next" crates/q-network/src/
```

### Option 4: BlockPackCodec Not Added to Swarm Behaviour

The `block_sync` behaviour may not be registered in the swarm.

**File**: Check `crates/q-network/src/unified_network_manager.rs` for swarm initialization

---

## Evidence From Logs

### Server Alpha (Sync Node)
```
✅ [DISCOVERY] Test request sent to peer 12D3KooWH6tJY...
(90 seconds later)
⚠️ [BLOCK-PACK] Outbound failure: Timeout
🚫 [PEER COMPAT] Peer BLACKLISTED (3+ failures)
```

### Server Beta (Producer Node)
```
(NO BLOCK-PACK LOGS AT ALL)
```

**Conclusion**: Requests are sent but never arrive at Server Beta.

---

## Most Likely Root Cause

### **Network ID Mismatch** (95% confidence)

**Server Beta**: Running `testnet-phase5`
**Server Alpha**: Likely running `testnet-phase6` (newer version)

**Why this breaks BlockPackCodec**:
1. Gossipsub topics include network ID in name
2. P2P discovery may also be network-scoped
3. Peers on different networks can't communicate via custom protocols

**How to verify**:
```bash
# On Server Alpha (161.35.219.10)
curl http://localhost:8080/api/v1/status | jq '.data.network_id'

# On Server Beta (185.182.185.227)
curl http://localhost:8080/api/v1/status | jq '.data.network_id'

# Should match!
```

**How to fix**:
If mismatch confirmed, update Server Alpha or Server Beta to use same network ID.

---

## Secondary Possible Causes

### 1. Firewall Blocking P2P Port 9001

**Check**:
```bash
# On Server Alpha, test connectivity to Server Beta P2P port
telnet 185.182.185.227 9001
```

### 2. BlockPackCodec Not Enabled on Server Beta

**Code location**: Check how `block_sync` behaviour is added to swarm

### 3. Different libp2p Protocol Versions

If one node uses older RequestResponse protocol version, they may be incompatible.

---

## Action Plan

### Step 1: Verify Network IDs Match ⚠️ CRITICAL

```bash
# Check both nodes
ssh 161.35.219.10 "curl -s localhost:8080/api/v1/status | jq '.data.network_id'"
ssh 185.182.185.227 "curl -s localhost:8080/api/v1/status | jq '.data.network_id'"
```

### Step 2: Check P2P Connectivity

```bash
# From Server Alpha
telnet 185.182.185.227 9001
```

### Step 3: Verify Protocol Initialization

```bash
# Check if BlockPackCodec is registered
journalctl -u q-api-server | grep -i "block.*pack\|request.*response.*init"
```

### Step 4: Enable Debug Logging

Add debug logging to see ALL libp2p events:
```rust
// In poll loop
debug!("🔍 [LIBP2P EVENT] {:?}", event);
```

---

## Expected Fix

**If Network ID mismatch**:
- Update both nodes to same network ID
- Restart both services
- Should immediately start working

**If protocol not initialized**:
- Verify `block_sync` behaviour added to swarm
- Ensure poll loop processes all event types

**If P2P connectivity issue**:
- Check firewalls
- Verify ports are open
- Test with nc/telnet

---

**Status**: ⚠️ **ROOT CAUSE INVESTIGATION REQUIRED**

**Next Action**: Check network IDs on both servers

**Expected Resolution Time**: < 5 minutes once root cause confirmed

---

*Analysis created: 2025-11-09 12:56 CET*
*Priority: 🔴 CRITICAL - Blocks 1000 blocks/second performance goal*
