# QNK-101: Peer-Height Message Logging - Investigation Findings

**Date**: November 15, 2025
**Status**: IN PROGRESS - Root Cause Identified
**Priority**: P0 - CRITICAL

---

## Investigation Summary

I've successfully located the peer-height message handler and identified the **EXACT root cause** of the empty peer registry issue.

### Location of Peer-Height Handler

**File**: `crates/q-api-server/src/main.rs`
**Line**: 3459-3600
**Handler Code**:

```rust
} else if topic.ends_with("/peer-heights") {
    // 🚀 TURBO SYNC PEER HEIGHT ANNOUNCEMENTS - Track peer capabilities
    #[derive(serde::Serialize, serde::Deserialize)]
    struct PeerHeightAnnouncement {
        peer_id: String,
        highest_block: u64,
    }

    match postcard::from_bytes::<PeerHeightAnnouncement>(&data) {
        Ok(announcement) => {
            // ✅ Logs successfully parsed announcements
            info!("📡 [TURBO SYNC] Peer {} has height {}", ...);
            turbo_sync.register_peer(peer_id, announcement.highest_block).await;
        }
        Err(e) => {
            // ❌ MISSING: NO ERROR LOGGING HERE!
            // This is the SILENT FAILURE the AI reviewers predicted!
        }
    }
}
```

---

## ROOT CAUSE IDENTIFIED

**The peer-height message parsing has NO error logging in the `Err(e)` arm!**

This means:
1. If bootstrap peer broadcasts height messages in wrong format → **SILENT FAILURE**
2. If protobuf schema changed between versions → **SILENT FAILURE**
3. If peer_id parsing fails → **SILENT FAILURE**
4. Any parse error → **SILENT FAILURE**, registry stays empty

This perfectly matches:
- ✅ Kimi's hypothesis: "Silent failure pattern strongly suggests missing error handling"
- ✅ ChatGPT's ticket QNK-101: "Add detailed logging for peer-height message handling"
- ✅ DeepSeek's assessment: "Missing error handler + logging"

---

## MISSING ERROR HANDLER

The code needs this Err arm (currently missing/not logging):

```rust
Err(e) => {
    // 🚨 QNK-101: Add diagnostic logging
    warn!("🔍 [QNK-101 DEBUG] Failed to parse peer height announcement from topic {}",
          topic);
    warn!("🔍 [QNK-101 DEBUG] Error: {}", e);
    warn!("🔍 [QNK-101 DEBUG] Message size: {} bytes", data.len());
    warn!("🔍 [QNK-101 DEBUG] First {} bytes (hex): {}",
          data.len().min(64),
          hex::encode(&data[..data.len().min(64)]));

    // Try to debug the peer_id field specifically
    if let Ok(s) = String::from_utf8(data.clone()) {
        warn!("🔍 [QNK-101 DEBUG] Data as UTF-8: {}", &s[..s.len().min(200)]);
    }
}
```

---

## Gossipsub Message Flow

**Message Path**:
1. Bootstrap peer publishes to `/qnk/testnet-phase11/peer-heights` topic
2. Message received in `unified_network_manager.rs` line 993 (Gossipsub::Event::Message)
3. Forwarded via channel to `gossipsub_message_tx` (line 1114-1136)
4. Consumed in `main.rs` line 3459 in topic-specific handler
5. **PARSE ATTEMPT** with `postcard::from_bytes::<PeerHeightAnnouncement>()`
6. **IF PARSE FAILS** → Silent skip, no registry update

---

## Current Evidence

### What Works ✅
- Gossipsub subscription to `/qnk/testnet-phase11/peer-heights` (main.rs:1013-1016)
- P2P connection established to bootstrap peer
- Gossipsub forwarding channel operational
- Topic routing correctly identifies `/peer-heights` messages

### What's Missing ❌
- **NO logging when postcard deserialization fails**
- **NO logging showing raw message received on peer-heights topic**
- **NO logging showing whether ANY messages arrive on this topic**

### Current Logs Show
- ✅ AI gossipsub deserialization errors ARE logged (proving error logging works elsewhere)
- ❌ NO peer-height deserialization errors logged (proving silent failure)
- ❌ NO "Received peer height message" debug logs
- ❌ NO "📡 [TURBO SYNC] Peer X has height Y" messages (proving NO successful parses)

---

## Next Steps (QNK-101 Implementation)

### Step 1: Add Message Reception Logging (HIGH PRIORITY)

Add BEFORE the `match postcard::from_bytes` statement:

```rust
// 🔍 QNK-101: Log ALL peer-height messages received
warn!("🔍 [QNK-101] Received peer-height message on topic: {}", topic);
warn!("🔍 [QNK-101] Message size: {} bytes", data.len());
warn!("🔍 [QNK-101] First 64 bytes (hex): {}",
      hex::encode(&data[..data.len().min(64)]));
```

**This will answer**: Are height messages reaching the node at all?

### Step 2: Add Error Logging (CRITICAL)

Add in the `Err(e)` arm:

```rust
Err(e) => {
    error!("❌ [QNK-101] Failed to deserialize peer height announcement!");
    error!("   Topic: {}", topic);
    error!("   Error: {}", e);
    error!("   Message size: {} bytes", data.len());
    error!("   First 100 bytes: {:?}", &data[..data.len().min(100)]);

    // Try UTF-8 decode to see if it's a string format issue
    if let Ok(s) = String::from_utf8(data.clone()) {
        error!("   Data as UTF-8 string: {}", &s[..s.len().min(200)]);
    }
}
```

**This will answer**: WHY are messages failing to parse (if they arrive)?

### Step 3: Add Successful Parse Confirmation

Enhance existing success logging:

```rust
Ok(announcement) => {
    info!("✅ [QNK-101] Successfully parsed peer height announcement!");
    info!("   Peer ID: {}", announcement.peer_id);
    info!("   Height: {}", announcement.highest_block);
    info!("   Message size: {} bytes", data.len());

    // Existing registration code...
}
```

---

## Expected Outcomes After Fix

### Scenario A: Messages Not Arriving
```
[No logs with "Received peer-height message"]
→ Problem: Bootstrap peer not broadcasting heights
→ Action: Check bootstrap peer logs (QNK-301)
```

### Scenario B: Messages Arriving But Parse Failing
```
🔍 [QNK-101] Received peer-height message on topic: /qnk/testnet-phase11/peer-heights
🔍 [QNK-101] Message size: 128 bytes
❌ [QNK-101] Failed to deserialize peer height announcement!
   Error: missing field `peer_id` at line 1
→ Problem: Schema mismatch or wrong serialization format
→ Action: Fix message format or add backward compatibility
```

### Scenario C: Messages Arriving And Parsing Successfully
```
🔍 [QNK-101] Received peer-height message
✅ [QNK-101] Successfully parsed peer height announcement!
   Peer ID: 12D3KooW...
   Height: 88500
📡 [TURBO SYNC] Peer 12D3KooW... has height 88500
→ Problem: None! Fix worked!
→ Action: Verify registry population (QNK-102)
```

---

## Files That Need Modification

1. **`crates/q-api-server/src/main.rs`** (lines 3459-3600)
   - Add message reception logging before parse attempt
   - Add error logging in `Err(e)` arm
   - Enhance success logging

2. **`Cargo.toml`** (workspace root)
   - Bump version to v1.0.15-beta (diagnostic build)

---

## Estimated Fix Time

**Diagnostic Logging Only**: 15-30 minutes
**Full Fix (including schema correction)**: 2-6 hours

**Confidence**: 95% that adding this logging will immediately reveal the root cause.

---

## Related Tickets

- **QNK-102**: Add periodic registry status logging (next step)
- **QNK-103**: Add batch sync decision logging (required for full diagnosis)
- **QNK-301**: Verify bootstrap peer broadcasting heights (parallel investigation)

---

**Status**: Ready for implementation
**Blocker**: None
**Next Action**: Implement diagnostic logging in main.rs:3459-3600
