# AI Chat Message Persistence Fix

**Date**: October 29, 2025  
**Issue**: AI assistant responses disappear when user navigates away from chat and returns  
**Status**: PARTIALLY FIXED

---

## Problem Identified

### Root Cause
AI chat messages were being stored only in React component state and not being persisted to the database when users navigated away from a chat before the AI response completed generating.

### Technical Details

1. **Frontend Issue** (`AIChatScreen.tsx:349`):
   - Comment: `// DON'T reload messages from backend - our UI state is already correct!`
   - Frontend intentionally DID NOT reload messages from database after generation completed
   - Messages only existed in React state, lost on navigation

2. **Backend Behavior** (`chat_api.rs:613`):
   - AI messages ARE being saved to database when stream completes successfully
   - Saves happen in `StreamEvent::Complete` handler
   - **BUT**: If EventSource connection closes before completion (user navigates away), the complete event never fires

3. **Partial Save Logic** (`chat_api.rs:562-586`):
   - There IS code to save partial responses when client disconnects
   - However, this only triggers if disconnection happens DURING token generation
   - If user navigates BEFORE generation starts, nothing gets saved

---

## Fix Applied

### Frontend Fix (`AIChatScreen.tsx`)

**Changed** (line 349-351):
```typescript
// Reload messages from backend to ensure persistence
// This ensures messages are fetched from the database on chat switch
await loadMessages(currentChatId);
```

**Previously**:
```typescript
// DON'T reload messages from backend - our UI state is already correct!
// The backend saves messages automatically via the streaming endpoint.
// Reloading causes messages to disappear if user sends another message quickly.
```

**Impact**: 
- ✅ Messages that successfully complete generation are now visible after returning to chat
- ✅ Frontend always fetches from database, ensuring persistence across sessions
- ❌ Messages still lost if user navigates away before generation completes

---

## Remaining Issue

**Scenario**: User navigates away before AI response finishes generating

**What Happens**:
1. User sends message
2. Backend saves user message ✅
3. Backend starts generating AI response
4. User switches to different chat or screen
5. EventSource connection closes
6. Backend's `complete` event never fires
7. AI response is NEVER saved to database ❌

**Evidence**:
```bash
# Test confirmed: Only user messages persisted when stream interrupted
$ curl "http://localhost:8080/api/chat/{chat_id}/messages"
{
  "data": [
    {"role": "user", "content": "Hello AI"},  // ✅ Saved
    // ❌ NO assistant response
  ]
}
```

---

## Complete Solution (TODO)

### Option 1: Save User Message + Placeholder Immediately
```rust
// In stream_message() - BEFORE starting generation
let placeholder_ai_message = ChatMessage {
    index: ai_message_index,
    role: "assistant".to_string(),
    content: "".to_string(),  // Empty placeholder
    timestamp: current_timestamp(),
    images: None,
    audio: None,
    generation_stats: None,
};

state.storage_engine.save_chat_message(&chat_id, &placeholder_ai_message).await?;
```

Then UPDATE the message as tokens arrive, ensuring it's always persisted.

### Option 2: Background Task for Message Saving
- Spawn tokio task that saves messages independent of connection
- Store cumulative_text in Arc<RwLock<String>>
- Periodic saves every N tokens or N seconds
- Final save on completion

### Option 3: WebSocket Instead of SSE
- Bidirectional connection allows backend to confirm saves
- Frontend can track save status
- Reconnection logic on navigation

---

## Testing

### Test 1: Full Stream Completion ✅
```bash
# AI inference engine running but VERY slow (0.4-0.8 tokens/sec)
$ journalctl -u q-api-server | grep mistralrs_core
Throughput (T/s) 0.60, Prefix cache hitrate 50.00%, 1 running, 0 waiting
```

Messages ARE saved when stream completes naturally.

### Test 2: Early Navigation ❌
```bash
# User navigates away after 5 seconds
# Generation takes 60+ seconds for 20 tokens
# Result: Only user message saved, no AI response
```

---

## Files Modified

1. `/opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/src/components/AIChatScreen.tsx`
   - Line 349-351: Added `await loadMessages(currentChatId)` to reload from database
   - Removed comment discouraging database reloads

2. Frontend rebuilt:
   - `npm run build` completed successfully
   - New bundle: `dist-final/assets/index-jHzZzmcy-1761773355268.js` (2.2MB)

---

## Recommendation

Implement **Option 1** (Placeholder + Update Pattern) as it's the simplest and most reliable:
- Zero-latency user experience (placeholder shown immediately)
- Guaranteed persistence even if user navigates away
- No need for background tasks or protocol changes
- Consistent with REST API behavior

**Implementation Priority**: HIGH - User data loss is unacceptable

---

*Document created: 2025-10-29 22:30 UTC*  
*Fix status: Partial - Frontend reloads from DB, backend still needs save-on-disconnect logic*
