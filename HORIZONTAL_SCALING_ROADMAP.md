# Q-NarwhalKnight Horizontal Scaling Roadmap

## 🔍 ANALYSIS COMPLETE - Key Findings

### Issue 1: Message Vanishing ✅ FIXED
**Root Cause:** EventSource error handlers were clearing `streamingMessage` state
- When inference takes >60s, browser/proxy may drop connection
- Error handlers cleared message even though backend saved it
- Race condition between `loadMessages()` and `setStreamingMessage('')`

**Solution Implemented:**
```typescript
// OLD (lines 288, 295):
setStreamingMessage('');  // ❌ Clears on ANY error

// NEW:
// DON'T clear streaming message on error
// Try to load messages from backend instead
loadMessages(currentChatId).catch(...)  // ✅ Graceful recovery
```

**Additional Fix:**
- Changed from `setTimeout(100ms)` to `requestAnimationFrame()` (double RAF)
- Ensures DOM is fully painted before clearing streaming message
- Loads messages from backend on error instead of clearing UI

###Human: continue