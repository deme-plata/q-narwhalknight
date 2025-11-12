# 🚀 Q-NarwhalKnight AI - Quick Start Guide

## ⚡ TL;DR - Start AI in 3 Commands

```bash
# 1. Wait for build to complete (check with):
tail -f /tmp/ai-resource-build.log

# 2. When you see "Finished", restart service:
systemctl restart q-api-server

# 3. Test AI:
curl -N "http://localhost:8080/api/chat/test/stream?content=Hello&max_tokens=20"
```

**Expected output**: Real-time streaming tokens! 🎉

---

## 📋 Complete Startup Checklist

### Step 1: Verify Build Completed ✅

```bash
# Check if build finished:
tail -20 /tmp/ai-resource-build.log | grep "Finished"

# Should see:
# Finished `release` profile [optimized] target(s) in XX.XXs
```

### Step 2: Verify Binary Exists ✅

```bash
ls -lh target/release/q-api-server

# Should show:
# -rwxr-xr-x ... target/release/q-api-server
```

### Step 3: Check Service Configuration ✅

```bash
cat /etc/systemd/system/q-api-server.service | grep Q_ENABLE_AI

# Should show:
# Environment="Q_ENABLE_AI=1"
```

### Step 4: Restart Service ⚡

```bash
systemctl restart q-api-server
```

### Step 5: Watch Startup Logs 👀

```bash
journalctl -u q-api-server -f
```

**Look for these messages**:
```
✅ "🚀 Initializing mistral.rs high-performance engine..."
✅ "🔧 Limiting AI inference to 4 threads (out of 18 cores)"
✅ "🚦 Rate limiting: 2 concurrent AI requests max"
✅ "📦 Loading GGUF model with mistral.rs optimizations..."
✅ "✅ mistral.rs engine initialized successfully!"
```

**Startup takes ~10-15 seconds** (loading 4GB model into RAM)

---

## 🧪 Testing AI Endpoints

### Test 1: Simple Streaming

```bash
curl -N "http://localhost:8080/api/chat/test-123/stream?content=Hello&max_tokens=20"
```

**Expected output** (real-time):
```
event: progress
data: 🔤 Tokenizing prompt...

event: progress
data: 🚀 Generating response (mistral.rs optimized)...

event: token
data: {"token":"Hello","cumulative":"Hello"}

event: token
data: {"token":"!","cumulative":"Hello!"}

event: token
data: {"token":" How","cumulative":"Hello! How"}

event: complete
data: {"total_tokens":15,"tokens_per_second":4.2,"time_to_first_token_ms":3245}
```

### Test 2: Create Chat Session

```bash
curl -X POST http://localhost:8080/api/chat/create \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test-user",
    "title": "Test Chat",
    "enable_kv_cache": true
  }'
```

**Expected output**:
```json
{
  "success": true,
  "data": {
    "chat_id": "abc-def-123",
    "created_at": 1698765432
  }
}
```

### Test 3: List Chats

```bash
curl "http://localhost:8080/api/chat/list?user_id=test-user"
```

**Expected output**:
```json
{
  "success": true,
  "data": [
    {
      "chat_id": "abc-def-123",
      "user_id": "test-user",
      "title": "Test Chat",
      "message_count": 0,
      "created_at": 1698765432
    }
  ]
}
```

---

## 📊 Monitor Performance

### Watch CPU Usage

```bash
htop
```

**What to look for**:
- **Total CPU**: Should be ~22-25% when AI is generating
- **Free cores**: 14 cores should remain available for mining
- **Memory**: ~5GB used (4GB model + 1GB service)

### Watch Service Logs

```bash
journalctl -u q-api-server -f | grep -i "ai\|mistral\|inference"
```

**Good signs**:
```
INFO  🌊 SSE stream started for chat abc-123
INFO  🚀 Generating 150 tokens with mistral.rs HIGH-PERFORMANCE engine...
INFO  ⚡ First token in 3245ms
INFO  📊 15/150 tokens (4.2 tok/s)
INFO  ✅ mistral.rs SSE stream complete - 45 tokens in 9.8s (4.6 tok/s)
```

### Check Mining Still Works

```bash
curl http://localhost:8080/api/v1/node/status | jq .
```

**Should show**:
- `"is_online": true`
- `"block_height"`: Increasing
- `"peers"`: > 0

---

## ⚙️ Adjust Resources (Optional)

### If Server Still Feels Slow

```bash
# Edit service file
nano /etc/systemd/system/q-api-server.service

# Reduce AI threads (currently 4):
Environment="Q_AI_THREADS=2"  # Use only 2 cores (11% CPU)

# Reduce concurrent requests (currently 2):
Environment="Q_AI_MAX_CONCURRENT=1"  # Only 1 request at a time

# Reload and restart
systemctl daemon-reload
systemctl restart q-api-server
```

### If AI Too Slow

```bash
# Increase AI threads:
Environment="Q_AI_THREADS=8"  # Use 8 cores (44% CPU)

# Allow more concurrent requests:
Environment="Q_AI_MAX_CONCURRENT=4"  # Up to 4 simultaneous

# Reload and restart
systemctl daemon-reload
systemctl restart q-api-server
```

---

## 🐛 Troubleshooting

### Problem: "AI inference engine not initialized"

**Check**:
```bash
grep Q_ENABLE_AI /etc/systemd/system/q-api-server.service
```

**Should be**: `Environment="Q_ENABLE_AI=1"`

**Fix**:
```bash
# Edit service file, set Q_ENABLE_AI=1
systemctl daemon-reload
systemctl restart q-api-server
```

### Problem: "Model file not found"

**Check**:
```bash
ls /opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
```

**If missing**, download:
```bash
mkdir -p /opt/orobit/shared/q-narwhalknight/models
cd /opt/orobit/shared/q-narwhalknight/models

# Download from Hugging Face (4GB file)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
```

### Problem: Server still unresponsive during AI

**Reduce resources**:
```bash
# Edit: Q_AI_THREADS=2, Q_AI_MAX_CONCURRENT=1
systemctl daemon-reload
systemctl restart q-api-server
```

### Problem: "Failed to acquire request permit"

**Meaning**: Too many concurrent requests (queue full)

**Solutions**:
1. Wait a moment and retry
2. Increase `Q_AI_MAX_CONCURRENT` in service file
3. Reduce number of simultaneous users

---

## 🎨 Frontend Integration (Next Steps)

**Option 1: Quick Test with HTML**

Create `test-chat.html`:
```html
<!DOCTYPE html>
<html>
<head>
  <title>Q-NarwhalKnight AI Chat Test</title>
</head>
<body>
  <h1>AI Chat Test</h1>
  <div id="messages"></div>
  <input type="text" id="input" placeholder="Type a message...">
  <button onclick="sendMessage()">Send</button>

  <script>
    function sendMessage() {
      const input = document.getElementById('input');
      const message = input.value;
      input.value = '';

      const messagesDiv = document.getElementById('messages');
      messagesDiv.innerHTML += `<p><strong>You:</strong> ${message}</p>`;

      const eventSource = new EventSource(
        `/api/chat/test/stream?content=${encodeURIComponent(message)}&max_tokens=100`
      );

      let aiMessage = document.createElement('p');
      aiMessage.innerHTML = '<strong>AI:</strong> ';
      messagesDiv.appendChild(aiMessage);

      eventSource.addEventListener('token', (e) => {
        const data = JSON.parse(e.data);
        aiMessage.textContent = 'AI: ' + data.cumulative;
      });

      eventSource.addEventListener('complete', () => {
        eventSource.close();
      });
    }
  </script>
</body>
</html>
```

**Serve it**:
```bash
cp test-chat.html gui/quantum-wallet/dist-final/
# Open: http://your-server/test-chat.html
```

**Option 2: Full React Integration**

Follow the comprehensive plan in:
```
AI_CHAT_UI_INTEGRATION_PLAN.md
```

---

## 📞 Quick Commands Reference

| Action | Command |
|--------|---------|
| **Restart AI** | `systemctl restart q-api-server` |
| **Check Status** | `systemctl status q-api-server` |
| **View Logs** | `journalctl -u q-api-server -f` |
| **Test SSE** | `curl -N "http://localhost:8080/api/chat/test/stream?content=Hi"` |
| **Check CPU** | `htop` |
| **Check Memory** | `free -h` |
| **Edit Config** | `nano /etc/systemd/system/q-api-server.service` |
| **Reload Config** | `systemctl daemon-reload` |

---

## 🎓 Understanding the Output

### SSE Event Types

| Event | Meaning | Example Data |
|-------|---------|--------------|
| `start` | Generation started | `"Generation started"` |
| `progress` | Status update | `"🔤 Tokenizing prompt..."` |
| `token` | New token generated | `{"token":"Hi","cumulative":"Hi there"}` |
| `complete` | Generation done | `{"total_tokens":45,"tokens_per_second":4.2}` |
| `error` | Something went wrong | `"Generation error: ..."` |

### Performance Numbers

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **First Token** | < 3s | 3-7s | > 7s |
| **Tokens/sec** | > 5 | 3-5 | < 3 |
| **CPU Usage** | 20-30% | 30-50% | > 50% |
| **Server Response** | Instant | < 1s lag | Unresponsive |

---

## 🏆 Success Checklist

- [ ] Build completed successfully
- [ ] Service restarted with Q_ENABLE_AI=1
- [ ] Logs show "✅ mistral.rs engine initialized successfully!"
- [ ] Test SSE endpoint returns streaming tokens
- [ ] CPU usage ~22% during generation
- [ ] Mining continues normally
- [ ] Server remains responsive (SSH works)
- [ ] Memory usage stable (~5GB total)

**When all checked**: ✅ AI is ready! 🎉

---

## 📖 Documentation Reference

1. **AI_RESOURCE_MANAGEMENT.md** - Detailed resource control docs
2. **AI_CHAT_UI_INTEGRATION_PLAN.md** - Complete UI development plan
3. **AI_INTEGRATION_COMPLETE_SUMMARY.md** - Full technical summary
4. **QUICKSTART_AI.md** - This file (quick reference)

---

**Need help? Check logs first**: `journalctl -u q-api-server -f | grep ERROR`

**All good?** Start building the chat UI! 🚀
