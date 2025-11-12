# Block Producer Stall at Height 840 - Diagnosis

**Date**: 2025-11-06
**Status**: 🔴 **CRITICAL - BLOCK PRODUCER STALLED**

---

## 🔴 **PROBLEM**

Node is stuck at height 840 with block producer completely stalled.

**Symptoms:**
- Current height: 840
- Block producer watchdog triggered: "Height unchanged for 60 seconds: 839"
- Mining solutions being queued but NOT processed
- No "BLOCK PRODUCER" or "Creating new block" logs
- No block creation activity despite active mining

**Evidence from logs:**
```
2025-11-06T07:57:15.696818Z ERROR q_api_server: 🚨 WATCHDOG: Block producer STALLED!
2025-11-06T07:57:15.696883Z ERROR q_api_server:    Height unchanged for 60 seconds: 839
2025-11-06T07:57:15.696887Z ERROR q_api_server:    Node height: 840
2025-11-06T07:57:15.696895Z ERROR q_api_server:    IMMEDIATE ACTION REQUIRED: Service needs restart
```

---

## 🔍 **ROOT CAUSE**

**Block Producer Task Hung or Deadlocked**

The block producer background task has stopped processing mining solutions. Possible causes:

1. **Deadlock** - Block producer waiting on a lock that never releases
2. **Channel Full** - Mining solution queue filled up and blocked
3. **Panic** - Block producer task panicked silently
4. **Resource Starvation** - No CPU time allocated to producer task

**Key Observations:**
- Mining solutions ARE being queued: `⚡ Mining submission queued (non-blocking)`
- Block producer is NOT creating blocks (no logs)
- Network height shows 814 (lower than our 840) - we're ahead but stuck
- Process is running (PID 305185) but not producing blocks

---

## ✅ **IMMEDIATE FIX**

### **Restart the Service**

The watchdog explicitly states "Service needs restart":

```bash
# Stop current service
systemctl stop q-api-server

# Wait for clean shutdown
sleep 5

# Start fresh instance
systemctl start q-api-server

# Monitor startup
journalctl -u q-api-server -f
```

**Expected Result:**
- Block producer resumes at height 841
- Mining solutions start being processed
- Blocks created and propagated to network

---

## 🔧 **PERMANENT FIX OPTIONS**

### **Option 1: Auto-Restart on Stall (Recommended)**

Update systemd service to automatically restart on failure:

```ini
[Service]
Restart=on-failure
RestartSec=10
# Restart if block producer stalls (exit code from watchdog)
```

### **Option 2: Increase Block Producer Priority**

Ensure block producer task gets CPU time:

```rust
// In block producer spawn
tokio::task::Builder::new()
    .name("block-producer")
    .spawn(async move {
        // Block producer logic
    });
```

### **Option 3: Add Deadlock Detection**

Implement timeout for block production:

```rust
// In block producer loop
tokio::select! {
    solution = rx.recv() => {
        // Process solution
    }
    _ = tokio::time::sleep(Duration::from_secs(30)) => {
        error!("Block producer timeout - no solution processed in 30s");
        // Force restart or recovery
    }
}
```

### **Option 4: Increase Solution Queue Size**

If channel is full:

```rust
// Increase from default (probably 100) to larger size
let (tx, rx) = tokio::sync::mpsc::channel(10000);
```

---

## 📊 **VERIFICATION STEPS**

After restart:

1. **Check block production resumes:**
   ```bash
   journalctl -u q-api-server -f | grep "BLOCK PRODUCER"
   ```

2. **Verify height increases:**
   ```bash
   watch -n1 'curl -s http://localhost:8080/api/v1/node/status | jq .data.current_height'
   ```

3. **Monitor for stall recurrence:**
   ```bash
   journalctl -u q-api-server -f | grep -E "(STALL|WATCHDOG)"
   ```

4. **Check mining rewards appear:**
   ```bash
   curl -s http://localhost:8080/api/v1/node/status | jq .data.balance
   ```

---

## 🐛 **DEBUG INFORMATION**

**Process Status:**
```
PID: 305185
Memory: 6.1GB RSS
CPU: 129% (saturated)
Uptime: ~34 minutes since last restart
```

**Network Status:**
```json
{
  "current_height": 840,
  "highest_network_height": 814,
  "is_syncing": false,
  "peer_count": null
}
```

**Other Issues Observed:**
- AI gossipsub deserialization errors (non-blocking)
- Turbo sync duplicate publish warnings (cosmetic)
- Mining still shows as "stalled" (13.2 minutes without solutions being PROCESSED)

---

## 💡 **LESSONS LEARNED**

1. **Watchdog is working correctly** - Detected stall in 60 seconds
2. **Need automatic recovery** - Manual restart shouldn't be required
3. **Block producer needs monitoring** - Deadlock detection critical
4. **Queue monitoring needed** - Track solution queue depth

---

**Next Steps:**
1. Restart service immediately
2. Monitor for recurrence
3. Implement auto-restart in systemd
4. Add deadlock detection to block producer
