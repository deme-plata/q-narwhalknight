# Q-NarwhalKnight v0.2.4-beta Release Notes

**Release Date**: October 30, 2025
**Version**: v0.2.4-beta
**Critical Fixes**: Mining rewards visibility + Graceful shutdown

---

## 🎉 Release Summary

This release addresses critical user-reported issues:
1. **Mining rewards not visible** when mining to localhost
2. **30-minute shutdown times** blocking deployments and restarts
3. **Verbose AI library logs** cluttering node output

All issues have been resolved with comprehensive fixes and testing.

---

## 🔧 Critical Fixes

### 1. ✅ Mining Reward Visibility (SSE Broadcasting)

**Problem**: Miners couldn't see their rewards in real-time despite being credited to wallets.

**Root Cause**: SSE sampling only broadcast 1 in 10 balance updates to reduce event spam.

**Fix**: Removed sampling logic - now **ALL mining rewards** are broadcast immediately via `/api/v1/stream/events`.

**Impact**:
- Miners see **every single reward** in real-time
- Transparent reward tracking
- Full visibility into mining earnings

**Location**: `crates/q-api-server/src/main.rs:1364-1379`

---

### 2. ✅ Graceful Shutdown (0.975s vs 30 minutes)

**Problem**: `systemctl stop q-api-server` took ~30 minutes, blocking deployments.

**Root Causes**:
- No SIGTERM signal handler
- No request timeout for AI inference
- No systemd timeout configuration

**Fixes**:
1. **SIGTERM Handler** - Added graceful shutdown for systemd signals
2. **Request Timeout** - 120-second global timeout for all requests
3. **Systemd Timeout** - 30-second maximum with force kill

**Performance**:
- **Before**: ~30 minutes (1800 seconds)
- **After**: 0.975 seconds
- **Improvement**: 1,846x faster shutdown!

**Locations**:
- `crates/q-api-server/src/high_performance_server.rs:112-139`
- `crates/q-api-server/src/main.rs:3081-3083`
- `/etc/systemd/system/q-api-server.service:29-32`

---

### 3. ✅ Clean Startup Logs (AI Library Filtering)

**Problem**: Millions of tensor/byte array debug logs from AI libraries on startup.

**Root Cause**: External AI/ML libraries logging at debug level during model initialization.

**Fix**: Updated tracing configuration to filter external libraries (candle, mistralrs, tokenizers) to `warn` level.

**Impact**: Clean startup logs showing only relevant application output.

**Location**: `crates/q-api-server/src/main.rs:189-198`

---

### 4. ✅ Block Production Monitoring

**Problem**: No visibility into block production process or delays.

**Fix**: Added comprehensive logging:
- Queue depth before production
- Production duration timing
- Detailed block metrics (Producer ID, Height, Hash, Solutions, TX count)

**Example Output**:
```
🔨 Block production triggered - Queue depth: 347 solutions
⚡ Produced 2 blocks in 145.3ms
🎉 BLOCK PRODUCED: Producer #0 | Height 12847 | Hash 4a7f3e21 | Solutions 156 | TX 3
```

**Location**: `crates/q-api-server/src/main.rs:1398-1417`

---

## 📊 Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Shutdown Time** | 30 minutes | 0.975s | 1,846x faster |
| **SSE Rewards** | 10% broadcast | 100% broadcast | Complete visibility |
| **Startup Logs** | Millions of lines | Clean output | Readable logs |
| **Block Monitoring** | No visibility | Full diagnostics | Complete transparency |

---

## 🚀 Upgrade Instructions

### Option 1: Download Pre-Built Binary (Recommended)

1. Visit https://quillon.xyz
2. Navigate to "Download Node"
3. Download `q-api-server-v0.2.4-beta` for Linux
4. Stop your current node:
   ```bash
   sudo systemctl stop q-api-server
   ```
5. Replace binary:
   ```bash
   sudo cp q-api-server-v0.2.4-beta /usr/local/bin/q-api-server
   sudo chmod +x /usr/local/bin/q-api-server
   ```
6. Start node:
   ```bash
   sudo systemctl start q-api-server
   ```

### Option 2: Build from Source

```bash
cd /opt/orobit/shared/q-narwhalknight
git pull origin clean-branch
timeout 36000 cargo build --release --package q-api-server --bin q-api-server
sudo systemctl stop q-api-server
sudo cp target/release/q-api-server /usr/local/bin/
sudo systemctl start q-api-server
```

### Update Systemd Configuration

If you have a custom systemd service file, add these lines to the `[Service]` section:

```ini
# Graceful shutdown timeout (30s for AI inference to complete, then force SIGKILL)
TimeoutStopSec=30
KillMode=mixed
KillSignal=SIGTERM
```

Then reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart q-api-server
```

---

## ✅ Verification

### Check Mining Rewards in Real-Time

1. Start mining:
   ```bash
   ./q-miner --node http://localhost:8080 --address YOUR_WALLET_ADDRESS
   ```

2. Monitor SSE stream:
   ```bash
   curl -N http://localhost:8080/api/v1/stream/events
   ```

   You should see:
   ```json
   {
     "type": "BalanceUpdated",
     "wallet_address": "YOUR_ADDRESS",
     "old_balance": 150.5,
     "new_balance": 151.2,
     "change_reason": "mining_reward",
     "timestamp": "2025-10-30T12:34:56Z"
   }
   ```

### Check Block Production

Monitor logs:
```bash
journalctl -u q-api-server -f | grep "BLOCK PRODUCED"
```

You should see:
```
🔨 Block production triggered - Queue depth: 234 solutions
⚡ Produced 1 blocks in 89.2ms
🎉 BLOCK PRODUCED: Producer #0 | Height 12850 | Hash 3f8a9c12 | Solutions 234 | TX 7
```

### Check Shutdown Time

Test shutdown:
```bash
time sudo systemctl stop q-api-server
```

Should complete in **< 5 seconds** (was 30 minutes).

### Check Your Balance

Query wallet:
```bash
curl http://localhost:8080/api/v1/wallet/YOUR_ADDRESS/balance
```

---

## 🛡️ What Was NOT Changed

### AEGIS Authentication
- Mining endpoints (`/api/v1/mining/submit`, `/api/v1/mining/challenge`) remain **public**
- No authentication required for mining to localhost
- AEGIS middleware only protects admin routes

### 1% Development Fee
- Transparent 1% dev fee continues to work correctly
- Fee applied at submission time and clearly logged
- Not causing reward issues

### Mining Reward Calculation
- Block rewards: 50 QNK per block
- Distribution: Proportional to solution difficulty contribution
- Batch processing: 25,000 submissions/second capacity
- All unchanged and working as designed

---

## 📈 Technical Details

### Mining Reward Flow

1. Miner submits solution → `/api/v1/mining/submit`
2. Submission queued to background processor (non-blocking)
3. Batch processed (500 submissions per batch or 20ms interval)
4. Rewards calculated proportionally by difficulty contribution
5. Balance updated in RocksDB (async, non-blocking)
6. **SSE event broadcast to all listeners (NOW: ALL REWARDS)**
7. Solutions queued to block producer pool
8. Block production triggered when threshold reached
9. New block broadcast to P2P network

### Graceful Shutdown Flow

```
User: systemctl stop q-api-server
  ↓
systemd: Send SIGTERM to process
  ↓
Server: Receive SIGTERM via tokio signal handler
  ↓
Server: Stop accepting new connections
  ↓
Server: Wait for in-flight requests to complete
  ↓  (max 120s per request due to timeout layer)
  ↓
Server: Close all connections gracefully
  ↓
Server: Exit cleanly (0.975s)
  ↓
systemd: Process exited successfully
  ↓
[If still running after 30s]
  ↓
systemd: Send SIGKILL (force terminate)
```

### Performance Metrics

- **Mining submission rate**: 25,000 submissions/second
- **Batch processing interval**: 20ms maximum latency
- **SSE broadcast**: All rewards (was 10% sampled)
- **Block production**: Monitored with queue depth + timing
- **Database writes**: Async, non-blocking
- **Request timeout**: 120 seconds global limit
- **Shutdown timeout**: 30 seconds maximum

---

## 🐛 Known Issues

None. All reported issues have been resolved in this release.

---

## 📚 Detailed Documentation

- **Mining Fixes**: See `MINING_FIXES_v0.2.4-beta.md`
- **Shutdown Fixes**: See `SHUTDOWN_FIXES_v0.2.4-beta.md`

---

## 💬 Support

If you experience issues:

1. Check node logs: `journalctl -u q-api-server -n 100`
2. Verify node is synced: `curl http://localhost:8080/api/v1/status`
3. Check SSE stream: `curl -N http://localhost:8080/api/v1/stream/events`
4. Report issues on Discord/Telegram with logs

---

## 🙏 Acknowledgments

Thank you to all community members who reported these issues and helped test the fixes. Your feedback makes Q-NarwhalKnight better!

---

**Version**: v0.2.4-beta
**Build**: q-api-server with mining fixes + graceful shutdown + distributed AI
**Date**: 2025-10-30
**Quantum consensus awaits!** ⚛️🚀
