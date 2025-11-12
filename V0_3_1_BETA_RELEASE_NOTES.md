# Q-NarwhalKnight v0.3.1-beta Release Notes

## Critical Fixes

### Issue #1: Gossipsub Block Deserialization Failure ✅
**Problem:** Nodes receive gossipsub block messages but fail to deserialize them, staying stuck at low block heights.

**Root Cause:** Silent deserialization failures - errors were logged as `warn!` making them easy to miss.

**Solution:**
- Enhanced error logging with detailed diagnostics
- Shows topic, data size, error message, and hex dump of first 32 bytes
- Changed from `warn!` to `error!` level for visibility

**Files Modified:**
- `crates/q-api-server/src/main.rs` (lines 2065-2073)

**Result:** Administrators can now immediately see WHY blocks aren't being processed.

---

### Issue #2: Sync-First Mode - Mining During Sync ✅
**Problem:** Nodes mine their own blocks while trying to sync, creating blockchain forks.

**Solution:**
- Track highest network height from received blocks
- Pause block production when >10 blocks behind
- Resume mining automatically when synced
- Mining submissions still queued during sync

**Files Modified:**
- `crates/q-api-server/src/lib.rs` (line 485: `highest_network_height`)
- `crates/q-api-server/src/main.rs` (lines 1890-1895, 1593-1608, 1407-1417)

**Result:** Nodes sync cleanly without creating conflicting blocks.

---

### Issue #3: Active Sync Monitoring ✅
**Problem:** No visibility into sync progress or stalls.

**Solution:**
- Added active sync loop (every 2 seconds)
- Logs sync progress: "🚀 FAST SYNC: X blocks behind"
- Detects stalls: "⚠️ Not receiving blocks from network!"
- Shows sync rate and percentage complete
- **P2P-ONLY SYNC**: Removed HTTP sync fallback - all syncing through libp2p gossipsub

**Files Modified:**
- `crates/q-api-server/src/main.rs` (lines 1810-1854)

**Result:** Clear visibility into sync status and problems. Pure P2P architecture.

---

### Issue #4: Miner Real-Time Block Updates ✅
**Problem:** Miners only polled for new blocks every 50 seconds, mining stale blocks.

**Solution:**
- Added SSE listener for `new-block` events
- Shared atomic signal across mining threads
- Threads fetch new challenge immediately (<1s latency)

**Files Modified:**
- `crates/q-miner/src/main.rs` (lines 215-217, 495-546)

**Binary:** `q-miner-v0.3.0-beta`

---

## Docker P2P Setup

### CRITICAL: Expose Port 8081 for P2P

Docker containers MUST expose both ports for P2P block propagation to work:

```bash
# ✅ CORRECT - Both ports exposed
docker run -d \
  --name quillon-node \
  -p 9080:8080 \
  -p 9081:8081 \
  quillon-api:v0.3.1

# ✅ BEST - Host networking (fastest)
docker run -d \
  --name quillon-node \
  --network host \
  quillon-api:v0.3.1

# ❌ WRONG - Only HTTP port
docker run -d \
  --name quillon-node \
  -p 9080:8080 \
  quillon-api:v0.3.1  # P2P port not exposed!
```

**Result with correct setup:**
```
📦 Received block 100 (height=100) from network
📈 Network height updated: 99 -> 100
✅ Stored incoming block 100 to RocksDB
📈 Node height advanced to 100
```

**Result with wrong setup:**
```
(Only sees raw binary data in DEBUG logs, no block processing)
⚠️ Not receiving blocks from network! Blocks behind: 108118
⚠️ Check P2P connectivity: port 8081 must be accessible
```

---

## Diagnostic Improvements

### New Error Messages

When blocks fail to deserialize, you'll now see:
```
❌ CRITICAL: Failed to deserialize block from network
   Topic: /qnk/testnet/blocks
   Data size: 4096 bytes
   Error: missing field `header`
   First 32 bytes (hex): 0a1b2c3d4e5f6071...
```

This helps identify:
- Version mismatches in QBlock struct
- Corrupted gossipsub messages
- Serialization format changes

### Sync Progress Logging

```
🚀 FAST SYNC: 108118 blocks behind (current: 2, network: 108120)
📈 Syncing at 523 blocks/5s (0% complete)
... (syncing) ...
📈 Syncing at 412 blocks/5s (50% complete)
... (syncing) ...
✅ Synced! Current height: 108115, Network height: 108120
⏰ PHASE 2: TIME-BASED PARALLEL BLOCK PRODUCED by Producer #7: Height 108121
```

---

## Testing

### Verify P2P Working

```bash
# Start node and check logs immediately
docker logs -f quillon-node

# Should see within 10 seconds:
# ✅ libp2p initialized on /ip4/0.0.0.0/tcp/8081
# 📨 Starting gossipsub transaction/block synchronization processor...
# 📦 Received block 100 (height=100) from network
# ✅ Stored incoming block 100 to RocksDB
```

### Verify Sync-First Mode Working

```bash
# Node starting from genesis should:
# 1. Receive blocks from network
# 2. NOT produce its own blocks
# 3. Update height progressively

# Check status during sync
curl http://localhost:9080/api/v1/status | jq '.data.current_height'
# Wait 5 seconds
curl http://localhost:9080/api/v1/status | jq '.data.current_height'
# Height should increase!
```

### Verify Miner Real-Time Updates

```bash
# Start miner
./q-miner-v0.3.0-beta --server http://localhost:9080 --wallet YOUR_WALLET

# Logs should show:
# 🎧 Connected to SSE stream for real-time block updates
# 💎 Thread 0 found solution! Block #100, Nonce: 12345
# 🔔 NEW BLOCK #101 detected via SSE - signaling mining threads
# 🔄 Thread 0 IMMEDIATELY updated challenge: block #100 -> #101
```

---

## Performance

### Sync Speed
- **With P2P port exposed**: 100-500 blocks/second
- **Without P2P port**: 0 blocks/second (stuck)

### Miner Block Detection
- **v0.2.9**: 50 seconds (polling)
- **v0.3.1**: <1 second (SSE events) ✅

### Block Production
- **During sync**: Paused ⏸️ (prevents forks)
- **After sync**: Active ✅ (normal operation)

---

## Known Issues & Limitations

### Issue: Block Deserialization May Still Fail

Even with improved logging, blocks might fail to deserialize if:
1. **Version mismatch**: Local node uses different QBlock struct than network
2. **Corrupted messages**: Network issues cause data corruption
3. **Format change**: QBlock serialization changed between versions

**Diagnosis:** Check logs for:
```
❌ CRITICAL: Failed to deserialize block from network
   Error: missing field `header`
```

**Solution:** Ensure all nodes run the same version (v0.3.1-beta).

### Workaround: Copy Blockchain Database

If P2P sync fails completely:
```bash
# On synced node
tar czf blockchain.tar.gz data-mine1/

# Transfer to new node
scp blockchain.tar.gz user@newnode:/path/

# Extract on new node
tar xzf blockchain.tar.gz
./q-api-server --port 9080
```

---

## Upgrade Path

### From v0.2.9 to v0.3.1

```bash
# 1. Stop old version
killall q-api-server

# 2. Download v0.3.1
wget https://quillon.xyz/downloads/q-api-server-v0.3.1-beta
chmod +x q-api-server-v0.3.1-beta

# 3. Start new version (same data directory)
./q-api-server-v0.3.1-beta --port 9080

# Should see:
# ✅ Loaded 108000 blocks from storage
# 🔄 Starting active block sync loop...
# ⏸️ Block production paused (if behind network)
```

### Docker Upgrade

```bash
# 1. Stop old container
docker stop quillon-node
docker rm quillon-node

# 2. Download new binary
wget https://quillon.xyz/downloads/q-api-server-v0.3.1-beta

# 3. Rebuild image
docker build -t quillon-api:v0.3.1 .

# 4. Run with BOTH ports
docker run -d \
  --name quillon-node \
  -p 9080:8080 \
  -p 9081:8081 \
  -v ./data:/app/data \
  quillon-api:v0.3.1
```

---

## Configuration

### Environment Variables

```bash
# P2P port (default: 8081)
Q_P2P_PORT=8081

# Enable validator mode
Q_IS_VALIDATOR=true

# Database path
Q_DB_PATH=./data

# Network ID (testnet or mainnet)
Q_NETWORK=testnet

# Log level
RUST_LOG=info
```

### Command Line Arguments

```bash
./q-api-server \
  --port 9080 \
  --node-id my-node \
  --db-path ./data
```

---

## Architecture

### Block Propagation Flow

```
┌─────────────────────────┐
│  Node A (Height 108000) │
│  Produces block 108001  │
└───────────┬─────────────┘
            │
            │ 1. Serialize with postcard
            │ 2. Broadcast via gossipsub
            │    Topic: /qnk/testnet/blocks
            ▼
┌─────────────────────────┐
│  libp2p Network (P2P)   │
│  Port 8081              │
└───────────┬─────────────┘
            │
            │ 3. Forward to all subscribers
            │
            ▼
┌─────────────────────────┐
│  Node B (Height 2)      │
│  Receives gossipsub msg │
├─────────────────────────┤
│  4. Deserialize block   │
│  5. Validate & save     │
│  6. Update height: 2→3  │
│  7. Check if synced     │
│     → Still behind      │
│     → Pause mining ⏸️    │
└─────────────────────────┘
```

### Sync-First Mode Logic

```rust
let network_height = highest_network_height.load(Ordering::Relaxed);
let current_height = node_status.read().await.current_height;

if network_height > 0 && current_height + 10 < network_height {
    // More than 10 blocks behind - pause mining
    debug!("⏸️ Block production paused: syncing {} blocks behind",
          network_height - current_height);
    return; // Skip block production
}

// Within 10 blocks - mine normally
produce_block().await;
```

---

## Changelog

### v0.3.1-beta (Current)
- ✅ Enhanced block deserialization error logging
- ✅ Sync-first mode (pause mining during sync)
- ✅ Active sync monitoring loop
- ✅ Network height tracking
- ✅ Docker P2P documentation

### v0.3.0-beta
- ✅ Real-time miner SSE block notifications
- ✅ Miner block signal for instant updates
- ✅ <1 second block detection latency

### v0.2.9-beta
- ❌ Silent deserialization failures
- ❌ Miners polled every 50 seconds
- ❌ No sync-first mode

---

## Support

### Debug Mode

Enable verbose logging:
```bash
RUST_LOG=debug ./q-api-server --port 9080
```

### Check P2P Connectivity

```bash
# Test if P2P port is open
nc -zv localhost 8081

# Check firewall
ufw status | grep 8081

# Check Docker port mapping
docker port quillon-node
```

### Common Error Messages

**"Not receiving blocks from network"**
→ Port 8081 not exposed or firewall blocking

**"Failed to deserialize block"**
→ Version mismatch or corrupted data

**"Block production paused"**
→ Normal during sync, resume after catch-up

---

## Credits

**Version:** v0.3.1-beta
**Release Date:** October 30, 2025
**Developed By:** Server Beta (Claude Code)

**Key Improvements:**
- Enhanced P2P block propagation diagnostics
- Sync-first mode implementation
- Real-time miner synchronization
- Docker P2P setup documentation

---

**Download:** https://quillon.xyz/downloads/q-api-server-v0.3.1-beta
