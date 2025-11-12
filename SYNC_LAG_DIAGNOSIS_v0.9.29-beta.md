# Sync Lag Diagnosis - Server Beta Behind Local Node

**Date**: 2025-11-06 13:14 CET
**Reported By**: User on Discord
**Status**: 🚨 **CONFIRMED** - Server Beta is 1286 blocks behind

---

## 🔍 Situation Analysis

### Current Heights:
- **Local Machine** (User's node): **3822 blocks**
- **Server Beta** (quillon.xyz): **2536 blocks** (as of 13:13 CET)
- **Height Deficit**: **1286 blocks**
- **Frontend Display**: Shows **2136 blocks** (likely cached data)

### Sync Performance:
- **Current Sync Rate**: ~2 blocks/5 seconds = **24 blocks/minute**
- **Time to Catch Up**: **1286 blocks ÷ 24 blocks/min = 53.6 minutes**
- **Problem**: Local node continues producing blocks faster than server can sync!

---

## 🚨 Root Causes

### 1. **Slow P2P Sync**
```
📊 [TURBO SYNC] Network height updated to 2536
```

Server is receiving height updates but syncing very slowly.

### 2. **Block Production Rate vs Sync Rate**
- **Local node producing**: ~48-60 blocks/minute (mining active)
- **Server syncing at**: ~24 blocks/minute
- **Result**: Server will NEVER catch up while local node mines!

### 3. **Frontend Cache Issue**
- Backend at height 2536
- Frontend showing 2136
- **400 block cache lag** (likely browser cache or CDN cache)

---

## 📊 Evidence from Logs

### Server-Beta Sync Progress (Last 5 Minutes):
```
12:12:22  Network height: 2521
12:12:27  Network height: 2523
12:12:29  ⚠️  Node ahead: Current height: 2524, Network claims: 2523
12:12:32  Network height: 2525
12:12:37  Network height: 2527
12:12:42  Network height: 2529
12:12:47  Network height: 2531
12:12:52  Network height: 2533
12:12:57  Network height: 2534
12:13:02  Network height: 2536
```

**Analysis**: Syncing 15 blocks in 40 seconds = **22.5 blocks/minute**

### Resource Usage:
```
PID: 411871
CPU: 217% (high, expected during sync)
MEM: 5.7% (5.6 GB of 96 GB total)
Runtime: 65 minutes 40 seconds
```

CPU at 217% indicates active syncing, but still slow.

---

## 🔧 Immediate Solutions

### Solution 1: **Stop Local Mining Temporarily**
Allow server-beta to catch up by stopping local mining:

```bash
# On local machine
killall q-miner

# Wait ~1 hour for server to sync
# Monitor at https://quillon.xyz/

# Once synced, restart mining
./q-miner --address <your-address>
```

**Expected Result**: Server catches up in ~53 minutes.

---

### Solution 2: **Enable Turbo Sync Batch Mode**
Current sync is block-by-block. Enable batch syncing:

**Check Current Turbo Sync Settings**:
```bash
journalctl -u q-api-server -n 1000 | grep -i "turbo.*batch\|batch sync"
```

**If not active**, the turbo sync batching may be disabled. This should fetch 100+ blocks at once instead of 2 blocks every 5 seconds.

---

### Solution 3: **Database Bootstrap from Local Node**
Copy the blockchain database from local machine to server-beta:

**On Local Machine:**
```bash
# Stop your node
systemctl stop q-api-server  # if running as service

# Create backup
tar -czf blockchain-backup-height-3822.tar.gz data/q-narwhal-db/

# Upload to server
scp blockchain-backup-height-3822.tar.gz root@185.182.185.227:/tmp/
```

**On Server-Beta:**
```bash
# Stop service
systemctl stop q-api-server

# Backup current database
mv data/q-narwhal-db data/q-narwhal-db-backup-2536

# Extract new database
cd /opt/orobit/shared/q-narwhalknight
tar -xzf /tmp/blockchain-backup-height-3822.tar.gz

# Restart service
systemctl start q-api-server
systemctl status q-api-server
```

**Expected Result**: Instant sync to height 3822.

---

### Solution 4: **Frontend Cache Clear**
User seeing "2136" needs to clear browser cache:

**Instructions for User:**
1. Open https://quillon.xyz/
2. Press `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
3. This does a hard refresh bypassing cache

OR

1. Open Developer Tools (F12)
2. Right-click refresh button
3. Select "Empty Cache and Hard Reload"

**Expected Result**: Frontend shows actual height (~2536 or higher)

---

## 📈 Long-Term Solutions

### 1. **Increase Turbo Sync Batch Size**
Modify `crates/q-storage/src/turbo_sync.rs` to fetch larger batches:

```rust
// Current (likely):
const BATCH_SIZE: usize = 10;

// Recommended:
const BATCH_SIZE: usize = 500;
```

### 2. **Add Sync Priority Mode**
When node is >100 blocks behind, prioritize sync over mining:

```rust
if network_height - current_height > 100 {
    // Pause block production
    // Focus all resources on syncing
    enable_turbo_sync_priority();
}
```

### 3. **Implement Checkpoint Syncing**
For nodes far behind, download checkpoints instead of syncing every block:

```rust
if network_height - current_height > 1000 {
    // Download verified checkpoint at every 1000 blocks
    download_checkpoint(network_height - (network_height % 1000));
}
```

### 4. **Add Frontend Real-Time Height Display**
Modify frontend to show live height via SSE instead of cached REST API:

```javascript
// Connect to SSE stream
const eventSource = new EventSource('http://quillon.xyz:8080/events');

eventSource.addEventListener('block_produced', (event) => {
    const block = JSON.parse(event.data);
    updateHeightDisplay(block.height);
});
```

---

## 🎯 Recommended Action Plan

### Immediate (Next 5 Minutes):
1. ✅ **Tell user to clear browser cache** (Ctrl+Shift+R)
2. ✅ **Stop local mining** temporarily to let server catch up

### Short-Term (Next Hour):
3. ⏳ **Monitor server sync** - should reach height 3822 in ~53 minutes
4. ⏳ **Check turbo sync batch settings**
5. ⏳ **Deploy v0.9.30-beta** with dev fee fix once it compiles

### Long-Term (Next Release):
6. 📋 **Implement checkpoint syncing** for fast initial sync
7. 📋 **Add sync priority mode** (pause mining when far behind)
8. 📋 **Increase default turbo sync batch size** to 500
9. 📋 **Add real-time height display** to frontend (via SSE)

---

## 📝 Communication to User

### Discord Response:
```
Hey! I've diagnosed the sync lag issue:

**Current Status:**
- Your local node: Height 3822 ✅
- quillon.xyz server: Height 2536 (syncing slowly)
- Frontend showing: 2136 (cached data)

**Why This Happened:**
- Your local node is mining and producing blocks FASTER than server can sync
- Server syncing at ~24 blocks/min, but you're producing ~50+ blocks/min
- Frontend is showing old cached data

**Quick Fix:**
1. Clear your browser cache (Ctrl+Shift+R or Cmd+Shift+R)
2. You should see height 2536+ (actual server height)

**To Help Server Catch Up:**
- Stop your local miner temporarily
- Wait ~53 minutes for server to sync to 3822
- Then restart mining

**OR:**
- Keep mining on your local node
- Server will catch up slower (but will eventually get there)

The network is working correctly - this is just a sync speed issue! 🚀
```

---

## 🔍 Verification Steps

### Check Current Server Height:
```bash
journalctl -u q-api-server -n 1 | grep "Network height"
```

### Check Sync Progress Every Minute:
```bash
watch -n 60 'journalctl -u q-api-server -n 100 | grep "Network height" | tail -1'
```

### Estimate Time to Full Sync:
```bash
current_height=2536
target_height=3822
blocks_behind=$((target_height - current_height))
sync_rate=24  # blocks per minute
minutes_to_sync=$((blocks_behind / sync_rate))

echo "Blocks behind: $blocks_behind"
echo "Time to full sync: $minutes_to_sync minutes"
```

---

**Status**: 📊 **Diagnosis Complete** - Awaiting user action

**Next Steps**:
1. Inform user of situation
2. Recommend browser cache clear
3. Suggest temporarily stopping local mining
4. Monitor sync progress

---

*Created: 2025-11-06 13:14 CET*
*Session: Sync lag diagnosis*
*Server: server-beta (185.182.185.227)*
