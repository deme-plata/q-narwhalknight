# Server Alpha Sync Diagnostic Guide

## Problem Statement

Server Alpha (syncing node) is NOT sending BlockPack requests to Server Beta (bootstrap), despite:
- ✅ v0.9.75-beta deployed on Server Beta
- ✅ BlockPackCodec initialized on Server Beta
- ✅ Server Beta ready to respond
- ❌ ZERO BlockPack requests received in 10+ minutes

## Diagnostic Commands for Server Alpha

Run these commands on **Server Alpha** (the syncing node) to diagnose the issue:

### 1. Verify v0.9.75-beta is Running

```bash
# Check if process is running
ps aux | grep q-api-server

# Check binary modification date (should be Nov 9, 2025 14:06+)
ls -lh ./q-api-server

# Check version string in logs (look for v0.9.75-beta indicators)
grep -i "optimistic\|eligible peers\|FAST SYNC DEBUG" <your-log-file>
```

**Expected in v0.9.75-beta logs**:
```
🚀 [FAST SYNC] Activating TURBO MODE with optimistic peer testing
📊 [FAST SYNC DEBUG] Peer registry: X total, Y blacklisted, Z eligible
```

**If you DON'T see "optimistic peer testing" in logs** → You're running old version!

### 2. Check Sync Loop Activation

```bash
# Check if sync loop is detecting height difference
grep -E "(blocks behind|network height|TURBO|FAST SYNC)" <your-log-file> | tail -20
```

**Expected**:
```
📊 [TURBO SYNC] Network height updated to 9080+
🚀 [FAST SYNC] Activating TURBO MODE
```

**If you DON'T see these** → Sync loop not detecting network height properly

### 3. Check Peer Discovery

```bash
# Check if Server Beta is discovered as a peer
grep -E "(Registered peer|discovered)" <your-log-file> | grep "12D3KooWRX3GGK9F"
```

**Expected**: Peer `12D3KooWRX3GGK9F` (Server Beta's bootstrap ID) should be registered

**If peer not found** → P2P connection issue

### 4. Check for BlockPack Request Sending

```bash
# Check if requests are being SENT
grep -E "(request_blocks_from_peer|FAST SYNC.*Requesting|BlockPack request sent)" <your-log-file> | tail -20
```

**Expected**:
```
📥 [FAST SYNC #1] Requesting 2000 blocks (266-2265) from peer
✅ [FAST SYNC #1] BlockPack request sent (expecting 2000 blocks)
```

**If you DON'T see "BlockPack request sent"** → Requests not being sent!

### 5. Check for Errors

```bash
# Check for any sync-related errors
grep -iE "(error|failed|timeout).*sync" <your-log-file> | tail -20

# Check for libp2p errors
grep -iE "(error|failed).*p2p" <your-log-file> | tail -20
```

## Common Issues and Solutions

### Issue 1: Old Binary Running

**Symptoms**:
- No "optimistic peer testing" in logs
- No "📊 [FAST SYNC DEBUG]" messages
- Still seeing HTTP fallback

**Solution**:
```bash
# Kill old process
pkill q-api-server

# Verify it's killed
ps aux | grep q-api-server

# Download v0.9.75-beta
wget http://185.182.185.227:80/downloads/q-api-server-v0.9.75-beta -O q-api-server
chmod +x q-api-server

# Start new version
./q-api-server --port 8080 2>&1 | tee server-alpha.log
```

### Issue 2: Sync Loop Not Triggering

**Symptoms**:
- Peer discovered but no sync requests sent
- Height difference detected but no FAST SYNC activation

**Possible causes**:
1. Current height too close to network height (difference < threshold)
2. Sync loop disabled or stuck
3. Storage engine issues

**Solution**:
Check logs for:
```
📊 [TURBO SYNC] Network height updated to X
```

If network height is NOT being updated → Peer height announcements not being processed

### Issue 3: P2P Connection Not Established

**Symptoms**:
- Server Beta's peer ID (`12D3KooWRX3GGK9F`) not in peer list
- No peer heights being registered

**Solution**:
```bash
# Check bootstrap peer configuration
grep -i "bootstrap" <your-log-file>

# Should see:
# "Connected to bootstrap peer 12D3KooWRX3GGK9F"
```

**If bootstrap not connected**:
1. Check firewall (port 9001 must be open)
2. Check bootstrap address in code
3. Check network connectivity to 185.182.185.227:9001

### Issue 4: Sync Threshold Not Met

**Symptoms**:
- Peer discovered
- Height detected
- But no sync requests sent

**Cause**: Server Alpha's height might be too close to network height

**Check**:
```bash
grep "blocks behind" <your-log-file>
```

Sync only triggers if **blocks_behind > 5**, so if Server Alpha is within 5 blocks of network height, it won't request blocks!

## Expected Full Sync Flow (v0.9.75-beta)

Here's what you should see in Server Alpha's logs when sync works correctly:

```
1. 📡 [TURBO SYNC] Peer 12D3KooWRX3GGK9F has height 9080
   📊 [TURBO SYNC] Network height updated to 9080

2. 🚀 [FAST SYNC] Activating TURBO MODE with optimistic peer testing
   📊 [FAST SYNC DEBUG] Peer registry: 2 total, 0 blacklisted, 2 eligible

3. Peer #1: 12D3KooWRX3GGK9F (height: 9080)

4. 🚀 [FAST SYNC] Sending 1 parallel BlockPack requests (chunk size: 2000)

5. 📥 [FAST SYNC #1] Requesting 2000 blocks (266-2265) from peer 12D3KooWRX3GGK9F
   ✅ [FAST SYNC #1] BlockPack request sent (expecting 2000 blocks)

6. [10 seconds later...]
   📨 [BLOCK-PACK] Received block pack response: 2000 blocks (heights 266-2265)
   ✅ [BLOCK-PACK] Forwarded 2000 blocks to consensus for validation

7. ✅ [FAST SYNC] Received 2000 blocks! (height: 266 → 2266)
   ⚡ [TURBO MODE] Speed: 2000 blocks/10s = 12000/min

8. [Repeat steps 4-7 until fully synced]
```

## If Nothing Works: Deep Dive

### Check Server Alpha's Network Manager

The issue might be in how Server Alpha's network manager is initialized. Check logs for:

```bash
# Network manager initialization
grep "Block sync request-response protocol initialized" <your-log-file>
```

Should see:
```
🔗 Block sync request-response protocol initialized (BlockPackCodec)
```

**If this is missing** → BlockPackCodec not initialized on Server Alpha!

### Check Request-Response Protocol

```bash
# Check if requests are even attempted
grep -i "send_request\|request_blocks" <your-log-file>
```

Should see attempts to call `request_blocks_from_peer()`

### Last Resort: Enable Debug Logging

Add to Server Alpha startup:
```bash
RUST_LOG=q_network=debug,q_api_server=debug ./q-api-server --port 8080
```

This will show ALL P2P and sync activity in detail.

## Reporting Back

Please provide the output of these commands:

1. **Version check**:
   ```bash
   ls -lh ./q-api-server
   grep "optimistic" <your-log-file> | head -5
   ```

2. **Peer status**:
   ```bash
   grep "Registered peer" <your-log-file> | tail -10
   grep "Network height updated" <your-log-file> | tail -5
   ```

3. **Sync attempts**:
   ```bash
   grep -E "(FAST SYNC|BlockPack request sent)" <your-log-file> | tail -10
   ```

4. **Errors**:
   ```bash
   grep -iE "(error|failed)" <your-log-file> | tail -20
   ```

This will help us pinpoint exactly where the sync process is failing!
