# Server Alpha Deployment Guide - v0.7.4-beta

**Target Server**: Server Alpha (161.35.219.10)
**Current Version**: v0.7.3-beta
**Target Version**: v0.7.4-beta
**Priority**: 🔥 HIGH - Fix messy height issue

---

## Pre-Deployment Status

**Server Beta** (Bootstrap Node):
- Version: v0.7.4-beta ✅
- Height: 2435+ blocks
- Status: Stable, no errors
- Role: Bootstrap peer for network

**Server Alpha** (Testing Node):
- Version: v0.7.3-beta (outdated)
- Status: Experiencing "messy height" issue
- Role: Testing node, miners connect here
- Issue: Height progression shows skips (1829→1831→1834)

---

## Deployment Steps for Server Alpha

### Step 1: Download Binary from Server Beta

On **Server Alpha**:

```bash
# Download v0.7.4-beta from Server Beta
wget https://quillon.xyz/downloads/q-api-server-v0.7.4-beta

# OR download latest (symlink)
wget https://quillon.xyz/downloads/q-api-server-linux-x86_64

# Verify download
ls -lh q-api-server-v0.7.4-beta
# Expected: ~112MB binary
```

### Step 2: Stop Current Service

```bash
# Stop the running q-api-server
killall q-api-server

# OR if running as systemd service
systemctl stop q-api-server

# Verify stopped
ps aux | grep q-api-server
# Should show no running processes
```

### Step 3: Deploy New Binary

```bash
# Make binary executable
chmod +x q-api-server-v0.7.4-beta

# Option A: Replace in-place
cp q-api-server-v0.7.4-beta /opt/orobit/shared/q-narwhalknight/target/release/q-api-server

# Option B: Run directly (testing)
./q-api-server-v0.7.4-beta --port 8080
```

### Step 4: Start Service

```bash
# If running from looksgoodbutslow9.ini location:
cd /opt/orobit/shared/q-narwhalknight
./target/release/q-api-server --port 8080 > /tmp/q-api-server.log 2>&1 &

# Check if running
ps aux | grep q-api-server
```

### Step 5: Verify Connection to Server Beta

Wait ~30 seconds for P2P connection, then check logs:

```bash
tail -100 /tmp/q-api-server.log | grep -E "CONNECTION|peer|Turbo Sync"
```

**Expected Output**:
```
✅ [CONNECTION] Successfully connected to peer: 12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN
📍 [CONNECTION] Endpoint: /ip4/185.182.185.227/tcp/9001
🚀 [TURBO SYNC] Starting sync from height 1831 to 2435
```

---

## Testing the Gap Detection Fix

### Test 1: Monitor Height Progression

Watch height progression in real-time:

```bash
watch -n 1 'curl -s http://localhost:8080/api/v1/status | jq .data.current_height'
```

**Expected Behavior** (v0.7.4-beta with fix):
```
Height: 1831
Height: 1832
Height: 1833
Height: 1834
...
```

**OLD Behavior** (v0.7.3-beta - broken):
```
Height: 1829
Height: 1831  ❌ SKIPPED 1830
Height: 1834  ❌ SKIPPED 1832, 1833
Height: 1838  ❌ SKIPPED 1835-1837
```

### Test 2: Watch for Gap Detection Logs

Monitor logs for gap detection messages:

```bash
tail -f /tmp/q-api-server.log | grep -E "Gap detected|Advanced height"
```

**Expected Logs** (if gaps are detected):
```
⚠️ [BATCH SYNC] Gap detected at height 1850 - requesting missing block from peers
   Current height: 1849, Highest stored: 1852 (gap prevents height advancement)
```

**Expected Logs** (when no gaps):
```
📈 [BATCH SYNC] Advanced height by 10 blocks to 1860 (no gaps) ⚡
```

### Test 3: Verify Mining Challenge Consistency

Check mining challenge matches current height:

```bash
CURRENT=$(curl -s http://localhost:8080/api/v1/status | jq -r '.data.current_height')
CHALLENGE=$(curl -s http://localhost:8080/api/v1/mining/challenge | jq -r '.data.block_height')

echo "Current height: $CURRENT"
echo "Mining challenge: $CHALLENGE"
echo "Match: $([ $CURRENT -eq $CHALLENGE ] && echo 'YES ✅' || echo 'NO ❌')"
```

---

## Success Criteria

### ✅ Server Alpha Must Show:

1. **Sequential Height Progression**
   - No skipped blocks in logs
   - Height advances: 1831→1832→1833→1834 (sequential)

2. **P2P Connection to Server Beta**
   - Connected to bootstrap peer: `12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN`
   - Peer count: 1 or more

3. **Turbo Sync Success**
   - Syncs from current height to Server Beta's height
   - All historical blocks processed correctly
   - Gap detection triggers when needed

4. **Mining Stability**
   - Mining challenges remain consistent
   - Miners don't see confusing height jumps
   - Rewards distributed correctly

---

## Troubleshooting

### Issue 1: Cannot Connect to Server Beta

**Symptoms**:
```
⚠️ Failed to dial bootstrap peer
🔌 Connected peers: 0
```

**Solutions**:
1. Verify Server Beta is running: `ssh 185.182.185.227 'systemctl status q-api-server'`
2. Check firewall on Server Alpha: `iptables -L | grep 9001`
3. Test network connectivity: `telnet 185.182.185.227 9001`

### Issue 2: Height Still Shows Skips

**Symptoms**:
```
Height: 1829 → 1831 (skipped 1830)
```

**Diagnosis**:
1. Check if v0.7.4-beta is actually running:
   ```bash
   strings /proc/$(pgrep q-api-server)/exe | grep "v0.7.4"
   ```

2. Check logs for gap detection messages:
   ```bash
   grep "Gap detected" /tmp/q-api-server.log
   ```

3. If gap detection is NOT triggering, binary may not be v0.7.4-beta

### Issue 3: Turbo Sync Stuck

**Symptoms**:
```
Height: 1831 (not increasing)
```

**Solutions**:
1. Check peer connection:
   ```bash
   curl http://localhost:8080/api/v1/status | jq .data.connected_peers
   ```

2. Restart service to retry sync:
   ```bash
   killall q-api-server
   ./target/release/q-api-server --port 8080 > /tmp/q-api-server.log 2>&1 &
   ```

---

## Post-Deployment Verification

### 1. Check Version (Optional - if version field is implemented)

```bash
curl http://localhost:8080/api/v1/status | jq .data.version
# Expected: "v0.7.4-beta" or similar
```

### 2. Verify Height Matches Server Beta (Eventually)

After Turbo Sync completes:

```bash
# Server Alpha height
curl http://localhost:8080/api/v1/status | jq .data.current_height

# Server Beta height (from Server Beta)
curl http://185.182.185.227:8080/api/v1/status | jq .data.current_height

# Heights should be within ~5 blocks of each other
```

### 3. Monitor for 30 Minutes

```bash
# Watch logs for errors
tail -f /tmp/q-api-server.log | grep -E "ERROR|CRITICAL|panic"

# Watch height progression
watch -n 5 'curl -s http://localhost:8080/api/v1/status | jq .data.current_height'
```

---

## Expected Timeline

**Total Time**: ~15 minutes

- **Download binary**: 1-2 minutes (112MB @ ~1MB/s)
- **Stop/Deploy/Start**: 1 minute
- **P2P Connection**: 30 seconds
- **Turbo Sync**: 5-10 minutes (syncing 600+ blocks)
- **Verification**: 2 minutes

---

## Rollback Procedure

If v0.7.4-beta causes issues:

```bash
# 1. Stop v0.7.4-beta
killall q-api-server

# 2. Restore v0.7.3-beta binary
# (Assuming you backed it up)
cp q-api-server-v0.7.3-beta.backup /opt/orobit/shared/q-narwhalknight/target/release/q-api-server

# 3. Restart service
./target/release/q-api-server --port 8080 > /tmp/q-api-server.log 2>&1 &

# 4. Verify rollback
curl http://localhost:8080/api/v1/status | jq .data.current_height
```

---

## Next Steps After Deployment

### Phase 1: Stability Testing (24 hours)
1. Monitor Server Alpha for crashes or errors
2. Verify height progression remains sequential
3. Test mining on Server Alpha (submit solutions)
4. Verify balances (note: balance consensus issue still exists)

### Phase 2: Multi-Node Testing (This week)
1. Add 3rd node to network
2. Test with multiple miners
3. Simulate network partitions
4. Chaos testing (random restarts)

### Phase 3: Balance Consensus Fix (Next 2-3 weeks)
1. Begin implementing v0.8.0-beta
2. Follow `BALANCE_CONSENSUS_IMPLEMENTATION_PLAN.md`
3. Implement `BalanceConsensusEngine`
4. Test on testnet-phase4 (new network for clean state)

---

## Contact Information

**If issues occur**:
1. Check logs: `tail -100 /tmp/q-api-server.log`
2. Check service status: `ps aux | grep q-api-server`
3. Verify network: `curl http://localhost:8080/api/v1/status`
4. Report to user with full error logs

---

## Key Files

**Binary Locations**:
- Download: `https://quillon.xyz/downloads/q-api-server-v0.7.4-beta`
- Deployed: `/opt/orobit/shared/q-narwhalknight/target/release/q-api-server`
- Log file: `/tmp/q-api-server.log`

**Configuration**:
- Bootstrap peer: `/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWRX3GGK9Fs3iM3BfqYNNJiHBDujac7EHwqWjaK1n1kzPN`
- Network ID: `testnet-phase3`
- API Port: `8080`
- P2P Port: `9001`

---

**Prepared by**: Claude Code (Server Beta)
**Target**: Server Alpha (161.35.219.10)
**Version**: v0.7.4-beta
**Priority**: Fix "messy height" issue
**Expected Outcome**: Sequential height progression, stable mining
