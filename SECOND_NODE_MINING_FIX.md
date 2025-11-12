# Second Node Mining Connection Fix

**Date**: 2025-10-30
**Issue**: Miner cannot connect to second node at 161.35.219.10:8080
**Status**: ⏳ NEEDS USER ACTION

## Problem Summary

The q-miner.exe cannot connect to the second node at `161.35.219.10:8080`:

```
ERROR q_miner: ❌ Thread 2 failed to fetch initial challenge: error sending request for url (http://161.35.219.10:8080/api/v1/mining/challenge)
ERROR q_miner:    Make sure q-api-server is running on http://161.35.219.10:8080
WARN q_miner: SSE stream error: http error: error trying to connect: tcp connect error
```

**Hash Rate**: 0.00 H/s (stuck, no mining happening)

## Root Cause Analysis

Tested connection from bootstrap server (185.182.185.227):

```bash
curl -v http://161.35.219.10:8080/api/v1/mining/challenge
```

**Result**: Connection timeout after 45+ seconds - **the second node is NOT responding**

### Possible Causes:

1. ❌ **Second node not running** - Docker container stopped or crashed
2. ❌ **Second node running OLD version** - Still on older version, not v0.2.9-beta
3. ❌ **Port 8080 not exposed** - Docker container not exposing port correctly
4. ❌ **Firewall blocking** - UFW or cloud firewall blocking port 8080
5. ❌ **API server crashed** - Process started but crashed during initialization

## Solution Steps

### Step 1: SSH to Second Node

```bash
ssh root@161.35.219.10
```

### Step 2: Check Docker Container Status

```bash
# Check if container is running
docker ps -a | grep q-test-node

# If stopped, check logs for crash reason:
docker logs q-test-node --tail 100
```

### Step 3: Download v0.2.9-beta Binary

The new binary with all fixes is available:

```bash
# Download the latest version
wget https://quillon.xyz/downloads/q-api-server-v0.2.9-beta -O /root/q-api-server-v0.2.9-beta

# Make executable
chmod +x /root/q-api-server-v0.2.9-beta
```

### Step 4: Restart Second Node with New Binary

```bash
# Stop existing container
docker stop q-test-node
docker rm q-test-node

# Run with new v0.2.9-beta binary
docker run -d \
  --name q-test-node \
  --network host \
  -v /root/data-node2:/data \
  -v /root/q-api-server-v0.2.9-beta:/usr/local/bin/q-api-server:ro \
  -e Q_DB_PATH=/data \
  -e Q_P2P_PORT=9001 \
  -e RUST_LOG=info \
  -e BOOTSTRAP_PEERS="/ip4/185.182.185.227/tcp/9000/p2p/12D3KooWGmYmguqn7YpvfNngw8GTbF7pEe5CWGvvgJRaVH2F3htz" \
  ubuntu:22.04 \
  /usr/local/bin/q-api-server --port 8080 --node-id node2
```

### Step 5: Verify Node Started Successfully

```bash
# Check logs
docker logs -f q-test-node

# Look for these success messages:
# ✅ REST API server started on 0.0.0.0:8080
# ✅ Mining endpoint initialized
# 🌐 Connected to bootstrap node
# 🤖 AI model loaded: Mistral-7B
```

### Step 6: Test Mining API Endpoint

From your local machine or bootstrap server:

```bash
# Test mining challenge endpoint
curl http://161.35.219.10:8080/api/v1/mining/challenge

# Should return JSON with challenge data:
# {"challenge":"...","difficulty":4,"timestamp":...}
```

### Step 7: Test Miner Connection

Run your miner again:

```bash
q-miner.exe --api-url http://161.35.219.10:8080 --threads 4
```

**Expected output**:
```
✅ Connected to API server at http://161.35.219.10:8080
🚀 Mining with 4 threads
⛏️  Hash rate: 125.43 H/s
✅ Submitted proof for challenge ...
```

## Verification Checklist

- [ ] SSH access to 161.35.219.10 confirmed
- [ ] Docker container running: `docker ps | grep q-test-node`
- [ ] Port 8080 accessible: `curl http://161.35.219.10:8080/health`
- [ ] Mining endpoint working: `curl http://161.35.219.10:8080/api/v1/mining/challenge`
- [ ] Node running v0.2.9-beta (check logs for version)
- [ ] Miner successfully connects and mines
- [ ] Hash rate > 0.00 H/s

## Firewall Configuration (if needed)

If port 8080 is blocked by firewall:

```bash
# UFW (Ubuntu Firewall)
ufw allow 8080/tcp
ufw reload

# Or cloud provider firewall (DigitalOcean example)
# Go to Networking > Firewalls
# Add inbound rule: TCP port 8080 from any source
```

## Expected Results After Fix

1. ✅ Second node running v0.2.9-beta
2. ✅ Mining API endpoint responds with challenge data
3. ✅ Miner connects successfully
4. ✅ Hash rate > 0 H/s
5. ✅ Mining rewards being earned and saved (no balance loss)
6. ✅ Distributed AI working between both nodes

## Critical Fixes in v0.2.9-beta

This version includes CRITICAL fixes that must be deployed:

### 1. Balance Loss Race Condition (FIXED)
- Users were losing ~50 QNK on every restart
- Fixed by persisting balances BEFORE releasing locks
- **Impact**: No more coin loss on restart

### 2. Loan System Compilation Errors (FIXED)
- 3 compilation errors resolved
- Loan applications now persist correctly
- Network broadcasting working

### 3. Version Compatibility
- Both nodes MUST run same version for proper coordination
- Bootstrap node already upgraded to v0.2.9-beta
- Second node needs upgrade

## Next Steps

1. **SSH to 161.35.219.10** and follow Steps 1-7 above
2. **Restart second node** with v0.2.9-beta binary
3. **Test mining connection** from your miner
4. **Monitor both nodes** for proper coordination

---

## Support Commands

### Check Second Node Status (from bootstrap)
```bash
curl -s http://161.35.219.10:8080/api/v1/node/info | jq '.'
curl -s http://161.35.219.10:8080/health
```

### Monitor Second Node Logs
```bash
docker logs -f q-test-node | grep -E "(✅|❌|⛏️|Mining)"
```

### Check Network Connectivity
```bash
ping -c 3 161.35.219.10
telnet 161.35.219.10 8080  # Should connect if port is open
```

---

**IMPORTANT**: The second node MUST be restarted with v0.2.9-beta for mining to work properly. The old version is incompatible with the new mining API changes.
