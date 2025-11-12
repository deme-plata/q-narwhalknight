# Server Alpha: Localhost Mining Issue - Fix Plan

**Date**: 2025-11-07
**Server**: Server Alpha (161.35.219.10)
**Issue**: Mining rewards going to localhost:8091 instead of syncing with bootstrap server (185.182.185.227:8080)
**Status**: ⚠️ **NETWORK SPLIT - Mining to separate blockchain**

---

## Problem Summary

**User Report**:
- Frontend connected to bootstrap server (185.182.185.227:8080)
- Miner sending rewards to localhost server (localhost:8091 or localhost:8090)
- **Result**: Mining rewards not visible in frontend because they're on different blockchain instances

**Root Cause**: Network split - two separate blockchain instances:
1. **Bootstrap Server (185.182.185.227)**: Main network blockchain
2. **Localhost Server (161.35.219.10)**: Isolated blockchain with mining rewards

---

## Current Architecture Problem

```
┌──────────────────────────────────────────────────────────────┐
│  Server Alpha (161.35.219.10)                                │
│                                                                │
│  ┌─────────────────┐           ┌─────────────────┐           │
│  │  Frontend GUI   │◄──────────│  Bootstrap API  │           │
│  │  (Browser)      │  HTTP     │  185.182...227  │           │
│  │                 │  :8080    │  :8080          │           │
│  └─────────────────┘           └─────────────────┘           │
│                                                                │
│  ❌ SEPARATE NETWORK SPLIT ❌                                │
│                                                                │
│  ┌─────────────────┐           ┌─────────────────┐           │
│  │  Miner Process  │──────────►│  Localhost API  │           │
│  │  q-miner        │  Mining   │  localhost:8091 │           │
│  │                 │  Rewards  │  (or :8090)     │           │
│  └─────────────────┘           └─────────────────┘           │
│                                                                │
└──────────────────────────────────────────────────────────────┘

Result: Mining rewards exist on localhost blockchain,
        but frontend shows balances from bootstrap blockchain
```

---

## Two Solutions

### Solution 1: Mine Directly to Bootstrap Server (RECOMMENDED)

**Concept**: Point miner to bootstrap server instead of localhost

**Advantages**:
- ✅ Simple configuration change
- ✅ No local node needed
- ✅ Immediate sync with main network
- ✅ Frontend works correctly

**Disadvantages**:
- ❌ Depends on bootstrap server availability
- ❌ Network latency affects mining

**Implementation**:
```bash
# On Server Alpha (161.35.219.10)
# Stop local miner if running
pkill -f q-miner

# Start miner pointing to bootstrap server
./q-miner-linux-x64 \
  --api-url http://185.182.185.227:8080 \
  --wallet-address qnke9578fdf77fa62a961af97636ffb9d1d1885d6a9831bb53f4519dbf97c01ebee \
  --threads 4

# Or if using environment variables:
export Q_API_URL=http://185.182.185.227:8080
export Q_WALLET_ADDRESS=qnke9578fdf77fa62a961af97636ffb9d1d1885d6a9831bb53f4519dbf97c01ebee
./q-miner-linux-x64
```

---

### Solution 2: Sync Localhost Node with Bootstrap (COMPLEX)

**Concept**: Run local node on Server Alpha that syncs with bootstrap server

**Advantages**:
- ✅ Full decentralization
- ✅ Local blockchain copy
- ✅ Works offline for mining

**Disadvantages**:
- ❌ Requires fixing libp2p connectivity (port 9001 issue from SERVER_ALPHA_LIBP2P_FAILURE_DIAGNOSIS.md)
- ❌ Requires blockchain sync
- ❌ Requires fork resolution if chains diverged
- ❌ More complex setup

**Implementation**:
```bash
# Step 1: Fix Docker container to expose P2P port
docker rm q-v0936-beta  # Stop current container

docker run -d \
  --name q-v0937-beta-synced \
  -p 8090:8080 \
  -p 9001:9001 \  # ← Critical: Expose P2P port!
  -e Q_HOST=0.0.0.0 \
  -e Q_P2P_PORT=9001 \
  -e Q_BOOTSTRAP_PEER=/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWLQok4vAPYLWSbUuj4LY4dLYcaJCeMp12GaEpDNQ6uJGJ \
  -v /opt/orobit/data:/data \
  ubuntu:22.04 ./q-api-server-v0.9.36-beta --port 8080

# Step 2: Wait for sync (monitor with enhanced logging from v0.9.37-beta)
docker logs q-v0937-beta-synced -f | grep -E "SYNC|libp2p|TURBO SYNC"

# Step 3: Point miner to localhost AFTER sync completes
./q-miner-linux-x64 \
  --api-url http://localhost:8090 \
  --wallet-address qnke9578fdf...
```

---

## Recommended Quick Fix (Solution 1)

### Step-by-Step Instructions

**1. Verify Bootstrap Server Connectivity**
```bash
# On Server Alpha, test connection
curl http://185.182.185.227:8080/api/v1/node/info

# Expected output:
{
  "current_height": 10700+,
  "node_id": "12D3KooWLQok4vAPYLWSbUuj4LY4dLYcaJCeMp12GaEpDNQ6uJGJ",
  ...
}
```

**2. Stop Localhost Miner**
```bash
# Find miner process
ps aux | grep q-miner

# Kill it
pkill -f q-miner
# Or if specific PID:
kill <PID>
```

**3. Download Latest Miner Binary**
```bash
cd /tmp
wget https://quillon.xyz/downloads/q-miner-linux-x64
chmod +x q-miner-linux-x64
```

**4. Start Miner Pointing to Bootstrap**
```bash
./q-miner-linux-x64 \
  --api-url http://185.182.185.227:8080 \
  --wallet-address qnke9578fdf77fa62a961af97636ffb9d1d1885d6a9831bb53f4519dbf97c01ebee \
  --threads 4 \
  > /tmp/miner.log 2>&1 &

echo "Miner started with PID: $!"
```

**5. Monitor Mining Activity**
```bash
# Check miner logs
tail -f /tmp/miner.log

# Expected output:
⛏️  Mining started: API=http://185.182.185.227:8080
💰 Wallet: qnke9578fdf...
🔨 Threads: 4
✅ Mining block at height 10701...
```

**6. Verify Balance Updates in Frontend**
```bash
# Wait 1-2 minutes for a block to be mined
# Then check wallet balance via API:
curl http://185.182.185.227:8080/api/v1/wallet/qnke9578fdf77fa62a961af97636ffb9d1d1885d6a9831bb53f4519dbf97c01ebee/balance

# Expected: Balance should increase after mining
```

---

## Configuration File for Persistent Mining

Create `/opt/orobit/miner-config.env`:
```bash
# Q-NarwhalKnight Miner Configuration
# Server Alpha mining to Bootstrap Server

# API endpoint (bootstrap server)
Q_API_URL=http://185.182.185.227:8080

# Wallet address for mining rewards
Q_WALLET_ADDRESS=qnke9578fdf77fa62a961af97636ffb9d1d1885d6a9831bb53f4519dbf97c01ebee

# Mining threads (adjust based on CPU cores)
Q_MINING_THREADS=4

# Logging
RUST_LOG=info
```

**Start miner with config**:
```bash
source /opt/orobit/miner-config.env
./q-miner-linux-x64 \
  --api-url "$Q_API_URL" \
  --wallet-address "$Q_WALLET_ADDRESS" \
  --threads "$Q_MINING_THREADS"
```

---

## Systemd Service for Automatic Miner Restart

Create `/etc/systemd/system/q-miner.service`:
```ini
[Unit]
Description=Q-NarwhalKnight Miner (Server Alpha)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/orobit
EnvironmentFile=/opt/orobit/miner-config.env
ExecStart=/opt/orobit/q-miner-linux-x64 \
  --api-url ${Q_API_URL} \
  --wallet-address ${Q_WALLET_ADDRESS} \
  --threads ${Q_MINING_THREADS}
Restart=always
RestartSec=10s
StandardOutput=append:/var/log/q-miner.log
StandardError=append:/var/log/q-miner.log

[Install]
WantedBy=multi-user.target
```

**Enable and start service**:
```bash
# Copy miner binary
cp q-miner-linux-x64 /opt/orobit/

# Create config file
cat > /opt/orobit/miner-config.env << 'EOF'
Q_API_URL=http://185.182.185.227:8080
Q_WALLET_ADDRESS=qnke9578fdf77fa62a961af97636ffb9d1d1885d6a9831bb53f4519dbf97c01ebee
Q_MINING_THREADS=4
RUST_LOG=info
EOF

# Install and start service
systemctl daemon-reload
systemctl enable q-miner.service
systemctl start q-miner.service

# Check status
systemctl status q-miner.service
journalctl -u q-miner.service -f
```

---

## Verification Checklist

After implementing the fix:

- [ ] **Miner connected to bootstrap**: Check logs for "API=http://185.182.185.227:8080"
- [ ] **Mining blocks successfully**: Look for "✅ Mining block at height..." messages
- [ ] **Balance increasing**: Frontend shows wallet balance growing
- [ ] **No localhost references**: Miner NOT connecting to localhost:8090 or localhost:8091
- [ ] **Network height matches**: Miner mining at same height as bootstrap server

**Test Commands**:
```bash
# 1. Check miner is running
ps aux | grep q-miner

# 2. Check miner logs
tail -100 /var/log/q-miner.log | grep -E "Mining|Balance|API"

# 3. Check wallet balance on bootstrap
curl http://185.182.185.227:8080/api/v1/wallet/qnke9578fdf77fa62a961af97636ffb9d1d1885d6a9831bb53f4519dbf97c01ebee/balance

# 4. Check network height
curl http://185.182.185.227:8080/api/v1/node/info | jq '.current_height'
```

---

## Troubleshooting

### Issue: Miner shows "Connection refused"
```
❌ Error: Connection refused (os error 111)
```

**Solution**: Verify bootstrap server is running and accessible
```bash
# Test from Server Alpha
curl http://185.182.185.227:8080/api/v1/node/info

# If this fails, check:
# 1. Is bootstrap server running?
systemctl status q-api-server  # On bootstrap server

# 2. Is firewall blocking?
ufw status  # On bootstrap server

# 3. Is nginx configured correctly?
nginx -t && systemctl status nginx  # On bootstrap server
```

### Issue: Mining rewards still go to localhost

**Solution**: Verify miner command line
```bash
# Check running miner process
ps aux | grep q-miner

# Should show:
--api-url http://185.182.185.227:8080

# NOT:
--api-url http://localhost:8090
--api-url http://localhost:8091
--api-url http://127.0.0.1:8090
```

### Issue: Balance not updating in frontend

**Possible causes**:
1. **Frontend cached**: Hard refresh browser (Ctrl+F5)
2. **Mining to wrong wallet**: Check wallet address in miner config
3. **Network split still present**: Verify no localhost node running

**Debug**:
```bash
# Check if localhost node is still running
netstat -tulpn | grep -E "8090|8091"

# If shows q-api-server:
pkill -f q-api-server  # Kill localhost node

# Restart miner
systemctl restart q-miner.service
```

---

## Long-term Architecture (Future)

Once libp2p connectivity is fixed (port 9001 exposed), transition to:

```
┌──────────────────────────────────────────────────────────────┐
│  Server Alpha (161.35.219.10)                                │
│                                                                │
│  ┌─────────────────┐           ┌─────────────────┐           │
│  │  Frontend GUI   │◄──────────│  Local API      │           │
│  │  (Browser)      │  HTTP     │  :8090          │           │
│  │                 │  :8090    │                 │           │
│  └─────────────────┘           └─────┬───────────┘           │
│                                       │                        │
│                                       │ P2P Sync              │
│                                       │ (libp2p               │
│                                       │  port 9001)           │
│  ┌─────────────────┐                 │                        │
│  │  Miner Process  │──────────────────┘                       │
│  │  q-miner        │  Mining Rewards                          │
│  │                 │  to local chain                          │
│  └─────────────────┘  (synced with                            │
│                        bootstrap)                             │
└──────────────────────────────────────────────────────────────┘
         │
         │ P2P gossipsub (:9001)
         │ TURBO SYNC enabled
         ▼
┌─────────────────────────────────┐
│  Bootstrap Server               │
│  (185.182.185.227)              │
│  Main Network Blockchain        │
└─────────────────────────────────┘
```

**Benefits of future architecture**:
- Full decentralization with local blockchain copy
- P2P sync for redundancy
- Offline mining capability
- Network resilience

**Prerequisites**:
1. Fix Docker port mapping (expose 9001)
2. Enable libp2p network manager
3. Deploy v0.9.37-beta with sync progress logging
4. Verify TURBO SYNC working (v0.9.36-beta fix)

---

## Status

**Current**: ⚠️ **NETWORK SPLIT** - Mining to localhost, frontend connected to bootstrap
**Quick Fix**: ✅ **READY TO DEPLOY** - Point miner to bootstrap server
**Long-term**: ⏳ **PENDING** - Requires libp2p connectivity fix

---

**Date**: 2025-11-07
**Documented By**: Claude Code (Server Beta)
**Priority**: 🔴 **CRITICAL** - User cannot see mining rewards

