# Server Alpha Phase 5 Upgrade Instructions

**Date**: 2025-11-06 08:15 CET
**Issue**: Server Alpha running Phase 4 (v0.9.18-beta) cannot connect to Server Beta Phase 5 network

---

## 🚨 DIAGNOSIS

**Server Alpha Status**:
- Container: `q-v0922-new`
- Version: `v0.9.18-beta-testnet` ❌ (OLD)
- Network ID: `testnet-phase4` ❌
- Gossipsub topics: `/qnk/testnet-phase4/*` ❌
- Bootstrap peer: Found ✅ (12D3KooWCcbvQBzW4PxWxSnUUb9i1nJCkDHkChYXRL1VBhQ8JkBn)
- Connection status: InsufficientPeers (expected - Phase 4 cannot see Phase 5 peers)

**Server Beta Status**:
- Version: `v0.9.25-beta` ✅
- Network ID: `testnet-phase5` ✅
- Gossipsub topics: `/qnk/testnet-phase5/*` ✅
- Status: Running and producing blocks ✅

**Problem**: Phase 4 and Phase 5 networks are INCOMPATIBLE by design. Different gossipsub topics prevent communication.

---

## ✅ SOLUTION: Download Phase 5 Binary

### Step 1: Download v0.9.25-beta Binary

```bash
# Download Phase 5 binary from Server Beta
wget https://quillon.xyz/downloads/q-api-server-v0.9.25-beta
chmod +x q-api-server-v0.9.25-beta

# Verify SHA256 checksum
wget https://quillon.xyz/downloads/q-api-server-v0.9.25-beta.sha256
sha256sum -c q-api-server-v0.9.25-beta.sha256
# Expected: OK
```

### Step 2: Stop Old Container

```bash
# Stop Phase 4 container
docker stop q-v0922-new
docker rm q-v0922-new
```

### Step 3: Run Phase 5 Binary (Option A - Direct)

```bash
# Run Phase 5 binary directly (no Docker)
./q-api-server-v0.9.25-beta --port 8090 --db-path=./data-phase5

# Expected output:
# - Network ID: testnet-phase5
# - Gossipsub: /qnk/testnet-phase5/*
# - Bootstrap: Connects to Server Beta
# - Sync: From genesis (height 0)
```

### Step 3: Run Phase 5 Binary (Option B - Docker)

```bash
# Run in Docker with Phase 5 binary
docker run -d \
  --name q-phase5 \
  -p 8090:8080 \
  -p 9002:9001 \
  -v $(pwd)/q-api-server-v0.9.25-beta:/app/q-api-server:ro \
  -v $(pwd)/data-phase5:/app/data \
  --restart unless-stopped \
  debian:bullseye \
  /app/q-api-server --port 8080 --db-path=/app/data

# Follow logs
docker logs -f q-phase5
```

---

## 🔍 VERIFICATION

### Test 1: Check Network ID
```bash
curl -s http://localhost:8090/api/v1/status | jq -r '.data.network_id'
# Expected: testnet-phase5
```

### Test 2: Check Gossipsub Topics
```bash
docker logs q-phase5 | grep "Subscribed to" | head -10
# Expected: /qnk/testnet-phase5/blocks, /qnk/testnet-phase5/peer-heights, etc.
```

### Test 3: Check Bootstrap Connection
```bash
curl -s http://localhost:8090/api/v1/status | jq '.data.peer_count'
# Expected: > 0 (connected to Server Beta)
```

### Test 4: Check Blockchain Sync
```bash
curl -s http://localhost:8090/api/v1/status | jq '.data.current_height'
# Expected: Increasing (syncing from Server Beta)
```

---

## 📊 EXPECTED BEHAVIOR

### After Upgrade to Phase 5:

1. **Network ID**: testnet-phase5 ✅
2. **Gossipsub Topics**: `/qnk/testnet-phase5/*` ✅
3. **Bootstrap Peer**: Found (Server Beta) ✅
4. **P2P Connection**: Established ✅
5. **Blockchain Sync**: Active (from height 0) ✅
6. **Block Production**: Receiving blocks from Server Beta ✅

### Logs Should Show:
```
✅ Discovered 2 bootstrap peer(s) automatically
📢 Subscribed to testnet-phase5 Gossipsub topic: /qnk/testnet-phase5/blocks
🌐 libp2p PeerConnected: 12D3KooWCcbvQBzW4PxWxSnUUb9i1nJCkDHkChYXRL1VBhQ8JkBn
📥 [GOSSIPSUB] Received block from peer (height: X)
🔄 [TURBO SYNC] Syncing to height X from peer
✅ Block X validated and saved
```

---

## 🚨 TROUBLESHOOTING

### Issue: "InsufficientPeers" Warning
**Cause**: Still on Phase 4 or network connection issue
**Fix**: Verify Phase 5 binary, check gossipsub topics in logs

### Issue: "current_height missing" Warning
**Cause**: Server Beta API compatibility (known issue in v0.9.25-beta)
**Fix**: Can be ignored - height discovery still works via alternative methods

### Issue: No Peer Connection
**Cause**: Firewall blocking port 9001
**Fix**: Check `ufw allow 9001/tcp` or Docker port mapping

### Issue: Database Locked
**Cause**: Old process still running
**Fix**: `pkill q-api-server` or `docker stop` old container

---

## 📋 QUICK CHECKLIST

- [ ] Downloaded v0.9.25-beta binary from quillon.xyz
- [ ] Verified SHA256 checksum
- [ ] Stopped old Phase 4 container/process
- [ ] Created fresh database directory (`data-phase5`)
- [ ] Started Phase 5 binary
- [ ] Verified network_id = testnet-phase5
- [ ] Verified gossipsub topics = /qnk/testnet-phase5/*
- [ ] Confirmed peer connection to Server Beta
- [ ] Verified blockchain syncing

---

## 🔗 DOWNLOAD LINK

**wget command**:
```bash
wget https://quillon.xyz/downloads/q-api-server-v0.9.25-beta
```

**Alternative (if wget fails)**:
```bash
curl -O https://quillon.xyz/downloads/q-api-server-v0.9.25-beta
```

---

## ✅ SUCCESS CRITERIA

Server Alpha Phase 5 upgrade is successful when:

1. ✅ Network ID: `testnet-phase5`
2. ✅ Gossipsub topics: `/qnk/testnet-phase5/*`
3. ✅ Peer count: > 0 (connected to Server Beta)
4. ✅ Blockchain height: Increasing (syncing)
5. ✅ No "InsufficientPeers" warnings after 30 seconds
6. ✅ Block reception: Receiving blocks via gossipsub

---

**Phase 5 network is ready and waiting for Server Alpha to join!** 🚀🌐✨
