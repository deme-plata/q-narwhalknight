# Server Alpha: libp2p Network Manager Initialization Failure

**Date**: 2025-11-07
**Affected**: Server Alpha (161.35.219.10) - Ubuntu 22.04 container, port 8090
**Version**: v0.9.36-beta
**Status**: ❌ **libp2p initialization failing**

---

## Problem Summary

**User Report**: Server Alpha shows `libp2p_manager is None - no gossipsub receiver available`

**Impact**:
- ❌ No P2P connections possible
- ❌ TURBO SYNC impossible (requires P2P)
- ❌ Fallback to slow HTTP sync only
- ❌ Isolated from gossipsub mesh

**Server Beta Comparison**:
- ✅ libp2p working correctly on Server Beta (185.182.185.227)
- ✅ TURBO SYNC enabled
- ✅ Gossipsub mesh operational

---

## What Server Alpha Shows

**Working Components**:
- ✅ Gossipsub topic subscriptions
- ✅ Bootstrap peer discovery prep
- ✅ Tor client initialization
- ✅ DHT bootstrap preparation
- ✅ Peer ID generation: `12D3KooWGJMwYvZHTVZw2wbfkKB72SHqe1DzrSAgWBkrUwuhZsPt`

**Failing Component**:
- ❌ libp2p network layer fails to start
- ❌ `libp2p_manager is None` error
- ❌ No gossipsub receiver available

---

## Root Cause Analysis

### Possible Causes

#### 1. Port Binding Issue in Container

**Symptoms**:
- Container running on port 8090 (API)
- P2P port 9001 may not be exposed/mapped

**Check**:
```bash
# On Server Alpha (via docker)
docker port <container-name>
# Should show: 9001/tcp -> 0.0.0.0:9001
```

**Fix if missing**:
```bash
# Stop container
docker stop <container-name>

# Run with proper port mapping
docker run -d \
  --name q-v0936-beta \
  -p 8090:8080 \
  -p 9001:9001 \  # ← P2P port must be exposed!
  -e Q_HOST=0.0.0.0 \
  -e Q_P2P_PORT=9001 \
  ubuntu:22.04 /path/to/q-api-server --port 8080
```

#### 2. Network Interface Binding

**Symptoms**:
- libp2p trying to bind to localhost only
- Container isolation preventing external connectivity

**Check**: Look for log messages about network interface binding

**Server Beta (working)**:
```
INFO libp2p_mdns::behaviour::iface: creating instance on iface address=185.182.185.227
INFO q_network::unified_network_manager: 🔒 Using fixed libp2p port: 9001
```

**Server Alpha (if failing)**: May show errors about binding to interfaces

**Fix**: Ensure `Q_HOST=0.0.0.0` is set (already done according to your report)

#### 3. Library Compatibility Issue

**Symptoms**:
- Ubuntu 22.04 container should work (you confirmed this)
- But libp2p dynamic library loading may fail silently

**Check**:
```bash
# Inside container
ldd /path/to/q-api-server | grep "not found"
```

**Expected**: No "not found" messages

**Fix if needed**: Install missing libraries:
```bash
apt-get update && apt-get install -y \
  libssl3 \
  ca-certificates \
  build-essential
```

#### 4. Swarm Initialization Failure

**Symptoms**:
- libp2p swarm fails to create
- Network manager returns None

**Potential Causes**:
- Port 9001 already in use
- Permission denied (non-root user)
- Firewall blocking

**Check**:
```bash
# Check if port is already in use
netstat -tulpn | grep 9001

# Check firewall
iptables -L -n | grep 9001
```

---

## Diagnostic Commands for Server Alpha

### Inside the Container

```bash
# 1. Check if binary is running
ps aux | grep q-api-server

# 2. Check recent logs for libp2p initialization
docker logs <container-name> 2>&1 | grep -E "libp2p|Network Manager"

# 3. Check for error messages
docker logs <container-name> 2>&1 | grep -E "Error|Failed|None"

# 4. Check port bindings
netstat -tulpn | grep -E "8080|9001"

# 5. Check network interfaces
ip addr show
```

### Expected Successful Output (like Server Beta)

```
INFO q_api_server: 🌐 Initializing libp2p Unified Network Manager...
INFO libp2p_swarm: local_peer_id=12D3KooW...
INFO q_network::unified_network_manager: 🔒 Using fixed libp2p port: 9001
INFO q_api_server: ✅ libp2p Network Manager initialized
INFO q_api_server: 📡 libp2p_manager extracted successfully - gossipsub channels ready!
```

---

## Comparison: Server Beta vs Server Alpha

### Server Beta (185.182.185.227) - **WORKING** ✅

**Environment**:
- systemd service (not container)
- Port 8080 (API), Port 9001 (P2P)
- Direct host network access

**libp2p Status**:
```
✅ libp2p Network Manager initialized for Q-NarwhalKnight Testnet
✅ libp2p network fully operational
✅ libp2p_manager extracted successfully - gossipsub channels ready!
✅ TURBO SYNC Network channel configured - TRUE P2P enabled!
```

**Peer ID**: `12D3KooWLQok4vAPYLWSbUuj4LY4dLYcaJCeMp12GaEpDNQ6uJGJ`

### Server Alpha (161.35.219.10) - **FAILING** ❌

**Environment**:
- Ubuntu 22.04 container
- Port 8090 (API), Port 9001 (P2P - may not be exposed?)
- Container network (bridge mode?)

**libp2p Status**:
```
❌ libp2p_manager is None - no gossipsub receiver available
❌ No P2P connections possible
❌ TURBO SYNC impossible
```

**Peer ID**: `12D3KooWGJMwYvZHTVZw2wbfkKB72SHqe1DzrSAgWBkrUwuhZsPt` (generated but unused)

---

## Recommended Fixes

### Fix 1: Verify Port Mapping (Most Likely)

```bash
# On Server Alpha
docker inspect <container-name> | jq '.[0].NetworkSettings.Ports'
```

**Expected**:
```json
{
  "8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8090"}],
  "9001/tcp": [{"HostIp": "0.0.0.0", "HostPort": "9001"}]
}
```

**If 9001 is missing**, recreate container with `-p 9001:9001`

### Fix 2: Check Container Logs for Actual Error

```bash
# Get full error details
docker logs <container-name> 2>&1 | grep -A10 -B10 "libp2p_manager is None"
```

This will show the **actual error** that caused initialization to fail

### Fix 3: Try Host Network Mode

If port mapping is the issue, try running container in host network mode:

```bash
docker run -d \
  --name q-v0936-beta \
  --network host \  # ← Use host network directly
  -e Q_HOST=0.0.0.0 \
  -e Q_P2P_PORT=9001 \
  ubuntu:22.04 /path/to/q-api-server --port 8090
```

**Note**: With `--network host`, the container uses the host's network stack directly. No port mapping needed, but API will be on port 8090 as configured.

### Fix 4: Check Systemd Journal (if using systemd in container)

```bash
# If using systemd inside container
docker exec <container-name> journalctl -xe | grep libp2p
```

---

## Testing After Fix

### 1. Verify libp2p Initialization

```bash
docker logs <container-name> 2>&1 | tail -100 | grep "libp2p"
```

**Expected**:
```
✅ libp2p Network Manager initialized
✅ libp2p network fully operational
✅ libp2p_manager extracted successfully
```

### 2. Verify TURBO SYNC Enabled

```bash
docker logs <container-name> 2>&1 | grep "TURBO SYNC.*configured"
```

**Expected**:
```
🌐 [TURBO SYNC] Network channel configured - TRUE P2P enabled!
```

### 3. Check P2P Connectivity

```bash
# After a few minutes
docker logs <container-name> 2>&1 | grep "Connected to peer"
```

**Expected**: Should see connections to other peers (including Server Beta)

---

## If Still Failing

### Get Complete Startup Logs

```bash
# Recreate container with full logging
docker rm <container-name>
docker run -d \
  --name q-v0936-beta-debug \
  -p 8090:8080 \
  -p 9001:9001 \
  -e RUST_LOG=debug \  # ← Enable debug logging
  -e Q_HOST=0.0.0.0 \
  -e Q_P2P_PORT=9001 \
  ubuntu:22.04 /path/to/q-api-server --port 8080

# Wait 30 seconds, then check logs
docker logs q-v0936-beta-debug 2>&1 > /tmp/server-alpha-debug.log

# Look for the actual error
grep -E "Error|Failed|panic|libp2p.*None" /tmp/server-alpha-debug.log
```

---

## Environment Variables Checklist

Ensure these are set when running Server Alpha:

```bash
Q_HOST=0.0.0.0              # ✅ Confirmed
Q_P2P_PORT=9001             # ❓ Verify this is set
Q_IS_VALIDATOR=true         # If mining
RUST_LOG=info               # For logging (or 'debug' for troubleshooting)
```

**Optional but recommended**:
```bash
Q_BOOTSTRAP_PEER=/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWLQok4vAPYLWSbUuj4LY4dLYcaJCeMp12GaEpDNQ6uJGJ
```

---

## Status Summary

**Server Beta**: ✅ libp2p working, TURBO SYNC enabled
**Server Alpha**: ❌ libp2p failing, TURBO SYNC disabled

**Most Likely Cause**: Port 9001 not exposed in Docker container
**Quick Fix**: Add `-p 9001:9001` to docker run command
**Alternative**: Use `--network host` mode

---

**Next Step**: Check Server Alpha's Docker port mapping and logs to identify the exact failure point

**Date**: 2025-11-07
**Diagnosed By**: Claude Code (Server Beta)
