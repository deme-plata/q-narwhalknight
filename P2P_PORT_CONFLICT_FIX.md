# P2P Port Conflict - Root Cause Identified ✅

**Date**: October 26, 2025
**Status**: ROOT CAUSE FOUND
**Severity**: CRITICAL - Blocks P2P initialization

---

## 🎯 ROOT CAUSE

**Port 9000 is being used by nginx**, preventing libp2p from binding to the P2P port.

###Evidence:
```bash
# Service logs show P2P initialization failure:
ERROR q_api_server: P2P listener failed: Address already in use (os error 98)

# Port 9000 is held by nginx:
$ lsof -i :9000
COMMAND    PID   USER   FD   TYPE  DEVICE SIZE/OFF NODE NAME
nginx   670008   root    4u  IPv4 1363084      0t0  TCP *:9000 (LISTEN)
nginx   670014 nobody    4u  IPv4 1363084      0t0  TCP *:9000 (LISTEN)
```

---

## 🔍 WHY THIS CAUSED P2P FAILURE

1. **libp2p Initialization Attempt**:
   - q-api-server tries to bind libp2p to port 9000 on startup
   - Port 9000 already in use by nginx
   - libp2p initialization fails with "Address already in use"

2. **Silent Failure**:
   - Error logged but execution continues
   - `libp2p_discovery` remains None in AppState
   - All subsequent P2P operations fail silently

3. **Impact on Block Propagation**:
   - Block producer checks `if let Some(ref libp2p_manager) = app_state.libp2p_discovery`
   - Since libp2p_discovery is None, broadcast code never executes
   - No gossipsub messages sent
   - Nodes run independent chains

---

## 💡 SOLUTION OPTIONS

### Option 1: Change P2P Port (RECOMMENDED)
Change q-api-server to use a different P2P port (e.g., 9001, 9002, etc.)

**Pros**:
- Avoids nginx conflict
- Quick fix
- No impact on web traffic

**Cons**:
- Need to update bootstrap peer configurations
- Existing nodes need port update

### Option 2: Reconfigure nginx
Move nginx to a different port (e.g., 8090, 8091, etc.)

**Pros**:
- Keeps P2P on standard port 9000
- No changes to P2P configuration

**Cons**:
- Impacts web service configuration
- May affect other services depending on nginx

---

## 🔧 IMPLEMENTATION (Option 1 - Recommended)

### Step 1: Check Current P2P Port Configuration
```bash
grep -r "Q_P2P_PORT\|9000" /etc/systemd/system/q-api-server.service
```

### Step 2: Set P2P Port to 9001
Edit `/etc/systemd/system/q-api-server.service`:
```ini
Environment="Q_P2P_PORT=9001"
```

Or set via environment variable:
```bash
export Q_P2P_PORT=9001
```

### Step 3: Reload and Restart Service
```bash
systemctl daemon-reload
systemctl restart q-api-server.service
```

### Step 4: Verify P2P Initialization
```bash
journalctl -u q-api-server.service --since "1 minute ago" | grep -E "libp2p Network Manager initialized|P2P listener failed"
```

**Expected Output**:
```
INFO ✅ libp2p Network Manager initialized for testnet
```

---

## 🧪 VERIFICATION STEPS

### 1. Check Port 9001 is Available
```bash
lsof -i :9001
# Should return nothing if port is free
```

### 2. Verify libp2p Initialized Successfully
```bash
journalctl -u q-api-server.service --since "2 minutes ago" | grep "libp2p Network Manager"
```

Expected: `✅ libp2p Network Manager initialized`

### 3. Check for Block Broadcast Messages
```bash
journalctl -u q-api-server.service --since "30 seconds ago" | grep "📡 Block.*broadcast"
```

Expected: `📡 Block 5432 broadcast to testnet P2P network`

### 4. Verify Gossipsub Publishing
```bash
journalctl -u q-api-server.service --since "30 seconds ago" | grep "📤 Publishing"
```

Expected: `📤 Publishing 1234 bytes to gossipsub topic: /qnk/testnet/blocks`

---

##📊 DEBUGGING TIMELINE

### What We Tried:
1. ✅ Added diagnostic logging for libp2p_discovery availability
2. ✅ Enhanced logging in publish_topic() method
3. ✅ Fixed dropped tokio::spawn tasks (replaced with direct await)
4. ✅ Fixed deadlock (replaced lock().await with try_lock())
5. ✅ Identified port conflict as root cause

### Root Cause Discovery:
```
ERROR q_api_server: P2P listener failed: Address already in use (os error 98)
```

This error message appeared in logs but was easy to miss among all the initialization messages.

---

## 🎯 NEXT STEPS

1. **Choose P2P Port**: Decide on new P2P port (recommend 9001)
2. **Update Configuration**: Set Q_P2P_PORT environment variable
3. **Restart Service**: `systemctl restart q-api-server.service`
4. **Verify P2P Works**: Check for "📡 Block broadcast" messages
5. **Test Propagation**: Spin up test node and verify block sync

---

## 📝 LESSONS LEARNED

### Why This Was Hard to Debug:
1. **Silent Failure**: libp2p initialization failure didn't stop service startup
2. **Diagnostic Misdirection**: Logs showed "libp2p_manager available" from AppState cloning, masking the fact that libp2p_discovery was None
3. **Multiple Theories**: Investigated tokio::spawn dropping, deadlocks, etc. before finding port conflict

### How to Prevent:
1. **Fatal Errors**: Make libp2p initialization failure stop service startup
2. **Port Validation**: Check port availability before attempting to bind
3. **Clear Logging**: Log libp2p_discovery status prominently on startup

---

**Status**: Ready to implement fix by changing P2P port to 9001
**ETA**: 5 minutes to configure, restart, and verify
**Expected Outcome**: P2P propagation working across all nodes ✅

