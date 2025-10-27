# Critical Peer Discovery Fix - Zero Peer Count Issue

## Problem Summary

Users were experiencing **zero peer count** because the bootstrap infrastructure was not properly configured.

## Root Cause Analysis

### Issue #1: Random P2P Ports
- libp2p was listening on **random ports** (e.g., `tcp/40937`) instead of the hardcoded port `9001`
- Bootstrap configuration in `crates/q-types/src/lib.rs:775` specifies `/ip4/185.182.185.227/tcp/9001/p2p/...`
- Nodes attempting to connect to port `9001` received "Connection Refused"

### Issue #2: Outdated Peer ID
- Hardcoded peer ID `12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG` was from an old bootstrap node
- Current running node has a different peer ID
- Peer ID changes on every restart (new keypair generated)

### Issue #3: Secondary Bootstrap Peer Unreachable
- Secondary bootstrap at `161.35.219.10:9001` is not responding
- HTTP peer ID discovery endpoint times out
- No fallback mechanism when peer ID fetch fails

## Solution Implemented

### Fix #1: Support Fixed P2P Ports via Environment Variable

**File**: `crates/q-network/src/unified_network_manager.rs`

Added support for `Q_P2P_PORT` environment variable:
- If `Q_P2P_PORT` is set, libp2p listens on that fixed port
- If not set, defaults to random port (port 0)
- Bootstrap nodes MUST set this to `9001`

```rust
let p2p_port = std::env::var("Q_P2P_PORT")
    .ok()
    .and_then(|p| p.parse::<u16>().ok())
    .unwrap_or(0); // 0 = random port (default)

if p2p_port > 0 {
    info!("🔒 Using fixed libp2p port: {}", p2p_port);
    swarm.listen_on(format!("/ip4/0.0.0.0/tcp/{}", p2p_port).parse()?)?;
    swarm.listen_on(format!("/ip6/::/tcp/{}", p2p_port).parse()?)?;
}
```

### Fix #2: Restart Bootstrap Node with Fixed Port

The existing systemd service `/etc/systemd/system/q-api-server.service` already has:
```
Environment="Q_P2P_PORT=9001"
```

After recompiling and restarting the service:
1. Node will listen on port `9001` (fixed)
2. New peer ID will be generated
3. Peer ID needs to be retrieved and published

### Fix #3: Get Peer ID from Bootstrap Node

After restart, get the peer ID using:
```bash
# Check systemd logs for peer ID
journalctl -u q-api-server -n 100 | grep "Local Peer ID"

# Or check via HTTP API (once /api/v1/peer-id is fixed)
curl http://185.182.185.227:8080/api/v1/peer-id
```

### Fix #4: Update Bootstrap Configuration

Once we have the new peer ID, update the hardcoded bootstrap configuration in:
- `crates/q-types/src/lib.rs:775` (testnet bootstrap_peers)
- Release binaries and documentation

## Deployment Steps

1. ✅ **Modified code** to support `Q_P2P_PORT` environment variable
2. ⏳ **Compiling** new binary with fixed code
3. **Restart service**: `systemctl restart q-api-server`
4. **Get peer ID** from logs
5. **Update configuration** with correct peer ID
6. **Publish new binary** with corrected bootstrap peers

## Verification

After restart, verify:
```bash
# Check if port 9001 is listening
ss -tlnp | grep 9001

# Check peer ID from logs
journalctl -u q-api-server -n 50 | grep "Local Peer ID"

# Verify peers can connect
# From another node, try to dial the bootstrap node
```

## For Users

### Temporary Workaround (Until Fix is Deployed)

If you're a user experiencing zero peers:
1. Try connecting to other users on your **local network** - mDNS discovery should work
2. Wait for the bootstrap node fix to be deployed
3. Update to the latest binary once published

### Permanent Fix

Once the bootstrap node is restarted with the fix:
1. The official bootstrap peer address will be published
2. New releases will have the corrected bootstrap configuration
3. Existing users should update their nodes to the latest version

## Technical Details

### Bootstrap Node Configuration

**Server**: 185.182.185.227 (this server)
**HTTP API Port**: 8080
**libp2p P2P Port**: 9001 (fixed)
**Database**: `./data-mine1`
**Systemd Service**: `/etc/systemd/system/q-api-server.service`

### Expected Bootstrap Address Format

After fix is deployed:
```
/ip4/185.182.185.227/tcp/9001/p2p/<NEW_PEER_ID>
```

Where `<NEW_PEER_ID>` will be a `12D3Koo...` identifier retrieved after restart.

## Next Steps

1. Wait for compilation to complete
2. Restart the systemd service
3. Extract the new peer ID
4. Update the codebase with correct bootstrap peer
5. Release new binaries
6. Announce the fix to users

---

**Status**: 🔧 Fix implemented, compilation in progress
**ETA**: ~10-15 minutes for compilation + restart
**Impact**: Will restore peer discovery for all testnet users
