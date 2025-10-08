# Bootstrap Peer Port Fix

## Problem Identified

The Windows node was unable to discover the Linux server due to a **hardcoded bootstrap peer with incorrect port**.

### Root Cause

**File**: `crates/q-network/src/unified_network_manager.rs:34`

**Original (broken)**:
```rust
const DEFAULT_BOOTSTRAP_PEER: &str = "/ip4/185.182.185.227/tcp/40735/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG";
```

**Fixed**:
```rust
const DEFAULT_BOOTSTRAP_PEER: &str = "/ip4/185.182.185.227/tcp/8081/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG";
```

### Port Mismatch

- **Hardcoded Port**: 40735 ❌
- **Actual Linux Server Port**: 8081 ✅

This caused the Windows node to attempt connections to the wrong port, resulting in no peer discovery.

## Behavior Analysis

### Windows Node Startup

The Windows node showed:
```
🚀 libp2p Zero-Knowledge Discovery initialized successfully!
📡 Active discovery mechanisms: mDNS (local network), Identify (peer exchange), Ping (keepalive)
```

However, **no DHT bootstrap logs appeared** because:
1. mDNS is disabled on Windows (only works on local networks)
2. The fallback to hardcoded bootstrap peer used wrong port
3. Connection to 185.182.185.227:40735 failed silently
4. No peers discovered = no network connectivity

### Expected Bootstrap Logs (after fix)

After fixing the port, Windows nodes should show:
```
ℹ️ Using default bootstrap peer: /ip4/185.182.185.227/tcp/8081/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG
📍 Added bootstrap peer: 12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG at /ip4/185.182.185.227/tcp/8081/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG
🚀 Kademlia DHT bootstrap initiated with 1 peers
✅ DHT bootstrap complete: X peers in routing table
```

## Code Flow

### Bootstrap Peer Loading (lines 148-179)

```rust
// Bootstrap from environment variable or use hardcoded default
let bootstrap_peers_str = std::env::var("Q_BOOTSTRAP_PEERS")
    .unwrap_or_else(|_| {
        info!("ℹ️ Using default bootstrap peer: {}", DEFAULT_BOOTSTRAP_PEER);
        DEFAULT_BOOTSTRAP_PEER.to_string()
    });
```

**Behavior**:
1. If `Q_BOOTSTRAP_PEERS` environment variable is set → use it
2. If NOT set → fallback to `DEFAULT_BOOTSTRAP_PEER` constant

**Before Fix**: DEFAULT_BOOTSTRAP_PEER had wrong port (40735)
**After Fix**: DEFAULT_BOOTSTRAP_PEER has correct port (8081)

## Solution

### Option 1: Use Environment Variable (Recommended)
```powershell
$env:Q_BOOTSTRAP_PEERS = "/ip4/185.182.185.227/tcp/8081/p2p/12D3KooWPaQogoQVq1XoNenW93So8TC9T8CahEoMto455j4jgYmG"
.\q-api-server.exe --port 8096
```

### Option 2: Rebuild with Fixed Default
```bash
# Fix is already applied to unified_network_manager.rs
# Rebuild Windows executable
cross build --release --target x86_64-pc-windows-gnu --package q-api-server
```

## Testing

After rebuilding with the fix, run on Windows:
```powershell
.\q-api-server.exe --port 8096
```

Expected output should include:
- ℹ️ Using default bootstrap peer message
- 📍 Added bootstrap peer message
- 🚀 Kademlia DHT bootstrap initiated
- ✅ DHT bootstrap complete with peer count

## Impact

- **Before**: Windows nodes could NOT connect to Linux server (wrong port)
- **After**: Windows nodes automatically connect to Linux server (correct port)
- **Compatibility**: Existing deployments with `Q_BOOTSTRAP_PEERS` environment variable continue to work

## Related Files

- `crates/q-network/src/unified_network_manager.rs` - Bootstrap peer configuration
- `LIBP2P_PHASE5B_COMPLETE.md` - Original bootstrap implementation documentation
- `windows-release/README.md` - Windows deployment guide

---

**Fix Applied**: 2025-10-07
**Status**: Needs rebuild and testing
