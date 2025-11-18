# libp2p Connection Layer Diagnosis - Server Alpha

**Date**: 2025-11-18
**Issue**: Bootstrap discovery works, but TCP connections fail to establish
**Status**: Bootstrap port fix applied ✅, deeper transport issue identified

---

## Current Situation

**What's Working:**
- ✅ Bootstrap discovery finds correct peer on port 9001
- ✅ Network components initialize properly
- ✅ Kademlia DHT adds peer to routing table

**What's NOT Working:**
- ❌ `SwarmEvent::ConnectionEstablished` never fires
- ❌ TCP connections time out or fail
- ❌ Peer count remains 0/0
- ❌ No handshakes exchanged

**Conclusion**: Discovery layer works, connection layer fails.

---

## Root Cause Hypotheses (Ranked by Likelihood)

### 1. **DNS Resolution Failure in Multiaddr** (HIGH)

**Theory**: Server Beta's multiaddr might use `/dns/` instead of `/ip4/`, and Server Alpha cannot resolve it.

**Check**:
```bash
# On Server Alpha - Check actual multiaddr being dialed:
RUST_LOG=libp2p_swarm=debug ./q-api-server 2>&1 | grep -E "Dialing.*185.182.185.227"

# Look for:
# Dialing /dns/quillon.xyz/tcp/9001/...  ❌ DNS might fail
# Dialing /ip4/185.182.185.227/tcp/9001/... ✅ Should work
```

**Fix**: Ensure `BOOTSTRAP_PEERS` uses `/ip4/` not `/dns/`:
```rust
// CORRECT:
"/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7"

// WRONG (if DNS fails):
"/dns/quillon.xyz/tcp/9001/p2p/12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7"
```

---

### 2. **Connection Limits Blocking Inbound** (MEDIUM-HIGH)

**Theory**: `connection_limits` behaviour may be rejecting connections.

**Check Configuration** (`unified_network_manager.rs`):
```rust
// Look for connection_limits configuration
let limits = ConnectionLimits::default()
    .with_max_pending_incoming(Some(10))
    .with_max_pending_outgoing(Some(10))
    .with_max_established_incoming(Some(50))
    .with_max_established_outgoing(Some(50))
    .with_max_established_per_peer(Some(5));
```

**Diagnostic**:
```bash
# Check for "connection limit" errors in logs:
journalctl -u q-api-server -n 200 | grep -i "limit\|reject\|deny"
```

**Fix**: Temporarily disable limits to test:
```rust
// In QNarwhalBehaviour initialization:
// Comment out connection_limits temporarily
// connection_limits: libp2p::connection_limits::Behaviour::new(limits),
```

---

### 3. **TCP Transport Not Configured** (MEDIUM)

**Theory**: libp2p transport stack might not include TCP, or uses incompatible upgrade.

**Check Transport Stack** (`unified_network_manager.rs` around line 400-500):
```rust
// Verify transport configuration includes TCP:
let transport = tcp::async_io::Transport::new(tcp::Config::default())
    .upgrade(upgrade::Version::V1)
    .authenticate(noise::Config::new(&keypair)?)
    .multiplex(yamux::Config::default())
    .boxed();
```

**Common Issues**:
- Missing `.authenticate()` step
- Incompatible `noise::Config` parameters
- Wrong yamux configuration

**Diagnostic**:
```bash
# Look for transport errors:
RUST_LOG=libp2p_tcp=debug,libp2p_core=debug ./q-api-server 2>&1 | \
  grep -E "(transport|upgrade|authenticate|multiplex)"
```

---

### 4. **Firewall Blocking Outbound on Server Alpha** (MEDIUM)

**Theory**: Server Alpha's firewall blocks outbound TCP to 185.182.185.227:9001.

**Test**:
```bash
# On Server Alpha:
# 1. Basic connectivity
nc -zv 185.182.185.227 9001

# 2. Check iptables
sudo iptables -L OUTPUT -n -v | grep -E "185.182.185.227|9001"

# 3. Check ufw
sudo ufw status | grep -E "9001|185.182"
```

**Fix**:
```bash
# Allow outbound to Server Beta P2P port
sudo iptables -A OUTPUT -p tcp -d 185.182.185.227 --dport 9001 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4
```

---

### 5. **Handshake Protocol Mismatch** (LOW-MEDIUM)

**Theory**: Handshake protocol version differs between Server Alpha and Server Beta.

**Check**:
```bash
# On both servers:
grep -r "HANDSHAKE_PROTOCOL\|/qnk/handshake" crates/q-network/src/

# Should match exactly:
pub const HANDSHAKE_PROTOCOL: &'static str = "/qnk/handshake/1.0.0";
```

**Diagnostic - Enable handshake logging**:
```bash
RUST_LOG=q_network::handshake_validator=trace ./q-api-server 2>&1 | \
  grep -E "handshake|protocol"
```

---

### 6. **Server Alpha Behind NAT Without Relay** (LOW)

**Theory**: Server Alpha is behind NAT and can't accept inbound, needs relay.

**Check**:
```bash
# On Server Alpha:
curl -4 ifconfig.me  # External IP
ip addr show         # Internal IP

# If they differ, Server Alpha is behind NAT
```

**Solution**: NAT traversal is already implemented (AutoNAT, Relay, DCUtR) but may need explicit relay configuration:

```rust
// Ensure Server Beta is configured as relay server
// In unified_network_manager.rs on Server Beta:
use libp2p::relay;

// Add relay server behaviour (not just client)
pub struct QNarwhalBehaviour {
    // ...
    relay_server: relay::Behaviour,  // Add this
}
```

---

## Advanced Diagnostics

### Comprehensive Debug Command

Run this on **Server Alpha** and share the output:

```bash
#!/bin/bash
echo "=== libp2p Connection Layer Diagnostics ==="
echo ""

echo "1. Network Connectivity Test"
nc -zv -w 5 185.182.185.227 9001
echo ""

echo "2. DNS Resolution Test"
dig +short quillon.xyz
nslookup quillon.xyz
echo ""

echo "3. Firewall Status"
sudo iptables -L OUTPUT -n -v | head -20
sudo ufw status
echo ""

echo "4. Current PeerID"
RUST_LOG=q_network=debug timeout 30 ./q-api-server 2>&1 | \
  grep -E "Local peer ID|PeerId" | head -1
echo ""

echo "5. Bootstrap Dial Attempts"
RUST_LOG=libp2p_swarm=debug timeout 30 ./q-api-server 2>&1 | \
  grep -E "Dialing|DialFailure|ConnectionEstablished" | head -10
echo ""

echo "6. Transport Errors"
RUST_LOG=libp2p_tcp=debug,libp2p_core=debug timeout 30 ./q-api-server 2>&1 | \
  grep -E "error|Error|failed|Failed" | head -10
echo ""

echo "7. Handshake Protocol Check"
grep -r "HANDSHAKE_PROTOCOL" crates/q-network/src/ | head -5
echo ""

echo "=== Diagnostics Complete ==="
```

### Targeted libp2p Event Logging

Add this to `unified_network_manager.rs` event loop (around line 933):

```rust
SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
    error!("❌ OUTGOING CONNECTION FAILED: peer={:?}, error={:?}", peer_id, error);

    // Log specific error types
    match error {
        libp2p::swarm::DialError::Transport(addrs) => {
            for (addr, err) in addrs {
                error!("  Transport failed for {}: {:?}", addr, err);
            }
        }
        libp2p::swarm::DialError::ConnectionLimit(_) => {
            error!("  Connection limit reached!");
        }
        libp2p::swarm::DialError::Denied { cause } => {
            error!("  Connection denied: {:?}", cause);
        }
        _ => {}
    }
}

SwarmEvent::IncomingConnectionError { local_addr, send_back_addr, error, .. } => {
    warn!("❌ INCOMING CONNECTION FAILED: local={}, remote={}, error={:?}",
          local_addr, send_back_addr, error);
}
```

---

## Systematic Fix Process

### Phase 1: Confirm Basic Connectivity (1 minute)

```bash
# From Server Alpha:
nc -zv 185.182.185.227 9001

# Expected: "Connection to 185.182.185.227 9001 port [tcp/*] succeeded!"
# If this fails, it's firewall/network, not libp2p
```

### Phase 2: Enable Maximum Debug Logging (30 seconds)

```bash
# On Server Alpha:
sudo systemctl stop q-api-server

RUST_LOG=debug,libp2p=trace,q_network=trace \
/path/to/q-api-server 2>&1 | tee /tmp/libp2p-full-debug.log &

# Let it run for 60 seconds
sleep 60

# Analyze failures
grep -E "(DialFailure|OutgoingConnectionError|Transport.*error)" /tmp/libp2p-full-debug.log
```

### Phase 3: Check for Specific Error Patterns

```bash
# Pattern 1: DNS resolution failures
grep -i "dns" /tmp/libp2p-full-debug.log

# Pattern 2: Connection refused (server not listening)
grep -i "refused" /tmp/libp2p-full-debug.log

# Pattern 3: Timeout (firewall blocking)
grep -i "timeout\|timed out" /tmp/libp2p-full-debug.log

# Pattern 4: Connection limit
grep -i "limit\|maximum" /tmp/libp2p-full-debug.log

# Pattern 5: Protocol negotiation failure
grep -i "negotiate\|protocol.*failed" /tmp/libp2p-full-debug.log
```

### Phase 4: Apply Targeted Fix

Based on the pattern found above, apply the corresponding fix from the hypotheses section.

---

## Quick Fixes to Try in Order

### Fix 1: Verify multiaddr is IP-based (30 seconds)
```bash
# Check current bootstrap config:
grep "BOOTSTRAP_PEERS" crates/q-network/src/unified_network_manager.rs

# Should be /ip4/ not /dns/
# If wrong, rebuild with corrected multiaddr
```

### Fix 2: Temporarily disable connection limits (2 minutes)
```rust
// In unified_network_manager.rs, comment out:
// connection_limits: libp2p::connection_limits::Behaviour::new(limits),

// Rebuild and test
cargo build --release --package q-api-server
```

### Fix 3: Add explicit error logging (5 minutes)
```rust
// Add the OutgoingConnectionError handler from "Targeted libp2p Event Logging" above
// This will tell us exactly why connections are failing
```

### Fix 4: Simplify transport stack (10 minutes)
```rust
// Try minimal transport configuration:
let transport = tcp::async_io::Transport::default()
    .upgrade(upgrade::Version::V1)
    .authenticate(noise::Config::new(&keypair)?)
    .multiplex(yamux::Config::default())
    .boxed();
```

---

## Expected Results When Fixed

**Immediate (within 10 seconds of startup):**
```
[INFO] 🔗 Dialing /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWC688...
[DEBUG] TCP connection attempt to 185.182.185.227:9001
[INFO] 🔗 ConnectionEstablished: peer=12D3KooWC688..., num_established=1
[DEBUG] 🤝 [HANDSHAKE] Sent handshake request to 12D3KooWC688...
```

**Within 30 seconds:**
```
[INFO] ✅ [HANDSHAKE] Peer validated our handshake successfully
[INFO] 📊 Healthy peer connections: 1/1
[INFO] 🚀 TURBO SYNC: Network height above ours, activating sync...
```

---

## Next Steps

1. **Run comprehensive debug command** (above) and share output
2. **Check Phase 1 connectivity** with `nc -zv`
3. **Enable Phase 2 debug logging** and identify error pattern
4. **Apply corresponding fix** from hypotheses section
5. **Verify** with expected results

The issue is definitely at the libp2p transport/connection layer, not discovery. Once we see the actual error logs, the fix will be straightforward.

---

## Related Files

- `crates/q-network/src/unified_network_manager.rs:400-600` - Transport configuration
- `crates/q-network/src/unified_network_manager.rs:933-1000` - Connection event handling
- `crates/q-network/src/unified_network_manager.rs:37-42` - Bootstrap configuration
- `crates/q-network/src/handshake_validator.rs` - Handshake protocol implementation

---

**Status**: Awaiting Server Alpha diagnostic output to pinpoint exact failure mode.
