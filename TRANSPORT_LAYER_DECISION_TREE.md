# Transport Layer Decision Tree - Systematic Diagnosis

**Date**: 2025-11-18
**Status**: Bootstrap discovery works ✅, ConnectionEstablished never fires ❌
**Goal**: Identify exact failure point between TCP dial and libp2p transport upgrade

---

## Decision Tree (Execute in Order)

### ✅ Step 1: Prove Raw TCP Connectivity (30 seconds)

**On Server Alpha - Run this first:**
```bash
echo "=== STEP 1: Raw TCP Test ==="
timeout 5 nc -zv 185.182.185.227 9001
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TCP connectivity SUCCESS - Problem is in libp2p layer"
    echo "→ Proceed to Step 2"
else
    echo "❌ TCP connectivity FAILED - Problem is at OS/network layer"
    echo "→ Run firewall diagnostics below"
fi
```

**Decision:**
- **If FAILS** (exit code != 0): **STOP HERE** - Fix OS/network layer first
- **If SUCCEEDS**: Problem is in libp2p transport/config - proceed to Step 2

---

### ❌ Step 1 Failed: OS/Network Layer Diagnostics

**On Server Beta (185.182.185.227):**
```bash
echo "=== Server Beta - Verify Listener ==="
ss -tlnp | grep 9001
lsof -i :9001

# Expected: q-api-server listening on 0.0.0.0:9001 or [::]:9001
# If NOT found:
#   - Check Q_P2P_BIND_ADDR environment variable
#   - Verify q-api-server is actually running
#   - Check it's binding to 0.0.0.0, not 127.0.0.1
```

**Firewall Check - Both Servers:**
```bash
echo "=== Firewall Status ==="
# Beta (inbound):
sudo iptables -L INPUT -n -v | grep 9001
sudo ufw status | grep 9001

# Alpha (outbound):
sudo iptables -L OUTPUT -n -v | grep 9001
sudo ufw status | grep 9001
```

**Common Fixes:**

1. **Server Beta not listening on public interface:**
   ```bash
   # Check environment:
   systemctl show q-api-server | grep Q_P2P_BIND_ADDR

   # Should be: Q_P2P_BIND_ADDR=0.0.0.0:9001
   # NOT: Q_P2P_BIND_ADDR=127.0.0.1:9001
   ```

2. **Firewall blocking:**
   ```bash
   # Beta - allow inbound:
   sudo ufw allow 9001/tcp
   sudo iptables -A INPUT -p tcp --dport 9001 -j ACCEPT

   # Alpha - allow outbound (usually not needed):
   sudo iptables -A OUTPUT -p tcp -d 185.182.185.227 --dport 9001 -j ACCEPT
   ```

**Once TCP works, restart from Step 1 to confirm, then proceed to Step 2.**

---

### ✅ Step 2: Identify Exact Dial Failure (2 minutes)

**TCP works, but libp2p doesn't connect. Need to see the actual error.**

**On Server Alpha - Add diagnostic logging:**

First, add this code to `crates/q-network/src/unified_network_manager.rs`:

```rust
// In the SwarmEvent match statement (around line 933 or 1928):
SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
    error!("❌ [P2P] OUTGOING CONNECTION FAILED");
    error!("   Peer: {:?}", peer_id);
    error!("   Error: {:?}", error);

    // Detailed error breakdown:
    match &error {
        libp2p::swarm::DialError::Transport(addrs) => {
            error!("   🚨 TRANSPORT ERROR:");
            for (addr, transport_err) in addrs {
                error!("      Address: {}", addr);
                error!("      Error: {:?}", transport_err);

                // Further breakdown of transport errors:
                use libp2p::TransportError;
                match transport_err {
                    TransportError::MultiaddrNotSupported(a) => {
                        error!("         → Multiaddr not supported: {}", a);
                    }
                    TransportError::Other(e) => {
                        error!("         → IO Error: {}", e);
                        // This is where ConnectionRefused, Timeout, etc appear
                    }
                }
            }
        }
        libp2p::swarm::DialError::ConnectionLimit(limit) => {
            error!("   🚨 CONNECTION LIMIT REACHED: {:?}", limit);
        }
        libp2p::swarm::DialError::Denied { cause } => {
            error!("   🚨 CONNECTION DENIED: {:?}", cause);
        }
        libp2p::swarm::DialError::NoAddresses => {
            error!("   🚨 NO ADDRESSES TO DIAL");
        }
        libp2p::swarm::DialError::WrongPeerId { obtained, endpoint } => {
            error!("   🚨 WRONG PEER ID: expected vs obtained");
            error!("      Obtained: {}", obtained);
            error!("      Endpoint: {:?}", endpoint);
        }
        libp2p::swarm::DialError::Aborted => {
            error!("   🚨 DIAL ABORTED");
        }
        libp2p::swarm::DialError::DialPeerConditionFalse(_) => {
            error!("   🚨 DIAL PEER CONDITION FALSE");
        }
        other => {
            error!("   🚨 OTHER DIAL ERROR: {:?}", other);
        }
    }
}
```

**Then run with full logging:**
```bash
cd /opt/orobit/shared/q-narwhalknight
cargo build --release --package q-api-server

# Run with maximum transport debugging:
RUST_LOG=debug,libp2p_swarm=trace,libp2p_tcp=debug,q_network=trace \
./target/release/q-api-server 2>&1 | tee /tmp/transport-diagnosis.log &

DIAGNOSIS_PID=$!
echo "Running diagnostic (PID: $DIAGNOSIS_PID)..."

# Wait for connection attempts (30 seconds):
sleep 30

# Kill the diagnostic run:
kill $DIAGNOSIS_PID

# Analyze the logs:
echo ""
echo "=== DIAL ATTEMPTS ==="
grep -E "Dialing" /tmp/transport-diagnosis.log | head -10

echo ""
echo "=== CONNECTION ERRORS ==="
grep -E "OUTGOING CONNECTION FAILED|TRANSPORT ERROR|CONNECTION LIMIT|DENIED" /tmp/transport-diagnosis.log

echo ""
echo "=== MULTIADDR FORMAT CHECK ==="
grep "Dialing /ip4/185.182.185.227" /tmp/transport-diagnosis.log && \
    echo "✅ Using /ip4/ format (correct)" || \
    echo "❌ NOT using /ip4/ - might be using /dns/"

grep "Dialing /dns/" /tmp/transport-diagnosis.log && \
    echo "⚠️  WARNING: Using /dns/ multiaddr - this may fail if DNS broken"
```

---

### 📊 Step 2 Results Interpretation

**Pattern A: DNS Multiaddr Issue**
```
Dialing /dns/quillon.xyz/tcp/9001/p2p/...
TRANSPORT ERROR: IO Error: ... DNS resolution failed
```
**Fix**: Change to `/ip4/` in `BOOTSTRAP_PEERS`:
```rust
// In unified_network_manager.rs:
const BOOTSTRAP_PEERS: &[&str] = &[
    "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7",
    // ^^^^ Must be /ip4/, not /dns/
];
```

---

**Pattern B: Connection Limit**
```
OUTGOING CONNECTION FAILED
CONNECTION LIMIT REACHED: ConnectionLimit { ... }
```
**Fix**: Temporarily disable connection limits for testing:
```rust
// In unified_network_manager.rs QNarwhalBehaviour initialization:
QNarwhalBehaviour {
    // ... other fields ...
    // connection_limits: libp2p::connection_limits::Behaviour::new(limits),
    // ^^^^ Comment this out temporarily
}
```

---

**Pattern C: Transport/Upgrade Failure**
```
OUTGOING CONNECTION FAILED
TRANSPORT ERROR: ... protocol negotiation failed / upgrade error
```
**Fix**: Transport stack mismatch between Alpha and Beta. Check:

1. **Verify both use same libp2p version:**
   ```bash
   grep "libp2p.*=" Cargo.toml
   # Should be identical on both servers
   ```

2. **Check transport config matches:**
   ```rust
   // Both must use identical:
   .upgrade(upgrade::Version::V1)  // Same version
   .authenticate(noise::Config::new(&keypair)?)  // Same auth
   .multiplex(yamux::Config::default())  // Same multiplexer
   ```

3. **Temporary debug: Use plaintext auth:**
   ```rust
   use libp2p::plaintext;

   let transport = tcp_transport
       .upgrade(upgrade::Version::V1)
       .authenticate(plaintext::Config::new(&local_keypair.public()))
       .multiplex(yamux::Config::default())
       .boxed();

   // If this works but Noise doesn't, the issue is in Noise config
   ```

---

**Pattern D: Wrong PeerID**
```
OUTGOING CONNECTION FAILED
WRONG PEER ID: expected vs obtained
  Obtained: 12D3KooWXXXX...
```
**Fix**: Update `BOOTSTRAP_PEERS` with actual PeerID from Beta logs:
```bash
# On Beta:
journalctl -u q-api-server | grep "Local peer ID" | tail -1
```

---

**Pattern E: IO Error (ConnectionRefused after TCP test worked)**
```
TRANSPORT ERROR: IO Error: ConnectionRefused
```
**This is weird** - `nc` worked but libp2p's TCP transport says refused. Possible causes:

1. **Server Beta restarted** between `nc` test and libp2p dial:
   ```bash
   # On Beta:
   systemctl status q-api-server
   journalctl -u q-api-server -n 20
   ```

2. **Different source port/interface** - libp2p binds differently than `nc`:
   ```bash
   # On Alpha during libp2p run:
   sudo ss -anp | grep 185.182.185.227:9001
   # Check what local port/interface is being used
   ```

3. **Connection tracking table full** (rare):
   ```bash
   sudo sysctl net.netfilter.nf_conntrack_count
   sudo sysctl net.netfilter.nf_conntrack_max
   ```

---

## 🧪 Step 3: Minimal Libp2p Dialer Test (10 minutes)

**If errors are still unclear, isolate with minimal test binary.**

Create `crates/q-network/examples/minimal_dialer.rs`:

```rust
//! Minimal libp2p dialer to test transport in isolation
//! Usage: cargo run --example minimal_dialer

use libp2p::{
    identity, noise, yamux,
    core::{transport::Transport, upgrade},
    swarm::{Swarm, SwarmEvent},
    tcp, Multiaddr, PeerId,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    println!("🔑 Local test peer ID: {}", local_peer_id);

    // Build exact same transport as main q-api-server
    let tcp_transport = tcp::async_io::Transport::new(tcp::Config::default());

    let transport = tcp_transport
        .upgrade(upgrade::Version::V1)
        .authenticate(noise::Config::new(&local_key)?)
        .multiplex(yamux::Config::default())
        .boxed();

    // Use simplest possible behaviour
    let behaviour = libp2p::ping::Behaviour::default();

    let mut swarm = Swarm::new(
        transport,
        behaviour,
        local_peer_id,
        libp2p::swarm::Config::with_tokio_executor(),
    );

    // Target Server Beta
    let target: Multiaddr = "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooWC688bzHi7djbkensGQMABzX9tY41LNasgd3g3FdwqQn7"
        .parse()?;

    println!("📡 Dialing {}", target);
    swarm.dial(target.clone())?;

    // Wait for connection result
    loop {
        match swarm.select_next_some().await {
            SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                println!("✅ SUCCESS: ConnectionEstablished");
                println!("   Peer: {}", peer_id);
                println!("   Endpoint: {:?}", endpoint);
                break;
            }
            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                eprintln!("❌ FAILURE: OutgoingConnectionError");
                eprintln!("   Peer: {:?}", peer_id);
                eprintln!("   Error: {:?}", error);
                break;
            }
            other => {
                println!("Event: {:?}", other);
            }
        }
    }

    Ok(())
}
```

**Run the minimal test:**
```bash
cd /opt/orobit/shared/q-narwhalknight
cargo run --example minimal_dialer

# If this SUCCEEDS but main q-api-server FAILS:
#   → Problem is in QNarwhalBehaviour (connection_limits, some other behaviour)
#
# If this ALSO FAILS with same error:
#   → Problem is in transport config or network layer
```

---

## 🎯 Success Criteria

**When everything works, you'll see this sequence in logs:**

```
[DEBUG] libp2p_tcp: Dialing 185.182.185.227:9001
[DEBUG] libp2p_swarm: Dialing /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWC688...
[DEBUG] libp2p_tcp: New outgoing connection to 185.182.185.227:9001
[DEBUG] libp2p_swarm: Upgrading connection to 12D3KooWC688...
[INFO]  🔗 ConnectionEstablished { peer_id: 12D3KooWC688..., endpoint: Dialer { ... } }
[DEBUG] 🤝 [HANDSHAKE] Sent handshake request to 12D3KooWC688...
[INFO]  ✅ [HANDSHAKE] Peer validated successfully
[INFO]  📊 Healthy peer connections: 1/1
```

**Failure modes will be obvious:**
- `ConnectionRefused` → Server not listening / firewall
- `TimedOut` → Firewall blocking packets
- `DNS resolution failed` → Using `/dns/` when it shouldn't
- `Connection limit` → limits too restrictive
- `Upgrade failed` / `Protocol negotiation` → Transport mismatch

---

## 📋 Quick Reference

| Symptom | Root Cause | Fix Location |
|---------|------------|--------------|
| `nc` fails | Firewall/listener | OS layer (iptables/ufw/systemd) |
| `/dns/` in dial logs | DNS multiaddr | `unified_network_manager.rs:39` |
| `ConnectionLimit` error | Limits too strict | `unified_network_manager.rs:80` |
| `Upgrade failed` | Transport mismatch | `unified_network_manager.rs:400-500` |
| `WrongPeerId` | Stale bootstrap config | `unified_network_manager.rs:39` |
| Works in minimal dialer, fails in main | Behaviour issue | `QNarwhalBehaviour` initialization |

---

## Next Steps

1. **Start with Step 1** - `nc -zv` test (30 seconds)
2. **Add OutgoingConnectionError logging** (Step 2)
3. **Run diagnosis** and match error pattern
4. **Apply targeted fix** based on pattern
5. **If still unclear**, run minimal_dialer test

**Report back with:**
- Step 1 result (TCP test pass/fail)
- Step 2 error pattern from logs
- Any specific error messages

This will pinpoint the exact issue immediately.
