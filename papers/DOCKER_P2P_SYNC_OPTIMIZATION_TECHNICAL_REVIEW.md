# Docker P2P Sync Optimization Technical Review

## Q-NarwhalKnight v1.2.2-beta: Achieving Full Speed Sync in Containerized Environments

**Date**: 2025-12-10
**Version**: v1.2.2-beta
**Status**: IMPLEMENTED

---

## Implementation Status

**v1.2.2-beta Implementation Complete:**

| Component | Status | File |
|-----------|--------|------|
| Address Filter Module | Implemented | `crates/q-network/src/address_filter.rs` |
| Directional Filtering | Implemented | `is_routable_peer_address()` - filters PEER addresses only |
| Network Manager Integration | Implemented | `crates/q-network/src/unified_network_manager.rs` |
| Environment Configuration | Implemented | `Q_FILTER_DOCKER_ADDRESSES`, `Q_ALLOW_PRIVATE_ADDRESSES`, `Q_EXTERNAL_ADDRESS` |
| Startup Logging | Implemented | `log_filter_configuration()` called at startup |
| Unit Tests | Implemented | 12 tests covering all filter scenarios |

**CRITICAL DESIGN NOTE - Directional Filtering:**

The filter is **DIRECTIONAL** - it ONLY filters **PEER addresses**, NOT our own listen addresses:
- `is_routable_peer_address()` - Use for PEER addresses from Identify/Kademlia
- Do NOT use for our own listen addresses (would break Identify announcements)
- Bootstrap peer addresses are NOT filtered (explicitly configured by users)

---

## Executive Summary

Docker containers running Q-NarwhalKnight nodes experience degraded P2P synchronization performance (30 b/s vs expected 50-100 b/s) due to libp2p address caching conflicts between Docker network interfaces and external addresses. This document proposes four solutions to achieve full-speed sync in containerized environments.

---

## Problem Statement

### Observed Behavior

| Metric | Native | Docker (host network) | Docker (bridge) |
|--------|--------|----------------------|-----------------|
| Sync Speed | 50-100 b/s | 30 b/s | 0 b/s (stuck) |
| Gossipsub | Working | Working | Working |
| Request-Response | Working | **Failing 50%** | **Failing 100%** |
| Address Discovery | Correct | Mixed | Incorrect |

### Root Cause Analysis

```
ERROR: Empty response buffer - peer may have closed connection prematurely
Peer: 12D3KooWC25N7aDTtYpKtC4LgvWoi3EFMj1ybzWcTzP4eaYs77c7

Cached addresses for peer:
  - /ip4/172.17.0.1/tcp/9001/p2p/12D3KooWC25N...  ← Docker bridge (INVALID)
  - /ip4/185.182.185.227/tcp/9001/p2p/12D3KooWC25N...  ← External (VALID)
```

**The problem**: libp2p's address book caches ALL discovered addresses for a peer, including Docker internal network addresses. When the request-response protocol selects a Docker internal address (172.x.x.x), the connection fails because:

1. Bridge networking: Container can't route to host's Docker bridge
2. Host networking: Address is technically reachable but points to local loopback

---

## Proposed Solutions

### Solution 1: Address Filtering at Discovery (Recommended)

**Implementation**: Filter out non-routable addresses before adding to Kademlia/address book.

```rust
// In crates/q-network/src/unified_network_manager.rs

/// Filter addresses to exclude non-routable Docker/container IPs
fn is_routable_address(addr: &Multiaddr) -> bool {
    for protocol in addr.iter() {
        if let Protocol::Ip4(ip) = protocol {
            // Filter Docker bridge networks (172.16.0.0/12)
            if ip.octets()[0] == 172 && (ip.octets()[1] >= 16 && ip.octets()[1] <= 31) {
                return false;
            }
            // Filter Docker default bridge (172.17.0.0/16)
            if ip.octets()[0] == 172 && ip.octets()[1] == 17 {
                return false;
            }
            // Filter private Class A (10.0.0.0/8) - common in container orchestration
            if ip.octets()[0] == 10 {
                return false;
            }
            // Filter link-local (169.254.0.0/16)
            if ip.octets()[0] == 169 && ip.octets()[1] == 254 {
                return false;
            }
            // Filter loopback
            if ip.is_loopback() {
                return false;
            }
        }
    }
    true
}

// Apply filter when adding peer addresses
fn add_peer_address_filtered(&mut self, peer_id: &PeerId, addr: Multiaddr) {
    if is_routable_address(&addr) {
        self.swarm.behaviour_mut().kademlia.add_address(peer_id, addr.clone());
        self.swarm.add_peer_address(*peer_id, addr);
    } else {
        debug!("🚫 [ADDR-FILTER] Skipping non-routable address: {}", addr);
    }
}
```

**Pros**: Simple, surgical fix. No protocol changes.
**Cons**: May filter legitimate private network deployments.
**Performance Impact**: Full speed (50-100 b/s expected)

---

### Solution 2: External Address Announcement via Environment Variable

**Implementation**: Allow explicit external address configuration.

```rust
// In crates/q-network/src/unified_network_manager.rs

fn configure_external_address(swarm: &mut Swarm<QNarwhalBehaviour>) {
    // Q_EXTERNAL_ADDRESS=/ip4/185.182.185.227/tcp/9001
    if let Ok(external_addr) = std::env::var("Q_EXTERNAL_ADDRESS") {
        if let Ok(addr) = external_addr.parse::<Multiaddr>() {
            swarm.add_external_address(addr.clone());
            info!("📢 [EXTERNAL] Configured external address: {}", addr);
        }
    }

    // Q_FILTER_DOCKER_ADDRESSES=true (default for containers)
    if std::env::var("Q_FILTER_DOCKER_ADDRESSES").unwrap_or_default() == "true" {
        // Enable address filtering
    }
}
```

**Docker Compose Example**:
```yaml
services:
  q-node:
    image: quillon/q-api-server:v1.2.2-beta
    environment:
      - Q_EXTERNAL_ADDRESS=/ip4/185.182.185.227/tcp/9001
      - Q_FILTER_DOCKER_ADDRESSES=true
    network_mode: host
```

**Pros**: User-configurable, works with any network topology.
**Cons**: Requires manual configuration.
**Performance Impact**: Full speed when configured correctly.

---

### Solution 3: Gossipsub-Based Block Sync Fallback

**Implementation**: When request-response fails, fall back to gossipsub broadcast for block sync.

```rust
// In crates/q-storage/src/turbo_sync.rs

pub async fn sync_with_fallback(
    &self,
    target_height: u64,
    network_manager: &UnifiedNetworkManager,
) -> Result<SyncResult> {
    // Try request-response first (fast path)
    match self.sync_via_request_response(target_height).await {
        Ok(result) => return Ok(result),
        Err(e) if e.is_connection_error() => {
            warn!("🔄 [FALLBACK] Request-response failed, using gossipsub sync");
        }
        Err(e) => return Err(e),
    }

    // Fallback: Request blocks via gossipsub (slower but more reliable)
    let request = BlockSyncRequest {
        start_height: self.current_height(),
        end_height: target_height,
        requester_peer_id: network_manager.local_peer_id().to_string(),
    };

    // Publish request to sync topic
    network_manager.publish_to_topic(
        "/qnk/testnet-phase15/sync-requests",
        &bincode::serialize(&request)?,
    ).await?;

    // Wait for responses via gossipsub
    self.wait_for_gossipsub_blocks(target_height, Duration::from_secs(30)).await
}
```

**New Gossipsub Topics**:
- `/qnk/{network}/sync-requests` - Block sync requests
- `/qnk/{network}/sync-responses` - Block sync responses (chunked)

**Pros**: Works through any NAT/container configuration.
**Cons**: Higher bandwidth (broadcast vs unicast), slower.
**Performance Impact**: 20-40 b/s (degraded but reliable).

---

### Solution 4: Hybrid Protocol with Address Scoring

**Implementation**: Score addresses by reliability and prefer high-scoring addresses.

```rust
// In crates/q-network/src/address_scorer.rs

#[derive(Default)]
pub struct AddressScorer {
    scores: HashMap<(PeerId, Multiaddr), AddressScore>,
}

#[derive(Clone)]
pub struct AddressScore {
    pub success_count: u32,
    pub failure_count: u32,
    pub last_success: Option<Instant>,
    pub address_type: AddressType,
}

#[derive(Clone, PartialEq)]
pub enum AddressType {
    PublicIPv4,      // Best: 185.x.x.x
    PublicIPv6,      // Good: 2001:x::
    PrivateClass192, // OK: 192.168.x.x (home networks)
    PrivateClass172, // Bad: 172.x.x.x (Docker)
    PrivateClass10,  // Bad: 10.x.x.x (Kubernetes)
    Loopback,        // Worst: 127.x.x.x
}

impl AddressScorer {
    pub fn get_best_address(&self, peer_id: &PeerId) -> Option<Multiaddr> {
        self.scores
            .iter()
            .filter(|((pid, _), _)| pid == peer_id)
            .max_by(|(_, a), (_, b)| a.reliability_score().cmp(&b.reliability_score()))
            .map(|((_, addr), _)| addr.clone())
    }

    pub fn record_success(&mut self, peer_id: &PeerId, addr: &Multiaddr) {
        let score = self.scores.entry((peer_id.clone(), addr.clone())).or_default();
        score.success_count += 1;
        score.last_success = Some(Instant::now());
    }

    pub fn record_failure(&mut self, peer_id: &PeerId, addr: &Multiaddr) {
        let score = self.scores.entry((peer_id.clone(), addr.clone())).or_default();
        score.failure_count += 1;
    }
}

impl AddressScore {
    fn reliability_score(&self) -> i32 {
        let type_bonus = match self.address_type {
            AddressType::PublicIPv4 => 100,
            AddressType::PublicIPv6 => 90,
            AddressType::PrivateClass192 => 50,
            AddressType::PrivateClass172 => -100, // Heavily penalize Docker
            AddressType::PrivateClass10 => -100,  // Heavily penalize Kubernetes
            AddressType::Loopback => -1000,
        };

        let success_rate = if self.success_count + self.failure_count > 0 {
            (self.success_count as f32 / (self.success_count + self.failure_count) as f32 * 100.0) as i32
        } else {
            50 // Unknown
        };

        type_bonus + success_rate
    }
}
```

**Integration Point**:
```rust
// Before sending request-response
let best_addr = address_scorer.get_best_address(&peer_id);
if let Some(addr) = best_addr {
    swarm.dial(addr)?;
}
```

**Pros**: Adaptive, learns from failures, self-optimizing.
**Cons**: More complex, requires state management.
**Performance Impact**: Gradually improves to full speed after learning period.

---

## Recommended Implementation Plan

### Phase 1: Quick Fix (v1.2.2-beta)

Implement **Solution 1** (Address Filtering) + **Solution 2** (Environment Config):

```rust
// crates/q-network/src/address_filter.rs

use libp2p::Multiaddr;
use libp2p::multiaddr::Protocol;
use std::net::Ipv4Addr;

/// Check if an address is routable for P2P connections
pub fn is_routable(addr: &Multiaddr) -> bool {
    // Check Q_ALLOW_PRIVATE_ADDRESSES env var for private network deployments
    let allow_private = std::env::var("Q_ALLOW_PRIVATE_ADDRESSES")
        .map(|v| v == "true")
        .unwrap_or(false);

    for protocol in addr.iter() {
        match protocol {
            Protocol::Ip4(ip) => {
                // Always filter loopback
                if ip.is_loopback() {
                    return false;
                }

                // Filter Docker/container networks unless explicitly allowed
                if !allow_private {
                    // Docker bridge: 172.16.0.0/12
                    if is_docker_network(ip) {
                        return false;
                    }
                    // Kubernetes/cloud: 10.0.0.0/8
                    if ip.octets()[0] == 10 {
                        return false;
                    }
                    // Link-local: 169.254.0.0/16
                    if ip.octets()[0] == 169 && ip.octets()[1] == 254 {
                        return false;
                    }
                }
            }
            _ => {}
        }
    }
    true
}

fn is_docker_network(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    // Docker default bridge: 172.17.0.0/16
    if octets[0] == 172 && octets[1] == 17 {
        return true;
    }
    // Docker custom networks: 172.16.0.0/12 (172.16-31.x.x)
    if octets[0] == 172 && (octets[1] >= 16 && octets[1] <= 31) {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filters_docker_bridge() {
        let addr: Multiaddr = "/ip4/172.17.0.1/tcp/9001".parse().unwrap();
        assert!(!is_routable(&addr));
    }

    #[test]
    fn test_allows_public_ip() {
        let addr: Multiaddr = "/ip4/185.182.185.227/tcp/9001".parse().unwrap();
        assert!(is_routable(&addr));
    }

    #[test]
    fn test_filters_kubernetes() {
        let addr: Multiaddr = "/ip4/10.244.1.5/tcp/9001".parse().unwrap();
        assert!(!is_routable(&addr));
    }
}
```

### Phase 2: Robustness (v1.2.3-beta)

Add **Solution 3** (Gossipsub Fallback) for environments where filtering isn't sufficient.

### Phase 3: Intelligence (v1.3.0)

Implement **Solution 4** (Address Scoring) for fully adaptive networking.

---

## Docker Deployment Guide (Post-Fix)

### Recommended Configuration

```bash
# Run with host networking (best performance)
docker run -d \
  --name q-node \
  --network=host \
  -e Q_DB_PATH=/data \
  -e Q_NETWORK_ID=testnet-phase15 \
  -e Q_P2P_PORT=9001 \
  -e Q_FILTER_DOCKER_ADDRESSES=true \
  -v /opt/q-data:/data \
  quillon/q-api-server:v1.2.2-beta

# Alternative: Bridge networking with explicit external address
docker run -d \
  --name q-node \
  -p 9001:9001 \
  -p 8080:8080 \
  -e Q_DB_PATH=/data \
  -e Q_NETWORK_ID=testnet-phase15 \
  -e Q_P2P_PORT=9001 \
  -e Q_EXTERNAL_ADDRESS=/ip4/YOUR_PUBLIC_IP/tcp/9001 \
  -v /opt/q-data:/data \
  quillon/q-api-server:v1.2.2-beta
```

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: q-node
spec:
  hostNetwork: true  # Required for P2P
  containers:
  - name: q-api-server
    image: quillon/q-api-server:v1.2.2-beta
    env:
    - name: Q_FILTER_DOCKER_ADDRESSES
      value: "true"
    - name: Q_EXTERNAL_ADDRESS
      valueFrom:
        fieldRef:
          fieldPath: status.hostIP
    ports:
    - containerPort: 9001
      hostPort: 9001
    - containerPort: 8080
      hostPort: 8080
```

---

## Expected Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Docker (host) sync | 30 b/s | **50-100 b/s** |
| Docker (bridge) sync | 0 b/s | **30-50 b/s** |
| Request-response success | 50% | **95%+** |
| Sync completion time | ~30 min | **~10 min** |

---

## Conclusion

The Docker P2P sync performance issue is caused by libp2p caching non-routable Docker network addresses. The recommended fix combines:

1. **Address filtering** to exclude Docker/Kubernetes internal addresses
2. **Environment configuration** for explicit external address announcement
3. **Fallback mechanisms** for maximum reliability

Implementation requires ~200 lines of new code in `q-network` crate with no protocol-breaking changes.

---

**Author**: Q-NarwhalKnight Development Team
**Review Status**: IMPLEMENTED (v1.2.2-beta)
**Implementation Date**: 2025-12-10

## Changelog

### v1.2.2-beta (2025-12-10)
- Created `address_filter.rs` module with comprehensive Docker/container address filtering
- Added `is_routable_peer_address()` function for directional PEER address filtering
- Added `score_address()` function for address quality scoring
- Added `filter_peer_addresses()` and `get_best_address()` helper functions
- Integrated filtering into `unified_network_manager.rs` ConnectionEstablished handler
- Added startup logging via `log_filter_configuration()`
- Added environment variable configuration: `Q_FILTER_DOCKER_ADDRESSES`, `Q_ALLOW_PRIVATE_ADDRESSES`, `Q_EXTERNAL_ADDRESS`
- **CRITICAL FIX**: Added `swarm.add_external_address()` call to register external address with libp2p
- This ensures Identify protocol announces ONLY the public IP, preventing Docker address caching
- 12 unit tests covering all filtering scenarios
