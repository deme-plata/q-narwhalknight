# #037: Tunnel system is dead code

**Priority**: LOW
**File(s)**: `crates/q-compute/src/tunnel.rs`, `crates/q-compute/src/orchestrator.rs`
**Risk**: Code maintenance burden with no runtime value

## Problem

`TunnelManager` is fully implemented (250+ lines of code with open/close/route/cleanup logic) but is never instantiated outside of its own unit tests. A grep of the entire codebase shows `TunnelManager::new` only appears in `tunnel.rs` test functions (lines 284, 299, 308).

The orchestrator's `ComputeStatus` has a `tunnels: Vec<TunnelInfo>` field that is always set to `Vec::new()` (orchestrator.rs line 154):

```rust
tunnels: Vec::new(), // Populated by tunnel module
```

The comment "Populated by tunnel module" indicates intent to wire this up, but it was never completed. The `TunnelWorkItem`, `TunnelPayload`, `ComputeTunnel`, and `TunnelManager` types exist but serve no runtime purpose.

Similarly, `cluster_peers: Vec<ComputePeerInfo>` is always empty (line 155).

## Fix

Two options:

**Option A (Remove):** If compute tunnels are not on the near-term roadmap, delete `tunnel.rs`, remove `pub mod tunnel` from `lib.rs`, and remove `TunnelInfo`/`TunnelType`/`ComputePeerInfo` types. This eliminates dead code maintenance.

**Option B (Wire):** If tunnels are planned:
1. Create a `TunnelManager` instance in the orchestrator.
2. Wire it into the P2P gossipsub layer to open tunnels when compute peers are discovered.
3. Populate `ComputeStatus.tunnels` from `tunnel_manager.tunnel_infos()`.
4. Route inference tasks through tunnels to remote peers.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
