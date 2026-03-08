# Issue #004: Auto-Tier Classification from Traffic Patterns

**Status**: Done
**Priority**: Low
**Component**: q-flux
**Assignee**: Server Beta
**Labels**: enhancement, libp2p

## Description

Unknown peers default to `PeerTier::Unknown` (10 Mbps, 1 connection). Based on observed behavior, peers should be auto-promoted:

- Connection >1hr + >100MB transferred → `Miner` (50 Mbps, 2 conns)
- Peer ID prefix matches known validator pattern → `Validator` (200 Mbps, 4 conns)
- Never auto-promote to Bootstrap/Supernode (pre-seeded only)

## Files to Change
- `crates/q-flux/src/worker.rs` — periodic task scanning PeerTracker
- `crates/q-flux/src/libp2p_aware.rs` — add `promote_tier()` method

## Depends On
- Issue #003 (needs byte tracking to measure transferred data)
