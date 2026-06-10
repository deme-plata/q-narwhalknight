# Issue #008: Weighted Routing for Super-Cluster Peers

**Status**: Open
**Priority**: Low
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: enhancement, super-cluster

## Description

Super-cluster failover currently uses simple round-robin across cluster peers. Add latency-weighted routing so faster peers get more traffic during failover.

## Approach
- Track response latency per cluster peer (rolling average)
- Use weighted random selection based on inverse latency
- Peers with lower latency get proportionally more requests

## Files to Change
- `crates/q-flux/src/upstream.rs` — add latency tracking to forward(), weighted selection in next_backend()
- `crates/q-flux/src/health.rs` — record probe latency alongside health status
