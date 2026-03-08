# Issue #009: Prometheus Metrics for libp2p Peers

**Status**: Done
**Priority**: Medium
**Component**: q-flux
**Assignee**: Server Beta
**Labels**: enhancement, observability, prometheus

## Description

The `/metrics` endpoint exposes global counters but no per-peer or per-tier breakdowns. Add Prometheus gauges/counters for:

- `qflux_libp2p_peers_total{tier="Bootstrap"}` — peer count by tier
- `qflux_libp2p_connections_active{tier="Miner"}` — active connections by tier
- `qflux_libp2p_bytes_total{direction="rx",tier="Bootstrap"}` — bytes by tier
- `qflux_libp2p_circuit_breaker_open` — number of peers with open breakers
- `qflux_libp2p_bandwidth_limited_total` — times bandwidth limit triggered

## Files to Change
- `crates/q-flux/src/metrics.rs` — add libp2p-specific counters
- `crates/q-flux/src/admin.rs` — render new metrics in /metrics endpoint

## Depends On
- Issue #003 (byte tracking) and #005 (peer stats endpoint)
