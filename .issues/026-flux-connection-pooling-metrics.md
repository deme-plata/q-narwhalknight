# Issue #026: Upstream Connection Pool Metrics

**Status**: Planned
**Priority**: Medium
**Component**: q-flux
**Labels**: observability, prometheus

## Description

The upstream connection pool (hyper Client per worker) currently lacks visibility into pool utilization. Under high load, pool exhaustion causes 502 errors. Add metrics to expose pool state for proactive alerting.

## Metrics to Add

- `q_flux_pool_idle_connections` (gauge) — idle keepalive connections per backend
- `q_flux_pool_active_connections` (gauge) — in-use connections per backend
- `q_flux_pool_waiters` (gauge) — requests waiting for a connection
- `q_flux_pool_timeouts_total` (counter) — connection pool acquisition timeouts
- `q_flux_upstream_request_duration_seconds` (histogram) — end-to-end upstream latency

## Approach

1. Wrap hyper Client with a tracking layer that instruments connect/idle/close events
2. Export via the existing `/metrics` Prometheus endpoint
3. Use per-backend labels so operators can identify hot/cold backends

## Files to Change

- `crates/q-flux/src/upstream.rs` — pool tracking
- `crates/q-flux/src/admin.rs` — export new metrics
- `crates/q-flux/src/metrics.rs` — new metric counters
