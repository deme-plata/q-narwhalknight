# Issue #028: Graceful Upstream Health Checks

**Status**: Done
**Priority**: High
**Component**: q-flux
**Labels**: reliability, health

## Description

The current HealthMap tracks backend health reactively (on request failure). Add proactive health checks that periodically probe upstream backends and super-cluster peers, with configurable check intervals, thresholds, and health endpoints.

## Approach

1. Spawn a background task per upstream backend
2. Periodically send HTTP GET to configured health endpoint (default: `/health`)
3. Track consecutive failures and mark backend unhealthy after threshold
4. Gradually restore traffic with "half-open" state (send 10% traffic, check success rate)
5. Support TCP-only checks (SYN probe) for non-HTTP backends
6. Different intervals for local backends (5s) vs cluster peers (30s)

## Config Example

```toml
[health_check]
interval_secs = 5
timeout_ms = 2000
unhealthy_threshold = 3
healthy_threshold = 2
path = "/health"
```

## Benefits

- Detect backend failures before user traffic hits them
- Faster failover: pre-computed health state vs first-request detection
- Configurable sensitivity: tight for local backends, loose for cluster peers

## Files to Change

- `crates/q-flux/src/health.rs` — proactive health check loop
- `crates/q-flux/src/config.rs` — health check config
- `crates/q-flux/src/upstream.rs` — use HealthMap for routing decisions
- `crates/q-flux/src/admin.rs` — expose health check results in /status
