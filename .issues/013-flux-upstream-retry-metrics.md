# Issue #013: Upstream Retry Metrics

**Status**: Done
**Priority**: Medium
**Component**: q-flux
**Assignee**: Server Beta
**Labels**: enhancement, observability, prometheus

## Description

The retry logic (Issue #011) had no observability — operators couldn't tell how often retries fire or whether they actually help. Added two new atomic counters:

- `upstream_retries` — total retry attempts
- `upstream_retry_successes` — retries that succeeded (saved a 502)

Exposed in:
- Prometheus `/metrics` endpoint as `q_flux_upstream_retries_total` and `q_flux_upstream_retry_successes_total`
- JSON `/status` endpoint
- `MetricsSnapshot` Display format

## Files Changed
- `crates/q-flux/src/metrics.rs` — new fields in MetricsInner, MetricsSnapshot, new methods
- `crates/q-flux/src/proxy.rs` — call `upstream_retry()` and `upstream_retry_success()`
- `crates/q-flux/src/admin.rs` — Prometheus + JSON export
