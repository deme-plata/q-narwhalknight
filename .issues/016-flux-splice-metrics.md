# Issue #016: Splice Zero-Copy Metrics

**Status**: Open
**Priority**: Low
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: observability, io_uring, prometheus

## Description

When splice zero-copy is activated (Issue #014), operators need visibility into:
- How many connections use splice vs fallback copy
- Bytes transferred via splice (kernel-side, no userspace touch)
- Splice failures (fd type mismatch, pipe errors)

## Metrics to Add

```
q_flux_splice_connections_active    — gauge: current splice connections
q_flux_splice_bytes_total           — counter: bytes moved via splice(2)
q_flux_splice_fallbacks_total       — counter: times splice failed, fell back to copy
```

## Files to Change
- `crates/q-flux/src/metrics.rs` — new counters
- `crates/q-flux/src/admin.rs` — Prometheus + JSON export
- `crates/q-flux/src/proxy.rs` — record splice vs fallback

## Depends On
- Issue #014 (splice wiring)
