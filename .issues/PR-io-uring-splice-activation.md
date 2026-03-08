# PR: q-flux io_uring Splice Zero-Copy Activation

## Summary

- Add upstream retry metrics (`upstream_retries`, `upstream_retry_successes`) for observability
- Fix h2_proxy.rs body clone bug in mining 503 retry path
- Close 3 issues, defer 3 low-priority complex issues
- Create 3 new issues for io_uring splice zero-copy activation roadmap
- Add global upstream semaphore config + structural config validation

## Changes

### Retry Metrics (Issue #013) - DONE
- `metrics.rs`: 2 new atomic counters, methods, snapshot fields
- `proxy.rs`: `metrics.upstream_retry()` on attempt, `upstream_retry_success()` on win
- `h2_proxy.rs`: Same for h2 503 mining retry path
- `admin.rs`: Prometheus `q_flux_upstream_retries_total` + `q_flux_upstream_retry_successes_total`

### Global Upstream Semaphore
- `config.rs`: `max_upstream_global` (default 512) - shared semaphore across all workers
- Prevents death spiral: 48 workers x 64 per-worker = 3072 concurrent → capped to 512

### Config Validation
- `config.rs`: `FluxConfig::validate()` - structural checks without filesystem access

### io_uring Splice Issues Created
- **#014**: Splice zero-copy for WebSocket/SSE passthrough (High)
- **#015**: io_uring config section + runtime feature detection (Medium)
- **#016**: Splice zero-copy metrics (Low)

### Issue Triage
- **#010 Done**: TLS drain watcher implemented
- **#013 Done**: Retry metrics (this PR)
- **#006, #007, #008 Deferred**: Complex, low-priority

## Issue Board

| # | Title | Status |
|---|-------|--------|
| 001-005 | Core features | Done |
| 006-008 | Complex libp2p/cluster | Deferred |
| 009-013 | Observability + reliability | Done |
| 014-016 | io_uring splice activation | Open |

## Test Results
- 108 q-flux tests pass
- Clean compilation (0 errors)

## io_uring Splice Architecture (Issues #014-016)

```
Current (bandwidth_limited_copy):
  client_socket → [16KB userspace buf] → upstream_socket
  2 copies per chunk (kernel→user→kernel)

Splice zero-copy (Issue #014):
  client_socket → [kernel pipe] → upstream_socket
  0 copies (data stays in kernel)

io_uring_loop.rs already has:
  - SpliceChannel (pipe management, cleanup)
  - splice_one_direction() (libc::splice wrapper)
  - splice_bidirectional() (full duplex)
  - BufferPool (registered buffers)
  - IoUringAcceptor (multishot accept)
  - Feature detection (probe_io_uring_features)
```

Key constraint: Splice only works with raw fds (plain TCP). TLS streams
need userspace decryption. The win is on the **upstream side** (q-flux →
backend on localhost is cleartext).
