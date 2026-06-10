# Issue #015: io_uring Config Section + Runtime Feature Detection

**Status**: Open
**Priority**: Medium
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: enhancement, io_uring, config

## Description

`io_uring_loop.rs` has `IoUringConfig` and `probe_io_uring_features()` but they're not wired into:
1. The TOML config (`q-flux.toml` has no `[io_uring]` section)
2. Startup detection (q-flux doesn't log whether io_uring is available)

## Approach

Add `[io_uring]` section to `FluxConfig`:

```toml
[io_uring]
enabled = false              # Master toggle (default: off)
queue_depth = 4096           # SQE ring depth
buffer_count = 1024          # Pre-allocated registered buffers
buffer_size = 16384          # 16KB per buffer
splice_pipe_size = 65536     # Pipe buffer for splice (64KB)
```

At startup, call `probe_io_uring_features()` and log capabilities:
```
io_uring: available=true, multishot_accept=true, provided_buffers=true
```

If `enabled = true` but kernel doesn't support io_uring, warn and fall back.

## Files to Change
- `crates/q-flux/src/config.rs` — add `IoUringConfig` section with serde defaults
- `crates/q-flux/src/main.rs` — probe features at startup, log result
