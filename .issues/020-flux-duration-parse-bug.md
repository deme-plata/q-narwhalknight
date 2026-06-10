# Issue #020: Duration Parser Bug — "ms" Matched by "s" Suffix

**Priority**: Medium
**Status**: Done
**Labels**: bug, config

## Problem

The `parse_duration()` function in `config.rs` checked suffixes in wrong order:

```rust
// OLD (broken):
if let Some(secs) = s.strip_suffix('s') {     // ← matches "100ms" (strips to "100m")
    ...
} else if let Some(ms) = s.strip_suffix("ms") { // ← never reached for "ms" inputs
    ...
}
```

`"100ms"` was matched by `strip_suffix('s')` first (since "ms" ends with 's'), producing `"100m"` which failed to parse as u64.

## Fix

Reorder: check `"ms"` before `'s'`:

```rust
// NEW (fixed):
if let Some(ms) = s.strip_suffix("ms") {       // ← check multi-char suffix first
    ...
} else if let Some(secs) = s.strip_suffix('s') {
    ...
}
```

Added test `test_parse_duration_variants` covering all 4 formats: `30s`, `100ms`, `5m`, `60` (raw seconds).

## Impact

Any config value using milliseconds (e.g., `health_check_timeout = "500ms"`) would fail to parse on startup. All default values use the `"Xs"` format, so this only affects users who explicitly set millisecond timeouts.
