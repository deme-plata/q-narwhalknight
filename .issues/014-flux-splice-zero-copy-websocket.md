# Issue #014: Splice Zero-Copy for WebSocket Passthrough

**Status**: Open
**Priority**: High
**Component**: q-flux
**Assignee**: Unassigned
**Labels**: performance, io_uring, zero-copy

## Description

WebSocket passthrough currently uses `bandwidth_limited_copy()` which does explicit `read()` into a 16KB userspace buffer then `write()` out — 2 kernel-user copies per chunk. With 400+ miners holding persistent WebSocket connections, this burns CPU on data copying.

Linux `splice(2)` moves data kernel-side via a pipe intermediary: `socket_a → pipe → socket_b` with zero userspace copies. The `SpliceChannel` and `splice_bidirectional()` are already fully implemented in `io_uring_loop.rs` (lines 614-836) but not wired into the WebSocket path.

## Approach

1. After WebSocket upgrade, extract raw fd from both client and upstream `TcpStream`
2. Create `SpliceChannel::new(65536)` for the connection
3. Run `splice_bidirectional()` in a loop on a blocking thread (splice is sync)
4. Fall back to `bandwidth_limited_copy()` if splice fails (e.g., TLS streams have no raw fd)
5. Gate behind config: `[server] zero_copy = true` (default false until validated)

## Key Constraint

Splice only works with raw fds (plain TCP). TLS streams (rustls `StreamOwned`) don't expose a raw fd — the data must pass through the TLS layer in userspace. So splice applies to:
- **Internal connections** (q-flux → backend on localhost, already cleartext)
- **NOT** client-facing connections (encrypted TLS)

This means splice benefits the **upstream half** only if both sides are plain TCP. For the common case (TLS client → cleartext upstream), the upstream→pipe→client direction still needs userspace TLS encryption.

The real win is SSE/WebSocket on the **upstream side** — avoiding double-copy between q-flux and `127.0.0.1:8080`.

## Files to Change
- `crates/q-flux/src/proxy.rs` — add splice path in `handle_websocket_upgrade()`
- `crates/q-flux/src/proxy.rs` — add splice path in `handle_sse_direct()`
- `crates/q-flux/src/config.rs` — add `zero_copy` toggle to ServerConfig

## Depends On
- io_uring_loop.rs SpliceChannel (already implemented)
