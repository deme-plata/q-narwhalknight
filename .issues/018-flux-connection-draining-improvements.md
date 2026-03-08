# Issue #018: Connection Draining Improvements

**Status**: Done (Phases 1-3)
**Priority**: Medium
**Component**: q-flux
**Labels**: reliability, deployment

## Description

Current connection draining on config reload (Issue #010) relies on Arc-based natural draining — old connections use the old TLS config, new connections get the new one. This works but has gaps:

1. **No active drain signal**: Long-lived connections (WebSocket, SSE) are never told to close
2. **No drain timeout enforcement**: Old connections can persist indefinitely
3. **No drain metrics**: No visibility into how many connections are still draining
4. **Binary upgrade**: No graceful handoff when replacing the q-flux binary itself

## Approach

### Phase 1: Drain Signal for Long-Lived Connections
- Track all active SSE and WebSocket connections in a `DashMap<ConnectionId, CancellationToken>`
- On config reload: trigger `CancellationToken::cancel()` for all tracked connections
- Connections receive the cancel signal and initiate graceful close (send close frame for WS, end stream for SSE)
- Clients reconnect automatically and get the new TLS config

### Phase 2: Drain Timeout
- After sending drain signal, start a timer (`drain_timeout_secs` from config, default 30s)
- After timeout, force-close any remaining old connections
- Log how many connections were force-closed vs gracefully drained

### Phase 3: Drain Metrics
- `q_flux_drain_active` gauge: connections currently draining
- `q_flux_drain_completed_total` counter: connections that drained gracefully
- `q_flux_drain_forced_total` counter: connections force-closed after timeout

### Phase 4: Binary Upgrade (SO_REUSEPORT handoff)
- New binary starts, binds same ports via SO_REUSEPORT
- Old binary stops accepting new connections
- Old binary drains existing connections (Phase 1-2)
- Old binary exits after drain completes or timeout

## Files to Change

- `crates/q-flux/src/proxy.rs` — track SSE/WS connections, handle cancel signal
- `crates/q-flux/src/worker.rs` — connection tracking map, drain trigger
- `crates/q-flux/src/config.rs` — drain config options
- `crates/q-flux/src/metrics.rs` — drain metrics
- `crates/q-flux/src/main.rs` — binary upgrade handoff (Phase 4)
