# Issue #010: Node Auto-Update — Systemd Config + SIGUSR1 Handler

**State**: `closed`
**Priority**: HIGH
**Labels**: `auto-update`, `systemd`, `signal-handling`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

The auto-update system needs two things to be production-ready:

1. **Systemd environment variables** — `Q_AUTO_UPDATE=1` must be set in the service file for the auto-updater to be active. Bootstrap nodes also need `Q_BOOTSTRAP_NODE=1` for co-signing.

2. **SIGUSR1 signal handler** — The auto-updater calls `apply_update()` which needs a way to trigger graceful restart. SIGUSR1 is the standard Unix pattern for this.

## Acceptance Criteria

- [ ] `Q_AUTO_UPDATE=1` added to Beta's systemd service file
- [ ] `Q_BOOTSTRAP_NODE=1` added to Beta's systemd service file
- [ ] SIGUSR1 handler in main.rs that triggers graceful shutdown
- [ ] Graceful shutdown: stop accepting new connections, flush pending writes, exit
- [ ] Gamma and Epsilon service files also updated (after testing on Beta)

## Technical Details

### Systemd Service Addition
```ini
[Service]
# ... existing env vars ...
Environment="Q_AUTO_UPDATE=1"
Environment="Q_BOOTSTRAP_NODE=1"
Environment="Q_AUTO_UPDATE_CHECK_INTERVAL=300"
Environment="Q_AUTO_UPDATE_RESTART_DELAY=30"
Environment="Q_AUTO_UPDATE_ROLLBACK_TIMEOUT=60"
Environment="Q_AUTO_UPDATE_MIN_PEERS=2"
```

### SIGUSR1 Handler (main.rs)
```rust
// Register SIGUSR1 for graceful restart (used by auto-updater)
let mut sigusr1 = tokio::signal::unix::signal(
    tokio::signal::unix::SignalKind::user_defined1()
)?;
tokio::spawn(async move {
    sigusr1.recv().await;
    info!("Received SIGUSR1 — initiating graceful restart for auto-update");
    // Set shutdown flag, wait for in-flight requests, exit
    shutdown_flag.store(true, Ordering::SeqCst);
});
```

## Files

- `/etc/systemd/system/q-api-server.service` — Add env vars
- `crates/q-api-server/src/main.rs` — SIGUSR1 handler
- `crates/q-api-server/src/node_auto_updater.rs` — Uses SIGUSR1 in apply_update()
