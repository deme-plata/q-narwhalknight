# PR #003: Node Auto-Update — Deploy Integration & Production Wiring

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `auto-update`, `infrastructure`, `deploy`
**Closes**: #009, #010, #011

---

## Summary

Wires the existing node auto-update system (1033 lines, already implemented) into the production deployment pipeline. The auto-updater engine, gossipsub topic, and admin API endpoints already exist — this PR connects the remaining pieces.

### What's included

- **Deploy script integration** — `safe-deploy.sh` now announces updates to the P2P network after successful deployment, computing SHA-256 + BLAKE3 checksums
- **Systemd configuration** — `Q_AUTO_UPDATE=1` and `Q_BOOTSTRAP_NODE=1` environment variables enabled
- **SIGUSR1 handler** — Graceful restart signal for auto-update binary swap
- **Missing types** — `AnnounceUpdateRequest`/`AnnounceUpdateResponse` structs
- **Email notification stub** — `send_update_notification_email()` logs instead of panicking

## Existing Infrastructure (already implemented, not in this PR)

| Component | File | Lines |
|-----------|------|-------|
| Auto-updater engine | `node_auto_updater.rs` | 1033 |
| Update announcement protocol | `update_announcement.rs` | ~300 |
| Admin API endpoints | `deploy_admin_api.rs` | ~200 |
| Rollback script | `rollback-check.sh` | ~60 |
| Gossipsub topic | `main.rs:2860` | Subscribed |
| Message forwarding | `main.rs:13249` | Wired |

## Files Changed (this PR)

| File | Change |
|------|--------|
| `scripts/safe-deploy.sh` | MODIFIED — Add announce call after deployment |
| `crates/q-api-server/src/main.rs` | MODIFIED — SIGUSR1 handler |
| `crates/q-api-server/src/deploy_admin_api.rs` | MODIFIED — Missing types + email stub |
| `/etc/systemd/system/q-api-server.service` | MODIFIED — Auto-update env vars |

## Test Plan

- [ ] `cargo check --package q-api-server` — compiles clean
- [ ] `cargo test --package q-api-server` — no regression
- [ ] Manual: Deploy via `safe-deploy.sh`, verify announce endpoint is called
- [ ] Manual: Check `journalctl` for "auto-update" announcement log
- [ ] Manual: Verify SIGUSR1 triggers graceful shutdown
- [ ] Verify: Rollback script still works (restart_marker detection)

## Risk Assessment

- **Consensus impact**: ZERO — auto-update is purely operational
- **Default state**: Auto-update is opt-in (`Q_AUTO_UPDATE=0` by default)
- **Safety**: 2-of-3 quorum, dual-hash verify, preflight check, 60s rollback watchdog
- **Rollback**: If auto-update fails, rollback-check.sh restores previous binary
