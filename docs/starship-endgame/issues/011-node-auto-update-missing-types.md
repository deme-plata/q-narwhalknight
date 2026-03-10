# Issue #011: Node Auto-Update — Missing Types and Email Stub

**State**: `closed` (no changes needed — types already exist)
**Priority**: MEDIUM
**Labels**: `auto-update`, `bugfix`, `types`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

The auto-update admin API (`deploy_admin_api.rs`) references types and functions that may not be defined:

1. `AnnounceUpdateRequest` / `AnnounceUpdateResponse` — Used in `admin_announce_update()` but may not be defined
2. `send_update_notification_email()` — Called after announcing but function doesn't exist (would panic)
3. Bootstrap co-signing — `is_bootstrap` flag is read but nodes don't actually co-sign foreign announcements

## Acceptance Criteria

- [ ] `AnnounceUpdateRequest` and `AnnounceUpdateResponse` structs defined with serde derives
- [ ] `send_update_notification_email()` implemented as a no-op stub (log instead of panic)
- [ ] Bootstrap co-signing: when a bootstrap node receives a valid quorum announcement for a version it's already running, re-sign and re-broadcast
- [ ] All code compiles without errors

## Files

- `crates/q-api-server/src/deploy_admin_api.rs` — Types + email stub
- `crates/q-api-server/src/node_auto_updater.rs` — Co-signing logic
- `crates/q-types/src/update_announcement.rs` — Types if needed
