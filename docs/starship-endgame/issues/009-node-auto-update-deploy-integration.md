# Issue #009: Node Auto-Update — Deploy Script Integration

**State**: `closed`
**Priority**: HIGH
**Labels**: `auto-update`, `deploy`, `infrastructure`
**Assigned**: Beta
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

The node auto-update system (`node_auto_updater.rs`, 1033 lines) is feature-complete but `safe-deploy.sh` does NOT announce updates to the network after a successful deployment. Other nodes have no way to discover new versions via P2P gossipsub.

## Problem

After `safe-deploy.sh` deploys a new binary and restarts the service, it should:
1. Compute SHA-256 + BLAKE3 checksums of the new binary
2. POST to `localhost:8080/api/v1/admin/update/announce`
3. The node then signs the announcement with its Ed25519 key and broadcasts via gossipsub

Currently, the entire auto-update P2P pipeline is dormant because no announcements are ever published.

## Acceptance Criteria

- [x] `node_auto_updater.rs` core engine implemented (1033 lines)
- [x] Admin endpoint `POST /api/v1/admin/update/announce` implemented
- [x] Gossipsub topic `/qnk/{network}/update-announcements` subscribed
- [x] Message forwarding from gossipsub to auto-updater channel
- [x] `safe-deploy.sh` computes checksums after build
- [x] `safe-deploy.sh` calls announce endpoint after successful restart
- [ ] `ha-deploy.sh` (if it exists) also calls announce endpoint (deferred — ha-deploy calls safe-deploy)
- [x] Announce includes: version, sha256, blake3, binary_size, download_url, release_notes

## Technical Details

```bash
# After successful deployment, add to safe-deploy.sh:
SHA256=$(sha256sum target/release/q-api-server | awk '{print $1}')
BLAKE3=$(b3sum target/release/q-api-server | awk '{print $1}')
SIZE=$(stat --format=%s target/release/q-api-server)
VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')

curl -s -X POST http://localhost:8080/api/v1/admin/update/announce \
  -H "Content-Type: application/json" \
  -d "{
    \"version\": \"$VERSION\",
    \"sha256_checksum\": \"$SHA256\",
    \"blake3_checksum\": \"$BLAKE3\",
    \"binary_size\": $SIZE,
    \"download_url\": \"https://quillon.xyz/downloads/q-api-server-v$VERSION\",
    \"mandatory\": false,
    \"release_notes\": \"v$VERSION automated deployment\"
  }"
```

## Files

- `scripts/safe-deploy.sh` — Add announce call after deployment
- `crates/q-api-server/src/deploy_admin_api.rs` — Existing announce endpoint
- `crates/q-api-server/src/node_auto_updater.rs` — Auto-updater engine
