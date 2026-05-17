# Issue #008: Make AEGIS localhost admin bypass fail closed

**State**: `open`
**Priority**: HIGH
**Labels**: `security`, `admin`, `authentication`, `aegis`
**Created**: 2026-05-17

## Finding

The AEGIS founder-auth middleware accepts `X-Admin-Local: true` as a localhost bypass and treats a missing `ConnectInfo<SocketAddr>` extension as local by default.

## Evidence

- `crates/q-api-server/src/aegis_auth_middleware.rs::verify_founder_signature` checks the raw `X-Admin-Local` header before parsing normal AEGIS headers.
- The locality test calls `.get::<ConnectInfo<SocketAddr>>()` and then `.unwrap_or(true)`, so missing connection metadata is treated as loopback.
- Protected Quillon Bank routes are guarded by this middleware in `crates/q-api-server/src/main.rs`, so any default-allow bypass in the middleware applies to founder-only banking routes.

## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/aegis_auth_middleware.rs:185-193` accepts `X-Admin-Local: true` and uses `.unwrap_or(true)` when `ConnectInfo` is absent.
- `crates/q-api-server/src/aegis_auth_middleware.rs:195-197` returns `Ok(next.run(request).await)` when `is_local` is true.
- `crates/q-api-server/src/main.rs:25310-25317` nests founder-only Quillon Bank protected routes behind `verify_founder_signature`.

## Impact

If the Axum service is ever run without `ConnectInfo` propagation, behind a proxy that strips socket metadata, or in a test/deployment path where the extension is missing, a remote caller can potentially satisfy founder authentication with only `X-Admin-Local: true`.

## Acceptance Criteria

- [ ] Missing `ConnectInfo` fails closed unless an explicit dev-only config flag enables local bypass.
- [ ] Local bypass only works for real loopback peer IPs and cannot be activated through forwarded headers.
- [ ] Founder/admin routes have tests proving `X-Admin-Local: true` is rejected when `ConnectInfo` is absent or non-loopback.
- [ ] Production startup logs clearly state whether local admin bypass is enabled.

## Suggested Fix

Replace `.unwrap_or(true)` with `.unwrap_or(false)`, require an explicit `Q_ENABLE_LOCAL_ADMIN_BYPASS=1`-style flag for development, and prefer a signed local admin token over a static header.
