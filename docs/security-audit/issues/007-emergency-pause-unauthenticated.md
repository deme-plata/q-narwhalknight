# Issue #007: Require founder authentication for emergency pause and resume

**State**: `open`
**Priority**: CRITICAL
**Labels**: `security`, `availability`, `admin`, `authentication`
**Created**: 2026-05-17

## Finding

Emergency pause and resume endpoints are registered as public POST routes, while the handlers only validate timestamp and non-empty reason. The code contains TODO comments for founder signature verification.

## Evidence

- `crates/q-api-server/src/main.rs` registers `POST /api/v1/emergency/pause` and `POST /api/v1/emergency/resume` directly on the main public router.
- `crates/q-api-server/src/handlers.rs::activate_emergency_pause` validates timestamp and reason, then sets `state.emergency_paused` to true.
- `crates/q-api-server/src/handlers.rs::resume_from_pause` validates timestamp and reason, then clears the pause flag.
- `crates/q-api-server/src/handlers.rs` explicitly notes `TODO: Verify founder signature` / `TODO: Verify founder signature using AEGIS-QL auth`.


## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/main.rs:24795-24797` registers emergency status, pause, and resume on the main router.
- `crates/q-api-server/src/handlers.rs:17228-17258` validates timestamp/reason, leaves founder verification as TODO, and sets `state.emergency_paused` to true.
- `crates/q-api-server/src/handlers.rs:17278-17314` validates timestamp/reason, leaves founder verification as TODO, and clears the pause flag.

## Impact

Any network caller who can reach the API can halt block production and transaction acceptance, then resume it, by supplying a fresh timestamp and reason. This is a direct availability and governance-control risk.

## Acceptance Criteria

- [ ] Pause and resume routes are protected by the existing founder/admin authentication middleware or equivalent multisig verification.
- [ ] The signed message covers operation, timestamp, reason, nonce, and route.
- [ ] Replayed pause/resume signatures are rejected.
- [ ] Tests prove anonymous callers cannot pause/resume and authorized founder/multisig callers can.

## Suggested Fix

Nest these endpoints under an admin router guarded by `verify_founder_signature`, or add a dedicated emergency-action verifier that requires multisig/founder authorization and nonce replay protection.
