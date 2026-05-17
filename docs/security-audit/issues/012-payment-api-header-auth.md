# Issue #012: Replace payment API header-string auth with cryptographic wallet auth

**State**: `open`
**Priority**: CRITICAL
**Labels**: `security`, `payments`, `authentication`, `usd`
**Created**: 2026-05-17

## Finding

Payment endpoints that move USD/QUGUSD trust a raw header string as the wallet identity. The helper returns `x-wallet-auth` or `Authorization: Bearer` text directly, and handlers authorize when that string equals the request wallet address. The payment rate limiter is also currently a stub that always returns `None`.

## Evidence

- `extract_wallet_from_headers` returns the raw `x-wallet-auth` header value or bearer token without signature verification.
- `withdraw_usd`, `convert_usd_to_qugusd`, and `transfer_usd` compare that raw string to the requested wallet address to authorize money movement.
- The payment rate-limit helper comments that it is not wired and returns `None`.
- `convert_usd_to_qugusd` directly debits USD storage and saves a QUGUSD token balance, rather than going through a signed/consensus-backed mint path.

## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/payment_api.rs:25-43` returns raw `x-wallet-auth` or bearer text as the wallet identity.
- `crates/q-api-server/src/payment_api.rs:97-116` documents the rate limiter as not wired and returns `None`.
- `crates/q-api-server/src/payment_api.rs:523-540` authorizes USD withdrawals by comparing the raw header-derived string to `request.wallet_address`.
- `crates/q-api-server/src/payment_api.rs:654-671` authorizes USD-to-QUGUSD conversion the same way.
- `crates/q-api-server/src/payment_api.rs:743-766` debits USD and directly saves the QUGUSD token balance.
- `crates/q-api-server/src/payment_api.rs:863-880` authorizes USD transfers by comparing raw header text to `request.from_wallet`.

## Impact

A caller can potentially spoof payment authority by sending a header whose value equals the target wallet address. If reachable in production, this can enable unauthorized USD withdrawals, USD transfers, or USD-to-QUGUSD conversion attempts, with no effective rate limiting.

## Acceptance Criteria

- [ ] Payment routes use the same `AuthenticatedWallet` extractor/canonical signed-message scheme as wallet-protected crypto routes.
- [ ] Authorization compares the authenticated wallet address bytes to parsed request addresses, not raw strings.
- [ ] Payment rate limiting is wired into `AppState` and tested.
- [ ] USD-to-QUGUSD conversion mints through an auditable, idempotent, consensus-backed path.
- [ ] Regression tests prove raw `x-wallet-auth: <wallet>` and bearer strings alone are rejected.

## Suggested Fix

Delete `extract_wallet_from_headers`, require `AuthenticatedWallet` on all money-moving payment handlers, bind signatures to method/body/nonce as in issue #002, and route QUGUSD issuance through the stablecoin state machine rather than direct token-balance writes.
