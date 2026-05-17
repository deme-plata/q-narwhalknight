# Issue #002: Bind wallet auth to method, body, query, and one-time nonce

**State**: `open`
**Priority**: HIGH
**Labels**: `security`, `authentication`, `replay`
**Created**: 2026-05-17

## Finding

Wallet authentication signs `SHA3-256(address + timestamp + request_path)` and accepts timestamps within a short window, but it does not bind the signature to HTTP method, query string, mutating request body, or a one-time nonce.

## Evidence

- `crates/q-api-server/src/wallet_auth.rs::AuthenticatedWallet::from_request_parts` builds `backend_path` from the original URI path, not the full URI including query.
- The signed message hash only includes address, timestamp, and path.
- The timestamp check is window-based; no used-nonce cache was found in the wallet-auth path.


## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/wallet_auth.rs:209` documents the signed format as `SHA3-256(address + timestamp + request_path)`.
- `crates/q-api-server/src/wallet_auth.rs:213-217` derives `backend_path` from `.path()`, which omits query parameters.
- `crates/q-api-server/src/wallet_auth.rs:218-222` hashes address, timestamp, and path only; method, body hash, and nonce are absent from the signed message.

## Impact

Any captured `X-Wallet-Auth` header can be replayed against the same path while the timestamp is valid. For POST/PUT-style APIs, a body-changing request is not cryptographically tied to the wallet signature.

## Acceptance Criteria

- [ ] The canonical signed message includes wallet address, timestamp, HTTP method, full original URI including query, request body hash for mutating methods, and a nonce/challenge ID.
- [ ] The server stores short-lived used nonces and rejects replays.
- [ ] Client signing code is updated to the same canonical format.
- [ ] Tests cover replay rejection, method mismatch, query mismatch, and changed-body rejection.

## Suggested Fix

Add a wallet-auth challenge endpoint or client-provided unique nonce recorded on first use. Canonicalize method/URI/body-hash consistently on both frontend and backend.
