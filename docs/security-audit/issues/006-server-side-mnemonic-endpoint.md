# Issue #006: Remove or lock down server-side mnemonic generation

**State**: `open`
**Priority**: HIGH
**Labels**: `security`, `wallet`, `privacy`, `key-management`
**Created**: 2026-05-17

## Finding

The API exposes `/api/v1/mnemonic` as a GET endpoint that generates seed material server-side and returns the mnemonic phrase and raw entropy to the caller.

## Evidence

- `crates/q-api-server/src/main.rs` registers `GET /api/v1/mnemonic` to `handlers::generate_mnemonic`.
- `crates/q-api-server/src/handlers.rs::generate_mnemonic` returns JSON fields including `mnemonic`, `words`, and `entropy`.
- The displayed wallet address is derived by `SHA3-256(mnemonic_phrase)` rather than a standard BIP39 seed/key derivation path.


## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/main.rs:24679` registers `GET /api/v1/mnemonic`.
- `crates/q-api-server/src/handlers.rs:7109-7136` generates entropy and a BIP39 mnemonic server-side.
- `crates/q-api-server/src/handlers.rs:7149-7155` derives a displayed wallet address by hashing the mnemonic phrase.
- `crates/q-api-server/src/handlers.rs:7157-7165` returns `mnemonic`, `words`, and raw hex `entropy` in the API response.

## Impact

A private cryptocurrency wallet should not rely on server-generated and server-transmitted seed material in production. Logs, proxies, browser history, TLS termination, or a compromised API host could expose wallet recovery material.

## Acceptance Criteria

- [ ] Production builds do not expose `/api/v1/mnemonic`.
- [ ] Mnemonics are generated client-side by default.
- [ ] If retained for development, the endpoint is gated behind an explicit dev-only flag and authentication.
- [ ] Raw entropy is never returned in API responses.
- [ ] Address derivation matches the wallet's actual BIP39 seed/key derivation path.

## Suggested Fix

Move mnemonic generation into the wallet client. Keep a dev-only diagnostic endpoint only when a clearly named environment flag is enabled.
