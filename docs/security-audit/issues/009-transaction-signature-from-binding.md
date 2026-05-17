# Issue #009: Bind transaction signatures to the debited `from` address

**State**: `open`
**Priority**: CRITICAL
**Labels**: `security`, `transactions`, `signature`, `consensus`
**Created**: 2026-05-17

## Finding

`Transaction::verify_signature` verifies Ed25519 signatures using a public key pulled from `tx.data` when present, without requiring that key to correspond to `tx.from`. Balance application later debits `tx.from`.

## Evidence

- `crates/q-types/src/lib.rs::verify_ed25519_signature` extracts the public key from `self.data[..32]` when the data field is at least 32 bytes, otherwise it falls back to `self.from`.
- `crates/q-types/src/lib.rs::verify_ed25519_signature` verifies only the transaction hash against that extracted public key.
- `crates/q-api-server/src/handlers.rs::submit_transaction` accepts direct submitted transactions after `request.transaction.verify_signature()`.
- `crates/q-storage/src/balance_consensus.rs` debits `block_tx.from` for both token and native transfers.

## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-types/src/lib.rs:2870-2878` selects the Ed25519 public key from the first 32 bytes of `tx.data` when available.
- `crates/q-types/src/lib.rs:2896-2900` verifies `self.hash()` against that public key and signature.
- `crates/q-api-server/src/handlers.rs:2593-2614` accepts `/api/v1/transactions` submissions after `verify_signature()` succeeds.
- `crates/q-storage/src/balance_consensus.rs:868-887` subtracts token balances from `block_tx.from`.
- `crates/q-storage/src/balance_consensus.rs:908-912` subtracts native QUG balances from `block_tx.from`.

## Impact

A transaction can carry an arbitrary public key in `data`, produce a valid signature for that key, and still name a different `from` address for balance debiting unless another layer rejects it. Signature validity must prove authority over the account being debited.

## Acceptance Criteria

- [ ] Signature verification derives the authorized account address from the verifying key and requires it to equal `tx.from`.
- [ ] Custom-token metadata no longer doubles as an arbitrary signing-key override without binding to `from`.
- [ ] Direct `/api/v1/transactions` submissions reject transactions where the signer key/address and `from` differ.
- [ ] Regression tests cover malicious `data[..32]` public-key substitution for QUG and custom-token transfers.

## Suggested Fix

Separate signer public key from arbitrary transaction data, include the signer key in the canonical signed payload, derive the account address from that key, and require exact equality with `tx.from` before mempool admission or block application.
