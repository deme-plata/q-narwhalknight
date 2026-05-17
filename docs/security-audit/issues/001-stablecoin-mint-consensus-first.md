# Issue #001: Make QUGUSD minting consensus-first and balance-safe

**State**: `open`
**Priority**: CRITICAL
**Labels**: `security`, `stablecoin`, `consensus`, `accounting`
**Created**: 2026-05-17

## Finding

`mint_qugusd` mutates and persists `state.collateral_vault` before the generated `StableMint` transaction is accepted by the normal block/state transition path. The helper-built transaction is unsigned, zero-fee, and its submit result is ignored.

## Evidence

- `crates/q-api-server/src/stablecoin_api.rs::mint_qugusd` parses the request amount, calls `vault_write.mint_qugusd(...)`, builds the API response, and persists the cloned vault before transaction submission.
- `crates/q-api-server/src/stablecoin_api.rs::mint_qugusd` records the generated tx in `optimistic_applied_txs` and assigns the result of `transaction_utils::submit_transaction(...)` to `_result` without checking rejection.
- `crates/q-api-server/src/transaction_utils.rs::TransactionBuilder::build_with_nonce` initializes `signature: vec![]`.
- `crates/q-api-server/src/transaction_utils.rs::create_stable_mint_transaction` creates `TransactionType::StableMint` with `.fee(0)` and no signing step.
- `crates/q-types/src/lib.rs::Transaction::verify_signature` requires signatures for non-coinbase transactions, so the generated local transaction is not equivalent to a valid user-signed spend.


## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/stablecoin_api.rs:624-628` parses QUG with `f64` and casts to `u128`.
- `crates/q-api-server/src/stablecoin_api.rs:631-638` mutates `state.collateral_vault` with `vault_write.mint_qugusd(...)`.
- `crates/q-api-server/src/stablecoin_api.rs:654-665` persists the cloned vault before submitting the generated transaction.
- `crates/q-api-server/src/stablecoin_api.rs:671-685` constructs the `StableMint` transaction and assigns `submit_transaction(...)` to `_result`.
- `crates/q-api-server/src/transaction_utils.rs:144` initializes builder-created transactions with an empty signature.
- `crates/q-api-server/src/transaction_utils.rs:405-414` creates `StableMint` with zero fee and no signing step.

## Impact

A node can persist local QUGUSD debt/collateral state before the source-of-truth ledger has proven the user owns and locked QUG. Nodes may diverge on vault state, and failed/rejected transactions can still leave local minted state behind.

## Acceptance Criteria

- [ ] `mint_qugusd` does not mutate or persist `CollateralVault` until a signed vault operation has been accepted through the same state-transition path used by blocks.
- [ ] The authenticated wallet balance is checked against the canonical spendable QUG balance before minting.
- [ ] QUG collateral locking/debiting and QUGUSD minting are atomic and replay-safe.
- [ ] `submit_transaction` errors/rejections are surfaced to the client.
- [ ] Regression tests cover insufficient QUG, unsigned `StableMint`, rejected submit, and successful exact-once locking.

## Suggested Fix

Move the vault mutation into block/state application for `TransactionType::StableMint`, require the wallet signature to cover the collateral amount and nonce, and treat the API handler as a transaction constructor/submitter only.
