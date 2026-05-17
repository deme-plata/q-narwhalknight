# PR #44 Verification Questions

**Date**: 2026-05-17
**Scope**: Follow-up verification for the GitHub collaboration questions covering issues #54–#58 and related ingestion paths.


## External GitHub Coordination Entry Point

For the next Codex session, start from **PR #44's latest comment thread** or the individual GitHub issue threads. The external discussion is cross-linked there and should be treated as the live coordination source for implementation work.

### Posted issue set

The GitHub tracker currently has 16 discoverable issues (#45–#60): the 12 findings mirrored in this audit folder plus four additional second-pass findings called out in the collaboration thread:

- #46 — admin-auth hardening
- #53 — CORS exposure
- #54 — `save_wallet_balance` regression
- #55 — missing block-apply signature verification

### Concrete fix-proposal comments to review first

Review these issue comments before editing production code, because they contain proposed diffs/tests that may supersede the documentation-only guidance here:

- #45 — emergency pause: new `FounderAuth` extractor design
- #46 — admin endpoints: same `FounderAuth` plus `Q_GENESIS_NODE` lockout
- #54 — `save_wallet_balance` regression: 3-line `return Ok(())` patch
- #55 — block-apply signature verification: `verify_all_tx_signatures` batch helper and three height-gated call sites
- #56 — signer/from binding: remove unsafe `data[..32]` signer-key extraction, but preserve/replace the production HTTP send dependency noted below
- #57 — payment header auth: replace `extract_wallet_from_headers` with `AuthenticatedWallet`
- #58 — AEGIS bypass: `.unwrap_or(false)` plus opt-in environment flag

### Local-access limitation

This container has no configured git remote and no GitHub CLI output, and public web search did not surface PR #44. The notes below are therefore based on the checked-out workspace plus the issue/PR mapping provided in the prompt. If a future session has GitHub access, it should re-open PR #44 and the issue comments directly before applying patches.

## Summary Answers

| Question | Short answer |
| --- | --- |
| 1. Are production callers relying on the `data[..32]` signer-pubkey path that #56 would break? | **Yes.** The HTTP `send_transaction` path writes the derived Ed25519 public key into `signed_transaction.data` for standard QUG/QUGUSD transfers, and `Transaction::verify_ed25519_signature` currently reads `data[..32]` as the verifying key. A patch that simply deletes `data[..32]` extraction will break this path unless signer public key/address binding is replaced with an explicit field or a new signed payload format. |
| 2. Classify `save_wallet_balance` callers for #54. | The current workspace already returns `Ok(())` from `save_wallet_balance`; the 3-line regression patch appears already present. Callers split into: consensus/storage wrappers, state-sync/import/finality overwrites, block-producer persistence of block-derived balances, and application-level direct debits/credits that should be audited separately because they bypass normal transaction consensus. |
| 3. Confirm q-storage block-apply entry-point names for #55. | Main q-storage sync entry points are `TurboSync::apply_block_pack` and `TurboSync::apply_blocks_vec`. Both call optional `BlockStateProcessor::process_block` and `BalanceConsensusEngine::process_block_mining_rewards_tx`. Balance consensus itself processes transaction loops in `process_block_mining_rewards_tx` plus related coinbase/direct variants. |
| 4. Audit other ingestion paths that may bypass tx signature verification. | Direct HTTP submit verifies signatures before inserting, and two P2P handlers check signatures before `tx_pool` insertion. However, `ProductionMempool::add_transaction` delegates to `TxValidator::validate_transaction`, whose `perform_validation` currently returns `Ok(true)`, so any caller that reaches the production mempool without a prior signature check relies on placeholder validation. Block-apply paths also need #55-style verification before balance/state application. |
| 5. Confirm `AuthenticatedWallet` body-hash coverage for #57. | **No body-hash coverage exists.** `AuthenticatedWallet` implements `FromRequestParts`, so it only sees request parts/headers, not the body. The signed challenge is `SHA3-256(address || timestamp || path)` and omits method, query, body hash, and nonce. Bearer-token auth also bypasses per-request body binding. |

## Q1: `data[..32]` Signer-Pubkey Dependency

`Transaction::verify_ed25519_signature` currently chooses the verifying key from `self.data[..32]` whenever `data.len() >= 32`; only otherwise does it fall back to `self.from`.

The production HTTP `send_transaction` path relies on that behavior:

- It derives an Ed25519 public key from the submitted mnemonic or OAuth2 vault key.
- It computes `derived_address = SHA3-256(derived_public_key)` and compares the request `from` address to that derived address or a legacy mnemonic hash address.
- It stores the Ed25519 public key in `signed_transaction.data` for standard QUG/QUGUSD transfers.

### Important nuance for #56

A patch that only deletes `data[..32]` extraction will likely break standard HTTP sends, because `tx.from` is an address/hash, not necessarily the raw Ed25519 verifying key. The safer direction is:

1. Add an explicit signer public-key field or canonical envelope.
2. Verify the signature with that signer key.
3. Derive the account address from the signer key.
4. Require the derived account address to equal `tx.from`.
5. Keep custom-token metadata separate from signer identity.

### Source anchors

- `crates/q-types/src/lib.rs:2870-2878` — verifier selects `data[..32]` as public key when present.
- `crates/q-types/src/lib.rs:2896-2900` — verifier checks `self.hash()` against that selected key.
- `crates/q-api-server/src/handlers.rs:4513-4530` — HTTP send derives Ed25519 key/public key from mnemonic.
- `crates/q-api-server/src/handlers.rs:4575-4597` — HTTP send derives an address from the public key and compares it with `from`.
- `crates/q-api-server/src/handlers.rs:4612-4626` — HTTP send stores the signature and writes public key material into `signed_transaction.data`.

## Q2: `save_wallet_balance` Caller Classification

`save_wallet_balance` currently does return `Ok(())`, so the proposed #54 regression fix appears already present in this workspace.

### Function status

- `crates/q-storage/src/lib.rs:4285-4319` — `save_wallet_balance` writes with `put_sync(...)` and returns `Ok(())`.

### Caller classes

| Class | Examples | Security interpretation |
| --- | --- | --- |
| Tests | `crates/q-storage/tests/balance_root_v1_advanced_tests.rs` | Test-only setup. |
| Consensus/storage wrappers | `BalanceStorage::add_balance`, `BalanceStorage::subtract_balance`, `BalanceStorage::set_balance` in `crates/q-storage/src/lib.rs` | Legitimate lower-level helpers; their callers still determine whether the mutation is consensus-authorized. |
| Finality/state-sync authoritative writes | `crates/q-storage/src/balance_finality_engine.rs`, `crates/q-api-server/src/state_sync_api.rs` | Intended overwrite/import paths; should remain narrowly gated by checkpoint/finality/BFT proof rules. |
| Block-derived persistence | `crates/q-api-server/src/main.rs` block-producer persistence of in-memory balances | Legitimate if balances were produced by validated block application; must be paired with #55 signature checks before block apply. |
| Application-level direct debits/credits | DCA, listing, qcredit, qno, email, quillon_bank, contracts, liquidity, calendar APIs | Highest audit risk. These mutate native balances directly outside normal transaction consensus and should each be reviewed for auth, atomicity, replay/idempotency, and whether they should become consensus transactions. |

### Representative source anchors

- `crates/q-api-server/src/state_sync_api.rs:852` — missing-wallet import writes absent peer balances.
- `crates/q-api-server/src/state_sync_api.rs:909-916` — bootstrap max-wins import writes higher peer balance.
- `crates/q-api-server/src/state_sync_api.rs:1558-1564` — finality records overwrite wallet balances.
- `crates/q-api-server/src/main.rs:20695-20699` — block producer persists block-derived balances.
- `crates/q-api-server/src/dca_api.rs:1188-1199` and `1230-1236` — DCA direct debit/credit persistence.
- `crates/q-api-server/src/qcredit_api.rs:253-261` and `317-325` — QCREDIT lock/unlock direct native balance persistence.
- `crates/q-api-server/src/qno_api.rs:343-349`, `532-538`, and `693-699` — QNO stake/unstake/claim persistence.
- `crates/q-api-server/src/email_api.rs:886-895` — email crypto transfer direct debit/credit persistence.
- `crates/q-api-server/src/liquidity_api.rs:1215-1218` and `1874-1877` — liquidity add/remove native balance persistence.

## Q3: Block-Apply Entry Points for #55

Confirmed q-storage names to target for block-apply signature verification:

- `TurboSync::apply_block_pack` — applies compressed block packs and processes state/balances.
- `TurboSync::apply_blocks_vec` — applies direct vectors of blocks.
- `BlockStateProcessor::process_block` — optional full state-sync processor called inside both TurboSync paths when configured.
- `BalanceConsensusEngine::process_block_mining_rewards_tx` — balance consensus processor called inside both TurboSync paths.

### Source anchors

- `crates/q-storage/src/turbo_sync.rs:4432-4436` — `apply_block_pack` signature.
- `crates/q-storage/src/turbo_sync.rs:4800-4804` — `apply_block_pack` calls `state_proc.process_block(block)`.
- `crates/q-storage/src/turbo_sync.rs:4868-4884` — `apply_block_pack` calls `process_block_mining_rewards_tx`.
- `crates/q-storage/src/turbo_sync.rs:5379-5383` — `apply_blocks_vec` signature.
- `crates/q-storage/src/turbo_sync.rs:5494` — `apply_blocks_vec` calls `state_proc.process_block(block)`.
- `crates/q-storage/src/turbo_sync.rs:5502-5520` — `apply_blocks_vec` calls `process_block_mining_rewards_tx`.
- `crates/q-storage/src/balance_consensus.rs:366-372` and `767-768` — balance consensus loops over block transactions.

## Q4: Other Ingestion Paths That May Bypass Signature Verification

### Paths that do verify before insertion

- `crates/q-api-server/src/handlers.rs:2593-2614` — direct `/api/v1/transactions` calls `request.transaction.verify_signature()` before pool insertion.
- `crates/q-api-server/src/main.rs:10020-10033` — one P2P gossip path rejects invalid signatures before `tx_pool.insert`.
- `crates/q-api-server/src/main.rs:15750-15760` — P2P mempool path rejects invalid signatures before `tx_pool.insert`.

### Gaps / risks

- `crates/q-narwhal-core/src/production_mempool.rs:312-316` calls `transaction_validator.validate_transaction(...)`, but `perform_validation` currently returns `Ok(true)` at `crates/q-narwhal-core/src/production_mempool.rs:823-836`. This makes the production mempool unsafe as the only validation boundary.
- `crates/q-api-server/src/transaction_utils.rs:216-227` inserts helper-built transactions into `tx_pool` and then production mempool without local `verify_signature()` enforcement in that helper. Some helper transactions may be system operations, but the bypass should be explicit and type-gated.
- Block-sync apply paths process `block.transactions` through state/balance processors without an obvious pre-loop all-transaction signature gate in the shown `apply_block_pack` / `apply_blocks_vec` paths. This matches the proposed #55 fix direction.
- Mining reward gossip at `crates/q-api-server/src/main.rs:10123-10133` inserts reward transactions as confirmed for tracking; this may be acceptable for coinbase/reward telemetry, but should be type-gated so it cannot admit arbitrary non-coinbase transfers.

## Q5: `AuthenticatedWallet` Body-Hash Coverage

`AuthenticatedWallet` does **not** cover request body bytes today.

Reasons:

- It implements `FromRequestParts`, not a full-body extractor, so it only receives headers/URI parts.
- The signed challenge hashes address, timestamp, and path only.
- It uses `.path()` from `OriginalUri`, not the full URI including query string.
- There is no method binding, body hash, or nonce in `AuthHeader`.
- Bearer-token and AIOC auth paths also produce `AuthenticatedWallet` without per-request body binding.

### Source anchors

- `crates/q-api-server/src/wallet_auth.rs:63-90` — `AuthHeader` has address/timestamp/scheme/signature fields, but no method/query/body hash/nonce.
- `crates/q-api-server/src/wallet_auth.rs:118-125` — `AuthenticatedWallet` is a `FromRequestParts` extractor.
- `crates/q-api-server/src/wallet_auth.rs:173-180` — replay defense is timestamp-window only.
- `crates/q-api-server/src/wallet_auth.rs:208-222` — challenge hashes address, timestamp, and path only.
- `crates/q-api-server/src/wallet_auth.rs:480-543` — bearer auth accepts stored access tokens without method/body binding.
- `crates/q-api-server/src/wallet_auth.rs:659-670` — AIOC HMAC covers wallet address and timestamp only.
- `crates/q-api-server/src/wallet_auth.rs:754-765` — WebSocket auth uses the same fixed path challenge.

## Commands Used

```bash
rg -n "save_wallet_balance\(" crates -g '*.rs'
rg -n "add_transaction\(|tx_pool\.insert|process_block_mining_rewards_tx\(|apply_block_pack\(|apply_blocks_vec\(" crates/q-api-server/src crates/q-storage/src crates/q-narwhal-core/src -g '*.rs'
rg -n "fn try_bearer_auth|try_bearer_auth|body" crates/q-api-server/src/wallet_auth.rs
nl -ba crates/q-types/src/lib.rs | sed -n '2828,2908p'
nl -ba crates/q-api-server/src/handlers.rs | sed -n '2590,2630p;4510,4630p'
nl -ba crates/q-storage/src/lib.rs | sed -n '4278,4335p;9690,9770p;9818,9838p'
nl -ba crates/q-storage/src/turbo_sync.rs | sed -n '4430,4455p;4770,4895p;5379,5528p'
nl -ba crates/q-narwhal-core/src/production_mempool.rs | sed -n '260,335p;823,870p'
nl -ba crates/q-api-server/src/wallet_auth.rs | sed -n '63,90p;118,245p;470,560p;659,670p;750,765p'
```
