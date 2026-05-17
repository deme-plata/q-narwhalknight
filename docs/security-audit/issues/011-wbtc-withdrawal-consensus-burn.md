# Issue #011: Make wBTC withdrawals burn through consensus and persist an atomic bridge record

**State**: `open`
**Priority**: HIGH
**Labels**: `security`, `bridge`, `bitcoin`, `accounting`, `consensus`
**Created**: 2026-05-17

## Finding

The wBTC withdrawal endpoint deducts the caller's local token balance and persists it directly before broadcasting a Bitcoin transaction. The deduction is not represented as a signed burn transaction accepted by consensus, and successful Bitcoin broadcasts do not appear to persist an atomic withdrawal/burn record in the shown handler.

## Evidence

- `withdraw_wbtc` is registered as an API route in `main.rs`.
- The handler deducts `state.token_balances` and saves the new token balance directly to storage.
- The handler then calls `bridge.send_withdrawal(...)` and only refunds on broadcast error.
- There is no visible construction/submission of a signed wBTC burn transaction before the on-chain BTC send in this handler.

## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/main.rs:25432-25433` registers `POST /api/v1/bitcoin/withdraw`.
- `crates/q-api-server/src/bitcoin_deposit_api.rs:354-362` accepts an optional wallet auth extractor and requires it at runtime.
- `crates/q-api-server/src/bitcoin_deposit_api.rs:428-457` checks, deducts, and persists the local wBTC balance.
- `crates/q-api-server/src/bitcoin_deposit_api.rs:459-472` broadcasts the Bitcoin withdrawal after the local deduction.
- `crates/q-api-server/src/bitcoin_deposit_api.rs:474-485` only handles the broadcast-error refund path.

## Impact

Bridge supply can diverge across nodes if the local deduction is not replayed through the canonical transaction/block path. Crashes or partial failures around local persistence and external Bitcoin broadcast can also leave incomplete withdrawal state without a durable idempotency key.

## Acceptance Criteria

- [ ] wBTC withdrawal first creates a signed burn/redeem transaction that consensus accepts exactly once.
- [ ] Bitcoin broadcast is keyed by a durable withdrawal ID tied to the accepted burn.
- [ ] Restart/retry logic is idempotent and cannot double-send BTC or resurrect burned wBTC.
- [ ] Tests cover mempool rejection, crash after burn before BTC broadcast, broadcast retry, and duplicate withdrawal requests.

## Suggested Fix

Split withdrawal into a state-machine: `Requested -> BurnAccepted -> BroadcastPending -> BroadcastConfirmed/RefundRequired`, store it durably, and make the token burn a consensus transaction rather than a local balance edit.
