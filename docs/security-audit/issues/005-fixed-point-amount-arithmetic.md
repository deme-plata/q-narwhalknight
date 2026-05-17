# Issue #005: Use integer or decimal-safe arithmetic for 24-decimal token amounts

**State**: `open`
**Priority**: HIGH
**Labels**: `security`, `accounting`, `precision`, `stablecoin`
**Created**: 2026-05-17

## Finding

Several 24-decimal money paths convert user-supplied string amounts or base-unit balances through `f64`, including stablecoin mint/redeem and vault display/math paths.

## Evidence

- `crates/q-api-server/src/stablecoin_api.rs::mint_qugusd` parses `request.qug_amount` with `parse::<f64>()`, multiplies by `1e24`, then casts to `u128`.
- `crates/q-api-server/src/stablecoin_api.rs::redeem_qug` repeats the same pattern for QUGUSD.
- The same handler formats base-unit values back through `as f64 / 1e24`.


## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/stablecoin_api.rs:617-628` parses `request.qug_amount` through `f64`, multiplies by `1e24`, and casts to `u128`.
- `crates/q-api-server/src/stablecoin_api.rs:640-651` formats base-unit mint results through `as f64 / 1e24`.
- `crates/q-api-server/src/stablecoin_api.rs:698-710` repeats `f64 * 1e24 as u128` parsing for redeem amounts.

## Impact

`f64` cannot exactly represent 24-decimal token values. Rounding, truncation, overflow, and scientific-notation surprises can corrupt balances or collateral ratios in a cryptocurrency accounting path.

## Acceptance Criteria

- [ ] User amount strings are parsed into base-unit integers by a decimal parser enforcing at most 24 fractional digits.
- [ ] No production money path casts floats to `u128` for balances, prices, collateral ratios, minted amounts, or burned amounts.
- [ ] Formatting is integer/decimal-safe at API boundaries.
- [ ] Tests cover tiny values, max-ish values, over-precision rejection, and exact round trips.

## Suggested Fix

Add a shared fixed-point parser/formatter for QUG/QUGUSD amounts and ban `parse::<f64>()`, `1e24`, and `as u128` in consensus/accounting code through tests or lint-like checks.
