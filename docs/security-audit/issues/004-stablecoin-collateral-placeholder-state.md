# Issue #004: Back stablecoin collateral calculations with real per-user state

**State**: `open`
**Priority**: HIGH
**Labels**: `security`, `stablecoin`, `collateral`, `accounting`
**Created**: 2026-05-17

## Finding

`QuantumCollateralManager` uses fixed placeholder collateral and position data instead of request/user-specific state.

## Evidence

- `crates/q-stablecoin/src/collateral.rs::calculate_quantum_collateral_value` ignores its request and returns `BigDecimal::from(1000)`.
- `crates/q-stablecoin/src/collateral.rs::get_quantum_position` ignores `user_id` and returns a position for `"test"` with fixed collateral amount and ratio.


## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-stablecoin/src/collateral.rs:21-25` ignores the mint request and returns fixed placeholder collateral value `1000`.
- `crates/q-stablecoin/src/collateral.rs:28-34` ignores `user_id` and returns a fixed `"test"` position.

## Impact

Mint/burn authorization can be based on fake collateral values. A caller may mint or release collateral using default placeholder state rather than authenticated positions.

## Acceptance Criteria

- [ ] Collateral positions are storage-backed and keyed by the authenticated user/request user.
- [ ] Collateral value is computed from request amount, collateral type, and an authenticated price source.
- [ ] Unknown users return `NotFound` or equivalent instead of a default position.
- [ ] Tests prove users cannot mint or release collateral using default placeholder state.

## Suggested Fix

Introduce persistent collateral-position storage, require authenticated ownership for position access, and make missing pricing/position data a hard error.
