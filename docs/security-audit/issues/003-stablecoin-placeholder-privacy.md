# Issue #003: Replace stablecoin placeholder proofs and signatures

**State**: `open`
**Priority**: HIGH
**Labels**: `security`, `privacy`, `cryptography`, `stablecoin`
**Created**: 2026-05-17

## Finding

The stablecoin privacy layer returns fixed placeholder proof and signature bytes for mint, burn, and quantum signature operations.

## Evidence

- `crates/q-stablecoin/src/privacy.rs::generate_mint_proof` returns `proof_data: vec![1, 2, 3, 4]`.
- `crates/q-stablecoin/src/privacy.rs::generate_burn_proof` returns the same fixed proof bytes.
- `crates/q-stablecoin/src/privacy.rs::generate_quantum_signature` returns the same fixed byte vector.


## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-stablecoin/src/privacy.rs:24-32` returns `proof_data: vec![1, 2, 3, 4]` for mint proofs.
- `crates/q-stablecoin/src/privacy.rs:35-43` returns the same fixed bytes for burn proofs.
- `crates/q-stablecoin/src/privacy.rs:46-47` returns `vec![1, 2, 3, 4]` for the quantum signature.

## Impact

Stablecoin operations can appear to carry cryptographic privacy/authenticity artifacts even though the artifacts are static placeholders. This creates a false privacy guarantee for a private cryptocurrency system.

## Acceptance Criteria

- [ ] Placeholder proof/signature generation is replaced by real proof construction and verification, or returns a clear not-implemented error.
- [ ] Mint/burn callers verify proofs before accepting privacy-enabled operations.
- [ ] Advanced privacy modes are feature-gated off until real proofs are available.
- [ ] Tests assert that `vec![1, 2, 3, 4]` and other placeholder values are never accepted as valid proofs.

## Suggested Fix

Add verifier functions beside proof generation, thread verification through `QuantumStablecoin::mint_orbusd` and `burn_orbusd`, and remove/soften runtime claims until the cryptography is implemented.
