# Issue #010: Replace transaction privacy proof placeholders with real proofs or disable claims

**State**: `open`
**Priority**: HIGH
**Labels**: `security`, `privacy`, `zk`, `cryptography`
**Created**: 2026-05-17

## Finding

The automatic transaction privacy proof generator claims Bulletproof/STARK privacy, but it constructs deterministic, valid-format byte blobs locally, includes clear amount material in the Bulletproof bytes, and treats proof-generation failures as non-fatal.

## Evidence

- The module-level comments advertise automatic privacy, amount hiding, STARK validity proofs, and nullifiers.
- `generate_bulletproof` says it creates a valid-format proof for now, inserts `amount_u64.to_le_bytes()` and the blinding bytes directly into the proof, then hashes deterministic labels for the body.
- `generate_stark_proof` builds a local trace and fills FRI-like query bytes with hashes/evaluations, without calling an external verifier-backed prover.
- `generate_proofs` logs proof failures but continues the transaction without failing.

## Verification Status

Verified against the current workspace on 2026-05-17. Source anchors checked with `nl -ba`:

- `crates/q-api-server/src/privacy_proof_generator.rs:5-12` advertises automatic ZK privacy, amount hiding, STARK validity, nullifiers, and LatticeGuard.
- `crates/q-api-server/src/privacy_proof_generator.rs:128-161` continues after Bulletproof/STARK generation failures.
- `crates/q-api-server/src/privacy_proof_generator.rs:219-227` creates a "valid-format" proof and writes amount/blinding material into the proof bytes.
- `crates/q-api-server/src/privacy_proof_generator.rs:229-271` derives the rest of the Bulletproof body from hashes and padding.
- `crates/q-api-server/src/privacy_proof_generator.rs:315-394` constructs STARK-like proof bytes from a local trace, hashes, evaluations, and zero constraints.

## Impact

Transactions can be marked as private even when no real range-proof/STARK verification security is present, and the generated proof bytes can leak or encode amount-derived information. Users may rely on privacy guarantees that are not actually enforced.

## Acceptance Criteria

- [ ] Replace placeholder proof construction with verifier-backed proof systems, or return explicit not-implemented errors.
- [ ] Never embed plaintext amount or blinding material in proof bytes advertised as amount-hiding.
- [ ] Privacy proof generation failure downgrades visible privacy status or rejects transactions when privacy is required.
- [ ] Add verifier tests proving placeholder byte blobs are rejected.

## Suggested Fix

Gate automatic privacy behind a feature flag until real proof generation and verification are wired. Keep `privacy_level` transparent unless verified proofs are attached and accepted by the same validation path used for mempool/block admission.
