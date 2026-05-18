# ADR: q-ivc ↔ q-recursive-proofs Adapter Boundary

**Date:** 2026-05-18
**Status:** Accepted for PR #79 production path
**Owners:** Recursive proofs / IVC maintainers
**Scope:** `crates/q-ivc`, `crates/q-recursive-proofs`

## Context

The repository currently contains two recursive-proof representations that can be
confused if they are wired directly into epoch proving paths:

1. `crates/q-recursive-proofs` owns the live protocol orchestration:
   `ProverNode::prove_epoch`, `ProverNode::verify_proof`, proof submissions, and
   `LightClient::verify_epoch_proof` all operate on `q_lattice_guard` proofs,
   `ArithmeticCircuit`, and `EpochPublicInputs`.
2. `crates/q-ivc` owns arkworks-compatible R1CS/Nova-style circuit research,
   gadgets, and host witness builders. Its crate docs explicitly mark the crate
   as scaffolded and list prerequisites before it can produce valid proofs over
   real chain data.

Without an explicit boundary, future work could accidentally add a third epoch
transition encoding: one shape for prover tasks, one for peer verification, and
another for light-client verification. That would make proof submissions appear
valid in one path and unverifiable in another.

## Decision

PR #79's production path uses **`q-recursive-proofs` LatticeGuard
`ArithmeticCircuit` circuits** as the canonical epoch transition proof system.

`q-ivc` arkworks/Nova circuits remain the research and future-backend track until
they have a complete production proving backend, audited circuit constraints, and
a migration ADR that replaces this decision.

The integration boundary is an adapter owned by `q-recursive-proofs`, tentatively
named:

```text
crates/q-recursive-proofs/src/ivc_adapter.rs
```

The adapter is the only module allowed to translate epoch task data into the
canonical LatticeGuard public-input vector and witness format used by the
production recursive-proof path.

## Canonical public-input encoding

The canonical epoch public inputs are the semantic fields already represented by
`EpochPublicInputs`:

| Order | Field | Scalar encoding |
| --- | --- | --- |
| 0 | `previous_state_root` | 8 little-endian `u32` limbs widened to `Scalar` |
| 1 | `current_state_root` | 8 little-endian `u32` limbs widened to `Scalar` |
| 2 | `epoch` | one `Scalar` |
| 3 | `height_range.0` | one `Scalar` |
| 4 | `height_range.1` | one `Scalar` |
| 5 | `validator_set_hash` | 8 little-endian `u32` limbs widened to `Scalar` |
| 6 | `signature_count` | one `Scalar` |
| 7 | `epoch_end_timestamp` | omitted from the fixed 28-scalar adapter vector |

The resulting vector length is 28 scalars. Three 32-byte roots/hashes occupy
24 scalars, and `epoch`, `height_range.0`, `height_range.1`, and
`signature_count` occupy the remaining 4 scalars. Including
`epoch_end_timestamp` in addition to `signature_count` would require 29 scalars,
so the 28-scalar adapter keeps `signature_count` and leaves timestamp transport
to the typed `EpochPublicInputs` serialization rather than the LatticeGuard
public-input vector. Any code path that proves, verifies, serializes,
deserializes, or serves a light-client proof must obtain this vector through the
adapter rather than hand-rolling conversions.

## Adapter responsibilities

The adapter should expose a small, typed surface rather than a second circuit
abstraction:

- Build `EpochPublicInputs` from `EpochProofTask` plus finalized epoch metadata.
- Encode those inputs into the canonical 28-scalar LatticeGuard vector.
- Build the LatticeGuard `ArithmeticCircuit` with the same epoch block count that
  the prover, peer verifier, and light client will use.
- Build the witness vector for the LatticeGuard circuit from fetched block,
  signature, previous-proof, and state-transition data.
- Reject inconsistent task metadata before proving, including mismatched block
  counts, state-root continuity failures, signature-count/public-input mismatch,
  and non-canonical height ranges.
- Provide a single test helper for synthetic epochs so conformance tests do not
  duplicate encoding logic.

The adapter must not expose arkworks field elements, Nova step-circuit state, or
`q-ivc` host witness internals to the production proof protocol.

## Required call-site changes

The implementation PR following this ADR should route these call sites through
`ivc_adapter`:

- `ProverNode::prove_epoch` must build the proof circuit, witness, and public
  inputs with the adapter. It must stop using its local partial public-input
  extractor.
- `ProverNode::verify_proof` must deserialize `EpochPublicInputs` and obtain the
  verification scalar vector and circuit shape from the adapter.
- `LightClient::verify_epoch_proof` must obtain the same verification scalar
  vector and circuit shape from the adapter.
- Light-client bootstrap proof verification must use the same adapter policy for
  public inputs rather than a separate bootstrap-only encoding.

## Crate boundary rules

- `q-recursive-proofs` is the production owner of epoch proof protocol data,
  canonical public-input encoding, LatticeGuard circuit construction, proof
  submission serialization, and light-client verification.
- `q-ivc` may provide arkworks/Nova gadgets, experimental circuits, and reference
  witness-building research. It must not define a production epoch-proof wire
  format unless a future ADR explicitly migrates the canonical proof system.
- Any future bridge from `q-ivc` to production must be implemented behind this
  adapter boundary and must prove byte-for-byte/scalar-for-scalar compatibility
  with the canonical `EpochPublicInputs` encoding above.

## Conformance testing requirement

The implementation PR must add a conformance test that constructs one synthetic
epoch and checks that the public inputs match across all production paths:

1. Prover path: task plus epoch metadata encoded by the adapter.
2. Peer verifier path: serialized `EpochProofSubmission` decoded and encoded by
   the adapter.
3. Light-client path: `EpochProof` encoded by the adapter.

The test must assert exact equality of the 28-scalar vector and must fail if any
path omits `validator_set_hash` or `signature_count`, includes a 29th
`epoch_end_timestamp` scalar, or uses 64-bit hash limbs instead of the canonical
32-bit limbs.

## Consequences

### Positive

- One canonical production proof representation for PR #79.
- Prover, peer verifier, and light client share public inputs and circuit shape.
- `q-ivc` can continue evolving without destabilizing the production
  LatticeGuard protocol.
- Future backend migration has a clear compatibility target.

### Negative / trade-offs

- Arkworks/Nova work in `q-ivc` is not the production proof path for PR #79.
- The adapter adds a small indirection layer before proving and verification.
- A future migration from LatticeGuard `ArithmeticCircuit` to `q-ivc`/Nova will
  require a new ADR and conformance proof rather than a transparent swap.

## Non-goals

- This ADR does not implement the adapter module.
- This ADR does not add or modify proof-generation code.
- This ADR does not claim that `q-ivc` circuits are production-ready.
