# ADR — Nova Folding Crate Selection for Phase 2 Recursive zk-SNARK

**Status:** Decided
**Date:** 2026-05-15
**Owners:** Server Beta + DeepSeek
**Replaces:** an earlier DeepSeek draft (`docs/nova-crate-evaluation-2026-05-15.md`, withdrawn) that presented benchmark numbers from non-compiling sketch code.

## Decision

**Adopt `arkworks-rs/nova`** for the Phase 2 step-circuit folding layer.

This is an **architectural** decision, not a performance decision. No benchmarks were run for this ADR. The reasoning is:

1. The entire IVC stack in `crates/q-ivc/` is arkworks 0.5 — `ark-ff`, `ark-r1cs-std`, `ark-relations`, `ark-groth16`, `ark-bls12-381`.
2. All current and planned δ-circuit gadgets (BLAKE3, NTT-256, sparse Merkle tree depth-256, Dilithium witness allocators) are written against `ConstraintSystemRef<F: PrimeField>` and `FpVar<F>` / `UInt8<F>` / `Boolean<F>`.
3. `arkworks-rs/nova` consumes `ConstraintSynthesizer<F>` directly via its `StepCircuit<F>` trait. **Zero bridge code.**
4. `microsoft/Nova` (the original Bellperson-based implementation) consumes `bellperson::ConstraintSystem<F>` via its own `StepCircuit<F>` trait. Adopting it requires:
   - A field-element bridge `ark_ff::Fp ↔ ff::Field` (round-trip via byte encoding — adds ~30 LOC and an extra constraint per cross).
   - Re-implementing or wrapping every gadget so it speaks both APIs.
   - Estimated minimum bridge surface: **~200 LOC** of `From`/`Into` plumbing per primitive, multiplied across BLAKE3, NTT-256, Merkle, Dilithium witness — **realistically ~800–1200 LOC of glue with associated test burden**.

The bridge cost is paid every time a new gadget is added, forever. The arkworks-native path pays it zero times. Even if `microsoft/Nova` were 2× faster at folding, the engineering carrying cost of the bridge would still tilt the decision toward arkworks for our specific gadget stack.

## What we explicitly are NOT claiming

- We are NOT claiming `arkworks-rs/nova` is faster than `microsoft/Nova`. The Microsoft implementation is widely understood to be more mature and likely faster on raw folding throughput today.
- We are NOT claiming a measured fold-time or verify-time number. None has been run on our circuit.
- The decision rests entirely on the bridge-cost argument above.

## Performance acceptance criteria (deferred to implementation)

Once the `StepCircuit<F>` impl for `DeltaStepCircuit` is wired (next implementation step), we will measure on Epsilon Docker (Debian 12, `rust:bookworm`):

| Metric | Target | If missed → |
|---|---|---|
| Fold time per step (toy 9-word circuit) | < 50 ms | acceptable — we have a 1 s/block budget |
| Fold time per step (full δ-circuit, K=10) | < 1 000 ms | revisit; consider per-component proving |
| Verify time (final compressed SNARK) | < 50 ms | revisit Microsoft + bridge |
| Compressed proof size | < 4 KB | acceptable for P2P fast-sync |

If the full-δ-circuit fold time exceeds 1 s and cannot be brought down by parallel proving across blocks, the decision is revisitable. In that scenario we would prototype the Microsoft bridge with cost estimates against a working baseline — not against a hypothetical one.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `arkworks-rs/nova` is younger / less battle-tested than Microsoft's | Medium | Advisory-mode rollout (whitepaper §6.2): proof is non-binding for ≥6 months; block-by-block validation is the soundness fallback. |
| Bug in arkworks-rs/nova's folding scheme | Low | Soundness is gated by Groth16 over compressed proof; folding-layer bug surfaces as verification failure, not silent acceptance. |
| Performance shortfall vs budget | Medium | Acceptance criteria above; rollback path is implementation-defined (swap StepCircuit impl, keep δ-circuit unchanged). |
| arkworks-rs/nova API churn breaking the StepCircuit impl | Low-Med | Pin to a known-good commit hash. Re-eval at every arkworks 0.5 → 0.6 jump. |

## Implementation handoff

- `crates/q-ivc/Cargo.toml` → add `arkworks-rs/nova` as a git dependency (pin commit SHA in same PR that lands `StepCircuit` impl)
- `crates/q-ivc/src/recursion/step_circuit.rs::DeltaStepCircuit` → impl `arkworks_nova::StepCircuit<Fr>` (estimated ~30 LOC; arity = `STEP_Z_LEN = 9`)
- `crates/q-ivc/src/recursion/fold.rs` (new) → `NovaFolder::fold_block` driver per whitepaper §5.2
- `crates/q-api-server/src/tip_watcher.rs:114` → replace `PHASE2-WIRE-POINT` marker (one line) once the above land

## Why this ADR exists

The earlier DeepSeek-authored evaluation included a benchmark table whose accompanying source code was annotated as using "hypothetical APIs" and placeholder values (`1234`, `789`, `abcd123`, `wxyz456`). Publishing decision-support material with fabricated numbers — even with the right conclusion — corrupts the project's evidence base. This ADR reaches the same decision from honest grounds: an architectural cost-of-bridging argument that needs no measurements to be valid.

This is also the template for future Phase 2/3 crate-selection decisions: state the architectural reasoning first, defer performance numbers to acceptance criteria that get measured against working code.
