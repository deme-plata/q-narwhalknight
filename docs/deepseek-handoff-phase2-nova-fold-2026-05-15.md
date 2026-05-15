# DeepSeek Handoff — Phase 2 Nova Fold (Recursive zk-SNARK)

**Date:** 2026-05-15
**Branch base:** `feature/safe-batched-sync-v1.0.2` at commit `0e0b227a4`
or later (the commit that landed your ExpandA byte-order fix from the
2026-05-15 peer review).

**Companion docs you should read first:**
- `docs/technical-review-recursive-snark-progress-2026-05-15.md` —
  the Phase 1 review you peer-reviewed earlier today
- `papers/quillon-recursive-lattice-snark-whitepaper-v2-2026-05-13.tex`
  §3.3, §4.2, §4.3 (Nova fold spec, prover throughput, verifier latency)
- `docs/deepseek-job-board-nova-phase2-2026-05-14.md` Jobs N2 + N3
- `crates/q-ivc/src/recursion/step_circuit.rs` — the Phase 2 boundary
  I've scaffolded (StepIO, StepCircuitAdapter, DeltaStepCircuit,
  fold_native)
- `crates/q-ivc/src/host/dilithium_witness.rs` — the host helpers
  you peer-reviewed

**Audience:** you (DeepSeek), continuing the same review-and-implement
cadence from this morning. The Server Beta in-tree author is refactoring
`DeltaBlockCircuit` in parallel to enable `synthesize_step`; you should
land the work below independently and we'll merge.

**Mainnet safety:** all of this lives in `crates/q-ivc/` which is NOT a
dep of `q-api-server`. Zero impact on the running production binary.
The recursive-proof rollout is gated through Phase 3 advisory mode for
6+ months before any consensus-affecting flip per whitepaper §6.2.

---

## TL;DR — your two coding tasks

| Task | What | Est. LOC | Est. time |
|---|---|---|---|
| **DS-1** | ExpandA conformance test against `pqcrystals-dilithium` reference | ~120 | 2 hrs |
| **DS-2** | Nova-crate evaluation: prototype `nova-snark` AND `arkworks-rs/nova` on a toy `StepCircuit`, report which we should adopt | ~300 + report | 1 engineering day |

Both are self-contained, exercise areas you already showed strong
context on (FIPS-204 byte format for DS-1; Nova folding for DS-2),
and unblock the next concrete piece of the Phase 2 rollout.

The Server Beta in-tree author is in parallel doing:
- Refactor `DeltaBlockCircuit::generate_constraints` to expose an
  inner method that takes pre-allocated z_in
- Implement `DeltaStepCircuit::synthesize_step` against that inner
  method
- Once you pick the Nova crate (DS-2), wire the chosen crate's
  `StepCircuit` trait to `DeltaStepCircuit::synthesize_step`

---

## DS-1 — ExpandA conformance test against pqcrystals-dilithium

You caught the byte-order bug in `expand_a_native` (commit `0e0b227a4`).
You explicitly recommended: *"cross-check against the `pqcrystals-dilithium`
reference implementation byte-for-byte on a few known seeds."* This task
does exactly that.

### Goal

Generate a known-good A matrix using `pqcrystals-dilithium`'s reference
implementation and assert byte-equality against `expand_a_native`'s
output for several test vectors. This locks in the byte order + the
23-bit rejection sampling for future regressions.

### File to create

`crates/q-ivc/tests/expand_a_conformance.rs` (NEW)

### Why an integration test (`tests/`) not a unit test (in
`src/host/dilithium_witness.rs`)

Bringing in `pqcrystals-dilithium` as a `[dev-dependencies]` for the
crate is fine, but importing the reference at module-test scope mixes
abstraction layers. Integration tests are the right home for
cross-crate conformance checks.

### What you'll need

The `pqcrystals-dilithium` Rust crate is at:
- crates.io: `pqcrystals-dilithium-sys` (FFI to the C reference)
- Alternative: `pqcrypto-dilithium` (pure Rust, already used in
  `crates/q-crypto-simd/` per the technical review §5.2 "Batched
  Dilithium signature verification") — **prefer this one** since it's
  already in the workspace.

Spec reference: FIPS-204 §3.2 Algorithm 4 RejBoundedPoly + Algorithm 8
ExpandA.

### Implementation sketch

```rust
// crates/q-ivc/tests/expand_a_conformance.rs
//
// Cross-check expand_a_native against pqcrypto-dilithium's reference
// ExpandA. Generates A from N test seeds via both implementations,
// asserts byte-equality.

use q_ivc::host::dilithium_witness::{expand_a_native, K, L, N};
use pqcrypto_dilithium::dilithium5; // or whichever crate exports a usable Expand

fn reference_expand_a(rho: &[u8; 32]) -> Vec<[u32; N]> {
    // Use pqcrypto-dilithium's internal ExpandA. If it's not exposed
    // publicly, you may need to reimplement the rejection-sampling
    // loop using SHAKE-128 from sha3 — but get the bytes from THEIR
    // SHAKE input/output to verify only the byte-ORDER is the
    // question, not the sampling logic.
    //
    // If pqcrypto-dilithium doesn't expose ExpandA directly:
    //   1. Generate a Dilithium5 keypair with their `keypair_from_seed`
    //      using the test seed.
    //   2. Their public key contains ρ in the first 32 bytes.
    //   3. The matrix A is implicitly committed via t = A·s1 + s2.
    //   4. Extract A by sampling polynomials independently and
    //      cross-checking. (More work — better to find a crate that
    //      exposes the expansion.)
    //
    // Alternative reference: jhwgh1968/dilithium5-rs (single-file
    // pure-Rust impl by an academic; may expose internals).
    todo!()
}

#[test]
fn expand_a_matches_reference_on_zero_seed() {
    let rho = [0u8; 32];
    let ours = expand_a_native(&rho);
    let theirs = reference_expand_a(&rho);
    for (idx, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
        assert_eq!(a, b, "ExpandA poly[{}] mismatch", idx);
    }
}

#[test]
fn expand_a_matches_reference_on_known_seeds() {
    // FIPS-204 Annex C provides intermediate values for ML-DSA-87 from
    // a known seed. Use those if available; otherwise pick 5 distinct
    // 32-byte seeds and assert byte-equality.
    let seeds: [[u8; 32]; 5] = [
        [0x01; 32],
        [0xAA; 32],
        [0x00, 0x01, /* ... fill with a deterministic pattern ... */ 0u8],
        // ... three more
    ];
    for seed in &seeds {
        let ours = expand_a_native(seed);
        let theirs = reference_expand_a(seed);
        assert_eq!(ours, theirs, "ExpandA divergence for seed {:?}", &seed[..4]);
    }
}
```

### Acceptance criteria

- Test file lives at `crates/q-ivc/tests/expand_a_conformance.rs`.
- At least 5 distinct test seeds.
- Test invokes a real `pqcrypto-dilithium` (or vetted alternative)
  implementation, not a self-implemented reference.
- If the reference doesn't expose ExpandA publicly, comment WHY the
  alternative path you chose is faithful to the spec (e.g., "extracted
  A from a keypair via A·s1 = t - s2, which only works if our
  signature mathematics is consistent").
- Tests must PASS — which means our `expand_a_native` is correct per
  the spec. If they fail, file the bug back and we'll fix.

### Cargo.toml addition

```toml
[dev-dependencies]
pqcrypto-dilithium = { workspace = true }  # if not already in dev-deps
```

Check `crates/q-ivc/Cargo.toml` — it may already be there from a
previous bit of work in this branch.

---

## DS-2 — Nova crate selection: prototype both, report

The whitepaper §3.3 + technical plan v2 Phase 2 explicitly defer the
choice between `nova-snark` (Microsoft) and `arkworks-rs/nova`
(community). The intent: "three engineering days prototyping both on
a toy step circuit before committing."

The Server Beta in-tree author's `DeltaStepCircuit` is Nova-crate-agnostic
right now (it implements a local `StepCircuitAdapter` trait). Once
you pick a crate, wiring the actual trait is a 30-LOC commit.

### Goal

Build a TOY step circuit in BOTH crates that does the same trivial
transition (e.g., `z_out = z_in + 1` for a 9-word z), measure prover
+ verifier time + proof size at chain lengths n ∈ {1, 10, 100,
1000}, write a short report recommending one with concrete numbers.

### Why this matters

The benchmark data informs:
- Per-block prover budget (whitepaper §4.2: <1s/block target on
  Epsilon-class hardware).
- Verifier latency target (5-10ms desktop, <250ms WASM).
- Memory footprint (Nova accumulates a relaxed R1CS instance; size
  matters for genesis-node steady state).
- Constraint-system compatibility (Microsoft's uses bellperson;
  arkworks uses arkworks `ConstraintSynthesizer`). Our gadgets are
  arkworks. The arkworks crate has zero adapter cost; the
  Microsoft crate needs a bellperson↔arkworks bridge (~200 LOC of
  field-element conversion).

### Files to create

```
crates/q-ivc-nova-bench/Cargo.toml           NEW
crates/q-ivc-nova-bench/src/main.rs          NEW
crates/q-ivc-nova-bench/src/microsoft.rs     NEW (~150 LOC)
crates/q-ivc-nova-bench/src/arkworks.rs      NEW (~150 LOC)
docs/nova-crate-evaluation-2026-05-15.md     NEW (report)
```

Don't add to the workspace members yet — make it a standalone crate
that the maintainer can pull in once you've picked. Add to workspace
once the choice is final.

### Toy circuit spec

A `StepCircuit` whose `synthesize`:
- Takes z_in ∈ F^9 (matching `STEP_Z_LEN` from `step_circuit.rs`).
- For each of 8 first words: enforces z_out[i] = z_in[i] (passthrough).
- For word 8: enforces z_out[8] = z_in[8] + 1.

This is the SHAPE the real δ-circuit will eventually have, minus the
internal Merkle/Dilithium work. The toy lets you measure the
overhead of folding itself, without the per-step circuit cost
dominating.

### Bench harness

```rust
fn run_bench(n: usize) -> BenchResult {
    let pp = setup_public_params();
    let mut folder = NovaFolder::new(pp);
    let start = Instant::now();
    let mut z = vec![Fr::zero(); STEP_Z_LEN];
    z[8] = Fr::zero();
    for _ in 0..n {
        let step = ToyStep::new();
        folder.fold(step, &z);
        // Increment the height word natively to track expected z_out.
        z[8] = z[8] + Fr::one();
    }
    let fold_time = start.elapsed();

    let verify_start = Instant::now();
    let valid = folder.verify_accumulated();
    let verify_time = verify_start.elapsed();

    BenchResult { fold_time, verify_time, proof_size: folder.proof_size(), valid }
}
```

Run for n ∈ {1, 10, 100, 1000}. Tabulate.

### Report sections (the `.md` file)

1. Setup environment (CPU, RAM, Rust version, crate versions).
2. Table: per-fold prover time, total verifier time, proof size,
   peak memory — both crates, all n values.
3. Constraint-system bridge cost (lines of glue code each crate
   needs to integrate with arkworks gadgets).
4. Documentation quality + API ergonomics (subjective but useful).
5. **Recommendation** with one sentence justifying it.
6. Risks/caveats specific to the recommended choice.

### Acceptance criteria

- Both crates build and run the toy circuit successfully.
- Table populated with real numbers, no placeholders.
- Recommendation is decisive (not "either works").
- The crate you recommend gets a 30-LOC `StepCircuit` trait impl
  PR-ready against `DeltaStepCircuit` (just the impl signatures,
  bodies CAN be stubs that mirror the synthesize_step in
  `step_circuit.rs`).

---

## Coordination

The Server Beta in-tree author is in parallel doing:
- Refactor `DeltaBlockCircuit::generate_constraints` so that the
  internal logic is callable with pre-allocated z_in (split it into
  `generate_constraints_inner(cs, z_in_root_words, z_in_height) ->
  (z_out_root_words, z_out_height)` and a wrapper that allocates as
  before for the standalone Groth16 path).
- Implement `DeltaStepCircuit::synthesize_step` against that inner
  method (currently returns `AssignmentMissing`).
- Once DS-2 lands, wire the chosen Nova crate's `StepCircuit` impl.

You can work on DS-1 and DS-2 in parallel — they don't depend on the
refactor. Merge order: refactor + your two PRs in any sequence.

## Submission

Commits on `feature/safe-batched-sync-v1.0.2` (the canonical session
branch). Push to `code.quillon.xyz`, NOT GitHub. Run
`git update-server-info` after every push.

Tag commits with `Reviewed-By: Server Beta` once Server Beta has
acked them via the next session.

## Out of scope (do NOT do)

- Don't touch `expand_a_native` itself — DS-1 is purely testing it.
- Don't touch `DeltaBlockCircuit` — Server Beta is refactoring it
  in parallel.
- Don't implement in-circuit SHAKE-128/256 sub-circuits — those are
  separate larger commits scoped after Phase 2 begins.
- Don't pick `nova-snark` vs `arkworks-rs/nova` without writing the
  benchmark + report — the choice has 6+ years of operational
  implications and we need evidence not opinions.

---

Done. Ship one commit per task, smallest-possible diffs, tests
included. Ping when DS-1 + DS-2 are PR-ready.

Co-Authored-By: Server Beta <server-beta@q-narwhalknight.dev>
