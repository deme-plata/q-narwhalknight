# DeepSeek Feedback — DS-1 and DS-2 Submissions

**Date:** 2026-05-15
**Subject:** Why DS-1 and DS-2 (handoff doc `deepseek-handoff-phase2-nova-fold-2026-05-15.md`) could not be merged as submitted; corrections for the next iteration.

Both submissions arrived with the right conclusions but unmergeable artifacts. This doc explains what failed and how to land next time.

---

## DS-1 — ExpandA Conformance Test

### Issues

1. **Non-existent crate.**
   The submission declares `pqcrystals-dilithium5-sys = "0.1"` as a dev-dependency. No such crate exists on crates.io. We do have `pqcrypto-dilithium = "0.5"` already in the workspace (binding to PQClean's safe API: `keypair`, `sign`, `open`), but PQClean does NOT expose `polyvec_matrix_expand` as a callable Rust symbol. So even the corrected crate name would not give you ExpandA in isolation.

2. **Wrong project namespace for the C symbol.**
   The submission FFI-binds to `PQCLEAN_DILITHIUM5_polyvec_matrix_expand`. PQClean and pq-crystals/dilithium are two different upstreams with different naming conventions:
   - pq-crystals exposes: `polyvec_matrix_expand` (no namespace prefix) when compiled with `-DDILITHIUM_MODE=5`.
   - PQClean exposes: `PQCLEAN_DILITHIUM5_polyvec_matrix_expand` (this exact name) BUT only inside its own internal headers — it is not part of the public Rust API surface, and the C source files would have to be vendored and built directly.

3. **Wrong FFI element type.**
   The signature uses `*mut [u32; 256]` for the matrix buffer. The pq-crystals reference uses `int32_t coeffs[256]` for `poly` — coefficients are stored as signed in the canonical representation. ExpandA output happens to land in `[0, q)` so cast bit-patterns coincide, but declaring the C side as `uint32_t*` is incorrect and would be flagged by `-Wstrict-aliasing` and rustc UB lints.

4. **Statistical "Annex C" claim is misleading.**
   The submission claims "FIPS-204 Annex C provides an intermediate value for ML-DSA-87". The seeds in the test are hand-crafted (`0x01`-repeats, `0xAA`-repeats, etc.), not from any KAT, and the inline comment then admits "We can't extract ρ without keygen". A reviewer would accept the test once but flag the prose later as misleading.

### What we landed instead

`crates/q-ivc/tests/expand_a_conformance.rs` (this PR):
- Six committed seeds (zero, ones, AA-pattern, FF, sequential, magic) with a stable row-major SHA-3-256 serialization.
- `EXPECTED_HASHES` constants gated with `#[ignore]` until an external reference produces them.
- `expand_a_print_hashes` bootstrap helper (also `#[ignore]`'d) that prints copy-paste-ready digests.
- Always-on regression checks for shape/determinism/seed-diversity that DO run on every `cargo test`.

`docs/expand-a-conformance-status.md` documents three closure paths (vendor C ref, golden-file pin, indirect via reference keygen) with effort estimates.

### Corrected DS-1 task for next iteration

**Goal:** un-ignore `expand_a_hash_lock_kat` with externally-computed `EXPECTED_HASHES`.

**Concrete steps for DeepSeek:**
1. Clone `https://github.com/pq-crystals/dilithium` (CC0 license — drop-in vendorable).
2. Create `crates/q-ivc/vendor/dilithium-ref/` and copy ONLY these files from the pq-crystals tree:
   - `ref/poly.c`, `ref/poly.h`
   - `ref/polyvec.c`, `ref/polyvec.h`
   - `ref/fips202.c`, `ref/fips202.h`
   - `ref/symmetric-shake.c`, `ref/symmetric.h`
   - `ref/params.h`
3. Add a shim file `crates/q-ivc/vendor/dilithium-ref/qnk_expand_a_shim.c`:
   ```c
   #include "params.h"
   #include "poly.h"
   #include "polyvec.h"

   /* Output buffer: K*L*N int32_t, row-major.
      Caller must allocate K*L*N*sizeof(int32_t) = 8*7*256*4 = 57344 bytes. */
   void qnk_expand_a(int32_t *out, const uint8_t rho[32]) {
       polyvecl mat[K];
       polyvec_matrix_expand(mat, rho);
       for (size_t i = 0; i < K; i++) {
           for (size_t j = 0; j < L; j++) {
               for (size_t n = 0; n < N; n++) {
                   out[(i*L + j)*N + n] = mat[i].vec[j].coeffs[n];
               }
           }
       }
   }
   ```
4. Add `crates/q-ivc/build.rs`:
   ```rust
   fn main() {
       cc::Build::new()
           .files(["vendor/dilithium-ref/poly.c",
                   "vendor/dilithium-ref/polyvec.c",
                   "vendor/dilithium-ref/fips202.c",
                   "vendor/dilithium-ref/symmetric-shake.c",
                   "vendor/dilithium-ref/qnk_expand_a_shim.c"])
           .define("DILITHIUM_MODE", "5")
           .include("vendor/dilithium-ref")
           .compile("dilithium_ref");
   }
   ```
   Add `cc = "1.0"` to `[build-dependencies]` and `links = "dilithium_ref"` to `[package]`.
5. Bind `qnk_expand_a` in `tests/expand_a_conformance.rs`:
   ```rust
   extern "C" {
       fn qnk_expand_a(out: *mut i32, rho: *const u8);
   }
   ```
6. Run `cargo test --test expand_a_conformance -- --ignored` once. Note the printed digests. Paste them into `EXPECTED_HASHES`. Remove `#[ignore]` from `expand_a_hash_lock_kat`.

The shim function makes this a single-symbol surface: no naming-convention guessing, no element-type confusion, no need to claim KAT provenance that doesn't exist.

---

## DS-2 — Nova Crate Evaluation

### Issues

1. **Benchmarks are fabricated.**
   The report's table presents fold/verify/proof-size numbers across 1/10/100/1000 steps for two crates. The accompanying source code is annotated:
   - `nova.get_final_snark()` — *"hypothetical API"*
   - `nova.verify_accumulated(...)` — *"hypothetical"*
   - Proof sizes — literally `1234` (microsoft.rs) and `789` (arkworks.rs), commented `// placeholder`
   - Commit hashes `abcd123` / `wxyz456` for both upstream crates

   The submission's own footnote: *"the exact API calls for nova-snark and ark-nova may differ; the above sketches the intended integration. The final implementation would use the actual trait signatures and folding methods. The benchmark harness collects real timing."* — past tense for code that was never run.

2. **Wrong type identifiers.**
   `microsoft.rs` declares `type F = nova_snark::provider::Bn256Engine;` and then uses `F::Scalar` as a field element. `Bn256Engine` is a curve/group bundle in `nova-snark`'s provider system, not a scalar field type, and the resulting code does not compile.

3. **Misleading conclusion provenance.**
   The conclusion ("adopt ark-nova") is correct — and is correct on architectural grounds. But the document presents the conclusion as data-driven from numbers that don't exist. This corrupts our evidence base.

### What we landed instead

`docs/nova-crate-decision-2026-05-15.md` reaches the same conclusion (ark-nova) on architectural grounds explicitly:
- Our entire gadget stack is arkworks 0.5 (`ark-ff`, `ark-r1cs-std`, etc.).
- `arkworks-rs/nova` consumes our `ConstraintSynthesizer<F>` directly — zero bridge.
- `microsoft/Nova` requires ~800–1200 LOC of `ark_ff::Fp ↔ ff::Field` glue, paid forever.
- Performance acceptance criteria are stated but deferred to measurement against working code.

### Corrected DS-2 task for next iteration

**Goal:** working `StepCircuit<F>` implementation for `DeltaStepCircuit` against `arkworks-rs/nova`.

**Concrete steps for DeepSeek:**
1. Identify a known-good commit SHA on `arkworks-rs/nova` `master`. Pin in `Cargo.toml`:
   ```toml
   [dependencies]
   ark-nova = { git = "https://github.com/arkworks-rs/nova", rev = "<commit-sha>" }
   ```
2. Read the trait at that revision and produce a 30-LOC impl block in `crates/q-ivc/src/recursion/step_circuit.rs` against the **actual** trait signature, not a hypothesized one.
3. Build with `cargo check --package q-ivc` inside Epsilon Docker (`rust:bookworm`) and confirm zero errors before committing.
4. ONLY THEN run benchmarks against the toy 9-word `DeltaStepCircuit` to populate the "Performance acceptance criteria" table in `docs/nova-crate-decision-2026-05-15.md`. Real timings, on a real machine, with the commit SHA of the run.

---

## General principles for next DeepSeek iterations

1. **No code marked "hypothetical" or "may differ" gets a benchmark table.** If the code can't compile, no numbers can be claimed.
2. **No placeholder constants in published artifacts.** `1234`, `789`, `abcd123` are immediate red flags.
3. **`cargo check --package <crate>`-clean is the floor.** Submissions that won't compile are work for the reviewer to either reject or fix — and we'd rather reject than fix.
4. **Verify external crate names against crates.io or the actual GitHub org before declaring a dep.** `pqcrystals-dilithium5-sys` vs the real `pqcrypto-dilithium` (already in our workspace) is a 30-second lookup.
5. **Prose claims need provenance.** "FIPS-204 Annex C provides…" must be either: (a) a direct paraphrase with a `§` and page citation, or (b) replaced with "our deterministic seeds chosen for diversity".
6. **Architectural arguments don't need fake numbers.** A correct decision from clean reasoning beats a correct decision from fabricated evidence every time.

---

## Output we'd love to see next

For the un-ignore-EXPECTED_HASHES path (the natural DS-1.1 follow-up): a single PR that vendors the 5 pq-crystals files + shim + build.rs + populated `EXPECTED_HASHES`, with a `cargo test --test expand_a_conformance` log attached showing all assertions passing on Epsilon Docker.

For the ark-nova StepCircuit path (the natural DS-2.1 follow-up): one PR with the pinned crate, the impl block, a `cargo check` log, and a benchmark commit that uses the real trait API.
