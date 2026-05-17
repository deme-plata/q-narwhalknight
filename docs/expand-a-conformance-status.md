# FIPS-204 ExpandA — External Conformance Status

**As of:** 2026-05-15
**Owner:** Server Beta (with DeepSeek follow-up)
**Status:** Byte-order fix landed (commit `0e0b227a4`). External cross-reference still **PENDING**.

---

## 1. What we have

| Coverage | Source | Confidence |
|---|---|---|
| Shape (56 polynomials of degree 256) | In-module test `expand_a_produces_k_times_l_polynomials` | High |
| Range (every coeff < q = 8 380 417) | In-module test `expand_a_coefficients_are_in_zq` | High |
| Determinism (same ρ → same A) | In-module test `expand_a_is_deterministic` | High |
| Seed diversity (different ρ → different A) | In-module test `expand_a_different_seeds_produce_different_matrices` | High |
| Uniform-ish distribution across [0, q) | In-module test `expand_a_coefficients_appear_uniformly_distributed` | Statistical sanity |
| **Byte-order correctness (ρ ∥ i ∥ j vs ρ ∥ j ∥ i)** | Commit `0e0b227a4` after DeepSeek peer review | **Code-review only** |

## 2. What we lack

A test that proves byte-by-byte equality with an independently-implemented FIPS-204 ExpandA reference. The byte-order bug DeepSeek caught (`0e0b227a4`) was latent — every in-module property test passed with the buggy seed order. **Only an external reference can detect that class of bug.**

## 3. The three paths to closure

### Path A — Vendor pq-crystals/dilithium C reference  *(recommended, ~2 hrs)*

- License: CC0 (public domain), drop-in compatible
- Files to vendor under `crates/q-ivc/vendor/dilithium-ref/`:
  - `poly.c` `poly.h` (polynomial ops including `poly_uniform`)
  - `polyvec.c` `polyvec.h` (matrix expand wrapper)
  - `fips202.c` `fips202.h` (SHAKE-128/256)
  - `symmetric-shake.c` `symmetric.h` (SHAKE-based stream)
  - `params.h` (ML-DSA-87 parameters)
- Add `cc` build script in `crates/q-ivc/build.rs` compiling with `-DDILITHIUM_MODE=5` and a tiny shim `expand_a_shim.c` exposing one C symbol:
  ```c
  void qnk_expand_a(int32_t *out_kl_n, const uint8_t rho[32]);
  ```
- Bind via `extern "C"` in `tests/expand_a_conformance.rs`
- Run once locally; copy resulting digests into `EXPECTED_HASHES`; un-ignore `expand_a_hash_lock_kat`

**Pros:** truly independent (different language, different toolchain), one-time cost, permanent coverage forever.
**Cons:** adds ~5 vendored .c files and a `cc` build dep, slightly slows test compilation.

### Path B — Golden-file pin from current impl  *(stop-gap, ~30 min)*

- Run `expand_a_print_hashes` from `tests/expand_a_conformance.rs` on a clean Docker build
- Paste the printed digests into `EXPECTED_HASHES`
- Un-ignore `expand_a_hash_lock_kat`

**Pros:** zero new deps, immediately catches *future* regressions.
**Cons:** does not prove *current* correctness — only pins it. If the byte-order bug had re-emerged today, this would lock the wrong hashes in.

**Use case:** acceptable if Path A is paired with it in the same PR (A produces the hashes, B is how they're committed). NOT acceptable as a standalone substitute for A.

### Path C — Indirect cross-check via reference keygen  *(complex, ~3 hrs)*

- Use `pqcrypto-dilithium 0.5` (already a workspace dep) to call `keypair()` with a deterministic seed
- Unpack the resulting `(pk, sk)` per FIPS-204 §4.2:
  - `pk = ρ ∥ t₁_packed`
  - `sk = ρ ∥ K ∥ tr ∥ s₁_packed ∥ s₂_packed ∥ t₀_packed`
- Compute `A := expand_a_native(ρ)` (our impl)
- Compute `t' := A · s₁ + s₂` (we already have NTT routines for this)
- Verify `HighBits(t', 2^d) == t₁`

**Pros:** end-to-end verification using a reference keygen result, no .c vendoring.
**Cons:** requires unpacking s₁/s₂/t₀ from `sk` byte layout (off-by-one risk), and t' = A · s₁ + s₂ computation must itself be correct (regress-risk on a different gadget). The dependency on our own NTT correctness weakens isolation.

## 4. Recommendation

**Path A + Path B in the same follow-up commit.**

A produces the digests from an independent toolchain; B is just the commit format. Together they give permanent, sound conformance without standing on our own NTT. Estimated total work: 2 hours.

Path C is interesting future work for *full FIPS-204 algorithm-level* conformance (signing, verification end-to-end), but is over-investment for closing the ExpandA gap alone.

## 5. Tracking

- This file commits the scaffolding: `crates/q-ivc/tests/expand_a_conformance.rs` (seeds + serialization locked, hash assertions gated).
- Closure ticket: when `EXPECTED_HASHES` are filled and `#[ignore]` removed, delete this file or move to `docs/closed/`.
