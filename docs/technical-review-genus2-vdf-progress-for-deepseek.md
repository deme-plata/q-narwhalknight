# Technical Review: Genus-2 VDF Progress Report — For DeepSeek Session Continuation

**Date:** 2026-04-14  
**Context:** You (DeepSeek) provided the initial `genus2_cantor.rs` implementation earlier in this session. This document reports what we built on top of it, what bugs we fixed, what's working, and what we need next.

---

## 1. What You Provided (Foundation)

Your code implemented:
- `Poly` — Polynomial arithmetic over F_p
- `JacobianElement` — Mumford representation with serialization
- `double_jacobian()` — Cantor's doubling via tangent method
- `add_distinct()` — Cantor's addition of distinct divisors
- `scalar_mul()` — Double-and-add
- `generate_wesolowski_proof()` / `verify_wesolowski_proof()` — Wesolowski protocol
- `miller_rabin()` — Primality test
- `hash_to_prime()` — Fiat-Shamir challenge derivation
- Test suite with small curve (p=13)

---

## 2. Bugs We Fixed

### Bug 1: `add_distinct()` — Missing Second GCD Step

**Your code:**
```rust
let (d, s, t) = u1.gcd_extended(&u2);
let v = (s*v1 + t*v2) mod d;
let u = (u1 * u2) / d^2;
```

**Correct Cantor (2-step composition):**
```rust
let (d1, e1, e2) = u1.gcd_extended(&u2);           // Step 1
let (d, c1, c3) = d1.gcd_extended(&(v1 + v2));     // Step 2 (MISSING)
let s1 = c1 * e1;
let s2 = c1 * e2;
let s3 = c3;
let u_composed = (u1 * u2) / d^2;                   // Uses d from step 2, not d1
let v_composed = (s1*u1*v2 + s2*u2*v1 + s3*(v1*v2 + f)) / d  mod u_composed;
```

The single-GCD version only works when `gcd(gcd(u1,u2), v1+v2) = 1`, which is common but not guaranteed.

### Bug 2: Polynomial Arithmetic Used `%` Instead of `mod_floor`

Rust's `%` for BigInt preserves the sign of the dividend: `-3 % 13 = -3`, not `10`. This caused negative polynomial coefficients that broke subsequent divisions. Fixed all Poly operations to use a `mod_p()` wrapper that calls `.mod_floor()`.

### Bug 3: Miller-Rabin Used `rand::thread_rng()` (API Incompatible)

The `rand` 0.9.x crate changed the `thread_rng()` API. We replaced random witnesses with deterministic ones:
- 12 fixed small primes: {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
- 8 hash-derived witnesses: `SHA3("miller-rabin-witness" || n_bytes || counter)`

This is more appropriate for a blockchain (reproducible, no RNG dependency).

### Bug 4: `Poly::mul()` Panicked on Empty Coefficients

Zero polynomials had empty `coeffs` vectors. Multiplication would compute `len - 1` which underflows for `usize`. Added explicit zero-check at top of `mul()`.

### Bug 5: `hash_to_curve_point()` Assumed `p ≡ 3 mod 4`

Your code used `y = f(x)^((p+1)/4)` which only works for `p ≡ 3 mod 4`. We implemented full Tonelli-Shanks algorithm that works for any odd prime. (Our specific prime IS `3 mod 4`, but the code should be generic.)

### Bug 6: `WesolowskiProof` Had No Serialization

The proof needed to travel from miner → server as bytes. Added:
```rust
impl WesolowskiProof {
    fn to_bytes(&self, field_bytes: usize) -> Vec<u8>;      // 145 bytes for pq128
    fn from_bytes(bytes: &[u8], curve: &CurveParams) -> Result<Self>;
}
```

### Bug 7: Proof Generation Materialized `2^T` as BigUint

For T=1,000,000, `BigUint::one() << T` creates a 125KB integer. We replaced with iterative long-division accumulation that computes `q = floor(2^T / c)` bit-by-bit during a single pass:

```rust
let mut r = BigInt::zero();
let mut pi = JacElement::identity();
for i in (0..iterations).rev() {
    r = (2 * r) mod c;
    pi = [2]pi;
    if r >= c {
        r -= c;
        pi = pi + g;    // bit i of q is 1
    }
}
```

This never allocates more than a 128-bit integer for `r`.

---

## 3. What's Working Now (19 Tests Passing)

| Test | Description | Status |
|------|-------------|--------|
| `test_mod_p_always_positive` | Field arithmetic never negative | PASS |
| `test_sqrt_mod_p` | Tonelli-Shanks for any prime | PASS |
| `test_poly_basics` | Add, mul, div over F_p | PASS |
| `test_poly_mul_zero` | Zero polynomial edge case | PASS |
| `test_generator_valid` | from_seed → valid element (v² ≡ f mod u) | PASS |
| `test_identity_doubling` | [2]O = O | PASS |
| `test_double_validates` | 2g, 4g valid on Jacobian | PASS |
| `test_add_identity` | g + O = g | PASS |
| `test_negate_and_add_to_identity` | g + (-g) = O | PASS |
| `test_scalar_mul_consistency` | [n]g matches iterative addition | PASS |
| `test_vdf_evaluation` | T sequential doublings, deterministic | PASS |
| `test_serialization_roundtrip` | Canonical bytes encode/decode (all degrees) | PASS |
| `test_miller_rabin` | Deterministic primality (no RNG) | PASS |
| `test_hash_to_curve_point` | Point on curve (y² = f(x)) | PASS |
| `test_from_seed` | Deterministic, different seeds → different elements | PASS |
| `test_wesolowski_proof_small` | Proof verifies + tamper detection | PASS |
| `test_proof_serialization_roundtrip` | Proof bytes → proof → verify (small curve) | PASS |
| `test_proof_serialization_pq128` | Proof 145 bytes, roundtrip on real 256-bit curve | PASS |
| `test_pq128_basic` | Doubling + scalar mul on real pq128 curve | PASS |

---

## 4. Benchmark Results (Release Mode, Server Beta — 4 cores)

| Metric | Value |
|--------|-------|
| Per doubling | **465 μs** |
| Throughput | **2,150 doublings/sec** |
| T for 2-second VDF | **~4,300 iterations** |
| Proof verification (T=50) | **29 ms** |
| Total miner cost (T=50) | **29 ms** (eval + proof) |

---

## 5. Server Integration (Completed)

### PATH A Verification (main.rs)

Replaced the stub (which only checked SHA3 hash of first 32 proof bytes) with real Wesolowski verification:

```rust
// Height-gated: only active when Q_GENUS2_VDF_ACTIVATION_HEIGHT is set
if !genus2_vdf_active {
    return None; // Reject VDF submissions before activation
}

// Thread-local curve params (zero alloc per submission)
thread_local! {
    static CURVE: CurveParams = CurveParams::pq128();
}

CURVE.with(|curve| {
    let seed = blake3(challenge_bytes || nonce_le_bytes);
    let g = JacElement::from_seed(seed.as_bytes(), curve);
    let y = JacElement::from_bytes(&vdf_output, curve);
    let proof = WesolowskiProof::from_bytes(&vdf_proof, curve);
    verify_proof(&g, &y, &proof, T, curve)  // O(log T) cost
});
```

### Challenge Endpoint (handlers.rs)

Added 5 new optional fields to `MiningChallengeResponse`:
```json
{
    "vdf_lane_active": true,
    "vdf_curve_id": "pq128",
    "vdf_target_iterations": 4300,
    "vdf_reward_share_bps": 5000,
    "blake3_reward_share_bps": 5000
}
```

All `skip_serializing_if = "Option::is_none"` — old miners see zero change.

---

## 6. What We Need From You (Enhancement Requests)

### Request A: Optimized Explicit Formulas for Degree-2 Doubling

The current doubling uses generic polynomial arithmetic (Poly GCD, div_rem, etc.). For VDF mining where millions of doublings happen, we need the explicit formulas from Lange 2002 that avoid polynomial GCD entirely.

For a genus-2 curve `y² = f(x)` with `h(x) = 0` (our case), and input `D = (u, v)` with `deg(u) = 2`:

The explicit formulas compute the 4 output coefficients `(u1', u0', v1', v0')` directly from `(u1, u0, v1, v0)` and the curve coefficients, using only field multiplications, additions, and one field inversion.

**Can you provide the explicit degree-2 doubling formulas for our curve `y² = x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀` in Rust?**

The function signature should be:
```rust
/// Optimized doubling for degree-2 elements using explicit formulas (Lange 2002).
/// Avoids polynomial GCD — uses only field arithmetic (mul, add, inv).
/// Returns Err if the element is degenerate (2v ≡ 0 mod u).
pub fn double_jacobian_explicit(
    u1: &BigInt, u0: &BigInt,
    v1: &BigInt, v0: &BigInt,
    curve: &CurveParams,
) -> Result<(BigInt, BigInt, BigInt, BigInt)>;  // (u1', u0', v1', v0')
```

This would give us a 5-10× speedup on doubling (the VDF hot path), reducing the per-doubling cost from ~465μs to potentially ~50-100μs.

**References:**
- "Handbook of Elliptic and Hyperelliptic Curve Cryptography" — Chapter 14.3
- "Formulae for Arithmetic on Genus 2 Hyperelliptic Curves" (Lange, 2005)
- Explicit formula database: https://www.hyperelliptic.org/EFD/g2p/

### Request B: Incremental Wesolowski Proof (Single-Pass)

Currently the prover does two passes:
1. **Eval pass:** Compute `y = [2^T]g` (T doublings)
2. **Proof pass:** Compute `π = [q]g` where `q = floor(2^T/c)` (another T doublings)

Total miner cost: 2× the VDF evaluation time.

**Can you provide a single-pass Wesolowski proof generation?**

The idea: during the evaluation pass, maintain a running quotient accumulator. Since `c = hash_to_prime(g, y)` depends on `y` which isn't known until the eval completes, we can't compute `q` during eval. 

HOWEVER, there's a known optimization: **delayed proof generation with segment caching.**

1. During eval, cache checkpoints every K steps: `g_0, g_K, g_2K, ..., g_T = y`
2. After eval, compute `c = hash_to_prime(g, y)`
3. For each segment `[iK, (i+1)K]`, compute the partial contribution to π
4. Total proof cost: O(T) but with better cache locality than a second full pass

Alternatively, is there a way to split the proof into multiple small proofs (one per segment) that can be verified independently? This would allow parallel proof generation.

### Request C: Montgomery Form for Constant-Time Field Arithmetic

For production, we should use Montgomery multiplication for field arithmetic instead of generic BigInt. This would:
- Eliminate variable-time modular reduction
- Use fixed-width 256-bit arithmetic (4 × u64 limbs)
- Achieve ~50ns per field multiplication (vs ~5μs with BigInt)

**Can you provide a `MontgomeryField256` implementation for our 256-bit prime?**

```rust
pub struct MontgomeryField256 {
    limbs: [u64; 4],  // 256-bit value in Montgomery form
}

impl MontgomeryField256 {
    fn mul(&self, other: &Self, p: &MontgomeryParams) -> Self;
    fn add(&self, other: &Self, p: &MontgomeryParams) -> Self;
    fn sub(&self, other: &Self, p: &MontgomeryParams) -> Self;
    fn inv(&self, p: &MontgomeryParams) -> Self;
    fn to_normal(&self, p: &MontgomeryParams) -> [u64; 4];
    fn from_normal(val: &[u64; 4], p: &MontgomeryParams) -> Self;
}
```

This is the single biggest performance win — it would take doubling from ~465μs to ~5-10μs, making T=100,000 feasible for a 2-second VDF (much more room for difficulty adjustment).

---

## 7. Architecture Decision: What NOT to Change

1. **Curve parameters** — Don't change `pq128`. The curve `y² = x⁵ + x² - 1` over the 256-bit prime is fixed.
2. **Wesolowski protocol** — Don't switch to Pietrzak. Wesolowski has simpler verification.
3. **Fiat-Shamir domain separator** — Must remain `"genus2-wesolowski-challenge-v1"` for protocol compatibility.
4. **Serialization format** — JacElement canonical bytes are 129 bytes (degree-2), WesolowskiProof is 145 bytes. Don't change.
5. **hash_to_curve_point domain separator** — Must remain `"genus2-hash-to-curve-v1"`.

---

## 8. Summary

Your foundation code was solid — the struct design, the Wesolowski protocol flow, and the test structure were all correct. The bugs were in the mathematical details (Cantor's 2-step GCD, mod_floor vs %, Tonelli-Shanks) and practical integration (serialization, deterministic Miller-Rabin).

We now have a working, tested, server-integrated VDF that's ready for Docker testing. The next step is performance optimization (Request A: explicit formulas) to bring the per-doubling cost down from 465μs to <100μs, which would make the VDF much more flexible for difficulty adjustment.

**Priority of requests: A > C > B** (explicit formulas first, then Montgomery, then single-pass proof).
