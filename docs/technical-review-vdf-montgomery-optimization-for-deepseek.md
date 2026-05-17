# Technical Review: Montgomery Field & Polynomial Optimization for Genus-2 VDF

**Date:** 2026-04-15
**Project:** Quillon Graph (Q-NarwhalKnight) — $1B market cap blockchain
**File:** `crates/q-vdf/src/genus2_cantor.rs`
**Context:** Genus-2 hyperelliptic curve VDF for CPU-fair mining (dual-lane with BLAKE3)

---

## 1. Current Performance & Goal

We have a working Genus-2 VDF over the curve y² = x⁵ + x² - 1 over a 256-bit prime field. VDF miners compute T=4300 sequential doublings in the Jacobian J(C), then generate a Wesolowski proof.

**Current performance (release mode):**
| Operation | Time | Per-doubling |
|-----------|------|-------------|
| Eval (T=4300 doublings) | ~2.24s | 521 µs |
| Proof generation | ~0.35s | — |
| **Total per proof** | **~2.6s** | — |

**Target:** 100-150 µs per doubling → T=4300 in ~0.5s → total ~0.7s per proof.

**Gap:** 521 µs vs 100 µs = **~5× slower than necessary.** The bottleneck is that all field arithmetic uses `BigInt::mod_floor()` (arbitrary-precision division) instead of Montgomery multiplication (fixed-width multiply + shift).

---

## 2. Curve & Field Parameters

```
Curve:  y² = x⁵ + x² - 1  (genus 2, hyperelliptic)
Prime:  p = 115792089237316195423570985008687907853269984665640564039457584007913129639747
        p = 2²⁵⁶ - 189  (256-bit, p ≡ 3 mod 4)
        p in hex: FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF43
Jacobian elements: Mumford representation (u(x), v(x)) with deg(u) ≤ 2, deg(v) < deg(u)
Stored as: (u1, u0, v1, v0, degree) where u(x) = x² + u1·x + u0, v(x) = v1·x + v0
```

For Montgomery form with R = 2²⁵⁶:
```
R mod p = 189  (since p = 2²⁵⁶ - 189)
R² mod p = 189² mod p = 35721
p⁻¹ mod 2⁶⁴ — computed via Newton iteration (6 rounds)
-p⁻¹ mod 2⁶⁴ — the "p_inv" used in CIOS
```

---

## 3. Problem A: Montgomery CIOS Carry Bug

### 3.1 Existing Implementation (lines 219-271)

We have a CIOS (Coarsely Integrated Operand Scanning) Montgomery multiplier that works correctly for **short chains** of operations (mul, add, sub all pass tests) but **diverges after ~256 sequential multiplications** (as required by square-and-multiply inversion).

```rust
/// Montgomery multiplication: z = x * y * R^{-1} mod p (CIOS method).
fn mont_mul(x: &[u64; 4], y: &[u64; 4], params: &MontgomeryParams, z: &mut [u64; 4]) {
    let mut t = [0u64; 5];
    for i in 0..4 {
        // Multiply-accumulate
        let mut carry: u128 = 0;
        for j in 0..4 {
            let prod = (x[j] as u128) * (y[i] as u128) + (t[j] as u128) + carry;
            t[j] = prod as u64;
            carry = prod >> 64;
        }
        t[4] = t[4].wrapping_add(carry as u64);

        // Montgomery reduction
        let m = t[0].wrapping_mul(params.p_inv);
        carry = 0;
        for j in 0..4 {
            let prod = (m as u128) * (params.p[j] as u128) + (t[j] as u128) + carry;
            t[j] = prod as u64;
            carry = prod >> 64;
        }
        t[4] = t[4].wrapping_add(carry as u64);

        // Shift right by one limb
        t[0] = t[1];
        t[1] = t[2];
        t[2] = t[3];
        t[3] = t[4];
        t[4] = 0;
    }

    // Final conditional subtraction: if t >= p, subtract p
    let mut borrow: u64 = 0;
    let mut tmp = [0u64; 4];
    for i in 0..4 {
        let a = t[i] as u128;
        let b = params.p[i] as u128 + borrow as u128;
        if a >= b {
            tmp[i] = (a - b) as u64;
            borrow = 0;
        } else {
            tmp[i] = (a + (1u128 << 64) - b) as u64;
            borrow = 1;
        }
    }
    if borrow == 0 {
        *z = tmp;
    } else {
        z.copy_from_slice(&t[..4]);
    }
}
```

### 3.2 Symptoms

```
test_montgomery_roundtrip ... ok     ✅ (from_bigint → to_bigint)
test_montgomery_arithmetic ... ok    ✅ (single mul, add, sub)
test_montgomery_inverse ... FAILED   ❌ (a * a^(p-2) diverges)
```

The inverse uses `a^(p-2)` via square-and-multiply (256 squarings + ~128 multiplications). After ~256 sequential `mont_mul` calls, the accumulated error makes `a * a⁻¹ ≠ 1`.

### 3.3 Suspected Root Causes

1. **Carry overflow in `t[4]`:** The line `t[4] = t[4].wrapping_add(carry as u64)` could lose a bit if t[4] already held a carry from the multiply-accumulate phase. After the reduction, t[4] should be at most 1, but `wrapping_add` silently truncates overflow.

2. **Final subtraction logic:** The conditional subtraction uses an `if a >= b` branch per limb, which is non-constant-time and may have an edge case when `t == p` exactly (should subtract) or when intermediate values exceed 2p (need double subtraction).

3. **p_inv computation:** Newton iteration for modular inverse:
   ```rust
   let mut inv: u64 = 1;
   for _ in 0..6 {
       inv = inv.wrapping_mul(2u64.wrapping_sub(p_limbs[0].wrapping_mul(inv)));
   }
   let p_inv = inv.wrapping_neg();
   ```
   6 iterations gives 2⁶ = 64 bits of convergence, which is correct. But verify: we need `p[0] * p_inv ≡ -1 mod 2⁶⁴`.

### 3.4 What We Need

A correct 4-limb (256-bit) CIOS Montgomery multiplier where:
- `mont_mul(a, b)` computes `a·b·R⁻¹ mod p` for R = 2²⁵⁶
- `from_bigint(x)` = `mont_mul(x_limbs, R² mod p)` → x in Montgomery form
- `to_bigint(xR)` = `mont_mul(xR, [1,0,0,0])` → x in normal form
- `inv(a)` = `a^(p-2)` via square-and-multiply with `mont_mul` → must round-trip correctly after 256+ sequential multiplications
- The final subtraction must handle t ∈ [0, 2p) correctly

**Tests that must pass:**
```rust
// 1. Roundtrip
let mont = from_bigint(42); assert_eq!(to_bigint(mont), 42);

// 2. Single ops
assert_eq!(to_bigint(mul(ma, mb)), (a * b) % p);
assert_eq!(to_bigint(add(ma, mb)), (a + b) % p);
assert_eq!(to_bigint(sub(ma, mb)), (a - b + p) % p);

// 3. Inverse (THE CRITICAL ONE)
let ma_inv = inv(ma);  // a^(p-2) via 256 squarings + ~128 muls
assert_eq!(to_bigint(mul(ma, ma_inv)), 1);

// 4. Long chain correctness
let mut x = from_bigint(7);
for _ in 0..1000 { x = mul(x, x); }
// Must match BigInt: 7^(2^1000) mod p
```

---

## 4. Problem B: Inline Polynomial Composition for Degree-2 Doubling

### 4.1 Context

The VDF hot loop calls `double_fast()` 4300 times per evaluation. Each doubling on the Jacobian involves:

1. **Inline GCD** (already optimized — 2×2 Cramer's rule, 2 mod_inv + 6 field ops)
2. **Compose:** `k = (f - v²)/u`, `l = t·k mod u`, `u_comp = u²`, `v_comp = v + u·l`
3. **Reduce:** `u_new = (f - v_comp²)/u_comp`, make monic, `v_new = -v_comp mod u_new`

Steps 2-3 currently use `pmul()` and `pdivrem()` helper functions that allocate `Vec<BigInt>` and call `BigInt::mod_floor()` per coefficient. For degree-2 polynomials, all sizes are known at compile time.

### 4.2 Current Code (lines 1290-1365 approximately)

After the inline GCD gives us `t_coeffs = [t0, t1]`:

```rust
// k = (f - v²) / u — exact division
let v_sq = pmul(&v_coeffs, &v_coeffs, &p);         // Vec alloc, 4 muls, 3 mod_p
let mut f_minus_vsq = vec![BigInt::zero(); 6];       // Vec alloc
for i in 0..6 { f_minus_vsq[i] = f[i].clone(); }    // 6 clones
for i in 0..v_sq.len() { f_minus_vsq[i] = (&f_minus_vsq[i] - &v_sq[i]).mod_floor(&p); }
let (k, _rem) = pdivrem(&f_minus_vsq, &u_coeffs, &p);  // Vec alloc, loop with mod_floor

// l = t * k mod u
let tk = pmul(&t_coeffs, &k, &p);                   // Vec alloc, muls
let (_, l) = pdivrem(&tk, &u_coeffs, &p);           // Vec alloc, loop

// u_comp = u², v_comp = v + u*l
let u_comp = pmul(&u_coeffs, &u_coeffs, &p);        // Vec alloc, 9 muls
let u_l = pmul(&u_coeffs, &[l0, l1], &p);           // Vec alloc
let mut v_comp = v_coeffs.to_vec();                  // Vec alloc
// ... accumulation with mod_floor per element

// u_new = (f - v_comp²) / u_comp, make monic
let vc_sq = pmul(&v_comp, &v_comp, &p);             // Vec alloc
let mut num = vec![BigInt::zero(); ...];             // Vec alloc
let (mut u_new, _) = pdivrem(&num, &u_comp, &p);    // Vec alloc, loop

// v_new = -v_comp mod u_new
let neg_vc: Vec<BigInt> = v_comp.iter().map(...).collect(); // Vec alloc
let (_, mut v_new) = pdivrem(&neg_vc, &u_new, &p);  // Vec alloc
```

**Total allocations per doubling: ~12 Vec<BigInt>** (each with 2-9 BigInt elements)
**Total mod_floor calls: ~60+** (each is a BigInt division)

### 4.3 What We Need

Replace the entire compose+reduce section with **fixed-size array arithmetic** that:

1. Uses `[BigInt; N]` arrays instead of `Vec<BigInt>` (zero allocation)
2. Exploits the known degrees: u is degree 2 (monic), v is degree ≤ 1, f is degree 5 (monic)
3. Computes polynomial products as explicit coefficient formulas (no loops for degree-2 × degree-2)
4. Does modular reduction (mod u) using the known relation `x² ≡ -u1·x - u0`
5. Minimizes `mod_floor` calls by deferring reduction

**Specifically, for u = x² + u1·x + u0 (monic), v = v1·x + v0:**

**Composition step (compute k, l, u_comp, v_comp):**
```
v² = v1²·x² + 2·v1·v0·x + v0²   (3 muls)
f - v² = (1-v1²)·x⁵ + ... (known coefficients from curve a0..a4)
k = (f - v²) / u   (exact division, degree 3 — can do by synthetic division with known coefficients)
l = t·k mod u  (reduce degree-≤4 product mod degree-2 → degree ≤ 1, use x² = -u1·x - u0)
u_comp = u² = x⁴ + 2u1·x³ + (u1² + 2u0)·x² + 2u0·u1·x + u0²  (3 muls: u1², u0², u0·u1)
u·l = (x² + u1·x + u0)(l1·x + l0) — degree 3 (6 muls but can share with u_comp)
v_comp = v + u·l   (coefficient-wise addition)
```

**Reduction step (compute u_new = (f - v_comp²)/u_comp, v_new = -v_comp mod u_new):**
```
v_comp is degree ≤ 3, v_comp² is degree ≤ 6
f - v_comp² is degree ≤ 6 (or 5 if leading cancels)
u_new = (f - v_comp²) / u_comp  (exact division, result degree 2, make monic)
v_new = -v_comp mod u_new  (reduce degree-3 poly mod degree-2, use x² = -u1_new·x - u0_new)
```

### 4.4 Ideal Output

A function `double_fast_inline()` that:
- Takes `(u1, u0, v1, v0: &BigInt, curve: &CurveParams)` 
- Returns `(u1_new, u0_new, v1_new, v0_new: BigInt)`
- Uses only stack-allocated `[BigInt; N]` or individual `BigInt` variables
- Does at most ~30 field multiplications (mul + mod_floor) and ~20 add/sub
- No `Vec` allocation, no `Poly` struct, no `pmul()`/`pdivrem()` calls
- Every intermediate polynomial is represented as explicit named coefficients

---

## 5. Integration: Using Montgomery Field in double_fast

Once both fixes work independently:

1. Replace `BigInt` field ops in `double_fast_inline()` with `MontField256` ops
2. Convert `(u1, u0, v1, v0)` to Montgomery form once at the start of `evaluate_vdf()`
3. All 4300 doublings happen in Montgomery form (no `BigInt` allocation)
4. Convert back to `BigInt` at the end
5. The curve constants `(a0, a1, a2, a3, a4)` and precomputed values are also in Montgomery form

**Expected speedup:** 
- BigInt mod_floor: ~500ns per operation (256-bit division)
- Montgomery mul: ~20ns per operation (4 × 4 = 16 multiply-accumulates + shift)
- With ~30 field muls per doubling: 30 × 500ns = 15µs → 30 × 20ns = 0.6µs
- Total doubling: currently 521µs → target ~50-100µs (dominated by inversions for GCD)

---

## 6. Deliverables Requested

### Deliverable A: Fixed Montgomery CIOS Multiplier
- Fix the carry bug in `mont_mul()` (or rewrite from scratch)
- Must pass the inverse test: `a * a^(p-2) mod p == 1` for all a
- Must work for our specific prime p = 2²⁵⁶ - 189
- Include the `MontgomeryParams`, `MontField256` structs (from_bigint, to_bigint, mul, add, sub, inv)
- All tests from Section 3.4 must pass

### Deliverable B: Inline Polynomial Composition
- Replace the compose+reduce section of `double_fast()` with explicit coefficient formulas
- No `Vec` allocation, no `Poly` struct usage
- Input: `(u1, u0, v1, v0, t0, t1, p, a0, a2)` where t is the GCD cofactor
- Output: `(u1_new, u0_new, v1_new, v0_new)`
- Must match the output of the current `double_fast()` for all inputs
- Target: ~30 field multiplications per doubling

### Deliverable C (Bonus): Combined Montgomery + Inline Doubling
- A `double_mont()` function that does the entire doubling in Montgomery form
- Input/output in Montgomery form
- The hot loop becomes: `for _ in 0..T { current = double_mont(&current, &curve_mont); }`

---

## 7. Test Vectors

### Field arithmetic:
```
p = 115792089237316195423570985008687907853269984665640564039457584007913129639747
a = 42
b = 123456789987654321
a * b mod p = 5185185179481481482
a^(-1) mod p = 88117163180622625322531798863572883174634940507538906315825634863122574773807
a * a^(-1) mod p = 1
```

### Doubling (from test_double_fast_pq128):
```
Curve: y² = x⁵ + x² - 1, p as above
Input: JacElement from_seed(b"test_vdf_input_data_1234567890ab", curve)
After 1 doubling: must match double_jacobian() output exactly
After 10 doublings: must match evaluate_vdf(g, 10, curve) exactly
```

### VDF pipeline (from bench_pq128_vdf_short):
```
T = 50
g = JacElement::from_seed(b"bench_vdf_seed_000", curve)
y = evaluate_vdf(g, 50, curve)
proof = generate_proof(g, y, 50, curve)
verify_proof(g, y, proof, 50, curve) == true
```

---

## 8. Existing Code Structure

The code lives in a single file `crates/q-vdf/src/genus2_cantor.rs` (~2250 lines). Key sections:

| Lines | Component | Status |
|-------|-----------|--------|
| 36-47 | `mod_p()`, `mod_inv()` | Working (slow — uses BigInt) |
| 49-125 | `sqrt_mod_p()` (Tonelli-Shanks) | Working |
| 127-344 | Montgomery field (`MontgomeryParams`, `MontField256`) | **BROKEN** — carry bug in `inv()` |
| 350-596 | `Poly` struct (arbitrary-degree polynomials) | Working but slow |
| 598-658 | `CurveParams` (pq128 curve) | Working |
| 660-810 | `JacElement` (Mumford representation) | Working |
| 900-1000 | `add_distinct()`, `cantor_reduce()` | Working |
| 1002-1056 | `double_jacobian()` (generic, slow) | Working (reference impl) |
| 1127-1170 | `pmul()`, `pdivrem()` (inline helpers) | Working but allocates |
| 1172-1356 | `double_fast()` (optimized degree-2) | Working, **needs Deliverable B** |
| 1368-1404 | `evaluate_vdf()` (hot loop) | Working |
| 1457-1519 | `generate_proof()` (Wesolowski) | Working (just optimized) |
| 1524-1600 | `verify_proof()` | Working |

---

## 9. Constraints

- **Rust only** — no C/assembly (yet). Pure safe Rust is fine.
- **No external Montgomery crate** — we want self-contained code in this file.
- `num-bigint` crate is available for BigInt/BigUint (already a dependency).
- The prime p = 2²⁵⁶ - 189 is special: very close to 2²⁵⁶, which simplifies some arithmetic.
- Must be `#[inline]` friendly — the doubling function is called 4300+ times in a tight loop.
- Correctness over speed — a correct but slightly slower Montgomery is better than a fast but subtly wrong one.

---

*This review is part of the Quillon Graph VDF mining system. The VDF ensures CPU-fair mining alongside GPU BLAKE3 mining in a 50/50 dual-lane reward split.*
