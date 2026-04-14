# Technical Review: Genus-2 Jacobian VDF — Implementation Request for DeepSeek

**Date:** 2026-04-14  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Purpose:** Request DeepSeek to implement the cryptographic core for a Genus-2 Hyperelliptic Curve VDF  
**Language:** Rust  
**Dependencies:** `num-bigint 0.4` (with `serde` + `rand` features), `num-traits 0.2`, `num-integer 0.1`, `sha3 0.10`

---

## 1. Problem Statement

We have a dual-lane mining system where:
- **GPU lane:** BLAKE3 proof-of-work (working, deployed)
- **CPU lane:** Genus-2 Jacobian VDF (infrastructure ready, **cryptographic core incomplete**)

The CPU lane uses sequential repeated doubling in the Jacobian of a genus-2 hyperelliptic curve as a Verifiable Delay Function. This is inherently sequential (GPUs can't parallelize it) and makes CPU mining profitable again.

**What we need implemented:**
1. Correct Cantor's algorithm for doubling in `J(C)` (Mumford representation)
2. Wesolowski proof generation adapted for Jacobian groups
3. Wesolowski proof verification (O(log T) cost)
4. Proper serialization/deserialization of Jacobian elements
5. Miller-Rabin primality test for Fiat-Shamir challenge generation

---

## 2. Mathematical Specification

### 2.1 The Curve

Genus-2 hyperelliptic curve over `F_p`:

```
C: y² = f(x) = x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀
```

For our 128-bit post-quantum security level (`pq128`):
```
p = 115792089237316195423570985008687907853269984665640564039457584007913129639747
f(x) = x⁵ + x² - 1    (i.e., a₄=0, a₃=0, a₂=1, a₁=0, a₀=-1)
```

This is a 256-bit prime field. The Jacobian `J(C)` has order approximately `p²`.

### 2.2 Mumford Representation

Points on `J(C)` are represented as pairs of polynomials `(u(x), v(x))` where:
- `u(x)` is monic, `deg(u) ≤ 2` (genus)
- `deg(v) < deg(u)`
- `u | (v² - f)` (i.e., `v(x)² ≡ f(x) mod u(x)`)

**Degree-2 (generic) element:**
```
u(x) = x² + u₁x + u₀
v(x) = v₁x + v₀
```

**Degree-1 element:**
```
u(x) = x + u₀     (monic degree 1, so u₁ is the implicit leading 1)
v(x) = v₀          (constant)
```

**Identity (neutral element):**
```
u(x) = 1, v(x) = 0, degree = 0
```

### 2.3 Cantor's Algorithm for Doubling (D → 2D)

Given `D = (u, v)` on `J(C)` where `C: y² = f(x)`, compute `2D`.

**Full algorithm (Cantor 1987, refined by Lange 2002):**

```
INPUT: D = (u, v) with u | (v² - f), deg(u) ≤ 2, deg(v) < deg(u)
OUTPUT: 2D = (u', v') in reduced Mumford form

STEP 1: COMPOSITION
  Compute d = gcd(u, 2v)
  
  If deg(u) = 2:
    Compute d₁ = gcd(u, 2v)  [extended Euclidean → d₁, s₁, s₂ such that d₁ = s₁·u + s₂·(2v)]
    
    If d₁ = 1 (typical case for doubling when gcd(u, 2v) = 1):
      Compute l = s₂ · (f - v²) / u  mod u
      u_composed = u²
      v_composed = v + u·l  mod u_composed
    
    If d₁ ≠ 1:
      Handle degenerate case (rare)

STEP 2: REDUCTION  
  While deg(u_composed) > 2:
    u' = (f - v_composed²) / u_composed     [exact polynomial division]
    Make u' monic (divide by leading coefficient)
    v' = -v_composed  mod u'
    u_composed = u'
    v_composed = v'

STEP 3: NORMALIZE
  Reduce all coefficients mod p
  Ensure u' is monic
  Ensure deg(v') < deg(u')

RETURN (u', v')
```

**For the specific case of doubling a degree-2 element (most common in VDF):**

The optimized explicit formulas from Lange 2002 are:

```
Given D = (u, v) where u = x² + u₁x + u₀, v = v₁x + v₀

1. Compute d = gcd(u, 2v₁x + 2v₀)
   Since u is degree 2 and 2v is degree 1 (or 0), use simplified extended GCD:
   
   If 2v₁ ≡ 0 mod p:
     d = gcd(u, 2v₀) — may be non-trivial
   Else:
     Use extended Euclidean to find s₁, s₂, d₁ with d₁ = s₁·u + s₂·(2v)

2. If d = 1 (generic case):
   Compute k = (f - v²) / u   [polynomial, exact division]
   This gives k(x) = x³ + k₂x² + k₁x + k₀  (degree 3 for genus-2)
   
   Compute l = s₂ · k mod u
   l(x) = l₁x + l₀  (reduced mod u, so degree < 2)
   
   u' = (l)² - k  expressed via polynomial arithmetic, then...
   
   Actually, the standard approach is:
   
   u_new = l² + l·(2v)/u - (f-v²)/u²
   ... this gets complex.

3. SIMPLER: Use the "add-and-reduce" formulation:
   
   Composition: U = u²  (degree 4)
                V = v + u·l  where l satisfies 2v + u·l ≡ 0 mod u (from extended GCD)
   
   Reduction step 1 (degree 4 → degree 2):
     u₁' = (f - V²) / U   [exact polynomial division, result is degree ≤ 3... no]
   
   Actually for doubling in genus 2, the composed divisor has degree 4 (= 2g),
   and we need exactly ONE reduction step to get back to degree ≤ 2.
```

### 2.4 Explicit Formulas for Genus-2 Doubling (Lange 2002)

The most efficient explicit formulas for doubling on genus-2 curves with `h(x) = 0` (our case, since `y² = f(x)` has no `h(x)y` term) are from Lange's thesis. For `D = (u, v)` with `deg(u) = 2`:

**Step 1: Compute resultant and inverse**
```
// Extended GCD of u and v' = 2v (derivative of v² w.r.t. affine coords)
// Since h = 0, the tangent calculation uses 2v directly.
//
// inv1 = 1 / (2v) mod u
// This means: find w such that w·(2v) ≡ 1 mod u
//
// Since u = x² + u₁x + u₀ and 2v = 2v₁x + 2v₀:
// Use extended Euclidean on polynomials over F_p

r = resultant(u, 2v)     // r ∈ F_p (scalar, since deg(u)=2, deg(2v)≤1)
If r = 0: degenerate case (handle separately)
inv_r = r⁻¹ mod p
// Cofactor: find w₁x + w₀ such that w·(2v) ≡ r mod u
```

**Step 2: Compute the "slope" polynomial `l`**
```
// l = [(f - v²)/u] · inv(2v) mod u
// This is the tangent line slope in Jacobian coordinates

k = (f - v²) / u         // Exact polynomial division (degree 3 result)
l = k · w mod u           // w is the cofactor from step 1, l has deg < 2
l(x) = l₁x + l₀
```

**Step 3: Compute new u (composition + reduction)**
```
// After composition: degree-4 polynomial, reduce to degree 2
u' = l² + l·h - f  ... no, for h=0:
u' = (l + v)² / u        // ... this isn't right either for genus-2

// CORRECT (Cantor's reduction for genus 2):
// Composed semi-reduced divisor has u_comp = u², v_comp = v + u·l
// Reduce: u_red = (f - v_comp²) / u_comp, then make monic
// Then: v_red = -v_comp mod u_red

u_comp = u·u = u²        // degree 4
v_comp = v + u·l          // degree ≤ 3

// Reduction:
u_red = (f - v_comp²) / u_comp      // exact division, result degree ≤ genus = 2
// Make monic (divide by leading coeff)
v_red = (-v_comp) mod u_red          // polynomial remainder
```

**Step 4: Normalize**
```
Reduce all coefficients mod p
Ensure u_red is monic (leading coeff = 1)
```

### 2.5 Wesolowski Proof Protocol for Jacobian Groups

The Wesolowski VDF protocol adapted for `J(C)`:

**SETUP:**
- Group: `J(C)` with operation denoted additively (repeated doubling = multiplication by 2^T)
- Input: `g ∈ J(C)` (derived from challenge hash)
- VDF computation: `y = [2^T]g` (T sequential doublings of g)

**PROVE(g, y, T):**
```
1. c = HashToPrime(g, y)            // Fiat-Shamir challenge (a prime number)
2. Compute q, r such that 2^T = q·c + r  where 0 ≤ r < c
   (i.e., q = floor(2^T / c), r = 2^T mod c)
3. π = [q]g                          // q doublings of g (accumulated during VDF eval)
4. Return proof = (y, π)
```

**VERIFY(g, y, π, T):**
```
1. c = HashToPrime(g, y)            // Same deterministic challenge
2. r = 2^T mod c                    // Modular exponentiation (fast)
3. Check: [c]π + [r]g == y          // c doublings of π, r doublings of g, add, compare
   Cost: O(log c + log r) doublings = O(log T) total
```

**Why this works:** If `π = [q]g`, then `[c]π + [r]g = [cq]g + [r]g = [cq + r]g = [2^T]g = y`. ✓

**Efficient proof accumulation during VDF evaluation:**

During the T sequential doublings, accumulate π incrementally:
```
g_current = g
pi = identity
for i in 0..T:
    // Check if bit i of q is set (long division style)
    // Maintain: 2^i mod c, and floor(2^i / c) bit by bit
    bit = (2^i / c) bit extraction  
    if bit_i_of_q == 1:
        pi = [2]pi + g_current     // (accumulate)
    else:
        pi = [2]pi
    g_current = [2]g_current       // This is the VDF step anyway
```

Actually, the simpler approach for proof generation:
```
// After computing y = [2^T]g:
c = HashToPrime(g, y)
q = floor(2^T / c)   // Big integer division
pi = [q]g             // q doublings of g (separate pass, or accumulated)
```

The proof generation requires O(T) work (same as evaluation) but can be computed AFTER the evaluation is done. For mining, the miner does:
1. Compute `y = [2^T]g` (the VDF output) — takes ~2 seconds
2. Compute `c = HashToPrime(g, y)` — instant
3. Compute `π = [q]g` where `q = floor(2^T/c)` — takes ~2 seconds (same cost as eval)

Total miner time: ~4 seconds. Verification: O(log T) doublings = milliseconds.

**Optimization:** Accumulate π during evaluation (single pass):
```
c is unknown during eval (depends on y which isn't known yet)
So we must do a second pass. OR use Pietrzak's protocol (halving, no second pass).
For simplicity: two-pass is fine. Miner spends 2× time, server verifies in ms.
```

### 2.6 HashToPrime

```
fn hash_to_prime(g: &JacobianElement, y: &JacobianElement) -> BigUint {
    let mut nonce: u64 = 0;
    loop {
        let mut hasher = Sha3_256::new();
        hasher.update(b"genus2-wesolowski-challenge-v1");
        hasher.update(&g.to_bytes());
        hasher.update(&y.to_bytes());
        hasher.update(&nonce.to_le_bytes());
        let hash = hasher.finalize();
        
        let candidate = BigUint::from_bytes_be(&hash[..16]);  // 128-bit prime
        candidate |= 1;  // Ensure odd
        
        if miller_rabin(&candidate, 40) {  // 40 rounds = negligible error
            return candidate;
        }
        nonce += 1;
    }
}
```

### 2.7 Scalar Multiplication in J(C)

`[n]D` means adding `D` to itself `n` times. For large `n`, use double-and-add:

```
fn scalar_mul(d: &JacobianElement, n: &BigUint, curve: &Genus2CurveParams) -> JacobianElement {
    let mut result = JacobianElement::identity();
    let mut base = d.clone();
    let bits = n.bits();
    
    for i in 0..bits {
        if n.bit(i) {
            result = add_jacobian(&result, &base, curve);
        }
        base = double_jacobian(&base, curve);
    }
    result
}
```

This requires both `double_jacobian()` AND `add_jacobian()` (addition of two different elements). Cantor's full algorithm handles both.

---

## 3. What We Need Implemented

### 3.1 Polynomial Arithmetic over F_p

```rust
/// Polynomial over F_p, represented as Vec<BigInt> of coefficients
/// poly[i] = coefficient of x^i (poly[0] = constant term)
pub struct Poly {
    pub coeffs: Vec<BigInt>,
    pub modulus: BigInt,  // The prime p
}

impl Poly {
    fn degree(&self) -> usize;
    fn is_zero(&self) -> bool;
    fn leading_coeff(&self) -> &BigInt;
    fn make_monic(&mut self);  // Divide all coefficients by leading coeff
    
    fn add(&self, other: &Poly) -> Poly;
    fn sub(&self, other: &Poly) -> Poly;
    fn mul(&self, other: &Poly) -> Poly;
    fn div_rem(&self, other: &Poly) -> (Poly, Poly);  // Quotient, remainder
    fn gcd_extended(&self, other: &Poly) -> (Poly, Poly, Poly);  // (gcd, s, t)
    fn eval(&self, x: &BigInt) -> BigInt;
    fn scale(&self, scalar: &BigInt) -> Poly;  // Multiply all coeffs by scalar
}
```

### 3.2 Cantor's Algorithm (Complete)

```rust
/// Add two divisors D1 + D2 in J(C)
pub fn add_jacobian(
    d1: &JacobianElement,
    d2: &JacobianElement,
    curve: &Genus2CurveParams,
) -> Result<JacobianElement>;

/// Double a divisor: 2D in J(C)
pub fn double_jacobian(
    d: &JacobianElement,
    curve: &Genus2CurveParams,
) -> Result<JacobianElement>;

/// Scalar multiplication: [n]D using double-and-add
pub fn scalar_mul(
    d: &JacobianElement,
    n: &BigUint,
    curve: &Genus2CurveParams,
) -> Result<JacobianElement>;

/// Negate a divisor: -D
pub fn negate_jacobian(
    d: &JacobianElement,
    curve: &Genus2CurveParams,
) -> JacobianElement;
```

### 3.3 Wesolowski Proof for Jacobian

```rust
/// Generate Wesolowski proof for VDF output
/// Prover has already computed y = [2^T]g
pub fn generate_wesolowski_proof(
    g: &JacobianElement,
    y: &JacobianElement,
    iterations: u64,
    curve: &Genus2CurveParams,
) -> Result<WesolowskiProofGenus2>;

/// Verify Wesolowski proof
/// Cost: O(log T) doublings (milliseconds for T = 1,000,000)
pub fn verify_wesolowski_proof(
    g: &JacobianElement,
    y: &JacobianElement,
    proof: &WesolowskiProofGenus2,
    iterations: u64,
    curve: &Genus2CurveParams,
) -> Result<bool>;

pub struct WesolowskiProofGenus2 {
    /// The proof element π = [floor(2^T / c)]g
    pub pi: JacobianElement,
    /// The Fiat-Shamir challenge (prime)
    pub challenge: BigUint,
}
```

### 3.4 Miller-Rabin Primality Test

```rust
/// Deterministic Miller-Rabin for numbers up to 128 bits
/// Uses 40 random witnesses for negligible false-positive probability
pub fn miller_rabin(n: &BigUint, rounds: u32) -> bool;
```

### 3.5 Proper Serialization

```rust
impl JacobianElement {
    /// Serialize to canonical byte representation
    /// Format: [degree: 1 byte][u1: 32 bytes BE][u0: 32 bytes BE][v1: 32 bytes BE][v0: 32 bytes BE]
    /// Total: 129 bytes for degree-2 elements (pq128 security)
    pub fn to_bytes_canonical(&self, field_size: usize) -> Vec<u8>;
    
    /// Deserialize from canonical bytes
    pub fn from_bytes_canonical(bytes: &[u8], curve: &Genus2CurveParams) -> Result<Self>;
    
    /// Validate that this element is on the Jacobian (v² ≡ f mod u)
    pub fn validate(&self, curve: &Genus2CurveParams) -> bool;
}
```

---

## 4. Existing Code Structure (Integration Points)

The implementation will live in `crates/q-vdf/src/genus2_vdf.rs`. Here's the existing struct that must be preserved:

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct JacobianElement {
    pub u1: BigInt,     // u₁ coefficient of u(x) = x² + u₁x + u₀
    pub u0: BigInt,     // u₀ coefficient
    pub v1: BigInt,     // v₁ coefficient of v(x) = v₁x + v₀  
    pub v0: BigInt,     // v₀ coefficient
    pub degree: usize,  // 0 (identity), 1, or 2
}

pub struct Genus2CurveParams {
    pub p: BigUint,     // Prime field modulus
    pub a4: BigInt,     // Coefficients of f(x) = x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀
    pub a3: BigInt,
    pub a2: BigInt,
    pub a1: BigInt,
    pub a0: BigInt,
}
```

The `Genus2VDF` struct's `evaluate()` method calls `double_jacobian()` T times. We need that function to be **correct**.

---

## 5. Test Vectors & Acceptance Criteria

### 5.1 Curve Equation Invariant

After EVERY doubling, the result must satisfy:
```
v(x)² ≡ f(x)  (mod u(x))   over F_p
```

Test: For 10,000 random elements, double each one and verify the curve equation holds.

### 5.2 Group Law Properties

```
1. Identity: [2]O = O  (doubling identity gives identity)
2. Associativity: [2]([2]D) = [4]D = [2]D + [2]D  (doubling twice = adding D+D+D+D)
3. Determinism: double(D) on two different machines → same result
4. Order: for random D, [n]D = O for some n dividing |J(C)|
```

### 5.3 VDF Correctness

```
For T ∈ {10, 100, 1000, 10000}:
  g = hash_to_element(b"test-seed")
  y = [2^T]g  (T sequential doublings)
  
  Verify: y is a valid Jacobian element (curve equation holds)
  Verify: y ≠ g (non-trivial output)
  Verify: same seed → same y (deterministic)
```

### 5.4 Wesolowski Proof Correctness

```
For T = 1000:
  g = hash_to_element(b"proof-test")
  y = [2^T]g
  (pi, c) = generate_proof(g, y, T)
  
  // Positive test
  assert!(verify_proof(g, y, pi, T) == true)
  
  // Negative tests (forgery detection)
  assert!(verify_proof(g, y, identity(), T) == false)       // wrong π
  assert!(verify_proof(g, identity(), pi, T) == false)      // wrong y
  assert!(verify_proof(identity(), y, pi, T) == false)      // wrong g
  assert!(verify_proof(g, y, pi, T+1) == false)             // wrong T
  
  // Different seed → different proof
  g2 = hash_to_element(b"different-seed")
  y2 = [2^T]g2
  assert!(verify_proof(g2, y2, pi, T) == false)             // proof from g doesn't work for g2
```

### 5.5 Performance Targets

On a mid-range CPU (e.g., i7-12700K single thread):
- Single doubling: < 5 μs
- T = 1,000,000 doublings: < 5 seconds
- Proof generation (second pass, T doublings): < 5 seconds  
- Proof verification: < 50 ms (for T = 1,000,000)
- Miller-Rabin (128-bit candidate, 40 rounds): < 1 ms

### 5.6 Security Requirements

1. **No shortcuts:** Computing y without T sequential doublings must be infeasible
2. **Soundness:** A forged proof must be rejected with overwhelming probability
3. **Determinism:** Same inputs → same outputs on all platforms (no floating point)
4. **No panics:** All operations must handle edge cases (zero elements, degenerate divisors)

---

## 6. Reference Implementation Guidance

### 6.1 Known-Good References

- **Cantor's algorithm:** Original paper "Computing in the Jacobian of a Hyperelliptic Curve" (Cantor, 1987)
- **Explicit formulas:** "Formulae for Arithmetic on Genus 2 Hyperelliptic Curves" (Lange, 2005) — Chapter 14 of "Handbook of Elliptic and Hyperelliptic Curve Cryptography"
- **Wesolowski VDF:** "Efficient Verifiable Delay Functions" (Wesolowski, EUROCRYPT 2019) — Section 3
- **Working implementation:** libpari/GP has genus-2 Jacobian arithmetic; SageMath `hyperelliptic_generic.py`

### 6.2 The Key Subtleties

1. **Polynomial division must be exact.** When computing `(f - v²) / u`, the result must have zero remainder. If it doesn't, the input element was invalid.

2. **All arithmetic is mod p.** Every coefficient operation must reduce mod p immediately to prevent BigInt explosion.

3. **Extended GCD on polynomials over F_p** uses the same algorithm as integer GCD but with polynomial division and coefficient inversion.

4. **Modular inverse** of a field element uses Fermat's little theorem: `a⁻¹ = a^(p-2) mod p`.

5. **Making a polynomial monic** means multiplying by the inverse of the leading coefficient.

6. **Degenerate cases in doubling:**
   - If `2v ≡ 0 mod u` (tangent is vertical): result is the identity
   - If `d = gcd(u, 2v) ≠ 1`: need to handle the non-coprime case

7. **The reduction step** for genus 2: after composition you get degree-4 `u_composed`. One reduction step takes it to degree ≤ 2. The reduction is:
   ```
   u_new = (f - v_composed²) / u_composed   (make monic)
   v_new = -v_composed mod u_new
   ```

8. **Addition of two DIFFERENT elements** (needed for verification):
   Uses the same Cantor framework but with `gcd(u₁, u₂)` instead of `gcd(u, 2v)`.

---

## 7. Deliverables

Please provide a single Rust module (`genus2_cantor.rs`) containing:

1. `struct Poly` with full polynomial arithmetic over F_p
2. `fn add_jacobian(d1, d2, curve) -> JacobianElement` — full Cantor addition
3. `fn double_jacobian(d, curve) -> JacobianElement` — optimized doubling
4. `fn negate_jacobian(d, curve) -> JacobianElement`
5. `fn scalar_mul(d, n, curve) -> JacobianElement` — double-and-add
6. `fn validate_element(d, curve) -> bool` — check v² ≡ f mod u
7. `fn generate_wesolowski_proof(g, y, T, curve) -> (JacobianElement, BigUint)` — (π, c)
8. `fn verify_wesolowski_proof(g, y, pi, T, curve) -> bool`
9. `fn hash_to_prime(g_bytes, y_bytes) -> BigUint` — Fiat-Shamir
10. `fn miller_rabin(n, rounds) -> bool`
11. `fn to_bytes_canonical(elem, field_bytes) -> Vec<u8>`
12. `fn from_bytes_canonical(bytes, curve) -> Result<JacobianElement>`
13. Comprehensive test suite covering all acceptance criteria from Section 5

**Constraints:**
- Use only: `num-bigint`, `num-traits`, `num-integer`, `sha3`, `anyhow`
- No `unsafe` code
- No floating point (integer-only arithmetic)
- All operations must be constant-time with respect to element representation (not secret data, but no data-dependent branches that could cause different element sizes)
- Reduce mod p after every multiplication to keep BigInt bounded

---

## 8. Integration Plan (Our Responsibility, Not DeepSeek's)

Once DeepSeek provides the module, we will:
1. Replace the broken `double_jacobian()` in `genus2_vdf.rs` with the correct version
2. Wire `generate_wesolowski_proof()` into the miner's submission pipeline
3. Wire `verify_wesolowski_proof()` into the server's PATH A verification
4. Add proper serialization to the mining protocol
5. Benchmark on real hardware to calibrate T
6. Delta Docker testing (48+ hours)
7. Height-gated mainnet activation

---

## 9. Quick Reference: What's Wrong With Current Code

| Function | File:Line | Bug |
|----------|-----------|-----|
| `double_jacobian()` | `genus2_vdf.rs:334-383` | Not Cantor's algorithm. Computes `u_new = 2·u₁, u₀_new = u₀ + u₁²` — this is nonsense. No polynomial GCD, no composition, no reduction. |
| `double_deg1()` | `genus2_vdf.rs:386-403` | Computes `2·u₀, 2·v₀` — this is scalar multiplication of coefficients, not Jacobian doubling |
| `generate_proof()` | `genus2_vdf.rs:407-437` | Only computes Fiat-Shamir hash, doesn't compute π = [q]g |
| `verify()` | `genus2_vdf.rs:441-476` | Only checks if hash matches, doesn't verify π^c · g^r == y |
| `from_bytes()` | `genus2_vdf.rs:229-244` | Returns zero element regardless of input |
| `from_hash()` | `genus2_vdf.rs:175-211` | Derives random coefficients but doesn't ensure v² ≡ f mod u |

All six must be replaced with correct implementations.
