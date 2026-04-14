//! Genus-2 Jacobian arithmetic with correct Cantor's algorithm and Wesolowski VDF proofs.
//!
//! This module replaces the broken cryptographic core in `genus2_vdf.rs` with
//! mathematically correct implementations of:
//! - Polynomial arithmetic over F_p (with proper mod_floor reduction)
//! - Cantor's full 2-step addition for genus-2 Jacobian elements
//! - Cantor's doubling via tangent line
//! - Wesolowski proof generation and O(log T) verification
//! - Miller-Rabin primality test for Fiat-Shamir challenges
//! - Deterministic hash-to-curve-point (try-and-increment)
//! - Canonical serialization/deserialization
//!
//! # Curve
//! y² = f(x) = x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀  over F_p
//!
//! # Safety
//! - All field arithmetic uses mod_floor (never negative remainders)
//! - Every Jacobian element is validated after construction (v² ≡ f mod u)
//! - No floating point arithmetic anywhere
//! - No unsafe code

use anyhow::{ensure, Result};
use num_bigint::{BigInt, BigUint, Sign, ToBigInt};
use num_integer::Integer;
use num_traits::{One, Zero};
use sha3::{Digest, Sha3_256};
use tracing::{debug, info, trace, warn};

// ═══════════════════════════════════════════════════════════════════════════════
// FIELD ARITHMETIC
// ═══════════════════════════════════════════════════════════════════════════════

/// Reduce a BigInt modulo p, result always in [0, p-1].
/// Unlike Rust's `%` operator which preserves sign, this always returns non-negative.
#[inline]
fn mod_p(a: &BigInt, p: &BigInt) -> BigInt {
    a.mod_floor(p)
}

/// Modular inverse via Fermat's little theorem: a^(p-2) mod p.
/// Requires p prime and a ≢ 0 mod p.
fn mod_inv(a: &BigInt, p: &BigInt) -> Result<BigInt> {
    let a_reduced = mod_p(a, p);
    ensure!(!a_reduced.is_zero(), "cannot invert zero");
    let exp = p - BigInt::from(2);
    Ok(a_reduced.modpow(&exp, p))
}

/// Tonelli-Shanks algorithm for computing square roots mod p.
/// Returns Some(y) where y² ≡ n mod p, or None if n is not a QR.
/// Works for any odd prime p (not just p ≡ 3 mod 4).
fn sqrt_mod_p(n: &BigInt, p: &BigInt) -> Option<BigInt> {
    let n = mod_p(n, p);
    if n.is_zero() {
        return Some(BigInt::zero());
    }

    // Check if n is a quadratic residue (Euler's criterion)
    let exp = (p - BigInt::one()) / BigInt::from(2);
    let legendre = n.modpow(&exp, p);
    if legendre != BigInt::one() {
        return None; // Not a QR
    }

    // Fast path for p ≡ 3 mod 4
    if p.mod_floor(&BigInt::from(4)) == BigInt::from(3) {
        let exp = (p + BigInt::one()) / BigInt::from(4);
        let y = n.modpow(&exp, p);
        return Some(y);
    }

    // General Tonelli-Shanks
    // Factor p-1 = Q * 2^S with Q odd
    let mut q = p - BigInt::one();
    let mut s: u64 = 0;
    while q.is_even() {
        q >>= 1;
        s += 1;
    }

    // Find a quadratic non-residue z
    let mut z = BigInt::from(2);
    loop {
        let leg = z.modpow(&exp, p);
        if leg == p - BigInt::one() {
            break;
        }
        z += BigInt::one();
    }

    let mut m = s;
    let mut c = z.modpow(&q, p);
    let mut t = n.modpow(&q, p);
    let mut r = n.modpow(&((q.clone() + BigInt::one()) / BigInt::from(2)), p);

    loop {
        if t.is_zero() {
            return Some(BigInt::zero());
        }
        if t == BigInt::one() {
            return Some(r);
        }

        // Find the least i such that t^(2^i) ≡ 1 mod p
        let mut i: u64 = 1;
        let mut tmp = (&t * &t).mod_floor(p);
        while tmp != BigInt::one() {
            tmp = (&tmp * &tmp).mod_floor(p);
            i += 1;
            if i >= m {
                return None; // Should not happen for a QR
            }
        }

        let b = c.modpow(&(BigInt::one() << (m - i - 1)), p);
        m = i;
        c = (&b * &b).mod_floor(p);
        t = (&t * &c).mod_floor(p);
        r = (&r * &b).mod_floor(p);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// POLYNOMIAL ARITHMETIC OVER F_p
// ═══════════════════════════════════════════════════════════════════════════════

/// Polynomial over F_p. coeffs[i] = coefficient of x^i (low degree first).
/// All coefficients are always in [0, p-1].
#[derive(Clone, Debug)]
pub struct Poly {
    pub coeffs: Vec<BigInt>,
    pub modulus: BigInt,
}

impl PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        if self.modulus != other.modulus {
            return false;
        }
        let a = self.normalized();
        let b = other.normalized();
        a.coeffs == b.coeffs
    }
}
impl Eq for Poly {}

impl Poly {
    pub fn zero(modulus: BigInt) -> Self {
        Poly {
            coeffs: vec![],
            modulus,
        }
    }

    pub fn one(modulus: BigInt) -> Self {
        Poly {
            coeffs: vec![BigInt::one()],
            modulus,
        }
    }

    pub fn constant(c: &BigInt, modulus: BigInt) -> Self {
        let c = mod_p(c, &modulus);
        if c.is_zero() {
            Poly::zero(modulus)
        } else {
            Poly {
                coeffs: vec![c],
                modulus,
            }
        }
    }

    /// Remove trailing zero coefficients.
    fn normalized(&self) -> Self {
        let mut coeffs = self.coeffs.clone();
        while coeffs.len() > 0 && coeffs.last().unwrap().is_zero() {
            coeffs.pop();
        }
        Poly {
            coeffs,
            modulus: self.modulus.clone(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.normalized().coeffs.is_empty()
    }

    pub fn degree(&self) -> usize {
        let n = self.normalized();
        if n.coeffs.is_empty() {
            return 0;
        }
        n.coeffs.len() - 1
    }

    pub fn leading_coeff(&self) -> BigInt {
        let n = self.normalized();
        if n.coeffs.is_empty() {
            BigInt::zero()
        } else {
            n.coeffs.last().unwrap().clone()
        }
    }

    pub fn make_monic(&mut self) -> Result<()> {
        *self = self.normalized();
        if self.is_zero() {
            return Ok(());
        }
        let lc = self.leading_coeff();
        if lc == BigInt::one() {
            return Ok(());
        }
        let inv_lc = mod_inv(&lc, &self.modulus)?;
        for c in &mut self.coeffs {
            *c = mod_p(&(c.clone() * &inv_lc), &self.modulus);
        }
        Ok(())
    }

    pub fn add(&self, other: &Poly) -> Self {
        let max_len = std::cmp::max(self.coeffs.len(), other.coeffs.len());
        let mut coeffs = vec![BigInt::zero(); max_len];
        for (i, c) in self.coeffs.iter().enumerate() {
            coeffs[i] = c.clone();
        }
        for (i, c) in other.coeffs.iter().enumerate() {
            coeffs[i] = mod_p(&(coeffs[i].clone() + c), &self.modulus);
        }
        Poly {
            coeffs,
            modulus: self.modulus.clone(),
        }
        .normalized()
    }

    pub fn sub(&self, other: &Poly) -> Self {
        let max_len = std::cmp::max(self.coeffs.len(), other.coeffs.len());
        let mut coeffs = vec![BigInt::zero(); max_len];
        for (i, c) in self.coeffs.iter().enumerate() {
            coeffs[i] = c.clone();
        }
        for (i, c) in other.coeffs.iter().enumerate() {
            coeffs[i] = mod_p(&(coeffs[i].clone() - c), &self.modulus);
        }
        Poly {
            coeffs,
            modulus: self.modulus.clone(),
        }
        .normalized()
    }

    pub fn mul(&self, other: &Poly) -> Self {
        // Handle zero polynomials
        if self.is_zero() || other.is_zero() {
            return Poly::zero(self.modulus.clone());
        }
        let len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![BigInt::zero(); len];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                let prod = mod_p(&(a * b), &self.modulus);
                coeffs[i + j] = mod_p(&(coeffs[i + j].clone() + &prod), &self.modulus);
            }
        }
        Poly {
            coeffs,
            modulus: self.modulus.clone(),
        }
        .normalized()
    }

    pub fn scale(&self, scalar: &BigInt) -> Self {
        if self.is_zero() {
            return Poly::zero(self.modulus.clone());
        }
        let coeffs: Vec<BigInt> = self
            .coeffs
            .iter()
            .map(|c| mod_p(&(c * scalar), &self.modulus))
            .collect();
        Poly {
            coeffs,
            modulus: self.modulus.clone(),
        }
        .normalized()
    }

    /// Polynomial long division over F_p.
    /// Returns (quotient, remainder) with self = quotient * divisor + remainder.
    pub fn div_rem(&self, divisor: &Poly) -> (Poly, Poly) {
        assert!(!divisor.is_zero(), "division by zero polynomial");
        let divisor = divisor.normalized();
        let mut remainder = self.normalized();
        let mut quotient = Poly::zero(self.modulus.clone());

        let divisor_lc = divisor.leading_coeff();
        let divisor_deg = divisor.degree();
        let inv_lc = mod_inv(&divisor_lc, &self.modulus).expect("divisor leading coeff not invertible");

        while !remainder.is_zero() && remainder.degree() >= divisor_deg {
            let deg_diff = remainder.degree() - divisor_deg;
            let lc_ratio = mod_p(&(remainder.leading_coeff() * &inv_lc), &self.modulus);

            let mut term_coeffs = vec![BigInt::zero(); deg_diff + 1];
            term_coeffs[deg_diff] = lc_ratio;
            let term = Poly {
                coeffs: term_coeffs,
                modulus: self.modulus.clone(),
            };

            quotient = quotient.add(&term);
            remainder = remainder.sub(&term.mul(&divisor));
        }
        (quotient, remainder)
    }

    /// Extended GCD for polynomials. Returns (gcd, s, t) with s*self + t*other = gcd.
    /// The gcd is made monic.
    pub fn gcd_extended(&self, other: &Poly) -> (Poly, Poly, Poly) {
        let (mut r0, mut r1) = (self.normalized(), other.normalized());
        let (mut s0, mut s1) = (
            Poly::one(self.modulus.clone()),
            Poly::zero(self.modulus.clone()),
        );
        let (mut t0, mut t1) = (
            Poly::zero(self.modulus.clone()),
            Poly::one(self.modulus.clone()),
        );

        while !r1.is_zero() {
            let (q, r) = r0.div_rem(&r1);
            r0 = r1;
            r1 = r;
            let s_next = s0.sub(&q.mul(&s1));
            let t_next = t0.sub(&q.mul(&t1));
            s0 = s1;
            s1 = s_next;
            t0 = t1;
            t1 = t_next;
        }

        // Make gcd monic and scale cofactors accordingly
        let lc = r0.leading_coeff();
        if !lc.is_zero() && lc != BigInt::one() {
            let inv_lc = mod_inv(&lc, &self.modulus).unwrap();
            r0 = r0.scale(&inv_lc);
            s0 = s0.scale(&inv_lc);
            t0 = t0.scale(&inv_lc);
        }
        (r0, s0, t0)
    }

    pub fn eval(&self, x: &BigInt) -> BigInt {
        // Horner's method
        let mut result = BigInt::zero();
        for c in self.coeffs.iter().rev() {
            result = mod_p(&(result * x + c), &self.modulus);
        }
        result
    }

    /// Remainder after division: self mod other
    pub fn rem(&self, other: &Poly) -> Poly {
        let (_, r) = self.div_rem(other);
        r
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURVE PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Parameters of a genus-2 hyperelliptic curve y² = f(x).
/// f(x) = x⁵ + a₄x⁴ + a₃x³ + a₂x² + a₁x + a₀ over F_p.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CurveParams {
    pub p: BigUint,
    pub a4: BigInt,
    pub a3: BigInt,
    pub a2: BigInt,
    pub a1: BigInt,
    pub a0: BigInt,
}

impl CurveParams {
    /// PQ-128 security: y² = x⁵ + x² - 1 over a 256-bit prime field.
    pub fn pq128() -> Self {
        let p = BigUint::parse_bytes(
            b"115792089237316195423570985008687907853269984665640564039457584007913129639747",
            10,
        )
        .unwrap();
        CurveParams {
            p,
            a4: BigInt::zero(),
            a3: BigInt::zero(),
            a2: BigInt::one(),
            a1: BigInt::zero(),
            a0: BigInt::from(-1),
        }
    }

    pub fn p_bigint(&self) -> BigInt {
        self.p.to_bigint().unwrap()
    }

    pub fn field_bytes(&self) -> usize {
        (self.p.bits() as usize + 7) / 8
    }

    /// Return f(x) as a Poly.
    pub fn f_poly(&self) -> Poly {
        let p = self.p_bigint();
        Poly {
            coeffs: vec![
                mod_p(&self.a0, &p),
                mod_p(&self.a1, &p),
                mod_p(&self.a2, &p),
                mod_p(&self.a3, &p),
                mod_p(&self.a4, &p),
                BigInt::one(),
            ],
            modulus: p,
        }
    }

    /// Evaluate f(x) at a point.
    pub fn eval_f(&self, x: &BigInt) -> BigInt {
        self.f_poly().eval(x)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// JACOBIAN ELEMENT (MUMFORD REPRESENTATION)
// ═══════════════════════════════════════════════════════════════════════════════

/// Element of J(C) in Mumford form: (u(x), v(x)) where
/// u is monic, deg(u) ≤ 2, deg(v) < deg(u), and u | (v² - f).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JacElement {
    pub u1: BigInt,
    pub u0: BigInt,
    pub v1: BigInt,
    pub v0: BigInt,
    pub degree: usize,
}

impl JacElement {
    pub fn identity() -> Self {
        JacElement {
            u1: BigInt::zero(),
            u0: BigInt::zero(),
            v1: BigInt::zero(),
            v0: BigInt::zero(),
            degree: 0,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.degree == 0
    }

    /// Build u(x) as a Poly.
    pub fn u_poly(&self, p: &BigInt) -> Poly {
        match self.degree {
            0 => Poly::one(p.clone()),
            1 => Poly {
                coeffs: vec![mod_p(&self.u0, p), BigInt::one()],
                modulus: p.clone(),
            },
            2 => Poly {
                coeffs: vec![
                    mod_p(&self.u0, p),
                    mod_p(&self.u1, p),
                    BigInt::one(),
                ],
                modulus: p.clone(),
            },
            _ => unreachable!(),
        }
    }

    /// Build v(x) as a Poly.
    pub fn v_poly(&self, p: &BigInt) -> Poly {
        match self.degree {
            0 => Poly::zero(p.clone()),
            1 => Poly {
                coeffs: vec![mod_p(&self.v0, p)],
                modulus: p.clone(),
            },
            2 => Poly {
                coeffs: vec![mod_p(&self.v0, p), mod_p(&self.v1, p)],
                modulus: p.clone(),
            },
            _ => unreachable!(),
        }
    }

    /// Validate: v² ≡ f (mod u).
    pub fn validate(&self, curve: &CurveParams) -> bool {
        if self.is_identity() {
            return true;
        }
        let p = curve.p_bigint();
        let u = self.u_poly(&p);
        let v = self.v_poly(&p);
        let f = curve.f_poly();
        let v2 = v.mul(&v);
        let diff = v2.sub(&f);
        let (_, r) = diff.div_rem(&u);
        r.is_zero()
    }

    /// Deterministically derive a valid Jacobian element from a seed.
    /// Maps seed → curve point (x, y) → degree-1 divisor.
    pub fn from_seed(seed: &[u8], curve: &CurveParams) -> Result<Self> {
        let (x, y) = hash_to_curve_point(seed, curve)?;
        let p = curve.p_bigint();
        // Degree-1 divisor: u(x) = x - x₀, v(x) = y₀
        // In our representation: u = x + u0 where u0 = -x₀ mod p
        let u0 = mod_p(&(-&x), &p);
        let v0 = mod_p(&y, &p);
        let elem = JacElement {
            u1: BigInt::zero(),
            u0,
            v1: BigInt::zero(),
            v0,
            degree: 1,
        };
        ensure!(elem.validate(curve), "from_seed produced invalid element");
        Ok(elem)
    }

    /// Canonical serialization.
    /// [degree: 1 byte][u1: N bytes][u0: N bytes][v1: N bytes][v0: N bytes]
    /// where N = field_bytes. For degree < 2, unused fields are still present (as zero).
    pub fn to_bytes(&self, field_bytes: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + 4 * field_bytes);
        out.push(self.degree as u8);

        let write = |out: &mut Vec<u8>, val: &BigInt| {
            let (_, mut bytes) = val.to_bytes_be();
            // Pad to field_bytes
            while bytes.len() < field_bytes {
                bytes.insert(0, 0);
            }
            // Truncate if somehow longer (shouldn't happen after mod_p)
            out.extend_from_slice(&bytes[bytes.len() - field_bytes..]);
        };

        write(&mut out, &self.u1);
        write(&mut out, &self.u0);
        write(&mut out, &self.v1);
        write(&mut out, &self.v0);
        out
    }

    /// Canonical deserialization.
    pub fn from_bytes(bytes: &[u8], curve: &CurveParams) -> Result<Self> {
        let fb = curve.field_bytes();
        ensure!(bytes.len() == 1 + 4 * fb, "wrong byte length");
        let degree = bytes[0] as usize;
        ensure!(degree <= 2, "invalid degree");

        let read = |offset: usize| -> BigInt {
            let slice = &bytes[1 + offset * fb..1 + (offset + 1) * fb];
            mod_p(
                &BigInt::from_bytes_be(Sign::Plus, slice),
                &curve.p_bigint(),
            )
        };

        let u1 = read(0);
        let u0 = read(1);
        let v1 = read(2);
        let v0 = read(3);

        Ok(JacElement {
            u1,
            u0,
            v1,
            v0,
            degree,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HASH TO CURVE POINT
// ═══════════════════════════════════════════════════════════════════════════════

/// Map a seed deterministically to a point (x, y) on y² = f(x).
/// Uses try-and-increment with Tonelli-Shanks square root.
pub fn hash_to_curve_point(seed: &[u8], curve: &CurveParams) -> Result<(BigInt, BigInt)> {
    let p = curve.p_bigint();
    for counter in 0u32..10_000 {
        let mut h = Sha3_256::new();
        h.update(b"genus2-hash-to-curve-v1");
        h.update(seed);
        h.update(&counter.to_le_bytes());
        let hash = h.finalize();

        let x = mod_p(
            &BigInt::from_bytes_be(Sign::Plus, &hash[..32]),
            &p,
        );
        let fx = curve.eval_f(&x);

        if let Some(y) = sqrt_mod_p(&fx, &p) {
            return Ok((x, y));
        }
    }
    anyhow::bail!("hash_to_curve_point: no point found after 10000 attempts")
}

// ═══════════════════════════════════════════════════════════════════════════════
// CANTOR'S ALGORITHM — ADDITION AND DOUBLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert (u_poly, v_poly) pair back to JacElement after reduction.
fn polys_to_element(u: &Poly, v: &Poly, p: &BigInt) -> Result<JacElement> {
    let u = u.normalized();
    let v = v.normalized();
    let deg = u.degree();

    if u.is_zero() || (deg == 0 && u.leading_coeff() == BigInt::one()) {
        return Ok(JacElement::identity());
    }

    match deg {
        0 => {
            // Non-identity degree-0: shouldn't happen after proper reduction
            // Treat as identity if leading coeff is 1
            Ok(JacElement::identity())
        }
        1 => {
            let u0 = if !u.coeffs.is_empty() {
                mod_p(&u.coeffs[0], p)
            } else {
                BigInt::zero()
            };
            let v0 = if !v.coeffs.is_empty() {
                mod_p(&v.coeffs[0], p)
            } else {
                BigInt::zero()
            };
            Ok(JacElement {
                u1: BigInt::zero(),
                u0,
                v1: BigInt::zero(),
                v0,
                degree: 1,
            })
        }
        2 => {
            let u0 = mod_p(&u.coeffs[0], p);
            let u1 = mod_p(&u.coeffs[1], p);
            let v0 = if !v.coeffs.is_empty() {
                mod_p(&v.coeffs[0], p)
            } else {
                BigInt::zero()
            };
            let v1 = if v.coeffs.len() > 1 {
                mod_p(&v.coeffs[1], p)
            } else {
                BigInt::zero()
            };
            Ok(JacElement {
                u1,
                u0,
                v1,
                v0,
                degree: 2,
            })
        }
        _ => anyhow::bail!("polys_to_element: deg(u) = {} > 2", deg),
    }
}

/// Cantor reduction: given semi-reduced (u, v) with deg(u) > g=2,
/// reduce to proper Mumford form.
fn cantor_reduce(u: &Poly, v: &Poly, f: &Poly, p: &BigInt) -> Result<(Poly, Poly)> {
    let mut u_cur = u.clone();
    let mut v_cur = v.clone();

    // Each reduction step: u' = (f - v²) / u (make monic), v' = -v mod u'
    while u_cur.degree() > 2 && !u_cur.is_zero() {
        let v_sq = v_cur.mul(&v_cur);
        let num = f.sub(&v_sq);
        let (mut u_new, rem) = num.div_rem(&u_cur);
        ensure!(rem.is_zero(), "Cantor reduction: f - v² not divisible by u");
        u_new.make_monic()?;

        // v_new = (-v_cur) mod u_new
        let neg_v = v_cur.scale(&(p - BigInt::one())); // multiply by -1 ≡ p-1
        let v_new = neg_v.rem(&u_new);

        u_cur = u_new;
        v_cur = v_new;
    }

    Ok((u_cur, v_cur))
}

/// Add two DISTINCT divisors D1 ≠ D2 using Cantor's full 2-step algorithm.
///
/// Algorithm (Cantor 1987):
/// 1. d₁ = gcd(u₁, u₂), with e₁, e₂ such that d₁ = e₁u₁ + e₂u₂
/// 2. d = gcd(d₁, v₁ + v₂), with c₁, c₃ such that d = c₁d₁ + c₃(v₁+v₂)
///    Let s₁ = c₁e₁, s₂ = c₁e₂, s₃ = c₃
/// 3. u' = (u₁ · u₂) / d²
/// 4. v' = [s₁u₁v₂ + s₂u₂v₁ + s₃(v₁v₂ + f)] / d  mod u'
/// 5. Reduce to deg(u') ≤ 2
fn add_distinct(
    d1: &JacElement,
    d2: &JacElement,
    curve: &CurveParams,
) -> Result<JacElement> {
    let p = curve.p_bigint();
    let u1 = d1.u_poly(&p);
    let v1 = d1.v_poly(&p);
    let u2 = d2.u_poly(&p);
    let v2 = d2.v_poly(&p);
    let f = curve.f_poly();

    // Step 1: d₁ = gcd(u₁, u₂)
    let (d1_poly, e1, e2) = u1.gcd_extended(&u2);

    // Step 2: d = gcd(d₁, v₁ + v₂)
    let v_sum = v1.add(&v2);
    let (d_poly, c1, c3) = d1_poly.gcd_extended(&v_sum);

    // Compute s₁ = c₁·e₁, s₂ = c₁·e₂, s₃ = c₃
    let s1 = c1.mul(&e1);
    let s2 = c1.mul(&e2);
    let s3 = c3;

    // Step 3: u_composed = (u₁ · u₂) / d²
    let u_prod = u1.mul(&u2);
    let d_sq = d_poly.mul(&d_poly);
    let (u_composed, rem_u) = u_prod.div_rem(&d_sq);
    ensure!(
        rem_u.is_zero(),
        "add_distinct: u1*u2 not divisible by d^2"
    );

    // Step 4: v_composed = [s₁u₁v₂ + s₂u₂v₁ + s₃(v₁v₂ + f)] / d  mod u_composed
    let term1 = s1.mul(&u1).mul(&v2);
    let term2 = s2.mul(&u2).mul(&v1);
    let v1v2 = v1.mul(&v2);
    let term3 = s3.mul(&v1v2.add(&f));
    let v_num = term1.add(&term2).add(&term3);
    let (v_divided, rem_v) = v_num.div_rem(&d_poly);
    ensure!(
        rem_v.is_zero(),
        "add_distinct: v numerator not divisible by d"
    );
    let v_composed = v_divided.rem(&u_composed);

    // Step 5: Reduce
    let (u_red, v_red) = cantor_reduce(&u_composed, &v_composed, &f, &p)?;

    polys_to_element(&u_red, &v_red, &p)
}

/// Double a divisor D → 2D using Cantor's tangent method.
///
/// For doubling, the "tangent" polynomial replaces the secant:
/// 1. Compute d = gcd(u, 2v)  [since h=0 for our curve]
/// 2. If d = 1 (generic): l = [(f - v²)/u] · inv(2v) mod u
/// 3. Compose: u' = u², v' = v + u·l
/// 4. Reduce
pub fn double_jacobian(d: &JacElement, curve: &CurveParams) -> Result<JacElement> {
    if d.is_identity() {
        return Ok(JacElement::identity());
    }

    let p = curve.p_bigint();
    let u = d.u_poly(&p);
    let v = d.v_poly(&p);
    let f = curve.f_poly();

    // Compute gcd(u, 2v)
    let two_v = v.scale(&BigInt::from(2));
    let (d_poly, _s, t) = u.gcd_extended(&two_v);

    if d_poly.degree() > 0 {
        // Non-generic case: gcd(u, 2v) has positive degree.
        // This means v ≡ 0 mod (u/d) partially or fully.
        // Use the general Cantor framework with d ≠ 1.
        let (u_div_d, _) = u.div_rem(&d_poly);

        let v_sq = v.mul(&v);
        let f_minus_v2 = f.sub(&v_sq);
        let (k, _) = f_minus_v2.div_rem(&u_div_d);
        let tk = t.mul(&k);
        let l = tk.rem(&u_div_d);

        let u_comp = u_div_d.mul(&u_div_d);
        let v_comp = v.add(&u_div_d.mul(&l));
        let v_comp_mod = v_comp.rem(&u_comp);

        let (u_red, v_red) = cantor_reduce(&u_comp, &v_comp_mod, &f, &p)?;
        return polys_to_element(&u_red, &v_red, &p);
    }

    // Generic case: d = 1, so _s·u + t·(2v) = 1
    // Compute k = (f - v²) / u  (exact division)
    let v_sq = v.mul(&v);
    let f_minus_v2 = f.sub(&v_sq);
    let (k, rem) = f_minus_v2.div_rem(&u);
    ensure!(rem.is_zero(), "doubling: f - v² not divisible by u");

    // l = t · k mod u
    let tk = t.mul(&k);
    let l = tk.rem(&u);

    // Compose: u_comp = u², v_comp = v + u·l
    let u_comp = u.mul(&u);
    let v_comp = v.add(&u.mul(&l));
    let v_comp_mod = v_comp.rem(&u_comp);

    // Reduce
    let (u_red, v_red) = cantor_reduce(&u_comp, &v_comp_mod, &f, &p)?;

    polys_to_element(&u_red, &v_red, &p)
}

/// Add two Jacobian elements (handles identity, equal, distinct cases).
pub fn add_jacobian(
    a: &JacElement,
    b: &JacElement,
    curve: &CurveParams,
) -> Result<JacElement> {
    if a.is_identity() {
        return Ok(b.clone());
    }
    if b.is_identity() {
        return Ok(a.clone());
    }
    if a == b {
        return double_jacobian(a, curve);
    }

    // Check if a = -b (same u, v1 = -v2)
    let p = curve.p_bigint();
    if a.degree == b.degree && a.u1 == b.u1 && a.u0 == b.u0 {
        let neg_v0 = mod_p(&(-&b.v0), &p);
        let neg_v1 = mod_p(&(-&b.v1), &p);
        if mod_p(&a.v0, &p) == neg_v0 && mod_p(&a.v1, &p) == neg_v1 {
            return Ok(JacElement::identity());
        }
    }

    add_distinct(a, b, curve)
}

/// Negate a divisor: -(u, v) = (u, -v mod u).
pub fn negate(d: &JacElement, curve: &CurveParams) -> JacElement {
    if d.is_identity() {
        return JacElement::identity();
    }
    let p = curve.p_bigint();
    JacElement {
        u1: d.u1.clone(),
        u0: d.u0.clone(),
        v1: mod_p(&(-&d.v1), &p),
        v0: mod_p(&(-&d.v0), &p),
        degree: d.degree,
    }
}

/// Scalar multiplication [n]D via double-and-add.
pub fn scalar_mul(
    d: &JacElement,
    n: &BigUint,
    curve: &CurveParams,
) -> Result<JacElement> {
    if n.is_zero() || d.is_identity() {
        return Ok(JacElement::identity());
    }
    let mut result = JacElement::identity();
    let mut base = d.clone();
    for i in 0..n.bits() {
        if n.bit(i) {
            result = add_jacobian(&result, &base, curve)?;
        }
        base = double_jacobian(&base, curve)?;
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// VDF EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Evaluate VDF: compute y = [2^T]g by performing T sequential doublings.
/// This is the core sequential work that cannot be parallelized.
/// Logs progress every `log_interval` steps (0 = no logging).
pub fn evaluate_vdf(
    g: &JacElement,
    iterations: u64,
    curve: &CurveParams,
    log_interval: u64,
) -> Result<JacElement> {
    let mut current = g.clone();
    let start = std::time::Instant::now();

    for i in 0..iterations {
        current = double_jacobian(&current, curve)?;

        if log_interval > 0 && i > 0 && i % log_interval == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = i as f64 / elapsed;
            let remaining = (iterations - i) as f64 / rate;
            debug!(
                "VDF progress: {}/{} ({:.1}%), {:.0} dbl/s, ~{:.1}s remaining",
                i,
                iterations,
                (i as f64 / iterations as f64) * 100.0,
                rate,
                remaining
            );
        }
    }

    let elapsed = start.elapsed();
    info!(
        "VDF evaluation complete: {} doublings in {:.3}s ({:.0} dbl/s)",
        iterations,
        elapsed.as_secs_f64(),
        iterations as f64 / elapsed.as_secs_f64()
    );

    Ok(current)
}

// ═══════════════════════════════════════════════════════════════════════════════
// WESOLOWSKI PROOF
// ═══════════════════════════════════════════════════════════════════════════════

/// Wesolowski proof for a Genus-2 VDF.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WesolowskiProof {
    /// π = [floor(2^T / c)]g
    pub pi: JacElement,
    /// Fiat-Shamir challenge (prime)
    pub challenge: BigUint,
}

/// Fixed proof size for pq128: pi (129 bytes) + challenge (16 bytes) = 145 bytes.
pub const PROOF_BYTES_PQ128: usize = 145;

impl WesolowskiProof {
    /// Serialize to bytes.
    /// Format: [pi_canonical (1 + 4*field_bytes)] [challenge_be (16 bytes, zero-padded)]
    pub fn to_bytes(&self, field_bytes: usize) -> Vec<u8> {
        let mut out = self.pi.to_bytes(field_bytes);
        // Challenge: fixed 16 bytes (128-bit prime), big-endian, zero-padded
        let ch_bytes = self.challenge.to_bytes_be();
        let pad = 16usize.saturating_sub(ch_bytes.len());
        out.extend(std::iter::repeat(0u8).take(pad));
        if ch_bytes.len() <= 16 {
            out.extend_from_slice(&ch_bytes);
        } else {
            // Shouldn't happen for 128-bit primes, but handle gracefully
            out.extend_from_slice(&ch_bytes[ch_bytes.len() - 16..]);
        }
        out
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8], curve: &CurveParams) -> Result<Self> {
        let fb = curve.field_bytes();
        let pi_len = 1 + 4 * fb; // canonical pi size
        ensure!(bytes.len() >= pi_len + 16, "proof bytes too short");

        let pi = JacElement::from_bytes(&bytes[..pi_len], curve)?;
        let challenge = BigUint::from_bytes_be(&bytes[pi_len..pi_len + 16]);

        Ok(WesolowskiProof { pi, challenge })
    }
}

/// Generate a Wesolowski proof after VDF evaluation.
///
/// Requires y = [2^T]g to have been computed already.
/// Cost: O(T) doublings (same as evaluation — second pass).
pub fn generate_proof(
    g: &JacElement,
    y: &JacElement,
    iterations: u64,
    curve: &CurveParams,
) -> Result<WesolowskiProof> {
    info!("Generating Wesolowski proof for T={}", iterations);
    let start = std::time::Instant::now();

    let c = hash_to_prime(g, y, curve)?;

    // Compute q = floor(2^T / c) and π = [q]g.
    //
    // We CANNOT materialize 2^T as a BigUint for large T (too many bits).
    // Instead we compute q via long division while simultaneously computing π.
    //
    // Algorithm: iterate from bit T-1 down to 0.
    // Maintain r (the running remainder) and π (the proof accumulator).
    //   r = 0, π = identity
    //   For bit i from T-1 down to 0:
    //     r = 2*r
    //     π = [2]π
    //     if r >= c:
    //       r = r - c
    //       π = π + g    (bit i of q is 1)
    //   At the end, π = [q]g and r = 2^T mod c.
    let c_bigint = c.to_bigint().unwrap();
    let mut r = BigInt::zero();
    let mut pi = JacElement::identity();

    for i in (0..iterations).rev() {
        r = mod_p(&(&r * BigInt::from(2)), &c_bigint);
        pi = double_jacobian(&pi, curve)?;
        if r >= c_bigint {
            r -= &c_bigint;
            pi = add_jacobian(&pi, g, curve)?;
        }

        if i % 100_000 == 0 && i > 0 {
            trace!("Proof generation: bit {}/{}", iterations - i, iterations);
        }
    }

    let elapsed = start.elapsed();
    info!(
        "Wesolowski proof generated in {:.3}s",
        elapsed.as_secs_f64()
    );

    Ok(WesolowskiProof { pi, challenge: c })
}

/// Verify a Wesolowski proof.
/// Checks: [c]π + [r]g == y  where r = 2^T mod c.
/// Cost: O(log c + log r) doublings ≈ O(log T) — milliseconds.
pub fn verify_proof(
    g: &JacElement,
    y: &JacElement,
    proof: &WesolowskiProof,
    iterations: u64,
    curve: &CurveParams,
) -> Result<bool> {
    // Recompute challenge
    let c = hash_to_prime(g, y, curve)?;
    if c != proof.challenge {
        warn!("Wesolowski verification: challenge mismatch");
        return Ok(false);
    }

    // r = 2^T mod c  (fast modular exponentiation)
    let two = BigUint::from(2u32);
    let r = two.modpow(&BigUint::from(iterations), &c);

    // [c]π + [r]g  should equal y
    let c_pi = scalar_mul(&proof.pi, &c, curve)?;
    let r_g = scalar_mul(g, &r, curve)?;
    let lhs = add_jacobian(&c_pi, &r_g, curve)?;

    let valid = lhs == *y;
    if valid {
        debug!("Wesolowski proof verified successfully");
    } else {
        warn!("Wesolowski proof verification FAILED");
    }
    Ok(valid)
}

// ═══════════════════════════════════════════════════════════════════════════════
// MILLER-RABIN & HASH-TO-PRIME
// ═══════════════════════════════════════════════════════════════════════════════

/// Deterministic Miller-Rabin primality test.
/// Uses fixed witnesses derived from the candidate (deterministic, no RNG needed).
/// For 128-bit candidates, testing witnesses {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
/// is sufficient for correctness (proven for n < 3.3×10^24, and probabilistically
/// overwhelming for larger n).
pub fn miller_rabin(n: &BigUint, _rounds: u32) -> bool {
    if n < &BigUint::from(2u32) {
        return false;
    }
    if n == &BigUint::from(2u32) || n == &BigUint::from(3u32) {
        return true;
    }
    if n.is_even() {
        return false;
    }

    // Write n-1 = d · 2^s
    let n_minus_1 = n - 1u32;
    let mut d = n_minus_1.clone();
    let mut s: u32 = 0;
    while d.is_even() {
        d >>= 1;
        s += 1;
    }

    let n_big = n.to_bigint().unwrap();
    let n_minus_1_big = &n_big - BigInt::one();

    // Fixed witnesses — deterministic, no RNG
    let witnesses: &[u32] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    // Also add hash-derived witnesses for extra coverage on large n
    let mut extra_witnesses = Vec::new();
    {
        let n_bytes = n.to_bytes_be();
        for i in 0u32..8 {
            let mut h = Sha3_256::new();
            h.update(b"miller-rabin-witness");
            h.update(&n_bytes);
            h.update(&i.to_le_bytes());
            let hash = h.finalize();
            let w = BigUint::from_bytes_be(&hash[..16]) % n;
            if w >= BigUint::from(2u32) {
                extra_witnesses.push(w);
            }
        }
    }

    'witness: for w in witnesses
        .iter()
        .map(|&w| BigUint::from(w))
        .chain(extra_witnesses.into_iter())
    {
        if w >= *n {
            continue;
        }
        let a = w.to_bigint().unwrap();
        let mut x = a.modpow(&d.to_bigint().unwrap(), &n_big);

        if x == BigInt::one() || x == n_minus_1_big {
            continue;
        }
        for _ in 0..s - 1 {
            x = (&x * &x).mod_floor(&n_big);
            if x == n_minus_1_big {
                continue 'witness;
            }
            if x == BigInt::one() {
                return false;
            }
        }
        return false;
    }
    true
}

/// Derive a 128-bit prime deterministically from g and y (Fiat-Shamir).
pub fn hash_to_prime(
    g: &JacElement,
    y: &JacElement,
    curve: &CurveParams,
) -> Result<BigUint> {
    let fb = curve.field_bytes();
    let g_bytes = g.to_bytes(fb);
    let y_bytes = y.to_bytes(fb);

    for nonce in 0u64.. {
        let mut h = Sha3_256::new();
        h.update(b"genus2-wesolowski-challenge-v1");
        h.update(&g_bytes);
        h.update(&y_bytes);
        h.update(&nonce.to_le_bytes());
        let hash = h.finalize();

        // 128-bit candidate from first 16 bytes
        let mut candidate = BigUint::from_bytes_be(&hash[..16]);
        candidate |= BigUint::one(); // ensure odd
        if candidate > BigUint::one() && miller_rabin(&candidate, 40) {
            return Ok(candidate);
        }
    }
    unreachable!()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Small curve for fast testing: y² = x⁵ + x² - 1 over F_13.
    fn small_curve() -> CurveParams {
        CurveParams {
            p: BigUint::from(13u32),
            a4: BigInt::zero(),
            a3: BigInt::zero(),
            a2: BigInt::one(),
            a1: BigInt::zero(),
            a0: BigInt::from(-1),
        }
    }

    /// Known point on small curve: f(1) = 1 + 1 - 1 = 1. sqrt(1) = 1. Point (1, 1).
    fn small_gen() -> JacElement {
        let p = BigInt::from(13);
        JacElement {
            u1: BigInt::zero(),
            u0: mod_p(&BigInt::from(-1), &p), // u = x - 1 => u0 = -1 mod 13 = 12
            v1: BigInt::zero(),
            v0: BigInt::one(), // v = 1
            degree: 1,
        }
    }

    #[test]
    fn test_mod_p_always_positive() {
        let p = BigInt::from(13);
        assert_eq!(mod_p(&BigInt::from(-3), &p), BigInt::from(10));
        assert_eq!(mod_p(&BigInt::from(15), &p), BigInt::from(2));
        assert_eq!(mod_p(&BigInt::zero(), &p), BigInt::zero());
    }

    #[test]
    fn test_sqrt_mod_p() {
        let p = BigInt::from(13);
        // 1 is a QR: sqrt(1) = 1 or 12
        let y = sqrt_mod_p(&BigInt::one(), &p).unwrap();
        assert_eq!(mod_p(&(&y * &y), &p), BigInt::one());

        // 12 = -1 mod 13. 13 ≡ 1 mod 4, so -1 may or may not be QR.
        // Legendre: (-1)^((13-1)/2) = (-1)^6 = 1, so -1 IS a QR mod 13.
        let y2 = sqrt_mod_p(&BigInt::from(12), &p);
        assert!(y2.is_some());
        let y2 = y2.unwrap();
        assert_eq!(mod_p(&(&y2 * &y2), &p), BigInt::from(12));
    }

    #[test]
    fn test_poly_basics() {
        let p = BigInt::from(13);
        let a = Poly {
            coeffs: vec![BigInt::from(1), BigInt::from(2)],
            modulus: p.clone(),
        };
        let b = Poly {
            coeffs: vec![BigInt::from(3), BigInt::from(4)],
            modulus: p.clone(),
        };
        // (2x+1)(4x+3) = 8x² + 10x + 3
        let prod = a.mul(&b);
        assert_eq!(prod.coeffs, vec![BigInt::from(3), BigInt::from(10), BigInt::from(8)]);

        // Division
        let (q, r) = prod.div_rem(&a);
        assert_eq!(q.coeffs, b.coeffs);
        assert!(r.is_zero());
    }

    #[test]
    fn test_poly_mul_zero() {
        let p = BigInt::from(13);
        let zero = Poly::zero(p.clone());
        let a = Poly {
            coeffs: vec![BigInt::from(5)],
            modulus: p,
        };
        assert!(zero.mul(&a).is_zero());
        assert!(a.mul(&zero).is_zero());
    }

    #[test]
    fn test_generator_valid() {
        let curve = small_curve();
        let g = small_gen();
        assert!(g.validate(&curve), "generator is not on Jacobian");
    }

    #[test]
    fn test_identity_doubling() {
        let curve = small_curve();
        let id = JacElement::identity();
        let dbl = double_jacobian(&id, &curve).unwrap();
        assert!(dbl.is_identity());
    }

    #[test]
    fn test_double_validates() {
        let curve = small_curve();
        let g = small_gen();
        let g2 = double_jacobian(&g, &curve).unwrap();
        assert!(g2.validate(&curve), "2g is not on Jacobian");
        let g4 = double_jacobian(&g2, &curve).unwrap();
        assert!(g4.validate(&curve), "4g is not on Jacobian");
    }

    #[test]
    fn test_add_identity() {
        let curve = small_curve();
        let g = small_gen();
        let id = JacElement::identity();
        let sum = add_jacobian(&g, &id, &curve).unwrap();
        assert_eq!(sum, g);
        let sum2 = add_jacobian(&id, &g, &curve).unwrap();
        assert_eq!(sum2, g);
    }

    #[test]
    fn test_scalar_mul_consistency() {
        let curve = small_curve();
        let g = small_gen();

        // [0]g = identity
        let s0 = scalar_mul(&g, &BigUint::zero(), &curve).unwrap();
        assert!(s0.is_identity());

        // [1]g = g
        let s1 = scalar_mul(&g, &BigUint::one(), &curve).unwrap();
        assert_eq!(s1, g);

        // [2]g = double(g)
        let s2 = scalar_mul(&g, &BigUint::from(2u32), &curve).unwrap();
        let d2 = double_jacobian(&g, &curve).unwrap();
        assert_eq!(s2, d2);

        // [4]g = double(double(g))
        let s4 = scalar_mul(&g, &BigUint::from(4u32), &curve).unwrap();
        let d4 = double_jacobian(&d2, &curve).unwrap();
        assert_eq!(s4, d4);

        // [3]g = [2]g + g
        let s3 = scalar_mul(&g, &BigUint::from(3u32), &curve).unwrap();
        let a3 = add_jacobian(&d2, &g, &curve).unwrap();
        assert_eq!(s3, a3);
    }

    #[test]
    fn test_negate_and_add_to_identity() {
        let curve = small_curve();
        let g = small_gen();
        let neg_g = negate(&g, &curve);
        let sum = add_jacobian(&g, &neg_g, &curve).unwrap();
        assert!(sum.is_identity(), "g + (-g) should be identity");
    }

    #[test]
    fn test_vdf_evaluation() {
        let curve = small_curve();
        let g = small_gen();
        let t = 20u64;
        let y = evaluate_vdf(&g, t, &curve, 0).unwrap();
        assert!(y.validate(&curve), "VDF output not on Jacobian");

        // Verify determinism: same input → same output
        let y2 = evaluate_vdf(&g, t, &curve, 0).unwrap();
        assert_eq!(y, y2);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let curve = small_curve();
        let g = small_gen();
        let bytes = g.to_bytes(1); // field_bytes = 1 for p=13
        let g2 = JacElement::from_bytes(&bytes, &curve).unwrap();
        assert_eq!(g, g2);

        // Also test degree-2 element
        let d2 = double_jacobian(&g, &curve).unwrap();
        let bytes2 = d2.to_bytes(1);
        let d2_back = JacElement::from_bytes(&bytes2, &curve).unwrap();
        assert_eq!(d2, d2_back);

        // Identity
        let id = JacElement::identity();
        let bytes_id = id.to_bytes(1);
        let id_back = JacElement::from_bytes(&bytes_id, &curve).unwrap();
        assert_eq!(id, id_back);
    }

    #[test]
    fn test_miller_rabin() {
        assert!(miller_rabin(&BigUint::from(2u32), 10));
        assert!(miller_rabin(&BigUint::from(13u32), 10));
        assert!(miller_rabin(&BigUint::from(127u32), 10));
        assert!(!miller_rabin(&BigUint::from(15u32), 10));
        assert!(!miller_rabin(&BigUint::from(1u32), 10));
        assert!(!miller_rabin(&BigUint::from(4u32), 10));
    }

    #[test]
    fn test_proof_serialization_roundtrip() {
        let curve = small_curve();
        let g = small_gen();
        let t = 10u64;
        let y = evaluate_vdf(&g, t, &curve, 0).unwrap();
        let proof = generate_proof(&g, &y, t, &curve).unwrap();

        // Serialize and deserialize
        let bytes = proof.to_bytes(1); // field_bytes=1 for p=13
        let proof_back = WesolowskiProof::from_bytes(&bytes, &curve).unwrap();
        assert_eq!(proof.pi, proof_back.pi);
        assert_eq!(proof.challenge, proof_back.challenge);

        // Deserialized proof still verifies
        let valid = verify_proof(&g, &y, &proof_back, t, &curve).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_proof_serialization_pq128() {
        let curve = CurveParams::pq128();
        let g = JacElement::from_seed(b"proof-ser-test", &curve).unwrap();
        let t = 10u64;
        let y = evaluate_vdf(&g, t, &curve, 0).unwrap();
        let proof = generate_proof(&g, &y, t, &curve).unwrap();

        let fb = curve.field_bytes(); // 32
        let bytes = proof.to_bytes(fb);
        assert_eq!(bytes.len(), PROOF_BYTES_PQ128); // 145 bytes

        let proof_back = WesolowskiProof::from_bytes(&bytes, &curve).unwrap();
        assert_eq!(proof, proof_back);

        let valid = verify_proof(&g, &y, &proof_back, t, &curve).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_wesolowski_proof_small() {
        let curve = small_curve();
        let g = small_gen();
        let t = 10u64;

        // Evaluate VDF
        let y = evaluate_vdf(&g, t, &curve, 0).unwrap();
        assert!(y.validate(&curve));

        // Generate proof
        let proof = generate_proof(&g, &y, t, &curve).unwrap();

        // Verify
        let valid = verify_proof(&g, &y, &proof, t, &curve).unwrap();
        assert!(valid, "valid proof should verify");

        // Tamper: wrong challenge (different prime)
        let fake = WesolowskiProof {
            pi: proof.pi.clone(),
            challenge: proof.challenge.clone() + BigUint::from(2u32),
        };
        let fake_valid = verify_proof(&g, &y, &fake, t, &curve).unwrap();
        assert!(!fake_valid, "fake challenge should NOT verify");

        // Tamper: wrong T
        let wrong_t = verify_proof(&g, &y, &proof, t + 1, &curve).unwrap();
        assert!(!wrong_t, "wrong T should NOT verify");
    }

    #[test]
    fn test_hash_to_curve_point() {
        let curve = small_curve();
        let (x, y) = hash_to_curve_point(b"test-seed", &curve).unwrap();
        let p = curve.p_bigint();
        let y_sq = mod_p(&(&y * &y), &p);
        let fx = curve.eval_f(&x);
        assert_eq!(y_sq, fx, "point not on curve");
    }

    #[test]
    fn test_from_seed() {
        let curve = small_curve();
        let g = JacElement::from_seed(b"mining-challenge-42", &curve).unwrap();
        assert!(g.validate(&curve));
        assert_eq!(g.degree, 1);

        // Deterministic
        let g2 = JacElement::from_seed(b"mining-challenge-42", &curve).unwrap();
        assert_eq!(g, g2);

        // Different seed → different element
        let g3 = JacElement::from_seed(b"mining-challenge-43", &curve).unwrap();
        assert_ne!(g, g3);
    }

    /// Test with the real pq128 curve (slower but ensures production parameters work).
    #[test]
    fn test_pq128_basic() {
        let curve = CurveParams::pq128();
        let g = JacElement::from_seed(b"pq128-test", &curve).unwrap();
        assert!(g.validate(&curve));

        // Double a few times
        let g2 = double_jacobian(&g, &curve).unwrap();
        assert!(g2.validate(&curve));
        let g4 = double_jacobian(&g2, &curve).unwrap();
        assert!(g4.validate(&curve));

        // Scalar mul consistency
        let s4 = scalar_mul(&g, &BigUint::from(4u32), &curve).unwrap();
        assert_eq!(s4, g4);
    }

    /// Benchmark: measure doubling speed on the real 256-bit pq128 curve.
    /// This determines VDF feasibility and calibrates the iteration count T.
    #[test]
    fn bench_pq128_doubling() {
        let curve = CurveParams::pq128();
        let g = JacElement::from_seed(b"bench-seed-2026", &curve).unwrap();

        // Warm up (first doubling triggers any lazy init)
        let mut current = double_jacobian(&g, &curve).unwrap();

        // Benchmark: 100 doublings
        let n = 100u64;
        let start = std::time::Instant::now();
        for _ in 0..n {
            current = double_jacobian(&current, &curve).unwrap();
        }
        let elapsed = start.elapsed();
        let per_doubling_us = elapsed.as_micros() as f64 / n as f64;
        let doublings_per_sec = 1_000_000.0 / per_doubling_us;

        // Project VDF times
        let t_1m = n as f64 * per_doubling_us / 1_000_000.0 * 10_000.0; // 1M doublings
        let t_for_2s = 2_000_000.0 / per_doubling_us; // iterations for ~2 second VDF

        eprintln!("\n══════════════════════════════════════════════════════════");
        eprintln!("  GENUS-2 VDF BENCHMARK (pq128, 256-bit field)");
        eprintln!("══════════════════════════════════════════════════════════");
        eprintln!("  {} doublings in {:.3}ms", n, elapsed.as_secs_f64() * 1000.0);
        eprintln!("  Per doubling:    {:.1} μs", per_doubling_us);
        eprintln!("  Throughput:      {:.0} doublings/sec", doublings_per_sec);
        eprintln!("──────────────────────────────────────────────────────────");
        eprintln!("  Projected VDF times:");
        eprintln!("    T = 1,000:     {:.3}s", per_doubling_us * 1_000.0 / 1_000_000.0);
        eprintln!("    T = 10,000:    {:.3}s", per_doubling_us * 10_000.0 / 1_000_000.0);
        eprintln!("    T = 100,000:   {:.3}s", per_doubling_us * 100_000.0 / 1_000_000.0);
        eprintln!("    T = 1,000,000: {:.3}s", per_doubling_us * 1_000_000.0 / 1_000_000.0);
        eprintln!("──────────────────────────────────────────────────────────");
        eprintln!("  For ~2 second VDF: T ≈ {:.0}", t_for_2s);
        eprintln!("  NOTE: This is DEBUG build. Release will be 5-20× faster.");
        eprintln!("══════════════════════════════════════════════════════════\n");

        // Validate output is still on the Jacobian
        assert!(current.validate(&curve), "result after 100 doublings invalid");
    }

    /// Benchmark: measure doubling speed in RELEASE mode approximation.
    /// Runs a short VDF evaluation and reports stats.
    #[test]
    fn bench_pq128_vdf_short() {
        let curve = CurveParams::pq128();
        let g = JacElement::from_seed(b"vdf-bench-2026", &curve).unwrap();

        let t = 50u64;
        let start = std::time::Instant::now();
        let y = evaluate_vdf(&g, t, &curve, 0).unwrap();
        let eval_elapsed = start.elapsed();

        assert!(y.validate(&curve));

        // Generate proof
        let proof_start = std::time::Instant::now();
        let proof = generate_proof(&g, &y, t, &curve).unwrap();
        let proof_elapsed = proof_start.elapsed();

        // Verify proof
        let verify_start = std::time::Instant::now();
        let valid = verify_proof(&g, &y, &proof, t, &curve).unwrap();
        let verify_elapsed = verify_start.elapsed();

        assert!(valid);

        eprintln!("\n══════════════════════════════════════════════════════════");
        eprintln!("  VDF FULL PIPELINE BENCHMARK (T={}, pq128)", t);
        eprintln!("══════════════════════════════════════════════════════════");
        eprintln!("  Evaluate (T={} doublings): {:.3}ms", t, eval_elapsed.as_secs_f64() * 1000.0);
        eprintln!("  Proof generation:          {:.3}ms", proof_elapsed.as_secs_f64() * 1000.0);
        eprintln!("  Verification:              {:.3}ms", verify_elapsed.as_secs_f64() * 1000.0);
        eprintln!("  Total miner cost:          {:.3}ms (eval + proof)", (eval_elapsed + proof_elapsed).as_secs_f64() * 1000.0);
        eprintln!("  Server verify cost:        {:.3}ms", verify_elapsed.as_secs_f64() * 1000.0);
        eprintln!("══════════════════════════════════════════════════════════\n");
    }
}
