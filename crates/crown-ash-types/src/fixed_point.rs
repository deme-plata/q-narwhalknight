//! Fixed-point arithmetic for deterministic on-chain simulation.
//! All values use i64 with a scale of 1000 (3 decimal places).
//! NO FLOATING POINT — ensures identical results on all nodes.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, Neg};

/// Scale factor: 1000 = 3 decimal places of precision.
pub const SCALE: i64 = 1000;

/// Fixed-point number: i64 with implicit /1000 divisor.
/// `FixedPoint(1500)` represents 1.500.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FixedPoint(pub i64);

impl FixedPoint {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(SCALE);
    pub const MAX: Self = Self(i64::MAX);
    pub const MIN: Self = Self(i64::MIN);

    /// Create from integer (e.g. `from_int(5)` → 5.000).
    pub const fn from_int(n: i64) -> Self {
        Self(n * SCALE)
    }

    /// Create from raw scaled value.
    pub const fn from_raw(raw: i64) -> Self {
        Self(raw)
    }

    /// Create from integer numerator and denominator: `from_fraction(3, 4)` → 0.750.
    pub const fn from_fraction(num: i64, den: i64) -> Self {
        Self(num * SCALE / den)
    }

    /// Raw scaled value.
    pub const fn raw(self) -> i64 {
        self.0
    }

    /// Integer part (truncated toward zero).
    pub const fn integer(self) -> i64 {
        self.0 / SCALE
    }

    /// Fractional part (always in 0..999 for positive, -999..0 for negative).
    pub const fn frac(self) -> i64 {
        self.0 % SCALE
    }

    /// Multiply two fixed-point values: `(a * b) / SCALE`.
    pub const fn mul_fp(self, rhs: Self) -> Self {
        Self((self.0 as i128 * rhs.0 as i128 / SCALE as i128) as i64)
    }

    /// Divide two fixed-point values: `(a * SCALE) / b`.
    ///
    /// # Panics
    /// Panics if `rhs` is zero. Prefer [`checked_div_fp`] in consensus-critical code.
    pub const fn div_fp(self, rhs: Self) -> Self {
        // Safety: caller guarantees rhs != 0. Use checked_div_fp for fallible division.
        Self((self.0 as i128 * SCALE as i128 / rhs.0 as i128) as i64)
    }

    /// Checked division: returns `None` if divisor is zero.
    /// **Use this in all consensus-critical code paths.**
    pub const fn checked_div_fp(self, rhs: Self) -> Option<Self> {
        if rhs.0 == 0 {
            None
        } else {
            Some(Self((self.0 as i128 * SCALE as i128 / rhs.0 as i128) as i64))
        }
    }

    /// Saturating division: returns `ZERO` if divisor is zero.
    /// Use when a zero-default is the correct domain behavior (e.g. averages over empty sets).
    pub const fn saturating_div_fp(self, rhs: Self) -> Self {
        if rhs.0 == 0 {
            Self::ZERO
        } else {
            Self((self.0 as i128 * SCALE as i128 / rhs.0 as i128) as i64)
        }
    }

    /// Clamp between lo and hi (inclusive).
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        if self.0 < lo.0 { lo } else if self.0 > hi.0 { hi } else { self }
    }

    /// Saturating add.
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    /// Saturating sub.
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Linear interpolation: self + (other - self) * t, where t is 0..1000 (0.000 to 1.000).
    pub fn lerp(self, other: Self, t: Self) -> Self {
        self + (other - self).mul_fp(t)
    }
}

impl Add for FixedPoint {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}

impl AddAssign for FixedPoint {
    fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; }
}

impl Sub for FixedPoint {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}

impl SubAssign for FixedPoint {
    fn sub_assign(&mut self, rhs: Self) { self.0 -= rhs.0; }
}

impl Mul<i64> for FixedPoint {
    type Output = Self;
    fn mul(self, rhs: i64) -> Self { Self(self.0 * rhs) }
}

impl Neg for FixedPoint {
    type Output = Self;
    fn neg(self) -> Self { Self(-self.0) }
}

impl Default for FixedPoint {
    fn default() -> Self { Self::ZERO }
}

impl fmt::Display for FixedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.0 < 0 { "-" } else { "" };
        let abs = self.0.unsigned_abs() as i64;
        write!(f, "{}{}.{:03}", sign, abs / SCALE, abs % SCALE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_arithmetic() {
        let a = FixedPoint::from_int(3);
        let b = FixedPoint::from_int(2);
        assert_eq!((a + b).integer(), 5);
        assert_eq!((a - b).integer(), 1);
        assert_eq!(a.mul_fp(b).integer(), 6);
        assert_eq!(a.div_fp(b).raw(), 1500); // 1.500
    }

    #[test]
    fn fraction() {
        let f = FixedPoint::from_fraction(3, 4);
        assert_eq!(f.raw(), 750);
        assert_eq!(f.integer(), 0);
        assert_eq!(f.frac(), 750);
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", FixedPoint::from_int(42)), "42.000");
        assert_eq!(format!("{}", FixedPoint::from_raw(1234)), "1.234");
        assert_eq!(format!("{}", FixedPoint::from_raw(-500)), "-0.500");
    }

    #[test]
    fn checked_div_zero_returns_none() {
        let a = FixedPoint::from_int(100);
        assert_eq!(a.checked_div_fp(FixedPoint::ZERO), None);
    }

    #[test]
    fn checked_div_nonzero_returns_some() {
        let a = FixedPoint::from_int(10);
        let b = FixedPoint::from_int(3);
        let result = a.checked_div_fp(b).unwrap();
        assert_eq!(result.raw(), 3333); // 3.333
    }

    #[test]
    fn saturating_div_zero_returns_zero() {
        let a = FixedPoint::from_int(100);
        assert_eq!(a.saturating_div_fp(FixedPoint::ZERO), FixedPoint::ZERO);
    }

    #[test]
    fn deterministic_mul() {
        // Verify identical results regardless of platform
        let a = FixedPoint::from_raw(123_456);
        let b = FixedPoint::from_raw(789_012);
        assert_eq!(a.mul_fp(b).raw(), 97_408_265);
    }
}
