//! 256-bit finite field arithmetic for Shamir secret sharing
//!
//! Implements modular arithmetic over the secp256k1 prime field.

use num_bigint::BigUint;
use num_traits::{One, Zero};
use num_integer::Integer;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Sub};
use zeroize::Zeroize;

use crate::config::FIELD_PRIME_SECP256K1;
use crate::error::{TemporalError, TemporalResult};

/// A 256-bit field element for Shamir secret sharing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Zeroize)]
pub struct FieldElement256 {
    /// Internal representation as bytes (big-endian)
    bytes: [u8; 32],
}

lazy_static::lazy_static! {
    /// The field prime as BigUint
    static ref PRIME: BigUint = BigUint::from_bytes_be(&FIELD_PRIME_SECP256K1);
}

impl FieldElement256 {
    /// Create a zero element
    pub fn zero() -> Self {
        Self { bytes: [0u8; 32] }
    }

    /// Create the multiplicative identity (one)
    pub fn one() -> Self {
        let mut bytes = [0u8; 32];
        bytes[31] = 1;
        Self { bytes }
    }

    /// Create from a u64 value
    pub fn from_u64(value: u64) -> Self {
        let mut bytes = [0u8; 32];
        bytes[24..32].copy_from_slice(&value.to_be_bytes());
        Self { bytes }
    }

    /// Create from bytes (big-endian), with modular reduction
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let value = BigUint::from_bytes_be(bytes);
        let reduced = value % &*PRIME;
        let reduced_bytes = reduced.to_bytes_be();

        let mut result = [0u8; 32];
        let start = 32 - reduced_bytes.len().min(32);
        result[start..].copy_from_slice(&reduced_bytes[..reduced_bytes.len().min(32)]);

        Self { bytes: result }
    }

    /// Create from a 32-byte array (with reduction)
    pub fn from_bytes_32(bytes: [u8; 32]) -> Self {
        Self::from_bytes(&bytes)
    }

    /// Get the bytes representation (big-endian)
    pub fn to_bytes(&self) -> [u8; 32] {
        self.bytes
    }

    /// Get as BigUint for internal operations
    fn to_biguint(&self) -> BigUint {
        BigUint::from_bytes_be(&self.bytes)
    }

    /// Create from BigUint with reduction
    fn from_biguint(value: BigUint) -> Self {
        let reduced = value % &*PRIME;
        let bytes = reduced.to_bytes_be();

        let mut result = [0u8; 32];
        let len = bytes.len().min(32);
        let start = 32 - len;
        result[start..].copy_from_slice(&bytes[..len]);

        Self { bytes: result }
    }

    /// Check if zero
    pub fn is_zero(&self) -> bool {
        self.bytes.iter().all(|&b| b == 0)
    }

    /// Modular addition
    pub fn add(&self, other: &Self) -> Self {
        let a = self.to_biguint();
        let b = other.to_biguint();
        Self::from_biguint(a + b)
    }

    /// Modular subtraction
    pub fn sub(&self, other: &Self) -> Self {
        let a = self.to_biguint();
        let b = other.to_biguint();
        // Handle underflow by adding prime
        let result = if a >= b {
            a - b
        } else {
            &*PRIME - (b - a)
        };
        Self::from_biguint(result)
    }

    /// Modular multiplication
    pub fn mul(&self, other: &Self) -> Self {
        let a = self.to_biguint();
        let b = other.to_biguint();
        Self::from_biguint(a * b)
    }

    /// Modular exponentiation
    pub fn pow(&self, exp: u64) -> Self {
        let base = self.to_biguint();
        let result = base.modpow(&BigUint::from(exp), &*PRIME);
        Self::from_biguint(result)
    }

    /// Modular multiplicative inverse using extended Euclidean algorithm
    pub fn inverse(&self) -> TemporalResult<Self> {
        if self.is_zero() {
            return Err(TemporalError::DivisionByZero);
        }

        let a = self.to_biguint();
        // Use Fermat's little theorem: a^(-1) = a^(p-2) mod p
        let exp = &*PRIME - BigUint::from(2u32);
        let result = a.modpow(&exp, &*PRIME);
        Ok(Self::from_biguint(result))
    }

    /// Modular division
    pub fn div(&self, other: &Self) -> TemporalResult<Self> {
        let inv = other.inverse()?;
        Ok(self.mul(&inv))
    }

    /// Negate (additive inverse)
    pub fn neg(&self) -> Self {
        if self.is_zero() {
            return self.clone();
        }
        Self::from_biguint(&*PRIME - self.to_biguint())
    }

    /// Generate a random field element
    pub fn random() -> TemporalResult<Self> {
        let mut bytes = [0u8; 32];
        getrandom::getrandom(&mut bytes)
            .map_err(|e| TemporalError::RandomnessFailed(e.to_string()))?;
        Ok(Self::from_bytes(&bytes))
    }
}

impl Add for FieldElement256 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        FieldElement256::add(&self, &other)
    }
}

impl Add for &FieldElement256 {
    type Output = FieldElement256;

    fn add(self, other: Self) -> FieldElement256 {
        FieldElement256::add(self, other)
    }
}

impl Sub for FieldElement256 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        FieldElement256::sub(&self, &other)
    }
}

impl Mul for FieldElement256 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        FieldElement256::mul(&self, &other)
    }
}

impl Mul for &FieldElement256 {
    type Output = FieldElement256;

    fn mul(self, other: Self) -> FieldElement256 {
        FieldElement256::mul(self, other)
    }
}

impl Default for FieldElement256 {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_zero() {
        let zero = FieldElement256::zero();
        assert!(zero.is_zero());
    }

    #[test]
    fn test_field_one() {
        let one = FieldElement256::one();
        assert!(!one.is_zero());
        assert_eq!(one.to_bytes()[31], 1);
    }

    #[test]
    fn test_field_add() {
        let a = FieldElement256::from_u64(100);
        let b = FieldElement256::from_u64(200);
        let c = a.add(&b);
        assert_eq!(c, FieldElement256::from_u64(300));
    }

    #[test]
    fn test_field_mul() {
        let a = FieldElement256::from_u64(7);
        let b = FieldElement256::from_u64(8);
        let c = a.mul(&b);
        assert_eq!(c, FieldElement256::from_u64(56));
    }

    #[test]
    fn test_field_inverse() {
        let a = FieldElement256::from_u64(7);
        let inv = a.inverse().unwrap();
        let product = a.mul(&inv);
        assert_eq!(product, FieldElement256::one());
    }

    #[test]
    fn test_field_div() {
        let a = FieldElement256::from_u64(100);
        let b = FieldElement256::from_u64(5);
        let c = a.div(&b).unwrap();
        assert_eq!(c, FieldElement256::from_u64(20));
    }

    #[test]
    fn test_field_random() {
        let r1 = FieldElement256::random().unwrap();
        let r2 = FieldElement256::random().unwrap();
        // Very unlikely to be equal
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_field_sub() {
        let a = FieldElement256::from_u64(100);
        let b = FieldElement256::from_u64(30);
        let c = a.sub(&b);
        assert_eq!(c, FieldElement256::from_u64(70));
    }

    #[test]
    fn test_field_sub_underflow() {
        let a = FieldElement256::from_u64(10);
        let b = FieldElement256::from_u64(20);
        let c = a.sub(&b);
        // Should wrap around (p - 10)
        let expected = c.add(&FieldElement256::from_u64(20));
        assert_eq!(expected, FieldElement256::from_u64(10));
    }
}
