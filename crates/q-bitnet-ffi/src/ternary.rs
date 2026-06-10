//! Ternary Tensor Operations for BitNet
//!
//! BitNet uses ternary weights {-1, 0, +1} which can be packed as 2 bits per weight.
//! This module provides efficient ternary tensor operations for:
//!
//! - **Packing**: 4 weights per byte (16x compression vs FP16)
//! - **Matmul**: Lookup table operations (no floating point!)
//! - **All-Reduce**: Bitwise operations for distributed sum
//!
//! ## Encoding
//!
//! Each weight is encoded as 2 bits:
//! - `00` = -1
//! - `01` =  0
//! - `10` = +1
//! - `11` = reserved (saturated sum result)
//!
//! This allows 4 weights per byte, giving 16x compression over FP16.

use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};
use tracing::debug;

/// Ternary value: -1, 0, or +1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TernaryValue {
    NegOne = 0b00,  // -1
    Zero = 0b01,    //  0
    PosOne = 0b10,  // +1
    Saturated = 0b11, // Used for sums that exceed [-1, +1]
}

impl TernaryValue {
    /// Create from i8 value
    #[inline]
    pub fn from_i8(val: i8) -> Self {
        match val {
            -1 => TernaryValue::NegOne,
            0 => TernaryValue::Zero,
            1 => TernaryValue::PosOne,
            _ => TernaryValue::Saturated,
        }
    }

    /// Convert to i8
    #[inline]
    pub fn to_i8(self) -> i8 {
        match self {
            TernaryValue::NegOne => -1,
            TernaryValue::Zero => 0,
            TernaryValue::PosOne => 1,
            TernaryValue::Saturated => 1, // Saturate to +1
        }
    }

    /// Convert to f32 for compatibility with existing code
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.to_i8() as f32
    }

    /// Get 2-bit encoding
    #[inline]
    pub fn to_bits(self) -> u8 {
        self as u8
    }

    /// Create from 2-bit encoding
    #[inline]
    pub fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => TernaryValue::NegOne,
            0b01 => TernaryValue::Zero,
            0b10 => TernaryValue::PosOne,
            _ => TernaryValue::Saturated,
        }
    }
}

impl Add for TernaryValue {
    type Output = i8;

    #[inline]
    fn add(self, other: Self) -> i8 {
        self.to_i8().saturating_add(other.to_i8())
    }
}

impl Mul for TernaryValue {
    type Output = TernaryValue;

    /// Ternary multiply: no floating point needed!
    /// -1 × -1 = +1, -1 × 0 = 0, -1 × +1 = -1, etc.
    #[inline]
    fn mul(self, other: Self) -> TernaryValue {
        match (self, other) {
            (TernaryValue::Zero, _) | (_, TernaryValue::Zero) => TernaryValue::Zero,
            (TernaryValue::NegOne, TernaryValue::NegOne) => TernaryValue::PosOne,
            (TernaryValue::PosOne, TernaryValue::PosOne) => TernaryValue::PosOne,
            (TernaryValue::NegOne, TernaryValue::PosOne) => TernaryValue::NegOne,
            (TernaryValue::PosOne, TernaryValue::NegOne) => TernaryValue::NegOne,
            _ => TernaryValue::Saturated,
        }
    }
}

/// Packed ternary values: 4 weights per byte
///
/// Layout within a byte (MSB to LSB):
/// ```text
/// [w3][w2][w1][w0]
///  ^    ^    ^   ^
///  |    |    |   +-- bits 0-1: weight 0
///  |    |    +------ bits 2-3: weight 1
///  |    +----------- bits 4-5: weight 2
///  +---------------- bits 6-7: weight 3
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedTernary {
    /// Packed bytes (4 ternary values per byte)
    pub data: Vec<u8>,
    /// Number of actual ternary values (may not be multiple of 4)
    pub len: usize,
}

impl PackedTernary {
    /// Create empty packed ternary
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }

    /// Create with capacity for `n` ternary values
    pub fn with_capacity(n: usize) -> Self {
        let bytes_needed = (n + 3) / 4; // Ceiling division
        Self {
            data: Vec::with_capacity(bytes_needed),
            len: 0,
        }
    }

    /// Pack a slice of ternary values
    pub fn from_values(values: &[TernaryValue]) -> Self {
        let bytes_needed = (values.len() + 3) / 4;
        let mut data = vec![0u8; bytes_needed];

        for (i, &val) in values.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= val.to_bits() << bit_offset;
        }

        Self {
            data,
            len: values.len(),
        }
    }

    /// Pack from i8 slice (quantize to ternary)
    pub fn from_i8_slice(values: &[i8]) -> Self {
        let ternary: Vec<TernaryValue> = values
            .iter()
            .map(|&v| TernaryValue::from_i8(v))
            .collect();
        Self::from_values(&ternary)
    }

    /// Pack from f32 slice using absmean quantization (BitNet style)
    ///
    /// Quantization formula:
    /// w_ternary = sign(w) × round(|w| / mean(|w|))
    /// Then clamp to {-1, 0, +1}
    pub fn from_f32_slice(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self::new();
        }

        // Calculate absmean
        let absmean: f32 = values.iter().map(|x| x.abs()).sum::<f32>() / values.len() as f32;
        let scale = if absmean > 1e-8 { 1.0 / absmean } else { 1.0 };

        // Quantize to ternary
        let ternary: Vec<TernaryValue> = values
            .iter()
            .map(|&w| {
                let scaled = w * scale;
                let quantized = scaled.round() as i8;
                TernaryValue::from_i8(quantized.clamp(-1, 1))
            })
            .collect();

        Self::from_values(&ternary)
    }

    /// Get a single ternary value
    #[inline]
    pub fn get(&self, idx: usize) -> Option<TernaryValue> {
        if idx >= self.len {
            return None;
        }
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        let bits = (self.data[byte_idx] >> bit_offset) & 0b11;
        Some(TernaryValue::from_bits(bits))
    }

    /// Set a single ternary value
    #[inline]
    pub fn set(&mut self, idx: usize, val: TernaryValue) {
        if idx >= self.len {
            return;
        }
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        let mask = !(0b11 << bit_offset);
        self.data[byte_idx] = (self.data[byte_idx] & mask) | (val.to_bits() << bit_offset);
    }

    /// Unpack to ternary values
    pub fn unpack(&self) -> Vec<TernaryValue> {
        (0..self.len)
            .map(|i| self.get(i).unwrap())
            .collect()
    }

    /// Unpack to i8 values
    pub fn unpack_i8(&self) -> Vec<i8> {
        self.unpack().into_iter().map(|v| v.to_i8()).collect()
    }

    /// Unpack to f32 values
    pub fn unpack_f32(&self) -> Vec<f32> {
        self.unpack().into_iter().map(|v| v.to_f32()).collect()
    }

    /// Get compression ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        if self.len == 0 {
            return 1.0;
        }
        let f32_bytes = self.len * 4;
        let packed_bytes = self.data.len();
        f32_bytes as f32 / packed_bytes as f32
    }

    /// Number of ternary values
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw bytes for network transfer (16x smaller than f32!)
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create from raw bytes
    pub fn from_bytes(data: Vec<u8>, len: usize) -> Self {
        Self { data, len }
    }
}

impl Default for PackedTernary {
    fn default() -> Self {
        Self::new()
    }
}

/// Ternary tensor with shape information
///
/// This is the main type for BitNet weight storage and computation.
/// Supports efficient:
/// - Storage: 16x compression vs FP16
/// - All-reduce: Bitwise operations
/// - Matmul: Lookup table based (no float multiply)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryTensor {
    /// Packed ternary data
    pub packed: PackedTernary,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Scale factor for dequantization (optional, for hybrid precision)
    pub scale: Option<f32>,
}

impl TernaryTensor {
    /// Create a new ternary tensor from f32 values
    pub fn from_f32(values: &[f32], shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        assert_eq!(values.len(), total_elements, "Shape mismatch");

        Self {
            packed: PackedTernary::from_f32_slice(values),
            shape,
            scale: None,
        }
    }

    /// Create from i8 values (already quantized)
    pub fn from_i8(values: &[i8], shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        assert_eq!(values.len(), total_elements, "Shape mismatch");

        Self {
            packed: PackedTernary::from_i8_slice(values),
            shape,
            scale: None,
        }
    }

    /// Create from packed bytes (for network transfer)
    pub fn from_bytes(data: Vec<u8>, len: usize, shape: Vec<usize>) -> Self {
        Self {
            packed: PackedTernary::from_bytes(data, len),
            shape,
            scale: None,
        }
    }

    /// Get raw bytes for all-reduce (16x smaller transfer!)
    pub fn to_bytes(&self) -> Vec<u8> {
        self.packed.data.clone()
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.packed.len()
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Reshape (no data copy)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(new_numel, self.numel(), "Shape mismatch in reshape");

        Self {
            packed: self.packed.clone(),
            shape: new_shape,
            scale: self.scale,
        }
    }

    /// Unpack to f32 tensor
    pub fn to_f32(&self) -> Vec<f32> {
        let values = self.packed.unpack_f32();
        if let Some(scale) = self.scale {
            values.into_iter().map(|v| v * scale).collect()
        } else {
            values
        }
    }

    /// Compression ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        self.packed.compression_ratio()
    }

    /// Slice along first dimension for tensor parallelism
    pub fn slice_dim0(&self, start: usize, end: usize) -> Self {
        if self.shape.is_empty() {
            return self.clone();
        }

        let dim0 = self.shape[0];
        assert!(start <= end && end <= dim0, "Invalid slice bounds");

        // Calculate elements per slice
        let elements_per_row: usize = self.shape[1..].iter().product();
        let start_idx = start * elements_per_row;
        let end_idx = end * elements_per_row;

        // Extract the slice
        let values: Vec<TernaryValue> = (start_idx..end_idx)
            .map(|i| self.packed.get(i).unwrap())
            .collect();

        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;

        Self {
            packed: PackedTernary::from_values(&values),
            shape: new_shape,
            scale: self.scale,
        }
    }

    /// Ternary dot product using lookup table
    ///
    /// This is the key operation for BitNet inference!
    /// No floating point multiply needed.
    ///
    /// Returns i32 accumulator (can exceed [-1, +1] range)
    pub fn dot_i32(&self, other: &TernaryTensor) -> i32 {
        assert_eq!(self.numel(), other.numel(), "Size mismatch for dot product");

        // Use lookup table for each pair
        // -1 × -1 = +1, -1 × 0 = 0, -1 × +1 = -1
        //  0 × -1 =  0,  0 × 0 = 0,  0 × +1 =  0
        // +1 × -1 = -1, +1 × 0 = 0, +1 × +1 = +1

        // LUT: indexed by (a << 2 | b) where a, b are 2-bit ternary encodings
        const LUT: [i8; 16] = [
            // a=00 (-1): b=00,01,10,11
            1, 0, -1, 0,
            // a=01 (0):  b=00,01,10,11
            0, 0, 0, 0,
            // a=10 (+1): b=00,01,10,11
            -1, 0, 1, 0,
            // a=11 (sat): b=00,01,10,11
            0, 0, 0, 0,
        ];

        let mut sum: i32 = 0;

        // Process 4 weights at a time (1 byte each)
        let full_bytes = self.numel() / 4;
        for i in 0..full_bytes {
            let a = self.packed.data[i];
            let b = other.packed.data[i];

            // Unpack and accumulate using LUT
            for j in 0..4 {
                let shift = j * 2;
                let a_bits = (a >> shift) & 0b11;
                let b_bits = (b >> shift) & 0b11;
                let lut_idx = ((a_bits as usize) << 2) | (b_bits as usize);
                sum += LUT[lut_idx] as i32;
            }
        }

        // Handle remaining elements
        for i in (full_bytes * 4)..self.numel() {
            let a = self.packed.get(i).unwrap();
            let b = other.packed.get(i).unwrap();
            sum += (a * b).to_i8() as i32;
        }

        sum
    }

    /// Element-wise sum of ternary tensors
    ///
    /// For all-reduce: sum of ternary values from multiple nodes.
    /// Result is i8 (can exceed ternary range).
    pub fn sum_tensors(tensors: &[&TernaryTensor]) -> Vec<i8> {
        if tensors.is_empty() {
            return vec![];
        }

        let len = tensors[0].numel();
        let mut result = vec![0i8; len];

        for tensor in tensors {
            assert_eq!(tensor.numel(), len, "Tensor size mismatch");
            for (i, val) in result.iter_mut().enumerate() {
                *val = val.saturating_add(tensor.packed.get(i).unwrap().to_i8());
            }
        }

        result
    }
}

/// SIMD-optimized ternary operations
#[cfg(target_arch = "x86_64")]
pub mod simd {
    use super::*;

    /// SIMD dot product for ternary tensors
    /// Uses AVX2 for 32 weights at a time
    #[cfg(target_feature = "avx2")]
    pub fn dot_simd(a: &TernaryTensor, b: &TernaryTensor) -> i32 {
        // TODO: Implement AVX2 version using vpshufb for LUT
        // For now, fall back to scalar
        a.dot_i32(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_value_encoding() {
        assert_eq!(TernaryValue::NegOne.to_bits(), 0b00);
        assert_eq!(TernaryValue::Zero.to_bits(), 0b01);
        assert_eq!(TernaryValue::PosOne.to_bits(), 0b10);

        assert_eq!(TernaryValue::from_bits(0b00), TernaryValue::NegOne);
        assert_eq!(TernaryValue::from_bits(0b01), TernaryValue::Zero);
        assert_eq!(TernaryValue::from_bits(0b10), TernaryValue::PosOne);
    }

    #[test]
    fn test_ternary_multiply() {
        use TernaryValue::*;

        assert_eq!(NegOne * NegOne, PosOne);
        assert_eq!(NegOne * Zero, Zero);
        assert_eq!(NegOne * PosOne, NegOne);
        assert_eq!(Zero * Zero, Zero);
        assert_eq!(PosOne * PosOne, PosOne);
    }

    #[test]
    fn test_packed_ternary() {
        let values = vec![
            TernaryValue::NegOne,
            TernaryValue::Zero,
            TernaryValue::PosOne,
            TernaryValue::NegOne,
            TernaryValue::PosOne,
        ];

        let packed = PackedTernary::from_values(&values);

        assert_eq!(packed.len(), 5);
        assert_eq!(packed.data.len(), 2); // 5 values needs 2 bytes

        // Verify unpacking
        let unpacked = packed.unpack();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_compression_ratio() {
        let pattern = [1.0f32, -1.0, 0.0, 1.0];
        let values: Vec<f32> = pattern.iter().cycle().take(400).cloned().collect();
        let tensor = TernaryTensor::from_f32(&values, vec![400]);

        // f32: 400 × 4 = 1600 bytes
        // packed: 400 / 4 = 100 bytes
        // ratio: 16x
        assert!(tensor.compression_ratio() >= 15.0);
    }

    #[test]
    fn test_dot_product() {
        let a = TernaryTensor::from_i8(&[1, -1, 0, 1], vec![4]);
        let b = TernaryTensor::from_i8(&[1, 1, 1, -1], vec![4]);

        // 1×1 + (-1)×1 + 0×1 + 1×(-1) = 1 - 1 + 0 - 1 = -1
        assert_eq!(a.dot_i32(&b), -1);
    }

    #[test]
    fn test_quantization() {
        let values = vec![0.9, -0.8, 0.1, 0.95, -0.92];
        let tensor = TernaryTensor::from_f32(&values, vec![5]);

        // Should quantize to approximately [1, -1, 0, 1, -1]
        let unpacked = tensor.packed.unpack_i8();
        assert_eq!(unpacked[0], 1);
        assert_eq!(unpacked[1], -1);
        assert_eq!(unpacked[3], 1);
        assert_eq!(unpacked[4], -1);
    }
}
