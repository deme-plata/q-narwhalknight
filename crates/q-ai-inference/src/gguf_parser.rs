//! GGUF File Format Parser
//!
//! This module implements parsing for the GGUF (GPT-Generated Unified Format) file format.
//! GGUF is a binary format used to store large language models with quantized weights.
//!
//! Format Specification:
//! - Magic bytes: "GGUF" (0x47475546)
//! - Version: u32 (currently 3)
//! - Tensor count: u64
//! - Metadata KV count: u64
//! - Metadata KV pairs: (key: string, value: typed value)
//! - Tensor info: (name, n_dimensions, dimensions[], type, offset)
//! - Alignment padding
//! - Tensor data
//!
//! Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use tracing::{debug, info, warn};

/// GGUF file magic bytes: "GGUF"
const GGUF_MAGIC: u32 = 0x46554747;

/// Current GGUF version
const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data (32 bytes)
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum GGUFQuantizationType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
}

impl GGUFQuantizationType {
    /// Get the quantization type from a u32 value
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1_M),
            _ => Err(anyhow!("Unknown quantization type: {}", value)),
        }
    }

    /// Get the block size in bytes for this quantization type
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,      // 16 nibbles (4-bit) + 2 bytes (fp16 delta + min)
            Self::Q4_1 => 20,      // 16 nibbles + 2 bytes delta + 2 bytes min
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,      // 32 bytes + 2 bytes (fp16 delta)
            Self::Q8_1 => 36,
            Self::Q2_K => 84,      // K-quant super-block
            Self::Q3_K => 110,
            Self::Q4_K => 144,     // Used by Mistral-7B Q4_K_M
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
            _ => 4,                // Default fallback
        }
    }

    /// Get the number of elements per block
    pub fn elements_per_block(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
            _ => 1,
        }
    }

    /// Calculate total size in bytes for a given number of elements
    pub fn calculate_size(&self, n_elements: usize) -> usize {
        let elements_per_block = self.elements_per_block();
        let n_blocks = (n_elements + elements_per_block - 1) / elements_per_block;
        n_blocks * self.block_size()
    }
}

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// Metadata value types
#[derive(Debug, Clone)]
pub enum GGUFMetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFMetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

/// Tensor metadata
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GGUFQuantizationType,
    pub offset: u64,
    pub size: usize,
}

/// GGUF file parser
pub struct GGUFParser {
    file: BufReader<File>,
    header: Option<GGUFHeader>,
    metadata: HashMap<String, GGUFMetadataValue>,
    tensors: Vec<TensorMetadata>,
    data_offset: u64,
}

impl GGUFParser {
    /// Open a GGUF file for parsing
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let file_size = file.metadata()?.len();

        info!(
            "📂 Opening GGUF file: {} ({:.2} GB)",
            path.as_ref().display(),
            file_size as f64 / 1_073_741_824.0
        );

        Ok(Self {
            file: BufReader::new(file),
            header: None,
            metadata: HashMap::new(),
            tensors: Vec::new(),
            data_offset: 0,
        })
    }

    /// Parse the GGUF file header and metadata
    pub fn parse_header(&mut self) -> Result<&GGUFHeader> {
        // Read magic bytes
        let magic = self.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(anyhow!("Invalid GGUF magic bytes: 0x{:X}", magic));
        }

        // Read version
        let version = self.read_u32()?;
        if version != GGUF_VERSION {
            warn!("GGUF version {} differs from expected {}", version, GGUF_VERSION);
        }

        // Read counts
        let tensor_count = self.read_u64()?;
        let metadata_kv_count = self.read_u64()?;

        info!(
            "✅ GGUF Header: version={}, tensors={}, metadata_kvs={}",
            version, tensor_count, metadata_kv_count
        );

        self.header = Some(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        });

        // Parse metadata key-value pairs
        for i in 0..metadata_kv_count {
            let key = self.read_string()?;
            let value = self.read_metadata_value()?;

            debug!("Metadata[{}]: {} = {:?}", i, key, value);
            self.metadata.insert(key, value);
        }

        Ok(self.header.as_ref().unwrap())
    }

    /// Parse tensor metadata
    pub fn parse_tensors(&mut self) -> Result<&Vec<TensorMetadata>> {
        let header = self.header.as_ref()
            .ok_or_else(|| anyhow!("Header not parsed yet"))?;

        info!("📊 Parsing {} tensor metadata entries...", header.tensor_count);

        for i in 0..header.tensor_count {
            let name = self.read_string()?;
            let n_dims = self.read_u32()? as usize;

            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(self.read_u64()? as usize);
            }

            let dtype_u32 = self.read_u32()?;
            let dtype = GGUFQuantizationType::from_u32(dtype_u32)?;

            let offset = self.read_u64()?;

            // Calculate tensor size
            let n_elements: usize = shape.iter().product();
            let size = dtype.calculate_size(n_elements);

            debug!(
                "Tensor[{}]: name={}, shape={:?}, dtype={:?}, offset={}, size={}",
                i, name, shape, dtype, offset, size
            );

            self.tensors.push(TensorMetadata {
                name,
                shape,
                dtype,
                offset,
                size,
            });
        }

        // Calculate data offset (after header, metadata, and tensor info)
        let current_pos = self.file.stream_position()?;

        // Align to GGUF_DEFAULT_ALIGNMENT
        self.data_offset = ((current_pos + GGUF_DEFAULT_ALIGNMENT - 1) / GGUF_DEFAULT_ALIGNMENT)
            * GGUF_DEFAULT_ALIGNMENT;

        info!("✅ Parsed {} tensors, data starts at offset {}",
            self.tensors.len(), self.data_offset);

        Ok(&self.tensors)
    }

    /// Get tensors for a specific layer range
    pub fn get_layer_tensors(&self, layer_start: usize, layer_end: usize) -> Vec<&TensorMetadata> {
        self.tensors
            .iter()
            .filter(|t| {
                // Extract layer number from tensor name
                // Mistral naming: "blk.{layer}.{component}"
                if let Some(layer_str) = t.name.strip_prefix("blk.") {
                    if let Some(dot_pos) = layer_str.find('.') {
                        if let Ok(layer) = layer_str[..dot_pos].parse::<usize>() {
                            return layer >= layer_start && layer <= layer_end;
                        }
                    }
                }
                false
            })
            .collect()
    }

    /// Read tensor data
    pub fn read_tensor_data(&mut self, tensor: &TensorMetadata) -> Result<Vec<u8>> {
        let absolute_offset = self.data_offset + tensor.offset;

        debug!(
            "📖 Reading tensor '{}' at offset {} (size: {} bytes)",
            tensor.name, absolute_offset, tensor.size
        );

        self.file.seek(SeekFrom::Start(absolute_offset))?;

        let mut buffer = vec![0u8; tensor.size];
        self.file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&GGUFMetadataValue> {
        self.metadata.get(key)
    }

    // Helper read methods
    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.file.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.file.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.file.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf)?)
    }

    fn read_metadata_value(&mut self) -> Result<GGUFMetadataValue> {
        let value_type = self.read_u32()?;

        match value_type {
            0 => Ok(GGUFMetadataValue::UInt8(self.read_u8()?)),
            1 => Ok(GGUFMetadataValue::Int8(self.read_u8()? as i8)),
            2 => {
                let mut buf = [0u8; 2];
                self.file.read_exact(&mut buf)?;
                Ok(GGUFMetadataValue::UInt16(u16::from_le_bytes(buf)))
            }
            3 => {
                let mut buf = [0u8; 2];
                self.file.read_exact(&mut buf)?;
                Ok(GGUFMetadataValue::Int16(i16::from_le_bytes(buf)))
            }
            4 => Ok(GGUFMetadataValue::UInt32(self.read_u32()?)),
            5 => Ok(GGUFMetadataValue::Int32(self.read_i32()?)),
            6 => Ok(GGUFMetadataValue::Float32(self.read_f32()?)),
            7 => Ok(GGUFMetadataValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GGUFMetadataValue::String(self.read_string()?)),
            9 => {
                // Array type
                let elem_type = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len);

                // Read array elements based on type
                for _ in 0..len {
                    // Recursively read based on elem_type
                    // For simplicity, we'll read strings for now
                    if elem_type == 8 {
                        arr.push(GGUFMetadataValue::String(self.read_string()?));
                    }
                }

                Ok(GGUFMetadataValue::Array(arr))
            }
            10 => Ok(GGUFMetadataValue::UInt64(self.read_u64()?)),
            11 => {
                let mut buf = [0u8; 8];
                self.file.read_exact(&mut buf)?;
                Ok(GGUFMetadataValue::Int64(i64::from_le_bytes(buf)))
            }
            12 => {
                let mut buf = [0u8; 8];
                self.file.read_exact(&mut buf)?;
                Ok(GGUFMetadataValue::Float64(f64::from_le_bytes(buf)))
            }
            _ => Err(anyhow!("Unknown metadata value type: {}", value_type)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_sizes() {
        assert_eq!(GGUFQuantizationType::F32.block_size(), 4);
        assert_eq!(GGUFQuantizationType::F16.block_size(), 2);
        assert_eq!(GGUFQuantizationType::Q4_K.block_size(), 144);

        assert_eq!(GGUFQuantizationType::Q4_K.elements_per_block(), 256);

        // 1024 elements with Q4_K
        let size = GGUFQuantizationType::Q4_K.calculate_size(1024);
        assert_eq!(size, 4 * 144); // 4 blocks of 256 elements each
    }

    #[test]
    fn test_quantization_type_from_u32() {
        assert_eq!(
            GGUFQuantizationType::from_u32(12).unwrap(),
            GGUFQuantizationType::Q4_K
        );
        assert_eq!(
            GGUFQuantizationType::from_u32(0).unwrap(),
            GGUFQuantizationType::F32
        );
        assert!(GGUFQuantizationType::from_u32(999).is_err());
    }
}
