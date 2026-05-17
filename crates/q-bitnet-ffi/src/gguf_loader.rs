//! GGUF Loader for BitNet Models
//!
//! Loads Microsoft BitNet 1.58-bit models from GGUF format.
//! BitNet uses I2_S (2-bit signed) quantization which maps perfectly
//! to our ternary representation {-1, 0, +1}.
//!
//! ## GGUF Format
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           GGUF HEADER                   │
//! │  magic: "GGUF" (0x46554747)             │
//! │  version: 3                             │
//! │  tensor_count: number of tensors        │
//! │  metadata_kv_count: number of KVs       │
//! ├─────────────────────────────────────────┤
//! │           METADATA KV PAIRS             │
//! │  architecture, quantization, etc.       │
//! ├─────────────────────────────────────────┤
//! │           TENSOR INFO                   │
//! │  name, shape, dtype, offset per tensor  │
//! ├─────────────────────────────────────────┤
//! │           TENSOR DATA                   │
//! │  Packed 2-bit weights for BitNet        │
//! └─────────────────────────────────────────┘
//! ```

use crate::engine::{BitNetConfig, BitNetEngine, BitNetLayer};
use crate::ternary::{PackedTernary, TernaryTensor, TernaryValue};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use tracing::{debug, info, warn};

/// GGUF magic number
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// GGUF tensor data types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgufDtype {
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
    // BitNet specific: 2-bit signed integers
    I2_S = 29,
}

impl GgufDtype {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(GgufDtype::F32),
            1 => Some(GgufDtype::F16),
            2 => Some(GgufDtype::Q4_0),
            8 => Some(GgufDtype::Q8_0),
            22 => Some(GgufDtype::IQ2_S),
            24 => Some(GgufDtype::I8),
            29 => Some(GgufDtype::I2_S),
            _ => None,
        }
    }

    /// Bytes per element (for packed formats, this is approximate)
    fn bytes_per_element(&self) -> f32 {
        match self {
            GgufDtype::F32 => 4.0,
            GgufDtype::F16 => 2.0,
            GgufDtype::Q4_0 | GgufDtype::Q4_1 => 0.5625, // 18 bytes per 32 elements
            GgufDtype::Q8_0 | GgufDtype::Q8_1 => 1.0625, // 34 bytes per 32 elements
            GgufDtype::I2_S | GgufDtype::IQ2_S => 0.25, // 2 bits = 0.25 bytes
            GgufDtype::I8 => 1.0,
            _ => 1.0,
        }
    }
}

/// GGUF tensor metadata
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgufDtype,
    pub offset: u64,
    pub size_bytes: usize,
}

/// GGUF file header and metadata
#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
}

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// BitNet GGUF Loader
pub struct BitNetGgufLoader {
    path: String,
    header: GgufHeader,
    file: BufReader<File>,
}

impl BitNetGgufLoader {
    /// Open a BitNet GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        info!("📂 Opening BitNet GGUF: {}", path_str);

        let file = File::open(&path)
            .map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;
        let mut reader = BufReader::new(file);

        // Parse header
        let header = Self::parse_header(&mut reader)?;

        info!("✅ GGUF loaded: {} tensors, version {}",
              header.tensor_count, header.version);

        Ok(Self {
            path: path_str,
            header,
            file: reader,
        })
    }

    /// Parse GGUF header
    fn parse_header(reader: &mut BufReader<File>) -> Result<GgufHeader> {
        // Read magic
        let mut magic_bytes = [0u8; 4];
        reader.read_exact(&mut magic_bytes)?;
        let magic = u32::from_le_bytes(magic_bytes);

        if magic != GGUF_MAGIC {
            return Err(anyhow!("Invalid GGUF magic: {:08x} (expected {:08x})",
                              magic, GGUF_MAGIC));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version < 2 || version > 3 {
            return Err(anyhow!("Unsupported GGUF version: {}", version));
        }

        // Read counts
        let mut count_bytes = [0u8; 8];
        reader.read_exact(&mut count_bytes)?;
        let tensor_count = u64::from_le_bytes(count_bytes);

        reader.read_exact(&mut count_bytes)?;
        let metadata_kv_count = u64::from_le_bytes(count_bytes);

        info!("📊 GGUF v{}: {} tensors, {} metadata entries",
              version, tensor_count, metadata_kv_count);

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = Self::read_kv(reader)?;
            debug!("  KV: {} = {:?}", key, value);
            metadata.insert(key, value);
        }

        // Parse tensor info
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let info = Self::read_tensor_info(reader)?;
            debug!("  Tensor: {} {:?} {:?}", info.name, info.shape, info.dtype);
            tensors.push(info);
        }

        // Calculate data offset (aligned to 32 bytes)
        let current_pos = reader.stream_position()?;
        let alignment = 32u64;
        let data_offset = (current_pos + alignment - 1) / alignment * alignment;

        Ok(GgufHeader {
            version,
            tensor_count,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Read a key-value pair
    fn read_kv(reader: &mut BufReader<File>) -> Result<(String, GgufValue)> {
        let key = Self::read_string(reader)?;
        let value = Self::read_value(reader)?;
        Ok((key, value))
    }

    /// Read a string (length-prefixed)
    fn read_string(reader: &mut BufReader<File>) -> Result<String> {
        let mut len_bytes = [0u8; 8];
        reader.read_exact(&mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| anyhow!("Invalid UTF-8 in string: {}", e))
    }

    /// Read a value
    fn read_value(reader: &mut BufReader<File>) -> Result<GgufValue> {
        let mut type_bytes = [0u8; 4];
        reader.read_exact(&mut type_bytes)?;
        let value_type = u32::from_le_bytes(type_bytes);

        match value_type {
            0 => {
                let mut b = [0u8; 1];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::U8(b[0]))
            }
            1 => {
                let mut b = [0u8; 1];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::I8(b[0] as i8))
            }
            2 => {
                let mut b = [0u8; 2];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::U16(u16::from_le_bytes(b)))
            }
            3 => {
                let mut b = [0u8; 2];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::I16(i16::from_le_bytes(b)))
            }
            4 => {
                let mut b = [0u8; 4];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::U32(u32::from_le_bytes(b)))
            }
            5 => {
                let mut b = [0u8; 4];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::I32(i32::from_le_bytes(b)))
            }
            6 => {
                let mut b = [0u8; 4];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::F32(f32::from_le_bytes(b)))
            }
            7 => {
                let mut b = [0u8; 1];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::Bool(b[0] != 0))
            }
            8 => {
                let s = Self::read_string(reader)?;
                Ok(GgufValue::String(s))
            }
            9 => {
                // Array
                let mut type_bytes = [0u8; 4];
                reader.read_exact(&mut type_bytes)?;
                let _elem_type = u32::from_le_bytes(type_bytes);

                let mut len_bytes = [0u8; 8];
                reader.read_exact(&mut len_bytes)?;
                let len = u64::from_le_bytes(len_bytes) as usize;

                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(Self::read_value(reader)?);
                }
                Ok(GgufValue::Array(values))
            }
            10 => {
                let mut b = [0u8; 8];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::U64(u64::from_le_bytes(b)))
            }
            11 => {
                let mut b = [0u8; 8];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::I64(i64::from_le_bytes(b)))
            }
            12 => {
                let mut b = [0u8; 8];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::F64(f64::from_le_bytes(b)))
            }
            _ => Err(anyhow!("Unknown value type: {}", value_type)),
        }
    }

    /// Read tensor info
    fn read_tensor_info(reader: &mut BufReader<File>) -> Result<GgufTensorInfo> {
        let name = Self::read_string(reader)?;

        // Number of dimensions
        let mut ndims_bytes = [0u8; 4];
        reader.read_exact(&mut ndims_bytes)?;
        let ndims = u32::from_le_bytes(ndims_bytes) as usize;

        // Shape
        let mut shape = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            let mut dim_bytes = [0u8; 8];
            reader.read_exact(&mut dim_bytes)?;
            shape.push(u64::from_le_bytes(dim_bytes) as usize);
        }

        // Data type
        let mut dtype_bytes = [0u8; 4];
        reader.read_exact(&mut dtype_bytes)?;
        let dtype_val = u32::from_le_bytes(dtype_bytes);
        let dtype = GgufDtype::from_u32(dtype_val)
            .ok_or_else(|| anyhow!("Unknown dtype: {}", dtype_val))?;

        // Offset
        let mut offset_bytes = [0u8; 8];
        reader.read_exact(&mut offset_bytes)?;
        let offset = u64::from_le_bytes(offset_bytes);

        // Calculate size
        let numel: usize = shape.iter().product();
        let size_bytes = (numel as f32 * dtype.bytes_per_element()) as usize;

        Ok(GgufTensorInfo {
            name,
            shape,
            dtype,
            offset,
            size_bytes,
        })
    }

    /// Get model configuration from metadata
    pub fn get_config(&self) -> BitNetConfig {
        let get_u32 = |key: &str, default: u32| -> u32 {
            self.header.metadata.get(key)
                .and_then(|v| match v {
                    GgufValue::U32(n) => Some(*n),
                    GgufValue::I32(n) => Some(*n as u32),
                    _ => None,
                })
                .unwrap_or(default)
        };

        // Try to extract from metadata or use defaults for BitNet 2B-4T
        let hidden_dim = get_u32("llama.embedding_length", 2048) as usize;
        let num_layers = get_u32("llama.block_count", 24) as usize;
        let num_heads = get_u32("llama.attention.head_count", 16) as usize;
        let num_kv_heads = get_u32("llama.attention.head_count_kv", 4) as usize;
        let intermediate_dim = get_u32("llama.feed_forward_length", 5632) as usize;
        let vocab_size = get_u32("llama.vocab_size", 32000) as usize;

        info!("📊 BitNet config from GGUF:");
        info!("   hidden_dim: {}", hidden_dim);
        info!("   num_layers: {}", num_layers);
        info!("   num_heads: {}", num_heads);
        info!("   num_kv_heads: {}", num_kv_heads);
        info!("   intermediate_dim: {}", intermediate_dim);
        info!("   vocab_size: {}", vocab_size);

        BitNetConfig {
            model_path: Some(self.path.clone()),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim: hidden_dim / num_heads,
            intermediate_dim,
            vocab_size,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            use_relu_squared: true,
        }
    }

    /// Load a tensor as TernaryTensor
    pub fn load_tensor(&mut self, name: &str) -> Result<TernaryTensor> {
        let info = self.header.tensors.iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("Tensor not found: {}", name))?
            .clone();

        debug!("Loading tensor: {} {:?} {:?}", info.name, info.shape, info.dtype);

        // Seek to tensor data
        self.file.seek(SeekFrom::Start(self.header.data_offset + info.offset))?;

        // Read raw bytes
        let mut data = vec![0u8; info.size_bytes];
        self.file.read_exact(&mut data)?;

        // Convert to ternary based on dtype
        let numel: usize = info.shape.iter().product();

        let packed = match info.dtype {
            GgufDtype::I2_S | GgufDtype::IQ2_S => {
                // Already 2-bit packed - perfect for ternary!
                // I2_S packs 4 values per byte
                PackedTernary::from_bytes(data, numel)
            }
            GgufDtype::I8 => {
                // Convert i8 to ternary
                let i8_data: Vec<i8> = data.iter().map(|&b| b as i8).collect();
                PackedTernary::from_i8_slice(&i8_data)
            }
            GgufDtype::F32 => {
                // Convert f32 to ternary using absmean quantization
                let f32_data: Vec<f32> = data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                PackedTernary::from_f32_slice(&f32_data)
            }
            _ => {
                warn!("Unsupported dtype {:?}, treating as packed ternary", info.dtype);
                PackedTernary::from_bytes(data, numel)
            }
        };

        Ok(TernaryTensor {
            packed,
            shape: info.shape,
            scale: None,
        })
    }

    /// Load all tensors into a BitNetEngine
    pub fn load_engine(&mut self) -> Result<BitNetEngine> {
        let config = self.get_config();
        let mut engine = BitNetEngine::new(config.clone());

        info!("📦 Loading {} layers from GGUF...", config.num_layers);

        for layer_idx in 0..config.num_layers {
            // Load attention weights
            let attn_q = self.load_tensor(&format!("blk.{}.attn_q.weight", layer_idx))
                .or_else(|_| self.load_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx)))
                .ok();

            let attn_k = self.load_tensor(&format!("blk.{}.attn_k.weight", layer_idx))
                .or_else(|_| self.load_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx)))
                .ok();

            let attn_v = self.load_tensor(&format!("blk.{}.attn_v.weight", layer_idx))
                .or_else(|_| self.load_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx)))
                .ok();

            let attn_output = self.load_tensor(&format!("blk.{}.attn_output.weight", layer_idx))
                .or_else(|_| self.load_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx)))
                .ok();

            // Load FFN weights
            let ffn_gate = self.load_tensor(&format!("blk.{}.ffn_gate.weight", layer_idx))
                .or_else(|_| self.load_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx)))
                .ok();

            let ffn_up = self.load_tensor(&format!("blk.{}.ffn_up.weight", layer_idx))
                .or_else(|_| self.load_tensor(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx)))
                .ok();

            let ffn_down = self.load_tensor(&format!("blk.{}.ffn_down.weight", layer_idx))
                .or_else(|_| self.load_tensor(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx)))
                .ok();

            // Update existing layer in engine
            if layer_idx < engine.layers.len() {
                let layer = &mut engine.layers[layer_idx];
                layer.attn_q = attn_q;
                layer.attn_k = attn_k;
                layer.attn_v = attn_v;
                layer.attn_o = attn_output;
                layer.ffn_gate = ffn_gate;
                layer.ffn_up = ffn_up;
                layer.ffn_down = ffn_down;

                if layer.is_complete() {
                    debug!("✅ Layer {} loaded completely", layer_idx);
                } else {
                    debug!("⚠️ Layer {} partially loaded", layer_idx);
                }
            }
        }

        // Load embedding and LM head
        if let Ok(emb) = self.load_tensor("token_embd.weight")
            .or_else(|_| self.load_tensor("model.embed_tokens.weight")) {
            info!("✅ Loaded token embedding");
            engine.token_embedding = Some(emb);
        }

        if let Ok(lm) = self.load_tensor("output.weight")
            .or_else(|_| self.load_tensor("lm_head.weight")) {
            info!("✅ Loaded LM head");
            engine.lm_head = Some(lm);
        }

        // Count loaded layers
        let complete_layers = engine.layers.iter().filter(|l| l.is_complete()).count();
        info!("🎉 BitNet engine loaded: {}/{} complete layers", complete_layers, config.num_layers);

        Ok(engine)
    }

    /// Get tensor names matching a pattern
    pub fn get_tensor_names(&self, pattern: &str) -> Vec<String> {
        self.header.tensors.iter()
            .filter(|t| t.name.contains(pattern))
            .map(|t| t.name.clone())
            .collect()
    }

    /// Get total model size in bytes
    pub fn model_size_bytes(&self) -> usize {
        self.header.tensors.iter().map(|t| t.size_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_dtype_bytes() {
        assert_eq!(GgufDtype::F32.bytes_per_element(), 4.0);
        assert_eq!(GgufDtype::I2_S.bytes_per_element(), 0.25);
        assert_eq!(GgufDtype::I8.bytes_per_element(), 1.0);
    }
}
