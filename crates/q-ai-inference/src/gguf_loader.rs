//! GGUF Model Loader using Candle's Built-in GGUF Support
//!
//! This module provides a high-level interface for loading GGUF model files
//! using Candle's native quantized tensor support. It handles:
//! - GGUF file parsing via candle_core::quantized::gguf_file
//! - Layer-wise weight extraction for distributed inference
//! - Q4_K_M quantized tensor loading
//! - Device-specific tensor placement (CUDA/Metal/CPU)
//!
//! This implementation leverages Candle's robust GGUF parser instead of
//! implementing custom binary parsing.

use anyhow::{anyhow, Context, Result};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

use crate::types::DeviceCapability;

/// Mistral-7B layer weight structure
#[derive(Debug)]
pub struct MistralLayerWeights {
    /// Layer index (0-31 for Mistral-7B)
    pub layer_idx: usize,

    /// Attention query projection weights
    pub attn_q: Option<QTensor>,

    /// Attention key projection weights
    pub attn_k: Option<QTensor>,

    /// Attention value projection weights
    pub attn_v: Option<QTensor>,

    /// Attention output projection weights
    pub attn_output: Option<QTensor>,

    /// FFN gate projection weights
    pub ffn_gate: Option<QTensor>,

    /// FFN up projection weights
    pub ffn_up: Option<QTensor>,

    /// FFN down projection weights
    pub ffn_down: Option<QTensor>,

    /// Attention normalization weights
    pub attn_norm: Option<QTensor>,

    /// FFN normalization weights
    pub ffn_norm: Option<QTensor>,
}

/// GGUF model loader for Mistral-7B
pub struct GGUFModelLoader {
    model_path: PathBuf,
    device: Device,
}

impl GGUFModelLoader {
    /// Create a new GGUF model loader
    pub fn new<P: AsRef<Path>>(model_path: P, capability: &DeviceCapability) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {}", model_path.display()));
        }

        // Determine device based on capability
        let device = match capability {
            DeviceCapability::CUDA { .. } => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0)?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    warn!("CUDA capability detected but CUDA feature not enabled, using CPU");
                    Device::Cpu
                }
            }
            DeviceCapability::Metal { .. } => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0)?
                }
                #[cfg(not(feature = "metal"))]
                {
                    warn!("Metal capability detected but Metal feature not enabled, using CPU");
                    Device::Cpu
                }
            }
            DeviceCapability::CPU { .. } => Device::Cpu,
        };

        info!(
            "🔧 GGUFModelLoader initialized: device={:?}, path={}",
            device,
            model_path.display()
        );

        Ok(Self { model_path, device })
    }

    /// Load weights for a single layer
    pub fn load_layer(
        &self,
        layer_idx: usize,
        device: &Device,
    ) -> Result<MistralLayerWeights> {
        let mut file = File::open(&self.model_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        self.load_single_layer(&mut file, &content, layer_idx)
    }

    /// Load weights for a specific layer range
    pub fn load_layer_range(
        &self,
        layer_start: usize,
        layer_end: usize,
    ) -> Result<Vec<MistralLayerWeights>> {
        info!(
            "📦 Loading layers {}-{} from {}",
            layer_start,
            layer_end,
            self.model_path.display()
        );

        // Open GGUF file
        let mut file = File::open(&self.model_path)?;
        let content = gguf_file::Content::read(&mut file)
            .context("Failed to read GGUF file content")?;

        info!(
            "✅ GGUF file loaded: {} tensors",
            content.tensor_infos.len()
        );

        // Log metadata
        self.print_metadata(&content)?;

        // Load layers
        let mut layer_weights = Vec::new();

        for layer_idx in layer_start..=layer_end {
            debug!("Loading layer {}", layer_idx);

            let weights = self.load_single_layer(&mut file, &content, layer_idx)?;
            layer_weights.push(weights);
        }

        info!(
            "✅ Successfully loaded {} layers ({}-{})",
            layer_weights.len(),
            layer_start,
            layer_end
        );

        Ok(layer_weights)
    }

    /// Load weights for a single transformer layer
    fn load_single_layer(
        &self,
        file: &mut File,
        content: &gguf_file::Content,
        layer_idx: usize,
    ) -> Result<MistralLayerWeights> {
        // Mistral GGUF naming convention:
        // blk.{N}.attn_q.weight, blk.{N}.attn_k.weight, etc.

        let prefix = format!("blk.{}", layer_idx);

        let mut weights = MistralLayerWeights {
            layer_idx,
            attn_q: None,
            attn_k: None,
            attn_v: None,
            attn_output: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_down: None,
            attn_norm: None,
            ffn_norm: None,
        };

        // Load attention weights
        weights.attn_q = self.try_load_tensor(
            file,
            content,
            &format!("{}.attn_q.weight", prefix),
        )?;

        weights.attn_k = self.try_load_tensor(
            file,
            content,
            &format!("{}.attn_k.weight", prefix),
        )?;

        weights.attn_v = self.try_load_tensor(
            file,
            content,
            &format!("{}.attn_v.weight", prefix),
        )?;

        weights.attn_output = self.try_load_tensor(
            file,
            content,
            &format!("{}.attn_output.weight", prefix),
        )?;

        // Load FFN weights
        weights.ffn_gate = self.try_load_tensor(
            file,
            content,
            &format!("{}.ffn_gate.weight", prefix),
        )?;

        weights.ffn_up = self.try_load_tensor(
            file,
            content,
            &format!("{}.ffn_up.weight", prefix),
        )?;

        weights.ffn_down = self.try_load_tensor(
            file,
            content,
            &format!("{}.ffn_down.weight", prefix),
        )?;

        // Load normalization weights
        weights.attn_norm = self.try_load_tensor(
            file,
            content,
            &format!("{}.attn_norm.weight", prefix),
        )?;

        weights.ffn_norm = self.try_load_tensor(
            file,
            content,
            &format!("{}.ffn_norm.weight", prefix),
        )?;

        debug!(
            "Layer {} loaded: Q={}, K={}, V={}, Out={}, Gate={}, Up={}, Down={}",
            layer_idx,
            weights.attn_q.is_some(),
            weights.attn_k.is_some(),
            weights.attn_v.is_some(),
            weights.attn_output.is_some(),
            weights.ffn_gate.is_some(),
            weights.ffn_up.is_some(),
            weights.ffn_down.is_some()
        );

        Ok(weights)
    }

    /// Try to load a tensor by name
    fn try_load_tensor(
        &self,
        file: &mut File,
        content: &gguf_file::Content,
        name: &str,
    ) -> Result<Option<QTensor>> {
        if let Some(tensor_info) = content.tensor_infos.get(name) {
            debug!(
                "Loading tensor '{}': shape={:?}, dtype={:?}",
                name, tensor_info.shape, tensor_info.ggml_dtype
            );

            let qtensor = tensor_info
                .read(file, content.tensor_data_offset, &self.device)
                .with_context(|| format!("Failed to read tensor '{}'", name))?;

            Ok(Some(qtensor))
        } else {
            warn!("Tensor '{}' not found in GGUF file", name);
            Ok(None)
        }
    }

    /// Print GGUF metadata
    fn print_metadata(&self, content: &gguf_file::Content) -> Result<()> {
        debug!("📋 GGUF Metadata:");

        if let Some(arch) = content.metadata.get("general.architecture") {
            debug!("  Architecture: {:?}", arch);
        }

        if let Some(name) = content.metadata.get("general.name") {
            debug!("  Model Name: {:?}", name);
        }

        if let Some(param_count) = content.metadata.get("general.parameter_count") {
            debug!("  Parameters: {:?}", param_count);
        }

        if let Some(layers) = content.metadata.get("llama.block_count") {
            debug!("  Layers: {:?}", layers);
        }

        if let Some(hidden) = content.metadata.get("llama.embedding_length") {
            debug!("  Hidden Size: {:?}", hidden);
        }

        if let Some(vocab) = content.metadata.get("llama.vocab_size") {
            debug!("  Vocab Size: {:?}", vocab);
        }

        debug!("  Total Tensors: {}", content.tensor_infos.len());

        Ok(())
    }

    /// Get embedding and output layers
    pub fn load_special_layers(&self) -> Result<SpecialLayers> {
        let mut file = File::open(&self.model_path)?;
        let content = gguf_file::Content::read(&mut file)?;

        let token_embd = self.try_load_tensor(
            &mut file,
            &content,
            "token_embd.weight",
        )?;

        let output_norm = self.try_load_tensor(
            &mut file,
            &content,
            "output_norm.weight",
        )?;

        let output = self.try_load_tensor(
            &mut file,
            &content,
            "output.weight",
        )?;

        Ok(SpecialLayers {
            token_embd,
            output_norm,
            output,
        })
    }

    /// List all tensor names in the GGUF file
    pub fn list_tensors(&self) -> Result<Vec<String>> {
        let mut file = File::open(&self.model_path)?;
        let content = gguf_file::Content::read(&mut file)?;

        let names: Vec<String> = content.tensor_infos.keys().cloned().collect();

        info!("📊 GGUF contains {} tensors", names.len());
        Ok(names)
    }

    /// Get total model size in bytes
    pub fn model_size(&self) -> Result<u64> {
        let metadata = std::fs::metadata(&self.model_path)?;
        Ok(metadata.len())
    }
}

/// Special model layers (embedding, output)
#[derive(Debug)]
pub struct SpecialLayers {
    pub token_embd: Option<QTensor>,
    pub output_norm: Option<QTensor>,
    pub output: Option<QTensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_loader_creation() {
        let capability = DeviceCapability::CPU {
            cores: 8,
            ram_gb: 16,
        };

        // Should fail with non-existent path
        let result = GGUFModelLoader::new("/nonexistent/model.gguf", &capability);
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Only run if model file exists
    fn test_load_mistral_7b() {
        let model_path = "/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf";

        if !std::path::Path::new(model_path).exists() {
            return; // Skip if model not downloaded
        }

        let capability = DeviceCapability::CPU {
            cores: 8,
            ram_gb: 16,
        };

        let loader = GGUFModelLoader::new(model_path, &capability).unwrap();

        // Test loading first 2 layers
        let weights = loader.load_layer_range(0, 1).unwrap();
        assert_eq!(weights.len(), 2);

        // Verify layer 0 has weights loaded
        assert!(weights[0].attn_q.is_some() || weights[0].attn_k.is_some());
    }
}
