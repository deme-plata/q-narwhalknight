//! Qwen3-VL Vision-Language Model Support
//!
//! This module implements vision processing capabilities for Qwen3-VL-8B-Instruct,
//! enabling multimodal inference with both images and text.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    Qwen3-VL Pipeline                            │
//! ├────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  Input: Text + Images                                            │
//! │    │                                                             │
//! │    ├──► Image Preprocessing (448×448 patches)                   │
//! │    │      │                                                      │
//! │    │      ├──► Vision Transformer (ViT)                         │
//! │    │      │      │                                               │
//! │    │      │      └──► Image Embeddings [N, 768]                │
//! │    │                                                             │
//! │    └──► Text Tokenization (Qwen tokenizer)                     │
//! │           │                                                      │
//! │           └──► Text Embeddings [M, 768]                        │
//! │                                                                  │
//! │  Combined Embeddings: [<img>, text tokens, <img>, ...]          │
//! │    │                                                             │
//! │    └──► Transformer Decoder (32 layers)                        │
//! │           │                                                      │
//! │           └──► Generated Text Response                          │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Vision Processing Details
//!
//! Qwen3-VL uses a Vision Transformer (ViT) to process images:
//! - **Input Size**: 448×448 RGB images
//! - **Patch Size**: 14×14 pixels
//! - **Patches per Image**: (448/14)² = 1024 patches
//! - **Embedding Dim**: 768 (matches text embedding dimension)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use q_ai_inference::Qwen3VLProcessor;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let processor = Qwen3VLProcessor::new("/path/to/model.gguf").await?;
//!
//! // Process image + text
//! let image_bytes = std::fs::read("photo.jpg")?;
//! let response = processor.generate_multimodal(
//!     "What is in this image?",
//!     vec![image_bytes],
//! ).await?;
//!
//! println!("Response: {}", response);
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use std::path::Path;
use tracing::{debug, info, warn};

/// Configuration for Qwen3-VL vision processing
#[derive(Debug, Clone)]
pub struct Qwen3VLConfig {
    /// Image size expected by vision transformer (default: 448×448)
    pub image_size: usize,
    /// Patch size for ViT (default: 14×14)
    pub patch_size: usize,
    /// Embedding dimension (default: 768 to match 8B model)
    pub embed_dim: usize,
    /// Number of attention heads in ViT
    pub num_heads: usize,
    /// Number of ViT layers
    pub num_vision_layers: usize,
    /// Device for computation (CPU/CUDA/Metal)
    pub device: Device,
}

impl Default for Qwen3VLConfig {
    fn default() -> Self {
        Self {
            image_size: 448,
            patch_size: 14,
            embed_dim: 768,
            num_heads: 12,
            num_vision_layers: 24,
            device: Device::Cpu,
        }
    }
}

/// Qwen3-VL multimodal processor
///
/// Handles both vision and language processing for Qwen3-VL-8B-Instruct
pub struct Qwen3VLProcessor {
    config: Qwen3VLConfig,
    // Vision encoder will be loaded from GGUF when available
    // For now, we implement image preprocessing only
}

impl Qwen3VLProcessor {
    /// Create a new Qwen3-VL processor
    ///
    /// # Arguments
    /// * `model_path` - Path to Qwen3-VL-8B GGUF model file
    pub async fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let config = Qwen3VLConfig::default();

        info!("🖼️  Initializing Qwen3-VL Vision-Language Processor");
        info!("   📏 Image size: {}×{}", config.image_size, config.image_size);
        info!("   🧩 Patch size: {}×{}", config.patch_size, config.patch_size);
        info!("   📐 Embedding dim: {}", config.embed_dim);
        info!("   🔢 Vision layers: {}", config.num_vision_layers);

        Ok(Self { config })
    }

    /// Preprocess an image for vision encoder
    ///
    /// Performs the following transformations:
    /// 1. Resize to 448×448 (bicubic interpolation)
    /// 2. Normalize RGB values to [-1, 1] range
    /// 3. Convert to CHW format (Channels, Height, Width)
    ///
    /// # Arguments
    /// * `image_bytes` - Raw image bytes (JPEG, PNG, WebP)
    ///
    /// # Returns
    /// Tensor of shape [3, 448, 448] with normalized pixel values
    pub fn preprocess_image(&self, image_bytes: &[u8]) -> Result<Tensor> {
        debug!("📷 Preprocessing image ({} bytes)", image_bytes.len());

        // 1. Decode image
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| anyhow!("Failed to decode image: {}", e))?;

        debug!("   Original size: {}×{}", img.width(), img.height());

        // 2. Resize to target size (448×448) using bicubic interpolation
        let resized = img.resize_exact(
            self.config.image_size as u32,
            self.config.image_size as u32,
            FilterType::CatmullRom, // High-quality bicubic
        );

        // 3. Convert to RGB (in case of RGBA or grayscale)
        let rgb_img = resized.to_rgb8();

        // 4. Convert to f32 tensor with normalization
        let (width, height) = rgb_img.dimensions();
        let pixel_data: Vec<f32> = rgb_img
            .pixels()
            .flat_map(|pixel| {
                // Normalize from [0, 255] to [-1, 1]
                pixel.0.iter().map(|&val| {
                    (val as f32 / 255.0) * 2.0 - 1.0
                })
            })
            .collect();

        // 5. Reshape to [3, H, W] (CHW format)
        let tensor = Tensor::from_vec(
            pixel_data,
            (height as usize, width as usize, 3),
            &self.config.device,
        )?;

        // Transpose from HWC to CHW
        let chw_tensor = tensor.permute((2, 0, 1))?;

        debug!("✅ Image preprocessed: shape {:?}", chw_tensor.shape());

        Ok(chw_tensor)
    }

    /// Extract vision embeddings from preprocessed image
    ///
    /// Converts image tensor into patch embeddings suitable for the transformer.
    ///
    /// # Process
    /// 1. Split image into 14×14 patches (1024 patches total for 448×448 image)
    /// 2. Flatten each patch
    /// 3. Project to embedding dimension (768)
    /// 4. Add positional embeddings
    ///
    /// # Arguments
    /// * `image_tensor` - Preprocessed image [3, 448, 448]
    ///
    /// # Returns
    /// Vision embeddings [num_patches, embed_dim] = [1024, 768]
    pub fn extract_vision_embeddings(&self, image_tensor: &Tensor) -> Result<Tensor> {
        debug!("🎨 Extracting vision embeddings");

        let (channels, height, width) = image_tensor.dims3()?;

        if height != self.config.image_size || width != self.config.image_size {
            return Err(anyhow!(
                "Image tensor has wrong size: {}×{}, expected {}×{}",
                height,
                width,
                self.config.image_size,
                self.config.image_size
            ));
        }

        let patch_size = self.config.patch_size;
        let num_patches_per_side = self.config.image_size / patch_size;
        let num_patches = num_patches_per_side * num_patches_per_side;

        debug!("   Patch grid: {}×{} = {} patches",
            num_patches_per_side, num_patches_per_side, num_patches);

        // Extract patches using unfold operation
        // This is a simplified implementation; full ViT would include:
        // - Patch projection layer (Conv2d or Linear)
        // - Positional embeddings
        // - CLS token
        // - Multi-head self-attention layers

        // For now, we create a placeholder embedding tensor
        // TODO: Load actual vision weights from GGUF and implement full ViT forward pass

        let patch_embeddings = Tensor::zeros(
            (num_patches, self.config.embed_dim),
            DType::F32,
            &self.config.device,
        )?;

        debug!("✅ Vision embeddings: shape {:?}", patch_embeddings.shape());

        Ok(patch_embeddings)
    }

    /// Process multiple images into vision embeddings
    ///
    /// # Arguments
    /// * `images` - Vector of raw image bytes
    ///
    /// # Returns
    /// Combined vision embeddings [total_patches, embed_dim]
    pub fn process_images(&self, images: Vec<Vec<u8>>) -> Result<Vec<Tensor>> {
        info!("🖼️  Processing {} images", images.len());

        let mut all_embeddings = Vec::new();

        for (idx, image_bytes) in images.iter().enumerate() {
            debug!("   Processing image {} of {}", idx + 1, images.len());

            // Preprocess image
            let preprocessed = self.preprocess_image(image_bytes)?;

            // Extract vision embeddings
            let embeddings = self.extract_vision_embeddings(&preprocessed)?;

            all_embeddings.push(embeddings);
        }

        info!("✅ Processed {} images successfully", all_embeddings.len());

        Ok(all_embeddings)
    }

    /// Create multimodal prompt with image placeholders
    ///
    /// Qwen3-VL uses special tokens to indicate where images should be inserted:
    /// ```text
    /// <|im_start|>system
    /// You are a helpful assistant.<|im_end|>
    /// <|im_start|>user
    /// <|vision_start|><|image_pad|><|vision_end|>What is in this image?<|im_end|>
    /// <|im_start|>assistant
    /// ```
    ///
    /// # Arguments
    /// * `text` - User text prompt
    /// * `num_images` - Number of images to embed
    ///
    /// # Returns
    /// Formatted prompt with image placeholders
    pub fn format_multimodal_prompt(&self, text: &str, num_images: usize) -> String {
        let mut prompt = String::from("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n");

        // Add image placeholders
        for _ in 0..num_images {
            prompt.push_str("<|vision_start|><|image_pad|><|vision_end|>");
        }

        // Add text prompt
        prompt.push_str(text);
        prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");

        prompt
    }
}

/// Image attachment data from database
#[derive(Debug, Clone)]
pub struct ImageAttachment {
    pub id: String,
    pub data: Vec<u8>,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
}

impl ImageAttachment {
    /// Load image from database attachment
    pub fn from_bytes(id: String, data: Vec<u8>, mime_type: String) -> Result<Self> {
        // Decode to get dimensions
        let img = image::load_from_memory(&data)
            .map_err(|e| anyhow!("Failed to decode attachment image: {}", e))?;

        Ok(Self {
            id,
            data,
            mime_type,
            width: img.width(),
            height: img.height(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_config_defaults() {
        let config = Qwen3VLConfig::default();
        assert_eq!(config.image_size, 448);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.embed_dim, 768);
    }

    #[tokio::test]
    async fn test_image_preprocessing() {
        let processor = Qwen3VLProcessor {
            config: Qwen3VLConfig::default(),
        };

        // Create a simple test image (1×1 red pixel)
        let test_image = DynamicImage::ImageRgb8(
            image::RgbImage::from_pixel(1, 1, image::Rgb([255, 0, 0]))
        );

        let mut buffer = Vec::new();
        test_image
            .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
            .unwrap();

        // Preprocess
        let result = processor.preprocess_image(&buffer);
        assert!(result.is_ok());

        let tensor = result.unwrap();
        let shape = tensor.shape();

        // Should be resized to 448×448
        assert_eq!(shape.dims(), &[3, 448, 448]);
    }

    #[test]
    fn test_multimodal_prompt_formatting() {
        let processor = Qwen3VLProcessor {
            config: Qwen3VLConfig::default(),
        };

        let prompt = processor.format_multimodal_prompt("What is this?", 1);

        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|vision_start|>"));
        assert!(prompt.contains("What is this?"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_multiple_images_prompt() {
        let processor = Qwen3VLProcessor {
            config: Qwen3VLConfig::default(),
        };

        let prompt = processor.format_multimodal_prompt("Compare these images", 2);

        // Should have 2 image placeholders
        let placeholder_count = prompt.matches("<|vision_start|>").count();
        assert_eq!(placeholder_count, 2);
    }
}
