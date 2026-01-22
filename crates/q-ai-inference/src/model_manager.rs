//! Model Manager - Lazy loading and switching between AI models
//!
//! This module provides efficient model management to prevent loading multiple large models
//! into RAM simultaneously. Only the currently requested model is kept in memory.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                      Model Manager                             │
//! │  ┌──────────────────────────────────────────────────────┐      │
//! │  │  Current Model: Mistral-7B (4.1 GB RAM)             │      │
//! │  │  ┌────────────────────────────────────────────┐     │      │
//! │  │  │  MistralRsEngine                           │     │      │
//! │  │  │  • Inference pipeline                      │     │      │
//! │  │  │  • KV-cache                                │     │      │
//! │  │  │  • Tokenizer                               │     │      │
//! │  │  └────────────────────────────────────────────┘     │      │
//! │  └──────────────────────────────────────────────────────┘      │
//! │                                                                 │
//! │  User requests Mistral-Small-24B:                              │
//! │  1. Unload Mistral-7B (frees 4.1 GB RAM)          ✅           │
//! │  2. Download Mistral-Small-24B if not cached      ⬇️           │
//! │  3. Load Mistral-Small-24B (uses 14 GB RAM)       ✅           │
//! │                                                                 │
//! │  Peak RAM: 14 GB (not 18.1 GB!)                                │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Lazy Loading**: Models loaded only when requested
//! - **Automatic Unloading**: Previous model released before loading new one
//! - **HTTP Download**: Fetch models from nginx if not cached locally
//! - **Memory Efficient**: Peak RAM = largest model size (not sum of all models)
//! - **Thread-Safe**: Arc<RwLock<>> for concurrent access
//!
//! ## Usage
//!
//! ```rust,no_run
//! use q_ai_inference::ModelManager;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut manager = ModelManager::new("/path/to/models".into()).await?;
//!
//! // Load Mistral-7B (4.1 GB)
//! let engine = manager.get_or_load_model("Mistral-7B-Instruct-v0.3").await?;
//! let response = engine.generate_stream("Hello", 50, |event| async { Ok(()) }).await?;
//!
//! // Switch to Mistral-Small-24B (14 GB)
//! // Mistral-7B automatically unloaded, freeing 4.1 GB
//! let engine = manager.get_or_load_model("Mistral-Small-3.2-24B-Instruct").await?;
//! let response = engine.generate_stream("Explain quantum computing", 100, |event| async { Ok(()) }).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::mistralrs_engine::{MistralRsConfig, MistralRsEngine};

/// Model metadata for tracking and management
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name (e.g., "Mistral-7B-Instruct-v0.3")
    pub name: String,
    /// Full GGUF filename (e.g., "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf")
    pub gguf_filename: String,
    /// Number of transformer layers
    pub layer_count: usize,
    /// Parameter count (billions)
    pub parameters_billions: f32,
    /// Approximate RAM usage in MB (Q4_K_M quantization)
    pub ram_usage_mb: usize,
    /// HTTP URL for downloading from nginx
    pub download_url: String,
}

impl ModelMetadata {
    /// Get model metadata by name
    pub fn from_name(name: &str, base_url: &str) -> Result<Self> {
        // Determine model-specific configuration
        let (gguf_filename, layer_count, parameters_billions, ram_usage_mb) =
            if name.contains("Kimi-K2") || name.contains("kimi-k2") {
                (
                    "Kimi-K2-unsloth.UD-TQ1_0.gguf".to_string(),
                    120, // Estimated layer count for 1T MoE model
                    1000.0, // 1 trillion parameters
                    245000, // 245 GB with UD-TQ1_0 quantization
                )
            } else if name.contains("Mistral-Small-3.2-24B") {
                (
                    "Mistral-Small-3.2-24B-Instruct-Q4_K_M.gguf".to_string(),
                    56, // 56 layers for 24B model
                    24.0,
                    14336, // 14 GB
                )
            } else if name.contains("Mistral-7B") {
                (
                    "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf".to_string(),
                    32, // 32 layers for 7B model
                    7.0,
                    4100, // 4.1 GB
                )
            } else if name.contains("Llama-7B") {
                (
                    "Llama-7B-Q4_K_M.gguf".to_string(),
                    32,
                    7.0,
                    4100,
                )
            } else if name.contains("Llama-13B") {
                (
                    "Llama-13B-Q4_K_M.gguf".to_string(),
                    40,
                    13.0,
                    7300,
                )
            } else if name.contains("Llama-70B") {
                (
                    "Llama-70B-Q4_K_M.gguf".to_string(),
                    80,
                    70.0,
                    38000,
                )
            } else if name.contains("Qwen3-VL-8B") || name.contains("qwen3-vl-8b") {
                (
                    "Qwen3-VL-8B-Instruct-Q4_K_M.gguf".to_string(),
                    32, // 32 layers for 8B model
                    8.0,
                    5120, // ~5.1 GB with Q4_K_M quantization
                )
            } else if name.contains("Qwen3-0.6B") || name.contains("qwen3-0.6b") {
                (
                    "Qwen3-0.6B-Q4_K_M.gguf".to_string(),
                    28, // 28 layers for Qwen3-0.6B
                    0.6,
                    400, // ~379 MB with Q4_K_M quantization
                )
            } else if name.contains("Qwen3-4B") || name.contains("qwen3-4b") {
                (
                    "Qwen3-4B-Q4_K_M.gguf".to_string(),
                    36, // 36 layers for Qwen3-4B
                    4.0,
                    2500, // ~2.4 GB with Q4_K_M quantization
                )
            } else if name.contains("Ministral-3B") || name.contains("ministral-3b") {
                // ⚠️ Ministral-3B has mistral3 architecture NOT supported by mistral.rs GGUF
                (
                    "Ministral-3B-Instruct-Q4_K_M.gguf".to_string(),
                    32, // 32 layers for 3B model
                    3.0,
                    2150, // ~2.15 GB with Q4_K_M quantization
                )
            } else {
                return Err(anyhow!("Unknown model: {}. Supported models: Qwen3-0.6B (fastest), Qwen3-4B (balanced), Mistral-7B, Mistral-Small-3.2-24B, Qwen3-VL-8B, Kimi-K2, Llama-7B/13B/70B", name));
            };

        let download_url = format!("{}/downloads/{}", base_url, gguf_filename);

        Ok(Self {
            name: name.to_string(),
            gguf_filename,
            layer_count,
            parameters_billions,
            ram_usage_mb,
            download_url,
        })
    }
}

/// Model Manager for lazy loading and efficient memory management
pub struct ModelManager {
    /// Base directory for model storage
    models_dir: PathBuf,
    /// Currently loaded model (if any)
    current_model: Arc<RwLock<Option<Arc<MistralRsEngine>>>>,
    /// Name of currently loaded model
    current_model_name: Arc<RwLock<Option<String>>>,
    /// Model metadata cache
    model_metadata: Arc<RwLock<HashMap<String, ModelMetadata>>>,
    /// Base URL for downloading models (nginx server)
    base_url: String,
}

impl ModelManager {
    /// Create a new model manager
    ///
    /// # Arguments
    /// * `models_dir` - Directory where models are stored/downloaded
    ///
    /// # Example
    /// ```rust,no_run
    /// # use q_ai_inference::ModelManager;
    /// # async fn example() -> anyhow::Result<()> {
    /// let manager = ModelManager::new("/opt/orobit/shared/q-narwhalknight/models".into()).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(models_dir: PathBuf) -> Result<Self> {
        info!("🗂️  Initializing Model Manager");
        info!("   📁 Models directory: {}", models_dir.display());

        // Ensure models directory exists
        tokio::fs::create_dir_all(&models_dir).await?;

        // Determine base URL for downloads (nginx server)
        let base_url = std::env::var("Q_MODEL_BASE_URL")
            .unwrap_or_else(|_| "http://quillon.xyz".to_string());

        info!("   🌐 Model download URL: {}/downloads/", base_url);

        Ok(Self {
            models_dir,
            current_model: Arc::new(RwLock::new(None)),
            current_model_name: Arc::new(RwLock::new(None)),
            model_metadata: Arc::new(RwLock::new(HashMap::new())),
            base_url,
        })
    }

    /// Get or load a model by name
    ///
    /// If the model is already loaded, returns it immediately.
    /// If a different model is loaded, unloads it first to free RAM.
    /// If the model file doesn't exist, downloads it from nginx.
    ///
    /// # Arguments
    /// * `model_name` - Name of model to load (e.g., "Mistral-7B-Instruct-v0.3")
    ///
    /// # Returns
    /// Arc reference to the loaded MistralRsEngine
    ///
    /// # Example
    /// ```rust,no_run
    /// # use q_ai_inference::ModelManager;
    /// # async fn example() -> anyhow::Result<()> {
    /// # let mut manager = ModelManager::new("/path/to/models".into()).await?;
    /// let engine = manager.get_or_load_model("Mistral-7B-Instruct-v0.3").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_or_load_model(&self, model_name: &str) -> Result<Arc<MistralRsEngine>> {
        // Check if this model is already loaded
        let current_name = self.current_model_name.read().await;
        if let Some(loaded_name) = current_name.as_ref() {
            if loaded_name == model_name {
                debug!("✅ Model '{}' already loaded, reusing existing instance", model_name);
                let model = self.current_model.read().await;
                if let Some(engine) = model.as_ref() {
                    return Ok(Arc::clone(engine));
                }
            }
        }
        drop(current_name);

        info!("🔄 Switching to model: {}", model_name);

        // Get model metadata
        let metadata = ModelMetadata::from_name(model_name, &self.base_url)?;

        info!("📊 Model metadata:");
        info!("   🗂️  GGUF file: {}", metadata.gguf_filename);
        info!("   🧠 Parameters: {:.1}B", metadata.parameters_billions);
        info!("   🔢 Layers: {}", metadata.layer_count);
        info!("   💾 RAM usage: {} MB (~{:.1} GB)", metadata.ram_usage_mb, metadata.ram_usage_mb as f32 / 1024.0);

        // Unload previous model to free RAM
        self.unload_current_model().await?;

        // Ensure model file exists (download if needed)
        let model_path = self.ensure_model_exists(&metadata).await?;

        // Load the new model
        info!("🚀 Loading model from: {}", model_path.display());
        let start_time = std::time::Instant::now();

        let config = MistralRsConfig {
            model_path: model_path.to_string_lossy().to_string(),
            enable_kv_cache: true,
            enable_distributed: false, // Default to local inference
            ..Default::default()
        };

        let engine = Arc::new(MistralRsEngine::with_config(config).await?);

        let load_time = start_time.elapsed();
        info!("✅ Model loaded successfully in {:.2}s", load_time.as_secs_f64());

        // Update current model
        *self.current_model.write().await = Some(Arc::clone(&engine));
        *self.current_model_name.write().await = Some(model_name.to_string());

        // Cache metadata
        self.model_metadata
            .write()
            .await
            .insert(model_name.to_string(), metadata);

        Ok(engine)
    }

    /// Unload the currently loaded model to free RAM
    async fn unload_current_model(&self) -> Result<()> {
        let current_name = self.current_model_name.read().await;
        if let Some(model_name) = current_name.as_ref() {
            info!("🗑️  Unloading previous model: {}", model_name);

            // Get RAM usage before unloading (for logging)
            let metadata = self.model_metadata.read().await;
            let ram_freed_mb = metadata
                .get(model_name)
                .map(|m| m.ram_usage_mb)
                .unwrap_or(0);

            drop(metadata);
            drop(current_name);

            // Drop the engine (this releases all RAM)
            *self.current_model.write().await = None;
            *self.current_model_name.write().await = None;

            // Force garbage collection hint (Rust doesn't have explicit GC, but this helps)
            tokio::task::yield_now().await;

            info!("✅ Model unloaded, freed ~{} MB (~{:.1} GB) RAM",
                ram_freed_mb, ram_freed_mb as f32 / 1024.0);
        }

        Ok(())
    }

    /// Ensure model file exists locally, downloading if necessary
    async fn ensure_model_exists(&self, metadata: &ModelMetadata) -> Result<PathBuf> {
        let model_path = self.models_dir.join(&metadata.gguf_filename);

        // Check if file already exists
        if model_path.exists() {
            let file_size = tokio::fs::metadata(&model_path).await?.len();
            debug!("✅ Model file exists: {} ({} MB)",
                model_path.display(), file_size / 1024 / 1024);
            return Ok(model_path);
        }

        // Model doesn't exist, download from nginx
        warn!("⬇️  Model file not found locally, downloading from nginx...");
        info!("   📥 URL: {}", metadata.download_url);
        info!("   💾 Size: ~{} MB (~{:.1} GB)", metadata.ram_usage_mb, metadata.ram_usage_mb as f32 / 1024.0);
        info!("   🎯 Destination: {}", model_path.display());

        self.download_model(&metadata.download_url, &model_path).await?;

        Ok(model_path)
    }

    /// Download a model file from HTTP URL
    async fn download_model(&self, url: &str, dest_path: &PathBuf) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let start_time = std::time::Instant::now();

        // Create HTTP client
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600)) // 1 hour timeout for large files
            .build()?;

        info!("📡 Starting download...");

        // Send GET request
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to start download: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Download failed with status: {}. Is nginx serving the file at {}?",
                response.status(),
                url
            ));
        }

        let total_size = response
            .content_length()
            .ok_or_else(|| anyhow!("Missing Content-Length header"))?;

        info!("📦 Total size: {} MB ({} bytes)", total_size / 1024 / 1024, total_size);

        // Create destination file
        let mut file = tokio::fs::File::create(dest_path)
            .await
            .map_err(|e| anyhow!("Failed to create file: {}", e))?;

        // Stream download with progress
        let mut downloaded: u64 = 0;
        let mut last_progress_percent: u64 = 0;
        let mut stream = response.bytes_stream();

        use futures::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow!("Download error: {}", e))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| anyhow!("Write error: {}", e))?;

            downloaded += chunk.len() as u64;

            // Show progress every 5%
            let progress_percent = (downloaded * 100) / total_size;
            if progress_percent >= last_progress_percent + 5 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let speed_mbps = (downloaded as f64 / 1024.0 / 1024.0) / elapsed;
                info!("   {}% complete ({}/{} MB, {:.1} MB/s)",
                    progress_percent,
                    downloaded / 1024 / 1024,
                    total_size / 1024 / 1024,
                    speed_mbps
                );
                last_progress_percent = progress_percent;
            }
        }

        file.sync_all()
            .await
            .map_err(|e| anyhow!("Failed to sync file: {}", e))?;

        let elapsed = start_time.elapsed();
        let average_speed = (total_size as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64();

        info!("✅ Download complete!");
        info!("   ⏱️  Time: {:.1}s ({:.0}m {:.0}s)",
            elapsed.as_secs_f64(),
            elapsed.as_secs() / 60,
            elapsed.as_secs() % 60
        );
        info!("   📊 Average speed: {:.1} MB/s", average_speed);

        Ok(())
    }

    /// Get metadata for the currently loaded model
    pub async fn get_current_model_info(&self) -> Option<ModelMetadata> {
        let name = self.current_model_name.read().await;
        if let Some(model_name) = name.as_ref() {
            let metadata = self.model_metadata.read().await;
            return metadata.get(model_name).cloned();
        }
        None
    }

    /// List all available models (in cache)
    pub async fn list_cached_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();

        let mut entries = tokio::fs::read_dir(&self.models_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                    models.push(filename.to_string());
                }
            }
        }

        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_mistral_7b() {
        let meta = ModelMetadata::from_name("Mistral-7B-Instruct-v0.3", "http://localhost").unwrap();
        assert_eq!(meta.layer_count, 32);
        assert_eq!(meta.parameters_billions, 7.0);
        assert_eq!(meta.ram_usage_mb, 4100);
    }

    #[test]
    fn test_model_metadata_mistral_small_24b() {
        let meta = ModelMetadata::from_name("Mistral-Small-3.2-24B-Instruct", "http://localhost").unwrap();
        assert_eq!(meta.layer_count, 56);
        assert_eq!(meta.parameters_billions, 24.0);
        assert_eq!(meta.ram_usage_mb, 14336);
    }

    #[test]
    fn test_model_metadata_unknown() {
        let result = ModelMetadata::from_name("UnknownModel-42B", "http://localhost");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_model_manager_creation() {
        let temp_dir = std::env::temp_dir().join("q-test-models");
        let manager = ModelManager::new(temp_dir.clone()).await;
        assert!(manager.is_ok());
        // Cleanup
        let _ = tokio::fs::remove_dir_all(temp_dir).await;
    }
}
