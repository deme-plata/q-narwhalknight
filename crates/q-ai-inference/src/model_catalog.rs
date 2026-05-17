//! Model Catalog — Auto-download registry for supported GGUF models
//!
//! v9.6.0: Provides a catalog of known models with download URLs,
//! SHA256 checksums, and metadata. Supports auto-download on first use.
//!
//! This is separate from `model_registry.rs` (on-chain model hash verification).
//! The catalog is for local model management; the registry is for decentralized trust.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn, error};

/// Known model identifiers for quick selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelId {
    Mistral7B,
    GLM4Flash,
    Llama3_8B,
    MistralSmall24B,
}

impl ModelId {
    /// Parse from env var or CLI string
    pub fn from_str_loose(s: &str) -> Option<Self> {
        let lower = s.to_lowercase();
        if lower.contains("glm4") || lower.contains("glm-4") {
            Some(ModelId::GLM4Flash)
        } else if lower.contains("llama3") || lower.contains("llama-3") {
            Some(ModelId::Llama3_8B)
        } else if lower.contains("mistral-small") || lower.contains("mistral24b") {
            Some(ModelId::MistralSmall24B)
        } else if lower.contains("mistral") {
            Some(ModelId::Mistral7B)
        } else {
            None
        }
    }

    /// Human-readable display name
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelId::Mistral7B => "Mistral-7B-Instruct-v0.3",
            ModelId::GLM4Flash => "GLM-4-9B-Chat",
            ModelId::Llama3_8B => "Meta-Llama-3-8B-Instruct",
            ModelId::MistralSmall24B => "Mistral-Small-3.2-24B-Instruct",
        }
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Catalog entry for a downloadable model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub id: ModelId,
    pub gguf_filename: String,
    pub download_url: String,
    pub size_bytes: u64,
    pub sha256: Option<String>,
    pub layers: u32,
    pub context_length: u32,
    pub recommended_threads: u32,
    pub family: String,
}

/// The model catalog with known entries
pub struct ModelCatalog {
    models_dir: PathBuf,
}

impl ModelCatalog {
    pub fn new(models_dir: PathBuf) -> Self {
        Self { models_dir }
    }

    /// Default models directory
    pub fn default_dir() -> PathBuf {
        PathBuf::from("/opt/orobit/shared/q-narwhalknight/models")
    }

    /// Get catalog entry for a model ID
    pub fn get_entry(id: ModelId) -> CatalogEntry {
        match id {
            ModelId::Mistral7B => CatalogEntry {
                id,
                gguf_filename: "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf".to_string(),
                download_url: "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf".to_string(),
                size_bytes: 4_370_000_000,
                sha256: None,
                layers: 32,
                context_length: 32768,
                recommended_threads: 4,
                family: "mistral".to_string(),
            },
            ModelId::GLM4Flash => CatalogEntry {
                id,
                gguf_filename: "glm-4-9b-chat-Q4_K_M.gguf".to_string(),
                download_url: "https://huggingface.co/bartowski/glm-4-9b-chat-GGUF/resolve/main/glm-4-9b-chat-Q4_K_M.gguf".to_string(),
                size_bytes: 5_500_000_000,
                sha256: None,
                layers: 40,
                context_length: 131072,
                recommended_threads: 6,
                family: "glm".to_string(),
            },
            ModelId::Llama3_8B => CatalogEntry {
                id,
                gguf_filename: "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf".to_string(),
                download_url: "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf".to_string(),
                size_bytes: 4_920_000_000,
                sha256: None,
                layers: 32,
                context_length: 8192,
                recommended_threads: 4,
                family: "llama".to_string(),
            },
            ModelId::MistralSmall24B => CatalogEntry {
                id,
                gguf_filename: "Mistral-Small-3.2-24B-Instruct-2503-Q4_K_M.gguf".to_string(),
                download_url: "https://huggingface.co/bartowski/Mistral-Small-3.2-24B-Instruct-2503-GGUF/resolve/main/Mistral-Small-3.2-24B-Instruct-2503-Q4_K_M.gguf".to_string(),
                size_bytes: 14_400_000_000,
                sha256: None,
                layers: 40,
                context_length: 32768,
                recommended_threads: 8,
                family: "mistral".to_string(),
            },
        }
    }

    /// Resolve model path — returns existing path or downloads the model
    pub async fn resolve_model_path(&self, id: ModelId) -> Result<PathBuf> {
        let entry = Self::get_entry(id);
        let model_path = self.models_dir.join(&entry.gguf_filename);

        if model_path.exists() {
            let metadata = tokio::fs::metadata(&model_path).await?;
            if metadata.len() > 100_000_000 {
                // Looks like a valid model file (>100MB)
                info!("📦 Model {} found at {:?} ({:.1} GB)", entry.id, model_path, metadata.len() as f64 / 1e9);
                return Ok(model_path);
            }
            warn!("📦 Model file at {:?} seems too small ({} bytes), re-downloading", model_path, metadata.len());
        }

        info!("📥 Model {} not found locally, downloading ({:.1} GB)...", entry.id, entry.size_bytes as f64 / 1e9);
        self.download_model(&entry, &model_path).await?;
        Ok(model_path)
    }

    /// Download a model from HuggingFace with progress logging
    async fn download_model(&self, entry: &CatalogEntry, dest: &Path) -> Result<()> {
        tokio::fs::create_dir_all(&self.models_dir).await?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(7200)) // 2 hours for large models
            .build()?;

        let response = client.get(&entry.download_url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to download {}: {}", entry.gguf_filename, e))?;

        if !response.status().is_success() {
            return Err(anyhow!("Download failed with HTTP {}: {}", response.status(), entry.download_url));
        }

        let total_size = response.content_length().unwrap_or(entry.size_bytes);
        let tmp_path = dest.with_extension("gguf.downloading");
        let mut file = tokio::fs::File::create(&tmp_path).await?;
        let mut downloaded: u64 = 0;
        let mut last_log: u64 = 0;
        let mut hasher = Sha256::new();

        let mut stream = response.bytes_stream();
        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow!("Download stream error: {}", e))?;
            file.write_all(&chunk).await?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;

            // Log every 500MB
            if downloaded - last_log > 500_000_000 {
                let pct = (downloaded as f64 / total_size as f64 * 100.0).min(100.0);
                info!("📥 Downloading {} — {:.1}% ({:.1}/{:.1} GB)",
                    entry.gguf_filename, pct,
                    downloaded as f64 / 1e9, total_size as f64 / 1e9);
                last_log = downloaded;
            }
        }

        file.flush().await?;
        drop(file);

        // Verify SHA256 if available
        if let Some(ref expected_hash) = entry.sha256 {
            let actual_hash = hex::encode(hasher.finalize());
            if actual_hash != *expected_hash {
                tokio::fs::remove_file(&tmp_path).await.ok();
                return Err(anyhow!(
                    "SHA256 mismatch for {}: expected {}, got {}",
                    entry.gguf_filename, expected_hash, actual_hash
                ));
            }
            info!("✅ SHA256 verified for {}", entry.gguf_filename);
        }

        // Atomic rename
        tokio::fs::rename(&tmp_path, dest).await?;
        info!("✅ Model {} downloaded successfully ({:.1} GB)", entry.gguf_filename, downloaded as f64 / 1e9);
        Ok(())
    }

    /// Resolve model from Q_AI_MODEL env var, falling back to default
    pub async fn resolve_from_env(&self) -> Result<(ModelId, PathBuf)> {
        let model_str = std::env::var("Q_AI_MODEL").unwrap_or_else(|_| "mistral".to_string());

        let model_id = ModelId::from_str_loose(&model_str).unwrap_or_else(|| {
            warn!("Unknown model '{}', falling back to Mistral-7B", model_str);
            ModelId::Mistral7B
        });

        let path = self.resolve_model_path(model_id).await?;
        Ok((model_id, path))
    }

    /// List all available models (both downloaded and catalog entries)
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        let all_ids = [ModelId::Mistral7B, ModelId::GLM4Flash, ModelId::Llama3_8B, ModelId::MistralSmall24B];

        for id in &all_ids {
            let entry = Self::get_entry(*id);
            let path = self.models_dir.join(&entry.gguf_filename);
            let downloaded = path.exists();
            let file_size = if downloaded {
                tokio::fs::metadata(&path).await.map(|m| m.len()).unwrap_or(0)
            } else {
                0
            };

            models.push(ModelInfo {
                id: *id,
                name: id.display_name().to_string(),
                family: entry.family.clone(),
                filename: entry.gguf_filename,
                context_length: entry.context_length,
                layers: entry.layers,
                download_size_bytes: entry.size_bytes,
                downloaded,
                local_size_bytes: file_size,
            });
        }

        models
    }
}

/// Model info for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: ModelId,
    pub name: String,
    pub family: String,
    pub filename: String,
    pub context_length: u32,
    pub layers: u32,
    pub download_size_bytes: u64,
    pub downloaded: bool,
    pub local_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_from_str() {
        assert_eq!(ModelId::from_str_loose("glm4-flash"), Some(ModelId::GLM4Flash));
        assert_eq!(ModelId::from_str_loose("GLM-4"), Some(ModelId::GLM4Flash));
        assert_eq!(ModelId::from_str_loose("llama3"), Some(ModelId::Llama3_8B));
        assert_eq!(ModelId::from_str_loose("mistral"), Some(ModelId::Mistral7B));
        assert_eq!(ModelId::from_str_loose("mistral-small"), Some(ModelId::MistralSmall24B));
        assert_eq!(ModelId::from_str_loose("unknown"), None);
    }

    #[test]
    fn test_catalog_entries() {
        let glm4 = ModelCatalog::get_entry(ModelId::GLM4Flash);
        assert_eq!(glm4.context_length, 131072);
        assert_eq!(glm4.layers, 40);
        assert_eq!(glm4.family, "glm");

        let mistral = ModelCatalog::get_entry(ModelId::Mistral7B);
        assert_eq!(mistral.context_length, 32768);
        assert_eq!(mistral.layers, 32);
    }
}
