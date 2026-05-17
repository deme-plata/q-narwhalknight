//! On-Chain Model Registry for AI Inference Integrity
//!
//! v6.0.0: Maintains a registry of verified model hashes. Workers MUST register
//! their model hash before advertising capability. Users request inference for
//! specific model_hash (not model name) to prevent model substitution attacks.
//!
//! ## Integrity Guarantees
//!
//! - **No substitution**: Worker can't serve Q4 quantization when user paid for Q8
//! - **Version pinning**: Model hash locks exact weights, not just name
//! - **Audit trail**: On-chain record of who registered which model and when
//! - **Deduplication**: Same file = same hash, regardless of who uploads

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Metadata for a registered model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// SHA3-256 hash of the model file (primary key)
    pub model_hash: [u8; 32],
    /// Human-readable model name (e.g., "Mistral-7B-Instruct-v0.3-Q4_K_M")
    pub name: String,
    /// Model family (e.g., "mistral", "llama", "qwen")
    pub family: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Quantization type (e.g., "Q4_K_M", "Q8_0", "F16")
    pub quantization: String,
    /// Approximate parameter count (e.g., 7_000_000_000 for 7B)
    pub parameter_count: u64,
    /// Context window size in tokens
    pub context_length: u32,
    /// Who registered this model (wallet address)
    pub registered_by: [u8; 32],
    /// Block height when registered
    pub registered_at: u64,
    /// Number of workers currently serving this model
    pub active_workers: u32,
    /// Whether this model is verified (admin-approved for promoted visibility)
    pub verified: bool,
}

/// On-chain model registry
pub struct ModelRegistry {
    /// Registered models: model_hash -> metadata
    models: Arc<RwLock<HashMap<[u8; 32], ModelMetadata>>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        info!("📚 Initializing Model Registry for AI inference integrity");
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new model. Returns error if model hash already registered.
    pub async fn register_model(&self, metadata: ModelMetadata) -> Result<()> {
        let mut models = self.models.write().await;
        if models.contains_key(&metadata.model_hash) {
            return Err(anyhow!(
                "Model hash {} already registered",
                hex::encode(metadata.model_hash)
            ));
        }

        info!(
            "📦 Model registered: {} ({}, {:.1} GB, {})",
            metadata.name,
            metadata.quantization,
            metadata.size_bytes as f64 / 1e9,
            hex::encode(&metadata.model_hash[..8])
        );

        models.insert(metadata.model_hash, metadata);
        Ok(())
    }

    /// Get metadata for a model by hash
    pub async fn get_model(&self, hash: &[u8; 32]) -> Option<ModelMetadata> {
        self.models.read().await.get(hash).cloned()
    }

    /// Check if a model hash is registered
    pub async fn is_registered(&self, hash: &[u8; 32]) -> bool {
        self.models.read().await.contains_key(hash)
    }

    /// List all registered models
    pub async fn list_models(&self) -> Vec<ModelMetadata> {
        self.models.read().await.values().cloned().collect()
    }

    /// Search models by name (case-insensitive substring match)
    pub async fn search(&self, query: &str) -> Vec<ModelMetadata> {
        let query_lower = query.to_lowercase();
        self.models.read().await.values()
            .filter(|m| m.name.to_lowercase().contains(&query_lower))
            .cloned()
            .collect()
    }

    /// Update active worker count for a model
    pub async fn update_worker_count(&self, hash: &[u8; 32], count: u32) {
        if let Some(model) = self.models.write().await.get_mut(hash) {
            model.active_workers = count;
        }
    }

    /// Mark a model as verified (admin operation)
    pub async fn verify_model(&self, hash: &[u8; 32]) -> Result<()> {
        let mut models = self.models.write().await;
        let model = models.get_mut(hash)
            .ok_or_else(|| anyhow!("Model not found"))?;
        model.verified = true;
        Ok(())
    }

    /// Get registry statistics
    pub async fn get_stats(&self) -> ModelRegistryStats {
        let models = self.models.read().await;
        ModelRegistryStats {
            total_models: models.len(),
            verified_models: models.values().filter(|m| m.verified).count(),
            total_workers_serving: models.values().map(|m| m.active_workers as u64).sum(),
            models_by_family: {
                let mut families: HashMap<String, usize> = HashMap::new();
                for m in models.values() {
                    *families.entry(m.family.clone()).or_insert(0) += 1;
                }
                families
            },
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryStats {
    pub total_models: usize,
    pub verified_models: usize,
    pub total_workers_serving: u64,
    pub models_by_family: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_lookup() {
        let registry = ModelRegistry::new();
        let hash = [42u8; 32];

        let metadata = ModelMetadata {
            model_hash: hash,
            name: "Mistral-7B-Q4_K_M".into(),
            family: "mistral".into(),
            size_bytes: 4_000_000_000,
            quantization: "Q4_K_M".into(),
            parameter_count: 7_000_000_000,
            context_length: 4096,
            registered_by: [1u8; 32],
            registered_at: 100,
            active_workers: 0,
            verified: false,
        };

        registry.register_model(metadata.clone()).await.unwrap();
        assert!(registry.is_registered(&hash).await);

        let found = registry.get_model(&hash).await.unwrap();
        assert_eq!(found.name, "Mistral-7B-Q4_K_M");
    }

    #[tokio::test]
    async fn test_no_duplicate_registration() {
        let registry = ModelRegistry::new();
        let hash = [42u8; 32];

        let metadata = ModelMetadata {
            model_hash: hash,
            name: "test".into(),
            family: "test".into(),
            size_bytes: 100,
            quantization: "Q4".into(),
            parameter_count: 1000,
            context_length: 512,
            registered_by: [1u8; 32],
            registered_at: 100,
            active_workers: 0,
            verified: false,
        };

        registry.register_model(metadata.clone()).await.unwrap();
        assert!(registry.register_model(metadata).await.is_err());
    }

    #[tokio::test]
    async fn test_search() {
        let registry = ModelRegistry::new();

        for (i, name) in ["Mistral-7B", "Llama-3-8B", "Mistral-Nemo"].iter().enumerate() {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            registry.register_model(ModelMetadata {
                model_hash: hash,
                name: name.to_string(),
                family: if name.contains("Mistral") { "mistral" } else { "llama" }.into(),
                size_bytes: 100,
                quantization: "Q4".into(),
                parameter_count: 1000,
                context_length: 512,
                registered_by: [1u8; 32],
                registered_at: 100,
                active_workers: 0,
                verified: false,
            }).await.unwrap();
        }

        let results = registry.search("mistral").await;
        assert_eq!(results.len(), 2);
    }
}
