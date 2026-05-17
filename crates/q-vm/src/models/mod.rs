//! Model registry for DAGKnight
use crate::contracts::{ModelRegistration, ResourceRequirements, ShardingCapability};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Model registry
pub struct ModelRegistry {
    models: RwLock<HashMap<String, ModelInfo>>,
}

/// Extended model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model registration info
    pub registration: ModelRegistration,
    /// Popularity score
    pub popularity: f64,
    /// Performance metrics
    pub performance: ModelPerformance,
    /// Quality metrics
    pub quality: ModelQuality,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Average tokens per second
    pub avg_tokens_per_second: f64,
    /// Average RAM usage in MB
    pub avg_ram_usage_mb: u64,
    /// Average GPU usage in MB
    pub avg_gpu_usage_mb: Option<u64>,
}

/// Model quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQuality {
    /// Average quality score (0-100)
    pub quality_score: f64,
    /// Number of ratings
    pub num_ratings: u64,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new model
    pub async fn register_model(&self, registration: ModelRegistration) -> bool {
        let mut models = self.models.write().await;

        if models.contains_key(&registration.model_id) {
            warn!("Model {} already registered", registration.model_id);
            return false;
        }

        let model_info = ModelInfo {
            registration: registration.clone(),
            popularity: 0.0,
            performance: ModelPerformance {
                avg_tokens_per_second: 0.0,
                avg_ram_usage_mb: registration.resources.min_memory_mb,
                avg_gpu_usage_mb: Some(registration.resources.min_gpu_memory_mb),
            },
            quality: ModelQuality {
                quality_score: 0.0,
                num_ratings: 0,
            },
        };

        models.insert(registration.model_id.clone(), model_info);
        info!("Registered model: {}", registration.model_id);

        true
    }

    /// Get model information
    pub async fn get_model(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }

    /// Update model performance
    pub async fn update_performance(
        &self,
        model_id: &str,
        tokens_per_second: f64,
        ram_usage_mb: u64,
        gpu_usage_mb: Option<u64>,
    ) -> bool {
        let mut models = self.models.write().await;

        if let Some(model) = models.get_mut(model_id) {
            // Update with exponential moving average
            let alpha = 0.1; // Weight for new observations

            model.performance.avg_tokens_per_second =
                (1.0 - alpha) * model.performance.avg_tokens_per_second + alpha * tokens_per_second;

            model.performance.avg_ram_usage_mb =
                ((1.0 - alpha) * model.performance.avg_ram_usage_mb as f64
                    + alpha * ram_usage_mb as f64) as u64;

            if let Some(gpu_usage) = gpu_usage_mb {
                model.performance.avg_gpu_usage_mb = Some(
                    ((1.0 - alpha) * model.performance.avg_gpu_usage_mb.unwrap_or(0) as f64
                        + alpha * gpu_usage as f64) as u64,
                );
            }

            // Increase popularity
            model.popularity += 0.1;
            return true;
        }

        warn!("Model {} not found for performance update", model_id);
        false
    }

    /// Update model quality
    pub async fn update_quality(&self, model_id: &str, quality_score: f64) -> bool {
        let mut models = self.models.write().await;

        if let Some(model) = models.get_mut(model_id) {
            // Update with weighted average
            let current_score = model.quality.quality_score;
            let num_ratings = model.quality.num_ratings;

            model.quality.quality_score =
                (current_score * num_ratings as f64 + quality_score) / (num_ratings as f64 + 1.0);

            model.quality.num_ratings += 1;

            return true;
        }

        warn!("Model {} not found for quality update", model_id);
        false
    }

    /// List all available models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }

    /// Find models that meet resource constraints
    pub async fn find_models_by_resources(
        &self,
        max_memory: u64,
        gpu_required: bool,
    ) -> Vec<ModelInfo> {
        let models = self.models.read().await;

        models
            .values()
            .filter(|m| {
                m.registration.resources.min_memory_mb <= max_memory
                    && (!gpu_required || m.registration.resources.min_gpu_memory_mb > 0)
            })
            .cloned()
            .collect()
    }

    /// Find models by sharding capability
    pub async fn find_models_by_sharding(&self, capability: ShardingCapability) -> Vec<ModelInfo> {
        let models = self.models.read().await;

        models
            .values()
            .filter(
                |m| match (capability.clone(), &m.registration.capabilities) {
                    (ShardingCapability::None, _) => true,
                    (
                        ShardingCapability::Horizontal,
                        ShardingCapability::Horizontal | ShardingCapability::Full,
                    ) => true,
                    (
                        ShardingCapability::Vertical,
                        ShardingCapability::Vertical | ShardingCapability::Full,
                    ) => true,
                    (ShardingCapability::Full, ShardingCapability::Full) => true,
                    _ => false,
                },
            )
            .cloned()
            .collect()
    }

    /// Initialize registry with default models
    pub async fn initialize_defaults(&self) {
        // Register some default models
        let models = [
            ModelRegistration {
                hash: [0; 32],
                owner: [0; 32],
                timestamp: 0,
                model_id: "llama2:7b".to_string(),
                description: "Meta's Llama 2 7B parameter model".to_string(),
                version: "2.0".to_string(),
                // memory_required: 16000,
                capabilities: ShardingCapability::Horizontal,
                resources: ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_mb: 16000,
                    min_gpu_memory_mb: 8000,
                    preferred_batch_size: 32,
                    // disk_space_mb: 14000,
                    // avg_exec_time_per_token_ms: 15.0,
                },
            },
            ModelRegistration {
                hash: [0; 32],
                owner: [0; 32],
                timestamp: 0,
                model_id: "deepseek-r1:1.5b".to_string(),
                description: "DeepSeek R1 1.5B parameter model".to_string(),
                version: "1.0".to_string(),
                // memory_required: 3000,
                capabilities: ShardingCapability::Full,
                resources: ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_mb: 4000,
                    min_gpu_memory_mb: 3000,
                    preferred_batch_size: 64,
                    // disk_space_mb: 3000,
                    // avg_exec_time_per_token_ms: 5.0,
                },
            },
            ModelRegistration {
                hash: [0; 32],
                owner: [0; 32],
                timestamp: 0,
                model_id: "mistral:7b".to_string(),
                description: "Mistral 7B parameter model".to_string(),
                version: "1.0".to_string(),
                // memory_required: 16000,
                capabilities: ShardingCapability::Vertical,
                resources: ResourceRequirements {
                    min_cpu_cores: 4,
                    min_memory_mb: 16000,
                    min_gpu_memory_mb: 7000,
                    preferred_batch_size: 32,
                    // disk_space_mb: 13500,
                    // avg_exec_time_per_token_ms: 12.0,
                },
            },
            ModelRegistration {
                hash: [0; 32],
                owner: [0; 32],
                timestamp: 0,
                model_id: "phi-2:3b".to_string(),
                description: "Microsoft's Phi-2 3B parameter model".to_string(),
                version: "2.0".to_string(),
                // memory_required: 6000,
                capabilities: ShardingCapability::Horizontal,
                resources: ResourceRequirements {
                    min_cpu_cores: 2,
                    min_memory_mb: 8000,
                    min_gpu_memory_mb: 4000,
                    preferred_batch_size: 48,
                    // disk_space_mb: 6000,
                    // avg_exec_time_per_token_ms: 8.0,
                },
            },
        ];

        for model in models {
            self.register_model(model).await;
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
