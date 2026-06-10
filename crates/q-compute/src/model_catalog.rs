#![allow(dead_code, non_camel_case_types)]
//! Model Catalog & Hot-Swap — Dynamic Model Management (Issue #025)
//!
//! Thread-safe registry of AI models available to the compute layer.
//! Tracks model metadata (family, quantization, size, tier), loaded state,
//! and provides hot-swap operations to transition between models without
//! restarting the node.
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │                      ModelCatalog                            │
//! │  Arc<RwLock<HashMap<model_id, ModelInfo>>>                   │
//! │                                                              │
//! │  register / unregister / list / get                          │
//! │  mark_loaded / mark_unloaded                                 │
//! │  find_by_family / find_by_tier                               │
//! │  hot_swap(old, new)                                          │
//! │  total_loaded_size_mb / can_fit_model                        │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Hot-Swap Flow
//!
//! 1. `hot_swap(old, new)` validates both model IDs exist in the catalog
//! 2. Atomically marks old model as unloaded and new model as loaded
//! 3. Caller is responsible for actual engine teardown / initialization
//!    (this layer only tracks metadata state)

use crate::inference_pool::ModelTier;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════════════════════════════

/// Model family -- the base architecture of the model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
    Llama,
    Mistral,
    Phi,
    Qwen,
    Custom(String),
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelFamily::Llama => write!(f, "llama"),
            ModelFamily::Mistral => write!(f, "mistral"),
            ModelFamily::Phi => write!(f, "phi"),
            ModelFamily::Qwen => write!(f, "qwen"),
            ModelFamily::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

impl std::str::FromStr for ModelFamily {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "llama" => Ok(ModelFamily::Llama),
            "mistral" => Ok(ModelFamily::Mistral),
            "phi" => Ok(ModelFamily::Phi),
            "qwen" => Ok(ModelFamily::Qwen),
            other => Ok(ModelFamily::Custom(other.to_string())),
        }
    }
}

/// GGUF quantization level -- determines model size vs quality trade-off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Quantization {
    /// 4-bit quantization, basic
    Q4_0,
    /// 4-bit quantization, K-means, medium quality
    Q4_K_M,
    /// 5-bit quantization, K-means, medium quality
    Q5_K_M,
    /// 8-bit quantization, highest quality quantized
    Q8_0,
    /// 16-bit floating point (half precision)
    F16,
    /// 32-bit floating point (full precision)
    F32,
}

impl Quantization {
    /// Human-readable label for this quantization.
    pub fn label(&self) -> &'static str {
        match self {
            Quantization::Q4_0 => "Q4_0",
            Quantization::Q4_K_M => "Q4_K_M",
            Quantization::Q5_K_M => "Q5_K_M",
            Quantization::Q8_0 => "Q8_0",
            Quantization::F16 => "F16",
            Quantization::F32 => "F32",
        }
    }

    /// Approximate bytes-per-parameter estimate for VRAM sizing.
    pub fn bytes_per_param(&self) -> f64 {
        match self {
            Quantization::Q4_0 => 0.5,    // 4 bits
            Quantization::Q4_K_M => 0.5625, // ~4.5 bits
            Quantization::Q5_K_M => 0.6875, // ~5.5 bits
            Quantization::Q8_0 => 1.0,     // 8 bits
            Quantization::F16 => 2.0,      // 16 bits
            Quantization::F32 => 4.0,      // 32 bits
        }
    }
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl std::str::FromStr for Quantization {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().replace('-', "_").as_str() {
            "Q4_0" | "Q4" => Ok(Quantization::Q4_0),
            "Q4_K_M" | "Q4KM" => Ok(Quantization::Q4_K_M),
            "Q5_K_M" | "Q5KM" => Ok(Quantization::Q5_K_M),
            "Q8_0" | "Q8" => Ok(Quantization::Q8_0),
            "F16" | "FP16" | "HALF" => Ok(Quantization::F16),
            "F32" | "FP32" | "FLOAT" | "FULL" => Ok(Quantization::F32),
            _ => Err(format!(
                "Unknown quantization: '{}'. Use: Q4_0, Q4_K_M, Q5_K_M, Q8_0, F16, F32",
                s
            )),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// ModelInfo
// ═══════════════════════════════════════════════════════════════════

/// Metadata for a single model registered in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique identifier (e.g., "mistral-7b-q4km")
    pub model_id: String,
    /// Human-readable name (e.g., "Mistral 7B Instruct v0.3")
    pub name: String,
    /// Model architecture family
    pub family: ModelFamily,
    /// On-disk / VRAM size in megabytes
    pub size_mb: u64,
    /// Quantization level
    pub quantization: Quantization,
    /// Maximum supported context window length (tokens)
    pub max_context_length: u32,
    /// Pricing / compute tier (from inference_pool)
    pub tier: ModelTier,
    /// Whether the model is currently loaded into memory / VRAM
    pub loaded: bool,
    /// Time taken to load the model (milliseconds), 0 if never loaded
    pub load_time_ms: u64,
}

// ═══════════════════════════════════════════════════════════════════
// ModelCatalog
// ═══════════════════════════════════════════════════════════════════

/// Thread-safe catalog of available AI models.
///
/// Provides registration, lookup, filtering, and hot-swap operations.
/// Interior mutability via `Arc<RwLock<...>>` allows sharing across
/// async tasks and threads without external synchronization.
#[derive(Clone)]
pub struct ModelCatalog {
    models: Arc<RwLock<HashMap<String, ModelInfo>>>,
}

impl ModelCatalog {
    /// Create a new empty catalog.
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a model in the catalog.
    ///
    /// If a model with the same `model_id` already exists, it is replaced
    /// (allowing metadata updates / re-registration).
    pub fn register_model(&self, info: ModelInfo) {
        let mut models = self.models.write();
        models.insert(info.model_id.clone(), info);
    }

    /// Remove a model from the catalog by ID.
    ///
    /// Returns `true` if the model was found and removed, `false` otherwise.
    pub fn unregister_model(&self, model_id: &str) -> bool {
        let mut models = self.models.write();
        models.remove(model_id).is_some()
    }

    /// List all registered models.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read();
        models.values().cloned().collect()
    }

    /// Get a single model by ID.
    pub fn get_model(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read();
        models.get(model_id).cloned()
    }

    /// Get all models that are currently loaded.
    pub fn get_loaded_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read();
        models.values().filter(|m| m.loaded).cloned().collect()
    }

    /// Mark a model as loaded. Returns `true` if the model exists in the catalog.
    pub fn mark_loaded(&self, model_id: &str) -> bool {
        let mut models = self.models.write();
        if let Some(model) = models.get_mut(model_id) {
            model.loaded = true;
            true
        } else {
            false
        }
    }

    /// Mark a model as unloaded. Returns `true` if the model exists in the catalog.
    pub fn mark_unloaded(&self, model_id: &str) -> bool {
        let mut models = self.models.write();
        if let Some(model) = models.get_mut(model_id) {
            model.loaded = false;
            true
        } else {
            false
        }
    }

    /// Find all models belonging to a specific family.
    pub fn find_by_family(&self, family: &ModelFamily) -> Vec<ModelInfo> {
        let models = self.models.read();
        models
            .values()
            .filter(|m| &m.family == family)
            .cloned()
            .collect()
    }

    /// Find all models in a specific pricing / compute tier.
    pub fn find_by_tier(&self, tier: ModelTier) -> Vec<ModelInfo> {
        let models = self.models.read();
        models
            .values()
            .filter(|m| m.tier == tier)
            .cloned()
            .collect()
    }

    /// Hot-swap: atomically mark the old model as unloaded and the new model
    /// as loaded.
    ///
    /// Both model IDs must be registered in the catalog. If either is missing,
    /// no state is changed and an error is returned.
    pub fn hot_swap(&self, old_model_id: &str, new_model_id: &str) -> Result<(), String> {
        let mut models = self.models.write();

        // Validate both models exist before mutating anything
        if !models.contains_key(old_model_id) {
            return Err(format!(
                "Old model '{}' not found in catalog",
                old_model_id
            ));
        }
        if !models.contains_key(new_model_id) {
            return Err(format!(
                "New model '{}' not found in catalog",
                new_model_id
            ));
        }

        // Atomically swap loaded state (single write-lock scope)
        if let Some(old) = models.get_mut(old_model_id) {
            old.loaded = false;
        }
        if let Some(new) = models.get_mut(new_model_id) {
            new.loaded = true;
        }

        Ok(())
    }

    /// Total VRAM / memory consumed by all currently loaded models (in MB).
    pub fn total_loaded_size_mb(&self) -> u64 {
        let models = self.models.read();
        models.values().filter(|m| m.loaded).map(|m| m.size_mb).sum()
    }

    /// Check whether a model can fit in the given available VRAM budget.
    ///
    /// Returns `true` if the model's `size_mb` is at most `available_vram_mb`.
    /// Returns `false` if the model is not registered or does not fit.
    pub fn can_fit_model(&self, model_id: &str, available_vram_mb: u64) -> bool {
        let models = self.models.read();
        match models.get(model_id) {
            Some(model) => model.size_mb <= available_vram_mb,
            None => false,
        }
    }

    /// Number of models in the catalog.
    pub fn len(&self) -> usize {
        self.models.read().len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.models.read().is_empty()
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a test ModelInfo with sensible defaults.
    fn make_model(id: &str, family: ModelFamily, size_mb: u64, tier: ModelTier) -> ModelInfo {
        ModelInfo {
            model_id: id.to_string(),
            name: format!("Test Model {}", id),
            family,
            size_mb,
            quantization: Quantization::Q4_K_M,
            max_context_length: 4096,
            tier,
            loaded: false,
            load_time_ms: 0,
        }
    }

    // ---------------------------------------------------------------
    // 1. Register and retrieve model
    // ---------------------------------------------------------------
    #[test]
    fn test_register_and_retrieve_model() {
        let catalog = ModelCatalog::new();
        let model = make_model("llama3-8b", ModelFamily::Llama, 4500, ModelTier::Medium);
        catalog.register_model(model);

        let retrieved = catalog.get_model("llama3-8b");
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.model_id, "llama3-8b");
        assert_eq!(retrieved.name, "Test Model llama3-8b");
        assert_eq!(retrieved.family, ModelFamily::Llama);
        assert_eq!(retrieved.size_mb, 4500);
        assert_eq!(retrieved.tier, ModelTier::Medium);
        assert!(!retrieved.loaded);
    }

    // ---------------------------------------------------------------
    // 2. List and filter by family
    // ---------------------------------------------------------------
    #[test]
    fn test_list_and_filter_by_family() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("llama3-8b", ModelFamily::Llama, 4500, ModelTier::Medium));
        catalog.register_model(make_model("mistral-7b", ModelFamily::Mistral, 4000, ModelTier::Medium));
        catalog.register_model(make_model("llama3-70b", ModelFamily::Llama, 38000, ModelTier::Large));
        catalog.register_model(make_model("phi-3-mini", ModelFamily::Phi, 2300, ModelTier::Small));

        let all = catalog.list_models();
        assert_eq!(all.len(), 4);

        let llamas = catalog.find_by_family(&ModelFamily::Llama);
        assert_eq!(llamas.len(), 2);
        assert!(llamas.iter().all(|m| m.family == ModelFamily::Llama));

        let mistrals = catalog.find_by_family(&ModelFamily::Mistral);
        assert_eq!(mistrals.len(), 1);
        assert_eq!(mistrals[0].model_id, "mistral-7b");

        let qwens = catalog.find_by_family(&ModelFamily::Qwen);
        assert!(qwens.is_empty());
    }

    // ---------------------------------------------------------------
    // 3. Filter by tier
    // ---------------------------------------------------------------
    #[test]
    fn test_filter_by_tier() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("phi-3-mini", ModelFamily::Phi, 2300, ModelTier::Small));
        catalog.register_model(make_model("llama3-8b", ModelFamily::Llama, 4500, ModelTier::Medium));
        catalog.register_model(make_model("mistral-7b", ModelFamily::Mistral, 4000, ModelTier::Medium));
        catalog.register_model(make_model("llama3-70b", ModelFamily::Llama, 38000, ModelTier::Large));

        let mediums = catalog.find_by_tier(ModelTier::Medium);
        assert_eq!(mediums.len(), 2);
        assert!(mediums.iter().all(|m| m.tier == ModelTier::Medium));

        let smalls = catalog.find_by_tier(ModelTier::Small);
        assert_eq!(smalls.len(), 1);
        assert_eq!(smalls[0].model_id, "phi-3-mini");

        let xls = catalog.find_by_tier(ModelTier::XL);
        assert!(xls.is_empty());
    }

    // ---------------------------------------------------------------
    // 4. Hot-swap between models
    // ---------------------------------------------------------------
    #[test]
    fn test_hot_swap_between_models() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("mistral-7b", ModelFamily::Mistral, 4000, ModelTier::Medium));
        catalog.register_model(make_model("llama3-8b", ModelFamily::Llama, 4500, ModelTier::Medium));

        // Mark mistral as loaded first
        catalog.mark_loaded("mistral-7b");
        assert!(catalog.get_model("mistral-7b").unwrap().loaded);
        assert!(!catalog.get_model("llama3-8b").unwrap().loaded);

        // Hot-swap: mistral -> llama
        let result = catalog.hot_swap("mistral-7b", "llama3-8b");
        assert!(result.is_ok());

        // mistral should be unloaded, llama should be loaded
        assert!(!catalog.get_model("mistral-7b").unwrap().loaded);
        assert!(catalog.get_model("llama3-8b").unwrap().loaded);
    }

    // ---------------------------------------------------------------
    // 5. Cannot hot-swap to unregistered model
    // ---------------------------------------------------------------
    #[test]
    fn test_hot_swap_to_unregistered_model_fails() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("mistral-7b", ModelFamily::Mistral, 4000, ModelTier::Medium));
        catalog.mark_loaded("mistral-7b");

        // Try to swap to a model that does not exist
        let result = catalog.hot_swap("mistral-7b", "nonexistent-model");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nonexistent-model"));

        // Original model should still be loaded (no partial mutation)
        assert!(catalog.get_model("mistral-7b").unwrap().loaded);
    }

    // ---------------------------------------------------------------
    // 6. Cannot hot-swap from unregistered model
    // ---------------------------------------------------------------
    #[test]
    fn test_hot_swap_from_unregistered_model_fails() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("llama3-8b", ModelFamily::Llama, 4500, ModelTier::Medium));

        let result = catalog.hot_swap("nonexistent-old", "llama3-8b");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nonexistent-old"));

        // Target model should remain unloaded
        assert!(!catalog.get_model("llama3-8b").unwrap().loaded);
    }

    // ---------------------------------------------------------------
    // 7. Mark loaded / unloaded state changes
    // ---------------------------------------------------------------
    #[test]
    fn test_mark_loaded_unloaded_state_changes() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("phi-3-mini", ModelFamily::Phi, 2300, ModelTier::Small));

        // Initially unloaded
        assert!(!catalog.get_model("phi-3-mini").unwrap().loaded);

        // Mark loaded
        assert!(catalog.mark_loaded("phi-3-mini"));
        assert!(catalog.get_model("phi-3-mini").unwrap().loaded);

        // Mark loaded again (idempotent)
        assert!(catalog.mark_loaded("phi-3-mini"));
        assert!(catalog.get_model("phi-3-mini").unwrap().loaded);

        // Mark unloaded
        assert!(catalog.mark_unloaded("phi-3-mini"));
        assert!(!catalog.get_model("phi-3-mini").unwrap().loaded);

        // Mark on nonexistent model returns false
        assert!(!catalog.mark_loaded("ghost-model"));
        assert!(!catalog.mark_unloaded("ghost-model"));
    }

    // ---------------------------------------------------------------
    // 8. Total loaded size calculation
    // ---------------------------------------------------------------
    #[test]
    fn test_total_loaded_size_mb() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("a", ModelFamily::Llama, 4000, ModelTier::Medium));
        catalog.register_model(make_model("b", ModelFamily::Mistral, 3000, ModelTier::Medium));
        catalog.register_model(make_model("c", ModelFamily::Phi, 2000, ModelTier::Small));

        // Nothing loaded yet
        assert_eq!(catalog.total_loaded_size_mb(), 0);

        // Load two models
        catalog.mark_loaded("a");
        catalog.mark_loaded("c");
        assert_eq!(catalog.total_loaded_size_mb(), 6000); // 4000 + 2000

        // Load the third
        catalog.mark_loaded("b");
        assert_eq!(catalog.total_loaded_size_mb(), 9000); // 4000 + 3000 + 2000

        // Unload one
        catalog.mark_unloaded("a");
        assert_eq!(catalog.total_loaded_size_mb(), 5000); // 3000 + 2000
    }

    // ---------------------------------------------------------------
    // 9. Can fit model (VRAM check)
    // ---------------------------------------------------------------
    #[test]
    fn test_can_fit_model_vram_check() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("small", ModelFamily::Phi, 2000, ModelTier::Small));
        catalog.register_model(make_model("big", ModelFamily::Llama, 40000, ModelTier::Large));

        // Small model fits in 4GB VRAM
        assert!(catalog.can_fit_model("small", 4000));
        // Small model fits exactly
        assert!(catalog.can_fit_model("small", 2000));
        // Small model does NOT fit in 1GB
        assert!(!catalog.can_fit_model("small", 1000));

        // Big model needs 40GB
        assert!(catalog.can_fit_model("big", 48000));
        assert!(!catalog.can_fit_model("big", 24000));

        // Unregistered model always returns false
        assert!(!catalog.can_fit_model("nonexistent", 999999));
    }

    // ---------------------------------------------------------------
    // 10. Duplicate registration (update)
    // ---------------------------------------------------------------
    #[test]
    fn test_duplicate_registration_updates() {
        let catalog = ModelCatalog::new();

        // Register initial model
        let model_v1 = ModelInfo {
            model_id: "mistral-7b".to_string(),
            name: "Mistral 7B v1".to_string(),
            family: ModelFamily::Mistral,
            size_mb: 4000,
            quantization: Quantization::Q4_0,
            max_context_length: 4096,
            tier: ModelTier::Medium,
            loaded: false,
            load_time_ms: 0,
        };
        catalog.register_model(model_v1);

        // Re-register with updated metadata
        let model_v2 = ModelInfo {
            model_id: "mistral-7b".to_string(),
            name: "Mistral 7B v2 (updated)".to_string(),
            family: ModelFamily::Mistral,
            size_mb: 4200,
            quantization: Quantization::Q4_K_M,
            max_context_length: 8192,
            tier: ModelTier::Medium,
            loaded: false,
            load_time_ms: 0,
        };
        catalog.register_model(model_v2);

        // Should still be 1 model, with updated fields
        let all = catalog.list_models();
        assert_eq!(all.len(), 1);

        let updated = catalog.get_model("mistral-7b").unwrap();
        assert_eq!(updated.name, "Mistral 7B v2 (updated)");
        assert_eq!(updated.size_mb, 4200);
        assert_eq!(updated.quantization, Quantization::Q4_K_M);
        assert_eq!(updated.max_context_length, 8192);
    }

    // ---------------------------------------------------------------
    // 11. Unregister removes model
    // ---------------------------------------------------------------
    #[test]
    fn test_unregister_removes_model() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("llama3-8b", ModelFamily::Llama, 4500, ModelTier::Medium));
        catalog.register_model(make_model("mistral-7b", ModelFamily::Mistral, 4000, ModelTier::Medium));

        assert_eq!(catalog.list_models().len(), 2);
        assert!(catalog.get_model("llama3-8b").is_some());

        // Unregister
        assert!(catalog.unregister_model("llama3-8b"));
        assert_eq!(catalog.list_models().len(), 1);
        assert!(catalog.get_model("llama3-8b").is_none());

        // Unregister nonexistent returns false
        assert!(!catalog.unregister_model("llama3-8b"));
        assert!(!catalog.unregister_model("ghost"));
    }

    // ---------------------------------------------------------------
    // 12. Find loaded models only
    // ---------------------------------------------------------------
    #[test]
    fn test_get_loaded_models_only() {
        let catalog = ModelCatalog::new();
        catalog.register_model(make_model("a", ModelFamily::Llama, 4000, ModelTier::Medium));
        catalog.register_model(make_model("b", ModelFamily::Mistral, 3000, ModelTier::Medium));
        catalog.register_model(make_model("c", ModelFamily::Phi, 2000, ModelTier::Small));

        // None loaded
        assert!(catalog.get_loaded_models().is_empty());

        // Load two
        catalog.mark_loaded("a");
        catalog.mark_loaded("c");
        let loaded = catalog.get_loaded_models();
        assert_eq!(loaded.len(), 2);
        let ids: Vec<&str> = loaded.iter().map(|m| m.model_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"c"));
        assert!(!ids.contains(&"b"));
    }

    // ---------------------------------------------------------------
    // 13. Enum Display / FromStr round-trips
    // ---------------------------------------------------------------
    #[test]
    fn test_model_family_display_and_parse() {
        assert_eq!(format!("{}", ModelFamily::Llama), "llama");
        assert_eq!(format!("{}", ModelFamily::Mistral), "mistral");
        assert_eq!(format!("{}", ModelFamily::Phi), "phi");
        assert_eq!(format!("{}", ModelFamily::Qwen), "qwen");
        assert_eq!(
            format!("{}", ModelFamily::Custom("deepseek".to_string())),
            "custom(deepseek)"
        );

        assert_eq!("llama".parse::<ModelFamily>().unwrap(), ModelFamily::Llama);
        assert_eq!("MISTRAL".parse::<ModelFamily>().unwrap(), ModelFamily::Mistral);
        assert_eq!("Phi".parse::<ModelFamily>().unwrap(), ModelFamily::Phi);
        assert_eq!("qwen".parse::<ModelFamily>().unwrap(), ModelFamily::Qwen);
        // Unknown strings become Custom
        assert_eq!(
            "deepseek".parse::<ModelFamily>().unwrap(),
            ModelFamily::Custom("deepseek".to_string())
        );
    }

    #[test]
    fn test_quantization_display_and_parse() {
        assert_eq!(Quantization::Q4_0.label(), "Q4_0");
        assert_eq!(Quantization::Q4_K_M.label(), "Q4_K_M");
        assert_eq!(format!("{}", Quantization::F16), "F16");
        assert_eq!(format!("{}", Quantization::F32), "F32");

        assert_eq!("Q4_0".parse::<Quantization>().unwrap(), Quantization::Q4_0);
        assert_eq!("q4".parse::<Quantization>().unwrap(), Quantization::Q4_0);
        assert_eq!("Q4_K_M".parse::<Quantization>().unwrap(), Quantization::Q4_K_M);
        assert_eq!("q5_k_m".parse::<Quantization>().unwrap(), Quantization::Q5_K_M);
        assert_eq!("Q8_0".parse::<Quantization>().unwrap(), Quantization::Q8_0);
        assert_eq!("f16".parse::<Quantization>().unwrap(), Quantization::F16);
        assert_eq!("fp16".parse::<Quantization>().unwrap(), Quantization::F16);
        assert_eq!("f32".parse::<Quantization>().unwrap(), Quantization::F32);
        assert_eq!("fp32".parse::<Quantization>().unwrap(), Quantization::F32);

        assert!("garbage".parse::<Quantization>().is_err());
    }

    // ---------------------------------------------------------------
    // 14. Serde round-trip for ModelInfo
    // ---------------------------------------------------------------
    #[test]
    fn test_model_info_serde_roundtrip() {
        let model = ModelInfo {
            model_id: "qwen2-7b".to_string(),
            name: "Qwen2 7B Chat".to_string(),
            family: ModelFamily::Qwen,
            size_mb: 4100,
            quantization: Quantization::Q5_K_M,
            max_context_length: 32768,
            tier: ModelTier::Medium,
            loaded: true,
            load_time_ms: 1234,
        };

        let json = serde_json::to_string(&model).unwrap();
        let parsed: ModelInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.model_id, "qwen2-7b");
        assert_eq!(parsed.family, ModelFamily::Qwen);
        assert_eq!(parsed.quantization, Quantization::Q5_K_M);
        assert_eq!(parsed.tier, ModelTier::Medium);
        assert!(parsed.loaded);
        assert_eq!(parsed.load_time_ms, 1234);
    }

    // ---------------------------------------------------------------
    // 15. Empty catalog edge cases
    // ---------------------------------------------------------------
    #[test]
    fn test_empty_catalog_operations() {
        let catalog = ModelCatalog::new();

        assert_eq!(catalog.list_models().len(), 0);
        assert!(catalog.get_loaded_models().is_empty());
        assert_eq!(catalog.total_loaded_size_mb(), 0);
        assert!(catalog.get_model("anything").is_none());
        assert!(!catalog.unregister_model("anything"));
        assert!(!catalog.mark_loaded("anything"));
        assert!(!catalog.mark_unloaded("anything"));
        assert!(!catalog.can_fit_model("anything", 999999));
        assert!(catalog.find_by_family(&ModelFamily::Llama).is_empty());
        assert!(catalog.find_by_tier(ModelTier::Small).is_empty());
        assert!(catalog.is_empty());
        assert_eq!(catalog.len(), 0);
    }

    // ---------------------------------------------------------------
    // 16. Quantization bytes_per_param ordering
    // ---------------------------------------------------------------
    #[test]
    fn test_quantization_bytes_per_param_ordering() {
        assert!(Quantization::Q4_0.bytes_per_param() < Quantization::Q4_K_M.bytes_per_param());
        assert!(Quantization::Q4_K_M.bytes_per_param() < Quantization::Q5_K_M.bytes_per_param());
        assert!(Quantization::Q5_K_M.bytes_per_param() < Quantization::Q8_0.bytes_per_param());
        assert!(Quantization::Q8_0.bytes_per_param() < Quantization::F16.bytes_per_param());
        assert!(Quantization::F16.bytes_per_param() < Quantization::F32.bytes_per_param());
    }
}
