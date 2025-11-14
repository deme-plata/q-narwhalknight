//! Q-AI-Inference: Distributed AI Inference using Candle
//!
//! This crate provides distributed AI inference capabilities across the Q-NarwhalKnight network.
//! Users can contribute computing power to collectively run large language models like Mistral-7B.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              Q-NarwhalKnight Network                    │
//! │                  (libp2p + Gossipsub)                   │
//! └───────────────────┬─────────────────────────────────────┘
//!                     │
//!     ┌───────────────┼───────────────┐
//!     │               │               │
//! Node A          Node B          Node C
//! Layers 0-10   Layers 11-21   Layers 22-32
//!     │               │               │
//!     └───────────────┼───────────────┘
//!                     │
//!           Distributed Inference
//!        (Mistral-7B-Instruct-v0.3)
//! ```

pub mod types;
pub mod gossipsub_handler;
pub mod capability_detector;
pub mod coordinator_election;
pub mod layer_assignment;
pub mod model_loader;
pub mod inference_pipeline;
pub mod gguf_loader;
pub mod mistral_model;
pub mod privacy;
pub mod kv_cache;
pub mod simple_kv_cache;
pub mod pipeline_parallel;
pub mod load_balancer;
pub mod mistral_integration;
pub mod tokenizer;
pub mod gguf_tokenizer;
pub mod sampling;
pub mod generation;
pub mod distributed_cache;
pub mod mistralrs_engine;
pub mod model_manager;
pub mod chat_templates;
pub mod distributed_engine;

// Re-export commonly used types
pub use types::{
    AIMessage, DeviceCapability, InferenceRequest, InferenceResponse, LayerAssignment, TensorData,
};
pub use gossipsub_handler::{
    AIGossipsubHandler, AIHandlerStats, InferenceRequestState, RequestStatus,
};
pub use capability_detector::CapabilityDetector;
pub use coordinator_election::{CoordinatorElection, ElectionCandidate};
pub use layer_assignment::{LayerAssignmentCoordinator, LayerAssignmentPlan};
pub use model_loader::{ModelConfig, ModelLoader, LoadedModel, ModelCache};
pub use inference_pipeline::{
    InferencePipeline, InferenceRequest as PipelineInferenceRequest,
    InferenceResponse as PipelineInferenceResponse, InferenceStatus, LayerResult,
    PipelineStatistics,
};
pub use gguf_loader::{GGUFModelLoader, MistralLayerWeights, SpecialLayers};
pub use mistral_model::{MistralConfig, MistralLayer, RMSNorm, RotaryEmbedding};
pub use privacy::{
    ComputationProof, EncryptedTensor, PrivacyConfig, PrivacyLayer, PrivacyMetrics,
    TensorMetadata,
};
pub use kv_cache::{
    CacheStatistics, KVCacheCoordinator, KVCacheEntry, SequenceCache, compute_cache_key,
};
pub use pipeline_parallel::{
    PipelineExecutor, PipelineRequest, PipelineResponse, PipelineStage, PipelineStats,
};
pub use load_balancer::{
    LoadBalancer, LoadBalancerStats, LoadBalancingStrategy, NodeMetrics,
};
pub use mistral_integration::{
    GenerationStats, IntegrationConfig, MistralIntegration,
};
pub use tokenizer::GgufTokenizer;
pub use mistralrs_engine::{MistralRsEngine, MistralRsConfig, StreamEvent};
pub use model_manager::{ModelManager, ModelMetadata};
pub use chat_templates::{format_chat_prompt, format_conversation, parse_kimi_k2_reasoning};
pub use distributed_engine::DistributedMistralEngine;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_imports() {
        // Verify types are exported correctly
        let _capability = DeviceCapability::CPU {
            cores: 8,
            ram_gb: 16,
        };
    }
}
