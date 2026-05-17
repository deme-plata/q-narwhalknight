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
pub mod engine_trait; // v5.1.0: Unified InferenceEngine trait for all backends (always compiled)
pub mod model_catalog; // v9.5.0: Model catalog for available inference models

// Feature-gated modules
#[cfg(feature = "legacy-mistralrs")]
pub mod mistralrs_engine;

// Stub module when legacy-mistralrs is not available - provides type compatibility
#[cfg(not(feature = "legacy-mistralrs"))]
pub mod mistralrs_engine {
    //! Stub module providing type compatibility when legacy-mistralrs feature is disabled.
    //! All constructors return errors at runtime.
    pub use crate::engine_trait::{GenerationStats, StreamEvent};
    use crate::privacy::PrivacyConfig;

    /// Stub MistralRsEngine - not functional without legacy-mistralrs feature
    pub struct MistralRsEngine;
    impl MistralRsEngine {
        pub async fn new(_model_path: &str) -> anyhow::Result<Self> {
            Err(anyhow::anyhow!("MistralRs engine not available: build with legacy-mistralrs feature"))
        }
        pub async fn with_config(_config: MistralRsConfig) -> anyhow::Result<Self> {
            Err(anyhow::anyhow!("MistralRs engine not available: build with legacy-mistralrs feature"))
        }
        pub async fn generate_stream<F, Fut>(
            &self, _prompt: &str, _max_tokens: usize, _callback: F,
        ) -> anyhow::Result<String>
        where
            F: FnMut(StreamEvent) -> Fut + Send,
            Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
        {
            Err(anyhow::anyhow!("MistralRs engine not available"))
        }
        pub fn engine_name(&self) -> &str { "mistralrs-stub" }
    }

    // Implement InferenceEngine trait for stub so it can be cast to Arc<dyn InferenceEngine>
    #[async_trait::async_trait]
    impl crate::engine_trait::InferenceEngine for MistralRsEngine {
        async fn generate_stream(
            &self, _prompt: &str, _max_tokens: usize,
            _tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
        ) -> anyhow::Result<String> {
            Err(anyhow::anyhow!("MistralRs engine not available"))
        }
        async fn generate(&self, _prompt: &str, _max_tokens: usize) -> anyhow::Result<String> {
            Err(anyhow::anyhow!("MistralRs engine not available"))
        }
        async fn get_stats(&self) -> GenerationStats {
            GenerationStats {
                tokens_generated: 0, prompt_tokens: 0, total_time_ms: 0.0,
                tokens_per_second: 0.0, time_to_first_token_ms: 0.0,
                kv_cache_hits: 0, kv_cache_misses: 0, speedup_factor: 0.0,
            }
        }
        fn engine_name(&self) -> &str { "mistralrs-stub" }
        fn model_hash(&self) -> [u8; 32] { [0u8; 32] }
    }

    /// Stub MistralRsConfig
    #[derive(Debug, Clone)]
    pub struct MistralRsConfig {
        pub model_path: String,
        pub enable_distributed: bool,
        pub privacy: PrivacyConfig,
        pub enable_kv_cache: bool,
        pub enable_pipeline: bool,
        pub enable_load_balancing: bool,
        pub temperature: f64,
        pub top_k: usize,
        pub top_p: f64,
        pub repeat_penalty: f64,
        pub max_seq_len: usize,
    }
    impl Default for MistralRsConfig {
        fn default() -> Self {
            Self {
                model_path: String::new(),
                enable_distributed: false,
                privacy: PrivacyConfig::default(),
                enable_kv_cache: true,
                enable_pipeline: false,
                enable_load_balancing: false,
                temperature: 0.7,
                top_k: 40,
                top_p: 0.95,
                repeat_penalty: 1.1,
                max_seq_len: 4096,
            }
        }
    }
}

pub mod model_manager;
pub mod chat_templates;
pub mod distributed_engine;
pub mod proof_of_inference;
pub mod worker_benchmark;
pub mod qwen3_vl;
pub mod weight_shard_manager; // v2.4.0: Tensor parallelism weight sharding
pub mod tensor_parallel_engine; // v2.4.0: Main tensor parallel inference engine
pub mod bitnet_integration; // v2.5.0: BitNet 1.58-bit integration for 16x faster tensor parallelism
pub mod bitnet_chat_engine; // v2.5.1: Native C++ BitNet chat engine via llama.cpp
#[cfg(feature = "llama-cpp")]
pub mod llama_cpp_engine; // v5.1.0: llama.cpp engine via llama-cpp-2 FFI (10-50x faster)
pub mod rpc_worker; // v5.1.0: llama.cpp RPC worker subprocess manager for pipeline parallelism

// v6.0.0: Decentralized AI inference — trustless verification + economic security
pub mod worker_registry;     // On-chain worker registration with staking
pub mod opml_verifier;        // opML dispute protocol (bisection + arbitration)
pub mod payment_settlement;   // Escrow, per-token payment, refund logic
pub mod model_registry;       // Model hash catalog + integrity verification

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

// mistralrs_engine types - always available (real or stub depending on feature)
pub use mistralrs_engine::{MistralRsEngine, MistralRsConfig};
// StreamEvent is now canonical in engine_trait, re-exported here for backward compat
pub use engine_trait::StreamEvent;

pub use model_manager::{ModelManager, ModelMetadata};
pub use chat_templates::{format_chat_prompt, format_conversation, parse_kimi_k2_reasoning};
pub use distributed_engine::DistributedMistralEngine;
pub use proof_of_inference::{
    ProofOfInferenceVerifier, InferenceProof, TokenProof, Challenge, ChallengeResponse,
    VerificationResult, SlashingRecord, ProofConfig,
};
pub use worker_benchmark::{
    WorkerBenchmarkVerifier, BenchmarkChallenge, BenchmarkResult, BenchmarkVerification,
    PerformanceThresholds, BenchmarkConfig,
};
pub use qwen3_vl::{
    Qwen3VLProcessor, Qwen3VLConfig, ImageAttachment,
};
pub use weight_shard_manager::{
    WeightShardManager, WeightShard, ShardConfig,
};
pub use tensor_parallel_engine::{
    TensorParallelEngine, TensorParallelConfig, TensorParallelStats,
    ShardedLayer, AllReduceRequest, AllReduceResponse,
};
pub use bitnet_integration::{
    BitNetTensorParallelEngine, BitNetTensorParallelConfig, BitNetStats as BitNetTPStats,
    BitNetPerformanceEstimate, calculate_bitnet_advantage,
};
pub use bitnet_chat_engine::{
    BitNetChatEngine, BitNetChatConfig, BitNetStreamEvent, BitNetGenerationStats,
};
// v5.1.0: Unified inference engine trait and llama.cpp engine
pub use engine_trait::InferenceEngine;
pub use engine_trait::{
    DeterministicConfig, DeterministicResult,
    compute_inference_commitment, compute_model_hash,
};
#[cfg(feature = "llama-cpp")]
pub use llama_cpp_engine::{LlamaCppEngine, LlamaCppConfig};
pub use rpc_worker::{RpcWorkerManager, RpcWorkerInfo, WorkerStatus as RpcWorkerStatus};

// v6.0.0: Decentralized AI inference exports
pub use worker_registry::{
    WorkerRegistry, WorkerRegistration, StakingTier, UnstakeRequest,
    WorkerRegistryStats, MIN_STAKE_BRONZE,
};
pub use opml_verifier::{
    OpMLVerifier, InferenceCommitment, DisputeState, DisputePhase,
    DisputeResolution, VerificationAction, OpMLMessage,
    VerificationStats as OpMLStats,
};
pub use payment_settlement::{
    PaymentSettlement, EscrowEntry, SettlementResult, PaymentTransfer,
    PaymentReason, PaymentStats, DEFAULT_PRICE_PER_TOKEN,
};
pub use model_registry::{
    ModelRegistry, ModelMetadata as AIModelMetadata, ModelRegistryStats,
};

// Re-export candle_core Device for use by q-network (ensures same version)
pub use candle_core::Device as CandleDevice;

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
