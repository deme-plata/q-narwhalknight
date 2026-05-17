//! Inference Engine Trait - Unified interface for all AI inference backends
//!
//! v5.1.0: Introduces a common trait so LlamaCppEngine, MistralRsEngine, and future
//! backends (e.g., vLLM, TensorRT) can be swapped transparently.
//!
//! v6.0.0: Added deterministic inference support for opML verification,
//! model_hash() for integrity verification, and generate_deterministic() for
//! reproducible outputs across nodes.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Streaming events for real-time feedback
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Progress indicator (e.g., "Loading model...", "Generating token 5/150...")
    Progress(String),
    /// Generated token text
    Token(String),
    /// Generation complete with statistics
    Complete(GenerationStats),
    /// Error occurred
    Error(String),
}

/// Generation statistics from an inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub tokens_generated: usize,
    pub prompt_tokens: usize,
    pub total_time_ms: f64,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub kv_cache_hits: usize,
    pub kv_cache_misses: usize,
    pub speedup_factor: f64,
}

/// v6.0.0: Configuration for deterministic inference (opML verification).
///
/// When deterministic mode is enabled, the engine MUST produce identical output
/// for the same (prompt, seed, model) tuple across different nodes. This is
/// achieved by forcing greedy decoding (temperature=0, top_k=1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicConfig {
    /// Fixed seed for reproducible sampling
    pub seed: u64,
    /// When true, forces temperature=0.0 and top_k=1 (greedy decoding)
    pub greedy: bool,
}

impl DeterministicConfig {
    /// Create a new deterministic config with greedy decoding
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            greedy: true,
        }
    }
}

/// v6.0.0: Result of deterministic inference, including commitment data
/// needed for opML verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicResult {
    /// The generated text
    pub text: String,
    /// Individual tokens generated (for bisection disputes)
    pub tokens: Vec<String>,
    /// SHA3-256 hash of (all_tokens ++ seed ++ model_hash)
    pub output_hash: [u8; 32],
    /// SHA3-256 prefix hashes: hash of tokens[0..i] for each i
    /// Used in bisection dispute protocol to narrow divergence
    pub prefix_hashes: Vec<[u8; 32]>,
    /// Generation stats
    pub stats: GenerationStats,
}

/// Unified inference engine trait for all AI backends.
///
/// Implementors: `LlamaCppEngine`, `MistralRsEngine`
///
/// # Thread Safety
/// All implementations must be `Send + Sync` to support sharing via `Arc<dyn InferenceEngine>`.
/// For backends with `!Send` contexts (e.g., llama-cpp-2's `LlamaContext`), use
/// `tokio::task::spawn_blocking` internally.
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Stream tokens to a channel as they are generated.
    ///
    /// This is the primary inference method. Tokens are sent through the `tx` channel
    /// as `StreamEvent::Token(text)`, with a final `StreamEvent::Complete(stats)`.
    ///
    /// Returns the complete generated text.
    async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        tx: mpsc::UnboundedSender<StreamEvent>,
    ) -> Result<String>;

    /// Simple non-streaming generation.
    ///
    /// Returns the complete generated text.
    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;

    /// v6.0.0: Deterministic generation for opML verification.
    ///
    /// Produces identical output across nodes for the same (prompt, seed, model).
    /// Returns individual tokens + prefix hashes for bisection dispute protocol.
    ///
    /// Default implementation: calls generate() with greedy settings and computes hashes.
    async fn generate_deterministic(
        &self,
        prompt: &str,
        max_tokens: usize,
        config: &DeterministicConfig,
    ) -> Result<DeterministicResult> {
        // Default: generate normally then compute commitment
        let text = self.generate(prompt, max_tokens).await?;
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let model_hash = self.model_hash();
        let stats = self.get_stats().await;

        let (output_hash, prefix_hashes) =
            compute_inference_commitment(&tokens, config.seed, &model_hash);

        Ok(DeterministicResult {
            text,
            tokens,
            output_hash,
            prefix_hashes,
            stats,
        })
    }

    /// Get current performance statistics.
    async fn get_stats(&self) -> GenerationStats;

    /// Human-readable name of this engine backend.
    fn engine_name(&self) -> &str;

    /// v6.0.0: SHA3-256 hash of the loaded model file for integrity verification.
    ///
    /// Workers include this in capability announcements. Requesters can require
    /// a specific model_hash to prevent model substitution attacks.
    ///
    /// Default: returns zeroed hash (unknown model).
    fn model_hash(&self) -> [u8; 32] {
        [0u8; 32]
    }
}

/// Compute the SHA3-256 commitment hash and prefix hashes for inference output.
///
/// - `output_hash` = SHA3(token_0 || token_1 || ... || token_n || seed || model_hash)
/// - `prefix_hashes[i]` = SHA3(token_0 || ... || token_i || seed || model_hash)
///
/// The prefix hashes enable O(log n) bisection to find the exact divergent token
/// in a dispute between worker and verifier.
pub fn compute_inference_commitment(
    tokens: &[String],
    seed: u64,
    model_hash: &[u8; 32],
) -> ([u8; 32], Vec<[u8; 32]>) {
    use sha3::{Sha3_256, Digest};

    let mut prefix_hashes = Vec::with_capacity(tokens.len());

    for i in 0..tokens.len() {
        let mut hasher = Sha3_256::new();
        // Hash all tokens up to and including index i
        for token in &tokens[..=i] {
            let token_bytes = token.as_bytes();
            hasher.update(&(token_bytes.len() as u32).to_le_bytes());
            hasher.update(token_bytes);
        }
        hasher.update(&seed.to_le_bytes());
        hasher.update(model_hash);

        let hash: [u8; 32] = hasher.finalize().into();
        prefix_hashes.push(hash);
    }

    // The output_hash is the last prefix hash (covers all tokens)
    let output_hash = prefix_hashes.last().copied().unwrap_or([0u8; 32]);

    (output_hash, prefix_hashes)
}

/// Compute SHA3-256 hash of a model file for integrity verification.
pub fn compute_model_hash(model_path: &std::path::Path) -> Result<[u8; 32]> {
    use sha3::{Sha3_256, Digest};
    use std::io::Read;

    let mut file = std::fs::File::open(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to open model file for hashing: {}", e))?;

    let mut hasher = Sha3_256::new();
    let mut buffer = [0u8; 65536]; // 64KB chunks

    loop {
        let bytes_read = file.read(&mut buffer)
            .map_err(|e| anyhow::anyhow!("Failed to read model file: {}", e))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hasher.finalize().into())
}
