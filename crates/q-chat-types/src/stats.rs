//! AI generation statistics types.

use serde::{Deserialize, Serialize};

/// Statistics produced by a single AI inference run.
///
/// Tracks token counts, latency, privacy overhead, and the number of
/// distributed nodes involved in the generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    /// Total tokens produced (prompt + completion).
    pub total_tokens: usize,
    /// Wall-clock latency in milliseconds for the full response.
    pub latency_ms: u64,
    /// Sustained throughput during generation (tokens / second).
    pub tokens_per_second: f64,
    /// Extra latency added by the privacy layer (Dandelion++, Tor routing, etc.).
    pub privacy_overhead_ms: u64,
    /// Time spent generating a ZK proof for the response (0 if disabled).
    pub zk_proof_time_ms: u64,
    /// How many inference nodes participated (1 = local, N = distributed).
    pub distributed_nodes_used: usize,
}

impl Default for GenerationStats {
    fn default() -> Self {
        Self {
            total_tokens: 0,
            latency_ms: 0,
            tokens_per_second: 0.0,
            privacy_overhead_ms: 0,
            zk_proof_time_ms: 0,
            distributed_nodes_used: 1,
        }
    }
}
