//! Verification API - Real-time monitoring of proof-of-inference and worker benchmarks
//!
//! This module provides Server-Sent Events (SSE) endpoints for the UI to display
//! live verification events:
//! - Merkle proof submissions
//! - Challenge-response events
//! - Worker benchmark results
//! - Slashing events
//! - Worker health status changes

use axum::{
    extract::{Query, State},
    response::sse::{Event, Sse},
    routing::get,
    Json, Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tracing::{debug, info};

use crate::AppState;

/// Verification event sent via SSE
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VerificationEvent {
    /// Worker submitted proof-of-inference
    ProofSubmitted {
        request_id: String,
        worker_node_id: String,
        merkle_root: String,
        token_count: usize,
        timestamp_ms: u64,
    },

    /// Coordinator issued challenge to worker
    ChallengeIssued {
        request_id: String,
        worker_node_id: String,
        token_indices: Vec<usize>,
        deadline_ms: u64,
        timestamp_ms: u64,
    },

    /// Worker responded to challenge
    ChallengeResponse {
        request_id: String,
        worker_node_id: String,
        proofs_count: usize,
        timestamp_ms: u64,
    },

    /// Proof verification result
    VerificationComplete {
        request_id: String,
        worker_node_id: String,
        result: String, // "valid" or "invalid" or "timeout"
        tokens_verified: usize,
        timestamp_ms: u64,
    },

    /// Worker slashed for invalid proof
    WorkerSlashed {
        worker_node_id: String,
        request_id: String,
        reason: String,
        amount_qbc: f64,
        timestamp_ms: u64,
    },

    /// Worker benchmark challenge issued
    BenchmarkChallengeIssued {
        challenge_id: String,
        worker_node_id: String,
        claimed_capability: String,
        benchmark_tokens: usize,
        deadline_ms: u64,
        timestamp_ms: u64,
    },

    /// Worker benchmark result submitted
    BenchmarkResultSubmitted {
        challenge_id: String,
        worker_node_id: String,
        tokens_generated: usize,
        time_ms: u64,
        tokens_per_second: f64,
        timestamp_ms: u64,
    },

    /// Benchmark verification result
    BenchmarkVerificationComplete {
        worker_node_id: String,
        result: String, // "passed" or "failed" or "timeout"
        score: Option<f64>,
        reason: Option<String>,
        timestamp_ms: u64,
    },

    /// Worker health status changed
    WorkerHealthChanged {
        worker_node_id: String,
        old_status: String,
        new_status: String,
        timestamp_ms: u64,
    },

    /// Worker failover/retry event
    FailoverEvent {
        request_id: String,
        failed_worker_id: String,
        retry_worker_id: Option<String>,
        failure_type: String,
        attempt: usize,
        timestamp_ms: u64,
    },
}

/// GET /api/verification/stream - SSE stream of verification events
pub async fn stream_verification_events(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    info!("📡 New verification stream client connected");

    // Create broadcast channel for verification events
    let (tx, rx) = broadcast::channel::<VerificationEvent>(100);

    // Store sender in AppState for verification systems to publish events
    // (This would be set up during AppState initialization)

    // Convert broadcast receiver to stream
    let stream = BroadcastStream::new(rx)
        .filter_map(|result| match result {
            Ok(event) => Some(Ok(Event::default()
                .event("verification")
                .data(serde_json::to_string(&event).unwrap_or_default()))),
            Err(_) => None, // Channel lag, skip
        });

    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(10))
            .text("keepalive"),
    )
}

/// GET /api/verification/stats - Get verification statistics
#[derive(Debug, Serialize)]
pub struct VerificationStats {
    pub total_proofs_submitted: u64,
    pub total_proofs_verified: u64,
    pub total_proofs_invalid: u64,
    pub total_slashing_events: u64,
    pub total_slashed_qbc: f64,
    pub total_benchmarks_issued: u64,
    pub total_benchmarks_passed: u64,
    pub total_benchmarks_failed: u64,
    pub active_workers: usize,
    pub healthy_workers: usize,
    pub unhealthy_workers: usize,
    pub average_verification_time_ms: f64,
}

pub async fn get_verification_stats(
    State(state): State<Arc<AppState>>,
) -> Json<VerificationStats> {
    // TODO: Integrate with actual ProofOfInferenceVerifier and WorkerBenchmarkVerifier stats

    let stats = VerificationStats {
        total_proofs_submitted: 0,
        total_proofs_verified: 0,
        total_proofs_invalid: 0,
        total_slashing_events: 0,
        total_slashed_qbc: 0.0,
        total_benchmarks_issued: 0,
        total_benchmarks_passed: 0,
        total_benchmarks_failed: 0,
        active_workers: 0,
        healthy_workers: 0,
        unhealthy_workers: 0,
        average_verification_time_ms: 0.0,
    };

    Json(stats)
}

/// GET /api/verification/slashing - Get slashing history
#[derive(Debug, Serialize)]
pub struct SlashingHistoryResponse {
    pub records: Vec<SlashingRecordJson>,
    pub total_slashed: f64,
}

#[derive(Debug, Serialize)]
pub struct SlashingRecordJson {
    pub worker_node_id: String,
    pub request_id: String,
    pub reason: String,
    pub amount_qbc: f64,
    pub slashed_at_ms: u64,
    pub evidence: String,
}

pub async fn get_slashing_history(
    State(state): State<Arc<AppState>>,
) -> Json<SlashingHistoryResponse> {
    // TODO: Integrate with ProofOfInferenceVerifier.get_all_slashing_records()

    let response = SlashingHistoryResponse {
        records: vec![],
        total_slashed: 0.0,
    };

    Json(response)
}

/// GET /api/verification/worker-health - Get all worker health statuses
#[derive(Debug, Serialize)]
pub struct WorkerHealthResponse {
    pub workers: Vec<WorkerHealthStatus>,
}

#[derive(Debug, Serialize)]
pub struct WorkerHealthStatus {
    pub worker_node_id: String,
    pub health: String, // "healthy" | "degraded" | "unhealthy" | "testing"
    pub reputation: f64,
    pub recent_failures: usize,
    pub total_requests: usize,
    pub success_rate: f64,
}

pub async fn get_worker_health(
    State(state): State<Arc<AppState>>,
) -> Json<WorkerHealthResponse> {
    // TODO: Integrate with FailoverManager.get_all_worker_health()

    let response = WorkerHealthResponse {
        workers: vec![],
    };

    Json(response)
}

/// Query parameters for benchmark history
#[derive(Debug, Deserialize)]
pub struct BenchmarkHistoryQuery {
    pub worker_id: Option<String>,
}

/// GET /api/verification/benchmarks - Get benchmark history
#[derive(Debug, Serialize)]
pub struct BenchmarkHistoryResponse {
    pub benchmarks: Vec<BenchmarkHistoryRecord>,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkHistoryRecord {
    pub challenge_id: String,
    pub worker_node_id: String,
    pub claimed_capability: String,
    pub tokens_generated: usize,
    pub time_ms: u64,
    pub tokens_per_second: f64,
    pub result: String, // "passed" | "failed" | "timeout"
    pub score: Option<f64>,
    pub timestamp_ms: u64,
}

pub async fn get_benchmark_history(
    State(state): State<Arc<AppState>>,
    Query(query): Query<BenchmarkHistoryQuery>,
) -> Json<BenchmarkHistoryResponse> {
    // TODO: Integrate with WorkerBenchmarkVerifier benchmark history

    let response = BenchmarkHistoryResponse {
        benchmarks: vec![],
    };

    Json(response)
}

/// Create verification API router
pub fn verification_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/stream", get(stream_verification_events))
        .route("/stats", get(get_verification_stats))
        .route("/slashing", get(get_slashing_history))
        .route("/worker-health", get(get_worker_health))
        .route("/benchmarks", get(get_benchmark_history))
}
