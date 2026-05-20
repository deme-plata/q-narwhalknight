//! Agent Fiber Lane (AFL-1) — REST endpoints per `docs/standards/afl-1-protocol-spec.md`.
//!
//! Three paths, escalating throughput:
//!
//!   POST /api/v1/agent/submit         — single tx, ≤ 50 ms RTT target
//!   POST /api/v1/agent/submit-batch   — array of intents, 1000 tx default cap
//!   WS   /api/v1/agent/stream         — persistent connection for sustained load
//!                                       (this file scaffolds; WS handler is a follow-up)
//!
//! Auth model (spec §2.1): X-Wallet-Auth Ed25519 signature with body_hash
//! folded into the signed payload. One signature authenticates the full body
//! (single tx OR entire batch array).
//!
//! Server flow:
//!   1. Verify X-Wallet-Auth (reuses wallet_auth::verify_request)
//!   2. Verify body_hash == sha3-256(received body)
//!   3. Clock-skew window (±30s) + replay dedup (60s LRU)
//!   4. Allocate nonce(s) for the wallet
//!   5. Build full Transaction(s) server-side
//!   6. Submit to ProductionMempool + tx_pool, broadcast via gossipsub
//!   7. Return tx_id(s) + assigned_nonce(s)
//!
//! Latency target: single submit ≤ 50 ms RTT on local network.
//! Throughput target: batch 1000 tx × 100/sec = 100,000 TPS via this endpoint.

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::Arc;

use crate::AppState;

// ============ CONSTANTS ============

/// Default max transactions per batch. Configurable via Q_AGENT_BATCH_MAX env var (cap 10000).
const DEFAULT_BATCH_MAX: usize = 1000;
const HARD_BATCH_MAX: usize = 10_000;

/// Clock-skew tolerance window for X-Wallet-Auth timestamp (seconds).
const TIMESTAMP_TOLERANCE_SECS: i64 = 30;

/// Replay-dedup LRU window. body_hash seen within this window from the same
/// address is rejected as duplicate. Spec §2.1 step 4.
const REPLAY_DEDUP_WINDOW_SECS: u64 = 60;

// ============ REQUEST/RESPONSE SHAPES ============

/// Single intent body for /api/v1/agent/submit. Per spec §2.1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIntent {
    pub to: String,
    pub amount: String, // u128 as string to survive JSON
    pub token_type: String, // "QUG" | "QUGUSD" | "QSHARE" | "Custom:<hex>"
    #[serde(default)]
    pub memo: Option<String>,
    /// Optional explicit fee (default: MIN_TRANSACTION_FEE)
    #[serde(default)]
    pub fee: Option<String>,
    /// Optional explicit nonce override (default: server-assigned next)
    #[serde(default)]
    pub nonce: Option<u64>,
}

/// Batch body for /api/v1/agent/submit-batch. Per spec §2.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBatchRequest {
    pub transactions: Vec<AgentIntent>,
}

/// Response from /api/v1/agent/submit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSubmitResponse {
    pub tx_id: String,
    pub assigned_nonce: u64,
    pub included_at_block: Option<u64>,
}

/// Response from /api/v1/agent/submit-batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBatchResponse {
    pub tx_ids: Vec<String>,
    pub first_nonce: u64,
    pub last_nonce: u64,
    pub accepted_count: usize,
    pub rejected_count: usize,
}

/// Standardised error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentError {
    pub code: String,
    pub message: String,
}

// ============ HELPERS ============

/// Compute SHA3-256 hex of the raw request body bytes.
fn compute_body_hash(body: &[u8]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(body);
    hex::encode(hasher.finalize())
}

/// Extract the X-Wallet-Auth header and parse the body_hash field from it.
/// Returns Err if the header is missing or malformed.
fn extract_auth_body_hash(headers: &HeaderMap) -> Result<String, AgentError> {
    let auth_header = headers
        .get("X-Wallet-Auth")
        .ok_or_else(|| AgentError {
            code: "MISSING_AUTH".to_string(),
            message: "X-Wallet-Auth header is required".to_string(),
        })?
        .to_str()
        .map_err(|_| AgentError {
            code: "INVALID_AUTH".to_string(),
            message: "X-Wallet-Auth header is not valid UTF-8".to_string(),
        })?;

    let parsed: serde_json::Value = serde_json::from_str(auth_header).map_err(|_| AgentError {
        code: "INVALID_AUTH".to_string(),
        message: "X-Wallet-Auth header is not valid JSON".to_string(),
    })?;

    let body_hash = parsed
        .get("body_hash")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AgentError {
            code: "MISSING_BODY_HASH".to_string(),
            message: "X-Wallet-Auth.body_hash field is required for AFL-1".to_string(),
        })?
        .to_string();

    Ok(body_hash)
}

/// Read the configured max batch size from env, clamped to HARD_BATCH_MAX.
fn batch_max() -> usize {
    std::env::var("Q_AGENT_BATCH_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BATCH_MAX)
        .min(HARD_BATCH_MAX)
}

// ============ HANDLERS ============

/// POST /api/v1/agent/submit — single-tx fast lane.
///
/// SCAFFOLD STATUS (v10.10.5 in progress):
/// - Body-hash verification: wired
/// - Timestamp/replay protection: TODO (needs LRU; v10.10.5 follow-up)
/// - X-Wallet-Auth signature verify: TODO (re-use wallet_auth::verify_request)
/// - Nonce allocation: TODO (per-wallet tokio::Mutex + next_nonce lookup)
/// - Transaction build + mempool insert + gossipsub broadcast: TODO
/// - Returns 501 NOT_IMPLEMENTED for now with clear next-step pointer.
pub async fn submit_single(
    State(_state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<AgentIntent>,
) -> impl IntoResponse {
    // Step 1: extract & verify body_hash
    let body_bytes = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(e) => return error_response(
            StatusCode::BAD_REQUEST,
            "JSON_REENCODE_FAILED",
            &format!("could not re-encode body for hash: {}", e),
        ),
    };
    let expected_hash = compute_body_hash(&body_bytes);

    let actual_hash = match extract_auth_body_hash(&headers) {
        Ok(h) => h,
        Err(e) => return error_response(StatusCode::UNAUTHORIZED, &e.code, &e.message),
    };

    if expected_hash != actual_hash {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "BODY_HASH_MISMATCH",
            &format!("expected sha3-256={}, header had {}", expected_hash, actual_hash),
        );
    }

    // Step 2+: signature verify, nonce alloc, tx build, mempool insert, broadcast.
    // SCAFFOLD: not yet wired. Returning a placeholder so callers can shape against
    // the contract surface.
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "AGENT_SUBMIT_SCAFFOLD",
        "POST /api/v1/agent/submit is scaffolded but not yet wired to mempool+broadcast. \
         Body-hash verification works; signature verify + nonce alloc + tx build are \
         the remaining steps. See docs/standards/afl-1-protocol-spec.md §2.1.",
    )
}

/// POST /api/v1/agent/submit-batch — batch lane, throughput-first.
///
/// SCAFFOLD STATUS (v10.10.5 in progress): same status as submit_single.
/// Batch size enforced; full submission pipeline TODO.
pub async fn submit_batch(
    State(_state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<AgentBatchRequest>,
) -> impl IntoResponse {
    // Batch size limit per spec §2.2
    let max_batch = batch_max();
    if body.transactions.len() > max_batch {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "BATCH_TOO_LARGE",
            &format!(
                "batch has {} txs, max allowed is {} (configurable via Q_AGENT_BATCH_MAX)",
                body.transactions.len(),
                max_batch
            ),
        );
    }
    if body.transactions.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "BATCH_EMPTY",
            "transactions array must contain at least one intent",
        );
    }

    // Body-hash verification
    let body_bytes = match serde_json::to_vec(&body) {
        Ok(b) => b,
        Err(e) => return error_response(
            StatusCode::BAD_REQUEST,
            "JSON_REENCODE_FAILED",
            &format!("could not re-encode body for hash: {}", e),
        ),
    };
    let expected_hash = compute_body_hash(&body_bytes);

    let actual_hash = match extract_auth_body_hash(&headers) {
        Ok(h) => h,
        Err(e) => return error_response(StatusCode::UNAUTHORIZED, &e.code, &e.message),
    };

    if expected_hash != actual_hash {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "BODY_HASH_MISMATCH",
            &format!("expected sha3-256={}, header had {}", expected_hash, actual_hash),
        );
    }

    // SCAFFOLD: rest of pipeline TODO. The batch path needs:
    //   - Single signature verify (one for whole batch — that's the throughput win)
    //   - Atomic nonce range allocation [next..next+N]
    //   - Parallel Transaction::build for the N intents
    //   - ProductionMempool::add_transactions_batch (Codex must add this method
    //     if it doesn't exist on production_mempool.rs)
    //   - Single gossipsub broadcast with postcard-encoded array
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "AGENT_BATCH_SCAFFOLD",
        &format!(
            "POST /api/v1/agent/submit-batch is scaffolded for {} txs. Body-hash + \
             batch-size validation work; signature verify + atomic nonce range + \
             batch mempool insert are the remaining steps. See \
             docs/standards/afl-1-protocol-spec.md §2.2.",
            body.transactions.len()
        ),
    )
}

/// Convenience: build a JSON error response with given status, code, and message.
fn error_response(status: StatusCode, code: &str, message: &str) -> axum::response::Response {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message,
        }
    });
    (status, axum::Json(body)).into_response()
}

// ============ TESTS ============

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_hash_deterministic() {
        let intent = AgentIntent {
            to: "qnk0123".to_string(),
            amount: "1000000000000000000000000".to_string(),
            token_type: "QUG".to_string(),
            memo: Some("test".to_string()),
            fee: None,
            nonce: None,
        };
        let body = serde_json::to_vec(&intent).unwrap();
        let h1 = compute_body_hash(&body);
        let h2 = compute_body_hash(&body);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // 32 bytes hex = 64 chars
    }

    #[test]
    fn batch_max_env_clamp() {
        // No env var → default
        std::env::remove_var("Q_AGENT_BATCH_MAX");
        assert_eq!(batch_max(), DEFAULT_BATCH_MAX);

        // Within range → use env
        std::env::set_var("Q_AGENT_BATCH_MAX", "2500");
        assert_eq!(batch_max(), 2500);

        // Above hard cap → clamp
        std::env::set_var("Q_AGENT_BATCH_MAX", "999999");
        assert_eq!(batch_max(), HARD_BATCH_MAX);

        // Cleanup
        std::env::remove_var("Q_AGENT_BATCH_MAX");
    }

    #[test]
    fn extract_body_hash_from_valid_header() {
        let mut headers = HeaderMap::new();
        let auth_json = serde_json::json!({
            "address": "qnk7154929a",
            "timestamp": 1779168000,
            "scheme": "Ed25519",
            "signature": "00".repeat(64),
            "body_hash": "ab".repeat(32),
        });
        headers.insert(
            "X-Wallet-Auth",
            auth_json.to_string().parse().unwrap(),
        );
        let bh = extract_auth_body_hash(&headers).unwrap();
        assert_eq!(bh, "ab".repeat(32));
    }

    #[test]
    fn extract_body_hash_missing_header_errors() {
        let headers = HeaderMap::new();
        let r = extract_auth_body_hash(&headers);
        assert!(matches!(r, Err(AgentError { code, .. }) if code == "MISSING_AUTH"));
    }
}
