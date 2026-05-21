//! Agent Fiber Lane (AFL-1) — REST endpoints per `docs/standards/afl-1-protocol-spec.md`.
//!
//! Two paths shipped in v10.10.7:
//!
//!   POST /api/v1/agent/submit         — single tx, ≤ 50 ms RTT target
//!   POST /api/v1/agent/submit-batch   — array of intents, 1000 tx default cap
//!
//! WebSocket (spec §2.3) deferred to v10.10.8+.
//!
//! Auth model (spec §2.1): X-Wallet-Auth Ed25519 signature with body_hash
//! folded into the signed payload. The AuthenticatedWallet extractor verifies
//! the signature + body_hash + timestamp window. One signature authenticates
//! the full body (single tx OR entire batch array).
//!
//! Server flow:
//!   1. X-Wallet-Auth verified by extractor (signature + body_hash + ts window)
//!   2. Allocate nonce(s) via state.nonce_tracker
//!   3. Build full Transaction(s) server-side from AgentIntent + the auth address
//!   4. Submit via transaction_utils::submit_transaction (mempool insert +
//!      gossipsub broadcast)
//!   5. Return tx_id(s) + assigned_nonce(s)
//!
//! Latency target: single submit ≤ 50 ms RTT on local network.
//! Throughput target: batch 1000 tx × 100/sec = 100,000 TPS via this endpoint.

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;
use crate::transaction_utils::{self, TransactionBuilder};
use crate::wallet_auth::AuthenticatedWallet;
use q_types::{TokenType, TransactionType};

// ============ CONSTANTS ============

/// Default max transactions per batch. Configurable via Q_AGENT_BATCH_MAX env var (cap 10000).
const DEFAULT_BATCH_MAX: usize = 1000;
const HARD_BATCH_MAX: usize = 10_000;

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
}

/// Batch body for /api/v1/agent/submit-batch. Per spec §2.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBatchRequest {
    pub transactions: Vec<AgentIntent>,
}

/// Per-tx submission result inside a batch response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBatchItem {
    pub tx_id: String,
    pub assigned_nonce: u64,
    pub status: String, // "queued" | "rejected"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>, // present if status == "rejected"
}

/// Response from /api/v1/agent/submit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSubmitResponse {
    pub tx_id: String,
    pub assigned_nonce: u64,
    pub queued_for_block: bool,
    pub broadcast_success: bool,
}

/// Response from /api/v1/agent/submit-batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBatchResponse {
    pub items: Vec<AgentBatchItem>,
    pub first_nonce: u64,
    pub last_nonce: u64,
    pub accepted_count: usize,
    pub rejected_count: usize,
}

// ============ HELPERS ============

/// Parse a wallet address from "qnk<64hex>" or raw "<64hex>". Returns 32-byte
/// address.
fn parse_wallet_addr(s: &str) -> Result<[u8; 32], String> {
    let clean = s.trim_start_matches("qnk").trim_start_matches("QNK");
    if clean.len() != 64 {
        return Err(format!(
            "expected 64-hex address (optionally 'qnk' prefixed), got {} chars",
            clean.len()
        ));
    }
    let bytes = hex::decode(clean).map_err(|e| format!("hex decode failed: {}", e))?;
    if bytes.len() != 32 {
        return Err(format!("expected 32 bytes, got {}", bytes.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// Parse amount string into u128, with defensive upper bound.
fn parse_amount(s: &str) -> Result<u128, String> {
    let a: u128 = s
        .parse()
        .map_err(|e| format!("amount not a u128: {}", e))?;
    if a == 0 {
        return Err("amount must be > 0".to_string());
    }
    // Defensive upper bound matching send_transaction_signed (~1B QUG in 24-dec)
    if a > 10u128.pow(33) {
        return Err("amount too large (cap: 1e33 raw units, ~1B in 24-decimal)".to_string());
    }
    Ok(a)
}

/// Resolve a token type string to its (TransactionType, TokenType, tx_data) triple.
/// Matches send_transaction_signed's resolution shape exactly so behavior is
/// consistent between the two paths.
fn resolve_token(token_str: &str) -> Result<(TransactionType, TokenType, Vec<u8>), String> {
    let upper = token_str.to_uppercase();
    if upper == "QUG" || upper == "NATIVE-QUG" {
        Ok((TransactionType::Transfer, TokenType::QUG, Vec::new()))
    } else if upper == "QUGUSD" || upper == "QUGUSD-STABLE" {
        Ok((TransactionType::TokenTransfer, TokenType::QUGUSD, q_types::QUGUSD_TOKEN_ADDRESS.to_vec()))
    } else if upper == "QSHARE" {
        Ok((TransactionType::TokenTransfer, TokenType::QSHARE, q_types::QSHARE_TOKEN_ADDRESS.to_vec()))
    } else {
        // Custom token by contract address
        let addr = parse_wallet_addr(token_str)
            .map_err(|e| format!("invalid token_type ({}): {}", token_str, e))?;
        Ok((TransactionType::TokenTransfer, TokenType::Custom(addr), addr.to_vec()))
    }
}

/// Build a Transaction from a validated AgentIntent + the authenticated sender.
/// Returns the built Transaction ready for mempool submission.
fn build_tx_from_intent(
    intent: &AgentIntent,
    from: [u8; 32],
    nonce: u64,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<q_types::Transaction, String> {
    let to = parse_wallet_addr(&intent.to)
        .map_err(|e| format!("invalid 'to' address: {}", e))?;
    let amount = parse_amount(&intent.amount)?;
    if from == to {
        return Err("from == to (no-op)".to_string());
    }
    let (tx_type, token_type, tx_data) = resolve_token(&intent.token_type)?;

    let mut tx = TransactionBuilder::new()
        .from(from)
        .to(to)
        .amount(amount)
        .token_type(token_type)
        .tx_type(tx_type)
        .data(tx_data)
        .build_with_nonce(nonce, now);

    // v10.10.15 fix: same bug as send_transaction_signed had pre-v10.10.14.
    // TransactionBuilder default-initializes fee: 0. After v10.9.58 PR #68
    // tightened ProductionMempool::perform_validation from Ok(true) stub to
    // real fee/signature/coinbase checks, any tx with fee=0 silently fails
    // block-inclusion (mempool accepts at queue-time, API returns success+
    // tx_id, no debit ever applies — the "ghost confirmation" pattern
    // observed 2026-05-21 on the 1-QUG handshake test, where send_signed
    // returned HTTP 200 but no balance moved).
    //
    // Honor caller-provided fee if present; otherwise use MIN_TRANSACTION_FEE_V1.
    tx.fee = match &intent.fee {
        Some(s) => s.parse::<u128>().unwrap_or(q_types::MIN_TRANSACTION_FEE_V1),
        None => q_types::MIN_TRANSACTION_FEE_V1,
    };

    if let Some(memo) = &intent.memo {
        tx.memo = Some(memo.clone());
        tx.id = transaction_utils::compute_transaction_id(&tx);
    } else {
        // Even without memo, the fee change above mutated tx, so recompute id
        // before returning. id is a function of all serialized fields.
        tx.id = transaction_utils::compute_transaction_id(&tx);
    }

    Ok(tx)
}

/// Read configured max batch size from env, clamped to HARD_BATCH_MAX.
fn batch_max() -> usize {
    std::env::var("Q_AGENT_BATCH_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BATCH_MAX)
        .min(HARD_BATCH_MAX)
}

/// Convenience: build a JSON error response with given status, code, and message.
fn error_response(status: StatusCode, code: &str, message: impl Into<String>) -> axum::response::Response {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message.into(),
        }
    });
    (status, axum::Json(body)).into_response()
}

// ============ HANDLERS ============

/// POST /api/v1/agent/submit — single-tx fast lane (spec §2.1).
///
/// AuthenticatedWallet extractor handles X-Wallet-Auth signature + body_hash +
/// timestamp window. After that we just build the tx and submit.
pub async fn submit_single(
    State(state): State<Arc<AppState>>,
    auth: AuthenticatedWallet,
    Json(intent): Json<AgentIntent>,
) -> impl IntoResponse {
    // Allocate nonce (server-side; spec §2.1 step 5).
    let nonce = state.nonce_tracker.get_and_increment(&auth.address);
    let now = chrono::Utc::now();

    let tx = match build_tx_from_intent(&intent, auth.address, nonce, now) {
        Ok(t) => t,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, "INVALID_INTENT", e),
    };
    let tx_id = tx.id;

    // Submit to mempool + broadcast (spec §2.1 step 7+8).
    let result = transaction_utils::submit_transaction(
        tx,
        &state.tx_pool,
        &state.tx_status,
        state.production_mempool.as_ref(),
        state.libp2p_discovery.as_ref(),
    ).await;

    let response = AgentSubmitResponse {
        tx_id: format!("0x{}", hex::encode(tx_id)),
        assigned_nonce: nonce,
        queued_for_block: result.queued_for_block,
        broadcast_success: result.broadcast_success,
    };
    (StatusCode::OK, axum::Json(response)).into_response()
}

/// POST /api/v1/agent/submit-batch — batch lane, throughput-first (spec §2.2).
///
/// One signature for the whole batch (verified by AuthenticatedWallet against
/// the JSON body's sha3-256). Atomic nonce range allocation, then submit each
/// tx. Returns per-tx outcome so partial failures are visible.
pub async fn submit_batch(
    State(state): State<Arc<AppState>>,
    auth: AuthenticatedWallet,
    Json(body): Json<AgentBatchRequest>,
) -> impl IntoResponse {
    let max_batch = batch_max();
    if body.transactions.len() > max_batch {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "BATCH_TOO_LARGE",
            format!("batch has {} txs, max allowed is {}", body.transactions.len(), max_batch),
        );
    }
    if body.transactions.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "BATCH_EMPTY",
            "transactions array must contain at least one intent",
        );
    }

    // v10.10.15: Parallel batch submission.
    //
    // Pre-v10.10.15 this loop was strictly sequential — `for ... await` on
    // submit_transaction per item, meaning a 1000-tx batch sequentialized
    // through ~5 ms × 1000 = ~5 s at handler level alone. Now we:
    //
    //  1. Allocate the FULL nonce range up front via NonceTracker::allocate_range
    //     (one atomic DashMap op for the whole batch, instead of N).
    //  2. Build + submit each tx in its own future.
    //  3. join_all the futures so they run concurrently on the tokio scheduler,
    //     bounded only by the mempool RwLock + libp2p mutex (the structural
    //     bottlenecks tracked for v10.10.16).
    //
    // Response ordering preserved (join_all returns in input order). Per-item
    // success/error semantics unchanged — best-effort batch atomicity per
    // AFL-1 spec §4.2.
    let now = chrono::Utc::now();
    let total = body.transactions.len();
    let first_nonce = state.nonce_tracker.allocate_range(&auth.address, total as u64);

    use futures::future::join_all;
    let item_futures = body.transactions.iter().enumerate().map(|(idx, intent)| {
        let state_arc = state.clone();
        let auth_addr = auth.address;
        let intent_clone = intent.clone();
        let nonce = first_nonce + idx as u64;
        async move {
            let tx = match build_tx_from_intent(&intent_clone, auth_addr, nonce, now) {
                Ok(t) => t,
                Err(reason) => {
                    return AgentBatchItem {
                        tx_id: format!("0x{}", "0".repeat(64)),
                        assigned_nonce: nonce,
                        status: "rejected".to_string(),
                        reason: Some(reason),
                    };
                }
            };
            let tx_id = tx.id;
            let result = transaction_utils::submit_transaction(
                tx,
                &state_arc.tx_pool,
                &state_arc.tx_status,
                state_arc.production_mempool.as_ref(),
                state_arc.libp2p_discovery.as_ref(),
            ).await;
            AgentBatchItem {
                tx_id: format!("0x{}", hex::encode(tx_id)),
                assigned_nonce: nonce,
                status: if result.queued_for_block { "queued".to_string() } else { "rejected".to_string() },
                reason: if result.queued_for_block { None } else { Some("mempool refused".to_string()) },
            }
        }
    });
    let items: Vec<AgentBatchItem> = join_all(item_futures).await;
    let accepted = items.iter().filter(|i| i.status == "queued").count();
    let rejected = total - accepted;

    let last_nonce = first_nonce + (total as u64).saturating_sub(1);
    let response = AgentBatchResponse {
        items,
        first_nonce,
        last_nonce,
        accepted_count: accepted,
        rejected_count: rejected,
    };
    (StatusCode::OK, axum::Json(response)).into_response()
}

// ============ TESTS ============

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_max_env_clamp() {
        std::env::remove_var("Q_AGENT_BATCH_MAX");
        assert_eq!(batch_max(), DEFAULT_BATCH_MAX);

        std::env::set_var("Q_AGENT_BATCH_MAX", "2500");
        assert_eq!(batch_max(), 2500);

        std::env::set_var("Q_AGENT_BATCH_MAX", "999999");
        assert_eq!(batch_max(), HARD_BATCH_MAX);

        std::env::remove_var("Q_AGENT_BATCH_MAX");
    }

    #[test]
    fn parse_wallet_addr_with_and_without_prefix() {
        let raw = "ab".repeat(32);
        let with_prefix = format!("qnk{}", raw);
        let a1 = parse_wallet_addr(&raw).unwrap();
        let a2 = parse_wallet_addr(&with_prefix).unwrap();
        assert_eq!(a1, a2);
    }

    #[test]
    fn parse_amount_rejects_zero() {
        assert!(parse_amount("0").is_err());
        assert!(parse_amount("1000000000000000000000").is_ok());
    }

    #[test]
    fn resolve_token_known_symbols() {
        let (_, t, _) = resolve_token("QUG").unwrap();
        assert!(matches!(t, TokenType::QUG));
        let (_, t, _) = resolve_token("QUGUSD").unwrap();
        assert!(matches!(t, TokenType::QUGUSD));
        let (_, t, _) = resolve_token("QSHARE").unwrap();
        assert!(matches!(t, TokenType::QSHARE));
    }

    #[test]
    fn resolve_token_custom_address() {
        let addr_hex = "ab".repeat(32);
        let (_, t, data) = resolve_token(&addr_hex).unwrap();
        assert!(matches!(t, TokenType::Custom(_)));
        assert_eq!(data.len(), 32);
    }
}
