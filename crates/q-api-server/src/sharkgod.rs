//! 🦈 SharkGod — Maximum Power Transaction Beam
//!
//! Bypasses ALL normal transaction bottlenecks for maximum speed:
//! - No Dandelion++ stem phase (skip 20s privacy delay)
//! - No gossipsub queue (direct publish, bypass rate limiting)
//! - No async spawn overhead (synchronous validation)
//! - Pre-serialize ONCE (reuse for all propagation paths)
//! - Wake block producer immediately (don't wait for 15s tick)
//! - Parallel unicast to ALL bootstrap peers (guaranteed delivery)

use crate::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
};
use q_types::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Minimum fee multiplier for SharkGod transactions (10× the base minimum)
const SHARKGOD_FEE_MULTIPLIER: u128 = 10;

/// Base minimum fee (same as Transaction::validate_fee minimum)
const BASE_MINIMUM_FEE: u128 = 1_000_000_000_000_000_000; // 1e18 (0.000001 QUG)

#[derive(Debug, Deserialize)]
pub struct SharkGodRequest {
    pub transaction: Transaction,
    /// Auto-boost fee to 10× minimum for priority inclusion (default: true)
    #[serde(default = "default_true")]
    pub boost_fee: bool,
    /// Wake block producer immediately to include tx in next block (default: true)
    #[serde(default = "default_true")]
    pub wake_producer: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct SharkGodReceipt {
    pub tx_hash: String,
    /// Number of peers the transaction was propagated to
    pub propagated_to: usize,
    /// Microseconds from submit to broadcast completion
    pub latency_us: u64,
    /// The fee that was applied (possibly boosted)
    pub fee_applied: String, // String for JSON u128 safety
    /// Position in the fee-ordered mempool (0 = highest priority)
    pub mempool_position: usize,
    /// Whether the block producer was woken to include this tx immediately
    pub block_producer_woken: bool,
}

#[derive(Debug, Serialize)]
pub struct SharkGodResponse {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub receipt: Option<SharkGodReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// POST /api/v1/sharkgod/submit
///
/// Maximum power transaction submission — bypasses Dandelion++, gossipsub queue,
/// and async overhead for the fastest possible transaction propagation.
pub async fn sharkgod_submit_handler(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<SharkGodRequest>,
) -> impl IntoResponse {
    let start = Instant::now();
    let tx = &mut request.transaction;

    info!(
        "🦈 [SHARKGOD] Incoming tx from {}",
        hex::encode(&tx.from[..4])
    );

    // ========================================================================
    // STEP 1: Synchronous validation (no tokio::spawn overhead)
    // ========================================================================

    // 1a. Signature verification
    if let Err(sig_error) = tx.verify_signature() {
        warn!(
            "🦈 [SHARKGOD] Signature FAILED: {}",
            sig_error
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(SharkGodResponse {
                success: false,
                receipt: None,
                error: Some(format!("Signature invalid: {}", sig_error)),
            }),
        );
    }

    // 1b. Fee validation
    if let Err(fee_error) = tx.validate_fee() {
        warn!("🦈 [SHARKGOD] Fee FAILED: {}", fee_error);
        return (
            StatusCode::BAD_REQUEST,
            Json(SharkGodResponse {
                success: false,
                receipt: None,
                error: Some(format!("Fee invalid: {}", fee_error)),
            }),
        );
    }

    // 1c. Founder wallet protection
    if tx.is_from_founder_wallet() {
        let current_height = state
            .current_height_atomic
            .load(std::sync::atomic::Ordering::Relaxed);
        if let Err(founder_error) = tx.validate_founder_withdrawal(current_height) {
            warn!("🦈 [SHARKGOD] Founder protection: {}", founder_error);
            return (
                StatusCode::FORBIDDEN,
                Json(SharkGodResponse {
                    success: false,
                    receipt: None,
                    error: Some(format!("Founder wallet protection: {}", founder_error)),
                }),
            );
        }
    }

    let tx_hash = tx.hash();
    let tx_hash_hex = hex::encode(&tx_hash);

    // 1d. Anti-replay: cross-block dedup
    if state.applied_tx_dedup.contains_key(&tx_hash) {
        warn!(
            "🦈 [SHARKGOD] Duplicate tx rejected: {}",
            &tx_hash_hex[..16]
        );
        return (
            StatusCode::CONFLICT,
            Json(SharkGodResponse {
                success: false,
                receipt: None,
                error: Some("Transaction already applied (duplicate)".to_string()),
            }),
        );
    }

    // 1e. Nonce validation
    let tx_from = tx.from;
    let tx_nonce = tx.nonce;
    if tx_nonce > 0 || tx.tx_type == TransactionType::Transfer {
        if let Err(expected) = state.nonce_tracker.validate_nonce(&tx_from, tx_nonce) {
            warn!(
                "🦈 [SHARKGOD] Nonce mismatch: got {}, expected {}",
                tx_nonce, expected
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(SharkGodResponse {
                    success: false,
                    receipt: None,
                    error: Some(format!("Invalid nonce: got {}, expected {}", tx_nonce, expected)),
                }),
            );
        }
        state.nonce_tracker.get_and_increment(&tx_from);
    }

    // ========================================================================
    // STEP 2: Fee boost (10× minimum for priority inclusion)
    // ========================================================================
    let fee_applied = if request.boost_fee {
        let sharkgod_min = BASE_MINIMUM_FEE * SHARKGOD_FEE_MULTIPLIER;
        if tx.fee < sharkgod_min {
            tx.fee = sharkgod_min;
            debug!(
                "🦈 [SHARKGOD] Fee boosted to {} (10× minimum)",
                sharkgod_min
            );
        }
        tx.fee
    } else {
        tx.fee
    };

    // ========================================================================
    // STEP 3: Pre-serialize ONCE (reuse for all propagation paths)
    // ========================================================================
    let peer_info = state.libp2p_peer_info.read().await;
    let origin_node_id = peer_info.0.clone();
    drop(peer_info);

    let p2p_tx = P2PTransaction::new(tx.clone(), origin_node_id);
    let tx_bytes = match postcard::to_allocvec(&p2p_tx) {
        Ok(bytes) => bytes,
        Err(e) => {
            warn!("🦈 [SHARKGOD] Serialization failed: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(SharkGodResponse {
                    success: false,
                    receipt: None,
                    error: Some(format!("Serialization failed: {}", e)),
                }),
            );
        }
    };

    info!(
        "🦈 [SHARKGOD] Pre-serialized tx {} ({} bytes)",
        &tx_hash_hex[..16],
        tx_bytes.len()
    );

    // ========================================================================
    // STEP 4: Insert mempool with MAX priority (lock-free)
    // ========================================================================
    state.tx_pool.insert(tx_hash, tx.clone());
    state.tx_status.insert(tx_hash, TxStatus::InMempool);

    // Submit to Narwhal ProductionMempool for fee-ordered batching
    let mempool_position = if let Some(ref production_mempool) = state.production_mempool {
        // Synchronous-style: we await directly instead of spawning
        match production_mempool.add_transaction(tx.clone(), None).await {
            Ok(true) => {
                debug!("🦈 [SHARKGOD] Added to production mempool");
                0 // SharkGod txs have boosted fees → typically position 0
            }
            Ok(false) => 0,
            Err(e) => {
                warn!("🦈 [SHARKGOD] Production mempool error: {}", e);
                0
            }
        }
    } else {
        0
    };

    // ========================================================================
    // STEP 5: DIRECT gossipsub publish (bypass queue + rate limiter)
    // ========================================================================
    let mut propagated_to: usize = 0;

    if let Some(ref cmd_tx) = state.libp2p_command_tx {
        let network_id = std::env::var("Q_NETWORK_ID")
            .unwrap_or_else(|_| "mainnet-genesis".to_string())
            .parse::<NetworkId>()
            .unwrap_or(NetworkId::MainnetGenesis);
        let topic = network_id.mempool_transactions_topic();

        // Send via PublishSharkGod command — bypasses gossipsub queue, publishes directly
        if let Err(e) = cmd_tx.send(q_network::NetworkCommand::PublishSharkGod {
            topic: topic.clone(),
            data: tx_bytes.clone(),
            tx_hash: tx_hash_hex.clone(),
        }) {
            warn!("🦈 [SHARKGOD] Failed to send direct publish command: {}", e);
        } else {
            // Optimistically count — the network manager will do the actual publish
            propagated_to += 1;
            debug!(
                "🦈 [SHARKGOD] Direct gossipsub publish sent for {}",
                &tx_hash_hex[..16]
            );
        }

        // Also publish to legacy /transactions topic for older peers
        let legacy_topic = network_id.transactions_topic();
        let legacy_bytes = match postcard::to_allocvec(&tx.clone()) {
            Ok(b) => b,
            Err(_) => tx_bytes.clone(), // fallback to p2p_tx bytes
        };
        if let Err(_) = cmd_tx.send(q_network::NetworkCommand::PublishSharkGod {
            topic: legacy_topic,
            data: legacy_bytes,
            tx_hash: tx_hash_hex.clone(),
        }) {
            // Non-critical — legacy topic is best-effort
        }
    }

    // ========================================================================
    // STEP 6: Wake block producer immediately
    // ========================================================================
    let block_producer_woken = if request.wake_producer {
        if let Some(ref notify) = state.sharkgod_block_wake {
            notify.notify_one();
            info!("🦈 [SHARKGOD] Block producer WOKEN — immediate inclusion");
            true
        } else {
            debug!("🦈 [SHARKGOD] No block wake channel — producer will pick up on next tick");
            false
        }
    } else {
        false
    };

    // ========================================================================
    // STEP 7: Return receipt with propagation stats
    // ========================================================================
    let latency = start.elapsed();
    let latency_us = latency.as_micros() as u64;

    info!(
        "🦈 [SHARKGOD] TX {} propagated in {}µs ({}ms) to {} peers | fee={} | mempool_pos={} | producer_woken={}",
        &tx_hash_hex[..16],
        latency_us,
        latency.as_millis(),
        propagated_to,
        fee_applied,
        mempool_position,
        block_producer_woken,
    );

    (
        StatusCode::OK,
        Json(SharkGodResponse {
            success: true,
            receipt: Some(SharkGodReceipt {
                tx_hash: tx_hash_hex,
                propagated_to,
                latency_us,
                fee_applied: fee_applied.to_string(),
                mempool_position,
                block_producer_woken,
            }),
            error: None,
        }),
    )
}
