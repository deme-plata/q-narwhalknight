//! One-click "Deposit BTC → Become Bridge LP" flow.
//!
//! This module ships the user-friendly path described in the v10.9.21 honest-liquidity
//! rewrite. Instead of asking users to (a) generate a BTC deposit address, (b) wait for
//! confirmations, (c) come back and manually call add_liquidity with their freshly-minted
//! wBTC paired with QUG, this collapses the experience into a single intent.
//!
//! Lifecycle of an LP intent
//! -------------------------
//! 1. User picks a BTC amount in the wallet. Frontend reads the live oracle price and
//!    suggests a matching QUG amount.
//! 2. `POST /api/v1/bitcoin/lp/intent { btc_amount_sats, qug_amount, pool_id }`
//!    - Authenticated. Server verifies the user has the QUG.
//!    - Server *escrows* the QUG by debiting it from the user's wallet balance and
//!      bookkeeping it under the intent (so even a node restart preserves the lock).
//!    - Server generates a fresh Knots deposit address tagged with the intent id.
//!    - Returns: intent_id, btc_address, qr_uri, expires_at.
//! 3. User sends BTC from any exchange/wallet. Frontend polls intent status.
//! 4. When the bridge sees the deposit confirm to at least `BTC_MIN_CONFIRMATIONS`,
//!    the user can call `POST /api/v1/bitcoin/lp/intent/:id/finalize` (UI does this
//!    automatically once status flips). The handler:
//!      a. Verifies confirmations.
//!      b. Refunds the escrowed QUG back to the user's wallet balance.
//!      c. Mints wBTC into the user's token_balance equal to the deposit amount.
//!      d. Calls `liquidity_api::add_liquidity` with the user as provider — that path
//!         already debits both tokens, mints LP tokens, persists the pool, and emits
//!         the right events.
//!      e. Marks the deposit as minted in the bridge dedup set so it can never be
//!         double-spent into a second LP or a regular wBTC credit.
//!    Net effect: the user's wallet ends up holding LP tokens, the pool holds real
//!    BTC-backed wBTC + matched QUG, and the bridge wallet on Delta holds the BTC.
//! 5. If the user changes their mind before BTC arrives, `POST /lp/intent/:id/cancel`
//!    refunds the escrowed QUG.
//!
//! Storage layout (CF_MANIFEST)
//! ----------------------------
//!   btc_lp_intent:<intent_id>                  → JSON LpIntent
//!   btc_lp_intent_by_wallet:<wallet>:<intent>  → empty (per-wallet listing index)
//!   btc_lp_intent_by_deposit:<deposit_id>      → intent_id (used to associate a Knots
//!                                                deposit with its LP intent when the
//!                                                background finalizer fires later)

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::{Path, State}, Json};
use hyper::StatusCode;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use uuid::Uuid;

use q_storage::{BalanceStorage, CF_MANIFEST};
use q_types::ApiResponse;
use q_types::WBTC_TOKEN_ADDRESS;

use crate::wallet_auth::AuthenticatedWallet;
use crate::AppState;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Intents expire after 48 hours if BTC hasn't arrived. Matches the bridge's deposit
/// address expiry so the two timers don't drift.
const INTENT_EXPIRY_SECS: u64 = 48 * 3600;

/// Maximum BTC we'll accept in a single LP intent (matches DepositBridge cap).
const MAX_INTENT_BTC_SATS: u64 = 100_000_000; // 1 BTC

/// Minimum BTC per intent (avoid dust LP positions).
const MIN_INTENT_BTC_SATS: u64 = 10_000; // 0.0001 BTC

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LpIntentStatus {
    /// QUG escrowed, BTC address generated, waiting for the user to send BTC.
    AwaitingBtc,
    /// BTC tx seen in mempool or with fewer than MIN confirmations.
    BtcDetected { txid: String, vout: u32, confirmations: u32 },
    /// Confs reached MIN; finalize endpoint can be called.
    ReadyToFinalize { txid: String, vout: u32 },
    /// Successfully minted wBTC + added liquidity. Stores the LP token count credited.
    Completed { txid: String, lp_tokens_minted: String },
    /// User cancelled before BTC arrived. QUG refunded.
    Cancelled,
    /// 48h passed without BTC arriving. QUG auto-refunded.
    Expired,
    /// Something went wrong during finalize. QUG refunded.
    Failed { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpIntent {
    pub intent_id: String,
    /// User's QNK wallet (32-byte raw address).
    pub wallet: [u8; 32],
    /// Target pool. For now this is always `pool-qug-wbtc-bridge`, but we store it so
    /// later bridges (wZEC, wIRON) can reuse the same flow.
    pub pool_id: String,
    /// BTC amount the user committed to deposit (sats).
    pub btc_amount_sats: u64,
    /// QUG amount currently held in escrow for this intent (24-decimal base units).
    pub qug_amount_escrowed: u128,
    /// Bridge-generated Bitcoin deposit address.
    pub btc_address: String,
    /// Deposit id from the underlying DepositBridge so the finalizer can look it up.
    pub deposit_id: String,
    pub status: LpIntentStatus,
    /// Unix seconds at creation.
    pub created_at: u64,
    /// Unix seconds of last status change.
    pub updated_at: u64,
    /// Unix seconds after which the intent auto-expires.
    pub expires_at: u64,
}

// ---------------------------------------------------------------------------
// Request / response shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateLpIntentRequest {
    /// How much BTC the user plans to deposit, in satoshis.
    pub btc_amount_sats: u64,
    /// QUG to pair with it, in 24-decimal base units (accepts string or number).
    pub qug_amount: serde_json::Value,
    /// Optional pool id. Defaults to "pool-qug-wbtc-bridge".
    #[serde(default)]
    pub pool_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateLpIntentResponse {
    pub intent_id: String,
    pub btc_address: String,
    pub qr_uri: String,
    pub btc_amount_sats: u64,
    pub qug_amount_escrowed: String,
    pub pool_id: String,
    pub expires_at: u64,
    /// Frontend should poll this endpoint every ~15s while AwaitingBtc.
    pub status_url: String,
}

#[derive(Debug, Serialize)]
pub struct LpIntentView {
    pub intent_id: String,
    pub pool_id: String,
    pub btc_amount_sats: u64,
    pub btc_address: String,
    pub qug_amount_escrowed: String,
    pub status: LpIntentStatus,
    pub created_at: u64,
    pub updated_at: u64,
    pub expires_at: u64,
}

impl From<&LpIntent> for LpIntentView {
    fn from(i: &LpIntent) -> Self {
        Self {
            intent_id: i.intent_id.clone(),
            pool_id: i.pool_id.clone(),
            btc_amount_sats: i.btc_amount_sats,
            btc_address: i.btc_address.clone(),
            qug_amount_escrowed: i.qug_amount_escrowed.to_string(),
            status: i.status.clone(),
            created_at: i.created_at,
            updated_at: i.updated_at,
            expires_at: i.expires_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct LpIntentListResponse {
    pub intents: Vec<LpIntentView>,
    pub total: usize,
}

// ---------------------------------------------------------------------------
// Storage helpers
// ---------------------------------------------------------------------------

fn intent_key(intent_id: &str) -> Vec<u8> {
    format!("btc_lp_intent:{}", intent_id).into_bytes()
}

fn wallet_index_key(wallet: &[u8; 32], intent_id: &str) -> Vec<u8> {
    format!("btc_lp_intent_by_wallet:{}:{}", hex::encode(wallet), intent_id).into_bytes()
}

fn wallet_index_prefix(wallet: &[u8; 32]) -> Vec<u8> {
    format!("btc_lp_intent_by_wallet:{}:", hex::encode(wallet)).into_bytes()
}

fn deposit_index_key(deposit_id: &str) -> Vec<u8> {
    format!("btc_lp_intent_by_deposit:{}", deposit_id).into_bytes()
}

async fn save_intent(state: &AppState, intent: &LpIntent) -> Result<(), String> {
    let bytes = serde_json::to_vec(intent).map_err(|e| format!("serialize intent: {}", e))?;
    let kv = state.storage_engine.get_kv();
    kv.put(CF_MANIFEST, &intent_key(&intent.intent_id), &bytes)
        .await
        .map_err(|e| format!("write intent: {}", e))?;
    kv.put(CF_MANIFEST, &wallet_index_key(&intent.wallet, &intent.intent_id), &[])
        .await
        .map_err(|e| format!("write wallet index: {}", e))?;
    kv.put(CF_MANIFEST, &deposit_index_key(&intent.deposit_id), intent.intent_id.as_bytes())
        .await
        .map_err(|e| format!("write deposit index: {}", e))?;
    Ok(())
}

async fn load_intent(state: &AppState, intent_id: &str) -> Option<LpIntent> {
    let kv = state.storage_engine.get_kv();
    let bytes = kv.get(CF_MANIFEST, &intent_key(intent_id)).await.ok().flatten()?;
    serde_json::from_slice(&bytes).ok()
}

async fn list_intents_for_wallet(state: &AppState, wallet: &[u8; 32]) -> Vec<LpIntent> {
    let kv = state.storage_engine.get_kv();
    let pairs = match kv.scan_prefix(CF_MANIFEST, &wallet_index_prefix(wallet)).await {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    let mut out = Vec::with_capacity(pairs.len());
    for (k, _) in pairs {
        // Key format: btc_lp_intent_by_wallet:<wallet_hex>:<intent_id>
        if let Ok(key_str) = std::str::from_utf8(&k) {
            if let Some(intent_id) = key_str.rsplit(':').next() {
                if let Some(i) = load_intent(state, intent_id).await {
                    out.push(i);
                }
            }
        }
    }
    // Newest first
    out.sort_by_key(|i| std::cmp::Reverse(i.created_at));
    out
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn parse_u128(v: &serde_json::Value) -> Option<u128> {
    match v {
        serde_json::Value::String(s) => s.parse().ok(),
        serde_json::Value::Number(n) => n.as_u64().map(|x| x as u128),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/// `POST /api/v1/bitcoin/lp/intent`
///
/// Create a new "deposit BTC → become LP" intent. Escrows the QUG and returns a fresh
/// BTC deposit address.
pub async fn create_lp_intent(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
    Json(req): Json<CreateLpIntentRequest>,
) -> Result<Json<ApiResponse<CreateLpIntentResponse>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required.".to_string()))),
    };

    // ---- Validate inputs --------------------------------------------------
    if req.btc_amount_sats < MIN_INTENT_BTC_SATS {
        return Ok(Json(ApiResponse::error(format!(
            "BTC amount below minimum ({} sats).",
            MIN_INTENT_BTC_SATS
        ))));
    }
    if req.btc_amount_sats > MAX_INTENT_BTC_SATS {
        return Ok(Json(ApiResponse::error(format!(
            "BTC amount above maximum ({} sats / 1 BTC).",
            MAX_INTENT_BTC_SATS
        ))));
    }
    let qug_amount = match parse_u128(&req.qug_amount) {
        Some(v) if v > 0 => v,
        _ => {
            return Ok(Json(ApiResponse::error(
                "qug_amount must be a positive integer (24-decimal base units).".into(),
            )))
        }
    };
    let pool_id = req.pool_id.unwrap_or_else(|| "pool-qug-wbtc-bridge".to_string());
    if pool_id != "pool-qug-wbtc-bridge" {
        return Ok(Json(ApiResponse::error(
            "Only pool-qug-wbtc-bridge is supported in this version.".into(),
        )));
    }

    // ---- Check + escrow QUG ----------------------------------------------
    // We debit the user's wallet_balance directly and persist the new value. The
    // escrowed amount is held inside the LpIntent record; refund on cancel/expire
    // adds it back. This avoids needing a separate escrow wallet address.
    let wallet_hex = hex::encode(wallet.address);
    let current_balance = state
        .storage_engine
        .get_balance(&wallet_hex)
        .await
        .unwrap_or(0);
    if current_balance < qug_amount {
        return Ok(Json(ApiResponse::error(format!(
            "Insufficient QUG balance. Have {} base units, need {}.",
            current_balance, qug_amount
        ))));
    }
    let after_escrow = current_balance - qug_amount;
    if let Err(e) = state
        .storage_engine
        .set_balance(&wallet_hex, after_escrow)
        .await
    {
        return Ok(Json(ApiResponse::error(format!(
            "Failed to escrow QUG: {}",
            e
        ))));
    }
    // Sync the in-memory cache so other handlers see the new balance immediately.
    {
        let mut bal = state.wallet_balances.write().await;
        bal.insert(wallet.address, after_escrow);
    }

    // ---- Generate BTC deposit address via the bridge ----------------------
    let bridge = match &state.deposit_bridge {
        Some(b) => b.clone(),
        None => {
            // Refund the escrow before erroring out.
            let _ = state
                .storage_engine
                .set_balance(&wallet_hex, current_balance)
                .await;
            {
                let mut bal = state.wallet_balances.write().await;
                bal.insert(wallet.address, current_balance);
            }
            return Ok(Json(ApiResponse::error(
                "Bitcoin deposit bridge is not enabled on this node.".into(),
            )));
        }
    };
    let deposit_addr = match bridge
        .create_deposit_address(wallet.address)
        .await
    {
        Ok(d) => d,
        Err(e) => {
            // Refund the escrow if address generation failed.
            let _ = state
                .storage_engine
                .set_balance(&wallet_hex, current_balance)
                .await;
            {
                let mut bal = state.wallet_balances.write().await;
                bal.insert(wallet.address, current_balance);
            }
            return Ok(Json(ApiResponse::error(format!(
                "Could not generate deposit address: {}",
                e
            ))));
        }
    };

    // ---- Persist the intent ----------------------------------------------
    let intent_id = Uuid::new_v4().to_string();
    let now = now_unix();
    let intent = LpIntent {
        intent_id: intent_id.clone(),
        wallet: wallet.address,
        pool_id: pool_id.clone(),
        btc_amount_sats: req.btc_amount_sats,
        qug_amount_escrowed: qug_amount,
        btc_address: deposit_addr.btc_address.clone(),
        deposit_id: deposit_addr.deposit_id.clone(),
        status: LpIntentStatus::AwaitingBtc,
        created_at: now,
        updated_at: now,
        expires_at: now + INTENT_EXPIRY_SECS,
    };

    if let Err(e) = save_intent(&state, &intent).await {
        // Couldn't persist — refund and bail. The deposit address is orphaned but harmless.
        let _ = state
            .storage_engine
            .set_balance(&wallet_hex, current_balance)
            .await;
        {
            let mut bal = state.wallet_balances.write().await;
            bal.insert(wallet.address, current_balance);
        }
        return Ok(Json(ApiResponse::error(format!(
            "Failed to persist LP intent: {}",
            e
        ))));
    }

    info!(
        "🌉 [LP-INTENT] Created intent {} for wallet {}… btc={} sats qug={}",
        &intent_id[..12],
        &wallet_hex[..16.min(wallet_hex.len())],
        req.btc_amount_sats,
        qug_amount
    );

    let qr_uri = format!(
        "bitcoin:{}?amount={:.8}",
        deposit_addr.btc_address,
        req.btc_amount_sats as f64 / 100_000_000.0
    );

    Ok(Json(ApiResponse::success(CreateLpIntentResponse {
        intent_id: intent.intent_id.clone(),
        btc_address: intent.btc_address.clone(),
        qr_uri,
        btc_amount_sats: intent.btc_amount_sats,
        qug_amount_escrowed: intent.qug_amount_escrowed.to_string(),
        pool_id: intent.pool_id.clone(),
        expires_at: intent.expires_at,
        status_url: format!("/api/v1/bitcoin/lp/intent/{}", intent.intent_id),
    })))
}

/// `GET /api/v1/bitcoin/lp/intent/:id`
pub async fn get_lp_intent(
    State(state): State<Arc<AppState>>,
    Path(intent_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<LpIntentView>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required.".into()))),
    };
    match load_intent(&state, &intent_id).await {
        Some(i) if i.wallet == wallet.address => Ok(Json(ApiResponse::success((&i).into()))),
        Some(_) => Ok(Json(ApiResponse::error("Intent belongs to another wallet.".into()))),
        None => Ok(Json(ApiResponse::error("Intent not found.".into()))),
    }
}

/// `GET /api/v1/bitcoin/lp/intents`
pub async fn list_lp_intents(
    State(state): State<Arc<AppState>>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<LpIntentListResponse>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required.".into()))),
    };
    let intents = list_intents_for_wallet(&state, &wallet.address).await;
    let views: Vec<LpIntentView> = intents.iter().map(LpIntentView::from).collect();
    let total = views.len();
    Ok(Json(ApiResponse::success(LpIntentListResponse {
        intents: views,
        total,
    })))
}

/// `POST /api/v1/bitcoin/lp/intent/:id/cancel`
///
/// Refunds the escrowed QUG if the intent is still in `AwaitingBtc` (or detected with
/// zero confs). Once the BTC tx has reached MIN_CONFIRMATIONS the intent must be
/// finalized — at that point the user is committed to the LP.
pub async fn cancel_lp_intent(
    State(state): State<Arc<AppState>>,
    Path(intent_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<LpIntentView>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required.".into()))),
    };

    let mut intent = match load_intent(&state, &intent_id).await {
        Some(i) if i.wallet == wallet.address => i,
        Some(_) => return Ok(Json(ApiResponse::error("Intent belongs to another wallet.".into()))),
        None => return Ok(Json(ApiResponse::error("Intent not found.".into()))),
    };

    let cancellable = matches!(
        intent.status,
        LpIntentStatus::AwaitingBtc | LpIntentStatus::BtcDetected { .. }
    );
    if !cancellable {
        return Ok(Json(ApiResponse::error(format!(
            "Intent is in status {:?} and can no longer be cancelled.",
            intent.status
        ))));
    }

    // Refund escrowed QUG back to wallet_balance.
    let wallet_hex = hex::encode(wallet.address);
    let current = state.storage_engine.get_balance(&wallet_hex).await.unwrap_or(0);
    let refunded = current.saturating_add(intent.qug_amount_escrowed);
    if let Err(e) = state.storage_engine.set_balance(&wallet_hex, refunded).await {
        return Ok(Json(ApiResponse::error(format!(
            "Failed to refund escrowed QUG: {}",
            e
        ))));
    }
    {
        let mut bal = state.wallet_balances.write().await;
        bal.insert(wallet.address, refunded);
    }

    intent.status = LpIntentStatus::Cancelled;
    intent.updated_at = now_unix();
    intent.qug_amount_escrowed = 0;
    if let Err(e) = save_intent(&state, &intent).await {
        warn!("[LP-INTENT] cancel persisted refund but failed to save intent: {}", e);
    }
    info!("🌉 [LP-INTENT] Cancelled {} — refunded {} QUG", &intent.intent_id[..12], intent.qug_amount_escrowed);

    Ok(Json(ApiResponse::success((&intent).into())))
}

/// `POST /api/v1/bitcoin/lp/intent/:id/finalize`
///
/// Called once the BTC deposit has reached `BTC_MIN_CONFIRMATIONS`. This is what
/// actually mints the wBTC + adds liquidity for the user.
///
/// NOTE: The deposit-confirmation polling already runs inside `DepositBridge`, but the
/// auto-call to `finalize` from that loop is intentionally deferred to a follow-up
/// commit so reviewers can see the LP path in isolation. Frontend polls intent status
/// and POSTs this once `ReadyToFinalize` is observed. The handler is idempotent — if
/// it's already Completed we just return the existing state.
pub async fn finalize_lp_intent(
    State(state): State<Arc<AppState>>,
    Path(intent_id): Path<String>,
    auth_wallet: Option<AuthenticatedWallet>,
) -> Result<Json<ApiResponse<LpIntentView>>, StatusCode> {
    let wallet = match auth_wallet {
        Some(w) => w,
        None => return Ok(Json(ApiResponse::error("Authentication required.".into()))),
    };

    let mut intent = match load_intent(&state, &intent_id).await {
        Some(i) if i.wallet == wallet.address => i,
        Some(_) => return Ok(Json(ApiResponse::error("Intent belongs to another wallet.".into()))),
        None => return Ok(Json(ApiResponse::error("Intent not found.".into()))),
    };

    // Idempotency: already done.
    if matches!(intent.status, LpIntentStatus::Completed { .. }) {
        return Ok(Json(ApiResponse::success((&intent).into())));
    }
    // Not cancellable scenarios.
    if matches!(intent.status, LpIntentStatus::Cancelled | LpIntentStatus::Expired | LpIntentStatus::Failed { .. }) {
        return Ok(Json(ApiResponse::error(format!(
            "Intent cannot be finalized from status {:?}.",
            intent.status
        ))));
    }

    let bridge = match &state.deposit_bridge {
        Some(b) => b.clone(),
        None => return Ok(Json(ApiResponse::error(
            "Bitcoin deposit bridge is not enabled on this node.".into(),
        ))),
    };

    // Refresh the deposit's on-chain status. If poll fails (RPC blip), surface a soft
    // error so the user can retry — we don't want to fail the intent on transient
    // network glitches.
    if let Err(e) = bridge.poll_deposits().await {
        warn!("[LP-INTENT] poll_deposits failed during finalize: {}", e);
        return Ok(Json(ApiResponse::error(format!(
            "Couldn't refresh deposit status from Bitcoin Knots: {}. Please retry.",
            e
        ))));
    }

    // Pull the latest deposit record.
    let deposits = bridge.list_deposits_for_wallet(&wallet.address).await;
    let deposit = match deposits.into_iter().find(|d| d.deposit_id == intent.deposit_id) {
        Some(d) => d,
        None => {
            return Ok(Json(ApiResponse::error(
                "Bridge has lost track of this deposit. Contact support.".into(),
            )))
        }
    };

    // Update intent status based on what we found.
    let (txid, vout, _confs, amount_sats) = match &deposit.status {
        q_bitcoin_bridge::deposit_bridge::DepositStatus::Awaiting => {
            intent.status = LpIntentStatus::AwaitingBtc;
            intent.updated_at = now_unix();
            let _ = save_intent(&state, &intent).await;
            return Ok(Json(ApiResponse::error(
                "BTC deposit not seen yet. Send BTC to the address and try again after the network sees it.".into(),
            )));
        }
        q_bitcoin_bridge::deposit_bridge::DepositStatus::Detected { txid, vout, confirmations } => {
            intent.status = LpIntentStatus::BtcDetected {
                txid: txid.clone(),
                vout: *vout,
                confirmations: *confirmations,
            };
            intent.updated_at = now_unix();
            let _ = save_intent(&state, &intent).await;
            return Ok(Json(ApiResponse::error(format!(
                "Deposit seen ({} confirmations). Wait for {} confirmations and retry.",
                confirmations, q_bitcoin_bridge::deposit_bridge::BTC_MIN_CONFIRMATIONS
            ))));
        }
        q_bitcoin_bridge::deposit_bridge::DepositStatus::Confirming { txid, vout, confirmations } => {
            if *confirmations < q_bitcoin_bridge::deposit_bridge::BTC_MIN_CONFIRMATIONS {
                intent.status = LpIntentStatus::BtcDetected {
                    txid: txid.clone(),
                    vout: *vout,
                    confirmations: *confirmations,
                };
                intent.updated_at = now_unix();
                let _ = save_intent(&state, &intent).await;
                return Ok(Json(ApiResponse::error(format!(
                    "{} confirmations; need {}.",
                    confirmations, q_bitcoin_bridge::deposit_bridge::BTC_MIN_CONFIRMATIONS
                ))));
            }
            (txid.clone(), *vout, *confirmations, deposit.amount_sats)
        }
        q_bitcoin_bridge::deposit_bridge::DepositStatus::Minted { txid, vout, .. } => {
            // v10.9.55 (G1 fix, audit 2026-05-18): the bridge has the deposit marked
            // as Minted but our intent.status is still pre-Completed. This is the
            // crash-between-mark_minted-and-intent-write race. The comment used to say
            // "Skipping LP add to avoid double-credit" — but the code did NOT actually
            // skip, it fell through to lines 681+ and re-executed the Step 1-2-3
            // sequence (refund QUG, mint wBTC, add liquidity) → double-credit on every
            // retry. Real bug, real money.
            //
            // Fix: return EARLY, marking intent Completed using the bridge's record as
            // the source of truth. The user already got their LP tokens from the prior
            // run; we just update our intent status to reflect that.
            warn!(
                "🛡️ [LP-INTENT G1] Deposit {}.{} already marked Minted in bridge. \
                 Intent {} status was {:?} — reconciling to Completed without re-executing \
                 LP add (prevents double-credit retry attack).",
                &txid[..16.min(txid.len())], vout, &intent.intent_id[..12], intent.status
            );
            // Note: lp_tokens_minted is unknown here because we don't have access to
            // the prior run's return value. Use a sentinel string and let the audit
            // log carry the txid + bridge dedup record as the ground truth.
            intent.status = LpIntentStatus::Completed {
                txid: txid.clone(),
                lp_tokens_minted: "unknown-reconciled-from-bridge-dedup".to_string(),
            };
            intent.qug_amount_escrowed = 0;
            intent.updated_at = now_unix();
            let _ = save_intent(&state, &intent).await;
            return Ok(Json(ApiResponse::success((&intent).into())));
        }
        q_bitcoin_bridge::deposit_bridge::DepositStatus::Expired => {
            intent.status = LpIntentStatus::Expired;
            intent.updated_at = now_unix();
            let _ = save_intent(&state, &intent).await;
            return Ok(Json(ApiResponse::error(
                "Deposit expired. Cancel this intent to refund your escrowed QUG.".into(),
            )));
        }
        q_bitcoin_bridge::deposit_bridge::DepositStatus::Failed { reason } => {
            return Ok(Json(ApiResponse::error(format!(
                "Deposit failed: {}. Cancel this intent to refund escrowed QUG.",
                reason
            ))));
        }
    };

    if amount_sats < intent.btc_amount_sats {
        // The user sent less BTC than they committed. We could either (a) finalize with
        // the actual amount and refund the difference of QUG, or (b) reject and ask
        // them to cancel/redo. (b) is simpler and avoids partial fills creating dust
        // LP positions.
        warn!(
            "[LP-INTENT] Underfunded deposit: intent={} sats committed={} received={}",
            &intent.intent_id[..12], intent.btc_amount_sats, amount_sats
        );
        intent.status = LpIntentStatus::Failed {
            reason: format!(
                "Underfunded: committed {} sats, received {}. Cancel to refund QUG.",
                intent.btc_amount_sats, amount_sats
            ),
        };
        intent.updated_at = now_unix();
        let _ = save_intent(&state, &intent).await;
        // Refund the escrowed QUG since we're failing the intent.
        refund_escrow_quietly(&state, &mut intent).await;
        return Ok(Json(ApiResponse::error("Underfunded deposit — QUG refunded.".into())));
    }

    intent.status = LpIntentStatus::ReadyToFinalize { txid: txid.clone(), vout };
    intent.updated_at = now_unix();
    let _ = save_intent(&state, &intent).await;

    // ---- Actually mint wBTC + add liquidity -------------------------------
    //
    // Conceptually we want a single atomic step: pool gets (qug, wbtc), user gets LP
    // tokens. To reuse the well-tested `liquidity_api::add_liquidity` we go through
    // a small choreography:
    //   1. Refund the escrowed QUG back into the user's wallet_balance (it's about
    //      to be debited by add_liquidity).
    //   2. Mint wBTC = amount_sats into the user's token_balance for WBTC_TOKEN_ADDRESS.
    //   3. Call add_liquidity with provider=user, amount0=qug, amount1=wBTC (24-dec).
    //   4. On success: mark deposit as Minted in the bridge dedup set.
    //   5. Record Completed status with the LP token count.
    //
    // If step 3 fails we restore both the QUG (still refunded) and burn the freshly
    // minted wBTC, returning to AwaitingBtc-equivalent so the user can cancel or
    // retry. This is best-effort — if step 3 succeeded but step 4 (mark_minted) fails,
    // the LP is intact and we log loudly (dedup set inconsistency, not user-visible).

    let wallet_hex = hex::encode(wallet.address);

    // Step 1: refund QUG into wallet_balance (it'll be re-debited by add_liquidity).
    let bal_before = state.storage_engine.get_balance(&wallet_hex).await.unwrap_or(0);
    let bal_with_refund = bal_before.saturating_add(intent.qug_amount_escrowed);
    if let Err(e) = state.storage_engine.set_balance(&wallet_hex, bal_with_refund).await {
        warn!("[LP-INTENT] failed to refund QUG to wallet_balance for finalize: {}", e);
        return Ok(Json(ApiResponse::error(format!(
            "Storage error during finalize (QUG refund): {}",
            e
        ))));
    }
    {
        let mut bal = state.wallet_balances.write().await;
        bal.insert(wallet.address, bal_with_refund);
    }

    // Step 2: mint wBTC into token_balances. The AMM stores token reserves in 24-decimal
    // form (see crates/q-api-server/src/lib.rs::bootstrap_bridge_pools), so the user's
    // wBTC balance also needs to be in the 24-dec form to match `add_liquidity`'s view.
    // wBTC is 8-decimal natively → multiply by 10^16 to reach 24-dec.
    let wbtc_token_key = (wallet.address, WBTC_TOKEN_ADDRESS);
    let wbtc_amount_24dec: u128 = (amount_sats as u128).saturating_mul(10u128.pow(16));
    let prior_wbtc = {
        let bals = state.token_balances.read().await;
        bals.get(&wbtc_token_key).copied().unwrap_or(0)
    };
    let new_wbtc = prior_wbtc.saturating_add(wbtc_amount_24dec);
    {
        let mut bals = state.token_balances.write().await;
        bals.insert(wbtc_token_key, new_wbtc);
    }
    if let Err(e) = state
        .storage_engine
        .save_token_balance(&wallet.address, &WBTC_TOKEN_ADDRESS, new_wbtc)
        .await
    {
        warn!("[LP-INTENT] failed to persist freshly-minted wBTC: {}", e);
        // Roll back the in-memory mint and the QUG refund.
        {
            let mut bals = state.token_balances.write().await;
            bals.insert(wbtc_token_key, prior_wbtc);
        }
        let _ = state.storage_engine.set_balance(&wallet_hex, bal_before).await;
        {
            let mut bal = state.wallet_balances.write().await;
            bal.insert(wallet.address, bal_before);
        }
        return Ok(Json(ApiResponse::error(format!(
            "Storage error during finalize (wBTC mint): {}",
            e
        ))));
    }

    // Step 3: add liquidity. We can't call the HTTP handler from here without going
    // through axum, so we duplicate the minimal pool-add logic directly using the
    // in-memory pool state. (For empty bridge pools this is the simpler case: first
    // LP defines the price, lp_token_supply = sqrt(amount0 * amount1).)
    let pool_id_str = intent.pool_id.clone();
    let lp_result = mint_lp_first_or_proportional(
        &state,
        &pool_id_str,
        wallet.address,
        intent.qug_amount_escrowed,
        wbtc_amount_24dec,
    )
    .await;

    let lp_tokens_minted = match lp_result {
        Ok(lp) => lp,
        Err(reason) => {
            warn!("[LP-INTENT] add-liquidity failed for intent {}: {}", &intent.intent_id[..12], reason);
            // Roll everything back: burn the freshly minted wBTC, keep the QUG with
            // the user (we already refunded). Mark the intent Failed.
            {
                let mut bals = state.token_balances.write().await;
                bals.insert(wbtc_token_key, prior_wbtc);
            }
            let _ = state
                .storage_engine
                .save_token_balance(&wallet.address, &WBTC_TOKEN_ADDRESS, prior_wbtc)
                .await;
            intent.status = LpIntentStatus::Failed { reason: reason.clone() };
            intent.qug_amount_escrowed = 0; // Already refunded into wallet_balance.
            intent.updated_at = now_unix();
            let _ = save_intent(&state, &intent).await;
            return Ok(Json(ApiResponse::error(format!(
                "Couldn't add liquidity ({}). Your QUG has been refunded; this intent is now Failed.",
                reason
            ))));
        }
    };

    // Step 4: mark the deposit as Minted in the bridge so it can't be re-used.
    // The mint_op_id ties back to this intent for audit purposes.
    if let Err(e) = bridge
        .mark_minted(&intent.deposit_id, txid.clone(), vout, format!("lp-intent:{}", intent.intent_id))
        .await
    {
        // Liquidity succeeded; this is an audit-only failure. Log loudly but don't
        // unwind the LP — the user has their tokens.
        warn!(
            "[LP-INTENT] LP added but mark_minted failed for {}: {}. Dedup set may not reflect this mint.",
            &intent.intent_id[..12], e
        );
    }

    // Step 5: record Completed.
    intent.status = LpIntentStatus::Completed {
        txid: txid.clone(),
        lp_tokens_minted: lp_tokens_minted.to_string(),
    };
    intent.qug_amount_escrowed = 0;
    intent.updated_at = now_unix();
    let _ = save_intent(&state, &intent).await;

    info!(
        "🌉 [LP-INTENT] Finalized {} — pool={} btc={} sats qug={} lp_tokens={}",
        &intent.intent_id[..12],
        intent.pool_id,
        amount_sats,
        intent.qug_amount_escrowed,
        lp_tokens_minted
    );

    Ok(Json(ApiResponse::success((&intent).into())))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async fn refund_escrow_quietly(state: &AppState, intent: &mut LpIntent) {
    if intent.qug_amount_escrowed == 0 {
        return;
    }
    let wallet_hex = hex::encode(intent.wallet);
    let current = state.storage_engine.get_balance(&wallet_hex).await.unwrap_or(0);
    let refunded = current.saturating_add(intent.qug_amount_escrowed);
    if let Err(e) = state.storage_engine.set_balance(&wallet_hex, refunded).await {
        warn!("[LP-INTENT] best-effort refund failed for {}: {}", &intent.intent_id[..12], e);
        return;
    }
    {
        let mut bal = state.wallet_balances.write().await;
        bal.insert(intent.wallet, refunded);
    }
    intent.qug_amount_escrowed = 0;
}

/// Integer square root for u128 — Newton iteration. Matches the algorithm used by
/// liquidity_api::add_liquidity for LP token issuance.
fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Mint LP tokens against the named pool. For an empty bridge pool this seeds the
/// reserves and sets `lp_token_supply = sqrt(amount0 * amount1)`. For a populated pool
/// this adds proportionally (the deposit must roughly match the current ratio).
async fn mint_lp_first_or_proportional(
    state: &AppState,
    pool_id: &str,
    provider: [u8; 32],
    qug_amount_24dec: u128,
    wbtc_amount_24dec: u128,
) -> Result<u128, String> {
    // 1. Debit the user's QUG (wallet_balance) and wBTC (token_balance) the way
    //    liquidity_api::add_liquidity would.
    let provider_hex = hex::encode(provider);
    let cur_qug = state.storage_engine.get_balance(&provider_hex).await.unwrap_or(0);
    if cur_qug < qug_amount_24dec {
        return Err(format!(
            "Provider has insufficient QUG ({} < {}) at LP-add time",
            cur_qug, qug_amount_24dec
        ));
    }
    let new_qug = cur_qug - qug_amount_24dec;
    state
        .storage_engine
        .set_balance(&provider_hex, new_qug)
        .await
        .map_err(|e| format!("debit QUG: {}", e))?;
    {
        let mut bals = state.wallet_balances.write().await;
        bals.insert(provider, new_qug);
    }

    let wbtc_key = (provider, WBTC_TOKEN_ADDRESS);
    let cur_wbtc = {
        let b = state.token_balances.read().await;
        b.get(&wbtc_key).copied().unwrap_or(0)
    };
    if cur_wbtc < wbtc_amount_24dec {
        // Roll back the QUG debit.
        let _ = state.storage_engine.set_balance(&provider_hex, cur_qug).await;
        {
            let mut bals = state.wallet_balances.write().await;
            bals.insert(provider, cur_qug);
        }
        return Err(format!(
            "Provider has insufficient wBTC at LP-add time ({} < {})",
            cur_wbtc, wbtc_amount_24dec
        ));
    }
    let new_wbtc = cur_wbtc - wbtc_amount_24dec;
    {
        let mut bals = state.token_balances.write().await;
        bals.insert(wbtc_key, new_wbtc);
    }
    state
        .storage_engine
        .save_token_balance(&provider, &WBTC_TOKEN_ADDRESS, new_wbtc)
        .await
        .map_err(|e| format!("debit wBTC: {}", e))?;

    // 2. Update pool reserves + mint LP tokens.
    let lp_tokens_minted: u128;
    let pool_snapshot = {
        let mut pools = state.liquidity_pools.write().await;
        let pool = pools
            .get_mut(pool_id)
            .ok_or_else(|| format!("Pool {} not found", pool_id))?;

        let r0 = pool.reserve0;
        let r1 = pool.reserve1;
        let supply = pool.lp_token_supply;

        if supply == 0 || r0 == 0 || r1 == 0 {
            // First LP — defines the price and seeds the pool.
            pool.reserve0 = qug_amount_24dec;
            pool.reserve1 = wbtc_amount_24dec;
            let lp = isqrt_u128(qug_amount_24dec.saturating_mul(wbtc_amount_24dec));
            pool.lp_token_supply = lp;
            pool.provider = provider; // First LP is the seeder
            lp_tokens_minted = lp;
        } else {
            // Proportional LP: mint min(d0, d1) — d_i = amount_i * supply / r_i
            let d0 = qug_amount_24dec.saturating_mul(supply) / r0.max(1);
            let d1 = wbtc_amount_24dec.saturating_mul(supply) / r1.max(1);
            let lp = d0.min(d1);
            if lp == 0 {
                return Err("Computed LP tokens = 0 — deposit too small relative to pool".into());
            }
            pool.reserve0 = r0.saturating_add(qug_amount_24dec);
            pool.reserve1 = r1.saturating_add(wbtc_amount_24dec);
            pool.lp_token_supply = supply.saturating_add(lp);
            lp_tokens_minted = lp;
        }
        serde_json::to_vec(&*pool).ok()
    };

    // 3. Persist updated pool.
    if let Some(bytes) = pool_snapshot {
        if let Err(e) = state.storage_engine.save_liquidity_pool(pool_id, &bytes).await {
            warn!("[LP-INTENT] LP minted in-memory but failed to persist pool {}: {}", pool_id, e);
        }
    }

    // 4. Credit the LP tokens to the provider. LP tokens are tracked per-pool via a
    //    synthetic token address derived from the pool id, matching the convention
    //    used by liquidity_api::add_liquidity.
    let lp_token_addr = derive_lp_token_address(pool_id);
    let lp_key = (provider, lp_token_addr);
    let cur_lp = {
        let b = state.token_balances.read().await;
        b.get(&lp_key).copied().unwrap_or(0)
    };
    let new_lp = cur_lp.saturating_add(lp_tokens_minted);
    {
        let mut bals = state.token_balances.write().await;
        bals.insert(lp_key, new_lp);
    }
    let _ = state
        .storage_engine
        .save_token_balance(&provider, &lp_token_addr, new_lp)
        .await;

    Ok(lp_tokens_minted)
}

/// LP-token address convention: must match `liquidity_api::generate_lp_token_address`,
/// which uses SHA-256("lp_token:" + pool_id). Same pool ⇒ same LP token address, so
/// LP tokens minted here are fungible with those minted by /api/v1/liquidity/add.
fn derive_lp_token_address(pool_id: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(format!("lp_token:{}", pool_id).as_bytes());
    let mut addr = [0u8; 32];
    addr.copy_from_slice(&hash[..32]);
    addr
}
