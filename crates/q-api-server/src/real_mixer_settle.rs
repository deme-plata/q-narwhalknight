//! Real, atomic, double-spend-safe mixer settlement endpoint.
//!
//! This is the honest counterpart to the fake `/api/v1/mixer/send` theatre. It runs
//! the crate's real LSAG ring-signature verification + Pedersen commitment binding
//! (`q_quantum_mixing::settlement::verify_spend`) and only then moves balances —
//! atomically, recording the spend's key image so a replay or concurrent duplicate
//! is rejected as a double-spend instead of minting a second credit.
//!
//! ## Scope notes (be honest about what is and is not wired)
//! - The spent-key-image set here is process-global and NOT yet persisted across
//!   restarts, so a restart currently clears double-spend history. Persisting it in
//!   a `q_storage` column family, committed in the same batch as the balance write,
//!   is the required follow-up before this fronts real value. Balances ARE persisted
//!   (via `storage_engine.save_wallet_balances`), as the rest of the server does.
//! - The caller must supply a valid ring signature. Producing it — server-side
//!   custodial Ristretto signing, or a wallet-client upgrade — is the remaining
//!   piece to point the consumer "send" button at this endpoint. Until then this is
//!   the real settlement primitive, live-wired to the actual balance store.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::LazyLock;

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use q_quantum_mixing::clsag::CLSAGSignature;
use q_quantum_mixing::settlement::{verify_spend, SettlementRequest};
use q_types::ApiResponse;

use crate::AppState;

/// Process-global set of spent key images. Accessed only while the caller holds the
/// wallet-balances write lock, so the check-and-insert is atomic with respect to
/// concurrent settlements. NOT persisted across restart yet (see module docs).
static SPENT_KEY_IMAGES: LazyLock<RwLock<HashSet<[u8; 32]>>> =
    LazyLock::new(|| RwLock::new(HashSet::new()));

/// Fee policy for the real mixer. Zero for now: the fake path charged 0.1% for a
/// service it never performed; a real fee is a separate product decision.
const MIXING_FEE: u64 = 0;

/// Treasury/fee account. Placeholder — wire to the real treasury before charging.
const FEE_ACCOUNT: [u8; 32] = [0xFE; 32];

#[derive(Debug, Deserialize)]
pub struct MixSettleRequest {
    /// 32-byte hex (optionally `qnk`-prefixed) sender account to debit.
    pub sender: String,
    /// 32-byte hex recipient account to credit.
    pub recipient: String,
    /// Amount in atomic units.
    pub amount: u64,
    /// 32-byte hex per-transfer nonce (bound into the signature).
    pub nonce: String,
    /// 32-byte hex Pedersen blinding for the amount commitment.
    pub mask: String,
    /// The LSAG ring signature authorizing the spend (serde).
    pub signature: CLSAGSignature,
}

#[derive(Debug, Serialize)]
pub struct MixSettleResponse {
    pub settled: bool,
    pub key_image: String,
    pub amount: u64,
    pub fee: u64,
    pub sender_balance: u128,
    pub recipient_balance: u128,
}

fn parse_addr(s: &str) -> Option<[u8; 32]> {
    let hex_part = s.strip_prefix("qnk").unwrap_or(s);
    let bytes = hex::decode(hex_part).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut a = [0u8; 32];
    a.copy_from_slice(&bytes);
    Some(a)
}

/// `POST /api/v1/mixer/settle` — verify a mixed transfer's cryptography and settle
/// it atomically against the real balance store.
pub async fn mixer_settle(
    State(state): State<Arc<AppState>>,
    Json(body): Json<MixSettleRequest>,
) -> Result<Json<ApiResponse<MixSettleResponse>>, StatusCode> {
    let (sender, recipient, nonce, mask) = match (
        parse_addr(&body.sender),
        parse_addr(&body.recipient),
        parse_addr(&body.nonce),
        parse_addr(&body.mask),
    ) {
        (Some(s), Some(r), Some(n), Some(m)) => (s, r, n, m),
        _ => {
            return Ok(Json(ApiResponse::error(
                "sender/recipient/nonce/mask must each be 32-byte hex".to_string(),
            )))
        }
    };

    let req = SettlementRequest {
        sender,
        recipient,
        amount: body.amount,
        nonce,
        mask,
        signature: body.signature,
    };

    // Step 1: cryptography only (LSAG ring verify + commitment binding). No ledger
    // I/O. Exactly the checks the in-crate reference `Settler` runs.
    let verified = match verify_spend(&req) {
        Ok(v) => v,
        Err(e) => {
            return Ok(Json(ApiResponse::error(format!(
                "verification failed: {e}"
            ))))
        }
    };

    let fee = MIXING_FEE;
    let total = verified.amount as u128 + fee as u128;

    // Step 2: atomic settle. The balance write lock is held across the key-image
    // check + insert + balance apply, so a replay or concurrent duplicate fails as
    // a double-spend instead of applying a second credit.
    let (sender_balance, recipient_balance) = {
        let mut balances = state.wallet_balances.write().await;
        let mut spent = SPENT_KEY_IMAGES.write().await;

        if spent.contains(&verified.key_image) {
            return Ok(Json(ApiResponse::error(
                "double-spend: key image already recorded".to_string(),
            )));
        }

        let sender_bal = *balances.get(&sender).unwrap_or(&0);
        if sender_bal < total {
            return Ok(Json(ApiResponse::error(format!(
                "insufficient funds: have {sender_bal}, need {total}"
            ))));
        }

        // Conservation: debit (amount+fee) from sender, credit amount to recipient,
        // credit fee to treasury. The three deltas sum to zero.
        let new_sender = sender_bal - total; // safe: checked sender_bal >= total above
        let recip_bal = *balances.get(&recipient).unwrap_or(&0);
        let new_recip = match recip_bal.checked_add(verified.amount as u128) {
            Some(v) => v,
            None => {
                return Ok(Json(ApiResponse::error(
                    "recipient balance overflow".to_string(),
                )))
            }
        };

        if fee > 0 {
            let fee_bal = *balances.get(&FEE_ACCOUNT).unwrap_or(&0);
            let new_fee = match fee_bal.checked_add(fee as u128) {
                Some(v) => v,
                None => {
                    return Ok(Json(ApiResponse::error(
                        "fee account overflow".to_string(),
                    )))
                }
            };
            balances.insert(FEE_ACCOUNT, new_fee);
        }
        balances.insert(sender, new_sender);
        balances.insert(recipient, new_recip);
        spent.insert(verified.key_image);

        (new_sender, new_recip)
    };

    // Persist balances (best-effort), mirroring the rest of the server. Key-image
    // persistence is the documented follow-up.
    let snapshot = state.wallet_balances.read().await.clone();
    if let Err(e) = state.storage_engine.save_wallet_balances(&snapshot).await {
        tracing::warn!("mixer_settle: balance persist failed: {e}");
    }

    tracing::info!(
        "✅ mixer_settle: moved {} atomic units, key image {} recorded",
        verified.amount,
        hex::encode(verified.key_image)
    );

    Ok(Json(ApiResponse::success(MixSettleResponse {
        settled: true,
        key_image: hex::encode(verified.key_image),
        amount: verified.amount,
        fee,
        sender_balance,
        recipient_balance,
    })))
}
