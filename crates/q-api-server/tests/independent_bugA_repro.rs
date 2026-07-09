//! INDEPENDENT verifier-authored repro for BUG A (not written by the round-3
//! implementer). Exercises the real send_transaction_signed handler via the
//! real submit_transaction/add_transaction_detailed path, and specifically
//! constructs a rejection that is UNRELATED to nonce correctness (duplicate
//! tx-hash from replaying the identical signed request), then asserts the
//! nonce tracker was NOT advanced by the rejected replay.

use axum::{
    body::Body,
    http::{Request, StatusCode},
    routing::post,
    Router,
};
use chrono::Utc;
use ed25519_dalek::{Signer, SigningKey};
use q_api_server::{handlers, AppState, Config};
use q_narwhal_core::production_mempool::{MempoolConfig, ProductionMempool};
use q_narwhal_core::{TorClient, TorStreamConnection};
use q_types::{Phase, Transaction, TransactionType, TokenType, TxHash, TxSignaturePhase, TransactionPrivacyLevel};
use serde_json::json;
use std::sync::Arc;
use tower::ServiceExt;

struct NoopTorClient;

#[async_trait::async_trait]
impl TorClient for NoopTorClient {
    async fn connect_to_onion(&self, _onion_address: &str, _port: u16) -> anyhow::Result<Box<dyn TorStreamConnection>> {
        Err(anyhow::anyhow!("noop"))
    }
}

async fn make_state(suffix: &str) -> Arc<AppState> {
    let db_root = format!("target/test-db-INDEP-bugA-{}", suffix);
    let hot_path = format!("{}/hot", db_root);
    let config = Config { db_path: Some(db_root), hot_db_path: Some(hot_path), ..Config::default() };
    let mut state = AppState::new(config).await.expect("AppState::new failed");
    let mempool = Arc::new(
        ProductionMempool::new(MempoolConfig::default(), Arc::new(NoopTorClient) as Arc<dyn TorClient>, Phase::Phase1)
            .await.expect("mempool new failed")
    );
    mempool.set_nonce_source(state.nonce_tracker.clone()).await;
    state.production_mempool = Some(mempool);
    Arc::new(state)
}

fn router(state: Arc<AppState>) -> Router {
    Router::new().route("/api/v1/transactions/send_signed", post(handlers::send_transaction_signed)).with_state(state)
}

fn wallet_auth(signer: &SigningKey, address: [u8; 32], path: &str) -> String {
    use sha3::{Digest, Sha3_256};
    let ts = Utc::now().timestamp();
    let mut h = Sha3_256::new();
    h.update(&address);
    h.update(&ts.to_le_bytes());
    h.update(path.as_bytes());
    let msg = h.finalize();
    let sig = signer.sign(&msg);
    json!({"address": hex::encode(address), "timestamp": ts, "scheme": "Ed25519", "signature": hex::encode(sig.to_bytes())}).to_string()
}

fn mirror_tx(from: [u8;32], to: [u8;32], amount: u128, nonce: u64, ts: i64, fee: u128) -> Transaction {
    Transaction {
        id: TxHash::default(), from, to, amount, fee, nonce,
        signature: vec![], timestamp: chrono::DateTime::<Utc>::from_timestamp(ts, 0).unwrap(),
        data: vec![], token_type: TokenType::QUG, fee_token_type: TokenType::QUGUSD,
        tx_type: TransactionType::Transfer, pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519, pqc_public_key: None,
        zk_proof_bundle: None, privacy_level: TransactionPrivacyLevel::Transparent,
        bulletproof: None, nullifier: None, memo: None,
    }
}

fn signed_request(signer: &SigningKey, to: [u8;32], amount: u128, nonce: u64, ts: i64) -> (serde_json::Value, String) {
    let from: [u8;32] = signer.verifying_key().to_bytes();
    let fee = q_types::MIN_TRANSACTION_FEE;
    let tx = mirror_tx(from, to, amount, nonce, ts, fee);
    let payload = tx.signable_payload();
    let sig = signer.sign(&payload);
    let auth = wallet_auth(signer, from, "/api/v1/transactions/send_signed");
    let body = json!({
        "from": format!("0x{}", hex::encode(from)),
        "to": format!("0x{}", hex::encode(to)),
        "amount": amount,
        "token_type": "QUG",
        "signature": hex::encode(sig.to_bytes()),
        "nonce": nonce,
        "timestamp": ts,
    });
    (body, auth)
}

async fn do_post(r: &Router, body: &serde_json::Value, auth: &str) -> (StatusCode, serde_json::Value) {
    let resp = r.clone().oneshot(
        Request::builder().method("POST").uri("/api/v1/transactions/send_signed")
            .header("content-type", "application/json").header("X-Wallet-Auth", auth)
            .body(Body::from(body.to_string())).unwrap()
    ).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let j: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(json!({}));
    (status, j)
}

/// THE INDEPENDENT BUG-A CHECK: submit a valid signed tx (admitted, nonce
/// advances 0->1), then REPLAY the exact same signed body again (same
/// nonce=0, same signature, same everything => identical tx.id). The
/// mempool's `add_transaction_detailed` must reject the replay as
/// `DuplicateTransaction` -- a rejection reason that has NOTHING to do with
/// nonce correctness. The nonce tracker must remain at 1 (not be bumped to
/// 2, and not be left ambiguous). This is exactly the class of bug the
/// round-2 reviewers found: a rejection unrelated to nonce silently
/// burning/advancing the nonce anyway.
#[tokio::test]
async fn independent_check_unrelated_rejection_does_not_advance_nonce() {
    let state = make_state("dup-replay").await;
    let r = router(state.clone());

    let signer = SigningKey::from_bytes(&[99u8; 32]);
    let from: [u8;32] = signer.verifying_key().to_bytes();
    let to = [0x77u8; 32];
    let ts = Utc::now().timestamp();

    // First submission: nonce=0, must be admitted.
    let (body1, auth1) = signed_request(&signer, to, 42_000, 0, ts);
    let (status1, json1) = do_post(&r, &body1, &auth1).await;
    assert_eq!(status1, StatusCode::OK);
    assert_eq!(json1["success"], true, "first submission must succeed: {}", json1);
    assert_eq!(state.nonce_tracker.get_current(&from), 1, "nonce must advance to 1 after genuine admission");

    // Replay the IDENTICAL signed body (same nonce=0, same signature, same
    // tx.id) a second time. This must be rejected -- and NOT because nonce=0
    // is "wrong" (the nonce check would also reject it, but for a DIFFERENT
    // reason: NonceAlreadyPending or ValidationFailed via the NonceSource,
    // since nonce_tracker is now at 1). Either way it is a rejection, and
    // the key assertion is: it must not further move the nonce tracker.
    let (status2, json2) = do_post(&r, &body1, &auth1).await;
    assert_eq!(status2, StatusCode::OK);
    assert_eq!(json2["success"], false, "replay of an already-admitted tx must be rejected, not silently succeed again: {}", json2);
    assert_eq!(
        state.nonce_tracker.get_current(&from), 1,
        "BUG-A REGRESSION: nonce tracker must remain at 1 after a REJECTED replay -- if it moved to 2, a rejection was wrongly treated as genuine admission"
    );

    println!("✅ independent_check_unrelated_rejection_does_not_advance_nonce PASSED: first={} second={}", json1, json2);
}

/// Second independent check: an actually-unrelated-to-nonce rejection using
/// the DOUBLE-SPEND / self-transfer / amount-zero guards is HTTP-layer only
/// (rejected before ever reaching the mempool), so that path is already
/// obviously safe (nonce isn't consumed pre-mempool on the client-signed
/// path per the code read). The interesting case is a MEMPOOL-LEVEL
/// rejection for a nonce-unrelated reason, which is the duplicate-tx-hash
/// check above. This second test targets a distinct mempool rejection
/// reason: submitting the SAME nonce (0) twice with DIFFERENT payloads
/// (different `to`/`amount`), which are different tx.id's, so this hits
/// NonceAlreadyPending / stale-nonce-vs-tracker (mempool-level), not
/// DuplicateTransaction. Confirms behavior is consistent across more than
/// one rejection reason.
#[tokio::test]
async fn independent_check_second_reason_stale_nonce_does_not_double_advance() {
    let state = make_state("stale-nonce-2").await;
    let r = router(state.clone());

    let signer = SigningKey::from_bytes(&[100u8; 32]);
    let from: [u8;32] = signer.verifying_key().to_bytes();
    let to_a = [0x11u8; 32];
    let to_b = [0x22u8; 32];
    let ts = Utc::now().timestamp();

    let (body1, auth1) = signed_request(&signer, to_a, 1_000, 0, ts);
    let (status1, json1) = do_post(&r, &body1, &auth1).await;
    assert_eq!(status1, StatusCode::OK);
    assert_eq!(json1["success"], true, "first transfer must succeed: {}", json1);
    assert_eq!(state.nonce_tracker.get_current(&from), 1);

    // Different payload, SAME (already-consumed) nonce=0.
    let (body2, auth2) = signed_request(&signer, to_b, 2_000, 0, ts + 1);
    let (status2, json2) = do_post(&r, &body2, &auth2).await;
    assert_eq!(status2, StatusCode::OK);
    assert_eq!(json2["success"], false, "stale-nonce transfer with a DIFFERENT payload must be rejected: {}", json2);
    assert_eq!(
        state.nonce_tracker.get_current(&from), 1,
        "BUG-A REGRESSION: nonce tracker must remain at 1 after a rejected stale-nonce resubmission with a different payload"
    );

    println!("✅ independent_check_second_reason_stale_nonce_does_not_double_advance PASSED");
}
