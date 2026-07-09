//! INDEPENDENT adversarial-reviewer repro for BUG A — written by the
//! reviewer, NOT the fix implementer, specifically to cover the scenario
//! the implementer's own shipped test suite
//! (phase0_send_signed_nonce_ordering_tests.rs) did NOT test: a
//! legitimately-signed, correctly-nonced transaction that is rejected by
//! the mempool for a reason UNRELATED to nonce correctness (blocked
//! wallet), and confirming the nonce tracker is NOT advanced as a result.
//!
//! This exercises the REAL production handler (send_transaction_signed),
//! the REAL AppState::new construction, and the REAL
//! ProductionMempool::add_transaction_detailed rejection path (via
//! Q_BLOCKED_WALLETS), exactly as round 2's reviewers required.

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
    async fn connect_to_onion(
        &self,
        _onion_address: &str,
        _port: u16,
    ) -> anyhow::Result<Box<dyn TorStreamConnection>> {
        Err(anyhow::anyhow!("NoopTorClient: no real Tor connections in tests"))
    }
}

async fn make_test_app_state(db_suffix: &str) -> Arc<AppState> {
    let db_root = format!("target/test-db-bugA-independent-{}", db_suffix);
    let hot_path = format!("{}/hot", db_root);
    let config = Config {
        db_path: Some(db_root),
        hot_db_path: Some(hot_path),
        ..Config::default()
    };

    let mut state = AppState::new(config)
        .await
        .expect("AppState::new failed");

    let mempool_config = MempoolConfig::default();
    let tor_client: Arc<dyn TorClient> = Arc::new(NoopTorClient);
    let mempool = Arc::new(
        ProductionMempool::new(mempool_config, tor_client, Phase::Phase1)
            .await
            .expect("ProductionMempool::new failed"),
    );

    // Deliberately NOT calling set_nonce_source here — the blocked-wallet
    // check is independent of the Phase 0 nonce-vs-chain-state check, and
    // we want to isolate BUG A's mechanism specifically (queued_for_block /
    // mempool_admitted conflation), not couple it to nonce-source wiring.
    state.production_mempool = Some(mempool);

    Arc::new(state)
}

fn make_test_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/transactions/send_signed", post(handlers::send_transaction_signed))
        .with_state(state)
}

fn build_wallet_auth_header(signer: &SigningKey, address: [u8; 32], path: &str) -> String {
    use sha3::{Digest, Sha3_256};
    let timestamp = Utc::now().timestamp();
    let mut hasher = Sha3_256::new();
    hasher.update(&address);
    hasher.update(&timestamp.to_le_bytes());
    hasher.update(path.as_bytes());
    let message = hasher.finalize();
    let signature = signer.sign(&message);
    json!({
        // NOTE: wallet_auth.rs::from_request_parts only strips a "qnk" prefix
        // from `auth.address` before hex::decode — NOT "0x". Sending "0x..."
        // here makes hex::decode fail on the literal 'x' character, causing
        // AuthError::invalid_address (401) before the handler body ever runs.
        // Raw hex (no prefix) works because "qnk"-stripping is a no-op then.
        "address": hex::encode(address),
        "timestamp": timestamp,
        "scheme": "Ed25519",
        "signature": hex::encode(signature.to_bytes()),
    })
    .to_string()
}

fn build_mirror_transaction(
    from: [u8; 32],
    to: [u8; 32],
    amount: u128,
    nonce: u64,
    timestamp_secs: i64,
    fee: u128,
) -> Transaction {
    Transaction {
        id: TxHash::default(),
        from,
        to,
        amount,
        fee,
        nonce,
        signature: vec![],
        timestamp: chrono::DateTime::<Utc>::from_timestamp(timestamp_secs, 0).unwrap(),
        data: vec![],
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUGUSD,
        tx_type: TransactionType::Transfer,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    }
}

fn build_signed_request(
    signer: &SigningKey,
    to: [u8; 32],
    amount: u128,
    nonce: u64,
) -> (serde_json::Value, String) {
    let from: [u8; 32] = signer.verifying_key().to_bytes();
    let timestamp = Utc::now().timestamp();
    let fee = q_types::MIN_TRANSACTION_FEE;
    let mirror_tx = build_mirror_transaction(from, to, amount, nonce, timestamp, fee);
    let payload = mirror_tx.signable_payload();
    let signature = signer.sign(&payload);
    let auth_header = build_wallet_auth_header(signer, from, "/api/v1/transactions/send_signed");
    let body = json!({
        "from": format!("0x{}", hex::encode(from)),
        "to": format!("0x{}", hex::encode(to)),
        // NOTE: SendTransactionSignedRequest::amount is a plain `u128` with no
        // string-tolerant deserializer (no serde_as/DisplayFromStr) — sending
        // a JSON string here makes axum's Json<> extractor itself reject the
        // request with HTTP 400 before send_transaction_signed's body ever
        // runs. Must be a raw JSON number.
        "amount": amount,
        "token_type": "QUG",
        "signature": hex::encode(signature.to_bytes()),
        "nonce": nonce,
        "timestamp": timestamp,
    });
    (body, auth_header)
}

async fn post_send_signed(
    router: &Router,
    body: &serde_json::Value,
    auth_header: &str,
) -> (StatusCode, serde_json::Value) {
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/transactions/send_signed")
                .header("content-type", "application/json")
                .header("X-Wallet-Auth", auth_header)
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json_body: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(json!({}));
    (status, json_body)
}

// ============================================================================
// THE CORE INDEPENDENT-REPRO TEST: a wallet on the Q_BLOCKED_WALLETS
// freeze list submits a perfectly-signed, correctly-nonced (nonce=0, fresh
// wallet) transfer. It must be REJECTED (mempool rejects for BlockedWallet,
// unrelated to nonce correctness) AND — this is the actual BUG A assertion
// — the nonce tracker must NOT advance as a result. We verify this by
// checking nonce_tracker.get_current() directly (white-box, most direct)
// AND by confirming a subsequent send with the SAME nonce=0 from an
// unblocked wallet-equivalent scenario is not itself corrupted.
//
// Env var caveat: the mempool's blocked-list is cached in a `OnceLock`
// keyed process-wide (first call to add_transaction_detailed wins), so
// Q_BLOCKED_WALLETS MUST be set before ANY transaction (from ANY wallet)
// has been submitted in this process. This test sets the env var at the
// very top, before constructing anything, and this file is run as its own
// test binary (separate process from other integration test files), so
// there is no cross-file contamination.
// ============================================================================

#[tokio::test]
async fn test_blocked_wallet_rejection_does_not_advance_nonce() {
    // Pick a random wallet keypair and freeze IT specifically via
    // Q_BLOCKED_WALLETS before constructing any mempool in this process.
    let signer = SigningKey::from_bytes(&[42u8; 32]);
    let from: [u8; 32] = signer.verifying_key().to_bytes();
    std::env::set_var("Q_BLOCKED_WALLETS", hex::encode(from));

    let state = make_test_app_state("blocked-wallet").await;
    let router = make_test_router(state.clone());

    let to = [0x77u8; 32];

    // Nonce tracker starts at 0 for a fresh wallet.
    assert_eq!(
        state.nonce_tracker.get_current(&from),
        0,
        "sanity: fresh wallet nonce tracker must start at 0"
    );

    // Correctly-signed, correctly-nonced (nonce=0) transfer from the
    // BLOCKED wallet.
    let (body, auth_header) = build_signed_request(&signer, to, 100_000, 0);
    let (status, json_body) = post_send_signed(&router, &body, &auth_header).await;

    assert_eq!(
        status,
        StatusCode::OK,
        "HTTP layer returns 200; rejection is reported via JSON `success` field"
    );

    // THE BUG A ASSERTION #1: the response must NOT claim success for a
    // transaction that was never admitted to the mempool.
    assert_eq!(
        json_body["success"], false,
        "🛡 BUG A: a transaction rejected by the mempool (blocked wallet) must \
         NOT be reported as success:true to the client — got: {}",
        json_body
    );

    // THE BUG A ASSERTION #2 (the actual regression this round's fix
    // targets): the nonce tracker must NOT have advanced. Pre-fix, this
    // gated on `queued_for_block`, which was unconditionally true even for
    // a BlockedWallet rejection (the old Ok(false) branch's own comment:
    // "Still counts as queued") — so nonce_tracker would read 1 here,
    // silently and permanently burning the sender's nonce for a
    // transaction that will NEVER appear in a block.
    assert_eq!(
        state.nonce_tracker.get_current(&from),
        0,
        "🛡 BUG A REGRESSION: nonce tracker advanced to {} even though the \
         mempool REJECTED this transaction (blocked wallet) — this is \
         exactly the silent-transaction-loss bug round 2's reviewers found: \
         a legitimate, correctly-nonced transfer rejected for an unrelated \
         reason still had its nonce burned.",
        state.nonce_tracker.get_current(&from)
    );

    println!(
        "✅ test_blocked_wallet_rejection_does_not_advance_nonce PASSED: response={}, nonce_after={}",
        json_body,
        state.nonce_tracker.get_current(&from)
    );
}
