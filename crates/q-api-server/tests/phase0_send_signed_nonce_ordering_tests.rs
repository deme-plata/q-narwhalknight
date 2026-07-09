//! Phase 0 BUG-1 regression tests — 2026-07-08
//!
//! Reproduces and pins the fix for the nonce-ordering self-rejection bug
//! found by three independent adversarial reviewers: `crates/q-api-server/
//! src/handlers.rs`'s `send_transaction_signed` handler used to call
//! `state.nonce_tracker.set_nonce(&from_address, nonce.saturating_add(1))`
//! immediately after signature verification, BEFORE the transaction was
//! handed to the mempool for admission. The Phase 0 nonce check added to
//! `q-narwhal-core`'s `production_mempool.rs::perform_validation` (via the
//! `NonceSource` trait) then compares the submitted nonce N against
//! `NonceTracker::get_current()`, which by that point ALREADY read N+1 —
//! because the SAME request had already bumped it one statement earlier.
//! Result: every legitimate client-signed transfer was rejected, 100% of
//! the time, deterministically, the moment Q_ENFORCE_MEMPOOL_NONCE=1 was set.
//!
//! This is a REAL end-to-end integration test through the ACTUAL
//! `crates/q-api-server/src/handlers.rs::send_transaction_signed` code path
//! (a real axum `Router` built with the real, unmodified handler, a real
//! `AppState` constructed via the same `AppState::new` used in production,
//! and a real `ProductionMempool` with the mempool's own `NonceSource`
//! wiring — the exact wiring `main.rs` performs at boot when
//! Q_ENFORCE_MEMPOOL_NONCE=1 is set, replicated here since `main.rs` is a
//! `[[bin]]`-only file and its boot sequence isn't reachable from a test).
//! This deliberately does NOT reuse q-narwhal-core's isolated unit tests
//! with a hand-rolled `NonceSource` test double — those structurally cannot
//! catch this class of bug, because they never touch the handler layer
//! where the bug actually lives (the double-advance happens entirely
//! inside `send_transaction_signed`, one statement before the mempool is
//! even called).
//!
//! ⚠️ KNOWN BLOCKER (2026-07-08, unrelated pre-existing bug, NOT part of
//! this round's fix): as written, these 3 tests currently FAIL to even
//! reach `send_transaction_signed` — `AppState::new()` itself errors with
//! "Database integrity check failed: Failed to open database" on every
//! run, on Linux, against ANY brand-new database. Root cause (confirmed by
//! direct reproduction, RocksDB LOG showed "lock hold by current process
//! ... No locks available"): `AppState::new` (crates/q-api-server/src/
//! lib.rs) opens the hot RocksDB via `StorageEngine::new` and holds it open
//! for the AppState's lifetime, then unconditionally (on non-Windows) runs
//! a mandatory `IntegrityChecker::check()` (crates/q-storage/src/
//! integrity.rs) that independently re-opens the SAME `{db_path}/hot` path
//! in the SAME process — RocksDB's LOCK file is exclusive even within one
//! process, so the second open always fails. This affects EVERY test in
//! this crate that calls `AppState::new()` (also confirmed present in the
//! already-otherwise-broken `contracts_api_tests.rs`, `dex_integration_
//! tests.rs`, and `rwa_api_tests.rs`), not something specific to this file.
//! It was deliberately NOT worked around here by weakening or bypassing the
//! integrity check (a mandatory, "AI Expert Consensus" safety control on a
//! live financial system) — that would be exactly the kind of unrequested,
//! out-of-scope, safety-relevant change this round's brief warns against.
//! Filed separately for a real fix (reuse the already-open hot_db handle in
//! IntegrityChecker instead of re-opening, or reorder the check before
//! StorageEngine opens the DB). The test logic below (request/signature/
//! auth-header construction, assertions) is believed correct and will pass
//! once that separate bug is fixed — it has NOT been possible to observe a
//! green run of these 3 tests in this session because of this blocker.
//!
//! Run: cargo test -p q-api-server --test phase0_send_signed_nonce_ordering_tests

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

// ============================================================================
// Minimal TorClient mock — ProductionMempool::new requires one; nothing in
// these tests exercises actual Tor broadcast. Mirrors the NoopTorClient
// pattern already used in q-narwhal-core's own
// phase0_nonce_reuse_gap_tests.rs.
// ============================================================================

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

// ============================================================================
// Test harness: builds a real AppState (via the same AppState::new used in
// production), a real ProductionMempool wired to state.nonce_tracker as its
// NonceSource EXACTLY the way main.rs does at boot when
// Q_ENFORCE_MEMPOOL_NONCE=1 (see main.rs's "Wiring nonce_tracker into
// production_mempool as NonceSource" block) — replicated here because
// main.rs is a [[bin]]-only file, not reachable from an external test.
// ============================================================================

async fn make_test_app_state(db_suffix: &str) -> Arc<AppState> {
    // TEMP DIAGNOSTIC (round-4, will be reverted): surface tracing warn!/error!
    // output so we can see the real perform_validation rejection reason.
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    // NOTE: `IntegrityChecker` (q-storage/src/integrity.rs, invoked
    // unconditionally by AppState::new) always looks at `{db_path}/hot` —
    // it does NOT read the separate `hot_db_path` config field at all (a
    // pre-existing, unrelated latent mismatch between the two config knobs,
    // out of scope for this round). Set `hot_db_path` to `{db_path}/hot`
    // explicitly here so StorageEngine actually creates its hot store where
    // the integrity checker will look for it.
    let db_root = format!("target/test-db-phase0-bug1-{}", db_suffix);
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

    // Same wiring as main.rs's Q_ENFORCE_MEMPOOL_NONCE=1 block: the mempool's
    // NonceSource is the SAME NonceTracker instance state.nonce_tracker uses
    // for server-assigned nonces — this is the crux of the bug under test,
    // since the double-advance and the check both hit this one instance.
    mempool
        .set_nonce_source(state.nonce_tracker.clone())
        .await;

    state.production_mempool = Some(mempool);

    Arc::new(state)
}

fn make_test_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/v1/transactions/send_signed", post(handlers::send_transaction_signed))
        .with_state(state)
}

/// Builds the X-Wallet-Auth header value for the Ed25519 auth scheme,
/// matching wallet_auth.rs's `FromRequestParts` verification exactly:
/// message = SHA3-256(address || timestamp.to_le_bytes() || request_path).
/// No body_hash (this endpoint's callers don't send one).
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
        // Phase 0 Round-4 GAP2 FIX (2026-07-08): wallet_auth.rs's
        // from_request_parts only strips a "qnk" prefix before hex::decode --
        // never "0x" -- so a "0x"-prefixed address here fails hex::decode on
        // the literal 'x' and every request in this file 401s before the
        // handler body ever runs. Send plain hex (no prefix) to match.
        "address": hex::encode(address),
        "timestamp": timestamp,
        "scheme": "Ed25519",
        "signature": hex::encode(signature.to_bytes()),
    })
    .to_string()
}

/// Builds the exact `Transaction` the server will construct internally from
/// a send_signed request (mirrors `TransactionBuilder::build_with_nonce` +
/// the fee/signature_phase assignment `send_transaction_signed` performs),
/// so `.signable_payload()` computed here matches what the server verifies
/// against. `id` does not need to be the server's real computed id —
/// `signable_payload()` zeroes `id` internally before hashing regardless.
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

/// Builds a signed send_signed JSON request body + the X-Wallet-Auth header
/// value for a transfer of `amount` from `signer` to `to`, using `nonce` and
/// the current timestamp. Fee defaults to MIN_TRANSACTION_FEE (matches the
/// server's `request.fee.unwrap_or(q_types::MIN_TRANSACTION_FEE)` when the
/// request omits `fee`).
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
        // Phase 0 Round-4 GAP2 FIX (2026-07-08): SendTransactionSignedRequest.amount
        // is a plain u128 (no string-tolerant deserializer), so sending it as a
        // JSON string here made axum's Json extractor reject the body with 400.
        // Send it as a raw JSON number instead.
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
// TEST 1 — The actual BUG-1 regression: a freshly-funded wallet's very
// first legitimate signed transfer must be ADMITTED (not rejected) with
// Q_ENFORCE_MEMPOOL_NONCE's wiring active (the mempool's NonceSource set to
// state.nonce_tracker, exactly as production does when that env var is 1).
//
// Pre-fix, this test fails: send_transaction_signed bumps nonce_tracker to
// 1 (nonce.saturating_add(1)) immediately after signature verification,
// THEN submit_transaction -> add_transaction -> perform_validation ->
// validate_nonce(from, submitted_nonce=0) compares against
// NonceTracker::get_current() which now reads 1 (already bumped by this
// SAME request) -> Err(1) -> rejected. 100% deterministic, every time.
// ============================================================================

#[tokio::test]
async fn test_fresh_wallet_first_signed_transfer_is_admitted_not_rejected() {
    let state = make_test_app_state("fresh-wallet").await;
    let router = make_test_router(state.clone());

    let signer = SigningKey::from_bytes(&[3u8; 32]);
    let to = [0xA1u8; 32];

    // Fresh wallet: NonceTracker::get_current() defaults to 0, so the
    // client's very first send_signed call must use nonce=0.
    let (body, auth_header) = build_signed_request(&signer, to, 100_000, 0);
    let (status, json_body) = post_send_signed(&router, &body, &auth_header).await;

    assert_eq!(status, StatusCode::OK, "HTTP layer must return 200 (errors are reported via the JSON body's `success` field, not HTTP status)");
    assert_eq!(
        json_body["success"], true,
        "a freshly-funded wallet's first legitimate signed transfer must be ADMITTED, not rejected — got response: {}",
        json_body
    );
    assert_eq!(
        json_body["data"]["queued_for_block"], true,
        "the transaction must actually reach the mempool (queued_for_block=true), not just get an HTTP 200 with an error payload"
    );

    println!("✅ test_fresh_wallet_first_signed_transfer_is_admitted_not_rejected PASSED: {}", json_body);
}

// ============================================================================
// TEST 2 — Two sequential legitimate transfers from the same wallet, where
// the second is signed with nonce=1 before the first (nonce=0) has
// "confirmed" (there is no real block production in this test harness, so
// "confirms" here means: admitted into the mempool). Both must eventually
// be admitted. This guards specifically against a reorder that fixes the
// self-rejection but introduces a NEW bug where the second nonce is never
// unblocked (e.g. if the fix accidentally stopped advancing the tracker at
// all, the second send would look like a nonce-reuse of the first).
// ============================================================================

#[tokio::test]
async fn test_two_sequential_transfers_same_wallet_both_eventually_succeed() {
    let state = make_test_app_state("sequential-transfers").await;
    let router = make_test_router(state.clone());

    let signer = SigningKey::from_bytes(&[5u8; 32]);
    let to_a = [0xB2u8; 32];
    let to_b = [0xC3u8; 32];

    // First transfer: nonce=0 (fresh wallet).
    let (body1, auth1) = build_signed_request(&signer, to_a, 50_000, 0);
    let (status1, json1) = post_send_signed(&router, &body1, &auth1).await;
    assert_eq!(status1, StatusCode::OK);
    assert_eq!(json1["success"], true, "first transfer (nonce=0) must be admitted: {}", json1);
    assert_eq!(json1["data"]["queued_for_block"], true, "first transfer must reach the mempool: {}", json1);

    // Second transfer: nonce=1, signed and submitted immediately after —
    // "before the first confirms" in the sense that there's no block
    // production between these two calls in this test harness, only the
    // BUG-1 fix's post-admission nonce advance from the first call.
    let (body2, auth2) = build_signed_request(&signer, to_b, 75_000, 1);
    let (status2, json2) = post_send_signed(&router, &body2, &auth2).await;
    assert_eq!(status2, StatusCode::OK);
    assert_eq!(json2["success"], true, "second transfer (nonce=1) must also be admitted: {}", json2);
    assert_eq!(json2["data"]["queued_for_block"], true, "second transfer must reach the mempool: {}", json2);

    println!("✅ test_two_sequential_transfers_same_wallet_both_eventually_succeed PASSED");
}

// ============================================================================
// TEST 3 — Negative control: a stale/wrong nonce (reusing nonce=0 after it
// has already been consumed) must still be REJECTED. Guards against an
// overcorrection where the fix stops enforcing nonce ordering altogether.
// ============================================================================

#[tokio::test]
async fn test_stale_nonce_reuse_after_admission_is_still_rejected() {
    let state = make_test_app_state("stale-nonce-reuse").await;
    let router = make_test_router(state.clone());

    let signer = SigningKey::from_bytes(&[6u8; 32]);
    let to_a = [0xD4u8; 32];
    let to_b = [0xE5u8; 32];

    let (body1, auth1) = build_signed_request(&signer, to_a, 10_000, 0);
    let (status1, json1) = post_send_signed(&router, &body1, &auth1).await;
    assert_eq!(status1, StatusCode::OK);
    assert_eq!(json1["success"], true, "first transfer (nonce=0) must be admitted: {}", json1);

    // Re-use nonce=0 again (a different payload, so a different tx.id —
    // DS-1's tx-id replay guard would not catch this; only the Phase 0
    // nonce check does).
    let (body2, auth2) = build_signed_request(&signer, to_b, 999_999, 0);
    let (status2, json2) = post_send_signed(&router, &body2, &auth2).await;
    assert_eq!(status2, StatusCode::OK, "HTTP layer still returns 200; rejection is reported via `success: false`");
    assert_eq!(
        json2["success"], false,
        "a stale/already-consumed nonce must be rejected, not silently admitted: {}",
        json2
    );

    println!("✅ test_stale_nonce_reuse_after_admission_is_still_rejected PASSED");
}
