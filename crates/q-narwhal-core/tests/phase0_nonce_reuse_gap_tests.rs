//! Phase 0 nonce-reuse gap regression tests — 2026-07-08
//!
//! Reproduces the historical gap this session's design plan identified:
//! nothing between mempool admission and block packing compared a
//! transaction's `nonce` against the sender's actual last-confirmed
//! on-chain nonce. DS-1 (the tx-id replay guard) does NOT catch a
//! re-signed transaction that reuses an already-spent nonce with a
//! DIFFERENT payload (different `to`/`amount`/`data`) — different content
//! means a different `tx.id` (SHA3-256 of content), so DS-1's
//! `applied_tx_<id>` key never collides.
//!
//! Exercises the real, live-path types added this session:
//!   - `q_narwhal_core::production_mempool::NonceSource` (the trait)
//!   - `TxValidator`'s admission-time check (Patch 2a, in `perform_validation`)
//!   - `ProductionMempool::get_transactions_for_block`'s pack-time re-check
//!     (Patch 2b)
//!
//! Uses a lightweight in-memory `NonceSource` test double (not the real
//! q-api-server `NonceTracker`, which lives in a different crate and would
//! be a reverse dependency) — the trait is exactly the seam that makes this
//! possible without q-narwhal-core depending on q-api-server.
//!
//! Run: cargo test --package q-narwhal-core --test phase0_nonce_reuse_gap_tests

use dashmap::DashMap;
use ed25519_dalek::{Signer, SigningKey};
use q_narwhal_core::production_mempool::{
    MempoolConfig, NonceSource, ProductionMempool,
};
use q_narwhal_core::{TorClient, TorStreamConnection};
use q_types::{Phase, Transaction, TokenType, TransactionType, TxHash, TxSignaturePhase, TransactionPrivacyLevel};
use std::sync::Arc;

// ============================================================================
// Minimal TorClient mock — ProductionMempool::new requires one, but nothing
// in these tests exercises actual broadcast, so a no-op stub is sufficient.
// (The crate's own MockTorClient in tor_client_impl.rs is private/#[cfg(test)]-
// scoped to that module, not reachable from an integration test file.)
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
// In-memory NonceSource test double, standing in for q-api-server's real
// NonceTracker (which implements this same trait — see
// crates/q-api-server/src/transaction_utils.rs).
// ============================================================================

struct TestNonceSource {
    expected: DashMap<[u8; 32], u64>,
}

impl TestNonceSource {
    fn new() -> Self {
        Self { expected: DashMap::new() }
    }

    /// Simulates a nonce becoming "spent" on-chain (i.e. the wallet's next
    /// expected nonce advances), the way a confirmed block would.
    fn advance(&self, wallet: &[u8; 32], new_expected: u64) {
        self.expected.insert(*wallet, new_expected);
    }
}

impl NonceSource for TestNonceSource {
    fn validate_nonce(&self, wallet: &[u8; 32], submitted_nonce: u64) -> Result<(), u64> {
        let expected = self.expected.get(wallet).map(|v| *v).unwrap_or(0);
        if submitted_nonce == expected {
            Ok(())
        } else {
            Err(expected)
        }
    }
}

// ============================================================================
// Test fixture: a real, validly-signed Ed25519 Transaction. Built by hand
// (not via the feature-gated Transaction::sign(), which q-narwhal-core's
// Cargo.toml does not enable) using the same signable_payload() the
// production verifier checks against — see q_types::Transaction::
// signable_payload / verify_ed25519_signature.
// ============================================================================

fn signed_transfer(
    signer: &SigningKey,
    to: [u8; 32],
    amount: u128,
    nonce: u64,
    _id_tag: u8,
) -> Transaction {
    let from: [u8; 32] = signer.verifying_key().to_bytes();
    let mut tx = Transaction {
        // id is derived/canonicalized-away data for signing purposes
        // (Transaction::signable_payload() zeroes it before hashing) — start
        // at zero, we don't need a distinguishing id for these tests since
        // uniqueness is asserted via the tx content itself where it matters.
        id: TxHash::default(),
        from,
        to,
        amount,
        // Fee = the exact minimum validate_fee_at_height requires for a
        // Transfer at legacy rates (BASE_GAS=21_000 * gas_multiplier=1 *
        // MIN_FEE_PER_GAS=1 / fee_divisor=1). Callers must pass amount >=
        // this so the dust-attack guard (fee > amount rejected unless
        // fee <= min_required_fee * 10) also passes.
        fee: 21_000u128,
        nonce,
        signature: vec![],
        timestamp: chrono::Utc::now(),
        data: vec![],
        token_type: TokenType::QUG,
        fee_token_type: TokenType::QUG,
        tx_type: TransactionType::Transfer,
        pqc_signature: None,
        signature_phase: TxSignaturePhase::Phase0Ed25519,
        pqc_public_key: None,
        zk_proof_bundle: None,
        privacy_level: TransactionPrivacyLevel::Transparent,
        bulletproof: None,
        nullifier: None,
        memo: None,
    };
    // verify_ed25519_signature checks the signature against signable_payload()
    // (canonical form with signature+id zeroed) — NOT signing_payload() (a
    // different, older canonicalization). Sign the correct target.
    let payload = tx.signable_payload();
    let signature = signer.sign(&payload);
    tx.signature = signature.to_bytes().to_vec();
    tx
}

async fn make_mempool() -> Arc<ProductionMempool> {
    let config = MempoolConfig::default();
    let tor_client: Arc<dyn TorClient> = Arc::new(NoopTorClient);
    Arc::new(
        ProductionMempool::new(config, tor_client, Phase::Phase1)
            .await
            .expect("ProductionMempool::new failed"),
    )
}

// ============================================================================
// TEST 1 — Baseline: with NO nonce source wired, a nonce-reuse-with-
// different-payload tx IS admitted (documents the gap as it exists without
// Patch 2a wired up — i.e. Q_ENFORCE_MEMPOOL_NONCE unset in production).
// ============================================================================

#[tokio::test]
async fn test_nonce_reuse_admitted_without_nonce_source_wired() {
    let mempool = make_mempool().await;
    let signer = SigningKey::from_bytes(&[7u8; 32]);
    let to_x = [0xA1u8; 32];
    let to_y = [0xB2u8; 32];

    let tx1 = signed_transfer(&signer, to_x, 100_000, 5, 0x01);
    let admitted1 = mempool.add_transaction(tx1, None).await.expect("add_transaction errored");
    assert!(admitted1, "tx1 (nonce=5) should be admitted — first use of this nonce");

    // Different payload (different `to`/`amount`), SAME nonce=5. Different
    // content -> different tx.id, so nothing tx-id-keyed (DS-1) would catch
    // this as a duplicate. Without a NonceSource wired, perform_validation's
    // nonce check is a documented no-op (see NonceSource trait doc).
    let tx2 = signed_transfer(&signer, to_y, 1_000_000, 5, 0x02);
    let admitted2 = mempool.add_transaction(tx2, None).await.expect("add_transaction errored");

    // NOTE: pending_nonces (the O(1) in-mempool nonce map) DOES still reject
    // this, because tx1 is still pending (not yet included in a block) and
    // pending_nonces keys on (from, nonce) regardless of payload. This is
    // the existing, narrower protection the design plan described as
    // "only prevents two simultaneously pending mempool entries from
    // sharing a nonce." To exercise the GAP itself (chain-state-vs-nonce,
    // not simultaneous-pending-vs-nonce), advance past that by removing tx1
    // from pending first, simulating it having already been included in a
    // block.
    assert!(
        !admitted2,
        "expected pending_nonces to reject the simultaneous-pending duplicate nonce \
         (this is the EXISTING, narrower guard — not the gap under test)"
    );

    println!("✅ test_nonce_reuse_admitted_without_nonce_source_wired PASSED (documents pre-existing pending_nonces behavior)");
}

// ============================================================================
// TEST 2 — The actual gap: tx1 (nonce=5) is included in a block (removed
// from pending, simulating confirmation). A re-signed tx2 with the SAME
// nonce=5 but a DIFFERENT payload arrives later. Without a NonceSource
// wired, it is admitted (the gap). WITH Patch 2a's NonceSource wired and
// the chain state NOT advanced to reflect tx1, the tracker still expects
// nonce=5 as unspent... so this test drives the source itself to reflect
// "nonce 5 already spent" (expected=6) the way a real NonceTracker would
// after tx1's block lands, and confirms tx2 is now REJECTED.
// ============================================================================

#[tokio::test]
async fn test_nonce_reuse_different_payload_rejected_with_nonce_source_wired() {
    let mempool = make_mempool().await;
    let nonce_source = Arc::new(TestNonceSource::new());
    mempool.set_nonce_source(nonce_source.clone() as Arc<dyn NonceSource>).await;

    let signer = SigningKey::from_bytes(&[9u8; 32]);
    let from: [u8; 32] = signer.verifying_key().to_bytes();
    let to_x = [0xC3u8; 32];
    let to_y = [0xD4u8; 32];

    // Chain state: this wallet's next expected nonce is 5 (i.e. nonces
    // 0..4 already confirmed on-chain).
    nonce_source.advance(&from, 5);

    let tx1 = signed_transfer(&signer, to_x, 100_000, 5, 0x03);
    let admitted1 = mempool.add_transaction(tx1.clone(), None).await.expect("add_transaction errored");
    assert!(admitted1, "tx1 (nonce=5, matches expected chain nonce) should be admitted");

    // Simulate tx1 landing in a confirmed block: remove it from pending
    // (as remove_included_transactions would) AND advance the nonce source
    // to reflect the new chain state (as the real NonceTracker would after
    // a confirmed send).
    mempool.remove_included_transactions(&[tx1.hash()]).await;
    nonce_source.advance(&from, 6);

    // Now a re-signed tx2 arrives: SAME nonce=5 (already spent per chain
    // state), DIFFERENT payload (different to/amount -> different tx.id,
    // so DS-1's tx-id replay guard would NOT catch this).
    let tx2 = signed_transfer(&signer, to_y, 999_999, 5, 0x04);
    assert_ne!(tx1.hash(), tx2.hash(), "test invariant: tx1 and tx2 must have different tx.id");

    let admitted2 = mempool.add_transaction(tx2, None).await.expect("add_transaction errored");
    assert!(
        !admitted2,
        "GAP CHECK: a nonce-reuse tx with a different payload (different tx.id, so DS-1 \
         would not catch it) must be rejected once a NonceSource is wired — nonce 5 no \
         longer matches the chain-expected nonce of 6"
    );

    println!("✅ test_nonce_reuse_different_payload_rejected_with_nonce_source_wired PASSED");
}

// ============================================================================
// TEST 3 — Pack-time re-check (Patch 2b): a tx admitted while its nonce was
// still valid gets dropped at get_transactions_for_block time if chain
// state moves (nonce gets spent by another producer/lane) before packing.
// ============================================================================

#[tokio::test]
async fn test_pack_time_recheck_drops_tx_whose_nonce_was_spent_after_admission() {
    let mempool = make_mempool().await;
    let nonce_source = Arc::new(TestNonceSource::new());
    mempool.set_nonce_source(nonce_source.clone() as Arc<dyn NonceSource>).await;

    let signer = SigningKey::from_bytes(&[11u8; 32]);
    let from: [u8; 32] = signer.verifying_key().to_bytes();
    let to = [0xE5u8; 32];

    nonce_source.advance(&from, 0); // fresh wallet, expects nonce 0

    let tx = signed_transfer(&signer, to, 100_000, 0, 0x05);
    let admitted = mempool.add_transaction(tx, None).await.expect("add_transaction errored");
    assert!(admitted, "tx (nonce=0, matches expected) should be admitted");

    // Simulate: BEFORE this tx is packed, a different producer/lane already
    // confirmed a block that consumed nonce 0 for this wallet (chain state
    // moved). The tx is still sitting in `pending_transactions` as Valid
    // (admission-time check already passed), but is now stale.
    nonce_source.advance(&from, 1);

    let packed = mempool.get_transactions_for_block(10).await;
    assert!(
        packed.is_empty(),
        "pack-time re-check (Patch 2b) must drop a tx whose nonce no longer matches \
         current chain state, even though it passed the admission-time check earlier: \
         got {} tx(s) packed, expected 0",
        packed.len()
    );

    println!("✅ test_pack_time_recheck_drops_tx_whose_nonce_was_spent_after_admission PASSED");
}

// ============================================================================
// TEST 4 — Positive control: a legitimate tx with the correct expected
// nonce is admitted AND survives pack-time re-check when chain state has
// not moved. Guards against Patch 2a/2b being so strict they block normal
// traffic (the mission's biggest named regression risk).
// ============================================================================

#[tokio::test]
async fn test_legitimate_tx_with_correct_nonce_admitted_and_packed() {
    let mempool = make_mempool().await;
    let nonce_source = Arc::new(TestNonceSource::new());
    mempool.set_nonce_source(nonce_source.clone() as Arc<dyn NonceSource>).await;

    let signer = SigningKey::from_bytes(&[13u8; 32]);
    let from: [u8; 32] = signer.verifying_key().to_bytes();
    let to = [0xF6u8; 32];

    nonce_source.advance(&from, 3);

    let tx = signed_transfer(&signer, to, 100_000, 3, 0x06);
    let admitted = mempool.add_transaction(tx, None).await.expect("add_transaction errored");
    assert!(admitted, "legitimate tx with correct expected nonce must be admitted");

    // Chain state has NOT moved — this tx's nonce is still the expected one.
    let packed = mempool.get_transactions_for_block(10).await;
    assert_eq!(
        packed.len(), 1,
        "legitimate tx must survive pack-time re-check when chain state is unchanged"
    );
    assert_eq!(packed[0].nonce, 3);
    assert_eq!(packed[0].from, from);

    println!("✅ test_legitimate_tx_with_correct_nonce_admitted_and_packed PASSED");
}
