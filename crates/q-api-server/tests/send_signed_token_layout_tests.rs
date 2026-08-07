//! v10.11.88 TOKEN-SEND regression tests — 2026-08-06 (grogu-qugusd-sendpath)
//!
//! Pins the 64-byte TokenTransfer `data` layout that `send_transaction_signed`
//! (crates/q-api-server/src/handlers.rs) now builds so agents (MCP clients)
//! can send QUGUSD / custom tokens:
//!
//!     data = token_addr(32) ‖ sender_pubkey(32)
//!
//! Why 64 bytes and not 32 — the fleet has TWO verifier conventions:
//!   1. q-types `verify_ed25519_signature`: candidate keys = `from`, data[..32];
//!      targets include p2p_signable_hash (which COVERS `data`). A 32-byte-data
//!      token tx can pass THIS verifier (the `from` candidate, v10.11.74)…
//!   2. …but `process_transaction_batch` (handlers.rs SIMD crediting pipeline)
//!      REQUIRES data.len() >= 64 for TokenTransfer and reads the pubkey at
//!      data[32..64]; a 32-byte-data tx is silently `continue`-skipped there.
//!      That silent skip is the layer that ate every agent token send.
//!
//! These tests deliberately do NOT go through the axum handler / AppState:
//! `AppState::new()` cannot be constructed in tests on Linux (known
//! pre-existing RocksDB double-open blocker — see the header comment of
//! phase0_send_signed_nonce_ordering_tests.rs). The contract lives at the
//! Transaction layer, which is exactly the seam the MCP client half must
//! reproduce byte-for-byte, so pinning it here is both sufficient and the
//! test-vector generator for the client implementation.
//!
//! Run with `-- --nocapture` to print the deterministic TEST VECTOR the
//! client half (quillon-wallet-mcp index.ts) must reproduce.

use chrono::TimeZone;
use ed25519_dalek::{Signer, SigningKey};
use q_api_server::transaction_utils::TransactionBuilder;
use q_types::{TokenType, TransactionType, QUGUSD_TOKEN_ADDRESS};

/// Deterministic fixture: fixed seed, fixed fields, fixed timestamp.
/// Ed25519 signing is deterministic (RFC 8032), so every byte of this
/// fixture — including the signature — is reproducible anywhere.
fn fixture() -> (SigningKey, q_types::Transaction) {
    // Fixed 32-byte seed (NOT a real wallet; test-vector only).
    let sk = SigningKey::from_bytes(&[7u8; 32]);
    let from: [u8; 32] = sk.verifying_key().to_bytes(); // address == pubkey (Ed25519 wallets)
    let to: [u8; 32] = [0x42u8; 32];

    // The layout under test: token_addr(32) ‖ sender_pubkey(32).
    let mut data = QUGUSD_TOKEN_ADDRESS.to_vec();
    data.extend_from_slice(&from);

    let ts = chrono::Utc.timestamp_opt(1_786_000_000, 0).unwrap();
    let tx = TransactionBuilder::new()
        .from(from)
        .to(to)
        .amount(1_000_000_000_000_000_000_000_000u128) // 1.0 QUGUSD (24 decimals)
        .fee(21_000u128)
        .token_type(TokenType::QUGUSD)
        .tx_type(TransactionType::TokenTransfer)
        .data(data)
        .build_with_nonce(7, ts);
    (sk, tx)
}

/// The core contract: a TokenTransfer with the 64-byte layout, signed over
/// p2p_signable_hash by the key whose pubkey == from, verifies.
#[test]
fn token_transfer_64byte_layout_signed_over_p2p_hash_verifies() {
    let (sk, mut tx) = fixture();

    assert_eq!(tx.data.len(), 64, "layout must be token_addr(32) ‖ pubkey(32)");
    assert_eq!(&tx.data[..32], QUGUSD_TOKEN_ADDRESS.as_slice(), "data[..32] = token addr");
    assert_eq!(&tx.data[32..64], tx.from.as_slice(), "data[32..64] = sender pubkey (crediting pipeline reads the key HERE)");

    let p2p_hash = tx.p2p_signable_hash();
    tx.signature = sk.sign(p2p_hash.as_ref()).to_bytes().to_vec();
    tx.signature_phase = q_types::TxSignaturePhase::Phase0Ed25519;

    tx.verify_signature()
        .expect("64-byte-layout token transfer signed over p2p_signable_hash must verify");
}

/// Documents WHY q-types verification alone was never the whole story: a
/// 32-byte-data token tx passes q-types (the v10.11.74 `from` candidate) but
/// fails the crediting pipeline's length requirement. If this test ever
/// fails on the first assertion, q-types got stricter and the handler
/// comment should be updated; if the length constant changes, the layout
/// contract itself moved — update the MCP client half in lockstep.
#[test]
fn thirty_two_byte_data_passes_qtypes_but_fails_crediting_length_rule() {
    let (sk, mut tx) = fixture();
    tx.data.truncate(32); // the pre-fix layout: token addr only
    tx.id = q_api_server::transaction_utils::compute_transaction_id(&tx);

    let p2p_hash = tx.p2p_signable_hash();
    tx.signature = sk.sign(p2p_hash.as_ref()).to_bytes().to_vec();
    tx.signature_phase = q_types::TxSignaturePhase::Phase0Ed25519;

    // q-types passes (from-candidate) — this is why the bug looked "verified"…
    tx.verify_signature()
        .expect("q-types accepts the from-candidate even with 32-byte data");

    // …but the crediting pipeline's rule (process_transaction_batch,
    // handlers.rs: required_len = 64 for TokenTransfer) drops it.
    let is_token_transfer = tx.tx_type == TransactionType::TokenTransfer;
    let required_len = if is_token_transfer { 64 } else { 32 };
    assert!(
        tx.data.len() < required_len,
        "32-byte data must be below the crediting pipeline's TokenTransfer minimum — \
         this asymmetry (q-types OK, crediting drop) IS the phantom-send mechanism"
    );
}

/// Tamper-evidence: the signature covers `data` via p2p_signable_hash, so a
/// post-signing data mutation (the v10.11.87 /send ordering bug, absent from
/// send_signed by construction) must break verification. Pins that the
/// send_signed handler MUST finalise data BEFORE the client-signature check.
#[test]
fn data_mutation_after_signing_breaks_verification() {
    let (sk, mut tx) = fixture();
    let p2p_hash = tx.p2p_signable_hash();
    tx.signature = sk.sign(p2p_hash.as_ref()).to_bytes().to_vec();
    tx.signature_phase = q_types::TxSignaturePhase::Phase0Ed25519;
    tx.verify_signature().expect("baseline verifies");

    tx.data[63] ^= 0x01; // mutate one pubkey byte after signing
    assert!(
        tx.verify_signature().is_err(),
        "signature must cover data — post-signing mutation may not verify"
    );
}

/// Pins the p2p_signable_payload wire prefix by hand-computable bytes, and
/// prints the full deterministic TEST VECTOR for the MCP client half.
#[test]
fn p2p_payload_layout_and_test_vector() {
    let (sk, mut tx) = fixture();
    let payload = tx.build_p2p_signable_payload();

    // Hand-verifiable prefix: version byte, from, to, amount(le,16).
    assert_eq!(payload[0], 0x01, "version byte");
    assert_eq!(&payload[1..33], tx.from.as_slice(), "from at [1..33]");
    assert_eq!(&payload[33..65], tx.to.as_slice(), "to at [33..65]");
    assert_eq!(&payload[65..81], tx.amount.to_le_bytes().as_slice(), "amount_le at [65..81]");
    // token_byte: QUG=0, QUGUSD=1, custom=2 — at [105] (65+16 fee+8 nonce+8 ts = 97… computed below)
    // offsets: 1 +32 +32 +16(amount) +16(fee) +8(nonce) +8(ts) = 113 → token_byte at [113]
    assert_eq!(payload[113], 1u8, "token_byte for QUGUSD");
    assert_eq!(
        &payload[114..118],
        (tx.data.len() as u32).to_le_bytes().as_slice(),
        "data_len_le"
    );
    assert_eq!(&payload[118..], tx.data.as_slice(), "data is the payload tail");
    assert_eq!(payload.len(), 118 + 64, "total payload length for a token transfer");

    let p2p_hash = tx.p2p_signable_hash();
    tx.signature = sk.sign(p2p_hash.as_ref()).to_bytes().to_vec();

    // ── TEST VECTOR (deterministic; run with --nocapture to capture) ──
    println!("──── send_signed TokenTransfer TEST VECTOR (v10.11.88) ────");
    println!("seed          : {}", hex::encode([7u8; 32]));
    println!("from (=pubkey): {}", hex::encode(tx.from));
    println!("to            : {}", hex::encode(tx.to));
    println!("amount        : {} (1.0 QUGUSD, 24 dec)", tx.amount);
    println!("fee           : {}", tx.fee);
    println!("nonce         : {}", tx.nonce);
    println!("timestamp     : {}", tx.timestamp.timestamp());
    println!("token_addr    : {}", hex::encode(QUGUSD_TOKEN_ADDRESS));
    println!("data (64B)    : {}", hex::encode(&tx.data));
    println!("p2p payload   : {}", hex::encode(&payload));
    println!("p2p hash      : {}", hex::encode(p2p_hash));
    println!("signature     : {}", hex::encode(&tx.signature));
}
