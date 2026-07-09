//! Phase 0 BUG-2 regression tests — 2026-07-08
//!
//! Pins the fix to `q-dag-knight`'s `vertex_creator::VertexCreator::
//! validate_vertex` — the function that is ACTUALLY on the live
//! vertex-ingestion path for network-gossiped vertices. Confirmed via grep:
//! the only two `.process_vertex(` call sites repo-wide are in this crate
//! (`mempool_integration.rs:295`, `lib.rs:170`), and both call DIFFERENT
//! functions — neither is `q_narwhal_core::NarwhalCore::process_vertex`/
//! `validate_vertex`, which (despite being correct, well-implemented code
//! modeled on `reliable_broadcast.rs`'s sibling) has NO production caller.
//! The real path for a peer-gossiped vertex is `mempool_integration.rs::
//! handle_peer_vertex`, which calls THIS crate's `vertex_creator::
//! validate_vertex` (see that function's doc comment for the full
//! reasoning).
//!
//! Before this fix, `validate_vertex` checked VDF proof validity, parent
//! references, and round consistency, but had NO signature check at all —
//! despite `Vertex.signature` being populated by `sign_vertex` whenever a
//! node creates its own vertex. Any peer could gossip a vertex with a
//! forged, empty, or corrupted signature and it would sail through
//! unnoticed.
//!
//! The fix ports the same Ed25519 verification logic already proven in
//! `q_narwhal_core::reliable_broadcast::verify_ed25519_signature` (reused
//! directly — the function was made `pub` for this — not re-derived) using
//! the same `H(id || round || tx_root || parents)` message construction
//! `sign_vertex` already signs.
//!
//! This is a standalone integration test file (not inline in
//! `vertex_creator.rs`'s own `#[cfg(test)] mod tests`) because
//! `q-dag-knight/src/lib.rs` has its OWN, separate inline test module with
//! extensive PRE-EXISTING, UNRELATED compile breakage (missing `.await`,
//! stale `Vertex`/`AnchorElectionResult`/`RandomnessSource` APIs — 26
//! baseline errors, confirmed via `cargo check -p q-dag-knight --lib`,
//! none touching vertex_creator.rs). Since all `#[cfg(test)] mod tests`
//! blocks in a crate's `src/` link into ONE `--lib` test binary, that
//! breakage would make it impossible to actually RUN any new inline tests
//! even if they compiled individually. A standalone `tests/*.rs` file is
//! its own independent binary and sidesteps this — the same reasoning the
//! prior round's `phase0_vertex_signature_tests.rs` /
//! `phase0_nonce_reuse_gap_tests.rs` already used for q-narwhal-core.
//!
//! Run: cargo test -p q-dag-knight --test phase0_dagknight_vertex_signature_tests

use ed25519_dalek::SigningKey;
use q_dag_knight::{QuantumVDF, QuantumVDFConfig, VDFSecurityLevel, Vertex, VertexCreator};
use std::sync::Arc;

/// Fast VDF config: `Classical` (pure SHA3 loop, no async QRNG spin-up)
/// with a small `base_difficulty` so `compute_proof` is instant in tests.
/// Must stay >= 8 — `compute_classical_proof` (in q-dag-knight's
/// quantum_vdf.rs) divides `difficulty / 8` to space out parallel
/// witnesses, so anything below 8 would panic on divide-by-zero.
fn fast_vdf_config() -> QuantumVDFConfig {
    QuantumVDFConfig {
        base_difficulty: 8,
        quantum_enhancement: 0.0,
        parallel_threads: 1,
        qrng_seed_interval: std::time::Duration::from_secs(60),
        security_level: VDFSecurityLevel::Classical,
    }
}

/// Builds a `VertexCreator` from a KNOWN signing key (rather than
/// `new_with_random_key`) so tests can compute the matching `proposer`
/// public key and deliberately construct mismatches. Returns the creator
/// plus its proposer's real Ed25519 public key.
async fn make_test_creator(key_seed: u8) -> (VertexCreator, [u8; 32]) {
    let node_id = [0u8; 32];
    let quantum_vdf = Arc::new(
        QuantumVDF::new(fast_vdf_config())
            .await
            .expect("QuantumVDF::new failed"),
    );
    let signing_key = SigningKey::from_bytes(&[key_seed; 32]);
    let proposer_pubkey = signing_key.verifying_key().to_bytes();
    let creator = VertexCreator::new(node_id, Arc::new(signing_key), quantum_vdf);
    (creator, proposer_pubkey)
}

/// Builds a genesis-parented vertex (no transactions, no parents) with a
/// genuinely valid signature (signed by `creator`) AND a genuinely valid
/// VDF proof — i.e. a vertex that should pass `validate_vertex` end to end.
/// Individual tests then corrupt one field to exercise a specific rejection
/// path. Uses its own freshly-constructed `QuantumVDF` (same config as
/// `creator`'s) rather than reaching into `creator`'s private `quantum_vdf`
/// field. The VDF challenge is an arbitrary fixed value: VDF verification
/// only checks internal self-consistency (recomputes from
/// `proof.challenge`/`proof.difficulty` and compares) — it does not need to
/// match the vertex's own content, so this is independent of the signature
/// check under test here.
async fn build_valid_vertex(creator: &VertexCreator, proposer_pubkey: [u8; 32]) -> Vertex {
    let vertex_id = [0u8; 32]; // genesis vertex id
    let tx_root = [0u8; 32]; // no transactions -> tx_root is all-zero
    let round = 0u64;
    let parents: Vec<[u8; 32]> = vec![];
    let signature = creator.sign_vertex(&vertex_id, round, &tx_root, &parents);

    let quantum_vdf = QuantumVDF::new(fast_vdf_config())
        .await
        .expect("QuantumVDF::new failed");
    let vdf_proof = quantum_vdf
        .compute_proof(&[0xABu8; 32])
        .await
        .expect("compute_proof failed")
        .proof;

    Vertex {
        id: vertex_id,
        round,
        proposer: proposer_pubkey,
        transactions: vec![],
        parents,
        vdf_proof,
        timestamp: 0,
        signature,
    }
}

#[tokio::test]
async fn test_validate_vertex_rejects_missing_signature() {
    let (creator, proposer_pubkey) = make_test_creator(7).await;
    let mut vertex = build_valid_vertex(&creator, proposer_pubkey).await;
    vertex.signature = vec![];

    let is_valid = creator
        .validate_vertex(&vertex)
        .await
        .expect("validate_vertex should not error, just return Ok(false)");
    assert!(!is_valid, "a vertex with a missing signature must be rejected");
    println!("✅ test_validate_vertex_rejects_missing_signature PASSED");
}

#[tokio::test]
async fn test_validate_vertex_rejects_corrupted_signature() {
    let (creator, proposer_pubkey) = make_test_creator(8).await;
    let mut vertex = build_valid_vertex(&creator, proposer_pubkey).await;
    // Corrupt one byte of an otherwise-valid signature.
    vertex.signature[0] ^= 0xFF;

    let is_valid = creator.validate_vertex(&vertex).await.unwrap();
    assert!(!is_valid, "a vertex with a corrupted signature must be rejected");
    println!("✅ test_validate_vertex_rejects_corrupted_signature PASSED");
}

#[tokio::test]
async fn test_validate_vertex_rejects_wrong_key_signature() {
    // Vertex claims to be from `proposer_pubkey` (creator's real key) but is
    // actually signed by a completely different key.
    let (creator, proposer_pubkey) = make_test_creator(9).await;
    let mut vertex = build_valid_vertex(&creator, proposer_pubkey).await;

    let (wrong_creator, _wrong_pubkey) = make_test_creator(99).await;
    let tx_root = [0u8; 32];
    vertex.signature =
        wrong_creator.sign_vertex(&vertex.id, vertex.round, &tx_root, &vertex.parents);

    let is_valid = creator.validate_vertex(&vertex).await.unwrap();
    assert!(
        !is_valid,
        "a vertex signed by a key that does not match its claimed proposer must be rejected"
    );
    println!("✅ test_validate_vertex_rejects_wrong_key_signature PASSED");
}

#[tokio::test]
async fn test_validate_vertex_accepts_correctly_signed_vertex() {
    // Positive control: guards against the signature check being so strict
    // it blocks legitimately-signed vertices (the biggest named regression
    // risk for this fix).
    let (creator, proposer_pubkey) = make_test_creator(11).await;
    let vertex = build_valid_vertex(&creator, proposer_pubkey).await;

    let is_valid = creator
        .validate_vertex(&vertex)
        .await
        .expect("validate_vertex should not error for a valid vertex");
    assert!(is_valid, "a correctly-signed vertex with matching VDF proof must validate");
    println!("✅ test_validate_vertex_accepts_correctly_signed_vertex PASSED");
}
