//! Phase 0 vertex-signature regression tests — 2026-07-08
//!
//! Pins Patch 1: `NarwhalCore::validate_vertex` (crates/q-narwhal-core/src/lib.rs)
//! previously had two literal `// TODO: Validate signature` / commented-out
//! `self.verify_vertex_signature(vertex)?;` lines and fell through to Ok(())
//! regardless of the vertex's signature. Any vertex — forged, unsigned, or
//! with a corrupted signature — was accepted by `process_vertex`.
//!
//! This crate's own note on scope (see NarwhalCore's live-topology audit,
//! this session): under the current single-producer/self-signed-certificate
//! topology, `q_narwhal_core::process_vertex`/`validate_vertex` is NOT on
//! the live money-moving path (that goes through main.rs's hand-built
//! Certificate + block_producer.rs instead). This is correctness-for-the-
//! future (multi-validator DAG activation), not an urgent live fix — but a
//! real bug, and the fix is pinned here so it cannot silently regress back
//! to Ok(()) before that topology change ships.
//!
//! Run: cargo test --package q-narwhal-core --test phase0_vertex_signature_tests

use chrono::Utc;
use ed25519_dalek::{Signer, SigningKey};
use q_narwhal_core::NarwhalCore;
use q_types::Vertex;

fn base_vertex(author_signer: &SigningKey, round: u64) -> Vertex {
    let author: [u8; 32] = author_signer.verifying_key().to_bytes();
    Vertex {
        id: [0x42u8; 32],
        round,
        author,
        // Empty transactions -> compute_tx_root returns [0u8;32] (see
        // NarwhalCore::compute_tx_root), so this matches with no txs.
        tx_root: [0u8; 32],
        parents: vec![],
        transactions: vec![],
        signature: vec![],
        timestamp: Utc::now(),
    }
}

/// Signs a vertex exactly the way the (already-correct) sibling
/// implementation in reliable_broadcast.rs::validate_vertex verifies it —
/// SHA3-256(id || round.to_le_bytes() || tx_root || parents) — which is
/// also exactly what Patch 1's new code in lib.rs::validate_vertex checks.
fn sign_vertex(vertex: &mut Vertex, signer: &SigningKey) {
    use sha3::{Digest, Sha3_256};
    let mut signing_data = Vec::new();
    signing_data.extend_from_slice(&vertex.id);
    signing_data.extend_from_slice(&vertex.round.to_le_bytes());
    signing_data.extend_from_slice(&vertex.tx_root);
    for parent in &vertex.parents {
        signing_data.extend_from_slice(parent);
    }
    let message_hash = Sha3_256::digest(&signing_data);
    let signature = signer.sign(&message_hash);
    vertex.signature = signature.to_bytes().to_vec();
}

// ============================================================================
// TEST 1 — A vertex with NO signature is now rejected (pre-Patch-1: Ok(())).
// ============================================================================

#[tokio::test]
async fn test_vertex_with_missing_signature_is_rejected() {
    let node_id = [1u8; 32];
    let narwhal = NarwhalCore::new(node_id);

    let author_signer = SigningKey::from_bytes(&[21u8; 32]);
    let vertex = base_vertex(&author_signer, 1); // signature left empty

    let result = narwhal.process_vertex(vertex).await;
    assert!(
        result.is_err(),
        "a vertex with a missing signature must be rejected post-Patch-1 \
         (pre-patch this returned Ok(()) unconditionally)"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.to_lowercase().contains("signature"),
        "expected a signature-related error message, got: {}",
        msg
    );

    println!("✅ test_vertex_with_missing_signature_is_rejected PASSED ({})", msg);
}

// ============================================================================
// TEST 2 — A vertex with a corrupted (present but invalid) signature is
// rejected.
// ============================================================================

#[tokio::test]
async fn test_vertex_with_corrupted_signature_is_rejected() {
    let node_id = [2u8; 32];
    let narwhal = NarwhalCore::new(node_id);

    let author_signer = SigningKey::from_bytes(&[22u8; 32]);
    let mut vertex = base_vertex(&author_signer, 1);
    sign_vertex(&mut vertex, &author_signer);

    // Corrupt one byte of an otherwise-valid signature.
    vertex.signature[0] ^= 0xFF;

    let result = narwhal.process_vertex(vertex).await;
    assert!(
        result.is_err(),
        "a vertex with a corrupted signature must be rejected post-Patch-1"
    );

    println!("✅ test_vertex_with_corrupted_signature_is_rejected PASSED ({})", result.unwrap_err());
}

// ============================================================================
// TEST 3 — A vertex signed by a DIFFERENT key than its `author` field
// claims is rejected (forged-author check).
// ============================================================================

#[tokio::test]
async fn test_vertex_signed_by_wrong_key_is_rejected() {
    let node_id = [3u8; 32];
    let narwhal = NarwhalCore::new(node_id);

    let claimed_author_signer = SigningKey::from_bytes(&[23u8; 32]);
    let actual_signer = SigningKey::from_bytes(&[24u8; 32]); // different key

    let mut vertex = base_vertex(&claimed_author_signer, 1);
    // Sign with the WRONG key while vertex.author still claims claimed_author_signer's pubkey.
    sign_vertex(&mut vertex, &actual_signer);

    let result = narwhal.process_vertex(vertex).await;
    assert!(
        result.is_err(),
        "a vertex signed by a key that does not match its claimed `author` must be rejected"
    );

    println!("✅ test_vertex_signed_by_wrong_key_is_rejected PASSED ({})", result.unwrap_err());
}

// ============================================================================
// TEST 4 — Positive control: a correctly-signed vertex clears the
// signature check (may still error later on the separate, not-yet-fixed
// parent-references TODO, or succeed outright — either is fine; what this
// test pins is that the signature check itself does NOT reject valid input).
// ============================================================================

#[tokio::test]
async fn test_correctly_signed_vertex_passes_signature_check() {
    let node_id = [4u8; 32];
    let narwhal = NarwhalCore::new(node_id);

    let author_signer = SigningKey::from_bytes(&[25u8; 32]);
    let mut vertex = base_vertex(&author_signer, 1);
    sign_vertex(&mut vertex, &author_signer);

    let result = narwhal.process_vertex(vertex).await;
    // The parent-references TODO is explicitly NOT part of this patch (still
    // a no-op), and later pipeline stages (vertex_store/reliable_broadcast)
    // may fail for unrelated setup reasons in this minimal test harness — so
    // this test only asserts the failure is NOT a signature-related one.
    if let Err(e) = &result {
        let msg = e.to_string().to_lowercase();
        assert!(
            !msg.contains("signature"),
            "a correctly-signed vertex must not fail the signature check, but got: {}",
            msg
        );
    }

    println!("✅ test_correctly_signed_vertex_passes_signature_check PASSED (result: {:?})",
        result.as_ref().map(|_| "Ok").map_err(|e| e.to_string()));
}
