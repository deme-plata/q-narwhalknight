//! Adversarial ground-truth tests, added 2026-07-08.
//!
//! Purpose: convert two source-reading-based claims from the design-review audit into
//! empirically observed facts, at zero risk to real funds (no server, no real wallet,
//! no mainnet state touched -- pure `cargo test` against the crate's own real crypto).
//!
//! Claim 1 (this file's main point): the ONLY double-spend guard for CLSAG ring
//! signatures is `CLSAGSigner.used_key_images`, a HashSet living on the signer object
//! itself. The existing test `test_clsag_double_spend_prevention` in clsag.rs's own
//! unit tests proves the guard works WITHIN a single persistent signer instance. But
//! `crates/q-api-server/src/privacy_service_api.rs::ring_signature_service` constructs
//! a BRAND NEW `CLSAGSigner::new(...)` on every single HTTP request (confirmed via
//! source read, line ~564) -- so in production there is never a second call against the
//! *same* signer object. This test reproduces that exact production pattern: same
//! wallet (same private key bytes), two independently-constructed signer instances
//! (simulating two separate HTTP requests), and checks whether the second "spend"
//! is rejected the way it must be for double-spend prevention to mean anything.
//!
//! Claim 2: the crate's existing Bulletproofs+ tests (`test_range_proof_out_of_range`,
//! `test_invalid_proof_rejection` in bulletproofs_pp.rs) are real, already in-tree
//! adversarial checks -- this file re-runs the same properties independently to confirm
//! they hold under `cargo test`, not just under a static read of the assertions.

use q_quantum_mixing::clsag::{
    CLSAGSigner, create_pedersen_commitment, generate_commitment_mask,
};
use q_quantum_mixing::quantum_entropy::QuantumEntropyPool;
use q_quantum_mixing::bulletproofs_pp::{BPPlusConfig, BPPlusRangeProof};
use std::sync::Arc;

/// THE core empirical question: does the double-spend guard survive across two
/// independently-constructed CLSAGSigner instances for the SAME wallet, the way
/// `ring_signature_service` actually instantiates it per request?
///
/// If this test's second signature SUCCEEDS, the production double-spend guard is
/// confirmed to be a no-op across API calls -- exactly as the source reading predicted,
/// now observed rather than inferred.
#[tokio::test]
async fn ground_truth_double_spend_guard_does_not_survive_fresh_instances() {
    // Same wallet both times: derive the same keypair from fixed private-key bytes,
    // exactly the shape `CLSAGSigner::from_private_key` expects a real wallet key to be.
    let wallet_private_key: [u8; 32] = [7u8; 32];

    // Also need a second ring member's public key -- CLSAG requires ring.len() >= 1,
    // real usage needs at least 2 for actual ambiguity, so give it a genuine decoy.
    let entropy_for_decoy = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let decoy = CLSAGSigner::new(entropy_for_decoy).await.unwrap();
    let decoy_pubkey = decoy.get_public_key();

    let mask = {
        let e = Arc::new(QuantumEntropyPool::new().await.unwrap());
        generate_commitment_mask(&e).await.unwrap()
    };
    let (commitment, _) = create_pedersen_commitment(1_000_000, &mask);

    // ---- "Request #1": server spins up a fresh entropy pool + fresh signer ----
    let entropy_1 = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer_1 = CLSAGSigner::from_private_key(wallet_private_key, entropy_1)
        .await
        .unwrap();
    let ring = vec![signer_1.get_public_key(), decoy_pubkey];

    let sig1 = signer_1
        .sign(b"withdraw request #1", &ring, &commitment, &mask)
        .await;
    assert!(sig1.is_ok(), "first withdrawal for this wallet must succeed");
    let key_image_1 = sig1.unwrap().get_key_image().to_owned();

    // ---- "Request #2": a SECOND, unrelated HTTP call for the SAME wallet,
    //      exactly as ring_signature_service does it in production: a fresh
    //      QuantumEntropyPool::new() + fresh CLSAGSigner::new(...)/from_private_key
    //      per call, with no shared state between requests. ----
    let entropy_2 = Arc::new(QuantumEntropyPool::new().await.unwrap());
    let mut signer_2 = CLSAGSigner::from_private_key(wallet_private_key, entropy_2)
        .await
        .unwrap();

    let sig2 = signer_2
        .sign(b"withdraw request #2 -- SAME WALLET, SHOULD BE REJECTED", &ring, &commitment, &mask)
        .await;

    let key_image_2 = signer_2.compute_key_image();
    assert_eq!(
        key_image_1, key_image_2,
        "sanity check: the same wallet must derive the same key image both times \
         (if this fails, the two 'requests' aren't actually the same spender and the \
         test below proves nothing)"
    );

    // THE ACTUAL FINDING: if the production per-request instantiation pattern is
    // unsafe, sig2 will be Ok -- the second signer's `used_key_images` set starts
    // empty, so it never sees the first signature's key image at all.
    if sig2.is_ok() {
        panic!(
            "CONFIRMED VULNERABLE: a second withdrawal request for the SAME wallet \
             (same key image {:x?}) succeeded because the double-spend guard lives on \
             a per-instance HashSet and production constructs a fresh instance per \
             HTTP request. This reproduces, empirically, the exact gap flagged by \
             source review: CLSAGSigner.used_key_images provides ZERO real double-spend \
             protection in the current ring_signature_service handler shape. A real fix \
             requires moving this check into persisted, atomically-checked state shared \
             across requests -- not a per-object in-memory set.",
            key_image_1
        );
    } else {
        println!(
            "Second signature was rejected ({:?}) -- double-spend guard held across \
             fresh instances. This would mean the vulnerability does NOT reproduce this \
             way and the real guard must live somewhere else than believed.",
            sig2.err()
        );
    }
}

/// Re-confirm, by actually running it, that a range proof for an out-of-range value
/// is genuinely rejected at construction time -- not just per the source-level
/// assertion in the crate's own test suite.
#[test]
fn ground_truth_bulletproof_rejects_out_of_range_amount() {
    let config = BPPlusConfig::new(8, 1).unwrap();
    let out_of_range_value = 300u64; // 2^8 = 256 is the ceiling for an 8-bit range

    let result = BPPlusRangeProof::prove(&config, out_of_range_value);
    assert!(
        result.is_err(),
        "an 8-bit range proof for value=300 must be rejected at construction; if this \
         assertion fails, the real range-proof primitive silently accepts amounts it \
         should reject"
    );
}

/// Re-confirm, by actually running it, that a tampered (bit-flipped) valid proof
/// fails verification rather than silently passing.
#[test]
fn ground_truth_bulletproof_rejects_tampered_proof() {
    use curve25519_dalek::scalar::Scalar;

    let config = BPPlusConfig::new(8, 1).unwrap();
    let (mut proof, _blinding) = BPPlusRangeProof::prove(&config, 42).unwrap();

    // Tamper with one scalar in the proof after honest construction.
    proof.t_hat = proof.t_hat + Scalar::ONE;

    let result = proof.verify(&config);
    assert!(
        result.is_err(),
        "a tampered Bulletproofs+ range proof must fail verification; if this \
         assertion fails, the real verifier silently accepts corrupted proofs"
    );
}
