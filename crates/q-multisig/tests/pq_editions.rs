//! Which post-quantum signature editions a multisig accepts (2026-10-19).
//!
//! The wallet page's "Register my keys" failed for every user with "Dilithium5 signature
//! verification failed": the page signed with dilithium-crystals (round-3.1 Dilithium,
//! 4,595-byte signature behind a 2-byte length header) and this crate + the register-key
//! handler verified with pqcrypto-dilithium 0.5 (the 2023 FIPS 204 draft). Same key size,
//! different maths. Both now go through `q_types::signature_verification::
//! verify_tx_pq_signature` (ML-DSA-87 final, then the draft) — the send path's verifier.
//!
//! The two fixtures are real artefacts, not hand-made bytes:
//! - `browser-mldsa87-register-key.json`: written by the patched wallet page's own code
//!   (`src/libp2p/postQuantumCrypto.mldsa.test.ts`, VEC_OUT=…) — the register-key
//!   challenge signed by `dilithium5Sign` with the key `generateDilithium5KeyPairFromMnemonic`
//!   derives from the test phrase below.
//! - `browser-round3-dilithium-crystals.json`: what the page signed BEFORE the fix
//!   (dilithium-crystals 1.0.6 `signDetached`, 4,597 bytes).

use ed25519_dalek::{Signer as _, SigningKey};
use fips204::ml_dsa_87;
use fips204::traits::{KeyGen, SerDes, Signer as _};
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _};
use sha3::{Digest, Sha3_256};

use q_multisig::{
    proposal::{MultisigAction, MultisigProposal, SignatureContribution},
    verify::{verify_member_signature, verify_proposal, VerifyError},
    wallet::{HybridPublicKey, Member, MultisigWallet},
};
use q_types::signature_verification::{verify_tx_pq_signature, PqVariant};

const PHRASE: &str =
    "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";

struct Vector {
    msg: Vec<u8>,
    pk: Vec<u8>,
    sig: Vec<u8>,
}

fn load(name: &str) -> Vector {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let h = |k: &str| hex::decode(j[k].as_str().unwrap()).unwrap();
    Vector { msg: h("msg"), pk: h("pk"), sig: h("sig") }
}

#[test]
fn browser_register_key_signature_verifies_as_mldsa87_final() {
    let v = load("browser-mldsa87-register-key.json");

    // The page's generateChallenge must be the server's register-key challenge:
    // SHA3-256(address(32) || timestamp_LE(8) || path). Inputs as in the TS test.
    let mut h = Sha3_256::new();
    h.update([0x11u8; 32]);
    h.update(1_792_420_000i64.to_le_bytes());
    h.update(b"/api/v1/multisig/register-key");
    assert_eq!(h.finalize().as_slice(), v.msg.as_slice(), "browser challenge != server challenge");

    assert_eq!(v.sig.len(), 4627);
    assert_eq!(verify_tx_pq_signature(&v.sig, &v.pk, &v.msg), Ok(PqVariant::MlDsa87Final));
}

#[test]
fn mnemonic_derivation_is_the_standard_not_one_library() {
    // noble (in the page) and fips204 (here) derive the same ML-DSA-87 key from the same
    // 32-byte seed — so any client implementing FIPS 204 can re-derive a wallet's PQ key.
    let v = load("browser-mldsa87-register-key.json");
    let seed: [u8; 32] = Sha3_256::digest(format!("qnk-dilithium-ml-dsa87-v1:{PHRASE}")).into();
    let (pk, _sk) = ml_dsa_87::KG::keygen_from_seed(&seed);
    assert_eq!(pk.into_bytes().as_slice(), v.pk.as_slice());
}

#[test]
fn old_round3_page_signature_is_refused_with_a_reason() {
    let v = load("browser-round3-dilithium-crystals.json");
    assert_eq!(v.sig.len(), 4597, "4,595-byte round-3.1 signature + 2-byte length header");
    let e = verify_tx_pq_signature(&v.sig, &v.pk, &v.msg).unwrap_err();
    assert!(e.contains("expected 4627"), "{e}");
    // Even with the header stripped it is a different edition, not a framing slip.
    assert!(verify_tx_pq_signature(&[&v.sig[2..], &[0u8; 32][..]].concat(), &v.pk, &v.msg).is_err());
}

enum Pq {
    Final(ml_dsa_87::PrivateKey),
    Draft(dilithium5::SecretKey),
}

struct Signer {
    ed: SigningKey,
    pq: Pq,
    member: Member,
}

fn signer(label: &str, seed_byte: u8, final_edition: bool) -> Signer {
    let ed = SigningKey::from_bytes(&[seed_byte; 32]);
    let (pq_pk, pq) = if final_edition {
        let (pk, sk) = ml_dsa_87::KG::keygen_from_seed(&[seed_byte; 32]);
        (pk.into_bytes().to_vec(), Pq::Final(sk))
    } else {
        let (pk, sk) = dilithium5::keypair();
        (pk.as_bytes().to_vec(), Pq::Draft(sk))
    };
    let member = Member {
        label: label.to_string(),
        pubkey: HybridPublicKey { ed25519: ed.verifying_key(), dilithium5: pq_pk },
    };
    Signer { ed, pq, member }
}

fn contribute(s: &Signer, p: &MultisigProposal) -> SignatureContribution {
    let payload = p.payload_hash();
    let dilithium5_signature = match &s.pq {
        Pq::Final(sk) => sk.try_sign(&payload, &[]).unwrap().to_vec(),
        Pq::Draft(sk) => dilithium5::detached_sign(&payload, sk).as_bytes().to_vec(),
    };
    SignatureContribution {
        member_addr: s.member.pubkey.member_address(),
        ed25519_sig: s.ed.sign(&payload),
        dilithium5_signature,
        at_unix: 1_792_420_000,
    }
}

fn transfer(wallet: &MultisigWallet) -> MultisigProposal {
    MultisigProposal::new(
        wallet.address,
        MultisigAction::Transfer {
            token: "QUG".to_string(),
            recipient: [0xAB; 32],
            amount_raw: 10_u128.pow(24),
            memo: None,
        },
        None,
        wallet.default_threshold,
        wallet.members.len(),
    )
}

#[test]
fn a_final_edition_member_and_a_draft_edition_member_co_sign_two_of_two() {
    let ceo = signer("ceo (browser, ML-DSA-87)", 3, true);
    let cfo = signer("cfo (old Rust client, draft)", 4, false);
    let wallet = MultisigWallet::new(vec![ceo.member.clone(), cfo.member.clone()], 2, "org").unwrap();
    let mut p = transfer(&wallet);
    assert_eq!(p.required, 2);

    let c1 = contribute(&ceo, &p);
    verify_member_signature(&wallet, &p, &c1).expect("ML-DSA-87 co-signature must count");
    p.add_signature(c1);
    let c2 = contribute(&cfo, &p);
    verify_member_signature(&wallet, &p, &c2).expect("draft co-signature must still count");
    p.add_signature(c2);
    verify_proposal(&wallet, &p).expect("2-of-2 with one signer of each edition");
}

#[test]
fn a_round3_sized_co_signature_is_bad_encoding_and_a_wrong_key_fails() {
    let ceo = signer("ceo", 5, true);
    let other = signer("other", 6, true);
    // A multisig needs at least two members; threshold 1 so one contribution decides.
    let wallet = MultisigWallet::new(vec![ceo.member.clone(), other.member.clone()], 1, "pair").unwrap();
    let p = transfer(&wallet);

    let mut c = contribute(&ceo, &p);
    c.dilithium5_signature = vec![0u8; 4597];
    assert!(matches!(verify_member_signature(&wallet, &p, &c), Err(VerifyError::Dilithium5BadEncoding(_))));

    // A correctly-shaped ML-DSA-87 signature from someone else's key must not count.
    let mut c = contribute(&ceo, &p);
    c.dilithium5_signature = contribute(&other, &p).dilithium5_signature;
    assert!(matches!(verify_member_signature(&wallet, &p, &c), Err(VerifyError::Dilithium5Failed(_))));
}

#[test]
fn the_real_pages_register_key_request_passes_the_handlers_checks() {
    // Captured from the BUILT wallet page in headless Chromium (register-key-e2e.mjs):
    // throwaway login with the test phrase → Organization & Multi-Sig → "Register my keys".
    // These are register_key()'s own steps, in its order.
    let path = format!("{}/tests/fixtures/browser-e2e-register-key-request.json", env!("CARGO_MANIFEST_DIR"));
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let addr: [u8; 32] = hex::decode(r["address"].as_str().unwrap().trim_start_matches("qnk"))
        .unwrap()
        .try_into()
        .unwrap();
    let mut h = Sha3_256::new();
    h.update(addr);
    h.update(r["timestamp"].as_i64().unwrap().to_le_bytes());
    h.update(b"/api/v1/multisig/register-key");
    let challenge = h.finalize();

    let ed_sig: [u8; 64] = hex::decode(r["ed25519_signature"].as_str().unwrap()).unwrap().try_into().unwrap();
    ed25519_dalek::VerifyingKey::from_bytes(&addr)
        .unwrap()
        .verify_strict(&challenge, &ed25519_dalek::Signature::from_bytes(&ed_sig))
        .expect("Ed25519 half");

    let sig = hex::decode(r["dilithium5_signature"].as_str().unwrap()).unwrap();
    let pk = hex::decode(r["dilithium5_public_key"].as_str().unwrap()).unwrap();
    assert_eq!(verify_tx_pq_signature(&sig, &pk, &challenge), Ok(PqVariant::MlDsa87Final));

    // Same phrase → same key, whether derived by the page at login or in the unit test.
    assert_eq!(pk, load("browser-mldsa87-register-key.json").pk);
}
