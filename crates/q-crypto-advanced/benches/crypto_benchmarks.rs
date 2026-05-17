//! Benchmarks for Advanced Cryptographic Primitives
//!
//! Compares performance of:
//! - AEGIS-256 vs AES-256-GCM
//! - FROST threshold signatures
//! - Circle STARKs vs traditional STARKs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use q_crypto_advanced::{
    aegis::{Aegis256, AegisKey, AegisNonce},
    frost::{FrostKeyGen, FrostSigner},
    circle_stark::{CircleStarkProver, CircleStarkVerifier, FIELD_MODULUS},
};
use std::collections::BTreeMap;

/// Add two numbers modulo FIELD_MODULUS (for Fibonacci trace generation)
fn add_mod(a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % FIELD_MODULUS as u128) as u64
}

fn bench_aegis_vs_aes(c: &mut Criterion) {
    let mut group = c.benchmark_group("AEGIS-256 vs AES-256-GCM");

    // Test different data sizes
    for size in [64, 1024, 16384, 65536, 1048576].iter() {
        let plaintext: Vec<u8> = (0..*size).map(|i| (i % 256) as u8).collect();
        let aad = b"benchmark-test";

        // AEGIS-256
        let aegis_key = AegisKey::generate();
        let aegis_nonce = AegisNonce::generate();

        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("AEGIS-256 Encrypt", size), size, |b, _| {
            b.iter(|| {
                Aegis256::encrypt(
                    black_box(&aegis_key),
                    black_box(&aegis_nonce),
                    black_box(&plaintext),
                    black_box(aad),
                )
            })
        });

        let ciphertext = Aegis256::encrypt(&aegis_key, &aegis_nonce, &plaintext, aad).unwrap();

        group.bench_with_input(BenchmarkId::new("AEGIS-256 Decrypt", size), size, |b, _| {
            b.iter(|| {
                Aegis256::decrypt(
                    black_box(&aegis_key),
                    black_box(&aegis_nonce),
                    black_box(&ciphertext),
                    black_box(aad),
                )
            })
        });
    }

    group.finish();
}

fn bench_frost_signing(c: &mut Criterion) {
    let mut group = c.benchmark_group("FROST Threshold Signatures");

    // Test different threshold configurations
    for (t, n) in [(2, 3), (3, 5), (5, 7), (7, 10)].iter() {
        let (key_shares, pubkey) = FrostKeyGen::generate_shares(*t, *n).unwrap();

        let mut signers: Vec<FrostSigner> = key_shares
            .into_iter()
            .map(FrostSigner::from_share)
            .collect();

        let message = b"Block #12345 to be signed by validator committee";

        group.bench_with_input(
            BenchmarkId::new("Keygen", format!("{}-of-{}", t, n)),
            &(*t, *n),
            |b, (t, n)| {
                b.iter(|| FrostKeyGen::generate_shares(black_box(*t), black_box(*n)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Round 1 (Commit)", format!("{}-of-{}", t, n)),
            &(*t, *n),
            |b, _| {
                b.iter(|| {
                    for signer in &mut signers {
                        signer.round1_commit();
                    }
                })
            },
        );

        // Full signing round
        let threshold = *t;  // Copy for use in closure
        group.bench_with_input(
            BenchmarkId::new("Full Sign", format!("{}-of-{}", t, n)),
            &(*t, *n),
            |b, _| {
                b.iter(|| {
                    // Round 1: Commit
                    let mut commitments = BTreeMap::new();
                    let mut nonces_list = Vec::new();

                    for signer in &mut signers {
                        let (commitment, nonces) = signer.round1_commit();
                        let id = *signer.frost_identifier();
                        commitments.insert(id, commitment);
                        nonces_list.push(nonces);
                    }

                    // Round 2: Sign (only t participants)
                    let mut sig_shares = BTreeMap::new();
                    for (i, signer) in signers.iter_mut().take(threshold as usize).enumerate() {
                        let sig_share = signer
                            .round2_sign(black_box(message), &commitments, Some(nonces_list[i].clone()))
                            .unwrap();
                        let id = *signer.frost_identifier();
                        sig_shares.insert(id, sig_share);
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_circle_stark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Circle STARK Proofs");

    // Test different trace sizes
    for log_size in [3, 4, 5, 6].iter() {
        let size = 1 << log_size;

        // Generate a simple Fibonacci-like trace
        let trace: Vec<Vec<u64>> = (0..size)
            .scan((1u64, 1u64), |state, _| {
                let result = vec![state.0, state.1];
                *state = (state.1, add_mod(state.0, state.1));
                Some(result)
            })
            .collect();

        let constraints = |curr: &[u64], next: &[u64]| -> Vec<u64> {
            vec![
                next[0].wrapping_sub(curr[1]) % 2147483647,
                next[1].wrapping_sub(curr[0].wrapping_add(curr[1])) % 2147483647,
            ]
        };

        let prover = CircleStarkProver::new(*log_size, 4, 8).unwrap();

        group.bench_with_input(
            BenchmarkId::new("Prove", format!("trace_size_{}", size)),
            &trace,
            |b, trace| {
                b.iter(|| prover.prove(black_box(trace), constraints))
            },
        );

        let proof = prover.prove(&trace, constraints).unwrap();
        let verifier = CircleStarkVerifier::new(size, 4);

        group.bench_with_input(
            BenchmarkId::new("Verify", format!("trace_size_{}", size)),
            &proof,
            |b, proof| {
                b.iter(|| verifier.verify(black_box(proof)))
            },
        );

        // Log proof size
        if let Ok(serialized) = bincode::serialize(&proof) {
            println!("Circle STARK proof size (trace_size={}): {} bytes", size, serialized.len());
        }
    }

    group.finish();
}

fn bench_key_derivation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Key Derivation");

    let password = b"secure_password_here_123!";
    let salt = b"random_salt_16bytes!";

    group.bench_function("AEGIS Key from Password (Argon2id + HKDF)", |b| {
        b.iter(|| AegisKey::derive_from_password(black_box(password), black_box(salt)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_aegis_vs_aes,
    bench_frost_signing,
    bench_circle_stark,
    bench_key_derivation,
);

criterion_main!(benches);
