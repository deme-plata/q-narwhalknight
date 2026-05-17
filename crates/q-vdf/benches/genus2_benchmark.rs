//! Benchmark: Genus-2 Jacobian VDF doubling speed on pq192 curve
//!
//! PURPOSE: Determine how many iterations we need for a 2-4 second VDF on CPU.
//! This is prerequisite P1 — no production code is modified.
//!
//! Run with: cargo bench --package q-vdf --bench genus2_benchmark

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use num_bigint::BigInt;
use std::time::Duration;

use q_vdf::genus2_vdf::{Genus2CurveParams, Genus2VDF, JacobianElement};

/// Benchmark a single Jacobian doubling on pq192 (384-bit prime)
fn bench_single_doubling_pq192(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 1000);

    // Create a representative element from a hash
    let seed = blake3::hash(b"benchmark-seed-pq192").as_bytes().to_vec();
    let elem = JacobianElement::from_hash(&seed, &curve)
        .expect("Failed to create JacobianElement from hash");

    c.bench_function("genus2_pq192_single_doubling", |b| {
        let mut current = elem.clone();
        b.iter(|| {
            current = vdf.double_jacobian_pub(&current)
                .expect("doubling failed");
        });
    });
}

/// Benchmark a single Jacobian doubling on pq128 (256-bit prime) for comparison
fn bench_single_doubling_pq128(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq128();
    let vdf = Genus2VDF::with_curve(curve.clone(), 1000);

    let seed = blake3::hash(b"benchmark-seed-pq128").as_bytes().to_vec();
    let elem = JacobianElement::from_hash(&seed, &curve)
        .expect("Failed to create JacobianElement from hash");

    c.bench_function("genus2_pq128_single_doubling", |b| {
        let mut current = elem.clone();
        b.iter(|| {
            current = vdf.double_jacobian_pub(&current)
                .expect("doubling failed");
        });
    });
}

/// Benchmark N sequential doublings on pq192 to measure VDF evaluation time
/// This tells us how many iterations we need for 2-4 seconds
fn bench_sequential_chain_pq192(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), 1000);

    let seed = blake3::hash(b"benchmark-chain-pq192").as_bytes().to_vec();
    let elem = JacobianElement::from_hash(&seed, &curve)
        .expect("Failed to create JacobianElement from hash");

    let mut group = c.benchmark_group("genus2_pq192_sequential_chain");
    group.sample_size(10); // fewer samples for long benchmarks
    group.measurement_time(Duration::from_secs(30));

    for &iterations in &[100, 500, 1000, 2000, 5000] {
        group.bench_with_input(
            BenchmarkId::new("iterations", iterations),
            &iterations,
            |b, &iters| {
                b.iter(|| {
                    let mut current = elem.clone();
                    for _ in 0..iters {
                        current = vdf.double_jacobian_pub(&current)
                            .expect("doubling failed");
                    }
                    current
                });
            },
        );
    }
    group.finish();
}

/// Benchmark from_hash (one-time cost per nonce attempt)
fn bench_from_hash_pq192(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq192();

    c.bench_function("genus2_pq192_from_hash", |b| {
        let mut nonce = 0u64;
        b.iter(|| {
            let mut input = [0u8; 40];
            input[..32].copy_from_slice(b"benchmark-challenge-hash-1234567");
            input[32..].copy_from_slice(&nonce.to_le_bytes());
            nonce += 1;
            let seed = blake3::hash(&input);
            JacobianElement::from_hash(seed.as_bytes(), &curve)
                .expect("from_hash failed")
        });
    });
}

/// Benchmark to_bytes / from_bytes (serialization round-trip)
fn bench_serialization_pq192(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq192();
    let seed = blake3::hash(b"benchmark-serialize-pq192").as_bytes().to_vec();
    let elem = JacobianElement::from_hash(&seed, &curve)
        .expect("Failed to create JacobianElement");

    c.bench_function("genus2_pq192_to_bytes", |b| {
        b.iter(|| {
            elem.to_bytes()
        });
    });
}

/// Benchmark SHA3-256(vdf_output) — the final hash step in mining
fn bench_output_hash(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq192();
    let seed = blake3::hash(b"benchmark-output-hash").as_bytes().to_vec();
    let elem = JacobianElement::from_hash(&seed, &curve)
        .expect("Failed to create JacobianElement");
    let bytes = elem.to_bytes();

    c.bench_function("genus2_sha3_output_hash", |b| {
        b.iter(|| {
            use sha3::{Digest, Sha3_256};
            let mut hasher = Sha3_256::new();
            hasher.update(&bytes);
            hasher.finalize()
        });
    });
}

/// COMPARISON: Benchmark BLAKE3 x100 (current mining) for reference
fn bench_blake3_x100(c: &mut Criterion) {
    c.bench_function("blake3_x100_current_mining", |b| {
        let challenge = [0x42u8; 32];
        let mut nonce = 0u64;
        b.iter(|| {
            let mut input = [0u8; 40];
            input[..32].copy_from_slice(&challenge);
            input[32..].copy_from_slice(&nonce.to_le_bytes());
            nonce += 1;

            let mut current = *blake3::hash(&input).as_bytes();
            for _ in 0..99 {
                current = *blake3::hash(&current).as_bytes();
            }
            current
        });
    });
}

criterion_group!(
    benches,
    bench_single_doubling_pq192,
    bench_single_doubling_pq128,
    bench_sequential_chain_pq192,
    bench_from_hash_pq192,
    bench_serialization_pq192,
    bench_output_hash,
    bench_blake3_x100,
);
criterion_main!(benches);
