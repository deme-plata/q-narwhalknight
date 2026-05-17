// SIMD Crypto Benchmarks Entry Point
// Criterion-based performance benchmarking for SIMD operations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use q_crypto_simd::*;
use q_types::{Hash256, PublicKey, Signature};
use tokio::runtime::Runtime;

fn criterion_benchmark(c: &mut Criterion) {
    bench_batch_verification(c);
    bench_simd_hashing(c);
    bench_merkle_computation(c);
    bench_memory_alignment(c);
}

/// Benchmark batch signature verification
fn bench_batch_verification(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let cpu_features = detect_cpu_features();

    let verifier = rt.block_on(async {
        BatchSignatureVerifier::new(&cpu_features, 64)
            .await
            .unwrap()
    });

    let mut group = c.benchmark_group("batch_verification");

    for batch_size in [1, 4, 8, 16, 32, 64].iter() {
        let (signatures, messages, public_keys) = generate_ed25519_test_data(*batch_size);
        let message_refs: Vec<&[u8]> = messages.iter().map(|m| m.as_slice()).collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("ed25519_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        verifier
                            .verify_batch(&signatures, &message_refs, &public_keys)
                            .await
                            .unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SIMD hash computation
fn bench_simd_hashing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let cpu_features = detect_cpu_features();

    let hasher = rt.block_on(async { SimdHasher::new(&cpu_features, 64).await.unwrap() });

    let mut group = c.benchmark_group("simd_hashing");

    for batch_size in [8, 16, 32, 64].iter() {
        let test_data = generate_hash_test_data(*batch_size, 1024);
        let data_refs: Vec<&[u8]> = test_data.iter().map(|d| d.as_slice()).collect();

        group.throughput(Throughput::Bytes((*batch_size * 1024) as u64));

        // Blake3 benchmark
        group.bench_with_input(
            BenchmarkId::new("blake3_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        hasher
                            .compute_batch(&data_refs, HashAlgorithm::Blake3)
                            .await
                            .unwrap()
                    })
                });
            },
        );

        // SHA-256 benchmark
        group.bench_with_input(
            BenchmarkId::new("sha256_batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        hasher
                            .compute_batch(&data_refs, HashAlgorithm::Sha256)
                            .await
                            .unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark merkle tree computation
fn bench_merkle_computation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let cpu_features = detect_cpu_features();

    let hasher = rt.block_on(async { SimdHasher::new(&cpu_features, 64).await.unwrap() });

    let mut group = c.benchmark_group("merkle_trees");

    for leaf_count in [16, 64, 256, 1024].iter() {
        let leaves = generate_hash_test_data(*leaf_count, 32);
        let leaf_refs: Vec<&[u8]> = leaves.iter().map(|l| l.as_slice()).collect();

        group.throughput(Throughput::Elements(*leaf_count as u64));
        group.bench_with_input(
            BenchmarkId::new("simd_merkle", leaf_count),
            leaf_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async { hasher.compute_merkle_root(&leaf_refs).await.unwrap() })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory alignment operations
fn bench_memory_alignment(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let cache = rt.block_on(async { SimdCache::new(64).await.unwrap() });

    let mut group = c.benchmark_group("memory_alignment");

    for data_size in [64, 256, 1024, 4096].iter() {
        let test_data = vec![0xAAu8; *data_size];

        group.throughput(Throughput::Bytes(*data_size as u64));
        group.bench_with_input(
            BenchmarkId::new("align_buffer", data_size),
            data_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let buffer = cache.align_buffer(&test_data).await.unwrap();
                        cache.return_buffer(buffer).await.unwrap();
                    })
                });
            },
        );
    }

    group.finish();
}

/// Generate Ed25519 test signatures
fn generate_ed25519_test_data(count: usize) -> (Vec<Signature>, Vec<Vec<u8>>, Vec<PublicKey>) {
    let mut signatures = Vec::new();
    let mut messages = Vec::new();
    let mut public_keys = Vec::new();

    for i in 0..count {
        // Create deterministic test data
        let mut sig_bytes = [0u8; 64];
        let mut key_bytes = [0u8; 32];

        for (j, byte) in sig_bytes.iter_mut().enumerate() {
            *byte = ((i * 64 + j) % 256) as u8;
        }

        for (j, byte) in key_bytes.iter_mut().enumerate() {
            *byte = ((i * 32 + j + 128) % 256) as u8;
        }

        signatures.push(Signature::Ed25519(sig_bytes));
        messages.push(format!("benchmark message {}", i).into_bytes());
        public_keys.push(PublicKey::Ed25519(key_bytes));
    }

    (signatures, messages, public_keys)
}

/// Generate test data for hashing
fn generate_hash_test_data(count: usize, size_per_item: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| {
            let mut data = vec![0u8; size_per_item];
            // Pseudo-random fill for realistic hashing scenarios
            for (j, byte) in data.iter_mut().enumerate() {
                *byte = ((i.wrapping_mul(251) + j.wrapping_mul(199)) % 256) as u8;
            }
            data
        })
        .collect()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
