//! Benchmark: Parallel VDF chains (GPU attack simulation)
//!
//! PURPOSE: Simulate what a GPU would do — run N independent VDF chains
//! in parallel and measure total solutions/sec vs single-thread.
//! If the ratio exceeds 20x, we need longer VDF or memory-hardness.
//!
//! This is prerequisite P2 — no production code is modified.
//!
//! Run with: cargo bench --package q-vdf --bench parallel_chains_benchmark

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use q_vdf::genus2_vdf::{Genus2CurveParams, Genus2VDF, JacobianElement};

/// Number of VDF iterations per chain (tune based on P1 benchmark results)
/// Start with 1000 — adjust once we know single-doubling speed
const VDF_ITERATIONS: u64 = 1000;

/// Run a single VDF chain: hash seed → N sequential doublings → output hash
fn run_single_chain(
    vdf: &Genus2VDF,
    curve: &Genus2CurveParams,
    challenge: &[u8; 32],
    nonce: u64,
    iterations: u64,
) -> [u8; 32] {
    // Step 1: seed = BLAKE3(challenge || nonce)
    let mut input = [0u8; 40];
    input[..32].copy_from_slice(challenge);
    input[32..].copy_from_slice(&nonce.to_le_bytes());
    let seed = blake3::hash(&input);

    // Step 2: map to Jacobian element
    let mut g = JacobianElement::from_hash(seed.as_bytes(), curve)
        .expect("from_hash failed");

    // Step 3: sequential doublings (the VDF — cannot be parallelized within a chain)
    for _ in 0..iterations {
        g = vdf.double_jacobian_pub(&g).expect("doubling failed");
    }

    // Step 4: output hash = SHA3-256(vdf_output)
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(&g.to_bytes());
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Benchmark: single-threaded VDF chains (baseline)
fn bench_single_thread_chains(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq192();
    let vdf = Genus2VDF::with_curve(curve.clone(), VDF_ITERATIONS);
    let challenge = [0x42u8; 32];

    let mut group = c.benchmark_group("parallel_chains_single_thread");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Run 5 sequential chains
    group.bench_function("5_chains_sequential", |b| {
        b.iter(|| {
            for nonce in 0..5u64 {
                run_single_chain(&vdf, &curve, &challenge, nonce, VDF_ITERATIONS);
            }
        });
    });

    group.finish();
}

/// Benchmark: multi-threaded VDF chains (GPU simulation)
/// Spawns N threads, each running independent VDF chains
fn bench_parallel_chains(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq192();
    let challenge = [0x42u8; 32];

    let mut group = c.benchmark_group("parallel_chains_gpu_simulation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let available_cores = num_cpus::get();

    for &num_threads in &[1, 2, 4, 8, 16, 32, 64] {
        if num_threads > available_cores * 2 {
            continue; // Skip unreasonable thread counts
        }

        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let chains_completed = Arc::new(AtomicU64::new(0));
                    let mut handles = Vec::new();

                    for thread_id in 0..threads {
                        let curve_clone = curve.clone();
                        let completed = Arc::clone(&chains_completed);

                        handles.push(std::thread::spawn(move || {
                            let vdf = Genus2VDF::with_curve(curve_clone.clone(), VDF_ITERATIONS);
                            // Each thread runs 1 chain with its own nonce
                            run_single_chain(
                                &vdf,
                                &curve_clone,
                                &challenge,
                                thread_id as u64,
                                VDF_ITERATIONS,
                            );
                            completed.fetch_add(1, Ordering::Relaxed);
                        }));
                    }

                    for h in handles {
                        h.join().expect("thread panicked");
                    }

                    chains_completed.load(Ordering::Relaxed)
                });
            },
        );
    }

    group.finish();
}

/// Quick throughput measurement: chains per second at different thread counts
/// Prints results directly for easy comparison
fn bench_throughput_comparison(c: &mut Criterion) {
    let curve = Genus2CurveParams::pq192();
    let challenge = [0x42u8; 32];

    let mut group = c.benchmark_group("throughput_chains_per_second");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let available_cores = num_cpus::get();

    // Single thread baseline
    group.bench_function("1_thread_baseline", |b| {
        let vdf = Genus2VDF::with_curve(curve.clone(), VDF_ITERATIONS);
        b.iter(|| {
            run_single_chain(&vdf, &curve, &challenge, 0, VDF_ITERATIONS)
        });
    });

    // All cores (simulates GPU with many independent chains)
    let max_threads = available_cores.min(64);
    group.bench_function(
        &format!("{}_threads_all_cores", max_threads),
        |b| {
            b.iter(|| {
                let mut handles = Vec::new();
                for tid in 0..max_threads {
                    let curve_clone = curve.clone();
                    handles.push(std::thread::spawn(move || {
                        let vdf = Genus2VDF::with_curve(curve_clone.clone(), VDF_ITERATIONS);
                        run_single_chain(&vdf, &curve_clone, &challenge, tid as u64, VDF_ITERATIONS)
                    }));
                }
                let mut results = Vec::new();
                for h in handles {
                    results.push(h.join().expect("thread panicked"));
                }
                results
            });
        },
    );

    group.finish();
}

/// COMPARISON: BLAKE3 x100 parallel (what GPU miners actually do today)
fn bench_blake3_parallel_comparison(c: &mut Criterion) {
    let challenge = [0x42u8; 32];
    let available_cores = num_cpus::get();

    let mut group = c.benchmark_group("blake3_x100_parallel_comparison");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    // Single thread BLAKE3 x100
    group.bench_function("blake3_x100_1_thread", |b| {
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

    // All cores BLAKE3 x100 (each thread tries different nonces)
    let max_threads = available_cores.min(64);
    group.bench_function(
        &format!("blake3_x100_{}_threads", max_threads),
        |b| {
            b.iter(|| {
                let completed = Arc::new(AtomicU64::new(0));
                let mut handles = Vec::new();
                for tid in 0..max_threads {
                    let comp = Arc::clone(&completed);
                    handles.push(std::thread::spawn(move || {
                        // Each thread does 100 nonce attempts
                        for n in 0..100u64 {
                            let nonce = tid as u64 * 1000 + n;
                            let mut input = [0u8; 40];
                            input[..32].copy_from_slice(&challenge);
                            input[32..].copy_from_slice(&nonce.to_le_bytes());
                            let mut current = *blake3::hash(&input).as_bytes();
                            for _ in 0..99 {
                                current = *blake3::hash(&current).as_bytes();
                            }
                        }
                        comp.fetch_add(100, Ordering::Relaxed);
                    }));
                }
                for h in handles {
                    h.join().expect("thread panicked");
                }
                completed.load(Ordering::Relaxed)
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread_chains,
    bench_parallel_chains,
    bench_throughput_comparison,
    bench_blake3_parallel_comparison,
);
criterion_main!(benches);
