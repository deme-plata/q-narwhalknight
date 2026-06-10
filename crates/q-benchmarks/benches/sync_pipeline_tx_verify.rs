//! TR-2026-004 Phase 4 measurement bench.
//!
//! Question: when sync receives a pack of N blocks with K tx-sigs each, does
//! routing tx-sig verification through `ParallelEd25519Verifier::verify_batch_chunked`
//! (what `crates/q-storage/src/sync_pipeline.rs::validation_stage` does)
//! actually beat a sequential `tx.verify_signature()` loop (what the current
//! `dag_sync_manager.rs` / `warp_sync.rs` paths do) by a margin large enough
//! to justify wiring SyncPipeline into the live block-pack receive path?
//!
//! Decision gate: wire-up proceeds (task #44) only if a representative load
//! (pack_size = 1024, txs_per_block = 32) shows the parallel path delivers
//! >=15% wall-clock reduction over the sequential path on Epsilon Docker.
//!
//! Sweep: total_sigs ∈ {128, 1024, 4096, 32768} — covering everything from a
//! near-empty pack to a saturated genesis-scan pack with 32 tx/block × 1024
//! blocks.
//!
//! Comparison is verification-only. Decompress / store / RocksDB write are
//! identical between baseline and pipeline and are excluded from this bench.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey, Signature};
use q_crypto_simd::parallel_ed25519::ParallelEd25519Verifier;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Generate a stable, deterministic set of (message, signature, pubkey)
/// triples in the format the parallel verifier expects (owned `Vec<u8>`s,
/// matching the `validation_stage` call site at sync_pipeline.rs:167-169).
fn fixture(n: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    // Deterministic seed so repeated bench runs produce the same workload.
    let mut rng = StdRng::seed_from_u64(0xDA6_C171F7);
    let mut messages = Vec::with_capacity(n);
    let mut signatures = Vec::with_capacity(n);
    let mut public_keys = Vec::with_capacity(n);

    for i in 0..n {
        // Real Quillon transactions hash to 32 bytes via Transaction::hash().
        // The sync_pipeline.rs validation_stage passes that 32-byte hash as
        // the verification message (see line 134: tx_item.hash().to_vec()).
        let mut msg = [0u8; 32];
        msg[..8].copy_from_slice(&(i as u64).to_le_bytes());
        msg[8..16].copy_from_slice(&(i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).to_le_bytes());

        let sk = SigningKey::generate(&mut rng);
        let sig = sk.sign(&msg);
        let vk: VerifyingKey = sk.verifying_key();

        messages.push(msg.to_vec());
        signatures.push(sig.to_bytes().to_vec());
        public_keys.push(vk.to_bytes().to_vec());
    }

    (messages, signatures, public_keys)
}

/// Sequential verify — what `dag_sync_manager.rs` and `warp_sync.rs` do today
/// when synced blocks arrive. Each signature is checked one by one with no
/// batching or rayon distribution.
fn verify_sequential(messages: &[Vec<u8>], signatures: &[Vec<u8>], public_keys: &[Vec<u8>]) -> usize {
    let mut ok_count = 0usize;
    for ((msg, sig), pk) in messages.iter().zip(signatures.iter()).zip(public_keys.iter()) {
        let Ok(vk) = VerifyingKey::from_bytes(pk.as_slice().try_into().unwrap_or(&[0u8; 32])) else { continue; };
        let Ok(signature) = Signature::from_slice(sig) else { continue; };
        if vk.verify(msg, &signature).is_ok() {
            ok_count += 1;
        }
    }
    ok_count
}

fn bench_sync_pipeline_vs_sequential(c: &mut Criterion) {
    // Pre-build a single verifier per bench run — matches sync_pipeline.rs:171
    // where the verifier is reconstructed per-batch. The Verifier's per-call
    // cost is negligible (just stores num_threads + chunk_size), so this is
    // representative even though we hoist the build out of the inner loop.
    let verifier = ParallelEd25519Verifier::new(num_cpus::get().max(1));

    let sizes = [128usize, 1024, 4096, 32768];

    let mut group = c.benchmark_group("sync_pipeline_tx_verify");
    group.sample_size(20);

    for &n in &sizes {
        let (messages, signatures, public_keys) = fixture(n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, &_n| {
            b.iter(|| {
                let ok = verify_sequential(black_box(&messages), black_box(&signatures), black_box(&public_keys));
                debug_assert_eq!(ok, n);
                ok
            });
        });

        group.bench_with_input(BenchmarkId::new("parallel_chunked", n), &n, |b, &_n| {
            b.iter(|| {
                let r = verifier
                    .verify_batch_chunked(black_box(&messages), black_box(&signatures), black_box(&public_keys))
                    .expect("batch verify");
                debug_assert_eq!(r.valid, n);
                r.valid
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sync_pipeline_vs_sequential);
criterion_main!(benches);
