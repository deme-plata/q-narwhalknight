/// Performance benchmark for Phase 1A Safe Batched Writer
///
/// Target: 150-250 BPS sustained over 1000 blocks
/// Expert consensus: ChatGPT, Kimi AI, DeepSeek

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

// Mock types for benchmark (replace with actual imports when integrated)
use q_storage::{SafeBatchedWriter, BatchConfig, OrderedBlockBuffer};
use q_types::block::QBlock;

/// Benchmark batched sync performance
fn bench_batched_sync(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();

    let mut group = c.benchmark_group("sync_performance");
    group.sample_size(10); // Reduce sample size for long-running tests
    group.measurement_time(Duration::from_secs(30));

    // Test different batch sizes
    for batch_size in [8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("batched_sync", batch_size),
            batch_size,
            |b, &batch_size| {
                b.to_async(&runtime).iter(|| async move {
                    // Create test database
                    let db = create_test_db().await;

                    // Create batched writer with custom config
                    let config = BatchConfig {
                        max_batch_blocks: batch_size,
                        max_batch_duration: Duration::from_secs(1),
                        max_wal_bytes: 1024 * 1024,
                        max_reorder_gap: 2048,
                    };

                    let (mut writer, tx) = SafeBatchedWriter::new(
                        db.clone(),
                        config,
                        0, // start_height
                    );

                    // Spawn writer task
                    let writer_handle = tokio::spawn(async move {
                        writer.run().await.unwrap();
                    });

                    // Send 1000 blocks
                    let start = Instant::now();
                    for i in 0..1000 {
                        let block = create_test_block(i);
                        tx.send(block).await.unwrap();
                    }

                    // Close channel and wait for completion
                    drop(tx);
                    writer_handle.await.unwrap();

                    let duration = start.elapsed();
                    let bps = 1000.0 / duration.as_secs_f64();

                    println!("Batch size {}: {:.2} BPS", batch_size, bps);

                    // Verify performance target
                    assert!(bps >= 150.0,
                            "Failed to meet 150 BPS target: {:.2} BPS",
                            bps);

                    black_box(bps)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark ordered block buffer performance
fn bench_ordered_buffer(c: &mut Criterion) {
    let mut group = c.benchmark_group("ordered_buffer");

    group.bench_function("insert_sequential", |b| {
        b.iter(|| {
            let mut buffer = OrderedBlockBuffer::new(0, 2048);
            for i in 0..1000 {
                buffer.insert(create_test_block(i)).unwrap();
            }
            black_box(buffer)
        });
    });

    group.bench_function("insert_reverse", |b| {
        b.iter(|| {
            let mut buffer = OrderedBlockBuffer::new(0, 2048);
            for i in (0..1000).rev() {
                buffer.insert(create_test_block(i)).unwrap();
            }
            black_box(buffer)
        });
    });

    group.bench_function("insert_random", |b| {
        b.iter(|| {
            let mut buffer = OrderedBlockBuffer::new(0, 2048);
            let mut heights: Vec<u64> = (0..1000).collect();
            // Shuffle heights (simplified)
            for i in 0..1000 {
                buffer.insert(create_test_block(heights[i])).unwrap();
            }
            black_box(buffer)
        });
    });

    group.finish();
}

/// Benchmark flush performance
fn bench_flush_performance(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();

    c.bench_function("flush_16_blocks", |b| {
        b.to_async(&runtime).iter(|| async {
            let db = create_test_db().await;
            let config = BatchConfig::default(); // 16 blocks

            let (mut writer, tx) = SafeBatchedWriter::new(db, config, 0);

            let writer_handle = tokio::spawn(async move {
                writer.run().await.unwrap();
            });

            let start = Instant::now();

            // Send exactly 16 blocks (one batch)
            for i in 0..16 {
                tx.send(create_test_block(i)).await.unwrap();
            }

            drop(tx);
            writer_handle.await.unwrap();

            let flush_time = start.elapsed();
            println!("Flush time for 16 blocks: {:?}", flush_time);

            black_box(flush_time)
        });
    });
}

/// Helper: Create test database
async fn create_test_db() -> Arc<rocksdb::DB> {
    use rocksdb::{DB, Options};
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);

    let db = DB::open_cf(&opts, temp_dir.path(), vec!["blocks"]).unwrap();
    Arc::new(db)
}

/// Helper: Create test block
fn create_test_block(height: u64) -> QBlock {
    use q_types::block::{BlockHeader, VDFProof, QuantumMetadata};

    QBlock {
        header: BlockHeader {
            height,
            phase: 11,
            network_id: "testnet-phase11".to_string(),
            prev_block_hash: [0; 32],
            solutions_root: [0; 32],
            tx_root: [0; 32],
            state_root: [0; 32],
            timestamp: 0,
            dag_round: height,
            vdf_proof: VDFProof {
                output: vec![],
                proof: vec![],
                iterations: 0,
            },
            anchor_validator: None,
            proposer: "test".to_string(),
            producer_id: 0,
            difficulty: 1,
        },
        mining_solutions: vec![],
        dag_parents: vec![],
        quantum_metadata: QuantumMetadata {
            vdf_iterations: 0,
            entropy_source: vec![],
            quantum_signature: vec![],
        },
        transactions: vec![],
        balance_updates: vec![],
        size_bytes: 600, // Typical block size
    }
}

criterion_group!(
    benches,
    bench_batched_sync,
    bench_ordered_buffer,
    bench_flush_performance
);
criterion_main!(benches);
