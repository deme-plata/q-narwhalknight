// Q-NarwhalKnight Sharding Performance Benchmarks
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use q_sharding::*;
use q_types::{create_test_state_key, create_test_state_value, create_test_transaction};
use tokio::runtime::Runtime;

fn bench_shard_routing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ShardConfig::default();

    let engine = rt.block_on(async { ShardingEngine::new(config).await.unwrap() });

    let test_tx = create_test_transaction();

    c.bench_with_input(
        BenchmarkId::new("shard_routing", "single_transaction"),
        &test_tx,
        |b, tx| {
            b.to_async(&rt)
                .iter(|| async { engine.route_transaction(tx).await.unwrap() });
        },
    );
}

fn bench_transaction_batch_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ShardConfig::default();

    let engine = rt.block_on(async { ShardingEngine::new(config).await.unwrap() });

    for batch_size in [10, 100, 1000].iter() {
        let transactions: Vec<_> = (0..*batch_size)
            .map(|_| create_test_transaction())
            .collect();

        c.bench_with_input(
            BenchmarkId::new("batch_processing", batch_size),
            &transactions,
            |b, txs| {
                b.to_async(&rt).iter(|| async {
                    engine.process_transaction_batch(txs.clone()).await.unwrap()
                });
            },
        );
    }
}

fn bench_state_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = ShardConfig::default();

    let state_shard =
        rt.block_on(async { state_shards::StateShard::new(0, config).await.unwrap() });

    let test_key = create_test_state_key();
    let test_value = create_test_state_value();

    c.bench_function("state_write", |b| {
        b.to_async(&rt).iter(|| async {
            state_shard
                .write_state(test_key.clone(), test_value.clone())
                .await
                .unwrap()
        });
    });

    c.bench_function("state_read", |b| {
        b.to_async(&rt)
            .iter(|| async { state_shard.read_state(&test_key).await.unwrap() });
    });
}

criterion_group!(
    benches,
    bench_shard_routing,
    bench_transaction_batch_processing,
    bench_state_operations
);
criterion_main!(benches);
