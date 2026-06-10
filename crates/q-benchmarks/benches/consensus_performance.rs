use criterion::{criterion_group, criterion_main, Criterion};
use q_benchmarks::{consensus_benchmark, BenchmarkConfig};
use tokio::runtime::Runtime;

fn bench_consensus_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();

    c.bench_function("consensus_performance", |b| {
        b.to_async(&rt).iter(|| async {
            consensus_benchmark::measure_consensus_performance(&config)
                .await
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_consensus_performance);
criterion_main!(benches);
