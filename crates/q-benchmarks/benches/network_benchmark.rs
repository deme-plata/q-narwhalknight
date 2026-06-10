use criterion::{criterion_group, criterion_main, Criterion};
use q_benchmarks::{network_benchmark, BenchmarkConfig};
use tokio::runtime::Runtime;

fn bench_network_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();

    c.bench_function("network_performance", |b| {
        b.to_async(&rt).iter(|| async {
            network_benchmark::measure_network_performance(&config)
                .await
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_network_performance);
criterion_main!(benches);
