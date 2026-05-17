use criterion::{criterion_group, criterion_main, Criterion};
use q_benchmarks::{tps_benchmark, BenchmarkConfig};
use tokio::runtime::Runtime;

fn bench_sharding_scalability(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig {
        target_tps: 25_000.0, // Phase 1 target
        duration_seconds: 30,
        ..BenchmarkConfig::default()
    };

    c.bench_function("sharding_tps_scaling", |b| {
        b.to_async(&rt)
            .iter(|| async { tps_benchmark::benchmark_tps_scaling(&config).await.unwrap() })
    });
}

criterion_group!(benches, bench_sharding_scalability);
criterion_main!(benches);
