use criterion::{criterion_group, criterion_main, Criterion};
use q_benchmarks::{memory_profiler, BenchmarkConfig};
use tokio::runtime::Runtime;

fn bench_memory_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();

    c.bench_function("memory_performance", |b| {
        b.to_async(&rt).iter(|| async {
            memory_profiler::measure_memory_performance(&config)
                .await
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_memory_performance);
criterion_main!(benches);
