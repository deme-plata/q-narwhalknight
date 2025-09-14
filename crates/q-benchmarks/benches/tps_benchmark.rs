use criterion::{criterion_group, criterion_main, Criterion};
use q_benchmarks::{measure_baseline_performance, BenchmarkConfig};
use tokio::runtime::Runtime;

fn bench_baseline_tps(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("baseline_tps_measurement", |b| {
        b.to_async(&rt)
            .iter(|| async { measure_baseline_performance().await.unwrap() })
    });
}

criterion_group!(benches, bench_baseline_tps);
criterion_main!(benches);
