use criterion::{black_box, criterion_group, criterion_main, Criterion};
use q_precision::{
    gas_optimization::{BatchProcessor, Operation, SolanaComparison},
    QAmount,
};
use std::str::FromStr;

fn benchmark_arithmetic_operations(c: &mut Criterion) {
    let a = QAmount::from_str("123.456789012345678901234567890123456").unwrap();
    let b = QAmount::from_str("987.654321098765432109876543210987654").unwrap();

    c.bench_function("addition", |bencher| {
        bencher.iter(|| black_box(black_box(a) + black_box(b)))
    });

    c.bench_function("multiplication", |bencher| {
        bencher.iter(|| black_box(black_box(a) * black_box(b)))
    });

    c.bench_function("division", |bencher| {
        bencher.iter(|| black_box(black_box(a) / black_box(b)))
    });

    c.bench_function("string_parsing", |bencher| {
        bencher.iter(|| {
            black_box(
                QAmount::from_str(black_box("0.123456789012345678901234567890123456")).unwrap(),
            )
        })
    });
}

fn benchmark_gas_operations(c: &mut Criterion) {
    c.bench_function("gas_calculation", |bencher| {
        bencher.iter(|| black_box(QAmount::calculate_gas_optimized_fee(black_box(1))))
    });

    let mut batch = BatchProcessor::new();
    let a = QAmount::from_str("1.0").unwrap();
    let b = QAmount::from_str("0.5").unwrap();

    c.bench_function("batch_processing", |bencher| {
        bencher.iter(|| {
            let mut batch = BatchProcessor::new();
            for _ in 0..100 {
                batch.add_operation(Operation::Add(black_box(a), black_box(b)));
                batch.add_operation(Operation::Mul(black_box(a), black_box(b)));
            }
            black_box(batch.execute_batch().unwrap())
        })
    });
}

criterion_group!(
    benches,
    benchmark_arithmetic_operations,
    benchmark_gas_operations
);
criterion_main!(benches);
