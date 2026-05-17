//! Performance benchmarks for Q-NarwhalKnight Virtual Machine
//!
//! This benchmark suite measures the performance of critical VM operations:
//! - State read/write operations
//! - Contract execution
//! - Memory management
//! - Consensus integration
//! - Concurrent access patterns

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use q_vm::{
    state::{InMemoryStateStorage, StateDB},
    vm::{CallData, StateAccess, VirtualMachine},
};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Benchmark basic state operations
fn bench_state_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state_db = Arc::new(StateDB::new_in_memory());

    let mut group = c.benchmark_group("state_operations");

    // Benchmark balance operations
    group.bench_function("set_balance", |b| {
        b.to_async(&rt).iter(|| async {
            for i in 0..100 {
                state_db
                    .set_balance(black_box(i), black_box(i * 1000))
                    .await
                    .unwrap();
            }
        });
    });

    // Prepare data for read benchmarks
    rt.block_on(async {
        for i in 0..1000 {
            state_db.set_balance(i, i * 1000).await.unwrap();
        }
    });

    group.bench_function("get_balance", |b| {
        b.to_async(&rt).iter(|| async {
            for i in 0..100 {
                let _balance = state_db.get_balance(black_box(i)).await.unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark storage operations
fn bench_storage_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state_db = Arc::new(StateDB::new_in_memory());

    let mut group = c.benchmark_group("storage_operations");

    // Test different data sizes
    let data_sizes = vec![32, 256, 1024, 4096];

    for size in data_sizes {
        group.bench_with_input(BenchmarkId::new("set_storage", size), &size, |b, &size| {
            let key = vec![0u8; 32];
            let value = vec![0u8; size];

            b.to_async(&rt).iter(|| async {
                state_db
                    .set_storage(
                        black_box(1),
                        black_box(key.clone()),
                        black_box(value.clone()),
                    )
                    .await
                    .unwrap();
            });
        });

        // Prepare data for read benchmarks
        rt.block_on(async {
            let key = vec![1u8; 32];
            let value = vec![0u8; size];
            state_db.set_storage(1, key, value).await.unwrap();
        });

        group.bench_with_input(BenchmarkId::new("get_storage", size), &size, |b, &_size| {
            let key = vec![1u8; 32];

            b.to_async(&rt).iter(|| async {
                let _value = state_db
                    .get_storage(black_box(1), black_box(&key))
                    .await
                    .unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark state root calculations
fn bench_state_root_calculation(c: &mut Criterion) {
    use q_vm::state::VmState;

    let mut group = c.benchmark_group("state_root");

    // Test with different state sizes
    let state_sizes = vec![10, 100, 1000];

    for size in state_sizes {
        group.bench_with_input(
            BenchmarkId::new("calculate_root", size),
            &size,
            |b, &size| {
                let mut state = VmState::default();

                // Populate state with test data
                for i in 0..size {
                    state.balances.insert(i, i * 1000);
                    state.nonces.insert(i, i);

                    let mut contract_storage = std::collections::HashMap::new();
                    contract_storage.insert(vec![i as u8; 32], vec![i as u8; 64]);
                    state.storage.insert(i, contract_storage);
                }

                b.iter(|| black_box(state.calculate_state_root()));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_state_operations,
    bench_storage_operations,
    bench_state_root_calculation
);
criterion_main!(benches);
