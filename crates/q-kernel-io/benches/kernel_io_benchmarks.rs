// Kernel I/O Benchmarks Entry Point
// Criterion-based performance benchmarking for kernel-level optimizations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use q_kernel_io::*;
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use tempfile::NamedTempFile;
use tokio::runtime::Runtime;

fn main_benchmarks(c: &mut Criterion) {
    bench_kernel_engine_performance(c);
    bench_numa_memory_performance(c);
    bench_zero_copy_operations(c);
    bench_io_throughput(c);
}

/// Benchmark kernel engine initialization and basic operations
fn bench_kernel_engine_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("kernel_engine");

    // Engine creation
    group.bench_function("engine_creation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = KernelIoConfig::default();
                KernelIoEngine::new(config).await.unwrap()
            })
        });
    });

    // Performance metrics collection
    let config = KernelIoConfig::default();
    let engine = rt.block_on(async { KernelIoEngine::new(config).await.unwrap() });

    group.bench_function("metrics_collection", |b| {
        b.iter(|| rt.block_on(async { engine.performance_metrics().await.unwrap() }));
    });

    group.finish();
}

/// Benchmark NUMA memory allocation performance
fn bench_numa_memory_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let numa_manager = Arc::new(rt.block_on(async { NumaManager::new(0).await.unwrap() }));

    let memory_manager =
        rt.block_on(async { KernelMemoryManager::new(&numa_manager, 4096).await.unwrap() });

    let mut group = c.benchmark_group("numa_memory");

    // Test different allocation sizes
    for size in [4096, 16384, 65536, 262144, 1048576].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        // NUMA-aware allocation
        group.bench_with_input(
            BenchmarkId::new("numa_allocation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        memory_manager
                            .allocate_numa_buffer(size, None)
                            .await
                            .unwrap()
                    })
                });
            },
        );

        // Standard allocation comparison
        group.bench_with_input(
            BenchmarkId::new("standard_allocation", size),
            size,
            |b, &size| {
                b.iter(|| ZeroCopyBuffer::new(size, 4096).unwrap());
            },
        );

        // Buffer pool reuse
        group.bench_with_input(
            BenchmarkId::new("buffer_pool_reuse", size),
            size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let buffer = memory_manager
                            .allocate_numa_buffer(size, None)
                            .await
                            .unwrap();
                        memory_manager.return_to_pool(buffer).await.unwrap();
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark zero-copy operations
fn bench_zero_copy_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let numa_manager = Arc::new(rt.block_on(async { NumaManager::new(0).await.unwrap() }));

    let memory_manager = Arc::new(
        rt.block_on(async { KernelMemoryManager::new(&numa_manager, 4096).await.unwrap() }),
    );

    let networking = rt.block_on(async { ZeroCopyNetworking::new(&memory_manager).await.unwrap() });

    let mut group = c.benchmark_group("zero_copy_operations");

    // Buffer acquisition from pool
    group.bench_function("get_send_buffer", |b| {
        b.iter(|| {
            rt.block_on(async {
                let buffer = networking.get_send_buffer().await.unwrap();
                networking.return_send_buffer(buffer).await.unwrap();
            })
        });
    });

    group.bench_function("get_receive_buffer", |b| {
        b.iter(|| {
            rt.block_on(async {
                let buffer = networking.get_receive_buffer().await.unwrap();
                networking.return_receive_buffer(buffer).await.unwrap();
            })
        });
    });

    // Memory-mapped storage operations
    for size in [16384, 65536, 262144].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("mmap_create_and_write", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let temp_file = NamedTempFile::new().unwrap();
                    let file_path = temp_file.path().to_str().unwrap();

                    let mut storage = MemoryMappedStorage::new(file_path, size, false).unwrap();

                    // Write test pattern
                    {
                        let slice = storage.as_mut_slice();
                        for (i, byte) in slice.iter_mut().enumerate() {
                            *byte = (i % 256) as u8;
                        }
                    }

                    storage.flush().unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("regular_file_write", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let temp_file = NamedTempFile::new().unwrap();
                    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
                    std::fs::write(&temp_file.path(), &data).unwrap();
                });
            },
        );
    }

    group.finish();
}

/// Benchmark I/O throughput with different techniques
#[cfg(target_os = "linux")]
fn bench_io_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let io_uring = rt.block_on(async { IoUringEngine::new(2048).await.unwrap() });

    let mut group = c.benchmark_group("io_throughput");

    // Single I/O operation throughput
    for size in [4096, 16384, 65536].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("io_uring_write", size),
            size,
            |b, &size| {
                let temp_file = NamedTempFile::new().unwrap();
                let buffer = vec![0u8; size];

                b.iter(|| {
                    rt.block_on(async {
                        let operation = UringOperation::Write {
                            fd: temp_file.as_raw_fd(),
                            buffer: buffer.clone(),
                            offset: 0,
                        };
                        io_uring.submit_operation(operation).await.unwrap()
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("standard_write", size),
            size,
            |b, &size| {
                let buffer = vec![0u8; size];

                b.iter(|| {
                    let temp_file = NamedTempFile::new().unwrap();
                    std::fs::write(&temp_file.path(), &buffer).unwrap();
                });
            },
        );
    }

    // Batch I/O operations
    for batch_size in [1, 4, 16, 64].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("io_uring_batch", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_files: Vec<_> = (0..batch_size)
                            .map(|_| NamedTempFile::new().unwrap())
                            .collect();

                        let operations: Vec<_> = temp_files
                            .iter()
                            .map(|f| UringOperation::Write {
                                fd: f.as_raw_fd(),
                                buffer: vec![0u8; 4096],
                                offset: 0,
                            })
                            .collect();

                        io_uring.submit_batch(operations).await.unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

#[cfg(not(target_os = "linux"))]
fn bench_io_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_throughput");

    // Simplified benchmark for non-Linux platforms
    for size in [4096, 16384, 65536].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("file_write", size), size, |b, &size| {
            let buffer = vec![0u8; size];

            b.iter(|| {
                let temp_file = NamedTempFile::new().unwrap();
                std::fs::write(&temp_file.path(), &buffer).unwrap();
            });
        });
    }

    group.finish();
}

/// Benchmark system optimization impact
fn bench_system_optimization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let config = KernelIoConfig {
        enable_io_uring: cfg!(target_os = "linux"),
        enable_numa_aware: true,
        enable_zero_copy: true,
        uring_queue_depth: 4096,
        ..Default::default()
    };

    let engine = rt.block_on(async {
        let mut engine = KernelIoEngine::new(config).await.unwrap();
        engine.optimize_system().await.unwrap();
        engine
    });

    let mut group = c.benchmark_group("system_optimization");

    // Test optimized vs unoptimized performance
    group.bench_function("optimized_workflow", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Allocate NUMA-local memory
                let buffer = engine.allocate_numa_memory(65536, None).await.unwrap();

                // Create memory-mapped storage
                let temp_file = NamedTempFile::new().unwrap();
                let file_path = temp_file.path().to_str().unwrap();
                let mut storage = engine
                    .create_memory_mapped_storage(file_path, 65536)
                    .await
                    .unwrap();

                // Copy data (zero-copy)
                {
                    let src = buffer.as_slice();
                    let dst = storage.as_mut_slice();
                    dst.copy_from_slice(&src[..65536]);
                }

                // Flush to disk
                storage.flush().unwrap();

                // Get performance metrics
                engine.performance_metrics().await.unwrap()
            })
        });
    });

    group.bench_function("standard_workflow", |b| {
        b.iter(|| {
            // Standard memory allocation
            let buffer = vec![0u8; 65536];

            // Standard file write
            let temp_file = NamedTempFile::new().unwrap();
            std::fs::write(&temp_file.path(), &buffer).unwrap();
        });
    });

    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let numa_manager = Arc::new(rt.block_on(async { NumaManager::new(0).await.unwrap() }));

    let memory_manager = Arc::new(
        rt.block_on(async { KernelMemoryManager::new(&numa_manager, 4096).await.unwrap() }),
    );

    let mut group = c.benchmark_group("concurrent_operations");

    // Concurrent memory allocations
    for thread_count in [1, 2, 4, 8].iter() {
        group.throughput(Throughput::Elements(*thread_count as u64));

        group.bench_with_input(
            BenchmarkId::new("concurrent_allocations", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    rt.block_on(async {
                        let futures: Vec<_> =
                            (0..thread_count)
                                .map(|_| {
                                    let manager = Arc::clone(&memory_manager);
                                    async move {
                                        manager.allocate_numa_buffer(16384, None).await.unwrap()
                                    }
                                })
                                .collect();

                        futures::future::join_all(futures).await
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    main_benchmarks,
    bench_system_optimization,
    bench_concurrent_operations
);
criterion_main!(benches);
