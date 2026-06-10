// Kernel I/O Performance Benchmarks
// Comprehensive benchmarking of kernel-level optimizations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use crate::*;  // Use crate:: instead of q_kernel_io:: since this is within the same crate
use std::net::SocketAddr;
use tokio::runtime::Runtime;

fn criterion_benchmark(c: &mut Criterion) {
    bench_kernel_io_engine(c);
    bench_io_uring_operations(c);
    bench_numa_allocations(c);
    bench_zero_copy_networking(c);
    bench_memory_mapped_storage(c);
}

/// Benchmark kernel I/O engine creation and initialization
fn bench_kernel_io_engine(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("kernel_io_engine_init", |b| {
        b.iter(|| {
            rt.block_on(async {
                let config = KernelIoConfig::default();
                KernelIoEngine::new(config).await.unwrap()
            })
        });
    });
}

/// Benchmark io_uring operations
#[cfg(target_os = "linux")]
fn bench_io_uring_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let engine = rt.block_on(async { IoUringEngine::new(1024).await.unwrap() });

    let mut group = c.benchmark_group("io_uring_operations");

    // Benchmark different operation sizes
    for size in [1024, 4096, 16384, 65536].iter() {
        let temp_file = NamedTempFile::new().unwrap();
        let buffer = vec![0u8; *size];

        group.throughput(Throughput::Bytes(*size as u64));

        // Write operation benchmark
        group.bench_with_input(BenchmarkId::new("write", size), size, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    let operation = UringOperation::Write {
                        fd: temp_file.as_raw_fd(),
                        buffer: buffer.clone(),
                        offset: 0,
                    };
                    engine.submit_operation(operation).await.unwrap()
                })
            });
        });

        // Read operation benchmark
        group.bench_with_input(BenchmarkId::new("read", size), size, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    let operation = UringOperation::Read {
                        fd: temp_file.as_raw_fd(),
                        buffer: vec![0u8; *size],
                        offset: 0,
                    };
                    engine.submit_operation(operation).await.unwrap()
                })
            });
        });
    }

    group.finish();
}

#[cfg(not(target_os = "linux"))]
fn bench_io_uring_operations(_c: &mut Criterion) {
    // Skip io_uring benchmarks on non-Linux platforms
}

/// Benchmark batch io_uring operations
#[cfg(target_os = "linux")]
fn bench_io_uring_batch(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let engine = rt.block_on(async { IoUringEngine::new(2048).await.unwrap() });

    let mut group = c.benchmark_group("io_uring_batch");

    for batch_size in [1, 4, 8, 16, 32, 64].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_write", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_files: Vec<_> = (0..*batch_size)
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

                        engine.submit_batch(operations).await.unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark NUMA memory allocations
fn bench_numa_allocations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let numa_manager = rt.block_on(async { NumaManager::new(0).await.unwrap() });

    let mut group = c.benchmark_group("numa_allocations");

    for size in [4096, 16384, 65536, 262144].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("numa_local", size), size, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    if let Some(node_id) = numa_manager.get_optimal_node() {
                        numa_manager.allocate_on_node(*size, node_id).await.unwrap()
                    } else {
                        std::ptr::null_mut()
                    }
                })
            });
        });

        group.bench_with_input(BenchmarkId::new("regular_alloc", size), size, |b, _| {
            b.iter(|| {
                use std::alloc::{alloc, Layout};
                unsafe {
                    let layout = Layout::from_size_align(*size, 64).unwrap();
                    alloc(layout)
                }
            });
        });
    }

    group.finish();
}

/// Benchmark zero-copy buffer operations
fn bench_zero_copy_buffers(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let numa_manager = Arc::new(rt.block_on(async { NumaManager::new(0).await.unwrap() }));

    let memory_manager =
        rt.block_on(async { KernelMemoryManager::new(&numa_manager, 64).await.unwrap() });

    let mut group = c.benchmark_group("zero_copy_buffers");

    for size in [1024, 4096, 16384, 65536].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("allocate", size), size, |b, _| {
            b.iter(|| {
                rt.block_on(async {
                    memory_manager
                        .allocate_numa_buffer(*size, None)
                        .await
                        .unwrap()
                })
            });
        });

        group.bench_with_input(BenchmarkId::new("clone_ref", size), size, |b, _| {
            let buffer = rt.block_on(async {
                memory_manager
                    .allocate_numa_buffer(*size, None)
                    .await
                    .unwrap()
            });

            b.iter(|| buffer.clone_ref());
        });
    }

    group.finish();
}

/// Benchmark zero-copy networking operations
fn bench_zero_copy_networking(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let numa_manager = Arc::new(rt.block_on(async { NumaManager::new(0).await.unwrap() }));

    let memory_manager =
        Arc::new(rt.block_on(async { KernelMemoryManager::new(&numa_manager, 64).await.unwrap() }));

    let networking = rt.block_on(async { ZeroCopyNetworking::new(&memory_manager).await.unwrap() });

    let mut group = c.benchmark_group("zero_copy_networking");

    // Benchmark buffer pool operations
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

    group.finish();
}

/// Benchmark memory-mapped storage operations
fn bench_memory_mapped_storage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_mapped_storage");

    for size in [4096, 16384, 65536, 262144].iter() {
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(BenchmarkId::new("create_mmap", size), size, |b, _| {
            b.iter(|| {
                let temp_file = NamedTempFile::new().unwrap();
                let file_path = temp_file.path().to_str().unwrap();

                MemoryMappedStorage::new(file_path, *size, false).unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("mmap_read", size), size, |b, _| {
            let temp_file = NamedTempFile::new().unwrap();
            let file_path = temp_file.path().to_str().unwrap();
            let storage = MemoryMappedStorage::new(file_path, *size, false).unwrap();

            b.iter(|| {
                let slice = storage.as_slice();
                let _checksum: u64 = slice.iter().map(|&b| b as u64).sum();
            });
        });

        group.bench_with_input(BenchmarkId::new("mmap_write", size), size, |b, _| {
            let temp_file = NamedTempFile::new().unwrap();
            let file_path = temp_file.path().to_str().unwrap();
            let mut storage = MemoryMappedStorage::new(file_path, *size, false).unwrap();

            b.iter(|| {
                let slice = storage.as_mut_slice();
                for i in 0..*size {
                    slice[i] = (i % 256) as u8;
                }
            });
        });
    }

    group.finish();
}

/// Benchmark comprehensive kernel I/O integration
fn bench_kernel_io_integration(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let config = KernelIoConfig {
        enable_io_uring: cfg!(target_os = "linux"),
        enable_numa_aware: true,
        enable_zero_copy: true,
        enable_memory_mapped: true,
        ..Default::default()
    };

    let engine = rt.block_on(async { KernelIoEngine::new(config).await.unwrap() });

    let mut group = c.benchmark_group("kernel_io_integration");

    // Benchmark complete workflow: allocate -> process -> deallocate
    group.bench_function("numa_allocation_workflow", |b| {
        b.iter(|| {
            rt.block_on(async {
                let buffer = engine.allocate_numa_memory(4096, None).await.unwrap();

                // Simulate processing
                {
                    let slice = buffer.as_slice();
                    let _sum: u64 = slice.iter().map(|&b| b as u64).sum();
                }

                // Buffer automatically deallocated when dropped
            })
        });
    });

    // Benchmark memory-mapped file operations
    group.bench_function("mmap_workflow", |b| {
        b.iter(|| {
            rt.block_on(async {
                let temp_file = NamedTempFile::new().unwrap();
                let file_path = temp_file.path().to_str().unwrap();

                let mut storage = engine
                    .create_memory_mapped_storage(file_path, 16384)
                    .await
                    .unwrap();

                // Write data
                {
                    let slice = storage.as_mut_slice();
                    slice[0] = 0xAA;
                    slice[16383] = 0xBB;
                }

                // Read data
                {
                    let slice = storage.as_slice();
                    let _checksum = slice[0] + slice[16383];
                }

                storage.flush().unwrap();
            })
        });
    });

    group.finish();
}

/// Benchmark performance comparison: kernel optimizations vs standard operations
fn bench_optimization_comparison(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let config = KernelIoConfig::default();
    let engine = rt.block_on(async { KernelIoEngine::new(config).await.unwrap() });

    let mut group = c.benchmark_group("optimization_comparison");

    let data_size = 65536;

    // Optimized allocation
    group.bench_function("optimized_allocation", |b| {
        b.iter(|| {
            rt.block_on(async { engine.allocate_numa_memory(data_size, None).await.unwrap() })
        });
    });

    // Standard allocation
    group.bench_function("standard_allocation", |b| {
        b.iter(|| vec![0u8; data_size]);
    });

    // Memory-mapped vs regular file I/O
    group.bench_function("mmap_file_io", |b| {
        b.iter(|| {
            rt.block_on(async {
                let temp_file = NamedTempFile::new().unwrap();
                let file_path = temp_file.path().to_str().unwrap();

                let mut storage = engine
                    .create_memory_mapped_storage(file_path, data_size)
                    .await
                    .unwrap();

                // Write pattern
                {
                    let slice = storage.as_mut_slice();
                    for (i, byte) in slice.iter_mut().enumerate() {
                        *byte = (i % 256) as u8;
                    }
                }

                storage.flush().unwrap();
            })
        });
    });

    group.bench_function("regular_file_io", |b| {
        b.iter(|| {
            let temp_file = NamedTempFile::new().unwrap();

            // Write pattern
            let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();
            std::fs::write(&temp_file.path(), &data).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    kernel_io_benchmarks,
    criterion_benchmark,
    bench_io_uring_batch,
    bench_zero_copy_buffers,
    bench_kernel_io_integration,
    bench_optimization_comparison
);

criterion_main!(kernel_io_benchmarks);
