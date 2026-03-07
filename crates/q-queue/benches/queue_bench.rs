use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use q_queue::{SpscQueue, MpscQueue, PersistentQueue};
use q_queue::persistent::{Segment, SegmentReader};
use std::sync::Arc;
use std::thread;

// ─── Ring Buffer Benchmarks (existing) ────────────────────────────────────

fn bench_spsc_roundtrip(c: &mut Criterion) {
    c.bench_function("spsc_push_pop_roundtrip", |b| {
        let q = SpscQueue::new(1024);
        b.iter(|| {
            q.push(42u64).unwrap();
            q.pop().unwrap();
        });
    });
}

fn bench_spsc_burst(c: &mut Criterion) {
    c.bench_function("spsc_burst_1000", |b| {
        let q = SpscQueue::new(2048);
        b.iter(|| {
            for i in 0..1000u64 {
                q.push(i).unwrap();
            }
            for _ in 0..1000 {
                q.pop().unwrap();
            }
        });
    });
}

fn bench_spsc_threaded_throughput(c: &mut Criterion) {
    c.bench_function("spsc_threaded_100k", |b| {
        b.iter_custom(|iters| {
            let total = 100_000u64 * iters as u64;
            let q = Arc::new(SpscQueue::new(8192));

            let q_prod = q.clone();
            let start = std::time::Instant::now();

            let producer = thread::spawn(move || {
                for i in 0..total {
                    while q_prod.push(i).is_err() {
                        std::hint::spin_loop();
                    }
                }
            });

            let mut received = 0u64;
            while received < total {
                if q.pop().is_some() {
                    received += 1;
                } else {
                    std::hint::spin_loop();
                }
            }

            producer.join().unwrap();
            start.elapsed()
        });
    });
}

fn bench_mpsc_4_producers(c: &mut Criterion) {
    c.bench_function("mpsc_4_producers_100k", |b| {
        b.iter_custom(|iters| {
            let per_producer = 25_000u64 * iters as u64;
            let q = Arc::new(MpscQueue::new(8192));
            let start = std::time::Instant::now();

            let producers: Vec<_> = (0..4)
                .map(|_| {
                    let q = q.clone();
                    thread::spawn(move || {
                        for i in 0..per_producer {
                            while q.push(i).is_err() {
                                std::hint::spin_loop();
                            }
                        }
                    })
                })
                .collect();

            let total = per_producer * 4;
            let mut received = 0u64;
            while received < total {
                if q.pop().is_some() {
                    received += 1;
                } else {
                    std::hint::spin_loop();
                }
            }

            for p in producers {
                p.join().unwrap();
            }
            start.elapsed()
        });
    });
}

// ─── Persistent Queue Benchmarks ──────────────────────────────────────────

/// Helper: create a fresh temporary directory for each benchmark iteration.
fn bench_temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "q-queue-bench-{}-{}-{}",
        std::process::id(),
        name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to create bench temp dir");
    dir
}

/// Benchmark 1: Persistent queue append throughput (single-threaded, no fsync).
///
/// Measures raw append speed — writes to OS page cache without explicit sync.
/// This is the hot-path performance users see when appending messages.
fn bench_persistent_append_throughput(c: &mut Criterion) {
    let payload = vec![0xABu8; 128]; // 128-byte messages

    let mut group = c.benchmark_group("persistent_append");
    group.throughput(Throughput::Elements(1));

    group.bench_function("128B_nosync", |b| {
        let dir = bench_temp_dir("append-nosync");
        let mut q = PersistentQueue::open(&dir, 16 * 1024 * 1024).unwrap();
        b.iter(|| {
            q.append(&payload).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dir);
    });

    // Also benchmark with varying message sizes
    for size in [64, 256, 1024, 4096] {
        let data = vec![0xCDu8; size];
        group.bench_with_input(
            BenchmarkId::new("nosync", format!("{}B", size)),
            &data,
            |b, data| {
                let dir = bench_temp_dir(&format!("append-nosync-{}", size));
                let mut q = PersistentQueue::open(&dir, 16 * 1024 * 1024).unwrap();
                b.iter(|| {
                    q.append(data).unwrap();
                });
                let _ = std::fs::remove_dir_all(&dir);
            },
        );
    }

    group.finish();
}

/// Benchmark 2: Persistent queue append + sync throughput.
///
/// Measures append followed by fsync — the durable-write path.
/// This is the performance when crash-safety is required on every message.
fn bench_persistent_append_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistent_append_sync");

    // Sync after every append (worst case — maximum durability)
    group.bench_function("128B_sync_every", |b| {
        let dir = bench_temp_dir("append-sync-every");
        let payload = vec![0xABu8; 128];
        let mut q = PersistentQueue::open(&dir, 16 * 1024 * 1024).unwrap();
        b.iter(|| {
            q.append(&payload).unwrap();
            q.sync().unwrap();
        });
        let _ = std::fs::remove_dir_all(&dir);
    });

    // Sync after a batch of 100 appends (amortized durability)
    group.bench_function("128B_sync_batch_100", |b| {
        let dir = bench_temp_dir("append-sync-batch");
        let payload = vec![0xABu8; 128];
        let mut q = PersistentQueue::open(&dir, 16 * 1024 * 1024).unwrap();
        b.iter(|| {
            for _ in 0..100 {
                q.append(&payload).unwrap();
            }
            q.sync().unwrap();
        });
        let _ = std::fs::remove_dir_all(&dir);
    });

    group.finish();
}

/// Benchmark 3: SegmentReader sequential read throughput.
///
/// Writes N messages to a segment, then benchmarks sequential read-back.
/// Measures the zero-copy read path including CRC32 verification.
fn bench_segment_reader_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_reader");

    for &msg_count in &[100, 1000, 10_000] {
        // Prepare a segment file with msg_count messages
        let dir = bench_temp_dir(&format!("reader-{}", msg_count));
        let payload = vec![0xEFu8; 128];
        {
            let mut seg = Segment::create(&dir, 0, (msg_count + 10) * 160).unwrap();
            for i in 0..msg_count as u64 {
                seg.append(&payload, i).unwrap();
            }
            seg.sync().unwrap();
        }
        let seg_path = dir.join("segment-0000000000000000.qlog");

        group.throughput(Throughput::Elements(msg_count as u64));
        group.bench_with_input(
            BenchmarkId::new("sequential_read", format!("{}_msgs", msg_count)),
            &seg_path,
            |b, path| {
                b.iter(|| {
                    let mut reader = SegmentReader::open(path).unwrap();
                    let mut count = 0u64;
                    while let Some((_seq, _data)) = reader.next() {
                        count += 1;
                    }
                    assert_eq!(count, msg_count as u64);
                    count
                });
            },
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    group.finish();
}

/// Benchmark 4: CRC32 checksum throughput on various payload sizes.
///
/// The CRC32 function is private to the persistent module, so we benchmark
/// it indirectly via Segment::append (which computes CRC32 on every write)
/// and SegmentReader::next (which verifies CRC32 on every read).
/// We isolate the CRC cost by comparing append of different sizes.
fn bench_crc32_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("crc32_via_append");

    for &size in &[64, 256, 1024, 4096, 16384, 65536] {
        let data = vec![0xA5u8; size];

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("append", format!("{}B", size)),
            &data,
            |b, data| {
                let dir = bench_temp_dir(&format!("crc32-{}", size));
                // Use a very large segment to avoid rotation overhead
                let seg_capacity = (size + 32) * 10_000;
                let mut seg = Segment::create(&dir, 0, seg_capacity).unwrap();
                let mut seq = 0u64;
                b.iter(|| {
                    // If segment full, create a new one
                    if seg.is_full() {
                        seq += 1;
                        seg = Segment::create(&dir, seq * 1_000_000, seg_capacity).unwrap();
                    }
                    seg.append(data, seq).unwrap();
                    seq += 1;
                });
                let _ = std::fs::remove_dir_all(&dir);
            },
        );
    }

    group.finish();

    // Also benchmark CRC verification (read path)
    let mut verify_group = c.benchmark_group("crc32_verify_read");

    for &size in &[64, 256, 1024, 4096, 16384] {
        let data = vec![0xB7u8; size];
        let msg_count = 1000;

        // Prepare segment
        let dir = bench_temp_dir(&format!("crc32-verify-{}", size));
        {
            let seg_capacity = (size + 32) * (msg_count + 10);
            let mut seg = Segment::create(&dir, 0, seg_capacity).unwrap();
            for i in 0..msg_count as u64 {
                seg.append(&data, i).unwrap();
            }
            seg.sync().unwrap();
        }
        let seg_path = dir.join("segment-0000000000000000.qlog");

        verify_group.throughput(Throughput::Bytes((size * msg_count) as u64));
        verify_group.bench_with_input(
            BenchmarkId::new("read_verify", format!("{}B_x{}", size, msg_count)),
            &seg_path,
            |b, path| {
                b.iter(|| {
                    let mut reader = SegmentReader::open(path).unwrap();
                    while reader.next().is_some() {}
                });
            },
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    verify_group.finish();
}

/// Benchmark 5: Compare SPSC vs MPSC vs Persistent queue throughput.
///
/// All three queue types benchmarked with the same workload:
/// 10,000 messages of 64 bytes each. Gives a direct comparison of
/// in-memory lock-free vs durable persistent queue performance.
fn bench_queue_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue_comparison_10k");
    let msg_count = 10_000u64;

    // SPSC: push 10k u64 values and pop them all
    group.throughput(Throughput::Elements(msg_count));
    group.bench_function("spsc_10k", |b| {
        let q = SpscQueue::new(16384);
        b.iter(|| {
            for i in 0..msg_count {
                q.push(i).unwrap();
            }
            for _ in 0..msg_count {
                q.pop().unwrap();
            }
        });
    });

    // MPSC: push 10k u64 values (single producer) and pop them all
    group.bench_function("mpsc_10k_single_producer", |b| {
        let q = MpscQueue::new(16384);
        b.iter(|| {
            for i in 0..msg_count {
                q.push(i).unwrap();
            }
            for _ in 0..msg_count {
                q.pop().unwrap();
            }
        });
    });

    // Persistent: append 10k x 64B messages (no sync, amortized segment creation)
    group.bench_function("persistent_10k_nosync", |b| {
        let dir = bench_temp_dir("comparison-persistent");
        let payload = vec![0xFFu8; 64];
        let mut q = PersistentQueue::open(&dir, 16 * 1024 * 1024).unwrap();
        b.iter(|| {
            for _ in 0..msg_count {
                q.append(&payload).unwrap();
            }
        });
        let _ = std::fs::remove_dir_all(&dir);
    });

    // Persistent: append 10k x 64B messages with sync at the end
    group.bench_function("persistent_10k_sync_end", |b| {
        let dir = bench_temp_dir("comparison-persistent-sync");
        let payload = vec![0xFFu8; 64];
        let mut q = PersistentQueue::open(&dir, 16 * 1024 * 1024).unwrap();
        b.iter(|| {
            for _ in 0..msg_count {
                q.append(&payload).unwrap();
            }
            q.sync().unwrap();
        });
        let _ = std::fs::remove_dir_all(&dir);
    });

    group.finish();
}

/// Benchmark: Segment rotation overhead.
///
/// Uses small segments (4KB) to force frequent rotation, measuring
/// the cost of creating new segment files during sustained writes.
fn bench_segment_rotation(c: &mut Criterion) {
    c.bench_function("segment_rotation_4kb", |b| {
        let dir = bench_temp_dir("rotation");
        let payload = vec![0xDDu8; 128];
        let mut q = PersistentQueue::open(&dir, 4096).unwrap(); // Small segments = frequent rotation
        b.iter(|| {
            q.append(&payload).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dir);
    });
}

criterion_group!(
    ring_benches,
    bench_spsc_roundtrip,
    bench_spsc_burst,
    bench_spsc_threaded_throughput,
    bench_mpsc_4_producers,
);

criterion_group!(
    persistent_benches,
    bench_persistent_append_throughput,
    bench_persistent_append_sync,
    bench_segment_reader_throughput,
    bench_crc32_throughput,
    bench_queue_comparison,
    bench_segment_rotation,
);

criterion_main!(ring_benches, persistent_benches);
