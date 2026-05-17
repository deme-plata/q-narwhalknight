//! Comprehensive benchmarks for q-queue lock-free and persistent queues.
//!
//! Run with: cargo bench --package q-queue
//!
//! Benchmark groups:
//!
//! 1. **spsc_roundtrip_latency** - Single push + pop round-trip on same thread
//!    (measures minimum queue overhead without cross-thread delay).
//!
//! 2. **spsc_throughput** - Producer thread pushes N items, consumer thread pops
//!    N items. Parameterized by N (10K, 100K). Reports items/sec.
//!
//! 3. **mpsc_throughput** - Multiple producers (1, 4, 8) push to a single
//!    consumer. Fixed total of 100K messages. Shows CAS contention scaling.
//!
//! 4. **persistent_append** - PersistentQueue::append throughput with 64-byte
//!    payloads. Parameterized by batch size (1K, 10K). Uses tempdir.
//!
//! 5. **persistent_read** - SegmentReader sequential read of 10K pre-written
//!    64-byte messages. Includes CRC32 verification overhead.
//!
//! 6. **spsc_vs_mpsc_overhead** - Identical 1-producer/1-consumer workload on
//!    both SpscQueue and MpscQueue. Isolates CAS overhead in MPSC.
//!
//! All in-memory benchmarks use 64-byte ([u8; 64]) payloads to focus on
//! queue coordination overhead rather than payload memcpy cost.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use q_queue::persistent::{PersistentQueue, SegmentReader};
use q_queue::{MpscQueue, SpscQueue};
use std::sync::Arc;
use std::thread;

// ---------------------------------------------------------------------------
// Shared constants and helpers
// ---------------------------------------------------------------------------

/// 64-byte payload. Fits in a cache line on most architectures, so benchmark
/// results reflect queue coordination cost, not data movement.
const MSG_SIZE: usize = 64;
type Msg = [u8; MSG_SIZE];

/// Construct a 64-byte message filled with `seed`.
#[inline]
fn make_msg(seed: u8) -> Msg {
    [seed; MSG_SIZE]
}

// ============================================================================
// 1. SPSC push/pop round-trip latency
// ============================================================================

/// Single-thread push-then-pop: measures the minimum round-trip overhead of
/// the SPSC queue (no cross-thread synchronization, no contention).
fn bench_spsc_roundtrip_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_roundtrip_latency");
    let q = SpscQueue::<Msg>::new(1024);

    group.bench_function("push_then_pop_64B", |b| {
        b.iter(|| {
            let msg = make_msg(0xAB);
            q.push(black_box(msg)).unwrap();
            let val = q.pop().unwrap();
            black_box(val);
        });
    });

    group.finish();
}

// ============================================================================
// 2. SPSC throughput (items/sec)
// ============================================================================

/// Cross-thread SPSC throughput: one producer thread pushes N items, one
/// consumer thread pops N items. Both spin-wait when the queue is full/empty.
/// Parameterized by N to show how throughput scales with batch size.
fn bench_spsc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_throughput");

    for &count in &[10_000u64, 100_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let q = Arc::new(SpscQueue::<Msg>::new(8192));
                    let q_prod = q.clone();
                    let q_cons = q.clone();

                    let producer = thread::spawn(move || {
                        let msg = make_msg(0xCD);
                        for _ in 0..count {
                            while q_prod.push(msg).is_err() {
                                std::hint::spin_loop();
                            }
                        }
                    });

                    let consumer = thread::spawn(move || {
                        let mut received = 0u64;
                        while received < count {
                            if let Some(val) = q_cons.pop() {
                                black_box(val);
                                received += 1;
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                    });

                    producer.join().unwrap();
                    consumer.join().unwrap();
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// 3. MPSC throughput with 1, 4, and 8 producers
// ============================================================================

/// Fan-in MPSC throughput: `num_producers` threads each push their share of
/// 100K total messages, while a single consumer thread drains the queue.
///
/// With 1 producer the CAS loop in MpscQueue::push never contends, so the
/// difference vs SPSC throughput at 1 producer shows the baseline CAS cost.
/// At 4 and 8 producers, CAS contention increases and throughput may drop.
fn bench_mpsc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("mpsc_throughput");
    let total_msgs: u64 = 100_000;

    for &num_producers in &[1u64, 4, 8] {
        let per_producer = total_msgs / num_producers;
        group.throughput(Throughput::Elements(total_msgs));
        group.bench_with_input(
            BenchmarkId::new("producers", num_producers),
            &num_producers,
            |b, &num_producers| {
                b.iter(|| {
                    let q = Arc::new(MpscQueue::<Msg>::new(8192));

                    let producers: Vec<_> = (0..num_producers)
                        .map(|id| {
                            let q = q.clone();
                            thread::spawn(move || {
                                let msg = make_msg(id as u8);
                                for _ in 0..per_producer {
                                    while q.push(msg).is_err() {
                                        std::hint::spin_loop();
                                    }
                                }
                            })
                        })
                        .collect();

                    let q_cons = q.clone();
                    let consumer = thread::spawn(move || {
                        let mut received = 0u64;
                        while received < total_msgs {
                            if let Some(val) = q_cons.pop() {
                                black_box(val);
                                received += 1;
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                    });

                    for p in producers {
                        p.join().unwrap();
                    }
                    consumer.join().unwrap();
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// 4. PersistentQueue append throughput
// ============================================================================

/// Append 64-byte messages to PersistentQueue backed by a tempdir.
/// Each criterion iteration gets a fresh queue (new tempdir) so segment
/// creation overhead is included. Measures the write path: header
/// serialization + CRC32 computation + file write.
fn bench_persistent_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistent_append");

    for &count in &[1_000u64, 10_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let dir = tempfile::tempdir().expect("failed to create tempdir");
                    // 1 MB segment avoids excessive rotation during the batch
                    let mut q =
                        PersistentQueue::open(dir.path(), 1_048_576).unwrap();
                    let msg = make_msg(0xEF);
                    for _ in 0..count {
                        q.append(black_box(&msg)).unwrap();
                    }
                    q.sync().unwrap();
                    // dir dropped here -> cleanup
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// 5. PersistentQueue read throughput (SegmentReader)
// ============================================================================

/// Pre-write 10K x 64-byte messages into a PersistentQueue, then benchmark
/// reading them all back through SegmentReader. The read path includes
/// CRC32 verification on every message.
fn bench_persistent_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistent_read");
    let count: u64 = 10_000;
    group.throughput(Throughput::Elements(count));

    // One-time setup: populate a segment directory.
    let dir = tempfile::tempdir().expect("failed to create tempdir");
    {
        // 4 MB segment is large enough to hold all 10K x 64-byte messages in
        // a single file (each message is 64 + 16 header = 80 bytes, total ~800 KB).
        let mut q = PersistentQueue::open(dir.path(), 4_194_304).unwrap();
        let msg = make_msg(0xDE);
        for _ in 0..count {
            q.append(&msg).unwrap();
        }
        q.sync().unwrap();
    }

    // Discover segment files to read.
    let q_for_files = PersistentQueue::open(dir.path(), 4_194_304).unwrap();
    let segment_files = q_for_files.segment_files().unwrap();

    group.bench_function(BenchmarkId::from_parameter(count), |b| {
        b.iter(|| {
            let mut total_read = 0u64;
            for path in &segment_files {
                let mut reader = SegmentReader::open(path).unwrap();
                while let Some((seq, data)) = reader.next_entry() {
                    black_box(seq);
                    black_box(&data);
                    total_read += 1;
                }
            }
            assert_eq!(total_read, count);
        });
    });

    group.finish();
    // dir dropped -> cleanup
}

// ============================================================================
// 6. Comparison: SPSC vs MPSC overhead (single producer)
// ============================================================================

/// Identical workload on SpscQueue and MpscQueue: 1 producer thread, 1
/// consumer thread, 100K x 64-byte messages. The only difference is the
/// push implementation:
///
/// - SPSC: plain atomic store (no CAS, no loop)
/// - MPSC: compare_exchange_weak loop (CAS)
///
/// The throughput delta between the two shows the cost of CAS coordination
/// even without actual contention from other producers.
fn bench_spsc_vs_mpsc_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_vs_mpsc_overhead");
    let count: u64 = 100_000;
    group.throughput(Throughput::Elements(count));

    // --- SPSC baseline ---
    group.bench_function("spsc_1p1c", |b| {
        b.iter(|| {
            let q = Arc::new(SpscQueue::<Msg>::new(8192));
            let q_prod = q.clone();
            let q_cons = q.clone();

            let producer = thread::spawn(move || {
                let msg = make_msg(0x11);
                for _ in 0..count {
                    while q_prod.push(msg).is_err() {
                        std::hint::spin_loop();
                    }
                }
            });

            let consumer = thread::spawn(move || {
                let mut received = 0u64;
                while received < count {
                    if let Some(val) = q_cons.pop() {
                        black_box(val);
                        received += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            });

            producer.join().unwrap();
            consumer.join().unwrap();
        });
    });

    // --- MPSC with single producer (CAS on producer path) ---
    group.bench_function("mpsc_1p1c", |b| {
        b.iter(|| {
            let q = Arc::new(MpscQueue::<Msg>::new(8192));
            let q_prod = q.clone();
            let q_cons = q.clone();

            let producer = thread::spawn(move || {
                let msg = make_msg(0x22);
                for _ in 0..count {
                    while q_prod.push(msg).is_err() {
                        std::hint::spin_loop();
                    }
                }
            });

            let consumer = thread::spawn(move || {
                let mut received = 0u64;
                while received < count {
                    if let Some(val) = q_cons.pop() {
                        black_box(val);
                        received += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
            });

            producer.join().unwrap();
            consumer.join().unwrap();
        });
    });

    group.finish();
}

// ============================================================================
// Criterion harness
// ============================================================================

criterion_group!(
    benches,
    bench_spsc_roundtrip_latency,
    bench_spsc_throughput,
    bench_mpsc_throughput,
    bench_persistent_append,
    bench_persistent_read,
    bench_spsc_vs_mpsc_overhead,
);

criterion_main!(benches);
