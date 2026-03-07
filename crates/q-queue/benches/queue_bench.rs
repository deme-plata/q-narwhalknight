use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use q_queue::{SpscQueue, MpscQueue};
use std::sync::Arc;
use std::thread;

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

criterion_group!(
    benches,
    bench_spsc_roundtrip,
    bench_spsc_burst,
    bench_spsc_threaded_throughput,
    bench_mpsc_4_producers,
);
criterion_main!(benches);
