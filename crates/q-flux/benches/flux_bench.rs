use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use q_flux::access_log::AccessEntry;
use q_flux::metrics::{LatencyHistogram, TokenBucket, RateLimiter};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

// ─── AccessEntry JSON Serialization Benchmarks ────────────────────────────

/// Benchmark 1: AccessEntry::to_json() serialization throughput.
///
/// Measures the hand-rolled JSON serializer that runs on every request
/// in the access log hot path. Tests with a realistic entry including
/// all optional fields populated.
fn bench_access_entry_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_entry_json");

    // Full entry with all fields populated (typical production entry)
    group.bench_function("full_entry", |b| {
        let entry = AccessEntry {
            timestamp: "2026-03-07T12:00:00.123Z".to_string(),
            client_addr: SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(192, 168, 1, 42)),
                54321,
            ),
            method: "POST".to_string(),
            path: "/api/v1/mining/submit".to_string(),
            status: 200,
            request_bytes: 2048,
            response_bytes: 512,
            latency: Duration::from_micros(4200),
            tls_version: Some("TLS1.3".to_string()),
            user_agent: Some("q-miner/9.1.9 (Linux x86_64)".to_string()),
            upstream_backend: Some("127.0.0.1:8080".to_string()),
        };
        b.iter(|| entry.to_json());
    });

    // Minimal entry — no TLS, no user agent (e.g., plain HTTP health check)
    group.bench_function("minimal_entry", |b| {
        let entry = AccessEntry {
            timestamp: "2026-03-07T12:00:00Z".to_string(),
            client_addr: SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
                8080,
            ),
            method: "GET".to_string(),
            path: "/health".to_string(),
            status: 200,
            request_bytes: 0,
            response_bytes: 2,
            latency: Duration::from_micros(50),
            tls_version: None,
            user_agent: None,
            upstream_backend: None,
        };
        b.iter(|| entry.to_json());
    });

    // Entry with special characters in path (requires JSON escaping)
    group.bench_function("escaped_path", |b| {
        let entry = AccessEntry {
            timestamp: "2026-03-07T12:00:00Z".to_string(),
            client_addr: SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1)),
                12345,
            ),
            method: "GET".to_string(),
            path: "/api/v1/search?q=\"hello world\"&filter=\\backslash\\".to_string(),
            status: 400,
            request_bytes: 128,
            response_bytes: 64,
            latency: Duration::from_millis(15),
            tls_version: Some("TLS1.2".to_string()),
            user_agent: Some("Mozilla/5.0 \"quoted\" agent".to_string()),
            upstream_backend: None,
        };
        b.iter(|| entry.to_json());
    });

    // Throughput measurement: batch of 1000 serializations
    group.throughput(Throughput::Elements(1000));
    group.bench_function("batch_1000", |b| {
        let entry = AccessEntry {
            timestamp: "2026-03-07T12:00:00.456Z".to_string(),
            client_addr: SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4)),
                9999,
            ),
            method: "POST".to_string(),
            path: "/api/v1/blocks".to_string(),
            status: 201,
            request_bytes: 4096,
            response_bytes: 256,
            latency: Duration::from_millis(3),
            tls_version: Some("TLS1.3".to_string()),
            user_agent: Some("q-node/9.1.9".to_string()),
            upstream_backend: Some("10.0.0.1:8080".to_string()),
        };
        b.iter(|| {
            for _ in 0..1000 {
                let _ = entry.to_json();
            }
        });
    });

    group.finish();
}

// ─── TokenBucket Rate Limiter Benchmarks ──────────────────────────────────

/// Benchmark 2: TokenBucket::try_acquire() throughput.
///
/// Measures the lock-free token bucket used for per-IP rate limiting.
/// Tests both the "tokens available" (fast path) and "refill needed" paths.
fn bench_token_bucket(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_bucket");

    // Fast path: tokens available, no refill needed.
    // High burst capacity means many acquires succeed without refill CAS.
    group.bench_function("acquire_fast_path", |b| {
        let bucket = TokenBucket::new(100_000, 1_000_000); // 100K/s, 1M burst
        let now_us = 1_000_000_000u64; // Arbitrary fixed timestamp
        b.iter(|| bucket.try_acquire(now_us));
    });

    // Refill path: advance time between each acquire to trigger refill logic.
    group.bench_function("acquire_with_refill", |b| {
        let bucket = TokenBucket::new(10_000, 100); // 10K/s, burst 100
        let mut now_us = 1_000_000_000u64;
        b.iter(|| {
            now_us += 1_000; // Advance 1ms each iteration to trigger refill
            bucket.try_acquire(now_us)
        });
    });

    // Rate-limited path: bucket is drained, every acquire returns false.
    group.bench_function("acquire_rate_limited", |b| {
        let bucket = TokenBucket::new(1, 1); // 1/s, burst 1
        let now_us = 1_000_000_000u64;
        // Drain the bucket
        bucket.try_acquire(now_us);
        b.iter(|| bucket.try_acquire(now_us)); // Should return false
    });

    // Sustained throughput: many acquires with realistic time advancement
    group.throughput(Throughput::Elements(1000));
    group.bench_function("sustained_1000_acquires", |b| {
        let bucket = TokenBucket::new(100_000, 100_000); // 100K/s
        let mut now_us = 1_000_000_000u64;
        b.iter(|| {
            for _ in 0..1000 {
                now_us += 10; // 10us between requests = 100K/s
                let _ = bucket.try_acquire(now_us);
            }
        });
    });

    group.finish();
}

// ─── LatencyHistogram Benchmarks ──────────────────────────────────────────

/// Benchmark 3: LatencyHistogram::observe() throughput.
///
/// Measures the lock-free histogram that records request latencies.
/// All operations are atomic fetch_add, so contention is minimal
/// on single-threaded benchmarks.
fn bench_latency_histogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_histogram");

    // Observe with a duration that hits the first bucket (<=1ms)
    group.bench_function("observe_1ms", |b| {
        let hist = LatencyHistogram::new();
        let duration = Duration::from_micros(500);
        b.iter(|| hist.observe(duration));
    });

    // Observe with a duration that hits a middle bucket (25-50ms)
    group.bench_function("observe_30ms", |b| {
        let hist = LatencyHistogram::new();
        let duration = Duration::from_millis(30);
        b.iter(|| hist.observe(duration));
    });

    // Observe with a duration that hits the last bucket (>5s, overflow)
    group.bench_function("observe_10s_overflow", |b| {
        let hist = LatencyHistogram::new();
        let duration = Duration::from_secs(10);
        b.iter(|| hist.observe(duration));
    });

    // Varied durations simulating real traffic distribution
    group.throughput(Throughput::Elements(1000));
    group.bench_function("observe_mixed_1000", |b| {
        let hist = LatencyHistogram::new();
        // Simulate realistic latency distribution:
        // 50% < 5ms, 30% 5-50ms, 15% 50-500ms, 5% > 500ms
        let durations: Vec<Duration> = (0..1000)
            .map(|i| match i % 20 {
                0..=9 => Duration::from_micros(500 + (i as u64 * 100) % 4500),   // <5ms
                10..=15 => Duration::from_millis(5 + (i as u64 * 3) % 45),        // 5-50ms
                16..=18 => Duration::from_millis(50 + (i as u64 * 7) % 450),      // 50-500ms
                _ => Duration::from_millis(500 + (i as u64 * 11) % 4500),          // >500ms
            })
            .collect();

        b.iter(|| {
            for d in &durations {
                hist.observe(*d);
            }
        });
    });

    // Prometheus export after many observations (tests formatting cost)
    group.bench_function("prometheus_export", |b| {
        let hist = LatencyHistogram::new();
        // Pre-fill with observations
        for i in 0..10_000u64 {
            hist.observe(Duration::from_micros(i * 100));
        }
        b.iter(|| hist.prometheus("q_flux_request_duration_seconds"));
    });

    group.finish();
}

// ─── RateLimiter Per-IP Benchmarks ────────────────────────────────────────

/// Benchmark 4: RateLimiter::check() per-IP throughput.
///
/// Measures the DashMap-backed per-IP rate limiter. This is the primary
/// ingress filter on every incoming request — performance here directly
/// impacts maximum request throughput.
fn bench_rate_limiter(c: &mut Criterion) {
    let mut group = c.benchmark_group("rate_limiter");

    // Single IP, repeated checks (hot DashMap entry)
    group.bench_function("single_ip_hot", |b| {
        let limiter = RateLimiter::new(100_000, 100_000, 1_000_000);
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        // Warm up the entry
        limiter.check(ip);
        b.iter(|| limiter.check(ip));
    });

    // Many distinct IPs — measures DashMap insert + lookup performance
    group.bench_function("unique_ips_cold", |b| {
        let limiter = RateLimiter::new(1000, 1000, 10_000_000);
        let mut ip_counter = 1u32;
        b.iter(|| {
            let ip = IpAddr::V4(Ipv4Addr::from(ip_counter));
            ip_counter = ip_counter.wrapping_add(1);
            if ip_counter == 0 {
                ip_counter = 1;
            }
            limiter.check(ip)
        });
    });

    // Realistic traffic: 256 distinct IPs, round-robin
    group.throughput(Throughput::Elements(256));
    group.bench_function("256_ips_round_robin", |b| {
        let limiter = RateLimiter::new(10_000, 10_000, 10_000_000);
        let ips: Vec<IpAddr> = (1..=255u8)
            .chain(std::iter::once(0u8)) // 256 IPs: 10.0.0.1 - 10.0.1.0
            .map(|i| IpAddr::V4(Ipv4Addr::new(10, 0, i / 255, ((i as u16 % 255) + 1) as u8)))
            .take(256)
            .collect();

        // Warm up all entries
        for &ip in &ips {
            limiter.check(ip);
        }

        b.iter(|| {
            for &ip in &ips {
                let _ = limiter.check(ip);
            }
        });
    });

    // Measure cleanup cost (removing stale entries)
    group.bench_function("cleanup_1000_entries", |b| {
        b.iter_batched(
            || {
                let limiter = RateLimiter::new(1000, 1000, 10_000_000);
                // Populate with 1000 IPs
                for i in 1..=1000u32 {
                    let ip = IpAddr::V4(Ipv4Addr::from(i));
                    limiter.check(ip);
                }
                limiter
            },
            |limiter| {
                limiter.cleanup();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    flux_benches,
    bench_access_entry_json,
    bench_token_bucket,
    bench_latency_histogram,
    bench_rate_limiter,
);
criterion_main!(flux_benches);
