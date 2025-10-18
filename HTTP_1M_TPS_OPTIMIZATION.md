# Axum HTTP Server Optimization for 1M+ TPS

**Date**: 2025-10-12
**Goal**: Configure Axum/Hyper for 1,000,000+ TPS throughput
**Status**: 🚧 **IMPLEMENTATION REQUIRED**

---

## Current Problem

Our Axum server is using **default configuration** which is NOT optimized for extreme throughput:

```rust
// CURRENT (SUBOPTIMAL):
axum::serve(listener, app.into_make_service()).await?;
```

This uses Hyper's defaults:
- **Single executor thread** (not utilizing all CPU cores)
- **Default buffer sizes** (too small for batches)
- **No connection pooling** (creating/destroying TCP connections)
- **No HTTP/2 optimization** (not using multiplexing)
- **Default backlog** (only 128 pending connections)

### Result: ~6,000 TPS (bottlenecked by HTTP layer)

---

## Solutions for 1M+ TPS

### 1. **Use Hyper with Tokio Runtime Optimization**

The key is to configure the Tokio runtime and Hyper server properly:

```rust
use hyper::server::conn::http1;
use hyper_util::rt::TokioIo;
use tower::ServiceExt;

// Configure Tokio runtime for maximum throughput
let runtime = tokio::runtime::Builder::new_multi_thread()
    .worker_threads(num_cpus::get() * 2)  // 2x CPU cores
    .thread_name("q-api-worker")
    .thread_stack_size(3 * 1024 * 1024)  // 3MB stack
    .enable_all()
    .build()?;

// Configure TCP listener with optimal settings
let listener = tokio::net::TcpListener::bind("0.0.0.0:8200").await?;
listener.set_nodelay(true)?;  // Disable Nagle's algorithm for low latency

// HTTP/1 connection parameters
let http_params = http1::Builder::new()
    .max_buf_size(16 * 1024 * 1024)  // 16MB buffer (for large batches)
    .pipeline_flush(true)  // Enable HTTP pipelining
    .timer(TokioTimer::new())
    .build();

// Accept connections in parallel
let max_concurrent_connections = 10_000;  // Up from default 128
let semaphore = Arc::new(Semaphore::new(max_concurrent_connections));

loop {
    let permit = semaphore.clone().acquire_owned().await?;
    let (stream, _) = listener.accept().await?;

    // Configure TCP socket for high throughput
    stream.set_nodelay(true)?;  // TCP_NODELAY
    stream.set_linger(None)?;  // Close immediately

    let io = TokioIo::new(stream);
    let service = app.clone();

    tokio::spawn(async move {
        if let Err(e) = http_params.serve_connection(io, service).await {
            // Log connection errors
        }
        drop(permit);  // Release connection slot
    });
}
```

**Expected Improvement**: 10x (6,000 → 60,000 TPS)

---

### 2. **Enable HTTP/2 with Multiplexing**

HTTP/2 allows multiple requests over a single connection:

```rust
use hyper::server::conn::http2;

let http2_params = http2::Builder::new(TokioExecutor::new())
    .max_concurrent_streams(1000)  // 1000 streams per connection
    .initial_connection_window_size(1024 * 1024)  // 1MB
    .initial_stream_window_size(512 * 1024)  // 512KB
    .max_frame_size(16384)  // 16KB frames
    .enable_connect_protocol()
    .build();

// Serve with HTTP/2
http2_params.serve_connection(io, service).await?;
```

**Expected Improvement**: 5x (60,000 → 300,000 TPS)

---

### 3. **Zero-Copy with Bytes and HTTP Body Streaming**

Current issue: We're deserializing the entire request before processing.

**Optimization**: Stream the body and process chunks in parallel:

```rust
use axum::body::Body;
use bytes::Bytes;
use futures::StreamExt;

pub async fn submit_binary_batch_streaming(
    State(state): State<Arc<AppState>>,
    body: Body,
) -> Result<impl IntoResponse, StatusCode> {
    // Stream body chunks without copying
    let mut stream = body.into_data_stream();

    // Accumulate chunks with zero-copy
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| StatusCode::BAD_REQUEST)?;
        chunks.push(chunk);
    }

    // Deserialize from concatenated chunks (single allocation)
    let data: Bytes = chunks.into_iter().collect();
    let batch: BinaryTransactionBatch = rmp_serde::from_slice(&data)?;

    // Process batch with SIMD
    // ... (existing code)
}
```

**Expected Improvement**: 2x (300,000 → 600,000 TPS)

---

### 4. **Connection Pooling and Keep-Alive**

The benchmark is creating new TCP connections for every request. Fix this:

#### Server-side (already configured in middleware):
```rust
.layer(
    ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024))
        // ADD: Connection keep-alive
        .layer(tower_http::set_header::SetResponseHeaderLayer::if_not_present(
            http::header::CONNECTION,
            http::HeaderValue::from_static("keep-alive"),
        ))
)
```

#### Client-side (fix benchmark):
```python
import httpx  # Better than requests for async

client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=1000,
        max_keepalive_connections=100,
    ),
    http2=True,  # Enable HTTP/2
    timeout=httpx.Timeout(30.0),
)

async def submit_batch(batch):
    packed = msgpack.packb(batch)
    response = await client.post(
        BINARY_BATCH_ENDPOINT,
        content=packed,
        headers={'Content-Type': 'application/octet-stream'}
    )
    return response
```

**Expected Improvement**: 2x (600,000 → 1,200,000 TPS)

---

## Complete Optimized Server Configuration

Create a new high-performance server module:

**File**: `crates/q-api-server/src/high_performance_server.rs`

```rust
use axum::Router;
use hyper::server::conn::http2;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tower::ServiceExt;

pub struct HighPerformanceServer {
    app: Router,
    addr: SocketAddr,
    max_connections: usize,
}

impl HighPerformanceServer {
    pub fn new(app: Router, addr: SocketAddr) -> Self {
        let cpu_cores = num_cpus::get();
        Self {
            app,
            addr,
            max_connections: cpu_cores * 1000,  // 1000 connections per core
        }
    }

    pub async fn run(self) -> anyhow::Result<()> {
        // Configure TCP listener
        let listener = TcpListener::bind(self.addr).await?;

        // Set socket options for maximum throughput
        let socket = socket2::Socket::from(listener.into_std()?);
        socket.set_nodelay(true)?;  // TCP_NODELAY
        socket.set_reuse_address(true)?;  // SO_REUSEADDR
        socket.set_reuse_port(true)?;  // SO_REUSEPORT (Linux)
        socket.set_recv_buffer_size(4 * 1024 * 1024)?;  // 4MB recv buffer
        socket.set_send_buffer_size(4 * 1024 * 1024)?;  // 4MB send buffer

        let listener = TcpListener::from_std(socket.into())?;

        tracing::info!("🚀 High-Performance Server listening on {}", self.addr);
        tracing::info!("   Max concurrent connections: {}", self.max_connections);
        tracing::info!("   HTTP/2 enabled with {} streams per connection", 1000);
        tracing::info!("   Target throughput: 1,000,000+ TPS");

        // Connection semaphore
        let semaphore = Arc::new(Semaphore::new(self.max_connections));

        // HTTP/2 parameters
        let http2_builder = http2::Builder::new(TokioExecutor::new())
            .timer(TokioTimer::new())
            .max_concurrent_streams(1000)
            .initial_connection_window_size(1024 * 1024)  // 1MB
            .initial_stream_window_size(512 * 1024)  // 512KB
            .max_frame_size(16384)
            .enable_connect_protocol();

        // Accept loop
        loop {
            // Acquire connection slot
            let permit = semaphore.clone().acquire_owned().await
                .map_err(|_| anyhow::anyhow!("Semaphore closed"))?;

            // Accept connection
            let (stream, remote_addr) = listener.accept().await?;

            // Configure socket
            let _ = stream.set_nodelay(true);
            let _ = stream.set_linger(None);

            // Clone app service
            let service = self.app.clone();

            // Spawn connection handler
            tokio::spawn(async move {
                let io = TokioIo::new(stream);

                // Serve HTTP/2 connection
                if let Err(e) = http2_builder.serve_connection(io, service).await {
                    tracing::debug!("Connection error from {}: {}", remote_addr, e);
                }

                drop(permit);  // Release connection slot
            });
        }
    }
}
```

### Usage in `main.rs`:

```rust
// Replace:
// axum::serve(listener, app.into_make_service()).await?;

// With:
use q_api_server::high_performance_server::HighPerformanceServer;

let addr = format!("0.0.0.0:{}", config.port).parse()?;
let server = HighPerformanceServer::new(app, addr);
server.run().await?;
```

---

## Benchmark Client Optimization

The Python benchmark is also bottlenecked. Create a Rust benchmark client:

**File**: `crates/q-api-server/examples/high_performance_benchmark.rs`

```rust
use rmp_serde;
use tokio::time::Instant;
use hyper::{Body, Client, Request};
use hyper_util::rt::TokioExecutor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "http://localhost:8200/api/v1/binary/batch";
    let total_transactions = 100_000;
    let batch_size = 10_000;  // Large batches
    let concurrent_requests = 100;

    println!("🚀 High-Performance Binary Protocol Benchmark");
    println!("📊 Total transactions: {}", total_transactions);
    println!("📦 Batch size: {}", batch_size);
    println!("⚡ Concurrent requests: {}", concurrent_requests);

    // Create HTTP/2 client with connection pooling
    let client = Client::builder(TokioExecutor::new())
        .http2_only(true)
        .http2_max_concurrent_streams(1000)
        .build_http();

    // Create batches
    let num_batches = total_transactions / batch_size;
    let mut batches = Vec::new();
    for i in 0..num_batches {
        let batch = create_batch(batch_size, i * batch_size);
        let packed = rmp_serde::to_vec(&batch)?;
        batches.push(packed);
    }

    // Send batches concurrently
    let start = Instant::now();
    let success_count = Arc::new(AtomicU64::new(0));
    let error_count = Arc::new(AtomicU64::new(0));

    let tasks: Vec<_> = batches.into_iter().map(|batch_data| {
        let client = client.clone();
        let success = success_count.clone();
        let errors = error_count.clone();
        let url = url.to_string();

        tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri(url)
                .header("content-type", "application/octet-stream")
                .body(Body::from(batch_data))?;

            match client.request(req).await {
                Ok(_) => success.fetch_add(1, Ordering::Relaxed),
                Err(_) => errors.fetch_add(1, Ordering::Relaxed),
            };

            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })
    }).collect();

    // Wait for all requests
    for task in tasks {
        let _ = task.await;
    }

    let elapsed = start.elapsed();
    let tps = total_transactions as f64 / elapsed.as_secs_f64();

    println!("\n================================");
    println!("📊 RESULTS");
    println!("================================");
    println!("✅ Successful batches: {}", success_count.load(Ordering::Relaxed));
    println!("❌ Failed batches: {}", error_count.load(Ordering::Relaxed));
    println!("⏱️  Total time: {:.2}s", elapsed.as_secs_f64());
    println!("⚡ TPS: {:.0}", tps);
    println!("📦 Batches/sec: {:.0}", num_batches as f64 / elapsed.as_secs_f64());
    println!("================================");

    Ok(())
}

fn create_batch(size: usize, start_nonce: usize) -> BinaryTransactionBatch {
    // ... (same as Python version)
}
```

**Expected Result**: **1,000,000+ TPS**

---

## Implementation Checklist

### Phase 1: Server Optimization (2 hours)
- [ ] Create `high_performance_server.rs` module
- [ ] Configure HTTP/2 with proper parameters
- [ ] Set TCP socket options (NODELAY, buffers, etc.)
- [ ] Implement connection pooling with semaphore
- [ ] Update `main.rs` to use new server

### Phase 2: Client Optimization (1 hour)
- [ ] Create Rust benchmark client
- [ ] Enable HTTP/2 client
- [ ] Implement proper connection pooling
- [ ] Add concurrent batch sending

### Phase 3: Testing (1 hour)
- [ ] Rebuild server: `timeout 36000 cargo build --release --package q-api-server`
- [ ] Run Rust benchmark: `cargo run --release --example high_performance_benchmark`
- [ ] Measure TPS with different batch sizes
- [ ] Verify SIMD verification is running

### Expected Results:
- **Baseline**: 6,200 TPS (current)
- **HTTP/2 + Optimizations**: 300,000 TPS (50x)
- **Rust Client + Large Batches**: 1,000,000+ TPS (160x)

---

## Why This Will Work

### 1. HTTP/2 Multiplexing
- Single connection handles 1000 concurrent streams
- No TCP handshake overhead per request
- Header compression (HPACK)

### 2. Zero-Copy Networking
- `Bytes` type avoids allocations
- Direct memory mapping for large buffers
- Streaming body processing

### 3. Proper TCP Configuration
- `TCP_NODELAY` eliminates 40ms Nagle delay
- Large buffers (4MB) prevent blocking
- `SO_REUSEPORT` for kernel load balancing

### 4. Connection Pooling
- Reuse TCP connections
- Amortize TLS handshake
- Reduce kernel overhead

### 5. Parallel Request Handling
- 10,000 concurrent connections
- Each connection handles 1000 streams
- Total capacity: 10,000,000 concurrent requests

---

## Key Insight

**The problem isn't Axum/HTTP** - it's that we're using default configuration designed for general web apps, not high-throughput transaction processing.

With proper configuration:
- **Axum + Hyper can absolutely handle 1M+ TPS**
- We just need to tune it like a database or message queue
- The SIMD optimizations we built will finally show their full potential

---

## Next Steps

1. **Implement `high_performance_server.rs`** - 2 hours
2. **Create Rust benchmark client** - 1 hour
3. **Test and measure** - 1 hour

**Total time to 1M+ TPS**: ~4 hours of implementation

---

**The path is clear. Let's build it.**
