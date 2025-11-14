# GPU Acceleration Plan for mistral.rs - 48 Hour Implementation
**Current State**: CPU at 2.7 tok/s (27x improvement from baseline)  
**Target State**: GPU at 100-200 tok/s (additional 37-74x improvement)  
**Timeline**: 48 hours to production GPU deployment  
**Hardware**: Bootstrap server (CPU-only) → Will add GPU when available

---

## Executive Summary

Based on feedback from DeepSeek, Kimi, and ChatGPT, plus your requirement to **stick with mistral.rs**, this plan focuses on:
1. Immediate: Optimize current CPU performance further (surgical fixes)
2. GPU-Ready: Prepare codebase for GPU without hardware
3. Bootstrap: Keep CPU-only server running efficiently until GPU arrives
4. Future: Clean GPU integration path when hardware is available

---

## Phase 1: Surgical CPU Optimizations (Today - 4 Hours)

### 1.1 Replace Mutex with RwLock (DeepSeek's Critical Fix)

**Problem**: `Mutex<Engine>` serializes ALL requests, even though inference is mostly read-only.

**File**: `crates/q-ai-inference/src/mistralrs_engine.rs`

**Change**:
```rust
// OLD (current)
static GLOBAL_ENGINE: OnceCell<Arc<tokio::sync::Mutex<MistralRsEngine>>> = OnceCell::new();

// NEW (allows concurrent reads)
use tokio::sync::RwLock;
static GLOBAL_ENGINE: OnceCell<Arc<RwLock<MistralRsEngine>>> = OnceCell::new();

pub fn get_global_engine() -> &'static Arc<RwLock<MistralRsEngine>> {
    GLOBAL_ENGINE.get_or_init(|| {
        info!("🚀 Initializing GLOBAL mistral.rs engine with RwLock");
        let engine = MistralRsEngine::new(MistralRsConfig::default())
            .expect("Failed to initialize global engine");
        Arc::new(RwLock::new(engine))
    })
}
```

**Usage**:
```rust
// For read-heavy operations (most of inference)
let engine = get_global_engine();
let eng = engine.read().await;  // Multiple readers allowed!
let response = eng.generate(&prompt, max_tokens).await?;
```

**Expected Impact**: 1.5-2x throughput under concurrent load (2.7 → 4-5 tok/s)

**Time**: 1 hour

### 1.2 Remove Request Queue + Use Single Worker (Per All AI Reviews)

**Problem**: We're using BOTH a queue AND a Mutex - double serialization.

**Solution**: Single worker owns the engine (no locks needed).

**File**: `crates/q-ai-inference/src/engine_worker.rs` (NEW)

```rust
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub struct WorkItem {
    pub prompt: String,
    pub max_tokens: usize,
    pub cancel: CancellationToken,
    pub response_tx: oneshot::Sender<Result<String>>,
}

pub struct EngineWorker {
    tx: mpsc::UnboundedSender<WorkItem>,
}

impl EngineWorker {
    pub fn new(config: MistralRsConfig) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<WorkItem>();
        
        tokio::spawn(async move {
            // Worker OWNS the engine (no Arc, no Mutex!)
            let mut engine = MistralRsEngine::new(config)
                .expect("Failed to initialize engine");
            
            // Warmup
            info!("🔥 Worker warming up engine...");
            let _ = engine.generate("warmup", 32).await;
            info!("✅ Worker ready!");
            
            // Process requests sequentially
            while let Some(work) = rx.recv().await {
                let WorkItem { prompt, max_tokens, cancel, response_tx } = work;
                
                // Check cancellation before starting
                if cancel.is_cancelled() {
                    let _ = response_tx.send(Err(anyhow::anyhow!("Cancelled")));
                    continue;
                }
                
                // Run inference
                let result = engine.generate(&prompt, max_tokens).await;
                let _ = response_tx.send(result);
            }
        });
        
        Self { tx }
    }
    
    pub async fn submit(&self, prompt: String, max_tokens: usize) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        let cancel = CancellationToken::new();
        
        self.tx.send(WorkItem {
            prompt,
            max_tokens,
            cancel,
            response_tx: tx,
        })?;
        
        rx.await?
    }
}

// Global worker (initialized once)
static GLOBAL_WORKER: OnceCell<EngineWorker> = OnceCell::new();

pub fn get_global_worker() -> &'static EngineWorker {
    GLOBAL_WORKER.get_or_init(|| {
        EngineWorker::new(MistralRsConfig::default())
    })
}
```

**Benefits**:
- No locks (worker owns engine)
- Simple cancellation support
- Clear ownership model
- Easy to reason about

**Expected Impact**: -10ms latency per request (removed mutex overhead)

**Time**: 2 hours

### 1.3 TCP_NODELAY + Streaming Improvements

**File**: `crates/q-api-server/src/main.rs`

```rust
use tokio::net::TcpSocket;

let socket = TcpSocket::new_v4()?;
socket.set_nodelay(true)?;  // Disable Nagle's algorithm
socket.bind(addr)?;
let listener = socket.listen(1024)?;

info!("🚀 Server listening on {} (TCP_NODELAY enabled)", addr);
axum::serve(listener, app).await?;
```

**Impact**: Tokens stream immediately instead of buffering

**Time**: 15 minutes

### 1.4 Backpressure + Fast Fail

**File**: `crates/q-ai-inference/src/engine_worker.rs`

```rust
const MAX_QUEUE_DEPTH: usize = 32;

let (tx, mut rx) = mpsc::channel::<WorkItem>(MAX_QUEUE_DEPTH);  // Bounded!

// In handler:
if worker.tx.try_send(work).is_err() {
    // Queue full - fail fast with 503
    return Err((
        StatusCode::SERVICE_UNAVAILABLE,
        "Server overloaded, try again in 5 seconds"
    ));
}
```

**Benefits**:
- No unbounded queues (prevents OOM)
- Fast fail when overloaded
- Clear capacity limits

**Expected Impact**: Prevents server degradation under load

**Time**: 30 minutes

### 1.5 Cancellation on Client Disconnect

**File**: `crates/q-api-server/src/chat_api.rs`

```rust
pub async fn stream_message(...) -> Sse<impl Stream<...>> {
    let (mut sender, body) = axum::body::Body::channel();
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    
    // Detect client disconnect
    tokio::spawn(async move {
        sender.closed().await;
        info!("🚫 Client disconnected, cancelling generation");
        cancel_clone.cancel();
    });
    
    // Pass cancel token to worker
    let response = worker.submit_cancellable(prompt, max_tokens, cancel).await?;
    
    // ... stream response ...
}
```

**Impact**: Stop wasting CPU on abandoned requests

**Time**: 30 minutes

---

## Phase 2: GPU-Ready Code (Today - 2 Hours)

### 2.1 Feature Flag for CUDA

**File**: `crates/q-ai-inference/Cargo.toml`

```toml
[features]
default = []
cuda = ["candle-core/cuda", "mistralrs/cuda"]
metal = ["candle-core/metal", "mistralrs/metal"]  # For future macOS support
```

**File**: `crates/q-ai-inference/src/mistralrs_engine.rs`

```rust
impl MistralRsEngine {
    pub fn new(config: MistralRsConfig) -> Result<Self> {
        // Auto-detect GPU and fallback to CPU
        #[cfg(feature = "cuda")]
        let device = match Device::new_cuda(0) {
            Ok(gpu) => {
                info!("🎮 Using CUDA GPU for inference!");
                gpu
            }
            Err(e) => {
                warn!("⚠️ CUDA init failed ({}), falling back to CPU", e);
                Device::Cpu
            }
        };
        
        #[cfg(feature = "metal")]
        let device = match Device::new_metal(0) {
            Ok(gpu) => {
                info!("🍎 Using Metal GPU for inference!");
                gpu
            }
            Err(e) => {
                warn!("⚠️ Metal init failed ({}), falling back to CPU", e);
                Device::Cpu
            }
        };
        
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        let device = {
            info!("💻 Using CPU for inference (compile with --features cuda for GPU)");
            Device::Cpu
        };
        
        // ... rest of initialization ...
        
        Ok(Self { device, model, config })
    }
}
```

**Benefits**:
- Code is GPU-ready NOW
- Compiles on CPU-only servers
- Auto-detects GPU when available
- Clear logging of acceleration status

**Time**: 1 hour

### 2.2 Build Configuration

**File**: `.cargo/config.toml` (NEW)

```toml
[build]
rustflags = ["-C", "target-cpu=native", "-C", "opt-level=3"]

[env]
RAYON_NUM_THREADS = { value = "16", relative = true }

[profile.release]
lto = "thin"
opt-level = 3
codegen-units = 1
panic = "abort"
```

**File**: `build_gpu.sh` (NEW)

```bash
#!/bin/bash
set -e

echo "🏗️ Building mistral.rs with GPU support..."

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    echo "🔧 Building with CUDA..."
    cargo build --release --features cuda --package q-api-inference
    cargo build --release --features cuda --package q-api-server
    
    echo "✅ GPU build complete!"
else
    echo "⚠️ No NVIDIA GPU found, building CPU-only version..."
    cargo build --release --package q-ai-inference
    cargo build --release --package q-api-server
    
    echo "ℹ️ To enable GPU: install CUDA toolkit and rebuild with --features cuda"
fi
```

**Time**: 30 minutes

### 2.3 Systemd Service with GPU Support

**File**: `/etc/systemd/system/q-api-server.service`

```ini
[Unit]
Description=Q-NarwhalKnight API Server with AI Inference
After=network.target

[Service]
Type=simple
User=orobit
WorkingDirectory=/opt/orobit/shared/q-narwhalknight
ExecStart=/opt/orobit/shared/q-narwhalknight/target/release/q-api-server

# Environment
Environment="RUST_LOG=info"
Environment="RUSTFLAGS=-C target-cpu=native"
Environment="RAYON_NUM_THREADS=16"

# GPU support (when available)
Environment="CUDA_VISIBLE_DEVICES=0"

# Resource limits
LimitNOFILE=65535
MemoryMax=24G

# Restart on failure
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

**Time**: 30 minutes

---

## Phase 3: Enhanced Observability (Today - 1 Hour)

### 3.1 Prometheus Metrics

**File**: `crates/q-ai-inference/src/metrics.rs` (ENHANCED)

```rust
use prometheus::{HistogramVec, IntCounterVec, IntGauge, Registry};
use once_cell::sync::Lazy;

static REGISTRY: Lazy<Registry> = Lazy::new(Registry::new);

static INFERENCE_TTFT: Lazy<HistogramVec> = Lazy::new(|| {
    let h = HistogramVec::new(
        histogram_opts!(
            "inference_ttft_seconds",
            "Time to first token",
            vec![0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]  // Buckets
        ),
        &["device"]  // Label: cpu or cuda
    ).unwrap();
    REGISTRY.register(Box::new(h.clone())).unwrap();
    h
});

static INFERENCE_THROUGHPUT: Lazy<HistogramVec> = Lazy::new(|| {
    let h = HistogramVec::new(
        histogram_opts!(
            "inference_throughput_toks_per_second",
            "Token generation throughput",
            vec![1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]
        ),
        &["device"]
    ).unwrap();
    REGISTRY.register(Box::new(h.clone())).unwrap();
    h
});

static QUEUE_DEPTH: Lazy<IntGauge> = Lazy::new(|| {
    let g = IntGauge::new("inference_queue_depth", "Current queue depth").unwrap();
    REGISTRY.register(Box::new(g.clone())).unwrap();
    g
});

static CANCELLATIONS: Lazy<IntCounterVec> = Lazy::new(|| {
    let c = IntCounterVec::new(
        opts!("inference_cancellations_total", "Total cancelled requests"),
        &["reason"]  // client_disconnect, timeout, error
    ).unwrap();
    REGISTRY.register(Box::new(c.clone())).unwrap();
    c
});

pub fn record_inference(device: &str, ttft_ms: u64, throughput: f64) {
    INFERENCE_TTFT
        .with_label_values(&[device])
        .observe(ttft_ms as f64 / 1000.0);
    
    INFERENCE_THROUGHPUT
        .with_label_values(&[device])
        .observe(throughput);
}

pub fn record_cancellation(reason: &str) {
    CANCELLATIONS.with_label_values(&[reason]).inc();
}

pub fn set_queue_depth(depth: i64) {
    QUEUE_DEPTH.set(depth);
}
```

**File**: `crates/q-api-server/src/handlers.rs`

```rust
// GET /metrics
pub async fn metrics() -> Result<String, StatusCode> {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    
    let metric_families = q_ai_inference::metrics::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    String::from_utf8(buffer)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}
```

**Grafana Dashboard Query Examples**:
```promql
# P95 Time to First Token
histogram_quantile(0.95, rate(inference_ttft_seconds_bucket[5m]))

# Current throughput
rate(inference_throughput_toks_per_second_sum[1m]) / rate(inference_throughput_toks_per_second_count[1m])

# Queue depth over time
inference_queue_depth

# Cancellation rate
rate(inference_cancellations_total[5m])
```

**Time**: 1 hour

---

## Phase 4: Smaller Model Option (Optional - 1 Hour)

### 4.1 Phi-3-Mini Integration

**Why**: 3.8B params = 2x faster on CPU than Mistral-7B

**File**: `crates/q-ai-inference/src/model_config.rs`

```rust
pub enum ModelChoice {
    Mistral7B,   // 7B params, higher quality
    Phi3Mini,    // 3.8B params, faster
}

impl ModelChoice {
    pub fn model_path(&self) -> &str {
        match self {
            Self::Mistral7B => "/home/orobit/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
            Self::Phi3Mini => "/home/orobit/models/Phi-3-mini-4k-instruct.Q4_K_M.gguf",
        }
    }
    
    pub fn expected_throughput_cpu(&self) -> f64 {
        match self {
            Self::Mistral7B => 2.7,   // Current performance
            Self::Phi3Mini => 5.0,    // Estimated (smaller model)
        }
    }
}
```

**Download Phi-3-Mini**:
```bash
cd /home/orobit/models
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

**Time**: 1 hour (if chosen)

---

## Phase 5: GPU Integration (When Hardware Arrives - 4 Hours)

### 5.1 CUDA Installation

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Verify
nvidia-smi
nvcc --version
```

### 5.2 Build with CUDA

```bash
cd /opt/orobit/shared/q-narwhalknight
./build_gpu.sh

# Verify GPU is used
./target/release/q-api-server
# Should see: "🎮 Using CUDA GPU for inference!"
```

### 5.3 Expected Performance

| Metric | CPU (Current) | GPU (RTX 4090) | Improvement |
|--------|---------------|----------------|-------------|
| TTFT | 2.1s | **0.2-0.5s** | **4-10x faster** |
| Throughput | 2.7 tok/s | **100-200 tok/s** | **37-74x faster** |
| 50-token response | 18.5s | **<1 second** | **18x faster** |

---

## Implementation Timeline

### Day 0 (Today) - CPU Optimizations
| Time | Task | Expected Improvement |
|------|------|---------------------|
| 1h | Replace Mutex with RwLock | 1.5-2x throughput |
| 2h | Single worker pattern | -10ms latency |
| 0.5h | TCP_NODELAY | Better streaming |
| 0.5h | Backpressure | Stability |
| 0.5h | Cancellation | CPU efficiency |
| 1h | GPU-ready code | Ready for GPU |
| 0.5h | Build scripts | Easy deployment |
| 1h | Enhanced metrics | Visibility |
| **7h total** | **All CPU optimizations** | **2x improvement** |

**Expected Result**: 2.7 tok/s → **4-5 tok/s** on CPU

### Day 1 (When GPU Arrives) - GPU Activation
| Time | Task |
|------|------|
| 2h | CUDA installation |
| 1h | Build with CUDA |
| 1h | Performance testing |
| **4h total** | **GPU ready** |

**Expected Result**: 4-5 tok/s → **100-200 tok/s**

---

## Success Criteria

### End of Day 0 (CPU Optimized):
- ✅ TTFT < 2 seconds (currently 2.1s)
- ✅ Throughput > 4 tok/s (currently 2.7 tok/s)
- ✅ No head-of-line blocking (RwLock)
- ✅ Clean cancellation support
- ✅ Prometheus metrics exposed
- ✅ Code compiles with --features cuda (GPU-ready)

### With GPU (Future):
- ✅ TTFT < 500ms (p95)
- ✅ Throughput > 100 tok/s (p95)
- ✅ Error rate < 0.1%
- ✅ Auto-fallback to CPU if GPU fails

---

## Risk Mitigation

### Risk 1: RwLock Doesn't Help
**Mitigation**: Keep single worker as fallback (already implemented)

### Risk 2: GPU Installation Fails
**Mitigation**: Code works on CPU by default, GPU is optional feature

### Risk 3: Performance Regresses
**Mitigation**: Metrics track every change, rollback if p95 degrades

### Risk 4: Memory Issues with GPU
**Mitigation**: Monitor `nvidia-smi`, adjust batch size, use smaller model

---

## Monitoring Alerts

```yaml
# Prometheus AlertManager

- alert: SlowInference
  expr: histogram_quantile(0.95, inference_ttft_seconds_bucket) > 2.0
  for: 5m
  severity: warning
  annotations:
    summary: "95th percentile TTFT > 2 seconds"

- alert: LowThroughput  
  expr: rate(inference_throughput_toks_per_second_sum[5m]) / rate(inference_throughput_toks_per_second_count[5m]) < 2.0
  for: 5m
  severity: warning
  annotations:
    summary: "Average throughput < 2 tok/s"

- alert: HighQueueDepth
  expr: inference_queue_depth > 10
  for: 2m
  severity: critical
  annotations:
    summary: "Inference queue backed up (>10 requests)"
```

---

## Final Recommendations

### For Bootstrap Server (CPU-Only):
1. ✅ Implement all Phase 1 optimizations (7 hours) → **2x improvement**
2. ✅ Add Prometheus metrics for visibility
3. ✅ Make code GPU-ready (feature flags)
4. ⏳ Wait for GPU hardware
5. ✅ When GPU arrives: `./build_gpu.sh` → **50-100x additional improvement**

### Don't Do:
- ❌ Don't try llama.cpp/vLLM now (stick with mistral.rs as requested)
- ❌ Don't remove distributed code yet (wait for GPU validation)
- ❌ Don't promise <1s responses until GPU is deployed

### Do This:
- ✅ Implement surgical CPU fixes TODAY
- ✅ Make code GPU-ready (no hardware required)
- ✅ Add comprehensive metrics
- ✅ Keep users informed of progress

---

**Next Action**: Start Phase 1 (CPU optimizations) - 7 hours of work for 2x improvement while waiting for GPU hardware.

