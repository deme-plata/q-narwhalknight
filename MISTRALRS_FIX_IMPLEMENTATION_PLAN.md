# mistral.rs Performance Fix - Implementation Plan
**Date**: 2025-11-13  
**Based on**: Technical reviews from Kimi, DeepSeek, and internal analysis  
**Priority**: CRITICAL (P0)  
**Timeline**: 48 hours to production-ready state

---

## Executive Summary

All three technical reviews converge on the same core issues:
1. **CPU-only execution is the killer** (200-3000x slower than GPU)
2. **Cold starts on every request** (60s model loading)
3. **Over-engineered distributed architecture** (solving non-existent problems)

**Consensus Solution**: 
- **Short-term** (Today): Fix cold starts with singleton pattern
- **Medium-term** (48h): Add GPU support OR use vLLM/llama.cpp
- **Long-term** (Week 2+): Remove distributed complexity

---

## Phase 0: Emergency Triage (Next 2 Hours)

### 0.1 Sanity Checks - Build Configuration

**Problem**: Might be running debug build or without optimizations.

**Fix**:
```bash
# Verify current build configuration
cd /opt/orobit/shared/q-narwhalknight
cargo build --release --package q-ai-inference

# Add CPU-specific optimizations
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
cargo build --release --package q-ai-inference

# Force Rayon to use physical cores only
export RAYON_NUM_THREADS=$(nproc)

# Pin to NUMA node 0 (if multi-socket)
numactl --cpunodebind=0 --membind=0 ./target/release/q-api-server
```

**Expected Improvement**: 2-5x speedup (0.1 → 0.5 tok/s)
**Time**: 30 minutes

### 0.2 Verify Model File Location

**Problem**: Model might be on slow disk or network mount.

**Check**:
```bash
# Verify model location and read speed
ls -lh /opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf

# Test disk I/O speed
dd if=/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf \
   of=/dev/null bs=1M count=4096 iflag=direct

# Should see >500 MB/s for SSD
```

**Action**: If model is on slow storage, copy to local NVMe/SSD.

**Expected Improvement**: 5-10s faster first token
**Time**: 15 minutes

### 0.3 Kill Background Processes

**Problem**: Other processes competing for CPU.

**Action**:
```bash
# Check CPU usage
htop

# Kill unnecessary services
systemctl stop postgresql  # If not needed
systemctl stop docker      # If not needed

# Verify server has dedicated CPU
ps aux | grep q-api-server
```

**Expected Improvement**: 20-30% throughput increase
**Time**: 15 minutes

---

## Phase 1: Singleton Engine (Today - Critical Fix)

### 1.1 Root Cause

**Current**: Model loads on EVERY request (60s penalty each time)
**Should Be**: Model loads ONCE on server start (60s penalty once)

### 1.2 Implementation

**File**: `crates/q-ai-inference/src/mistralrs_engine.rs`

**Add**:
```rust
use once_cell::sync::OnceCell;
use std::sync::Arc;

// Global singleton (loaded once per process)
static GLOBAL_ENGINE: OnceCell<Arc<tokio::sync::Mutex<MistralRsEngine>>> = OnceCell::new();

pub fn get_global_engine() -> &'static Arc<tokio::sync::Mutex<MistralRsEngine>> {
    GLOBAL_ENGINE.get_or_init(|| {
        info!("🚀 Initializing GLOBAL mistral.rs engine (ONE TIME ONLY)");
        let engine = MistralRsEngine::new(MistralRsConfig::default())
            .expect("Failed to initialize global engine");
        Arc::new(tokio::sync::Mutex::new(engine))
    })
}

// Warmup function (call on server start)
pub async fn warmup_global_engine() -> Result<()> {
    info!("🔥 Warming up global engine...");
    let engine = get_global_engine();
    let mut eng = engine.lock().await;
    
    // Generate dummy tokens to allocate KV-cache
    let warmup_prompt = "Hello";
    let _response = eng.generate(warmup_prompt, 32).await?;
    
    info!("✅ Global engine warmed up and ready!");
    Ok(())
}
```

**File**: `crates/q-api-server/src/main.rs`

**Change**:
```rust
#[tokio::main]
async fn main() -> Result<()> {
    // ... existing setup ...
    
    // WARMUP ENGINE BEFORE ACCEPTING REQUESTS
    info!("🔥 Pre-warming AI inference engine...");
    q_ai_inference::warmup_global_engine().await?;
    info!("✅ AI engine ready!");
    
    // Start HTTP server
    let listener = TcpListener::bind(&addr).await?;
    info!("🚀 Server listening on {}", addr);
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

**File**: `crates/q-api-server/src/chat_api.rs`

**Change**:
```rust
// OLD (creates new engine every time - SLOW!)
// let engine = MistralRsEngine::new(config).await?;

// NEW (use singleton - FAST!)
let engine = q_ai_inference::get_global_engine();
let mut eng = engine.lock().await;
let response = eng.generate_stream(prompt, max_tokens, callback).await?;
```

### 1.3 Expected Results

**Before**:
- First token: 60 seconds (model loading)
- Each request: Cold start

**After**:
- Server startup: 60 seconds (ONE TIME)
- First token: 2-5 seconds (just generation)
- Subsequent requests: 2-5 seconds (NO cold start)

**Speedup**: 12-30x for first token (60s → 2-5s)

### 1.4 Dependencies

Add to `crates/q-ai-inference/Cargo.toml`:
```toml
once_cell = "1.21"
```

**Time to Implement**: 2 hours
**Time to Test**: 1 hour
**Total**: 3 hours

---

## Phase 2: Request Queue (Today - Stability Fix)

### 2.1 Root Cause

**Problem**: Multiple concurrent requests thrash CPU caches and fight for memory bandwidth.

### 2.2 Implementation

**File**: `crates/q-ai-inference/src/request_queue.rs` (NEW)

```rust
use tokio::sync::{mpsc, oneshot};

pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub response_tx: oneshot::Sender<Result<String>>,
}

pub struct RequestQueue {
    tx: mpsc::UnboundedSender<InferenceRequest>,
}

impl RequestQueue {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<InferenceRequest>();
        
        // Spawn worker that processes requests sequentially
        tokio::spawn(async move {
            let engine = get_global_engine();
            
            while let Some(req) = rx.recv().await {
                let mut eng = engine.lock().await;
                let result = eng.generate(&req.prompt, req.max_tokens).await;
                let _ = req.response_tx.send(result);
            }
        });
        
        Self { tx }
    }
    
    pub async fn submit(&self, prompt: String, max_tokens: usize) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(InferenceRequest {
            prompt,
            max_tokens,
            response_tx: tx,
        })?;
        
        rx.await?
    }
}

// Global queue
static GLOBAL_QUEUE: OnceCell<RequestQueue> = OnceCell::new();

pub fn get_global_queue() -> &'static RequestQueue {
    GLOBAL_QUEUE.get_or_init(|| RequestQueue::new())
}
```

**File**: `crates/q-api-server/src/chat_api.rs`

**Change**:
```rust
// Submit to queue (prevents concurrent thrashing)
let response = q_ai_inference::get_global_queue()
    .submit(formatted_prompt, max_tokens)
    .await?;
```

### 2.3 Expected Results

**Before**: N concurrent requests = N× memory thrashing
**After**: Sequential processing = predictable latency

**Improvement**: 30-50% throughput increase under load

**Time to Implement**: 2 hours

---

## Phase 3: Observability (Today - Diagnosis)

### 3.1 Add Performance Metrics

**File**: `crates/q-ai-inference/src/metrics.rs` (NEW)

```rust
use std::time::Instant;

pub struct InferenceMetrics {
    pub ttft_ms: u64,           // Time to first token
    pub decode_rate: f64,        // Tokens per second
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub total_time_ms: u64,
}

pub fn record_inference(metrics: InferenceMetrics) {
    info!(
        "📊 TTFT: {}ms | Decode: {:.1} tok/s | Prefill: {} | Decode: {} | Total: {}ms",
        metrics.ttft_ms,
        metrics.decode_rate,
        metrics.prefill_tokens,
        metrics.decode_tokens,
        metrics.total_time_ms
    );
}
```

**Expected Output**:
```
📊 TTFT: 2340ms | Decode: 1.2 tok/s | Prefill: 128 | Decode: 150 | Total: 127850ms
```

**Time to Implement**: 1 hour

---

## Phase 4: GPU Support (48 Hours - If Hardware Available)

### Option A: Enable CUDA in mistral.rs

**File**: `crates/q-ai-inference/Cargo.toml`

**Change**:
```toml
[features]
default = ["cuda"]  # Enable by default if GPU present
cuda = [
    "candle-core/cuda",
    "mistralrs/cuda",
]
```

**File**: `crates/q-ai-inference/src/mistralrs_engine.rs`

**Change**:
```rust
#[cfg(feature = "cuda")]
let device = Device::new_cuda(0)?;

#[cfg(not(feature = "cuda"))]
let device = Device::Cpu;

info!("🎮 Using device: {:?}", device);
```

**Build**:
```bash
# Install CUDA toolkit (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Verify
nvidia-smi

# Build with CUDA
cargo build --release --features cuda --package q-ai-inference
```

**Expected Performance**: 100-200 tok/s (200-400x speedup)

**Time**: 4 hours (2h install + 2h build + test)

### Option B: Switch to vLLM (Kimi's Recommendation)

**Why**: Production-proven, better performance, less maintenance

**Implementation**:
```bash
# Install vLLM
pip install vllm

# Start vLLM server
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096
```

**File**: `crates/q-api-server/src/chat_api.rs`

**Change**:
```rust
// Use OpenAI-compatible client
let client = reqwest::Client::new();
let response = client
    .post("http://localhost:8000/v1/completions")
    .json(&serde_json::json!({
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "stream": true,
    }))
    .send()
    .await?;
```

**Expected Performance**: 200-400 tok/s (continuous batching)

**Pros**:
- Battle-tested production code
- Better throughput (continuous batching)
- OpenAI-compatible API
- Better observability

**Cons**:
- Python dependency
- Separate process to manage

**Time**: 3 hours (1h install + 2h integration)

### Option C: Switch to llama.cpp (DeepSeek's Suggestion)

**Why**: Better CPU performance than mistral.rs, GPU support, battle-tested

**Implementation**:
```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc) LLAMA_CUDA=1  # Or LLAMA_OPENBLAS=1 for CPU

# Start server
./llama-server \
  -m /path/to/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf \
  -c 4096 \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 35  # GPU layers (or 0 for CPU-only)
```

**Expected Performance**:
- CPU: 2-5 tok/s (5-10x better than mistral.rs)
- GPU: 100-150 tok/s

**Time**: 2 hours (1h build + 1h integration)

---

## Phase 5: Architectural Cleanup (Week 2)

### 5.1 Remove Distributed AI Complexity

**Delete** (DeepSeek's "kill dead paths" recommendation):
```bash
# Remove unused files
rm crates/q-network/src/distributed_ai_coordinator.rs
rm crates/q-network/src/distributed_ai_worker.rs
rm crates/q-ai-inference/src/pipeline_executor.rs
rm crates/q-ai-inference/src/load_balancer.rs

# Total: ~3,000 lines deleted
```

**File**: `crates/q-ai-inference/src/lib.rs`

**Simplify**:
```rust
// OLD (complex, unused)
pub struct MistralRsConfig {
    pub enable_distributed: bool,      // ❌ Remove
    pub enable_pipeline: bool,         // ❌ Remove
    pub enable_load_balancing: bool,   // ❌ Remove
    // ...
}

// NEW (simple, focused)
pub struct MistralRsConfig {
    pub model_path: String,
    pub max_seq_len: usize,
    pub temperature: f64,
    pub enable_kv_cache: bool,  // Keep (useful)
}
```

**Expected Improvement**: 
- -3,000 lines of code
- -10-50ms latency per request
- Easier to maintain

**Time**: 4 hours

---

## Phase 6: SLOs and Performance Testing (Week 2)

### 6.1 Define Service Level Objectives

**File**: `crates/q-ai-inference/src/slo.rs` (NEW)

```rust
pub struct InferenceSLO {
    pub ttft_p95_ms: u64,        // 95th percentile time to first token
    pub decode_p95_toks: f64,    // 95th percentile tokens/sec
    pub error_rate_percent: f64, // Error rate
}

pub const PRODUCTION_SLO: InferenceSLO = InferenceSLO {
    ttft_p95_ms: 2000,      // < 2 seconds to first token
    decode_p95_toks: 50.0,  // > 50 tok/s average
    error_rate_percent: 0.5, // < 0.5% errors
};
```

### 6.2 Add Performance Benchmarks

**File**: `crates/q-ai-inference/benches/inference_bench.rs` (NEW)

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");
    
    for prompt_len in [128, 512, 2048] {
        group.bench_with_input(
            BenchmarkId::new("ttft", prompt_len),
            &prompt_len,
            |b, &len| {
                b.iter(|| {
                    // Measure time to first token
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_inference);
criterion_main!(benches);
```

**Run**:
```bash
cargo bench --package q-ai-inference
```

**Time**: 3 hours

---

## Implementation Timeline

### Day 0 (Today) - Emergency Fixes
| Time | Task | Owner | Expected Improvement |
|------|------|-------|---------------------|
| 2h | Phase 0: Sanity checks | Dev | 2-5x |
| 3h | Phase 1: Singleton engine | Dev | 12-30x |
| 2h | Phase 2: Request queue | Dev | 1.5x |
| 1h | Phase 3: Observability | Dev | N/A (diagnosis) |
| **8h total** | **End of Day 0** | | **24-150x improvement** |

**Expected Result**: 0.1 tok/s → 2-15 tok/s (usable but slow)

### Day 1 (Tomorrow) - GPU Integration
| Time | Task | Owner | Expected Improvement |
|------|------|-------|---------------------|
| 4h | Option A: CUDA setup | DevOps | 100-200x |
| OR 3h | Option B: vLLM setup | DevOps | 200-400x |
| OR 2h | Option C: llama.cpp | DevOps | 100-150x |
| 2h | Integration testing | QA | N/A |
| 1h | Performance validation | QA | N/A |
| **7h total** | **End of Day 1** | | **Additional 50-100x** |

**Expected Result**: 2-15 tok/s → 100-400 tok/s (production-ready)

### Week 2 - Cleanup
| Time | Task | Owner |
|------|------|-------|
| 4h | Phase 5: Remove distributed code | Dev |
| 3h | Phase 6: SLOs and benchmarks | Dev |
| **7h total** | **Week 2** | |

---

## Success Criteria

### Minimum Viable (Day 0 End)
- ✅ First token < 5 seconds (down from 60s)
- ✅ Throughput > 1 tok/s (up from 0.1-0.5)
- ✅ No cold starts after server warmup
- ✅ Metrics logging for all requests

### Production Ready (Day 1 End)
- ✅ First token < 1 second (95th percentile)
- ✅ Throughput > 50 tok/s (95th percentile)
- ✅ Error rate < 0.5%
- ✅ Performance benchmarks passing

### Long-term Health (Week 2 End)
- ✅ -3,000 lines of dead code removed
- ✅ SLOs defined and monitored
- ✅ Automated performance regression tests
- ✅ Documentation updated

---

## Risk Mitigation

### Risk 1: GPU Not Available
**Mitigation**: Implement Option C (llama.cpp) for better CPU performance (2-5 tok/s)

### Risk 2: Singleton Pattern Breaks Streaming
**Mitigation**: Use `Arc<Mutex<Engine>>` with short lock durations, or implement lock-free queue

### Risk 3: Performance Doesn't Improve Enough
**Mitigation**: Fall back to cloud API (Groq) as temporary solution while debugging

### Risk 4: Code Changes Break Existing Features
**Mitigation**: Comprehensive testing before deployment, feature flags for rollback

---

## Monitoring and Alerts

### Metrics to Track
```rust
// Prometheus metrics
inference_ttft_seconds{p50,p95,p99}
inference_throughput_toks_per_second{p50,p95,p99}
inference_requests_total
inference_errors_total
inference_queue_depth
```

### Alerts
```yaml
- alert: SlowInference
  expr: inference_ttft_seconds{p95} > 2.0
  for: 5m
  severity: critical
  
- alert: LowThroughput
  expr: inference_throughput_toks_per_second{p95} < 50
  for: 5m
  severity: warning
```

---

## Final Recommendations

### For CPU-Only Environment (Bootstrap Server)
1. ✅ Implement Phase 0-3 TODAY (singleton + queue + metrics)
2. ✅ Switch to llama.cpp for 5-10x CPU improvement
3. ✅ Use smaller model (Phi-3-mini 3.8B) for 2x speed vs quality trade-off
4. ⚠️ Set expectations: "This will be slow (~2-5 tok/s), GPU recommended"

### For Production Environment
1. ✅ Add GPU (RTX 4070+ or cloud GPU)
2. ✅ Use vLLM for best performance and operational maturity
3. ✅ Set SLOs and monitor continuously
4. ✅ Keep cloud API fallback for resilience

### Don't Do This
- ❌ Keep current distributed AI code (dead weight)
- ❌ Ship CPU-only as primary experience (too slow)
- ❌ Ignore performance testing (blindness)
- ❌ Promise fast AI without GPU (dishonest)

---

**Next Steps**: Execute Phase 0-1 immediately (8 hours), then decide on GPU strategy based on hardware availability.

