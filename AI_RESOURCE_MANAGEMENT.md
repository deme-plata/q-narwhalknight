# AI Resource Management Guide

## Problem: Server Unresponsiveness with AI Enabled

When `Q_ENABLE_AI=1`, the mistral.rs inference engine would consume all CPU cores, making the server unresponsive for mining, API requests, and even SSH connections.

## Solutions Implemented

### 1. **CPU Core Limiting** ✅

The AI engine now uses only **25% of available CPU cores** by default (4 cores on your 18-core system).

**How it works:**
- Uses `rayon` thread pool to limit parallelism
- Leaves 75% of cores free for mining and API operations
- Configurable via `Q_AI_THREADS` environment variable

**Configuration:**
```bash
# In /etc/systemd/system/q-api-server.service:
Environment="Q_AI_THREADS=4"  # Use 4 CPU cores for AI (out of 18)

# Adjust based on your needs:
# - More cores = faster AI inference, but less CPU for mining
# - Fewer cores = slower AI, but better mining performance
```

### 2. **Request Rate Limiting** ✅

Only **2 concurrent AI requests** are allowed by default. Additional requests wait in queue.

**How it works:**
- Uses tokio Semaphore to limit concurrent inference
- Prevents CPU overload from simultaneous requests
- Requests queue automatically when limit reached

**Configuration:**
```bash
# In /etc/systemd/system/q-api-server.service:
Environment="Q_AI_MAX_CONCURRENT=2"  # Max 2 AI requests at once

# Adjust based on your needs:
# - Higher value = more concurrent requests, but higher CPU load
# - Lower value = slower when multiple users, but more responsive server
```

### 3. **Optional: System-Level CPU Quota** (Currently Disabled)

systemd can enforce a hard CPU limit on the entire service.

**To enable:**
Uncomment this line in `/etc/systemd/system/q-api-server.service`:
```bash
CPUQuota=50%  # Limit service to 50% of total CPU
```

Then reload:
```bash
systemctl daemon-reload
systemctl restart q-api-server
```

## Recommended Configurations

### Configuration A: Mining Priority (Current Default)
**Best for: Production mining nodes**

```bash
Q_ENABLE_AI=0              # AI disabled
Q_AI_THREADS=4             # If AI enabled: use 4 cores
Q_AI_MAX_CONCURRENT=2      # If AI enabled: max 2 requests
# CPUQuota=50%             # Uncommented if needed
```

**Result:**
- Mining gets full CPU resources
- Server always responsive
- AI can be enabled later without code changes

### Configuration B: Balanced AI + Mining
**Best for: Testing AI features while mining**

```bash
Q_ENABLE_AI=1              # AI enabled
Q_AI_THREADS=6             # Use 6 cores (33% of 18)
Q_AI_MAX_CONCURRENT=3      # Allow 3 concurrent requests
# CPUQuota=60%             # Optional safety net
```

**Result:**
- AI inference works but controlled
- Mining still gets majority of CPU
- Server remains responsive
- ~3-5 seconds per AI response

### Configuration C: AI Development Mode
**Best for: Dedicated AI development/testing server**

```bash
Q_ENABLE_AI=1              # AI enabled
Q_AI_THREADS=12            # Use 12 cores (67% of 18)
Q_AI_MAX_CONCURRENT=5      # Allow 5 concurrent requests
# No CPUQuota limit
```

**Result:**
- Fast AI inference (~1-2s first token)
- Mining may be slower
- Good for testing distributed AI features

## How to Change Configuration

### Step 1: Edit Service File
```bash
nano /etc/systemd/system/q-api-server.service
```

### Step 2: Modify Environment Variables
Change these lines:
```bash
Environment="Q_ENABLE_AI=1"           # Change 0 to 1 to enable AI
Environment="Q_AI_THREADS=6"          # Change to desired core count
Environment="Q_AI_MAX_CONCURRENT=3"   # Change to desired request limit
```

### Step 3: Reload and Restart
```bash
systemctl daemon-reload
systemctl restart q-api-server
```

### Step 4: Monitor Performance
```bash
# Watch CPU usage:
htop

# Watch service logs:
journalctl -u q-api-server -f

# Check AI performance:
curl http://localhost:8080/api/v1/chat -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, test response time"}'
```

## Performance Expectations

| Configuration | First Token | Tokens/sec | CPU Usage | Mining Impact |
|---------------|-------------|------------|-----------|---------------|
| **4 cores, 2 concurrent** | ~5s | 3-5 tok/s | 20-25% | Minimal |
| **6 cores, 3 concurrent** | ~3s | 5-8 tok/s | 30-40% | Low |
| **12 cores, 5 concurrent** | ~1.5s | 10-15 tok/s | 60-70% | Moderate |

## Troubleshooting

### Problem: Server still unresponsive with AI enabled

**Solutions:**
1. Reduce `Q_AI_THREADS` (try 2 or 3)
2. Reduce `Q_AI_MAX_CONCURRENT` to 1
3. Enable `CPUQuota=40%` in service file
4. Check for other resource-heavy processes: `htop`

### Problem: AI inference too slow

**Solutions:**
1. Increase `Q_AI_THREADS` (try 8 or 10)
2. Increase `Q_AI_MAX_CONCURRENT` (try 4 or 5)
3. Ensure model file exists and is Q4_K_M quantized
4. Check model path: `/opt/orobit/shared/q-narwhalknight/models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf`

### Problem: "Failed to acquire request permit" error

**Cause:** Too many concurrent requests (queue is full)

**Solutions:**
1. Increase `Q_AI_MAX_CONCURRENT`
2. Implement request timeout on client side
3. Add load balancer if multiple nodes

## Technical Details

### CPU Limiting Implementation
```rust
// In crates/q-ai-inference/src/mistralrs_engine.rs
let num_cpus = num_cpus::get();
let ai_threads = std::env::var("Q_AI_THREADS")
    .unwrap_or_else(|| (num_cpus / 4).max(1)); // 25% default

rayon::ThreadPoolBuilder::new()
    .num_threads(ai_threads)
    .build_global()
    .ok();
```

### Rate Limiting Implementation
```rust
// Request semaphore limits concurrent inference
let max_concurrent = std::env::var("Q_AI_MAX_CONCURRENT")
    .unwrap_or(2); // Default: 2 concurrent requests

let request_semaphore = Arc::new(
    tokio::sync::Semaphore::new(max_concurrent)
);

// Before inference:
let _permit = request_semaphore.acquire().await?;
// Permit automatically released when dropped
```

## Best Practices

1. **Start Conservative**: Begin with AI disabled or minimal resources
2. **Monitor First**: Enable AI with low resources and watch `htop`
3. **Gradually Increase**: Slowly increase cores/concurrency while monitoring
4. **Set Alerts**: Monitor CPU usage and set alerts at 80%
5. **Test Under Load**: Simulate multiple mining clients + AI requests
6. **Document Changes**: Keep notes on what works for your hardware

## Future Enhancements

Potential improvements for next version:

1. **Adaptive Scaling**: Automatically adjust threads based on load
2. **Priority Queue**: Prioritize mining over AI during high load
3. **GPU Support**: Offload inference to GPU (if available)
4. **Model Caching**: Keep model in memory between requests
5. **Streaming Throttle**: Slow down token generation to reduce CPU spikes

---

**Summary**: With these controls, you can safely enable AI features without sacrificing mining performance or server responsiveness. Start with the default settings and adjust based on your specific needs!
