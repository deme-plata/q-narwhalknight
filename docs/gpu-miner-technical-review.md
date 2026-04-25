# Q-NarwhalKnight GPU Miner — Technical Review for Enhancement

## Architecture Overview

The miner (`crates/q-miner/`, `crates/q-mining/`) implements a hybrid BLAKE3+VDF proof-of-work in OpenCL. Each GPU work item computes:
```
h = BLAKE3(challenge[32] || nonce_le[8])    // 40-byte input
for 99 rounds: h = BLAKE3(h)               // VDF chain
if h < difficulty_target: SOLUTION
```
100 BLAKE3 hashes per nonce, all in GPU registers with no global memory reads after initial challenge/target upload.

---

## Current Optimizations (What's Already Done)

- Persistent pre-allocated buffers (no alloc/free per dispatch)
- Conditional upload: challenge/target only re-uploaded on block change
- Challenge pre-converted to `u32[8]` on CPU (eliminates per-thread byte unpacking)
- `__constant` address space for challenge (hardware constant cache)
- `#pragma unroll 3` on 99-round VDF loop
- `reqd_work_group_size(256,1,1)` for register allocation hints
- Kernel binary caching (`~/.config/q-miner/kernel-cache/`)
- Adaptive work size (auto-tunes between 64K–64M to target 200–600ms dispatch)
- Two-phase submit/readback: all GPUs submit simultaneously before any readback (parallel multi-GPU)

---

## Known Bottlenecks & Areas for Improvement

### 1. BLAKE3 Inner Loop — Register Pressure

The VDF chain `for 99 rounds: h = BLAKE3(h)` runs 99 compressions. Each BLAKE3 compression needs 16 state words (`uint s[16]`), 16 message words, and 8 output words. On GPUs with limited registers per thread (e.g., 32 on older AMD), this causes register spilling to local memory, tanking throughput.

**To investigate:**
- Profile register usage with AMD ROCm Profiler or NVIDIA Nsight Compute
- Consider reducing `work_group_size` from 256 to 64 or 128 (fewer threads = more registers per thread)
- Try explicit `__private` declarations to hint compiler

### 2. BLAKE3 Compression — Wasted Message Words

For `blake3_hash_32` (32-byte input, the VDF step), `block[8..15]` are always zero. The compression function computes `G(...)` on these zero words, doing useless work. A specialized `blake3_compress_32` that skips zero message words could reduce ~12% of the G-operation count in the VDF loop.

**Proposed kernel optimization:**
```c
// Specialized for 32-byte input: block[8..15] = 0
// In the message schedule, wherever MSG_SCHED[r][col] >= 8, substitute mx/my = 0
// This eliminates ~12 of 112 G-operation operand loads per round per VDF iteration
void blake3_compress_32_opt(const uint cv[8], const uint data[8], uint output[8]) {
    uint s[16];
    // ... same setup ...
    for (int r = 0; r < 7; r++) {
        // Pre-check which indices are >= 8 and substitute 0 at compile time
        // Compiler should constant-fold the zero additions away
    }
}
```

### 3. `meets_target` — Branch Divergence

Most nonces fail at byte 0 of the difficulty target. The current implementation accesses `target[]` from `__global` memory inside a nested loop. On a warp/wavefront where most threads terminate at byte 0, the few surviving threads force the entire warp to continue executing.

**Improvement:** Pre-load the first 4 bytes of target into a register, compare word 0 of hash against target word 0 first as a fast-reject:
```c
bool meets_target(const uint hash[8], __global const uchar* target) {
    // Fast-reject using first byte only (catches ~99% of misses immediately)
    uchar h0 = (uchar)(hash[0] & 0xFFu);
    uchar t0 = target[0];
    if (h0 > t0) return false;
    if (h0 < t0) return true;
    // Full comparison only for the rare survivors past byte 0
    for (int i = 1; i < 32; i++) {
        uchar hb = (uchar)((hash[i >> 2] >> ((i & 3) * 8)) & 0xFFu);
        uchar tb = target[i];
        if (hb < tb) return true;
        if (hb > tb) return false;
    }
    return true;
}
```

### 4. Non-Temporal Stores for Results

`found_nonce`, `found_hash`, `found_flag` use `atomic_cmpxchg` which serializes the winning thread (correct). However, the `found_hash` write in the winning thread is 32 bytes written byte-by-byte via a loop. Packing into 8x `uint` writes reduces transactions:
```c
// Instead of byte loop, write 8 uint words
__global uint* found_hash_words = (__global uint*)found_hash;
for (int i = 0; i < 8; i++) found_hash_words[i] = h[i];
```

### 5. Multi-GPU Load Balancing

Current: each GPU gets a fixed nonce range per dispatch. Faster GPUs finish first and sit idle waiting for the slowest to finish before the next cycle begins.

**Improvement:** Work-stealing nonce queue. A shared `AtomicU64` nonce counter on the CPU side. When a GPU finishes, it atomically increments the counter by its work size and immediately starts the next chunk, without waiting for a synchronized `mine_batch` call. Eliminates inter-GPU idle time entirely.

### 6. Solution Callback Latency

When a GPU finds a solution, the flow is: GPU writes → `queue.finish()` → CPU reads flag → CPU reads nonce+hash → serializes to JSON → HTTP POST. This adds 10–50ms latency after the kernel finishes before the solution reaches the pool.

**Improvement:** Pre-allocate the HTTP request body buffer. On `found_flag[0] != 0`, do solution read + submission inline before returning `BatchResult`, saving one round-trip through the mining loop outer code.

### 7. VDF Loop Unrolling Strategy

`#pragma unroll 3` unrolls the 99-round loop 3×. At 7 rounds per BLAKE3 and 16 G-operations per round = 112 G-ops per BLAKE3, each unrolled iteration is 336 G-ops. Full unroll (99×) would be 11,088 G-ops — too large for instruction cache on most GPUs.

**To investigate:** Try `#pragma unroll 9` (99 = 11×9) or `#pragma unroll 11` (99 = 9×11). This doubles/triples the code size but may improve instruction throughput on high-end GPUs with large L1 instruction caches. Benchmark each on the target hardware — optimal value is GPU-specific.

### 8. CPU VDF Lane — SIMD Opportunity

`vdf_lane.rs` runs Genus-2 Jacobian VDF on CPU (Rust, `crates/q-vdf/`). This is inherently sequential per nonce (VDF design), but the 256-bit modular multiplications use scalar arithmetic. On x86-64 with AVX2/AVX-512, processing 4 or 8 field elements in parallel using SIMD lanes (interleaved nonce candidates) could multiply CPU VDF throughput by 4–8× while preserving the sequential-per-nonce property.

### 9. Challenge Polling in VDF Lane

`vdf_lane.rs` polls `/api/v1/mining/vdf-challenge` on a timer. If a new block arrives between polls, the VDF thread wastes up to the poll interval computing on a stale challenge.

**Fix:** Wire the VDF thread to the same `Arc<AtomicU64>` new-block signal that the BLAKE3 threads use. The signal is incremented immediately when the SSE stream delivers a new block, allowing the VDF thread to abort the current computation and restart within milliseconds.

### 10. Kernel Compilation Flags

Current: `-cl-mad-enable -cl-no-signed-zeros -cl-denorms-are-zero`

**Additional flags to test (benchmark each — driver-dependent):**
- `-cl-fast-relaxed-math` — enables approximate math; safe here since we use only integer ops
- `-cl-strict-aliasing` — allows more aggressive alias analysis
- NVIDIA-specific: `-nv-opt-level=3 -nv-maxrregcount=64`
- AMD-specific: `-O3 -amdgpu-function-calls=false`

---

## Benchmark Targets (per GPU, single RTX 3080)

| Metric | Current (estimated) | Target |
|--------|---------------------|--------|
| Hash rate | ~500 MH/s | 800+ MH/s |
| GPU utilization | ~95% | 99%+ |
| Dispatch time | 200–600ms adaptive | 300–500ms steady |
| Multi-GPU overhead | ~0% (parallel submit) | ~0% |
| Solution submission latency | 10–50ms | <5ms |

---

## Files to Focus On

| File | Purpose |
|------|---------|
| `crates/q-mining/src/gpu.rs` | Core OpenCL dispatch, kernel source, adaptive tuning |
| `crates/q-miner/src/gpu/opencl.rs` | Alternative OpenCL backend (double-buffered, `opencl-mining` feature) |
| `crates/q-miner/src/vdf_lane.rs` | CPU Genus-2 VDF thread |
| `crates/q-miner/src/main.rs` | Mining loop, challenge polling, solution submission |
| `crates/q-vdf/src/genus2_vdf.rs` | Genus-2 Jacobian field arithmetic (CPU VDF) |

---

## OpenCL Kernel Source Location

The full kernel is the `BLAKE3_KERNEL_SOURCE` constant in `crates/q-mining/src/gpu.rs` starting at line 61. It is a single-file OpenCL C program (~280 lines) compiled at runtime on first use and cached as a binary.
