# GPU Mining Technical Review -- Q-NarwhalKnight

**Date:** 2026-03-27 (updated)
**Version:** v10.1.8 (GPU optimization release — async dispatch, kernel tweaks, compilation caching)
**Scope:** All GPU mining code across `q-mining` and `q-miner` crates
**Purpose:** Reference document for AI systems and developers working on this codebase

---

## 1. Architecture Overview

GPU mining in Q-NarwhalKnight is split across two crates with distinct roles:

### q-mining (library crate) -- `crates/q-mining/`

The `q-mining` crate houses the **production GPU mining implementation** that is actually compiled into and used by the miner binary. Key file:

- **`src/gpu.rs`** (~1000 lines) -- Contains the `GPUMiner` struct, the inline OpenCL kernel source (`BLAKE3_KERNEL_SOURCE`), persistent buffer management, conditional upload, adaptive work sizing, kernel compilation caching, per-GPU calibration, multi-GPU dispatch, and the `mine_batch()` / `mine_batch_multi()` / `calibrate()` APIs. This is the **real, working** GPU mining code.

### q-miner (binary crate) -- `crates/q-miner/`

The `q-miner` crate contains the miner executable and has its own `src/gpu/` module directory with **higher-level abstractions and alternative backend implementations**. These are mostly architectural scaffolding that is NOT wired into the production mining loop:

- **`src/gpu/mod.rs`** (318 lines) -- Trait definitions (`GpuMiner`, `GpuMiningBackend`, `MiningKernel`), multi-GPU coordinator stubs, work distribution, solution collection. Defines the `MultiGpuCoordinator` that uses the trait-based backends.
- **`src/gpu/cuda.rs`** (400 lines) -- CUDA mining via `cudarc`. Feature-gated behind `cuda-mining`. Has a `CudaMiningKernel` that compiles PTX from `kernels/dag_knight_vdf.cu`. The mining loop (`cuda_mining_loop`) is a CPU-side BLAKE3 simulation, not actual GPU dispatch.
- **`src/gpu/opencl.rs`** (571 lines) -- OpenCL mining via `opencl3`. Feature-gated behind `opencl-mining`. Has its own `OpenClMiner` struct, kernel loading from `kernels/dag_knight_vdf.cl`, and a mining loop. This is a **different, independent** implementation from `q-mining::gpu`.
- **`src/gpu/multi_gpu.rs`** (560 lines) -- Multi-GPU device detection, work distribution, capacity-based load balancing. Well-structured but not connected to the production path.
- **`src/gpu/vulkan.rs`** (51 lines) -- Placeholder/stub. Returns immediately, hashes nothing.
- **`src/gpu/kernels/dag_knight_vdf.cl`** (258 lines) -- Standalone OpenCL kernel file used by q-miner's `OpenClMiner`.
- **`src/gpu/kernels/dag_knight_vdf.cu`** (216 lines) -- CUDA kernel file used by q-miner's `CudaMiningKernel`.

### The Critical Distinction

The **production mining path** in `q-miner/src/main.rs` (line ~1862) uses:

```rust
#[cfg(feature = "opencl-mining")]
// Uses q_mining::GPUMiner (from crates/q-mining/src/gpu.rs)
match q_mining::GPUMiner::new(q_mining::GPUMinerConfig::default()) {
    Ok(mut gpu_miner) => {
        // ... spawns spawn_blocking thread calling gpu_miner.mine_batch()
    }
}
```

This pulls `GPUMiner` from the **q-mining** crate, NOT from q-miner's own `src/gpu/` module. The q-miner GPU module (`mod.rs`, `cuda.rs`, `opencl.rs`, `multi_gpu.rs`) defines a parallel set of abstractions that are architecturally interesting but not on the production path.

### Dependency Graph

```
q-miner (binary)
  |
  +-- q-mining (library, optional, feature = "opencl-mining" activates "gpu-mining")
  |     |
  |     +-- opencl3 (conditional on feature "gpu-mining")
  |     +-- src/gpu.rs  <-- PRODUCTION GPU CODE
  |
  +-- src/gpu/mod.rs         <-- Unused in production
  +-- src/gpu/cuda.rs        <-- Unused in production (cudarc)
  +-- src/gpu/opencl.rs      <-- Unused in production (separate OpenCL impl)
  +-- src/gpu/multi_gpu.rs   <-- Unused in production
  +-- src/gpu/vulkan.rs      <-- Stub
```

---

## 2. Build System

### Feature Flags

In `q-miner/Cargo.toml`:

```toml
[features]
default = ["cpu-mining", "network", "cli", "jemalloc", "tui", "tor-support", "p2p"]

# Mining backends
cpu-mining = []
cuda-mining = ["cudarc", "half"]
opencl-mining = ["opencl3", "q-mining/gpu-mining"]   # <-- Activates production GPU code
vulkan-mining = ["vulkano", "vulkano-shaders"]
```

In `q-mining/Cargo.toml`:

```toml
[features]
default = ["mining-vrf"]
gpu-mining = ["opencl3"]
```

Key observations:

1. **GPU mining is NOT in the default feature set.** Users must explicitly compile with `--features opencl-mining` or `--features cuda-mining`.
2. The `opencl-mining` feature in q-miner activates two things: the local `opencl3` dependency AND `q-mining/gpu-mining` (the production code).
3. The `cuda-mining` feature pulls in `cudarc` and `half` but does NOT activate `q-mining/gpu-mining`. CUDA and OpenCL are independent feature paths.
4. The production mining loop in `main.rs` only checks `#[cfg(feature = "opencl-mining")]` -- CUDA devices go through a completely separate (and less mature) code path in q-miner's own GPU module.

### Conditional Compilation Strategy

The `q-mining/src/gpu.rs` file uses `#[cfg(feature = "gpu-mining")]` extensively:

```rust
#[cfg(feature = "gpu-mining")]
use opencl3::{ ... };

pub struct GPUMiner {
    #[cfg(feature = "gpu-mining")]
    contexts: Vec<GPUContext>,
    // v10.1.8: adaptive_work_size moved to GPUContext (per-GPU sizing)
}

#[cfg(feature = "gpu-mining")]
pub fn mine_batch(&mut self, ...) -> Result<BatchResult> { /* real implementation */ }

#[cfg(not(feature = "gpu-mining"))]
pub fn mine_batch(&mut self, ...) -> Result<BatchResult> {
    Err(anyhow!("GPU mining not available. Compile with --features gpu-mining"))
}
```

This pattern is clean -- the same struct compiles with or without GPU support, with stub methods returning errors when the feature is disabled.

### Dependencies

| Dependency | Version | Used By | Purpose |
|---|---|---|---|
| `opencl3` | 0.9 | q-mining (production) + q-miner (unused) | OpenCL 3.0 Rust bindings |
| `cudarc` | 0.10 | q-miner (cuda.rs, not production) | CUDA Runtime API bindings |
| `half` | 2.3 | q-miner (with cuda-mining) | Half-precision floats for GPU |
| `vulkano` | 0.34 | q-miner (vulkan.rs, stub) | Vulkan compute bindings |
| `blake3` | 1.5 | Both | BLAKE3 hash (CPU reference) |
| `sha3` | 0.10 | q-mining (hybrid_mining.rs) | SHA-3 for VDF challenge/PoW |

---

## 3. OpenCL Implementation (Production)

### File: `crates/q-mining/src/gpu.rs`

This is the most important GPU file in the project. It implements a complete BLAKE3 + VDF proof-of-work scheme with v10.1.7 performance optimizations.

### Algorithm

The mining algorithm performs 100 sequential BLAKE3 hashes per nonce candidate:

```
input  = challenge_hash[32 bytes] || nonce_le[8 bytes]   (40 bytes)
h      = BLAKE3(input)                                    (initial hash)
for _ in 0..99:
    h  = BLAKE3(h)                                        (VDF chain)
if h < difficulty_target:
    SOLUTION FOUND
```

This is a VDF-style proof: the 99 sequential re-hashing steps cannot be parallelized for a single nonce, ensuring a minimum time per candidate. GPU parallelism comes from evaluating millions of nonce candidates simultaneously, each performing its own 100-hash chain.

### Inline Kernel Source

The OpenCL kernel is embedded as a Rust string constant `BLAKE3_KERNEL_SOURCE` (approximately 255 lines of OpenCL C). This is the most carefully implemented GPU code in the project.

**BLAKE3 Constants:**

```c
__constant uint BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};
```

These are the standard BLAKE3 initialization vectors (same as BLAKE2s and SHA-256 fractional parts).

**Message Schedule:**

```c
__constant uchar MSG_SCHED[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    { 2, 6, 3,10, 7, 0, 4,13, 1,11,12, 5, 9,14,15, 8},
    // ... 7 permutations total
};
```

This implements the BLAKE3 message word permutation schedule for 7 rounds, which is correct per the BLAKE3 specification.

**G Mixing Macro:**

```c
#define G(s, a, b, c, d, mx, my) \
    s[a] = s[a] + s[b] + (mx);  \
    s[d] = rotr32(s[d] ^ s[a], 16u); \
    s[c] = s[c] + s[d];         \
    s[b] = rotr32(s[b] ^ s[c], 12u); \
    s[a] = s[a] + s[b] + (my);  \
    s[d] = rotr32(s[d] ^ s[a], 8u);  \
    s[c] = s[c] + s[d];         \
    s[b] = rotr32(s[b] ^ s[c], 7u);
```

The rotation amounts (16, 12, 8, 7) match the BLAKE3 spec exactly. The macro is inlined for GPU register optimization.

**Compression Function:**

The `blake3_compress()` function implements single-block BLAKE3 compression with proper state initialization:
- Upper half: chaining value (key or IV)
- Lower half: IV constants, counter, block length, and flags
- 7 rounds with message schedule permutation
- Output: XOR of upper and lower halves

**Flags Usage:**

```c
#define CHUNK_START 1u
#define CHUNK_END   2u
#define ROOT        8u
```

Both `blake3_hash_40` and `blake3_hash_32` use flags `CHUNK_START | CHUNK_END | ROOT` (value 11). This is correct for single-block, single-chunk hashing -- the entire input fits in one BLAKE3 block.

**v10.1.7 Change -- Challenge Word Precompute (Phase 2A):**

The kernel signature changed from `__global const uchar* challenge` (raw bytes) to `__constant uint* challenge_words` (pre-converted u32 words in constant address space). The CPU converts the 32-byte challenge to 8 little-endian u32 words once, eliminating the per-work-item byte-to-uint conversion loop that previously ran inside `blake3_hash_40()`:

```c
// OLD (pre-v10.1.7): Per-work-item byte unpacking (8 iterations per thread)
for (int i = 0; i < 8; i++) {
    block[i] = (uint)challenge[i*4]
             | ((uint)challenge[i*4+1] << 8)
             | ((uint)challenge[i*4+2] << 16)
             | ((uint)challenge[i*4+3] << 24);
}

// NEW (v10.1.8): Direct copy from __constant pre-converted words
block[0] = challenge_words[0]; block[1] = challenge_words[1];
// ... (8 direct assignments)
```

This saves 8 global memory reads + 24 shift-OR operations per work item. With 1M+ work items, the aggregate saving is significant.

**v10.1.8 Changes -- Kernel-Level Optimizations (GPU-003):**

1. **`__constant` address space for challenge buffer**: The challenge parameter uses `__constant uint*` instead of `__global const uint*`. This leverages the GPU's hardware constant cache (typically 64KB per CU), which is optimized for broadcast reads where all work items read the same address. Since every work item reads the same 32 bytes of challenge data, this is a perfect use case.

2. **`reqd_work_group_size(256,1,1)`**: The `__attribute__((reqd_work_group_size(256,1,1)))` on the main `blake3_mine` kernel tells the OpenCL compiler the exact work-group size at compile time, enabling better register allocation and occupancy calculations.

3. **`#pragma unroll 3`**: The 99-round VDF loop uses partial unrolling (99 = 33×3). This reduces loop overhead by 2/3 while keeping code size manageable. Full unrolling of 99 iterations would bloat the kernel binary excessively.

4. **IV private copy**: BLAKE3_IV is `__constant` but NVIDIA's OpenCL compiler requires matching address spaces for function parameters. Both `blake3_hash_40()` and `blake3_hash_32()` copy IV into private `uint iv[8]` arrays before passing to `blake3_compress()`.

```c
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void blake3_mine(
    __constant uint* challenge_words,  // hardware constant cache
    __global const uchar* target,
    const ulong nonce_start,
    __global ulong* found_nonce,
    __global uchar* found_hash,
    __global uint* found_flag
) {
    // ...
    #pragma unroll 3
    for (int vdf = 0; vdf < 99; vdf++) {
        blake3_hash_32(h, tmp);
        // ...
    }
}
```

**Target Comparison:**

```c
bool meets_target(const uint hash[8], __global const uchar* target) {
    for (int i = 0; i < 8; i++) {
        uint h = hash[i];
        for (int j = 0; j < 4; j++) {
            uchar hb = (uchar)((h >> (j * 8)) & 0xFFu);
            uchar tb = target[i * 4 + j];
            if (hb < tb) return true;
            if (hb > tb) return false;
        }
    }
    return true;
}
```

This compares hash bytes in little-endian word order (byte 0 of word 0 first). This must match the server-side validation exactly.

**Atomic Solution Claiming:**

```c
uint old = atomic_cmpxchg(found_flag, 0u, 1u);
if (old == 0u) {
    *found_nonce = nonce;
    // write hash bytes...
}
```

Uses `atomic_cmpxchg` (CAS) so that only the first work item to find a solution writes it. This is correct -- subsequent finders are silently dropped. The hash is then converted from uint[8] to byte[32] in little-endian order.

### Buffer Management (v10.1.7: Persistent Buffers)

**Before v10.1.7**, `dispatch_blake3_kernel` allocated 5 OpenCL buffers per dispatch and freed them at function exit. Each `Buffer::create()` call is a synchronous IPC round-trip to the GPU driver (5-50 microseconds each). At high dispatch rates this was a measurable bottleneck.

**v10.1.7 (Phase 1: Buffer Reuse)** moved all 5 buffers into the `GPUContext` struct, allocated once during `initialize_contexts()`:

```rust
#[cfg(feature = "gpu-mining")]
struct GPUContext {
    device: Device,
    context: Context,
    queue: CommandQueue,
    program: Program,
    kernel: Kernel,
    // Pre-allocated persistent buffers (Phase 1: buffer reuse)
    challenge_buf: Buffer<cl_uint>,    // 8 × u32, READ_ONLY  (Phase 2A: words)
    target_buf: Buffer<cl_uchar>,      // 32 bytes, READ_ONLY
    found_nonce_buf: Buffer<cl_ulong>, // 1 × u64, WRITE_ONLY
    found_hash_buf: Buffer<cl_uchar>,  // 32 bytes, WRITE_ONLY
    found_flag_buf: Buffer<cl_uint>,   // 1 × u32, READ_WRITE
    // Track what's currently uploaded to skip redundant transfers
    cached_challenge: [u8; 32],
    cached_target: [u8; 32],
}
```

**Conditional Upload:** Challenge and target are only re-uploaded when they change. Within a single block (same challenge), only the 4-byte `found_flag` zero-write is needed per dispatch. On new block: 3 writes (challenge words, target, flag). This eliminates 2 of 3 buffer writes in the steady-state case.

```rust
// Phase 1+2: Conditional upload — only re-upload when challenge/target changes
// GPU-001 (v10.1.8): CL_FALSE (non-blocking) — in-order queue guarantees writes
// complete before kernel executes. CPU doesn't block waiting for transfer.
if ctx.cached_challenge != *challenge {
    let challenge_words = challenge_bytes_to_words(challenge);
    ctx.queue.enqueue_write_buffer(&mut ctx.challenge_buf, CL_FALSE, 0, &challenge_words, &[])?;
    ctx.cached_challenge = *challenge;
}
if ctx.cached_target != *target {
    ctx.queue.enqueue_write_buffer(&mut ctx.target_buf, CL_FALSE, 0, target, &[])?;
    ctx.cached_target = *target;
}
// found_flag must be zeroed before every dispatch
ctx.queue.enqueue_write_buffer(&mut ctx.found_flag_buf, CL_FALSE, 0, &[0u32], &[])?;
```

**v10.1.8 (GPU-001): Non-blocking buffer writes** -- All three `enqueue_write_buffer` calls changed from `CL_TRUE` (blocking) to `CL_FALSE` (non-blocking). On an in-order command queue, the OpenCL driver guarantees commands execute in submission order, so the kernel launch is implicitly serialized after the writes. The CPU no longer blocks waiting for the GPU to acknowledge the transfer — it immediately proceeds to set kernel args and enqueue the kernel. Readback after `queue.finish()` remains `CL_TRUE` since the data is needed immediately for the if-check.

### Dispatch Logic

```rust
let global_work_size = [work_size];
let local_work_size = [self.config.local_work_size];  // Default: 256

ctx.queue.enqueue_nd_range_kernel(
    ctx.kernel.get(), 1,
    std::ptr::null(),
    global_work_size.as_ptr(),
    local_work_size.as_ptr(),
    &[],
)?;
ctx.queue.finish()?;  // Synchronous wait
```

The `queue.finish()` call blocks the thread until the GPU completes. This is acceptable because the GPU mining runs in a `spawn_blocking` thread (not on the tokio async executor).

### Adaptive Work Size (v10.1.7 Phase 4 + v10.1.8 GPU-002: Per-GPU)

Instead of a fixed 2^20 work size, v10.1.7 introduced adaptive sizing. **v10.1.8 (GPU-002)** moved `adaptive_work_size` from the shared `GPUMiner` struct into each `GPUContext`, enabling independent per-GPU tuning:

```rust
const MIN_WORK_SIZE: usize = 1 << 16;  // 65K (floor)
const MAX_WORK_SIZE: usize = 1 << 23;  // 8M (cap)
const DISPATCH_TARGET_LOW_MS: u128 = 100;
const DISPATCH_TARGET_HIGH_MS: u128 = 400;

// v10.1.8: Per-GPU adaptive tuning (in mine_batch and mine_batch_multi)
let ctx = &mut self.contexts[gpu_idx];
if dispatch_ms < 100 { ctx.adaptive_work_size = (ctx.adaptive_work_size * 3 / 2).min(MAX_WORK_SIZE); }
if dispatch_ms > 400 { ctx.adaptive_work_size = (ctx.adaptive_work_size * 2 / 3).max(MIN_WORK_SIZE); }
```

This allows fast GPUs (RTX 4090) to ramp up to 8M work items while slow GPUs auto-reduce to avoid timeouts. The work size is always rounded down to a multiple of `local_work_size` (256). In a mixed GPU setup (e.g., RTX 4090 + GTX 1660), each GPU converges to its own optimal work size independently.

### Startup Calibration (v10.1.8: GPU-002)

The `calibrate()` method benchmarks 4 work sizes (64K, 256K, 1M, 4M) per GPU and picks the highest throughput as the starting point. This eliminates the cold-start ramp-up period where the adaptive algorithm starts at 1M and takes several iterations to converge:

```rust
pub fn calibrate(&mut self) -> Result<()> {
    let test_sizes: [usize; 4] = [1 << 16, 1 << 18, 1 << 20, 1 << 22];
    let dummy_challenge = [0u8; 32];
    let dummy_target = [0xFFu8; 32]; // easy target — no solution expected

    for (gpu_idx, ctx) in self.contexts.iter_mut().enumerate() {
        // Benchmark each size, measure throughput = items / elapsed
        // Pick the size with highest throughput
        ctx.adaptive_work_size = best_size;
    }
    Ok(())
}
```

The `current_work_size()` method exposes the current adaptive value for diagnostics/TUI display.

### Multi-GPU Support (v10.1.7: Phase 3)

v10.1.7 added `mine_batch_multi()` which dispatches across all initialized GPU contexts:

```rust
pub fn mine_batch_multi(&mut self, challenge_hash: &[u8; 32], target: &[u8; 32], nonce_start: u64) -> Result<BatchResult>
```

**Nonce partitioning:** Work is split evenly across GPUs. GPU `i` of `N` gets `nonce_start + i * (work_size / N)`. The last GPU gets any remainder.

**Sequential dispatch:** Each GPU context is dispatched sequentially (due to `&mut GPUContext` requirement from persistent buffers). For most users with 1-2 GPUs this is fine. If only 1 GPU is present, `mine_batch_multi()` falls back to `mine_batch()`.

**v10.1.8 (GPU-002): Per-GPU timing** -- `mine_batch_multi()` was rewritten to use per-GPU timing and per-GPU adaptive sizing. Each GPU's dispatch is timed individually and its `ctx.adaptive_work_size` is adjusted based on that GPU's performance, not an average.

**Nonce partitioning (v10.1.8):** Nonces are assigned sequentially across GPUs based on each GPU's individual work size. GPU 0 starts at `nonce_start`, GPU 1 at `nonce_start + gpu0_work`, etc. This allows faster GPUs to process more nonces per batch.

**First-solution-wins:** After all dispatches complete, the first GPU to find a solution gets priority. The solution includes `gpu_index` for diagnostics.

**Note:** The production mining loop in `main.rs` currently calls `mine_batch()` (single-GPU). Switching to `mine_batch_multi()` is a one-line change when multi-GPU users request it.

---

## 4. Mining Loop Optimizations (v10.1.7)

### File: `crates/q-miner/src/main.rs` (lines ~1906-2008)

The v10.1.7 mining loop has three key improvements over the prior version:

### 4.1 Cached Challenge Decode

**Before:** Every dispatch iteration called `hex_to_bytes(&chal.challenge_hash)` and `hex_to_bytes(&chal.difficulty_target)`, each allocating a `Vec<u8>` via `hex::decode()`. At ~5 dispatches/second, this was ~10 heap allocations/second for data that changes once per block (~1 second).

**After:** Challenge and target are decoded once per block signal change and stored in stack-local `[u8; 32]` arrays:

```rust
let mut cached_challenge: [u8; 32] = [0u8; 32];
let mut cached_target: [u8; 32] = [0u8; 32];
let mut cached_block_height: u64 = 0;
let mut cached_vdf_iterations: u32 = 0;
let mut challenge_ready = false;

while gpu_is_running.load(Ordering::Relaxed) {
    let sig = gpu_new_block_signal.load(Ordering::Relaxed);
    if sig != last_signal || !challenge_ready {
        // Re-decode only on new block
        last_signal = sig;
        gpu_nonce = u64::MAX / 2;
        // ... decode and cache ...
    }
    // Dispatch with cached values
    gpu_miner.mine_batch(&cached_challenge, &cached_target, gpu_nonce);
}
```

### 4.2 Block Signal at Loop Top

**Before:** The block-signal check was at the BOTTOM of the loop (after dispatch). This meant a new block was detected one full dispatch cycle late (~200ms wasted hashing stale work).

**After:** The block-signal check is at the TOP of the loop. Combined with the cached challenge, this means:
1. New block signal detected immediately at next iteration start
2. Nonce reset happens before the next dispatch
3. Challenge re-decoded only when signal changes

### 4.3 Mutability Change

`mine_batch()` changed from `&self` to `&mut self` to support persistent buffer mutation and adaptive work size updates. The `GPUMiner` is owned by the `spawn_blocking` closure (not shared across threads), so this is safe.

```rust
Ok(mut gpu_miner) => {
    // gpu_miner is moved into spawn_blocking, &mut self is valid
    tokio::task::spawn_blocking(move || {
        gpu_miner.mine_batch(...);  // &mut self
    })
}
```

---

## 5. CUDA Implementation

### File: `crates/q-miner/src/gpu/cuda.rs`

The CUDA backend uses `cudarc` for device management and NVRTC for PTX compilation. **This is NOT on the production path.**

**Critical Issue: The CUDA mining loop (`cuda_mining_loop`) does NOT actually use the GPU kernel.** It calls `compute_dag_knight_hash()` which runs BLAKE3 on the CPU, and uses 1000 VDF rounds instead of 99.

**Status:** Not production-ready. Would need to be rewritten to match the BLAKE3+VDF algorithm from `gpu.rs`.

---

## 6. Multi-GPU Support (q-miner module, unused)

### File: `crates/q-miner/src/gpu/multi_gpu.rs`

The `MultiGPUMiner` provides device detection, load balancing strategies (Equal, CapacityBased, Dynamic, Manual), and work distribution. It has test coverage but is NOT connected to the production mining path.

**Note:** The production `GPUMiner` in `q-mining/src/gpu.rs` now has its own built-in multi-GPU support via `mine_batch_multi()` (v10.1.7), which is simpler and directly integrated.

---

## 7. Hybrid Mining

### File: `crates/q-mining/src/hybrid_mining.rs`

The hybrid mining system is an architectural design for splitting block rewards 50/50 between CPU miners (VDF proofs) and GPU miners (SHA-3 PoW hashes). It references `GPUMiner` under `#[cfg(feature = "gpu-mining")]` but uses it with `GPUMiningJob` which feeds it the same BLAKE3 mining.

**v10.1.7 change:** `Arc<GPUMiner>` changed to `Arc<tokio::sync::Mutex<GPUMiner>>` to accommodate the `&mut self` requirement on `mine_batch()`.

**Status:** Design-only framework, not used in production.

---

## 8. Performance Characteristics

### Production OpenCL (q-mining/src/gpu.rs)

| Parameter | Value (v10.1.8) | Notes |
|---|---|---|
| Initial work size | Calibrated per GPU | `calibrate()` benchmarks 4 sizes (v10.1.8) |
| Min work size | 65,536 (2^16) | Floor for adaptive sizing |
| Max work size | 8,388,608 (2^23) | Cap for adaptive sizing |
| Local work size | 256 | Fixed via `reqd_work_group_size` (v10.1.8) |
| Hashes per work item | 100 | 1 initial BLAKE3 + 99 VDF rounds |
| Dispatch target | 100-400ms | Adaptive work size window (per-GPU) |
| Buffer allocs per dispatch | 0 | Persistent buffers (v10.1.7) |
| Buffer writes blocking | No | `CL_FALSE` non-blocking (v10.1.8) |
| Challenge uploads per block | 1 | Conditional upload (v10.1.7) |
| Challenge address space | `__constant` | Hardware constant cache (v10.1.8) |
| Kernel startup (cached) | <100ms | Compilation caching (v10.1.8) |
| Kernel startup (first run) | 2-10s | Compiled from source, then cached |
| Nonce start (GPU) | u64::MAX / 2 | Avoids CPU nonce collision |

**Estimated Throughput:**

Each work item performs 100 BLAKE3 single-block compressions. BLAKE3 compression involves 7 rounds of the G mixing function applied to a 16-word state. On a modern GPU (e.g., RTX 3080 with 8704 CUDA cores at ~1.7 GHz), rough estimates:

- BLAKE3 single-block compression: ~100-200 ns per work item (GPU-optimized)
- 100 sequential compressions per work item: ~10-20 us
- With 1M work items and 68 SMs (RTX 3080): batch time ~150-300 ms
- Effective hash rate: 1M nonces / 0.2s = ~5 MH/s (nonce candidates per second)
- Effective BLAKE3 hash rate: ~500 MH/s (individual BLAKE3 operations)
- With adaptive sizing ramping to 8M work items: up to ~40 MH/s on fast GPUs

### Optimization Impact Summary (v10.1.7 + v10.1.8)

| Optimization | Version | Mechanism | Estimated Impact |
|---|---|---|---|
| Persistent buffers | v10.1.7 | Allocate 5 buffers once, reuse | -25-250µs/dispatch overhead |
| Conditional upload | v10.1.7 | Skip challenge/target re-upload within same block | -10-100µs/dispatch (steady state) |
| Challenge word precompute | v10.1.7 | CPU converts [u8;32]→[u32;8], kernel skips byte unpack | ~2-5% on first hash |
| Adaptive work size | v10.1.7 | Auto-tune dispatch size [65K, 8M] based on latency | Up to 8x more work per dispatch |
| Multi-GPU | v10.1.7 | Split nonce range across all GPU contexts | Near-linear scaling with GPU count |
| Cached challenge decode | v10.1.7 | Decode hex once per block, not per dispatch | Eliminates ~10 heap allocs/second |
| Block signal at loop top | v10.1.7 | Detect new blocks immediately, not after dispatch | Saves ~200ms of stale work per block |
| Non-blocking writes | v10.1.8 | `CL_FALSE` for all buffer uploads | -5-15µs/dispatch (CPU not blocked) |
| Per-GPU adaptive sizing | v10.1.8 | Each GPU tunes independently | +0-10% multi-GPU throughput |
| Startup calibration | v10.1.8 | Benchmark 4 sizes, pick optimal start | Instant ramp-up (no cold-start) |
| `__constant` challenge | v10.1.8 | Hardware constant cache for broadcast reads | ~5-10% kernel throughput |
| Loop unrolling | v10.1.8 | `#pragma unroll 3` on 99-round VDF loop | Reduced loop overhead |
| Work-group size hint | v10.1.8 | `reqd_work_group_size(256,1,1)` | Better register allocation |
| Kernel compilation cache | v10.1.8 | Save binary to `~/.config/q-miner/kernel-cache/` | Startup: 2-10s → <100ms |

---

## 9. Runtime Requirements

### OpenCL Mining (Production Path)

To compile and run with GPU mining:

```bash
cargo build --release --package q-miner --features opencl-mining
```

**User Requirements:**
- **OpenCL ICD loader**: `libOpenCL.so` (Linux) or OpenCL.dll (Windows)
  - Linux: `apt install ocl-icd-libopencl1` (runtime) or `ocl-icd-opencl-dev` (build)
- **GPU Driver with OpenCL 1.2+ support**:
  - NVIDIA: Driver 470+ (comes with OpenCL ICD)
  - AMD: ROCm or AMDGPU-PRO driver
  - Intel: Intel OpenCL runtime or NEO driver
- **No CUDA toolkit needed** for the OpenCL path

### Pre-built Binary (v10.1.8)

A portable GPU miner binary is available, built on Debian 12 (GLIBC 2.34) inside Docker:

```bash
wget https://quillon.xyz/downloads/q-miner-gpu-v10.1.8 && chmod +x q-miner-gpu-v10.1.8
# or generic name:
wget https://quillon.xyz/downloads/q-miner-gpu-linux-x64 && chmod +x q-miner-gpu-linux-x64
```

**Compatibility:**
- GLIBC requirement: 2.34 (compatible with Debian 12+, Ubuntu 22.04+)
- Links `libOpenCL.so.1` dynamically (users must have OpenCL runtime installed)
- Binary size: ~23MB

**Docker Build Command (reproducible):**

```bash
docker run --rm \
  -v $(pwd):/src \
  -v /path/to/target-cache:/src/target \
  -w /src \
  rust:bookworm \
  bash -c '
    apt-get update -qq
    apt-get install -y -qq libssl-dev pkg-config cmake clang libudev-dev \
      libclang-dev ocl-icd-opencl-dev opencl-headers
    cargo build --release --package q-miner --features opencl-mining
  '
```

### CUDA Mining (q-miner path, not production)

```bash
cargo build --release --package q-miner --features cuda-mining
```

Requires NVIDIA GPU, CUDA Toolkit, and driver 525+. Not production-ready.

---

## 10. Known Issues and Gaps

### Resolved in v10.1.7

1. **~~Buffer Allocation Per Dispatch~~** -- FIXED. Buffers are now persistent in `GPUContext`, allocated once during `initialize_contexts()`.

2. **~~Single-GPU Only in Production~~** -- PARTIALLY FIXED. `mine_batch_multi()` added. The production loop still calls `mine_batch()` (single-GPU) but switching is a one-line change.

3. **~~Fixed Work Size~~** -- FIXED. Adaptive work size auto-tunes between 65K and 8M based on dispatch latency.

4. **~~Per-Dispatch Hex Decode~~** -- FIXED. Mining loop now caches decoded challenge/target, only re-decoding on new block signal.

5. **~~Late Block Detection~~** -- FIXED. Block signal check moved from bottom to top of mining loop.

### Remaining Issues

6. **Algorithm Mismatch Between Implementations**

   There are at least 3 different mining algorithms across the codebase:

   | Location | Algorithm | VDF Rounds | Hash Function |
   |---|---|---|---|
   | `q-mining/src/gpu.rs` (PRODUCTION) | BLAKE3 + 99 VDF | 100 total | BLAKE3 |
   | `q-miner/src/gpu/cuda.rs` (CPU fallback) | BLAKE3 + 1000 VDF | 1001 total | BLAKE3 |
   | `q-miner/src/gpu/kernels/dag_knight_vdf.cu` | BLAKE3-like + variable VDF | 1000-2000 | Simplified BLAKE3 |
   | `q-miner/src/gpu/kernels/dag_knight_vdf.cl` | BLAKE3-like + 1024 VDF | 1024 + 8 quantum | Simplified BLAKE3 |
   | `q-mining/src/hybrid_mining.rs` | SHA-3 PoW | N/A | SHA3-256 |

   Only `q-mining/src/gpu.rs` matches the server-side validation. All others would produce invalid solutions.

7. **CUDA Kernel Does Not Execute on GPU** -- The `cuda_mining_loop()` runs BLAKE3 on the CPU despite loading PTX.

8. **Non-Compliant BLAKE3 in Kernel Files** -- The `.cl` and `.cu` files in `q-miner/src/gpu/kernels/` implement incorrect BLAKE3 variants that would not produce valid solutions.

9. **Temperature/Power Monitoring is Placeholder** -- `GPUMiningStats` has `temperature` and `power_draw` fields but they are never populated with real values.

10. **No Error Recovery for GPU Context** -- If a GPU dispatch fails, the mining loop logs the error and sleeps 5 seconds, then retries with the same context. There is no context re-initialization or device health checking.

11. **Sequential Multi-GPU Dispatch** -- `mine_batch_multi()` dispatches to GPUs sequentially (due to `&mut GPUContext`). For 2+ GPUs, parallel dispatch via threads would improve throughput. This is acceptable for 1-2 GPU setups but suboptimal for mining rigs with 4+ GPUs. **v10.1.8 improved this** with per-GPU timing and per-GPU adaptive sizing, but dispatch is still sequential.

12. **~~No Kernel Compilation Caching~~** -- FIXED in v10.1.8 (GPU-004). Compiled OpenCL binary is cached to `~/.config/q-miner/kernel-cache/{hash}.clbin`. Cache key = hash(source + device_name + driver_version). Subsequent startups load with `create_program_with_binary()` (<100ms vs 2-10s).

### Minor Issues

13. **Dead Code Warnings Suppressed** -- `#[allow(dead_code)]` on `GPUContext.device` and `GPUContext.program` fields.

14. **Inconsistent OpenCL API Usage** -- `q-mining/src/gpu.rs` uses `enqueue_nd_range_kernel` with raw pointers, while `q-miner/src/gpu/opencl.rs` uses the higher-level `ExecuteKernel` builder pattern.

15. **The `intensity` Config Field is Unused** -- `GPUMinerConfig.intensity` (default 80) is stored but never consulted.

---

## 11. Recommendations

### High Priority

1. **Unify GPU Mining Implementations** -- Remove or clearly mark as "example/reference" the q-miner GPU module (`cuda.rs`, `opencl.rs`, kernel files). The production code in `q-mining/src/gpu.rs` is the only correct implementation.

2. **Wire `mine_batch_multi()` into Production Loop** -- One-line change in `main.rs` for users with multiple GPUs.

3. **~~GPU Kernel Compilation Caching~~** -- DONE (v10.1.8, GPU-004). Saves compiled binary to `~/.config/q-miner/kernel-cache/`. Startup savings: 2-10s → <100ms.

### Medium Priority

4. **Parallel Multi-GPU Dispatch** -- Use `std::thread::scope` or per-GPU worker threads with `unsafe impl Send for GPUContext` (valid per OpenCL spec §3.3) to dispatch to multiple GPUs simultaneously.

5. **Add GPU Kernel Correctness Test** -- Dispatch a known input to the GPU, read back the result, compare against CPU reference. Currently `test_blake3_vdf_reference` only tests CPU determinism.

6. **Real Temperature/Power Monitoring** -- NVIDIA via NVML, AMD via ROCm SMI. Important for thermal throttling protection and dashboard stats.

7. **~~Non-Blocking Transfers~~** -- DONE (v10.1.8, GPU-001). All 3 `enqueue_write_buffer` calls use `CL_FALSE`. In-order queue guarantees correctness without explicit event dependencies.

### Low Priority

8. **CUDA Backend Rewrite** -- If CUDA support is desired, rewrite `cuda.rs` to use the same BLAKE3+VDF algorithm with proper GPU kernel dispatch.

9. **Vulkan Backend** -- Expand the stub using `vulkano-shaders` for SPIR-V compute. Useful for systems with Vulkan but not OpenCL.

10. **Stratum Protocol** -- The `pool-mining` feature flag exists but has no implementation. Important for GPU miners connecting to pools.

---

## Appendix A: File Reference

| File | Lines | Status | Purpose |
|---|---|---|---|
| `crates/q-mining/src/gpu.rs` | ~1000 | **PRODUCTION** | OpenCL BLAKE3+VDF kernel + GPUMiner (v10.1.8 optimized) |
| `crates/q-mining/src/hybrid_mining.rs` | 870 | Design only | CPU VDF + GPU SHA-3 hybrid framework |
| `crates/q-mining/src/lib.rs` | ~520 | Active | Crate root, re-exports GPU types |
| `crates/q-mining/Cargo.toml` | 77 | Active | Feature flag: `gpu-mining = ["opencl3"]` |
| `crates/q-miner/src/gpu/mod.rs` | 318 | Unused | Trait definitions, coordinator stubs |
| `crates/q-miner/src/gpu/cuda.rs` | 400 | Unused/Broken | CUDA backend (CPU fallback loop) |
| `crates/q-miner/src/gpu/opencl.rs` | 571 | Unused | Alternative OpenCL backend |
| `crates/q-miner/src/gpu/multi_gpu.rs` | 560 | Unused | Multi-GPU detection and load balancing |
| `crates/q-miner/src/gpu/vulkan.rs` | 51 | Stub | Placeholder |
| `crates/q-miner/src/gpu/kernels/dag_knight_vdf.cl` | 258 | Unused/Wrong algo | OpenCL kernel (non-compliant BLAKE3) |
| `crates/q-miner/src/gpu/kernels/dag_knight_vdf.cu` | 216 | Unused/Wrong algo | CUDA kernel (non-compliant BLAKE3) |
| `crates/q-miner/Cargo.toml` | 207 | Active | Feature flags: cuda/opencl/vulkan-mining |
| `crates/q-miner/src/main.rs` | ~2030+ | Active | Production GPU mining loop |
| `crates/q-miner/src/shared_state.rs` | ~375 | Active | `GpuDeviceSnapshot` for TUI display |

## Appendix B: Production Mining Flow (v10.1.8)

```
main.rs:~1862  #[cfg(feature = "opencl-mining")]
    |
    v
q_mining::GPUMiner::new(GPUMinerConfig::default())
    |
    +-- enumerate_devices()          -- scan OpenCL platforms for GPUs
    +-- initialize_contexts()        -- for each GPU:
    |     |                             1. Try load_cached_kernel() from disk
    |     |                             2. If miss: compile from source, save_kernel_cache()
    |     |                             3. Allocate 5 PERSISTENT buffers
    |     +-- (optional) calibrate() -- benchmark 4 work sizes per GPU
    |
    v
tokio::task::spawn_blocking(move || { ... })
    |
    v
LOOP:
    |
    +-- Check new_block_signal (TOP of loop — immediate detection)
    |     If changed: reset nonce to u64::MAX/2, re-decode challenge/target
    |     If unchanged: skip decode, use cached values
    |
    +-- gpu_miner.mine_batch(&cached_challenge, &cached_target, gpu_nonce)
    |     |
    |     +-- dispatch_blake3_kernel(&mut ctx, challenge, target, nonce, work_size)
    |     |     |
    |     |     +-- Conditional upload: challenge words (only if changed)  [CL_FALSE]
    |     |     +-- Conditional upload: target (only if changed)           [CL_FALSE]
    |     |     +-- Always: zero found_flag (4 bytes)                      [CL_FALSE]
    |     |     +-- Set kernel args (0-5) using persistent buffer refs
    |     |     +-- enqueue_nd_range_kernel(global=ctx.adaptive_work_size, local=256)
    |     |     +-- queue.finish()  [BLOCKING — waits for kernel + all queued writes]
    |     |     +-- Read found_flag  [CL_TRUE — need result for if-check]
    |     |     +-- If flag!=0: read found_nonce, found_hash
    |     |     +-- Return Option<(u64, [u8; 32])>
    |     |
    |     +-- Per-GPU adaptive sizing: measure dispatch_ms, adjust ctx.adaptive_work_size
    |     +-- Update stats (dispatches, total_hashes)
    |     +-- Return BatchResult { solution, hashes }
    |
    +-- Update nonce: gpu_nonce += result.hashes
    +-- Update hashrate (every 1s)
    +-- If solution found:
    |     +-- Build solution JSON
    |     +-- Build P2PMiningSubmission
    |     +-- Send to solution_submit_tx channel
    |
    +-- GOTO LOOP
```

## Appendix C: OpenCL Kernel Correctness Checklist

The inline kernel in `q-mining/src/gpu.rs` (`BLAKE3_KERNEL_SOURCE`):

- [x] BLAKE3 IV constants match specification
- [x] Message schedule permutation table (7 rounds) matches specification
- [x] G function rotation amounts: 16, 12, 8, 7 (correct)
- [x] Compression function state layout: CV[8] + IV[4] + counter[2] + block_len + flags
- [x] Output extraction: XOR upper and lower state halves
- [x] 40-byte hash: correct flag combination (CHUNK_START | CHUNK_END | ROOT)
- [x] 32-byte hash: correct flag combination (CHUNK_START | CHUNK_END | ROOT)
- [x] Nonce encoding: little-endian u64 at bytes 32-39
- [x] VDF chain: 99 rounds of BLAKE3(h) after initial hash = 100 total
- [x] Target comparison: byte-wise, little-endian word order
- [x] Atomic solution claiming via CAS
- [x] First-writer-wins semantics (subsequent solutions dropped)
- [x] Challenge accepted as pre-converted uint[8] words (v10.1.7)
- [x] Challenge uses `__constant` address space for constant cache (v10.1.8)
- [x] `reqd_work_group_size(256,1,1)` attribute on main kernel (v10.1.8)
- [x] `#pragma unroll 3` on 99-round VDF loop (v10.1.8)
- [x] IV copied from `__constant` to private for NVIDIA compat (v10.1.8)
- [ ] **NOT VERIFIED**: GPU output matches CPU reference (no GPU test exists)
- [ ] **NOT VERIFIED**: Endianness correctness across GPU architectures (AMD vs NVIDIA)

## Appendix D: v10.1.8 Optimization Stack

```
┌──────────────────────────────────────────────────────────────────┐
│               v10.1.8 GPU Optimization Stack                     │
│           (builds on v10.1.7 persistent buffers)                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Mining Loop (main.rs)                                           │
│  ├─ Cached challenge decode (1x per block, not 1x per dispatch) │
│  ├─ Block signal at top of loop (immediate new-block detection)  │
│  └─ Mutable GPUMiner (&mut self for persistent state)           │
│                                                                  │
│  GPUMiner (gpu.rs)                                               │
│  ├─ Persistent buffers (allocate once, reuse across dispatches)  │
│  ├─ Conditional upload (skip if challenge/target unchanged)      │
│  ├─ Non-blocking writes (CL_FALSE on in-order queue) [v10.1.8] │
│  ├─ Challenge word precompute (CPU [u8;32]→[u32;8])             │
│  ├─ Per-GPU adaptive work size (each GPU tunes independently)   │
│  │   [v10.1.8 — moved from GPUMiner to GPUContext]              │
│  ├─ Startup calibration (benchmark 4 sizes per GPU) [v10.1.8]  │
│  ├─ Multi-GPU dispatch (mine_batch_multi, per-GPU timing)        │
│  └─ Kernel compilation caching (~/.config/q-miner/) [v10.1.8]  │
│                                                                  │
│  OpenCL Kernel (BLAKE3_KERNEL_SOURCE)                            │
│  ├─ __constant challenge buffer (hardware constant cache)        │
│  ├─ reqd_work_group_size(256,1,1) (register optimization)       │
│  ├─ #pragma unroll 3 on VDF loop (99 = 33×3)                    │
│  └─ IV private copy for NVIDIA compat                            │
│                                                                  │
│  Portable Build                                                  │
│  ├─ Docker build on Debian 12 (GLIBC 2.34)                      │
│  ├─ Compatible: Debian 12+, Ubuntu 22.04+                        │
│  ├─ Links libOpenCL.so.1 (users need GPU driver + ICD)           │
│  └─ ~23MB binary, published to quillon.xyz/downloads             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
