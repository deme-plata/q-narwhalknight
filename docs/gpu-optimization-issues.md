# GPU Mining Optimization — Issue Tracker

**Project:** Q-NarwhalKnight GPU Miner (`crates/q-mining/src/gpu.rs`)
**Base version:** v10.1.7 (persistent buffers, conditional upload, adaptive work size)
**Target version:** v10.1.8 (async dispatch, kernel tweaks, compilation caching)
**Branch:** `gpu/v10.1.8-optimizations`

---

## Open Issues

### GPU-001: Non-blocking flag zeroing + async readback
- **Priority:** HIGH
- **Branch:** `gpu/phase-1-async-dispatch`
- **Status:** 🔵 In Progress
- **Impact:** -5-15µs per dispatch overhead
- **Description:** `dispatch_blake3_kernel()` uses `CL_TRUE` (blocking) for the `found_flag` zero-write. Use `CL_FALSE` and chain as event dependency for kernel launch. Also use non-blocking readback after `queue.finish()`.
- **Files:** `crates/q-mining/src/gpu.rs` (dispatch_blake3_kernel)
- **Assigned:** Claude Agent

### GPU-002: Per-GPU adaptive work size + initial calibration
- **Priority:** HIGH
- **Branch:** `gpu/phase-2-per-gpu-adaptive`
- **Status:** 🔵 In Progress
- **Impact:** +0-10% multi-GPU, instant ramp-up
- **Description:** Move `adaptive_work_size` from `GPUMiner` into `GPUContext` (per-GPU sizing). Add startup calibration: benchmark 4 work sizes (64K, 256K, 1M, 4M) and pick optimal starting point.
- **Files:** `crates/q-mining/src/gpu.rs` (GPUContext, GPUMiner, initialize_contexts, mine_batch_multi)
- **Assigned:** Claude Agent

### GPU-003: Kernel-level optimizations
- **Priority:** MEDIUM
- **Branch:** `gpu/phase-3-kernel-tweaks`
- **Status:** 🔵 In Progress
- **Impact:** +10-20% throughput
- **Description:**
  1. Store `challenge_words` in `__local` memory (1 global read per work-group instead of per-item)
  2. Add `#pragma unroll` hints for 99-round VDF loop
  3. Add `__attribute__((reqd_work_group_size(256,1,1)))` to main kernel
  4. Use `__constant` qualifier for challenge buffer (hardware constant cache)
- **Files:** `crates/q-mining/src/gpu.rs` (BLAKE3_KERNEL_SOURCE)
- **Assigned:** Claude Agent

### GPU-004: Kernel compilation caching
- **Priority:** MEDIUM
- **Branch:** `gpu/phase-4-kernel-cache`
- **Status:** 🔵 In Progress
- **Impact:** Startup 2-10s → <100ms
- **Description:** After compiling OpenCL kernel from source, save binary to `~/.config/q-miner/kernel-cache/{hash}.clbin`. On next startup, try `create_program_with_binary()` first. Cache key = sha256(source + device_name + driver_version).
- **Files:** `crates/q-mining/src/gpu.rs` (initialize_contexts)
- **Assigned:** Claude Agent

### GPU-005: Build & deploy v10.1.8
- **Priority:** HIGH
- **Branch:** `gpu/v10.1.8-optimizations` (merge target)
- **Status:** ⚪ Blocked on GPU-001..GPU-004
- **Impact:** Ship to users
- **Description:** Merge all phases, bump version to 10.1.8, Docker build on Epsilon (Debian 12), deploy to downloads, verify GLIBC compat.
- **Files:** `Cargo.toml`, Docker build on Epsilon

---

## Closed Issues

### GPU-000: Persistent buffers, conditional upload, challenge precompute, adaptive sizing (v10.1.7)
- **Status:** ✅ Closed (2026-03-27)
- **Branch:** `feature/safe-batched-sync-v1.0.2`
- **Summary:** Eliminated 5 buffer alloc/free per dispatch, conditional challenge/target upload, CPU-side u8→u32 conversion, auto-tuning work size 65K-8M.

---

## Future Issues (v10.2.0+)

### GPU-010: Parallel multi-GPU dispatch (per-GPU threads)
- **Priority:** MEDIUM
- **Status:** ⚪ Planned
- **Description:** Wrap each GPUContext in Mutex, spawn thread per GPU, use channel for first-solution-wins. Currently mine_batch_multi() dispatches sequentially.

### GPU-011: Double buffering
- **Priority:** LOW
- **Status:** ⚪ Planned
- **Description:** Two sets of persistent buffers, alternating between compute and readback. Requires out-of-order queue or 2 queues per device.

### GPU-012: Hardware-specific tuning
- **Priority:** LOW
- **Status:** ⚪ Planned
- **Description:** Detect GPU vendor/arch, adjust local_work_size (NVIDIA=256, AMD GCN=64, RDNA=256, Intel=128).

### GPU-013: Real temperature/power monitoring
- **Priority:** LOW
- **Status:** ⚪ Planned
- **Description:** NVML for NVIDIA, ROCm SMI for AMD. Populate GPUMiningStats temperature/power_draw fields.

### GPU-014: Stratum protocol for pool mining
- **Priority:** LOW
- **Status:** ⚪ Planned
- **Description:** Implement pool-mining feature flag with stratum v2 protocol.
