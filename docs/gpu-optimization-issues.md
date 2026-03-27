# GPU Mining Optimization — Issue Tracker

**Project:** Q-NarwhalKnight GPU Miner (`crates/q-mining/src/gpu.rs`)
**Base version:** v10.1.7 (persistent buffers, conditional upload, adaptive work size)
**Target version:** v10.1.8 (async dispatch, kernel tweaks, compilation caching)
**Branch:** `feature/safe-batched-sync-v1.0.2`

---

## Open Issues

### GPU-005: Build & deploy v10.1.8
- **Priority:** HIGH
- **Branch:** `feature/safe-batched-sync-v1.0.2`
- **Status:** 🔵 In Progress
- **Impact:** Ship to users
- **Description:** Docker build on Epsilon (Debian 12), deploy to downloads, verify GLIBC compat.
- **Files:** `Cargo.toml`, Docker build on Epsilon

---

## Closed Issues

### GPU-004: Kernel compilation caching
- **Status:** ✅ Closed (2026-03-27)
- **Branch:** `feature/safe-batched-sync-v1.0.2`
- **Commit:** `996b37e0`
- **Summary:** Save compiled OpenCL binary to `~/.config/q-miner/kernel-cache/{hash}.clbin`. Cache key = hash(source + device_name + driver_version). On subsequent starts, load with `create_program_with_binary()`. Falls back to source compilation if cached binary is invalid. Startup time: 2-10s → <100ms.

### GPU-003: Kernel-level optimizations
- **Status:** ✅ Closed (2026-03-27)
- **Branch:** `feature/safe-batched-sync-v1.0.2`
- **Commit:** `996b37e0`
- **Summary:** `__constant` qualifier for challenge buffer (hardware constant cache). `__attribute__((reqd_work_group_size(256,1,1)))` on main kernel. `#pragma unroll 3` for 99-round VDF loop (99 = 33×3). IV copied from `__constant` to private address space for NVIDIA OpenCL compat.

### GPU-002: Per-GPU adaptive work size + initial calibration
- **Status:** ✅ Closed (2026-03-27)
- **Branch:** `feature/safe-batched-sync-v1.0.2`
- **Commit:** `996b37e0`
- **Summary:** Moved `adaptive_work_size` from `GPUMiner` to `GPUContext` (per-GPU). `mine_batch_multi()` rewritten for per-GPU timing/tuning. Added `calibrate()` method: benchmarks 4 work sizes (64K, 256K, 1M, 4M), picks highest throughput as starting point.

### GPU-001: Non-blocking flag zeroing + async readback
- **Status:** ✅ Closed (2026-03-27)
- **Branch:** `feature/safe-batched-sync-v1.0.2`
- **Commit:** `996b37e0`
- **Summary:** All 3 `enqueue_write_buffer` calls in `dispatch_blake3_kernel()` now use `CL_FALSE` (non-blocking). On an in-order queue, the GPU driver guarantees writes complete before kernel executes, but the CPU doesn't block. Reads remain `CL_TRUE` since data is needed immediately.

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
