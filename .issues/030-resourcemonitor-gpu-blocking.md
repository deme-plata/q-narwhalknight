# #030: ResourceMonitor GPU query blocks async runtime

**Priority**: HIGH
**File(s)**: `crates/q-compute/src/resource_monitor.rs`
**Risk**: Tokio runtime stalls, increased latency on all async tasks

## Problem

`get_gpu_stats()` (line 207) calls `std::process::Command::new("nvidia-smi")` synchronously. This function is called from inside the `tokio::spawn` async sampling loop that runs every 100ms.

`std::process::Command::output()` blocks the current thread while waiting for the child process to exit. On systems without an NVIDIA GPU, `nvidia-smi` will fail after a brief PATH search, but on systems WITH a GPU, nvidia-smi can take 50-500ms to execute — consuming most or all of the 100ms sampling interval. This blocks the tokio worker thread, starving other async tasks (mining submission handling, SSE streaming, P2P message processing).

Even on systems without nvidia-smi, the failed exec attempt still briefly blocks the thread for PATH resolution.

## Fix

1. Replace `std::process::Command` with `tokio::process::Command` and `.await` the output.
2. Cache the nvidia-smi result and only re-query every 1-2 seconds instead of every 100ms. GPU utilization does not change meaningfully in 100ms.
3. On first call, detect whether nvidia-smi exists. If not, set a flag and skip all future calls to avoid repeated failed exec attempts.
4. Consider using the NVML library (`nvml-wrapper` crate) for direct GPU queries without process spawning.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
