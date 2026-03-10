# Issue #003: GPU Mining Acceleration

**State**: `open`
**Priority**: HIGH
**Labels**: `starship-endgame`, `gpu`, `mining`
**Assigned**: Epsilon
**Branch**: (not started)
**Created**: 2026-03-08
**Updated**: 2026-03-10

---

## Description

Add GPU hash computation for 10-100x mining speedup. Auto-detect GPU vendor, compile appropriate shader/kernel.

## Acceptance Criteria

- [ ] OpenCL backend (AMD + Intel + NVIDIA)
- [ ] CUDA backend (optional feature flag)
- [ ] Vulkan compute fallback
- [ ] GPU memory pool (avoid alloc/dealloc per hash)
- [ ] Benchmark: CPU vs GPU hash rate comparison

## Notes

Epsilon has 10Gbit bandwidth and is the GPU target. The mining hash function needs a GPU-friendly compute shader. Consider wgpu for cross-platform Vulkan/Metal/DX12 support.

## Blocked By

- #001 (orchestrator assigns GPU layers)
