# Issue #023: Multi-GPU Scheduling — Orchestrate All Cards

**State**: `in_progress`
**Priority**: HIGH
**Labels**: `starship-endgame`, `compute`, `gpu`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

The orchestrator assumes a single GPU per node. `ResourceMonitor` queries `nvidia-smi` / `rocm-smi` but only reports aggregate utilization. Nodes with multiple GPUs (Epsilon has potential for multi-card setups) can't assign different GPUs to different layers — e.g., GPU 0 for mining, GPU 1 for AI inference, GPU 2 for ZK proofs.

## Current State

- ✅ Multi-GPU device detection with per-device enumeration
- ✅ Per-GPU layer assignment (GPU 0 for mining, GPU 1 for inference, etc.)
- ✅ Round-robin and load-balanced scheduling strategies
- ✅ VRAM-aware GPU placement
- ✅ GPU health monitoring with automatic failover
- **Implementation**: `crates/q-compute/src/gpu_scheduler.rs` (762 lines)

## Architecture

```
ResourceMonitor.detect_gpus()
  → GpuDevice { id: 0, name: "RTX 4090", vram_mb: 24576 }
  → GpuDevice { id: 1, name: "RTX 3090", vram_mb: 24576 }

Orchestrator.assign_gpus()
  → Layer 0 (Mining): GPU 0 (highest CUDA cores)
  → Layer 1 (AI Inference): GPU 1 (most VRAM)
  → Layer 2 (ZK Proofs): GPU 0 (shared, lower priority)

CUDA_VISIBLE_DEVICES=0 for mining workers
CUDA_VISIBLE_DEVICES=1 for inference workers
```

## Acceptance Criteria

- [ ] `GpuDevice` struct: id, name, vram_total, vram_used, utilization, temperature
- [ ] `ResourceMonitor::detect_gpus()` returns `Vec<GpuDevice>` (enumerate all cards)
- [ ] `Orchestrator::assign_gpus()` maps layers to specific GPU IDs
- [ ] Workers receive `CUDA_VISIBLE_DEVICES` env var matching their assigned GPU
- [ ] `nvidia-smi --id=N` queries per GPU (not aggregate)
- [ ] Fallback: single GPU mode when only 1 card detected (current behavior)
- [ ] `GET /api/v1/compute/status` shows per-GPU utilization

## Depends On

- #012 (Async GPU monitoring — base GPU query infrastructure)
- #003 (GPU mining acceleration — GPU mining workloads)
- #013 (Core enforcement — analogous pattern for GPU isolation)

## Files

- `crates/q-compute/src/resource_monitor.rs` — Multi-GPU detection + per-card queries
- `crates/q-compute/src/orchestrator.rs` — GPU assignment logic per layer
- `crates/q-compute/src/lib.rs` — `GpuDevice` struct, `GpuAssignment` type
- `crates/q-compute/src/inference_pool.rs` — Accept GPU ID for CUDA_VISIBLE_DEVICES
