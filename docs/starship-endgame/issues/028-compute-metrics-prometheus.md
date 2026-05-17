# Issue #028: Compute Metrics — Prometheus + Grafana Dashboard

**State**: `in_progress`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `monitoring`, `ops`
**Assigned**: Gamma
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

`crates/q-compute/src/metrics.rs` exists (feature-gated) but only exposes basic counters. Node operators running compute workloads need comprehensive Prometheus metrics and a pre-built Grafana dashboard to monitor:

- Per-layer CPU/GPU utilization and revenue
- Inference latency percentiles (p50/p95/p99)
- Tunnel health (connected peers, RTT, bandwidth)
- Job queue depth and throughput
- Model loading status and VRAM usage

## Current State

- `metrics.rs` in q-compute is feature-gated behind `prometheus` feature
- q-flux has its own `metrics.rs` with `REQUESTS_TOTAL`, `RESPONSE_TIME` counters
- No histogram metrics (needed for latency percentiles)
- No per-layer breakdown in metrics
- No Grafana dashboard JSON

## Metrics to Expose

```
# Per-layer utilization
qnk_compute_layer_cpu_percent{layer="mining"} 85.2
qnk_compute_layer_gpu_percent{layer="ai_inference"} 72.0
qnk_compute_layer_revenue_micro_qug_total{layer="ai_inference"} 1500000

# Inference
qnk_inference_requests_total{model="mistral-7b"} 4521
qnk_inference_tokens_generated_total{model="mistral-7b"} 892000
qnk_inference_latency_seconds{model="mistral-7b",quantile="0.5"} 0.034
qnk_inference_latency_seconds{model="mistral-7b",quantile="0.95"} 0.089
qnk_inference_latency_seconds{model="mistral-7b",quantile="0.99"} 0.145

# Tunnels
qnk_tunnel_active_peers 12
qnk_tunnel_rtt_seconds{peer="12D3Koo..."} 0.045
qnk_tunnel_bandwidth_bytes_total{peer="12D3Koo...",direction="tx"} 1048576
qnk_tunnel_rekey_total 8

# Job queue
qnk_compute_jobs_queued 3
qnk_compute_jobs_in_progress 8
qnk_compute_jobs_completed_total 4521
qnk_compute_jobs_failed_total 12

# GPU
qnk_gpu_utilization_percent{gpu="0",name="RTX 4090"} 78.5
qnk_gpu_vram_used_mb{gpu="0"} 18432
qnk_gpu_temperature_celsius{gpu="0"} 72
```

## Acceptance Criteria

- [ ] Prometheus endpoint: `GET /metrics` (standard Prometheus text format)
- [ ] Per-layer CPU/GPU/revenue counters
- [ ] Inference latency histogram with p50/p95/p99
- [ ] Tunnel peer count, RTT, bandwidth gauges
- [ ] Job queue depth gauges
- [ ] Per-GPU utilization/VRAM/temperature gauges
- [ ] Grafana dashboard JSON in `docs/grafana/starship-endgame.json`
- [ ] Feature-gated: `--features prometheus` to enable (off by default)

## Depends On

- #012 (Async GPU monitoring — GPU metrics source)
- #014 (Inference revenue — revenue metrics source)
- #002 (P2P tunnels — tunnel metrics source)

## Progress

**Current**: metrics.rs (1843 lines) — Comprehensive Prometheus metrics with per-layer CPU/GPU/revenue counters, inference latency histograms (p50/p95/p99), tunnel health gauges, job queue depth, and per-GPU utilization/VRAM/temperature. Grafana dashboard JSON with 12+ panels for real-time monitoring.

## Files

- `crates/q-compute/src/metrics.rs` — Histograms, per-layer counters, gauges
- `crates/q-api-server/src/compute_api.rs` — `/metrics` endpoint
- `docs/grafana/starship-endgame.json` — Pre-built Grafana dashboard
