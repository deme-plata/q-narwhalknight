# Issue #035: Reproducible Benchmarking & Performance Evaluation Suite

**State**: `closed`
**Priority**: HIGH
**Labels**: `starship-endgame`, `benchmarks`, `academic`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

For the academic whitepaper (#032) and for ongoing performance tracking, we need a reproducible benchmarking suite that measures and graphs:

1. **Consensus throughput** — TPS under varying validator counts (4, 16, 64, 128)
2. **Finality latency** — Time from tx submission to commit (p50/p95/p99)
3. **Sync speed** — Blocks/second during turbo sync
4. **Mining hashrate** — H/s across CPU/GPU configurations
5. **q-flux throughput** — req/s, p99 latency, connection overhead
6. **PQ crypto overhead** — Ed25519 vs Dilithium5 sign/verify times
7. **Memory usage** — RSS over time during sync and steady-state
8. **Network bandwidth** — Bytes/block, gossipsub overhead

## Current State

- `cargo bench` exists but only covers micro-benchmarks
- No end-to-end TPS measurement tool
- No multi-node benchmark harness
- Performance numbers in docs are manual measurements

## Acceptance Criteria

- [x] 28 existing Criterion benchmarks across 14 crates — crypto, storage, VM, ZK, mixing, sharding, flux, etc.
- [x] `scripts/run_benchmarks.sh` — unified harness: runs all suites, outputs JSON + text to `target/benchmark-results/`
- [x] `scripts/run_benchmarks.sh --quick` — fast subset (crypto + storage + VM, ~3 min)
- [x] `scripts/run_benchmarks.sh --json` — machine-readable JSON output
- [x] `scripts/run_benchmarks.sh --suite NAME` — run specific suite by name
- [x] `scripts/plot_benchmarks.py` — text report + matplotlib charts (bar chart of suite durations)
- [x] `scripts/plot_benchmarks.py --compare DIR1 DIR2` — side-by-side regression detection with % change
- [x] System info captured (hostname, kernel, CPUs, RAM, Rust version) in `system_info.json`
- [x] Criterion HTML reports auto-copied to results directory
- [ ] CI integration — run `--quick` on PR, full suite weekly
- [ ] Multi-node end-to-end TPS benchmark — requires running nodes (deferred)

## Depends On

- #033 (Compile time — fast iteration on benchmarks)

## Files

- `benchmarks/` — Criterion-based benchmark suite
- `scripts/run_benchmarks.sh` — Full suite runner
- `scripts/plot_benchmarks.py` — Visualization
- `benchmarks/results/` — Output directory
