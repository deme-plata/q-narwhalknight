# PR #016: Distributed Compute & Security — 7 Modules

**State**: `open`
**Head**: `feature/safe-batched-sync-v1.0.2`
**Base**: `main`
**Author**: Server Beta
**Created**: 2026-03-10
**Labels**: `starship-endgame`, `compute`, `security`, `p2p`
**Closes**: #016 (Bridge Verification), #017 (Marketplace), #018 (Tensor Parallelism), #024 (Tunnel Key Rotation), #025 (Model Catalog), #026 (Job WAL), #028 (Prometheus Metrics)

---

## Summary

Seven modules covering bridge security, AI marketplace, tensor parallelism, key rotation, model management, job persistence, and observability.

### What's included

- **bridge_verification.rs** (1089 lines) — Multi-peer attestation, 2-of-3 quorum, 13 tests
- **marketplace.rs** (926 lines) — Proof-of-Useful-Work, 6 work types, bid/ask, 11 tests
- **tensor_parallel.rs** (563 lines) — Cross-node model splitting, pipeline scheduling, 13 tests
- **tunnel_rekey.rs** (477 lines) — X25519 key rotation, zeroize, forward secrecy, 14 tests
- **model_catalog.rs** (762 lines) — 8 GGUF models, VRAM-aware hot-swap, 16 tests
- **job_wal.rs** (1153 lines) — Append-only WAL, state machine, recovery, 14 tests
- **metrics.rs** (1846 lines) — Self-contained Prometheus, histograms, 23 tests

### Total: 6,816 lines, 104 tests

### Files changed

| File | Lines | Change |
|------|-------|--------|
| `crates/q-compute/src/bridge_verification.rs` | 1089 | NEW |
| `crates/q-compute/src/marketplace.rs` | 926 | NEW |
| `crates/q-compute/src/tensor_parallel.rs` | 563 | NEW |
| `crates/q-compute/src/tunnel_rekey.rs` | 477 | NEW |
| `crates/q-compute/src/model_catalog.rs` | 762 | NEW |
| `crates/q-compute/src/job_wal.rs` | 1153 | NEW |
| `crates/q-compute/src/metrics.rs` | 1846 | NEW |
| `crates/q-compute/src/lib.rs` | +14 | Module declarations |
| `crates/q-compute/Cargo.toml` | +1 | zeroize dependency |
| `docs/grafana/starship-endgame-compute.json` | 777 | NEW Grafana dashboard |

### Test plan

- [x] `cargo test --package q-compute` — all 104 new tests pass
- [x] `cargo check --package q-compute` — compiles clean
