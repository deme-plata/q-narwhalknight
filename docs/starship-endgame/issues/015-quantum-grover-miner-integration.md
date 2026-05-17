# Issue #015: Quantum Grover Miner — Rust FFI Integration

**State**: `in_progress`
**Priority**: HIGH
**Labels**: `starship-endgame`, `quantum`, `mining`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10
**Updated**: 2026-03-10

## Progress

- `grover_backend.rs`: 1303 lines implemented
  - GroverMiningBackend trait implementation
  - QuantumCircuitSimulator with QPanda/Qiskit fallback
  - Classical hybrid mining (20-bit Grover + 44-bit brute force)
  - MEV protection via quantum RNG for nonce ordering
  - 22 comprehensive unit tests

---

## Description

The `q-grover/` directory contains a Python implementation of Grover's algorithm for quantum-accelerated mining (20-bit quantum + 44-bit classical hybrid approach). This needs to be integrated into the Rust mining pipeline as a pluggable mining backend, accessible via the compute orchestrator.

## Architecture

```
q-api-server
  └── q-mining (existing Rust miner)
        └── GroverBackend (new)
              ├── QPanda FFI (hardware quantum)
              ├── Qiskit Simulator (software fallback)
              └── Classical Hybrid (20-bit Grover + 44-bit brute)
```

## Implementation Plan

1. Create `crates/q-grover-ffi/` — Rust crate wrapping `q-grover/` Python via PyO3 or subprocess
2. Add `GroverMiningBackend` trait implementation for `q-mining`
3. Wire into orchestrator as Layer 0 variant (quantum mining path)
4. Benchmark: compare hash rate vs classical on same hardware
5. MEV protection: use quantum RNG for nonce selection order

## Acceptance Criteria

- [ ] `crates/q-grover-ffi/` crate with Rust→Python bridge
- [ ] `GroverMiningBackend` implements mining trait
- [ ] Orchestrator can switch between classical/quantum mining
- [ ] Benchmark: quantum vs classical hash rates published
- [ ] MEV protection via quantum RNG nonce ordering
- [ ] Graceful fallback to classical if no quantum backend available

## Depends On

- #001 (Orchestrator manages mining layer)
- #003 (GPU acceleration — Grover uses GPU for simulation)

## Files (planned)

- `q-grover/` — Existing Python quantum miner
- `crates/q-grover-ffi/src/lib.rs` — NEW: Rust FFI bridge
- `crates/q-mining/src/grover_backend.rs` — NEW: Mining backend impl
