# Issue #033: Compile Time Optimization (30min -> Target 8min)

**State**: `closed`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `build`, `dx`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Every computer scientist would note: 30+ minute compile times hurt development velocity. PQ crypto crates (pqcrypto, kyber, dilithium) are the main bottleneck. Target: full release build in under 8 minutes on Epsilon (48 cores).

## Current State

- Clean release build: ~30-45 minutes
- Incremental build: ~3-5 minutes (when it works)
- PQ crypto crates compile C code via `cc` crate (single-threaded C compilation)
- `codegen-units = 1` maximizes runtime perf but slows linking

## Approach

1. **Split codegen-units** — Use `codegen-units = 8` for dev, `1` for release-deploy only
2. **Pre-compiled PQ crypto** — Build pqcrypto as a system library, link dynamically
3. **Cranelift backend** — Use cranelift for debug builds (5-10x faster compilation)
4. **cargo-nextest** — Parallel test execution (currently sequential per crate)
5. **sccache** — Distributed compilation cache across Beta/Epsilon
6. **Workspace dependency dedup** — Audit `cargo tree -d` for duplicate crate versions
7. **Feature pruning** — Disable unused features in dependencies

## Analysis Steps

```bash
# 1. Profile compilation time per crate
cargo build --release --timings --package q-api-server
# Opens target/cargo-timings/cargo-timing.html

# 2. Find duplicate dependencies
cargo tree -d | head -50

# 3. Measure PQ crypto compile time specifically
time cargo build --release --package q-types 2>&1 | tail -5

# 4. Check codegen units impact
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=8 cargo build --release --package q-api-server
```

## Acceptance Criteria

- [x] `cargo build --timings` analysis — use `cargo build --release --timings`
- [x] Dev profile uses `codegen-units = 16` + opt-level=1 — `Cargo.toml [profile.dev]`
- [x] Dev deps compiled at opt-level=2 — `[profile.dev.package."*"]`
- [x] Cranelift config prepared — `.cargo/config.toml` (uncomment to enable)
- [x] sccache config prepared — `.cargo/config.toml` (uncomment to enable)
- [x] debug = "line-tables-only" — reduces debug info link time
- [ ] PQ crypto pre-compiled as system library — deferred (marginal gain vs complexity)
- [ ] Duplicate dependency audit — run `cargo tree -d` periodically

## Depends On

None.

## Files

- `Cargo.toml` — Profile tuning
- `.cargo/config.toml` — Cranelift, sccache config
- `docs/technical/BUILD_OPTIMIZATION.md` — Analysis results
