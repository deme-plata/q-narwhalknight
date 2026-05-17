# Issue #031: Binary Size Reduction (86MB -> Target 40MB)

**State**: `closed`
**Priority**: MEDIUM
**Labels**: `starship-endgame`, `optimization`, `build`
**Assigned**: Epsilon
**Branch**: `feature/safe-batched-sync-v1.0.2`
**Created**: 2026-03-10

---

## Description

Bryan Cantrill would say: "86MB is too fat." The q-api-server binary is 86MB, which impacts download times for miners (especially on slow connections). Target: reduce to ~40MB without losing functionality.

## Current State

- Binary: 86MB (release, LTO=thin, codegen-units=1)
- Includes: AI inference (mistral.rs), post-quantum crypto, WASM VM, DEX, TUI, Tor client
- Many features are optional but compiled in by default

## Approach

1. **Feature-gate heavy dependencies** — AI inference, Tor, WASM VM behind cargo features
2. **Strip debug symbols** — `strip = true` in release profile
3. **Optimize PQ crypto** — pqcrypto crates are huge; evaluate lighter alternatives
4. **Dead code elimination** — identify unused code paths with `cargo-bloat`
5. **Compress with UPX** — optional self-extracting binary (50-60% reduction)
6. **Split into separate binaries** — `q-api-server-lite` without AI/Tor for miners who don't need them

## Analysis Steps

```bash
# 1. Identify largest contributors
cargo bloat --release --package q-api-server -n 30

# 2. Check section sizes
size target/release/q-api-server

# 3. Strip symbols (quick win)
strip target/release/q-api-server
ls -lh target/release/q-api-server

# 4. Feature-gated build
cargo build --release --package q-api-server --no-default-features --features "tui,mining"
```

## Acceptance Criteria

- [x] Add `strip = "symbols"` to release profile — already in `[profile.release]`
- [x] Feature-gate AI inference behind `ai` feature — q-ai-inference + candle-core optional
- [x] Feature-gate ZK subsystem behind `zk` feature — q-zk-stark, q-zk-snark, q-lattice-guard, q-recursive-proofs optional
- [x] Feature-gate DEX subsystem behind `dex` feature — q-dex, q-oracle, q-quillon-bank optional
- [x] Default build unchanged — `default = ["tui", "ai", "zk", "dex"]`
- [x] Slim build documented — `cargo build --release --no-default-features --features tui`
- [ ] Source-level `#[cfg(feature)]` guards for slim builds — follow-up (9 files, ~24 import sites)
- [ ] Analyze with `cargo-bloat` — run periodically

## Depends On

None.

## Files

- `Cargo.toml` — Feature gates in workspace and package manifests
- `crates/q-api-server/Cargo.toml` — Package features
- `.cargo/config.toml` — Release profile tuning
