# Technical Review: Build-Time Reduction Plan

**Date:** 2026-05-13
**Context:** Q-NarwhalKnight is on mainnet at $1.45–2B paper market cap, 17.9M blocks live. Every release cycle has needed 27–42 minutes per incremental `cargo build --release --package q-api-server` on Epsilon Docker. We did 7 binaries in this session alone — that's ~4 hours of build wait inside one workday.
**Constraint:** **Mainnet correctness is non-negotiable.** No change may alter observable behavior, public API surface, signature semantics, panic patterns, or feature gating. No `unsafe` introduction. Soak-test gates every shipped change.

---

## Why This Matters

A 30-minute build cycle defines how fast we can land safety fixes. This week's pattern:

1. Find a real bug in soak (e.g., Phase 2 OOM, pointer auto-repair, premature gap=0 fire).
2. Write a ~50-LOC patch.
3. Wait ~30 minutes for incremental rebuild.
4. Spin a fresh Docker container, wait ~5–25 minutes for soak signal.
5. Discover another issue, GOTO 2.

**The build is the bottleneck of the entire correctness loop.** Halving it nearly doubles the rate at which we can ship safe consensus fixes. That's the whole game.

This review focuses on **safe** reductions — changes that have no plausible mainnet risk. Anything that touches consensus paths, signature handling, or balance arithmetic is explicitly out of scope.

---

## Observed Build Metrics (This Week)

| Phase | Time |
|-------|------|
| Cold `cargo build --release` of q-api-server (all deps) | ~60 min |
| Incremental (touched q-storage + q-api-server only) | 27–42 min |
| Final `rustc` of `q-api-server` bin alone (LTO + codegen-units=1) | 12–20 min |
| Cold `cargo test --package q-ivc --lib` (PQ deps from scratch) | ~6 min |
| Incremental `cargo test --package q-ivc --lib` | ~5 sec |

Workspace stats:
- **93 crates** in `crates/`
- `q-api-server/src/main.rs` is **25,784 lines** (largest single Rust file by far)
- `q-api-server` Cargo.toml lists **151 direct dependencies**
- Release profile uses `lto = "thin"`, `codegen-units = 1` — the biggest single contributor to long bin-link time

The single dominant cost is the final `rustc q-api-server bin` with LTO. It's CPU-bound on one core for 12–20 minutes because LTO+codegen-units=1 force the entire bin into one optimization unit.

---

## Ranked Optimization Opportunities

Each item below has an estimated time savings and a risk classification. Risk levels:
- **Green** — no plausible behavior change; pure tooling/dep hygiene
- **Yellow** — small behavior change possible only at error boundaries; soak-test catches
- **Red** — touches consensus or signature paths; out of scope this review

### Tier 1 (Green, ship in v10.9.15)

**1.1 Split `q-api-server/src/main.rs` into a thin bin + a fat lib.** Already partially done (`lib.rs` exists). Move the remaining 25K lines of `main.rs` into the lib so the bin crate only orchestrates. The bin then re-links in seconds; only the lib has to recompile when code changes. Estimated saving on incremental: **5–10 min**.

  *Risk*: Green. No semantics change — just a `pub mod` reshuffling. Test by binary-byte-comparing the output.

**1.2 Enable cargo's `cranelift` codegen for `cargo test` (not release).** `rustc -Z codegen-backend=cranelift` skips LLVM for debug/test builds. Test compile times typically drop 30–50%. Release builds unchanged. Estimated saving on `cargo test`: **30–50%**, ~3 min on cold q-ivc test.

  *Risk*: Green. Cranelift is now stabilized for non-release codegen. Production binary unaffected.

**1.3 `RUSTFLAGS="-Z share-generics=y"` (nightly) or `-C share-generics=y` (stable on workspaces).** Allows generic monomorphizations to be shared between crates instead of recompiled in each. For the heavy `serde::Serialize` / `Result<_, _>` patterns we have everywhere, this is significant. Estimated saving on incremental: **10–20%**.

  *Risk*: Green. Pure linker optimization. Output binary semantics identical.

**1.4 Workspace `[profile.release.package.*]` overrides** to reduce optimization for non-hot crates:

  ```toml
  [profile.release.package.crown-ash-narrative]
  codegen-units = 16
  
  [profile.release.package.q-tui]
  codegen-units = 16
  ```

  Crates not on the consensus or mining hot path don't need codegen-units=1. Keep it for `q-storage`, `q-types`, `q-api-server`, `q-network`, `q-vdf`, `q-quantum-crypto`, `blake3`. Estimated saving on cold build: **5–8 min**.

  *Risk*: Green. Per-crate optimization tuning. The few crates we keep at codegen-units=1 are the ones doing the cryptographic and network hot paths. Audit the override list with the existing maintainer before shipping.

**1.5 Remove unused workspace dependencies.** Need to grep each crate's `Cargo.toml` against actual `use` statements. Likely 5–15 dependencies in the workspace are listed but never used (drift from refactors). Estimated saving: **2–4 min** on cold, marginal on incremental.

  *Risk*: Green if done with `cargo-machete` (or the manual equivalent). Each removal is mechanical and obvious — either the crate is referenced or it isn't.

**Tier 1 total estimated saving: 20–30 min cold, 10–15 min incremental.** This is the easy win.

---

### Tier 2 (Yellow, ship in v10.10.0 after careful audit)

**2.1 Disable default features on heavy deps where we don't use them.**

  Likely candidates (need verification):
  - `tokio` — we already use specific features
  - `reqwest` — `default-features = false, features = ["json", "rustls-tls"]` would drop the openssl dep
  - `serde` — `default-features = false` then enable per use site
  - `chrono` — usually uses `["clock", "std"]` only
  - `image` — drop `["default"]`, keep only formats we read

  Each one needs a careful check that we don't actually use a default feature. Easy to break log formatting (chrono Local timezone) or JSON parsing (serde derive). Soak test catches obvious breakage; subtle changes (e.g., a chrono `Local::now()` falling back to UTC) require explicit testing.

  Estimated saving on cold build: **3–5 min** per cleanly-trimmed major dep.

  *Risk*: Yellow. A misconfigured feature flag can silently change runtime behavior at runtime (e.g., chrono timezones). Each must be tested. Audit deps one at a time, soak each independently.

**2.2 Move proc-macro heavy code to compile-time generation.**

  Heavy proc macros are `serde_derive`, `tokio::main`, `axum::handler`. Most of these are stable and shared, so they cache well. But macros like `tracing::instrument` and `derive(Debug)` on huge enums generate enormous code.

  Strategy: audit which derives are actually used. Many places have `#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]` on types that only need 1–2 of those derives. Trimming to actual usage reduces both compile time and binary size.

  Estimated saving: **5–10 min** on cold across the workspace if done thoroughly.

  *Risk*: Yellow. Removing `Debug` from a type that's actually `dbg!`'d elsewhere is a compile error (safe). Removing `Clone` from a type that's actually cloned is a compile error (safe). Removing `Deserialize` from a struct serialized in a runtime config file is a *silent* runtime breakage (not safe without test). Audit-only at first; convert in batches with soak gates.

**2.3 Use `LazyLock` instead of `lazy_static!`.**

  `lazy_static!` expands to several hundred lines of macro per invocation. `std::sync::LazyLock` (stable since 1.80) is the modern replacement and adds zero macro overhead. Grep across the codebase for `lazy_static!` and convert call-by-call.

  Estimated saving: minor (1–2 min) but cleans up tech debt.

  *Risk*: Yellow. The conversion is mechanical but each `LazyLock` has a slightly different initialization order semantics than `lazy_static`. In practice the difference is irrelevant for almost all uses.

**Tier 2 total estimated saving: 8–15 min cold, 3–5 min incremental.**

---

### Tier 3 (Yellow, longer-term refactoring)

**3.1 Split `q-api-server` into multiple smaller crates.**

  25K lines in `main.rs` is the single biggest crate-compile bottleneck. Logical splits already implied by the existing module structure:

  - `q-api-server-core` — AppState + boot + signal handlers (small, stable)
  - `q-api-server-routes` — axum routes + handler glue (the HTTP layer)
  - `q-api-server-sync-loop` — turbo sync, gap fill, batch sync (the hot path)
  - `q-api-server-mining-loop` — block production, miner-link, finality (the hot path)
  - `q-api-server-dex` — DEX swap, AMM, pools, prices (already a fat module)
  - `q-api-server-bin` — just `main()`

  Each smaller crate recompiles independently. Touching one of the bridge APIs no longer triggers a re-link of the entire 25K-line binary.

  Estimated saving on incremental: **10–15 min** on most touch patterns, because the LTO unit is much smaller.

  *Risk*: Yellow → could surface real refactoring bugs (e.g., a global static moved across crate boundaries). Plan it as a separate v10.x.x dedicated release with full soak. Probably 2–3 weeks of careful work. Worth it.

**3.2 Replace `huggingface/candle` git deps with crates.io versions OR vendor them.**

  Currently `Cargo.lock` pulls `candle` from a git URL, which forces a fresh git fetch + recompile on every clean. Pinning to crates.io versions or vendoring (with `cargo vendor`) eliminates that. Estimated saving on cold: **2–3 min**.

  *Risk*: Yellow. If we depend on a feature only in the git fork, vendoring is safer than switching. Need to check what's actually used.

**Tier 3 total estimated saving: 12–18 min on cold, 10–15 min on incremental (after split lands).**

---

### Tier 4 (DO NOT do — explicitly)

These are commonly proposed but are *unsafe* for a mainnet-live project:

- **Disable LTO.** Saves 5–8 min on bin link but causes a ~10–15% throughput regression on mining hashing (blake3 inlining) and signature verification. **Not acceptable at mainnet.**
- **Switch `codegen-units = 1` → higher.** Same issue: throughput regression on hot paths.
- **Strip features from `pqcrypto-dilithium`, `pqcrypto-kyber`.** These are the post-quantum primitives. Their feature flags map to cryptographic backends. Any change requires NIST-level review.
- **Convert `panic = "abort"` to `unwind`.** Increases binary size and adds runtime cost. Has consensus implications (a paniclayer that unwinds could leave RocksDB in a partially-committed state on some platforms). Out of scope.
- **Replace `tikv_jemallocator` with system allocator.** jemalloc fragmentation tuning is something we already rely on for the OOM fix landed today. Changing allocators is a Tier-R3 (consensus-class) change.

---

## Implementation Plan

### Phase A — v10.9.15 / "tooling-only" (this week)

Land Tier 1 items as one cohesive PR:

1. **1.1** Split `q-api-server::main.rs` into bin + lib (verify byte-identical binary via reproducible build)
2. **1.3** `share-generics=y` in `.cargo/config.toml`
3. **1.4** Per-crate `codegen-units` overrides for non-hot crates
4. **1.5** Run `cargo-machete`, remove unused deps in batches

  Goal: drop incremental build from 30 min → **15–20 min**.

  Validation: full mainnet-safety test suite (per CLAUDE.md), 24h soak on Alpha Docker before deploying to Beta. Binary diff against v10.9.14 should show only file-naming changes, no instruction differences.

### Phase B — v10.10.0 / "feature flag audit" (next 2 weeks)

Land Tier 2 items one at a time:

1. **2.1** Audit and trim default features on `reqwest`, `chrono`, `image`, `tracing-subscriber`
2. **2.2** Trim unused derives across the workspace using a script (search for `#[derive(.*)]` on types that grep returns ≤1 usage of the relevant trait)
3. **2.3** Migrate `lazy_static!` → `LazyLock` mechanically

  Goal: incremental build down to **10–12 min**.

  Validation: Phase A's soak gate, plus extra attention to log formatting (chrono changes) and JSON parsing (serde feature changes).

### Phase C — v10.11.0 / "crate split" (~1 month, dedicated sprint)

Land Tier 3 items as a structural release:

1. **3.1** Crate-split `q-api-server` into 6 sub-crates
2. **3.2** Vendor or crate.io-pin the `candle` git deps

  Goal: incremental build down to **5–8 min** in the common case (most edits only touch one sub-crate). Full from-scratch build down to ~30 min.

  Validation: this is the riskiest of the three phases. Plan: cut a branch, soak for 7 days continuously on Alpha + Delta Docker before merging. Compare wallet balances, balance_root, and tip catch-up speed against v10.10.x baseline. Only merge if all signals are clean.

---

## What This Is *Not*

This plan does not:
- Touch any code on the consensus path (block validation, balance arithmetic, signature verification, BAL-001 enforcement)
- Change any cryptographic primitive
- Modify the Dilithium / Kyber / SQIsign integration
- Alter peer-to-peer protocol versions
- Change RocksDB layout, column families, or compaction
- Affect the IVC SNARK work in `crates/q-ivc`

Every change is a *tooling* or *compilation arrangement* change. The output binary should behave bit-identically on the consensus paths.

---

## What This Is

A roadmap to halve the build time without changing the chain. The single highest-leverage piece is the q-api-server crate split (Phase C), because that's where the multi-minute LTO link lives. Phase A is the quick win that buys us breathing room. Phase B is the cleanup that prevents the slowness from creeping back.

If we land Phase A in the next release, we ship safety fixes ~50% faster within the week. That ROI compounds across every future bug found in soak.

— Server Beta, 2026-05-13
