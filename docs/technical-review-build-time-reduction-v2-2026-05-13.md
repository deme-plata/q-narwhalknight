# Technical Review: Build-Time Reduction Plan — V2 (Amended)

**Date:** 2026-05-13
**Supersedes:** `docs/technical-review-build-time-reduction-2026-05-13.md` (V1)
**Reason for amendment:** V1 made three factual errors and overstated one structural win. V2 corrects them and adds two real high-impact items (`mold`, `cargo-chef`) that V1 missed.

---

## Errata from V1

1. **`-C share-generics=y` is nightly-only.** V1 listed it as a stable Tier-1 release-build optimization. It isn't — it requires `-Z share-generics=y` on nightly. Removed from the stable release pipeline.
2. **Cranelift codegen backend is nightly-only.** V1 described it as "stabilized for non-release codegen." That isn't accurate as of the Rust toolchain currently in use; `rustc-codegen-cranelift-preview` is still nightly. Demoted to a separate dev-only experiment.
3. **"Binary-byte-identical" validation is too strong.** Refactors that don't change semantics can still change symbol ordering, DWARF layout, and object metadata. Validation now uses behavioral gates instead (test suite + golden API snapshots + soak + perf benchmark).
4. **Fat-lib + thin-bin doesn't reliably save 5–10 min on incremental.** Only when edits land in the thin bin (rare). When edits land in the fat lib (the common case), the lib rebuild + final ThinLTO link still dominates. Reframed as incremental-invalidation hygiene and Phase C preparation, not a guaranteed time win.

V1 had these wrong, and reviewer caught all four. V2 reflects the corrections.

---

## What Changed Conceptually

V1 framed Phase A as "the easy 15-min win." V2 corrects this:

> **Phase A buys hygiene + some incremental relief; Phase C is the real compile-time unlock.**

The single highest-leverage operational change we can ship today is **switching the Docker build to `cargo-chef`** for dependency-layer caching — that's what cuts per-iteration soak time from ~60 min to ~5 min on code-only edits. The single biggest real-time saving on the `rustc q-api-server` *final-link* step itself is **switching the release linker to `mold`**, but the size of that delta depends on what the current toolchain already uses (Rust 1.90 ships `rust-lld` by default on `x86_64-unknown-linux-gnu`, which has already absorbed some of the historical GNU-`ld` overhead). We measure first, then choose.

The structural unlock — splitting `q-api-server` into smaller crates — remains a Phase C project. That's what removes per-edit invalidation surface so most edits don't trigger the 12–20 min bin link at all.

---

## Revised Phase A (v10.9.15 — this week)

### A.1. Measurement first

Before changing anything, capture the baseline:

```bash
cargo build --release --package q-api-server --timings
CARGO_LOG=cargo::core::compiler::fingerprint=info cargo build --release --package q-api-server 2>&1 | tee fingerprint.log
```

The `--timings` HTML pinpoints which crates dominate (likely `q-api-server`, `q-storage`, `q-network`, `wasmtime`, `librocksdb-sys`, `candle-core`). The fingerprint log says *why* each crate is dirty on an incremental build — sometimes a `build.rs` script or env-var change is triggering rebuilds we don't expect.

Record a baseline: cold time, incremental time (after touching `q-storage` + `q-api-server`), and final `rustc q-api-server` bin time as separate measurements. We can't claim "saved N minutes" without numbers.

### A.2. Switch the release linker to `mold`

Add to `.cargo/config.toml`:

```toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

Install `mold` in the Docker build image. Single change, drops final-link time substantially.

**Risk: Green.** A linker does not change generated code. Validation: build, run the full mainnet-safety test suite, compare API/RPC outputs on a known block range, run a mining-throughput micro-benchmark. Soak on Alpha for 24h. If signals are clean → ship.

**Estimated saving on the bin link step alone: 3–8 min.** This is the single biggest Phase A item by ROI.

### A.3. Add `cargo-chef` to the Docker build

Currently Epsilon's Docker build recompiles all dependencies on every code change because the source tree mounts everything together. `cargo-chef` precomputes a dependency-only layer that's cached as long as `Cargo.toml` / `Cargo.lock` don't change. On any pure code edit, the Docker build skips ~60 min of dependency compilation and only rebuilds the few crates that actually changed.

**Risk: Green.** It's a Dockerfile change, no Rust code touched. Validates by spot-checking that a code-only change produces a binary that passes the test suite and has the right `cargo tree` lock state.

**Estimated saving on Docker cold-ish builds: 30–50 min.** (Cold local builds unaffected.) This is the single biggest Phase A item for our actual workflow (which is Docker-on-Epsilon).

### A.4. Dependency hygiene

```bash
cargo install cargo-machete  # if not already present
cargo machete --with-metadata
```

Remove confirmed-unused workspace deps in small PRs. Add a CI check (or pre-commit) so drift doesn't return.

**Risk: Green.** Mechanical removal. Each removal is either obviously safe (no `use` references in the crate) or it's a compile error (caught immediately).

**Estimated saving: 1–3 min on cold, marginal on incremental.** Worth doing as ongoing hygiene.

### A.5. Per-crate `[profile.release.package.*]` overrides

Apply to clearly non-hot crates only:

```toml
[profile.release.package.crown-ash-narrative]
codegen-units = 16

[profile.release.package.q-tui]
codegen-units = 16

# ... etc, ONLY for crates verified non-hot
```

**Keep `codegen-units = 1`** on:
- `q-storage`, `q-types`, `q-api-server` (consensus + runtime hot)
- `q-network`, `q-vdf`, `q-quantum-crypto` (network + crypto hot)
- `blake3`, `pqcrypto-*` (mining + signature hot)

**Risk: Yellow.** Lowering codegen-units on a hot crate causes a measurable throughput regression. The override list must be audited before shipping. Validation: mining-throughput micro-benchmark, request-latency percentile check, blake3 throughput test.

**Estimated saving: 3–5 min on cold builds. Marginal on incremental.** Smaller than V1 claimed because monomorphization of hot-crate generics happens in the hot crate's codegen unit anyway.

### A.6. `lazy_static!` → `std::sync::LazyLock` (stable since 1.80)

Grep for `lazy_static!`, migrate one-by-one. Keep changes small, one batch per PR.

**Risk: Yellow.** `LazyLock` uses `OnceLock` semantics rather than `lazy_static!`'s `Once`+`spinlock`. For cold paths (logging, config) the difference is invisible. For hot paths (which shouldn't be using `lazy_static!` anyway), audit each site. The reviewer flagged this risk correctly.

**Estimated saving: minor (1–2 min on cold).** Mostly a tech-debt cleanup that compounds with later phases.

### A.7. `main.rs` → `lib.rs` source split (reframed)

V1 claimed 5–10 min incremental savings. V2 reframes: the source split improves incremental-invalidation boundaries, reduces parser/typecheck churn in the bin target on rare edits limited to startup glue, and **prepares for Phase C crate splitting**. We will not claim guaranteed time savings.

**Risk: Yellow (downgraded from Green).** Symbol ordering can shift, but behavior cannot. Validation drops "binary-byte-identical" in favor of:
- `cargo test --workspace --locked` clean
- Release `--locked` build clean
- Mainnet-safety test suite green
- Deterministic API snapshot diff on a known block range
- Balance-root + tip + wallet-state comparison under soak
- Mining throughput regression check
- No feature-set changes per `cargo tree -e features`
- No public symbol/API changes that downstream MCPs depend on

**Estimated saving: 0–5 min on incremental edits that touch only the orchestrator.** Most edits don't.

### A.8. Items removed entirely from Phase A

- ❌ **`-C share-generics=y` on stable release builds** — V1 listed this; it's nightly-only. Demoted to a dev-experiment, not in the release pipeline.
- ❌ **Cranelift backend as a "ship-ready" change** — V1 listed this; also nightly. Demoted to a separate experimental track for `cargo test` only, gated by toolchain-policy decision.

---

## Revised Phase B (v10.10.0 — next 2 weeks)

Largely unchanged from V1; reviewer's risk ratings adopted:

### B.1. Default-feature audit

Trim `default-features = false` on heavy deps, enable only what we actually use:
- `reqwest` (already partially trimmed; verify rustls-tls vs openssl)
- `chrono` (drop `clock` if not actively using `Local::now()`)
- `image` (formats we read only)
- `tracing-subscriber` (only the layers we use)

**Risk: Yellow.** A wrong feature-flag can silently change runtime behavior (e.g., chrono timezone fallback, serde derive mode). Each audit goes via PR with explicit before/after `cargo tree -e features` diff. Reviewer flagged this correctly.

### B.2. Trim unused derives (mechanical, but careful)

Before removing any derive:
1. `grep -r '{:?}' crates/` — every type used in `Debug` formatting needs `Debug`
2. `grep -r 'clone()' crates/` — every type cloned needs `Clone`
3. `grep -r 'Serialize\|Deserialize' crates/` — every serialized type needs serde derives, **including types serialized at runtime via reflection**. This last category is the trap: if a struct is only ever serialized into a runtime JSON-RPC body, the static grep won't find it.
4. Remove derives only in small PRs, soak gate each.

**Risk: Yellow.** `Serialize` and `Deserialize` are runtime-surface-affecting. The reviewer's amplification is correct.

**Estimated saving: 3–5 min on cold across the workspace.**

### B.3. `LazyLock` migration (moved into Phase A above)

Migrated to Phase A.6 because `LazyLock` has been stable since 1.80 (the reviewer's correction).

---

## Phase C (v10.11.0 — dedicated sprint, ~4 weeks)

Largely unchanged from V1, but with clearer reasoning:

### C.1. Crate-split `q-api-server`

This is **the real compile-time unlock**. The 25K-line `main.rs` (plus its module siblings) all link into the same release LTO unit. Touching any handler triggers a full bin re-link.

Proposed split:
- `q-api-server-core` — AppState + boot + signal handlers (~2K lines, rarely changes)
- `q-api-server-routes` — axum routes + handler glue (~6K lines)
- `q-api-server-sync` — turbo sync, gap fill, batch sync (~5K lines, hot path)
- `q-api-server-mining` — block production, miner-link, finality (~3K lines, hot path)
- `q-api-server-dex` — DEX/AMM (~3K lines)
- `q-api-server-bridges` — Bitcoin/Zcash/Ethereum/IronFish bridges (~3K lines)
- `q-api-server-bin` — `main()` only (~200 lines)

Each smaller crate compiles independently. The final bin still has to ThinLTO-link the whole tree, but incremental rebuilds only touch the changed crate. Most edits touch one crate.

**Estimated saving on incremental edits: 10–15 min on common touch patterns.**

**Risk: Yellow → could surface real refactoring bugs (globals, statics, trait implementations crossing crate boundaries). Plan: dedicated sprint, ~4 weeks, branch + 7-day soak before merge.

### C.2. Vendor or crates.io-pin the `candle` git deps

`Cargo.lock` pulls `candle-core` from a git URL. Every clean build refetches and recompiles. `cargo vendor` or pinning to a crates.io version solves this.

**Risk: Yellow.** Need to verify we don't depend on a feature only present in the git fork.

**Estimated saving: 2–3 min on cold.**

---

## Validation Strategy (Replaces V1's binary-diff)

For every shipped phase:

1. `cargo test --workspace --locked` — clean exit
2. `cargo build --release --package q-api-server --locked` — clean exit
3. Mainnet safety test suite (per CLAUDE.md "Mandatory Testing Protocol") — all green
4. Spin a fresh Docker container with the new binary, soak 24h on Alpha
5. Diff `cargo tree -e features` against prior version — only intended feature changes
6. API snapshot regression: query 5–10 known endpoints (`/api/v1/status`, `/api/v1/integrity/*`, etc.), compare bodies byte-for-byte against a stored golden
7. Replay a known block range, compare resulting `balance_root` against the live Epsilon `balance_root` at the same height
8. Mining throughput micro-benchmark: hash rate before/after must be within ±2%
9. Request-latency p50/p99 check on a hot endpoint (block-pack serve) — no regression

This replaces V1's "binary-byte-identical" goal which the reviewer correctly identified as too strong.

---

## Updated Expected Savings (Confidence-Weighted)

| Phase | Confidence | Saving on cold | Saving on incremental |
|-------|-----------|---------------|----------------------|
| A.2 (mold linker) | **High** | 3–8 min | 3–8 min |
| A.3 (cargo-chef in Docker) | High | 30–50 min in Docker | — |
| A.4 (machete) | Medium | 1–3 min | marginal |
| A.5 (per-crate codegen-units) | Medium | 3–5 min | marginal |
| A.6 (LazyLock migration) | Low | 1–2 min | marginal |
| A.7 (main.rs split) | Low–Medium | 0 | 0–5 min (rare touches only) |
| B.1 (feature audit) | Medium | 3–5 min | marginal |
| B.2 (derive trimming) | Medium | 3–5 min | marginal |
| C.1 (crate split) | **High** | 5–8 min | **10–15 min on common edits** |
| C.2 (vendor candle) | Medium | 2–3 min | — |

**Net realistic target:**
- Local cold build: 60 min → 35–40 min after Phase A, → 25–30 min after Phase C
- Local incremental: 30 min → 20–25 min after Phase A, → **5–10 min after Phase C on most edits**
- Docker rebuild (with `cargo-chef`): 60 min → **~3–5 min on code-only changes**

The Docker-with-chef number is the one that changes day-to-day developer experience most. That's what we should pursue first.

---

## What Reviewer Got Right and Where V2 Adopts the Correction

| Reviewer point | V2 action |
|---|---|
| `share-generics=y` is nightly | Removed from release pipeline |
| Cranelift is nightly | Demoted to dev-only experiment |
| Binary-byte-identical is too strong | Replaced with behavioral validation gates |
| Fat lib + thin bin saves less than claimed | Reframed as Phase C prep, no guaranteed savings |
| Mold linker is the real Phase A win | Added as new A.2 |
| `cargo-chef` for Docker builds | Added as new A.3 |
| `cargo build --timings` first | Added as new A.1 |
| `LazyLock` is stable since 1.80 | Kept in Phase A (was Phase B in V1) |
| Watch out for runtime serde derives | Explicit caveat added to B.2 |

---

## What This Is *Still* Not

Identical to V1: no consensus path changes, no crypto changes, no LTO disable, no codegen-units change on hot crates, no allocator switch.

---

## Recommended Order of Operations

1. **Today**: A.1 (measurement) + A.4 (machete dep removal). Both are commit-safe and produce data.
2. **This week**: A.2 (mold linker) + A.3 (cargo-chef Docker). The two biggest real wins.
3. **Next week**: A.5 + A.6 + A.7, soak-gated per change.
4. **Sprint after**: Phase B feature audit.
5. **Quarterly**: Phase C crate split as a dedicated release.

The Docker-side win from `cargo-chef` (A.3) is the single biggest operational improvement to our workflow. The mold linker (A.2) is the single biggest real-time saving on the bin link itself. The two together should drop the inner-loop dev experience from "30 min per soak iteration" to "5–10 min per soak iteration" without touching a single line of mainnet code.

— Server Beta, 2026-05-13
