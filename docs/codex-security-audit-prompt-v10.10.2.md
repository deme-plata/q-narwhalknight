# Codex security audit prompt — v10.10.2 (post-recovery)

**Branch to audit:** `mirror/v10.10.2` on `github.com/deme-plata/q-narwhalknight` (or `agent/cross-shard-simd-validation` if accessing local-git)
**Date:** 2026-05-20
**Trigger:** 45 commits added in a single recovery session, 28 of 29 Codex PRs cherry-picked onto the buildable lineage. The recovery is documented in `docs/development-workflow-going-forward.md` and TR-2026-008 / TR-2026-009.

## What we want from you

A focused security audit of the **delta** between `agent/cross-shard-simd-validation` @ `514ad291` (pre-recovery baseline) and HEAD (`mirror/v10.10.2`). 45 commits, 28 PRs of content, three concrete categories of risk we need eyes on:

1. **Cherry-pick correctness** — Did any commit lose intent during cherry-pick? Auto-merges happened on `crates/q-api-server/src/handlers.rs`, `crates/q-network/src/unified_network_manager.rs`, `crates/q-api-server/src/main.rs`. Manual conflict resolutions on `crates/q-api-server/src/agent_panel/mod.rs` (kept rescue's full docstring + activated PR #100's `pub mod scorers;`) and `Cargo.lock` (kept ours).

2. **Auth + balance surface changes** — Five PRs in this delta touch authentication or balance correctness paths. We need verification each works as intended and doesn't introduce a new bypass.

3. **New attack surface from added features** — Scorer chain (PRs #100-103), Twitter MCP (PR #88, #95), AFL-1 (PR #87, #91), Agent Activity Panel (PR #90), QSHARE-1 (PR #96), WebGPU miner PoC (PR #94 partial), SSE auth-via-query (PR #106). Each is a new endpoint, payload field, or signing path. Confirm none accepts unauthenticated input that touches balance, validator quorum, or admin state.

## What's IN the delta (high-trust changes — confirm they did what they claimed)

| Commit | PR | What it changes | Audit concern |
|---|---|---|---|
| `039b48bc` | #66 | `save_wallet_balance` singular — adds `return Ok(());` after max-wins error log so stale callers can't overwrite higher on-disk balance | Verify the early-return short-circuits before `put_sync()`. Confirm no other writer to `wallet_balance_*` keys exists that bypasses this guard. |
| `6ba9e068` | #67 | AEGIS local-admin: replaces `unwrap_or(true)` with `unwrap_or_else(|| env_var == "1")` for the `Q_ENABLE_LOCAL_ADMIN_BYPASS` env opt-in | Confirm no other code path admits `X-Admin-Local` without going through `aegis_auth_middleware::verify_founder_signature`. Confirm the env var is the only opt-in. |
| `fda33515` | #68 | `ProductionMempool::perform_validation` — replaces `Ok(true)` stub with `is_coinbase` carve-out + `verify_signature` + `validate_fee` | Confirm there's no other path into the mempool that skips this validator. Confirm `is_coinbase()` correctly identifies the carve-out (no spoofable). |
| `776e0a65` | #92 | `X-Wallet-Auth` body_hash extension — adds optional `X-Wallet-Body-Hash` header that gets folded into the signed payload | Verify that when present, the chain checks `sha3-256(received body) == hash` before signature verification. Verify backwards-compat when header absent. |
| `038a316e` | #106 | SSE auth-via-query — browser EventSource can't send headers, so auth signature is passed as `?auth=<base64>` query param | Verify query param is constant-time-compared to expected. Verify the URL with the auth blob is not logged (or is redacted). Verify no replay window past timestamp tolerance. |

## What's NEW in the delta (validate the surface)

| PR | New endpoint(s) / module | Audit concern |
|---|---|---|
| #98 | `GET /api/v1/p2p/known-peers` (public, read-only) | Confirm read-only enforcement. Confirm no metadata leak about peer reputation/private state. Self-healing peer registry — what's the auth model for entries getting auto-added vs evicted? |
| #100 | `crates/q-api-server/src/agent_panel/scorers/mod.rs` (142 LOC) | Scorers are pure functions of inputs → output a `ScoreReport`. Confirm no scorer reads from the wallet store or anything mutable. Confirm score values are bounded (no negative or NaN injection). |
| #101 | tx response now includes optional `score` field | Confirm score is computed from tx alone (not from caller-supplied data that could be spoofed). |
| #102 | swap response includes optional `score` field | Same as above. Confirm scorer for swap doesn't grant any preferential treatment based on score (i.e., scoring is informational only). |
| #103 | `gui/quantum-wallet/src/components/ScoreBreakdown.tsx` UI | Pure render of server-supplied score. Confirm no XSS surface in the breakdown labels. |
| #87 | `crates/q-trading-bot/src/bin/send_with_memo.rs` (CLI binary), `AGENT.md`, `docs/agent-fiber-lane-spec.md`, `docs/openapi.yaml` | The OpenAPI 3.1 spec is a contract — confirm it accurately reflects the implementation and doesn't promise more auth than is enforced. The `send_with_memo` binary signs with the agent wallet — confirm it doesn't expose the seed. |
| #91 | `docs/standards/afl-1-protocol-spec.md` (BIP-style spec) | Pure spec doc, no code. Verify the spec doesn't claim behaviors that the implementation doesn't deliver. |
| #88, #95 | `tools/quillon-twitter-mcp/` + Rust sidecar `tools/quillon-twitter-mcp/crates/x-algorithm-scorer/` | Confirm the Twitter MCP TS server only signs transactions with an explicitly-passed signing key (never embedded). Confirm the Rust scorer is deterministic and has no side effects. |
| #90 | `docs/agent-activity-panel-spec.md` | Spec doc. Verify the trust-tier indicator (🟢 local / 🔵 signed-by-X / 🟣 delegated-via-fiber-lane) maps to actual auth verification, not just label assertions. |
| #96 | QSHARE-1 protocol + `qcredit-dca` CLI subcommand on q-trading-bot | The CLI takes `--wallet`, `--amount`, `--tier` (bronze/silver/gold/platinum), `--interval`, `--min-balance-floor`, `--max-cycles`, `--dry-run`. Confirm the trading bot signs all transactions via X-Wallet-Auth (not embedded seeds) and respects `--min-balance-floor` (never drains the wallet below it). Confirm `--dry-run` is honored everywhere. |
| #94 (partial) | `gui/webgpu-miner/` PoC + `gui/quantum-wallet/src/components/{WebGpuMinerModal,AgentTerminalModal}.tsx` | Browser-native SHA3-256 mining. Confirm the WebGPU side cannot exfiltrate the mining-share signing key beyond browser storage. Confirm the AgentTerminalModal does NOT accept arbitrary shell commands. |
| #93 | Rescued files: `RecentActivityPanel.tsx` (32 KB), `q-trading-bot/src/{wallet_auth,observer}.rs`, `slint-wallet/src/config.rs`, integration tests | The 32 KB activity panel renders Transaction objects from API — confirm no XSS via tx memo fields. The trading-bot `wallet_auth.rs` is a real Ed25519 signer — confirm the seed never crosses a process boundary. |

## What's NOT in the delta (still-deferred work — call out gaps if you find them)

| Item | Status | Auditor action |
|---|---|---|
| PR #44 docs (security-audit/ISSUES.md, PR44_VERIFICATION_QUESTIONS.md, technical-review-private-crypto-audit) | Deferred — base docs not yet in tree | If the base docs include audit findings that should have been integrated, flag them. |
| PR #94 deferred 4 commits (`6a5b70d2d` session unblockers, `9a4656870` MCP v1.3.0 agent-mode signing, `4c0b6c12e` v10.10.2 out-of-box defaults, plus 3 v10.10.1 wire-up fixes) | Deferred — main.rs 135-LOC conflict on lattice-tip producer-hook | Especially `9a4656870` MCP v1.3.0 has security implications (agent-mode signing on balance/mining reads). Flag if this is critical to land. |
| PR #97 (multisig brief) + PR #99 (slint wallet settings) | Need per-commit triage; labelled "docs" but have feature-branch baseline | Inspect each PR — distinguish docs vs code. Flag any code-changes that should ride a future release. |

## Cross-cutting checks to run

1. **Run the cargo build** (`cargo build --release --package q-api-server --package q-miner` on Epsilon Docker / Debian 12). The v10.9.58 build succeeded; we have not yet built v10.10.2. If any cherry-pick combination broke the build, surface that here.

2. **Run the mainnet safety test suite** (the 125+ critical tests per CLAUDE.md). Specifically:
   - `cargo test --package q-storage --test mainnet_critical_tests`
   - `cargo test --package q-storage --test balance_integrity_tests`
   - `cargo test --package q-types --test signature_verification_tests`
   - `cargo test --package q-storage --test sync_down_protection_tests`

3. **Spot-check the 3 auth-touching commits in the binary** — string-grep for the new log lines:
   - `SKIPPED (max-wins:` from PR #66
   - `Q_ENABLE_LOCAL_ADMIN_BYPASS=1 for unix-socket deployments` from PR #67
   - `[MEMPOOL] reject: signature invalid` from PR #68
   - All three were verified present in the v10.9.58 binary; should also be in v10.10.2.

4. **Verify GPG signing chain** — all 45 new commits should show `G` (good signature) in `git log --pretty='%h %G?' -45`. If any show `E`, `N`, `B`, that's a flag.

5. **Confirm no committed secrets** — Run secret-detection on the 45 commits. Earlier check found nothing matching `glpat-`, `ghp_`, or `sk-` patterns, but a broader scan (`detect-secrets` or `gitleaks`) is worth running.

## Output format we'd like

For each finding:
- **Severity:** CRITICAL / HIGH / MEDIUM / LOW / INFO
- **Affected file(s) + line(s):** Concrete pointer
- **Summary:** One sentence
- **Evidence:** Diff snippet or specific behavior observed
- **Recommended action:** Whether to fix in-place, defer to a follow-up PR, or document as accepted risk

Aggregate findings into:
- `docs/codex-audit-findings-v10.10.2.md` — full report
- A summary table at the top of the report (severity counts)

If anything is **CRITICAL** and a fix is straightforward, open a PR with the fix and reference this prompt. The cherry-pick discipline applies: target `mirror/v10.10.2` or the next release line.

## Known good outcomes (don't flag these)

- Author attribution mix: many commits are authored by "Server Beta" while others are "Viktor S. Kristensen" or "Viktor". Both are the project owner; both signatures verify against `79A42E8E0ACF2EBB68493D8AD46383A143940FC4`.
- Cargo.toml comment growth: the workspace.package version comment accumulates release notes across versions. This is by convention.
- The `tools/quillon-wallet-mcp/package-lock.json` shows as modified in working tree but isn't part of our delta — leftover from earlier sessions.

## Reference materials

- `docs/development-workflow-going-forward.md` — the development doctrine
- `docs/technical-reviews/TR-2026-008-branch-hygiene-and-v10957-recovery.md`
- `docs/technical-reviews/TR-2026-009-release-branch-source-binary-mismatch.md`
- `docs/agent-fiber-lane-spec.md` — implementation contract for AFL-1 (PR #87)
- `docs/standards/afl-1-protocol-spec.md` — BIP-style standard (PR #91)
- `docs/openapi.yaml` — REST API contract (PR #87)
- `CLAUDE.md` — operations playbook (includes balance integrity non-negotiables)

Thanks. The recovery story this audits is in TR-2026-008 / TR-2026-009; the workflow that produced these 45 commits is in `docs/development-workflow-going-forward.md`. Both are required reading before starting.
