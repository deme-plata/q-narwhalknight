# Codex Validation Tasks — 2026-05-17

Two read-only validation tasks for Codex. Each is self-contained and produces a CRITICAL/HIGH/MEDIUM/LOW punch list. Run them independently.

## Task 1 — BalanceRootV1 (20M activation) + BalanceRootV2 (SMT shadow)

**File:** [`codex-validation-balance-root-2026-05-17.md`](codex-validation-balance-root-2026-05-17.md)

**Urgency:** HIGH — BalanceRootV1 enforcement activates at block height **20,000,000**. Current tip is ~18.1M, leaving roughly **22 days** at 1 bps. Any determinism crack discovered here must be fixed before the activation block or the network forks.

**Scope:** `crates/q-consensus-guard/src/upgrade_gate.rs`, `crates/q-storage/src/balance_smt.rs`, `crates/q-storage/src/lib.rs` (balance-root sections), activation tests, DeepSeek handoff doc.

**Output:** ranked punch list; explicit confirmation if no CRITICAL/HIGH issues are found.

## Task 2 — `tip-blake3-fs-v1` 10ms tip proof

**File:** [`codex-validation-tip-proof-2026-05-17.md`](codex-validation-tip-proof-2026-05-17.md)

**Urgency:** MEDIUM — already serving in production (`/api/v1/proof/tip`). DeepSeek reviewed §0 forgery (fixed v10.9.41) and §8 step-count length-extension (fixed v10.9.51). This task asks for an independent second pass — find what DeepSeek missed plus validate the existing fixes are complete.

**Scope:** `crates/q-recursive-proofs/src/tip_proof_v1.rs` and consumers/producers, existing review doc, spec.

**Output:** ranked punch list; do not repeat DeepSeek findings.

## How to run

Each prompt file is self-contained — paste the contents into Codex. The prompts include the codebase root, the file:line locations to audit, the specific questions to answer, and explicit out-of-scope guardrails. No additional context required.

Recommended order: Task 1 first (time-critical), Task 2 second.
