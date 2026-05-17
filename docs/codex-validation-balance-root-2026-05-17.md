# Codex Validation Prompt 1 — BalanceRootV1 (20M activation) + BalanceRootV2 (SMT shadow)

**Repo:** `/opt/orobit/shared/q-narwhalknight` (Q-NarwhalKnight, Rust workspace, mainnet running)
**Mode:** READ-ONLY validation. Find soundness gaps, determinism bugs, and activation hazards. Do NOT modify code. Produce a punch list ranked CRITICAL / HIGH / MEDIUM / LOW.

## Background

Two related consensus features:

1. **BalanceRootV1** — activates at block height **20,000,000** on mainnet (currently dormant; tip ~17.67M; ~15 days away). After activation, `BlockHeader.balance_state_root` MUST equal a flat-hash commitment over all wallet balances; verifiers reject mismatches.

2. **BalanceRootV2** — Sparse Merkle Tree (SMT) commitment. Currently DORMANT (`u64::MAX`). Runs in shadow mode (`Q_BALANCE_ROOT_V2_SHADOW=1`): node side-computes the SMT root each block and logs `MISMATCH` if it disagrees with v1. Activation gated on: (a) one-week zero-MISMATCH soak, (b) cross-node determinism, (c) reorg correctness, (d) activation rebuild path verified.

Mainnet integrity is non-negotiable — see `CLAUDE.md` "BALANCE INTEGRITY — NON-NEGOTIABLE RULES" (an old replay bug destroyed Epsilon's 3200→1484 QUG). Any determinism crack here = consensus split → all user funds at risk.

## Files to audit (in priority order)

1. **`crates/q-consensus-guard/src/upgrade_gate.rs`** lines 1–220 — Upgrade enum, activation heights, MAINNET_UPGRADES schedule. Confirm BalanceRootV1=20,000,000 mainnet, BalanceRootV2=u64::MAX. Look for: off-by-one at activation boundary, network-ID confusion (testnet vs mainnet schedule swap), missing upgrade-gate consultation at any consensus call site.

2. **`crates/q-storage/src/balance_smt.rs`** (743 lines) — Full SMT implementation. Audit deeply:
   - `leaf_hash_raw`, `node_hash_raw`, `precompute_empty_subtrees` — hash domain separation, big-endian vs little-endian balance encoding, empty-subtree precomputation correctness (do all 256 levels initialize consistently across nodes regardless of build flags?)
   - `addr_bit`, `flip_bit`, `node_key`, `sibling_key` — address ordering (MSB-first vs LSB-first must match across all nodes), key collision risk between depths
   - `apply_to_batch`, `apply_to_batch_internal`, `apply_one` — batch ordering: if two updates target the same address, which wins? Sorted by address? By insertion order? Could a HashMap iteration order leak in?
   - `update_batch`, `fold_to_root`, `fold_to_root_writing` — root recomputation must be byte-identical regardless of WriteBatch internal ordering
   - `rebuild_from_balances` — used at activation height; if N nodes call this with the same wallet snapshot, MUST produce byte-identical root. Look for: HashMap iteration, Vec ordering, signed/unsigned conversion, locale-sensitive ops
   - `prove`, `verify` (SmtProof) — soundness of proof verification

3. **`crates/q-storage/src/lib.rs`** lines 4556–4750 — `rebuild_balance_smt_from_wallet_table`, `compute_balance_state_root_v1`. Confirm:
   - V1 hash function: `// Do NOT change endianness — it would invalidate BalanceRootV1` (line 4730) — check the canonical byte layout
   - V1 wallet iteration ordering: is it a sorted scan, or RocksDB iterator order? (RocksDB iterator order IS deterministic within a single DB, but across nodes only if everyone inserts the same keys — verify this assumption)
   - The `BalanceSmt::open` path (line 776) — does it correctly persist root across restarts?

4. **`crates/q-storage/tests/balance_root_v2_activation_tests.rs`** — Read the existing test coverage. Identify what scenarios are NOT tested.

5. **`crates/q-api-server/tests/state_root_activation_tests.rs`** — Same.

6. **`docs/deepseek-handoff-balance-root-v2-activation-2026-05-14.md`** — DeepSeek's design handoff. Note its claims and check whether the code matches.

## Specific questions to answer

### Determinism (CRITICAL — any failure = consensus split)
- D1. If 4 nodes (Beta/Gamma/Delta/Epsilon) call `rebuild_balance_smt_from_wallet_table()` on identical wallet tables at activation height (20M), do they produce byte-identical roots? Trace every `HashMap`, `unordered iter`, `f64`, or platform-dependent op in the call graph.
- D2. Is wallet-table iteration order guaranteed across RocksDB versions and across nodes? Sort by address before SMT input?
- D3. Are balance values stored as u128 (uniformly little/big endian) or is there any signed/unsigned coercion in the leaf-hash path?

### Activation boundary (CRITICAL)
- A1. Block at height 19,999,999 with v1-only header — accept? Block at 20,000,000 with v1 header (no v2) — accept under V1 enforcement? Block at 20,000,000 with v1+v2 — accept?
- A2. What happens during a reorg across the 20M boundary? If chain forks at 19,999,998, does the rebuild get triggered correctly on both branches? Is `safe_floor` updated atomically with the SMT root?
- A3. Cold start at height 20,000,001 on a fresh node — does the bootstrap path (turbo_sync) correctly compute the SMT root from genesis, or does it skip and trust the checkpoint? If checkpoint-trusting, is the checkpoint's SMT root validated independently?

### Shadow-mode soak (HIGH — needed before activation)
- S1. The MISMATCH log line: does it include both v1 and v2 root values so we can diagnose disagreements?
- S2. Is the shadow computation atomic with the canonical v1 path? Could a `save_wallet_balances` race with `BalanceSmt::apply_to_batch` produce a transient MISMATCH that isn't a real bug?
- S3. If shadow mode is OFF on a node and ON on its peer, does ANYTHING about block validation/propagation diverge between them? (Must be a pure observer — NO side effects on consensus.)

### Reorg correctness (HIGH)
- R1. On a multi-block reorg that undoes wallet-balance writes, is the SMT also rewound? If so, where? If not, after reorg does SMT root match the wallets?
- R2. Does the SMT persist its root only on commit, or eagerly? If eagerly, a crash mid-reorg could leave SMT in an inconsistent state.

### Attack vectors (MEDIUM)
- ATK1. A malicious producer crafts a block with valid v1-root but wrong v2-root. Post-activation: rejected. Pre-activation: shadow logs MISMATCH but accepts → does the attacker gain anything by polluting MISMATCH logs?
- ATK2. SMT collision on `node_key(depth, addr)`: is depth folded into the key with sufficient domain separation? Could a malicious wallet pick an address that collides with an inner-node key at a different depth?
- ATK3. Empty-subtree precomputation: if a node's `precompute_empty_subtrees()` ever returns different bytes from another node's (e.g., due to build-flag or library-version differences), the SMT root diverges. Is there a startup self-check that the precomputed table hashes to a known constant?

## Output format

Markdown punch list:

```
## CRITICAL — must fix before 20M activation
- [issue title] — file:line — description — proposed mitigation

## HIGH — must fix before V2 activation
...

## MEDIUM — improvements

## LOW — nits and questions
```

If you find ZERO critical/high issues, say so explicitly with the evidence trail. Don't pad findings.

## What I am NOT asking you to do
- Don't rewrite the SMT.
- Don't add new tests (just note which scenarios lack coverage).
- Don't propose architectural changes; this code is shipping in ~15 days.
- Don't audit unrelated subsystems (sync, networking, DEX) — only the balance-root code path.
