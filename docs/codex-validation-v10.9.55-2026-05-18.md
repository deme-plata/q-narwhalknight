# Codex Validation Prompt — v10.9.55

**Repo:** `https://github.com/deme-plata/q-narwhalknight`
**Branch:** `release/v10.9.55`
**Mode:** READ-ONLY validation. Find bugs, regressions, edge cases, and missed mitigations BEFORE compile. Do NOT modify code. Produce a ranked CRITICAL / HIGH / MEDIUM / LOW punch list.

## Background

v10.9.55 ships three classes of changes at once:

1. **Consensus-critical pre-20M fixes** from your earlier review (2026-05-17 of `docs/codex-validation-balance-root-2026-05-17.md`): C1 fail-closed on BalanceRoot computation error, C2 SMT key collision fix, C3+H4 SMT rebuild determinism.
2. **Defense-in-depth** against the Mar 2026 kill-9-during-compaction loss pattern: RocksDB `force_consistency_checks` + `paranoid_file_checks` plus a systemd-hardening operator runbook.
3. **Sync sparse-chain awareness**: per-height presence index in CF_MANIFEST + new `synced_through` pointer so the sync loop advances past dead heights instead of wedging.

Context document: `docs/technical-review-sparse-chain-truth-v1.md` (this branch) — the empirical evidence that the chain is sparse-by-design (DAG-Knight produces blocks at ~93-96% of heights post-15M) plus historical pre-7M damage from compaction loss + v10.2.8 cleanup bug. Verify that the v10.9.55 code aligns with the doc's claims.

The 20,000,000 BalanceRootV1 enforcement activation is ~1.88M blocks away (~22 days at 1 bps). Any consensus-fork hazard found here MUST be fixed before that block, or the network forks.

## Files to audit (in priority order)

| File | What changed | What to check |
|---|---|---|
| `crates/q-api-server/src/main.rs` lines ~12506-12560 | C1: BalanceRootV1 enforcement fail-closed + 3-attempt retry with 100/500/2000ms backoff | (a) Does the retry actually retry on the right error class? (b) Could the retry mask a legitimate bug? (c) Is the total ~2.6s retry budget safe for the block-receive loop, or does it back up the queue? (d) Is the fail-closed branch reachable for ALL post-20M paths, or are there other call sites where computation fails open? |
| `crates/q-storage/src/balance_smt.rs` lines ~38-75 (constants), ~98-145 (node_key) | C2: domain-byte key encoding (b'N' nodes / b'L' leaves) | (a) Can ANY two (depth, addr) pairs still collide? Specifically check depth=255 vs leaf. (b) Migration: existing DBs may have old-format SMT keys from shadow mode runs. Does the rebuild path correctly clear them, or do they coexist in CF_BALANCE_SMT? (c) Is `KEY_PERSISTED_ROOT = b"R__root__"` distinct from any possible node_key output? (d) byte-identical roots required across nodes — confirm the new encoding is deterministic across endianness/build flags. |
| `crates/q-storage/src/balance_smt.rs` lines ~270-310 (rebuild_from_balances) | C3 + H4: truncate via delete_range_cf + sort updates | (a) Is `[0x00u8]` to `[0xFFu8; 64]` a safe range that covers all keys without touching unrelated CF data? (KEY_PERSISTED_ROOT 'R'=0x52 is inside the range — confirm intentional.) (b) Is delete_range_cf atomic with the rebuild writes, or could a crash mid-rebuild leave the CF inconsistent? (c) `sort_unstable_by_key(|(addr, _)| *addr)` — confirm `[u8; 32]` lexicographic ordering is byte-identical across all platforms. |
| `crates/q-storage/src/kv.rs` lines ~745, ~824, ~1340 | CF hardening: `force_consistency_checks(true)` + `paranoid_file_checks(true)` on CF_BLOCKS, CF_TRANSACTIONS, CF_QUANTUM_METADATA | (a) Is `set_paranoid_file_checks` available in rust-rocksdb 0.22? If not, the build fails. (b) Startup cost: with 48K SST files on Epsilon's CF_BLOCKS, how long does paranoid_file_checks take? Is the startup-time impact acceptable, or should it be gated by an env var for emergency cold-start? (c) Does force_consistency_checks change existing-DB-open behaviour in a way that could reject a legitimate but historically-imperfect DB? |
| `crates/q-storage/src/lib.rs` lines ~443-460 (HEIGHT_PRESENT_PREFIX), ~1369-1373, ~1685-1687 (marker writes), ~1964-1990 (is_height_present) | Task 3: per-height presence markers in CF_MANIFEST | (a) Are the marker writes atomic with the qblock:height: write? Confirm they share the same WriteBatch. (b) Tier-2 fallback in is_height_present checks `qblock:height:N` — does this give correct answers for the DAG-format-only old blocks documented in the sparse-chain review? (c) Key format `h_present:{:011}` — is 11 digits enough? 99,999,999,999 is plenty, but confirm no off-by-one. |
| `crates/q-storage/src/lib.rs` lines ~698-714 (struct field), ~824 (init), ~837-855 (load), ~1996-2050 (getter + advance) | Task 4: synced_through pointer | (a) On a clean upgrade from v10.9.54 → v10.9.55, the qblock:synced_through key doesn't exist. The init logic uses `persisted.max(initial_height)` — confirm this never regresses the pointer relative to actual sync progress. (b) `advance_synced_through` uses CAS loop + atomic load + persist. Race condition: thread A advances to 100, persists; thread B advances to 200, persists. Could B's persist hit RocksDB before A's, leaving the on-disk value at 100 when memory says 200? Examine the ordering. (c) Best-effort persist on failure — confirm acceptable risk envelope. |
| `crates/q-storage/src/turbo_sync.rs` lines ~7003-7030, ~7654-7665 | Task 4: synced_through used in sync_to_height | (a) `local_height = synced_through` if synced_through > local_height — does this break the safety check at line 7019 (`target_height < local_height && local_height > 1000`)? If synced_through is ahead of contiguous, target may legitimately be lower than synced_through during a tip-recheck. (b) Does the known-gap auto-advance at line 7071+ interact correctly with the new pointer? (c) advance_synced_through at end of sync_to_height: is this called too early (before the block-pack-receive task has actually persisted)? |
| `Cargo.toml` line 85 | Version bump 10.9.54 → 10.9.55 | Sanity: matches the running-version checks the deploy script enforces (`ha-deploy.sh` rejects if version matches running Beta version). |
| `docs/technical-review-sparse-chain-truth-v1.md` | New strategic doc | Cross-check the doc's claims against the code: does the patch list in section 6 actually match what shipped? Are the "things to never do again" rules enforced by code (where possible)? |
| `docs/v10.9.55-systemd-hardening.md` | Operator runbook | Cross-check: are the proposed systemd directives compatible with systemd 245+ (Debian 11/12 ship with this)? Is `SendSIGKILL=yes` actually a valid directive (or is it implied by TimeoutStopSec elapsing)? |

## Specific questions to answer

### CRITICAL — could fork the network or cause data loss

**Q1.** Can C1's retry-with-backoff ever produce a different verdict than no-retry on the same input? E.g., a non-deterministic `compute_balance_root_for_block` that returns different results on subsequent calls. If yes: which call's result wins, and is that the correct one for consensus?

**Q2.** With C2's new key encoding, do existing SMT keys from BalanceRootV2 shadow-mode runs (if any) become orphaned in `cf_balance_smt`? Confirm the rebuild path (`rebuild_from_balances` with `delete_range_cf`) is invoked BEFORE any new query that depends on the new encoding.

**Q3.** Does `force_consistency_checks=true` change the open semantics on a v10.9.54 DB in a way that could refuse a healthy DB? Specifically: does the existing CF_BLOCKS data on Beta/Gamma/Delta/Epsilon pass force_consistency_checks, or would the v10.9.55 upgrade brick the DB?

**Q4.** synced_through advances on `sync_to_height` Ok(()) return — but what about the case where `sync_to_height` short-circuits early (line 7005 "already synced") without doing any work? Does the advance still happen, or does it stall the next iteration?

### HIGH — performance regressions or operational hazards

**Q5.** `paranoid_file_checks` startup cost on Epsilon's 48,306-SST CF_BLOCKS: estimate. Is it minutes or seconds? If minutes, this is an operator surprise — gate behind env var.

**Q6.** Marker writes add one extra WriteBatch put per block. At 8.17M existing blocks (none have markers yet) plus 18.1M+ new blocks expected over the next year (per the user's "100M-blocks-in-a-year" projection), is the CF_MANIFEST size impact acceptable? Estimate.

**Q7.** Tier-2 fallback in `is_height_present` makes a synchronous `db.get` on CF_BLOCKS for pre-v10.9.55 blocks. Is this on the API hot path? If yes, fresh-node bootstrap could be slower than today until markers are backfilled.

**Q8.** Does the systemd diff in `v10.9.55-systemd-hardening.md` apply cleanly to the existing service file on each server? Specifically: KillMode=mixed conflicts with any other directive? Does Epsilon's NON-standard service file (binary path `/opt/orobit/.../q-api-server-v889`) need special handling?

### MEDIUM — code quality / coverage

**Q9.** Is there a regression test for the C2 collision? Specifically: a test with an address ending `& 0x03 == 0` that exercises both depth-254 and depth-256 paths and asserts they produce distinct keys?

**Q10.** Is there a determinism test for C3+H4? Two engines, identical wallet maps in HashMaps with different hash seeds, rebuild both, assert byte-identical roots?

**Q11.** Is there a test that synced_through survives restart? Init from disk, advance, kill node, reopen, confirm in-memory value matches what was persisted?

### LOW — informational / spec drift

**Q12.** Does the sparse-chain-truth doc's percentage figures (93-96% present in 15M-18M, ~3% in 0-7M) reproduce when re-running the documented `ldb` scan commands today? Bit-rot check.

## Output format

Markdown punch list:

```
## CRITICAL — must fix before merge or compile
- [issue title] — file:line — risk description — proposed mitigation

## HIGH — performance/operational hazards
...

## MEDIUM — code quality / test coverage gaps

## LOW — informational / spec drift
```

If zero CRITICAL/HIGH issues are found, say so explicitly with the evidence trail (which paths you traced, which determinism hazards you eliminated). Do not pad findings.

## What I am NOT asking you to do

- Don't rewrite anything — propose edits only, in the punch list.
- Don't audit unrelated subsystems (DEX, mining, P2P) — only the files listed above.
- Don't repeat findings from your 2026-05-17 BalanceRoot review unless they were missed in implementation; reference them by section instead.
- Don't propose architectural changes — v10.9.55 is a focused release, not a rewrite. Note structural issues but propose minimal fixes for this version.

## Out of band

After this validation pass, the operator will compile-check the branch on Epsilon Docker (Debian 12 / rust:bookworm) and run the existing 4000+ test suite via `scripts/safe-deploy.sh test-all`. Any CRITICAL findings here must be fixed before that compile to avoid wasting a 25-minute cycle.
