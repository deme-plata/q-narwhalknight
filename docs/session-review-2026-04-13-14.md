# Session Review: 2026-04-13 to 2026-04-14

**Duration:** ~18 hours continuous  
**Network:** Q-NarwhalKnight mainnet-genesis ($1B market cap)  
**Operator:** Demetri  
**AI Assistant:** Claude Code (Opus 4.6)  
**Final State:** Network stable, balances restored, critical bugs fixed

---

## What We Accomplished

### Bugs Found & Fixed

| Bug | Severity | Root Cause | Fix | Status |
|-----|----------|-----------|-----|--------|
| **Balance reorg handler** | CRITICAL | `perform_balance_reorg()` uses `saturating_sub` which zeros balances when old_reward > current_balance. Pre-existing since v0.9.31. | Disabled with `if false {}` | **DEPLOYED** (v10.3.3) |
| **DEX double deduction** | CRITICAL | Swap handler deducts directly from RocksDB AND submits consensus tx which deducts again. Every swap loses 2x. | Removed direct deduction, let balance_consensus handle via Swap tx | **DEPLOYED** (v10.3.2+) |
| **Ghost 4200 QUG on startup** | MEDIUM | 10-second window between RocksDB load and authority sync shows stale values | `startup_sync_complete` flag, balance API returns null during sync | **DEPLOYED** |
| **Epsilon memory stall** | HIGH | Jemalloc arena fragmentation: 41GB retained dirty pages from high-throughput block/sync processing | Root cause identified, per-arena purge fix designed | **NOT YET CODED** |
| **Hashrate crash on restart** | HIGH | q-flux had no cluster peers configured, all miners lost connection during restart | Added Beta/Gamma/Delta as failover peers in q-flux.toml | **DEPLOYED** |

### Features Deployed

| Feature | Status | Notes |
|---------|--------|-------|
| **LWMA difficulty adjustment** | LIVE on Epsilon + Beta | Block rate converging from ~3.46 to 1.0 bps. Activation height 14,900,000 (already passed). |
| **Phase A: Difficulty-weighted rewards** | LIVE | Harder solutions earn proportionally more. |
| **DEX safety gate** | LIVE | `dex_ready` flag blocks swaps during bootstrap. |
| **Balance debug logging** | LIVE | `🔴 [BALANCE WRITE]` on all 16 write paths. Caught the reorg handler bug. |
| **Persistent processed-block dedup** | LIVE | `processed_balance_block:{hash}` keys in CF_MANIFEST survive restarts. |
| **q-flux cluster failover** | LIVE on Epsilon | Miners route to Beta/Gamma/Delta during Epsilon restarts. |

### Documents Written

| Document | Path | Purpose |
|----------|------|---------|
| DEX balance corruption v1-v6 | `docs/technical-review-dex-balance-corruption-v*.md` | Root cause analysis (evolved through 6 versions as understanding deepened) |
| Balance restoration emergency | `docs/technical-review-balance-restoration-emergency.md` | Plan A (chain rebuild) and Plan B (Beta authority sync) |
| Plan B: Beta restore | `docs/technical-review-plan-b-beta-balance-restore.md` | Final approved approach |
| Epsilon memory stall | `docs/technical-review-epsilon-stall-uptime-v3.md` | Jemalloc fragmentation analysis |
| Startup ghost balance + hashrate | `docs/technical-review-startup-ghost-balance-hashrate-drop.md` | Two startup issues |
| Dual-lane VDF implementation | `docs/technical-review-dual-lane-implementation-v1.md` | Code audit: 3 crypto blockers, 8-10 week fix path |
| Dragon Ball ASIC review | `docs/technical-review-mining-for-dragon-ball-asic.md` | Hardware spec for Dragon Ball miner |
| BTC bridge RocksDB persistence | `docs/technical-review-btc-bridge-rocksdb-persistence-v2.md` | Prefix keys in CF_MANIFEST approach |
| Mining Phases B-D v2 | `docs/technical-review-mining-phases-B-C-D-v2.md` | LWMA + VDF lane design |
| Dual-lane mining whitepaper | `papers/dual-lane-mining-whitepaper.md` | Community-ready paper on CPU-fair mining |
| MCP security audit | (in conversation) | 5 critical + 5 high vulnerabilities |
| Comprehensive test suite | `scripts/delta-comprehensive-test.sh` | 12 tests for Delta Docker |

---

## Current State of All Nodes

| Node | Binary | Balance State | Reorg Handler | Authority Sync |
|------|--------|--------------|---------------|----------------|
| **Epsilon** | v10.3.3 | Correct (restored from Beta via authority sync, 274 wallets) | DISABLED | From Beta (one-time, ran successfully) |
| **Beta** | v10.3.3 | Correct (129+5 QUG confirmed by user) | DISABLED | REMOVED (Beta is the source of truth) |
| **Gamma** | v10.3.0 | INFLATED (P2P one-way sync, never received DEX deductions) | ACTIVE (old binary!) | None |
| **Delta** | v10.3.2-restore3 | Test node, stopped | N/A | None |

### CRITICAL: Do NOT restart Gamma without deploying v10.3.3 first. The reorg handler is still active on Gamma's old binary.

---

## Priority Items for Next Session

### P0: Chain Rebuild Investigation (DOCKER ONLY — NEVER PRODUCTION)

The `rebuild_balances_from_chain()` function crashes silently. This is the most serious architectural concern.

**What we know:**
- Scans 15M block heights sequentially
- Runs for ~3 minutes, consumes ~50s CPU
- Process exits cleanly (no panic, no OOM, no backtrace)
- RUST_BACKTRACE=full produced no output
- Disabling preflight check didn't help (initially thought it was the cause, but process still exits)
- The rebuild NEVER reaches the completion log ("Rebuilt X wallets")

**What to investigate (ON DELTA DOCKER ONLY):**
1. Add progress logging inside the rebuild loop (every 100K heights)
2. Run the rebuild in a standalone binary (not inside the full server)
3. Check if a specific block causes deserialization failure
4. Check if the HashMap grows too large
5. Check if a background tokio task is calling process::exit()
6. Check if `purge_and_rebuild_balances` itself panics in a catch_unwind

**Why this matters:** If ALL nodes lose their balance cache, we cannot recover from chain data alone. The blockchain's "verify from data" guarantee is broken.

### P1: Deploy v10.3.3 to Gamma

Gamma has the OLD binary with the reorg handler ACTIVE. If Gamma restarts, balances get destroyed. Deploy v10.3.3 (reorg handler disabled) to Gamma.

### P1: Jemalloc Memory Stall Fix

Epsilon and Beta both hit 35-52GB RSS from jemalloc fragmentation. The fix (per-arena purge via `tikv-jemalloc-ctl`) is designed but not coded. Needs:
- Add `tikv-jemalloc-ctl` dependency
- Per-arena `dirty_decay_ms` + `muzzy_decay_ms` set to 1000ms at startup
- Periodic `arena.{i}.purge` in the health watchdog (every 300s)
- DeepSeek review correction: use `arena.<i>.dirty_decay_ms` (per-arena), not `arenas.dirty_decay_ms` (global defaults for new arenas only)

### P1: Remove Q_BALANCE_AUTHORITY_PEER from Epsilon

After balances stabilize (monitor for 24-48 hours), remove the authority sync from Epsilon. It should not be a permanent dependency on Beta.

### P2: VDF Crypto Implementation (8-10 weeks)

Three blockers found in the code audit:
1. Cantor's doubling algorithm is mathematically incorrect (missing polynomial reduction)
2. Wesolowski proof generation is a stub (no actual proof)
3. Server verification accepts any proof (security theater)

Full implementation plan in `docs/technical-review-dual-lane-implementation-v1.md`.

### P2: MCP Security Fixes

5 critical vulnerabilities:
1. No transaction confirmation (AI prompt injection → fund drain)
2. No wallet address validation (path traversal)
3. `curl | bash` install with no checksums
4. Global auth token in memory
5. Hardcoded API URLs not validated

### P2: ChatGPT Desktop MCP Support

The MCP server works with Claude Code. ChatGPT Desktop also supports MCP. Setup script needs to detect and configure both.

### P3: BTC Bridge Implementation

Design approved (prefix keys in CF_MANIFEST, serialized mint executor). Not yet coded.

---

## Key Decisions Made

| Decision | Rationale | Reversible? |
|----------|-----------|-------------|
| Disable reorg handler permanently | Root cause of balance corruption, saturating_sub zeros balances | Yes — re-enable if a correct implementation is built |
| Use Beta as balance authority for Epsilon | Beta had verified correct values, chain rebuild crashes | Yes — remove env var |
| Skip token import in authority sync | QUGUSD ($24M) must not be touched | Yes — re-enable token import |
| Remove direct DEX deduction from swap handler | Was causing double deduction | Architectural change — should stay |
| LWMA activation at height 14,900,000 | Already past, activates immediately on deploy | Can be reverted by setting activation to u64::MAX |

---

## Files Changed (Key Commits)

```
17f8f91b fix(v10.3.3): Plan B — restore QUG from Beta authority sync, protect QUGUSD
bedcd2da EMERGENCY fix(v10.3.2): Disable balance reorg handler — ROOT CAUSE of balance loss
b35a1f15 fix(v10.3.2): EMERGENCY — Persistent dedup prevents balance reset on restart
afd3c92d feat(v10.3.2): Ghost balance fix + q-flux cluster failover + miner multi-server
8575dbef fix(v10.3.2): Remove direct balance deduction from swap handler — fixes double deduction
89e2ccf6 feat(v10.3.1): Balance write debugging + v5 corruption review
d56741a3 feat(v10.3.0): Phase B.2 — LWMA difficulty wired to challenge endpoint
b39aaf09 fix(v10.3.0): PQC block verification — auto-learn keys from block headers
11ce4697 feat(v10.3.0): Phase A — Difficulty-weighted mining rewards
```

---

## Backups

| Backup | Location | Size | Contains |
|--------|----------|------|----------|
| Beta RocksDB (correct balances) | `/opt/orobit/shared/q-narwhalknight/data-mainnet-genesis/hot-backup-beta-correct/` on Beta | 49GB | Full RocksDB with verified correct wallet_balance_ values |

---

## Lessons Learned

1. **Pre-existing bugs are the most dangerous.** The reorg handler existed since v0.9.31 and was never reviewed. Our peer reviews only covered our changes, not legacy code.

2. **Canary testing has limits.** Delta (1 miner) cannot reproduce Epsilon's reorg frequency (400+ miners). Production-load testing requires production traffic.

3. **The balance cache is NOT just a cache.** It's treated as authoritative state by the API, the DEX, and the frontend. Losing it loses user-visible funds. It needs checksums, snapshots, and multi-node verification.

4. **`saturating_sub` is dangerous in financial code.** It silently produces zero instead of failing. Financial operations should use checked arithmetic and fail loudly.

5. **Always backup before deploying to a $1B mainnet.** The 49GB Beta RocksDB backup saved us.

6. **Debug logging catches what theory misses.** The `🔴 [BALANCE WRITE]` logging caught both the DEX double-deduction and the reorg handler — two bugs that five rounds of peer review didn't find.

---

## How to Start Next Session

```
1. Read this document first
2. Check all 4 nodes are healthy:
   ssh root@89.149.241.126 "systemctl status q-api-server | head -5"  # Epsilon
   systemctl status q-api-server | head -5                            # Beta
   ssh root@109.205.176.60 "systemctl status q-api-server | head -5"  # Gamma
   ssh root@5.79.79.158 "systemctl status q-api-server | head -5"     # Delta

3. Check master wallet balance on Epsilon:
   ssh root@89.149.241.126 "journalctl -u q-api-server --since '1 minute ago' | grep 'BALANCE TX.*efca' | tail -1"
   # Should show [100-1K] range

4. Priority: investigate chain rebuild crash ON DELTA DOCKER ONLY
5. Priority: deploy v10.3.3 to Gamma (reorg handler still active there!)
6. Priority: code the jemalloc per-arena purge fix
```
