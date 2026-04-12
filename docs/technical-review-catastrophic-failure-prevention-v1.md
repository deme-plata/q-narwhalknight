# Q-NarwhalKnight Catastrophic Failure Guard: Technical Review

**Version**: 1.0  
**Date**: 2026-04-12  
**Network valuation**: $1B mainnet  
**Prepared for**: Peer review by DeepSeek, ChatGPT, and engineering team  

---

## Executive Summary

Q-NarwhalKnight has substantial safety infrastructure reflecting hard-won experience from real incidents. The sync-down protection, emission controller, pointer integrity system, and memory limiter are all well-engineered with proper test coverage.

**The three highest-risk gaps are:**
1. **Removed balance watermark without replacement** — creates a window for re-inflation on restart
2. **No off-site backups** — all safety systems fail if the disk fails
3. **No external alerting** — problems are logged but nobody is paged

For a $1B mainnet, these three issues should be resolved before the next incident, not after.

---

## 1. Catastrophic Failure Inventory

### 1.1 Sync-Down Attack
**Impact**: Deletion of all blocks above a malicious height  
**Status**: PROTECTED  
**Defence**: Hard abort in `turbo_sync.rs:5805` — `if target_height < local_height && local_height > 1000 { return Err("SAFETY ABORT") }`  
**Gap**: Nodes below height 1000 unprotected (acceptable for bootstrap). Fork detection degrades with <3 peers.

### 1.2 Balance Corruption (State Inconsistency)
**Impact**: Balances diverge across nodes  
**Status**: PARTIALLY PROTECTED  
**Has happened**: Yes — "111x inflation bug" (LRU dedup cache lost on restart → turbo_sync re-credited coinbase), and v8.5.7 balance reconciliation "destroyed all wallet balances"  
**Defence**: LRU-bounded processed_blocks cache (500K entries), coinbase merkle root + signature + amount verification  
**Gap**: **P0 CRITICAL** — The `balance_processed_watermark` was removed in v10.2.1 due to a race condition. The sole dedup is now the in-memory LRU cache, which is lost on restart. If turbo_sync replays blocks after restart, coinbase rewards could be re-credited.

### 1.3 Double-Spend
**Impact**: Arbitrary value creation  
**Status**: PROTECTED  
**Defence**: Nonce validation, balance debit check, TX hash dedup, atomic write batches  
**Gap**: Need to verify the gossipsub message handler enforces the same nonce check as the test code.

### 1.4 Chain Split / Fork
**Impact**: Network split, double-spend across forks  
**Status**: PROTECTED  
**Defence**: ForkDetector with 67% consensus threshold, max auto-reorg depth 1000  
**Gap**: Fork detection is advisory (detect + log, no auto-resolution). Only 3-4 production nodes — losing one drops below the 3-peer minimum.

### 1.5 Database Corruption (kill -9)
**Impact**: Pointer corruption, height regression  
**Status**: WELL PROTECTED  
**Has happened**: Yes — pointer corrupted from 12,114 to 353  
**Defence**: PointerIntegrityChecker (auto-repair on startup), SafeBatchedWriter (atomic writes), PreflightVerifier (opt-in), BackgroundIntegrityMonitor (60s interval)  
**Gap**: Preflight check is opt-in (`Q_PREFLIGHT_CHECK=1`). Should be mandatory on a $1B chain.

### 1.6 Key Compromise
**Impact**: Forged blocks with stolen validator identity  
**Status**: UNPROTECTED  
**Gap**: **P2** — No key rotation, no multi-sig, no HSM, no revocation mechanism. Single key per validator.

### 1.7 Emission Overflow
**Impact**: Hyperinflation  
**Status**: STRONGLY PROTECTED  
**Defence**: Hard cap at QUG_MAX_SUPPLY, era cap at 64 eras, remaining_supply cap, bounded correction factor [0.01, 5.0], u128 overflow tests  
**Gap**: Negligible — f64 rounding in correction factor over 256 years. Tests confirm it's dust-level.

### 1.8 Smart Contract Escape
**Impact**: Arbitrary code execution  
**Status**: PARTIALLY PROTECTED  
**Defence**: Wasmer sandbox, gas metering on host functions, state isolation  
**Gap**: **P2** — No Wasmer fuel metering (loops between host calls unmetered), no per-contract memory limit.

### 1.9 DEX Pool Drain
**Impact**: Liquidity stolen via arithmetic exploits  
**Status**: PARTIALLY PROTECTED  
**Defence**: BigDecimal arithmetic, basis-point fees, constant-product invariant  
**Gap**: **P1** — Pool state is in-memory only (restart loses all pools). No minimum liquidity lock. No re-entrancy guard.

### 1.10 Memory Exhaustion → P2P Death
**Impact**: Node death, potential WAL corruption  
**Status**: PROTECTED (v10.3.0)  
**Has happened**: Yes — 4 times in 24 hours on Epsilon  
**Defence**: MemoryLimiter with swap awareness, health watchdog with cgroup monitoring, block cache cap (4GB), MemoryHigh raised to 50GB  
**Gap**: Watchdog is log-only — no auto-restart, no cache eviction, no alerting.

### 1.11 Silent Pruning
**Impact**: Irrecoverable block body loss  
**Status**: FIXED (v10.3.0)  
**Has happened**: Yes — all nodes pruned blocks for first 20 days  
**Defence**: PruningConfig default changed to Full, scheduler requires explicit Q_PRUNING_MODE env var

---

## 2. Backup and Recovery

| Aspect | Status | Detail |
|--------|--------|--------|
| Automated backups | PARTIAL | Hourly incremental to local disk. IPFS disabled. |
| Off-site replication | **MISSING** | All backups on same disk as database |
| kill -9 recovery | GOOD | PointerIntegrityChecker auto-repairs on startup |
| Database repair tools | GOOD | 5+ repair binaries for various corruption types |
| Preflight verification | OPT-IN | Requires `Q_PREFLIGHT_CHECK=1` — should be default |
| RTO | ~10 min | Pointer repair + preflight + peer sync |
| RPO | ~1 hour | Backup interval; WAL may have unflushed data |

---

## 3. Defence in Depth Checklist

| # | Protection | Status |
|---|-----------|--------|
| 1 | Automated hourly backups | PARTIAL (local only) |
| 2 | Off-site backup replication | **NO** |
| 3 | Cross-node state hash comparison | **NO** |
| 4 | Balance reconciliation (sum = emitted - burned) | PARTIAL (test exists, not in production) |
| 5 | Block signature verification on turbo sync | **NO** |
| 6 | Rate limiting on P2P balance updates | PARTIAL |
| 7 | Graceful degradation at 0 peers | PARTIAL (detection only) |
| 8 | Automatic restart on OOM | **NO** |
| 9 | Preflight check before serving | PARTIAL (opt-in) |
| 10 | Pointer integrity on startup | YES |
| 11 | Sync-down protection | YES |
| 12 | Emission cap enforcement | YES |
| 13 | Fork detection | YES |
| 14 | Double-processing prevention | YES (LRU cache) |
| 15 | Database repair tooling | YES |
| 16 | WASM gas metering | PARTIAL |
| 17 | Key rotation / revocation | **NO** |
| 18 | Multi-sig block production | **NO** |
| 19 | Circuit breaker for network failures | YES |
| 20 | Memory pressure backoff | YES |

---

## 4. Priority-Ordered Gaps

### P0 — Critical (fix before next incident)

| # | Gap | Risk If Deferred | Effort |
|---|-----|-------------------|--------|
| 1 | No external alerting (Discord/webhook) | Node death goes unnoticed for hours | 1-2 days |
| 2 | No off-site backups (S3/R2) | Disk failure = total loss | 2-3 days |
| 3 | Balance watermark removed without replacement | Re-inflation on restart | 3-5 days |
| 4 | Preflight check not mandatory | Corrupted node serves bad data | 1 day |

### P1 — High (fix within 2 weeks)

| # | Gap | Risk If Deferred | Effort |
|---|-----|-------------------|--------|
| 5 | No supply invariant runtime check | Balance corruption undetected | 2-3 days |
| 6 | No block sig verification in turbo sync | Malicious blocks accepted during sync | 3-5 days |
| 7 | DEX pool state is in-memory only | Restart loses all DEX state | 3-5 days |
| 8 | Health watchdog is log-only | No auto-restart on sustained failure | 1-2 days |
| 9 | Only 3-4 production nodes | Fork detection non-functional with <3 peers | Infrastructure |

### P2 — Medium (fix within 1 month)

| # | Gap | Effort |
|---|-----|--------|
| 10 | No key rotation mechanism | 1-2 weeks |
| 11 | WASM execution not fully metered | 3-5 days |
| 12 | No clock drift detection | 1 day |
| 13 | Integrity monitor only checks last 1000 blocks | 2-3 days |

### P3 — Low (fix within quarter)

| # | Gap | Effort |
|---|-----|--------|
| 14 | No Prometheus metrics export | 2-3 days |
| 15 | No formal RTO/RPO documentation | 1 day |

---

## 5. What's Strong

These systems are well-engineered and reflect real incident experience:

- **Emission controller**: The most mathematically rigorous component. Hard caps at every level, u128 arithmetic, bounded correction factors, comprehensive tests.
- **Sync-down protection**: Hard abort with loud logging. Cannot be bypassed.
- **Pointer integrity**: Auto-repair on startup with parallel scan. Multiple repair utilities.
- **Memory limiter**: Swap-aware, RAM-tiered, 4 pressure levels with exponential backoff.
- **Health watchdog** (v10.3.0): Cgroup-aware, stall detection, zero-peer detection, cache cleanup.

---

## 6. Questions for Peer Reviewers

1. Should the balance watermark be restored as a persistent on-disk bloom filter (compact, survives restart) or as a RocksDB column family (exact, higher disk cost)?
2. Is the emission controller's f64 correction factor a real risk over 256 years, or is the u128 total_supply cap sufficient to catch any drift?
3. For turbo sync signature verification: should we verify ALL block signatures (expensive, ~50ms × 200 blocks per batch = 10s), or sample (e.g., verify 10% randomly)?
4. The DEX "quantum slippage reduction" gives traders MORE tokens than constant-product dictates. Is this an intentional LP subsidy or a pricing bug?
5. With 3-4 production nodes, what is the minimum viable validator set for meaningful fork detection? Is 5 sufficient, or do we need 7+ (for BFT 2f+1)?
