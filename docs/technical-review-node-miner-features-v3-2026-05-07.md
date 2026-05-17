# Q-NarwhalKnight — Node & Miner Technical Review v3
**Date:** 2026-05-07  
**Version Reviewed:** 10.6.1  
**Reviewer:** Server Beta (Claude Code)  
**Branch:** feature/safe-batched-sync-v1.0.2  
**Status:** Iteration 3 — Post-Incident Audit (Bridge Implementation, UX Hardening, Open Issues Triage)  
**Supersedes:** `technical-review-node-miner-features-v2-2026-05-06.md`

---

## Changelog from v2

| Section | Change |
|---------|--------|
| §1 Executive Summary | Updated critical finding count; 3 new FIXED items |
| §3.11 Bridge Stubs | **Major revision** — Bitcoin deposit bridge now IMPLEMENTED (was stub) |
| §7 Frontend Wallet | Rate limiter fix documented; Zcash UX overhaul documented |
| §10 Issues Register | 3 issues closed (I-004, I-008, I-009); DEX-001/002 CRITICAL remain open |
| §11 Roadmap | Priority order adjusted; bridge work complete |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Node (q-api-server) — Feature Review](#3-node-q-api-server--feature-review)
4. [Miner (q-miner) — Feature Review](#4-miner-q-miner--feature-review)
5. [Cryptography Layer](#5-cryptography-layer)
6. [DeFi & Smart Contracts](#6-defi--smart-contracts)
7. [Frontend Wallet](#7-frontend-wallet)
8. [Infrastructure & Deployment](#8-infrastructure--deployment)
9. [Feature Completeness Matrix](#9-feature-completeness-matrix)
10. [Issues Register — v3 (Triaged)](#10-issues-register--v3-triaged)
11. [Recommendations Roadmap](#11-recommendations-roadmap)

---

## 1. Executive Summary

Q-NarwhalKnight v10.6.1 marks the first release with a **fully operational Bitcoin deposit bridge**, moving from stub implementation to live HD-wallet-derived deposit address generation via Delta node's Bitcoin Knots v28.1 RPC. Three issues from v2 are closed; the DEX trading engine disconnect (DEX-001/002) remains the highest-priority open issue.

### Changes Since v2 (Fixed ✅)

| ID | Area | Fix Summary | Version |
|----|------|-------------|---------|
| **I-004** | Bitcoin bridge | `DepositBridge` fully implemented — HD address generation from `qug-bridge` wallet via Bitcoin Knots RPC on Delta (5.79.79.158:8332). Was `active: false` stub. | v10.6.1 |
| **I-008** | Bridge status display | `get_bridge_status()` now checks `|| state.deposit_bridge.is_some()` — bridge showed "offline" even when fully initialized | v10.6.1 |
| **I-009** | Frontend rate limiter | `RequestRateLimiter` max concurrent increased 10 → 20 — at 10, background SSE polling consumed all slots, blocking user actions with "Authentication error: Rate limiter timeout" | v10.6.1 |

### Still Open — Critical

| ID | Severity | Area | Finding |
|----|----------|------|---------|
| **DEX-001** | 🔴 Critical | DEX trading engine | `execute_quantum_trade()` does not update pool reserves; x×y=k never verified after swap |
| **DEX-002** | 🔴 Critical | DEX concurrency | No atomic write lock over read→compute→write during swap |

### Still Open — High / Medium

| ID | Severity | Area | Finding |
|----|----------|------|---------|
| **DEX-003** | 🟠 High | DEX slippage | `max_slippage_bps` in `TradeRequest` never validated in execution |
| **DEX-004** | 🟠 High | DEX reserves | No `MIN_POOL_RESERVE`; dust reserves → trillion-dollar prices |
| **POOL-001** | 🟠 High | Stratum dedup | Share dedup HashSet cleared entirely at 100k entries |
| **POOL-002** | 🟠 High | Stratum difficulty | Min difficulty not enforced synchronously |
| **POOL-003** | 🟡 Medium | Stratum replay | Extranonce2 in dedup key enables replay |
| **POOL-004** | 🟡 Medium | Stratum race | `clean_jobs=false` bug |
| **POOL-005** | 🟡 Medium | Pool always-on | Default enabled, silent port error |
| **VDF-001** | 🟡 Medium | VDF lane | 4–7 s/proof single-lane bottleneck |
| **BAL-001** | 🟡 Medium | BalanceRootV1 | Zero fallback causes chain halt on persistent storage error |
| **ARCH-001** | 🟡 Medium | Code size | `main.rs` 24,609 lines / `main()` ~22,762 lines |
| **ARCH-002** | 🟡 Medium | Code size | `handlers.rs` 17,019 lines |
| **DEX-005** | 🟡 Medium | DEX fee | Fee deducted from output but full amount added to reserves |
| **DEX-006** | 🟢 Low | DEX overflow | k overflows u128 at 24-decimal scale — mitigated by BigDecimal |
| **WIN-001** | 🟢 Low | Windows storage | Sled OOM via `panic::catch_unwind`; no io_uring parity |

**Good news (unchanged from v2):** Mining reward double-credit path is **SAFE** (3-layer persistent dedup). BalanceRootV1 at `height >= 18,600,000` is **correctly implemented**. Atomic swap opcodes are **safe no-ops**. u128 P2P serialization is **properly handled**. systemd service file contains **no secrets**.

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Q-NarwhalKnight Stack                           │
├────────────────┬─────────────────┬───────────────┬───────────────────┤
│  Web Wallet    │  Slint Wallet   │  q-miner CLI  │  External Miners  │
│ (React/TS)     │ (Rust/Slint)    │ CPU+GPU+VDF   │ (Stratum pool)    │
├────────────────┴─────────────────┴───────────────┴───────────────────┤
│              REST API · SSE · WebSocket  (Axum 0.7)                  │
│              228 routes · AEGIS-QL auth · governor rate limit        │
├──────────────┬───────────────┬────────────────┬───────────────────────┤
│  Block       │  Balance      │  Emission      │  Smart Contracts      │
│  Producer    │  Consensus    │  Controller    │  WASM (Wasmer 4.0)    │
│  SegQueue    │  3-layer dedup│  64 eras·21M   │  25+ contract types   │
├──────────────┴───────────────┴────────────────┴───────────────────────┤
│              DAG-Knight Consensus Engine                              │
│   Bullshark BFT · δ=1 · <50ms finality · Quantum VDF anchor         │
│   Homological fork detection (Betti H₀/H₁) · Genus-2 VDF            │
├──────────────────────────────────────────────────────────────────────┤
│              libp2p 0.56  (Tokio async)                              │
│  TCP · QUIC · WS · DCUTR · Gossipsub · Kademlia DHT                 │
│  Turbo Sync 150–250 BPS · 16 parallel streams · Dandelion++ Tor     │
├──────────────────────────────────────────────────────────────────────┤
│              Storage Layer                                            │
│  RocksDB (Linux/macOS) · Sled (Windows) · 12 column families        │
│  Argon2+AES-GCM · LZ4/Zstd · Parallel state applicator             │
└──────────────────────────────────────────────────────────────────────┘
```

**Exact file sizes (verified):**

| File | Lines |
|------|-------|
| `crates/q-api-server/src/main.rs` | **24,609** |
| `crates/q-api-server/src/handlers.rs` | **17,019** |
| `crates/q-miner/src/main.rs` | 4,778 |
| `crates/q-mining-pool/src/` | ~3,500 (multi-file) |
| `qug-v1-rtl/rtl/xcrypto/` | ~2,100 (SystemVerilog) |

---

## 3. Node (q-api-server) — Feature Review

### 3.1–3.10: Unchanged from v2

Sections 3.1 through 3.10 (Startup, Consensus, Block Production, Storage, P2P, REST API, SSE Streaming, BalanceRootV1, Emission Controller, Mining Pool) are unchanged from v2. All findings remain valid. See v2 for full detail.

Key unchanged verdicts:
- **Mining reward dedup: SAFE** (3-layer persistent, 5 tests verified)
- **BalanceRootV1 gate: CORRECTLY IMPLEMENTED** (`height >= 18,600,000`, no off-by-one)
- **Emission arithmetic: Pure u128, deterministic** (no floating-point drift)
- **systemd service file: CLEAN** (no secrets in Environment= lines)

---

### 3.11 Bitcoin Deposit Bridge — IMPLEMENTED (was stub in v2)

**Status change:** v2 listed the Bitcoin bridge as a stub returning `{"active": false}`. v10.6.1 ships a **fully operational deposit bridge**.

#### Architecture

```
User (Frontend)
      │
      ▼ POST /api/v1/bitcoin-bridge/deposit/address
      │
      ├── Auth check (AEGIS-QL session)
      │
      ├── state.deposit_bridge.is_some()? ──No──► 404 "not enabled"
      │
      ▼
DepositBridge::generate_deposit_address(user_pubkey)
      │
      ├── Derive HD path: m/44'/0'/0'/0/{user_index}  
      │   (from qug-bridge HD wallet on Delta)
      │
      ▼
Bitcoin Knots v28.1 RPC (Delta: 5.79.79.158:8332)
      │
      ├── wallet: qug-bridge
      ├── method: getnewaddress / deriveaddresses
      │
      ▼
Returns bc1q... native SegWit address (unique per user)
```

#### Initialization Path

**File:** `crates/q-api-server/src/main.rs:4019`

```rust
if let Some(bridge_config) = DepositBridgeConfig::from_env() {
    match DepositBridge::new(bridge_config, dep_event_tx).await {
        Ok(bridge) => {
            state.deposit_bridge = Some(Arc::new(bridge));
            info!("₿ Bitcoin deposit bridge initialized (Delta RPC: {})", btc_rpc_url);
        }
        Err(e) => {
            warn!("₿ Bitcoin deposit bridge init failed: {}", e);
        }
    }
}
```

`DepositBridgeConfig::from_env()` reads `BTC_RPC_URL`, `BTC_RPC_USER`, `BTC_RPC_PASS`. Returns `None` (disabling the bridge silently) if `BTC_RPC_PASS` is empty.

#### Bridge Status API Fix (I-008 — FIXED)

**File:** `crates/q-api-server/src/bitcoin_bridge_api.rs:742`

```rust
// v2 (broken): only checked atomic_swap_manager
let connected = state.atomic_swap_manager.is_some();

// v10.6.1 (fixed): checks either component
let connected = state.atomic_swap_manager.is_some() || state.deposit_bridge.is_some();
```

Before the fix, the bridge could be fully initialized (`state.deposit_bridge = Some(...)`) but the status endpoint still returned `"bridge_enabled": false`, showing "Bridge Offline" in the UI.

#### Epsilon Production Configuration

Epsilon (`89.149.241.126`) `/.env` now contains:

```
BTC_RPC_URL=http://5.79.79.158:8332
BTC_RPC_USER=qugbridge
BTC_RPC_PASS=<redacted>
```

Delta's Bitcoin Knots v28.1 is synced to block **948,305** with the `qug-bridge` HD wallet loaded. The bridge is confirmed alive via `/api/v1/bitcoin-bridge/status`:

```json
{
  "connected": true,
  "bridge_enabled": true,
  "bridge_version": "Knots v28.1",
  "network_height": 948305
}
```

#### HiBT Donation Tracking

The HiBT fundraising modal tracks `bc1qnqdj5kuka522kctk4v99l22jpjut3lums2kepl` via the mempool.space API fallback when `bitcoin_rpc_client` is unavailable. On-chain balance verified as **0 BTC** (no donations received as of 2026-05-07). The modal is functional and will display deposits as they arrive.

---

### 3.12 Frontend API Rate Limiter (I-009 — FIXED)

**File:** `gui/quantum-wallet/src/services/api.ts`

#### Root Cause

The `RequestRateLimiter` class caps concurrent HTTP requests to prevent API flooding. Prior to v10.6.1 the cap was **10 slots**. With 14+ active screens each running background polling (Dashboard SSE heartbeats, Explorer block polling, balance updates, SSE keep-alives), all 10 slots were consistently occupied by background tasks.

When the user clicked "Generate Deposit Address", the call was queued behind 10 in-flight requests and timed out after 5,000 ms:

```
Error path:
createDepositAddress()
  → authenticatedRequest('/api/v1/bitcoin-bridge/deposit/address', ...)
    → this.request(...)
      → rateLimiter.acquire(maxWaitMs=5000)  ← TIMEOUT
        → throw Error("Rate limiter timeout after 5000ms")
          → caught: "Authentication error: Rate limiter timeout after 5000ms — too many concurrent requests"
```

The error was incorrectly wrapped as "Authentication error", misleading operators into investigating session tokens rather than concurrency.

#### Fix

```typescript
// Before:
const rateLimiter = new RequestRateLimiter(10); // Max 10 concurrent requests

// After (v10.6.1):
const rateLimiter = new RequestRateLimiter(20); // Max 20 concurrent requests
```

**Remaining risk:** If all 20 slots fill (e.g., a user with many simultaneous tabs or extremely heavy SSE polling), the issue recurs. A longer-term fix would be to prioritize user-initiated requests over background polling in the queue, or to exempt background polls from the shared limiter.

---

## 4. Miner (q-miner) — Feature Review

No changes from v2. All findings (§4.1–4.8) remain valid. Key unchanged:
- CPU mining: SAFE, thread-partitioned nonce space
- GPU mining: OpenCL 3.0, persistent buffers, adaptive dispatch
- VDF-001: Single-lane 4–7 s/proof bottleneck still open
- POOL-001 through POOL-005: Stratum vulnerabilities still open

---

## 5. Cryptography Layer

No changes from v2. Classical + PQ + ZK layer unchanged (§5.1–5.4).

---

## 6. DeFi & Smart Contracts

### 6.1 DEX — Constant-Product AMM

**DEX-001 and DEX-002 remain OPEN and CRITICAL.** No code changes to `q-dex/src/trading.rs` or `q-dex/src/liquidity.rs` in this release cycle.

The current behavior is: trades execute, fees are charged, statistics are recorded — but pool reserve state (`token_a_reserve`, `token_b_reserve`) is **never mutated by swaps**. The DEX functions as a price discovery engine but does not settle on-chain. This is either by design (the DEX displays quotes but settlement happens at the transaction layer) or a missing integration.

**Operator note:** Until DEX-001/002 are fixed, the DEX should not be advertised as a finalized settlement layer. Users swapping tokens may see price discovery without actual reserve changes reflected in future price quotes.

---

## 7. Frontend Wallet

### 7.1 Quantum Wallet — Rate Limiter Fix

See §3.12. The 10→20 slot increase resolves the immediate user-facing issue. The underlying architecture (shared limiter for all request types) remains a medium-term design debt.

### 7.2 Zcash Wallet Modal — UX Overhaul

**File:** `gui/quantum-wallet/src/components/ZcashWalletModal.tsx`

The Zcash modal was completely redesigned in this release:

| Property | Before | After (v10.6.1) |
|----------|--------|-----------------|
| Background | Standard dark | Deep purple-black linear gradient `(#0d050f → #130620 → #0a0414)` |
| Border | Default | `1.5px solid rgba(147,51,234,0.45)` with purple glow |
| Balance display | Small text | Large `4xl` font, ZEC gold `#f4b728` with text-shadow |
| Animations | None | `motion` + `AnimatePresence` from framer-motion |
| QR code | None | `QRCodeSVG` with `zcash:${zAddress}` URI scheme |
| Address copy | Basic | Clipboard icon with `CheckCircle` confirmation flash |
| Tabs | Basic | Pill-style with purple active state |
| Status footer | None | Zebra block height display |
| Empty states | None | Dedicated empty state UI for history tab |

**Zcash connectivity:** The modal reads from Zebra RPC on Delta (port 8232). Zebra is confirmed synced to block **3,334,024**. The `get_z_address`, `get_z_balance`, and transaction history endpoints all require authentication and are functional.

---

## 8. Infrastructure & Deployment

### 8.1 Multi-Server HA Topology (Unchanged)

| Server | Role | Bandwidth | Notes |
|--------|------|-----------|-------|
| Beta (185.182.185.227) | Primary bootstrap | 100 Mbit | Nginx weight=10 |
| Gamma (109.205.176.60) | HA backup | 1 Gbit | Nginx weight=1 |
| Delta (5.79.79.158) | Canary + BTC/ZEC RPC | 1 Gbit | Hosts Bitcoin Knots + Zebra |
| Epsilon (89.149.241.126) | 10 Gbit supernode | 10 Gbit | DNS primary (quillon.xyz) |

### 8.2 v10.6.1 Deployment Notes

Deployment was performed **manually** (user explicitly requested non-HA path):

1. Built on Beta: `cargo build --release --package q-api-server`
2. Beta: `cp --remove-destination` trick (avoids "Text file busy" for running binary)
3. `systemctl restart q-api-server` on Beta
4. `scp` to Epsilon + `systemctl restart q-api-server` on Epsilon
5. Frontend rebuilt (`npm run build` in `gui/quantum-wallet/`) + rsync to Epsilon `/home/orobit/q-narwhalknight/dist-final/`

**Gamma (109.205.176.60) was NOT updated** and is still running v10.6.0. The ha-deploy.sh rolling pipeline was bypassed for this release; Gamma remains one version behind.

**Action required:** Deploy v10.6.1 to Gamma via `./scripts/ha-deploy.sh verify-gamma` or full rolling deploy.

---

## 9. Feature Completeness Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| DAG-Knight consensus | ✅ Production | δ=1, Genus-2 VDF, homological fork detection |
| Block production | ✅ Production | 15 s default, SIMD Merkle, 250 coinbase/block |
| RocksDB storage | ✅ Production | 12 CFs, AES-GCM, Warp Sync |
| libp2p P2P | ✅ Production | TCP/QUIC/WS, Kademlia, DCUTR |
| Turbo Sync | ✅ Production | 150–250 BPS, 16 parallel streams |
| REST API | ✅ Production | 228+ endpoints |
| SSE streaming | ✅ Production | <50 ms, 20+ event types |
| BalanceRootV1 | ✅ Production | Activates at height 18,600,000 (SAFE) |
| Mining reward dedup | ✅ Production | 3-layer persistent (SAFE) |
| Ed25519 / Dilithium5 / Kyber1024 | ✅ Production | Q0/Q1/Q2 phases |
| SQIsign / FROST | ✅ Integrated | `advanced-crypto` feature |
| Circle STARKs | ✅ Integrated | Private transactions |
| Dandelion++ | ✅ Production | Mandatory tx relay |
| Tor (Arti) | ✅ Integrated | `Q_ENABLE_TOR=1` |
| Stratum pool | ✅ Production | PPLNS — security issues (POOL-001..005) open |
| CPU miner | ✅ Production | Multi-thread, AVX2/512 |
| GPU miner (OpenCL) | ✅ Production | Persistent buffers, adaptive dispatch |
| Genus-2 VDF mining | ✅ Production | Single-lane bottleneck (VDF-001 open) |
| FPGA RTL (Xcrypto) | 🟡 Prototype | Kintex-7 BLAKE3 pipeline |
| FPGA RTL (Xlattice NTT) | ⚠️ Stub | Phase 1B; poly.add/mul implemented |
| React wallet (14+ screens) | ✅ Production | SSE sync, DEX, mining |
| Zcash wallet modal | ✅ Production | Redesigned v10.6.1; Zebra integration |
| Slint native wallet | ✅ Integrated | GPU mining, auto-update |
| DEX (constant product) | ⚠️ Partial | Price discovery works; reserve mutation disconnected (DEX-001) |
| WASM smart contracts | ✅ Integrated | Wasmer 4.0, 25+ types |
| AI inference (Mistral) | ✅ Integrated | CUDA/Metal optional |
| **Bitcoin deposit bridge** | **✅ Production** | **HD address generation via Delta Knots RPC (was stub in v2)** |
| Ethereum bridge | ⚠️ Not implemented | No handler functions |
| Monero bridge | ⚠️ Not implemented | No handler functions |
| Atomic swaps (0x90–0x93) | ⚠️ No-op | Silent pass, fee only |
| Sharding | ⚠️ Disabled | Crate exists, disabled in production |
| Windows / Sled storage | 🟡 Limited | OOM panic-catch; no io_uring parity |
| HA rolling deploy | ✅ Production | Zero-downtime 4-server pipeline |

---

## 10. Issues Register — v3 (Triaged)

### ✅ Closed Since v2

| ID | Area | Resolution | Version |
|----|------|------------|---------|
| **I-004** | Bitcoin bridge | `DepositBridge` fully implemented with Delta RPC | v10.6.1 |
| **I-008** | Bridge status | `|| state.deposit_bridge.is_some()` fix in `get_bridge_status()` | v10.6.1 |
| **I-009** | Rate limiter | Max concurrent requests 10 → 20 | v10.6.1 |

### 🔴 Critical (Open)

| ID | Area | Description | File:Line |
|----|------|-------------|----------|
| **DEX-001** | DEX trading | `execute_quantum_trade()` does not update pool reserves; x×y=k never verified | `q-dex/src/trading.rs:242` |
| **DEX-002** | DEX concurrency | No atomic lock over read→compute→write during swap; stale reserve reads | `q-dex/src/liquidity.rs:19` |

### 🟠 High (Open)

| ID | Area | Description | File:Line |
|----|------|-------------|----------|
| **DEX-003** | DEX slippage | `max_slippage_bps` exists in TradeRequest but never validated | `q-dex/src/trading.rs` |
| **DEX-004** | DEX reserves | No `MIN_POOL_RESERVE`; dust reserves → astronomical prices | `q-dex/src/liquidity.rs` |
| **POOL-001** | Stratum dedup | HashSet full-clear at 100k; all seen shares re-accepted | `q-mining-pool/src/share.rs:260` |
| **POOL-002** | Stratum difficulty | Min difficulty not enforced synchronously at TCP handler | `q-mining-pool/src/stratum.rs:511` |

### 🟡 Medium (Open)

| ID | Area | Description | File:Line |
|----|------|-------------|----------|
| **DEX-005** | DEX invariant | Fee deducted from output but full amount added to reserves | `q-dex/src/liquidity.rs:476` |
| **POOL-003** | Stratum replay | Miner-controlled `extranonce2` in dedup key enables replay | `q-mining-pool/src/share.rs:234` |
| **POOL-004** | Stratum race | `clean_jobs=false` bug; stale shares accepted in block-found window | `q-mining-pool/src/pool.rs:397` |
| **POOL-005** | Pool always-on | Default enabled; port error silent | `main.rs:4284` |
| **VDF-001** | VDF bottleneck | 4–7 s/proof; most evaluations discarded at high BPS | `q-miner/src/vdf_lane.rs:142` |
| **BAL-001** | BalanceRootV1 | Zero fallback on compute failure → chain halt | `block_producer.rs:983` |
| **ARCH-001** | Code size | `main.rs` 24,609 lines; `main()` ~22,762 lines | `q-api-server/src/main.rs` |
| **ARCH-002** | Code size | `handlers.rs` 17,019 lines; 70+ handlers | `q-api-server/src/handlers.rs` |
| **S-002** | Rate limiting | AI/ZK endpoints share limiter with lightweight endpoints | `main.rs` route setup |
| **I-010** | Rate limiter arch | Shared rate limiter for all request types; user actions compete with background polls | `api.ts:RequestRateLimiter` |

### 🟢 Low / Confirmed Acceptable

| ID | Area | Status |
|----|------|--------|
| **DEX-006** | DEX overflow | k overflows u128 at 24-decimal scale — mitigated by BigDecimal |
| **WIN-001** | Windows storage | Sled OOM via panic::catch_unwind |
| **I-002** | FPGA Xlattice NTT | Stubbed; not blocking mining (Phase 1B) |
| **S-001** | systemd secrets | Service file clean — no secrets |
| **u128 serde** | P2P serialization | Custom module correctly serializes as STRING |
| **DoubleCredit** | Mining rewards | 3-layer persistent dedup confirmed SAFE |
| **BalanceRoot gate** | Activation | `height >= 18,600,000` correct |
| **Bridge stubs** | Ethereum/Monero | Safe empty responses |
| **AtomicSwap** | 0x90–0x93 | Silent no-op with fee |

---

## 11. Recommendations Roadmap

### Priority 1 — Fix Before Next Mainnet DEX Promotion

**DEX-001 + DEX-002: Connect trading engine to pool state (atomic under write lock)**

```rust
// In execute_quantum_trade(), after computing output:
let mut pools = self.liquidity_manager.quantum_pools.write().await;
let pool = pools.get_mut(&pair_id).ok_or(DexError::PoolNotFound)?;
let new_reserve_in = pool.reserve_in.checked_add(amount_in_with_fee)
    .ok_or(DexError::Overflow)?;
let new_reserve_out = pool.reserve_out.checked_sub(amount_out)
    .ok_or(DexError::InsufficientReserve)?;
// Verify k ≥ k_initial (fees should increase k, not decrease it)
let new_k = new_reserve_in * new_reserve_out;
ensure!(new_k >= pool.k_invariant, DexError::InvariantViolation);
pool.reserve_in = new_reserve_in;
pool.reserve_out = new_reserve_out;
pool.k_invariant = new_k;
```

**DEX-003: Enforce slippage before compute**
```rust
let price_impact_bps = compute_price_impact(&pool, amount_in, amount_out);
ensure!(price_impact_bps <= request.max_slippage_bps, DexError::SlippageExceeded);
```

**DEX-004: Add MIN_POOL_RESERVE constant**
```rust
const MIN_POOL_RESERVE: u128 = 10u128.pow(22); // 0.01 token at 24 decimals
ensure!(new_reserve_a >= MIN_POOL_RESERVE && new_reserve_b >= MIN_POOL_RESERVE,
    DexError::ReserveTooLow);
```

### Priority 2 — Stratum Pool Security Hardening

- **POOL-001:** Replace `HashSet` with rolling LRU deque (no full-clear)
- **POOL-002:** Enforce `share_difficulty >= pool.min_difficulty` synchronously
- **POOL-003:** Change dedup key to `hash(job_id ‖ nonce)` (exclude extranonce2)
- **POOL-004:** Pass `clean_jobs = true` to `create_job()` on block found

### Priority 3 — Rate Limiter Architecture

**I-010:** Separate user-initiated vs. background polling request queues:
```typescript
const userRateLimiter = new RequestRateLimiter(15);     // User actions
const backgroundRateLimiter = new RequestRateLimiter(10); // Polling
```

User actions should never be blocked by background polling. Priority inversion was the root cause of I-009.

### Priority 4 — VDF Lane Scaling

**VDF-001:** Expose `Q_VDF_ITERATIONS_CAP` env var. Cap iterations so evaluation completes in ≤80% of expected block interval. At 10 BPS (100 ms/block), cap to ~50 ms effective work → ~25 squarings (far below 2,048 default). Consider parallel VDF evaluation for multiple lanes.

### Priority 5 — BalanceRootV1 Startup Health Check

**BAL-001:** Add to startup sequence (before accepting mining submissions at height ≥ 18,600,000):
```rust
if current_height >= 18_599_900 {
    match storage.compute_balance_root_for_block().await {
        Ok(_) => info!("BalanceRootV1 health check: PASS"),
        Err(e) => {
            error!("CRITICAL: BalanceRootV1 compute failed: {} — refusing to start at this height", e);
            std::process::exit(1);
        }
    }
}
```

### Priority 6 — Deploy v10.6.1 to Gamma

Gamma is one release behind (v10.6.0). Run:
```bash
./scripts/ha-deploy.sh verify-gamma
./scripts/ha-deploy.sh restore
```

---

*Next review (v4) should focus on: DEX-001/002 fix verification, POOL hardening status, BalanceRootV1 pre-activation monitoring (current height vs. 18,600,000).*
