# Q-NarwhalKnight — Node & Miner Technical Review v2
**Date:** 2026-05-06  
**Version Reviewed:** 10.6.0  
**Reviewer:** Server Beta (Claude Code)  
**Branch:** feature/safe-batched-sync-v1.0.2  
**Status:** Iteration 2 — Deep Audit (Security, Correctness, Architecture)  
**Supersedes:** `technical-review-node-miner-features-v1-2026-05-06.md`

---

## Changelog from v1

| Section | Change |
|---------|--------|
| §3.3 Mining reward path | Full trace added; dedup layers verified; double-credit verdict: **SAFE** |
| §6.1 DEX invariant | 8 findings including **CRITICAL** race condition and missing invariant check |
| §3.8 BalanceRootV1 | Activation guard verified; determinism tests mapped; fallback risk identified |
| §8 Bridge stubs | Atomic swap no-op behavior confirmed safe |
| §4.8 Stratum security | 5 vulnerabilities found: dedup, share difficulty, replay, race window |
| §5 Cryptography | u128 serialization boundary confirmed safe; custom serde module documented |
| §4.4 VDF lane | Bottleneck quantified: 4–7 s/proof, 30 s server timeout, staleness risk |
| §9 Systemd secrets | Service file audited: **CLEAN** |
| §10.4 Windows parity | Sled OOM panic-catch documented; io_uring fallback noted |
| Line counts | main.rs: 24,609 lines; handlers.rs: 17,019 lines (exact) |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Node (q-api-server) — Deep Feature Review](#3-node-q-api-server--deep-feature-review)
4. [Miner (q-miner) — Deep Feature Review](#4-miner-q-miner--deep-feature-review)
5. [Cryptography Layer](#5-cryptography-layer)
6. [DeFi & Smart Contracts](#6-defi--smart-contracts)
7. [Frontend Wallet](#7-frontend-wallet)
8. [Infrastructure & Deployment](#8-infrastructure--deployment)
9. [Feature Completeness Matrix](#9-feature-completeness-matrix)
10. [Issues Register — v2 (Verified & Expanded)](#10-issues-register--v2-verified--expanded)
11. [Recommendations Roadmap](#11-recommendations-roadmap)

---

## 1. Executive Summary

Q-NarwhalKnight (QNK) is a Rust-based modular blockchain node (89 crates, ~41K+ lines in the two primary source files alone) implementing a quantum-enhanced DAG-BFT consensus protocol (DAG-Knight). The system targets 10+ BPS block production, progressive post-quantum cryptographic hardening, and a full DeFi stack including a constant-product AMM DEX, WASM smart contracts, and cross-chain bridge stubs.

This iteration deepens the v1 inventory with verified code-level findings across five critical areas: mining reward accounting, DEX invariant correctness, BalanceRootV1 activation safety, Stratum pool security, and architecture scalability.

### Critical Findings Summary

| ID | Severity | Area | Finding |
|----|----------|------|---------|
| **DEX-001** | 🔴 Critical | DEX trading engine | No constant-product invariant check after swap; pool reserves not updated by trading engine |
| **DEX-002** | 🔴 Critical | DEX concurrency | Concurrent swap race: two simultaneous swaps use stale reserve reads, violating x×y=k |
| **DEX-003** | 🟠 High | DEX slippage | `max_slippage_bps` field exists in request but is never validated in trading engine |
| **DEX-004** | 🟠 High | DEX reserves | No `MIN_POOL_RESERVE` constant; reserve can drain to dust creating trillion-dollar prices |
| **POOL-001** | 🟠 High | Stratum dedup | Share dedup HashSet cleared in full at 100k entries; same share resubmittable after clear |
| **POOL-002** | 🟠 High | Stratum difficulty | Minimum share difficulty NOT enforced synchronously; async rejection only |
| **POOL-003** | 🟡 Medium | Stratum replay | No per-worker nonce sequence tracking; extranonce2 variation bypasses dedup |
| **POOL-004** | 🟡 Medium | Stratum race | Stale-share race window between job invalidation and new job creation/broadcast |
| **VDF-001** | 🟡 Medium | VDF lane | Single-lane bottleneck; ~4–7 s/proof; solutions may reference stale challenges at high BPS |
| **BAL-001** | 🟡 Medium | BalanceRootV1 | Compute failure falls back to `[0u8;32]` — consistently-failing compute deadlocks the chain |
| **ARCH-001** | 🟡 Medium | Code size | `main.rs` 24,609 lines; `main()` fn alone is ~22,762 lines — maintenance risk |
| **ARCH-002** | 🟡 Medium | Code size | `handlers.rs` 17,019 lines; 70+ handlers mixed in one file |
| **POOL-005** | 🟡 Medium | Pool always-on | Mining pool default-enabled; `Q_ENABLE_MINING_POOL=0` required to disable |
| **DEX-005** | 🟡 Medium | DEX fee/reserve | Fee deducted from output calc but full amount added to reserves — small invariant leak |
| **WIN-001** | 🟢 Low | Windows storage | Sled OOM handling via `panic::catch_unwind`; no RocksDB parity for io_uring |

**Good news:** Mining reward double-credit path is **SAFE** (3-layer persistent dedup). BalanceRootV1 activation guard is **correctly implemented** at `height >= 18,600,000`. Bridge stubs are **safe no-ops**. u128 P2P serialization boundary is **properly handled**. systemd service file contains **no secrets**.

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

## 3. Node (q-api-server) — Deep Feature Review

### 3.1 Startup & Configuration

**Entry point:** `crates/q-api-server/src/main.rs` (24,609 lines). The `main()` function begins at line 1,847 and spans ~22,762 lines — the entire node initialization and routing is inlined into a single function. This is the largest identified maintenance risk in the codebase (see §10 ARCH-001).

**Startup sequence** (verified in iteration 2):

| Step | Line | Action |
|------|------|--------|
| 1 | ~9–15 | Memory allocator: jemalloc (Unix) or mimalloc (Windows) |
| 2 | ~1,856 | `prctl(41,1,0,0,0)` — disable THP per-process |
| 3 | ~1,876 | rustls CryptoProvider install (MUST be first) |
| 4 | ~1,879 | `.env` load (non-fatal) |
| 5 | ~1,887 | `validate_env_config()` |
| 6 | ~2,085 | clap CLI parse (20+ flags) |
| 7+ | ~2,086 | Subsystem init (storage → consensus → P2P → API) |

**Key environment variables** (complete list):

| Variable | Default | Notes |
|----------|---------|-------|
| `Q_NETWORK_ID` | `testnet` | Checked BEFORE `--network` CLI flag |
| `Q_DB_PATH` | `./data` | Must be ABSOLUTE PATH on Epsilon |
| `Q_WORKER_THREADS` | CPU count | Tokio worker threads |
| `Q_DISABLE_AI` | `0` (enabled) | Set to `1` to skip AI inference |
| `Q_DISABLE_DEX` | `0` (enabled) | Set to `1` to disable DEX |
| `Q_ENABLE_TOR` | `0` | Set to `1` for Tor anonymity |
| `Q_ENABLE_MINING_POOL` | `1` (enabled) | Must set to `0` to disable |
| `Q_STRATUM_PORT` | `3333` | Stratum listen port |
| `Q_ROCKSDB_WRITE_RATE_MB` | `200` | MB/s cap |
| `ROCKSDB_BLOCK_CACHE_MB` | RAM/3 | Explicit cap required to prevent OOM |
| `Q_TOR_BOOTSTRAP_TIMEOUT` | `120` | Set to `5` in production to avoid slow start |
| `Q_SKIP_CHECKPOINT` | `0` | For genesis sync |
| `RUST_LOG` | `info` | Must be `warn` on Epsilon (DEBUG fills 40 GB root) |
| `Q_TURBO_PARALLEL_STREAMS` | `16` | Sync parallelism |
| `Q_TURBO_CHUNK_SIZE` | `2500` | Blocks per sync chunk |

---

### 3.2 Consensus Engine (DAG-Knight)

**Crate:** `crates/q-dag-knight/src/`

DAG-Knight is a Bullshark-family asynchronous BFT consensus protocol operating on a DAG of vertices. Sub-50 ms finality achieved using δ=1 commit rule.

| Property | Value | Source |
|----------|-------|--------|
| Byzantine tolerance | f = ⌊(n-1)/3⌋ | Algorithm design |
| Finality target | <50 ms | δ=1 commit rule |
| Anchor election | Quantum VDF entropy | `quantum_vdf.rs` |
| Fork detection | Homological (Betti H₀, H₁) | `homological_consensus.rs` |
| VDF base difficulty | 2,048 (with 50% quantum boost = 2,355 effective squarings) | `quantum_vdf.rs:133` |
| Genus-2 VDF | Hyperelliptic curves, Wesolowski proofs | `genus2_vdf_integration.rs` |
| QRNG refresh | 30 seconds | `quantum_beacon.rs` |

**Vertex structure** (abbreviated):
```rust
Vertex {
  id: [u8; 32],
  round: u64,
  author: NodeId,
  parents: Vec<VertexId>,
  transactions: Vec<Transaction>,
  signature: Vec<u8>,    // Ed25519 or Dilithium5
  timestamp: DateTime<Utc>,
}
```

**Block structure** (abbreviated):
```rust
Block {
  height: u64,
  hash: [u8; 32],               // BLAKE3(all fields)
  prev_hash: [u8; 32],
  transactions: Vec<Transaction>,
  dag_round: u64,
  dag_parents: Vec<VertexId>,
  balance_root: Option<[u8; 32]>,  // BalanceRootV1, height ≥ 18,600,000
  finality_cert: Option<BullsharkCert>,
}
```

---

### 3.3 Block Production & Mining Reward Accounting

**File:** `crates/q-api-server/src/block_producer.rs`

#### Block Production Flow

| Step | Detail |
|------|--------|
| Trigger | Time-based (15 s default) OR ≥1 queued solution OR pending mempool txs |
| Solution drain | Lock-free `crossbeam::SegQueue::pop()`, max 250/block |
| Safety valve | If queue depth >5,000: discard oldest to prevent unbounded growth |
| Coinbase creation | `BalanceConsensusEngine::calculate_block_reward()` |
| Fee distribution | dev_fee_bps (100 bps = 1%), operator split, miner remainder |
| Merkle root | SIMD-accelerated (AVX-512, 8–10× speedup) with scalar fallback |
| Validation | ConsensusGuard height monotonicity + BalanceRootV1 gate |
| Broadcast | Gossipsub `/qnk/{network}/blocks` (MessagePack, u128 as STRING) |
| SSE events | NewBlock, BalanceUpdated, MiningReward emitted (<50 ms) |

#### Mining Reward Security Audit — Verdict: SAFE ✅

The reward path was fully traced through 5 code layers (verified in iteration 2):

**Layer 1 — HTTP submission** (`handlers.rs:8834`):
- Challenge freshness check: rejects challenges >5 minutes old (`line 9043`)
- Server-side difficulty override: replaces client's difficulty with canonical challenge (`line 9057`)
- Zero-lock fast path (`lines 9075–9092`): only format validation + `try_send` to shard channel; returns 200 immediately
- Nonce dedup disabled at HTTP layer (`line 9061`, v8.9.0): dedup moved to batch processor

**Layer 2 — Background batch processor**:
- VDF verification with 30-second hard timeout (`main.rs:15963`)
- Real validation and stats updates happen asynchronously

**Layer 3 — Balance update** (`balance_consensus.rs:680`):
Three-layer deduplication prevents double-credit:

```
Dedup Layer 1 (PERSISTENT): RocksDB key "processed_balance_block:{hash}"
  → Survives node restarts (written atomically with balance update)
  → Line 700-728 checks this FIRST

Dedup Layer 2 (IN-MEMORY): LRU cache, 100k entries (~5 MB)
  → Fast secondary check
  → Falls back to persistent on cache miss

Dedup Layer 3 (ATOMIC TX): Balance update + dedup key in single RocksDB tx
  → Partial updates impossible
```

**Verified test coverage** (`crates/q-storage/tests/balance_consensus_integration.rs`):
- `test_double_processing_safety` (`line 132`): Second call returns `AlreadyProcessed`, balance unchanged
- `test_deterministic_consensus` (`line 58`): Two independent nodes produce identical balances
- `test_five_node_consensus` (`line 178`): 5 nodes, 10 blocks, identical final state

**Emission controller determinism** (`emission_controller.rs:1181`):
- Pure `u128` integer math — no floating-point drift across nodes
- Same `(block_height, timestamp)` → same reward on every node
- Budget correction factor bounded to [0.01, 3.0]

**Conclusion:** No double-credit, no missed credit, no reward drift between nodes.

---

### 3.4 Storage Layer

**Crate:** `crates/q-storage/src/`

#### Backend Selection

| Platform | Backend | Notes |
|----------|---------|-------|
| Linux / macOS | RocksDB 0.22 | Multi-threaded, column families |
| Windows | Sled 0.34 | Pure Rust; different performance/safety profile (see WIN-001) |

#### RocksDB Column Families (12 verified)

| CF | Content |
|----|---------|
| `CF_BLOCKS` | Blocks by height |
| `CF_BALANCES` | Wallet balances (u128) |
| `CF_STATE` | Smart contract state |
| `CF_TRANSACTIONS` | Transactions by TxHash |
| `CF_VERTICES` | DAG vertices |
| `CF_CERTIFICATES` | Bullshark certs |
| `CF_CONTRACTS` | Contract registry |
| `CF_MANIFEST` | Node metadata + dedup keys |
| `CF_PEER_TRUST` | Peer reputation |
| `CF_MINING_STATS` | Per-miner stats |
| `CF_EMAIL` | Blockchain email |
| `CF_CALENDAR` | Calendar events |

#### Windows / Sled Storage (WIN-001)

Sled emulates column families using separate trees (`kv_sled.rs:117`). Key limitations:

- **OOM behavior:** `apply_batch()` can **panic** instead of returning `Err`. Handled via `std::panic::catch_unwind` (`kv_sled.rs:128`). Falls back to individual inserts (slower, higher memory pressure).
- **Cache overshoot:** Sled's page cache overshoots `cache_capacity` by up to 10× under burst writes. Default cap lowered from 128 MB to 64 MB in v9.3.3. Peak usage ~640 MB on configured 64 MB.
- **io_uring:** Not available on Windows; async I/O falls back to standard Tokio I/O.
- **Batch chunk size:** 2,000 ops/batch (`kv_sled.rs:223`) to limit memory spike.

---

### 3.5 P2P Networking & Turbo Sync

**Crate:** `crates/q-network/src/`

#### Gossipsub Topics

| Topic | Purpose |
|-------|---------|
| `/qnk/{net}/blocks` | Block announcements |
| `/qnk/{net}/mempool-txs` | Transaction relay |
| `/qnk/{net}/peer-heights` | Height sync |
| `/qnk/{net}/turbo-sync-{shard}` | Batch block sync |
| `/qnk/{net}/mining-solutions` | Solution relay |
| `/qnk/{net}/ai-inference-{model}` | Distributed AI tensors |

#### Turbo Sync Performance

| Config | Value |
|--------|-------|
| Parallel streams | 16 per peer |
| Chunk size | 2,500 blocks |
| Compression | Zstd level 1 |
| Chunk timeout | 30 seconds |
| Peak speed (Epsilon 10 Gbit) | ~1,100 blocks/sec |
| Full sync (~11.4M blocks) | ~5.5 hours |

#### u128 Serialization Boundary (A-005 — Verified SAFE ✅)

All u128 fields in Transaction use a custom `u128_serde` module (`q-types/src/lib.rs:10`):

```rust
// lib.rs:2192 — Transaction.amount
#[serde(with = "u128_serde")]
pub amount: Amount,  // u128

// u128_serde::serialize (line 29-43):
// ALWAYS serializes as STRING for MessagePack P2P safety
// "MessagePack silently truncates u128 to u64 — always use string"
serializer.serialize_str(&value.to_string())
```

Block, Certificate, and PeerHeightWithProof structs contain **no u128 fields** — only `u64`, `[u8; 32]`, and `Vec<u8>`. No unprotected u128 in P2P message paths found.

**rmp-serde version:** 1.3 (used in q-network). Does not natively handle u128 correctly; custom module is essential.

---

### 3.6 REST API Surface

**Framework:** Axum 0.7 · 228+ endpoints · AEGIS-QL auth · governor 0.6 rate limiting

Full API surface unchanged from v1. Key notes:

- **Balance/wallet endpoints require auth.** Curl without auth tokens returns empty — not an error.
- **Rate limiting:** Token bucket per IP. AI inference and ZK proof endpoints share the same limiter as lightweight endpoints — consider per-category limits (see §10 S-002).

#### Handler Organization (A-002 — 17,019 lines verified)

70+ handler functions in a single file. Top handlers mapped:

| Handler | Line | Domain |
|---------|------|--------|
| `health_check` | 297 | Status |
| `node_status` | 796 | Status |
| `get_block_by_height` | 735 | Blockchain |
| `submit_transaction` | 2,134 | Transactions |
| `send_transaction` | 3,771 | Transactions |
| `get_wallet_transaction_history` | 4,807 | Wallet |
| `bitcoin_bridge_status` | 5,356 | Bridges |
| `get_mining_challenge` | ~8,600 | Mining |
| `submit_mining_solution` | 8,834 | Mining |

---

### 3.7 Real-Time Streaming (SSE/WebSocket)

```
GET /api/v1/events?wallet_address=...&headers_only=true&miner_mode=true
WS  /api/v1/ws?wallet_address=...
```

| Property | Value |
|----------|-------|
| Latency target | <50 ms |
| Throughput | 10K+ events/sec |
| Buffer | 1,000-event broadcast channel |
| Miner mode bandwidth | 2–5 KB/s (vs 111 KB/s full) |
| Implementation | Tokio `broadcast::Sender` + `crossbeam::SegQueue` |

---

### 3.8 BalanceRootV1 Activation Guard

**Full audit conducted in iteration 2. Verdict: CORRECTLY IMPLEMENTED with one caveat.**

#### Gate Condition

**File:** `block_producer.rs:970`
```rust
let state_root = if q_consensus_guard::is_upgrade_active(
    q_consensus_guard::Upgrade::BalanceRootV1,
    next_height,   // uses next_height, not current
) { ... }
```

**Upgrade gate** (`q-consensus-guard/src/upgrade_gate.rs:138`):
```rust
upgrades.insert(Upgrade::BalanceRootV1, UpgradeConfig {
    activation_height: 18_600_000,
    mandatory: true,
    min_version: "10.6.0".to_string(),
});
```

`is_active()` uses `height >= activation_height` (line 234). **No off-by-one.**

#### Peer Block Validation

When a block arrives from a peer at height ≥ 18,600,000 (`main.rs:11220`):

```rust
if block.header.state_root == [0u8; 32] {
    error!("REJECT block {} — missing balance root");
    return;  // Block rejected
}
let local_root = storage.compute_balance_root_for_block().await?;
if local_root != block.header.state_root {
    error!("REJECT block {} — balance root mismatch");
    return;  // Block rejected
}
```

Validation happens **before** applying transactions. Malicious peers cannot inject blocks with wrong or missing balance roots.

#### Determinism Guarantee

5 tests in `crates/q-storage/tests/balance_determinism_tests.rs` prove:
1. Two nodes with identical state → identical roots
2. Root survives restart (stored in RocksDB)
3. Insertion order does not affect root
4. Zero-balance wallets excluded deterministically
5. Single unit balance change produces different root

#### Known Risk (BAL-001)

**File:** `block_producer.rs:983`
```rust
Ok(root) => Some(root),
Err(e) => {
    warn!("Failed to compute balance root: {} — using zero sentinel", e);
    Some([0u8; 32])  // ← FALLBACK
}
```

If `compute_balance_root_for_block()` fails consistently (e.g. storage corruption), the producer emits blocks with `state_root = [0u8;32]`. Peers at height ≥ 18,600,000 will reject these blocks (zero root is explicitly rejected at line 11231). **Effect: the chain halts.** This is safe (no incorrect state accepted) but requires operator intervention.

**Recommendation:** Add a startup health check that verifies `compute_balance_root_for_block()` succeeds before accepting mining submissions near activation height.

---

### 3.9 Emission Controller

| Parameter | Value |
|-----------|-------|
| Max supply | 21,000,000 QUG |
| Decimal precision | 24 |
| Genesis timestamp | 1771761600 (2026-02-22 12:00 UTC) |
| Era 0 annual emission | 2,625,000 QUG |
| Halving interval | 126,230,400 s (4 × 365.25 days) |
| Total eras | 64 (~256 years) |
| Max reward/block | 2 QUG |
| Correction factor bounds | [0.01, 3.0] |
| Arithmetic | Pure u128 (no floating-point drift) |

---

### 3.10 Mining Pool Server (Stratum)

**Crate:** `crates/q-mining-pool/src/`  
**Version:** 2.2.1-beta  
**Protocol:** Stratum v1 (JSON-RPC over TCP)  
**Port:** `Q_STRATUM_PORT` (default 3333)  
**Payout:** PPLNS

**POOL-005:** Pool is **always enabled by default** (`Q_ENABLE_MINING_POOL` defaults to `"1"` at `main.rs:4284`). Must explicitly set `Q_ENABLE_MINING_POOL=0` to disable. Port binding failure is non-fatal (pool task exits, server continues), but operator may not notice the pool is down.

Detailed security findings in §4.8.

---

### 3.11 Bridge Stubs

**Audit verdict: SAFE ✅**

**Bitcoin bridge** (`handlers.rs:5356`): Returns `{"active": false, connected_peers: 0, ...}` with HTTP 200. No phantom state created.

**Ethereum, Zcash, Monero bridges:** Handler functions do not exist in the codebase. Endpoints in API documentation refer to future work.

**Atomic swap opcodes (0x90–0x93):**
- `validate_tx_type()` returns `Ok(())` (no rejection)
- `StateProcessor.execute_tx()` hits the `_ => { warn!("Unimplemented"); Ok(()) }` branch
- Fee IS deducted and nonce IS incremented
- **No escrow state created, no phantom swaps possible**
- Net effect: user loses transaction fee (~0.000021 QUG), tx stored in chain as a no-op

---

## 4. Miner (q-miner) — Deep Feature Review

### 4.1 Mining Algorithm

**Legacy mode** (below `GENUS2_VDF_MINING` activation): 100× iterated BLAKE3. Each thread: `blake3(challenge ‖ nonce)` → 99 further `blake3(h)` rounds.

**Genus-2 VDF mode** (above activation): Sequential Jacobian squaring over hyperelliptic curve (non-parallelizable). Produces Wesolowski proofs (O(log T) verification). Challenge specifies `vdf_target_iterations` (default: 4,300).

---

### 4.2 CPU Mining Engine

| Property | Value | Source |
|----------|-------|--------|
| Thread model | `spawn_blocking` per thread | Keeps tokio scheduler free |
| Core affinity | Pinned via `core_affinity` | `cpu/mod.rs:59` |
| Nonce partition | `thread_id << 48` | Cross-thread collision prevention |
| Batch size | `intensity × 100,000` nonces | Intensity 1–10, default 7 |
| SIMD detection | AVX2, AVX-512, NEON | Auto-selected |
| Allocator | jemalloc (Linux), mimalloc (Windows) | Memory efficiency |
| Hash counter | Flushed every 1,024 hashes | Reduces atomic contention |
| Stale detection | `new_block_signal` checked every 512 nonces | Lock-free atomic |

---

### 4.3 GPU Mining Engine

| Property | Value |
|----------|-------|
| Framework | OpenCL 3.0 (primary); CUDA via `cudarc` (feature flag) |
| Kernel | BLAKE3 × 100 rounds |
| Kernel optimizations | `#pragma unroll 9`, `__constant` challenge cache, persistent buffers |
| Dispatch strategy | Adaptive [100 ms, 400 ms] per kernel |
| Multi-GPU | Per-GPU independent auto-tuning (v10.1.7+) |
| Kernel cache | `~/.config/q-miner/kernel-cache/` |
| GPU nonce space | Starts at `u64::MAX / 2` (no CPU collision) |
| Stale detection | `gpu_new_block_signal` checked between dispatches |

---

### 4.4 VDF Lane — Bottleneck Analysis (VDF-001)

**File:** `crates/q-miner/src/vdf_lane.rs`

#### Measured performance

VDF difficulty: 2,048 base × (1 + 0.5 × 0.3) = **~2,355 effective squarings** on genus-2 Jacobian curve.

Estimated wall-clock per evaluation: **4–7 seconds** (genre-2 squarings take ~2–3 ms each on modern hardware, `vdf_lane.rs:10` comment confirms single-core sequential).

#### Staleness detection

```rust
// vdf_lane.rs:152-156
if new_block_signal.load(Ordering::Relaxed) != block_signal_before {
    debug!("VDF: new block arrived during eval, restarting");
    continue;  // DISCARD work, start over
}
```

At a 10 BPS block rate (100 ms/block) and 4–7 s VDF evaluation, the lane discards work on virtually every attempt. Even at 0.1 BPS (10 s/block), the VDF lane completes roughly 1–2 proofs per block arrival.

#### Server-side timeout

```rust
// main.rs:15963
tokio::time::timeout(
    Duration::from_secs(30),
    tokio::task::spawn_blocking(|| verify_vdf_proof(...))
)
```

30-second hard timeout on server-side VDF verification. Prevents mining loop saturation if verifier stalls. On timeout, solution is rejected.

#### Security implication

VDF solutions submitted for stale challenges (block N-1) are caught by the **challenge freshness check** (`handlers.rs:9043` — rejects >5 min old challenges). At practical block rates this is fine. Only a concern if block interval is shorter than VDF evaluation time and the operator configures very high VDF iterations.

**Recommendation:** Expose `Q_VDF_ITERATIONS_CAP` to cap per-challenge iterations to a value that completes within 80% of the expected block interval.

---

### 4.5 Pool vs Solo vs P2P Modes

| Mode | Protocol | Reward | Notes |
|------|----------|--------|-------|
| Solo | HTTP POST `/api/v1/mining/submit` | Direct to wallet | Best for large miners |
| Pool | Stratum v1 JSON-RPC over TCP | PPLNS pool payout | Default: `stratum+tcp://quillon.xyz:3333` |
| P2P | libp2p gossipsub | CRDT-based PPLNS | <50 ms challenge relay |

---

### 4.6 Communication with Node

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/mining/challenge` | GET | Fetch current challenge |
| `/api/v1/mining/submit` | POST | Submit solution (zero-lock fast path) |
| `/api/v1/status` | GET | Node health |
| `/api/v1/miner/device-login` | POST | MinerLink device auth |
| `/api/v1/events` (SSE) | GET | Real-time block signals |

**Challenge fields:** `challenge_hash`, `difficulty_target`, `block_height`, `vdf_iterations`, `block_reward`, `expires_at`

---

### 4.7 FPGA/RTL Implementation

**Path:** `qug-v1-rtl/rtl/`  
**Target:** Xilinx Kintex-7 XC7K325T at 100 MHz  
**Design:** 16-core RISC-V (RV32IMC) SoC with two custom ISA extensions

#### Xcrypto (custom-0, `0x0B`) — IMPLEMENTED ✅

Hardware-accelerated BLAKE3 in `rtl/xcrypto/`:
- `blake3_round.sv`: 2-stage pipelined round unit
- `xcrypto_unit.sv`: Top-level dispatch
- Instructions: `blake3.init`, `blake3.round`, `blake3.chain`, `blake3.finalize`, `blake3.ldmsg`, `blake3.status`
- Throughput target: 1 hash / 14 cycles @ 100 MHz → ~114 MH/s (16 cores)

#### Xlattice (custom-1, `0x2B`) — PARTIALLY STUBBED ⚠️

**Status confirmed in iteration 2** (`xlattice_unit.sv:93`):

| Operation | funct7 | Status |
|-----------|--------|--------|
| `ntt.fwd` | 0 | **STUB** — returns 0 |
| `ntt.inv` | 1 | **STUB** — returns 0 |
| `poly.add` | 2 | **IMPLEMENTED** (`mod_add_256.sv`) |
| `poly.mul` | 3 | **IMPLEMENTED** (`mod_mul_256.sv`, 15 KB pipelined) |
| `poly.reduce` | 4 | **STUB** — returns 0 |

**Technical note** (`doc/technical-review-rtl-v4.md §10`): NTT is needed for Dilithium5 signature verification on a full-node card (Phase 1B), NOT for mining. The current ASIC mining product only needs Xcrypto (BLAKE3). Estimated NTT-256 implementation: ~512 cycles using existing `mod_mul_256`, ~4,000 LUTs + 16 DSPs.

**Opcode encoding** (`qug_pkg.sv`): `OPC_CUSTOM_1 = 7'b010_1011` (0x2B) properly defined. Decoder recognizes instructions; execution is stubbed.

---

### 4.8 Stratum Pool Security Audit

**Critical findings from iteration 2:**

#### POOL-001 — Share Dedup Cache Full-Clear Vulnerability 🟠 HIGH

**File:** `share.rs:260`
```rust
if cache.len() >= self.max_cache_size {  // 100,000 entries
    cache.clear();  // Full eviction — ALL dedup state lost
}
cache.insert(unique_id);
```

A miner can submit 100,001 unique shares to flush the dedup cache, then resubmit any previous share without detection. With multiple workers, this threshold is reachable quickly.

**Fix:** Replace `HashSet` + clear with a bloom filter or LRU-eviction deque.

#### POOL-002 — Share Difficulty Not Enforced Synchronously 🟠 HIGH

**File:** `stratum.rs:511`
```rust
Ok(Some(StratumResponse::success(msg.id, json!(true))))
// Always returns "true" immediately; validation deferred to pool.rs
```

Workers receive acceptance confirmation before difficulty validation. No minimum share difficulty enforced at the TCP handler layer. Vardiff minimum (`VardiffConfig.min_difficulty`) adjusts over time but has no hard floor.

**Fix:** Enforce `share_difficulty >= pool.min_difficulty` synchronously before sending `true` to worker.

#### POOL-003 — Replay Attack via Extranonce2 Variation 🟡 Medium

**File:** `share.rs:234`
```rust
let unique_id = format!("{}:{}:{}", submission.job_id, submission.nonce, &submission.extranonce2);
```

The dedup key includes `extranonce2`, which is **miner-controlled**. A malicious miner can submit the same `(job_id, nonce)` pair with different `extranonce2` values — each generates a distinct unique_id, bypassing dedup. Each variant produces the same block hash (nonce determines the PoW), but the pool may credit multiple shares.

**Fix:** Dedup key should be `hash(block_header)` or `(job_id, nonce)` only, excluding miner-controlled extranonce2.

#### POOL-004 — Stale-Share Race Window 🟡 Medium

**File:** `pool.rs:395`
```rust
self.job_manager.invalidate_all();
self.job_manager.create_job(template, false)?;  // ← clean_jobs should be true
```

On block found: jobs invalidated, then new job created. In the window between these two operations, workers have no valid job. Shares submitted in this window reference the now-invalid old job but the new job hasn't been broadcast yet. Some old-job shares may still pass the 120-second expiry check.

**Fix:** Pass `clean_jobs = true` to `create_job` so stale shares are explicitly marked invalid in the new job broadcast.

#### POOL-005 — Pool Default-Enabled (Confirmed)

`main.rs:4284`: `Q_ENABLE_MINING_POOL` defaults to `"1"`. Port 3333 bound unconditionally when enabled. Port-in-use error is logged and pool task exits silently — operator may not notice.

---

## 5. Cryptography Layer

### 5.1 Classical Cryptography

| Algorithm | Use | Location |
|-----------|-----|---------|
| Ed25519 | Phase 0 signing, address derivation | `q-wallet` |
| SHA3-256 | Block/tx hash, address hash | `q-types` |
| BLAKE3 | Mining PoW, Merkle trees | `q-miner`, `q-crypto-simd` |
| AES-256-GCM | Wallet + storage encryption | `q-wallet`, `q-storage` |
| Argon2id | Key derivation (64 MB cost, 4 iter) | `q-wallet`, `q-storage` |
| BIP39 | 12-word mnemonic | `q-wallet` |
| AEGIS-256 | Authenticated encryption (2–5× AES-GCM speed, AES-NI) | `q-crypto-advanced` |

### 5.2 Post-Quantum Cryptography

| Algorithm | Standard | Signature size | Phase |
|-----------|----------|---------------|-------|
| Dilithium5 | ML-DSA (NIST) | ~3,300 B | Q1/Q2 |
| Kyber1024 | ML-KEM (NIST) | — (KEM) | Q1/Q2 |
| SQIsign | Isogeny-based (NIST candidate) | 204 B | Advanced |
| SPHINCS+ | Hash-based (NIST) | 7,856 B | Alternative |
| Genus-2 VDF | Hyperelliptic curves | — | Mining |
| FROST | Threshold Schnorr (IACR 2025/1024) | — | Validator committee |
| Ring-LWE L-VRF | Lattice VRF | — | Anchor election |

**Phase deployment:**

| Phase | Signing | Notes |
|-------|---------|-------|
| Q0 | Ed25519 | Legacy |
| Q1 | Ed25519 + Dilithium5 | Hybrid, defense-in-depth |
| Q2 | Dilithium5 | Full PQ |

### 5.3 Zero-Knowledge Proof Systems

| System | Proof size | Trusted setup | Quantum-safe | Use |
|--------|-----------|--------------|-------------|-----|
| Circle STARKs (IACR 2024/278) | ~60 KB | None | ✅ | Private txs |
| Groth16/PLONK SNARKs | 96–192 B | Required | ❌ | Contract proofs |
| Bulletproofs v2 (IACR 2024/313) | O(log n) | None | ❌ | Range proofs |
| Recursive SNARKs | Succinct | None | ❌ | Light client bootstrap |

### 5.4 Privacy Primitives

| Primitive | Status |
|-----------|--------|
| Dandelion++ | Production — mandatory for tx relay |
| Tor (Arti embedded) | Integrated — `Q_ENABLE_TOR=1` |
| Ring signatures (0x82) | Production — `RingTransfer` tx type |
| Shielded transfers (0x83) | Integrated — Circle STARK + nullifier set |
| AEGIS-QL access control | Production — middleware for API |
| Quantum mixing pool | Integrated — `/api/v1/mixer/` |

---

## 6. DeFi & Smart Contracts

### 6.1 DEX — Constant-Product AMM Audit

**AMM model:** Quantum-enhanced constant product (x × y = k)  
**Fee:** 0.30% total (0.05% protocol + 0.25% LP)

**Full audit results from iteration 2:**

#### DEX-001 — No Invariant Check After Swap 🔴 CRITICAL

**File:** `crates/q-dex/src/trading.rs:242`

`execute_quantum_trade()` computes a price with quantum-physics adjustments (uncertainty principle, entanglement, wave function collapse) and returns `QuantumExecutionResult` **without updating pool reserves and without verifying that x×y=k is preserved**.

Code flow:
```
execute_quantum_trade()
  → apply_uncertainty_principle()    // adds physics noise
  → calculate_entanglement_effect()  // adds adjustments
  → collapse_wave_function()         // adds golden_ratio + entanglement
  → calculate_quantum_fees()
  → return QuantumExecutionResult    // ← no reserve update, no k check
```

Pool reserves ARE updated in `liquidity.rs:update_pool_reserves()`, but this function is called only for add/remove liquidity, not for swaps. **Swaps do not mutate pool reserves.**

**Impact:** The DEX records trade statistics and charges fees but the on-chain pool state (reserves) never changes from the trading engine. Either swaps are not actually finalizing on-chain (meaning the DEX is price-discovery only), or there is a missing integration between `trading.rs` and `liquidity.rs`.

#### DEX-002 — Concurrent Swap Race Condition 🔴 CRITICAL

**File:** `crates/q-dex/src/liquidity.rs:19`

```rust
pub struct QuantumLiquidityManager {
    pub quantum_pools: Arc<RwLock<HashMap<String, QuantumLiquidityPool>>>,
}
```

The lock covers the pool map but `execute_quantum_trade()` does not hold a write lock for the duration of the read-compute-write cycle. Two simultaneous swaps can:

1. Both read pool state `(A, B, k)` under read lock
2. Both compute outputs based on stale state
3. Both attempt to update reserves

Since reserves are never actually updated by swaps (DEX-001), this is currently non-exploitable. However, if the reserve update path is reconnected, this race becomes a direct invariant violation.

#### DEX-003 — Slippage Not Enforced 🟠 HIGH

**File:** `crates/q-dex/src/types.rs:182`

```rust
pub max_slippage_bps: u16,  // present in request
```

**File:** `crates/q-dex/src/trading.rs`

The `execute_quantum_trade()` function never reads `max_slippage_bps`. A user requesting 0.5% max slippage receives no protection — the trade executes regardless of actual price impact.

#### DEX-004 — No Minimum Pool Reserve 🟠 HIGH

No `MIN_POOL_RESERVE` or `MINIMUM_POOL_RESERVE` constant found anywhere in `crates/q-dex/`. A liquidity provider can remove reserves until dust remains, creating near-zero denominator in `get_quantum_price()` → division producing astronomical prices.

Historical precedent: "broken pools with 8-decimal reserves caused trillion-dollar prices" (from codebase comments).

**Fix:** Enforce `reserve_a >= MIN_POOL_RESERVE && reserve_b >= MIN_POOL_RESERVE` on remove liquidity.

#### DEX-005 — Fee/Reserve Mismatch 🟡 Medium

**File:** `crates/q-dex/src/liquidity.rs:476`

Output calculated using `amount_in_with_fee` (fee-reduced input):
```rust
let amount_in_with_fee = amount_in * (1000 - fee_rate * 1000) / 1000;
let amount_out = amount_in_with_fee * reserve_out / (reserve_in + amount_in_with_fee);
```

But reserves updated with full `amount_in`:
```rust
pool.token_a_reserve = &pool.token_a_reserve + amount_a;  // full amount, not fee-reduced
```

Small invariant leak: k increases slightly per swap (by the fee amount). This is intentional in some AMM designs (fees accumulate as LP value) but appears unintentional here since there is no explicit fee accounting ledger.

#### DEX-006 — 24-Decimal Overflow Risk 🟡 Medium (Mitigated by BigDecimal)

**File:** `crates/q-dex/src/liquidity.rs:418`

```rust
pool.quantum_k_invariant = &pool.token_a_reserve * &pool.token_b_reserve;
```

With 1M tokens at 24 decimals: reserves = 10^30 base units. k = 10^60 — overflows `u128` (max ~3.4×10^38).

**Mitigation:** DEX uses `BigDecimal` (arbitrary precision). Overflow prevented. Performance degrades with very large numbers but no correctness issue.

**Risk:** If the codebase is ever migrated to `u128` integer arithmetic (as historical versions were), this would become a critical exploit.

### 6.2 Smart Contract VM

**Runtime:** Wasmer 4.0.0 (JIT) + Wasmtime 14.0.0  
**Language:** WebAssembly  
**Deployment:** Post-quantum Dilithium5 signatures (v3.7.4)

25+ contract types across 6 domains (DeFi, RWA, derivatives, governance, identity, utility). WASM gas metering, cross-contract calls, parallel execution.

### 6.3 Transaction Taxonomy

46 distinct opcodes across 11 domains (0x00–0xFF). Key domains:

| Range | Domain |
|-------|--------|
| 0x00–0x0F | Core (Transfer, Coinbase, Burn, Fee) |
| 0x20–0x2F | DEX (Pool, Swap, FlashLoan) |
| 0x80–0x8F | Privacy (Dandelion++, Ring, Shielded) |
| 0x90–0x9F | Cross-chain atomic swaps (currently no-op) |
| 0xF0–0xFF | System (EmergencyPause, StateCheckpoint) |

### 6.4 Token System

| Token | Decimals | Max supply | Notes |
|-------|---------|-----------|-------|
| QUG | 24 | 21,000,000 | Native, deflationary |
| QUGUSD | 24 | Collateral-bounded | Algorithmic stablecoin |
| QCREDIT | 24 | 1:1 QUG lock | Yield vault (v8.5.5) |
| QUSD | 24 | Founder-controlled | Issuer stablecoin |
| wBTC/wETH | 8/18 | 1:1 collateral | Bridge tokens |
| VAULT/FORGE | 0 | Physical device | RWA tokens |

---

## 7. Frontend Wallet

### 7.1 Quantum Wallet (React/TypeScript)

14+ navigable screens with SSE-driven real-time sync. Key verified details:

- **Balance sanity:** `≤ 21,000,000 QUG` max enforced in frontend — corrupted SSE data rejected
- **Block-height monotonic tracking:** Prevents balance "zigzag" from out-of-order events
- **DEX cooldown:** Prevents stale balance overwrites after swaps
- **QCI formula:** `peers_score (30%) + block_production_score (30%) + health_score (40%)`
- **Particle physics login:** `ψ(x,t) = A·e^(i(kx−ωt+φ))·sin(nπx/L)` driven by live `/api/v1/health`
- **Multi-server failover:** Return-to-primary every 60 s; 30 s minimum failover cooldown

### 7.2 Slint Native Wallet

**Framework:** Slint v1.9 (Rust backend)  
**Features:** BIP39 create/restore, QR code, SSE balance, OAuth2 (PKCE), CPU+OpenCL mining, auto-update binary v1.5, POS mode

---

## 8. Infrastructure & Deployment

### 8.1 Multi-Server HA Topology

| Server | Role | Bandwidth | Nginx weight |
|--------|------|-----------|-------------|
| Beta (185.182.185.227) | Primary bootstrap | 100 Mbit | 10 |
| Gamma (109.205.176.60) | HA backup | 1 Gbit | 1 |
| Delta (5.79.79.158) | Canary | 1 Gbit | canary |
| Epsilon (89.149.241.126) | 10 Gbit supernode | 10 Gbit | DNS primary |

**Nginx:** `ip_hash` sticky sessions. Ensures same user always hits same backend — prevents balance flickering between nodes with slightly different states.

### 8.2 Rolling Deployment Pipeline

**Script:** `scripts/ha-deploy.sh`

| Step | Action | Duration |
|------|--------|---------|
| `verify-delta` | Deploy to canary, 7+ min soak | ~7 min |
| `verify-gamma` | Deploy to backup, 90 s stability | ~2 min |
| `promote` | Gamma weight=10, Beta weight=1 | Immediate |
| `deploy-beta` | Replace primary binary | ~2 min |
| `restore` | Beta weight=10, Gamma weight=1 | Immediate |

**Key guards:** lock file prevents concurrent deploys; version bump enforced; max 10 min health wait per server; auto-rollback on failure.

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
| Stratum pool | ✅ Production | PPLNS — security issues (see §4.8) |
| CPU miner | ✅ Production | Multi-thread, AVX2/512 |
| GPU miner (OpenCL) | ✅ Production | Persistent buffers, adaptive dispatch |
| Genus-2 VDF mining | ✅ Production | Sequential lane bottleneck noted |
| FPGA RTL (Xcrypto) | 🟡 Prototype | Kintex-7 BLAKE3 pipeline |
| FPGA RTL (Xlattice NTT) | ⚠️ Stub | Phase 1B; poly.add/mul implemented |
| React wallet (14 screens) | ✅ Production | SSE sync, DEX, mining |
| Slint native wallet | ✅ Integrated | GPU mining, auto-update |
| DEX (constant product) | ⚠️ Partial | AMM price logic present; reserve updates disconnected from trading engine (DEX-001) |
| WASM smart contracts | ✅ Integrated | Wasmer 4.0, 25+ types |
| AI inference (Mistral) | ✅ Integrated | CUDA/Metal optional |
| Bitcoin bridge | 🟡 Stub | `active: false`; no phantom state |
| Other bridges | ⚠️ Not implemented | No handler functions |
| Atomic swaps (0x90–0x93) | ⚠️ No-op | Silent pass, fee only |
| Sharding | ⚠️ Disabled | Crate exists, disabled in production |
| Windows / Sled storage | 🟡 Limited | OOM panic-catch; no io_uring parity |
| HA rolling deploy | ✅ Production | Zero-downtime 4-server pipeline |

---

## 10. Issues Register — v2 (Verified & Expanded)

### 🔴 Critical

| ID | Area | Description | File:Line |
|----|------|-------------|----------|
| **DEX-001** | DEX trading | `execute_quantum_trade()` does not update pool reserves and does not verify x×y=k invariant after swap | `q-dex/src/trading.rs:242` |
| **DEX-002** | DEX concurrency | No atomic lock over read→compute→write during swap; concurrent swaps use stale reserve reads | `q-dex/src/liquidity.rs:19` |

### 🟠 High

| ID | Area | Description | File:Line |
|----|------|-------------|----------|
| **DEX-003** | DEX slippage | `max_slippage_bps` in `TradeRequest` never validated in execution path | `q-dex/src/trading.rs` |
| **DEX-004** | DEX reserves | No `MIN_POOL_RESERVE` constant; reserves can drain to dust (trillion-dollar price risk) | `q-dex/src/liquidity.rs` |
| **POOL-001** | Stratum dedup | Share dedup `HashSet` cleared entirely at 100k entries; previously-seen shares re-accepted after flush | `q-mining-pool/src/share.rs:260` |
| **POOL-002** | Stratum difficulty | Min share difficulty not enforced synchronously; workers receive `true` before validation | `q-mining-pool/src/stratum.rs:511` |

### 🟡 Medium

| ID | Area | Description | File:Line |
|----|------|-------------|----------|
| **DEX-005** | DEX invariant | Fee deducted from output calc but full amount added to reserves — untracked invariant leak | `q-dex/src/liquidity.rs:476` |
| **POOL-003** | Stratum replay | Dedup key includes miner-controlled `extranonce2`; same nonce with varied extranonce2 bypasses dedup | `q-mining-pool/src/share.rs:234` |
| **POOL-004** | Stratum race | `clean_jobs=false` bug; stale shares accepted during block-found → new-job window | `q-mining-pool/src/pool.rs:397` |
| **POOL-005** | Pool always-on | `Q_ENABLE_MINING_POOL` defaults to `"1"`; must explicitly disable; port error is silent | `main.rs:4284` |
| **VDF-001** | VDF bottleneck | Single VDF lane ~4–7 s/proof; high BPS rates cause most evaluations to be discarded | `q-miner/src/vdf_lane.rs:142` |
| **BAL-001** | BalanceRootV1 | Compute failure → `[0u8;32]` fallback → blocks rejected by peers → chain halt | `block_producer.rs:983` |
| **ARCH-001** | Code size | `main.rs` 24,609 lines; `main()` fn ~22,762 lines | `q-api-server/src/main.rs:1847` |
| **ARCH-002** | Code size | `handlers.rs` 17,019 lines; 70+ handlers in one file | `q-api-server/src/handlers.rs` |
| **S-002** | Rate limiting | AI inference + ZK proof endpoints share rate limiter with lightweight endpoints | `main.rs` route setup |
| **WIN-001** | Windows storage | Sled OOM handled via `panic::catch_unwind`; cache overshoots 10× configured value | `q-storage/src/kv_sled.rs:128` |
| **DEX-006** | DEX overflow | k = reserve_a × reserve_b overflows u128 at 24-decimal scale — mitigated by BigDecimal | `q-dex/src/liquidity.rs:418` |

### 🟢 Low / Confirmed Acceptable

| ID | Area | Description | Status |
|----|------|-------------|--------|
| **I-002** | FPGA Xlattice | NTT ops stubbed; poly.add/mul implemented; NTT is Phase 1B, not blocking mining | Documented |
| **CUDA** | GPU | CUDA available via feature flag but OpenCL is preferred path | Acceptable |
| **Bridges** | Cross-chain | Bridge endpoints return safe empty/false responses; no phantom state possible | SAFE |
| **AtomicSwap** | 0x90–0x93 | Silent no-op with fee deduction; no escrow created | SAFE |
| **S-001** | systemd | Service file audited: no secrets, no passphrase in `Environment=` | CLEAN |
| **u128** | Serialization | Custom `u128_serde` module correctly serializes as STRING for P2P | SAFE |
| **DoubleCredit** | Mining rewards | 3-layer persistent dedup; no double-credit possible | SAFE |
| **BalanceRoot gate** | BalanceRootV1 | `height >= 18,600,000` correct; determinism proven by 5 tests | SAFE |

---

## 11. Recommendations Roadmap

### Priority 1 — Fix Before Next Mainnet Feature Release

**DEX-001 + DEX-002: Connect trading engine to pool state**

The trading engine (`trading.rs`) and liquidity manager (`liquidity.rs`) are disconnected — swaps compute prices but never write reserve state. Implement:

```rust
// In execute_quantum_trade(), after computing output:
let mut pools = self.liquidity_manager.quantum_pools.write().await;
let pool = pools.get_mut(&pair_id).ok_or(DexError::PoolNotFound)?;
// Atomic under write lock:
let new_reserve_in = pool.reserve_in.checked_add(amount_in)?;
let new_reserve_out = pool.reserve_out.checked_sub(amount_out)?;
// Verify invariant:
assert!(new_reserve_in * new_reserve_out >= pool.k_invariant);
pool.reserve_in = new_reserve_in;
pool.reserve_out = new_reserve_out;
```

**DEX-003: Enforce slippage**

```rust
let max_out = expected_out * (10_000 + request.max_slippage_bps as u128) / 10_000;
let min_out = expected_out * (10_000 - request.max_slippage_bps as u128) / 10_000;
if actual_out < min_out {
    return Err(DexError::SlippageExceeded { expected: min_out, actual: actual_out });
}
```

**DEX-004: Add minimum reserve constant**

```rust
const MIN_POOL_RESERVE: u128 = 10_u128.pow(22);  // 10^22 base units (0.00001 tokens at 24 decimals)

fn remove_liquidity(...) -> Result<()> {
    if new_reserve_a < MIN_POOL_RESERVE || new_reserve_b < MIN_POOL_RESERVE {
        return Err(LiquidityError::BelowMinimumReserve);
    }
    ...
}
```

### Priority 2 — Security Hardening (Stratum Pool)

**POOL-001: Replace HashSet with bounded bloom filter**

```rust
use bloomfilter::Bloom;
// 1M capacity, 1% false-positive rate
let mut bloom = Bloom::new_for_fp_rate(1_000_000, 0.01);
// LRU-evict only on restart, never wipe during operation
```

**POOL-002: Synchronous difficulty floor**

```rust
if share.difficulty < self.config.min_difficulty {
    return Ok(Some(StratumResponse::error(msg.id, 23, "difficulty below minimum")));
}
// Only then return Ok(true)
```

**POOL-003: Dedup key exclude extranonce2**

```rust
let unique_id = format!("{}:{}", submission.job_id, submission.nonce);
// extranonce2 excluded — miner-controlled field cannot influence dedup
```

**POOL-004: Fix clean_jobs flag**

```rust
self.job_manager.create_job(template, true)?;  // true = clean_jobs
```

### Priority 3 — Architecture Refactoring

**ARCH-001 / ARCH-002: Split monolithic files**

Proposed structure for `main.rs` (24,609 lines → 5 focused modules):
```
crates/q-api-server/src/
  main.rs          (~100 lines — binary entry only)
  init.rs          — subsystem initialization
  config.rs        — CLI + env var parsing
  routes.rs        — Axum router setup
  tasks.rs         — background task spawning
```

Proposed structure for `handlers.rs` (17,019 lines → domain modules):
```
handlers/
  wallet.rs        — wallet create/import/sign
  blockchain.rs    — block/tx lookup
  mining.rs        — challenge/submit/stats
  network.rs       — peers/P2P/health
  analytics.rs     — supply/emission/metrics
  admin.rs         — admin operations
  bridges.rs       — bridge status stubs
  dex.rs           — swap/pool/oracle
  contracts.rs     — deploy/call
```

### Priority 4 — Operational

**BAL-001:** Add pre-flight check before height 18,600,000 that verifies `compute_balance_root_for_block()` succeeds; block mining submissions with a warning if it fails. Prevents silent chain halt at activation.

**VDF-001:** Add `Q_VDF_ITERATIONS_CAP` env var. Cap per-challenge iterations to `block_interval_ms * 0.8 / single_squaring_ms`. Prevents evaluation time exceeding block interval.

**POOL-005:** Change `Q_ENABLE_MINING_POOL` default from `"1"` to `"0"`. Bootstrap-only nodes should not run pool servers by default.

**S-002:** Add separate, lower rate limits for compute-intensive endpoints (AI inference: 1 req/5s; ZK proof: 1 req/10s; standard: 100 req/s).

---

*Document status: Iteration 2 — Deep audit complete. DEX trading/reserve integration and Stratum pool hardening are the highest-priority action items for the next development sprint.*
