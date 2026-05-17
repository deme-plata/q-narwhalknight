# Q-NarwhalKnight — Node & Miner Technical Review v1
**Date:** 2026-05-06  
**Version Reviewed:** 10.6.0  
**Reviewer:** Server Beta (Claude Code)  
**Branch:** feature/safe-batched-sync-v1.0.2  
**Status:** Iteration 1 — Full Feature Inventory

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Node (q-api-server) — Deep Feature Review](#3-node-q-api-server--deep-feature-review)
   - 3.1 Startup & Configuration
   - 3.2 Consensus Engine (DAG-Knight)
   - 3.3 Block Production
   - 3.4 Storage Layer
   - 3.5 P2P Networking & Turbo Sync
   - 3.6 REST API Surface (228+ endpoints)
   - 3.7 Real-Time Streaming (SSE/WebSocket)
   - 3.8 Balance & State Management
   - 3.9 Emission Controller
   - 3.10 Mining Pool Server (Stratum)
   - 3.11 AI Inference Engine
   - 3.12 Privacy & Tor Integration
4. [Miner (q-miner) — Deep Feature Review](#4-miner-q-miner--deep-feature-review)
   - 4.1 Mining Algorithm
   - 4.2 CPU Mining Engine
   - 4.3 GPU Mining Engine
   - 4.4 VDF Lane
   - 4.5 Pool vs Solo vs P2P Modes
   - 4.6 Communication with Node
   - 4.7 FPGA/RTL Implementation
5. [Cryptography Layer](#5-cryptography-layer)
   - 5.1 Classical Cryptography
   - 5.2 Post-Quantum Cryptography
   - 5.3 Zero-Knowledge Proof Systems
   - 5.4 Privacy Primitives
6. [DeFi & Smart Contracts](#6-defi--smart-contracts)
   - 6.1 DEX (q-dex)
   - 6.2 Smart Contract VM (q-vm)
   - 6.3 Transaction Type Taxonomy
   - 6.4 Token System
7. [Frontend Wallet](#7-frontend-wallet)
   - 7.1 Quantum Wallet (React)
   - 7.2 Slint Native Wallet
8. [Infrastructure & Deployment](#8-infrastructure--deployment)
   - 8.1 Multi-Server HA Topology
   - 8.2 Rolling Deployment Pipeline
9. [Feature Completeness Matrix](#9-feature-completeness-matrix)
10. [Issues & Improvement Areas — Iteration 1](#10-issues--improvement-areas--iteration-1)

---

## 1. Executive Summary

Q-NarwhalKnight (QNK) is a Rust-based modular blockchain node implementing a quantum-enhanced DAG-BFT consensus protocol (DAG-Knight). The system is architectured as an 89-crate Cargo workspace targeting 10+ BPS block production, 80K+ TPS throughput, and progressive post-quantum cryptographic hardening.

**Version 10.6.0** introduces BalanceRootV1 — balance state roots embedded in block headers, activated at mainnet height 18,600,000. This enables light-client balance verification without full chain replay.

### Key Statistics

| Metric | Value |
|--------|-------|
| Current version | 10.6.0 |
| Total crates | 89 |
| Main source file (main.rs) | ~23,900 lines |
| REST API endpoints | 228+ |
| Transaction types | 46 distinct opcodes |
| Mining algorithm | BLAKE3×100 + Genus-2 VDF |
| Max supply | 21,000,000 QUG |
| Decimal precision | 24 decimals |
| Consensus finality target | <50 ms |
| Block production target | 10–80 BPS |
| Minimum Rust version | 1.86 |
| License | Apache-2.0 |

### Production Infrastructure

| Server | Role | Bandwidth | Cores | RAM |
|--------|------|-----------|-------|-----|
| Beta (185.182.185.227) | Primary bootstrap | 100 Mbit | 24 | 64 GB |
| Gamma (109.205.176.60) | HA backup | 1 Gbit | 24 | 64 GB |
| Delta (5.79.79.158) | Canary | 1 Gbit | 24 | 64 GB |
| Epsilon (89.149.241.126) | 10 Gbit supernode | 10 Gbit | 48 | 64 GB |

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Q-NarwhalKnight Stack                           │
├────────────────┬─────────────────┬───────────────┬───────────────────┤
│   Web Wallet   │  Slint Wallet   │   q-miner CLI │  External Miners  │
│ (React/TS)     │ (Rust/Slint)    │ (CPU+GPU+VDF) │ (Stratum pool)    │
├────────────────┴─────────────────┴───────────────┴───────────────────┤
│                    REST API + SSE + WebSocket                         │
│              Axum 0.7  ·  228 routes  ·  rate-limited                │
├──────────────┬───────────────┬────────────────┬───────────────────────┤
│  Block       │  Balance      │  Emission      │  Smart Contracts      │
│  Producer    │  Consensus    │  Controller    │  (WASM, Wasmer 4.0)   │
├──────────────┴───────────────┴────────────────┴───────────────────────┤
│                  DAG-Knight Consensus Engine                          │
│      Bullshark-style BFT  ·  δ=1  ·  <50ms finality target           │
│      Homological fork detection  ·  Quantum VDF anchor election       │
├──────────────────────────────────────────────────────────────────────┤
│               libp2p 0.56 P2P Network (Tokio async)                  │
│  TCP · QUIC · WS · DCUTR  ·  Gossipsub  ·  Kademlia DHT             │
│  Turbo Sync (150–250 BPS)  ·  Dandelion++ Tor anonymity              │
├──────────────────────────────────────────────────────────────────────┤
│                    Storage Layer                                       │
│         RocksDB (Linux/macOS)  ·  Sled (Windows)                     │
│   Argon2+AES-GCM encryption  ·  LZ4/Zstd compression                │
│   Parallel state applicator  ·  LRU block cache                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Node (q-api-server) — Deep Feature Review

### 3.1 Startup & Configuration

**Entry point:** `crates/q-api-server/src/main.rs` (~23,900 lines)

#### Startup Sequence

1. **Memory allocator selection** — jemalloc (Unix) or mimalloc (Windows). Prevents gossipsub OOM at 60+ msg/sec.
2. **THP disable** — `prctl(41,1,0,0,0)` disables Transparent Huge Pages per-process to prevent jemalloc fragmentation under load.
3. **rustls CryptoProvider install** — Must be first, before any tokio worker spawns; avoids `CryptoProvider` panic in reqwest/tungstenite.
4. **.env file load** — Non-fatal; loads Stripe keys, Tor config, custom overrides.
5. **`validate_env_config()`** — Warns on invalid configuration values.
6. **CLI parsing (clap)** — 20+ flags including `--port`, `--network`, `--tui`, `--validator-key`, `--admin-wallet`, `--cheap-ssd`, `--setup`, `--experimental-fast-sync`, `--compute-mode`.
7. **Subsystem init** — Storage, balance consensus, block producer, P2P network, Tor, mining pool, AI inference, REST API.

#### Key Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `Q_NETWORK_ID` | Network identifier (overrides `--network`) | `testnet` |
| `Q_DB_PATH` | Database directory path | `./data` |
| `Q_WORKER_THREADS` | Tokio worker count | CPU count |
| `Q_DISABLE_AI` | Skip AI inference engine | `0` |
| `Q_DISABLE_DEX` | Disable DEX/swap | `0` |
| `Q_ENABLE_TOR` | Enable Tor anonymity layer | `0` |
| `Q_ENABLE_MINING_POOL` | Enable Stratum pool server | `0` |
| `Q_ROCKSDB_WRITE_RATE_MB` | RocksDB write limit (MB/s) | `200` |
| `ROCKSDB_BLOCK_CACHE_MB` | RocksDB block cache size | Auto (RAM/3) |
| `Q_TOR_BOOTSTRAP_TIMEOUT` | Tor startup timeout (s) | `120` |
| `Q_SKIP_CHECKPOINT` | Skip sync checkpoint validation | `0` |
| `RUST_LOG` | Log level (`warn`/`info`/`debug`) | `info` |

**Critical note on Epsilon:** `Q_DB_PATH` MUST be an absolute path (`/home/orobit/data-mainnet-genesis`). Relative path + `WorkingDirectory=/` resolves to the 40 GB root partition, filling it instantly.

---

### 3.2 Consensus Engine (DAG-Knight)

**Crate:** `crates/q-dag-knight/src/`

#### Algorithm

DAG-Knight is a Bullshark-family asynchronous BFT consensus protocol operating on a directed acyclic graph (DAG) of vertices. It achieves sub-50ms finality using δ=1 commit rule.

| Property | Value |
|----------|-------|
| Byzantine tolerance | f = ⌊(n-1)/3⌋ |
| Finality target | <50 ms |
| Commit rule | δ=1 (aggressive) |
| Anchor election | Quantum VDF entropy |
| Fork detection | Homological (Betti numbers H₀, H₁) |

#### Key Components

**QuantumVDF** (`quantum_vdf.rs`):
```
base_difficulty: 1024
quantum_enhancement: 70%
parallel_threads: 4
security_level: PostQuantum
```
Time-lock puzzle providing entropy for anchor election. QRNG seed refreshed every 30 seconds.

**QuantumAnchorElection** (`anchor_election.rs`):
Selects the "anchor" vertex in even rounds using VDF output + optional Ring-LWE L-VRF for quantum-enhanced randomness.

**HomologicalConsensus** (`homological_consensus.rs`):
Detects forks using algebraic topology. Betti numbers H₀ (connected components) and H₁ (cycles) track DAG topology to resolve ambiguous commit decisions.

**Genus-2 VDF** (`genus2_vdf_integration.rs`):
Uses hyperelliptic curve group law for time-lock puzzles resistant to quantum speedup (Shor's algorithm does not apply). Produces Wesolowski proofs for O(log T) verification. Enabled by default.

**Vertex structure:**
```
Vertex {
  id: [u8; 32],
  round: u64,
  author: NodeId,
  tx_root: TxHash,
  parents: Vec<VertexId>,
  transactions: Vec<Transaction>,
  signature: Vec<u8>,        // Ed25519 or Dilithium5
  timestamp: DateTime<Utc>,
}
```

**Block structure** (simplified):
```
Block {
  height: u64,
  hash: [u8; 32],          // BLAKE3(all fields)
  prev_hash: [u8; 32],
  timestamp: DateTime<Utc>,
  proposer: NodeId,
  transactions: Vec<Transaction>,
  dag_round: u64,
  dag_parents: Vec<VertexId>,
  balance_root: Option<[u8; 32]>,   // BalanceRootV1, active at height 18,600,000
  finality_cert: Option<BullsharkCert>,
}
```

---

### 3.3 Block Production

**File:** `crates/q-api-server/src/block_producer.rs`

#### Block Production Triggers

- **Time-based:** Every 15 seconds (configurable `block_interval_secs`)
- **Solution-based:** Minimum 1 queued mining solution
- **Mempool-based:** Pending user transactions (fee-ordered)

#### Production Flow

1. Drain lock-free `SegQueue<MiningSolution>` (max 250 per block, v7.2.6 regression fix)
2. Create coinbase transactions (emission rewards from `BalanceConsensusEngine`)
3. Pull user transactions from `ProductionMempool` (highest fee first)
4. Compute Merkle root — SIMD-accelerated (AVX-512, 8–10× speedup) with scalar fallback
5. Sign with validator keypair if present (Dilithium5 or Ed25519+Dilithium5)
6. Enforce BalanceRootV1 invariant at height ≥ 18,600,000
7. Publish to gossipsub topic `/qnk/{network}/blocks` (MessagePack, u128 as STRING)
8. Emit SSE events (NewBlock, BalanceUpdated, MiningReward)

#### Performance Characteristics

| Metric | Target |
|--------|--------|
| Block production rate | 10–80 BPS |
| Coinbase transactions/block | Up to 250 |
| Solution queue latency | ~100 ms (lock-free) |
| Block broadcast latency | ~200–500 ms (gossipsub) |
| End-to-end finality | ~1–2 s |

---

### 3.4 Storage Layer

**Crate:** `crates/q-storage/src/`

#### Backend Selection

| Platform | Backend | Notes |
|----------|---------|-------|
| Linux / macOS | RocksDB 0.22 | Multi-threaded, column families |
| Windows | Sled 0.34 | Pure Rust, cross-platform |

#### RocksDB Column Families

| CF | Content |
|----|---------|
| `CF_BLOCKS` | Serialized blocks indexed by height |
| `CF_BALANCES` | Wallet balances (u128 per address) |
| `CF_STATE` | Smart contract state (k/v pairs) |
| `CF_TRANSACTIONS` | Transactions indexed by TxHash |
| `CF_VERTICES` | DAG vertices by VertexId |
| `CF_CERTIFICATES` | Bullshark certificates |
| `CF_CONTRACTS` | Deployed contracts + metadata |
| `CF_MANIFEST` | Node metadata (chain tip, genesis, upgrade flags) |
| `CF_PEER_TRUST` | Peer reputation scores |
| `CF_MINING_STATS` | Per-miner block counts & hashrate |
| `CF_EMAIL` | Blockchain email inbox |
| `CF_CALENDAR` | Decentralized calendar events |

#### Storage Features

- **Encryption-at-rest:** Argon2 key derivation + AES-GCM authenticated encryption
- **Compression:** LZ4 (fast decompression) and Zstd (variable ratio)
- **Warp Sync:** Batch signature verification (25–50× faster than serial)
- **Parallel State Applicator:** Rayon-based multi-threaded state updates
- **Fork Detection:** Homological fork analysis
- **Async Pipeline:** Deferred batch writes for throughput
- **Memory-mapped files:** memmap2 for zero-copy block access

#### Serialization Formats

| Format | Usage | Notes |
|--------|-------|-------|
| Bincode | Storage backend | Native u128 support |
| MessagePack (rmp-serde) | P2P gossipsub | u128 serialized as STRING (critical fix v3.2.7) |
| Postcard | Transactions | Compact binary |
| JSON | REST API responses | Human-readable |

#### Known Issues & Fixes

- **`save_safe_floor()` must use `put_sync()`** (fsync) to survive kill-9 restarts (v7.4.1)
- **Fast recovery threshold lowered** from `>10,000` to `>1,000` blocks (v7.4.1)
- **`StorageEngine::sync_wal()`** was a no-op; fixed to call `hot_db.sync_wal()` (v7.4.1)
- **Cargo incremental build stale constants** — always `cargo clean --package <crate>` after constant changes

---

### 3.5 P2P Networking & Turbo Sync

**Crate:** `crates/q-network/src/`  
**Library:** libp2p 0.56

#### Transports

| Transport | Multiaddr pattern | Use |
|-----------|------------------|-----|
| TCP | `/ip4/.../tcp/9001` | Standard |
| QUIC | `/ip4/.../udp/9001/quic-v1` | NAT traversal |
| WebSocket | `/ip4/.../tcp/9001/ws` | Browser compatibility |
| Relay | Full relay path | DCUTR hole-punching |

#### Gossipsub Topics

| Topic | Purpose |
|-------|---------|
| `/qnk/{net}/blocks` | Block announcements |
| `/qnk/{net}/mempool-txs` | Transaction relay |
| `/qnk/{net}/peer-heights` | Height synchronization |
| `/qnk/{net}/turbo-sync-{shard}` | Batched block sync |
| `/qnk/{net}/mining-solutions` | Mining solution relay |
| `/qnk/{net}/ai-inference-{model}` | Distributed AI tensor forwarding |

**Gossipsub parameters:** mesh d=8, d_low=6, d_high=12, heartbeat=1s, history=12 heartbeats.

#### Turbo Sync

Batch block synchronization achieving 150–250 BPS (25–50× faster than serial).

| Config | Default |
|--------|---------|
| `Q_TURBO_PARALLEL_STREAMS` | 16 parallel streams per peer |
| `Q_TURBO_CHUNK_SIZE` | 2,500 blocks/chunk |
| `Q_TURBO_COMPRESSION_LEVEL` | 1 (fast Zstd) |
| `Q_TURBO_CHUNK_TIMEOUT_SECS` | 30 s |

**Expected sync performance on Epsilon (10 Gbit, 48 cores):**
- 0–500K blocks: ~280 blocks/sec (warmup)
- 500K–3M blocks: ~1,100 blocks/sec (peak)
- Full sync (~11.4M blocks): ~5.5 hours

#### Critical Safety Rules

- **Sync-down prevention:** Application layer enforces `network_height > current_height + 5` before triggering sync. Database layer adds a hard abort if `target_height < local_height && local_height > 1,000`.
- **Block-pack semaphore:** Max 4 concurrent block-pack responses (prevents OOM under parallel sync load — v9.1.9 fix).
- **gossipsub flood_publish OOM guard:** Semaphore prevents unbounded memory growth under message storms.

---

### 3.6 REST API Surface (228+ endpoints)

**Framework:** Axum 0.7  
**Auth:** AEGIS-QL post-quantum middleware, OAuth2, API keys, wallet signature  
**Rate limiting:** governor 0.6 (token bucket with burst)

#### API Modules Summary

| Module | Endpoint prefix | Key operations |
|--------|----------------|----------------|
| Status | `/api/v1/status`, `/api/v1/node/status` | Node health, height, peers |
| Blockchain | `/api/v1/blocks/`, `/api/v1/transactions/` | Block/tx lookup |
| Wallets | `/api/v1/wallets/`, `/api/v1/wallet/` | Create, import, balance, sign |
| Mining | `/api/v1/mining/` | Challenge, submit, stats, diagnostics |
| Transactions | `/api/v1/transactions/send`, `/api/v1/estimate-fee` | Send, fee estimate |
| Consensus | `/api/v1/validators/`, `/api/v1/consensus/` | Validator list, resonance metrics |
| Emission | `/api/v1/totalsupply`, `/api/v1/circulatingsupply`, `/api/v1/emission/` | Supply data |
| DEX | `/api/v1/defi/`, `/api/v1/dex/` | Swaps, pools, oracle prices, volume |
| Smart Contracts | `/api/v1/contracts/` | Deploy, query, call |
| Banking | `/api/v1/quillon/`, `/api/v1/stablecoin/` | CDP, bank balance, QUGUSD |
| Privacy | `/api/v1/mixer/`, `/api/v1/quantum-mixer/` | Mixing pool join/status |
| Bridges | `/api/v1/bitcoin/`, `/api/v1/ethereum/`, `/api/v1/zcash/`, `/api/v1/monero/` | Cross-chain |
| AI | `/api/v1/chat`, `/api/v1/ai/` | LLM chat, completion, translation |
| Email | `/api/v1/email/` | Send, inbox |
| Calendar | `/api/v1/calendar/` | Create events, fetch |
| SSE | `/api/v1/events` | Real-time event stream |
| Admin | `/api/v1/admin/` | Nginx/flux stats, bridge freeze |

**Authentication note:** All balance/wallet endpoints require session auth. Do NOT attempt to curl balance endpoints without valid auth tokens — they return empty.

---

### 3.7 Real-Time Streaming (SSE/WebSocket)

**File:** `crates/q-api-server/src/streaming.rs`

```
GET /api/v1/events?wallet_address=...&headers_only=true&miner_mode=true
WS  /api/v1/ws?wallet_address=...&headers_only=true
```

#### Event Types (selected)

| Event | Trigger |
|-------|---------|
| `NewBlock` | Block produced and committed |
| `BalanceUpdated` | Wallet balance changed |
| `MiningReward` | Mining solution accepted |
| `MiningStats` | Periodic hashrate / difficulty update |
| `DexSwapExecuted` | Swap completed |
| `SmartContractEvent` | Contract log emitted |
| `ValidatorActivation` | New validator registered |
| `NetworkHealthUpdate` | Peer count / sync quality change |
| `TokenPriceUpdate` | Oracle price feed update |

#### Performance

- **Event latency target:** <50 ms from production to client
- **Throughput:** 10K+ events/sec per node
- **Miner mode:** Reduces bandwidth from ~111 KB/s to 2–5 KB/s (mining-only events)
- **Implementation:** Tokio `broadcast::Sender` (1,000-event buffer) + lock-free `crossbeam::SegQueue`

---

### 3.8 Balance & State Management

**In-memory store:** `RwLock<HashMap<[u8; 32], u128>>` in `AppState`

**Update flow:**
1. Validate `from_balance ≥ amount + fee`
2. On block finalization, apply all transactions atomically
3. If height ≥ 18,600,000: compute and embed `BalanceRoot` in block header

**BalanceRootV1 (v10.6.0):**

| Property | Value |
|----------|-------|
| Activation height | 18,600,000 |
| Algorithm | SHA3-256 Merkle tree over (address, balance) pairs |
| Purpose | Light-client balance verification |
| Computation | `BalanceConsensusEngine::compute_balance_root_for_block()` |

---

### 3.9 Emission Controller

**File:** `crates/q-storage/src/emission_controller.rs`

| Parameter | Value |
|-----------|-------|
| Max supply | 21,000,000 QUG |
| Decimal precision | 24 |
| Genesis timestamp | 1771761600 (2026-02-22 12:00 UTC) |
| Era 0 annual emission | 2,625,000 QUG |
| Halving interval | 4 years (126,230,400 s) |
| Total eras | 64 |
| Total emission duration | ~256 years |
| Max reward per block | 2 QUG |

**Adaptive reward formula:**
```
R(t) = annual_target(era) / (λ × seconds_per_year)
```
where λ = measured block rate. This keeps annual emission constant regardless of block rate.

**Budget-based error correction:** Negative-feedback correction factor f ∈ [0.01, 3.0] corrects for cumulative over/under-emission. Guarantees convergence to 21M total supply.

---

### 3.10 Mining Pool Server (Stratum)

**Crate:** `crates/q-mining-pool/`  
**Version:** 2.2.1-beta  
**Port:** `Q_STRATUM_PORT` (default 3333)  
**Protocol:** Stratum v1 (JSON-RPC over TCP)  
**Payout:** PPLNS (Pay-Per-Last-N-Shares)

Currently enabled unconditionally at startup (cannot be disabled via env var).

---

### 3.11 AI Inference Engine

**Crate:** `crates/q-ai-inference/`  
**Controlled by:** `Q_DISABLE_AI=1` to skip

| Option | Details |
|--------|---------|
| Models | Mistral-7B, LLaMA-2, BitNet (configurable via `Q_AI_MODEL`) |
| Backend | mistral.rs (llama.cpp / Candle) |
| GPU | CUDA (`--features cuda`), Metal (macOS, `--features metal`) |
| API endpoints | `/api/v1/chat`, `/api/v1/ai/complete`, `/api/v1/ai/translate` |
| Token economy | `AICreditPurchase / AICreditSpend` transaction types (0x50–0x55) |

---

### 3.12 Privacy & Tor Integration

**Crates:** `crates/q-tor-client/`, `crates/q-tor-circuit/`, `crates/q-dandelion/`

| Feature | Implementation |
|---------|----------------|
| Embedded Tor client | Arti 0.9 (no system Tor required) |
| Circuit management | Dedicated circuit pool, 4 circuits per validator |
| Transaction anonymity | Dandelion++ (stem→fluff, ~10-minute anonymity set) |
| Routing | Onion network exit for gossipsub |
| Startup timeout | `Q_TOR_BOOTSTRAP_TIMEOUT=5` recommended (prevents 120 s startup block) |

---

## 4. Miner (q-miner) — Deep Feature Review

**Crate:** `crates/q-miner/src/main.rs` (4,778 lines)  
**Binary version:** 10.5.3  
**ELF:** x86-64, stripped, GNU/Linux 3.2.0+

### 4.1 Mining Algorithm

**Below GENUS2_VDF_MINING activation height — Legacy mode:**
- 100-round iterated BLAKE3 (1 initial hash + 99 additional rounds)
- Nonce: 8 bytes, partitioned by thread ID

**Above activation height — Genus-2 VDF mode:**
- Genus-2 Jacobian curve sequential squaring (non-parallelizable by design)
- Generates Wesolowski proofs (O(log T) verification)
- VDF iterations specified per-block in the challenge
- Target check: lexicographic comparison of 32-byte BLAKE3 output against difficulty target

---

### 4.2 CPU Mining Engine

**File:** `crates/q-miner/src/cpu/mod.rs` (650+ lines)

| Feature | Details |
|---------|---------|
| Threading | `tokio::task::spawn_blocking` per thread, pinned via `core_affinity` |
| Nonce partition | `thread_id << 48` (prevents cross-thread collision) |
| Batch size | `intensity × 100,000` nonces/batch (intensity 1–10, default 7) |
| SIMD detection | AVX2, AVX-512, NEON (automatic selection) |
| Allocator | jemalloc (Linux), mimalloc (Windows) |
| Hash counter flush | Every 1,024 hashes (reduces atomic contention) |
| Stale detection | `new_block_signal` atomic checked every 512 nonces |

---

### 4.3 GPU Mining Engine

**File:** `crates/q-mining/src/gpu.rs` (1,000+ lines)

| Feature | Details |
|---------|---------|
| Framework | OpenCL 3.0 (primary), CUDA via `cudarc` (feature: `cuda-mining`) |
| Algorithm on GPU | BLAKE3 × 100 rounds (same as CPU) |
| Kernel optimization | `#pragma unroll 9`, `__constant` challenge cache, persistent buffers |
| Dispatch | Adaptive [100 ms, 400 ms] per kernel execution |
| Multi-GPU | Independent per-GPU auto-tuning (v10.1.7+) |
| Kernel cache | `~/.config/q-miner/kernel-cache/` |
| GPU nonce space | Starts at `u64::MAX / 2` (avoids CPU collision) |
| Stale detection | `gpu_new_block_signal` atomic checked between dispatches |

---

### 4.4 VDF Lane

**File:** `crates/q-miner/src/vdf_lane.rs` (247 lines)

A single dedicated thread running Genus-2 Jacobian squaring — fundamentally sequential (this is the VDF guarantee). Checks `new_block_signal` before each evaluation. Produces Wesolowski proofs submitted via the solution channel.

In hybrid CPU+GPU mode, the VDF lane occupies 1 core while CPU threads and GPU operate in parallel.

---

### 4.5 Pool vs Solo vs P2P Modes

| Mode | Protocol | Reward destination | Notes |
|------|----------|--------------------|-------|
| Solo | HTTP POST to `/api/v1/mining/submit` | Wallet address directly | Best for large miners |
| Pool | Stratum v1 JSON-RPC over TCP | PPLNS pool payout | Default: `stratum+tcp://quillon.xyz:3333` |
| Decentralized P2P | libp2p gossipsub | CRDT-based PPLNS | `--force-p2p` flag; P2P challenge relay <50ms |

**Failover:** Up to 3 attempts with 2 s + 5 s backoff. Supports comma-separated `--server` list.  
**Deduplication:** `HashSet<[u8; 32]>` in centralized solution submitter prevents double-submission.

---

### 4.6 Communication with Node

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/mining/challenge` | GET | Fetch current challenge |
| `/api/v1/mining/submit` | POST | Submit solution |
| `/api/v1/status` | GET | Node health check |
| `/api/v1/miner/device-login` | POST | MinerLink device auth |
| `/api/v1/events` (SSE) | GET | Real-time block notifications |

**Challenge fields:** `challenge_hash`, `difficulty_target`, `block_height`, `vdf_iterations`, `block_reward`, `expires_at`

---

### 4.7 FPGA/RTL Implementation

**Path:** `qug-v1-rtl/rtl/`  
**Target:** Xilinx Kintex-7 XC7K325T at 100 MHz  
**Design:** 16-core RISC-V (RV32IMC) SoC with custom ISA extensions

#### Custom ISA Extensions

**Xcrypto (custom-0, opcode 0x0B)** — Hardware BLAKE3:
- `blake3.init`, `blake3.round`, `blake3.chain`, `blake3.finalize`, `blake3.ldmsg`, `blake3.status`
- Implemented in `rtl/xcrypto/` (SystemVerilog)
- 2-stage pipelined round unit (`blake3_round.sv`)

**Xlattice (custom-1, opcode 0x2B)** — Post-quantum NTT/polynomial ops:
- `ntt.fwd`, `ntt.inv`, `poly.add`, `poly.mul`, `poly.reduce`
- **Status: Planned — not yet implemented in RTL**

#### Performance Targets

| Metric | Target |
|--------|--------|
| Clock frequency | 100 MHz (FPGA prototype) |
| BLAKE3 throughput | 1 hash / 14 cycles |
| Mining hashrate (16 cores) | ~114 MH/s (FPGA) |
| Power (FPGA estimate) | ~12 W |
| ASIC tapeout target | TSMC 7nm |

---

## 5. Cryptography Layer

### 5.1 Classical Cryptography

| Algorithm | Use | Location |
|-----------|-----|---------|
| Ed25519 | Transaction signing (Phase 0), address derivation | `q-wallet` |
| SHA3-256 | Block hash, transaction hash, address hash | `q-types` |
| BLAKE3 | Mining PoW, Merkle trees | `q-miner`, `q-crypto-simd` |
| AES-256-GCM | Wallet encryption, storage encryption | `q-wallet`, `q-storage` |
| Argon2id | Key derivation (64 MB memory cost) | `q-wallet`, `q-storage` |
| BIP39 | 12-word mnemonic generation | `q-wallet` |

### 5.2 Post-Quantum Cryptography

| Algorithm | Standard | Signature size | Use |
|-----------|----------|---------------|-----|
| Dilithium5 | ML-DSA (NIST) | ~3,300 bytes | Block signing (Q1/Q2) |
| Kyber1024 | ML-KEM (NIST) | — | Wallet encryption KEM |
| SQIsign | Isogeny-based (NIST candidate) | 204 bytes | Compact PQ signatures (Phase Advanced) |
| SPHINCS+ | Hash-based (NIST) | 7,856 bytes | Alternative stateless signing |
| Genus-2 VDF | Hyperelliptic curves | — | Time-lock, mining puzzle |
| FROST | Threshold Schnorr (IACR 2025/1024) | — | t-of-n validator committee |
| Ring-LWE L-VRF | Lattice VRF | — | Anchor election |

**Deployment phases:**

| Phase | Signing | Notes |
|-------|---------|-------|
| Q0 (Classical) | Ed25519 only | Legacy wallets |
| Q1 (Hybrid) | Ed25519 + Dilithium5 | Defense-in-depth |
| Q2 (PQ) | Dilithium5 only | Full quantum resistance |

### 5.3 Zero-Knowledge Proof Systems

| System | Proof size | Trusted setup | Quantum-safe | Use |
|--------|-----------|--------------|-------------|-----|
| Circle STARKs (IACR 2024/278) | ~60 KB | None | Yes | Private transactions |
| Groth16/PLONK SNARKs | 96–192 bytes | Required | No | Smart contract proofs |
| Bulletproofs v2 (IACR 2024/313) | O(log n) | None | No | Range proofs (confidential amounts) |
| Recursive SNARKs | Succinct | None | No | Light client bootstrap (~10 ms) |

### 5.4 Privacy Primitives

| Primitive | Implementation |
|-----------|---------------|
| Dandelion++ | Stem-phase random routing → fluff-phase broadcast |
| Ring signatures | `RingTransfer (0x82)` transaction type |
| Stealth addresses | `StealthCreate (0x81)` transaction type |
| Shielded transfers | Circle STARK proofs + nullifier set |
| Tor routing | Arti embedded client, dedicated circuit pool |
| AEGIS-256 | Authenticated encryption (2–5× faster than AES-GCM) |

**Privacy tiers in DEX:** Basic (0), Enhanced (1, Tor), Maximum (2, multi-hop + ZK), Quantum (3, PQ + maximum routing).

---

## 6. DeFi & Smart Contracts

### 6.1 DEX (q-dex)

**AMM model:** Quantum-enhanced constant product (x × y = k)  
**Fee structure:** 0.30% total (0.05% protocol + 0.25% LP)  
**Slippage protection:** 0.50% maximum (50 bps)

**Quantum DEX parameters (physics-inspired):**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Uncertainty factor | 0.1618 (φ-scaled) | Volatility modeling |
| Decoherence time | 300 seconds | Quantum state lifetime |
| Max leverage | 10.0× | Derivative positions |
| Liquidation threshold | 80% (8,000 bps) | CDP safety |
| Wave collapse threshold | 5% | Price movement trigger |
| Entanglement strength | 0.707 (√2/2) | Cross-pool correlation |

**Yield farming (v8.6.0):**
- Multiplier: 1.1× (110%)
- Impermanent loss protection: 60%
- Slippage reduction: 3.50%

**DexScreener integration** for external price tracking.

### 6.2 Smart Contract VM (q-vm)

**Runtime:** Wasmer 4.0.0 (JIT) + Wasmtime 14.0.0 (hot/cold)  
**Language:** WebAssembly (WASM)  
**Features:** Gas metering, cross-contract calls, parallel execution, post-quantum deployment signatures (Dilithium5)

**Contract categories:**

| Category | Examples |
|----------|---------|
| Token standards | SecureToken, AdvancedToken, RwaToken, OrbusdStablecoin |
| DeFi infrastructure | MultisigWallet, Governance, PrivateDex, TimelockVault |
| Advanced DeFi | LendingPool, YieldFarming, StakingContract, InsuranceProtocol |
| Real World Assets | RealEstateToken, CarbonCreditToken, ArtCollectibleToken, EquityToken |
| Derivatives | OptionsContract, PredictionMarket, SyntheticAssets |
| Utility | NftMarketplace, IdentityContract, BridgeContract |

### 6.3 Transaction Type Taxonomy

46 distinct transaction opcodes across 11 functional domains:

| Range | Domain |
|-------|--------|
| 0x00–0x0F | Core (Transfer, Coinbase, Burn, Fee) |
| 0x10–0x1F | Custom tokens (Create, Mint, Transfer, Burn, Freeze) |
| 0x20–0x2F | DEX (Pool, Swap, LimitOrder, FlashLoan) |
| 0x30–0x3F | Smart contracts (Deploy, Call, Upgrade, Destroy) |
| 0x40–0x4F | Stablecoin/Vault (VaultLock, StableMint, Liquidate, Oracle) |
| 0x50–0x5F | AI compute credits (Purchase, Spend, Transfer, ProviderRegister) |
| 0x60–0x6F | Governance (Proposal, Vote, Execute, Delegate) |
| 0x70–0x7F | Staking (Stake, Unstake, ClaimRewards, Slash) |
| 0x80–0x8F | Privacy (Dandelion++, Stealth, Ring, Shielded) |
| 0x90–0x9F | Cross-chain atomic swaps (Initiate, Claim, Refund) |
| 0xF0–0xFF | System (EmergencyPause, StateCheckpoint, Genesis) |

### 6.4 Token System

| Token | Decimals | Max supply | Type |
|-------|---------|-----------|------|
| QUG | 24 | 21,000,000 | Native mining token |
| QUGUSD | 24 | Unlimited (collateralized) | Algorithmic stablecoin |
| QCREDIT | 24 | Unlimited (1:1 QUG lock) | Yield vault token |
| QUSD | 24 | Founder-controlled | Issuer stablecoin |
| wBTC | 8 | 1:1 BTC collateral | Bridge token |
| wETH | 18 | 1:1 ETH collateral | Bridge token |
| wZEC, wIRON | 8 | Bridge collateral | Privacy bridge tokens |
| VAULT, FORGE | 0 | Physical device 1:1 | RWA tokens |

**Fee system:**
- Base gas: 21,000 units
- Standard min fee: 0.00021 QUG (reduced to 0.000021 QUG at `REDUCED_FEES_V1` height)

---

## 7. Frontend Wallet

### 7.1 Quantum Wallet (React/TypeScript)

**Path:** `gui/quantum-wallet/src/App.tsx`  
**Framework:** React with TypeScript  
**Build output:** `gui/quantum-wallet/dist-final/`

#### Navigable Screens (14+)

| Screen | Component | Key Features |
|--------|-----------|-------------|
| Dashboard | `Dashboard.tsx` | Live balance, QCI, block height, mining rewards |
| Transactions | `TransactionScreenV2` | Send/receive QUG and tokens |
| Explorer | `ExplorerScreen.tsx` | Block browser, P2P data fetching, DAG 3D popup |
| DEX | `DexScreen` | Token swaps, pool management |
| Mining | `MiningScreen.tsx` | Solo + pool mining, Stratum stats, downloads |
| Email | — | Blockchain email inbox/compose |
| Bank | — | CDP, lending, QUGUSD |
| AI Chat | — | LLM assistant via WebSocket |
| RWA Marketplace | — | Real-world asset tokenization |
| Analytics | — | Network and wallet analytics |
| Map | — | Geographic node distribution |
| POS Mode | — | Point-of-sale payments (`/pos` route) |
| Download Node | — | Node binary distribution |
| Settings | — | Preferences, logout |

#### Key Frontend Features

- **Multi-server failover:** Automatic detection + return-to-primary monitoring (60 s interval)
- **SSE real-time sync:** Instant balance updates on receive, 50 ms debounce on send
- **Balance sanity checks:** Enforces 21M QUG max supply cap; rejects corrupted SSE data
- **Block-height monotonic tracking:** Prevents balance "zigzag" from out-of-order events
- **DEX cooldown tracking:** Prevents stale balance overwrites after swap
- **Quantum Coherence Index (QCI):** Composite health score (peers 30% + block production 30% + health 40%)
- **P2P-first data fetching:** Blockchain data from peers before API fallback
- **Performance mode:** Disables QuantumBackground for lower-end devices
- **Network change detection:** Auto-clears localStorage cache on network switch
- **MinerLink:** WebSocket connection to personal hardware miners

#### Login Screen Particle Physics

The login background renders a string-theoretic resonance visualization:
```
ψ(x,t) = A · e^(i(kx - ωt + φ)) · sin(nπx/L)
```
Particle properties driven by live `/api/v1/health` data — stake weight → size, consensus priority → oscillation frequency, phase → color (0–2π mapped to spectrum). Network pulses emit on block height changes.

### 7.2 Slint Native Wallet

**Path:** `gui/slint-wallet/`  
**Framework:** Slint v1.9 (native desktop rendering, Rust backend)

| Feature | Details |
|---------|---------|
| Wallet management | Create (BIP39), restore (mnemonic), QR code generation |
| Balance tracking | SSE-based `EventSource` |
| OAuth2 | Local server on localhost:8000, PKCE flow |
| Mining | CPU mining + OpenCL GPU (feature: `opencl`) |
| Auto-updater | Self-replace binary v1.5 |
| POS mode | Merchant payment interface |
| Renderer | GPU with software fallback |

---

## 8. Infrastructure & Deployment

### 8.1 Multi-Server HA Topology

```
                    ┌─────────────────────┐
                    │  quillon.xyz (Nginx) │
                    │  ip_hash sticky SSL  │
                    └────────┬────────────┘
                             │
               ┌─────────────┴────────────┐
               ▼                          ▼
    ┌──────────────────┐       ┌──────────────────┐
    │  Server Beta     │       │  Server Gamma    │
    │ 185.182.185.227  │       │ 109.205.176.60   │
    │  Primary w=10    │       │  Backup w=1      │
    └──────────────────┘       └──────────────────┘
               │     P2P Gossipsub     │
               └───────────────────────┘
                         ▲
              ┌──────────┘
    ┌──────────────────┐       ┌──────────────────┐
    │  Server Delta    │       │ Server Epsilon   │
    │  5.79.79.158     │       │ 89.149.241.126   │
    │  Canary          │       │ 10 Gbit Supernode│
    └──────────────────┘       └──────────────────┘
```

**Nginx configuration:** `ip_hash` sticky sessions prevent balance flickering between servers.

### 8.2 Rolling Deployment Pipeline

**Script:** `scripts/ha-deploy.sh`

| Step | Action | User impact |
|------|--------|-------------|
| 1. `verify-delta` | SCP binary to Delta, 7+ min soak | None — canary node |
| 2. `verify-gamma` | SCP binary to Gamma, 90 s stability check | None — backup |
| 3. `promote` | Nginx: Gamma w=10, Beta w=1 | None — Gamma serves |
| 4. `deploy-beta` | Stop Beta, replace binary, restart | None — Gamma primary |
| 5. `restore` | Nginx: Beta w=10, Gamma w=1 | None — traffic shifts back |

**Safeguards:**
- Lock file prevents concurrent deployments
- Max 10 minutes per server for health
- Min 90 second stability soak
- Auto-rollback on health failure
- Version bump enforcement (deploy aborts if version unchanged)

**Mandatory before deploying:**
1. Run critical test suite (125+ mainnet-safety tests)
2. Bump workspace version in `Cargo.toml`
3. `cargo build --release --package q-api-server`
4. `./scripts/ha-deploy.sh full -y`

---

## 9. Feature Completeness Matrix

| Feature Area | Status | Notes |
|-------------|--------|-------|
| DAG-Knight consensus | ✅ Production | Genus-2 VDF, Bullshark BFT, homological fork detection |
| Block production | ✅ Production | 15s default interval, SIMD Merkle, 250 coinbase/block |
| RocksDB storage | ✅ Production | Multi-CF, encryption, LRU cache, Warp Sync |
| libp2p P2P | ✅ Production | TCP/QUIC/WS, gossipsub, Kademlia DHT, DCUTR |
| Turbo Sync | ✅ Production | 150–250 BPS, 16 parallel streams |
| REST API | ✅ Production | 228+ endpoints, rate limiting, CORS |
| SSE/WebSocket streaming | ✅ Production | <50ms latency, 20+ event types |
| BalanceRootV1 | ✅ Production | Activates at height 18,600,000 |
| Emission controller | ✅ Production | 64 eras, 21M QUG, adaptive formula |
| Ed25519 signing | ✅ Production | Phase 0 classical |
| Dilithium5 signing | ✅ Production | Phase 1/2 post-quantum |
| Kyber1024 KEM | ✅ Production | Wallet encryption |
| SQIsign | ✅ Integrated | Feature: `advanced-crypto` |
| FROST threshold sigs | ✅ Integrated | Feature: `advanced-crypto` |
| Circle STARKs | ✅ Integrated | Private transactions |
| Bulletproofs v2 | ✅ Integrated | Range proofs |
| Dandelion++ | ✅ Production | Mandatory for transaction relay |
| Tor (Arti embedded) | ✅ Integrated | `Q_ENABLE_TOR=1` |
| Mining pool (Stratum) | ✅ Production | PPLNS, v2.2.1-beta |
| CPU miner | ✅ Production | Multi-threaded, core affinity, AVX2/512 |
| GPU miner (OpenCL) | ✅ Production | Persistent buffers, adaptive dispatch |
| CUDA mining | 🟡 Partial | Feature flag available, not primary |
| Genus-2 VDF mining | ✅ Production | Above activation height |
| FPGA RTL (Xcrypto) | 🟡 Prototype | Kintex-7 target, BLAKE3 pipeline |
| FPGA RTL (Xlattice) | ⚠️ Planned | NTT/polynomial ops not implemented |
| Quantum Wallet (React) | ✅ Production | 14+ screens, SSE sync, DEX, mining |
| Slint native wallet | ✅ Integrated | GPU mining, auto-updater |
| DEX (constant product) | ✅ Production | 0.30% fees, slippage protection |
| WASM smart contracts | ✅ Integrated | Wasmer 4.0, 25+ contract types |
| AI inference (Mistral) | ✅ Integrated | `Q_DISABLE_AI=0`, CUDA/Metal |
| Bitcoin bridge | 🟡 Partial | Endpoint stubs, HTLC design |
| Ethereum bridge | 🟡 Partial | Endpoint stubs |
| Zcash/Monero bridges | 🟡 Partial | Privacy-bridge endpoints stubs |
| Sharding | ⚠️ Disabled | `ShardingEngine` exists, disabled in production |
| Recursive SNARKs | ✅ Integrated | `Q_ENABLE_RECURSIVE_PROOFS=1` |
| Resonance protocol | 🟡 Optional | Feature: `resonance`, alternative consensus |
| K-gauge (ConsensusGuard) | ✅ Production | Mainnet height-monotonicity enforcement |
| Quillon Bank (CDP) | ✅ Integrated | QUGUSD CDP, collateral ratio |
| Stablecoin (QUGUSD) | ✅ Production | 24-decimal, oracle-backed |
| QCREDIT yield vault | ✅ Integrated | v8.5.5 |
| HA rolling deploy | ✅ Production | Zero-downtime 4-server pipeline |

---

## 10. Issues & Improvement Areas — Iteration 1

This section identifies areas for review in subsequent iterations. Severity: 🔴 Critical / 🟠 High / 🟡 Medium / 🟢 Low.

### 10.1 Active Known Issues

| ID | Severity | Area | Description |
|----|----------|------|-------------|
| I-001 | 🔴 | Stratum pool | Mining pool always enabled at startup; `Q_ENABLE_MINING_POOL` does not disable it. Consumes port 3333 even on nodes not intended as pool servers. |
| I-002 | 🟠 | Xlattice RTL | FPGA Xlattice extension (NTT/polynomial ops for post-quantum) is architecturally defined but not implemented in RTL. Blocks FPGA PQ mining validation. |
| I-003 | 🟠 | CUDA GPU mining | CUDA is feature-flagged but not the primary GPU path; OpenCL is preferred. May limit NVIDIA GPU mining performance relative to OpenCL. |
| I-004 | 🟡 | Cross-chain bridges | Bitcoin, Ethereum, Zcash, Monero bridge endpoints exist but are stubs. No live on-chain bridge logic. Users may see endpoints but no functional swaps. |
| I-005 | 🟡 | Sharding disabled | `ShardingEngine` crate exists with adaptive strategies but is disabled in all production builds. High-throughput scaling path blocked. |
| I-006 | 🟡 | Hierarchical cache | Phase 2 `q-cache` crate disabled by default. RocksDB LRU cache is the only caching layer in production. |
| I-007 | 🟢 | MEMORY.md line limit | Memory index at 247 lines (limit 200). Lower-priority entries truncated from context window. Index needs pruning. |
| I-008 | 🟢 | `gui/slint-wallet` Vulkan | Vulkan GPU support listed in `Cargo.toml` features but marked "planned" in miner. |

### 10.2 Architectural Observations

| ID | Area | Observation |
|----|------|-------------|
| A-001 | main.rs size | At ~23,900 lines, `crates/q-api-server/src/main.rs` is a significant maintenance burden. Consider extracting subsystem initialization into dedicated modules. |
| A-002 | handlers.rs size | At 749 KB+, `handlers.rs` is very large. Splitting into feature-specific handler files would improve navigability and compile times. |
| A-003 | Emission on Epsilon | If Epsilon ever re-syncs from genesis via P2P, emission controller state becomes incorrect (balance watermarks computed locally, not replicated). The authoritative DB must always be `/home/orobit/data-mainnet-genesis/`. |
| A-004 | Windows Sled storage | Windows uses Sled 0.34 instead of RocksDB. Sled lacks multi-column-family support and SIMD compression. Windows node performance will differ significantly from Linux. |
| A-005 | u128 serialization boundary | u128 must be serialized as STRING in MessagePack (P2P) but uses native encoding in Bincode (storage). This boundary is a recurring source of subtle bugs when adding new message types. |
| A-006 | VDF lane bottleneck | Single VDF lane thread is a sequential bottleneck. At high block rates, VDF proof production may lag block arrivals. Consider whether parallel VDF lanes with coordination are feasible. |

### 10.3 Security Observations

| ID | Severity | Description |
|----|----------|-------------|
| S-001 | 🟠 | `Q_ENCRYPTION_PASSPHRASE` is set in the systemd service `Environment=` field, visible via `systemctl show`. Should be sourced from a secrets file with restricted permissions. |
| S-002 | 🟡 | Rate limiting (governor) is configured at the route level. DDoS amplification vectors through expensive endpoints (AI inference, ZK proof generation) should have separate, lower rate limits. |
| S-003 | 🟡 | BalanceRootV1 activates at height 18,600,000 (~June 2026 estimated). Until activation, light clients cannot cryptographically verify balances — full trust in bootstrap nodes required. |

### 10.4 Items for Next Iteration

The following areas were identified for deeper analysis in iteration 2:

1. **Mining reward accounting path** — Trace coinbase creation through `BalanceConsensusEngine` to P2P propagation and final balance update. Verify no double-credit or missed credit scenarios.
2. **DEX invariant verification** — Audit constant-product invariant enforcement under concurrent swaps. Check for rounding errors in 24-decimal arithmetic.
3. **BalanceRootV1 activation path** — Verify the guard logic in `block_producer.rs` and `consensus_guard.rs` for height ≥ 18,600,000. Confirm there is no off-by-one at activation.
4. **Stratum pool security** — Review extranonce handling and share difficulty validation for potential grinding attacks.
5. **Bridge stub security** — Ensure stub bridge endpoints return appropriate errors and cannot be used to create phantom swap state.

---

*Document status: Iteration 1 — Full feature inventory complete. Next iteration will deepen analysis on mining reward path, DEX invariants, and BalanceRootV1 activation.*
