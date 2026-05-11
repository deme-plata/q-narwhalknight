# Technical Review: Consensus Layer Correctness & Performance
**Date:** 2026-05-11  
**Branch:** feature/safe-batched-sync-v1.0.2  
**Version:** v10.9.0  
**Scope:** DAG-Knight finality, BFT threshold enforcement, VDF anchor election, SIMD/io_uring/async optimisations, real-world throughput baseline

---

## Executive Summary

Q-NarwhalKnight implements a sophisticated DAG-BFT consensus architecture with genuine Bracha reliable broadcast, fee-ordered Narwhal mempool, and 4 parallel lock-free block producers. However, three structural gaps prevent the system from being scientifically correct at the consensus layer:

1. **Finality is implicit** — no cryptographic finality certificate is produced; a block accepted into the chain is treated as final with no verifiable proof
2. **VDF anchor election exists in the DAG layer but is disconnected from block production** — every block header contains `anchor_validator: None`
3. **BFT threshold infrastructure is present but runs with a single active validator** — the 2f+1 enforcement code is correct but there is nobody to form a quorum with

On the performance side, five concrete bottlenecks prevent reaching theoretical throughput: scalar Merkle tree computation, scalar wallet balance sorting, a fully-stubbed io_uring integration, gossipsub block validation running inline in the event loop, and no true async pipelining between network fetch and disk write in turbo sync.

This document defines the root cause of each issue, the correct fix, and a priority-ordered implementation plan.

---

## 1. Current Architecture (Verified Reality)

### 1.1 Block Production

| Property | Configured Value | Source |
|---|---|---|
| Parallel producers | 4 | `lib.rs:2913` — `let num_producers = 4` |
| Block interval | 0 (produce as fast as possible) | `config.rs:54` |
| Max user txs/block | 5,000 | `block_producer.rs:2053` |
| Max solutions/block | 250 | `block_producer.rs:61` (reduced from 10K; 10K stalled serialisation) |
| Solution distribution | Round-robin across 4 producers | `block_producer.rs:3156` |
| Block uniqueness | `producer_id` embedded in header | `main.rs:17374` |
| "Dual-lane mining" | Code complete, activation height = `u64::MAX` (never fires) | `block_producer.rs:1828` |

### 1.2 Mempool

Bracha's reliable broadcast protocol is **fully implemented** in `crates/q-narwhal-core/src/reliable_broadcast.rs` (lines 85–230). Three phases are active: SEND → ECHO (2f+1) → READY (2f+1) → DELIVER. The ProductionMempool is fee-ordered (highest fee first, then oldest first), holds 10,000 transactions max with a 5-minute eviction TTL, and enforces a 1MB per-transaction size limit. There is **no byte-size limit per block** — only the 5,000-transaction count cap.

### 1.3 Measured Throughput

From the benchmark file (`/tps-benchmark-results.txt`) and the status endpoint calculation (`handlers.rs:878–916`):

- **API-level TPS:** 457–500 TPS (500 txs in 1.09s, 100% success rate)
- **Latency P50:** 41ms — **P99:** 1,004ms
- **Block production queue depth:** 0–2 solutions in normal operation (drained immediately)
- **Theoretical ceiling:** 4 producers × 5,000 txs/block × ~1 block/sec/producer = **20,000 TPS** — not approached in practice

The gap between 457 TPS observed and 20,000 TPS theoretical is explained entirely by the bottlenecks in sections 3 and 4 below.

---

## 2. Consensus Correctness Gaps

### 2.1 Gap 1 — Implicit Finality (No Cryptographic Certificate)

**Root cause:**  
`crates/q-types/src/block.rs` defines a `FinalityCertificate` struct (lines 599–619) with fields for `commit_round`, `validator_signatures`, and a `quorum_bitmap`. This structure is **never populated**. Finality is determined solely by whether the gossipsub handler accepts a block and writes it to RocksDB. There is no verifiable proof that a quorum of validators agreed.

**DAG commit path** (`crates/q-dag-knight/src/lib.rs:213–227`):  
- Anchor elected on even rounds (line 174)
- `evaluate_commit_with_vrf()` called (line 201)
- Committed vertices stored in `HashMap<Round, Vec<VertexId>>` (line 214)
- **Nothing is signed or aggregated**

**Delta value:** Hardcoded `delta = 1` in `crates/q-dag-knight/src/commit_logic.rs:90` with the comment "Aggressive delta=1 for sub-50ms finality (was 4)". At ~1 second per block this means finality is ~2 seconds. The code at line 195 attempts an "enhanced delta" of `delta * 3/4` when VRF entropy quality > 0.8 — this would give δ = 0.75, meaning same-round finality, but it only fires if post-quantum VRF is active.

**Fix — Finality Certificate:**  
Wire the existing `FinalityCertificate` struct. When DAG-Knight commits a round, collect the commit decision, hash it (`Blake3(commit_round || anchor_vertex_id || block_hash)`), store it as `block.finality_certificate`, and expose it via a new endpoint `/api/v1/blocks/:height/finality`. Initially with a single validator this is a self-signed certificate (degenerate but verifiable). As additional validators come online, upgrade to BLS threshold aggregation (3-of-4 for f=1).

**Files to change:** `crates/q-dag-knight/src/lib.rs`, `crates/q-types/src/block.rs`, `crates/q-api-server/src/main.rs` (gossipsub handler), `crates/q-api-server/src/handlers.rs` (new endpoint).

---

### 2.2 Gap 2 — VDF Anchor Election Not Wired to Block Headers

**Root cause:**  
The anchor election is computed correctly inside the DAG layer (`crates/q-dag-knight/src/anchor_election.rs:205–335`) — the VRF runs, an anchor vertex is elected on even rounds, and the result is available as `AnchorElectionResult`. However, the result is **never passed back** to the block producer. Every block header sets `anchor_validator: None` (`block_producer.rs:1048` — the TODO comment has been there since v1.0.x).

**VDF proof status (REAL, not a stub):**  
The `vdf_proof` field in every block header contains a genuine SHA3-based VDF output with a Wesolowski verification proof (`block_producer.rs:918–940`). The comment at line 926 explicitly states "ACTUAL VDF output, NOT parent hash". Adaptive difficulty scaling parameters are included. The Genus-2 post-quantum VDF is compiled behind `#[cfg(feature = "advanced-crypto")]` and falls back to SHA3-based VDF when disabled (`anchor_election.rs:132`).

**What's missing:**  
The VDF output feeds into anchor election entropy (`anchor_election.rs:228–253`) but the elected anchor is discarded before it reaches the block header. This means the chain has no on-chain record of which validator proposed each block, making equivocation detection impossible and Byzantine accountability unenforceable.

**Fix — Wire Anchor to Block Header:**  
1. In `block_producer.rs`, after `vdf_proof` is generated (line ~930), call `dag_knight.get_last_elected_anchor()` or pass the `AnchorElectionResult` through the produce-block channel.
2. Set `block.header.anchor_validator = Some(elected_anchor.validator_id)`.
3. Add anchor validation in the gossipsub block handler — reject blocks above a future activation height where `anchor_validator == None`.
4. Long-term: make anchor election the **sole** block production trigger (replace the current timer-based produce loop with DAG-round-based triggering).

**Files to change:** `crates/q-dag-knight/src/lib.rs` (expose last anchor), `crates/q-api-server/src/block_producer.rs` (set field), `crates/q-api-server/src/main.rs` (gossipsub validation).

---

### 2.3 Gap 3 — BFT Threshold Code Correct, No Quorum Possible

**Root cause:**  
The vote counting is implemented correctly. `crates/q-narwhal-core/src/consensus_voting.rs:430`:
```rust
if accept_count >= self.config.byzantine_threshold {
    vote_tally.threshold_met = true;
}
```
`byzantine_threshold` is computed as `2f + 1` at line 48.

DAG-Knight is initialised with `f = 3` (10 validators, quorum of 7) at `main.rs:7128`.  
NarwhalCore BFT uses `f = 1` (4 validators, quorum of 3) at `main.rs:7209`.

The problem is not the code — it is that only **one node is running the validator role**. With a single validator `f` is effectively 0 and every block self-certifies. The BFT guarantees stated in the whitepaper require at least 4 validators (for f=1) to hold under Byzantine conditions.

**Fix — Multi-Validator Bootstrap:**  
This is an operational and protocol change, not just code:
1. Define a `ValidatorSet` in genesis config with 4 entries (Beta, Gamma, Delta, Epsilon peer IDs and signing keys).
2. Each bootstrap node registers as a validator on startup using its P2P identity key.
3. Route DAG vertex certificates between validators over the existing gossipsub `/qnk/mainnet-genesis/blocks` topic, or a new dedicated `/qnk/mainnet-genesis/vertices` topic.
4. Enforce 2f+1 quorum before committing rounds — gate block production behind DAG commit, not behind timer.

This is the largest structural change and should be activated at a future height (BAL-001 enforcement style) to avoid breaking the live chain.

---

## 3. Performance Bottlenecks — SIMD & Async

### 3.1 Bottleneck 1 — Scalar Transaction Merkle Root (HIGH)

**Location:** `crates/q-api-server/src/block_producer.rs:2076–2090`

```rust
fn compute_tx_merkle_root(&self, transactions: &[Transaction]) -> TxHash {
    let mut hasher = Sha3_256::new();
    for tx in transactions {   // ← scalar sequential loop
        hasher.update(tx.hash());
    }
    hasher.finalize().into()
}
```

This is not a true Merkle tree — it is a running hash. For 5,000 transactions this processes ~160KB of data sequentially on a single core. The `SimdMerkleTree` struct already exists in `crates/q-crypto-simd/src/simd_merkle.rs` with AVX-512 (8× throughput) and AVX2 (4×) implementations. It is **not imported or called** from `block_producer.rs`.

**Fix:**
```rust
// In block_producer.rs, replace compute_tx_merkle_root body:
use q_crypto_simd::SimdMerkleTree;
let leaves: Vec<[u8; 32]> = transactions.iter().map(|tx| tx.hash()).collect();
SimdMerkleTree::compute_root(&leaves)
```
Add `q-crypto-simd` to `q-api-server/Cargo.toml` dependencies. Expected gain: **4–8× on blocks with >100 transactions**.

---

### 3.2 Bottleneck 2 — Scalar Wallet Sort in Balance Root (HIGH)

**Location:** `crates/q-storage/src/lib.rs:4544`

```rust
sorted.sort_by_key(|(addr, _)| *addr);  // scalar sort over millions of entries
```

`compute_balance_root_for_block()` runs on every locally produced block above h=17,742,000. With ~1,346 wallets today this is cheap, but as wallet count grows this becomes the critical path for block production latency. Rayon is already in `Cargo.toml` at v1.10 and Blake3 is compiled with `rayon` feature.

**Fix:**
```rust
use rayon::slice::ParallelSliceMut;
sorted.par_sort_by_key(|(addr, _)| *addr);
```
For the leaf hashing loop, use `rayon::iter::ParallelIterator`:
```rust
let leaves: Vec<[u8; 32]> = sorted
    .par_iter()
    .map(|(addr, amount)| {
        let mut h = blake3::Hasher::new();
        h.update(addr.as_slice());
        h.update(&amount.to_be_bytes());
        *h.finalize().as_bytes()
    })
    .collect();
```
Expected gain: **linear speedup with core count** (Epsilon has 48 cores — ~30× for sort + hash).

---

### 3.3 Bottleneck 3 — Gossipsub Block Validation Inline in Event Loop (HIGH)

**Location:** `crates/q-api-server/src/main.rs:~10594–10950`

The gossipsub event loop processes all topics in a single `while let Some((topic, data)) = gossipsub_rx.recv().await` loop. Block validation (signature verification, PQC checks, storage writes) happens **inline** — not in a spawned task. If a block write takes 100ms, every gossipsub message (new blocks, peer heights, mining submissions) is blocked for that duration. The channel is bounded at 10,000 messages; under sustained block production this fills and messages are dropped.

**Fix:**
```rust
if topic.ends_with("/blocks") {
    let state = app_state_gossip.clone();
    tokio::spawn(async move {
        handle_received_block(state, data).await;
    });
    continue;  // gossipsub loop returns immediately
}
```
Move all block validation and storage writes into `handle_received_block()`. The gossipsub loop becomes a pure dispatcher — receive, clone Arc, spawn, continue. Use a semaphore (e.g. `Arc<Semaphore::new(16)>`) on spawned block handlers to prevent unbounded parallelism under a block flood.

Expected gain: **eliminates head-of-line blocking**, reduces P99 latency from ~1,000ms toward P50 latency.

---

### 3.4 Bottleneck 4 — io_uring Integration Incomplete (MEDIUM)

**Location:** `crates/q-kernel-io/src/uring.rs:210`

The `IoUringEngine` struct and operation types are defined but the implementation contains:
```
// TODO: tokio-uring API has changed, implement proper operations
```

All RocksDB reads and writes go through 55+ `spawn_blocking` calls (`crates/q-storage/src/kv.rs:1548` and elsewhere). This moves blocking work off the tokio reactor but does not achieve true async I/O — it occupies threads in tokio's blocking pool, which serialises under load.

The `AsyncStorageEngine` (`crates/q-storage/src/async_engine.rs`) is a dedicated OS thread with micro-batching (1,024 writes per batch, 4ms max wait). This reduces RocksDB write amplification but the underlying I/O is still synchronous.

**Fix plan:**
1. Update `q-kernel-io/uring.rs` to use current `tokio-uring` API (v0.4+). Key changes: `tokio_uring::fs::File::open()`, `file.read_at()`, `file.write_at()`.
2. Replace the RocksDB WAL sync path (`lib.rs:save_qblock`) with an io_uring write for the WAL file directly.
3. Keep `spawn_blocking` for compaction and column-family operations (RocksDB's C++ layer is not io_uring-aware).
4. Gate behind `#[cfg(target_os = "linux")]` with `spawn_blocking` fallback.

Expected gain: **5–10× disk write throughput** for the block storage hot path on Linux (Epsilon, Beta, Gamma, Delta are all Linux).

---

### 3.5 Bottleneck 5 — Turbo Sync Not Truly Pipelined (MEDIUM)

**Location:** `crates/q-storage/src/turbo_sync.rs`

The turbo sync claims "Download → Decompress → Verify → Apply simultaneously" in its header comment (lines 1–19). In practice, the Apply phase writes to RocksDB via `spawn_blocking`, which cannot truly overlap with the network fetch phase because both contend for tokio's blocking thread pool. When the pool is saturated (512 threads default), apply tasks queue behind fetch tasks.

**Fix — Dual-channel pipeline:**
```
Network fetch task  ──→  channel(capacity=8)  ──→  Storage apply task
     (tokio async)                                  (dedicated OS thread)
```
Use a dedicated `std::thread` (not spawn_blocking) for the apply phase with a fixed-size `ArrayQueue<BatchedBlocks>` between fetch and apply. The fetch task never waits on disk — it fills the queue. The apply thread drains it. This gives true network/disk overlap.

Expected gain: **20–40% faster turbo sync** on fast networks like Epsilon (10Gbit).

---

## 4. Finality Time Measurement

### 4.1 What Exists

`FinalityMetrics` (`block_producer.rs:207–248`) tracks:
- `avg_production_latency_us` — time from produce_block call to block object creation
- `avg_broadcast_latency_us` — time from block object to gossipsub publish
- Rolling 200-block window with `(epoch_ms, tx_count)` — added in v10.9.0

### 4.2 What Is Missing

True end-to-end finality requires measuring: **block produced on node A → block seen and committed on nodes B, C, D**. This is not tracked anywhere. There is no gossipsub acknowledgement mechanism and no "block seen" confirmation path.

### 4.3 Fix — Peer Confirmation Tracking

Add a `block_propagation_map: Arc<DashMap<[u8; 32], BlockPropagationState>>` to AppState:

```rust
struct BlockPropagationState {
    produced_at_ms: u64,
    seen_by: Vec<PeerId>,
    confirmed_at_ms: Option<u64>,  // set when seen_by.len() >= 2f+1
}
```

When a locally-produced block is broadcast, insert into the map. When the gossipsub handler receives a `PeerHeightAnnouncement` at or above a block's height from a peer, record that peer as "seen". When 2f+1 peers (or all known peers if f=0) have seen the block, record `confirmed_at_ms` and log:

```
INFO [FINALITY] Block 17749730 confirmed by 3/3 peers in 847ms
     (production: 12ms, broadcast: 3ms, propagation: 832ms)
```

This gives the first real end-to-end finality measurements and feeds the metrics endpoint.

---

## 5. Priority Matrix

| # | Fix | Impact | Effort | Risk |
|---|-----|--------|--------|------|
| 1 | Offload gossipsub block validation to spawned tasks | Eliminates P99 latency spike, fixes head-of-line blocking | Low | Low |
| 2 | `par_sort` + parallel leaf hashing in `compute_balance_root_for_block` | 30× on Epsilon (48 cores) | Low | Low |
| 3 | Wire `SimdMerkleTree` into `compute_tx_merkle_root` | 4–8× Merkle throughput | Low | Low |
| 4 | Peer confirmation tracking for finality measurement | Baseline data for all future consensus work | Medium | Low |
| 5 | Complete io_uring integration in q-kernel-io | 5–10× disk write throughput | Medium | Medium |
| 6 | True pipeline in turbo sync (dedicated apply thread) | 20–40% faster sync | Medium | Medium |
| 7 | Wire anchor election result into block headers | Enables Byzantine accountability | Medium | Medium |
| 8 | Cryptographic finality certificate (self-signed initially) | Verifiable finality proof | Medium | Low |
| 9 | Multi-validator BFT quorum (4 bootstrap nodes as validators) | Genuine BFT guarantees, requires protocol change | High | High |

Items 1–3 can be implemented and deployed in a single release with no protocol change and no risk to existing state.  
Items 4–6 are self-contained infrastructure improvements.  
Items 7–8 require a height-gated activation (BAL-001 style).  
Item 9 is a major protocol upgrade requiring multi-node coordination, new gossipsub topics, and a validator key ceremony.

---

## 6. Expected Throughput After Items 1–3

With 4 parallel producers, 5,000 txs/block, gossipsub validation offloaded, and SIMD-parallel balance root:

| Metric | Today | After items 1–3 | After items 1–6 |
|--------|-------|-----------------|-----------------|
| API TPS | 457 | ~2,000–5,000 | ~10,000+ |
| Block production latency | ~12ms | ~5ms | ~3ms |
| Balance root computation | ~N×hash_time | ~N/48×hash_time (Epsilon) | same |
| Turbo sync speed | ~570 blocks/sec | ~570 blocks/sec | ~800 blocks/sec |
| Finality visibility | none | none | measured |
| Finality guarantee | implicit | implicit | implicit + measured |
| Finality proof | none | none | self-signed cert |

---

## 7. References

| File | Key Lines | Topic |
|---|---|---|
| `crates/q-dag-knight/src/lib.rs` | 174–227 | Anchor election, round commit |
| `crates/q-dag-knight/src/commit_logic.rs` | 90, 195 | Delta=1, enhanced delta |
| `crates/q-dag-knight/src/anchor_election.rs` | 205–335 | VDF entropy → anchor result |
| `crates/q-types/src/block.rs` | 304–326, 599–619 | VDFProof, FinalityCertificate |
| `crates/q-narwhal-core/src/reliable_broadcast.rs` | 85–230 | Bracha 3-phase protocol |
| `crates/q-narwhal-core/src/consensus_voting.rs` | 430, 48 | 2f+1 threshold |
| `crates/q-narwhal-core/src/production_mempool.rs` | 220–481 | Fee ordering, capacity |
| `crates/q-api-server/src/block_producer.rs` | 61, 1048, 2053, 2076, 3156 | Lanes, anchor None, tx cap, scalar Merkle, round-robin |
| `crates/q-storage/src/lib.rs` | 4528–4558 | compute_balance_root_for_block |
| `crates/q-crypto-simd/src/simd_merkle.rs` | 1–200 | SimdMerkleTree (AVX2/AVX-512) |
| `crates/q-kernel-io/src/uring.rs` | 210 | io_uring stub |
| `crates/q-storage/src/async_engine.rs` | 29–38, 96 | Micro-batch config, OS thread |
| `crates/q-storage/src/kv.rs` | 1548 | spawn_blocking pattern (×55) |
| `crates/q-api-server/src/main.rs` | 7128, 7209, 10594 | f values, gossipsub inline validation |
| `crates/q-api-server/src/lib.rs` | 2913 | num_producers = 4 |
| `crates/q-api-server/src/handlers.rs` | 878–916, 2134 | TPS calculation, tx submission |
