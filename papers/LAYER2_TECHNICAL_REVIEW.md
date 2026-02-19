# Q-NarwhalKnight Layer 2 Scaling Architecture: Technical Review

**Version**: 1.0.0
**Date**: February 2026
**Authors**: Q-NarwhalKnight Core Team
**Status**: Technical Design Review
**Target Crate**: `q-l2-bridge`, `q-l2-executor`, `q-l2-sequencer`

---

## 1. Executive Summary

Q-NarwhalKnight (QNK) achieves 48,000+ TPS on its DAG-Knight L1 consensus layer, with sub-3s finality and post-quantum cryptographic guarantees. However, as DeFi activity grows across the DEX, stablecoin (QUGUSD), and custom token ecosystems, the demand profile increasingly includes high-frequency microtransactions, game-state updates, and complex multi-step DeFi operations that benefit from dedicated execution environments with lower latency and reduced per-transaction cost.

This document specifies a Layer 2 (L2) scaling architecture for Q-NarwhalKnight that inherits the L1's post-quantum security guarantees while delivering:

- **500,000+ TPS** on L2 through batched state transitions
- **Sub-50ms transaction confirmation** via off-chain sequencing
- **10-100x cost reduction** through amortized L1 settlement
- **Full EVM/VittuaVM compatibility** for existing smart contracts
- **Hybrid rollup design** supporting both optimistic and ZK proof modes

The architecture draws directly from the optimization patterns proven in the QNK miner: dedicated OS threads for CPU-bound work, channel-based communication between subsystems, SIMD-accelerated batch processing, core-affinity pinning, and lock-free atomic state sharing. These patterns, which produce measurable throughput gains in the mining pipeline, translate directly to L2 sequencer and prover performance.

---

## 2. Architecture Overview

### 2.1 Design Principles Derived from q-miner

The QNK miner (`crates/q-miner`) demonstrates several architectural patterns that are directly applicable to L2 execution:

| Miner Pattern | L2 Application |
|---------------|----------------|
| `std::thread::spawn` for CPU-bound work instead of tokio tasks | Dedicated prover threads, separate from async sequencer I/O |
| `core_affinity::set_for_current(core_ids[thread_id])` | Pin prover threads to NUMA-local cores for cache coherency |
| `AtomicU64` for lock-free hash rate sharing | Lock-free state root and batch counter sharing across subsystems |
| `mpsc::channel` for share submission | Transaction ingress queue, proof submission pipeline |
| Thread-local counters flushed every 1024 iterations | Batch state diffs accumulated locally, flushed per block |
| `tokio::runtime::Handle::block_on()` for async-from-sync | Prover threads dispatch settlement transactions to L1 without blocking |
| Pre-allocated buffers with in-place nonce overwrite | Pre-allocated state transition buffers, zero-copy serialization |

### 2.2 System Topology

```
                          L2 Users / DApps
                               |
                    ┌──────────┴──────────┐
                    |    L2 RPC Gateway    |   Port 8090
                    |  (Axum HTTP + WS)   |
                    └──────────┬──────────┘
                               |
              ┌────────────────┼────────────────┐
              |                |                |
     ┌────────┴────────┐  ┌───┴───┐  ┌────────┴────────┐
     |   Sequencer     |  | Mempool|  | State Manager   |
     | (Batch Builder) |  | (FIFO +|  | (RocksDB +      |
     |                 |  |  Prio) |  |  mmap state)     |
     └────────┬────────┘  └───┬───┘  └────────┬────────┘
              |                |                |
              └────────────────┼────────────────┘
                               |
                    ┌──────────┴──────────┐
                    |   Proof Generator   |
                    |  (N dedicated OS    |
                    |   threads, pinned)  |
                    └──────────┬──────────┘
                               |
                    ┌──────────┴──────────┐
                    |   L1 Settlement     |
                    |  Bridge Contract    |
                    |  (q-l2-bridge)      |
                    └──────────┬──────────┘
                               |
                    ┌──────────┴──────────┐
                    |   QNK L1 (DAG-      |
                    |   Knight Consensus) |
                    └─────────────────────┘
```

### 2.3 Core Data Structures

```rust
/// L2 transaction submitted by users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Transaction {
    pub tx_id: [u8; 32],
    pub from: [u8; 32],
    pub to: [u8; 32],
    pub value: u128,
    pub data: Vec<u8>,
    pub nonce: u64,
    pub gas_limit: u64,
    pub gas_price: u64,
    pub signature: L2Signature,
    pub l2_chain_id: u32,
}

/// Post-quantum L2 signature (Dilithium3 for speed, upgradable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum L2Signature {
    /// Ed25519 for backward compatibility (Phase 0)
    Classical([u8; 64]),
    /// Dilithium3 for post-quantum security (Phase 1+)
    PostQuantum(Vec<u8>),
    /// Hybrid: both must verify
    Hybrid {
        classical: [u8; 64],
        post_quantum: Vec<u8>,
    },
}

/// A batch of L2 transactions committed together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Batch {
    pub batch_index: u64,
    pub transactions: Vec<L2Transaction>,
    pub pre_state_root: [u8; 32],
    pub post_state_root: [u8; 32],
    pub timestamp: u64,
    pub sequencer_signature: L2Signature,
    /// Compressed state diff (LZ4) for data availability
    pub state_diff_compressed: Vec<u8>,
}

/// Settlement proof posted to L1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SettlementProof {
    /// Optimistic: no proof, subject to challenge period
    Optimistic {
        batch_index: u64,
        state_root: [u8; 32],
        challenge_deadline: u64,
    },
    /// ZK validity proof: immediately final
    ZkValidity {
        batch_index: u64,
        state_root: [u8; 32],
        proof: ZkProofData,
    },
}

/// ZK proof payload (STARK or SNARK depending on configuration)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkProofData {
    pub proof_type: ProofType,
    pub proof_bytes: Vec<u8>,
    pub public_inputs: Vec<[u8; 32]>,
    pub verification_key_hash: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofType {
    /// From q-zk-stark: transparent, post-quantum safe, larger proofs
    Stark,
    /// From q-zk-snark: smaller proofs, faster verification, requires trusted setup
    Snark,
    /// Recursive composition: SNARK wrapping STARK for compact PQ-safe proofs
    RecursiveSnarkOfStark,
}
```

---

## 3. State Channels

State channels provide the lowest-latency L2 primitive: two parties lock collateral on L1, exchange signed state updates off-chain, and settle the final state back to L1 when the channel closes.

### 3.1 Channel Lifecycle

```
1. OPEN:   Party A deposits 100 QUG into L1 bridge contract
           Party B deposits 50 QUG into L1 bridge contract
           Channel ID = H(A || B || nonce || block_height)

2. UPDATE: Off-chain signed state updates (sub-1ms latency)
           State_n = { A: 80 QUG, B: 70 QUG, nonce: n }
           Both parties sign each state transition

3. CLOSE:  Either party submits latest signed state to L1
           Challenge period: 256 L1 blocks (~12 minutes)
           If no dispute: balances distributed per final state
           If dispute: counterparty submits higher-nonce state
```

### 3.2 Channel State Machine

```rust
/// State channel between two L2 participants
pub struct StateChannel {
    pub channel_id: [u8; 32],
    pub participant_a: [u8; 32],
    pub participant_b: [u8; 32],
    pub balance_a: u128,
    pub balance_b: u128,
    pub nonce: u64,
    pub status: ChannelStatus,
    /// Both participants must sign every state update
    pub latest_sigs: (L2Signature, L2Signature),
}

pub enum ChannelStatus {
    /// Channel is open, off-chain updates active
    Active,
    /// Close requested, challenge period running
    Closing { deadline_block: u64, closer: [u8; 32] },
    /// Challenge period expired, funds distributed
    Settled,
    /// Dispute resolved via higher-nonce state
    Disputed { resolution_block: u64 },
}

/// Off-chain state update (exchanged directly between parties)
pub struct ChannelUpdate {
    pub channel_id: [u8; 32],
    pub nonce: u64,
    pub balance_a: u128,
    pub balance_b: u128,
    pub app_data: Vec<u8>,  // Arbitrary application state
    pub sig_a: L2Signature,
    pub sig_b: L2Signature,
}
```

### 3.3 Performance Characteristics

State channels achieve the theoretical minimum latency for bilateral transactions: one network round-trip between participants. With QNK's libp2p gossipsub infrastructure (12ms median RTT on direct connections, <300ms through Tor), state channels deliver:

- **Throughput**: Unlimited (bounded only by network bandwidth between participants)
- **Latency**: 1 RTT (12ms direct, <300ms Tor)
- **L1 cost**: 2 transactions total (open + close), amortized over channel lifetime
- **Security**: Disputes resolved on L1 with full DAG-Knight finality guarantees

---

## 4. Rollup Design

### 4.1 Sequencer Architecture

The sequencer is the central coordination point for L2 transaction ordering. Following the miner's architecture, it separates I/O-bound work (transaction ingestion, RPC serving) from CPU-bound work (state execution, proof generation) using dedicated OS threads.

```rust
/// L2 Sequencer — mirrors q-miner's thread architecture
pub struct L2Sequencer {
    /// Transaction ingress from RPC (async, tokio-managed)
    tx_ingress: mpsc::Receiver<L2Transaction>,

    /// Batch builder runs on dedicated OS thread (like mining_thread)
    batch_builder_handle: Option<std::thread::JoinHandle<()>>,

    /// Proof generators: N dedicated OS threads pinned to cores
    /// Mirrors: std::thread::Builder::new().name("miner-{}").spawn()
    prover_handles: Vec<std::thread::JoinHandle<()>>,

    /// Lock-free batch counter (mirrors hash_counter: Arc<AtomicU64>)
    batch_counter: Arc<AtomicU64>,

    /// Current batch being assembled
    /// Mirrors: current_work: Arc<RwLock<Option<WorkUnit>>>
    current_batch: Arc<RwLock<Option<L2Batch>>>,

    /// Signal for new batch ready (mirrors new_block_signal)
    new_batch_signal: Arc<AtomicU64>,

    /// Graceful shutdown flag (mirrors is_running)
    is_running: Arc<AtomicBool>,

    /// Tokio handle for async-from-sync dispatch
    /// Mirrors: tokio_handle = tokio::runtime::Handle::current()
    tokio_handle: tokio::runtime::Handle,
}

impl L2Sequencer {
    pub fn start(&mut self, num_prover_threads: usize) {
        self.is_running.store(true, Ordering::SeqCst);

        // Spawn batch builder on dedicated OS thread
        let is_running = self.is_running.clone();
        let current_batch = self.current_batch.clone();
        let new_batch_signal = self.new_batch_signal.clone();
        let batch_counter = self.batch_counter.clone();

        self.batch_builder_handle = Some(
            std::thread::Builder::new()
                .name("l2-batch-builder".into())
                .spawn(move || {
                    batch_builder_thread(
                        is_running, current_batch,
                        new_batch_signal, batch_counter,
                    );
                })
                .expect("Failed to spawn batch builder thread")
        );

        // Spawn prover threads pinned to cores
        // Mirrors: q-miner pool_mining_thread core_affinity pattern
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        for i in 0..num_prover_threads {
            let is_running = self.is_running.clone();
            let current_batch = self.current_batch.clone();
            let new_batch_signal = self.new_batch_signal.clone();
            let tokio_handle = self.tokio_handle.clone();
            let core_id = core_ids.get(i).cloned();

            self.prover_handles.push(
                std::thread::Builder::new()
                    .name(format!("l2-prover-{}", i))
                    .spawn(move || {
                        // Pin to core for cache locality (NUMA-aware)
                        if let Some(core) = core_id {
                            core_affinity::set_for_current(core);
                        }
                        prover_thread(
                            i, is_running, current_batch,
                            new_batch_signal, tokio_handle,
                        );
                    })
                    .expect("Failed to spawn prover thread")
            );
        }
    }
}
```

### 4.2 Optimistic Rollup Mode

In optimistic mode, the sequencer posts batch state roots to L1 without proofs. A challenge period of 256 L1 blocks (~12 minutes at QNK's block rate) allows any observer to submit a fraud proof if the state transition is invalid.

**Settlement flow:**

1. Sequencer executes N transactions, producing `post_state_root`
2. Sequencer posts `(batch_index, post_state_root, state_diff_hash)` to L1 bridge contract
3. Challenge window opens (256 blocks)
4. If no challenge: batch is finalized, withdrawals enabled
5. If challenged: L1 contract re-executes disputed transaction using on-chain state diff data

**Fraud proof structure:**

```rust
/// Fraud proof submitted by a verifier to challenge an invalid batch
pub struct FraudProof {
    pub batch_index: u64,
    pub tx_index: u32,
    /// Merkle proof of the transaction within the batch
    pub tx_inclusion_proof: Vec<[u8; 32]>,
    /// Pre-state for the disputed transaction (Merkle proof into state trie)
    pub pre_state_proof: StateProof,
    /// The transaction itself
    pub transaction: L2Transaction,
    /// Expected post-state root after correct execution
    pub correct_post_state: [u8; 32],
}

/// Sparse Merkle tree proof for a subset of state
pub struct StateProof {
    pub account_proofs: Vec<AccountStateProof>,
    pub storage_proofs: Vec<StorageSlotProof>,
}

pub struct AccountStateProof {
    pub address: [u8; 32],
    pub nonce: u64,
    pub balance: u128,
    pub code_hash: [u8; 32],
    pub storage_root: [u8; 32],
    pub merkle_siblings: Vec<[u8; 32]>,
}
```

### 4.3 ZK Rollup Mode

In ZK mode, the sequencer generates a validity proof for each batch before posting to L1. The proof guarantees that the state transition is correct; no challenge period is needed.

QNK has two existing ZK crates:

- **`q-zk-stark`**: Transparent setup, post-quantum safe (hash-based), larger proofs (~100KB)
- **`q-zk-snark`**: Compact proofs (~256 bytes), fast verification, requires trusted setup

The L2 uses a **recursive composition** strategy: the prover generates a STARK proof (post-quantum safe, no trusted setup), then wraps it in a SNARK for compact on-chain verification. This is implemented in `q-recursive-proofs`.

```rust
/// Prover thread — runs on dedicated OS thread, pinned to core
/// Mirrors: q-miner's mining_thread() pattern
fn prover_thread(
    thread_id: usize,
    is_running: Arc<AtomicBool>,
    current_batch: Arc<RwLock<Option<L2Batch>>>,
    new_batch_signal: Arc<AtomicU64>,
    tokio_handle: tokio::runtime::Handle,
) {
    let mut last_signal = 0u64;

    while is_running.load(Ordering::SeqCst) {
        let signal = new_batch_signal.load(Ordering::Relaxed);
        if signal == last_signal {
            // No new batch; spin-wait with backoff
            // Mirrors: miner waiting for new work
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }
        last_signal = signal;

        let batch = {
            let guard = current_batch.blocking_read();
            guard.clone()
        };

        if let Some(batch) = batch {
            // Phase 1: Generate STARK proof (CPU-intensive, post-quantum safe)
            let stark_proof = generate_stark_proof(&batch);

            // Phase 2: Wrap STARK in SNARK for compact verification
            let snark_wrapper = recursive_snark_wrap(&stark_proof);

            // Phase 3: Submit to L1 via async bridge
            // Mirrors: tokio_handle.block_on() for async-from-sync I/O
            let proof = SettlementProof::ZkValidity {
                batch_index: batch.batch_index,
                state_root: batch.post_state_root,
                proof: ZkProofData {
                    proof_type: ProofType::RecursiveSnarkOfStark,
                    proof_bytes: snark_wrapper,
                    public_inputs: vec![
                        batch.pre_state_root,
                        batch.post_state_root,
                    ],
                    verification_key_hash: VERIFICATION_KEY_HASH,
                },
            };

            tokio_handle.block_on(async {
                submit_proof_to_l1(proof).await
            }).ok();
        }
    }
}
```

---

## 5. Settlement Layer

The L1 bridge contract is a VittuaVM smart contract deployed on QNK L1 that holds deposited assets and validates settlement proofs.

### 5.1 Bridge Contract Interface

```rust
/// L1 Bridge contract state (deployed on VittuaVM)
pub struct L2BridgeContract {
    /// Total assets locked for L2
    pub total_locked: u128,
    /// Mapping: L2 batch_index -> SettlementRecord
    pub batches: BTreeMap<u64, SettlementRecord>,
    /// Mapping: user_address -> pending_withdrawal
    pub pending_withdrawals: BTreeMap<[u8; 32], WithdrawalRequest>,
    /// Current verified L2 state root
    pub verified_state_root: [u8; 32],
    /// Sequencer bond (slashed on fraud)
    pub sequencer_bond: u128,
    /// Challenge period length in L1 blocks
    pub challenge_period: u64,
}

pub struct SettlementRecord {
    pub batch_index: u64,
    pub state_root: [u8; 32],
    pub submission_block: u64,
    pub status: SettlementStatus,
    pub proof: Option<ZkProofData>,
}

pub enum SettlementStatus {
    /// Posted, challenge period running (optimistic only)
    Pending,
    /// ZK proof verified or challenge period expired without dispute
    Finalized,
    /// Fraud proof accepted, batch reverted
    Reverted,
}

pub struct WithdrawalRequest {
    pub user: [u8; 32],
    pub amount: u128,
    pub token: [u8; 32],
    pub l2_batch_index: u64,
    pub merkle_proof: Vec<[u8; 32]>,
}
```

### 5.2 Deposit and Withdrawal Flow

**Deposit (L1 to L2):**

1. User calls `bridge.deposit(amount, l2_recipient)` on L1
2. L1 contract locks `amount` QUG/tokens
3. Sequencer observes deposit event via L1 SSE stream
4. Sequencer credits `l2_recipient` on L2 in the next batch
5. Deposit is confirmed once the L2 batch containing the credit is settled

**Withdrawal (L2 to L1):**

1. User submits withdrawal transaction on L2
2. L2 burns the user's balance and records withdrawal in batch state diff
3. Batch is posted to L1 with proof
4. After finalization: user calls `bridge.complete_withdrawal(proof)` on L1
5. L1 contract verifies Merkle proof against finalized state root and releases funds

---

## 6. Data Availability

Every L2 batch must make its state diff data available so that any party can reconstruct the L2 state independently. QNK uses a tiered data availability (DA) strategy.

### 6.1 Tiered DA Architecture

| Tier | Storage | Latency | Cost | Retention |
|------|---------|---------|------|-----------|
| **Hot** | L1 calldata (in settlement TX) | Immediate | High | Permanent |
| **Warm** | IPFS via `q-ipfs-storage` | Seconds | Low | Pinned indefinitely |
| **Cold** | RocksDB snapshot archive | Local | Free | Configurable |

### 6.2 Integration with q-ipfs-storage

The existing `q-ipfs-storage` crate provides LZ4-compressed content-addressed storage backed by RocksDB. The L2 data availability layer uses it as follows:

```rust
/// Data availability manager for L2 state diffs
pub struct L2DataAvailability {
    /// IPFS storage backend (from q-ipfs-storage)
    ipfs: Arc<IpfsStorage>,
    /// Local RocksDB for fast retrieval
    local_db: Arc<RocksDBKV>,
    /// LZ4 compression for state diffs
    compressor: lz4_flex::frame::FrameEncoder<Vec<u8>>,
}

impl L2DataAvailability {
    /// Publish a batch's state diff to all DA tiers
    pub async fn publish_state_diff(
        &self,
        batch: &L2Batch,
    ) -> Result<DataAvailabilityCertificate> {
        // 1. Compress state diff with LZ4 (same as q-ipfs-storage)
        let compressed = lz4_flex::compress_prepend_size(
            &batch.state_diff_compressed
        );

        // 2. Store in local RocksDB (hot path for local queries)
        let key = format!("l2_da_batch_{}", batch.batch_index);
        self.local_db.put(key.as_bytes(), &compressed)?;

        // 3. Pin to IPFS for decentralized availability
        let cid = self.ipfs.store_block(&compressed).await?;

        // 4. Return DA certificate (hash + IPFS CID + size)
        Ok(DataAvailabilityCertificate {
            batch_index: batch.batch_index,
            state_diff_hash: blake3::hash(&compressed).into(),
            ipfs_cid: cid,
            compressed_size: compressed.len() as u64,
            uncompressed_size: batch.state_diff_compressed.len() as u64,
        })
    }
}

pub struct DataAvailabilityCertificate {
    pub batch_index: u64,
    pub state_diff_hash: [u8; 32],
    pub ipfs_cid: String,
    pub compressed_size: u64,
    pub uncompressed_size: u64,
}
```

---

## 7. Fraud Proofs and Validity Proofs

### 7.1 Fraud Proof Verification (Optimistic Mode)

When a verifier detects an invalid state transition, they submit a `FraudProof` to the L1 bridge contract. The contract must re-execute the disputed transaction in isolation:

1. Verify the transaction is included in the challenged batch (Merkle proof)
2. Reconstruct the pre-state from the `StateProof` and previous verified state root
3. Execute the transaction against the pre-state
4. Compare the resulting post-state root against the sequencer's claim
5. If mismatch: revert the batch, slash the sequencer's bond, reward the challenger

The re-execution environment uses a minimal VittuaVM interpreter embedded in the bridge contract, operating on the sparse state provided by the fraud proof. This keeps on-chain gas cost proportional to the single disputed transaction, not the entire batch.

### 7.2 ZK Validity Proof Generation

The prover pipeline mirrors the miner's batch processing architecture:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
| TX Ingress  | --> | Batch Builder| --> | STARK Prover | --> | SNARK Wrap |
| (async,     |     | (OS thread,  |     | (N OS threads|     | (1 thread, |
|  tokio)     |     |  batch=1024) |     |  core-pinned)|     |  GPU opt)  |
└─────────────┘     └──────────────┘     └──────────────┘     └────────────┘
      |                                                              |
      | mpsc::channel                              tokio_handle.block_on()
      |                                                              |
      v                                                              v
 Transaction                                                  L1 Settlement
   Pool                                                         Bridge
```

**SIMD optimization for proof generation:**

The STARK prover benefits from the same SIMD acceleration patterns used in the miner's `avx2_hash_batch` and `avx512_hash_batch` functions. The FRI (Fast Reed-Solomon IOP) polynomial evaluation step is embarrassingly parallel:

```rust
/// SIMD-accelerated polynomial evaluation for STARK FRI layer
/// Mirrors: q-miner cpu/mod.rs::optimizations::avx2_hash_batch
#[cfg(target_feature = "avx2")]
pub fn avx2_fri_evaluate(
    coefficients: &[u64],
    evaluation_points: &[u64],
    results: &mut [u64],
    modulus: u64,
) {
    use std::arch::x86_64::*;

    unsafe {
        let mod_vec = _mm256_set1_epi64x(modulus as i64);

        for (chunk_idx, point_chunk) in evaluation_points.chunks(4).enumerate() {
            let mut accum = _mm256_setzero_si256();
            let points = _mm256_loadu_si256(point_chunk.as_ptr() as *const __m256i);
            let mut power = _mm256_set1_epi64x(1);

            for coeff in coefficients {
                let c = _mm256_set1_epi64x(*coeff as i64);
                // accum += coeff * power (mod modulus)
                let term = _mm256_mul_epu32(c, power);
                accum = _mm256_add_epi64(accum, term);
                // power *= evaluation_point (mod modulus)
                power = _mm256_mul_epu32(power, points);
            }

            _mm256_storeu_si256(
                results[chunk_idx * 4..].as_mut_ptr() as *mut __m256i,
                accum,
            );
        }
    }
}
```

---

## 8. Cross-Layer Communication

### 8.1 L1 to L2 Message Passing

L1 contracts can send messages to L2 by emitting events that the sequencer monitors. This enables L1 DeFi protocols to trigger L2 actions (e.g., liquidations, oracle updates).

```rust
/// Cross-layer message from L1 to L2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1ToL2Message {
    pub message_id: [u8; 32],
    pub sender_l1: [u8; 32],
    pub target_l2: [u8; 32],
    pub value: u128,
    pub data: Vec<u8>,
    pub l1_block_height: u64,
    pub l1_tx_hash: [u8; 32],
}

/// Cross-layer message from L2 to L1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2ToL1Message {
    pub message_id: [u8; 32],
    pub sender_l2: [u8; 32],
    pub target_l1: [u8; 32],
    pub value: u128,
    pub data: Vec<u8>,
    pub l2_batch_index: u64,
    /// Merkle proof of message inclusion in L2 state
    pub inclusion_proof: Vec<[u8; 32]>,
}
```

### 8.2 SSE-Based Event Relay

The existing QNK L1 SSE infrastructure (used by the miner for `new_block_signal`) is extended for L2 event relay:

```rust
/// L2 bridge monitors L1 events via SSE (same pattern as miner SSE listener)
async fn l2_bridge_sse_listener(
    l1_sse_url: String,
    is_running: Arc<AtomicBool>,
    deposit_tx: mpsc::Sender<L1ToL2Message>,
) {
    // Mirrors: start_sse_listener() in q-miner/src/main.rs
    let client = eventsource_client::ClientBuilder::for_url(&l1_sse_url)
        .expect("Invalid SSE URL")
        .build();

    let mut stream = client.stream();

    while is_running.load(Ordering::Relaxed) {
        match stream.next().await {
            Some(Ok(event)) => {
                if event.event_type == "l2-deposit" {
                    if let Ok(msg) = serde_json::from_str::<L1ToL2Message>(&event.data) {
                        let _ = deposit_tx.send(msg).await;
                    }
                }
            }
            Some(Err(e)) => {
                tracing::warn!("L1 SSE error: {}, reconnecting...", e);
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
            None => break,
        }
    }
}
```

---

## 9. Performance Targets

| Metric | L1 (Current) | L2 Optimistic | L2 ZK | State Channel |
|--------|-------------|---------------|--------|---------------|
| **TPS** | 48,000 | 500,000 | 200,000 | Unlimited* |
| **Confirmation** | 2.3-2.9s | 50ms | 50ms | 12ms (1 RTT) |
| **Finality** | 2.9s | 12min (challenge) | 50ms + proof time | Immediate** |
| **TX Cost** | 1x | 0.01x | 0.05x | 0 (amortized) |
| **Security** | Full PQ | L1 fraud proofs | L1 validity proofs | L1 dispute |
| **DA** | On-chain | IPFS + calldata | IPFS + calldata | None needed |

*State channel throughput is bounded by network bandwidth between participants.
**State channel finality is immediate between participants; L1 settlement requires close + challenge period.

### 9.1 Throughput Analysis

The L2 sequencer's throughput is determined by the batch execution rate. Following the miner's optimization of thread-local counters flushed every 1024 iterations, the L2 executor accumulates state diffs in thread-local buffers and flushes per-batch:

```rust
/// Thread-local state diff accumulator
/// Mirrors: miner's local_hash_count flushed every 1024 hashes
struct ThreadLocalStateDiff {
    /// Modified accounts (address -> new state)
    dirty_accounts: HashMap<[u8; 32], AccountState>,
    /// Flush threshold (number of TX before merging to global state)
    flush_threshold: usize,
    /// Local TX counter
    local_tx_count: u64,
}

impl ThreadLocalStateDiff {
    fn apply_transaction(&mut self, tx: &L2Transaction, global: &Arc<RwLock<GlobalState>>) {
        // Execute TX against local dirty state
        self.local_tx_count += 1;

        // Flush to global state periodically (reduces lock contention)
        if self.local_tx_count as usize % self.flush_threshold == 0 {
            let mut global = global.blocking_write();
            for (addr, state) in self.dirty_accounts.drain() {
                global.accounts.insert(addr, state);
            }
        }
    }
}
```

---

## 10. Rust Implementation Plan

### 10.1 Crate Structure

```
crates/
  q-l2-bridge/          # L1 bridge contract + settlement logic
    src/
      lib.rs            # Bridge contract state machine
      deposit.rs        # L1 -> L2 deposit handling
      withdrawal.rs     # L2 -> L1 withdrawal + Merkle proofs
      fraud_proof.rs    # Fraud proof verification (optimistic mode)
      settlement.rs     # Batch settlement + proof validation

  q-l2-executor/        # L2 state execution engine
    src/
      lib.rs            # State transition function
      state_trie.rs     # Sparse Merkle trie for L2 state
      vm.rs             # VittuaVM execution context for L2
      parallel.rs       # Parallel TX execution engine

  q-l2-sequencer/       # Batch building + transaction ordering
    src/
      lib.rs            # Sequencer main loop
      mempool.rs        # Priority mempool with gas price ordering
      batch_builder.rs  # Batch assembly + state root computation
      rpc.rs            # L2 JSON-RPC endpoint

  q-l2-prover/          # ZK proof generation
    src/
      lib.rs            # Prover coordinator
      stark.rs          # STARK proof generation (q-zk-stark integration)
      snark.rs          # SNARK wrapper (q-zk-snark integration)
      recursive.rs      # Recursive composition (SNARK-of-STARK)
      simd.rs           # SIMD-accelerated FRI evaluation
```

### 10.2 Key Trait Definitions

```rust
/// State transition function: the core of L2 execution
#[async_trait]
pub trait L2StateTransition: Send + Sync {
    /// Execute a batch of transactions, returning the new state root
    async fn execute_batch(
        &self,
        pre_state: &StateRoot,
        transactions: &[L2Transaction],
    ) -> Result<(StateRoot, StateDiff)>;

    /// Verify a single transaction against a given pre-state
    /// (used for fraud proof verification)
    async fn verify_transaction(
        &self,
        pre_state: &StateProof,
        transaction: &L2Transaction,
    ) -> Result<StateRoot>;
}

/// Proof generation engine
#[async_trait]
pub trait L2ProofGenerator: Send + Sync {
    /// Generate a proof for a state transition
    async fn generate_proof(
        &self,
        pre_state_root: &StateRoot,
        post_state_root: &StateRoot,
        transactions: &[L2Transaction],
    ) -> Result<ZkProofData>;

    /// Verify a proof (used by L1 bridge contract)
    fn verify_proof(
        &self,
        proof: &ZkProofData,
        public_inputs: &[StateRoot],
    ) -> Result<bool>;
}
```

---

## 11. Integration with Existing Systems

### 11.1 DEX on L2

The existing `q-dex` AMM pools can operate on L2 with identical constant-product logic. Pool state is maintained in the L2 state trie, and liquidity providers deposit via the L1 bridge. This reduces swap cost by ~100x while maintaining the same AMM invariants.

Cross-layer arbitrage is possible: an arbitrageur observes a price discrepancy between L1 and L2 DEX pools, executes a swap on the cheaper layer, bridges the output, and closes the position on the other layer. The bridge's finality delay (challenge period for optimistic, proof time for ZK) introduces a natural friction that limits risk-free arbitrage to price differences exceeding the bridge cost.

### 11.2 Stablecoin (QUGUSD) on L2

QUGUSD can be bridged to L2 for fast payments. The L2 bridge contract tracks total bridged QUGUSD supply and enforces the invariant that `l2_supply + l1_supply = total_supply`. Minting and burning remain on L1 (where the collateral vault resides), while transfers and payments execute on L2.

### 11.3 Custom Tokens on L2

Any token deployed via VittuaVM on L1 can be bridged to L2. The bridge contract maintains a token registry mapping L1 contract addresses to L2 token IDs. Token transfers on L2 are 10-100x cheaper than L1, making L2 the natural home for high-frequency token trading.

---

## 12. Security Model

### 12.1 Post-Quantum Safety Inheritance

The L2 inherits L1's post-quantum cryptographic guarantees through several mechanisms:

1. **L2 signatures**: Dilithium3 (NIST PQC standard) for L2 transaction signing, with hybrid Ed25519+Dilithium mode for backward compatibility. The signature type is encoded in `L2Signature::Hybrid`.

2. **Proof system**: STARK proofs are inherently post-quantum safe (based on hash functions, not elliptic curves). The recursive SNARK wrapper uses lattice-based assumptions (from `q-zk-snark`) that are believed PQ-safe.

3. **Bridge security**: The L1 bridge contract verifies proofs using QNK L1's post-quantum-safe execution environment. Fraud proofs are verified on-chain with the same PQ signature verification used for L1 transactions.

4. **State commitments**: All state roots use BLAKE3 (hash-based, PQ-safe). Merkle proofs use the same hash function.

### 12.2 Sequencer Failure Modes

| Failure Mode | Impact | Mitigation |
|--------------|--------|------------|
| Sequencer goes offline | L2 halts, no new batches | Forced inclusion via L1: users can submit TX directly to bridge |
| Sequencer censors transactions | Specific users blocked | Forced inclusion + sequencer rotation via governance |
| Sequencer posts invalid state | Incorrect balances | Fraud proofs (optimistic) / ZK verification (ZK mode) |
| Sequencer withholds data | Cannot verify state | DA requirement: state diffs on IPFS before settlement accepted |
| Sequencer double-spends | Conflicting state roots | L1 contract enforces monotonic batch_index, single state root chain |

### 12.3 Economic Security

The sequencer must post a bond of at least 10,000 QUG to the bridge contract. If a fraud proof succeeds, the bond is slashed: 50% to the challenger, 50% burned. This creates an economic incentive for honest sequencing and an incentive for external verifiers to monitor L2 state transitions.

---

## 13. Conclusion

The Q-NarwhalKnight Layer 2 architecture extends the proven optimization patterns from the miner codebase -- dedicated OS threads, core-affinity pinning, channel-based communication, batch processing, and lock-free atomics -- into a high-performance scaling solution. By leveraging existing infrastructure (q-ipfs-storage for DA, q-zk-stark/q-zk-snark for proofs, VittuaVM for execution, libp2p SSE for event relay), the L2 can be built as a focused set of new crates (q-l2-bridge, q-l2-executor, q-l2-sequencer, q-l2-prover) without requiring changes to the L1 consensus layer.

The hybrid rollup design (optimistic for fast deployment, ZK for long-term security) provides a pragmatic migration path. State channels offer an additional scaling primitive for bilateral high-frequency interactions. Together, these L2 constructs position Q-NarwhalKnight to support millions of daily active users while maintaining the post-quantum security guarantees that distinguish the network from classical blockchain platforms.

---

*This document is a technical design review. Implementation timelines, activation heights, and economic parameters are subject to change based on testnet results and community governance.*
