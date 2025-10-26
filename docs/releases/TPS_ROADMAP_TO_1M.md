# TPS Optimization Roadmap: 4K → 200K → 1M TPS

## Current Status

**Benchmark Results:** 4,138 TPS
**Target:** 200,000+ TPS (baseline), 1,000,000+ TPS (with optimizations)
**Gap:** 48x to 241x improvement needed

## Root Cause Analysis

### Problem: Consensus Bypass
The API server is **NOT** using the DAG-Knight/Narwhal/Bullshark consensus at all!

**Current Flow (4K TPS):**
```
Transaction → API Handler → HashMap update → Simple tx_pool → DONE
```

**Required Flow (200K+ TPS):**
```
Transaction → ProductionMempool → Narwhal Batching → DAG Vertices →
Bullshark Finality → Parallel Processing → DONE
```

### Evidence
**File:** `crates/q-api-server/src/handlers.rs:419-423`
```rust
// Add to transaction pool
{
    let mut tx_pool = state.tx_pool.write().await;
    tx_pool.insert(tx_hash, signed_transaction.clone());  // Just a HashMap!
}
```

**File:** `crates/q-api-server/src/lib.rs`
```rust
pub narwhal_core: Option<Arc<NarwhalCore>>,  // Set to None!
```

## Available High-Performance Components

### ✅ Already Implemented
1. **DAG-Knight Consensus** - `crates/q-dag-knight/`
   - Parallel vertex processing
   - Zero-message complexity BFT
   - Quantum VDF for randomness

2. **Narwhal Mempool** - `crates/q-narwhal-core/production_mempool.rs`
   - Reliable broadcast (Bracha's protocol)
   - Tor-based p2p
   - Transaction batching
   - Anti-spam protection

3. **Parallel Workers** - `crates/q-narwhal-core/parallel_workers.rs`
   - Multi-threaded vertex processing
   - Lock-free data structures

4. **SIMD Crypto** - `crates/q-crypto-simd/`
   - Vectorized signature verification
   - Batch hash computation

5. **Kernel I/O** - `crates/q-kernel-io/`
   - io_uring support
   - NUMA-aware memory allocation
   - Zero-copy networking

## Optimization Phases

### Phase 1: Enable Consensus (→ 50K TPS)
**Time:** 2-4 hours
**Impact:** 12x improvement

1. Initialize `ProductionMempool` in API server
2. Route transactions through mempool
3. Connect mempool to DAG-Knight
4. Enable basic vertex processing

**Changes needed:**
- `crates/q-api-server/src/lib.rs`: Initialize NarwhalCore
- `crates/q-api-server/src/handlers.rs`: Route through mempool
- `crates/q-api-server/src/main.rs`: Start consensus engine

### Phase 2: Parallel Processing (→ 200K TPS)
**Time:** 1-2 hours
**Impact:** 4x improvement

1. Enable parallel worker threads (16-32 workers)
2. Activate batch processing in mempool
3. Use lock-free DAG vertex storage
4. Optimize RocksDB settings for throughput

**Changes needed:**
- Enable `parallel_workers` in Narwhal
- Configure `num_workers` based on CPU cores
- Tune RocksDB for write-heavy workload

### Phase 3: SIMD Optimizations (→ 500K TPS)
**Time:** 2-3 hours
**Impact:** 2.5x improvement

1. Enable SIMD signature verification
2. Batch cryptographic operations
3. Vectorize hash computations
4. Use AVX2/AVX-512 instructions

**Changes needed:**
- Link `q-crypto-simd` crate
- Enable `target-cpu=native` in build flags
- Batch verify signatures in groups of 64

### Phase 4: Kernel I/O (→ 1M TPS)
**Time:** 3-4 hours
**Impact:** 2x improvement

1. Replace tokio with io_uring for networking
2. Enable zero-copy packet processing
3. Use NUMA-aware memory allocation
4. Optimize syscall batching

**Changes needed:**
- Integrate `q-kernel-io` crate
- Configure io_uring with SQ/CQ depth
- Pin worker threads to NUMA nodes

## Quick Start: Phase 1 Implementation

### Step 1: Update AppState
```rust
// crates/q-api-server/src/lib.rs
pub struct AppState {
    // ... existing fields ...
    pub mempool: Option<Arc<ProductionMempool>>,  // Add this
    pub dag_consensus: Option<Arc<DAGKnightConsensus>>,  // Add this
}
```

### Step 2: Initialize in main.rs
```rust
// Initialize mempool
let mempool_config = MempoolConfig {
    max_transactions: 1_000_000,
    max_age: Duration::from_secs(300),
    min_fee_per_byte: 1,
};

let tor_client = Arc::new(SimpleTorClient::new().await?);
let mempool = Arc::new(
    ProductionMempool::new(mempool_config, tor_client, Phase::Phase0).await?
);

// Initialize DAG-Knight
let dag_consensus = Arc::new(
    DAGKnightConsensus::new(node_id, 16).await?  // 16 workers
);

// Add to AppState
state.mempool = Some(mempool);
state.dag_consensus = Some(dag_consensus);
```

### Step 3: Update Transaction Handler
```rust
// crates/q-api-server/src/handlers.rs
pub async fn send_transaction(...) -> Result<...> {
    // ... existing validation ...

    // Route through consensus instead of HashMap
    if let Some(mempool) = &state.mempool {
        mempool.add_transaction(signed_transaction, None).await?;

        // Mempool automatically batches and creates DAG vertices
        // DAG-Knight processes them in parallel
    }

    // ...
}
```

## Expected Results by Phase

| Phase | TPS | Latency (p99) | CPU Usage | Description |
|-------|-----|---------------|-----------|-------------|
| Current | 4K | 104ms | 22% | HashMap storage only |
| Phase 1 | 50K | 80ms | 45% | Basic consensus enabled |
| Phase 2 | 200K | 50ms | 75% | Parallel processing |
| Phase 3 | 500K | 30ms | 85% | SIMD crypto |
| Phase 4 | 1M+ | 15ms | 90% | io_uring kernel I/O |

## Monitoring & Validation

### Metrics to Track
1. **Mempool size** - Should stay small (< 10K txs)
2. **DAG vertex count** - Should grow linearly
3. **Worker utilization** - All cores should be active
4. **Batch size** - Target 10K transactions per batch
5. **Finality latency** - < 2 seconds for Bullshark

### Benchmark Commands
```bash
# Phase 1 validation
cargo run --release --bin tps-benchmark

# Expected: 50K+ TPS after Phase 1
# Expected: 200K+ TPS after Phase 2
# Expected: 1M+ TPS after Phase 4
```

## Next Steps

1. **Immediate:** Implement Phase 1 consensus integration
2. **Short-term:** Enable parallel workers (Phase 2)
3. **Medium-term:** SIMD optimizations (Phase 3)
4. **Long-term:** Kernel I/O for 1M+ TPS (Phase 4)

## References

- DAG-Knight paper: Zero-message BFT consensus
- Narwhal/Tusk paper: 130K+ TPS demonstrated
- Bullshark paper: 125K+ TPS demonstrated
- Q-NarwhalKnight combines all three for 200K+ baseline

---

**Note:** The consensus components are already implemented! We just need to wire them together in the API server.
