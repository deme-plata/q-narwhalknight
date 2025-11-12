# Mining Reward Architecture - Q-NarwhalKnight v0.7.3-beta

## Overview

Mining rewards in Q-NarwhalKnight testnet Phase 3 are processed **locally** on each node and do NOT automatically synchronize across the network.

## Architecture

### Mining Submission Flow

```
┌─────────────────────────────────────────────────────────────┐
│ MINER                                                       │
│ Solves VDF puzzle, finds valid nonce                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ POST /api/v1/mining/submit
┌─────────────────────────────────────────────────────────────┐
│ SERVER ALPHA (161.35.219.10:8080)                          │
│                                                             │
│ 1. Validate solution (handlers.rs:3990-3992)               │
│ 2. Queue to mining_submission_tx (handlers.rs:4095)        │
│ 3. Background processor updates balance (main.rs:2585)     │
│ 4. Solution added to pending_solutions queue               │
│    (block_producer.rs:76 - Arc<SegQueue>)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ Every ~2 seconds
┌─────────────────────────────────────────────────────────────┐
│ BLOCK PRODUCTION (Server Alpha)                            │
│                                                             │
│ 1. Drain pending_solutions queue (block_producer.rs:220)   │
│ 2. Create block with mining_solutions array                │
│ 3. Broadcast block via gossipsub                           │
│    Topic: /qnk/testnet-phase3/blocks                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼ P2P gossipsub broadcast
┌─────────────────────────────────────────────────────────────┐
│ SERVER BETA (185.182.185.227:8080)                         │
│                                                             │
│ 1. Receives block via gossipsub                            │
│ 2. Validates block structure and VDF proof                 │
│ 3. Stores block in RocksDB (kv.rs:503)                     │
│ 4. ❌ DOES NOT re-process mining_solutions                 │
│ 5. ❌ DOES NOT update balances from received block         │
└─────────────────────────────────────────────────────────────┘
```

## Key Finding

**Mining rewards are LOCAL to the node that accepts the submission.**

- ✅ Server Alpha wallet qnke9578fdf... likely has balance on Server Alpha
- ❌ Server Beta wallet qnke9578fdf... has ZERO balance on Server Beta
- ✅ This is EXPECTED BEHAVIOR for testnet Phase 3

## Code Locations

### 1. Mining Submission Handler
**File**: `crates/q-api-server/src/handlers.rs:3946-4129`

```rust
pub async fn submit_mining_solution(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MiningSolutionRequest>,
) -> Result<Json<ApiResponse<MiningSolutionResponse>>, StatusCode> {
    // Validate solution
    if !verify_mining_difficulty(&hash, &difficulty_target) {
        return Ok(Json(ApiResponse::error("Solution does not meet difficulty target")));
    }

    // Queue to background processor (LOCAL processing)
    if let Some(tx) = &state.mining_submission_tx {
        let submission = crate::MiningSubmission { ... };
        match tx.send(submission) {
            Ok(_) => { /* Success - queued locally */ }
            ...
        }
    }
}
```

### 2. Background Mining Processor
**File**: `crates/q-api-server/src/main.rs:2569-2641`

```rust
let mut batch_buffer: Vec<q_api_server::MiningSubmission> = Vec::with_capacity(500);

while let Some(submission) = mining_rx.recv().await {
    batch_buffer.push(submission);
    
    // Process batch every 500 submissions OR 20ms
    if batch_buffer.len() >= 500 || last_batch_process.elapsed().as_millis() >= 20 {
        // Update balances in LOCAL database
        for submission in &batch_buffer {
            // Balance update happens HERE - LOCAL ONLY
            storage.update_balance(
                submission.miner_address_str.clone(),
                miner_reward,
                ChangeReason::MiningReward
            ).await;
        }
    }
}
```

### 3. Block Producer
**File**: `crates/q-api-server/src/block_producer.rs:215-242`

```rust
pub async fn produce_block(&mut self) -> Option<QBlock> {
    // Drain LOCAL pending_solutions queue
    let mut solutions = Vec::with_capacity(self.config.max_solutions_per_block);
    
    while solutions.len() < self.config.max_solutions_per_block {
        // LOCK-FREE drain from LOCAL queue
        if let Some(solution) = self.pending_solutions.pop() {
            solutions.push(solution);
        } else {
            break;
        }
    }
    
    // Block is created with these LOCAL solutions
    // Solutions are broadcast in the block, but NOT re-processed by receivers
}
```

## Implications

### For Testnet Phase 3

1. **Mining directly to Server Beta** (bootstrap node):
   - ✅ Balance visible on Server Beta
   - ✅ Can be queried via API on Server Beta
   - ✅ Wallet can spend on Server Beta

2. **Mining to Server Alpha**:
   - ✅ Balance visible on Server Alpha
   - ❌ NOT visible on Server Beta
   - ❌ Cannot query from Server Beta API

3. **Network behavior**:
   - ✅ Blocks propagate correctly
   - ✅ Mining solutions are included in blocks
   - ❌ Balances do NOT synchronize
   - ❌ Each node maintains independent state

### For Mainnet

**This architecture MUST change before mainnet launch.**

Mainnet requires **consensus on balances** across all nodes:
- ✅ All nodes must agree on account balances
- ✅ Mining solutions must be re-processed by all validators
- ✅ UTXO or account state must be synchronized
- ✅ Blockchain state must be deterministic

## Recommended Solutions

### Short-term (Testnet Phase 3)
1. **Documentation**: Inform users to query the SAME node they mined to
2. **Mining pool**: Direct all miners to Server Beta (bootstrap node)
3. **Accept limitation**: Phase 3 is for testing RocksDB persistence, not consensus

### Long-term (Mainnet)
1. **State synchronization**: Re-process mining_solutions when receiving blocks
2. **UTXO model**: Track unspent transaction outputs with cryptographic proofs
3. **Account model**: Maintain global state tree synchronized via consensus
4. **Validator consensus**: All validators must agree on balance changes

## Current Status

**v0.7.3-beta Testnet Phase 3**:
- ✅ RocksDB persistence verified (blocks survive restart)
- ✅ Network connectivity working (7 peers)
- ✅ Block propagation working (gossipsub)
- ⚠️ Balance synchronization NOT implemented (expected)
- 🎯 Focus: Test RocksDB fixes before mainnet launch

---

**Generated**: 2025-11-02  
**Version**: v0.7.3-beta  
**Network**: testnet-phase3
