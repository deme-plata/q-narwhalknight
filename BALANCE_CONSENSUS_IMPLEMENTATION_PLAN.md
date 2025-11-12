# Balance Consensus Implementation Plan

**Version**: v0.8.0-beta
**Priority**: 🚨 CRITICAL - Mainnet Blocker
**Scope**: Single-node system (NO light client features)
**Timeline**: 2-3 weeks development + 1-2 weeks testing

---

## Executive Summary

**Problem**: Mining rewards are processed LOCALLY per node, causing balance divergence across the network.

**Solution**: Implement deterministic balance updates triggered by block reception via gossipsub, ensuring ALL nodes process identical state transitions.

**Exclusions**:
- ❌ Light client state proofs (not needed for single-node system)
- ❌ Cross-shard transfers (no sharding)
- ❌ Advanced state pruning (future optimization)

---

## Current Architecture (BROKEN)

```
Mining Solution Submitted
   ↓
[Mining Processor] → Update LOCAL balance ❌
   ↓
Block Produced
   ↓
Gossipsub Broadcast → Other nodes receive block
   ↓
Other nodes: NO balance update ❌
   ↓
RESULT: State divergence across network
```

**Evidence**:
```bash
# Server Alpha (miner)
curl http://localhost:8080/api/v1/balance/qnke9578... → 5000 coins ✅

# Server Beta (bootstrap node)
curl http://localhost:8080/api/v1/balance/qnke9578... → 0 coins ❌
```

---

## Target Architecture (FIXED)

```
Mining Solution Submitted
   ↓
[Block Producer] → Create block with solutions
   ↓
Gossipsub Broadcast → ALL nodes receive block
   ↓
[Balance Consensus Engine] → Process mining rewards
   ↓
Update balances deterministically ✅
   ↓
RESULT: Identical state across ALL nodes
```

---

## Implementation Plan

### Phase 1: Core Balance Consensus Engine (Week 1)

**File**: `crates/q-storage/src/balance_consensus.rs` (NEW)

**Components**:

1. **BalanceConsensusEngine** - Deterministic balance processor
   ```rust
   pub struct BalanceConsensusEngine {
       genesis_timestamp: u64,
       dev_wallet: String,
       processed_blocks: RwLock<HashMap<[u8; 32], bool>>, // Prevent double-processing
   }
   ```

2. **Key Functions**:
   - `process_block_mining_rewards()` - Process ALL solutions in a block
   - `calculate_block_reward()` - Deterministic reward calculation (MUST match mining processor)
   - `verify_solution_for_block()` - Validate solutions meet difficulty

3. **Error Handling**:
   ```rust
   pub enum BalanceConsensusError {
       AlreadyProcessed([u8; 32]),  // Block already processed (normal)
       InvalidSolution { block_height, solution_index },  // Reject block
       ZeroReward(u64),  // Halving complete
       Storage(StorageError),  // Database error
       BatchOperation(String),  // Atomic update failed
   }
   ```

**Deliverables**:
- [ ] `crates/q-storage/src/balance_consensus.rs` implemented
- [ ] Unit tests for deterministic reward calculation
- [ ] Unit tests for double-processing prevention
- [ ] Integration tests with RocksDB storage

---

### Phase 2: Gossipsub Integration (Week 1)

**File**: `crates/q-api-server/src/main.rs` (MODIFY)

**Changes**:

1. **Initialize Engine on Startup**:
   ```rust
   let balance_engine = Arc::new(BalanceConsensusEngine::new(
       GENESIS_TIMESTAMP,
       DEV_WALLET_ADDRESS.to_string()
   ));
   ```

2. **Add to Gossipsub Block Handler** (around line 1789):
   ```rust
   GossipsubEvent::Message { message, .. } => {
       if message.topic == blocks_topic.hash() {
           match serde_json::from_slice::<QBlock>(&message.data) {
               Ok(block) => {
                   // ✅ CONSENSUS-CRITICAL: Process mining rewards BEFORE storing
                   match balance_engine.process_block_mining_rewards(&*storage, &block).await {
                       Ok(updates) => {
                           info!("💰 Processed {} balance updates for block {}",
                                 updates.len(), block.header.height);
                       }
                       Err(BalanceConsensusError::AlreadyProcessed(_)) => {
                           debug!("Block {} already processed", block.header.height);
                       }
                       Err(e) => {
                           error!("❌ CRITICAL: Balance consensus failed: {:?}", e);
                           continue; // REJECT BLOCK
                       }
                   }

                   // Store block AFTER successful state update
                   storage.store_block(block).await?;
               }
               Err(e) => error!("Failed to deserialize block: {:?}", e);
           }
       }
   }
   ```

**Deliverables**:
- [ ] Gossipsub handler modified
- [ ] Batch sync handler modified (same logic)
- [ ] Integration tests with mock gossipsub messages
- [ ] Verify block rejection on consensus failure

---

### Phase 3: Remove Local Balance Updates (Week 2)

**File**: `crates/q-api-server/src/block_producer.rs` (MODIFY)

**Changes**:

1. **Remove Balance Updates from Mining Processor** (around line 300-400):
   ```rust
   // ❌ DELETE THIS:
   // let current_balance = storage.get_balance(&miner_address).await?;
   // let new_balance = current_balance + miner_reward;
   // storage.set_balance(&miner_address, new_balance).await?;

   // ✅ BALANCES NOW PROCESSED VIA GOSSIPSUB WHEN BLOCK IS RECEIVED
   info!("Mining solution added to block {} (balance update via consensus)",
         block_header.height);
   ```

2. **Update Mining Processor Tests**:
   - Remove assertions that check local balance updates
   - Add comments explaining consensus-based updates

**Deliverables**:
- [ ] Local balance updates removed from block producer
- [ ] Dev fee processing removed from block producer
- [ ] Mining processor tests updated
- [ ] Verify balances ONLY update via gossipsub

---

### Phase 4: Turbo Sync Integration (Week 2)

**File**: `crates/q-storage/src/turbo_sync.rs` (MODIFY)

**Changes**:

1. **Add Balance Processing to Sync Loop**:
   ```rust
   pub async fn sync_to_height(
       &self,
       target_height: u64,
       balance_engine: &BalanceConsensusEngine,
   ) -> Result<SyncStats> {
       for height in current_height..=target_height {
           let blocks = fetch_batch_from_peer(height, batch_size).await?;

           for block in blocks {
               // Process mining rewards for historical blocks
               match balance_engine.process_block_mining_rewards(storage, &block).await {
                   Ok(updates) => {
                       stats.balance_updates += updates.len();
                   }
                   Err(BalanceConsensusError::AlreadyProcessed(_)) => {
                       // Skip already processed blocks
                   }
                   Err(e) => {
                       error!("Failed to process block {}: {:?}", block.header.height, e);
                       return Err(e.into());
                   }
               }

               storage.store_block(block).await?;
           }
       }

       Ok(stats)
   }
   ```

2. **Add Sync Stats Tracking**:
   ```rust
   pub struct SyncStats {
       pub blocks_synced: u64,
       pub balance_updates: usize,
       pub start_time: Instant,
   }
   ```

**Deliverables**:
- [ ] Turbo sync processes balances for historical blocks
- [ ] Sync stats include balance update counts
- [ ] Integration tests with 1000+ block sync
- [ ] Verify new nodes catch up correctly

---

### Phase 5: Testing & Validation (Week 3)

**Test Suite**: `tests/balance_consensus_tests.rs` (NEW)

**Test Coverage**:

1. **Determinism Test**:
   ```rust
   #[tokio::test]
   async fn test_balance_consensus_determinism() {
       // Two nodes process same block → identical balances
       let node1 = TestNode::new().await;
       let node2 = TestNode::new().await;
       let block = create_test_block_with_solutions(50);

       node1.process_block(&block).await.unwrap();
       node2.process_block(&block).await.unwrap();

       assert_eq!(node1.get_all_balances().await, node2.get_all_balances().await);
   }
   ```

2. **Double-Spend Prevention**:
   ```rust
   #[tokio::test]
   async fn test_double_processing_prevention() {
       let node = TestNode::new().await;
       let block = create_test_block_with_solutions(1);

       // First processing succeeds
       assert!(node.process_block(&block).await.is_ok());

       // Second processing fails gracefully
       let result = node.process_block(&block).await;
       assert!(matches!(result, Err(BalanceConsensusError::AlreadyProcessed(_))));
   }
   ```

3. **Network Consensus Simulation**:
   ```rust
   #[tokio::test]
   async fn test_5_node_network_consensus() {
       let nodes = vec![
           TestNode::new().await,
           TestNode::new().await,
           TestNode::new().await,
           TestNode::new().await,
           TestNode::new().await,
       ];

       // Process 100 blocks across all nodes
       for height in 1..=100 {
           let block = create_random_block(height);
           for node in &nodes {
               node.process_block(&block).await.unwrap();
           }

           // Verify all nodes have identical state
           assert_consistent_balances(&nodes).await;
       }
   }
   ```

4. **Turbo Sync Recovery**:
   ```rust
   #[tokio::test]
   async fn test_turbo_sync_balance_recovery() {
       // Node 1 produces 1000 blocks
       let node1 = TestNode::new().await;
       produce_blocks(&node1, 1000).await;

       // Node 2 syncs from genesis
       let node2 = TestNode::new().await;
       node2.turbo_sync_from(&node1, 1000).await.unwrap();

       // Verify identical balances after sync
       assert_eq!(node1.get_all_balances().await, node2.get_all_balances().await);
   }
   ```

**Deliverables**:
- [ ] 10+ comprehensive tests covering all scenarios
- [ ] Benchmark tests for 10k block processing
- [ ] Chaos testing (network partitions, crashes)
- [ ] Performance profiling (ensure <1ms overhead per block)

---

### Phase 6: Deployment (Week 4)

**Rollout Strategy**:

1. **Testnet Phase 4 Deployment**:
   ```bash
   # Build v0.8.0-beta with balance consensus
   timeout 36000 cargo build --release --package q-api-server

   # Deploy to Server Beta (bootstrap node)
   sudo systemctl stop q-api-server
   sudo cp target/release/q-api-server /usr/local/bin/q-api-server-v0.8.0-beta
   sudo ln -sf /usr/local/bin/q-api-server-v0.8.0-beta /usr/local/bin/q-api-server

   # CRITICAL: Reset database for clean state
   rm -rf ./data-mine3

   # Restart with monitoring
   sudo systemctl start q-api-server
   journalctl -u q-api-server -f | grep -E "Balance|CRITICAL"
   ```

2. **Monitoring Dashboard**:
   - Balance updates processed per block
   - Consensus failures (should be ZERO)
   - Block processing latency
   - Network-wide balance consistency checks

3. **Validation Tests**:
   ```bash
   # Test 1: Mine on Server Alpha
   # Verify balance appears on BOTH Server Alpha AND Server Beta

   # Test 2: Turbo sync new node
   # Verify balances match bootstrap node

   # Test 3: 24-hour stability test
   # Monitor for any consensus failures
   ```

**Deliverables**:
- [ ] v0.8.0-beta deployed to testnet-phase4
- [ ] 24-hour stability monitoring report
- [ ] Balance consistency verified across 3+ nodes
- [ ] No consensus failures observed

---

## Files to Create/Modify

### New Files:
1. `crates/q-storage/src/balance_consensus.rs` - Core consensus engine (~300 lines)
2. `tests/balance_consensus_tests.rs` - Comprehensive test suite (~500 lines)
3. `BALANCE_CONSENSUS_IMPLEMENTATION_PLAN.md` - This document

### Modified Files:
1. `crates/q-api-server/src/main.rs` - Gossipsub integration (~50 lines)
2. `crates/q-api-server/src/block_producer.rs` - Remove local updates (~50 lines deleted)
3. `crates/q-storage/src/turbo_sync.rs` - Historical balance processing (~100 lines)
4. `crates/q-storage/src/lib.rs` - Export new consensus module (~5 lines)

**Total LOC**: ~1000 lines (300 new engine + 500 tests + 200 modifications)

---

## Risk Mitigation

### Risk 1: Reward Calculation Mismatch

**Problem**: If `BalanceConsensusEngine::calculate_block_reward()` differs from mining processor, nodes diverge.

**Mitigation**:
- Extract reward calculation into shared utility function
- Both mining processor AND consensus engine use same function
- Add determinism tests comparing both implementations

### Risk 2: Block Double-Processing

**Problem**: Gossipsub may deliver same block multiple times.

**Mitigation**:
- Track processed blocks in `processed_blocks` HashMap
- Return `Ok` on second processing (idempotent)
- Add test for double-processing scenario

### Risk 3: Atomic Update Failure

**Problem**: Balance update fails mid-block (50 solutions processed, 50 remaining).

**Mitigation**:
- Use RocksDB batch writes (atomic)
- If batch fails, entire block is rejected
- Add rollback test for partial updates

### Risk 4: Turbo Sync Performance

**Problem**: Processing 100k historical blocks may take too long.

**Mitigation**:
- Benchmark 10k block processing (target <10 seconds)
- Use batch operations for efficiency
- Add progress reporting every 1000 blocks

---

## Success Criteria

### Testnet Phase 4:
- ✅ ALL nodes show identical balances for same wallet
- ✅ Mining rewards appear on ALL nodes, not just miner
- ✅ New nodes sync balances correctly via Turbo Sync
- ✅ Zero consensus failures in 24-hour stability test
- ✅ <1ms overhead per block for balance processing

### Mainnet Launch:
- ✅ 1-week testnet run with 10+ nodes, zero divergence
- ✅ Load testing with 1000+ blocks/hour
- ✅ Exchange integration testing (balance queries)
- ✅ Block explorer integration (transaction history)

---

## Exclusions (NOT in Scope)

The following features from `critical11.rs` are EXCLUDED:

1. ❌ **Light Client State Proofs** - Not needed for single-node system
2. ❌ **State Root in Block Header** - Future optimization (Phase 2)
3. ❌ **State Merkle Trees** - Future optimization (Phase 2)
4. ❌ **Cross-Shard Transfers** - No sharding in current design
5. ❌ **Advanced State Pruning** - Future optimization
6. ❌ **Formal Verification** - Research project

**Rationale**: Focus on MINIMUM viable fix for mainnet launch. Advanced features can be added post-mainnet.

---

## Timeline

**Week 1**:
- [ ] Implement `BalanceConsensusEngine` (3 days)
- [ ] Integrate with gossipsub (2 days)

**Week 2**:
- [ ] Remove local balance updates (1 day)
- [ ] Integrate with Turbo Sync (2 days)
- [ ] Write comprehensive tests (2 days)

**Week 3**:
- [ ] Test on testnet-phase4 (3 days)
- [ ] 24-hour stability test (1 day)
- [ ] Performance benchmarking (1 day)

**Week 4**:
- [ ] Multi-node testing (3 days)
- [ ] Final validation (2 days)
- [ ] **READY FOR MAINNET** ✅

---

## Next Steps

1. **Review this plan** - Confirm approach aligns with vision
2. **Begin Week 1 implementation** - Start with `balance_consensus.rs`
3. **Create feature branch**: `git checkout -b feature/balance-consensus-v0.8.0`
4. **Track progress** via TodoWrite tool

---

**Prepared by**: Claude Code (Server Beta)
**Version**: Implementation Plan v1.0
**Target Release**: v0.8.0-beta
**Mainnet Impact**: 🚨 CRITICAL - Unblocks mainnet launch

---

## Appendix: Key Code Locations

**Mining Processor** (local balance updates to REMOVE):
- `crates/q-api-server/src/block_producer.rs:300-400`

**Gossipsub Block Handler** (where to ADD consensus engine):
- `crates/q-api-server/src/main.rs:1789-1825`

**Turbo Sync** (where to ADD historical balance processing):
- `crates/q-storage/src/turbo_sync.rs:100-200`

**Reward Calculation** (MUST be deterministic):
- Extract from `crates/q-mining/src/lib.rs:calculate_mining_reward()`
- Share between mining processor AND consensus engine
