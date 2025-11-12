# Balance Consensus Not Processing Locally Produced Blocks

**Date**: 2025-11-06
**Status**: 🔴 **CRITICAL BUG - Locally produced blocks bypass balance consensus**

---

## 🔴 **PROBLEM**

Mining rewards show 0 balance despite actively producing blocks with coinbase transactions.

**Symptoms:**
- Block producer actively creating blocks (height 1050+)
- Coinbase transactions being created: "💰 Created 101 coinbase transactions"
- Balance queries all return 0: "has 0 QUG"
- No balance consensus processing logs for locally produced blocks

**Evidence:**
```
2025-11-06T08:29:31.910447Z  INFO q_api_server::block_producer: 🏗️  Producing block: height=1050, solutions=100
2025-11-06T08:29:31.910689Z  INFO q_api_server::block_producer: 🎉 Producer #0 created block at height 1050
2025-11-06T08:29:44.046295Z  INFO q_api_server::block_producer: 💰 Created 101 coinbase transactions: 1000000 dev fee, 990000 miner rewards
2025-11-06T08:29:42.469948Z  INFO q_api_server::handlers: 🔐 Authenticated balance query: 5ae7f344fd6b0877 has 0 QUG
```

NO logs showing:
- `add_balance_tx`
- `Processed.*balance`
- Balance consensus processing local blocks

---

## 🔍 **ROOT CAUSE**

**Balance consensus only processes blocks received via gossipsub, NOT locally produced blocks!**

### Architecture Gap:

1. **When receiving blocks** (`main.rs:2612`):
   ```rust
   // ✅ Balance consensus IS called
   let updates = match balance_engine.process_block_mining_rewards_tx(&tx, &block).await {
       Ok(updates) => { /* Process updates */ }
   }
   ```

2. **When producing blocks** (`block_producer.rs`):
   ```rust
   // ❌ Balance consensus NOT called
   let block = QBlock {
       transactions: coinbase_transactions, // Created but never processed!
       balance_updates: vec![], // Empty!
   };
   // Block saved to storage but balance consensus never invoked
   ```

### The Flow:

```
Producer creates block → Save to RocksDB → Broadcast to network
                              ↓
                        ❌ MISSING: Process balance consensus

Network receives block → Process via gossipsub → ✅ Balance consensus called
```

**Result**: Only blocks received from OTHER nodes update balances. Your own blocks don't!

---

## ✅ **THE FIX**

### **Option 1: Process Produced Blocks Through Balance Consensus (RECOMMENDED)**

After the block producer creates a block, immediately process it through balance consensus:

**Location**: `crates/q-api-server/src/main.rs` (in the block production task)

```rust
// After block is created and saved
for block in produced_blocks {
    // Save block to storage
    storage.save_qblock(&block).await?;

    // ✅ NEW: Process balance consensus for locally produced blocks
    let updates = balance_engine.process_block_mining_rewards(
        &storage,
        &block
    ).await?;

    debug!("💰 Processed {} balance updates for locally produced block {}",
           updates.len(), block.height);

    // Broadcast to network
    network.publish_block(&block).await?;
}
```

### **Option 2: Add Balance Consensus to Block Producer**

Integrate balance consensus directly into block producer:

**Location**: `crates/q-api-server/src/block_producer.rs`

```rust
pub async fn produce_block(
    &mut,
    solutions: Vec<MiningSolution>,
    storage: &QStorage,
    balance_engine: &BalanceConsensusEngine, // ✅ Add parameter
) -> Result<QBlock> {
    // Create block with coinbase transactions
    let block = QBlock { /* ... */ };

    // ✅ Process balance consensus immediately
    let updates = balance_engine.process_block_mining_rewards(
        storage,
        &block
    ).await?;

    Ok(block)
}
```

### **Option 3: Self-Broadcast (Current Workaround)**

Make the node receive its own blocks via gossipsub so they get processed:

```rust
// After creating block
network.publish_block(&block).await?;

// ✅ Subscribe to own blocks (gossipsub will deliver back to us)
// Balance consensus will process when received
```

**Downside**: Adds unnecessary network round-trip

---

## 📊 **VERIFICATION**

After implementing the fix, you should see:

```bash
journalctl -u q-api-server -f | grep -E "balance|💰"
```

Expected output:
```
INFO q_storage: 💰 Processed 101 balance updates for block 1051
INFO q_api_server::handlers: 🔐 Authenticated balance query: ... has 99000 QUG
```

Check balance:
```bash
curl -s http://localhost:8080/api/v1/node/status | jq .data.balance
# Should show non-zero value
```

---

## 🐛 **WHY THIS WASN'T CAUGHT**

1. **Testnet had multiple nodes** - Other nodes' blocks were processed correctly
2. **Solo mining scenario** - Only discovered when one node produces ALL blocks
3. **Balance consensus is new** - Added recently, integration not complete
4. **Logs were confusing** - "Created coinbase transactions" looked successful

---

## 💡 **IMMEDIATE WORKAROUND**

Start a second node that connects to this one. When blocks are broadcast and received back, balance consensus will process them!

```bash
# On another machine or port
Q_DB_PATH=./data-node2 Q_P2P_PORT=9002 ./q-api-server --port 8081
```

This works because:
- Node 1 produces block → broadcasts
- Node 2 receives block → processes balance consensus
- Node 1 receives block from node 2 → processes balance consensus ✅

---

## 📝 **RECOMMENDED FIX (Option 1)**

**File**: `crates/q-api-server/src/main.rs`
**Location**: After blocks are produced (around line 3500-3600 in block production task)

**Add**:
```rust
// Process balance consensus for locally produced blocks
if let Err(e) = balance_engine_clone.process_block_mining_rewards(
    &app_state_producer.storage_engine,
    &block
).await {
    error!("Failed to process balance consensus for produced block {}: {}",
           block.height, e);
}
```

This ensures parity between received blocks and produced blocks.

---

**Next Steps**: Implement Option 1 in v0.9.28-beta
