# Local Mining on User Nodes: Technical Review & Implementation Plan

**Version**: v1.0.84-beta
**Date**: 2025-12-02
**Status**: Critical Architecture Gap Identified

---

## Executive Summary

After extensive testing, we've identified that **local mining rewards are not being propagated to the network**. The miner finds valid solutions, but:

1. Solutions are accepted locally by the Docker test node
2. Mined blocks exist only on the local node's database
3. **Blocks are NOT propagated via libp2p to the bootstrap node**
4. Balance updates are local-only and don't reach consensus
5. When the node syncs, locally-mined blocks get overwritten by network blocks

**Root Cause**: The current architecture relies on HTTP API for block submission, but blocks need to be propagated via libp2p gossipsub to be recognized by the network.

---

## Current Architecture (Broken)

```
┌─────────────────┐     HTTP POST      ┌─────────────────┐
│   q-miner       │ ──────────────────►│  Local Node     │
│   (CPU mining)  │   /submit_solution │  (Docker)       │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                │ Block saved locally
                                                │ Balance updated locally
                                                ▼
                                       ┌─────────────────┐
                                       │  Local RocksDB  │
                                       │  (isolated)     │
                                       └─────────────────┘
                                                │
                                                │ ❌ NO P2P PROPAGATION
                                                │
                                       ┌─────────────────┐
                                       │  Bootstrap Node │
                                       │  (Server Beta)  │ ◄── Network doesn't see the block!
                                       └─────────────────┘
```

### What Happens Now:

1. **Miner submits solution** to local node via HTTP
2. **Local node accepts** and creates a block
3. **Block is stored** in local RocksDB
4. **Balance is updated** in local balance store
5. **NO gossipsub message** is sent to network
6. **Network continues** with its own chain
7. **Local node eventually syncs** from network, overwriting local blocks
8. **Local balance reverts** to network-agreed balance

---

## Required Architecture (Correct)

```
┌─────────────────┐     HTTP POST      ┌─────────────────┐
│   q-miner       │ ──────────────────►│  Local Node     │
│   (CPU mining)  │   /submit_solution │  (User's Node)  │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                │ 1. Block created
                                                │ 2. Balance updated
                                                │ 3. Block stored
                                                ▼
                                       ┌─────────────────┐
                                       │ libp2p Network  │
                                       │ Manager         │
                                       └────────┬────────┘
                                                │
                                                │ 4. gossipsub.publish(
                                                │      "/qnk/testnet/blocks",
                                                │      SerializedBlock
                                                │    )
                                                ▼
                              ┌─────────────────────────────────┐
                              │        Gossipsub Network        │
                              │                                 │
                    ┌─────────┴─────────┐         ┌─────────────┴─────────┐
                    │   Bootstrap Node  │         │    Other Peers        │
                    │   (Server Beta)   │         │    (Full Network)     │
                    └───────────────────┘         └───────────────────────┘
                              │                             │
                              │ 5. Receive block            │
                              │ 6. Validate block           │
                              │ 7. Add to DAG               │
                              │ 8. Update balances          │
                              │ 9. Reach consensus          │
                              ▼                             ▼
                    ┌───────────────────┐         ┌───────────────────────┐
                    │ Network Consensus │ ◄──────►│ All nodes agree on    │
                    │ (DAG-Knight)      │         │ miner's block!        │
                    └───────────────────┘         └───────────────────────┘
```

---

## Implementation Requirements

### 1. Block Production Must Trigger P2P Propagation

**File**: `crates/q-api-server/src/block_producer.rs`

Current flow:
```rust
// Current (broken)
async fn produce_block(solution: MiningSolution) -> Result<Block> {
    let block = create_block(solution)?;
    storage.store_block(&block).await?;  // Local only!
    balance_store.update(miner_address, reward)?;  // Local only!
    Ok(block)
}
```

Required flow:
```rust
// Required (correct)
async fn produce_block(
    solution: MiningSolution,
    network_manager: Arc<UnifiedNetworkManager>
) -> Result<Block> {
    let block = create_block(solution)?;

    // Store locally first
    storage.store_block(&block).await?;
    balance_store.update(miner_address, reward)?;

    // CRITICAL: Propagate to network via gossipsub
    network_manager.broadcast_block(&block).await?;

    Ok(block)
}
```

### 2. UnifiedNetworkManager Must Support Block Broadcasting

**File**: `crates/q-network/src/unified_network_manager.rs`

Add method:
```rust
impl UnifiedNetworkManager {
    /// Broadcast a newly mined block to the network
    pub async fn broadcast_block(&self, block: &Block) -> Result<()> {
        let topic = format!("/qnk/{}/blocks", self.network_id);
        let message = serialize_block(block)?;

        // Publish to gossipsub
        self.swarm.behaviour_mut()
            .gossipsub
            .publish(IdentTopic::new(topic), message)?;

        info!("📡 Broadcast mined block {} to network", block.height);
        Ok(())
    }
}
```

### 3. Network Peers Must Validate and Accept External Blocks

**File**: `crates/q-api-server/src/main.rs`

The gossipsub message handler must:
```rust
// Handle incoming block from gossipsub
async fn handle_gossipsub_block(
    block: Block,
    storage: Arc<Storage>,
    dag_manager: Arc<DagManager>,
) -> Result<()> {
    // 1. Verify block signature
    if !verify_block_signature(&block)? {
        warn!("Invalid block signature from peer");
        return Err(anyhow!("Invalid signature"));
    }

    // 2. Verify PoW solution
    if !verify_pow_solution(&block)? {
        warn!("Invalid PoW solution");
        return Err(anyhow!("Invalid PoW"));
    }

    // 3. Verify block fits in DAG
    if !dag_manager.can_add_block(&block)? {
        debug!("Block doesn't fit current DAG state");
        return Ok(()); // Not an error, just orphan
    }

    // 4. Add to storage and DAG
    storage.store_block(&block).await?;
    dag_manager.add_block(&block)?;

    // 5. Update miner's balance (CRITICAL)
    let reward = calculate_block_reward(block.height);
    balance_store.credit(block.miner_address, reward)?;

    info!("✅ Accepted external block {} from {}", block.height, block.miner_address);
    Ok(())
}
```

---

## The DAG-Knight Challenge

In a DAG-based system (unlike linear blockchain), multiple blocks can exist at the same height:

```
Height 100:  [Block A] ────┐
                          ├──► [Block C @ Height 101]
Height 100:  [Block B] ────┘
```

This means:
- Multiple miners can produce valid blocks simultaneously
- All valid blocks should be included in the DAG
- Rewards should go to ALL miners whose blocks are included
- Final ordering happens via DAG-Knight consensus

### DAG-Knight Integration Requirements

**File**: `crates/q-dagknight/src/lib.rs`

```rust
impl DagKnight {
    /// Add a new block to the DAG
    pub fn add_block(&mut self, block: Block) -> Result<()> {
        // Verify parents exist
        for parent_hash in &block.parents {
            if !self.blocks.contains_key(parent_hash) {
                return Err(anyhow!("Missing parent block"));
            }
        }

        // Add to DAG
        self.blocks.insert(block.hash.clone(), block.clone());

        // Update tips (blocks with no children)
        self.update_tips(&block);

        // Recalculate consensus ordering
        self.recalculate_ordering()?;

        Ok(())
    }
}
```

---

## libp2p Network Crate Structure

### Required Components

```
crates/q-network/
├── src/
│   ├── lib.rs                    # Module exports
│   ├── unified_network_manager.rs # Main network orchestrator
│   ├── gossipsub/
│   │   ├── mod.rs                # Gossipsub configuration
│   │   ├── topics.rs             # Topic definitions
│   │   └── message_handler.rs    # Incoming message processing
│   ├── kademlia/
│   │   ├── mod.rs                # DHT for peer discovery
│   │   └── bootstrap.rs          # Bootstrap node connection
│   ├── request_response/
│   │   ├── mod.rs                # Block sync protocol
│   │   └── block_pack_codec.rs   # Serialization
│   └── block_propagation/
│       ├── mod.rs                # NEW: Block broadcast system
│       ├── broadcaster.rs        # Outgoing blocks
│       └── receiver.rs           # Incoming blocks validation
```

### Topic Definitions

```rust
// crates/q-network/src/gossipsub/topics.rs

pub struct NetworkTopics {
    /// Block propagation topic
    pub blocks: IdentTopic,

    /// Peer height announcements
    pub peer_heights: IdentTopic,

    /// Transaction propagation (future)
    pub transactions: IdentTopic,

    /// Mining difficulty adjustments
    pub difficulty: IdentTopic,
}

impl NetworkTopics {
    pub fn new(network_id: &str) -> Self {
        Self {
            blocks: IdentTopic::new(format!("/qnk/{}/blocks", network_id)),
            peer_heights: IdentTopic::new(format!("/qnk/{}/peer-heights", network_id)),
            transactions: IdentTopic::new(format!("/qnk/{}/transactions", network_id)),
            difficulty: IdentTopic::new(format!("/qnk/{}/difficulty", network_id)),
        }
    }
}
```

---

## Implementation Phases

### Phase 1: Block Broadcasting (Week 1)
- [ ] Add `broadcast_block()` to UnifiedNetworkManager
- [ ] Wire block producer to call broadcast after local storage
- [ ] Add gossipsub topic subscription for blocks
- [ ] Basic incoming block handler

### Phase 2: Block Validation (Week 1-2)
- [ ] PoW verification for incoming blocks
- [ ] Signature verification
- [ ] Parent block existence check
- [ ] Duplicate block detection

### Phase 3: DAG Integration (Week 2)
- [ ] Proper DAG insertion for external blocks
- [ ] Balance updates for external block miners
- [ ] Orphan block handling
- [ ] Fork resolution via DAG-Knight

### Phase 4: Testing & Hardening (Week 3)
- [ ] Multi-node test with 3+ nodes
- [ ] Mining from different nodes simultaneously
- [ ] Network partition recovery
- [ ] Balance consistency verification

---

## Critical Code Locations

| Component | File | Function |
|-----------|------|----------|
| Block Production | `crates/q-api-server/src/block_producer.rs` | `produce_block()` |
| Mining API | `crates/q-api-server/src/handlers.rs` | `submit_mining_solution()` |
| Network Manager | `crates/q-network/src/unified_network_manager.rs` | `new()`, event loop |
| Gossipsub | `crates/q-network/src/unified_network_manager.rs` | `configure_gossipsub()` |
| Balance Store | `crates/q-storage/src/lib.rs` | `BalanceStore` |
| DAG-Knight | `crates/q-dagknight/src/lib.rs` | `DagKnight` |

---

## Testing Plan

### Test 1: Local Block Propagation
```bash
# Start bootstrap node (Server Beta)
systemctl start q-api-server

# Start test node (Docker)
docker run -d --name q-node-test \
  -p 8099:8080 -p 9099:9001 \
  q-narwhalknight:v1.0.84

# Start miner pointed at test node
./q-miner --mode solo --wallet $WALLET --server http://localhost:8099

# Check bootstrap node for mined blocks
curl http://185.182.185.227:8080/api/v1/blocks/latest

# Blocks should appear on bootstrap within seconds!
```

### Test 2: Balance Consensus
```bash
# Check wallet balance on local node
curl http://localhost:8099/api/v1/balance/$WALLET

# Check same wallet on bootstrap
curl http://185.182.185.227:8080/api/v1/balance/$WALLET

# Balances should match!
```

### Test 3: Multi-Miner Competition
```bash
# Start 3 miners on different nodes
# All mining to different wallets
# Verify all get rewards proportional to blocks found
```

---

## Conclusion

The current mining system is fundamentally broken because it lacks P2P block propagation. The fix requires:

1. **Wiring block production to gossipsub broadcast**
2. **Handling incoming blocks from network peers**
3. **Proper DAG integration for externally-mined blocks**
4. **Balance consensus across all nodes**

This is a ~2-3 week implementation to get working correctly. The good news is that all the underlying pieces exist (libp2p, gossipsub, DAG-Knight) - they just need to be connected properly.

---

## References

- libp2p-rust: https://github.com/libp2p/rust-libp2p
- Gossipsub Spec: https://github.com/libp2p/specs/blob/master/pubsub/gossipsub/gossipsub-v1.1.md
- DAG-Knight Paper: https://eprint.iacr.org/2022/1488
- Q-NarwhalKnight Architecture: `/opt/orobit/shared/q-narwhalknight/papers/`
