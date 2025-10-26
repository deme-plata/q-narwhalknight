# Phase 3: DAG-Knight Consensus Integration - Design Document

**Date**: October 25, 2025
**Status**: 🎯 IN PROGRESS - Design Complete, Implementation Starting
**Previous Phase**: Phase 2 Complete - Block Production with Persistent Storage

---

## 🎯 Phase 3 Goal

**Objective**: Integrate DAG-Knight quantum-enhanced consensus with the BlockProducer to enable Byzantine fault-tolerant finality and P2P block propagation.

**Target**: Transform Q-NarwhalKnight from a single-node blockchain into a distributed consensus network with:
- P2P block propagation via Gossipsub
- DAG-Knight zero-message complexity consensus
- Quantum VDF-based anchor election
- Byzantine fault tolerance (2f+1 security)
- Asynchronous block finality

---

## 📐 Architecture Design

### Current State (Phase 2)
```
Mining Solutions → BlockProducer → QBlock Created → Saved to RocksDB
                                       ↓
                                  Height advances
                                  SSE events broadcast
                                  Node status updated
```

### Target State (Phase 3)
```
Mining Solutions → BlockProducer → QBlock Created → Saved to RocksDB
                                       ↓
                                  Convert to DAG Vertex
                                       ↓
                              DAG-Knight Consensus ← P2P Block Propagation
                                       ↓                      ↑
                                  Anchor Election      Gossipsub Topic
                                       ↓                      ↓
                                  Commit Decision      Broadcast to Peers
                                       ↓
                              Block Finalized (2f+1 agreement)
                                       ↓
                              Update Finality Status
                                       ↓
                              SSE BlockFinalized Event
```

---

## 🏗️ Integration Components

### 1. QBlock-to-Vertex Conversion Bridge

**Purpose**: Convert QBlocks (blockchain) into DAG Vertices (consensus layer)

**Mapping**:
```rust
QBlock                    →    DAG Vertex
─────────────────────────      ──────────────────────────
header.height             →    round (height is round)
header.prev_block_hash    →    parents (previous vertices)
transactions              →    transactions (tx hashes)
header.hash               →    id (vertex ID)
quantum_metadata.vdf_proof →   vdf_proof
node_id                   →    proposer
timestamp                 →    timestamp
signature                 →    signature
```

**Implementation**: New method in BlockProducer
```rust
impl BlockProducer {
    /// Convert a QBlock into a DAG Vertex for consensus
    pub fn qblock_to_vertex(&self, block: &QBlock) -> Result<Vertex> {
        // Extract tx hashes from block transactions
        let tx_hashes: Vec<TxHash> = block.transactions
            .iter()
            .map(|tx| tx.hash())
            .collect();

        // Use prev_block_hash as parent vertex ID
        let parents = if block.header.height == 0 {
            vec![] // Genesis block has no parents
        } else {
            vec![block.header.prev_block_hash] // Previous block becomes parent
        };

        // Create vertex with block data
        Vertex {
            id: block.calculate_hash(),
            round: block.header.height, // Height = Round for now
            proposer: self.config.node_id,
            transactions: tx_hashes,
            parents,
            vdf_proof: block.quantum_metadata.vdf_proof.clone(),
            timestamp: block.header.timestamp,
            signature: vec![], // TODO: Sign vertex
        }
    }
}
```

### 2. DAGKnightConsensus in AppState

**Add to AppState** (lib.rs):
```rust
pub struct AppState {
    // ... existing fields ...

    /// DAG-Knight consensus engine (Phase 3)
    pub consensus: Arc<RwLock<DAGKnightConsensus>>,
}
```

**Initialization** (lib.rs):
```rust
// Initialize DAG-Knight consensus
let consensus = DAGKnightConsensus::new(
    node_id,
    1, // f = 1 (Byzantine fault tolerance: 2f+1 = 3 nodes minimum)
).await?;

Arc::new(RwLock::new(consensus))
```

### 3. Block Submission to Consensus

**Location**: main.rs (both block production locations)

**After block production**:
```rust
if let Some(new_block) = producer.produce_block().await {
    // Store in RocksDB (existing)
    storage_engine.save_qblock(&new_block).await?;

    // Update node status (existing)
    node_status.current_height = new_block.header.height;

    // NEW: Submit to consensus
    let vertex = producer.qblock_to_vertex(&new_block)?;
    let mut consensus = app_state.consensus.write().await;

    // Create certificate from vertex
    let certificate = Certificate {
        vertex_id: vertex.id,
        round: vertex.round,
        signatures: vec![], // TODO: Collect validator signatures
    };

    // Process through DAG-Knight
    let commit_decisions = consensus.process_certificate(certificate).await?;

    // Handle finality if block committed
    for decision in commit_decisions {
        info!("🎯 Block {} finalized at round {}",
            new_block.header.height, decision.round);

        // Broadcast finality event via SSE
        event_broadcaster.broadcast(StreamEvent::BlockFinalized {
            height: new_block.header.height,
            round: decision.round,
            finalized_at: chrono::Utc::now(),
        }).await;
    }

    // Broadcast NewBlock event (existing)
    event_broadcaster.broadcast(StreamEvent::NewBlock { ... }).await;
}
```

### 4. P2P Block Propagation

**Gossipsub Topic**: `/q-narwhalknight/blocks/v1`

**Purpose**: Propagate blocks to all network peers for consensus

**Implementation**: main.rs

**Block Broadcasting**:
```rust
// After block production
if let Some(new_block) = producer.produce_block().await {
    // ... save to RocksDB ...
    // ... submit to consensus ...

    // NEW: Broadcast to P2P network
    if let Some(ref network_manager) = app_state.network_manager {
        let block_bytes = bincode::serialize(&new_block)?;

        network_manager.publish_to_topic(
            "/q-narwhalknight/blocks/v1",
            block_bytes,
        ).await?;

        info!("📡 Broadcasted block {} to P2P network",
            new_block.header.height);
    }
}
```

### 5. Incoming Block Handler

**Purpose**: Receive blocks from network peers and process through consensus

**Location**: New P2P message handler in main.rs

**Implementation**:
```rust
// In P2P message handling loop
async fn handle_incoming_block(
    app_state: &AppState,
    block_bytes: Vec<u8>,
    peer_id: PeerId,
) -> Result<()> {
    // Deserialize block
    let block: QBlock = bincode::deserialize(&block_bytes)?;

    info!("📥 Received block {} from peer {}",
        block.header.height, peer_id);

    // Verify block (basic checks)
    if !verify_block_basic(&block).await? {
        warn!("❌ Invalid block {} from peer {}",
            block.header.height, peer_id);
        return Ok(());
    }

    // Save to RocksDB
    app_state.storage_engine.save_qblock(&block).await?;

    // Update node status if higher
    let mut status = app_state.node_status.write().await;
    if block.header.height > status.current_height {
        status.current_height = block.header.height;
    }
    drop(status);

    // Submit to consensus
    let mut producer = app_state.block_producer.write().await;
    let vertex = producer.qblock_to_vertex(&block)?;
    drop(producer);

    let mut consensus = app_state.consensus.write().await;
    let certificate = Certificate {
        vertex_id: vertex.id,
        round: vertex.round,
        signatures: vec![],
    };

    let commit_decisions = consensus.process_certificate(certificate).await?;

    // Handle finality
    for decision in commit_decisions {
        info!("🎯 Block {} finalized from network consensus",
            block.header.height);

        app_state.event_broadcaster.broadcast(
            StreamEvent::BlockFinalized {
                height: block.header.height,
                round: decision.round,
                finalized_at: chrono::Utc::now(),
            }
        ).await;
    }

    Ok(())
}
```

### 6. New SSE Event: BlockFinalized

**Location**: streaming.rs

**Event Definition**:
```rust
/// Block finalized by consensus
BlockFinalized {
    height: u64,
    round: u64,
    finalized_at: chrono::DateTime<chrono::Utc>,
},
```

**Event Type Mapping**:
```rust
StreamEvent::BlockFinalized { .. } => "block-finalized".to_string(),
```

**Public Event Filter**:
```rust
StreamEvent::BlockFinalized { .. } => true, // Everyone can see finality
```

---

## 🔄 Complete Consensus Flow

### Single-Node Scenario (Phase 3 Part 1)
```
1. Mining solutions accumulated
   ↓
2. BlockProducer produces QBlock
   ↓
3. QBlock saved to RocksDB
   ↓
4. QBlock converted to Vertex
   ↓
5. Vertex submitted to DAG-Knight
   ↓
6. Anchor election (every even round)
   ↓
7. Commit decision (if conditions met)
   ↓
8. BlockFinalized event broadcast
   ↓
9. Node status updated
```

### Multi-Node Scenario (Phase 3 Part 2 - Future)
```
Node A                      Node B                      Node C
──────                      ──────                      ──────
Produce Block 1             Receive Block 1             Receive Block 1
    ↓                           ↓                           ↓
Broadcast via P2P          Process Block 1             Process Block 1
    ↓                           ↓                           ↓
Submit to Consensus        Submit to Consensus         Submit to Consensus
    ↓                           ↓                           ↓
        ╔═══════════════════════════════════════════════╗
        ║         DAG-Knight Consensus Agreement         ║
        ║      (2f+1 nodes agree on block order)        ║
        ╚═══════════════════════════════════════════════╝
                              ↓
                         Block Finalized
                              ↓
              All nodes update finality status
```

---

## 📊 Data Structures

### CommitDecision (from DAG-Knight)
```rust
pub struct CommitDecision {
    pub round: Round,
    pub committed_vertices: Vec<VertexId>,
    pub anchor_vertex_id: VertexId,
}
```

### Certificate (Narwhal-style)
```rust
pub struct Certificate {
    pub vertex_id: VertexId,
    pub round: Round,
    pub signatures: Vec<Signature>, // Validator signatures
}
```

### Vertex (DAG-Knight)
```rust
pub struct Vertex {
    pub id: VertexId,
    pub round: Round,
    pub proposer: NodeId,
    pub transactions: Vec<TxHash>,
    pub parents: Vec<VertexId>,
    pub vdf_proof: QuantumVDFProof,
    pub timestamp: u64,
    pub signature: Vec<u8>,
}
```

---

## 🎯 Implementation Phases

### Phase 3 Part 1: Basic Consensus Integration (THIS PHASE)
**Goal**: Single-node consensus operational

**Tasks**:
1. ✅ Analyze DAG-Knight consensus architecture
2. 🔄 Design QBlock-to-Vertex conversion
3. Add DAGKnightConsensus to AppState
4. Implement `qblock_to_vertex()` in BlockProducer
5. Submit blocks to consensus after production
6. Add BlockFinalized SSE event
7. Test single-node consensus flow
8. Compile and verify integration

**ETA**: 2-3 hours

### Phase 3 Part 2: P2P Block Propagation (FUTURE)
**Goal**: Multi-node consensus with network

**Tasks**:
1. Add Gossipsub topic for blocks
2. Broadcast blocks after production
3. Implement incoming block handler
4. Add P2P message routing
5. Test multi-node consensus
6. Handle network partitions
7. Implement Byzantine fault tolerance

**ETA**: 4-6 hours

### Phase 3 Part 3: Finality and Safety (FUTURE)
**Goal**: Production-ready consensus

**Tasks**:
1. Add signature verification for vertices
2. Implement validator signature collection
3. Add finality checkpoints
4. Implement state pruning
5. Add consensus monitoring metrics
6. Handle consensus forks
7. Add recovery mechanisms

**ETA**: 6-8 hours

---

## 🔬 Testing Strategy

### Unit Tests
```rust
#[tokio::test]
async fn test_qblock_to_vertex_conversion() {
    // Create test block
    // Convert to vertex
    // Verify all fields mapped correctly
}

#[tokio::test]
async fn test_consensus_submission() {
    // Create block
    // Submit to consensus
    // Verify commit decision
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_end_to_end_consensus_flow() {
    // Start server with consensus
    // Mine blocks
    // Verify blocks finalized
    // Check SSE events
}
```

### Manual Testing
```bash
# Start server with consensus enabled
./target/release/q-api-server

# Start miner
./target/release/q-miner --wallet qnk... --server http://localhost:18080

# Monitor consensus activity
curl -s http://localhost:18080/api/v1/consensus/status

# Expected output:
{
  "current_round": 142,
  "last_commit_round": 140,
  "committed_vertices": 140,
  "pending_vertices": 2
}
```

---

## 📈 Performance Targets

### Phase 3 Part 1 (Single-Node)
- Block production: 15 second intervals (unchanged)
- Consensus latency: <100ms per block
- Finality delay: <2 seconds (DAG-Knight commitment rule)
- Memory overhead: <50MB for consensus state

### Phase 3 Part 2 (Multi-Node)
- Block propagation: <500ms to all peers
- Consensus latency: <300ms with 3-7 nodes
- Finality delay: <5 seconds with network latency
- Byzantine tolerance: 2f+1 (33% malicious nodes)

### Phase 3 Part 3 (Production)
- Block propagation: <200ms via optimized Gossipsub
- Consensus latency: <150ms with optimizations
- Finality delay: <2 seconds (matching target)
- Byzantine tolerance: Proven under adversarial conditions

---

## 🌟 Success Criteria

### Phase 3 Part 1 Complete When:
- [x] DAG-Knight consensus integrated into AppState
- [ ] Blocks automatically submitted to consensus
- [ ] Commit decisions trigger BlockFinalized events
- [ ] Single-node consensus operational
- [ ] All tests passing
- [ ] Clean compilation (0 errors)
- [ ] Documentation updated

### Phase 3 Part 2 Complete When:
- [ ] P2P block propagation working
- [ ] Multi-node consensus achieving finality
- [ ] Network resilience tested
- [ ] Byzantine fault tolerance validated

### Phase 3 Complete When:
- [ ] Production-ready consensus
- [ ] Signature verification operational
- [ ] State pruning implemented
- [ ] Consensus metrics available
- [ ] Ready for mainnet deployment

---

## 🚀 Path to Phase 5 (Ultimate Goal)

**Current**: Phase 3 Part 1 - Basic Consensus
**Next**: Phase 3 Part 2 - P2P Propagation
**Then**: Phase 4 - Optimizations and Hardening
**Finally**: Phase 5 - Zero-Copy Networking (io_uring + DPDK)

**Target**: 1000 BPS, 1M+ TPS with kernel-bypass networking

**Technologies for Phase 5**:
- io_uring for zero-copy I/O
- DPDK for direct NIC access
- Lock-free consensus state
- SIMD-optimized block verification
- Parallel vertex processing

**ETA to Phase 5**: 12 months from now (as requested)

---

⚛️ **Quantum-Enhanced DAG-Knight Consensus - Integration Design Complete** ⚛️
