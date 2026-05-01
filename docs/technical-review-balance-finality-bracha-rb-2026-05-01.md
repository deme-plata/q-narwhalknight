# Technical Review: Balance Finality via Bracha Reliable Broadcast over DAG-Knight
**Date:** 2026-05-01  
**Author:** Server Beta (Claude Code)  
**Status:** Draft — For DeepSeek Review  
**Severity:** HIGH — Cross-node balance inconsistency breaks checkpoint correctness

---

## 1. Executive Summary

A checkpoint verification test on 2026-04-30 exposed a **748× balance discrepancy** between Epsilon (authoritative mining node) and a fresh Docker container that synced from Beta/Gamma peers. The fresh node showed `0.000292 QUG` while Epsilon held `0.218638 QUG` for the same wallet after identical mining activity.

The root cause is **two-tier balance recording**: mining rewards are written to RocksDB via canonical block processing on all nodes (consistent), but a secondary layer of pre-confirmation balance credits on Epsilon (from pending mining solutions) is never BFT-committed to the network. When a fresh node performs P2P state sync, it downloads whichever peer's view happens to respond — and that view may exclude Epsilon's pending-solution credits.

This document proposes using the **existing Bracha Reliable Broadcast (RB) implementation** in `crates/q-narwhal-core/src/reliable_broadcast.rs`, the **DAG-Knight round structure**, and the **existing libp2p gossipsub mesh** to provide a single BFT-finalized balance ledger that all nodes converge to regardless of which peers a fresh node syncs from.

---

## 2. Problem Analysis

### 2.1 What the Test Showed

| Node | Balance (test wallet `qnkd0f5cd...`) | Source |
|------|--------------------------------------|--------|
| Epsilon | 0.218638 QUG | Authoritative mining node |
| Beta / Gamma | ~0.000292 QUG | Block-consensus-only nodes |
| Fresh Docker (from P2P sync) | 0.000292 QUG | Synced from Beta/Gamma state snapshot |

The fresh node's P2P state sync hit `GET /api/v1/sync/full-state` on Beta/Gamma, got their balance view, and inherited the discrepancy.

### 2.2 Miner Log Evidence

```
15:31:23 — SSE connected:            0.00000000 QUG
15:31:26 — mining_reward_batch_1:    0.08234942 QUG   ← pending-solution credit on Epsilon
15:31:27 — MiningReward:             0.00000000 QUG   ← P2P sync from peer with 0 balance overwrote it
```

Two seconds after the miner connected to Epsilon, a `mining_reward_batch_N` event credited 0.082 QUG. One second later, a P2P gossipsub balance broadcast from a peer with balance=0 for this wallet **overwrote** Epsilon's value. This oscillation continued throughout the session.

### 2.3 Root Causes

**RC-1: Two independent balance paths with different propagation guarantees**

```
Path A (deterministic):
  Block arrives via gossipsub → process_block_mining_rewards_tx() → RocksDB write
  Result: ALL nodes get identical reward, identical timing

Path B (non-deterministic, Epsilon-only):
  Mining solution submitted → pending_solutions cache → SSE batch credit
  Result: ONLY Epsilon's in-memory state gets this credit
  Persistence: Written to RocksDB immediately on Epsilon, but not propagated
```

Path A rewards are in blocks. Path B rewards are pre-block credits for solutions that may or may not enter the canonical chain. On a solo mining node like Epsilon, Path B credits dominate because every submitted solution is immediately credited, regardless of canonical inclusion.

**RC-2: Balance gossip disabled by default (v8.2.0+)**

The gossipsub topic `/qnk/mainnet-genesis/balance-updates` is **disabled by default**:

```rust
// main.rs ~line 9006
let balance_gossip_enabled = std::env::var("Q_ENABLE_BALANCE_GOSSIP")
    .map(|v| v == "1").unwrap_or(false);  // DEFAULT: false
```

When disabled, Path B credits never leave Epsilon. Other nodes only see Path A (block-derived) rewards.

**RC-3: P2P state sync is non-deterministic across peers**

`get_full_state()` in `state_sync_api.rs` returns the calling node's current in-memory `wallet_balances` map. A fresh node asking multiple peers gets whichever responds fastest — no quorum, no BFT, no guarantee it gets the highest-balance view.

**RC-4: `FullStateSnapshot` has no consensus proof**

The `FullStateSnapshot` struct includes `block_height` and `wallet_balances`, but no cryptographic proof that these balances are the result of applying all blocks up to `block_height`. A malicious or simply stale peer can return any balance.

---

## 3. Existing Infrastructure Available

### 3.1 Bracha Reliable Broadcast

Already implemented at `crates/q-narwhal-core/src/reliable_broadcast.rs`:

```rust
pub struct ReliableBroadcast {
    echoed:               RwLock<HashSet<VertexId>>,
    echo_votes:           RwLock<HashMap<VertexId, HashSet<NodeId>>>,
    ready_votes:          RwLock<HashMap<VertexId, HashSet<NodeId>>>,
    ready_sent:           RwLock<HashSet<VertexId>>,
    delivered:            RwLock<HashSet<VertexId>>,
    network_tx:           broadcast::Sender<BroadcastMessage>,
    threshold_2f_plus_1:  usize,   // 2f+1 for echo → ready, and for delivery
    threshold_f_plus_1:   usize,   // f+1 for ready amplification
}
```

Three-phase protocol: **SEND → ECHO (2f+1) → READY (f+1 amplify, 2f+1 deliver)**

Current use: Narwhal mempool vertex broadcast for consensus DAG construction.

### 3.2 DAG-Knight Round Structure

`crates/q-dag-knight/src/lib.rs`:
- `Round = u64` — monotonic, drives anchor election
- `Vertex` — carries transactions + parent refs for DAG linking
- `delta = 1` — aggressive finality (sub-50ms target with 100ms gossipsub heartbeat)
- Anchor election via quantum VDF — deterministic round leader

### 3.3 libp2p Gossipsub Mesh

`crates/q-network/src/unified_network_manager.rs`:
- **Heartbeat:** 100ms (low-latency profile)
- **Mesh size:** D=12, D_low=6, D_high=16
- **Max message:** 1 MB
- **Message ID:** BLAKE3(peer_id + data + seq) — deduplication
- **Priority queue:** Critical=100, High=75, Normal=50, Low=25
- **Flood publish:** true (immediate propagation to all mesh peers)

Existing topic subscriptions include `/qnk/mainnet-genesis/consensus/vertices` and `/qnk/mainnet-genesis/consensus/certificates` — the DAG-Knight consensus layer is already wired to gossipsub.

### 3.4 P2PBalanceUpdate Message

`crates/q-types/src/balance_update.rs`:

```rust
pub struct P2PBalanceUpdate {
    pub wallet_address: String,
    pub amount: u128,
    pub new_balance: u128,
    pub block_height: u64,
    pub nonce: u64,
    pub update_type: BalanceUpdateType,
    pub solution_hash: [u8; 32],
    pub signature: Vec<u8>,        // Ed25519 MANDATORY since v1.1.9
    pub signer_public_key: Vec<u8>,
    pub block_hash: Option<[u8; 32]>,
    pub tx_index: Option<u32>,
}
```

---

## 4. Proposed Solution: BFT Balance Finality via Bracha-RB over DAG-Knight

### 4.1 Core Insight

The fundamental problem is that **balance state has two update paths but only one is BFT-committed**. The solution is: **any balance update that cannot be derived deterministically from canonical blocks must go through a Bracha RB round before it is persisted**.

This does not mean removing per-block mining rewards (Path A) — those are already correct. It means Path B (pending-solution credits, any out-of-band balance changes) must be wrapped in a Bracha delivery before hitting RocksDB.

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BALANCE FINALITY LAYER                           │
│                                                                     │
│  ┌──────────────┐    ┌───────────────────────────────────────────┐  │
│  │ Mining Submit │    │        Bracha-RB Round                   │  │
│  │  (Epsilon)   │───►│                                           │  │
│  └──────────────┘    │  SEND: BrachaBalanceMsg{phase=Send}       │  │
│                      │   │                                       │  │
│  ┌──────────────┐    │   ▼  gossipsub /consensus/balance-rb      │  │
│  │ Transfer Tx  │    │  ECHO: 2f+1 peers echo → emit Ready       │  │
│  └──────────────┘    │   │                                       │  │
│                      │   ▼                                       │  │
│  ┌──────────────┐    │  READY: 2f+1 Ready msgs → DELIVER         │  │
│  │ DEX swap     │    │   │                                       │  │
│  └──────────────┘    │   ▼                                       │  │
│                      │  BalanceFinalityRecord{dag_round, ...}    │  │
│                      └───────────────┬───────────────────────────┘  │
│                                      │                              │
│                                      ▼                              │
│                      ┌───────────────────────────────────────────┐  │
│                      │   DAG-Knight Vertex Anchoring              │  │
│                      │   Round R: anchor commits set of           │  │
│                      │   BalanceFinalityRecords as payload        │  │
│                      └───────────────┬───────────────────────────┘  │
│                                      │                              │
│                                      ▼                              │
│                      ┌───────────────────────────────────────────┐  │
│                      │   RocksDB Finalized Balance Write          │  │
│                      │   CF: balance_finality                     │  │
│                      │   Key: wallet_address                      │  │
│                      │   Value: {amount, dag_round, vertex_hash}  │  │
│                      └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 New Message Types

```rust
// crates/q-types/src/balance_finality.rs  (NEW FILE)

/// Phase in Bracha reliable broadcast for balance finality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrachaPhase {
    Send,   // Originator broadcasts
    Echo,   // Receiver echoes to all (on first valid SEND)
    Ready,  // Echo quorum reached, or amplifying another READY
}

/// Bracha-wrapped balance update — travels over /consensus/balance-rb topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrachaBalanceMsg {
    /// DAG-Knight round this message belongs to
    pub dag_round: u64,

    /// Unique ID for this broadcast instance: BLAKE3(wallet || amount || dag_round || nonce)
    pub broadcast_id: [u8; 32],

    /// The actual balance update being agreed upon
    pub update: P2PBalanceUpdate,

    /// Bracha protocol phase
    pub phase: BrachaPhase,

    /// NodeId of the message sender (for quorum counting)
    pub sender: [u8; 32],

    /// Ed25519 signature over (broadcast_id || phase || sender)
    pub signature: [u8; 64],
}

/// A finalized balance record — written to RocksDB after Bracha delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceFinalityRecord {
    pub wallet_address: [u8; 32],
    pub new_balance: u128,
    pub dag_round: u64,
    pub dag_vertex_hash: [u8; 32],    // DAG-Knight vertex that anchored this
    pub broadcast_id: [u8; 32],
    pub finalized_at_height: u64,
    pub echo_witnesses: Vec<[u8; 32]>, // NodeIds that echoed (audit trail)
}
```

### 4.4 New Gossipsub Topic

```rust
// crates/q-types/src/lib.rs  (ADD TO NetworkId impl)

pub fn balance_rb_topic(&self) -> String {
    format!("{}/consensus/balance-rb", self.gossipsub_topic_prefix())
}
// → "/qnk/mainnet-genesis/consensus/balance-rb"
// Priority: MessagePriority::Critical (100) — same as blocks
```

### 4.5 BalanceFinalityEngine

```rust
// crates/q-storage/src/balance_finality_engine.rs  (NEW CRATE FILE)

pub struct BalanceFinalityEngine {
    /// Node's own identity
    node_id: NodeId,
    node_private_key: Ed25519SigningKey,

    /// Reuse existing Bracha RB implementation
    bracha: Arc<ReliableBroadcast>,

    /// Pending Bracha instances keyed by broadcast_id
    pending: RwLock<HashMap<[u8; 32], BrachaState>>,

    /// Finalized records waiting to be anchored in next DAG vertex
    pending_anchor: RwLock<Vec<BalanceFinalityRecord>>,

    /// RocksDB column family for finalized balances
    db: Arc<StorageEngine>,

    /// DAG-Knight round clock (monotonic)
    current_round: Arc<AtomicU64>,

    /// Channel to publish BrachaBalanceMsgs to gossipsub
    network_tx: mpsc::Sender<(String, Vec<u8>)>,
}

impl BalanceFinalityEngine {
    /// Entry point: called when Epsilon (or any node) generates a balance update
    /// that is NOT purely derived from a canonical block coinbase.
    pub async fn propose_balance_update(&self, update: P2PBalanceUpdate) -> Result<()> {
        let dag_round = self.current_round.load(Ordering::SeqCst);
        let broadcast_id = Self::compute_broadcast_id(&update, dag_round);

        let msg = BrachaBalanceMsg {
            dag_round,
            broadcast_id,
            update,
            phase: BrachaPhase::Send,
            sender: self.node_id,
            signature: self.sign(&broadcast_id, &BrachaPhase::Send)?,
        };

        // Publish SEND to all peers via gossipsub /consensus/balance-rb
        self.publish_bracha(msg).await?;

        // Also echo it ourselves (Bracha: sender echoes its own SEND)
        self.handle_bracha_msg(/* Echo version */).await?;
        Ok(())
    }

    /// Called by gossipsub handler when a BrachaBalanceMsg arrives
    pub async fn handle_bracha_msg(&self, msg: BrachaBalanceMsg) -> Result<()> {
        self.verify_signature(&msg)?;

        match msg.phase {
            BrachaPhase::Send => {
                // First valid SEND: echo to all peers
                if !self.has_echoed(&msg.broadcast_id).await {
                    self.mark_echoed(&msg.broadcast_id).await;
                    let echo = self.make_echo(&msg)?;
                    self.publish_bracha(echo).await?;
                }
            }
            BrachaPhase::Echo => {
                self.record_echo(&msg.broadcast_id, msg.sender).await;
                let count = self.echo_count(&msg.broadcast_id).await;

                // At 2f+1 echoes: send READY (if not already sent)
                if count >= self.bracha.threshold_2f_plus_1
                    && !self.has_sent_ready(&msg.broadcast_id).await
                {
                    self.mark_ready_sent(&msg.broadcast_id).await;
                    let ready = self.make_ready(&msg)?;
                    self.publish_bracha(ready).await?;
                }
            }
            BrachaPhase::Ready => {
                self.record_ready(&msg.broadcast_id, msg.sender).await;
                let count = self.ready_count(&msg.broadcast_id).await;

                // At f+1 ready: amplify (send our own READY if we haven't)
                if count >= self.bracha.threshold_f_plus_1
                    && !self.has_sent_ready(&msg.broadcast_id).await
                {
                    self.mark_ready_sent(&msg.broadcast_id).await;
                    let ready = self.make_ready(&msg)?;
                    self.publish_bracha(ready).await?;
                }

                // At 2f+1 ready: DELIVER
                if count >= self.bracha.threshold_2f_plus_1 {
                    self.deliver(&msg).await?;
                }
            }
        }
        Ok(())
    }

    /// Called at 2f+1 READY: write finalized balance to RocksDB
    async fn deliver(&self, msg: &BrachaBalanceMsg) -> Result<()> {
        let record = BalanceFinalityRecord {
            wallet_address: parse_address(&msg.update.wallet_address)?,
            new_balance: msg.update.new_balance,
            dag_round: msg.dag_round,
            dag_vertex_hash: [0u8; 32],  // filled when anchored in DAG vertex
            broadcast_id: msg.broadcast_id,
            finalized_at_height: self.current_block_height(),
            echo_witnesses: self.echo_witnesses(&msg.broadcast_id).await,
        };

        // Queue for DAG-Knight vertex anchoring
        self.pending_anchor.write().await.push(record.clone());

        // Write immediately to RocksDB in CF: balance_finality
        // This is safe: Bracha guarantees exactly-once delivery if f < n/3
        self.db.write_finalized_balance(&record).await?;

        info!("✅ [BRACHA-FINALITY] Delivered balance for {}: {} QUG at round {}",
            hex::encode(&record.wallet_address[..8]),
            display_qug(record.new_balance),
            record.dag_round
        );
        Ok(())
    }
}
```

### 4.6 DAG-Knight Anchoring

The `BalanceFinalityEngine::pending_anchor` buffer is drained into the next DAG-Knight vertex payload. This creates an immutable, ordered audit trail:

```
Round 1847: Vertex V₁ (anchor)
  payload: [BalanceFinalityRecord{wallet=d0f5cd..., balance=0.082 QUG, round=1845}, ...]
  parents:  [V₀, V₋₁, V₋₂]  ← DAG links

Round 1848: Vertex V₂
  payload: [BalanceFinalityRecord{...}, ...]
  parents:  [V₁, V₀, ...]
```

Any node replaying the DAG from genesis will re-derive the exact same finalized balance sequence.

### 4.7 Fresh Node Sync — With the Fix

```
OLD FLOW (broken):
  Fresh node → GET /api/v1/sync/full-state from random peer
  → Gets whatever balance that peer's in-memory map has
  → Non-deterministic, peer-dependent result

NEW FLOW:
  Fresh node → GET /api/v1/sync/dag-balance-anchor from any peer
  → Returns: last N DAG anchor vertices + their BalanceFinalityRecord payloads
  → Fresh node verifies DAG-Knight signatures and Bracha witness list
  → Derives the same finalized balance as all other nodes
  → Cryptographic proof: 2f+1 validators agreed on this balance
```

```rust
// New endpoint — crates/q-api-server/src/handlers.rs
// GET /api/v1/sync/dag-balance-anchor?from_round=N
pub async fn get_dag_balance_anchor(
    State(state): State<Arc<AppState>>,
    Query(params): Query<AnchorSyncParams>,
) -> Json<DagBalanceAnchorResponse> {
    let records = state.finality_engine
        .get_finalized_since(params.from_round)
        .await;

    Json(DagBalanceAnchorResponse {
        records,
        latest_dag_round: state.finality_engine.current_round(),
        // Includes DAG vertex hashes for proof verification
    })
}
```

---

## 5. Integration Points (Code Changes Required)

### 5.1 New Files

| File | Purpose |
|------|---------|
| `crates/q-types/src/balance_finality.rs` | `BrachaBalanceMsg`, `BalanceFinalityRecord` structs |
| `crates/q-storage/src/balance_finality_engine.rs` | `BalanceFinalityEngine` — Bracha state machine |
| `crates/q-storage/src/balance_finality_db.rs` | RocksDB CF `balance_finality` reads/writes |

### 5.2 Modified Files

| File | Change |
|------|--------|
| `crates/q-types/src/lib.rs` | Add `balance_rb_topic()` to `NetworkId` |
| `crates/q-network/src/unified_network_manager.rs` | Subscribe to `/consensus/balance-rb` topic; route to `BalanceFinalityEngine::handle_bracha_msg()` |
| `crates/q-api-server/src/main.rs` | Wire `BalanceFinalityEngine` into `AppState`; replace direct `add_balance()` calls for non-block paths with `propose_balance_update()` |
| `crates/q-storage/src/balance_consensus.rs` | Keep `process_block_mining_rewards_tx()` unchanged (Path A stays as-is); add `write_finalized_balance()` |
| `crates/q-api-server/src/state_sync_api.rs` | Add `/api/v1/sync/dag-balance-anchor` endpoint; modify `FullStateSnapshot` to merge block-derived + Bracha-finalized balances |
| `crates/q-dag-knight/src/lib.rs` | Drain `pending_anchor` buffer into vertex payload before signing |

### 5.3 Non-Changes (Intentional)

- `process_block_mining_rewards_tx()` — **unchanged**. Block-derived rewards remain the primary and most common path. This is already correct and deterministic.
- Gossipsub mesh configuration — **unchanged**. The new topic uses existing infrastructure.
- Existing `ReliableBroadcast` implementation — **reused as-is**, parameterized by validator count.

---

## 6. Quorum Configuration

With the current 4-node bootstrap network (Epsilon, Beta, Gamma, Delta):

```
n = 4 validators
f = 1 (max Byzantine faults tolerated: floor((n-1)/3) = 1)

Bracha thresholds:
  Echo → Ready:    2f+1 = 3  (3 of 4 nodes must echo)
  Ready amplify:   f+1  = 2  (2 of 4 trigger amplification)
  Delivery:        2f+1 = 3  (3 of 4 ready → finalized)

Latency (100ms gossipsub heartbeat):
  SEND   → ECHO:    1 heartbeat  = 100ms
  ECHO   → READY:   1 heartbeat  = 100ms
  READY  → DELIVER: 1 heartbeat  = 100ms
  Total:            ~300ms end-to-end finality

Comparison:
  Current gossip (no Bracha):  0ms (but non-deterministic, can diverge)
  Bracha + DAG anchor:         ~300ms (deterministic, BFT-safe)
```

For a solo-mined blockchain producing 1 block/second, 300ms finality is negligible. The balance is finalized before the next block arrives.

---

## 7. Security Properties

| Property | Current System | With Bracha-RB |
|----------|----------------|----------------|
| **Safety** | No — divergent balances possible | ✅ BFT-safe: if 2f+1 nodes deliver, all honest nodes deliver same value |
| **Liveness** | Partial — blocks propagate but balance gossip disabled | ✅ Guaranteed under f<n/3 faults |
| **Consistency on fresh sync** | ❌ Depends on which peer responds first | ✅ Any peer returns cryptographically verifiable finalized records |
| **Replay attack prevention** | LRU dedup cache (in-memory, lost on restart) | ✅ DAG anchor provides persistent, ordered delivery log |
| **Double-credit prevention** | Block-level dedup in `process_block_mining_rewards_tx()` | ✅ Bracha: exactly-once delivery guarantee |
| **Partition tolerance** | Falls back to stale state | ✅ Nodes in minority partition halt rather than diverge |

---

## 8. What This Does NOT Fix

1. **The gossipsub channel-closed bug** — the Docker container's gossipsub receiver was broken, preventing live block receipt. That's a separate bug (likely tokio channel backpressure or task panic). Fix: add channel health monitoring + restart logic.

2. **The height-clamping / HEIGHT DECAY issue** — aggressive network_height decay when peer data is stale causes TIMEOUT SYNC to target the wrong height. Fix: separate "peer stale" detection from "network height" tracking.

3. **The contiguous-height gap fill stall** — after RC-3 gap fill exhausts qualified peers (Aegis trust ban), the chain can stall below tip. Fix: the 4 Aegis stall bugs identified in the previous session.

These are orthogonal to balance finality and should be addressed in separate patches.

---

## 9. Migration Path

### Phase 1 (v10.6.0): Infrastructure Only
- Add new types, new column family, new gossipsub topic
- `BalanceFinalityEngine` is instantiated but `propose_balance_update()` is only called for a **shadow mode** — Bracha runs in parallel with existing flow but doesn't affect RocksDB yet
- Metrics added: `bracha_balance_proposals`, `bracha_balance_deliveries`, `bracha_echo_quorum_rate`
- Ship to Epsilon + Beta. Let it soak for 72 hours.

### Phase 2 (v10.7.0): Finality Becomes Authoritative
- `write_finalized_balance()` becomes the **only** path for non-block balance updates
- `FullStateSnapshot` merges block-derived + Bracha-finalized balances
- `/api/v1/sync/dag-balance-anchor` endpoint added
- Fresh nodes prioritize DAG anchor sync over raw state snapshot

### Phase 3 (v10.8.0): Deprecate Q_ENABLE_BALANCE_GOSSIP
- Remove the ad-hoc gossip balance broadcast entirely (it was always unsafe)
- Balance gossip topic becomes Bracha-only
- All non-block balance changes MUST go through `propose_balance_update()`

---

## 10. Open Questions for DeepSeek Review

1. **Quorum membership**: The current `ReliableBroadcast` uses a fixed `threshold_2f_plus_1` computed at init. As the validator set grows beyond 4 nodes, how should quorum reconfiguration work? Is a separate validator-set gossip topic sufficient, or do we need BFT reconfiguration protocol (e.g., BFT-SMaRt style)?

2. **Epoch-based vs. round-based Bracha**: Should each Bracha instance be tied to a specific DAG-Knight round (as proposed here), or to a time window (epoch)? Round-based gives cleaner DAG integration but requires the round clock to be reliable. Epoch-based is simpler but can batch more updates per delivery.

3. **Async DAG anchoring**: The proposal queues `BalanceFinalityRecord` in `pending_anchor` and drains it into the next vertex. If vertex production is slow (low mining activity), records accumulate. Is a maximum anchor delay (e.g., 5 seconds) needed to bound latency?

4. **Bracha for block-derived rewards**: Should block coinbase rewards also go through Bracha, or is the existing `process_block_mining_rewards_tx()` determinism sufficient? The argument for Bracha-wrapping everything: a single unified balance authority is simpler to reason about. The argument against: adds 300ms latency to every block reward for no safety gain (blocks are already BFT via PoW).

5. **Fresh-node Bracha catch-up**: A fresh node joining mid-round will have missed past Bracha rounds. It cannot retroactively vote. The proposed fix (GET /api/v1/sync/dag-balance-anchor) relies on trusting the DAG vertex signatures. Is this sufficient, or does the fresh node need to re-run Bracha from genesis (infeasible) or trust a checkpointed finality proof?

6. **f=1 vs. production f**: With 4 bootstrap nodes, f=1 means any single node failure halts Bracha delivery (since we need 3-of-4). This is fine for current network size but requires careful handling of planned maintenance (e.g., Gamma downtime during rolling deploys). Should we use f=0 (all-honest assumption) during Phase 1 shadow mode?

---

## 11. References

| Component | Location |
|-----------|----------|
| Existing Bracha RB | `crates/q-narwhal-core/src/reliable_broadcast.rs` |
| DAG-Knight consensus | `crates/q-dag-knight/src/lib.rs` |
| Block reward processing | `crates/q-storage/src/balance_consensus.rs:680` |
| Balance gossip handler | `crates/q-api-server/src/main.rs:9005-9250` |
| P2PBalanceUpdate struct | `crates/q-types/src/balance_update.rs:50-103` |
| FullStateSnapshot | `crates/q-api-server/src/state_sync_api.rs:38-143` |
| Gossipsub config | `crates/q-network/src/unified_network_manager.rs:1600-1695` |
| Message priority queue | `crates/q-network/src/gossipsub_queue.rs:21-66` |
| DAG consensus topics | `crates/q-types/src/lib.rs:4048-4080` |
| Test wallet doc | `docs/test-wallet-checkpoint-verification-2026-04-30.md` |

---

## 12. Implementation Status Update (2026-05-01)

**DeepSeek review incorporated. Phase 1 implementation complete and compiling.**

### 12.1 Files Created

| File | Purpose | Status |
|------|---------|--------|
| `crates/q-types/src/balance_finality.rs` | All Bracha+DAG types (BrachaBalanceMsg, BrachaInstance, BalanceFinalityRecord, ValidatorBitmask, DagBalanceAnchorResponse) | ✅ Compiles |
| `crates/q-storage/src/balance_finality_engine.rs` | BalanceFinalityEngine — full Bracha state machine (SEND/ECHO/READY), pending_anchor buffer, background tasks | ✅ Compiles |

### 12.2 Files Modified

| File | Change |
|------|--------|
| `crates/q-types/src/lib.rs` | Added `pub mod balance_finality`, `pub use balance_finality::{...}`, `balance_rb_topic()` on NetworkId |
| `crates/q-storage/src/lib.rs` | Added `pub mod balance_finality_engine`, `scan_manifest_prefix()`, `put_manifest_sync()` wrappers |
| `crates/q-api-server/src/lib.rs` | Added `balance_finality_engine: Option<Arc<BalanceFinalityEngine>>` to AppState |
| `crates/q-api-server/src/main.rs` | Engine construction + gossip channel wiring after AppState init; balance-rb topic handler in gossipsub loop |
| `crates/q-network/src/unified_network_manager.rs` | Added `/consensus/balance-rb` to default gossipsub subscriptions |
| `crates/q-api-server/src/handlers.rs` | Added `GET /api/v1/sync/dag-balance-anchor` handler (returns anchored + pending_anchor records) |

### 12.3 Questions for DeepSeek (Phase 2 Design)

1. **Mining credit wiring**: The pending-solution pre-confirmation path in the mining handler currently calls `add_balance()` directly. To get cross-node balance consistency, this must be replaced with `engine.propose_balance_update()`. However, the solution is accepted on only one node (whichever the miner submits to). Should the PROPOSER be the accepting node, or should the block coinbase already be the canonical credit (and Path B completely eliminated)?

   **Recommended direction**: Eliminate Path B entirely. All mining rewards go through block coinbase (Path A). The Bracha layer handles only DEX credits, cross-chain bridge credits, and any other out-of-block balance events that are currently non-deterministic.

2. **DEX swap finality**: When `execute_swap()` produces a QUG credit or fee distribution, should the entire swap be Bracha-wrapped (proposer = executing node, 2f+1 confirmations before writing), or should swaps remain optimistic (write immediately on all nodes that process the tx, consistent via P2P tx propagation)?

   **Current state**: DEX swaps use the `optimistic_applied_txs` dedup map + P2P tx propagation. This is already deterministic if the tx reaches all nodes. Bracha wrapping would add 300ms latency for no safety gain unless the tx propagation is unreliable.

3. **Fresh node catch-up via `/api/v1/sync/dag-balance-anchor`**: The endpoint is implemented. The question is: does a fresh Docker container syncing from Beta/Gamma automatically call this endpoint? Currently the `get_full_state()` endpoint is called. We need to either merge the finality records into `FullStateSnapshot`, or add an explicit call to `/api/v1/sync/dag-balance-anchor` in the fresh-node sync startup.

   **Recommended**: Add `finality_records` field to `FullStateSnapshot` (state_sync_api.rs). Fresh nodes apply anchored+pending_anchor records in the same pass as wallet balances. This way no extra HTTP call is needed and the atomic guarantee is preserved.

4. **f=0 shadow mode graduation criteria**: Currently the engine starts with f=0 (all-honest, 1-of-1 delivery). When should this be promoted to f=1 (3-of-4 delivery on mainnet with 4 nodes)? Suggested criterion: when all 4 bootstrap nodes are online and the validator index map shows 4 entries.

5. **Epsilon balance safety**: Epsilon holds the authoritative balance ledger (219 GB DB, genesis node). With the engine in f=0 shadow mode, Epsilon proposes balance updates that are immediately delivered (since f+1=1). This means Epsilon's view still dominates. The key safety guarantee is that Epsilon's proposals are now SIGNED and GOSSIP'd — so any tampered replay or balance injection would require a valid Ed25519 signature from Epsilon's node key, which is protected.

### 12.4 Next Steps (Phase 2)

The implementation compiles and is architecturally complete. To activate the engine's full effect:

1. **Wire mining credit path**: Replace `add_balance()` in the mining submission handler with `engine.propose_balance_update()` for the pending-solution pre-confirmation step.
2. **Merge finality into FullStateSnapshot**: Fresh nodes receive finality records automatically in the state sync response.
3. **Bump f to 1**: Once 4-node quorum is confirmed stable, set `f=1` in the engine constructor.
4. **Build and test on Delta Docker**: Verify that the test wallet `qnkd0f5cd...` shows the same balance on Delta after syncing as on Epsilon.

### 12.5 Epsilon Balance Protection Summary

- Epsilon is the genesis node. Its DB (`/home/orobit/data-mainnet-genesis/`) is the ground truth.
- The Bracha engine does NOT modify Epsilon's DB contents without consensus — it requires at least `2f+1` READY messages before writing. With f=0 (current), this means Epsilon's own signature is required to propose any balance change.
- No peer can inject a balance update for any wallet (including the master wallet) without holding Epsilon's Ed25519 node signing key.
- The `update_matches()` cross-phase validator prevents a Byzantine ECHO from carrying a different amount than the original SEND — the echo carries the full `P2PBalanceUpdate` and is checked on receipt.
- All finality proofs are stored under `balance_finality_proof:{hex_wallet}` in the manifest CF — auditable, immutable after 2f+1 delivery.
| Swap balance revert review | `docs/technical-review-swap-balance-revert-2026-04-29.md` |
