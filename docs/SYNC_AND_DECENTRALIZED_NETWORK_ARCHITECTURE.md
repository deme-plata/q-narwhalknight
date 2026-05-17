# Q-NarwhalKnight Sync & Decentralized Network Architecture

## Technical Review v1.0.87-beta

### Executive Summary

The Q-NarwhalKnight blockchain implements a hybrid sync architecture that enables:
1. **Data Redundancy**: Full blockchain replication across all nodes
2. **Decentralized Mining**: Users mine on localhost while connected to any node
3. **Real-time Balance Updates**: Frontend UI reflects balance changes instantly
4. **Coherent Network State**: All nodes maintain identical blockchain state

---

## 1. What Gets Synced: Block Anatomy

Every synced block contains the complete cryptographic proof chain:

```rust
pub struct QBlock {
    // Block Header - Cryptographic commitments
    pub header: BlockHeader,

    // Mining Proof-of-Work solutions
    pub mining_solutions: Vec<MiningSolution>,

    // DAG vertex references (DAG-Knight ordering)
    pub dag_parents: Vec<VertexId>,

    // Quantum consensus metadata
    pub quantum_metadata: QuantumMetadata,

    // Financial transactions
    pub transactions: Vec<Transaction>,

    // Balance state changes (deterministic across nodes)
    pub balance_updates: Vec<BalanceUpdate>,

    // Block size tracking
    pub size_bytes: usize,
}
```

### 1.1 Block Header (Cryptographic Root)

```rust
pub struct BlockHeader {
    pub height: u64,                    // Monotonically increasing
    pub phase: u8,                      // Network phase (prevents cross-phase sync)
    pub network_id: String,             // "testnet-phase8", "mainnet"
    pub prev_block_hash: [u8; 32],      // Chain backbone (Bitcoin-style)
    pub solutions_root: [u8; 32],       // Merkle root of mining solutions
    pub tx_root: [u8; 32],              // Merkle root of transactions
    pub state_root: [u8; 32],           // World state after this block
    pub timestamp: u64,                 // Unix epoch seconds
    pub dag_round: u64,                 // DAG-Knight consensus round
    pub vdf_proof: VDFProof,            // Quantum VDF anchor election
    pub anchor_validator: Option<String>, // Elected anchor for this round
    pub proposer: NodeId,               // Block creator
    pub producer_id: u8,                // Parallel producer lane (0-7)
    pub total_difficulty: u128,         // Cumulative difficulty
}
```

### 1.2 Mining Solutions (Proof-of-Work)

Each block contains valid PoW solutions that prove computational work:

```rust
pub struct MiningSolution {
    pub nonce: u64,                     // Winning nonce
    pub hash: [u8; 32],                 // Result hash (meets difficulty)
    pub difficulty_target: [u8; 32],    // Target this solution meets
    pub miner_address: [u8; 32],        // Reward recipient
    pub timestamp: u64,                 // When solution was found
}
```

### 1.3 Transactions (Value Transfer)

```rust
pub struct Transaction {
    pub id: TxHash,
    pub from: Address,                  // Sender (0x00...00 = coinbase)
    pub to: Address,                    // Recipient
    pub amount: u64,                    // QNK amount (in smallest unit)
    pub fee: u64,                       // Transaction fee
    pub nonce: u64,                     // Replay protection
    pub signature: Vec<u8>,             // Cryptographic signature
    pub timestamp: DateTime<Utc>,
    pub token_type: TokenType,          // QNK or QUGUSD
    pub tx_type: TransactionType,       // Coinbase, Transfer, Contract, etc.
}
```

### 1.4 Balance Updates (Deterministic State)

Critical for ensuring all nodes have identical balance state:

```rust
pub struct BalanceUpdate {
    pub address: String,                // Wallet address
    pub old_balance: u64,               // Balance before update
    pub new_balance: u64,               // Balance after update
    pub reason: String,                 // "mining_reward", "transfer", "dev_fee"
    pub timestamp: u64,
}
```

---

## 2. Sync Protocol Flow

### 2.1 Initial Connection

```
┌─────────────────┐         ┌─────────────────┐
│  New Node       │         │  Bootstrap Node │
│  (Server Alpha) │         │  (Server Beta)  │
│  Height: 0      │         │  Height: 456000 │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ 1. libp2p Connection      │
         │ ─────────────────────────>│
         │                           │
         │ 2. Peer Height Exchange   │
         │ <─────────────────────────│
         │    "My height: 456000"    │
         │                           │
         │ 3. BlockPackRequest       │
         │ ─────────────────────────>│
         │    start: 1, end: 2000    │
         │                           │
         │ 4. BlockPackResponse      │
         │ <─────────────────────────│
         │    [2000 blocks]          │
         │                           │
         └───────────────────────────┘
```

### 2.2 Batch Sync Pipeline (v1.0.87-beta)

The sync uses intelligent pipelining with safeguards:

```rust
// Pipeline cap prevents runaway requests
const MAX_PIPELINE_AHEAD: u64 = 15_000;

// Duplicate request prevention
if outstanding_requests.contains(start_height) {
    return; // Skip duplicate
}

// Request batching
let batch_size = 2000; // Blocks per request
let concurrent_requests = 5; // Parallel requests
```

### 2.3 Block Processing Pipeline

```
                    ┌──────────────────┐
                    │ libp2p Receive   │
                    │ BlockPackResponse│
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Validate Phase   │ ← Reject wrong network phase
                    │ & Network ID     │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │ Fast Sync Path    │         │ Fallback Path     │
    │ (SafeBatchedWriter)│        │ (Direct Write)    │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼─────────┐
                    │ Process Coinbase │ ← Extract mining rewards
                    │ Transactions     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Update Balances  │ ← wallet_balances HashMap
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Persist to       │ ← RocksDB atomic write
                    │ RocksDB          │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Update Height    │ ← current_height_atomic
                    │ Atomic           │
                    └──────────────────┘
```

---

## 3. Balance Synchronization

### 3.1 The Balance Consensus Problem

**Challenge**: How do users mining on their local node see their balance update in the frontend when connected to the bootstrap node's API?

**Solution**: Three-layer balance synchronization:

```
┌─────────────────────────────────────────────────────────┐
│                    LAYER 1: In-Memory                   │
│                                                         │
│  wallet_balances: Arc<RwLock<HashMap<Address, u64>>>   │
│                                                         │
│  - Instant updates on mining reward                     │
│  - Frontend reads this for UI display                   │
│  - Updated by coinbase transaction processing           │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Every 15 seconds
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   LAYER 2: RocksDB                      │
│                                                         │
│  Column Family: "balances"                              │
│  Key: wallet_address                                    │
│  Value: u64 balance                                     │
│                                                         │
│  - Persistent across restarts                           │
│  - Atomic batch writes                                  │
│  - Survives node crashes                                │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Via Block Sync
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  LAYER 3: Blockchain                    │
│                                                         │
│  Each block contains:                                   │
│  - Coinbase transactions (mining rewards)               │
│  - Balance update records                               │
│  - Cryptographic proof chain                            │
│                                                         │
│  - Authoritative source of truth                        │
│  - Replicated to all nodes                              │
│  - Immutable history                                    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Coinbase Transaction Flow

When a miner submits a valid solution:

```rust
// 1. Block producer creates coinbase transaction
Transaction {
    from: [0u8; 32],           // Zero address = coinbase
    to: miner_address,          // Miner's wallet
    amount: block_reward,       // 0.00008584 QNK (current)
    tx_type: TransactionType::Coinbase,
}

// 2. Block is broadcast via gossipsub
gossipsub.publish("/qnk/testnet-phase8/blocks", block);

// 3. Syncing nodes process coinbase transactions
for tx in block.transactions {
    if tx.is_coinbase() {
        // Update in-memory balance
        wallet_balances.insert(tx.to, current + tx.amount);

        // Persist to RocksDB
        storage.add_balance(&tx.to, tx.amount).await;
    }
}
```

### 3.3 Frontend Balance Display

The frontend (React/TypeScript) polls the API:

```typescript
// Frontend polling (every 2 seconds)
const response = await fetch('/api/balance?address=qnk...');
const { balance } = await response.json();

// API handler reads from in-memory HashMap
async fn get_balance(address: &str) -> u64 {
    let balances = state.wallet_balances.read().await;
    balances.get(address).copied().unwrap_or(0)
}
```

---

## 4. Post-Sync State

### 4.1 What Changes After Sync Completes

```
┌─────────────────────────────────────────────────────────┐
│                 NODE STATE AFTER SYNC                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ Blockchain Data (RocksDB)                          │
│     └── All blocks from height 1 to current            │
│     └── Complete transaction history                    │
│     └── Mining solution proofs                          │
│     └── DAG parent references                           │
│                                                         │
│  ✅ Balance State                                       │
│     └── In-memory: wallet_balances HashMap             │
│     └── On-disk: RocksDB "balances" column family      │
│     └── Identical to bootstrap node                     │
│                                                         │
│  ✅ Height Pointer                                      │
│     └── qblock:latest = current height                  │
│     └── current_height_atomic = current height          │
│                                                         │
│  ✅ DAG Consensus State                                 │
│     └── Vertex store populated                          │
│     └── DAG-Knight ordering established                 │
│     └── Ready for block production                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Mining After Sync

Once synced, the node can participate in mining:

```
┌──────────────────┐      ┌──────────────────┐
│  Local Miner     │      │  Local Node      │
│  (q-miner)       │      │  (q-api-server)  │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         │ GET /api/mining/challenge
         │ ────────────────────────>
         │                         │
         │ {height, challenge, diff}
         │ <────────────────────────
         │                         │
         │ [Compute PoW locally]   │
         │                         │
         │ POST /api/mining/submit │
         │ ────────────────────────>
         │                         │
         │                    ┌────▼────┐
         │                    │ Validate│
         │                    │ Solution│
         │                    └────┬────┘
         │                         │
         │                    ┌────▼────┐
         │                    │ Create  │
         │                    │ Block   │
         │                    └────┬────┘
         │                         │
         │                    ┌────▼────┐
         │                    │Broadcast│
         │                    │via P2P  │
         │                    └────┬────┘
         │                         │
         │ {reward: 0.00008584}   │
         │ <────────────────────────
         │                         │
└─────────────────────────────────────────┘
```

### 4.3 Balance Visibility Across Network

**Scenario**: User mines on their local node, checks balance on bootstrap node

```
┌─────────────────┐         ┌─────────────────┐
│  User's Node    │         │  Bootstrap Node │
│  (localhost)    │         │  (quillon.xyz)  │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ 1. Mine block locally     │
         │ ─────────────────────────>│
         │    [Block via gossipsub]  │
         │                           │
         │                      ┌────▼────┐
         │                      │ Process │
         │                      │ Block   │
         │                      └────┬────┘
         │                           │
         │                      ┌────▼────┐
         │                      │ Update  │
         │                      │ Balances│
         │                      └────┬────┘
         │                           │
         │ 2. Check balance          │
         │ <─────────────────────────│
         │    (via API or frontend)  │
         │                           │
         │    {balance: 415.787 QNK} │
         │ <─────────────────────────│
         │                           │
└─────────────────────────────────────────┘
```

---

## 5. Data Redundancy Architecture

### 5.1 Replication Topology

```
                    ┌─────────────────┐
                    │  Bootstrap Node │
                    │  (Primary)      │
                    │  Full Chain     │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼───────┐ ┌──────▼──────┐ ┌───────▼───────┐
    │  Node A       │ │  Node B     │ │  Node C       │
    │  Full Chain   │ │  Full Chain │ │  Full Chain   │
    │  Copy         │ │  Copy       │ │  Copy         │
    └───────────────┘ └─────────────┘ └───────────────┘
```

### 5.2 Consistency Guarantees

| Property | Guarantee |
|----------|-----------|
| Block Order | Deterministic (height-based) |
| Balance State | Eventually consistent (within ~3 seconds) |
| Transaction History | Immutable once synced |
| Mining Rewards | Exactly-once processing |

### 5.3 Failure Recovery

If a node goes offline and reconnects:

1. **Height Check**: Compare local height with network
2. **Gap Detection**: Identify missing blocks
3. **Catch-up Sync**: Request missing blocks via `BlockPackRequest`
4. **Balance Reconciliation**: Reprocess coinbase transactions
5. **Resume Mining**: Generate new mining challenges

---

## 6. Performance Characteristics

### 6.1 Sync Speed (v1.0.87-beta)

| Metric | Value |
|--------|-------|
| Blocks per batch | 2,000 |
| Concurrent requests | 5 |
| Pipeline depth | 15,000 blocks max |
| Typical sync rate | 300-500 blocks/sec |
| Full sync (456K blocks) | ~15-25 minutes |

### 6.2 Resource Usage

| Resource | Usage |
|----------|-------|
| Memory | ~500MB - 2GB during sync |
| Disk | ~10GB for full chain |
| Network | ~100 Mbps burst during sync |
| CPU | Multi-threaded block validation |

---

## 7. Security Considerations

### 7.1 Sync Safety (v1.0.87-beta)

```rust
// CRITICAL: Prevent sync-down attacks
if target_height < local_height && local_height > 1000 {
    error!("🚨 SAFETY ABORT: Refusing to sync down");
    return Err("Sync-down prevented");
}
```

### 7.2 Phase Isolation

```rust
// Blocks from wrong phase are rejected
if block.header.network_id != "testnet-phase8" {
    return Err("Phase mismatch");
}
```

### 7.3 Duplicate Prevention

```rust
// v1.0.87-beta: Skip duplicate requests
if outstanding_requests.contains(start_height) {
    debug!("Skipping duplicate request");
    return Ok(());
}
```

---

## 8. Summary

The Q-NarwhalKnight sync architecture achieves:

1. **Complete Data Redundancy**: Every node holds full blockchain
2. **Decentralized Mining**: Any node can produce blocks
3. **Real-time Balance Updates**: <3 second propagation
4. **Network Coherence**: Deterministic state across all nodes
5. **Failure Resilience**: Automatic catch-up on reconnect

This enables the core user experience: **Mine locally, see balance globally**.
