# Decentralized Mining Pool Technical Review

## Q-NarwhalKnight v2.3.0 Architecture Proposal

**Author:** Claude Code
**Date:** 2025-12-26
**Status:** Technical Design Review

---

## 1. Executive Summary

This document outlines the architecture for transforming the Q-NarwhalKnight mining pool from a centralized Stratum server to a **fully decentralized P2P mining pool** leveraging our existing libp2p-rust infrastructure. The goal is to eliminate single points of failure while maintaining fair reward distribution via PPLNS.

### Key Benefits
- **No central operator** - Pool runs as a DAO on the network itself
- **Censorship resistant** - Any node can accept miners
- **Trustless payouts** - Rewards distributed via on-chain smart contracts
- **Geographic distribution** - Miners connect to nearest pool node
- **Fault tolerant** - Pool survives individual node failures

---

## 2. Current Centralized Architecture

```
                    ┌─────────────────────────────────┐
                    │     CENTRALIZED POOL SERVER     │
                    │                                 │
   Miner A ────────►│  Stratum Server (:3333)         │
   Miner B ────────►│  Share Validator                │
   Miner C ────────►│  PPLNS Calculator               │
   Miner D ────────►│  Payout Processor               │
                    │  Worker Manager                 │
                    └─────────────────────────────────┘
                                   │
                                   ▼
                           Q-NarwhalKnight
                              Blockchain
```

### Current Limitations

| Issue | Impact |
|-------|--------|
| Single point of failure | Pool downtime = no mining |
| Geographic centralization | High latency for distant miners |
| Trust requirement | Operator controls payouts |
| Censorship risk | Operator can ban miners |
| Regulatory target | Single entity to regulate |

---

## 3. Decentralized Pool Architecture

### 3.1 High-Level Design

```
                           ┌─────────────────────────────────────────┐
                           │         GOSSIPSUB MESH NETWORK          │
                           │                                         │
    ┌──────────────┐       │   /qnk/pool/shares                      │
    │ Pool Node A  │◄─────►│   /qnk/pool/blocks-found                │
    │ Stratum:3333 │       │   /qnk/pool/pplns-state                 │
    │ Workers: 50  │       │   /qnk/pool/payouts                     │
    └──────────────┘       │                                         │
           ▲               └─────────────────────────────────────────┘
           │                              ▲           ▲
    Miners A,B,C                          │           │
                                          │           │
    ┌──────────────┐              ┌──────────────┐   │
    │ Pool Node B  │◄────────────►│ Pool Node C  │◄──┘
    │ Stratum:3333 │              │ Stratum:3333 │
    │ Workers: 75  │              │ Workers: 30  │
    └──────────────┘              └──────────────┘
           ▲                             ▲
           │                             │
    Miners D,E,F,G                  Miners H,I,J
```

### 3.2 Core Components

#### 3.2.1 Distributed Share Registry

Shares are broadcast via gossipsub and stored in a distributed hash table (DHT):

```rust
// crates/q-mining-pool/src/distributed/share_registry.rs

use libp2p::gossipsub::{Topic, TopicHash};
use libp2p::kad::{Kademlia, Record};

/// Distributed share with cryptographic proof
#[derive(Clone, Serialize, Deserialize)]
pub struct DistributedShare {
    /// Unique share ID (hash of share content)
    pub share_id: [u8; 32],

    /// Worker ID (wallet.worker_name)
    pub worker_id: WorkerId,

    /// Share difficulty
    pub difficulty: f64,

    /// Block template hash (proves work was for correct block)
    pub block_template_hash: [u8; 32],

    /// Nonce that solves the share
    pub nonce: u64,

    /// VDF proof (prevents share grinding)
    pub vdf_proof: Vec<u8>,

    /// Timestamp (unix millis)
    pub timestamp: u64,

    /// Pool node that received this share
    pub receiving_node: PeerId,

    /// Signature from receiving node (attestation)
    pub node_signature: [u8; 64],
}

/// Share registry synchronized via gossipsub + Kademlia DHT
pub struct DistributedShareRegistry {
    /// Local share cache (ring buffer, last N hours)
    local_cache: RwLock<VecDeque<DistributedShare>>,

    /// Kademlia DHT for share persistence
    kad: Arc<Mutex<Kademlia<MemoryStore>>>,

    /// Gossipsub topic for share announcements
    shares_topic: Topic,

    /// Bloom filter for duplicate detection
    seen_shares: RwLock<BloomFilter>,

    /// Our node's signing key
    node_key: ed25519::Keypair,
}

impl DistributedShareRegistry {
    /// Broadcast a new share to the network
    pub async fn broadcast_share(&self, share: DistributedShare) -> Result<()> {
        // 1. Validate share locally
        self.validate_share(&share)?;

        // 2. Sign the share as attestation
        let signed = self.sign_share(share)?;

        // 3. Store in local cache
        self.local_cache.write().push_back(signed.clone());

        // 4. Store in DHT for persistence
        let record = Record {
            key: signed.share_id.to_vec().into(),
            value: bincode::serialize(&signed)?,
            publisher: None,
            expires: Some(Instant::now() + Duration::from_hours(24)),
        };
        self.kad.lock().await.put_record(record, Quorum::Majority)?;

        // 5. Gossip to all pool nodes
        let message = bincode::serialize(&signed)?;
        self.gossipsub.publish(self.shares_topic.clone(), message)?;

        Ok(())
    }
}
```

#### 3.2.2 Distributed PPLNS State Machine

The PPLNS window is synchronized using a CRDT (Conflict-free Replicated Data Type):

```rust
// crates/q-mining-pool/src/distributed/pplns_crdt.rs

use crdts::{GCounter, Map, Orswot};

/// CRDT-based PPLNS state that merges automatically
#[derive(Clone, Serialize, Deserialize)]
pub struct DistributedPPLNS {
    /// Worker difficulty contributions (G-Counter per worker)
    /// Key: wallet_address, Value: total difficulty contributed
    worker_difficulty: Map<String, GCounter<PeerId>, PeerId>,

    /// Shares in current window (OR-Set with timestamps)
    /// Automatically handles concurrent additions
    window_shares: Orswot<ShareId, PeerId>,

    /// Current round number (last block height found)
    round_number: u64,

    /// Network difficulty at round start
    network_difficulty: f64,

    /// Window size = N * network_difficulty
    n_factor: f64,

    /// Vector clock for state versioning
    vclock: VClock<PeerId>,
}

impl DistributedPPLNS {
    /// Merge state from another node (CRDT merge is commutative + idempotent)
    pub fn merge(&mut self, other: &Self) {
        // G-Counters merge by taking max of each node's contribution
        self.worker_difficulty.merge(other.worker_difficulty.clone());

        // OR-Set merges handle concurrent add/remove correctly
        self.window_shares.merge(other.window_shares.clone());

        // Vector clock merge
        self.vclock.merge(&other.vclock);

        // Round number takes max (monotonically increasing)
        self.round_number = self.round_number.max(other.round_number);
    }

    /// Add share to PPLNS window
    pub fn add_share(&mut self, node_id: PeerId, share: &DistributedShare) {
        // Increment worker's difficulty counter
        self.worker_difficulty
            .entry(share.worker_id.wallet().to_string())
            .or_default()
            .increment(node_id, share.difficulty as u64);

        // Add to share set
        self.window_shares.add(share.share_id, node_id);

        // Increment our vector clock
        self.vclock.increment(node_id);

        // Trim window if over size
        self.trim_window();
    }

    /// Calculate rewards (deterministic from CRDT state)
    pub fn calculate_rewards(&self, block_reward: u64) -> Vec<RewardEntry> {
        let total_difficulty: u64 = self.worker_difficulty
            .iter()
            .map(|(_, counter)| counter.read())
            .sum();

        if total_difficulty == 0 {
            return vec![];
        }

        // Dev fee: 1%, Pool fee: 1.5%
        let dev_fee = block_reward * 100 / 10_000;
        let pool_fee = block_reward * 150 / 10_000;
        let miner_rewards = block_reward - dev_fee - pool_fee;

        self.worker_difficulty
            .iter()
            .map(|(wallet, counter)| {
                let difficulty = counter.read();
                let proportion = difficulty as f64 / total_difficulty as f64;
                RewardEntry {
                    wallet_address: wallet.clone(),
                    amount: (miner_rewards as f64 * proportion) as u64,
                    proportion,
                }
            })
            .filter(|r| r.amount > 0)
            .collect()
    }
}
```

#### 3.2.3 Gossipsub Topics for Pool Coordination

```rust
// crates/q-mining-pool/src/distributed/topics.rs

/// Pool-specific gossipsub topics
pub struct PoolTopics {
    /// Share announcements (high volume)
    pub shares: Topic,

    /// Block found notifications (rare, high priority)
    pub blocks_found: Topic,

    /// PPLNS state synchronization (periodic)
    pub pplns_state: Topic,

    /// Payout batch announcements
    pub payouts: Topic,

    /// Pool node heartbeats
    pub heartbeat: Topic,

    /// Block template distribution
    pub block_templates: Topic,
}

impl PoolTopics {
    pub fn new(network_id: &str) -> Self {
        Self {
            shares: Topic::new(format!("/qnk/{}/pool/shares", network_id)),
            blocks_found: Topic::new(format!("/qnk/{}/pool/blocks-found", network_id)),
            pplns_state: Topic::new(format!("/qnk/{}/pool/pplns-state", network_id)),
            payouts: Topic::new(format!("/qnk/{}/pool/payouts", network_id)),
            heartbeat: Topic::new(format!("/qnk/{}/pool/heartbeat", network_id)),
            block_templates: Topic::new(format!("/qnk/{}/pool/templates", network_id)),
        }
    }
}
```

---

## 4. Consensus Mechanisms

### 4.1 Block Found Consensus

When a miner finds a block, we need consensus that:
1. The share was valid
2. The block was submitted to the network
3. The block was accepted by the blockchain

```rust
// crates/q-mining-pool/src/distributed/block_consensus.rs

/// Block found announcement with multi-node attestation
#[derive(Clone, Serialize, Deserialize)]
pub struct BlockFoundAnnouncement {
    /// Block hash
    pub block_hash: [u8; 32],

    /// Block height
    pub height: u64,

    /// Share that found the block
    pub winning_share: DistributedShare,

    /// Block reward (from coinbase)
    pub reward: u64,

    /// Announcing node
    pub announcer: PeerId,

    /// Announcer's signature
    pub announcer_sig: [u8; 64],

    /// Attestations from other pool nodes
    pub attestations: Vec<NodeAttestation>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NodeAttestation {
    pub node_id: PeerId,
    pub timestamp: u64,
    pub signature: [u8; 64],
    pub block_verified: bool,
}

impl BlockFoundAnnouncement {
    /// Check if we have enough attestations (2f+1 for BFT)
    pub fn has_quorum(&self, total_pool_nodes: usize) -> bool {
        let required = (total_pool_nodes * 2 / 3) + 1;
        let valid_attestations = self.attestations
            .iter()
            .filter(|a| a.block_verified)
            .count();
        valid_attestations >= required
    }
}
```

### 4.2 Payout Consensus

Payouts require consensus to prevent double-payments:

```rust
// crates/q-mining-pool/src/distributed/payout_consensus.rs

/// Payout batch with multi-signature authorization
#[derive(Clone, Serialize, Deserialize)]
pub struct PayoutBatch {
    /// Batch ID (hash of contents)
    pub batch_id: [u8; 32],

    /// Round this payout is for
    pub round: u64,

    /// Individual payouts
    pub payouts: Vec<Payout>,

    /// Total amount being paid
    pub total_amount: u64,

    /// PPLNS state hash at time of calculation
    pub pplns_state_hash: [u8; 32],

    /// Multi-sig from pool nodes (threshold signature)
    pub threshold_signature: Option<ThresholdSignature>,

    /// Node signatures for this batch
    pub node_votes: Vec<PayoutVote>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PayoutVote {
    pub node_id: PeerId,
    pub approved: bool,
    pub signature: [u8; 64],
    pub timestamp: u64,
}

/// FROST threshold signature for payout authorization
#[derive(Clone, Serialize, Deserialize)]
pub struct ThresholdSignature {
    /// Aggregated signature (t-of-n)
    pub signature: [u8; 64],

    /// Participating signers
    pub signers: Vec<PeerId>,

    /// Threshold required
    pub threshold: usize,
}
```

---

## 5. Integration with Existing libp2p Infrastructure

### 5.1 Leveraging q-network Crate

Our existing `q-network` crate provides the foundation:

```rust
// crates/q-network/src/pool_integration.rs

use crate::gossipsub::GossipsubNetwork;
use crate::kademlia::KademliaNetwork;

/// Extend existing network with pool functionality
impl GossipsubNetwork {
    /// Subscribe to all pool topics
    pub async fn join_mining_pool(&mut self, network_id: &str) -> Result<PoolTopics> {
        let topics = PoolTopics::new(network_id);

        // Subscribe to each topic
        self.subscribe(&topics.shares)?;
        self.subscribe(&topics.blocks_found)?;
        self.subscribe(&topics.pplns_state)?;
        self.subscribe(&topics.payouts)?;
        self.subscribe(&topics.heartbeat)?;
        self.subscribe(&topics.block_templates)?;

        info!("📡 Joined decentralized mining pool network");

        Ok(topics)
    }

    /// Configure gossipsub for pool traffic patterns
    pub fn configure_for_pool(&mut self) {
        // Shares topic: high volume, can tolerate some loss
        self.set_topic_params(&topics.shares, TopicParams {
            mesh_size: 8,
            mesh_low: 4,
            mesh_high: 12,
            gossip_factor: 0.25,  // Lower gossip for high-volume
            history_length: 5,
            heartbeat_interval: Duration::from_secs(1),
        });

        // Blocks found: low volume, critical
        self.set_topic_params(&topics.blocks_found, TopicParams {
            mesh_size: 12,
            mesh_low: 8,
            mesh_high: 16,
            gossip_factor: 1.0,  // Maximum redundancy
            history_length: 10,
            heartbeat_interval: Duration::from_millis(500),
        });
    }
}
```

### 5.2 Pool Node Discovery via Kademlia

```rust
// crates/q-mining-pool/src/distributed/discovery.rs

/// Discover pool nodes via Kademlia DHT
pub struct PoolNodeDiscovery {
    kad: Arc<Mutex<Kademlia<MemoryStore>>>,
    known_pool_nodes: RwLock<HashMap<PeerId, PoolNodeInfo>>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PoolNodeInfo {
    pub peer_id: PeerId,
    pub stratum_port: u16,
    pub multiaddrs: Vec<Multiaddr>,
    pub worker_count: u32,
    pub hashrate: f64,
    pub uptime: Duration,
    pub region: String,
    pub last_seen: Instant,
}

impl PoolNodeDiscovery {
    /// Register ourselves as a pool node
    pub async fn register_as_pool_node(&self, info: PoolNodeInfo) -> Result<()> {
        // Store under well-known key in DHT
        let key = format!("/qnk/pool/nodes/{}", info.peer_id);
        let record = Record {
            key: key.into_bytes().into(),
            value: bincode::serialize(&info)?,
            publisher: Some(info.peer_id),
            expires: Some(Instant::now() + Duration::from_hours(1)),
        };
        self.kad.lock().await.put_record(record, Quorum::One)?;
        Ok(())
    }

    /// Find nearest pool nodes (for miner connection)
    pub async fn find_nearest_pool_nodes(&self, count: usize) -> Vec<PoolNodeInfo> {
        // Query DHT for pool nodes
        let key = b"/qnk/pool/nodes/".to_vec();
        let records = self.kad.lock().await
            .get_record(key.into())
            .await;

        // Parse and sort by latency
        let mut nodes: Vec<PoolNodeInfo> = records
            .filter_map(|r| bincode::deserialize(&r.value).ok())
            .collect();

        // Measure latency to each
        for node in &mut nodes {
            node.latency = self.measure_latency(&node.peer_id).await;
        }

        nodes.sort_by(|a, b| a.latency.cmp(&b.latency));
        nodes.truncate(count);
        nodes
    }
}
```

---

## 6. Anti-Cheat Mechanisms

### 6.1 Share Validation Proofs

Every share must include a VDF proof to prevent:
- Share grinding (pre-computing shares)
- Share withholding attacks
- Timestamp manipulation

```rust
// crates/q-mining-pool/src/distributed/share_proof.rs

/// Share proof using Genus-2 VDF
pub struct ShareProof {
    /// VDF input (hash of block template + nonce)
    pub input: [u8; 32],

    /// VDF output
    pub output: [u8; 32],

    /// VDF proof (allows fast verification)
    pub proof: Vec<u8>,

    /// Required iterations (based on share difficulty)
    pub iterations: u64,
}

impl ShareProof {
    /// Verify the VDF proof (O(log n) verification)
    pub fn verify(&self) -> bool {
        use q_vdf::genus2::Genus2VDF;

        let vdf = Genus2VDF::new();
        vdf.verify(&self.input, &self.output, &self.proof, self.iterations)
    }
}

/// Validate share is not pre-computed
pub fn validate_share_timing(share: &DistributedShare) -> bool {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Share must be within 30 seconds of current time
    let age = now.saturating_sub(share.timestamp);
    age < 30_000
}
```

### 6.2 Sybil Resistance for Pool Operators

Pool operators must stake QUG to participate:

```rust
// crates/q-mining-pool/src/distributed/operator_stake.rs

/// Pool operator registration with stake requirement
pub struct PoolOperatorRegistry {
    /// Minimum stake to operate a pool node (100 QUG)
    pub min_stake: u64,

    /// Registered operators
    pub operators: HashMap<PeerId, OperatorRegistration>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorRegistration {
    pub peer_id: PeerId,
    pub stake_tx: TransactionId,
    pub stake_amount: u64,
    pub registered_at: u64,
    pub slashable_until: u64,
}

impl PoolOperatorRegistry {
    /// Check if peer is a valid pool operator
    pub fn is_valid_operator(&self, peer_id: &PeerId) -> bool {
        self.operators.get(peer_id)
            .map(|r| r.stake_amount >= self.min_stake)
            .unwrap_or(false)
    }

    /// Slash operator for misbehavior
    pub async fn slash_operator(&mut self, peer_id: &PeerId, reason: SlashReason) -> Result<()> {
        if let Some(registration) = self.operators.get(peer_id) {
            // Create slash transaction
            let slash_amount = match reason {
                SlashReason::FakeShare => registration.stake_amount / 10,  // 10%
                SlashReason::DoublePayout => registration.stake_amount,     // 100%
                SlashReason::Downtime => registration.stake_amount / 100,   // 1%
            };

            // Submit slash to blockchain
            self.submit_slash_tx(peer_id, slash_amount).await?;
        }
        Ok(())
    }
}
```

---

## 7. Miner Experience

### 7.1 Automatic Pool Node Selection

Miners connect to the nearest responsive pool node:

```rust
// q-miner enhancement for decentralized pool

/// Connect to decentralized pool
pub async fn connect_to_decentralized_pool(wallet: &str) -> Result<StratumConnection> {
    // 1. Query bootstrap nodes for pool node list
    let pool_nodes = discover_pool_nodes().await?;

    // 2. Measure latency to each
    let mut ranked: Vec<(PoolNodeInfo, Duration)> = vec![];
    for node in pool_nodes {
        if let Ok(latency) = measure_stratum_latency(&node).await {
            ranked.push((node, latency));
        }
    }
    ranked.sort_by_key(|(_, lat)| *lat);

    // 3. Connect to best node with fallback
    for (node, latency) in ranked.iter().take(3) {
        match connect_stratum(&node.stratum_addr()).await {
            Ok(conn) => {
                info!("Connected to pool node {} ({}ms)", node.peer_id, latency.as_millis());
                return Ok(conn);
            }
            Err(e) => {
                warn!("Failed to connect to {}: {}", node.peer_id, e);
            }
        }
    }

    Err(anyhow!("No pool nodes available"))
}
```

### 7.2 Failover Between Pool Nodes

```rust
/// Miner connection with automatic failover
pub struct ResilientPoolConnection {
    primary: Option<StratumConnection>,
    backup_nodes: Vec<PoolNodeInfo>,
    current_job: Arc<RwLock<Option<MiningJob>>>,
}

impl ResilientPoolConnection {
    /// Handle connection failure with seamless failover
    pub async fn handle_disconnect(&mut self) -> Result<()> {
        warn!("Primary pool node disconnected, failing over...");

        // Try backup nodes
        for node in &self.backup_nodes {
            match connect_stratum(&node.stratum_addr()).await {
                Ok(conn) => {
                    self.primary = Some(conn);

                    // Resume with current job (shares still valid)
                    info!("Failover successful to {}", node.peer_id);
                    return Ok(());
                }
                Err(_) => continue,
            }
        }

        // Refresh node list if all backups failed
        self.backup_nodes = discover_pool_nodes().await?;
        Err(anyhow!("All pool nodes unreachable"))
    }
}
```

---

## 8. Implementation Phases

### Phase 1: P2P Share Broadcasting (v2.3.0)
- [ ] Add gossipsub topics for pool coordination
- [ ] Implement DistributedShare structure with VDF proofs
- [ ] Share validation across multiple nodes
- [ ] Basic CRDT-based PPLNS state

### Phase 2: Multi-Node Stratum (v2.4.0)
- [ ] Pool node discovery via Kademlia DHT
- [ ] Miner auto-selection of nearest pool node
- [ ] Failover between pool nodes
- [ ] PPLNS state synchronization

### Phase 3: Decentralized Payouts (v2.5.0)
- [ ] FROST threshold signatures for payout authorization
- [ ] On-chain payout smart contract
- [ ] Operator stake/slash mechanism
- [ ] Trustless reward distribution

### Phase 4: Full Decentralization (v3.0.0)
- [ ] Remove any centralized coordinator
- [ ] DAO governance for pool parameters
- [ ] Geographic distribution optimization
- [ ] Cross-pool interoperability

---

## 9. Performance Considerations

### 9.1 Bandwidth Requirements

| Component | Messages/sec | Size/msg | Bandwidth |
|-----------|-------------|----------|-----------|
| Share announcements | 1000 | 256 bytes | 250 KB/s |
| PPLNS state sync | 0.1 | 10 KB | 1 KB/s |
| Block templates | 0.2 | 2 KB | 0.4 KB/s |
| Heartbeats | 1 | 128 bytes | 0.1 KB/s |
| **Total** | | | **~252 KB/s** |

### 9.2 Latency Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Share propagation | <100ms | 95th percentile |
| Block found broadcast | <50ms | Critical path |
| PPLNS state convergence | <5s | After block found |
| Failover time | <2s | Miner reconnection |

---

## 10. Security Analysis

### 10.1 Threat Model

| Threat | Mitigation |
|--------|------------|
| Fake shares | VDF proofs + multi-node validation |
| Share withholding | Random share auditing |
| Double payouts | Threshold signatures |
| Sybil pool nodes | Stake requirement (100 QUG) |
| Eclipse attacks | Diverse peer connections |
| Timestamp manipulation | NTP + relative ordering |

### 10.2 Cryptographic Primitives

- **Share proofs**: Genus-2 Jacobian VDF
- **Node signatures**: Ed25519 (fast) or Dilithium5 (PQ)
- **Threshold signatures**: FROST (Flexible Round-Optimized Schnorr Threshold)
- **State hashing**: Blake3
- **Merkle proofs**: SHA3-256

---

## 11. Conclusion

By leveraging our existing libp2p infrastructure, we can transform the Q-NarwhalKnight mining pool into a truly decentralized system. The key innovations are:

1. **CRDT-based PPLNS** - Conflict-free state synchronization
2. **VDF share proofs** - Anti-grinding protection
3. **Threshold payouts** - Trustless reward distribution
4. **DHT-based discovery** - Geographic optimization

This architecture maintains the simplicity of pool mining for end users while eliminating the trust and censorship concerns of centralized pools.

---

## Appendix A: Message Formats

```protobuf
// pool_messages.proto

message DistributedShare {
  bytes share_id = 1;
  string worker_id = 2;
  double difficulty = 3;
  bytes block_template_hash = 4;
  uint64 nonce = 5;
  bytes vdf_proof = 6;
  uint64 timestamp = 7;
  bytes receiving_node = 8;
  bytes node_signature = 9;
}

message BlockFoundAnnouncement {
  bytes block_hash = 1;
  uint64 height = 2;
  DistributedShare winning_share = 3;
  uint64 reward = 4;
  bytes announcer = 5;
  bytes announcer_sig = 6;
  repeated NodeAttestation attestations = 7;
}

message PPLNSState {
  map<string, uint64> worker_difficulty = 1;
  repeated bytes window_share_ids = 2;
  uint64 round_number = 3;
  double network_difficulty = 4;
  bytes state_hash = 5;
  bytes vector_clock = 6;
}
```

---

## Appendix B: Configuration

```toml
# pool_node.toml

[pool]
name = "Q-NarwhalKnight Decentralized Pool"
network_id = "mainnet"

[pool.operator]
# Stake transaction (required)
stake_tx = "abc123..."
# Minimum 100 QUG stake
min_stake = 100_000_000_000

[pool.stratum]
port = 3333
max_connections = 10000

[pool.gossipsub]
# Share topic configuration
share_mesh_size = 8
share_gossip_factor = 0.25

# Block found topic configuration
block_mesh_size = 12
block_gossip_factor = 1.0

[pool.pplns]
n_factor = 2.0
sync_interval_ms = 1000

[pool.payouts]
# FROST threshold (t-of-n)
threshold = 5
total_signers = 7
min_payout = 100_000_000  # 0.1 QUG
```
