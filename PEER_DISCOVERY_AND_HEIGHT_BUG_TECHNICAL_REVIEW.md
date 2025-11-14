# Technical Review: "No New Peers to Process" Bug & Height Advancement Failure

**Status:** CRITICAL - Affecting User Nodes (Not Bootstrap Node)
**Severity:** P0 - Prevents Local Mining and Chain Progression
**First Reported:** Multiple user reports, consistent reproduction
**Last Updated:** 2025-11-14

---

## Executive Summary

Users report a critical bug where their nodes fail to discover peers and cannot advance local blockchain height, despite successfully receiving blocks via gossipsub. The bootstrap node (185.182.185.227) functions correctly, but user nodes exhibit two distinct but related failure modes:

1. **Peer Discovery Failure:** Debug log shows "no new peers to process"
2. **Height Advancement Failure:** Local height stuck at 1 while claiming to be synced at network height

This creates a **false sync state** where nodes appear healthy but cannot mine or produce blocks locally.

---

## Bug Profile

### Symptom 1: Peer Discovery Failure

**Log Evidence:**
```
DEBUG no new peers to process
```

**Characteristics:**
- Appears repeatedly in debug logs
- Node fails to discover or connect to peers
- Even when bootstrap peer is specified correctly
- Network connectivity exists (can reach bootstrap node)

### Symptom 2: Height Advancement Failure (Sequential Processing Bug)

**Log Evidence:**
```
✅ [SYNCED] Height: 78390 (fully synced)
📨 Gossipsub BLOCK...height=78390
✅ Producer #6: Created block at height 1
⚠️ [v1.0.1-beta] Block created but height NOT advanced - MUST call advance_height()
```

**Characteristics:**
- Node successfully receives blocks via gossipsub
- Reports "SYNCED" status at network height (e.g., 78,390)
- Local height remains stuck at height 1
- Block producers create blocks but height never advances
- Warning about missing `advance_height()` call

---

## Environment Comparison

### Working Environment (Bootstrap Node)

**Node:** 185.182.185.227
**Configuration:**
- Network ID: testnet-phase11
- Peer ID: 12D3KooWDs4efUkG8TewmRotHegCi9FpjyUM4V9DGruzx3nkyWUv
- P2P Port: 9001
- API Port: 8080
- Status: ✅ Fully operational

**Evidence of Correct Operation:**
```bash
curl http://185.182.185.227:8080/api/v1/status
{
  "peer_id": "12D3KooWDs4efUkG8TewmRotHegCi9FpjyUM4V9DGruzx3nkyWUv",
  "network_id": "testnet-phase11",
  "version": "v0.9.103-beta",
  "bootstrap_node": true,
  "status": "ready"
}
```

**Mining Challenges Work:**
```bash
curl http://185.182.185.227:8080/api/v1/mining/challenge
{
  "success": true,
  "data": {
    "block_height": 78081,  # Current network height
    "challenge_hash": "...",
    "difficulty_target": "..."
  }
}
```

### Failing Environment (User Nodes)

**Symptoms:**
1. Peer discovery fails ("no new peers to process")
2. Height stuck at 1 despite receiving network blocks
3. False sync status (claims synced but local height frozen)
4. Mining challenges for height 1 (useless)

**Configuration Attempts:**
- Bootstrap peer correctly specified
- Network ID matches
- Firewall rules checked
- Direct connectivity to bootstrap node verified

---

## Code Analysis

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Node                            │
│                                                         │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │  Peer Discovery  │      │  Block Reception │       │
│  │   (libp2p mdns)  │      │   (gossipsub)    │       │
│  └────────┬─────────┘      └────────┬─────────┘       │
│           │                          │                  │
│           │ "no new peers"           │ ✅ Works         │
│           ▼                          ▼                  │
│  ┌──────────────────┐      ┌──────────────────┐       │
│  │  Peer Manager    │      │  Block Validator │       │
│  │  (Empty)         │      │  (Processes OK)  │       │
│  └────────┬─────────┘      └────────┬─────────┘       │
│           │                          │                  │
│           ▼                          ▼                  │
│  ┌──────────────────────────────────────────┐         │
│  │  Sequential Block Producer                │         │
│  │  - Creates blocks at height 1              │         │
│  │  - ⚠️ advance_height() never called       │         │
│  │  - Local height stuck at 1                 │         │
│  └──────────────────────────────────────────┘         │
│                                                         │
│  Status: ✅ [SYNCED] Height: 78390 (FALSE)            │
│  Reality: Local height = 1 (STUCK)                     │
└─────────────────────────────────────────────────────────┘
```

### Critical Code Paths

#### 1. Peer Discovery System

**Location:** Likely in `crates/q-network/src/` (UnifiedNetworkManager)

**Expected Behavior:**
```rust
// On startup
1. Initialize libp2p Swarm with mDNS + Kademlia
2. Connect to bootstrap peers
3. Discover additional peers via DHT
4. Maintain peer connections
```

**Actual Behavior (User Nodes):**
```rust
// Logs show:
DEBUG no new peers to process
// Meaning:
- mDNS discovery returns empty results
- Bootstrap connection fails or doesn't propagate
- DHT queries return no peers
- Peer list remains empty
```

#### 2. Height Advancement System

**Location:** `crates/q-api-server/src/block_producer.rs`

**Expected Behavior:**
```rust
async fn produce_block() {
    // 1. Create block at current height
    let block = create_block_at_height(current_height);

    // 2. Validate and store block
    storage.store_block(block).await?;

    // 3. CRITICAL: Advance height for next block
    storage.advance_height().await?;  // ← THIS IS MISSING
    current_height_atomic.fetch_add(1, Ordering::SeqCst);
}
```

**Actual Behavior (User Nodes):**
```rust
async fn produce_block() {
    // 1. Create block at current height
    let block = create_block_at_height(1);  // Always height 1

    // 2. Store block (succeeds)
    storage.store_block(block).await?;

    // 3. Height advancement NEVER HAPPENS
    // ⚠️ Warning logged but no action taken
    warn!("Block created but height NOT advanced - MUST call advance_height()");

    // 4. Next iteration still at height 1
}
```

---

## Root Cause Analysis

### Hypothesis 1: Bootstrap Peer Connection Failure

**Theory:** User nodes cannot establish initial connection to bootstrap peer

**Evidence FOR:**
- Debug log: "no new peers to process"
- Peer discovery fails immediately

**Evidence AGAINST:**
- Network connectivity verified (can curl bootstrap node)
- Bootstrap peer address correctly specified
- No firewall blocking (tested)

**Verdict:** Unlikely to be the primary cause, but may be a contributing factor

### Hypothesis 2: mDNS Discovery Scoped to Local Network

**Theory:** mDNS only discovers peers on same local network, bootstrap peer ignored

**Evidence FOR:**
- mDNS (multicast DNS) is designed for LAN discovery
- Bootstrap peer on different network may not be discovered via mDNS
- "no new peers" suggests mDNS returned empty list

**Evidence AGAINST:**
- Bootstrap peer should be added explicitly, not via mDNS
- libp2p supports direct peer connections

**Verdict:** HIGHLY LIKELY - mDNS misconfiguration or bootstrap peer not added to peer list

### Hypothesis 3: Missing advance_height() Call in Sequential Producer

**Theory:** Block producer creates blocks but never advances height counter

**Evidence FOR:**
- Explicit warning in logs: "MUST call advance_height()"
- Local height stuck at 1 across all user nodes
- Block creation succeeds but height frozen
- Warning format: `[v1.0.1-beta]` suggests old code

**Evidence AGAINST:**
- None - this is clearly a bug

**Verdict:** CONFIRMED BUG - `advance_height()` is not called after block creation

### Hypothesis 4: False Sync Status from Passive Reception

**Theory:** Node reports synced based on gossipsub reception, not local production

**Evidence FOR:**
- Node receives blocks via gossipsub (passive)
- Reports sync status based on received block height
- Local production stuck at height 1 (active)
- Creates false impression of healthy sync

**Evidence AGAINST:**
- None - this is observable behavior

**Verdict:** CONFIRMED - Sync status is misleading

---

## Technical Deep Dive

### Part 1: Peer Discovery Failure

#### Expected libp2p Initialization

```rust
// crates/q-network/src/unified_network_manager.rs (hypothetical)

pub async fn new(config: NetworkConfig) -> Result<Self> {
    // 1. Create libp2p Swarm
    let mut swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(tcp::Config::default(), noise::Config::new, yamux::Config::default)?
        .with_behaviour(|key| {
            // 2. Add mDNS for local peer discovery
            let mdns = mdns::tokio::Behaviour::new(mdns::Config::default(), key.public().to_peer_id())?;

            // 3. Add Kademlia DHT for global peer discovery
            let mut kad_config = KademliaConfig::default();
            let kad = Kademlia::new(key.public().to_peer_id(), MemoryStore::new(key.public().to_peer_id()));

            // 4. Add Gossipsub for message propagation
            let gossipsub = gossipsub::Behaviour::new(...)?;

            Behaviour { mdns, kad, gossipsub }
        })?
        .build();

    // 5. CRITICAL: Add bootstrap peers manually
    for bootstrap_addr in config.bootstrap_peers {
        swarm.dial(bootstrap_addr)?;  // ← MUST HAPPEN
    }

    Ok(Self { swarm })
}
```

#### Bug Location (User Nodes)

**Possible Issue #1: Bootstrap Peer Not Dialed**
```rust
// Bootstrap peer specified in config but never dialed
let bootstrap_peer = "/ip4/185.182.185.227/tcp/9001/p2p/12D3KooW...";
// ❌ Missing: swarm.dial(bootstrap_peer)?;
```

**Possible Issue #2: mDNS Only Mode**
```rust
// Only relying on mDNS discovery (local network only)
// When no local peers exist, discovery returns empty
if let Some(mdns_event) = mdns.next().await {
    match mdns_event {
        MdnsEvent::Discovered(peers) => {
            debug!("mDNS discovered {} peers", peers.len());
            if peers.is_empty() {
                debug!("no new peers to process");  // ← THIS LOG
            }
        }
    }
}
// ❌ Never tries bootstrap peer from config
```

**Possible Issue #3: Network ID Mismatch**
```rust
// Bootstrap peer has network ID "testnet-phase11"
// User node has network ID "testnet" (default)
// Peer rejection due to protocol mismatch
```

### Part 2: Height Advancement Failure

#### Expected Sequential Block Production

```rust
// crates/q-api-server/src/block_producer.rs

pub struct SequentialBlockProducer {
    storage: Arc<StorageEngine>,
    current_height: Arc<AtomicU64>,
}

impl SequentialBlockProducer {
    pub async fn produce_block(&self, producer_id: u8) -> Result<Block> {
        // 1. Load current height
        let height = self.current_height.load(Ordering::SeqCst);

        // 2. Create block at current height
        let block = Block::new(
            height,
            producer_id,
            Vec::new(), // transactions
            chrono::Utc::now(),
        );

        // 3. Store block in database
        self.storage.store_block(&block).await?;

        // 4. ✅ MUST ADVANCE HEIGHT FOR NEXT ROUND
        self.storage.advance_height(height + 1).await?;
        self.current_height.fetch_add(1, Ordering::SeqCst);

        info!("✅ Producer #{}: Created block at height {} and advanced to {}",
              producer_id, height, height + 1);

        Ok(block)
    }
}
```

#### Actual Buggy Implementation (User Nodes)

```rust
// ❌ BUG: advance_height() never called

pub async fn produce_block(&self, producer_id: u8) -> Result<Block> {
    let height = self.current_height.load(Ordering::SeqCst);  // Always 1

    let block = Block::new(height, producer_id, Vec::new(), chrono::Utc::now());

    self.storage.store_block(&block).await?;

    // ❌ MISSING: self.storage.advance_height(height + 1).await?;
    // ❌ MISSING: self.current_height.fetch_add(1, Ordering::SeqCst);

    warn!("⚠️ [v1.0.1-beta] Block created but height NOT advanced - MUST call advance_height()");

    // Next iteration: height still 1

    Ok(block)
}
```

**Why This Happens:**

**Theory A: Dead Code Path**
```rust
// advance_height() exists but is never called
// Possibly in a conditional branch that's never taken
if some_condition_that_never_happens {
    self.storage.advance_height(height + 1).await?;
}
```

**Theory B: Missing Implementation**
```rust
// Function exists but is a no-op
pub async fn advance_height(&self, new_height: u64) -> Result<()> {
    // TODO: Implement height advancement
    warn!("advance_height() called but not implemented");
    Ok(())
}
```

**Theory C: Async Timing Issue**
```rust
// advance_height() called but doesn't await
let _ = self.storage.advance_height(height + 1);  // ❌ Not awaited
// Height update lost
```

---

## Impact Analysis

### User Experience Impact

**Scenario:** User starts node for mining

1. **Start node:** `./q-api-server --port 8080`
2. **Check status:** Node reports "SYNCED" at height 78,390
3. **Start miner:** `./q-miner --wallet qnkXXX --server localhost:8080`
4. **Miner gets challenge:** Challenge for height 1 (not 78,390)
5. **Miner solves:** Solution submitted for height 1
6. **Network rejects:** Height 1 is ancient (network at 78,390)
7. **Result:** 100% rejection rate, zero rewards

**User Perception:** "Mining doesn't work on localhost, only bootstrap works"

### Network Impact

**Current State:**
- Bootstrap node: 1 working producer (produces valid blocks)
- User nodes: ~10-100 nodes (all stuck at height 1)
- Network consensus: Works (bootstrap node drives chain)
- Mining participation: Near zero (user mining broken)

**Long-term Risk:**
- Network centralization (only bootstrap node mines)
- User frustration (can't participate despite setup)
- Reduced network security (single point of failure)

---

## Comparison with Bootstrap Node

### Why Bootstrap Node Works

**Configuration Differences:**

1. **Peer Discovery:**
   ```
   Bootstrap: No peers needed (IS the bootstrap)
   User Node: Needs bootstrap peer (fails to connect)
   ```

2. **Height Advancement:**
   ```
   Bootstrap: advance_height() works correctly
   User Node: advance_height() never called (bug)
   ```

3. **Binary Version:**
   ```
   Bootstrap: v0.9.103-beta (working)
   User Node: v0.9.90-beta-testnet (broken)
   ```

4. **Network Role:**
   ```
   Bootstrap: bootstrap_node=true (authoritative)
   User Node: bootstrap_node=false (follower)
   ```

### Code Path Divergence

**Hypothesis:** Bootstrap node uses different block production code path

```rust
// Bootstrap node (working):
if self.is_bootstrap_node {
    // Production code path with advance_height()
    self.produce_block_with_advancement().await?;
} else {
    // Follower code path WITHOUT advance_height()
    self.produce_block_without_advancement().await?;  // ❌ BUG
}
```

**Evidence:**
- Bootstrap node at height 78,390+ (continuously advancing)
- User nodes stuck at height 1 (never advancing)
- Both use same binary version (allegedly)
- Different behavior suggests different code paths

---

## Proposed Fixes

### Fix 1: Peer Discovery (Immediate - P0)

**Location:** `crates/q-network/src/unified_network_manager.rs`

**Change Required:**
```rust
pub async fn new(config: NetworkConfig) -> Result<Self> {
    let mut swarm = /* ... */;

    // ✅ FIX: Explicitly dial bootstrap peers on startup
    for bootstrap_addr in config.bootstrap_peers {
        info!("🔗 Dialing bootstrap peer: {}", bootstrap_addr);
        match swarm.dial(bootstrap_addr.clone()) {
            Ok(_) => info!("✅ Bootstrap dial initiated"),
            Err(e) => error!("❌ Failed to dial bootstrap: {}", e),
        }
    }

    // ✅ FIX: Add bootstrap peers to Kademlia routing table
    if let Some(kad) = swarm.behaviour_mut().kad.as_mut() {
        for bootstrap_addr in config.bootstrap_peers {
            if let Some(peer_id) = extract_peer_id(&bootstrap_addr) {
                kad.add_address(&peer_id, bootstrap_addr.clone());
            }
        }
    }

    Ok(Self { swarm })
}
```

**Expected Result:**
- Bootstrap peer connection established
- DHT queries succeed
- Additional peers discovered
- "no new peers" log disappears

### Fix 2: Height Advancement (Critical - P0)

**Location:** `crates/q-api-server/src/block_producer.rs`

**Change Required:**
```rust
pub async fn produce_block(&self, producer_id: u8) -> Result<Block> {
    let height = self.current_height.load(Ordering::SeqCst);

    let block = Block::new(height, producer_id, Vec::new(), chrono::Utc::now());

    self.storage.store_block(&block).await?;

    // ✅ FIX: ALWAYS call advance_height() after successful block storage
    self.storage.advance_height(height + 1).await?;
    self.current_height.fetch_add(1, Ordering::SeqCst);

    info!("✅ Producer #{}: Created block at height {} → {}",
          producer_id, height, height + 1);

    Ok(block)
}
```

**Alternative Fix (If advance_height() is missing):**
```rust
// In StorageEngine
pub async fn advance_height(&self, new_height: u64) -> Result<()> {
    // ✅ Implement proper height advancement
    self.db.put(b"current_height", &new_height.to_le_bytes())?;
    self.height_cache.store(new_height, Ordering::SeqCst);

    info!("📈 Height advanced to {}", new_height);
    Ok(())
}
```

**Expected Result:**
- Local height advances from 1 → 2 → 3 → ...
- Block producers create sequential blocks
- Mining challenges reflect current local height
- Node can participate in consensus

### Fix 3: Sync Status Accuracy (Important - P1)

**Location:** `crates/q-api-server/src/handlers.rs` (status endpoint)

**Change Required:**
```rust
pub async fn get_status(State(state): State<Arc<AppState>>) -> Json<ApiResponse> {
    let local_height = state.current_height_atomic.load(Ordering::Acquire);
    let network_height = state.highest_network_height.load(Ordering::Acquire);

    // ✅ FIX: Accurate sync status calculation
    let is_synced = if network_height == 0 {
        false  // Network height unknown
    } else {
        let blocks_behind = network_height.saturating_sub(local_height);
        blocks_behind <= 10  // Consider synced if within 10 blocks
    };

    let status_text = if !is_synced {
        format!("SYNCING ({} blocks behind)", network_height - local_height)
    } else {
        "SYNCED".to_string()
    };

    Json(ApiResponse::success(StatusResponse {
        local_height,      // ← Report BOTH heights
        network_height,    // ← Don't hide the truth
        is_synced,
        status: status_text,
        // ...
    }))
}
```

**Expected Result:**
- Status correctly reports "SYNCING (78389 blocks behind)"
- Users understand node is not actually synced
- Mining disabled until true sync achieved (via P0 hotfix)

---

## Testing Strategy

### Test Case 1: Fresh Node Startup

**Setup:**
1. Clean data directory
2. Configure bootstrap peer
3. Start node

**Expected Behavior:**
```
[INFO] 🔗 Dialing bootstrap peer: /ip4/185.182.185.227/tcp/9001/p2p/...
[INFO] ✅ Bootstrap dial initiated
[INFO] 📡 Discovered 1 peer via gossipsub
[INFO] 📈 Height advanced to 2
[INFO] 📈 Height advanced to 3
[INFO] ✅ [SYNCING] Height: 3 (78387 blocks behind)
```

**Success Criteria:**
- No "no new peers to process" log
- Height advances beyond 1
- Sync status accurate

### Test Case 2: Mining Challenge Validity

**Setup:**
1. Start node
2. Wait for sync (may take time)
3. Request mining challenge

**Expected Behavior:**
```bash
curl http://localhost:8080/api/v1/mining/challenge

# When syncing (local=5, network=78390):
{
  "success": false,
  "error": "Node is syncing: 78385 blocks behind network..."
}

# When synced (local=78390, network=78390):
{
  "success": true,
  "data": {
    "block_height": 78390,  # ← Current local height
    "challenge_hash": "...",
    "difficulty_target": "..."
  }
}
```

**Success Criteria:**
- Mining blocked during sync
- Challenge height matches local height when synced
- No stale challenges (height 1)

### Test Case 3: Block Production Continuity

**Setup:**
1. Start node
2. Monitor block production
3. Check height progression

**Expected Behavior:**
```
[INFO] ✅ Producer #1: Created block at height 1 → 2
[INFO] ✅ Producer #2: Created block at height 2 → 3
[INFO] ✅ Producer #3: Created block at height 3 → 4
[INFO] ✅ Producer #4: Created block at height 4 → 5
```

**Success Criteria:**
- Height advances after each block
- No "height NOT advanced" warnings
- Continuous sequential progression

---

## Diagnostic Commands

### For User to Run on Failing Node

```bash
# 1. Check peer discovery
echo "=== Peer Discovery Test ==="
grep "no new peers" /var/log/q-api-server/latest.log

# 2. Check height advancement
echo "=== Height Advancement Test ==="
grep "height NOT advanced" /var/log/q-api-server/latest.log

# 3. Check local vs network height
echo "=== Sync Status Test ==="
curl -s http://localhost:8080/api/v1/status | jq '.data | {local_height, network_height, is_synced}'

# 4. Check mining challenge
echo "=== Mining Challenge Test ==="
curl -s http://localhost:8080/api/v1/mining/challenge | jq '.data.block_height // .error'

# 5. Check bootstrap connectivity
echo "=== Bootstrap Connectivity Test ==="
curl -s http://185.182.185.227:8080/api/v1/status | jq '.data.peer_id'

# 6. Check libp2p peer connections
echo "=== P2P Peer Test ==="
grep "Discovered.*peer" /var/log/q-api-server/latest.log | tail -10
```

### Expected Healthy Output

```bash
=== Peer Discovery Test ===
# (empty - no "no new peers" logs)

=== Height Advancement Test ===
# (empty - no "height NOT advanced" warnings)

=== Sync Status Test ===
{
  "local_height": 78390,
  "network_height": 78390,
  "is_synced": true
}

=== Mining Challenge Test ===
78390

=== Bootstrap Connectivity Test ===
"12D3KooWDs4efUkG8TewmRotHegCi9FpjyUM4V9DGruzx3nkyWUv"

=== P2P Peer Test ===
[INFO] 📡 Discovered 1 peer via gossipsub
[INFO] 🔗 Connected to peer: 12D3KooW...
```

### Expected Buggy Output (Current User Experience)

```bash
=== Peer Discovery Test ===
DEBUG no new peers to process
DEBUG no new peers to process
DEBUG no new peers to process

=== Height Advancement Test ===
⚠️ [v1.0.1-beta] Block created but height NOT advanced - MUST call advance_height()
⚠️ [v1.0.1-beta] Block created but height NOT advanced - MUST call advance_height()

=== Sync Status Test ===
{
  "local_height": 1,         # ← STUCK
  "network_height": 78390,   # ← FALSE CLAIM
  "is_synced": true          # ← FALSE
}

=== Mining Challenge Test ===
1   # ← USELESS (network at 78390)

=== Bootstrap Connectivity Test ===
"12D3KooWDs4efUkG8TewmRotHegCi9FpjyUM4V9DGruzx3nkyWUv"  # ← Bootstrap reachable

=== P2P Peer Test ===
DEBUG no new peers to process
```

---

## Questions for External AI Consultation

### Priority 1: Peer Discovery

1. **libp2p Bootstrap Dialing:**
   - How to properly dial bootstrap peers in libp2p-rs?
   - Should bootstrap peers be added to Kademlia routing table explicitly?
   - Does mDNS discovery interfere with bootstrap peer connections?

2. **Network ID Matching:**
   - Does libp2p require exact protocol version matching?
   - How to debug peer rejection due to protocol mismatch?
   - Is there a way to force peer acceptance despite version differences?

3. **Diagnostic Techniques:**
   - How to enable verbose libp2p logging for peer discovery?
   - What metrics indicate successful bootstrap peer connection?
   - How to verify DHT routing table is populated?

### Priority 2: Height Advancement

1. **Block Production Flow:**
   - Where in the code should `advance_height()` be called?
   - Should height advancement be atomic with block storage?
   - How to ensure height consistency across database and atomics?

2. **Code Path Analysis:**
   - Why would bootstrap node and user nodes have different block production paths?
   - Could `bootstrap_node` flag cause code path divergence?
   - How to identify missing function calls in Rust async code?

3. **Database Transactions:**
   - Should block storage and height advancement be in same transaction?
   - How to recover if block is stored but height not advanced?
   - What are best practices for sequential block production?

### Priority 3: Sync Status Accuracy

1. **Gossipsub vs Local Production:**
   - Should sync status be based on received blocks or produced blocks?
   - How to distinguish passive reception from active production?
   - What is industry standard for "synced" definition?

2. **Height Sources:**
   - Should status report both local and network height?
   - How to calculate "blocks behind" accurately?
   - What threshold defines "synced" (10 blocks? 100 blocks?)?

---

## Conclusion

The "no new peers to process" bug and height advancement failure are two distinct but related issues:

1. **Peer Discovery Bug:** User nodes fail to connect to bootstrap peer, likely due to missing explicit dial or mDNS-only configuration

2. **Height Advancement Bug:** Block producers create blocks but never call `advance_height()`, causing local chain to freeze at height 1

3. **False Sync Status:** Node reports synced based on gossipsub reception (passive) while local production is stuck (active)

**Combined Impact:** User nodes cannot mine locally despite appearing healthy in status checks.

**Recommended Fix Priority:**
1. **P0:** Fix height advancement bug (add `advance_height()` call)
2. **P0:** Fix peer discovery bug (explicit bootstrap dial)
3. **P1:** Fix sync status accuracy (report both heights)

**External Consultation Needed:**
- libp2p bootstrap peer connection best practices
- Rust async block production flow verification
- Distributed system sync status standards

This technical review is ready for consultation with external AI systems to identify root causes and validate proposed fixes.

---

**Document Status:** Ready for External Consultation
**Created:** 2025-11-14
**Author:** Technical Analysis - Server Beta
**For Review By:** External AI Systems (Claude, GPT-4, Gemini, etc.)
