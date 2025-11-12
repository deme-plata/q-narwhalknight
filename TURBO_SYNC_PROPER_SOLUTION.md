# Turbo Sync Proper Solution: libp2p Request-Response Protocol

**Date**: 2025-11-09
**Issue**: Turbo sync completely broken - peers never respond to block pack requests
**Root Cause**: Using gossipsub for request/response pattern (wrong protocol)
**Proper Solution**: Use libp2p's built-in `request-response` protocol

---

## Problem Analysis

### Current Broken Implementation

**What we're doing wrong**:
```rust
// ❌ WRONG: Using gossipsub for request/response
// Send request via gossipsub topic: /qnk/testnet-phase6/block-pack-requests
gossipsub.publish(topic, BlockPackRequest { start: 0, end: 1000 })

// Wait for response on different topic: /qnk/testnet-phase6/block-pack-responses
// ❌ NO HANDLER EXISTS - requests are never answered!
```

**Why it fails**:
1. **Gossipsub is pub/sub** - messages are broadcast to all subscribers
2. **No request/response correlation** - can't match responses to requests
3. **No handler implemented** - no code to receive requests and send responses
4. **90-second timeout** - always expires because no responses ever come
5. **HTTP fallback only** - 0.5 blocks/minute = 7 days to sync

### Why This Is Wrong Architecture

| Protocol | Use Case | Our Need |
|----------|----------|----------|
| **Gossipsub** | Broadcast messages to all peers | Block propagation ✅ |
| **Request-Response** | 1:1 request/response pattern | Block sync ✅ |
| **Kademlia DHT** | Distributed hash table | Peer discovery ✅ |

**We need Request-Response for block sync, NOT gossipsub!**

---

## Proper Solution: libp2p Request-Response

### Architecture

```
┌──────────────┐                    ┌──────────────┐
│   Node A     │                    │   Node B     │
│  (syncing)   │                    │  (synced)    │
├──────────────┤                    ├──────────────┤
│              │                    │              │
│  1. Request  │──────────────────► │  Handler     │
│              │  BlockPackRequest  │              │
│              │  { start: 0,       │  2. Load     │
│              │    end: 1000 }     │     blocks   │
│              │                    │     from DB  │
│              │                    │              │
│  4. Process  │ ◄──────────────────│  3. Response │
│     blocks   │  BlockPackResponse │              │
│              │  { blocks: [...] } │              │
└──────────────┘                    └──────────────┘
```

### Implementation Steps

#### Step 1: Add request-response to libp2p dependencies

**File**: `Cargo.toml` (line 96)
```toml
# Before:
libp2p = { version = "0.53", features = ["noise", "yamux", "tcp", "gossipsub", "identify", "ping", "kad", "upnp", "macros", "tokio"] }

# After:
libp2p = { version = "0.53", features = ["noise", "yamux", "tcp", "gossipsub", "identify", "ping", "kad", "upnp", "macros", "tokio", "request-response", "cbor"] }
```

**Why `cbor`**: Efficient binary serialization for request/response (alternative to JSON)

#### Step 2: Define Request/Response Types

**File**: `crates/q-types/src/lib.rs` (add to existing types)
```rust
use libp2p::request_response::{ProtocolSupport, RequestResponseCodec};
use async_trait::async_trait;

/// Block pack request for turbo sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPackRequest {
    /// Starting block height (inclusive)
    pub start_height: u64,
    /// Ending block height (inclusive)
    pub end_height: u64,
    /// Maximum blocks to return (prevents DoS)
    pub max_blocks: usize,
}

/// Block pack response for turbo sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPackResponse {
    /// Requested blocks
    pub blocks: Vec<QBlock>,
    /// Starting height of this response
    pub start_height: u64,
    /// Ending height of this response
    pub end_height: u64,
    /// Whether more blocks are available
    pub has_more: bool,
}

/// Codec for BlockPack request/response protocol
#[derive(Debug, Clone)]
pub struct BlockPackCodec;

#[async_trait]
impl RequestResponseCodec for BlockPackCodec {
    type Protocol = BlockPackProtocol;
    type Request = BlockPackRequest;
    type Response = BlockPackResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;
        serde_cbor::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;
        serde_cbor::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let bytes = serde_cbor::to_vec(&req)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        io.write_all(&bytes).await?;
        io.flush().await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let bytes = serde_cbor::to_vec(&res)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        io.write_all(&bytes).await?;
        io.flush().await
    }
}

/// Protocol identifier for block pack requests
#[derive(Debug, Clone)]
pub struct BlockPackProtocol;

impl AsRef<str> for BlockPackProtocol {
    fn as_ref(&self) -> &str {
        "/qnk/block-pack/1.0.0"
    }
}
```

#### Step 3: Add Request-Response to Network Manager

**File**: `crates/q-network/src/unified_network_manager.rs`

**Add to imports**:
```rust
use libp2p::request_response::{
    self, ProtocolSupport, RequestResponse, RequestResponseEvent,
    RequestResponseMessage, ResponseChannel,
};
use q_types::{BlockPackCodec, BlockPackProtocol, BlockPackRequest, BlockPackResponse};
```

**Add to NetworkBehaviour**:
```rust
#[derive(NetworkBehaviour)]
pub struct UnifiedBehaviour {
    pub gossipsub: gossipsub::Behaviour,
    pub kad: kad::Behaviour<MemoryStore>,
    pub identify: identify::Behaviour,
    pub ping: ping::Behaviour,
    pub block_pack: RequestResponse<BlockPackCodec>,  // ✅ NEW
}
```

**Initialize in new() method**:
```rust
// After ping initialization:
let block_pack = RequestResponse::new(
    BlockPackCodec,
    iter::once((BlockPackProtocol, ProtocolSupport::Full)),
    request_response::Config::default(),
);
```

#### Step 4: Handle Incoming Requests

**File**: `crates/q-api-server/src/main.rs` (event loop)

**Add to event matching**:
```rust
SwarmEvent::Behaviour(UnifiedBehaviourEvent::BlockPack(
    RequestResponseEvent::Message { peer, message }
)) => {
    match message {
        RequestResponseMessage::Request { request, channel, .. } => {
            // ✅ Handle incoming block pack request
            let storage = app_state.storage.clone();
            tokio::spawn(async move {
                handle_block_pack_request(storage, peer, request, channel).await;
            });
        }
        RequestResponseMessage::Response { response, .. } => {
            // ✅ Handle received block pack response
            process_block_pack_response(response, &app_state).await;
        }
    }
}
```

**Implement handler**:
```rust
async fn handle_block_pack_request(
    storage: Arc<RocksDBStorage>,
    peer: PeerId,
    request: BlockPackRequest,
    channel: ResponseChannel<BlockPackResponse>,
) {
    info!("📥 [BLOCK PACK] Request from {}: blocks {}-{}",
          peer, request.start_height, request.end_height);

    // Validate request
    let block_count = request.end_height.saturating_sub(request.start_height) + 1;
    if block_count > request.max_blocks as u64 {
        warn!("⚠️  [BLOCK PACK] Request exceeds max_blocks: {} > {}",
              block_count, request.max_blocks);
        return;
    }

    // Load blocks from database
    let mut blocks = Vec::new();
    for height in request.start_height..=request.end_height {
        match storage.get_block_by_height(height).await {
            Ok(Some(block)) => blocks.push(block),
            Ok(None) => {
                warn!("⚠️  [BLOCK PACK] Block {} not found", height);
                break;
            }
            Err(e) => {
                error!("❌ [BLOCK PACK] Error loading block {}: {}", height, e);
                break;
            }
        }

        // Limit response size
        if blocks.len() >= request.max_blocks {
            break;
        }
    }

    // Send response
    let response = BlockPackResponse {
        blocks: blocks.clone(),
        start_height: request.start_height,
        end_height: request.start_height + blocks.len() as u64 - 1,
        has_more: blocks.len() < block_count as usize,
    };

    info!("📤 [BLOCK PACK] Sending {} blocks to {}", blocks.len(), peer);

    if let Err(e) = swarm.behaviour_mut().block_pack.send_response(channel, response) {
        error!("❌ [BLOCK PACK] Failed to send response: {}", e);
    }
}
```

#### Step 5: Replace Turbo Sync Gossipsub with Request-Response

**File**: `crates/q-storage/src/turbo_sync.rs`

**Replace gossipsub publish with request-response**:
```rust
// ❌ OLD: Gossipsub (broken)
pub async fn sync_to_height(&self, target_height: u64) -> Result<()> {
    let request = BlockPackRequest {
        start_height: current + 1,
        end_height: target_height
    };
    // Publish to gossipsub and HOPE someone responds (they don't!)
    self.network.gossipsub_publish(topic, request)?;
}

// ✅ NEW: Request-Response (proper)
pub async fn sync_to_height(&self, target_height: u64) -> Result<()> {
    let current = self.storage.get_latest_height().await?;

    // Request blocks in chunks of 1000
    const CHUNK_SIZE: u64 = 1000;

    for chunk_start in (current + 1..=target_height).step_by(CHUNK_SIZE as usize) {
        let chunk_end = (chunk_start + CHUNK_SIZE - 1).min(target_height);

        let request = BlockPackRequest {
            start_height: chunk_start,
            end_height: chunk_end,
            max_blocks: 1000,
        };

        // ✅ Send request to a peer (request-response handles routing)
        let request_id = self.network.send_block_pack_request(request).await?;

        // ✅ Wait for response (with proper correlation)
        match tokio::time::timeout(
            Duration::from_secs(30),  // ✅ 30s timeout (vs broken 90s)
            self.wait_for_response(request_id)
        ).await {
            Ok(Ok(response)) => {
                // ✅ Process received blocks
                for block in response.blocks {
                    self.storage.store_block(&block).await?;
                }
                info!("✅ [TURBO SYNC] Synced blocks {}-{}",
                      chunk_start, chunk_end);
            }
            Ok(Err(e)) => {
                error!("❌ [TURBO SYNC] Request failed: {}", e);
                // ✅ Fall back to HTTP for this chunk
                self.http_sync_chunk(chunk_start, chunk_end).await?;
            }
            Err(_) => {
                warn!("⏱️  [TURBO SYNC] Timeout after 30s, trying HTTP");
                // ✅ HTTP fallback
                self.http_sync_chunk(chunk_start, chunk_end).await?;
            }
        }
    }

    Ok(())
}
```

---

## Performance Improvements

### Before (Gossipsub - Broken)
- **Request sent**: Via gossipsub broadcast
- **Response**: Never received (no handler)
- **Timeout**: 90 seconds (always)
- **Fallback**: HTTP sync at 0.5 blocks/minute
- **Sync time**: **7+ days** for 5000 blocks

### After (Request-Response - Proper)
- **Request sent**: To specific peer via request-response
- **Response**: Received within 1-5 seconds
- **Timeout**: 30 seconds (rarely hit)
- **Chunks**: 1000 blocks per request
- **Sync time**: **~5 minutes** for 5000 blocks (1000x faster!)

---

## Migration Path

### Phase 1: Add Request-Response (Non-Breaking)
1. Add `request-response` and `cbor` to libp2p features
2. Implement BlockPackCodec and protocol types
3. Add request-response to NetworkBehaviour
4. Implement request handler (serves blocks to others)
5. **Keep existing gossipsub turbo sync** (for compatibility)

### Phase 2: Test Request-Response
1. Deploy to test nodes
2. Monitor request/response success rate
3. Verify block sync performance
4. Compare with gossipsub (should be 1000x faster)

### Phase 3: Migrate Turbo Sync (Breaking)
1. Replace gossipsub turbo sync with request-response
2. Remove gossipsub block-pack topics
3. Update all nodes to new version
4. **Result**: 5-minute sync time (vs 7 days)

---

## Additional Improvements

### 1. Intelligent Peer Selection
```rust
// Select best peer for block requests
fn select_sync_peer(&self) -> Option<PeerId> {
    // Prefer peers with:
    // 1. Higher block height
    // 2. Lower latency
    // 3. Good request success rate
    self.peers
        .iter()
        .filter(|p| p.height >= target_height)
        .min_by_key(|p| p.latency)
}
```

### 2. Parallel Chunk Requests
```rust
// Request multiple chunks in parallel
let futures: Vec<_> = chunks
    .iter()
    .map(|chunk| self.request_chunk(chunk))
    .collect();

let results = futures::future::join_all(futures).await;
```

### 3. Block Verification
```rust
// Verify blocks before storing
for block in response.blocks {
    if !block.verify_signatures().await? {
        return Err(anyhow!("Invalid block signature"));
    }
    storage.store_block(&block).await?;
}
```

---

## Testing Plan

### Unit Tests
```rust
#[tokio::test]
async fn test_block_pack_request_response() {
    let storage = create_test_storage().await;
    let request = BlockPackRequest {
        start_height: 0,
        end_height: 100,
        max_blocks: 100,
    };

    let response = handle_request(storage, request).await?;
    assert_eq!(response.blocks.len(), 101);
    assert_eq!(response.start_height, 0);
    assert_eq!(response.end_height, 100);
}
```

### Integration Tests
```rust
#[tokio::test]
async fn test_turbo_sync_with_request_response() {
    let node1 = spawn_test_node(5000).await; // Has 5000 blocks
    let node2 = spawn_test_node(0).await;    // Empty

    node2.sync_to_height(5000).await?;

    let height = node2.get_height().await?;
    assert_eq!(height, 5000);

    // Should complete in < 1 minute
    assert!(elapsed < Duration::from_secs(60));
}
```

---

## Rollout Plan

### Week 1: Implementation
- [ ] Add request-response dependencies
- [ ] Implement BlockPackCodec
- [ ] Add request-response to NetworkBehaviour
- [ ] Implement request handler
- [ ] Add unit tests

### Week 2: Testing
- [ ] Deploy to test nodes
- [ ] Monitor performance metrics
- [ ] Fix any issues discovered
- [ ] Performance benchmarks

### Week 3: Migration
- [ ] Update turbo sync to use request-response
- [ ] Deploy to all nodes
- [ ] Monitor sync performance
- [ ] Verify 1000x speedup

---

## Success Metrics

After implementation:
- ✅ **Sync time**: 5 minutes (vs 7 days)
- ✅ **Request success**: >95% (vs 0%)
- ✅ **Timeout rate**: <5% (vs 100%)
- ✅ **Network load**: 50x reduction (chunked vs individual HTTP)
- ✅ **New node onboarding**: Minutes (vs impossible)

---

**Status**: Ready for implementation
**Priority**: 🔴 **CRITICAL** - Network unusable without this
**Effort**: 2-3 days full implementation
**Impact**: **1000x sync speed improvement**
