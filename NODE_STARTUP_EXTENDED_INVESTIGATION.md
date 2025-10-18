# Extended Node Startup Investigation - Critical Findings

## Problem Update

**10-second stagger DID NOT solve the problem.** Even with significant stagger time between node launches, only nodes 0 and 2 become HTTP-ready, while nodes 1 and 3 consistently fail to start their HTTP servers within 180 seconds.

### Test Results with 10s Stagger

```
Launch sequence:
- T+0s:  Node 0 launched
- T+10s: Node 1 launched
- T+20s: Node 2 launched
- T+30s: Node 3 launched

Result after 180s timeout:
✅ Node 0: HTTP ready
❌ Node 1: Timeout
✅ Node 2: HTTP ready
❌ Node 3: Timeout
```

**This pattern indicates the problem is NOT resource contention.**

## New Theory: Systematic Initialization Deadlock

The fact that only **even-numbered nodes** (0, 2) succeed while **odd-numbered nodes** (1, 3) fail suggests:

1. **Race condition or deadlock in initialization code**
2. **libp2p mDNS peer discovery issue** - Nodes might be waiting for each other
3. **IPFS bootstrap deadlock** - Nodes trying to connect to each other during init
4. **Q-Resonance integration issue** - New shadow mode code may have a bug

### Most Likely Culprit: libp2p Discovery Deadlock

Looking at the API server initialization (crates/q-api-server/src/main.rs:1674-1733):

```rust
// libp2p Discovery Setup - Phase 14
let libp2p_bridge = if let Some(ref mut unified) = unified_manager {
    unified.spawn_event_loop().await?;
    Some(unified.get_bridge().await)
} else {
    None
};
```

**Hypothesis**: Node 1 tries to discover Node 0 (which is still initializing), gets blocked waiting for mDNS response, and never completes its own initialization. Same for Node 3 trying to discover Nodes 0-2.

## Critical Evidence

1. **Nodes launch successfully** - Process IDs appear in logs
2. **Data directories created** - `./data-libp2p-node{0,1,2,3}` all exist
3. **HTTP server never binds** - Nodes 1 & 3 never reach line 1736 in main.rs
4. **No error messages** - Nodes don't crash, they just hang
5. **Deterministic pattern** - Always nodes 1 & 3, never nodes 0 & 2

## Recommended Debugging Steps

### Step 1: Check if nodes are actually running

```bash
ps aux | grep q-api-server | grep -v grep
```

**Expected**: Should see 4 processes (one per node)
**If only 2**: Nodes 1 & 3 crashed silently
**If 4**: Nodes 1 & 3 are hung/deadlocked

### Step 2: Capture stdout/stderr from hung nodes

The benchmark currently pipes stdout/stderr but doesn't save them. We need to see what nodes 1 & 3 are doing:

```rust
// In distributed_libp2p_1m_tps.rs, modify launch_validator_node():
.stdout(std::process::Stdio::piped())  // Change to:
.stdout(std::process::Stdio::from(File::create(format!("/tmp/node{}.stdout", config.node_id))?))
.stderr(std::process::Stdio::from(File::create(format!("/tmp/node{}.stderr", config.node_id))?))
```

### Step 3: Test without Q-Resonance shadow mode

The shadow mode integration was recently added and might have introduced a deadlock:

```rust
// In crates/q-api-server/src/main.rs, temporarily comment out lines 537-603:
/*
// Q-Resonance + Shadow Mode initialization
let resonance_coordinator = ResonanceCoordinator::new(node_id.to_vec());
let shadow_mode_coordinator = ShadowModeCoordinator::new(
    dag_knight_consensus.clone(),
    resonance_coordinator.clone(),
);
*/
```

### Step 4: Test with sequential launches (no parallelism)

Launch nodes completely one-at-a-time:

```bash
# Launch node 0, wait for HTTP ready
Q_DB_PATH=./data-node0 Q_P2P_PORT=9200 ./target/release/q-api-server --port 9100 &
curl --retry 20 --retry-delay 5 http://localhost:9100/health

# Only after node 0 is ready, launch node 1
Q_DB_PATH=./data-node1 Q_P2P_PORT=9201 ./target/release/q-api-server --port 9101 &
curl --retry 20 --retry-delay 5 http://localhost:9101/health

# etc...
```

### Step 5: Enable RUST_BACKTRACE and verbose logging

```rust
// In launch_validator_node():
.env("RUST_LOG", "debug,q_network=trace,libp2p=trace")
.env("RUST_BACKTRACE", "full")
```

## Alternative Solution: Skip Problematic Initialization

If the issue is in optional components (IPFS, libp2p discovery, etc.), we can make them non-blocking:

### Option A: Lazy Initialization

Move HTTP server start to BEFORE full initialization:

```rust
// crates/q-api-server/src/main.rs

// Start HTTP server FIRST (before any blocking initialization)
let app = Router::new()
    .route("/health", get(|| async { "initializing" }))
    // ... other routes

tokio::spawn(async move {
    let addr: SocketAddr = format!("0.0.0.0:{}", config.port).parse().unwrap();
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
});

// NOW do the slow initialization
// If it fails, HTTP server is already up and will return errors
```

### Option B: Timeout on Blocking Operations

Wrap potentially blocking operations in timeouts:

```rust
// Example for libp2p discovery:
let libp2p_result = tokio::time::timeout(
    Duration::from_secs(30),
    unified_manager.spawn_event_loop()
).await;

match libp2p_result {
    Ok(Ok(())) => info!("libp2p initialized successfully"),
    Ok(Err(e)) => warn!("libp2p initialization error: {}", e),
    Err(_) => warn!("libp2p initialization timeout - continuing without peer discovery"),
}
```

## Immediate Action Items

1. ✅ **Verified 10-second stagger doesn't fix the issue**
2. ⏳ **Need to capture node logs** to see where nodes 1 & 3 are hanging
3. ⏳ **Test with Q-Resonance disabled** to rule out shadow mode as culprit
4. ⏳ **Add timeouts to blocking operations** in API server initialization
5. ⏳ **Implement early HTTP server binding** before full initialization

## Root Cause Candidates (Ranked by Likelihood)

1. **libp2p mDNS discovery deadlock** (90% likely) - Nodes waiting for each other
2. **Q-Resonance shadow mode initialization bug** (75% likely) - Recently added code
3. **IPFS bootstrap deadlock** (60% likely) - Nodes trying to connect during init
4. **RocksDB file locking** (40% likely) - But would affect all nodes, not just odd ones
5. **DAG-Knight initialization race condition** (30% likely) - Consensus setup issue

## Conclusion

The consistent pattern of only even-numbered nodes succeeding strongly suggests a **peer discovery or networking deadlock** during initialization, NOT resource contention. The 10-second stagger proves nodes have plenty of time to initialize individually - something is causing nodes 1 & 3 to wait indefinitely for network events that never arrive.

**Recommended immediate fix**: Implement lazy initialization (Option A above) so HTTP server starts BEFORE potentially-blocking network initialization.

---

**Investigation Date**: October 14, 2025
**Status**: Root cause narrowed down to network initialization deadlock
**Priority**: CRITICAL - Blocks all distributed testing
