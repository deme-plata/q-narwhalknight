# Console Visualization Fix

## Problem
The console says "No peers connected yet" and "0 transactions" even though:
- ConnectionManager successfully connected to peer 185.182.185.227:8081
- Transactions are being submitted
- Everything is working, just not displayed

## Root Cause
The visualization reads from `node_status.connected_peers` and `node_status.tx_pool_size` which are never updated. The real data is in:
- `ConnectionManager.active_peers` - HashMap with actual connections
- Transaction pool in DAGKnight consensus

## Fix Required

### 1. Add Connection Count Method
**File:** `crates/q-network/src/connection_manager.rs`

Add this method after line 118:

```rust
/// Get number of active connections
pub async fn get_active_connection_count(&self) -> usize {
    self.active_peers.read().await.len()
}
```

### 2. Expose ConnectionManager in AppState
**File:** `crates/q-api-server/src/lib.rs`

Find `pub struct AppState` and add:
```rust
pub connection_manager: Option<Arc<ConnectionManager>>,
```

### 3. Update Stats from Real Data
**File:** `crates/q-api-server/src/main.rs` (around line 640)

Replace the stats update loop to read from connection_manager:

```rust
loop {
    interval.tick().await;

    // Get real connection count from ConnectionManager
    let connected_peers = if let Some(ref conn_mgr) = app_state_updater.connection_manager {
        conn_mgr.get_active_connection_count().await
    } else {
        0
    };

    // Get real transaction count from consensus
    let (current_tx, current_blocks) = if let Some(ref consensus) = app_state_updater.dag_consensus {
        // Read from actual consensus state
        (consensus.transaction_count().await, consensus.block_count().await)
    } else {
        // Fallback to node_status
        let node_status = app_state_updater.node_status.read().await;
        (node_status.tx_pool_size as u64, node_status.current_height)
    };

    update_stats(stats_handle_updater.clone(), |stats| {
        stats.total_transactions = current_tx;
        stats.total_blocks = current_blocks;
        stats.connected_peers = connected_peers;
        // ... rest of stats
    }).await;
}
```

### 4. Wire Up Connection Manager
**File:** `crates/q-api-server/src/main.rs` (where connection_manager is created)

When creating AppState, pass the connection_manager:

```rust
let app_state = Arc::new(AppState {
    // ... other fields ...
    connection_manager: connection_manager.clone(),
});
```

## Quick Test
After fix, you should see:
- `Connected Peers: 1` (shows the 185.182.185.227:8081 connection)
- Transaction count increases when you submit transactions
- Network Status changes from "❌ Isolated" to "✅ Connected"

## Current Workaround
The system is working correctly - ConnectionManager logs show successful connection. Only the display is wrong.
