# Q-NarwhalKnight GUI - Full Node Feature

## Overview

The Q-NarwhalKnight GUI now supports two operational modes:

1. **Light Client** (Default - ~30MB)
   - Connects to remote API server
   - Minimal resource usage
   - Perfect for wallets and basic usage

2. **Full Node** (Optional - ~150MB)
   - Runs embedded blockchain node
   - Complete consensus participation
   - Mining capabilities
   - Full sovereignty

## Building Different Versions

### Light Client (Default)
```bash
# Linux
cargo build --release --package qnk-gui

# Windows
cross build --release --target x86_64-pc-windows-gnu --package qnk-gui
```

### Full Node Version
```bash
# Linux with embedded node
cargo build --release --package qnk-gui --features full-node

# Windows with embedded node
cross build --release --target x86_64-pc-windows-gnu --package qnk-gui --features full-node
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                Q-NarwhalKnight GUI                       │
│                                                          │
│  User Selection:                                         │
│    ○ Light Client Mode (Remote Server)                  │
│    ● Full Node Mode (Embedded)                          │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  IF Full Node Enabled:                                  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Embedded Components:                              │ │
│  │  • DAG-Knight Consensus Engine                     │ │
│  │  • Narwhal Mempool (Reliable Broadcast)           │ │
│  │  • libp2p P2P Network Layer                       │ │
│  │  • Mining Engine (Optional)                        │ │
│  │  • Local Storage Backend                           │ │
│  │  • Tor Client (Privacy Layer)                      │ │
│  │  • Built-in API Server (localhost:8080)           │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  Node Status Display:                                   │
│    🟢 Running                                           │
│    Block Height: 1,245                                  │
│    Peers: 12 connected                                  │
│    Sync: 100%                                           │
│    Mining: Active                                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Implementation Details

### Cargo Features

The `full-node` feature enables the embedded node dependencies:

```toml
[features]
full-node = ["embedded-node"]
embedded-node = [
    "q-dag-knight",      # Consensus engine
    "q-narwhal-core",    # Mempool
    "q-network",         # P2P networking
    "q-api-server",      # Local API
    "q-wallet",          # Wallet management
    "q-storage",         # Blockchain storage
    "q-mining",          # Mining engine
    "q-tor-client"       # Tor integration
]
```

### Module Structure

- **`src/embedded_node.rs`** - Node management module
  - `EmbeddedNode::new()` - Initialize node
  - `EmbeddedNode::start()` - Start full node
  - `EmbeddedNode::stop()` - Stop full node
  - `EmbeddedNode::get_status()` - Get node metrics

### Conditional Compilation

The code uses conditional compilation to support both modes:

```rust
#[cfg(feature = "embedded-node")]
{
    // Full node code
    info!("Starting embedded consensus node...");
    self.start_dag_knight().await?;
    self.start_mempool().await?;
    self.start_network().await?;
}

#[cfg(not(feature = "embedded-node"))]
{
    // Light client code
    warn!("Full node feature not compiled - using remote server");
}
```

## User Interface

### Settings Panel (Planned)

```
┌─────────────────────────────────────┐
│  Settings                           │
├─────────────────────────────────────┤
│                                     │
│  Node Mode:                         │
│    ○ Light Client                   │
│    ● Full Node                      │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  Full Node Status             │ │
│  │                               │ │
│  │  Status: 🟢 Running           │ │
│  │  Synced: 1,245/1,245 blocks   │ │
│  │  Peers: 12 connected          │ │
│  │  Sync Progress: ████████ 100% │ │
│  │                               │ │
│  │  Mining: ☑ Enabled            │ │
│  │  Hash Rate: 1.2 MH/s          │ │
│  │                               │ │
│  │  Storage: 2.3 GB              │ │
│  │  Location: ./data/blockchain  │ │
│  └───────────────────────────────┘ │
│                                     │
│  [Start Node]  [Stop Node]          │
│                                     │
└─────────────────────────────────────┘
```

## Performance Comparison

| Feature              | Light Client | Full Node    |
|---------------------|--------------|--------------|
| Binary Size         | ~30 MB       | ~150 MB      |
| RAM Usage           | 50-100 MB    | 500-2000 MB  |
| Disk Space          | 0 MB         | 2-10 GB      |
| Startup Time        | < 1 second   | 10-30 seconds|
| Trust Model         | Server Trust | Trustless    |
| Mining              | ❌           | ✅            |
| Consensus Voting    | ❌           | ✅            |
| Network Resilience  | ❌           | ✅            |

## Benefits of Full Node Mode

### For Users:
- ✅ **Sovereignty**: No dependence on external servers
- ✅ **Privacy**: All transactions processed locally
- ✅ **Mining**: Earn rewards by participating in consensus
- ✅ **Resilience**: Works even if remote servers are down
- ✅ **Security**: Validate all blockchain rules yourself

### For the Network:
- ✅ **Decentralization**: More nodes = stronger network
- ✅ **Consensus**: More validators = higher security
- ✅ **Availability**: More peers = better uptime
- ✅ **Tor Integration**: Enhanced privacy for all users

## Next Steps

### Phase 1: Infrastructure (Current)
- ✅ Feature flag system
- ✅ Embedded node module structure
- ⏳ Build system for both versions

### Phase 2: UI Integration
- ⏳ Add settings toggle
- ⏳ Node status display
- ⏳ Start/stop controls

### Phase 3: Full Implementation
- ⏳ Integrate q-api-server
- ⏳ Connect DAG-Knight consensus
- ⏳ Enable P2P networking
- ⏳ Add mining controls

### Phase 4: Polish
- ⏳ Sync progress indicator
- ⏳ Peer management UI
- ⏳ Mining statistics
- ⏳ Resource usage monitoring

## Building Instructions

### Quick Start

```bash
# Build light client (default)
cargo build --release --package qnk-gui

# Build full node version
cargo build --release --package qnk-gui --features full-node

# Both binaries will be functionally identical in UI
# Full node version includes embedded blockchain capabilities
```

### Distribution

- **Light Client**: Distribute to general users
- **Full Node**: Distribute to power users, miners, validators

## Configuration

Full node configuration via environment variables:

```bash
# Data directory
export QNK_DATA_DIR=./data/blockchain

# P2P listen port
export QNK_P2P_PORT=9001

# Enable mining
export QNK_ENABLE_MINING=true

# Mining threads
export QNK_MINING_THREADS=4

# Tor integration
export QNK_ENABLE_TOR=true
```

## Future Enhancements

- **Auto-Update**: Automatic blockchain sync
- **Pruning**: Option to prune old blocks
- **Snapshot Sync**: Fast sync from trusted checkpoint
- **Light Mining**: CPU-friendly mining mode
- **Mobile Support**: Android/iOS full node builds
- **Bootstrap Peers**: Auto-discover network peers

---

**Note**: The full node feature is currently in development. Light client mode is fully functional and recommended for most users.
