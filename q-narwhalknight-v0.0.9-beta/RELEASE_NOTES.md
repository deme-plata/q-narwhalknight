# Q-NarwhalKnight v0.0.9-beta Release Notes

## 🎉 Peer Discovery Fix - Production Ready

**Release Date**: October 23, 2025

### 🐛 Critical Bug Fixes

#### **Peer Discovery Event Loop Fix**
- **Issue**: libp2p event loop was terminating on dial errors, preventing peer discovery
- **Root Cause**: `swarm.dial(addr)?` propagated errors instead of logging and continuing
- **Fix**: Changed error handling to log failures while keeping event loop alive
- **Impact**: Nodes now continuously discover and connect to peers via mDNS and Kademlia DHT
- **File**: `crates/q-network/src/unified_network_manager.rs:343-348`

### ✨ Improvements

#### **Real-Time Peer Counting**
- Added thread-safe atomic counter for accurate peer tracking
- Peer count updates instantly on connection/disconnection events
- Console and frontend UI show live peer count
- Files: `unified_network_manager.rs:109, 480-482, 595, 611`

#### **Enhanced Discovery Logging**
- Added detailed dial attempt logs
- Connection success/failure visibility
- Better debugging for network issues

### 🌐 Network Discovery

**Multi-Layer Peer Discovery**:
- ✅ **mDNS** (Local network, <1 second)
- ✅ **Kademlia DHT** (Global internet, 5-30 seconds)
- ✅ **Identify Protocol** (Peer exchange)
- ✅ **Bootstrap Nodes** (185.182.185.227:8081)

### 📊 Features

- Austrian Economics tokenomics (0.5 QUG block reward, 1s block time)
- TUI (Terminal User Interface) enabled by default
- Real-time explorer with privacy protection
- Mining dashboard with accurate statistics
- WebSocket/SSE real-time updates

### 🔧 Configuration

**Bootstrap Nodes**:
```bash
# Default (automatic)
./q-api-server --port 8080

# Custom bootstrap
export Q_BOOTSTRAP_PEERS="/ip4/1.2.3.4/tcp/8081/p2p/12D3Koo..."
./q-api-server --port 8080
```

### 📦 Package Contents

- `q-api-server` - Full node with REST API, consensus, and mining
- `README.md` - Quick start guide
- `RELEASE_NOTES.md` - This file

### 🚀 Quick Start

```bash
# Extract
tar -xzf q-narwhalknight-linux-v0.0.9-beta.tar.gz
cd q-narwhalknight-v0.0.9-beta/bin

# Run node
./q-api-server --port 8080

# Run with TUI
./q-api-server --port 8080 --tui

# Check peer count
curl http://localhost:8080/api/v1/status | jq '.data.connected_peers'
```

### 🔗 Resources

- **Website**: https://quillon.xyz
- **Explorer**: https://quillon.xyz/explorer
- **Mining Guide**: https://quillon.xyz/mining
- **API Docs**: http://localhost:8080/api/v1/status

### 🐛 Known Issues

None critical. Peer discovery now works reliably on both local and remote networks.

### 📈 Upgrade Path

**From v0.0.8-beta**:
1. Stop old node: `killall q-api-server`
2. Backup data: `cp -r ./data ./data-backup`
3. Replace binary with v0.0.9-beta
4. Restart: `./q-api-server --port 8080`
5. Verify peer count increases

---

**Built with ❤️ by the Q-NarwhalKnight Team**
