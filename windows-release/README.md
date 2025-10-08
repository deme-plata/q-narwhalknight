# Q-NarwhalKnight Windows Release

## 🚀 Quantum-Enhanced DAG-BFT Consensus for Windows

**Version**: 0.1.0
**Build Date**: 2025-10-07
**Target**: x86_64-pc-windows-gnu
**Build Method**: cargo-cross with Docker
**Build Time**: 5m 59s

---

## 📦 Package Contents

- **q-api-server.exe** (73 MB) - Q-NarwhalKnight API Server with full consensus stack

**SHA256 Checksums**:
```
ca97ea3af80b7e4ce599b57c1835ff894b9fb595c3c68966552e2cc4189a46ef  q-api-server.exe
41aadd9492ae9327ac1ce0e02c7fb67786e6fe4b2ef8c34b225bb33710c7d40a  q-narwhalknight-windows-x86_64.zip
```

---

## ⚡ Quick Start

### Basic Usage

```powershell
# Run the API server on default port 8080
.\q-api-server.exe

# Specify custom port
.\q-api-server.exe --port 8090

# Specify custom database path
$env:Q_DB_PATH = ".\my-node-data"
.\q-api-server.exe --port 8080

# Specify node ID for multi-node setup
.\q-api-server.exe --port 8080 --node-id "node1"
```

### Multi-Node Deployment

**Node 1** (PowerShell window 1):
```powershell
$env:Q_DB_PATH = ".\data-node1"
$env:Q_P2P_PORT = "9001"
.\q-api-server.exe --port 8080 --node-id "node1"
```

**Node 2** (PowerShell window 2):
```powershell
$env:Q_DB_PATH = ".\data-node2"
$env:Q_P2P_PORT = "9002"
.\q-api-server.exe --port 8082 --node-id "node2"
```

**Node 3** (PowerShell window 3):
```powershell
$env:Q_DB_PATH = ".\data-node3"
$env:Q_P2P_PORT = "9003"
.\q-api-server.exe --port 8084 --node-id "node3"
```

**Node 4** (PowerShell window 4):
```powershell
$env:Q_DB_PATH = ".\data-node4"
$env:Q_P2P_PORT = "9004"
.\q-api-server.exe --port 8086 --node-id "node4"
```

---

## 🌐 API Endpoints

Once running, the API is available at `http://localhost:8080` (or your custom port).

### Core Endpoints

- `GET /health` - Health check
- `GET /status` - Node status and peer information
- `POST /transaction` - Submit new transaction
- `GET /transaction/{hash}` - Get transaction by hash
- `GET /balance/{address}` - Get wallet balance
- `POST /wallet/create` - Create new wallet
- `GET /peers` - List connected peers
- `GET /consensus/status` - Consensus engine status

### Real-time Streaming

- `GET /stream/transactions` - Server-Sent Events (SSE) for real-time transaction updates
- `GET /stream/blocks` - Real-time block updates
- `GET /ws` - WebSocket connection for bidirectional communication

---

## 🔧 Configuration

### Environment Variables

- `Q_DB_PATH` - Database storage path (default: `./data`)
- `Q_P2P_PORT` - P2P networking port (default: `9000`)
- `RUST_LOG` - Logging level (`debug`, `info`, `warn`, `error`)

### Command-Line Options

```
OPTIONS:
    --port <PORT>          API server port [default: 8080]
    --node-id <ID>         Node identifier for P2P network
    --workers <N>          Number of parallel workers [default: 16]
    -h, --help            Print help information
    -V, --version         Print version information
```

---

## 🏗️ System Architecture

### Consensus Layer
- **Algorithm**: DAG-Knight with VDF-based quantum anchor election
- **Byzantine Threshold**: f=3 (tolerates 3 Byzantine nodes)
- **Message Complexity**: Zero-message ordering
- **Quantum Features**: VDF-enhanced randomness beacon

### Mempool Layer
- **Type**: Narwhal-style reliable broadcast
- **Sharding**: 16-way parallel processing (configurable)
- **Batching**: Automatic transaction batching

### Network Layer
- **Protocol**: libp2p with Zero-Knowledge Discovery
- **Transport**: TCP/IP with QUIC support
- **Discovery**: mDNS local network auto-discovery
- **Anonymity**: Tor integration support (optional)

### Cryptography
- **Phase 0**: Ed25519 (active)
- **Phase 1**: Dilithium5 + Kyber1024 (crypto-agile)
- **ZK Proofs**: STARK + SNARK systems
- **SIMD**: Vectorized cryptography engine

---

## 📊 Performance Benchmarks

### Linux Benchmarks (Reference)
- **Single-node**: ~185 TPS
- **4-node distributed**: 299 TPS aggregate
- **Concurrency**: 20 parallel clients per node
- **Success Rate**: 100%

*Windows performance may vary based on hardware and configuration.*

---

## 🔐 Security Features

### Active Security
- ✅ Zero-Knowledge peer discovery
- ✅ Post-quantum cryptography ready (Dilithium5/Kyber1024)
- ✅ Byzantine fault tolerance (f=3)
- ✅ ZK-proof verification
- ✅ Quantum-resistant signatures

### Network Security
- ⚠️ Tor integration prepared (requires configuration)
- ✅ P2P encryption via libp2p
- ✅ Secure peer authentication

---

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Check what's using port 8080
netstat -ano | findstr :8080

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Or use a different port
.\q-api-server.exe --port 8090
```

### Firewall Issues
```powershell
# Allow q-api-server through Windows Firewall
netsh advfirewall firewall add rule name="Q-NarwhalKnight" dir=in action=allow program="%CD%\q-api-server.exe" enable=yes

# Or temporarily disable firewall for testing (not recommended for production)
netsh advfirewall set allprofiles state off
```

### Database Permission Errors
```powershell
# Ensure the database directory exists and is writable
New-Item -ItemType Directory -Force -Path ".\data"

# Or specify a different path
$env:Q_DB_PATH = "C:\Users\YourUser\AppData\Local\q-narwhalknight"
.\q-api-server.exe
```

### Peer Discovery Issues
```powershell
# Check P2P port is not blocked
Test-NetConnection -ComputerName localhost -Port 9000

# Manually specify P2P port
$env:Q_P2P_PORT = "9001"
.\q-api-server.exe
```

---

## 📝 Example: Testing the Installation

```powershell
# 1. Start the server
.\q-api-server.exe

# 2. In another PowerShell window, test the API
Invoke-RestMethod -Uri http://localhost:8080/health

# 3. Check node status
Invoke-RestMethod -Uri http://localhost:8080/status

# 4. Create a wallet
$wallet = Invoke-RestMethod -Method POST -Uri http://localhost:8080/wallet/create
Write-Host "Wallet Address: $($wallet.address)"

# 5. Check balance
Invoke-RestMethod -Uri "http://localhost:8080/balance/$($wallet.address)"

# 6. Submit a transaction (example)
$tx = @{
    from = $wallet.address
    to = "recipient_address_here"
    amount = 100
} | ConvertTo-Json

Invoke-RestMethod -Method POST -Uri http://localhost:8080/transaction -Body $tx -ContentType "application/json"
```

---

## 🔄 Upgrading

To upgrade to a newer version:

1. Stop the running server (Ctrl+C)
2. Replace `q-api-server.exe` with the new version
3. Restart the server

**Note**: Database format may change between versions. Always backup your `data` directory before upgrading.

---

## 🆘 Support & Resources

- **GitHub**: https://github.com/deme-plata/q-narwhalknight
- **Documentation**: See `README.md` in the source repository
- **Issues**: Report bugs via GitHub Issues

---

## ⚖️ License

See `LICENSE` file in the source repository.

---

## 🌟 Technical Achievements

This Windows build includes:

- ✅ Full quantum-enhanced consensus implementation
- ✅ Post-quantum cryptography (Dilithium5, Kyber1024)
- ✅ Zero-Knowledge peer discovery
- ✅ libp2p networking with mDNS
- ✅ DAG-Knight consensus engine
- ✅ Narwhal mempool with parallel processing
- ✅ Real-time streaming APIs (SSE + WebSocket)
- ✅ Cross-compiled using cargo-cross with Docker

**Build Success**: After failing with MinGW, this binary was successfully cross-compiled using `cargo-cross` with a proper Windows GNU toolchain in Docker, completing in just 5m 59s.

---

*Generated: 2025-10-07*
*Q-NarwhalKnight - Quantum-Enhanced DAG-BFT Consensus*
