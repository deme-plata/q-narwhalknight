# Q-NarwhalKnight Miner

High-performance CPU/GPU miner for the Q-NarwhalKnight quantum-enhanced consensus network.

## 🚀 Features

### Mining Capabilities
- **CPU Mining**: Multi-threaded VDF computation and DAG consensus participation
- **NVIDIA CUDA**: High-performance GPU mining with CUDA kernels
- **OpenCL**: Cross-platform GPU computing support
- **Vulkan Compute**: Modern GPU compute pipeline

### Network Integration
- **Anonymous Mining**: Tor-enabled pool connections
- **P2P Discovery**: Bitcoin-based peer discovery
- **Pool Mining**: Stratum protocol support
- **Solo Mining**: Direct network participation

### User Experience
- **Cross-Platform**: Windows, Linux, macOS support
- **GUI Dashboard**: Real-time mining statistics
- **CLI Tools**: Command-line mining and monitoring
- **Auto-Detection**: Hardware capability detection

## 📦 Installation

### Quick Install (Linux)
```bash
curl -sSL https://github.com/deme-plata/q-narwhalknight/releases/latest/download/install.sh | bash
```

### Quick Install (Windows)
```powershell
Invoke-WebRequest -Uri "https://github.com/deme-plata/q-narwhalknight/releases/latest/download/install.ps1" | Invoke-Expression
```

### Quick Install (macOS)
```bash
brew install q-narwhalknight/tap/q-miner
```

### Manual Build
```bash
git clone https://github.com/deme-plata/q-narwhalknight.git
cd q-narwhalknight
cargo build --package q-miner --release --features cuda-mining,gui
```

## 🎯 Quick Start

### Solo Mining
```bash
# CPU mining
./q-miner --mode solo --threads 8

# GPU mining (CUDA)
./q-miner --mode solo --gpu --cuda --devices 0,1

# GUI mode
./q-miner-gui
```

### Pool Mining
```bash
# Connect to anonymous pool
./q-miner --mode pool \
  --pool-url stratum+tor://pool.qnarwhal.onion:4444 \
  --wallet-address qnk1...your_address \
  --gpu --cuda
```

### Benchmark Mode
```bash
# Hardware benchmarking
./q-miner-benchmark --all-devices

# Algorithm benchmarking  
./q-miner-benchmark --algorithm vdf --iterations 1000000
```

## ⚙️ Configuration

### Automatic Configuration
The miner auto-detects optimal settings:
- CPU cores and threads
- GPU devices and memory
- Network connectivity
- Power efficiency settings

### Manual Configuration
Create `~/.q-miner/config.toml`:
```toml
[mining]
mode = "pool"  # "solo" or "pool"
algorithm = "dag-knight-vdf"
difficulty_target = "auto"

[hardware]
cpu_threads = 0  # 0 = auto-detect
gpu_enabled = true
cuda_devices = [0, 1]
memory_limit_gb = 8

[network]
pool_url = "stratum+tor://pool.qnarwhal.onion:4444"
tor_enabled = true
p2p_enabled = true
max_peers = 32

[wallet]
address = "qnk1...your_address"
fee_priority = "medium"

[logging]
level = "info"
file = "~/.q-miner/logs/miner.log"
```

## 🔧 Advanced Usage

### Multi-GPU Setup
```bash
# NVIDIA multi-GPU
./q-miner --cuda --devices 0,1,2,3 --threads-per-gpu 2048

# Mixed GPU vendors
./q-miner --cuda --devices 0,1 --opencl --devices 2,3
```

### Performance Tuning
```bash
# High-performance mode
./q-miner --performance-mode extreme \
  --power-limit 300W \
  --memory-clock +1000 \
  --core-clock +200

# Efficiency mode
./q-miner --performance-mode efficient \
  --power-limit 150W \
  --fan-curve auto
```

### Monitoring & Analytics
```bash
# Real-time monitoring
./q-miner --monitor --web-ui http://localhost:8090

# Export performance data
./q-miner --export-stats --format prometheus \
  --output ./mining-metrics.prom
```

## 🌐 Network Architecture

### Triple-Layer Anonymity Mining
```
┌─────────────────┐    🧅 Tor Network    ┌─────────────────┐
│   Your Miner    │◄──► Anonymous Pool ◄──►│ Q-Knight Network│
│                 │    Mining Traffic     │                 │
│ CPU: 8 threads  │                      │ Consensus: DAG  │
│ GPU: RTX 4090   │    👻 DNS-Phantom    │ Finality: 2.3s │
│ Hash: 2.1GH/s   │    Steganographic    │ TPS: 48,000+   │
└─────────────────┘    Communication     └─────────────────┘
```

### Mining Pool Protocol
- **Stratum v2**: Enhanced security and efficiency
- **Tor Integration**: Anonymous connections
- **Quantum-Ready**: Post-quantum cryptography
- **Fair Distribution**: DAG-based work allocation

## 📊 Performance Targets

### Hardware Performance (Typical)
| Hardware | Hash Rate | Power | Efficiency |
|----------|-----------|-------|------------|
| Intel i9-13900K | 150 MH/s | 125W | 1.2 MH/W |
| AMD Ryzen 9 7950X | 180 MH/s | 105W | 1.7 MH/W |
| RTX 4090 | 8.5 GH/s | 450W | 18.9 MH/W |
| RTX 4080 | 6.2 GH/s | 320W | 19.4 MH/W |
| RTX 3080 | 4.1 GH/s | 320W | 12.8 MH/W |

### Network Performance
- **Latency**: <50ms pool connection
- **Throughput**: 10,000+ solutions/second
- **Efficiency**: 99.5% valid share submission
- **Uptime**: 99.9% connection stability

## 🛡️ Security Features

### Anonymous Mining
- **Tor Integration**: All pool connections via Tor
- **IP Protection**: No IP address leakage
- **Payment Privacy**: Confidential transactions
- **Hardware Fingerprinting**: Randomized device IDs

### Quantum Security
- **Post-Quantum Crypto**: Dilithium5/Kyber1024
- **VDF Security**: Verifiable delay functions
- **Quantum-Safe Pools**: End-to-end protection
- **Future-Proof**: Algorithm agility

## 🎮 User Experience

### Auto-Setup Wizard
1. **Hardware Detection**: Scan CPU/GPU capabilities
2. **Wallet Setup**: Generate or import wallet
3. **Pool Selection**: Choose optimal mining pool
4. **Performance Test**: Benchmark and optimize
5. **Start Mining**: One-click activation

### Real-Time Dashboard
- Live hash rate and earnings
- Hardware temperature/power monitoring  
- Network status and peer count
- Historical performance charts
- Profit calculator and ROI analysis

## 🚀 Distribution Strategy

### Release Packages
- **Windows**: `.msi` installer with GUI
- **Linux**: `.deb`, `.rpm`, `.tar.gz`, AppImage
- **macOS**: `.dmg` with notarized binary
- **Docker**: Multi-arch containers
- **Snap/Flatpak**: Universal Linux packages

### Auto-Updates
- **Background Updates**: Seamless version upgrades
- **Security Patches**: Automatic critical fixes
- **Mining Algorithm**: Hot-swappable algorithms
- **Pool Integration**: Dynamic pool discovery

This design provides a sophisticated, user-friendly miner that leverages the full power of modern hardware while maintaining the anonymity and quantum-security principles of Q-NarwhalKnight.