# Q-NarwhalKnight Node Distribution & Deployment Guide

**Version:** 1.0.0  
**Target Platform:** Linux  
**Architecture:** Quantum-Enhanced DAG-BFT Consensus System  

---

## 📋 Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Binary Distribution Strategy](#binary-distribution-strategy)
3. [Build & Package Instructions](#build--package-instructions)
4. [Deployment Options](#deployment-options)
5. [User Onboarding Process](#user-onboarding-process)
6. [Network Bootstrap & Discovery](#network-bootstrap--discovery)
7. [Configuration & Optimization](#configuration--optimization)
8. [Troubleshooting](#troubleshooting)

---

## 🏗️ System Architecture Overview

### Core System Components
- **42 specialized crates** with quantum consensus, privacy, and DeFi capabilities
- **4 primary binary targets** for different use cases
- **Triple-layer anonymity** (Tor + DNS-phantom + BEP-44 DHT)
- **ZK privacy** (STARK/SNARK proofs with GPU acceleration)
- **AI integration** (Mistral.rs with 320+ dependencies)

### Key Technological Stack
```
┌─────────────────────────────────────────────────┐
│                 USER LAYER                      │
│  GUI Client • CLI Tools • Web Interface        │
├─────────────────────────────────────────────────┤
│               APPLICATION LAYER                 │
│  q-api-server • q-miner • phantom-node         │
├─────────────────────────────────────────────────┤
│                CONSENSUS LAYER                  │
│  DAG-Knight • Narwhal Mempool • VDF            │
├─────────────────────────────────────────────────┤
│                NETWORKING LAYER                 │
│  Tor • DNS-Phantom • BEP-44 DHT • Bitcoin Bridge│
├─────────────────────────────────────────────────┤
│                PRIVACY LAYER                    │
│  ZK-STARK • ZK-SNARK • Quantum Crypto          │
├─────────────────────────────────────────────────┤
│                STORAGE LAYER                    │
│  RocksDB • Quantum Storage • Sharding          │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Binary Distribution Strategy

### Primary Distribution Target: `q-api-server`
**Recommended for 95% of users**

**Features:**
- ✅ Complete quantum consensus node
- ✅ Full REST API for wallet operations
- ✅ Triple-layer anonymity networking
- ✅ Real-time streaming (WebSocket/SSE)  
- ✅ Automatic peer discovery
- ✅ Bitcoin bridge integration
- ✅ ZK privacy features
- ✅ Web-based GUI interface

**Use Cases:**
- Individual node operators
- DeFi participants
- Privacy-focused users
- General blockchain users

### Secondary Distribution Options

#### 1. `phantom-node` - Steganographic Networking
```bash
# Lightweight DNS steganography node
./phantom-node --stealth --tx-rate 100
```
**Target Users:** Privacy researchers, covert network participants

#### 2. `q-miner` - High-Performance Mining
```bash
# CPU/GPU mining with CUDA support
./q-miner --cpu-threads 16 --gpu-cuda --pool mining.example.com:4444
```
**Target Users:** Miners, computing power contributors

#### 3. `qnk-gui` - Desktop Client
```bash
# Desktop GUI with embedded node
./qnk-gui --embedded-node
```
**Target Users:** Desktop users, non-technical users

---

## 🔧 Build & Package Instructions

### Prerequisites
```bash
# Install Rust 1.86+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install system dependencies
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev libclang-dev

# Optional: CUDA support for mining
sudo apt install -y nvidia-cuda-toolkit
```

### Complete Build Process
```bash
# Clone repository
git clone https://github.com/quantum-dag-labs/Q-NarwhalKnight.git
cd Q-NarwhalKnight

# Build all binaries with extended timeout for complex dependencies
timeout 36000 cargo build --release --workspace

# Verify binary creation
ls -la target/release/
# Should show: q-api-server, phantom-node, q-miner, qnk-gui
```

### Create Distribution Packages

#### Minimal Package (Primary Distribution)
```bash
#!/bin/bash
# Package: q-narwhalknight-minimal.tar.gz (~50MB)

mkdir -p dist/minimal
cp target/release/q-api-server dist/minimal/
cp -r gui/ui/ dist/minimal/gui/
cp README.md LICENSE dist/minimal/

# Create startup script
cat > dist/minimal/start-node.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Q-NarwhalKnight Node..."
./q-api-server --node-id $(hostname)-$(date +%s) --port 8080
EOF
chmod +x dist/minimal/start-node.sh

# Strip binaries for size
strip dist/minimal/q-api-server

# Create package
cd dist && tar -czf q-narwhalknight-minimal.tar.gz minimal/
```

#### Standard Package (Recommended)
```bash
#!/bin/bash
# Package: q-narwhalknight-standard.tar.gz (~200MB)

mkdir -p dist/standard
cp target/release/{q-api-server,phantom-node} dist/standard/
cp -r gui/ scripts/ dist/standard/
cp README.md LICENSE DEPLOYMENT_GUIDE.md dist/standard/

# Create configuration template
cat > dist/standard/node.toml << 'EOF'
[node]
port = 8080
enable_tor = true
enable_dns_phantom = true

[privacy]
stealth_mode = false
max_tx_rate = 100

[mining]
enabled = false
threads = 0
EOF

cd dist && tar -czf q-narwhalknight-standard.tar.gz standard/
```

#### Complete Package (Advanced Users)
```bash
#!/bin/bash
# Package: q-narwhalknight-complete.tar.gz (~800MB)

mkdir -p dist/complete
cp target/release/* dist/complete/
cp -r gui/ papers/ scripts/ docs/ dist/complete/
cp README.md LICENSE DEPLOYMENT_GUIDE.md CLAUDE.md dist/complete/

# Include development tools
mkdir -p dist/complete/dev-tools/
cp -r benchmarks/ examples/ tests/ dist/complete/dev-tools/

cd dist && tar -czf q-narwhalknight-complete.tar.gz complete/
```

---

## 🚀 Deployment Options

### Option 1: Single Binary Deployment (Recommended)

#### Quick Start
```bash
# Download and extract
wget https://releases.q-narwhal.network/q-narwhalknight-minimal.tar.gz
tar -xzf q-narwhalknight-minimal.tar.gz
cd minimal/

# Run node
./start-node.sh

# Access web interface
open http://localhost:8080
```

#### Manual Configuration
```bash
# Basic node
./q-api-server --node-id my-node --port 8080

# Privacy-enhanced node
./q-api-server \
  --node-id privacy-node \
  --port 8080 \
  --enable-tor \
  --enable-dns-phantom \
  --stealth-mode

# Mining node
./q-api-server \
  --node-id mining-node \
  --port 8080 \
  --enable-mining \
  --mining-threads 8
```

### Option 2: Docker Deployment

#### Create Dockerfile
```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    tor \
    dnsutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries
COPY target/release/q-api-server /usr/local/bin/
COPY target/release/phantom-node /usr/local/bin/
COPY target/release/q-miner /usr/local/bin/
COPY gui/ /app/gui/

# Create user
RUN useradd -m -s /bin/bash qnode

# Expose ports
EXPOSE 8080 8081 9050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

USER qnode
WORKDIR /app

CMD ["q-api-server", "--port", "8080"]
```

#### Docker Compose Setup
```yaml
version: '3.8'

services:
  qnk-node:
    build: .
    container_name: q-narwhal-node
    ports:
      - "8080:8080"
      - "8081:8081"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - RUST_LOG=info
      - QNK_NODE_ID=docker-node-1
      - QNK_ENABLE_TOR=true
    restart: unless-stopped
    
  tor:
    image: torproject/tor:latest
    container_name: qnk-tor
    ports:
      - "9050:9050"
    volumes:
      - ./tor-config:/etc/tor
    restart: unless-stopped
```

### Option 3: Package Manager Distribution

#### Debian/Ubuntu Package
```bash
# Create .deb package structure
mkdir -p qnk-package/DEBIAN
mkdir -p qnk-package/usr/local/bin
mkdir -p qnk-package/etc/qnarwhalknight
mkdir -p qnk-package/usr/share/applications
mkdir -p qnk-package/usr/share/doc/qnarwhalknight

# Copy binaries
cp target/release/q-api-server qnk-package/usr/local/bin/

# Create control file
cat > qnk-package/DEBIAN/control << EOF
Package: qnarwhalknight
Version: 1.0.0
Section: net
Priority: optional
Architecture: amd64
Depends: libc6, libssl3, tor
Maintainer: Quantum-DAG Labs <contact@quantum-dag-labs.org>
Description: Quantum-Enhanced DAG-BFT Consensus System
 Q-NarwhalKnight is a quantum-ready blockchain consensus system
 with advanced privacy features and steganographic networking.
EOF

# Build package
dpkg-deb --build qnk-package qnarwhalknight_1.0.0_amd64.deb
```

#### Snap Package
```yaml
name: qnarwhalknight
version: '1.0.0'
summary: Quantum-Enhanced Blockchain Consensus
description: |
  Q-NarwhalKnight provides quantum-ready blockchain consensus
  with advanced privacy and steganographic networking features.

base: core20
confinement: strict

apps:
  qnk:
    command: q-api-server
    daemon: simple
    plugs: [network, network-bind, home]
    
  miner:
    command: q-miner
    plugs: [network, hardware-observe]
    
  phantom:
    command: phantom-node
    plugs: [network, network-bind]

parts:
  qnarwhalknight:
    plugin: rust
    source: .
    rust-features: [default]
```

---

## 🎯 User Onboarding Process

### Phase 1: Simple Setup (5 minutes)
```bash
# 1. Download single binary
wget https://releases.q-narwhal.network/q-api-server
chmod +x q-api-server

# 2. Start node with automatic configuration
./q-api-server --auto-config

# 3. Access web interface
# Browser automatically opens to http://localhost:8080

# 4. Create quantum wallet
# Follow GUI prompts for mnemonic generation

# 5. Node automatically:
#    - Generates secure node ID
#    - Establishes peer connections  
#    - Syncs blockchain state
#    - Joins consensus network
```

### Phase 2: Privacy Enhancement (10 minutes)
```bash
# Enable full privacy stack
export QNK_ENABLE_TOR=true
export QNK_ENABLE_DNS_PHANTOM=true
export QNK_STEALTH_MODE=true

# Restart with privacy features
./q-api-server --privacy-mode

# Verify anonymity layers
curl localhost:8080/api/v1/security/tor/status
curl localhost:8080/api/v1/dns/phantom/status
```

### Phase 3: Advanced Features (30 minutes)
```bash
# Enable mining
./q-api-server --enable-mining --mining-threads $(nproc)

# Join specific networks
./q-api-server --target-beta  # Connect to Server Beta

# Enable DeFi features
curl -X POST localhost:8080/api/v1/defi/dex/enable
curl -X POST localhost:8080/api/v1/defi/oracle/subscribe

# Deploy ZK applications
curl -X POST localhost:8080/api/v1/zk/stark/deploy \
  -H "Content-Type: application/json" \
  -d '{"circuit": "privacy_mixer", "constraints": 1000000}'
```

---

## 🌐 Network Bootstrap & Discovery

### Automatic Peer Discovery
The node uses multiple discovery mechanisms:

1. **DNS-Phantom Steganographic Discovery** (Primary)
   - Embeds peer info in DNS queries
   - Covert channel through DNS infrastructure
   - Resistance to traffic analysis

2. **BEP-44 DHT Discovery** (Secondary)  
   - BitTorrent DHT for peer announcements
   - Encrypted friend-only mode
   - Decoy traffic generation

3. **Bitcoin Network Bridge** (Fallback)
   - Anonymous peer discovery via Bitcoin network
   - Bitcoin transaction steganography
   - Bridge to existing crypto ecosystem

4. **Manual Peer Addition** (Backup)
   ```bash
   curl -X POST localhost:8080/api/v1/network/peers/add \
     -H "Content-Type: application/json" \
     -d '{"address": "peer1.q-narwhal.network:8080"}'
   ```

### Bootstrap Node Configuration
```toml
# /etc/qnarwhalknight/bootstrap.toml
[bootstrap]
nodes = [
  "bootstrap-1.q-narwhal.network:8080",
  "bootstrap-2.q-narwhal.network:8080", 
  "bootstrap-3.q-narwhal.network:8080"
]

[discovery]
dns_phantom_enabled = true
bep44_enabled = true
bitcoin_bridge_enabled = true
stealth_mode = false

[tor]
enabled = true
circuits = 4
rotation_interval = 600  # 10 minutes
```

---

## ⚙️ Configuration & Optimization

### Performance Tuning
```bash
# High-throughput configuration (6M+ TPS target)
export QNK_SIMD_CRYPTO=true        # Enable SIMD cryptographic acceleration
export QNK_KERNEL_IO=true          # Enable io_uring and NUMA optimizations
export QNK_SHARDING_ENABLED=true   # Enable cross-shard communication
export QNK_CACHE_LEVEL=3           # Maximum caching level

# Memory optimization
export QNK_MEMORY_POOL_SIZE=8GB
export QNK_WORKER_THREADS=$(nproc)
export QNK_MAX_CONNECTIONS=10000

# Restart with optimizations
./q-api-server --high-performance
```

### Security Hardening
```bash
# Maximum security configuration
export QNK_QUANTUM_CRYPTO=true     # Enable post-quantum cryptography
export QNK_ZK_PRIVACY=true         # Enable ZK-STARK/SNARK privacy
export QNK_TOR_ISOLATION=true      # Strict Tor circuit isolation
export QNK_DNS_PHANTOM_STEALTH=true # Maximum steganography

# Network security
export QNK_FIREWALL_MODE=strict
export QNK_IP_WHITELIST_ONLY=true
export QNK_RATE_LIMITING=aggressive

./q-api-server --security-hardened
```

### Resource Monitoring
```bash
# Check node performance
curl localhost:8080/api/v1/analytics/performance

# Monitor network health  
curl localhost:8080/api/v1/network/analytics

# View consensus metrics
curl localhost:8080/api/v1/consensus/dag-knight

# Check privacy status
curl localhost:8080/api/v1/security/tor/circuits
curl localhost:8080/api/v1/dns/phantom/peers
```

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### Issue: Node fails to start
```bash
# Check dependencies
ldd target/release/q-api-server

# Check permissions
chmod +x q-api-server
chown $USER:$USER q-api-server

# Check ports
netstat -tulpn | grep :8080
sudo ufw allow 8080

# Check logs
./q-api-server --log-level debug
```

#### Issue: Tor connection fails
```bash
# Install Tor
sudo apt install tor
sudo systemctl start tor

# Check Tor status
sudo systemctl status tor
curl --socks5 127.0.0.1:9050 http://check.torproject.org

# Configure Tor for Q-NarwhalKnight
echo "SOCKSPort 9050" | sudo tee -a /etc/tor/torrc
sudo systemctl restart tor
```

#### Issue: Peer discovery not working
```bash
# Check DNS phantom status
curl localhost:8080/api/v1/dns/phantom/status

# Force discovery
curl -X POST localhost:8080/api/v1/network/discovery/trigger

# Manual peer addition
curl -X POST localhost:8080/api/v1/network/peers/add \
  -d '{"address": "peer.example.com:8080"}'

# Check firewall
sudo ufw status
sudo ufw allow out 53    # DNS
sudo ufw allow out 8080  # P2P
```

#### Issue: Mining not working
```bash
# Check GPU support
nvidia-smi  # For NVIDIA GPUs
rocm-smi   # For AMD GPUs

# Enable mining
./q-miner --cpu-threads $(nproc) --gpu-enabled

# Check mining status
curl localhost:8080/api/v1/mining/status
```

#### Issue: High resource usage
```bash
# Reduce resource consumption
export QNK_WORKER_THREADS=2
export QNK_MEMORY_POOL_SIZE=1GB
export QNK_CACHE_LEVEL=1

# Monitor resources
htop
iostat -x 1
free -h

# Optimize configuration
./q-api-server --low-resource-mode
```

### Debug Information Collection
```bash
#!/bin/bash
# debug-info.sh - Collect debugging information

echo "=== Q-NarwhalKnight Debug Information ===" > debug-info.txt
date >> debug-info.txt
echo "" >> debug-info.txt

echo "--- System Information ---" >> debug-info.txt
uname -a >> debug-info.txt
lscpu >> debug-info.txt
free -h >> debug-info.txt
df -h >> debug-info.txt
echo "" >> debug-info.txt

echo "--- Network Status ---" >> debug-info.txt
netstat -tulpn | grep 8080 >> debug-info.txt
curl -s localhost:8080/api/v1/status >> debug-info.txt
echo "" >> debug-info.txt

echo "--- Node Logs ---" >> debug-info.txt
journalctl -u qnarwhalknight --no-pager -n 100 >> debug-info.txt

echo "Debug information collected in debug-info.txt"
```

### Performance Benchmarks
```bash
# Run comprehensive benchmarks
timeout 36000 cargo bench --workspace

# API performance test
wrk -t12 -c400 -d30s http://localhost:8080/api/v1/status

# Consensus performance
curl localhost:8080/api/v1/consensus/benchmarks

# Network throughput
iperf3 -c peer.example.com -p 8081 -t 60
```

---

## 📞 Support & Resources

### Documentation
- **Main Repository:** https://github.com/quantum-dag-labs/Q-NarwhalKnight
- **API Documentation:** http://localhost:8080/docs (when node is running)
- **Technical Papers:** `/papers/` directory in repository

### Community
- **Discord:** https://discord.gg/qnarwhalknight
- **Telegram:** https://t.me/qnarwhalknight
- **Forum:** https://forum.quantum-dag-labs.org

### Reporting Issues
```bash
# Collect debug info first
./debug-info.sh

# Submit issue with:
# 1. debug-info.txt contents
# 2. Steps to reproduce
# 3. Expected vs actual behavior
# 4. Configuration used
```

---

## 🔄 Updates & Maintenance

### Automatic Updates
```bash
# Enable automatic updates
echo "auto-update = true" >> ~/.config/qnarwhalknight/config.toml

# Check for updates
curl localhost:8080/api/v1/system/update/check

# Apply updates
curl -X POST localhost:8080/api/v1/system/update/apply
```

### Manual Updates
```bash
# Download new version
wget https://releases.q-narwhal.network/q-api-server-latest
chmod +x q-api-server-latest

# Stop node gracefully
curl -X POST localhost:8080/api/v1/system/shutdown

# Backup data
cp -r ~/.local/share/qnarwhalknight ~/.local/share/qnarwhalknight.backup

# Replace binary
mv q-api-server-latest q-api-server

# Restart node
./q-api-server
```

---

**Last Updated:** $(date)  
**Guide Version:** 1.0.0  
**Compatible Node Versions:** 1.0.0+

---

*This guide covers the complete deployment and distribution strategy for Q-NarwhalKnight nodes. For technical support, consult the troubleshooting section or reach out to the community channels.*