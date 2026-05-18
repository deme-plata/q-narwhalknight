# Quillon Graph

![Quillon Graph — quantum-resistant DAG-BFT blockchain](docs/images/architecture-hero.png)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/built_with-Rust-orange?logo=rust)](https://www.rust-lang.org/)
[![Network](https://img.shields.io/badge/network-mainnet--genesis-green)](https://quillon.xyz)
[![Signed commits](https://img.shields.io/badge/commits-GPG_signed-success)](https://github.com/deme-plata/q-narwhalknight/commits/main)

> Quantum-resistant DAG-BFT blockchain — DAG-Knight consensus over a Narwhal mempool, post-quantum signatures (Dilithium-5, Kyber-1024), BLAKE3-keyed commitments, and a Sparse Merkle Tree balance commitment. Production network running on mainnet-genesis with multi-server HA.

Live at **[quillon.xyz](https://quillon.xyz)**. Repository tag: `v10.9.55` and counting.

---

## At a glance

| | |
|---|---|
| **Consensus** | DAG-Knight with VDF-based anchor election |
| **Mempool** | Narwhal with reliable broadcast (Bracha RB) |
| **Signatures** | Dilithium-5 + Ed25519 hybrid |
| **KEM** | Kyber-1024 |
| **Hash** | BLAKE3-keyed |
| **Block time** | ~1 second |
| **Network** | mainnet-genesis (live), 4 production bootstrap nodes |

---

## System architecture

![System architecture diagram](docs/images/architecture-diagram.png)

The original phased description is preserved below for historical context. See `docs/technical-review-sparse-chain-truth-v1.md` for the empirical chain-state truth and `docs/v10.9.55-balance-replay-removal-2026-05-18.md` for the most recent operational lesson.

---

🌟 **Quantum-Enhanced DAG-BFT Consensus with Distributed AI & Triple-Layer Anonymity**

Q-NarwhalKnight is a blockchain consensus system that combines the efficiency of DAG-Knight consensus with Narwhal mempool, quantum-ready cryptographic primitives, and production-grade Tor integration for complete network anonymity. This implementation provides a phased approach to quantum-resistance, starting with classical cryptography (Phase 0) and progressively upgrading to full quantum protocols (Phase 4).

## 🔬 Architecture Overview

Q-NarwhalKnight implements a four-tier quantum threat model with seamless cryptographic agility:

### Phase 0: Classical Foundation (Current Implementation)
- **Consensus**: DAG-Knight with quantum-enhanced anchor election
- **Mempool**: Narwhal with reliable broadcast (Bracha's protocol)
- **Networking**: libp2p with Ed25519 + QUIC transport
- **Ordering**: Zero-message complexity BFT with VDF-based randomness

### Phase 1: Post-Quantum Transition (Implemented)
- **Crypto-Agile Framework**: Seamless algorithm negotiation
- **Signatures**: Dilithium5 (post-quantum digital signatures)  
- **Key Exchange**: Kyber1024 (post-quantum KEM)
- **Hybrid Security**: Classical + post-quantum dual protection

### Phase 2-4: Quantum Future (Planned)
- **Quantum Random Number Generation** (QRNG)
- **Quantum Key Distribution** (QKD) networking
- **Lattice-based Verifiable Random Functions**
- **STARK-only zkVM for quantum-resistant proofs**

## 🚀 Quick Start

**Current release line:** `v10.9.x` · **Network:** `mainnet-genesis` · **Live frontend:** [quillon.xyz](https://quillon.xyz)

You don't have to build the source to use Quillon Graph. Most users only need the wallet (in the browser) or the miner (one binary download).

### Option 1 — Use the web wallet (no install)

Open **[quillon.xyz](https://quillon.xyz)** in any modern browser. You can create or import a wallet, send QUG, swap on the DEX, and watch the live chain. Everything is in-browser — keys never leave your machine.

> The web wallet validates BIP39 mnemonics with checksum. Don't use weak phrases like `test test test ...` — they collide to shared addresses (see [`docs/brainwallet_bip39_incident.md`](docs/brainwallet_bip39_incident.md)).

### Option 2 — Run the miner against the live network

```bash
# Download the latest miner (linux x86_64)
wget https://quillon.xyz/downloads/q-miner-linux-x64
chmod +x q-miner-linux-x64

# Solo-mine against quillon.xyz (TLS, load-balanced)
./q-miner-linux-x64 \
  --mode solo \
  --wallet qnk<your-64-char-hex-address> \
  --threads 8 \
  --intensity 9 \
  --server https://quillon.xyz

# Benchmark your hardware first
./q-miner-linux-x64 --benchmark --threads 16 --duration 60
```

`--server https://quillon.xyz` routes through the production nginx load balancer (Beta primary, Gamma backup) so you don't need to pin to any single node.

### Option 3 — Run a full node

```bash
# Download the latest node binary (linux x86_64, glibc 2.36+)
wget https://quillon.xyz/downloads/q-api-server-linux-x86_64
chmod +x q-api-server-linux-x86_64

# Start (auto-creates ./data-mainnet-genesis/)
./q-api-server-linux-x86_64 --port 8080
```

The node auto-discovers bootstrap peers (Epsilon supernode at `89.149.241.126:9001`, plus Beta/Gamma/Delta) and syncs from genesis. First sync takes ~5.5 hours on a 10 Gbit/s line. The REST API + SSE stream listen on `:8080`.

> ⚠️ Verify the binary path matches your distro: Debian 12 / Ubuntu 22.04+ work. Older glibc (< 2.36) will fail with `GLIBC_2.38 not found`.

### Option 4 — Build from source

```bash
# Prerequisites: Rust 1.70+, build-essential, libssl-dev, pkg-config, cmake, clang, libudev-dev
git clone https://github.com/deme-plata/q-narwhalknight.git
cd q-narwhalknight

# Build (use a 10-hour timeout; post-quantum crates take ~30 min on first build)
timeout 36000 cargo build --release --package q-api-server
timeout 36000 cargo build --release --package q-miner

# Binaries land in target/release/
./target/release/q-api-server --port 8080
```

### Configuration

The node reads `./.env` and CLI args. Minimal `.env`:

```env
Q_NETWORK_ID=mainnet-genesis
Q_DB_PATH=./data-mainnet-genesis
Q_P2P_PORT=9001
RUST_LOG=warn
ROCKSDB_BLOCK_CACHE_MB=2048
```

For the consensus / crypto parameters see [`docs/spec-10ms-verification-2026-05-16.tex`](docs/spec-10ms-verification-2026-05-16.tex) and [`docs/phase1-pq-pure-kravspesifikasjon.md`](docs/phase1-pq-pure-kravspesifikasjon.md).

## 🏗️ Project Structure

```
Q-NarwhalKnight/
├── crates/
│   ├── q-types/           # Core type definitions and primitives
│   ├── q-wallet/          # Wallet management and key handling  
│   ├── q-api-server/      # REST API and real-time streaming
│   ├── q-visualizer/      # Quantum state visualization
│   ├── q-narwhal-core/    # Narwhal mempool implementation
│   ├── q-dag-knight/      # DAG-Knight consensus engine
│   └── q-network/         # libp2p networking with crypto-agility
├── papers/                # Academic papers and documentation
└── docs/                  # Additional documentation
```

## 🎯 Key Features

### ⚡ High Performance
- **Zero-message complexity**: No additional consensus communication overhead
- **Parallel processing**: Asynchronous vertex processing and validation
- **Stream processing**: Real-time updates with <50ms latency
- **Scalable architecture**: Supports thousands of validators

### 🔐 Quantum-Ready Security
- **Cryptographic agility**: Hot-swappable algorithm suites
- **Post-quantum signatures**: Dilithium5 lattice-based signatures
- **Hybrid security**: Classical + quantum-resistant dual protection
- **VDF-based randomness**: Quantum-enhanced verifiable delay functions

### 🌐 Advanced Networking & Anonymity
- **libp2p integration**: Modern P2P networking with QUIC transport
- **Tor integration**: Embedded Arti Tor client for complete network anonymity
- **Triple-layer anonymity**: Tor + Quantum mixing + VDF-based unlinkability
- **Gossip protocol**: Efficient message propagation
- **Peer discovery**: Capability-aware peer management with .onion support
- **Network resilience**: Byzantine-fault-tolerant networking
- **Zero IP leakage**: All validator communications routed through Tor circuits

### 📊 Consensus Innovation
- **DAG-Knight ordering**: Deterministic transaction ordering
- **Quantum anchor election**: VDF-based leader selection
- **Narwhal mempool**: High-throughput transaction batching
- **Commit protocols**: Multiple commit paths for optimal latency

### ⛏️ High-Performance Mining
- **Multi-threaded CPU mining**: Optimized DAG-Knight PoW with VDF verification
- **Remote mining support**: Connect to any node via `--server` parameter
- **Real-time updates**: Server-Sent Events (SSE) for instant reward notifications
- **Adaptive difficulty**: Dynamic adjustment based on network hashrate
- **Quantum-resistant PoW**: VDF-enhanced mining algorithm
- **Benchmark mode**: Performance testing and hardware optimization

### 💰 **Transparent Development Fee** (v0.2.0-beta)

Q-NarwhalKnight implements a **1% development fee** on all mining rewards to ensure sustainable, long-term project development.

**Reward Split:**
- **99%** → Miner (you)
- **1%** → Development Fund (founder wallet)

**Example:** If block reward is 2.0 QNK, you receive **1.98 QNK** and **0.02 QNK** funds development.

**What This Funds:**
- Core protocol development and bug fixes
- Post-quantum cryptography research (Phases 2-4)
- Network infrastructure and bootstrap nodes
- Security audits and penetration testing
- Academic publications and peer review
- Community support and documentation

**Transparency:**
- **Founder Wallet**: `qnk8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f1a2b3c4d5e6f7a8b9c0d1e2f3a`
- **Full Documentation**: See [`DEVELOPMENT_FEE_TRANSPARENCY.md`](./DEVELOPMENT_FEE_TRANSPARENCY.md)
- **Open Source**: All fee logic is visible in the source code
- **On-Chain Verification**: Founder wallet balance is publicly auditable

**Security:** The network uses AEGIS-QL post-quantum authentication to ensure only authorized miners can participate, preventing unauthorized forks that bypass the development fee.

## 🔬 Research & Papers

This implementation is based on cutting-edge research in:
- **DAG-Knight**: Asynchronous BFT consensus ([DISC 2021](https://arxiv.org/abs/2102.08325))
- **Narwhal**: High-throughput mempool ([EuroSys 2022](https://arxiv.org/abs/2105.11827))
- **Post-Quantum Cryptography**: NIST standardized algorithms
- **Quantum Networking**: QKD and quantum-enhanced protocols

See `papers/quantum-aesthetics.pdf` for our comprehensive analysis of quantum consensus aesthetics.

## 🧪 Testing & Development

### Unit Tests
```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p q-dag-knight

# Run with logging
RUST_LOG=debug cargo test
```

### Integration Tests
```bash
# End-to-end consensus test
cargo test --test integration_consensus

# Network layer tests  
cargo test --test network_integration

# Performance benchmarks
cargo bench
```

### Load Testing
```bash
# Simulate high-throughput scenario
cargo run --bin q-load-tester -- \
    --nodes 4 \
    --transactions-per-second 1000 \
    --duration 60s
```

## 🌟 API Documentation

### REST Endpoints
```
POST   /wallets              # Create new wallet
GET    /wallets/{id}          # Get wallet info
POST   /transactions         # Submit transaction
GET    /consensus/status      # Get consensus state
GET    /network/peers         # List connected peers
GET    /metrics              # Prometheus metrics
```

### WebSocket Streams
```
/ws/blocks                   # Real-time block notifications
/ws/transactions            # Transaction confirmations
/ws/consensus              # Consensus state updates
/ws/quantum/visualization  # Quantum state visualization
```

### Server-Sent Events
```
/stream/consensus          # Consensus events
/stream/network           # Network events  
/stream/quantum/beacons   # Quantum beacon updates
```

## 🎨 Quantum Visualization

Q-NarwhalKnight includes advanced quantum state visualization:

- **Rainbow-box quantum states**: Multi-dimensional qubit representation
- **DAG entanglement patterns**: Moiré interference visualization
- **QKD photon waterfalls**: Real-time quantum key distribution
- **STARK proof fractals**: Zero-knowledge proof visualization

Access visualizations at `/quantum/visualization` or via WebSocket streams.

## 💼 Dual Wallet Implementations

Q-NarwhalKnight offers two complementary wallet implementations:

### 🦀 Slint Native Wallet (`/gui`)
- **Technology**: Rust + Slint UI Framework
- **Platform**: Native desktop application (Linux, Windows, macOS)
- **Performance**: Zero-overhead native performance
- **Features**:
  - DAG consensus visualization with real-time updates
  - Network topology monitoring
  - Tor anonymity integration
  - Quantum entropy visualization
  - Narwhal mempool explorer
  - Phase transition monitoring
  - LVRF randomness visualization
  - VDF progress tracking
  - Canvas-based quantum visualizations

### ⚛️ React TypeScript Wallet (`/gui/quantum-wallet`)
- **Technology**: Vite + React + TypeScript
- **Platform**: Web-based PWA (Progressive Web App)
- **Performance**: Optimized with code splitting and lazy loading
- **Features**:
  - Modern responsive UI with dark mode
  - Wallet authentication with post-quantum signatures
  - Real-time transaction monitoring via SSE
  - DEX trading interface with liquidity pools
  - Mining dashboard with GPU/CPU support
  - CDP vault management for stablecoins
  - QR code scanner for mobile transactions
  - Session timeout with encrypted key storage
  - Token management and custom token support
  - Stripe payment integration
  - Download node packages directly from UI

Both wallets share the same backend API and provide complementary user experiences - native performance for power users and web accessibility for general users.

## 🤝 Contributing

We welcome contributions! Please see `CLAUDE.md` for detailed contribution guidelines and multi-server development processes.

### Development Setup
1. **Clone and build** the repository
2. **Review** architecture in `docs/` 
3. **Check** open issues and project roadmap
4. **Submit** pull requests with comprehensive tests

### Multi-Server Development
See `CLAUDE.md` for instructions on:
- Setting up distributed development environments
- Using shared `/mnt` folders across servers
- Coordinating through GitLab pipelines
- Contributing code via Claude Code integration

## 📋 Roadmap

### Phase 0 (Current) - Classical Foundation
- [x] DAG-Knight consensus engine
- [x] Narwhal mempool with reliable broadcast  
- [x] libp2p networking with gossip protocol
- [x] REST API and real-time streaming
- [x] Quantum state visualization
- [ ] Performance optimization and benchmarking

### Phase 1 (In Progress) - Post-Quantum Transition
- [x] Cryptographic agility framework
- [x] Dilithium5/Kyber1024 integration
- [x] Algorithm negotiation protocol
- [ ] Hybrid classical+post-quantum mode
- [ ] Migration tools and compatibility

### Phase 2 (Planned) - Quantum Enhancement
- [ ] QRNG hardware integration
- [ ] Lattice-based VRF implementation
- [ ] Quantum-enhanced VDF protocols
- [ ] STARK-only zkVM integration

### Phase 3-4 (Research) - Full Quantum
- [ ] QKD networking protocols
- [ ] Quantum fair queueing
- [ ] Advanced quantum consensus protocols
- [ ] Quantum error correction integration

## 🏷️ Version History

- **v0.0.3-beta** (Current): Production Beta Release
  - Dual wallet implementations: Slint (native Rust) + React TypeScript (web)
  - Full wallet authentication with post-quantum signatures (SHA-512 + noble-ed25519)
  - DEX integration with liquidity pools and real-time swap functionality
  - Session management with auto-timeout and encrypted key storage
  - Mining dashboard with SSE real-time updates
  - Transaction tunneling for privacy-preserving transfers
  - Database replication for high availability
  - AEGIS-QL quantum entanglement coordination language
  - Higgs-Hydro quantum vacuum simulation
  - Reticular chemistry robots for material synthesis
  - ZK-STARK batch prover integration
  - Comprehensive API documentation and testing suite
  - IPFS + RocksDB distributed storage layer
  - QUG/QUGUSD stablecoin with CDP vaults
  - Nitro Points loyalty system
  - Payment gateway integration (Stripe)

- **v0.0.2-beta**: Enhanced Features Release
  - Post-quantum wallet authentication
  - Real-time DEX with liquidity management
  - Privacy enhancements and transaction tunneling
  - Database replication architecture

- **v0.0.1-alpha**: Initial implementation with Phase 0 consensus and Phase 1 crypto-agility

- **v0.1.0** (Planned): Production-ready Phase 0 with performance optimizations
- **v0.2.0** (Planned): Complete Phase 1 post-quantum transition

## 📄 License

Apache-2.0 License - see `LICENSE` file for details.

## 🙏 Acknowledgments

- **DAG-Knight team** for the foundational consensus research
- **Narwhal/Bullshark authors** for mempool design
- **libp2p community** for networking infrastructure  
- **NIST** for post-quantum cryptography standards
- **Quantum-DAG Labs** for quantum consensus research

---

**Building the quantum-ready future of distributed consensus** ⚛️🚀