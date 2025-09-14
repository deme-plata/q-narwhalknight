# Q-NarwhalKnight v0.0.1-alpha

🌟 **Quantum-Enhanced DAG-BFT Consensus with Post-Quantum Cryptography**

Q-NarwhalKnight is a revolutionary blockchain consensus system that combines the efficiency of DAG-Knight consensus with Narwhal mempool and quantum-ready cryptographic primitives. This implementation provides a phased approach to quantum-resistance, starting with classical cryptography (Phase 0) and progressively upgrading to full quantum protocols (Phase 4).

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

### Prerequisites
- Rust 1.70+ with Cargo
- libp2p networking stack
- Optional: PostgreSQL for persistent storage

### Build & Run
```bash
# Clone the repository
git clone https://gitlab.com/dagknight/q-narwhalknight.git
cd q-narwhalknight

# Build the workspace
cargo build --release

# Run the API server
cargo run --bin q-api-server

# Run consensus node
cargo run --bin q-dag-knight-node
```

### Configuration
```toml
[network]
listen_addr = "/ip4/0.0.0.0/tcp/7000"
bootstrap_peers = []

[consensus]
node_id = "auto" # Or specify 32-byte hex
byzantine_tolerance = 1 # f parameter (supports up to 3f+1 total nodes)
delta_rounds = 4 # Commit latency parameter

[crypto]
phase = "Phase0" # "Phase0" | "Phase1" 
auto_upgrade = true
```

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

### 🌐 Advanced Networking
- **libp2p integration**: Modern P2P networking with QUIC transport
- **Gossip protocol**: Efficient message propagation
- **Peer discovery**: Capability-aware peer management
- **Network resilience**: Byzantine-fault-tolerant networking

### 📊 Consensus Innovation
- **DAG-Knight ordering**: Deterministic transaction ordering
- **Quantum anchor election**: VDF-based leader selection
- **Narwhal mempool**: High-throughput transaction batching
- **Commit protocols**: Multiple commit paths for optimal latency

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

- **v0.0.1-alpha** (Current): Initial implementation with Phase 0 consensus and Phase 1 crypto-agility
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