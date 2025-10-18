# Q-NarwhalKnight Linux v0.0.2-beta

## Post-Quantum Blockchain Node - Beta 2 Release

### What's New in v0.0.2-beta

✅ **Enhanced Shadow Mode Logging** - Real-time consensus metrics
✅ **ZK-STARK Batch Prover** - 535x efficiency gain for privacy proofs
✅ **Transaction Tunneling** - Ultra-low-latency fast paths (30-55% faster)
✅ **Improved Performance** - Overall 1.8-2.5x throughput increase

### Installation

1. Extract this archive:
```bash
tar -xzf q-narwhalknight-linux-v0.0.2-beta.tar.gz
cd q-narwhalknight-linux-v0.0.2-beta
```

2. Make the binary executable:
```bash
chmod +x q-api-server
```

3. Run the node:
```bash
./q-api-server --port 8080
```

### Configuration (Optional)

Set environment variables for custom configuration:

```bash
export Q_DB_PATH=./data          # Database directory
export Q_P2P_PORT=9001           # P2P networking port
export RUST_LOG=info             # Logging level
export Q_ENABLE_TUNNELING=true   # Enable transaction tunneling

./q-api-server --port 8080
```

### System Requirements

- **OS:** Ubuntu 20.04+, Debian 11+, or compatible Linux
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 10GB free space
- **CPU:** 2+ cores (4+ recommended for optimal performance)

### API Endpoints

- **Node Status:** http://localhost:8080/api/v1/status
- **Consensus Metrics:** http://localhost:8080/api/v1/consensus/resonance/status
- **Submit Transaction:** http://localhost:8080/api/v1/transactions

### Features

🔐 **Post-Quantum Security**
- Dilithium5 signatures (NIST PQC standard)
- Kyber1024 key exchange

⚡ **High Performance**
- DAG-Knight consensus with Resonance shadow mode
- ZK-STARK batch proving (5-10x efficiency)
- Transaction tunneling for whitelisted paths

🌐 **Network**
- libp2p-based P2P networking
- Automatic peer discovery (mDNS + DHT)
- NAT traversal with UPnP

### Troubleshooting

**Port already in use:**
```bash
./q-api-server --port 8090  # Use different port
```

**Database errors:**
```bash
rm -rf data/  # Clear database and restart
```

**Network issues:**
```bash
export Q_P2P_PORT=9002  # Use different P2P port
./q-api-server --port 8080
```

### Support

- **Documentation:** https://github.com/deme-plata/q-narwhalknight
- **Issues:** https://github.com/deme-plata/q-narwhalknight/issues

### License

Apache 2.0 - See LICENSE file

---

**Version:** v0.0.2-beta  
**Build Date:** October 16, 2025  
**Architecture:** x86_64 Linux
