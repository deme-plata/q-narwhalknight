===========================================
Q-NarwhalKnight Linux Build - v0.0.1-beta
===========================================

Quantum-Enhanced DAG-BFT Consensus System for Linux

BUILD INFORMATION:
- Version: v0.0.1-beta
- Target: x86_64-unknown-linux-gnu
- Build Date: October 12, 2025
- Optimizations: Release mode with LTO

INCLUDED FILES:
- q-api-server (41MB) - Main consensus server executable
- README-LINUX.txt - This file
- README.md - Project documentation

SYSTEM REQUIREMENTS:
- Linux (Ubuntu 20.04+, Debian 11+, or similar)
- x86_64 architecture
- 4GB RAM minimum (8GB+ recommended)
- 10GB free disk space
- Internet connection for P2P networking

QUICK START:

1. Extract all files to a directory (e.g., ~/Q-NarwhalKnight/)

   tar -xzf q-narwhalknight-linux-v0.0.1-beta.tar.gz
   cd q-narwhalknight-linux

2. Make executable (if needed):

   chmod +x q-api-server

3. Start the server:

   ./q-api-server --port 8080

4. The server will start on http://localhost:8080

5. Test the API:

   curl http://localhost:8080/api/v1/status

CONFIGURATION:

Environment Variables:
- Q_DB_PATH - Database storage path (default: ./data)
- Q_P2P_PORT - P2P networking port (default: 9000)

Example with custom database:

  export Q_DB_PATH=~/q-data
  ./q-api-server --port 8080

WALLET OPERATIONS:

1. Request Faucet (get test tokens):

   curl -X POST http://localhost:8080/api/v1/faucet \
     -H "Content-Type: application/json" \
     -d '{"wallet_address": "qnkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'

2. Check Balance:

   curl http://localhost:8080/api/v1/wallets/qnkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/balance

3. Send Transaction:

   curl -X POST http://localhost:8080/api/v1/transactions/send \
     -H "Content-Type: application/json" \
     -d '{"from": "qnkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "to": "qnkbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "amount": 5}'

API ENDPOINTS:

- GET  /api/v1/status - Server status and version
- GET  /api/v1/wallets/{address}/balance - Check wallet balance
- POST /api/v1/faucet - Request test tokens
- POST /api/v1/transactions/send - Send QNK tokens
- GET  /api/v1/transactions/{hash} - Get transaction details
- GET  /api/v1/dag/stats - DAG consensus statistics

FEATURES:

✓ DAG-Knight Consensus - Zero-message complexity BFT
✓ Parallel Transaction Processing - 16 worker threads
✓ Quantum-Ready Cryptography - Phase 0 (Ed25519) & Phase 1 (Dilithium5/Kyber1024)
✓ Real-time Balance Updates - Correct transaction processing
✓ P2P Networking - libp2p with automatic peer discovery
✓ Persistent Storage - RocksDB-based transaction and balance storage

PERFORMANCE:

- Throughput: 48k+ TPS (transactions per second)
- Latency: <2.3s finality with local validators
- Scalability: 50+ concurrent peer connections

RUNNING AS A SERVICE (systemd):

Create a systemd service file:

sudo nano /etc/systemd/system/q-narwhalknight.service

[Unit]
Description=Q-NarwhalKnight Quantum Consensus Node
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/q-narwhalknight-linux
Environment="Q_DB_PATH=/var/lib/q-narwhalknight"
ExecStart=/home/youruser/q-narwhalknight-linux/q-api-server --port 8080
Restart=always

[Install]
WantedBy=multi-user.target

Then enable and start:

sudo systemctl daemon-reload
sudo systemctl enable q-narwhalknight
sudo systemctl start q-narwhalknight
sudo systemctl status q-narwhalknight

TROUBLESHOOTING:

Issue: Port 8080 already in use
Solution: Use a different port: ./q-api-server --port 8081

Issue: Database locked
Solution: Ensure no other instance is running. Delete ./data directory if needed.

Issue: Network connection errors
Solution: Check firewall settings:
  sudo ufw allow 8080/tcp
  sudo ufw allow 9000/tcp

Issue: Permission denied
Solution: Make executable: chmod +x q-api-server

MULTI-NODE SETUP:

To run multiple nodes for testing:

# Terminal 1 - Node 1
Q_DB_PATH=./data-node1 Q_P2P_PORT=9001 ./q-api-server --port 8001 --node-id node1

# Terminal 2 - Node 2
Q_DB_PATH=./data-node2 Q_P2P_PORT=9002 ./q-api-server --port 8002 --node-id node2

# Terminal 3 - Node 3
Q_DB_PATH=./data-node3 Q_P2P_PORT=9003 ./q-api-server --port 8003 --node-id node3

Nodes will automatically discover each other via P2P networking.

SUPPORT:

GitHub: https://github.com/deme-plata/q-narwhalknight
Documentation: See included technical papers
License: MIT License

TECHNICAL DETAILS:

Architecture:
- Consensus: DAG-Knight with VDF-based anchor election
- Mempool: Narwhal reliable broadcast protocol
- Cryptography: Crypto-agile (Ed25519 → Dilithium5 transition)
- Networking: libp2p with Tor integration (Phase 2)
- Storage: RocksDB with atomic transaction guarantees

Build Configuration:
- Compiler: rustc 1.70+
- Optimizations: --release with LTO and codegen-units=1
- Static linking: All dependencies bundled

CHANGELOG (v0.0.1-beta):

[October 12, 2025]
✓ Fixed parallel worker race condition in transaction processing
✓ Implemented real-time balance updates
✓ Added persistent transaction history
✓ Fixed wallet balance calculation accuracy
✓ Completed Linux distribution package

For the latest updates and detailed documentation, visit:
https://github.com/deme-plata/q-narwhalknight

===========================================
End of README - Happy quantum mining! ⚛️🚀
===========================================
