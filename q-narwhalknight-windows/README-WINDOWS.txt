===========================================
Q-NarwhalKnight Windows Build - v0.0.1-beta
===========================================

Quantum-Enhanced DAG-BFT Consensus System for Windows

BUILD INFORMATION:
- Version: v0.0.1-beta
- Target: x86_64-pc-windows-gnu
- Build Date: October 11, 2025
- Static Linking: YES (all dependencies bundled)

INCLUDED FILES:
- q-api-server.exe (81MB) - Main consensus server executable

SYSTEM REQUIREMENTS:
- Windows 10/11 (x86_64)
- 4GB RAM minimum (8GB+ recommended)
- 10GB free disk space
- Internet connection for P2P networking

QUICK START:

1. Extract all files to a directory (e.g., C:\Q-NarwhalKnight\)

2. Open Command Prompt or PowerShell in that directory

3. Start the server:
   q-api-server.exe --port 8080

4. The server will start on http://localhost:8080

5. Test the API:
   curl http://localhost:8080/api/v1/status

CONFIGURATION:

Environment Variables:
- Q_DB_PATH - Database storage path (default: ./data)
- Q_P2P_PORT - P2P networking port (default: 9000)

Example with custom database:
  set Q_DB_PATH=C:\Q-NarwhalKnight\data
  q-api-server.exe --port 8080

WALLET OPERATIONS:

1. Request Faucet (get test tokens):
   curl -X POST http://localhost:8080/api/v1/faucet ^
     -H "Content-Type: application/json" ^
     -d "{\"wallet_address\": \"qnkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}"

2. Check Balance:
   curl http://localhost:8080/api/v1/wallets/qnkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/balance

3. Send Transaction:
   curl -X POST http://localhost:8080/api/v1/transactions/send ^
     -H "Content-Type: application/json" ^
     -d "{\"from\": \"qnkaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", \"to\": \"qnkbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\", \"amount\": 5}"

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

TROUBLESHOOTING:

Issue: Port 8080 already in use
Solution: Use a different port: q-api-server.exe --port 8081

Issue: Database locked
Solution: Ensure no other instance is running. Delete ./data directory if needed.

Issue: Network connection errors
Solution: Check firewall settings. Allow q-api-server.exe through Windows Firewall.

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
- Compiler: rustc 1.70+ via cross-compilation
- Static linking: libgfortran, libgcc, OpenBLAS bundled
- Optimizations: --release with LTO and codegen-units=1

CHANGELOG (v0.0.1-beta):

[October 11, 2025]
✓ Fixed parallel worker race condition in transaction processing
✓ Added static linking for Windows (no DLL dependencies)
✓ Implemented real-time balance updates
✓ Added persistent transaction history
✓ Fixed wallet balance calculation accuracy

For the latest updates and detailed documentation, visit:
https://github.com/deme-plata/q-narwhalknight

===========================================
End of README - Happy quantum mining! ⚛️🚀
===========================================
