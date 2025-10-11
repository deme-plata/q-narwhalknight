Q-NarwhalKnight Windows Build

Version: 0.0.1-beta
Architecture: x86_64-pc-windows-gnu
Built: Sat 11 Oct 17:46:43 CEST 2025

## Running the Server

1. Open Command Prompt or PowerShell
2. Navigate to this directory
3. Run: q-api-server.exe --port 8080

## Requirements
- Windows 10/11 (64-bit)
- No additional dependencies required (statically linked)

## Default Ports
- HTTP API: 8080
- P2P Network: 8081

## Environment Variables (optional)
- Q_DB_PATH: Database storage path (default: ./data)
- Q_P2P_PORT: P2P listening port (default: 9001)
- RUST_LOG: Log level (default: info)

## Quantum Features
✅ DAG-Knight consensus
✅ Post-quantum cryptography (Dilithium5 + Kyber1024)
✅ Zero-knowledge proofs (STARK + SNARK)
✅ Real-time WebSocket streaming
✅ High-performance SIMD cryptography

## Support
GitHub: https://github.com/deme-plata/q-narwhalknight

