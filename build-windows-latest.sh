#!/bin/bash
# Q-NarwhalKnight Windows Build Script with Latest Features
# Builds for x86_64-pc-windows-gnu target with all Phase 5b capabilities

set -e

echo "🪟 Q-NarwhalKnight Windows Build - Latest Features"
echo "=================================================="
echo ""
echo "Features included:"
echo "  ✅ Dual-stack peer discovery (mDNS + Kademlia DHT)"
echo "  ✅ Gossipsub consensus messaging"
echo "  ✅ libp2p 0.53 integration"
echo "  ✅ Bootstrap peer support"
echo "  ✅ Quantum consensus (DAG-Knight + Narwhal)"
echo "  ✅ Post-quantum cryptography (Dilithium5 + Kyber1024)"
echo "  ✅ SIMD optimizations"
echo "  ✅ Parallel workers (16x performance)"
echo "  ✅ io_uring kernel I/O (Linux target)"
echo ""

# Install Windows target if not already installed
echo "📦 Ensuring Windows target is installed..."
rustup target add x86_64-pc-windows-gnu

# Install mingw-w64 cross-compiler if not available
if ! command -v x86_64-w64-mingw32-gcc &> /dev/null; then
    echo "📦 Installing MinGW-w64 cross-compiler..."
    sudo apt-get update
    sudo apt-get install -y mingw-w64
fi

# Set environment variables for Windows build
export CC_x86_64_pc_windows_gnu=x86_64-w64-mingw32-gcc
export CXX_x86_64_pc_windows_gnu=x86_64-w64-mingw32-g++
export AR_x86_64_pc_windows_gnu=x86_64-w64-mingw32-ar
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc

# Build configuration
export RUSTFLAGS="-C target-cpu=native"

echo ""
echo "🔨 Building Q-NarwhalKnight for Windows (x86_64-pc-windows-gnu)..."
echo ""

# Build with 10-hour timeout (as per CLAUDE.md)
timeout 36000 cargo build --release \
    --target x86_64-pc-windows-gnu \
    --package q-api-server \
    2>&1 | tee /tmp/windows-build.log

# Check build status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Windows build completed successfully!"
    echo ""
    echo "📦 Binary location:"
    echo "   ./target/x86_64-pc-windows-gnu/release/q-api-server.exe"
    echo ""

    # Show binary size
    BINARY_SIZE=$(du -h target/x86_64-pc-windows-gnu/release/q-api-server.exe | cut -f1)
    echo "📊 Binary size: $BINARY_SIZE"
    echo ""

    # Create distribution package
    echo "📦 Creating Windows distribution package..."
    mkdir -p ./dist-windows
    cp target/x86_64-pc-windows-gnu/release/q-api-server.exe ./dist-windows/

    # Create README for Windows users
    cat > ./dist-windows/README.txt << 'EOF'
Q-NarwhalKnight - Quantum-Enhanced DAG-BFT Consensus
====================================================

Windows Installation Instructions
==================================

1. Extract this package to a directory (e.g., C:\Q-NarwhalKnight)

2. Run from Command Prompt or PowerShell:
   q-api-server.exe --port 8080

3. Environment Variables (optional):

   Network Configuration:
   - Q_P2P_PORT=9301          # libp2p listening port
   - Q_BOOTSTRAP_PEERS        # Comma-separated bootstrap peer multiaddrs
                              # Example: /ip4/1.2.3.4/tcp/9301/p2p/12D3Koo...

   Storage:
   - Q_DB_PATH=./data         # Database directory

   Logging:
   - RUST_LOG=info            # Log level (trace, debug, info, warn, error)
   - RUST_LOG=info,q_network::unified_network_manager=debug  # Detailed network logs

4. Example: Start with Bootstrap Peer

   Set environment variable:
   set Q_BOOTSTRAP_PEERS=/ip4/185.182.185.227/tcp/9301/p2p/12D3KooWAP3iKGmF1RAYHMc1cDtHrN69cLzXDXq7yCZk1MJdsHWT

   Run:
   q-api-server.exe --port 8080

5. Example: Custom Configuration

   set Q_DB_PATH=C:\Q-NarwhalKnight\data
   set Q_P2P_PORT=9301
   set RUST_LOG=info,q_network=debug
   q-api-server.exe --port 8080

Features
========

✅ Dual-Stack Peer Discovery
   - mDNS for local network discovery (~50ms)
   - Kademlia DHT for global internet discovery (5-30s)
   - Bootstrap peer support for clearnet connectivity

✅ Consensus
   - DAG-Knight zero-message ordering
   - Narwhal mempool with reliable broadcast
   - Quantum-enhanced VDF anchor election
   - BFT with 3f+1 validator tolerance

✅ Cryptography
   - Phase 0: Ed25519 (classical)
   - Phase 1: Dilithium5 + Kyber1024 (post-quantum)
   - Crypto-agile framework for seamless migration

✅ Performance
   - SIMD vectorization
   - 16 parallel workers
   - Target: 349,072 TPS (projected)
   - Kernel I/O optimization (io_uring on Linux)

API Endpoints
=============

Once running, access the API at:
- http://localhost:8080/health
- http://localhost:8080/api/transactions
- http://localhost:8080/api/balances
- http://localhost:8080/metrics (Prometheus metrics)

Network Discovery
=================

The node will automatically:
1. Discover peers on the same local network via mDNS
2. Connect to bootstrap peers if Q_BOOTSTRAP_PEERS is set
3. Discover additional peers via Kademlia DHT
4. Form Gossipsub mesh for consensus messaging

Firewall Configuration
======================

Allow incoming connections on:
- TCP port 8080 (API server)
- TCP port 9301 (libp2p P2P, default Q_P2P_PORT)

For NAT traversal, ensure port forwarding is configured.

Support
=======

Documentation: https://github.com/deme-plata/q-narwhalknight
Issues: https://github.com/deme-plata/q-narwhalknight/issues

Built with: Rust + libp2p 0.53 + Quantum Consensus
License: MIT
EOF

    # Create batch file for easy startup
    cat > ./dist-windows/start-node.bat << 'EOF'
@echo off
echo Q-NarwhalKnight - Starting Node
echo ===============================
echo.

REM Set default configuration
if not defined Q_DB_PATH set Q_DB_PATH=.\data
if not defined Q_P2P_PORT set Q_P2P_PORT=9301
if not defined RUST_LOG set RUST_LOG=info

echo Configuration:
echo   Database: %Q_DB_PATH%
echo   P2P Port: %Q_P2P_PORT%
echo   Log Level: %RUST_LOG%
echo.

REM Check for bootstrap peers
if defined Q_BOOTSTRAP_PEERS (
    echo Bootstrap Peers: %Q_BOOTSTRAP_PEERS%
    echo.
)

REM Create data directory if it doesn't exist
if not exist "%Q_DB_PATH%" mkdir "%Q_DB_PATH%"

echo Starting Q-NarwhalKnight API Server...
echo.

q-api-server.exe --port 8080

pause
EOF

    # Create batch file for bootstrap node configuration
    cat > ./dist-windows/start-with-bootstrap.bat << 'EOF'
@echo off
echo Q-NarwhalKnight - Starting with Bootstrap Peer
echo ===============================================
echo.

REM Configure bootstrap peer (EDIT THIS LINE)
set Q_BOOTSTRAP_PEERS=/ip4/185.182.185.227/tcp/9301/p2p/12D3KooWAP3iKGmF1RAYHMc1cDtHrN69cLzXDXq7yCZk1MJdsHWT

REM Set default configuration
set Q_DB_PATH=.\data
set Q_P2P_PORT=9301
set RUST_LOG=info,q_network::unified_network_manager=debug

echo Configuration:
echo   Database: %Q_DB_PATH%
echo   P2P Port: %Q_P2P_PORT%
echo   Log Level: %RUST_LOG%
echo   Bootstrap: %Q_BOOTSTRAP_PEERS%
echo.

REM Create data directory
if not exist "%Q_DB_PATH%" mkdir "%Q_DB_PATH%"

echo Starting Q-NarwhalKnight with Kademlia DHT bootstrap...
echo.

q-api-server.exe --port 8080

pause
EOF

    echo "✅ Distribution package created: ./dist-windows/"
    echo ""
    echo "📋 Package contents:"
    ls -lh ./dist-windows/
    echo ""
    echo "🎉 Windows build complete!"
    echo ""
    echo "To deploy on Windows:"
    echo "  1. Copy ./dist-windows/ to Windows machine"
    echo "  2. Run start-node.bat (local network only)"
    echo "     OR"
    echo "     Edit start-with-bootstrap.bat and run (connects to global network)"
    echo ""
else
    echo ""
    echo "❌ Windows build failed!"
    echo ""
    echo "Check logs: /tmp/windows-build.log"
    exit 1
fi
