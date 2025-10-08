#!/bin/bash
# Q-NarwhalKnight Windows Build Script
# Cross-compiles miner for Windows x86_64

set -e

echo "🪟 Q-NarwhalKnight Windows x86_64 Build"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if target is installed
if ! rustup target list --installed | grep -q "x86_64-pc-windows-gnu"; then
    echo -e "${YELLOW}Installing Windows target...${NC}"
    rustup target add x86_64-pc-windows-gnu
fi

# Check if MinGW is installed
if ! command -v x86_64-w64-mingw32-gcc &> /dev/null; then
    echo -e "${YELLOW}MinGW not found. Please install: apt-get install mingw-w64${NC}"
    exit 1
fi

echo -e "${CYAN}Building Q-Miner for Windows x86_64...${NC}"
echo ""

# Build the miner with optimizations
RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
    cargo build \
    --release \
    --target x86_64-pc-windows-gnu \
    --package q-miner \
    --bin q-miner

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo "Windows executable:"
    ls -lh target/x86_64-pc-windows-gnu/release/q-miner.exe
    echo ""

    # Create downloads directory
    mkdir -p gui/quantum-wallet/dist-final/downloads

    # Copy to downloads
    cp target/x86_64-pc-windows-gnu/release/q-miner.exe \
       gui/quantum-wallet/dist-final/downloads/q-miner-windows-x64.exe

    echo -e "${GREEN}✓ Copied to: gui/quantum-wallet/dist-final/downloads/q-miner-windows-x64.exe${NC}"
    echo ""
    echo "File size: $(du -h target/x86_64-pc-windows-gnu/release/q-miner.exe | cut -f1)"
    echo ""
    echo -e "${CYAN}Ready for download at: http://localhost:8080/downloads/q-miner-windows-x64.exe${NC}"
    echo ""
    echo "To test on Windows:"
    echo "  1. Download q-miner-windows-x64.exe"
    echo "  2. Open PowerShell or CMD"
    echo "  3. Run: .\\q-miner-windows-x64.exe --mode solo --wallet YOUR_ADDRESS --threads 8"
    echo ""
else
    echo -e "${YELLOW}❌ Build failed${NC}"
    exit 1
fi