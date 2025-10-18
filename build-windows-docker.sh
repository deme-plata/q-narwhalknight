#!/bin/bash
# Q-NarwhalKnight Windows Cross-Compilation Script
# Uses Docker to build Windows binaries with proper MinGW-w64 toolchain

set -e

echo "🪟 Q-NarwhalKnight Windows Build (Docker)"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${BLUE}📦 Building Docker image for Windows cross-compilation...${NC}"
docker build -f Dockerfile.windows -t q-narwhalknight-windows-builder .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker image build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker image built successfully${NC}"
echo ""

# Create output directory
mkdir -p q-narwhalknight-windows

echo -e "${BLUE}🔨 Compiling Q-NarwhalKnight for Windows (x86_64)...${NC}"
echo -e "${YELLOW}⏳ This may take 10-30 minutes depending on your system...${NC}"
echo ""

# Run the build inside Docker container
docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    q-narwhalknight-windows-builder \
    cargo build --release --target x86_64-pc-windows-gnu --package q-miner --bin q-miner

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Windows compilation failed!${NC}"
    echo "Check the error messages above for details."
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Compilation successful!${NC}"
echo ""

# Copy the Windows executable
echo -e "${BLUE}📦 Packaging Windows binaries...${NC}"
cp target/x86_64-pc-windows-gnu/release/q-miner.exe q-narwhalknight-windows/
cp README.md q-narwhalknight-windows/ 2>/dev/null || echo "Note: README.md not found"
cp LICENSE q-narwhalknight-windows/ 2>/dev/null || echo "Note: LICENSE not found"

# Check if executable was created
if [ -f "q-narwhalknight-windows/q-miner.exe" ]; then
    SIZE=$(du -h q-narwhalknight-windows/q-miner.exe | cut -f1)
    echo -e "${GREEN}✅ Windows executable created: q-miner.exe (${SIZE})${NC}"
else
    echo -e "${RED}❌ Failed to create Windows executable!${NC}"
    exit 1
fi

# Copy required MinGW DLLs
echo -e "${BLUE}📦 Extracting required Windows DLLs...${NC}"
docker run --rm \
    -v "$(pwd)/q-narwhalknight-windows:/output" \
    q-narwhalknight-windows-builder \
    bash -c "cp /usr/x86_64-w64-mingw32/lib/libwinpthread-1.dll /output/ 2>/dev/null || true"

# Create README for Windows users
cat > q-narwhalknight-windows/README-WINDOWS.txt << 'EOF'
Q-NarwhalKnight Miner - Windows Edition
========================================

Quantum-Enhanced DAG-BFT Consensus Mining

QUICK START
-----------
1. Open Command Prompt or PowerShell
2. Navigate to this directory
3. Run: q-miner.exe --wallet YOUR_WALLET_ADDRESS --threads 4 --intensity 5

EXAMPLE
-------
q-miner.exe --wallet qnka96c3f02158455d4de43549296f9b984e0f43c3f7ae79e227905dd5378ea4df5 --threads 8 --intensity 7

PARAMETERS
----------
--wallet      Your QNK wallet address (required)
--threads     Number of CPU threads to use (default: 4)
--intensity   Mining intensity 1-10 (default: 5)
--mode        Mining mode: solo or pool (default: solo)
--api-url     API server URL (default: http://localhost:8080)

SYSTEM REQUIREMENTS
-------------------
- Windows 10/11 (64-bit)
- 4GB RAM minimum
- Multi-core CPU recommended
- Internet connection

TROUBLESHOOTING
---------------
If you get "VCRUNTIME140.dll not found" error:
- Install Visual C++ Redistributable from Microsoft

If mining is slow:
- Reduce --threads value
- Reduce --intensity value
- Close other applications

SUPPORT
-------
GitHub: https://github.com/deme-plata/q-narwhalknight
Documentation: https://q-narwhalknight.dev/docs

LICENSE
-------
Apache 2.0
EOF

echo -e "${GREEN}✅ README created for Windows users${NC}"
echo ""

# Create a zip archive
echo -e "${BLUE}📦 Creating distribution archive...${NC}"
ZIP_NAME="q-narwhalknight-windows-v0.0.2-beta.zip"

if command -v zip &> /dev/null; then
    cd q-narwhalknight-windows
    zip -r "../${ZIP_NAME}" .
    cd ..
    echo -e "${GREEN}✅ Archive created: ${ZIP_NAME}${NC}"
else
    echo -e "${YELLOW}⚠️  'zip' command not found. Creating tar.gz instead...${NC}"
    tar -czf "q-narwhalknight-windows-v0.0.2-beta.tar.gz" q-narwhalknight-windows/
    echo -e "${GREEN}✅ Archive created: q-narwhalknight-windows-v0.0.2-beta.tar.gz${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Windows build complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}📦 Output directory:${NC} q-narwhalknight-windows/"
echo -e "${BLUE}📦 Executable:${NC} q-narwhalknight-windows/q-miner.exe"

if [ -f "${ZIP_NAME}" ]; then
    echo -e "${BLUE}📦 Distribution:${NC} ${ZIP_NAME}"
fi

echo ""
echo -e "${YELLOW}🎯 Next Steps:${NC}"
echo "1. Test the executable on a Windows machine"
echo "2. Upload ${ZIP_NAME} to GitHub releases"
echo "3. Update documentation with Windows installation guide"
echo ""
echo -e "${GREEN}Happy mining! ⛏️${NC}"
