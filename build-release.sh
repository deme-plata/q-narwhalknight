#!/bin/bash
# Q-NarwhalKnight Release Build Script
# Builds essential binaries for distribution

set -e

echo "🚀 Building Q-NarwhalKnight Release Binaries"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cd "$(dirname "$0")"

# Build core API server
echo -e "${BLUE}📦 Building q-api-server...${NC}"
if cargo build --release --package q-api-server; then
    echo -e "${GREEN}✅ q-api-server built successfully${NC}"
else
    echo -e "${RED}❌ Failed to build q-api-server${NC}"
    exit 1
fi

# Build DAG-Knight VM
echo -e "${BLUE}🌊 Building dagknight VM...${NC}"
if cargo build --release --package q-vm; then
    echo -e "${GREEN}✅ dagknight VM built successfully${NC}"
else
    echo -e "${RED}❌ Failed to build dagknight VM${NC}"
    exit 1
fi

# Build other essential components
echo -e "${BLUE}🔗 Building core components...${NC}"

CORE_PACKAGES=(
    "q-types"
    "q-dag-knight"
    "q-narwhal-core" 
    "q-quantum-rng"
    "q-network"
    "q-storage"
)

for package in "${CORE_PACKAGES[@]}"; do
    echo -e "${YELLOW}  - Building $package...${NC}"
    if cargo build --release --package "$package" --lib; then
        echo -e "    ${GREEN}✅ $package built${NC}"
    else
        echo -e "    ${YELLOW}⚠️  $package build failed (non-critical)${NC}"
    fi
done

echo ""
echo -e "${GREEN}🎉 Release Build Summary${NC}"
echo "========================="

# Check what binaries we have
RELEASE_DIR="target/release"
if [ -f "$RELEASE_DIR/q-api-server" ]; then
    echo -e "${GREEN}✅ q-api-server binary ready${NC}"
    ls -lh "$RELEASE_DIR/q-api-server"
fi

if [ -f "$RELEASE_DIR/dagknight" ]; then
    echo -e "${GREEN}✅ dagknight binary ready${NC}"
    ls -lh "$RELEASE_DIR/dagknight"
fi

# Create distribution directory
DIST_DIR="dist"
mkdir -p "$DIST_DIR"

# Copy binaries to distribution
echo -e "${BLUE}📦 Creating distribution package...${NC}"
if [ -f "$RELEASE_DIR/q-api-server" ]; then
    cp "$RELEASE_DIR/q-api-server" "$DIST_DIR/"
fi
if [ -f "$RELEASE_DIR/dagknight" ]; then
    cp "$RELEASE_DIR/dagknight" "$DIST_DIR/"
fi

# Copy install script
cp install.sh "$DIST_DIR/"
chmod +x "$DIST_DIR/install.sh"

echo -e "${GREEN}✅ Distribution ready in $DIST_DIR/${NC}"
echo ""
echo -e "${CYAN}🌐 To serve via HTTP:${NC}"
echo -e "  nginx: Point document root to $(pwd)/$DIST_DIR"
echo -e "  python: cd $DIST_DIR && python3 -m http.server 8080"
echo ""
echo -e "${CYAN}📥 Installation command for users:${NC}"
echo -e "  curl -fsSL https://yourdomain.com/install.sh | bash"