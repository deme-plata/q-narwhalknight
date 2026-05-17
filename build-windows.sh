#!/bin/bash
# Windows Build Script for Q-NarwhalKnight
# Builds Windows .exe using MinGW cross-compilation

set -e  # Exit on error

echo "🪟 Q-NarwhalKnight Windows Build Script"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v x86_64-w64-mingw32-gcc &> /dev/null; then
    echo -e "${RED}❌ MinGW gcc not found${NC}"
    echo "Install with: apt-get install gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64"
    exit 1
fi

if ! rustup target list | grep -q "x86_64-pc-windows-gnu (installed)"; then
    echo -e "${YELLOW}⚠️  Windows target not installed, installing...${NC}"
    rustup target add x86_64-pc-windows-gnu
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo ""

# Clean previous build (optional)
if [ "$1" == "--clean" ]; then
    echo "🧹 Cleaning previous build..."
    cargo clean --target x86_64-pc-windows-gnu
    echo ""
fi

# Build
echo "🔨 Building Windows executable..."
echo "   Target: x86_64-pc-windows-gnu"
echo "   Mode: Release"
echo "   Package: q-api-server"
echo ""

# Use 10-hour timeout as specified in CLAUDE.md
if timeout 36000 cargo build --release --target x86_64-pc-windows-gnu --package q-api-server; then
    echo ""
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo ""

    # Show build info
    EXE_PATH="target/x86_64-pc-windows-gnu/release/q-api-server.exe"
    if [ -f "$EXE_PATH" ]; then
        SIZE=$(du -h "$EXE_PATH" | cut -f1)
        echo "📦 Output:"
        echo "   Location: $EXE_PATH"
        echo "   Size: $SIZE"
        echo ""

        # Create a package directory
        PACKAGE_DIR="q-narwhalknight-windows"
        echo "📁 Creating package directory: $PACKAGE_DIR"
        mkdir -p "$PACKAGE_DIR"

        # Copy executable
        cp "$EXE_PATH" "$PACKAGE_DIR/"

        # Copy Windows DLLs if they exist
        if [ -d "q-narwhalknight-windows" ]; then
            echo "   Copying Windows DLLs..."
            cp q-narwhalknight-windows/*.dll "$PACKAGE_DIR/" 2>/dev/null || true
        fi

        # Create README
        cat > "$PACKAGE_DIR/README.txt" << 'EOF'
Q-NarwhalKnight Windows Release

To run the server:
1. Extract all files to a folder
2. Open Command Prompt in that folder
3. Run: q-api-server.exe --port 8080

Environment Variables (optional):
- Q_DB_PATH=./data        - Database directory
- Q_P2P_PORT=9001         - P2P networking port
- RUST_LOG=info           - Logging level

For more information, visit:
https://github.com/deme-plata/q-narwhalknight
EOF

        # Create archive
        ARCHIVE="q-narwhalknight-windows-$(date +%Y%m%d).zip"
        echo "   Creating archive: $ARCHIVE"
        zip -r "$ARCHIVE" "$PACKAGE_DIR/" > /dev/null 2>&1 || echo "   (zip not available, skipping archive)"

        echo ""
        echo -e "${GREEN}🎉 Package complete!${NC}"
        echo "   Directory: $PACKAGE_DIR/"
        if [ -f "$ARCHIVE" ]; then
            echo "   Archive: $ARCHIVE"
        fi
        echo ""
        echo "📝 Enhanced Logging Features:"
        echo "   ✅ Shadow mode real-time metrics"
        echo "   ✅ API endpoint request logging"
        echo "   ✅ Performance comparison tracking"
        echo "   ✅ Migration readiness indicators"
    else
        echo -e "${RED}❌ Build output not found at $EXE_PATH${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${RED}❌ Build failed!${NC}"
    echo "Check the error messages above for details."
    exit 1
fi
