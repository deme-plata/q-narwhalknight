#!/bin/bash
# Q-NarwhalKnight Windows GUI Build Script
# Builds Slint GUI + embedded full node for Windows x86_64

set -e

echo "🪟 Q-NarwhalKnight Windows GUI Builder"
echo "========================================"
echo ""

# Configuration
IMAGE_NAME="qnk-windows-builder"
CONTAINER_NAME="qnk-windows-build"
OUTPUT_DIR="$(pwd)/build-output"
BUILD_MODE="${1:-gui-only}" # gui-only or full-node

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📦 Build mode: $BUILD_MODE"
echo "📂 Output directory: $OUTPUT_DIR"
echo ""

# Build Docker image
echo "🐳 Building Docker image for Windows cross-compilation..."
docker build -f Dockerfile.windows -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
    echo "❌ Docker image build failed!"
    exit 1
fi

echo "✅ Docker image built successfully"
echo ""

# Run build
echo "🔨 Starting Windows GUI build..."
echo "⏱️  This may take 1-3 hours depending on build mode..."
echo ""

if [ "$BUILD_MODE" == "full-node" ]; then
    echo "🌟 Building with embedded full node (windows-full feature)..."
    docker run --rm \
        -v "$(pwd):/workspace" \
        -v "$OUTPUT_DIR:/output" \
        --name $CONTAINER_NAME \
        $IMAGE_NAME \
        sh -c "timeout 36000 cargo build --release --target x86_64-pc-windows-gnu --package qnk-gui --features windows-full && cp target/x86_64-pc-windows-gnu/release/qnk-gui.exe /output/QNarwhalKnight-FullNode-Windows-v0.1.0.exe"
else
    echo "🎨 Building standalone GUI (connects to external node)..."
    docker run --rm \
        -v "$(pwd):/workspace" \
        -v "$OUTPUT_DIR:/output" \
        --name $CONTAINER_NAME \
        $IMAGE_NAME \
        sh -c "timeout 36000 cargo build --release --target x86_64-pc-windows-gnu --package qnk-gui && cp target/x86_64-pc-windows-gnu/release/qnk-gui.exe /output/QNarwhalKnight-GUI-Windows-v0.1.0.exe"
fi

BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Windows build completed successfully!"
    echo "📦 Output files:"
    ls -lh "$OUTPUT_DIR"/*.exe 2>/dev/null || echo "⚠️  No .exe files found in output"
    echo ""
    echo "🧪 Testing with Wine (if available)..."
    if command -v wine64 &> /dev/null; then
        wine64 "$OUTPUT_DIR"/*.exe --version 2>/dev/null || echo "⚠️  Wine test skipped"
    else
        echo "ℹ️  Wine not installed, skipping test"
    fi
else
    echo ""
    echo "❌ Windows build failed with exit code $BUILD_EXIT_CODE"
    exit $BUILD_EXIT_CODE
fi

echo ""
echo "🎉 Build process complete!"
echo "📋 Next steps:"
echo "   1. Test on Windows: Copy .exe to Windows machine"
echo "   2. Verify GUI loads correctly"
echo "   3. Test wallet creation and transactions"
echo "   4. Test quantum metrics visualization"
echo ""
