#!/bin/bash

set -e

echo "🏗️  Building Q-NarwhalKnight API Server for Windows..."

# Create build output directory
mkdir -p build-output

# Build Docker image
echo "📦 Building Docker image..."
docker build -t qnk-windows-api-builder -f Dockerfile.windows-api .

echo "✅ Docker image built successfully"
echo ""
echo "🔨 Starting Windows API Server build..."
echo "⏱️  This may take 1-3 hours due to post-quantum crypto dependencies..."
echo ""

# Run the build in Docker
docker run --rm \
    -v "$(pwd)":/workspace \
    -v "$(pwd)/build-output":/build-output \
    qnk-windows-api-builder \
    sh -c "timeout 36000 cargo build --release --target x86_64-pc-windows-gnu --package q-api-server && \
           cp target/x86_64-pc-windows-gnu/release/q-api-server.exe /build-output/QNarwhalKnight-API-Windows-v0.0.3-beta.exe && \
           echo '✅ Build complete!' && \
           ls -lh /build-output/QNarwhalKnight-API-Windows-v0.0.3-beta.exe"

echo ""
echo "🎉 Windows API Server build complete!"
echo "📦 Output: build-output/QNarwhalKnight-API-Windows-v0.0.3-beta.exe"
