#!/bin/bash
# GPU-Accelerated Build Script for Q-NarwhalKnight
# Automatically detects CUDA and builds with GPU support when available

set -e  # Exit on error

echo "🚀 Q-NarwhalKnight GPU-Accelerated Build Script"
echo "================================================"
echo ""

# Detect CUDA
CUDA_AVAILABLE=false
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo "✅ CUDA detected: v$CUDA_VERSION"
    CUDA_AVAILABLE=true
else
    echo "⚠️  CUDA not detected (nvcc not in PATH)"
    echo "   GPU acceleration will be DISABLED"
fi

echo ""
echo "Build Configuration:"
echo "-------------------"

# Determine build features
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "✅ GPU Support: ENABLED (CUDA)"
    echo "   Expected Performance: 100-200 tok/s"
    BUILD_FEATURES="--features q-ai-inference/cuda"
else
    echo "❌ GPU Support: DISABLED (CPU-only)"
    echo "   Expected Performance: 2-5 tok/s"
    BUILD_FEATURES=""
fi

echo ""
echo "Starting build..."
echo ""

# Build with appropriate features
if [ "$1" = "--release" ]; then
    echo "📦 Building RELEASE binaries..."
    cargo build --release --workspace $BUILD_FEATURES

    echo ""
    echo "✅ Build complete!"
    echo ""
    echo "Binaries location:"
    echo "  API Server: target/release/q-api-server"
    echo "  Miner: target/release/q-miner"

    if [ "$CUDA_AVAILABLE" = true ]; then
        echo ""
        echo "🎯 GPU-accelerated build successful!"
        echo "   Run with: ./target/release/q-api-server"
        echo "   AI inference will use CUDA automatically"
    fi
else
    echo "📦 Building DEBUG binaries..."
    cargo build --workspace $BUILD_FEATURES

    echo ""
    echo "✅ Build complete!"
    echo ""
    echo "Binaries location:"
    echo "  API Server: target/debug/q-api-server"
    echo "  Miner: target/debug/q-miner"
fi

echo ""
echo "To force CPU-only build: cargo build --workspace"
echo "To force CUDA build:     cargo build --workspace --features q-ai-inference/cuda"
echo ""
