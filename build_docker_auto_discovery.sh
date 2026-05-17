#!/bin/bash
set -euo pipefail

echo "🐳 Building Q-NarwhalKnight Auto-Discovery Docker Environment"
echo "============================================================"

# Build the latest binaries first
echo "🔧 Building latest Q-NarwhalKnight binaries..."
cargo build --release --bin q-api-server

# Check if binary was built successfully
if [ ! -f "target/release/q-api-server" ]; then
    echo "❌ Binary build failed - q-api-server not found"
    exit 1
fi

echo "✅ Binary built successfully"

# Build Docker image
echo "🐳 Building Docker image with auto-discovery capabilities..."
docker build -t qnarwhal:latest -f docker/Dockerfile.qnarwhal .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker image build failed"
    exit 1
fi

# Create Docker network if it doesn't exist
echo "🌐 Creating Docker network..."
docker network create qnarwhal-mesh --subnet=172.20.0.0/16 2>/dev/null || echo "Network already exists"

echo "🚀 Starting auto-discovery test environment..."
echo ""
echo "This will start:"
echo "  • DNS-Phantom Hub (steganographic discovery)"
echo "  • Tor Proxy (anonymous networking)"
echo "  • 1 Beta Coordinator"
echo "  • 3 Alpha Nodes (auto-discovery enabled)"
echo "  • 2 Validator Nodes"
echo "  • 1 Network Monitor"
echo ""

# Start the containers
docker-compose -f docker-compose-auto-discovery.yml up -d

echo "✅ Auto-discovery environment started!"
echo ""
echo "🔍 Container Status:"
docker-compose -f docker-compose-auto-discovery.yml ps

echo ""
echo "📊 Access Points:"
echo "  • DNS-Phantom Hub: http://localhost:8080"
echo "  • Beta Coordinator: http://localhost:8180" 
echo "  • Alpha Node 1: http://localhost:8280"
echo "  • Alpha Node 2: http://localhost:8281"
echo "  • Alpha Node 3: http://localhost:8282"
echo "  • Network Monitor: http://localhost:8380"
echo "  • Tor Proxy: SOCKS5 on localhost:9050"
echo ""
echo "🎯 Test Commands:"
echo "  docker-compose -f docker-compose-auto-discovery.yml logs -f alpha-node-1"
echo "  docker-compose -f docker-compose-auto-discovery.yml logs -f beta-coordinator"
echo "  curl http://localhost:8180/api/mesh/stats"
echo ""
echo "🌟 Auto-discovery test environment is ready!"