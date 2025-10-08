#!/bin/bash
set -euo pipefail

echo "🐳 Testing Docker Auto-Discovery Environment"
echo "============================================"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found - please install Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found - please install docker-compose"
    exit 1
fi

echo "✅ Docker and docker-compose are available"

# Wait for binary to be ready
echo "⏳ Waiting for q-api-server binary to be built..."
while [ ! -f "target/release/q-api-server" ]; do
    echo "   Still building..."
    sleep 5
done

echo "✅ Binary is ready"

# Build Docker image
echo "🔨 Building qnarwhal Docker image..."
docker build -t qnarwhal:latest -f docker/Dockerfile.qnarwhal . || {
    echo "❌ Docker build failed"
    exit 1
}

echo "✅ Docker image built successfully"

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose-auto-discovery.yml down --remove-orphans 2>/dev/null || true

# Create Docker network
echo "🌐 Creating Docker network..."
docker network create qnarwhal-mesh --subnet=172.20.0.0/16 2>/dev/null || echo "Network already exists"

# Start the auto-discovery test environment
echo "🚀 Starting auto-discovery test environment..."
docker-compose -f docker-compose-auto-discovery.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 15

# Check container status
echo "📊 Container Status:"
docker-compose -f docker-compose-auto-discovery.yml ps

echo ""
echo "🔍 Testing Auto-Discovery Functionality..."

# Test 1: Check DNS-Phantom Hub
echo "Testing DNS-Phantom Hub health..."
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ DNS-Phantom Hub is responding"
else
    echo "⚠️ DNS-Phantom Hub not responding yet"
fi

# Test 2: Check Beta Coordinator
echo "Testing Beta Coordinator health..."
if curl -sf http://localhost:8180/health > /dev/null 2>&1; then
    echo "✅ Beta Coordinator is responding"
else
    echo "⚠️ Beta Coordinator not responding yet"
fi

# Test 3: Check Alpha Nodes
for i in {1..3}; do
    port=$((8279 + i))
    echo "Testing Alpha Node $i health..."
    if curl -sf http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ Alpha Node $i is responding"
    else
        echo "⚠️ Alpha Node $i not responding yet"
    fi
done

echo ""
echo "📋 Live Container Logs (last 10 lines each):"

# Show recent logs from key containers
echo ""
echo "🔍 DNS-Phantom Hub logs:"
docker-compose -f docker-compose-auto-discovery.yml logs --tail=10 dns-phantom-hub

echo ""
echo "🤝 Beta Coordinator logs:"
docker-compose -f docker-compose-auto-discovery.yml logs --tail=10 beta-coordinator

echo ""
echo "🚀 Alpha Node 1 logs:"
docker-compose -f docker-compose-auto-discovery.yml logs --tail=10 alpha-node-1

echo ""
echo "📊 Summary:"
echo "  • Total containers: $(docker-compose -f docker-compose-auto-discovery.yml ps -q | wc -l)"
echo "  • Running containers: $(docker-compose -f docker-compose-auto-discovery.yml ps -q --status=running | wc -l)"
echo ""
echo "🎯 Test Commands Available:"
echo "  docker-compose -f docker-compose-auto-discovery.yml logs -f alpha-node-1"
echo "  docker-compose -f docker-compose-auto-discovery.yml logs -f beta-coordinator"
echo "  curl http://localhost:8180/api/mesh/stats"
echo ""
echo "🛑 To stop the environment:"
echo "  docker-compose -f docker-compose-auto-discovery.yml down"