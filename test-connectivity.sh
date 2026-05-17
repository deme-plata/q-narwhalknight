#!/bin/bash

# Q-NarwhalKnight Docker Connectivity Test Script
# Tests real node connectivity between Docker containers

echo "🚀 Q-NarwhalKnight Docker Connectivity Test"
echo "============================================"

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker compose down --volumes --remove-orphans 2>/dev/null || true
docker system prune -f

# Build the Docker images
echo "🔨 Building Docker images..."
docker compose build

# Start the containers
echo "🌐 Starting 2-node test network..."
docker compose up -d

# Wait for containers to start
echo "⏳ Waiting for containers to initialize..."
sleep 10

# Check container status
echo "📊 Container status:"
docker compose ps

# Test node-alpha health
echo ""
echo "🔍 Testing Node Alpha health..."
if curl -f http://localhost:8080/health --max-time 5; then
    echo "✅ Node Alpha is healthy"
else
    echo "❌ Node Alpha health check failed"
fi

echo ""
echo "🔍 Testing Node Beta health..."
if curl -f http://localhost:8081/health --max-time 5; then
    echo "✅ Node Beta is healthy"
else
    echo "❌ Node Beta health check failed"
fi

# Check container logs
echo ""
echo "📝 Node Alpha logs (last 10 lines):"
docker compose logs --tail=10 node-alpha

echo ""
echo "📝 Node Beta logs (last 10 lines):"
docker compose logs --tail=10 node-beta

# Test network connectivity between containers
echo ""
echo "🌐 Testing inter-container network connectivity..."
docker exec qnk-node-alpha ping -c 3 node-beta || echo "❌ Alpha->Beta ping failed"
docker exec qnk-node-beta ping -c 3 node-alpha || echo "❌ Beta->Alpha ping failed"

# Test port connectivity
echo ""
echo "🔌 Testing port connectivity..."
docker exec qnk-node-alpha nc -zv node-beta 8334 || echo "❌ Alpha cannot reach Beta:8334"
docker exec qnk-node-beta nc -zv node-alpha 8334 || echo "❌ Beta cannot reach Alpha:8334"

echo ""
echo "🏁 Connectivity test complete!"
echo "Check logs above for any connection issues."
echo ""
echo "To monitor live logs: docker compose logs -f"
echo "To stop network: docker compose down"