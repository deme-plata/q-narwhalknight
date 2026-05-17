#!/bin/bash
# Q-NarwhalKnight API Server Restart Script
# Compiled with mistral.rs high-performance optimizations
# Date: October 29, 2025

set -e

echo "=========================================="
echo "Q-NarwhalKnight API Server Restart"
echo "mistral.rs High-Performance Edition"
echo "=========================================="
echo ""

# Check if binary exists
BINARY_PATH="/opt/orobit/shared/q-narwhalknight/target/release/q-api-server"

if [ ! -f "$BINARY_PATH" ]; then
    echo "❌ ERROR: Binary not found at $BINARY_PATH"
    echo "   Please wait for compilation to complete"
    exit 1
fi

echo "✅ Binary found: $BINARY_PATH"
ls -lh "$BINARY_PATH"
echo ""

# Check service status before restart
echo "📊 Current service status:"
systemctl status q-api-server --no-pager | head -10
echo ""

# Stop the service
echo "🛑 Stopping q-api-server service..."
sudo systemctl stop q-api-server
sleep 2
echo "✅ Service stopped"
echo ""

# Start the service with new binary
echo "🚀 Starting q-api-server with mistral.rs optimizations..."
sudo systemctl start q-api-server
sleep 3
echo ""

# Check service status
echo "📊 New service status:"
systemctl status q-api-server --no-pager | head -15
echo ""

# Check if service is running
if systemctl is-active --quiet q-api-server; then
    echo "✅ SUCCESS: q-api-server is running with mistral.rs optimizations!"
    echo ""
    echo "📊 Recent logs:"
    sudo journalctl -u q-api-server -n 50 --no-pager | tail -20
    echo ""
    echo "=========================================="
    echo "Expected Log Messages:"
    echo "  🚀 Initializing mistral.rs high-performance engine..."
    echo "  📦 Loading GGUF model with mistral.rs optimizations..."
    echo "  ✅ mistral.rs engine initialized successfully!"
    echo "=========================================="
else
    echo "❌ ERROR: Service failed to start"
    echo ""
    echo "📋 Last 50 log lines:"
    sudo journalctl -u q-api-server -n 50 --no-pager
    exit 1
fi

echo ""
echo "🧪 Testing API endpoint..."
timeout 5 curl -s http://localhost:8080/api/health > /dev/null && \
    echo "✅ API endpoint responding" || \
    echo "⚠️  API endpoint not yet responding (give it a few seconds)"

echo ""
echo "=========================================="
echo "✅ Restart Complete!"
echo "=========================================="
echo ""
echo "To monitor logs in real-time:"
echo "  sudo journalctl -u q-api-server -f"
echo ""
echo "To test AI inference:"
echo "  curl -N 'http://localhost:8080/api/chat/test/stream?content=Hello'"
echo ""
