#!/usr/bin/env bash

# Q-NarwhalKnight HiveOS Miner Installation Script
# Version: 1.0

set -e

echo "=================================="
echo "Q-NarwhalKnight HiveOS Miner Setup"
echo "=================================="
echo ""

# Check if running on HiveOS
if [[ ! -d /hive ]]; then
    echo "❌ Error: This script must be run on HiveOS"
    exit 1
fi

echo "✓ HiveOS detected"
echo ""

# Download and extract miner package
echo "📦 Downloading Q-NarwhalKnight miner package..."
cd /tmp
wget -q https://quillon.xyz/downloads/q-miner-hiveos-v1.0.tar.gz
wget -q https://quillon.xyz/downloads/q-miner-linux-x64

echo "✓ Downloaded successfully"
echo ""

# Create miner directory
echo "📁 Creating miner directory..."
sudo mkdir -p /hive/miners/custom/q-miner
cd /hive/miners/custom/q-miner

# Extract package
echo "📦 Extracting package..."
sudo tar -xzf /tmp/q-miner-hiveos-v1.0.tar.gz --strip-components=1

# Copy miner binary
echo "📄 Installing miner binary..."
sudo cp /tmp/q-miner-linux-x64 1.0/q-miner
sudo chmod +x 1.0/q-miner
sudo chmod +x 1.0/*.sh

echo "✓ Installation complete"
echo ""

# Display configuration instructions
echo "=================================="
echo "Configuration Instructions"
echo "=================================="
echo ""
echo "1. Go to HiveOS Dashboard > Flight Sheets"
echo "2. Click 'Create Flight Sheet'"
echo "3. Configure as follows:"
echo ""
echo "   Coin: Custom"
echo "   Wallet: Your QNK wallet address (qnk...)"
echo "   Pool: https://quillon.xyz:8080"
echo "   Miner: Custom"
echo "   Installation URL: /hive/miners/custom/q-miner/1.0"
echo "   Miner config template: %WAL%"
echo ""
echo "4. Optional: Set extra config (JSON):"
echo '   {"threads": 8, "log_level": "info"}'
echo ""
echo "5. Apply flight sheet to your rig"
echo ""
echo "=================================="
echo "Testing Miner"
echo "=================================="
echo ""

# Test miner binary
if /hive/miners/custom/q-miner/1.0/q-miner --version 2>/dev/null; then
    echo "✓ Miner binary is working"
else
    echo "⚠️  Could not verify miner binary (may need wallet to test)"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "For detailed setup guide, visit:"
echo "https://quillon.xyz/hiveos-setup"
echo ""
echo "Need help? Join our Discord:"
echo "https://discord.gg/quillon"
echo ""
