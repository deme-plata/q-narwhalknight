#!/usr/bin/env bash

# Q-NarwhalKnight HiveOS Miner Installation Script
# Version: 10.4.11

set -e

echo "=================================="
echo "Q-NarwhalKnight HiveOS Miner Setup"
echo "=================================="
echo ""

# Check if running on HiveOS
if [[ ! -d /hive ]]; then
    echo "Error: This script must be run on HiveOS"
    exit 1
fi

echo "HiveOS detected"
echo ""

# Download and extract miner package
echo "Downloading Q-NarwhalKnight miner package..."
cd /tmp
wget -q https://quillon.xyz/downloads/q-miner-hiveos.tar.gz -O q-miner-hiveos.tar.gz
wget -q https://quillon.xyz/downloads/q-miner-linux-x64 -O q-miner-linux-x64

echo "Downloaded successfully"
echo ""

# Create miner directory
echo "Creating miner directory..."
sudo mkdir -p /hive/miners/custom/q-miner/10.4.11
cd /hive/miners/custom/q-miner

# Extract package
echo "Extracting package..."
sudo tar -xzf /tmp/q-miner-hiveos.tar.gz --strip-components=1

# Copy miner binary
echo "Installing miner binary..."
sudo cp /tmp/q-miner-linux-x64 10.4.11/q-miner
sudo chmod +x 10.4.11/q-miner
sudo chmod +x 10.4.11/*.sh

echo "Installation complete"
echo ""
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
echo "   Pool: https://quillon.xyz"
echo "   Miner: Custom"
echo "   Installation URL: /hive/miners/custom/q-miner/10.4.11"
echo "   Miner config template: %WAL%"
echo ""
echo "4. Optional extra config (JSON):"
echo '   {"threads": 8, "log_level": "info"}'
echo ""
echo "5. Apply flight sheet to your rig"
echo ""

# Test miner binary
if /hive/miners/custom/q-miner/10.4.11/q-miner --version 2>/dev/null; then
    echo "Miner binary verified"
fi

echo ""
echo "Done! (v10.4.11)"
echo "Help: https://discord.gg/quillon"
