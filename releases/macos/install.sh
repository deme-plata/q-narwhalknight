#!/bin/bash
#
# Q-NarwhalKnight macOS Installer
# Quantum-Enhanced DAG-BFT Consensus Node
#
# Usage: ./install.sh

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        Q-NarwhalKnight macOS Installer v0.0.3-beta           ║"
echo "║     Quantum-Enhanced DAG-BFT Consensus Node                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    BINARY="q-api-server-macos-x86_64"
    echo "✓ Detected: Intel Mac (x86_64)"
elif [ "$ARCH" = "arm64" ]; then
    BINARY="q-api-server-macos-aarch64"
    echo "✓ Detected: Apple Silicon (arm64)"
else
    echo "❌ Error: Unsupported architecture: $ARCH"
    exit 1
fi

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "❌ Error: Binary file '$BINARY' not found in current directory"
    echo ""
    echo "Please download the appropriate binary for your Mac:"
    echo "  Intel: q-api-server-macos-x86_64"
    echo "  Apple Silicon: q-api-server-macos-aarch64"
    exit 1
fi

echo ""
echo "🔧 Installing Q-NarwhalKnight..."
echo ""

# Make binary executable
chmod +x "$BINARY"
echo "✓ Made binary executable"

# Install to /usr/local/bin
echo ""
echo "📦 Installing to /usr/local/bin (requires sudo)..."
sudo cp "$BINARY" /usr/local/bin/q-api-server
echo "✓ Binary installed to /usr/local/bin/q-api-server"

# Remove quarantine attribute
echo "🔓 Removing macOS quarantine attribute..."
sudo xattr -r -d com.apple.quarantine /usr/local/bin/q-api-server 2>/dev/null || true
echo "✓ Quarantine removed"

# Create config directory
CONFIG_DIR="$HOME/.q-narwhalknight"
mkdir -p "$CONFIG_DIR/data"
echo "✓ Created config directory: $CONFIG_DIR"

# Create default config if it doesn't exist
if [ ! -f "$CONFIG_DIR/config.toml" ]; then
    cat > "$CONFIG_DIR/config.toml" << 'EOF'
[node]
port = 8080
node_id = "my-mac-node"
data_path = "$HOME/.q-narwhalknight/data"

[network]
bootstrap_peers = []
max_peers = 50

[consensus]
phase = 0  # 0=Classical, 1=Post-Quantum

[api]
enable_cors = true
allowed_origins = ["http://localhost:5173", "https://wallet.quillon.xyz"]
EOF
    echo "✓ Created default configuration"
else
    echo "ℹ Using existing configuration"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Installation Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "🚀 Quick Start:"
echo ""
echo "   1. Start the node:"
echo "      q-api-server --port 8080"
echo ""
echo "   2. Check node status:"
echo "      curl http://localhost:8080/api/v1/health"
echo ""
echo "   3. Access the Quantum Wallet:"
echo "      https://wallet.quillon.xyz"
echo ""
echo "📁 Configuration:"
echo "   Config file: $CONFIG_DIR/config.toml"
echo "   Data directory: $CONFIG_DIR/data"
echo ""
echo "📖 Documentation:"
echo "   https://api.quillon.xyz"
echo ""
echo "💬 Support:"
echo "   Discord: https://discord.gg/jEhaYtAhfx"
echo "   GitHub: https://github.com/deme-plata/q-narwhalknight"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "⚡ Optional: Run as a service (launchd)"
echo ""
echo "Create ~/Library/LaunchAgents/com.quillon.q-narwhalknight.plist"
echo "See README.md for details."
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
