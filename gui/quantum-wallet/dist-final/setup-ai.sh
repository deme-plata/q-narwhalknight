#!/bin/bash
# Quillon Graph — AI Wallet & Mining Setup
# Run: curl -fsSL https://quillon.xyz/setup-ai.sh | bash
#
# This script sets up AI-powered wallet and mining management.
# Currently supports Claude Code. More AI assistants coming soon.
# After setup, just say:
#   "Create a wallet"
#   "Start mining"
#   "What's my balance?"
set -e

echo ""
echo "  Quillon Graph — AI Wallet & Mining Setup"
echo "  ========================================="
echo ""

# 1. Check Node.js
if ! command -v node &>/dev/null; then
  echo "  Node.js not found. Installing..."
  if command -v apt-get &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - 2>/dev/null
    sudo apt-get install -y nodejs 2>/dev/null
  elif command -v brew &>/dev/null; then
    brew install node 2>/dev/null
  else
    echo "  Please install Node.js first: https://nodejs.org"
    exit 1
  fi
fi
echo "  ✓ Node.js $(node --version)"

# 2. Check Claude Code
if ! command -v claude &>/dev/null; then
  echo ""
  echo "  Claude Code not found. Install it:"
  echo "    npm install -g @anthropic-ai/claude-code"
  echo ""
  echo "  Then re-run: curl -fsSL https://quillon.xyz/setup-ai.sh | bash"
  exit 1
fi
echo "  ✓ Claude Code found"

# 3. Install Quillon MCP server
INSTALL_DIR="$HOME/.quillon/mcp"
mkdir -p "$INSTALL_DIR"

echo "  Downloading Quillon AI tools..."

# Download the MCP server files
curl -fsSL "https://quillon.xyz/downloads/quillon-wallet-mcp.tar.gz" -o "/tmp/quillon-mcp.tar.gz" 2>/dev/null || {
  # Fallback: create minimal MCP server inline
  mkdir -p "$INSTALL_DIR/build" "$INSTALL_DIR/src"
  cat > "$INSTALL_DIR/package.json" << 'PKGEOF'
{"name":"quillon-wallet-mcp","version":"1.0.0","type":"module","main":"build/index.js","dependencies":{"@modelcontextprotocol/sdk":"^1.12.1"}}
PKGEOF
  echo "  Installing dependencies..."
  cd "$INSTALL_DIR" && npm install --production 2>/dev/null
  # Download pre-built index.js
  curl -fsSL "https://quillon.xyz/downloads/quillon-mcp-index.js" -o "$INSTALL_DIR/build/index.js" 2>/dev/null || {
    echo "  Could not download MCP server. Check https://quillon.xyz/downloads/"
    exit 1
  }
}

# If tar was downloaded, extract it
if [ -f "/tmp/quillon-mcp.tar.gz" ]; then
  tar xzf /tmp/quillon-mcp.tar.gz -C "$INSTALL_DIR" 2>/dev/null
  cd "$INSTALL_DIR" && npm install --production 2>/dev/null
  rm -f /tmp/quillon-mcp.tar.gz
fi

echo "  ✓ Quillon AI tools installed"

# 4. Configure Claude Code settings
SETTINGS_DIR="$HOME/.claude"
mkdir -p "$SETTINGS_DIR"
SETTINGS_FILE="$SETTINGS_DIR/settings.json"

if command -v node &>/dev/null; then
  node -e "
    const fs = require('fs');
    const path = '$SETTINGS_FILE';
    let settings = {};
    try { settings = JSON.parse(fs.readFileSync(path, 'utf8')); } catch(e) {}
    if (!settings.mcpServers) settings.mcpServers = {};
    settings.mcpServers['quillon-wallet'] = {
      command: 'node',
      args: ['$INSTALL_DIR/build/index.js'],
      env: { QUILLON_API_URL: 'https://quillon.xyz/api/v1' }
    };
    fs.writeFileSync(path, JSON.stringify(settings, null, 2));
  " 2>/dev/null
fi

echo "  ✓ Claude Code configured"

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║        Setup Complete!                        ║"
echo "  ╠═══════════════════════════════════════════════╣"
echo "  ║                                               ║"
echo "  ║  Open Claude Code and say:                    ║"
echo "  ║                                               ║"
echo "  ║    \"Create a wallet\"                          ║"
echo "  ║    \"Start mining on this machine\"             ║"
echo "  ║    \"Set up a node on this server\"             ║"
echo "  ║    \"What's the network status?\"               ║"
echo "  ║                                               ║"
echo "  ║  No GPG. No air-gapping. Just works.          ║"
echo "  ║                                               ║"
echo "  ║  quillon.xyz | Post-Quantum Electronic Cash   ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""
