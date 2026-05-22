#!/bin/bash
# Quillon Graph — AI Wallet & Mining Setup
# Run: curl -fsSL https://quillon.xyz/setup-ai.sh | bash
#
# This script sets up AI-powered wallet and mining management.
# Currently supports Claude Code and Cursor. Auto-detects what you have.
# After setup, just say:
#   "Create a wallet"
#   "Start mining"
#   "What's my balance?"
set -e

echo ""
echo "  Quillon Graph — AI Wallet & Mining Setup"
echo "  ========================================="
echo ""

# v2.9.0: detect Windows-via-Git-Bash / MSYS / Cygwin and redirect to the
# PowerShell version. The bash path below assumes apt-get or brew is
# available, which neither is on Windows. Adrian (Cursor on Windows) hit
# this on 2026-05-22 — script gave up with "Please install Node.js first:
# https://nodejs.org" instead of pointing to the PowerShell installer.
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "  ⚠ Detected Git Bash / MSYS on Windows."
    echo "  This bash script can't install Node.js on Windows."
    echo "  Please run the PowerShell version instead:"
    echo ""
    echo "    irm https://quillon.xyz/setup-ai.ps1 | iex"
    echo ""
    echo "  Or download manually:"
    echo "    curl -fsSL https://quillon.xyz/setup-ai.ps1 -o setup-ai.ps1"
    echo "    powershell -ExecutionPolicy Bypass -File setup-ai.ps1"
    exit 1
    ;;
esac

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

# 2. Detect AI clients (Claude Code, Cursor, Codex, or any combination)
HAS_CLAUDE=0
HAS_CURSOR=0
HAS_CODEX=0

if command -v claude &>/dev/null; then
  HAS_CLAUDE=1
  echo "  ✓ Claude Code found"
fi

# Cursor detection: look for the config dir (works on Linux/macOS/Git-Bash on Windows)
if [ -d "$HOME/.cursor" ] || command -v cursor &>/dev/null; then
  HAS_CURSOR=1
  echo "  ✓ Cursor found"
fi

# Codex (OpenAI / ChatGPT 5.5) detection
if command -v codex &>/dev/null || [ -d "$HOME/.codex" ]; then
  HAS_CODEX=1
  echo "  ✓ Codex (ChatGPT 5.5) found"
fi

if [ "$HAS_CLAUDE" = "0" ] && [ "$HAS_CURSOR" = "0" ] && [ "$HAS_CODEX" = "0" ]; then
  echo ""
  echo "  No supported AI client found."
  echo ""
  echo "  Install ONE of:"
  echo "    Claude Code:  npm install -g @anthropic-ai/claude-code"
  echo "    Cursor:       https://cursor.sh"
  echo "    Codex:        npm install -g @openai/codex"
  echo ""
  echo "  Then re-run: curl -fsSL https://quillon.xyz/setup-ai.sh | bash"
  exit 1
fi

# 3. Install Quillon MCP server
INSTALL_DIR="$HOME/.quillon/mcp"
mkdir -p "$INSTALL_DIR"

echo "  Downloading Quillon AI tools..."

# Download the MCP server tarball
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

echo "  ✓ Quillon AI tools installed at $INSTALL_DIR"

# 4. Configure each detected client
MCP_INDEX="$INSTALL_DIR/build/index.js"

# 4a. Claude Code → $HOME/.claude/settings.json (mcpServers key)
if [ "$HAS_CLAUDE" = "1" ]; then
  SETTINGS_DIR="$HOME/.claude"
  mkdir -p "$SETTINGS_DIR"
  SETTINGS_FILE="$SETTINGS_DIR/settings.json"
  node -e "
    const fs = require('fs');
    const path = '$SETTINGS_FILE';
    let settings = {};
    try { settings = JSON.parse(fs.readFileSync(path, 'utf8')); } catch(e) {}
    if (!settings.mcpServers) settings.mcpServers = {};
    settings.mcpServers['quillon-wallet'] = {
      command: 'node',
      args: ['$MCP_INDEX'],
      env: { QUILLON_API_URL: 'https://quillon.xyz/api/v1' }
    };
    fs.writeFileSync(path, JSON.stringify(settings, null, 2));
  " 2>/dev/null
  echo "  ✓ Claude Code configured at $SETTINGS_FILE"
fi

# 4b. Cursor → $HOME/.cursor/mcp.json (standalone file, mcpServers key)
if [ "$HAS_CURSOR" = "1" ]; then
  CURSOR_DIR="$HOME/.cursor"
  mkdir -p "$CURSOR_DIR"
  CURSOR_FILE="$CURSOR_DIR/mcp.json"
  node -e "
    const fs = require('fs');
    const path = '$CURSOR_FILE';
    let cfg = {};
    try { cfg = JSON.parse(fs.readFileSync(path, 'utf8')); } catch(e) {}
    if (!cfg.mcpServers) cfg.mcpServers = {};
    cfg.mcpServers['quillon-wallet'] = {
      command: 'node',
      args: ['$MCP_INDEX'],
      env: { QUILLON_API_URL: 'https://quillon.xyz/api/v1' }
    };
    fs.writeFileSync(path, JSON.stringify(cfg, null, 2));
  " 2>/dev/null
  echo "  ✓ Cursor configured at $CURSOR_FILE"
  echo "    → Restart Cursor or reload window for MCP to load"
fi

# 4c. Codex (ChatGPT 5.5) → $HOME/.codex/config.toml (TOML format)
if [ "$HAS_CODEX" = "1" ]; then
  CODEX_DIR="$HOME/.codex"
  mkdir -p "$CODEX_DIR"
  CODEX_FILE="$CODEX_DIR/config.toml"
  # Codex CLI uses TOML; mcp_servers section is keyed by server name.
  # If config.toml exists, append or replace the quillon-wallet block;
  # else create from scratch.
  if [ -f "$CODEX_FILE" ] && grep -q "\[mcp_servers.quillon-wallet\]" "$CODEX_FILE"; then
    echo "  ✓ Codex config already has quillon-wallet entry at $CODEX_FILE"
  else
    cat >> "$CODEX_FILE" << CODEXEOF

[mcp_servers.quillon-wallet]
command = "node"
args = ["$MCP_INDEX"]

[mcp_servers.quillon-wallet.env]
QUILLON_API_URL = "https://quillon.xyz/api/v1"
CODEXEOF
    echo "  ✓ Codex configured at $CODEX_FILE"
    echo "    → Restart Codex (codex --reload) for MCP to load"
  fi
fi

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║        Setup Complete!                        ║"
echo "  ╠═══════════════════════════════════════════════╣"
echo "  ║                                               ║"
if [ "$HAS_CLAUDE" = "1" ]; then
echo "  ║  Open Claude Code and say:                    ║"
echo "  ║                                               ║"
echo "  ║    \"Create a wallet\"                          ║"
echo "  ║    \"Start mining on this machine\"             ║"
echo "  ║    \"Set up a node on this server\"             ║"
echo "  ║    \"What's the network status?\"               ║"
echo "  ║                                               ║"
fi
if [ "$HAS_CURSOR" = "1" ]; then
echo "  ║  In Cursor (Agent mode, after reload):        ║"
echo "  ║                                               ║"
echo "  ║    \"What Quillon tools are available?\"        ║"
echo "  ║    \"Show my QUG balance\"                      ║"
echo "  ║    \"Check node sync status\"                   ║"
echo "  ║                                               ║"
fi
echo "  ║  No GPG. No air-gapping. Just works.          ║"
echo "  ║                                               ║"
echo "  ║  quillon.xyz | Post-Quantum Electronic Cash   ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""
