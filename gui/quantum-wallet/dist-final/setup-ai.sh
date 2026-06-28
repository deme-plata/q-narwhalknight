#!/bin/bash
# Quillon Graph — AI Wallet & Mining Setup
# Run: curl -fsSL https://quillon.xyz/setup-ai.sh | bash
#
# This script sets up AI-powered wallet and mining management.
# Supports Claude Code, Cursor, Codex, Qwen Coder, and Grok Build. Auto-detects what you have.
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

install_mcp_bundle() {
  target_dir="$1"
  label="$2"
  mkdir -p "$target_dir"

  if [ -f "/tmp/quillon-mcp.tar.gz" ]; then
    echo "  → Installing $label MCP bundle..."
    tar xzf /tmp/quillon-mcp.tar.gz -C "$target_dir" 2>/dev/null
  fi

  if [ ! -f "$target_dir/package.json" ]; then
    mkdir -p "$target_dir/build"
    # Complete deps — the MCP signs transactions with @noble (signTransferV72) and
    # validates tool args with zod. A package.json missing these produced a
    # load-broken install whenever the tarball path failed.
    cat > "$target_dir/package.json" << 'PKGEOF'
{"name":"quillon-wallet-mcp","version":"2.19.1","type":"module","main":"build/index.js","dependencies":{"@modelcontextprotocol/sdk":"^1.12.1","@noble/curves":"^1.6.0","@noble/hashes":"^1.5.0","zod":"^3.23.8"}}
PKGEOF
  fi

  if [ -f "$target_dir/build/index.js" ]; then
    echo "  → $label MCP: prebuilt build/index.js found"
    cd "$target_dir" && npm install --omit=dev --no-audit --no-fund 2>/dev/null
  else
    echo "  → $label MCP: no prebuilt output; installing build dependencies"
    cd "$target_dir" && npm install --no-audit --no-fund 2>/dev/null
    if npm run 2>/dev/null | grep -q " build"; then
      npm run build 2>/dev/null
    fi
  fi

  if [ ! -f "$target_dir/build/index.js" ]; then
    echo "  → $label MCP: downloading prebuilt fallback (index.js + wallet_auth.js)"
    mkdir -p "$target_dir/build"
    # The build is multi-file: index.js imports ./wallet_auth.js (the signer).
    # Fetch BOTH or the MCP fails to load (the old fallback grabbed only index.js).
    curl -fsSL "https://quillon.xyz/downloads/quillon-mcp-index.js" -o "$target_dir/build/index.js" 2>/dev/null || true
    curl -fsSL "https://quillon.xyz/downloads/quillon-mcp-wallet-auth.js" -o "$target_dir/build/wallet_auth.js" 2>/dev/null || true
    cd "$target_dir" && npm install --omit=dev --no-audit --no-fund 2>/dev/null
  fi

  if [ ! -f "$target_dir/build/index.js" ]; then
    echo "  ✗ $label MCP install incomplete: build/index.js missing"
    exit 1
  fi

  echo "  ✓ $label MCP ready: $target_dir/build/index.js"
}

# 2. Detect AI clients (Claude Code, Cursor, Codex, or any combination)
HAS_CLAUDE=0
HAS_CURSOR=0
HAS_CODEX=0
HAS_QWEN=0
HAS_GROK=0

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

if command -v qwen &>/dev/null || command -v qwen-coder &>/dev/null || [ -d "$HOME/.qwen" ]; then
  HAS_QWEN=1
  echo "  ✓ Qwen Coder found"
fi

if command -v grok &>/dev/null || command -v grok-build &>/dev/null || [ -d "$HOME/.grok" ]; then
  HAS_GROK=1
  echo "  ✓ Grok Build CLI found"
fi

if [ "$HAS_CLAUDE" = "0" ] && [ "$HAS_CURSOR" = "0" ] && [ "$HAS_CODEX" = "0" ] && [ "$HAS_QWEN" = "0" ] && [ "$HAS_GROK" = "0" ]; then
  echo ""
  echo "  No supported AI client found."
  echo ""
  echo "  Install ONE of:"
  echo "    Claude Code:  npm install -g @anthropic-ai/claude-code"
  echo "    Cursor:       https://cursor.sh"
  echo "    Codex:        npm install -g @openai/codex"
  echo "    Qwen Coder:   create ~/.qwen or install the Qwen Coder CLI"
  echo "    Grok Build:   create ~/.grok or install the Grok Build CLI"
  echo ""
  echo "  Then re-run: curl -fsSL https://quillon.xyz/setup-ai.sh | bash"
  exit 1
fi

# 3. Install Quillon MCP server
INSTALL_DIR="$HOME/.quillon/mcp"
mkdir -p "$INSTALL_DIR"

echo "  Downloading Quillon AI tools..."

# Download the MCP server tarball (preferred path). If it fails, do NOT hand-build
# a broken minimal install here — install_mcp_bundle below reconstructs correctly
# from the published prebuilt files (index.js + wallet_auth.js) and a complete
# package.json (incl. @noble + zod). Pre-creating files here would block that.
curl -fsSL "https://quillon.xyz/downloads/quillon-wallet-mcp.tar.gz" -o "/tmp/quillon-mcp.tar.gz" 2>/dev/null || {
  echo "  ⚠ tarball download failed — reconstructing from prebuilt files..."
  rm -f /tmp/quillon-mcp.tar.gz
}

install_mcp_bundle "$INSTALL_DIR" "base"

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

# 4d. Qwen Coder → dedicated ~/.qwen/quillon path + config + fast alias
if [ "$HAS_QWEN" = "1" ]; then
  QWEN_HOME="$HOME/.qwen/quillon"
  QWEN_MCP_DIR="$QWEN_HOME/mcp"
  QWEN_BIN_DIR="$QWEN_HOME/bin"
  QWEN_CONFIG="$QWEN_HOME/quillon-mcp.json"
  mkdir -p "$QWEN_HOME" "$QWEN_MCP_DIR" "$QWEN_BIN_DIR"

  if [ ! -f "/tmp/quillon-mcp.tar.gz" ]; then
    cp -R "$INSTALL_DIR"/. "$QWEN_MCP_DIR"/ 2>/dev/null || true
  fi
  install_mcp_bundle "$QWEN_MCP_DIR" "Qwen Coder"

  QWEN_MCP_INDEX="$QWEN_MCP_DIR/build/index.js"
  cat > "$QWEN_CONFIG" << QWENEOF
{
  "name": "quillon-agentic-money-wallet",
  "client": "qwen-coder",
  "transport": "stdio",
  "command": "node",
  "args": ["$QWEN_MCP_INDEX"],
  "env": {
    "QUILLON_API_URL": "https://quillon.xyz/api/v1",
    "QUILLON_CLIENT": "qwen-coder",
    "QUILLON_AGENT_MODE": "proposer"
  },
  "quickstart": [
    "Read quillon://agentic-money/primer and quillon://privacy-xauth",
    "List Quillon wallet MCP tools",
    "Create a wallet",
    "Use qwen_fast_status first; it batches wallet_identity, signed balance, network status, and market summary",
    "Use qwen_trade_prepare for read-only DEX prep; it batches balance, quote, slippage, and policy verdict",
    "Balances are private and require signed X-Wallet-Auth; the MCP signs automatically",
    "Run strategy_dry_run before any DEX trade if qwen_trade_prepare says READY_FOR_HUMAN_CONFIRM",
    "Use execute_strategy only after explicit human confirmation"
  ],
  "qwen_fast_path": {
    "first_status_tool": "qwen_fast_status",
    "first_trade_tool": "qwen_trade_prepare",
    "avoid": "Do not spawn a fresh MCP server for each long-running mining/status step; use one composite tool call or setup_miner for persistent mining.",
    "schema_note": "Trading dry-runs use strategy_type, not type. Valid values: swap, limit_order, dca, arbitrage."
  },
  "privacy_model": {
    "graph": "private DAG-Knight graph with ZK-SNARK/ZK-STARK privacy posture",
    "balance_reads": "signed X-Wallet-Auth required; MCP signs automatically",
    "agent_rule": "authentication required does not mean zero balance; call wallet_identity then get_balance/portfolio_overview"
  },
  "capabilities": {
    "remote_signing_mode": "planned: QR/notification approval without exposing seed to the model",
    "dry_run_first": "available: strategy_dry_run and execute_strategy confirm=false",
    "async_status": "available where the MCP client renders server log notifications",
    "natural_language_errors": "available in signed wallet/DEX tool errors",
    "code_to_contract_generator": "available: code_to_contract dry-run generator",
    "multi_agent_council": "available: council_consensus proposer/risk/codex review"
  }
}
QWENEOF

  cat > "$QWEN_BIN_DIR/quillon-qwen" << QWENALIASEOF
#!/bin/sh
exec node "$QWEN_MCP_INDEX" "\$@"
QWENALIASEOF
  chmod +x "$QWEN_BIN_DIR/quillon-qwen"

  if ! grep -qs "$QWEN_BIN_DIR" "$HOME/.profile" "$HOME/.bashrc" 2>/dev/null; then
    echo "export PATH=\"\$PATH:$QWEN_BIN_DIR\"" >> "$HOME/.profile"
  fi

echo "  ✓ Qwen Coder configured at $QWEN_CONFIG"
echo "    → Fast alias: $QWEN_BIN_DIR/quillon-qwen"
echo "    → Add this MCP command in Qwen Coder: node $QWEN_MCP_INDEX"
  echo "    → First tool to call: qwen_fast_status"
  echo "    → First trade prep:  qwen_trade_prepare"
fi

# 4e. Grok Build CLI → local config + remote MCP connector config.
if [ "$HAS_GROK" = "1" ]; then
  GROK_HOME="$HOME/.grok/quillon"
  GROK_MCP_DIR="$GROK_HOME/mcp"
  GROK_BIN_DIR="$GROK_HOME/bin"
  GROK_LOCAL_CONFIG="$GROK_HOME/grok-local-mcp.json"
  GROK_REMOTE_CONFIG="$GROK_HOME/grok-remote-mcp.json"
  mkdir -p "$GROK_HOME" "$GROK_MCP_DIR" "$GROK_BIN_DIR"

  if [ ! -f "/tmp/quillon-mcp.tar.gz" ]; then
    cp -R "$INSTALL_DIR"/. "$GROK_MCP_DIR"/ 2>/dev/null || true
  fi
  install_mcp_bundle "$GROK_MCP_DIR" "Grok Build"

  GROK_MCP_INDEX="$GROK_MCP_DIR/build/index.js"
  cat > "$GROK_LOCAL_CONFIG" << GROKEOF
{
  "name": "quillon-grok-agentic-money",
  "client": "grok-build-cli",
  "personality": "chaotic-friendly, truth-seeking, policy-bounded",
  "constitution_resource": "quillon://constitution/grok",
  "transport": "stdio",
  "command": "node",
  "args": ["$GROK_MCP_INDEX"],
  "env": {
    "QUILLON_API_URL": "https://quillon.xyz/api/v1",
    "QUILLON_CLIENT": "grok-build-cli",
    "QUILLON_AGENT_MODE": "proposer"
  },
  "boot_prompt": "Use prompt grok-agent-boot before wallet/trading work."
}
GROKEOF

  cat > "$GROK_REMOTE_CONFIG" << GROKREMOTEEOF
{
  "name": "quillon-grok-remote-mcp",
  "client": "grok-remote-connector",
  "personality": "chaotic-friendly, truth-seeking, policy-bounded",
  "constitution_resource": "quillon://constitution/grok",
  "transport": "streamable-http",
  "url": "https://quillon.xyz/mcp/grok",
  "fallback_local": {
    "command": "node",
    "args": ["$GROK_MCP_INDEX", "--http", "8787"]
  },
  "policy": {
    "default_mode": "proposer",
    "dry_run_first": true,
    "confirm_required": true,
    "never_expose_seed": true,
    "private_graph": true,
    "balance_reads": "signed X-Wallet-Auth via MCP"
  }
}
GROKREMOTEEOF

  cat > "$GROK_BIN_DIR/quillon-grok" << GROKALIASEOF
#!/bin/sh
exec node "$GROK_MCP_INDEX" "\$@"
GROKALIASEOF
  cat > "$GROK_BIN_DIR/quillon-grok-http" << GROKHTTPALIASEOF
#!/bin/sh
PORT="\${1:-8787}"
exec node "$GROK_MCP_INDEX" --http "\$PORT"
GROKHTTPALIASEOF
  chmod +x "$GROK_BIN_DIR/quillon-grok" "$GROK_BIN_DIR/quillon-grok-http"

  if ! grep -qs "$GROK_BIN_DIR" "$HOME/.profile" "$HOME/.bashrc" 2>/dev/null; then
    echo "export PATH=\"\$PATH:$GROK_BIN_DIR\"" >> "$HOME/.profile"
  fi

  echo "  ✓ Grok Build CLI configured at $GROK_LOCAL_CONFIG"
  echo "  ✓ Grok remote MCP config at $GROK_REMOTE_CONFIG"
  echo "    → Local:  $GROK_BIN_DIR/quillon-grok"
  echo "    → Remote: $GROK_BIN_DIR/quillon-grok-http 8787"
fi

rm -f /tmp/quillon-mcp.tar.gz

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
if [ "$HAS_QWEN" = "1" ]; then
echo "  ║  In Qwen Coder:                              ║"
echo "  ║                                               ║"
echo "  ║    \"List Quillon wallet MCP tools\"           ║"
echo "  ║    \"Create a wallet\"                         ║"
echo "  ║    \"Run qwen_fast_status\"                    ║"
echo "  ║    \"Run qwen_trade_prepare for token swap\"   ║"
echo "  ║                                               ║"
fi
if [ "$HAS_GROK" = "1" ]; then
echo "  ║  In Grok Build CLI:                          ║"
echo "  ║                                               ║"
echo "  ║    \"Load grok-agent-boot\"                    ║"
echo "  ║    \"Read agent_constitution grok\"            ║"
echo "  ║    \"Run market_scan then strategy_dry_run\"   ║"
echo "  ║                                               ║"
fi
echo "  ║  No GPG. No air-gapping. Just works.          ║"
echo "  ║  Private graph: signed balance reads via MCP. ║"
echo "  ║                                               ║"
echo "  ║  quillon.xyz | Post-Quantum Electronic Cash   ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""
