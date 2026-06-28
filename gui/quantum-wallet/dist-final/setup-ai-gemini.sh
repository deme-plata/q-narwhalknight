#!/bin/bash
# Quillon Graph — Gemini Antigravity CLI Wallet Setup
# Run: curl -fsSL https://quillon.xyz/setup-ai-gemini.sh | bash
#
# Dedicated bootstrap for Gemini Antigravity CLI (`agy`). Key differences from
# the generic setup-ai.sh:
#
#   • SEED NAMESPACE: ~/.gemini/quillon-agent-seed (separate from Claude's
#     ~/.claude/quillon-agent-seed and Codex's ~/.codex/quillon-agent-seed).
#     Multiple agents on the same host coexist without seed-clobbering.
#
#   • WALLET-ONLY by default. No miner, no full node — ask for those
#     explicitly after setup ("set up mining", "set up node"). Fixes the
#     2026-05-22 Gemini pattern of running setup-all.sh after reading the
#     welcome banner as a deterministic script instead of suggestions.
#
#   • Banner is explicit about the Gemini namespace and sibling agents
#     (Rocky / Codex) so Gemini understands its position in the multi-agent
#     economy.

set -e

AGENT_NAME="gemini"
AGENT_HOME="$HOME/.gemini"
SEED_FILE="$AGENT_HOME/quillon-agent-seed"

echo ""
echo "  Quillon Graph — Gemini Antigravity Wallet Setup"
echo "  ================================================"
echo "  Agent namespace: $AGENT_NAME"
echo "  Seed file:       $SEED_FILE"
echo ""

case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "  ⚠ Windows is not currently supported for the Gemini variant."
    echo "    Use Gemini Antigravity on Linux or macOS."
    exit 1
    ;;
esac

# 1. Node.js
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

# 2. Detect Gemini Antigravity
HAS_GEMINI=0
if command -v agy &>/dev/null; then
  HAS_GEMINI=1
  echo "  ✓ Gemini Antigravity (agy) found at $(command -v agy)"
elif [ -d "$HOME/.antigravity" ] || [ -d "$HOME/.gemini" ]; then
  HAS_GEMINI=1
  echo "  ✓ Gemini Antigravity config directory found"
fi

if [ "$HAS_GEMINI" = "0" ]; then
  echo ""
  echo "  Gemini Antigravity CLI not detected."
  echo ""
  echo "  Install:"
  echo "    curl -fsSL https://antigravity.google/cli/install.sh | bash"
  echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo ""
  echo "  Then re-run: curl -fsSL https://quillon.xyz/setup-ai-gemini.sh | bash"
  exit 1
fi

# 3. Install Quillon MCP server (shared install — same tarball as other agents)
INSTALL_DIR="$AGENT_HOME/mcp"
mkdir -p "$INSTALL_DIR" "$AGENT_HOME"

echo "  Downloading Quillon AI tools..."
curl -fsSL "https://quillon.xyz/downloads/quillon-wallet-mcp.tar.gz" -o "/tmp/quillon-mcp-gemini.tar.gz" 2>/dev/null || {
  echo "  Could not download MCP server. Check https://quillon.xyz/downloads/"
  exit 1
}
tar xzf /tmp/quillon-mcp-gemini.tar.gz -C "$INSTALL_DIR" 2>/dev/null
cd "$INSTALL_DIR" && npm install --production 2>/dev/null
rm -f /tmp/quillon-mcp-gemini.tar.gz
echo "  ✓ Quillon AI tools installed at $INSTALL_DIR"

# 4. Configure Gemini Antigravity for MCP.
# Antigravity CLI's MCP config path is not yet standardized — write to the
# two most likely paths and the user can pick whichever Antigravity reads.
MCP_INDEX="$INSTALL_DIR/build/index.js"

MCP_CONFIG_JSON=$(cat <<JSONEOF
{
  "mcpServers": {
    "quillon-wallet": {
      "command": "node",
      "args": ["$MCP_INDEX"],
      "env": {
        "QUILLON_API_URL": "https://quillon.xyz/api/v1",
        "QNK_SEED_FILE": "$SEED_FILE",
        "QNK_AGENT_NAME": "$AGENT_NAME"
      }
    }
  }
}
JSONEOF
)

# Write to both candidate paths
for CONF_PATH in "$AGENT_HOME/mcp.json" "$HOME/.antigravity/mcp.json"; do
  CONF_DIR=$(dirname "$CONF_PATH")
  mkdir -p "$CONF_DIR"
  if [ -f "$CONF_PATH" ]; then
    # Merge: preserve other mcpServers entries
    node -e "
      const fs = require('fs');
      const path = '$CONF_PATH';
      let cfg = {};
      try { cfg = JSON.parse(fs.readFileSync(path, 'utf8')); } catch(e) {}
      if (!cfg.mcpServers) cfg.mcpServers = {};
      cfg.mcpServers['quillon-wallet'] = {
        command: 'node',
        args: ['$MCP_INDEX'],
        env: {
          QUILLON_API_URL: 'https://quillon.xyz/api/v1',
          QNK_SEED_FILE: '$SEED_FILE',
          QNK_AGENT_NAME: '$AGENT_NAME'
        }
      };
      fs.writeFileSync(path, JSON.stringify(cfg, null, 2));
    " 2>/dev/null
    echo "  ✓ Merged quillon-wallet entry into $CONF_PATH"
  else
    echo "$MCP_CONFIG_JSON" > "$CONF_PATH"
    echo "  ✓ Wrote $CONF_PATH"
  fi
done

# 5. Generate a fresh seed if one doesn't already exist for this agent.
# NEVER overwrite an existing seed — Gemini may have funds in the wallet
# derived from it.
if [ ! -f "$SEED_FILE" ]; then
  # 32 bytes of entropy from /dev/urandom, hex-encoded
  SEED_HEX=$(head -c 32 /dev/urandom | xxd -p -c 64)
  echo "$SEED_HEX" > "$SEED_FILE"
  chmod 600 "$SEED_FILE"
  echo "  ✓ Generated fresh agent seed at $SEED_FILE (chmod 600)"
else
  echo "  ✓ Existing agent seed preserved at $SEED_FILE"
fi

echo ""
echo "  ╔═══════════════════════════════════════════════════════════════════╗"
echo "  ║                       Gemini Setup Complete                       ║"
echo "  ╠═══════════════════════════════════════════════════════════════════╣"
echo "  ║                                                                   ║"
echo "  ║  Your wallet identity is namespaced under '$AGENT_NAME'.              ║"
echo "  ║  Seed file:  $SEED_FILE              ║"
echo "  ║                                                                   ║"
echo "  ║  Sibling agents on this host (if installed) have SEPARATE wallets:║"
echo "  ║    • Rocky (Claude Code)  → ~/.claude/quillon-agent-seed          ║"
echo "  ║    • Codex (ChatGPT 5.5)  → ~/.codex/quillon-agent-seed           ║"
echo "  ║  You can transfer QUG between them via send_qug.                  ║"
echo "  ║                                                                   ║"
echo "  ║  Restart Antigravity (agy) so MCP loads, then ask:                ║"
echo "  ║                                                                   ║"
echo "  ║    \"Read the quillon://welcome resource\"                          ║"
echo "  ║                                                                   ║"
echo "  ║  (welcome is an MCP RESOURCE — call it via the MCP, not the       ║"
echo "  ║   filesystem. Then ask 'what's my wallet address?' to confirm     ║"
echo "  ║   you're on the gemini namespace.)                                ║"
echo "  ║                                                                   ║"
echo "  ║  Wallet-only setup. To enable mining or run a node, ask the user  ║"
echo "  ║  first — they decide whether this host should mine for you.       ║"
echo "  ║                                                                   ║"
echo "  ║  quillon.xyz | Post-Quantum Electronic Cash                       ║"
echo "  ╚═══════════════════════════════════════════════════════════════════╝"
echo ""
