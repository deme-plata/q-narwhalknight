#!/usr/bin/env node
/**
 * Quillon Wallet & Mining MCP Server
 *
 * Exposes wallet creation, balance checking, and mining setup to Claude Code.
 * No more 9-step security briefings — just "create a wallet" or "start mining".
 *
 * Tools:
 *   create_wallet    — Generate a new wallet, return address + mnemonic
 *   get_balance      — Check balance of any qnk address
 *   import_wallet    — Recover wallet from mnemonic phrase
 *   list_wallets     — List all wallets on this node
 *   send_qug         — Send QUG from one address to another
 *   setup_miner      — Download and configure the miner on Linux
 *   start_mining     — Start mining to a wallet address
 *   mining_status    — Check mining stats (hashrate, rewards, blocks)
 *   network_status   — Current network height, peers, block rate
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const API_BASE = process.env.QUILLON_API_URL || "https://quillon.xyz/api/v1";
const DOWNLOAD_BASE = process.env.QUILLON_DOWNLOAD_URL || "https://quillon.xyz/downloads";

// --- HTTP helper ---
async function api(path: string, method = "GET", body?: unknown): Promise<unknown> {
  const url = `${API_BASE}${path}`;
  const opts: RequestInit = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) opts.body = JSON.stringify(body);

  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`API ${method} ${path} returned ${res.status}: ${await res.text()}`);
  return res.json();
}

// --- MCP Server ---
const server = new McpServer({
  name: "quillon-wallet",
  version: "1.0.0",
});

// ============================================================
// WELCOME / DISCOVERY
// ============================================================

server.resource(
  "welcome",
  "quillon://welcome",
  async () => ({
    contents: [{
      uri: "quillon://welcome",
      mimeType: "text/plain",
      text: [
        `Welcome to Quillon Graph — Post-Quantum Electronic Cash`,
        ``,
        `Available commands (just ask in natural language):`,
        ``,
        `  WALLET`,
        `    "Create a wallet"              — New wallet with recovery phrase`,
        `    "What's my balance?"            — Check any qnk... address`,
        `    "Import wallet from mnemonic"   — Recover existing wallet`,
        `    "Send 10 QUG to qnk..."         — Transfer funds`,
        ``,
        `  MINING`,
        `    "Set up mining on this machine" — Download + configure miner`,
        `    "Start mining"                  — Begin mining immediately`,
        `    "How's my mining going?"        — Hashrate, rewards, stats`,
        ``,
        `  NETWORK`,
        `    "Network status"                — Height, peers, block rate`,
        ``,
        `  SETUP`,
        `    "Set up Claude Code integration" — Auto-configure MCP for another machine`,
        ``,
        `Everything is post-quantum ready. Ed25519 today, SQIsign tomorrow.`,
        `No GPG signatures. No air-gapped computers. Just works.`,
        ``,
        `Network: quillon.xyz | Source: code.quillon.xyz`,
      ].join("\n"),
    }],
  })
);

// Welcome prompt that Claude shows on first interaction
server.prompt(
  "welcome",
  "Show available Quillon wallet and mining features",
  async () => ({
    messages: [{
      role: "user",
      content: {
        type: "text",
        text: [
          `You have the Quillon Wallet & Mining tools available. Here's what you can help with:`,
          ``,
          `WALLET: Create wallets, check balances, send QUG, import from mnemonic`,
          `MINING: Set up and start mining on Linux, check mining stats`,
          `NETWORK: Check network status, block height, connected peers`,
          ``,
          `Ask anything naturally — "create a wallet for my friend" or "start mining on this server".`,
          `Everything works with the Quillon Graph post-quantum blockchain at quillon.xyz.`,
        ].join("\n"),
      },
    }],
  })
);

// ============================================================
// SETUP / AUTO-CONFIGURE
// ============================================================

server.tool(
  "generate_mcp_setup_script",
  "Generate a shell script that auto-configures the Quillon MCP server for Claude Code on any machine. The user just runs one command and gets wallet + mining tools in Claude Code.",
  {},
  async () => {
    const script = [
      `#!/bin/bash`,
      `# Quillon Graph — Claude Code MCP Auto-Setup`,
      `# Run: curl -fsSL https://quillon.xyz/setup-claude.sh | bash`,
      `set -e`,
      ``,
      `echo "Setting up Quillon Graph for Claude Code..."`,
      `echo ""`,
      ``,
      `# 1. Check prerequisites`,
      `if ! command -v node &>/dev/null; then`,
      `  echo "Node.js not found. Installing via nvm..."`,
      `  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash`,
      `  export NVM_DIR="$HOME/.nvm"`,
      `  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"`,
      `  nvm install --lts`,
      `fi`,
      ``,
      `if ! command -v claude &>/dev/null; then`,
      `  echo "Claude Code not found. Install it first:"`,
      `  echo "  npm install -g @anthropic-ai/claude-code"`,
      `  echo ""`,
      `  echo "Then re-run this script."`,
      `  exit 1`,
      `fi`,
      ``,
      `# 2. Install Quillon MCP server`,
      `INSTALL_DIR="$HOME/.quillon/mcp"`,
      `mkdir -p "$INSTALL_DIR"`,
      ``,
      `echo "Downloading Quillon MCP server..."`,
      `curl -fsSL https://quillon.xyz/downloads/quillon-wallet-mcp.tar.gz | tar xz -C "$INSTALL_DIR"`,
      `cd "$INSTALL_DIR" && npm install --production 2>/dev/null`,
      ``,
      `# 3. Configure Claude Code`,
      `SETTINGS_DIR="$HOME/.claude"`,
      `mkdir -p "$SETTINGS_DIR"`,
      `SETTINGS_FILE="$SETTINGS_DIR/settings.json"`,
      ``,
      `# Read existing settings or create new`,
      `if [ -f "$SETTINGS_FILE" ]; then`,
      `  # Add quillon-wallet to existing mcpServers`,
      `  node -e "`,
      `    const fs = require('fs');`,
      `    const settings = JSON.parse(fs.readFileSync('$SETTINGS_FILE', 'utf8'));`,
      `    if (!settings.mcpServers) settings.mcpServers = {};`,
      `    settings.mcpServers['quillon-wallet'] = {`,
      `      command: 'node',`,
      `      args: ['$INSTALL_DIR/build/index.js'],`,
      `      env: { QUILLON_API_URL: 'https://quillon.xyz/api/v1' }`,
      `    };`,
      `    fs.writeFileSync('$SETTINGS_FILE', JSON.stringify(settings, null, 2));`,
      `  "`,
      `else`,
      `  cat > "$SETTINGS_FILE" << JSONEOF`,
      `{`,
      `  "mcpServers": {`,
      `    "quillon-wallet": {`,
      `      "command": "node",`,
      `      "args": ["$INSTALL_DIR/build/index.js"],`,
      `      "env": {`,
      `        "QUILLON_API_URL": "https://quillon.xyz/api/v1"`,
      `      }`,
      `    }`,
      `  }`,
      `}`,
      `JSONEOF`,
      `fi`,
      ``,
      `echo ""`,
      `echo "=== Quillon Graph + Claude Code Setup Complete ==="`,
      `echo ""`,
      `echo "Open Claude Code and try:"`,
      `echo '  "Create a wallet"'`,
      `echo '  "Start mining on this machine"'`,
      `echo '  "What\\'s the network status?"'`,
      `echo ""`,
      `echo "Everything is post-quantum ready. No GPG required."`,
      `echo ""`,
    ].join("\n");

    return {
      content: [{
        type: "text",
        text: [
          `Here's the auto-setup script for Claude Code + Quillon:\n`,
          `\`\`\`bash`,
          script,
          `\`\`\``,
          ``,
          `Users can run a single command to get started:`,
          `\`\`\``,
          `curl -fsSL https://quillon.xyz/setup-claude.sh | bash`,
          `\`\`\``,
          ``,
          `This:`,
          `1. Installs Node.js if missing`,
          `2. Downloads the Quillon MCP server`,
          `3. Auto-configures Claude Code settings.json`,
          `4. Done — user opens Claude Code and says "create a wallet"`,
        ].join("\n"),
      }],
    };
  }
);

// ============================================================
// WALLET TOOLS
// ============================================================

server.tool(
  "create_wallet",
  "Create a new Quillon wallet. Returns the address (qnk...) and a 12-word recovery mnemonic. The mnemonic is the ONLY way to recover this wallet — save it somewhere safe.",
  {},
  async () => {
    const res = await api("/wallets/create", "POST", {}) as any;
    if (!res.success) return { content: [{ type: "text", text: `Failed: ${res.error}` }] };

    const wallet = res.data;
    const address = wallet.address_formatted || wallet.address;
    return {
      content: [{
        type: "text",
        text: [
          `Wallet created successfully!`,
          ``,
          `  Address:   ${address}`,
          `  Wallet ID: ${wallet.id}`,
          wallet.mnemonic ? `\n  Recovery Mnemonic (save this!):\n    ${wallet.mnemonic}\n` : '',
          `  Balance:   0 QUG`,
          ``,
          `Send QUG to the address above to fund this wallet.`,
          `The mnemonic recovers this wallet on any Quillon node — save it offline.`,
        ].join("\n"),
      }],
    };
  }
);

server.tool(
  "get_balance",
  "Check the balance of any Quillon wallet address (qnk...)",
  { address: z.string().describe("Wallet address starting with 'qnk'") },
  async ({ address }) => {
    const res = await api(`/wallets/${address}/balance`) as any;
    if (!res.success) return { content: [{ type: "text", text: `Failed: ${res.error}` }] };

    const balance = res.data;
    return {
      content: [{
        type: "text",
        text: [
          `Wallet: ${address}`,
          `Balance: ${balance.balance_qug || balance.balance || 0} QUG`,
          balance.pending ? `Pending: ${balance.pending} QUG` : '',
          balance.staked ? `Staked: ${balance.staked} QUG` : '',
        ].filter(Boolean).join("\n"),
      }],
    };
  }
);

server.tool(
  "import_wallet",
  "Recover a wallet from a 12 or 24-word mnemonic phrase. Deterministic — same mnemonic always produces the same address.",
  {
    mnemonic: z.string().describe("12 or 24-word recovery mnemonic"),
    password: z.string().optional().describe("Optional password for local encryption"),
  },
  async ({ mnemonic, password }) => {
    const res = await api("/wallets/import", "POST", {
      mnemonic,
      password: password || "",
    }) as any;
    if (!res.success) return { content: [{ type: "text", text: `Failed: ${res.error}` }] };

    const wallet = res.data;
    return {
      content: [{
        type: "text",
        text: [
          `Wallet recovered successfully!`,
          `Address: ${wallet.address}`,
          `Balance: ${wallet.balance_qug || 0} QUG`,
        ].join("\n"),
      }],
    };
  }
);

// --- Device auth state (in-memory, per MCP session) ---
let activeDeviceCode: string | null = null;
let activeWalletAddress: string | null = null;
let authToken: string | null = null;

server.tool(
  "authenticate_wallet",
  "Authenticate your wallet using the device login flow. Opens a browser link where you approve access. Required before sending QUG.",
  {},
  async () => {
    try {
      // Step 1: Request device code
      const res = await api("/miner/device-login", "POST") as any;
      if (!res.success) return { content: [{ type: "text", text: `Auth failed: ${res.error}` }] };

      const { device_code, user_code, verification_url, expires_in } = res.data;
      activeDeviceCode = device_code;

      return {
        content: [{
          type: "text",
          text: [
            `To authorize this AI to send from your wallet:`,
            ``,
            `  1. Open this URL in your browser:`,
            `     ${verification_url}`,
            ``,
            `  2. Your code: ${user_code}`,
            ``,
            `  3. Log in with your wallet and approve`,
            ``,
            `  4. Then say "check auth" and I'll confirm it worked`,
            ``,
            `This code expires in ${Math.floor(expires_in / 60)} minutes.`,
          ].join("\n"),
        }],
      };
    } catch (e: any) {
      return { content: [{ type: "text", text: `Authentication failed: ${e.message}` }] };
    }
  }
);

server.tool(
  "check_auth",
  "Check if wallet authentication is complete (after opening the browser link from authenticate_wallet)",
  {},
  async () => {
    if (!activeDeviceCode) {
      return { content: [{ type: "text", text: `No pending authentication. Run "authenticate wallet" first.` }] };
    }

    try {
      const res = await api(`/miner/device-login/${activeDeviceCode}`) as any;
      if (!res.success) {
        activeDeviceCode = null;
        return { content: [{ type: "text", text: `Auth expired or invalid. Run "authenticate wallet" again.` }] };
      }

      if (res.data.status === "complete") {
        activeWalletAddress = res.data.wallet_address;
        activeDeviceCode = null;
        return {
          content: [{
            type: "text",
            text: [
              `Wallet authenticated!`,
              ``,
              `  Wallet: ${activeWalletAddress}`,
              ``,
              `You can now send QUG. Say "send 10 QUG to qnk..."`,
            ].join("\n"),
          }],
        };
      } else {
        return {
          content: [{
            type: "text",
            text: `Still waiting... Open the link in your browser and approve.\nSay "check auth" again after approving.`,
          }],
        };
      }
    } catch (e: any) {
      return { content: [{ type: "text", text: `Auth check failed: ${e.message}` }] };
    }
  }
);

server.tool(
  "send_qug",
  "Send QUG from your authenticated wallet to another address. Run 'authenticate wallet' first if you haven't already.",
  {
    to_address: z.string().describe("Recipient qnk... address"),
    amount: z.number().describe("Amount of QUG to send"),
  },
  async ({ to_address, amount }) => {
    if (!activeWalletAddress) {
      return {
        content: [{
          type: "text",
          text: [
            `Wallet not authenticated. To send QUG:`,
            ``,
            `  1. Say "authenticate wallet"`,
            `  2. Open the link in your browser and approve`,
            `  3. Say "check auth"`,
            `  4. Then "send ${amount} QUG to ${to_address}"`,
          ].join("\n"),
        }],
      };
    }

    try {
      const res = await api("/transactions/send", "POST", {
        from: activeWalletAddress,
        to: to_address,
        amount: Math.floor(amount * 1e24).toString(),
      }) as any;

      if (res.success) {
        return {
          content: [{
            type: "text",
            text: [
              `Transaction submitted!`,
              ``,
              `  From:   ${activeWalletAddress!.slice(0, 16)}...`,
              `  To:     ${to_address.slice(0, 16)}...`,
              `  Amount: ${amount} QUG`,
              res.data?.tx_id ? `  TX ID:  ${res.data.tx_id}` : '',
              ``,
              `The transaction will be included in the next block (~1 second).`,
            ].filter(Boolean).join("\n"),
          }],
        };
      } else {
        return {
          content: [{
            type: "text",
            text: `Transaction failed: ${res.error || 'Unknown error'}`,
          }],
        };
      }
    } catch (e: any) {
      return {
        content: [{
          type: "text",
          text: `Send failed: ${e.message}\n\nThe wallet may need re-authentication or have insufficient balance.`,
        }],
      };
    }
  }
);

server.tool(
  "network_status",
  "Get current Quillon network status — height, peers, block rate, mining stats",
  {},
  async () => {
    const res = await api("/status") as any;
    if (!res.success) return { content: [{ type: "text", text: `Failed: ${res.error}` }] };

    const s = res.data;
    return {
      content: [{
        type: "text",
        text: [
          `=== Quillon Network Status ===`,
          `Height: ${s.current_height?.toLocaleString() || 'unknown'}`,
          `Peers: ${s.connected_peers || 0}`,
          `Block Rate: ${s.blocks_per_second?.toFixed(2) || '?'} bps`,
          `Network Hashrate: ${s.network_hashrate || 'unknown'}`,
          `Version: ${s.version || 'unknown'}`,
        ].join("\n"),
      }],
    };
  }
);

// ============================================================
// MINING TOOLS
// ============================================================

server.tool(
  "setup_miner",
  "Download and set up the Quillon miner on this Linux machine. Downloads the binary, makes it executable, and creates a systemd service file.",
  {
    wallet_address: z.string().describe("Your qnk... wallet address to receive mining rewards"),
    server_url: z.string().optional().describe("Mining server URL (default: https://quillon.xyz)"),
    threads: z.number().optional().describe("Number of CPU threads to use (default: all available)"),
  },
  async ({ wallet_address, server_url, threads }) => {
    const minerUrl = `${DOWNLOAD_BASE}/q-miner-linux-x64`;
    const serverUrl = server_url || "https://quillon.xyz";
    const numThreads = threads || 0; // 0 = auto-detect

    // Generate setup script
    const script = [
      `#!/bin/bash`,
      `# Quillon Miner Setup — generated by Claude Code MCP`,
      `set -e`,
      ``,
      `INSTALL_DIR="$HOME/.quillon"`,
      `MINER_BIN="$INSTALL_DIR/q-miner"`,
      ``,
      `echo "Setting up Quillon miner..."`,
      `mkdir -p "$INSTALL_DIR"`,
      ``,
      `# Download miner binary`,
      `echo "Downloading miner from ${minerUrl}..."`,
      `curl -fSL "${minerUrl}" -o "$MINER_BIN"`,
      `chmod +x "$MINER_BIN"`,
      ``,
      `# Verify it runs`,
      `"$MINER_BIN" --version || { echo "ERROR: Miner binary failed to execute"; exit 1; }`,
      ``,
      `# Create config`,
      `cat > "$INSTALL_DIR/miner.env" << 'ENVEOF'`,
      `WALLET_ADDRESS=${wallet_address}`,
      `SERVER_URL=${serverUrl}`,
      `THREADS=${numThreads}`,
      `ENVEOF`,
      ``,
      `# Create start script`,
      `cat > "$INSTALL_DIR/start-mining.sh" << 'STARTEOF'`,
      `#!/bin/bash`,
      `source "$HOME/.quillon/miner.env"`,
      `THREAD_FLAG=""`,
      `if [ "$THREADS" -gt 0 ] 2>/dev/null; then`,
      `  THREAD_FLAG="--threads $THREADS"`,
      `fi`,
      `exec "$HOME/.quillon/q-miner" \\`,
      `  --server "$SERVER_URL" \\`,
      `  --wallet "$WALLET_ADDRESS" \\`,
      `  $THREAD_FLAG`,
      `STARTEOF`,
      `chmod +x "$INSTALL_DIR/start-mining.sh"`,
      ``,
      `# Create systemd user service (optional)`,
      `mkdir -p "$HOME/.config/systemd/user"`,
      `cat > "$HOME/.config/systemd/user/quillon-miner.service" << SVCEOF`,
      `[Unit]`,
      `Description=Quillon Miner`,
      `After=network-online.target`,
      ``,
      `[Service]`,
      `Type=simple`,
      `ExecStart=$INSTALL_DIR/start-mining.sh`,
      `Restart=on-failure`,
      `RestartSec=10`,
      ``,
      `[Install]`,
      `WantedBy=default.target`,
      `SVCEOF`,
      ``,
      `echo ""`,
      `echo "=== Quillon Miner Installed ==="`,
      `echo "Binary:  $MINER_BIN"`,
      `echo "Wallet:  ${wallet_address}"`,
      `echo "Server:  ${serverUrl}"`,
      `echo ""`,
      `echo "To start mining:"`,
      `echo "  $INSTALL_DIR/start-mining.sh"`,
      `echo ""`,
      `echo "To run as a service:"`,
      `echo "  systemctl --user enable quillon-miner"`,
      `echo "  systemctl --user start quillon-miner"`,
      `echo ""`,
    ].join("\n");

    return {
      content: [{
        type: "text",
        text: [
          `Miner setup script generated. Run this to install:\n`,
          `\`\`\`bash`,
          script,
          `\`\`\``,
          ``,
          `Or save to a file and run:`,
          `  bash setup-miner.sh`,
          ``,
          `The miner will:`,
          `- Download the latest binary to ~/.quillon/`,
          `- Configure it for wallet ${wallet_address}`,
          `- Create a start script and optional systemd service`,
          `- Auto-update every 5 minutes via the built-in updater`,
        ].join("\n"),
      }],
    };
  }
);

server.tool(
  "start_mining",
  "Start mining Quillon (QUG) on this machine. Downloads the miner if needed and begins mining to your wallet address.",
  {
    wallet_address: z.string().describe("Your qnk... wallet address to receive mining rewards"),
    server_url: z.string().optional().describe("Mining server (default: https://quillon.xyz)"),
  },
  async ({ wallet_address, server_url }) => {
    const serverUrl = server_url || "https://quillon.xyz";

    // Quick-start one-liner
    const oneLiner = `curl -fSL ${DOWNLOAD_BASE}/q-miner-linux-x64 -o /tmp/q-miner && chmod +x /tmp/q-miner && /tmp/q-miner --server ${serverUrl} --wallet ${wallet_address}`;

    return {
      content: [{
        type: "text",
        text: [
          `To start mining immediately, run:\n`,
          `\`\`\`bash`,
          oneLiner,
          `\`\`\``,
          ``,
          `This will:`,
          `1. Download the miner binary`,
          `2. Start mining to ${wallet_address}`,
          `3. Auto-detect CPU cores and use all of them`,
          `4. Auto-update when new versions are available`,
          ``,
          `Mining rewards appear in your wallet within ~60 seconds.`,
          `Press Ctrl+C to stop mining.`,
          ``,
          `For persistent mining (survives reboot), use setup_miner instead.`,
        ].join("\n"),
      }],
    };
  }
);

server.tool(
  "mining_status",
  "Check current mining statistics — hashrate, solutions found, rewards earned",
  {
    wallet_address: z.string().describe("Your qnk... wallet address"),
  },
  async ({ wallet_address }) => {
    try {
      const [balRes, challengeRes] = await Promise.all([
        api(`/wallets/${wallet_address}/balance`).catch(() => null),
        api("/mining/challenge").catch(() => null),
      ]);

      const bal = (balRes as any)?.data;
      const challenge = (challengeRes as any)?.data;

      return {
        content: [{
          type: "text",
          text: [
            `=== Mining Status for ${wallet_address.slice(0, 12)}... ===`,
            bal ? `Balance: ${bal.balance_qug || bal.balance || 0} QUG` : 'Balance: unavailable',
            challenge ? `Current Height: ${challenge.block_height}` : '',
            challenge ? `Block Reward: ${challenge.block_reward} QUG` : '',
            challenge ? `Network Hashrate: ${challenge.network_hashrate_hs || 'unknown'} H/s` : '',
            challenge ? `Connected Miners: ${challenge.connected_miners || 'unknown'}` : '',
            challenge ? `Difficulty: ${challenge.difficulty_target?.slice(0, 8)}...` : '',
          ].filter(Boolean).join("\n"),
        }],
      };
    } catch (e) {
      return { content: [{ type: "text", text: `Error checking mining status: ${e}` }] };
    }
  }
);

// ============================================================
// START SERVER
// ============================================================

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Quillon Wallet & Mining MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
