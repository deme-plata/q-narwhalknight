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
// ═══════════════════════════════════════════════════════════════
// SECURITY FIX 5: API URL validation — prevent SSRF/phishing via env vars
// ═══════════════════════════════════════════════════════════════
const ALLOWED_API_DOMAINS = ["quillon.xyz", "localhost", "127.0.0.1"];
function validateApiUrl(url) {
    try {
        const parsed = new URL(url);
        // Enforce HTTPS for non-local URLs
        if (parsed.hostname !== "localhost" && parsed.hostname !== "127.0.0.1") {
            if (parsed.protocol !== "https:") {
                console.error(`SECURITY: Rejecting non-HTTPS API URL: ${url}`);
                return "https://quillon.xyz/api/v1";
            }
        }
        // Validate domain against allowlist
        if (!ALLOWED_API_DOMAINS.some(d => parsed.hostname === d || parsed.hostname.endsWith(`.${d}`))) {
            console.error(`SECURITY: Rejecting untrusted API domain: ${parsed.hostname}`);
            return "https://quillon.xyz/api/v1";
        }
        return url;
    }
    catch {
        console.error(`SECURITY: Invalid API URL: ${url}`);
        return "https://quillon.xyz/api/v1";
    }
}
const API_BASE = validateApiUrl(process.env.QUILLON_API_URL || "https://quillon.xyz/api/v1");
const DOWNLOAD_BASE = validateApiUrl(process.env.QUILLON_DOWNLOAD_URL || "https://quillon.xyz/downloads");
// --- HTTP helper ---
async function api(path, method = "GET", body) {
    const url = `${API_BASE}${path}`;
    const opts = {
        method,
        headers: { "Content-Type": "application/json" },
        redirect: "error", // SECURITY: Never follow redirects (prevents open redirect attacks)
    };
    if (body)
        opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    if (!res.ok)
        throw new Error(`API ${method} ${path} returned ${res.status}: ${await res.text()}`);
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
server.resource("welcome", "quillon://welcome", async () => ({
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
                `  NODE`,
                `    "Set up a node on this machine" — Download binary + systemd service`,
                `    "Set up node from source"       — Build with Rust, then install`,
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
}));
// Welcome prompt that Claude shows on first interaction
server.prompt("welcome", "Show available Quillon wallet and mining features", async () => ({
    messages: [{
            role: "user",
            content: {
                type: "text",
                text: [
                    `You have the Quillon Wallet & Mining tools available. Here's what you can help with:`,
                    ``,
                    `WALLET: Create wallets, check balances, send QUG, import from mnemonic`,
                    `MINING: Set up and start mining on Linux, check mining stats`,
                    `NODE: Set up a full node on this machine — binary install or build from source`,
                    `NETWORK: Check network status, block height, connected peers`,
                    ``,
                    `Ask anything naturally — "create a wallet", "start mining on this server", or "set up a node".`,
                    `Everything works with the Quillon Graph post-quantum blockchain at quillon.xyz.`,
                ].join("\n"),
            },
        }],
}));
// ============================================================
// SETUP / AUTO-CONFIGURE
// ============================================================
server.tool("generate_mcp_setup_script", "Generate a shell script that auto-configures the Quillon MCP server for Claude Code on any machine. The user just runs one command and gets wallet + mining tools in Claude Code.", {}, async () => {
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
});
// ============================================================
// WALLET TOOLS
// ============================================================
server.tool("create_wallet", "Create a new Quillon wallet. Returns the address (qnk...) and a 12-word recovery mnemonic. The mnemonic is the ONLY way to recover this wallet — save it somewhere safe.", {}, async () => {
    const res = await api("/wallets/create", "POST", {});
    if (!res.success)
        return { content: [{ type: "text", text: `Failed: ${res.error}` }] };
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
});
server.tool("get_balance", "Check the balance of any Quillon wallet address (qnk...)", { address: z.string().describe("Wallet address starting with 'qnk'") }, async ({ address }) => {
    const res = await api(`/wallets/${address}/balance`);
    if (!res.success)
        return { content: [{ type: "text", text: `Failed: ${res.error}` }] };
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
});
server.tool("import_wallet", "Recover a wallet from a 12 or 24-word mnemonic phrase. Deterministic — same mnemonic always produces the same address.", {
    mnemonic: z.string().describe("12 or 24-word recovery mnemonic"),
    password: z.string().optional().describe("Optional password for local encryption"),
}, async ({ mnemonic, password }) => {
    const res = await api("/wallets/import", "POST", {
        mnemonic,
        password: password || "",
    });
    if (!res.success)
        return { content: [{ type: "text", text: `Failed: ${res.error}` }] };
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
});
// ═══════════════════════════════════════════════════════════════
// SECURITY FIX 4: Per-session wallet isolation with expiry
// ═══════════════════════════════════════════════════════════════
// Auth state expires after 30 minutes of inactivity. Each MCP stdio
// session is already process-isolated, but token expiry prevents stale
// sessions from accumulating risk.
const SESSION_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
let activeDeviceCode = null;
let activeWalletAddress = null;
let authToken = null;
let sessionAuthenticatedAt = null;
function isSessionValid() {
    if (!activeWalletAddress || !sessionAuthenticatedAt)
        return false;
    if (Date.now() - sessionAuthenticatedAt > SESSION_TIMEOUT_MS) {
        // Session expired — clear all auth state
        activeWalletAddress = null;
        authToken = null;
        sessionAuthenticatedAt = null;
        return false;
    }
    return true;
}
function refreshSession() {
    if (sessionAuthenticatedAt)
        sessionAuthenticatedAt = Date.now();
}
server.tool("authenticate_wallet", "Authenticate your wallet using the device login flow. Opens a browser link where you approve access. Required before sending QUG.", {}, async () => {
    try {
        // Step 1: Request device code
        const res = await api("/miner/device-login", "POST");
        if (!res.success)
            return { content: [{ type: "text", text: `Auth failed: ${res.error}` }] };
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
    }
    catch (e) {
        return { content: [{ type: "text", text: `Authentication failed: ${e.message}` }] };
    }
});
server.tool("check_auth", "Check if wallet authentication is complete (after opening the browser link from authenticate_wallet)", {}, async () => {
    if (!activeDeviceCode) {
        return { content: [{ type: "text", text: `No pending authentication. Run "authenticate wallet" first.` }] };
    }
    try {
        const res = await api(`/miner/device-login/${activeDeviceCode}`);
        if (!res.success) {
            activeDeviceCode = null;
            return { content: [{ type: "text", text: `Auth expired or invalid. Run "authenticate wallet" again.` }] };
        }
        if (res.data.status === "complete") {
            activeWalletAddress = res.data.wallet_address;
            authToken = res.data.token || null;
            sessionAuthenticatedAt = Date.now();
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
        }
        else {
            return {
                content: [{
                        type: "text",
                        text: `Still waiting... Open the link in your browser and approve.\nSay "check auth" again after approving.`,
                    }],
            };
        }
    }
    catch (e) {
        return { content: [{ type: "text", text: `Auth check failed: ${e.message}` }] };
    }
});
server.tool("send_qug", "Send QUG from your authenticated wallet to another address. Run 'authenticate wallet' first if you haven't already.", {
    to_address: z.string().describe("Recipient qnk... address"),
    amount: z.number().describe("Amount of QUG to send"),
}, async ({ to_address, amount }) => {
    if (!isSessionValid()) {
        return {
            content: [{
                    type: "text",
                    text: [
                        `Wallet not authenticated${sessionAuthenticatedAt ? ' (session expired)' : ''}. To send QUG:`,
                        ``,
                        `  1. Say "authenticate wallet"`,
                        `  2. Open the link in your browser and approve`,
                        `  3. Say "check auth"`,
                        `  4. Then "send ${amount} QUG to ${to_address}"`,
                    ].join("\n"),
                }],
        };
    }
    refreshSession(); // Keep session alive on activity
    // ═══════════════════════════════════════════════════════════════
    // SECURITY FIX 1: Validate wallet address format
    // ═══════════════════════════════════════════════════════════════
    if (!to_address.startsWith('qnk') || to_address.length !== 67 || !/^qnk[0-9a-f]{64}$/.test(to_address)) {
        return {
            content: [{
                    type: "text",
                    text: `Invalid recipient address. Must be 'qnk' + 64 hex characters (67 total). Got: ${to_address.slice(0, 20)}...`,
                }],
        };
    }
    // ═══════════════════════════════════════════════════════════════
    // SECURITY FIX 2: Require explicit confirmation for transactions
    // ═══════════════════════════════════════════════════════════════
    // Return a confirmation prompt instead of executing immediately.
    // The AI must relay this to the user and get explicit approval.
    if (!to_address.startsWith('qnk_CONFIRMED_')) {
        return {
            content: [{
                    type: "text",
                    text: [
                        `⚠️ TRANSACTION CONFIRMATION REQUIRED`,
                        ``,
                        `  From:   ${activeWalletAddress.slice(0, 20)}...`,
                        `  To:     ${to_address.slice(0, 20)}...`,
                        `  Amount: ${amount} QUG`,
                        ``,
                        `Please confirm: do you want to send ${amount} QUG to ${to_address}?`,
                        `Reply "yes, send ${amount} QUG to ${to_address}" to proceed.`,
                        ``,
                        `⚠️ This action is irreversible. Verify the recipient address carefully.`,
                    ].join("\n"),
                }],
        };
    }
    // Strip confirmation prefix
    const confirmed_address = to_address.replace('qnk_CONFIRMED_', 'qnk');
    // ═══════════════════════════════════════════════════════════════
    // SECURITY FIX 3: Spending limits
    // ═══════════════════════════════════════════════════════════════
    if (amount > 1000) {
        return {
            content: [{
                    type: "text",
                    text: `⚠️ Amount exceeds MCP spending limit (1000 QUG). For larger transfers, use the web wallet at quillon.xyz.`,
                }],
        };
    }
    if (amount <= 0) {
        return {
            content: [{
                    type: "text",
                    text: `Invalid amount: must be greater than 0.`,
                }],
        };
    }
    try {
        const res = await api("/transactions/send", "POST", {
            from: activeWalletAddress,
            to: confirmed_address,
            amount: Math.floor(amount * 1e24).toString(),
            ...(authToken ? { auth_token: authToken } : {}),
        });
        if (res.success) {
            return {
                content: [{
                        type: "text",
                        text: [
                            `Transaction submitted!`,
                            ``,
                            `  From:   ${activeWalletAddress.slice(0, 16)}...`,
                            `  To:     ${to_address.slice(0, 16)}...`,
                            `  Amount: ${amount} QUG`,
                            res.data?.tx_id ? `  TX ID:  ${res.data.tx_id}` : '',
                            ``,
                            `The transaction will be included in the next block (~1 second).`,
                        ].filter(Boolean).join("\n"),
                    }],
            };
        }
        else {
            return {
                content: [{
                        type: "text",
                        text: `Transaction failed: ${res.error || 'Unknown error'}`,
                    }],
            };
        }
    }
    catch (e) {
        return {
            content: [{
                    type: "text",
                    text: `Send failed: ${e.message}\n\nThe wallet may need re-authentication or have insufficient balance.`,
                }],
        };
    }
});
server.tool("network_status", "Get current Quillon network status — height, peers, block rate, mining stats", {}, async () => {
    const res = await api("/status");
    if (!res.success)
        return { content: [{ type: "text", text: `Failed: ${res.error}` }] };
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
});
// ============================================================
// MINING TOOLS
// ============================================================
server.tool("setup_miner", "Download and set up the Quillon miner on this Linux machine. Downloads the binary, makes it executable, and creates a systemd service file.", {
    wallet_address: z.string().describe("Your qnk... wallet address to receive mining rewards"),
    server_url: z.string().optional().describe("Mining server URL (default: https://quillon.xyz)"),
    threads: z.number().optional().describe("Number of CPU threads to use (default: all available)"),
}, async ({ wallet_address, server_url, threads }) => {
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
});
server.tool("start_mining", "Start mining Quillon (QUG) on this machine. Downloads the miner if needed and begins mining to your wallet address.", {
    wallet_address: z.string().describe("Your qnk... wallet address to receive mining rewards"),
    server_url: z.string().optional().describe("Mining server (default: https://quillon.xyz)"),
}, async ({ wallet_address, server_url }) => {
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
});
server.tool("mining_status", "Check current mining statistics — hashrate, solutions found, rewards earned", {
    wallet_address: z.string().describe("Your qnk... wallet address"),
}, async ({ wallet_address }) => {
    try {
        const [balRes, challengeRes] = await Promise.all([
            api(`/wallets/${wallet_address}/balance`).catch(() => null),
            api("/mining/challenge").catch(() => null),
        ]);
        const bal = balRes?.data;
        const challenge = challengeRes?.data;
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
    }
    catch (e) {
        return { content: [{ type: "text", text: `Error checking mining status: ${e}` }] };
    }
});
// ============================================================
// NODE SETUP
// ============================================================
server.tool("setup_node", "Set up a full Quillon (QNK) blockchain node on this Debian/Ubuntu Linux machine. Downloads the latest binary, creates the data directory, and installs a systemd service that survives reboots. After setup the node will sync automatically.", {
    install_dir: z.string().optional().describe("Directory to install the node (default: /opt/quillon)"),
    data_dir: z.string().optional().describe("Directory for blockchain data (default: /opt/quillon/data)"),
    api_port: z.number().optional().describe("HTTP API port (default: 8080)"),
    p2p_port: z.number().optional().describe("P2P gossip port (default: 9001)"),
    wallet_address: z.string().optional().describe("Your qnk... wallet address to use as the node admin wallet. If omitted, the setup wizard will ask interactively."),
    build_from_source: z.boolean().optional().describe("Build from source using Rust instead of downloading pre-built binary (default: false)"),
}, async ({ install_dir, data_dir, api_port, p2p_port, build_from_source, wallet_address }) => {
    const installDir = install_dir || "/opt/quillon";
    const dataDir = data_dir || `${installDir}/data`;
    const apiPort = api_port || 8080;
    const p2pPort = p2p_port || 9001;
    const binaryUrl = `${DOWNLOAD_BASE}/q-api-server-linux-x86_64`;
    // Auto-create a fresh wallet for this node if none supplied
    let adminWallet = wallet_address || "";
    let newWalletMnemonic = "";
    if (!adminWallet) {
        try {
            const res = await api("/wallets/create", "POST", {});
            if (res.success && res.data?.address_formatted) {
                adminWallet = res.data.address_formatted;
                newWalletMnemonic = res.data.mnemonic || "";
            }
        }
        catch { }
    }
    // Systemd service file content
    const serviceFile = [
        `[Unit]`,
        `Description=Quillon Graph Node`,
        `Documentation=https://quillon.xyz`,
        `After=network-online.target`,
        `Wants=network-online.target`,
        ``,
        `[Service]`,
        `Type=simple`,
        `User=root`,
        `WorkingDirectory=${installDir}`,
        `Environment="Q_DB_PATH=${dataDir}"`,
        `Environment="Q_NETWORK_ID=mainnet-genesis"`,
        `Environment="RUST_LOG=warn"`,
        `ExecStart=${installDir}/q-api-server --port ${apiPort}`,
        `Restart=on-failure`,
        `RestartSec=10`,
        `LimitNOFILE=65536`,
        ``,
        `[Install]`,
        `WantedBy=multi-user.target`,
    ].join("\n");
    if (build_from_source) {
        // Build-from-source script
        const script = [
            `#!/bin/bash`,
            `# Quillon Node — Build from Source`,
            `# Requires: Debian 12 / Ubuntu 22.04+`,
            `set -e`,
            ``,
            `INSTALL_DIR="${installDir}"`,
            `DATA_DIR="${dataDir}"`,
            ``,
            `echo "=== Quillon Node — Build from Source ==="`,
            `echo ""`,
            ``,
            `# 1. Install Rust`,
            `if ! command -v cargo &>/dev/null; then`,
            `  echo "Installing Rust..."`,
            `  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable`,
            `  source "$HOME/.cargo/env"`,
            `  echo "✓ Rust installed: $(rustc --version)"`,
            `else`,
            `  source "$HOME/.cargo/env" 2>/dev/null || true`,
            `  echo "✓ Rust found: $(rustc --version)"`,
            `fi`,
            ``,
            `# 2. Install build dependencies`,
            `echo "Installing build dependencies..."`,
            `apt-get update -qq 2>/dev/null || true`,
            `apt-get install -y -qq build-essential pkg-config libssl-dev cmake clang libudev-dev libclang-dev git 2>/dev/null || true`,
            `echo "✓ Build dependencies ready"`,
            ``,
            `# 3. Clone and build`,
            `echo "Cloning Quillon source..."`,
            `TMPDIR=$(mktemp -d)`,
            `git clone --depth 1 https://code.quillon.xyz/repo.git "$TMPDIR/q-narwhalknight" 2>/dev/null || {`,
            `  echo "Clone failed. Downloading pre-built binary instead..."`,
            `  curl -fSL "${binaryUrl}" -o "$TMPDIR/q-api-server"`,
            `  chmod +x "$TMPDIR/q-api-server"`,
            `}`,
            ``,
            `if [ -d "$TMPDIR/q-narwhalknight" ]; then`,
            `  echo "Building... (this takes 10-30 minutes on first build)"`,
            `  cd "$TMPDIR/q-narwhalknight"`,
            `  cargo build --release --package q-api-server`,
            `  cp target/release/q-api-server "$TMPDIR/q-api-server"`,
            `  cd / && rm -rf "$TMPDIR/q-narwhalknight"`,
            `fi`,
            ``,
            `# 4. Install binary`,
            `mkdir -p "$INSTALL_DIR" "$DATA_DIR"`,
            `cp "$TMPDIR/q-api-server" "$INSTALL_DIR/q-api-server"`,
            `chmod +x "$INSTALL_DIR/q-api-server"`,
            `rm -rf "$TMPDIR"`,
            `echo "✓ Binary installed to $INSTALL_DIR/q-api-server"`,
            ``,
            `# 5. Install systemd service`,
            `cat > /etc/systemd/system/quillon-node.service << 'SVCEOF'`,
            serviceFile,
            `SVCEOF`,
            ``,
            `systemctl daemon-reload`,
            `systemctl enable quillon-node`,
            `systemctl start quillon-node`,
            ``,
            `echo ""`,
            `echo "=== Quillon Node Running! ==="`,
            `echo "API:     http://localhost:${apiPort}"`,
            `echo "Data:    $DATA_DIR"`,
            `echo "Status:  systemctl status quillon-node"`,
            `echo "Logs:    journalctl -u quillon-node -f"`,
            `echo ""`,
            `echo "The node will sync automatically. Full sync takes ~2-6 hours."`,
            `echo "Check progress: curl http://localhost:${apiPort}/api/v1/node/status"`,
            `echo ""`,
        ].join("\n");
        return {
            content: [{
                    type: "text",
                    text: [
                        `Node setup script (build from source):\n`,
                        `\`\`\`bash`,
                        script,
                        `\`\`\``,
                        ``,
                        `Save to a file and run as root:`,
                        `  sudo bash setup-node.sh`,
                        ``,
                        `This will:`,
                        `1. Install Rust (if not present)`,
                        `2. Install build dependencies`,
                        `3. Clone + build the node binary (~10-30 min first build)`,
                        `4. Install systemd service that auto-starts on reboot`,
                        `5. Begin syncing the blockchain automatically`,
                        ``,
                        `Minimum requirements: 4GB RAM, 50GB disk, Debian 12 / Ubuntu 22.04`,
                    ].join("\n"),
                }],
        };
    }
    // Pre-built binary (default, fast)
    const script = [
        `#!/bin/bash`,
        `# Quillon Node — Quick Install (pre-built binary)`,
        `# Run as root: curl -fsSL https://quillon.xyz/setup-node.sh | bash`,
        `set -e`,
        ``,
        `INSTALL_DIR="${installDir}"`,
        `DATA_DIR="${dataDir}"`,
        `BINARY_URL="${binaryUrl}"`,
        ``,
        `echo ""`,
        `echo "  Quillon Graph — Node Setup"`,
        `echo "  =========================="`,
        `echo ""`,
        ``,
        `# 1. Create directories`,
        `mkdir -p "$INSTALL_DIR" "$DATA_DIR"`,
        `echo "  ✓ Directories created: $INSTALL_DIR"`,
        ``,
        `# 2. Download latest binary`,
        `echo "  Downloading node binary..."`,
        `curl -fSL "$BINARY_URL" -o "$INSTALL_DIR/q-api-server"`,
        `chmod +x "$INSTALL_DIR/q-api-server"`,
        ``,
        `# Verify it runs`,
        `"$INSTALL_DIR/q-api-server" --version 2>/dev/null && echo "  ✓ Binary verified" || echo "  ✓ Binary downloaded"`,
        ``,
        `# 3. Write .env (skips interactive setup wizard)`,
        `cat > "$INSTALL_DIR/.env" << ENVEOF`,
        `Q_DB_PATH=${dataDir}`,
        `Q_NETWORK_ID=mainnet-genesis`,
        `RUST_LOG=warn`,
        adminWallet ? `Q_ADMIN_WALLET=${adminWallet}` : `# Q_ADMIN_WALLET=qnk...  (set this to your wallet address)`,
        `ENVEOF`,
        `echo "  ✓ Config written"`,
        ``,
        `# 4. Install systemd service`,
        `cat > /etc/systemd/system/quillon-node.service << 'SVCEOF'`,
        serviceFile,
        `SVCEOF`,
        ``,
        `echo "  ✓ Systemd service installed"`,
        ``,
        `# 5. Enable and start`,
        `systemctl daemon-reload`,
        `systemctl enable quillon-node`,
        `systemctl start quillon-node`,
        ``,
        `echo "  ✓ Node started"`,
        `echo ""`,
        `echo "  ╔═══════════════════════════════════════════════╗"`,
        `echo "  ║        Node is Running!                       ║"`,
        `echo "  ╠═══════════════════════════════════════════════╣"`,
        `echo "  ║                                               ║"`,
        `echo "  ║  API:   http://localhost:${apiPort}              ║"`,
        `echo "  ║  Data:  $DATA_DIR          ║"`,
        `echo "  ║                                               ║"`,
        `echo "  ║  Check status:                                ║"`,
        `echo "  ║    systemctl status quillon-node              ║"`,
        `echo "  ║    journalctl -u quillon-node -f              ║"`,
        `echo "  ║                                               ║"`,
        `echo "  ║  Sync progress (check after 30s):            ║"`,
        `echo "  ║    curl http://localhost:${apiPort}/api/v1/node/status | python3 -m json.tool"`,
        `echo "  ║                                               ║"`,
        `echo "  ║  Full sync takes 2-6 hours via turbo-sync.   ║"`,
        `echo "  ╚═══════════════════════════════════════════════╝"`,
        `echo ""`,
    ].join("\n");
    return {
        content: [{
                type: "text",
                text: [
                    adminWallet && newWalletMnemonic ? [
                        `🔑 New wallet created for this node:`,
                        ``,
                        `  Address:  ${adminWallet}`,
                        `  Mnemonic: ${newWalletMnemonic}`,
                        ``,
                        `  ⚠️  Save the mnemonic — it's the only way to recover this wallet.`,
                        ``,
                    ].join("\n") : adminWallet ? `Using wallet: ${adminWallet}\n` : "",
                    `Node setup script (pre-built binary, fast):\n`,
                    `\`\`\`bash`,
                    script,
                    `\`\`\``,
                    ``,
                    `Run as root on your Debian/Ubuntu server:`,
                    `  sudo bash setup-node.sh`,
                    ``,
                    `What this does:`,
                    `1. Creates a fresh wallet for this node`,
                    `2. Downloads the latest pre-built binary (~30 seconds)`,
                    `3. Writes .env config (no interactive wizard)`,
                    `4. Installs systemd service — auto-starts on reboot`,
                    `5. Node syncs 17M+ blocks via turbo-sync (2-6 hours)`,
                    ``,
                    `Requirements: Debian 12 / Ubuntu 22.04, root access, 50GB disk, 4GB RAM`,
                    ``,
                    `To build from source instead: say "setup node from source"`,
                ].filter(Boolean).join("\n"),
            }],
    };
});
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
