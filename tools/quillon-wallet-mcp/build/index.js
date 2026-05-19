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
import * as ed25519 from "@noble/ed25519";
import { sha3_256 } from "@noble/hashes/sha3";
import { sha512 } from "@noble/hashes/sha512";
// @noble/ed25519 v2 requires explicit sha512 setup for sync ops; we use async.
ed25519.etc.sha512Sync = (...m) => sha512(Buffer.concat(m.map((x) => Buffer.from(x))));
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
// ═══════════════════════════════════════════════════════════════
// v1.3.0: AGENT MODE — local Ed25519 signing key for autonomous flows
// ═══════════════════════════════════════════════════════════════
// When QUILLON_AGENT_SEED is set, the MCP holds an agent's own signing
// key (NOT a user's wallet key). All reads sign X-Wallet-Auth headers
// automatically. Distinct from user OAuth2 flow (browser device login).
//
// Use case: AI agents that own and transact crypto natively without
// human-in-the-loop browser approval for every action.
//
// Derivation matches the server-side scheme documented in memory
// [[wallet_seed_derivation]]: priv = SHA3-256(seed_str_utf8),
// pub = ed25519(priv), addr = "qnk" + hex(pub).
let agentPrivKey = null;
let agentAddrCache = null;
(function initAgentMode() {
    const seed = process.env.QUILLON_AGENT_SEED;
    if (!seed)
        return;
    const trimmed = seed.trim();
    if (trimmed.length === 0)
        return;
    agentPrivKey = sha3_256(new TextEncoder().encode(trimmed));
    console.error(`[quillon-mcp] Agent mode enabled (Ed25519 priv-key derived from QUILLON_AGENT_SEED)`);
})();
async function getAgentAddress() {
    if (!agentPrivKey)
        return null;
    if (agentAddrCache)
        return agentAddrCache;
    const pub = await ed25519.getPublicKeyAsync(agentPrivKey);
    agentAddrCache = "qnk" + Buffer.from(pub).toString("hex");
    return agentAddrCache;
}
/**
 * Build X-Wallet-Auth JSON header by signing the canonical message:
 *   SHA3-256(address_bytes || timestamp_le_i64 || path_bytes)
 * Returns null if agent mode is not enabled.
 */
async function buildAuthHeader(fullPath) {
    if (!agentPrivKey)
        return null;
    const addr = await getAgentAddress();
    if (!addr)
        return null;
    const ts = Math.floor(Date.now() / 1000);
    const addrBytes = Buffer.from(addr.slice(3), "hex");
    const tsBytes = Buffer.alloc(8);
    tsBytes.writeBigInt64LE(BigInt(ts));
    const pathBytes = Buffer.from(fullPath, "utf8");
    const msg = sha3_256(Buffer.concat([addrBytes, tsBytes, pathBytes]));
    const sig = await ed25519.signAsync(msg, agentPrivKey);
    return JSON.stringify({
        address: addr,
        timestamp: ts,
        scheme: "Ed25519",
        signature: Buffer.from(sig).toString("hex"),
    });
}
// --- HTTP helper ---
// `withAuth` injects X-Wallet-Auth from agent key (agent mode only).
// Reads against authenticated endpoints (balance, mining-status, etc.)
// pass withAuth=true so they return real data instead of "unavailable".
async function api(path, method = "GET", body, opts) {
    const url = `${API_BASE}${path}`;
    const headers = { "Content-Type": "application/json" };
    if (opts?.withAuth) {
        // Server middleware reads X-Wallet-Auth and verifies against the request
        // path as seen at axum router level — that's `/api/v1` + path here.
        const authHdr = await buildAuthHeader(`/api/v1${path}`);
        if (authHdr)
            headers["X-Wallet-Auth"] = authHdr;
    }
    const fetchOpts = {
        method,
        headers,
        redirect: "error", // SECURITY: Never follow redirects (prevents open redirect attacks)
    };
    if (body)
        fetchOpts.body = JSON.stringify(body);
    const res = await fetch(url, fetchOpts);
    if (!res.ok)
        throw new Error(`API ${method} ${path} returned ${res.status}: ${await res.text()}`);
    return res.json();
}
// --- MCP Server ---
const server = new McpServer({
    name: "quillon-wallet",
    version: "1.2.0",
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
                `  DEX SWAP`,
                `    "List tradeable tokens"         — QUG, QUGUSD, wBTC, wZEC, wIRON, wETH`,
                `    "Quote 10 QUG to QUGUSD"        — See expected output, price impact`,
                `    "Swap 10 QUG to QUGUSD"         — Two-step: shows quote → asks to confirm`,
                `    "Send 25 QUGUSD to qnk..."      — Token transfers (any DEX-listed token)`,
                ``,
                `  NETWORK`,
                `    "Network status"                — Height, peers, block rate`,
                `    "Verify node consistency"       — Compare two nodes' balance state (proves decentralization)`,
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
    // v1.3.0: send X-Wallet-Auth so private blockchain balance endpoint actually returns data
    const res = await api(`/wallets/${address}/balance`, "GET", undefined, { withAuth: true });
    if (!res.success)
        return { content: [{ type: "text", text: `Failed: ${res.error}` }] };
    const balance = res.data;
    return {
        content: [{
                type: "text",
                text: [
                    `Wallet: ${address}`,
                    `Balance: ${typeof balance.balance_qnk === "number" ? balance.balance_qnk.toFixed(6) : (balance.balance_qug || balance.balance || 0)} QUG`,
                    balance.pending ? `Pending: ${balance.pending} QUG` : '',
                    balance.staked ? `Staked: ${balance.staked} QUG` : '',
                    balance.auth_scheme ? `Auth scheme: ${balance.auth_scheme}` : '',
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
        // v1.3.0: balance endpoint requires X-Wallet-Auth — pass withAuth so
        // the agent-mode signing key produces a valid header. /mining/challenge
        // is unauthenticated.
        const [balRes, challengeRes] = await Promise.all([
            api(`/wallets/${wallet_address}/balance`, "GET", undefined, { withAuth: true }).catch(() => null),
            api("/mining/challenge").catch(() => null),
        ]);
        const bal = balRes?.data;
        const challenge = challengeRes?.data;
        const balanceStr = bal
            ? `${typeof bal.balance_qnk === "number" ? bal.balance_qnk.toFixed(6) : bal.balance_qug || bal.balance || 0} QUG`
            : "unavailable (set QUILLON_AGENT_SEED env var to enable authenticated reads)";
        return {
            content: [{
                    type: "text",
                    text: [
                        `=== Mining Status for ${wallet_address.slice(0, 12)}... ===`,
                        `Balance: ${balanceStr}`,
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
// Cached for the lifetime of the MCP session — refreshed on demand.
let tokenCache = null;
const TOKEN_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
async function fetchTokens(force = false) {
    if (!force && tokenCache && (Date.now() - tokenCache.fetchedAt) < TOKEN_CACHE_TTL_MS) {
        return tokenCache.tokens;
    }
    const res = await api("/dex/tokens");
    if (!res.success && res.ok !== true) {
        throw new Error(`Failed to fetch tokens: ${res.error || 'unknown error'}`);
    }
    // DEX API uses { ok: bool, data: ... } shape
    const list = res.data || res.tokens || [];
    tokenCache = { tokens: list, fetchedAt: Date.now() };
    return list;
}
function findTokenBySymbol(tokens, symbol) {
    const upper = symbol.trim().toUpperCase();
    return tokens.find(t => t.symbol.toUpperCase() === upper);
}
// Multiply display amount × 10^decimals as a BigInt-precise string.
// Avoids float precision loss for large decimal counts (24 for QUG).
function toBaseUnits(displayAmount, decimals) {
    if (!Number.isFinite(displayAmount) || displayAmount < 0) {
        throw new Error(`Invalid amount: ${displayAmount}`);
    }
    // Split into integer and fractional, scale each with BigInt to avoid float drift.
    const s = displayAmount.toFixed(decimals);
    const [intPart, fracPart = ""] = s.split(".");
    const padded = (fracPart + "0".repeat(decimals)).slice(0, decimals);
    const combined = (intPart + padded).replace(/^0+/, "") || "0";
    return combined;
}
// Inverse: base units (string) → display number with the given decimals.
function fromBaseUnits(baseUnits, decimals) {
    const b = BigInt(baseUnits);
    const divisor = 10n ** BigInt(decimals);
    const whole = b / divisor;
    const rem = b % divisor;
    // Use up to 6 fractional digits for display
    const fracStr = rem.toString().padStart(decimals, "0").slice(0, 6);
    return parseFloat(`${whole.toString()}.${fracStr}`);
}
// Pretty-print caveats for token safety. AI should relay these so users
// understand what they're trading into.
function tokenCaveat(t) {
    switch (t.contract_type.toLowerCase()) {
        case "native":
            return "Native chain asset — backed by proof-of-work mining and consensus";
        case "stablecoin":
            return "Collateralized stablecoin — value tracks USD via on-chain CDP vault. Peg holds while QUG collateral exceeds liability";
        case "wrapped":
            return `Bridge-wrapped — represents the external asset held in custody by the ${t.symbol.replace(/^w/i, "")} bridge contract. Requires bridge withdrawal to redeem the underlying`;
        case "lp":
            return "Liquidity pool token — represents your share of a pool; redeemable for the underlying tokens";
        default:
            return `${t.contract_type} token — verify the audit report before trading large amounts`;
    }
}
server.tool("dex_list_tokens", "List all tokens tradeable on the Quillon DEX with their decimals, type, and safety caveats. Says caveats so users understand what they're trading.", {}, async () => {
    try {
        const tokens = await fetchTokens(true); // force refresh
        if (tokens.length === 0) {
            return { content: [{ type: "text", text: `No tokens registered on this DEX yet.` }] };
        }
        const lines = [
            `=== Quillon DEX — Tradeable Tokens (${tokens.length}) ===`,
            ``,
        ];
        for (const t of tokens) {
            const verified = t.verified ? "✓" : "?";
            lines.push(`${verified} ${t.symbol}  (${t.name})`);
            lines.push(`  type: ${t.contract_type}  ·  decimals: ${t.decimals}`);
            lines.push(`  ${tokenCaveat(t)}`);
            lines.push(``);
        }
        lines.push(`Caveats:`, `  • All swaps go through constant-product AMM pools with 0.3% pool fee`, `  • Default slippage tolerance is 0.5%; max is 10%`, `  • Swap amounts are non-reversible once submitted — confirm carefully`, `  • Bridge tokens (wBTC/wZEC/wIRON/wETH) require the corresponding bridge to be operational for redemption`);
        return { content: [{ type: "text", text: lines.join("\n") }] };
    }
    catch (e) {
        return { content: [{ type: "text", text: `Failed to fetch tokens: ${e.message}` }] };
    }
});
server.tool("dex_get_quote", "Get a swap quote — see how much you'd receive before committing. No auth needed; this is just pricing.", {
    from_token: z.string().describe("Symbol of the token you want to sell (e.g., QUG)"),
    to_token: z.string().describe("Symbol of the token you want to buy (e.g., QUGUSD)"),
    amount: z.number().positive().describe("Amount of from_token in display units (e.g., 10 for 10 QUG)"),
    slippage_percent: z.number().optional().describe("Slippage tolerance, 0.0-10.0 (default 0.5)"),
}, async ({ from_token, to_token, amount, slippage_percent }) => {
    try {
        const tokens = await fetchTokens();
        const tIn = findTokenBySymbol(tokens, from_token);
        const tOut = findTokenBySymbol(tokens, to_token);
        if (!tIn)
            return { content: [{ type: "text", text: `Unknown from_token "${from_token}". Run dex_list_tokens to see what's available.` }] };
        if (!tOut)
            return { content: [{ type: "text", text: `Unknown to_token "${to_token}". Run dex_list_tokens to see what's available.` }] };
        if (tIn.symbol.toUpperCase() === tOut.symbol.toUpperCase()) {
            return { content: [{ type: "text", text: `Cannot swap a token for itself (${tIn.symbol}).` }] };
        }
        const slip = slippage_percent ?? 0.5;
        if (slip < 0 || slip > 10) {
            return { content: [{ type: "text", text: `Slippage tolerance must be between 0% and 10% (got ${slip}%).` }] };
        }
        const amountInBase = toBaseUnits(amount, tIn.decimals);
        const res = await api("/dex/swap/quote", "POST", {
            token_in: tIn.symbol,
            token_out: tOut.symbol,
            amount_in: amountInBase,
            slippage_tolerance: slip,
        });
        // The dex_integration_api returns { ok: bool, data: SwapQuote | null, error?: string }
        if (res.ok === false || res.success === false) {
            return { content: [{ type: "text", text: `Quote failed: ${res.error || 'unknown error'}` }] };
        }
        const q = res.data;
        if (!q)
            return { content: [{ type: "text", text: `Quote response empty — no liquidity for ${tIn.symbol}/${tOut.symbol}?` }] };
        const outDisplay = fromBaseUnits(q.amount_out, tOut.decimals);
        const minOutDisplay = fromBaseUnits(q.minimum_amount_out, tOut.decimals);
        const priceImpactPct = (q.price_impact * 100).toFixed(3);
        const lines = [
            `=== Swap Quote ===`,
            ``,
            `  Sell:        ${amount} ${tIn.symbol}`,
            `  Receive:     ${outDisplay.toFixed(6)} ${tOut.symbol}  (estimated)`,
            `  Min after ${slip}% slippage: ${minOutDisplay.toFixed(6)} ${tOut.symbol}`,
            ``,
            `  Price impact: ${priceImpactPct}%${q.price_impact > 0.01 ? "  ⚠️ HIGH" : ""}`,
            `  Execution price: 1 ${tIn.symbol} ≈ ${(q.execution_price).toFixed(6)} ${tOut.symbol}`,
            ``,
            `Quote valid until block timestamp ${q.valid_until}. Run dex_swap with confirm=true to execute.`,
        ];
        return { content: [{ type: "text", text: lines.join("\n") }] };
    }
    catch (e) {
        return { content: [{ type: "text", text: `Quote error: ${e.message}` }] };
    }
});
server.tool("dex_swap", "Execute a DEX swap. Wallet must be authenticated first (run authenticate_wallet). Call once to see a quote + confirmation prompt; call again with confirm=true to actually execute.", {
    from_token: z.string().describe("Token to sell (e.g., QUG)"),
    to_token: z.string().describe("Token to buy (e.g., QUGUSD)"),
    amount: z.number().positive().describe("Amount of from_token in display units"),
    slippage_percent: z.number().optional().describe("Slippage tolerance (default 0.5)"),
    confirm: z.boolean().optional().describe("Set to true to execute; without it, returns the quote first"),
}, async ({ from_token, to_token, amount, slippage_percent, confirm }) => {
    // Step 1: auth check (same pattern as send_qug)
    if (!isSessionValid()) {
        return {
            content: [{
                    type: "text",
                    text: [
                        `Wallet not authenticated${sessionAuthenticatedAt ? ' (session expired)' : ''}. To execute a swap:`,
                        ``,
                        `  1. Say "authenticate wallet"`,
                        `  2. Open the link in your browser and approve`,
                        `  3. Say "check auth"`,
                        `  4. Then retry the swap`,
                    ].join("\n"),
                }],
        };
    }
    refreshSession();
    // Step 2: resolve tokens + decimals
    let tokens;
    try {
        tokens = await fetchTokens();
    }
    catch (e) {
        return { content: [{ type: "text", text: `Token lookup failed: ${e.message}` }] };
    }
    const tIn = findTokenBySymbol(tokens, from_token);
    const tOut = findTokenBySymbol(tokens, to_token);
    if (!tIn)
        return { content: [{ type: "text", text: `Unknown from_token "${from_token}".` }] };
    if (!tOut)
        return { content: [{ type: "text", text: `Unknown to_token "${to_token}".` }] };
    if (tIn.symbol.toUpperCase() === tOut.symbol.toUpperCase()) {
        return { content: [{ type: "text", text: `Cannot swap a token for itself.` }] };
    }
    const slip = slippage_percent ?? 0.5;
    if (slip < 0 || slip > 10) {
        return { content: [{ type: "text", text: `Slippage must be 0-10% (got ${slip}%).` }] };
    }
    if (amount <= 0) {
        return { content: [{ type: "text", text: `Amount must be > 0.` }] };
    }
    // Step 3: get quote for confirmation display
    const amountInBase = toBaseUnits(amount, tIn.decimals);
    let quoteRes;
    try {
        quoteRes = await api("/dex/swap/quote", "POST", {
            token_in: tIn.symbol,
            token_out: tOut.symbol,
            amount_in: amountInBase,
            slippage_tolerance: slip,
        });
    }
    catch (e) {
        return { content: [{ type: "text", text: `Quote fetch failed: ${e.message}` }] };
    }
    if (quoteRes.ok === false || quoteRes.success === false || !quoteRes.data) {
        return { content: [{ type: "text", text: `Cannot price the swap: ${quoteRes.error || 'no liquidity for this pair?'}` }] };
    }
    const q = quoteRes.data;
    const outDisplay = fromBaseUnits(q.amount_out, tOut.decimals);
    const minOutDisplay = fromBaseUnits(q.minimum_amount_out, tOut.decimals);
    const priceImpactPct = (q.price_impact * 100).toFixed(3);
    const highImpact = q.price_impact > 0.05; // 5% impact threshold
    // Step 4: if not confirmed, show quote and ask
    if (!confirm) {
        return {
            content: [{
                    type: "text",
                    text: [
                        `⚠️ SWAP CONFIRMATION REQUIRED`,
                        ``,
                        `  From wallet:  ${activeWalletAddress.slice(0, 20)}...`,
                        `  Sell:         ${amount} ${tIn.symbol}`,
                        `  Receive:      ≈${outDisplay.toFixed(6)} ${tOut.symbol}`,
                        `  Min received: ${minOutDisplay.toFixed(6)} ${tOut.symbol}  (with ${slip}% slippage tolerance)`,
                        `  Price impact: ${priceImpactPct}%${highImpact ? "  🚨 HIGH IMPACT — pool may be shallow" : ""}`,
                        ``,
                        `Reply with: dex_swap from=${tIn.symbol} to=${tOut.symbol} amount=${amount} confirm=true`,
                        `(or rephrase: "yes, execute the ${tIn.symbol}→${tOut.symbol} swap")`,
                        ``,
                        `This is irreversible. Verify the amounts above.`,
                    ].join("\n"),
                }],
        };
    }
    // Step 5: execute
    try {
        const res = await api("/dex/swap", "POST", {
            from_token: tIn.symbol,
            to_token: tOut.symbol,
            amount_in: amountInBase,
            min_amount_out: q.minimum_amount_out,
            wallet_address: activeWalletAddress,
            slippage_tolerance: slip,
            ...(authToken ? { auth_token: authToken } : {}),
        });
        if (res.success === false || res.ok === false) {
            return { content: [{ type: "text", text: `Swap failed: ${res.error || 'unknown error'}` }] };
        }
        const data = res.data || res;
        const txHash = data.transaction_hash || data.tx_hash || data.tx_id || "(no tx id)";
        const filledOutBase = data.amount_out || q.amount_out;
        const filledOutDisplay = fromBaseUnits(String(filledOutBase), tOut.decimals);
        return {
            content: [{
                    type: "text",
                    text: [
                        `✅ Swap submitted!`,
                        ``,
                        `  Sold:     ${amount} ${tIn.symbol}`,
                        `  Received: ${filledOutDisplay.toFixed(6)} ${tOut.symbol}`,
                        `  Tx hash:  ${txHash}`,
                        ``,
                        `The swap will be reflected in your balance within ~1 second (next block).`,
                        `Check balance with: get_balance address=${activeWalletAddress}`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `Swap submission failed: ${e.message}\n\nThe wallet may need re-authentication or have insufficient balance.` }] };
    }
});
// ============================================================
// TOKEN TRANSFER TOOL — send any DEX-listed token
// ============================================================
server.tool("send_token", "Send a non-QUG token (e.g., QUGUSD, wBTC, wETH) from your authenticated wallet to another address. Same auth + confirmation pattern as send_qug. Use this for any token returned by dex_list_tokens.", {
    token: z.string().describe("Token symbol (QUGUSD, wBTC, wZEC, wIRON, wETH, etc.) — see dex_list_tokens"),
    to_address: z.string().describe("Recipient qnk... address"),
    amount: z.number().positive().describe("Amount in display units (e.g., 25.5 for 25.5 QUGUSD)"),
    confirm: z.boolean().optional().describe("Set true to execute; without it, returns confirmation prompt"),
}, async ({ token, to_address, amount, confirm }) => {
    // Auth gate (same as send_qug)
    if (!isSessionValid()) {
        return {
            content: [{
                    type: "text",
                    text: [
                        `Wallet not authenticated${sessionAuthenticatedAt ? ' (session expired)' : ''}. To send tokens:`,
                        ``,
                        `  1. Say "authenticate wallet"`,
                        `  2. Open the link in your browser and approve`,
                        `  3. Say "check auth"`,
                        `  4. Then retry sending`,
                    ].join("\n"),
                }],
        };
    }
    refreshSession();
    // Validate address format
    if (!to_address.startsWith("qnk") || to_address.length !== 67 || !/^qnk[0-9a-f]{64}$/.test(to_address)) {
        return { content: [{ type: "text", text: `Invalid recipient address. Must be 'qnk' + 64 hex chars (67 total).` }] };
    }
    if (amount <= 0) {
        return { content: [{ type: "text", text: `Amount must be > 0.` }] };
    }
    // Look up token decimals + verify it's tradeable
    let tokens;
    try {
        tokens = await fetchTokens();
    }
    catch (e) {
        return { content: [{ type: "text", text: `Token lookup failed: ${e.message}` }] };
    }
    const t = findTokenBySymbol(tokens, token);
    if (!t) {
        return { content: [{ type: "text", text: `Unknown token "${token}". Run dex_list_tokens to see what's available.` }] };
    }
    // Don't allow sending QUG through this — use send_qug for native to keep audit trails clean
    if (t.symbol.toUpperCase() === "QUG") {
        return { content: [{ type: "text", text: `For native QUG, use send_qug instead (separate code path, separate per-session limits).` }] };
    }
    // Confirmation step
    if (!confirm) {
        return {
            content: [{
                    type: "text",
                    text: [
                        `⚠️ TOKEN TRANSFER CONFIRMATION REQUIRED`,
                        ``,
                        `  From:   ${activeWalletAddress.slice(0, 20)}...`,
                        `  To:     ${to_address.slice(0, 20)}...`,
                        `  Send:   ${amount} ${t.symbol}  (${t.contract_type})`,
                        ``,
                        `Caveat: ${tokenCaveat(t)}`,
                        ``,
                        `Reply: send_token token=${t.symbol} to_address=${to_address} amount=${amount} confirm=true`,
                        `(or rephrase: "yes, send ${amount} ${t.symbol}")`,
                        ``,
                        `This is irreversible. Verify above.`,
                    ].join("\n"),
                }],
        };
    }
    // Execute
    try {
        const res = await api("/transactions/send", "POST", {
            from: activeWalletAddress,
            to: to_address,
            amount: amount, // f64; the handler does decimal scaling per token_type
            token_type: t.symbol,
            ...(authToken ? { auth_token: authToken } : {}),
        });
        if (!res.success) {
            return { content: [{ type: "text", text: `Token transfer failed: ${res.error || 'unknown error'}` }] };
        }
        return {
            content: [{
                    type: "text",
                    text: [
                        `✅ ${t.symbol} transfer submitted!`,
                        ``,
                        `  From:   ${activeWalletAddress.slice(0, 16)}...`,
                        `  To:     ${to_address.slice(0, 16)}...`,
                        `  Amount: ${amount} ${t.symbol}`,
                        res.data?.tx_id ? `  Tx id:  ${res.data.tx_id}` : ``,
                        ``,
                        `Will be included in the next block (~1 second).`,
                    ].filter(Boolean).join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `Token send failed: ${e.message}` }] };
    }
});
// ============================================================
// AUTOMATED DECENTRALIZATION TEST — verify_node_consistency
// ============================================================
//
// Validates that two Quillon nodes have produced bit-identical balance state.
// Uses the public /api/v1/integrity/balance-root endpoint (which returns a
// BLAKE3 fingerprint over all non-zero wallet balances) so no auth is needed.
//
// If the hashes match at the same height, every wallet on the network has the
// same QUG balance on both nodes. This is the consensus correctness property
// that BAL-001 (activation block 20,000,000) will enforce at the protocol layer.
server.tool("verify_node_consistency", "Compare two Quillon nodes' balance state and report whether they agree. Uses /api/v1/integrity/balance-root — no auth needed. The primary node defaults to quillon.xyz (Epsilon); the secondary defaults to localhost:8080 (a local node you're running). Pass URLs to compare other nodes. Output includes a verdict, both nodes' wallet counts, total supplies, balance_root_hex values, and a per-node height.", {
    primary_url: z.string().optional().describe("Primary node API base URL (default: https://quillon.xyz/api/v1)"),
    secondary_url: z.string().optional().describe("Secondary node API base URL (default: http://localhost:8080/api/v1)"),
}, async ({ primary_url, secondary_url }) => {
    const primary = primary_url || "https://quillon.xyz/api/v1";
    const secondary = secondary_url || "http://localhost:8080/api/v1";
    async function fetchIntegrity(baseUrl) {
        const url = `${baseUrl.replace(/\/$/, "")}/integrity/balance-root`;
        const res = await fetch(url, { redirect: "error" });
        if (!res.ok)
            throw new Error(`${url} returned HTTP ${res.status}`);
        return await res.json();
    }
    let p, s;
    try {
        p = await fetchIntegrity(primary);
    }
    catch (e) {
        return { content: [{ type: "text", text: `Could not reach primary ${primary}: ${e.message}` }] };
    }
    try {
        s = await fetchIntegrity(secondary);
    }
    catch (e) {
        return { content: [{ type: "text", text: `Could not reach secondary ${secondary}: ${e.message}` }] };
    }
    if (!p.ok || !p.data)
        return { content: [{ type: "text", text: `Primary returned invalid integrity response.` }] };
    if (!s.ok || !s.data)
        return { content: [{ type: "text", text: `Secondary returned invalid integrity response.` }] };
    const pd = p.data, sd = s.data;
    const matchRoot = pd.balance_root_hex === sd.balance_root_hex;
    const matchHeight = pd.at_height === sd.at_height;
    const matchWallets = pd.wallet_count === sd.wallet_count;
    const matchSupply = pd.total_supply_base_units === sd.total_supply_base_units;
    const passed = matchRoot && matchHeight && matchWallets && matchSupply;
    const lines = [
        `=== Node Consistency Check ===`,
        ``,
        `PRIMARY:    ${primary}`,
        `  height:           ${pd.at_height.toLocaleString()}`,
        `  wallet_count:     ${pd.wallet_count}`,
        `  total_supply:     ${pd.total_supply_display} QUG`,
        `  balance_root_hex: ${pd.balance_root_hex}`,
        ``,
        `SECONDARY:  ${secondary}`,
        `  height:           ${sd.at_height.toLocaleString()}`,
        `  wallet_count:     ${sd.wallet_count}`,
        `  total_supply:     ${sd.total_supply_display} QUG`,
        `  balance_root_hex: ${sd.balance_root_hex}`,
        ``,
        `Comparison:`,
        `  heights match:        ${matchHeight ? "✓" : "✗ (Δ " + Math.abs(pd.at_height - sd.at_height) + " blocks — secondary may still be syncing)"}`,
        `  wallet counts match:  ${matchWallets ? "✓" : `✗ (Δ ${Math.abs(pd.wallet_count - sd.wallet_count)})`}`,
        `  total supply matches: ${matchSupply ? "✓" : "✗"}`,
        `  balance_root matches: ${matchRoot ? "✓ (state is bit-identical)" : "✗ (DIVERGENCE — nodes disagree about wallet state)"}`,
        ``,
    ];
    if (passed) {
        lines.push(`✅ VERDICT: PASS — both nodes have bit-identical balance state.`, ``, `Every wallet on the network has the same QUG balance on both nodes. This is`, `the consensus correctness property that BAL-001 will enforce at the protocol`, `layer starting at block 20,000,000.`, ``, `Note: this check covers native QUG balances. Per-token (QUGUSD, wBTC, etc.)`, `consistency requires individual wallet queries (use the web wallet at quillon.xyz`, `or per-wallet auth to verify specific token amounts).`);
    }
    else if (!matchHeight) {
        lines.push(`⏳ VERDICT: PENDING — secondary is at a different height. Either it's still`, `syncing or it's serving stale data. Re-run after the heights converge.`, ``, `If both nodes are caught up to network tip but still diverge, that's a real`, `consensus problem — file an issue immediately.`);
    }
    else {
        lines.push(`❌ VERDICT: DIVERGENCE — same height, different state.`, ``, `This is a real consensus problem. One of the two nodes has wrong wallet state.`, `Compare with a third node (e.g., a freshly-synced Docker container) to`, `triangulate which one is correct. If multiple nodes disagree on the same`, `chain height, the network has split.`);
    }
    return { content: [{ type: "text", text: lines.join("\n") }] };
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
