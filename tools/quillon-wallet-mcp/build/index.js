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
// v2.0.0: X-Wallet-Auth seed-derived signing for production v10.9.55+ endpoints.
// See ./wallet_auth.ts for the signing algorithm (matches raw_balance.mjs exactly).
import { loadSeed, deriveKeys, signXWalletAuth, SeedNotFoundError, SignatureError, } from "./wallet_auth.js";
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
// --- HTTP helpers ---
// API_BASE includes /api/v1 (e.g., "https://quillon.xyz/api/v1"); when we sign
// an X-Wallet-Auth header the server expects the FULL request path including
// the prefix, so we extract the path portion of API_BASE once and prepend it
// to the per-call suffix.
const API_PATH_PREFIX = (() => {
    try {
        return new URL(API_BASE).pathname.replace(/\/$/, "");
    }
    catch {
        return "/api/v1";
    }
})();
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
// v2.0.0: signed variant for endpoints that require X-Wallet-Auth.
// On 401 we re-load the seed and retry ONCE (handles fresh-seed-after-launch).
// On final failure throws a SignatureError with diagnostic content the agent
// can act on (which seed source, which derived address, server reply).
async function apiSigned(path, method = "GET", body, opts) {
    const url = `${API_BASE}${path}`;
    const signPath = `${API_PATH_PREFIX}${path}`;
    const doRequest = async (attempt) => {
        const auth = signXWalletAuth(signPath, { seedArg: opts?.seed });
        const init = {
            method,
            headers: {
                "Content-Type": "application/json",
                "X-Wallet-Auth": auth.header,
            },
            redirect: "error",
        };
        if (body)
            init.body = JSON.stringify(body);
        const res = await fetch(url, init);
        // Stash the derived address + source so the error path can report them.
        res._qnk = {
            addr: auth.address,
            src: auth.source,
            attempt,
        };
        return res;
    };
    let res = await doRequest(1);
    if (res.status === 401) {
        res = await doRequest(2);
    }
    if (!res.ok) {
        const ctx = res._qnk;
        const bodyText = await res.text();
        throw new SignatureError(`Signed API ${method} ${path} returned ${res.status} (attempt ${ctx?.attempt ?? "?"}). ` +
            `Seed source: ${ctx?.src ?? "?"}. Derived address: ${ctx?.addr ?? "?"}. ` +
            `Server reply: ${bodyText}. ` +
            `Hints: confirm the address matches the wallet you intended; check clock drift (server tolerance ±5min); ` +
            `if mid-session seed change, verify the new seed file content.`, ctx?.addr);
    }
    return res.json();
}
// --- MCP Server ---
const server = new McpServer({
    name: "quillon-wallet",
    version: "2.1.0",
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
server.tool("get_balance", "Check the balance of any Quillon wallet address (qnk...). Production v10.9.55+ requires X-Wallet-Auth; the MCP signs automatically using the configured seed (file ~/.claude/quillon-agent-seed → QNK_SEED env, override per-call with `seed`). Returns balance in QUG.", {
    address: z.string().optional().describe("Wallet address starting with 'qnk'. If omitted, uses the address derived from the configured seed (your own wallet)."),
    seed: z.string().optional().describe("Optional 64-char hex seed to override the configured seed for this call only."),
}, async ({ address, seed }) => {
    try {
        // If no address given, derive from seed (signs as the same wallet).
        const targetAddress = address ?? (() => {
            const { seed: s } = loadSeed({ seedArg: seed });
            return deriveKeys(s).address;
        })();
        const res = await apiSigned(`/wallets/${targetAddress}/balance`, "GET", undefined, { seed });
        if (!res.success)
            return { content: [{ type: "text", text: `Failed: ${res.error}` }] };
        const balance = res.data;
        return {
            content: [{
                    type: "text",
                    text: [
                        `Wallet: ${targetAddress}`,
                        `Balance: ${balance.balance_qnk ?? balance.balance_qug ?? balance.balance ?? 0} QUG`,
                        balance.pending ? `Pending: ${balance.pending} QUG` : "",
                        balance.staked ? `Staked: ${balance.staked} QUG` : "",
                        `Auth: ${balance.auth_scheme ?? "Ed25519"} (${balance.privacy_mode ?? "authenticated"})`,
                    ].filter(Boolean).join("\n"),
                }],
        };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError) {
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        }
        if (e instanceof SignatureError) {
            return { content: [{ type: "text", text: `🔒 ${e.message}` }] };
        }
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
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
    // v10.10.0 fix: /api/v1/status nests current_height under `upgrades`, not top-level.
    // The old top-level read returned "unknown" silently for the past several versions.
    // Documented in AGENT.md §4 gotchas; fix here so MCP callers see the real chain height.
    const height = s.upgrades?.current_height ?? s.current_height;
    return {
        content: [{
                type: "text",
                text: [
                    `=== Quillon Network Status ===`,
                    `Height: ${height?.toLocaleString?.() ?? height ?? 'unknown'}`,
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
// v2.3.0: All DEX/AMM-side amounts (amount_in, amount_out, minimum_amount_out)
// are denominated in 24-decimal universal AMM units regardless of the token's
// native decimal count. See [[dex_token_bugs]] memory: "ALL AMM amounts in
// 24-decimal". Using a token's native decimal count when converting AMM
// amounts produced the May-21 dex_get_quote bug where 1000 QUGUSD→QUG
// reported a receive of 3.4 × 10^15 QUG (off by 10^16).
const AMM_DECIMALS = 24;
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
        // v2.3.0: AMM amounts are universally 24-decimal — see AMM_DECIMALS comment.
        const amountInBase = toBaseUnits(amount, AMM_DECIMALS);
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
        const outDisplay = fromBaseUnits(q.amount_out, AMM_DECIMALS);
        const minOutDisplay = fromBaseUnits(q.minimum_amount_out, AMM_DECIMALS);
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
server.tool("dex_swap", "Execute a DEX swap. v2.1.1: signs via the configured seed (file → QNK_SEED env); no browser-auth required. Call once with confirm omitted to see the quote; call again with confirm=true to execute.", {
    from_token: z.string().describe("Token to sell (e.g., QUG)"),
    to_token: z.string().describe("Token to buy (e.g., QUGUSD)"),
    amount: z.number().positive().describe("Amount of from_token in display units"),
    slippage_percent: z.number().optional().describe("Slippage tolerance (default 0.5)"),
    confirm: z.boolean().optional().describe("Set to true to execute; without it, returns the quote first"),
    seed: z.string().optional().describe("Optional seed override (otherwise: file → QNK_SEED env)"),
}, async ({ from_token, to_token, amount, slippage_percent, confirm, seed }) => {
    // v2.1.1: seed-derived auth — no browser flow.
    // Derive our qnk... address from the same seed apiSigned() will use, so
    // the wallet_address in the body matches the X-Wallet-Auth signer.
    let signerAddress;
    try {
        const { seed: rawSeed } = loadSeed({ seedArg: seed });
        signerAddress = deriveKeys(rawSeed).address;
    }
    catch (e) {
        return { content: [{ type: "text", text: `No wallet seed available: ${e.message}` }] };
    }
    // Resolve tokens + decimals
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
    // v2.3.0: AMM amounts are universally 24-decimal — see AMM_DECIMALS comment.
    const amountInBase = toBaseUnits(amount, AMM_DECIMALS);
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
    const outDisplay = fromBaseUnits(q.amount_out, AMM_DECIMALS);
    const minOutDisplay = fromBaseUnits(q.minimum_amount_out, AMM_DECIMALS);
    const priceImpactPct = (q.price_impact * 100).toFixed(3);
    const highImpact = q.price_impact > 0.05; // 5% impact threshold
    // If not confirmed, show quote and ask
    if (!confirm) {
        return {
            content: [{
                    type: "text",
                    text: [
                        `⚠️ SWAP CONFIRMATION REQUIRED`,
                        ``,
                        `  From wallet:  ${signerAddress.slice(0, 20)}...`,
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
    // Execute — seed-signed X-Wallet-Auth, no vault session.
    try {
        const res = await apiSigned("/dex/swap", "POST", {
            from_token: tIn.symbol,
            to_token: tOut.symbol,
            amount_in: amountInBase,
            min_amount_out: q.minimum_amount_out,
            wallet_address: signerAddress,
            slippage_tolerance: slip,
        }, { seed });
        if (res.success === false || res.ok === false) {
            return { content: [{ type: "text", text: `Swap failed: ${res.error || 'unknown error'}` }] };
        }
        const data = res.data || res;
        // v2.1.1: server returns `transaction_id` per memory entry
        // first_agentic_loop_closed.md (the 2026-05-17 finding). Try that
        // first; older variants kept for compatibility.
        const txHash = data.transaction_id || data.transaction_hash || data.tx_hash || data.tx_id || "(no tx id)";
        const filledOutBase = data.amount_out || q.amount_out;
        // v2.3.0: server returns amount_out in 24-decimal AMM base
        const filledOutDisplay = fromBaseUnits(String(filledOutBase), AMM_DECIMALS);
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
                        `Check balance with: get_balance address=${signerAddress}`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `Swap submission failed: ${e.message}\n\nVerify the seed file exists at ~/.claude/quillon-agent-seed (or QNK_SEED env), and that the wallet has sufficient balance for both the swap and gas.` }] };
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
// QSHARE-1 TOOLS — L3 autonomous treasury share
// Per docs/standards/qshare-treasury-protocol-spec.md
// Phase 2 scaffolding: tools work against the on-chain contract
// once the REST API exposes /api/v1/qshare/* endpoints.
// ============================================================
server.tool("qshare_nav", "Read the current NAV (net asset value) per QSHARE token in QUG units. NAV = total QUG-equivalent treasury / circulating QSHARE supply. Returns nav_per_qshare (raw u128 with decimals=24), total_treasury_qug_equivalent, circulating_qshare, and the block height at which NAV was computed. Use this to compare against market price (qshare_premium_ratio) for arbitrage decisions.", {
    api_url: z.string().optional().describe("Quillon API base URL (default: https://quillon.xyz/api/v1)"),
}, async ({ api_url }) => {
    const base = (api_url || "https://quillon.xyz/api/v1").replace(/\/$/, "");
    try {
        const res = await fetch(`${base}/qshare/state`, { redirect: "error" });
        if (!res.ok) {
            return { content: [{ type: "text", text: `qshare/state endpoint returned HTTP ${res.status}. The /api/v1/qshare/* endpoints are spec'd but not yet exposed by q-api-server. See docs/standards/qshare-treasury-protocol-spec.md §5 for the expected shape.` }] };
        }
        const data = await res.json();
        return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
    }
    catch (e) {
        return { content: [{ type: "text", text: `Network error: ${e.message}` }] };
    }
});
server.tool("qshare_premium_ratio", "Read the current QSHARE/QUG premium ratio (×1000 basis points). Premium ratio = market_price_twap / nav_per_qshare. Above 1500 means a mint trigger is available; below 950 means a buyback trigger is available; between is the neutral zone. Returns premium_ratio_bps, nav_per_qshare, market_price_twap, and the eligibility window (next_mint_eligible_at_height, next_buyback_eligible_at_height).", {
    api_url: z.string().optional().describe("Quillon API base URL (default: https://quillon.xyz/api/v1)"),
}, async ({ api_url }) => {
    const base = (api_url || "https://quillon.xyz/api/v1").replace(/\/$/, "");
    try {
        const res = await fetch(`${base}/qshare/premium`, { redirect: "error" });
        if (!res.ok) {
            return { content: [{ type: "text", text: `qshare/premium endpoint returned HTTP ${res.status}. The /api/v1/qshare/* endpoints are spec'd but not yet exposed by q-api-server. See docs/standards/qshare-treasury-protocol-spec.md §5.` }] };
        }
        const data = await res.json();
        const ratio_x1000 = data.premium_ratio_bps || 0;
        const human = (ratio_x1000 / 1000).toFixed(3);
        let zone = "neutral";
        if (ratio_x1000 >= 1500)
            zone = "MINT eligible (premium ≥ 1.5×)";
        else if (ratio_x1000 <= 950)
            zone = "BUYBACK eligible (discount ≤ 0.95×)";
        return { content: [{ type: "text", text: `Premium ratio: ${human}× (${ratio_x1000} bps) — zone: ${zone}\n\n${JSON.stringify(data, null, 2)}` }] };
    }
    catch (e) {
        return { content: [{ type: "text", text: `Network error: ${e.message}` }] };
    }
});
server.tool("qshare_mint", "Attempt to trigger an autonomous QSHARE mint. Permissionless — the contract gates by premium threshold (≥ 1.5×), cooldown (360 blocks default), and pool depth. If accepted, caller earns a bounty (~0.5% of accumulated QUG, capped at 1 QUG). Calls POST /api/v1/qshare/try_mint_signed (v10.10.7+). Seed defaults to file → env per usual; per-call `seed` overrides.", {
    seed: z.string().optional().describe("Override the configured seed for this call (otherwise: file → QNK_SEED env)"),
    dry_run: z.boolean().optional().describe("If true, describe what would happen without submitting"),
}, async ({ seed, dry_run }) => {
    try {
        if (dry_run) {
            const { seed: s, source } = loadSeed({ seedArg: seed });
            const { address } = deriveKeys(s);
            return {
                content: [{
                        type: "text",
                        text: `[DRY RUN] POST ${API_BASE}/qshare/try_mint_signed\n` +
                            `Seed source: ${source}\n` +
                            `Caller: ${address}\n` +
                            `Contract gates: premium ≥ 1.5×, cooldown ≥ 360 blocks, pool depth.`,
                    }],
            };
        }
        const res = await apiSigned("/qshare/try_mint_signed", "POST", {}, { seed });
        return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        if (e instanceof SignatureError)
            return { content: [{ type: "text", text: `🔒 ${e.message}` }] };
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
server.tool("qshare_buyback", "Attempt to trigger a QSHARE buyback. Permissionless — gates by discount ≤ 0.95×, cooldown ≥ 720 blocks, pool depth, AND accrued-yield availability (principal is never spent). Tighter pool cap than mint (0.1% vs 0.5%). Calls POST /api/v1/qshare/try_buyback_signed (v10.10.7+).", {
    seed: z.string().optional().describe("Override the configured seed for this call"),
    dry_run: z.boolean().optional().describe("If true, describe what would happen without submitting"),
}, async ({ seed, dry_run }) => {
    try {
        if (dry_run) {
            const { seed: s, source } = loadSeed({ seedArg: seed });
            const { address } = deriveKeys(s);
            return {
                content: [{
                        type: "text",
                        text: `[DRY RUN] POST ${API_BASE}/qshare/try_buyback_signed\n` +
                            `Seed source: ${source}\n` +
                            `Caller: ${address}\n` +
                            `Contract gates: discount ≤ 0.95×, cooldown ≥ 720 blocks, pool depth, accrued yield available.`,
                    }],
            };
        }
        const res = await apiSigned("/qshare/try_buyback_signed", "POST", {}, { seed });
        return { content: [{ type: "text", text: JSON.stringify(res, null, 2) }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        if (e instanceof SignatureError)
            return { content: [{ type: "text", text: `🔒 ${e.message}` }] };
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
// ============================================================
// v2.0.0 — DISCOVERY + AGENT-PANEL + ENGINE PULSE + BATCH SUBMIT
// ============================================================
server.tool("wallet_info", "One-shot discovery: address derived from the configured seed, seed source actually used, server endpoint + version, and current balance. Call this first in any new session — it confirms which wallet the MCP will sign as.", {
    seed: z.string().optional().describe("Optional seed override (otherwise: file → QNK_SEED env)"),
}, async ({ seed }) => {
    try {
        const { seed: s, source } = loadSeed({ seedArg: seed });
        const { address } = deriveKeys(s);
        // Pull status + engine pulse + balance in parallel.
        // engine_pulse.sync.current_height is the LIVE tip; status.upgrades.current_height
        // is the upgrade-evaluation height which lags by minutes-to-hours and was misleading
        // in early v2.0.0 (showed h=18,131,330 while real tip was h=18,224,847).
        const [status, pulse, balance] = await Promise.all([
            api("/status").catch(() => null),
            api("/engine/pulse").catch(() => null),
            apiSigned(`/wallets/${address}/balance`, "GET", undefined, { seed }).catch(() => null),
        ]);
        const lines = [
            `address:       ${address}`,
            `seed source:   ${source}`,
            `endpoint:      ${API_BASE}`,
        ];
        if (status?.data) {
            lines.push(`server status: ${status.data.status ?? "?"}`);
            lines.push(`network_id:    ${status.data.network_id ?? "?"}`);
        }
        else {
            lines.push(`server status: UNREACHABLE`);
        }
        if (pulse?.data?.sync?.current_height !== undefined) {
            lines.push(`tip height:    ${pulse.data.sync.current_height}`);
            if (pulse.data.version)
                lines.push(`server version: ${pulse.data.version}`);
        }
        else if (status?.data?.upgrades?.current_height !== undefined) {
            // Fallback if engine/pulse unavailable; this number is older.
            lines.push(`tip height:    ${status.data.upgrades.current_height} (upgrade-eval, may be stale)`);
        }
        if (balance?.data) {
            lines.push(`balance:       ${balance.data.balance_qnk ?? balance.data.balance_qug ?? "?"} QUG`);
            lines.push(`auth scheme:   ${balance.data.auth_scheme ?? "Ed25519"}`);
        }
        else {
            lines.push(`balance:       (failed to read — seed correct? server reachable?)`);
        }
        return { content: [{ type: "text", text: lines.join("\n") }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
server.tool("engine_pulse", "Live engine vitals: sync heights, mining counters, P2P bytes, mempool size, gap-to-tip. No auth needed. Useful as a first-look health check before deeper queries.", {}, async () => {
    try {
        const res = await api("/engine/pulse");
        return { content: [{ type: "text", text: JSON.stringify(res?.data ?? res, null, 2) }] };
    }
    catch (e) {
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
server.tool("agent_panel", "Fetch the agent activity panel for a wallet (v10.10.7+, route /api/v1/agent/panel/:addr). With mode=owner the server requires X-Wallet-Auth matching :addr; the MCP signs automatically using the configured seed. With mode=embed it's a public read.", {
    address: z.string().optional().describe("Wallet address; omit to use the address derived from the configured seed (only valid with mode=owner)"),
    mode: z.enum(["owner", "embed"]).optional().describe("'owner' (default) shows the full panel and requires signing; 'embed' is the public truncated view"),
    zone: z.enum(["now", "queued", "done"]).optional().describe("Filter to one zone"),
    seed: z.string().optional().describe("Optional seed override"),
}, async ({ address, mode, zone, seed }) => {
    try {
        const effectiveMode = mode ?? "owner";
        // Owner mode needs a wallet match — derive from seed if address omitted.
        const targetAddr = address ?? (() => {
            const { seed: s } = loadSeed({ seedArg: seed });
            return deriveKeys(s).address;
        })();
        const qs = new URLSearchParams();
        qs.set("mode", effectiveMode);
        if (zone)
            qs.set("zone", zone);
        const path = `/agent/panel/${targetAddr}?${qs.toString()}`;
        const res = effectiveMode === "owner"
            ? await apiSigned(path, "GET", undefined, { seed })
            : await api(path);
        return { content: [{ type: "text", text: JSON.stringify(res?.data ?? res, null, 2) }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        if (e instanceof SignatureError)
            return { content: [{ type: "text", text: `🔒 ${e.message}` }] };
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
server.tool("agent_submit", "Submit a single agent transaction (v10.10.7+ AFL-1, route POST /api/v1/agent/submit). The MCP signs as the caller automatically. Intent should match the AFL-1 spec; the server constructs + broadcasts the underlying Transaction.", {
    intent: z.record(z.any()).describe("Agent intent object per AFL-1 protocol (see docs/standards/afl-1-protocol-spec.md). Required fields depend on intent kind."),
    seed: z.string().optional().describe("Optional seed override"),
}, async ({ intent, seed }) => {
    try {
        const res = await apiSigned("/agent/submit", "POST", { intent }, { seed });
        return { content: [{ type: "text", text: JSON.stringify(res?.data ?? res, null, 2) }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        if (e instanceof SignatureError)
            return { content: [{ type: "text", text: `🔒 ${e.message}` }] };
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
server.tool("agent_submit_batch", "Submit a batch of agent intents in one call (v10.10.7+ AFL-1, route POST /api/v1/agent/submit-batch). Each intent is processed independently; the response includes a per-item result so partial failures don't lose the whole batch. Default cap is 1000 intents per call (Q_AGENT_BATCH_MAX, hard cap 10000).", {
    intents: z.array(z.record(z.any())).describe("Array of intent objects per AFL-1 spec"),
    seed: z.string().optional().describe("Optional seed override"),
}, async ({ intents, seed }) => {
    try {
        const res = await apiSigned("/agent/submit-batch", "POST", { intents }, { seed });
        return { content: [{ type: "text", text: JSON.stringify(res?.data ?? res, null, 2) }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        if (e instanceof SignatureError)
            return { content: [{ type: "text", text: `🔒 ${e.message}` }] };
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
server.tool("qshare_bootstrap_pool", "Seed the genesis QSHARE/QUG AMM pool (v10.10.7+, route POST /api/v1/qshare/bootstrap_pool). Founder-style call: deposits initial liquidity so the premium-ratio math has reserves to read against. Future v10.10.8 will add an AEGIS-QL founder gate; today this just requires a signed call.", {
    qug_amount: z.number().describe("Amount of QUG to deposit into the pool"),
    qshare_amount: z.number().describe("Amount of QSHARE to deposit (typically minted from initial allocation)"),
    seed: z.string().optional().describe("Optional seed override"),
}, async ({ qug_amount, qshare_amount, seed }) => {
    try {
        const res = await apiSigned("/qshare/bootstrap_pool", "POST", { qug_amount, qshare_amount }, { seed });
        return { content: [{ type: "text", text: JSON.stringify(res?.data ?? res, null, 2) }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        if (e instanceof SignatureError)
            return { content: [{ type: "text", text: `🔒 ${e.message}` }] };
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
// ============================================================
// v2.1.0 — X-ALGORITHM SURFACE (score tweet drafts + panel breakdown)
// ============================================================
server.tool("score_tweet_draft", "Score a tweet draft for engagement using the x-algorithm-scorer Layer-1 heuristic sidecar (proxies http://localhost:8090/score on the q-api-server host). Returns per-action probabilities (favorite, reply, repost, quote, block, mute, report) and variant suggestions for low-scoring drafts. Sidecar implements hand-engineered features (text-length sweet-spot, question-mark boost, all-caps risk, inflammatory-keyword list) — Layer-2 ML wrapper of xAI Phoenix engine deferred to v2+. No auth needed (read-only inference). Sidecar must be running on the same host as the MCP; configure via XALGO_SCORER_URL env.", {
    text: z.string().describe("Tweet draft to score"),
    recent_tweets: z.array(z.string()).optional().describe("Recent tweets for context (improves variant suggestions)"),
    target_audience: z.string().optional().describe("Brief description of target audience"),
}, async ({ text, recent_tweets, target_audience }) => {
    const url = process.env.XALGO_SCORER_URL || "http://localhost:8090/score";
    try {
        const body = { text };
        if (recent_tweets || target_audience) {
            body.context = {};
            if (recent_tweets)
                body.context.recent_tweets = recent_tweets;
            if (target_audience)
                body.context.target_audience = target_audience;
        }
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            redirect: "error",
        });
        if (!res.ok) {
            return {
                content: [{
                        type: "text",
                        text: `Sidecar at ${url} returned ${res.status}.\n` +
                            `Is x-algorithm-scorer running? Start it with:\n` +
                            `  cd tools/quillon-twitter-mcp/crates/x-algorithm-scorer && cargo run --release\n` +
                            `Then it listens on :8090.`,
                    }],
            };
        }
        const data = await res.json();
        return { content: [{ type: "text", text: JSON.stringify(data, null, 2) }] };
    }
    catch (e) {
        return {
            content: [{
                    type: "text",
                    text: `Sidecar at ${url} unreachable (${e?.message ?? e}).\n` +
                        `Either x-algorithm-scorer isn't running here, or XALGO_SCORER_URL needs to point at the host where it does.\n` +
                        `Quick start (on this host): cd tools/quillon-twitter-mcp/crates/x-algorithm-scorer && cargo run --release`,
                }],
        };
    }
});
server.tool("agent_panel_breakdown", "Same as agent_panel but renders the full ScoreReport.components for each task so you can see WHY each was ranked. Each candidate's TaskCandidate already has score: Option<ScoreReport> in the wire format — this tool just makes the breakdown legible instead of dropping it. Use mode=owner for full task list (needs signing); mode=embed for public truncated view.", {
    address: z.string().optional().describe("Wallet address; omit to use the address derived from the configured seed (only valid with mode=owner)"),
    mode: z.enum(["owner", "embed"]).optional().describe("'owner' (default) shows the full panel and requires signing; 'embed' is the public truncated view"),
    zone: z.enum(["now", "queued", "done"]).optional().describe("Filter to one zone"),
    seed: z.string().optional().describe("Optional seed override"),
}, async ({ address, mode, zone, seed }) => {
    try {
        const effectiveMode = mode ?? "owner";
        const targetAddr = address ?? (() => {
            const { seed: s } = loadSeed({ seedArg: seed });
            return deriveKeys(s).address;
        })();
        const qs = new URLSearchParams();
        qs.set("mode", effectiveMode);
        if (zone)
            qs.set("zone", zone);
        const path = `/agent/panel/${targetAddr}?${qs.toString()}`;
        const res = effectiveMode === "owner"
            ? await apiSigned(path, "GET", undefined, { seed })
            : await api(path);
        // Render the panel with score breakdowns expanded.
        const data = res?.data ?? res;
        const out = [];
        out.push(`Panel for ${data.wallet ?? targetAddr} (viewer: ${data.viewer_mode ?? effectiveMode})`);
        out.push(`Computed at: ${data.computed_at ?? "?"}`);
        out.push("");
        for (const zoneName of ["now", "queued", "done"]) {
            const tasks = data.zones?.[zoneName] ?? [];
            if (!tasks.length)
                continue;
            out.push(`=== ${zoneName.toUpperCase()} (${tasks.length}) ===`);
            for (const t of tasks) {
                const score = t.score?.total !== undefined ? `score=${t.score.total.toFixed(3)}` : "score=?";
                out.push(`• ${t.task_type ?? "?"}/${t.status ?? "?"} ${score} — ${t.label ?? t.task_id?.slice(0, 16) ?? "?"}`);
                if (t.score?.components?.length) {
                    for (const c of t.score.components) {
                        const explanation = c.explanation ? ` (${c.explanation})` : "";
                        out.push(`    ${c.name}: value=${c.value?.toFixed(3) ?? "?"} weight=${c.weight?.toFixed(2) ?? "?"}${explanation}`);
                    }
                }
                out.push("");
            }
        }
        return { content: [{ type: "text", text: out.join("\n") || "(panel empty)" }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        if (e instanceof SignatureError)
            return { content: [{ type: "text", text: `🔒 ${e.message}` }] };
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
server.tool("score_tx_dry", "Dry-score a candidate tx without submitting it. Runs the q-api-server's TxScorer against the current mempool + reserves snapshot, returns the full ScoreReport (4 components: balance_delta_health 0.35, fee_burden 0.20, mempool_pressure 0.20, reserve_utilization 0.25). Requires POST /api/v1/agent/score-tx-dry endpoint on the server (v10.10.10+). Lets the agent ask 'how would my candidate tx rank?' before committing.", {
    to_address: z.string().describe("Recipient qnk address"),
    amount_qug: z.number().describe("Amount of QUG to send (display units)"),
    fee_qug: z.number().optional().describe("Override fee; defaults to current mempool median"),
    seed: z.string().optional().describe("Optional seed override"),
}, async ({ to_address, amount_qug, fee_qug, seed }) => {
    try {
        const body = { to_address, amount_qug };
        if (fee_qug !== undefined)
            body.fee_qug = fee_qug;
        const res = await apiSigned("/agent/score-tx-dry", "POST", body, { seed });
        return { content: [{ type: "text", text: JSON.stringify(res?.data ?? res, null, 2) }] };
    }
    catch (e) {
        if (e instanceof SeedNotFoundError)
            return { content: [{ type: "text", text: `🔑 ${e.message}` }] };
        if (e instanceof SignatureError) {
            // Common case until v10.10.10 ships: server returns 404 for unknown route.
            return {
                content: [{
                        type: "text",
                        text: `🔒 ${e.message}\n\nIf the error is HTTP 404, /api/v1/agent/score-tx-dry isn't deployed yet. ` +
                            `That endpoint lands in v10.10.10. Production is currently v10.9.55.`,
                    }],
            };
        }
        return { content: [{ type: "text", text: `Failed: ${e?.message ?? e}` }] };
    }
});
async function fetchEnginePulse() {
    const raw = await api("/engine/pulse");
    if (raw?.success === false)
        throw new Error(raw.error ?? "engine_pulse returned success=false");
    return (raw?.data ?? raw);
}
/** Is the caller signing as the node's admin wallet? Operator-only tools
 *  gate on this. We compare the seed-derived address to whatever the node
 *  declared as its admin in /status (cheap heuristic; the server is the
 *  real authority — these tools just don't display fields the caller
 *  shouldn't see). */
async function isOperatorSelf(seedArg) {
    try {
        const { seed } = loadSeed({ seedArg });
        const { address } = deriveKeys(seed);
        const status = await api("/status");
        const adminAddr = (status?.data?.admin_wallet || status?.admin_wallet || "").toLowerCase();
        return adminAddr.length > 0 && adminAddr === address.toLowerCase();
    }
    catch {
        return false;
    }
}
function fmtBytes(n) {
    if (n < 1024)
        return `${n} B`;
    if (n < 1024 * 1024)
        return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3)
        return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(2)} GB`;
}
function fmtNum(n) {
    return Number(n).toLocaleString("en-US");
}
server.tool("chain_overview", "Scientist module — high-level chain stats anyone analyzing Quillon Graph needs first: height, sync status, mempool depth, mining health, peer bytes, version. Public — no admin gating. Use this as the entry point before drilling into mining_network or speed_report.", {}, async () => {
    try {
        const p = await fetchEnginePulse();
        const ageMs = Date.now() - p.ts_unix_ms;
        return {
            content: [{
                    type: "text",
                    text: [
                        `=== Quillon Graph — Chain Overview ===`,
                        ``,
                        `  Version:               ${p.version}`,
                        `  Tip height:            ${fmtNum(p.sync.current_height)}`,
                        `  Contiguous:            ${fmtNum(p.sync.contiguous_height)}`,
                        `  Network max seen:      ${fmtNum(p.sync.highest_network_height)}`,
                        `  Gap to tip:            ${p.sync.gap_to_tip} blocks  ${p.sync.is_caught_up ? "✓ caught up" : "⚠ syncing"}`,
                        ``,
                        `  Mempool size:          ${p.mempool.tx_pool_size} pending tx (tracked: ${p.mempool.tx_status_tracked})`,
                        `  Wallets seen:          ${fmtNum(p.wallets.known_count)}`,
                        ``,
                        `  Mining health:         ${p.mining.is_healthy ? "✓ healthy" : "⚠ unhealthy"}`,
                        `  Solutions submitted:   ${fmtNum(p.mining.solutions_submitted_total)}`,
                        `  Solutions accepted:    ${fmtNum(p.mining.solutions_accepted_total)}`,
                        `  Acceptance ratio:      ${p.mining.accept_ratio_pct.toFixed(2)}%`,
                        ``,
                        `  P2P bytes in:          ${fmtBytes(p.p2p.bytes_in_total)}`,
                        `  P2P bytes out:         ${fmtBytes(p.p2p.bytes_out_total)}`,
                        ``,
                        `  K-param (decentralization EMA): ${p.consensus.decentralization_ema.toFixed(2)}`,
                        ``,
                        `Data freshness: ${ageMs} ms.  Endpoint: GET /api/v1/engine/pulse`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `chain_overview failed: ${e?.message ?? e}` }] };
    }
});
server.tool("mining_network", "Scientist module — mining network power for the explorer's network-power modal: solutions per second (smoothed), accept ratio, time-since-last-solution, miner health. Use to argue 'is this chain alive + secured by real work'.", {}, async () => {
    try {
        const p = await fetchEnginePulse();
        const sinceLastSolutionMs = Date.now() - p.mining.last_solution_unix_ms;
        const sinceLastSolutionSec = (sinceLastSolutionMs / 1000).toFixed(1);
        const submitted = p.mining.solutions_submitted_total;
        const accepted = p.mining.solutions_accepted_total;
        const rejectedTotal = submitted - accepted;
        return {
            content: [{
                    type: "text",
                    text: [
                        `=== Mining Network Power ===`,
                        ``,
                        `  Health:                ${p.mining.is_healthy ? "✓ healthy" : "⚠ degraded"}`,
                        `  Last solution:         ${sinceLastSolutionSec}s ago`,
                        ``,
                        `  Solutions submitted:   ${fmtNum(submitted)}`,
                        `  Solutions accepted:    ${fmtNum(accepted)}`,
                        `  Solutions rejected:    ${fmtNum(rejectedTotal)}  (${((rejectedTotal / Math.max(1, submitted)) * 100).toFixed(2)}%)`,
                        `  Accept ratio:          ${p.mining.accept_ratio_pct.toFixed(2)}%`,
                        ``,
                        `  Notes for analysts:`,
                        `   - Accept ratio > 95% = mempool tracks current difficulty cleanly`,
                        `   - Last-solution < 2s = continuous block production (1 bps target)`,
                        `   - High rejection often signals miner is on stale difficulty`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `mining_network failed: ${e?.message ?? e}` }] };
    }
});
server.tool("speed_report", "Scientist module — SPEED. The pitch every DAG-Knight analyst will lead with: block time, finality latency, tip-proof verification time. Combines /engine/pulse with /proof/tip telemetry. Useful for comparing throughput against L1/L2 alternatives.", {}, async () => {
    try {
        const p = await fetchEnginePulse();
        let proofMs = "?";
        let proofVersion = "?";
        let proofSize = "?";
        try {
            const proof = await api("/proof/tip");
            const d = proof?.data ?? proof;
            if (d) {
                proofMs = (d.last_verify_us ? (d.last_verify_us / 1000).toFixed(2) : (d.verify_ms?.toFixed?.(2) ?? "?")) + " ms";
                proofVersion = d.proof_version ?? "?";
                proofSize = d.wire_size_bytes ? `${d.wire_size_bytes} bytes` : "?";
            }
        }
        catch { /* /proof/tip optional */ }
        const sinceLastSolutionMs = Date.now() - p.mining.last_solution_unix_ms;
        const blocksPerSecApprox = sinceLastSolutionMs > 0 ? (1000 / sinceLastSolutionMs).toFixed(3) : "n/a";
        return {
            content: [{
                    type: "text",
                    text: [
                        `=== Speed Report — Why Quillon Graph is fast ===`,
                        ``,
                        `  Block production:`,
                        `    Target:              ~1 block/sec (DAG-Knight cadence)`,
                        `    Observed instant:    ${blocksPerSecApprox} blk/s (from last solution timestamp)`,
                        `    Tip height:          ${fmtNum(p.sync.current_height)}`,
                        ``,
                        `  Finality:`,
                        `    DAG-Knight is BFT — finality is ANCHOR-COMMITTED, not probabilistic.`,
                        `    Once an anchor includes a vertex, it's final (no reorgs at depth ≥ 1 anchor).`,
                        `    Typical anchor-to-finality: < 3 seconds.`,
                        ``,
                        `  Trustless bootstrap (tip-proof):`,
                        `    Scheme:              ${proofVersion}`,
                        `    Wire size:           ${proofSize}`,
                        `    Verify time:         ${proofMs}  (target: < 10 ms)`,
                        `    Post-quantum:        BLAKE3 Fiat-Shamir (~128-bit Grover-quantum)`,
                        ``,
                        `  Sync throughput (recent benchmarks):`,
                        `    Peak block-pack:     ~3,348 blocks/sec to fresh node from Epsilon (10Gbit)`,
                        `    Steady-state:        ~570 blocks/sec average over full sync`,
                        ``,
                        `  Network observed:`,
                        `    Bytes in:            ${fmtBytes(p.p2p.bytes_in_total)}`,
                        `    Bytes out:           ${fmtBytes(p.p2p.bytes_out_total)}`,
                        ``,
                        `  Comparison cheat-sheet (typical L1 numbers, NOT endorsements):`,
                        `    Bitcoin   ~10 min blocks, probabilistic finality (~60 min)`,
                        `    Ethereum  ~12 sec slots, single-slot finality (~12 sec since Pectra)`,
                        `    Solana    ~400 ms slots, deterministic ~12 sec`,
                        `    Quillon   ~1 sec blocks, anchor finality ~3 sec, < 10 ms trustless bootstrap`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `speed_report failed: ${e?.message ?? e}` }] };
    }
});
server.tool("k_parameter", "Scientist module — the K-parameter (decentralization gauge). It's the EMA of the effective number of independent validators contributing recent anchors. Higher = more decentralized. Lower = bus-factor risk. Public.", {}, async () => {
    try {
        const p = await fetchEnginePulse();
        const k = p.consensus.decentralization_ema;
        const interpretation = k >= 50
            ? "✓ healthy — diverse anchor authorship"
            : k >= 20
                ? "moderate — visible diversity but watch trend"
                : k >= 5
                    ? "⚠ low — small set carrying anchors; investigate validator participation"
                    : "🚨 critical — single-operator risk; chain is effectively centralized";
        return {
            content: [{
                    type: "text",
                    text: [
                        `=== K-Parameter — Decentralization EMA ===`,
                        ``,
                        `  Current value:    ${k.toFixed(4)}`,
                        `  Interpretation:   ${interpretation}`,
                        ``,
                        `  What this measures:`,
                        `    Exponentially weighted moving average of the effective number`,
                        `    of independent validators that have contributed anchors in the`,
                        `    recent window. Computed entirely on-chain from block producer`,
                        `    keys; not self-reported.`,
                        ``,
                        `  Thresholds (rule of thumb):`,
                        `    K ≥ 50   healthy decentralization`,
                        `    20 ≤ K < 50   moderate, recoverable`,
                        `    5 ≤ K < 20   degraded, alertable`,
                        `    K < 5    critical — single-operator failure mode`,
                        ``,
                        `  Source: GET /api/v1/engine/pulse → consensus.decentralization_ema`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `k_parameter failed: ${e?.message ?? e}` }] };
    }
});
server.tool("operator_stats", "Scientist module (OPERATOR-ONLY) — node operator income, dev-fee config, and fee-tx counters. Returns the public summary if the caller's seed-derived address isn't the node's admin wallet, full breakdown otherwise. Gated by isOperatorSelf().", {
    seed: z.string().optional().describe("Optional seed override"),
}, async ({ seed }) => {
    try {
        const p = await fetchEnginePulse();
        const isOp = await isOperatorSelf(seed);
        const f = p.fees;
        const lines = [`=== Node Fee / Operator Stats ===`, ``];
        lines.push(`  Dev fee:               ${(f.dev_fee_bps / 100).toFixed(2)}% (${f.dev_fee_bps} bps)`);
        if (isOp) {
            lines.push(`  Operator fee:          ${(f.operator_fee_promille / 10).toFixed(2)}% (${f.operator_fee_promille} ‰)`);
            lines.push(`  Operator tx count:     ${fmtNum(f.operator_fee_tx_count)}`);
            lines.push(`  Earned this session:   ${fmtNum(f.operator_fees_earned_session)} (raw units)`);
            lines.push(`  Earned total:          ${fmtNum(f.operator_fees_earned_total)} (raw units)`);
            lines.push(``);
            lines.push(`  (Authenticated as operator — full breakdown shown.)`);
        }
        else {
            lines.push(``);
            lines.push(`  (Operator-specific fields hidden — call with the node's admin`);
            lines.push(`   seed to see operator_fee_promille / earned totals.)`);
        }
        return { content: [{ type: "text", text: lines.join("\n") }] };
    }
    catch (e) {
        return { content: [{ type: "text", text: `operator_stats failed: ${e?.message ?? e}` }] };
    }
});
server.tool("science_summary", "Scientist module — composite one-shot answer for 'tell me about this chain.' Bundles chain_overview + mining_network + speed_report + k_parameter into a single readable card. Use when pitching DAG-Knight to someone in 30 seconds.", {}, async () => {
    try {
        const p = await fetchEnginePulse();
        const sinceLastSolutionSec = ((Date.now() - p.mining.last_solution_unix_ms) / 1000).toFixed(1);
        const k = p.consensus.decentralization_ema;
        return {
            content: [{
                    type: "text",
                    text: [
                        `╔══════════════════════════════════════════════════════════════╗`,
                        `║         Quillon Graph — Science Summary                      ║`,
                        `╠══════════════════════════════════════════════════════════════╣`,
                        ``,
                        `  Speed`,
                        `    • ~1 block/sec, anchor finality ~3s, tip-proof verify <10ms`,
                        `    • Last solution: ${sinceLastSolutionSec}s ago (mining live)`,
                        ``,
                        `  Scale`,
                        `    • Tip:        ${fmtNum(p.sync.current_height)} blocks`,
                        `    • Wallets:    ${fmtNum(p.wallets.known_count)}`,
                        `    • P2P traffic: ${fmtBytes(p.p2p.bytes_in_total + p.p2p.bytes_out_total)} cumulative`,
                        ``,
                        `  Health`,
                        `    • Mining: ${p.mining.is_healthy ? "healthy" : "DEGRADED"}, accept ${p.mining.accept_ratio_pct.toFixed(1)}%`,
                        `    • Sync: ${p.sync.is_caught_up ? "caught up" : "syncing"}, gap ${p.sync.gap_to_tip}`,
                        `    • Mempool: ${p.mempool.tx_pool_size} pending`,
                        ``,
                        `  Decentralization`,
                        `    • K-parameter (EMA): ${k.toFixed(2)}  ${k >= 20 ? "✓" : k >= 5 ? "⚠" : "🚨"}`,
                        ``,
                        `  Crypto stack`,
                        `    • Block sigs: Hybrid Ed25519 + Dilithium5 (FIPS 204, NIST L5 PQ)`,
                        `    • Hashing: BLAKE3 + SHA3-256`,
                        `    • Tip proof: BLAKE3 Fiat-Shamir (~128-bit Grover-quantum)`,
                        ``,
                        `  Why analysts should care`,
                        `    • DAG-Knight = single-canonical-consensus + parallel execution`,
                        `      (sharding-grade throughput without cross-shard 2PC tax)`,
                        `    • Post-quantum from day 1, not bolted on later`,
                        `    • Trustless bootstrap = wallet verifies chain tip in <10ms,`,
                        `      no light-client trust assumption`,
                        ``,
                        `Drill down: chain_overview, mining_network, speed_report, k_parameter,`,
                        `             operator_stats (if you're the node operator).`,
                        ``,
                        `Source: GET /api/v1/engine/pulse (v${p.version})`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `science_summary failed: ${e?.message ?? e}` }] };
    }
});
// ============================================================
// v2.3.0: VISIBILITY-GAP FIXES + KILLER FEATURE + DEPLOY TOOL
// ============================================================
// Added in response to the May-21 "test water" trading session that
// exposed three visibility gaps + the 10^16 decimal-display bug:
//   1. get_token_balance   — non-QUG token balances (was QUG-only)
//   2. tx_status           — confirm a tx landed without inferring from balance
//   3. arb_scan            — KILLER: triangular arbitrage across all DEX pools
//   4. deploy_token        — create a new ERC20-style token (used for GROK
//                            commemorative + future drops)
// ============================================================
server.tool("get_token_balance", "Get the balance of any token (not just QUG) for any wallet. v2.3.0: closes the visibility gap that made multi-token trading dangerous — previously we could see only QUG via get_balance, so a successful QUG→TOKEN swap left us blind to whether the TOKEN actually arrived.", {
    symbol: z.string().describe("Token symbol (e.g., QUGUSD, wBTC, GROK). Run dex_list_tokens for the full set."),
    address: z.string().optional().describe("Wallet address. If omitted, uses the seed-derived agent wallet."),
    seed: z.string().optional().describe("Optional seed override for the address derivation when `address` is omitted."),
}, async ({ symbol, address, seed }) => {
    try {
        let target = address;
        if (!target) {
            const { seed: rawSeed } = loadSeed({ seedArg: seed });
            target = deriveKeys(rawSeed).address;
        }
        const tokens = await fetchTokens();
        const t = findTokenBySymbol(tokens, symbol);
        if (!t) {
            return { content: [{ type: "text", text: `Unknown token symbol "${symbol}". Run dex_list_tokens to see what's available.` }] };
        }
        if (t.symbol.toUpperCase() === "QUG") {
            // QUG has its own /balance endpoint; delegate.
            const res = await apiSigned(`/wallets/${target}/balance`, "GET", undefined, { seed });
            const qug = res?.balance ?? res?.data?.balance ?? 0;
            return { content: [{ type: "text", text: `${target}\n  ${symbol}: ${qug} QUG` }] };
        }
        // For all other tokens: hit /wallets/<addr>/tokens which returns the
        // full token-balance map. AMM/contract storage is 24-decimal universally.
        const res = await api(`/wallets/${target}/tokens`, "GET");
        const list = res?.tokens ?? res?.data?.tokens ?? res?.data ?? [];
        const entry = Array.isArray(list)
            ? list.find((e) => (e.symbol || "").toUpperCase() === t.symbol.toUpperCase())
            : null;
        if (!entry) {
            return { content: [{ type: "text", text: `${target}\n  ${symbol}: 0 (no balance found)` }] };
        }
        // entry.balance is the 24-decimal AMM base; convert for display.
        const raw = entry.balance ?? entry.amount ?? "0";
        const display = fromBaseUnits(String(raw), AMM_DECIMALS);
        return { content: [{ type: "text", text: `${target}\n  ${symbol}: ${display.toFixed(6)}\n  (raw base: ${raw}, scale: ${AMM_DECIMALS} decimals)` }] };
    }
    catch (e) {
        return { content: [{ type: "text", text: `get_token_balance failed: ${e?.message ?? e}` }] };
    }
});
server.tool("tx_status", "Confirm a transaction landed on-chain. v2.3.0: removes the May-21 ambiguity where a successful swap's hash returned but balance changes from mining masked whether the swap actually executed. Pass the tx hash from dex_swap / send_qug / send_token / agent_submit / deploy_token.", {
    tx_hash: z.string().describe("Transaction hash (0x-prefixed hex). 64 hex chars after the 0x."),
}, async ({ tx_hash }) => {
    try {
        const h = tx_hash.trim();
        const normalized = h.startsWith("0x") ? h : `0x${h}`;
        // Try the canonical endpoint first; fall back to mempool/recent if the
        // tx hasn't been mined yet.
        const res = await api(`/transactions/${normalized}`, "GET");
        const tx = res?.transaction ?? res?.data ?? res;
        if (!tx || res?.success === false) {
            // Not in confirmed history; probe mempool.
            const mp = await api(`/mempool/${normalized}`, "GET").catch(() => null);
            const inMempool = mp?.found === true || mp?.data?.found === true || mp?.transaction;
            if (inMempool) {
                return { content: [{ type: "text", text: `tx ${normalized.slice(0, 20)}…\n  Status: ⏳ PENDING (in mempool, awaiting block inclusion)` }] };
            }
            return { content: [{ type: "text", text: `tx ${normalized.slice(0, 20)}…\n  Status: ❓ NOT FOUND\n  Either the tx hasn't propagated yet, the hash is wrong, or it was rejected during validation. Try again in 2 seconds; if still NOT FOUND after ~10 s the tx never landed.` }] };
        }
        const height = tx.block_height ?? tx.height ?? "?";
        const status = tx.status ?? "confirmed";
        const ts = tx.timestamp ?? "?";
        const fromAddr = tx.from ?? tx.sender ?? "?";
        const toAddr = tx.to ?? tx.recipient ?? "?";
        const amount = tx.amount ?? tx.value ?? "?";
        const fee = tx.fee ?? "?";
        const ttype = tx.tx_type ?? tx.transaction_type ?? "?";
        return {
            content: [{
                    type: "text",
                    text: [
                        `tx ${normalized.slice(0, 20)}…`,
                        `  Status:   ✅ ${status.toString().toUpperCase()}`,
                        `  Block:    #${height}`,
                        `  Type:     ${ttype}`,
                        `  From:     ${String(fromAddr).slice(0, 20)}…`,
                        `  To:       ${String(toAddr).slice(0, 20)}…`,
                        `  Amount:   ${amount}`,
                        `  Fee:      ${fee}`,
                        `  Timestamp: ${ts}`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `tx_status failed: ${e?.message ?? e}` }] };
    }
});
// ────────────────────────────────────────────────────────────────────
// KILLER FEATURE: arb_scan
// ────────────────────────────────────────────────────────────────────
// Scans every pool in the DEX, computes triangular arbitrage loops
// (A → B → C → A), and ranks them by net return after the 0.3% pool
// fee charged on each of the three hops. Returns the top N loops with
// the exact path, input amount, expected output, and net profit.
//
// What an LLM agent dreams of: a one-call "find me money" function.
// Pure read — does not execute anything. Caller decides whether to
// run a sequence of dex_swap calls to capture the arb.
// ────────────────────────────────────────────────────────────────────
server.tool("arb_scan", "🎯 KILLER FEATURE — find triangular arbitrage opportunities across DEX pools. Pure-read scan: computes A→B→C→A loops, ranks by net return after 0.3% × 3 fees, returns the top opportunities. Caller decides whether to execute via dex_swap. Useful when the agent has spare QUG and wants to harvest mispricings between meme/native/wrapped pools.", {
    base_amount: z.number().positive().optional().describe("Probe input amount in QUG to size each loop (default 10). Larger probes reveal which loops can absorb size; smaller probes are cheaper to test."),
    min_profit_pct: z.number().optional().describe("Filter: only return loops with >= this net profit %. Default 0.1 (0.1%)."),
    top_n: z.number().optional().describe("How many opportunities to return (default 10)."),
    base_token: z.string().optional().describe("Loop start/end token symbol (default QUG)."),
}, async ({ base_amount, min_profit_pct, top_n, base_token }) => {
    try {
        const probe = base_amount ?? 10;
        const minPct = min_profit_pct ?? 0.1;
        const topN = top_n ?? 10;
        const base = (base_token ?? "QUG").toUpperCase();
        const tokens = await fetchTokens();
        const baseToken = findTokenBySymbol(tokens, base);
        if (!baseToken) {
            return { content: [{ type: "text", text: `Unknown base token "${base}". Pick one from dex_list_tokens.` }] };
        }
        // Build the set of intermediate-token candidates. Skip the base; skip
        // synthetic/auth/test tokens that have no real pool depth.
        const intermediates = tokens
            .filter(t => t.symbol.toUpperCase() !== base)
            .filter(t => !/^(AUTHTEST|TBORK|__)/i.test(t.symbol));
        const probeBase = toBaseUnits(probe, AMM_DECIMALS);
        // Helper: probe a single A→B quote. Returns receive in AMM-base, or null
        // if no pool / quote fails.
        const probeQuote = async (from, to, amountInBase) => {
            try {
                const res = await api("/dex/swap/quote", "POST", {
                    token_in: from,
                    token_out: to,
                    amount_in: amountInBase,
                    slippage_tolerance: 0.5,
                });
                if (res?.ok === false || res?.success === false)
                    return null;
                const out = res?.data?.amount_out ?? res?.amount_out;
                return out ? String(out) : null;
            }
            catch {
                return null;
            }
        };
        // Scan all base→B→C→base triangles. For 26 tokens this is ~625 triangles
        // × 3 hops = ~1875 quotes. We run with bounded concurrency to avoid
        // overwhelming the API; the average quote is ~30 ms so total scan is
        // ~10–20 s in the worst case.
        const opportunities = [];
        // Cache base→B and B→base hops so we don't requote in every triangle.
        const baseToB = new Map();
        const bToBase = new Map();
        for (const t of intermediates) {
            baseToB.set(t.symbol, await probeQuote(base, t.symbol, probeBase));
        }
        for (const t of intermediates) {
            // Probe a tiny back-leg amount just to confirm the pool exists.
            bToBase.set(t.symbol, await probeQuote(t.symbol, base, toBaseUnits(1, AMM_DECIMALS)));
        }
        // Now scan B→C middle legs for triangles. Only pairs where both legs
        // (base→B and C→base) exist are worth probing the middle.
        const viableTokens = intermediates.filter(t => baseToB.get(t.symbol));
        for (const tB of viableTokens) {
            const bAmount = baseToB.get(tB.symbol);
            if (!bAmount)
                continue;
            for (const tC of viableTokens) {
                if (tC.symbol === tB.symbol)
                    continue;
                if (!bToBase.get(tC.symbol))
                    continue;
                const cAmount = await probeQuote(tB.symbol, tC.symbol, bAmount);
                if (!cAmount)
                    continue;
                const backAmount = await probeQuote(tC.symbol, base, cAmount);
                if (!backAmount)
                    continue;
                // Net profit: backAmount / probeBase - 1
                const back = BigInt(backAmount);
                const probeB = BigInt(probeBase);
                if (probeB === 0n)
                    continue;
                // Compute (back - probeB) / probeB * 100 using BigInt for precision,
                // then convert to a Number at the end (safe since the ratio fits).
                const profitRaw = back - probeB;
                const profitPct = Number(profitRaw * 10000n / probeB) / 100;
                if (profitPct >= minPct) {
                    opportunities.push({
                        path: [base, tB.symbol, tC.symbol, base],
                        profitPct,
                        outBase: backAmount,
                        gas_cost_qug: 0.001, // 3 × 0.0003 estimated tx fee
                    });
                }
            }
        }
        opportunities.sort((a, b) => b.profitPct - a.profitPct);
        const top = opportunities.slice(0, topN);
        if (top.length === 0) {
            return { content: [{ type: "text", text: `=== Arb Scan Results ===\n\nNo profitable triangles ≥ ${minPct}% found.\nScanned ${intermediates.length} intermediate tokens (~${viableTokens.length * (viableTokens.length - 1)} triangles).\nProbe size: ${probe} ${base}.\n\nMarkets look efficient. Try smaller probe size, lower min_profit_pct, or different base_token.` }] };
        }
        const lines = [
            `=== Arb Scan — Top ${top.length} Triangles ===`,
            `Probe: ${probe} ${base}  ·  Min profit: ${minPct}%  ·  Pools scanned: ${viableTokens.length}`,
            ``,
        ];
        for (const [i, op] of top.entries()) {
            const finalOut = fromBaseUnits(op.outBase, AMM_DECIMALS);
            lines.push(`  ${i + 1}. ${op.path.join(" → ")}`);
            lines.push(`     Net profit: ${op.profitPct.toFixed(3)}%  ·  Final: ${finalOut.toFixed(6)} ${base}  ·  Gas est: ${op.gas_cost_qug.toFixed(3)} QUG`);
        }
        lines.push(``);
        lines.push(`To execute the top loop manually:`);
        lines.push(`  1. dex_swap from=${top[0].path[0]} to=${top[0].path[1]} amount=${probe} confirm=true`);
        lines.push(`  2. (read get_token_balance ${top[0].path[1]} to get exact received amount)`);
        lines.push(`  3. dex_swap from=${top[0].path[1]} to=${top[0].path[2]} amount=<step1_received> confirm=true`);
        lines.push(`  4. (read get_token_balance ${top[0].path[2]})`);
        lines.push(`  5. dex_swap from=${top[0].path[2]} to=${top[0].path[3]} amount=<step3_received> confirm=true`);
        lines.push(``);
        lines.push(`⚠ Quotes can shift between dispatches. Use small probe + tight slippage. Pool depth not modeled — top loops may have <0.1% impact at probe size but reorder at 10× probe.`);
        return { content: [{ type: "text", text: lines.join("\n") }] };
    }
    catch (e) {
        return { content: [{ type: "text", text: `arb_scan failed: ${e?.message ?? e}` }] };
    }
});
// ────────────────────────────────────────────────────────────────────
// deploy_token — used for GROK commemorative + future commemoratives
// ────────────────────────────────────────────────────────────────────
server.tool("deploy_token", "Deploy a new ERC20-style token on Quillon. Costs 1 QUG. v2.3.0: convenience wrapper for /api/v1/contracts/deploy with sensible defaults. Use this for commemorative drops (e.g., GROK alongside the existing CLAI) or app-token launches.", {
    name: z.string().min(2).max(64).describe("Full token name (e.g., 'Grok AI Commemorative')."),
    symbol: z.string().min(2).max(12).describe("Ticker symbol (e.g., 'GROK'). Uppercase recommended."),
    decimals: z.number().int().min(0).max(24).optional().describe("Decimals (default 0 for collectible/commemorative, 24 for utility tokens that need fractional units)."),
    initial_supply: z.number().positive().optional().describe("Initial supply minted to deployer (default 1_000_000). In display units; deploy will scale by 10^decimals."),
    description: z.string().optional().describe("Free-text description that ends up in the contract metadata."),
    seed: z.string().optional().describe("Optional seed override (otherwise: file → QNK_SEED env)."),
    confirm: z.boolean().optional().describe("Set to true to actually deploy. Without it, returns a dry-run summary."),
}, async ({ name, symbol, decimals, initial_supply, description, seed, confirm }) => {
    try {
        const dec = decimals ?? 0;
        const supply = initial_supply ?? 1_000_000;
        const desc = description ?? `Commemorative ${symbol} token on Quillon Graph`;
        let signerAddress;
        try {
            const { seed: rawSeed } = loadSeed({ seedArg: seed });
            signerAddress = deriveKeys(rawSeed).address;
        }
        catch (e) {
            return { content: [{ type: "text", text: `No wallet seed available: ${e.message}` }] };
        }
        // Strip "qnk" prefix for the deploy endpoint (it wants raw hex address).
        const ownerHex = signerAddress.startsWith("qnk") ? signerAddress.slice(3) : signerAddress;
        const supplyBase = toBaseUnits(supply, dec);
        if (!confirm) {
            return {
                content: [{
                        type: "text",
                        text: [
                            `⚠ DEPLOY DRY RUN — pass confirm=true to execute`,
                            ``,
                            `  Name:           ${name}`,
                            `  Symbol:         ${symbol}`,
                            `  Decimals:       ${dec}`,
                            `  Initial supply: ${supply.toLocaleString()} ${symbol}`,
                            `  Supply (base):  ${supplyBase}`,
                            `  Description:    ${desc}`,
                            `  Deployer:       ${signerAddress}`,
                            `  Cost:           1 QUG (deployment fee)`,
                            ``,
                            `When you confirm, an LP pool is NOT auto-created. To make ${symbol}`,
                            `tradeable on the DEX, follow the deploy with an add_liquidity call`,
                            `pairing ${symbol} with QUG.`,
                        ].join("\n"),
                    }],
            };
        }
        const res = await apiSigned("/contracts/deploy", "POST", {
            contract_type: "TOKEN",
            owner: ownerHex,
            parameters: {
                name,
                symbol,
                decimals: dec,
                initialSupply: supplyBase,
                description: desc,
            },
        }, { seed });
        if (res?.success === false) {
            return { content: [{ type: "text", text: `Deploy failed: ${res.error ?? 'unknown error'}\n\nIf the error mentions 'rate limit', wait an hour. If it mentions 'insufficient balance', the deployer needs ≥ 1 QUG.` }] };
        }
        const data = res?.data ?? res;
        const txHash = data.transaction_hash ?? data.tx_hash ?? data.deployment_tx ?? "(no tx hash in response)";
        const contractAddr = data.contract_address ?? data.address ?? "(pending)";
        return {
            content: [{
                    type: "text",
                    text: [
                        `✅ Deploy submitted!`,
                        ``,
                        `  Name:      ${name} (${symbol})`,
                        `  Contract:  ${contractAddr}`,
                        `  Tx hash:   ${txHash}`,
                        `  Supply:    ${supply.toLocaleString()} minted to deployer`,
                        ``,
                        `Next steps:`,
                        `  1. tx_status tx_hash=${txHash}  → confirm landed`,
                        `  2. get_token_balance symbol=${symbol}  → verify supply received`,
                        `  3. (optional) seed an LP pool against QUG so the token is tradeable`,
                        ``,
                        `Sentiment check on the deploy tx with score_tx_dry to_address=${signerAddress} amount_qug=1 — see how the agent_panel x-algo ranks it.`,
                    ].join("\n"),
                }],
        };
    }
    catch (e) {
        return { content: [{ type: "text", text: `deploy_token failed: ${e?.message ?? e}` }] };
    }
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
