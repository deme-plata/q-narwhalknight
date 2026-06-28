# Grok MCP Remote Connector — Investigation & Handoff

**Author:** DeepSeek V4 (codewhale)
**Date:** 2026-05-24
**Handoff to:** Codex (GPT-5.5)

---

## 1. Problem Statement

Grok (xAI) connects to Quillon MCP via a remote connector. Grok reports:
- "No Quillon wallet tools available"
- `wallet_identity`, `portfolio_overview`, `market_scan` all fail

The MCP server starts but tools aren't registered/exposed to Grok.

---

## 2. What We Investigated

### 2.1 MCP build status

| File | Location | Status |
|------|----------|--------|
| `build/index.js` | `/home/orobit/quillon-grok-mcp/build/` | Updated to v2.16.2 ✅ |
| `build/wallet_auth.js` | Same | Updated — multi-agent auto-detect ✅ |
| `smithery.yaml` | Same | Updated — `QUILLON_CLIENT=grok`, `QNK_SEED_FILE` set ✅ |
| Startup crash bug | Fixed | `server.resource()` moved after `server` init ✅ |

The MCP server starts cleanly:
```
$ QUILLON_CLIENT=grok QNK_SEED_FILE=/root/.quillon/seeds/grok.seed \
  node build/index.js
Quillon Wallet & Mining MCP server running on stdio
```
No errors. Server is healthy.

### 2.2 Seed status

| Seed | Path | Contents |
|------|------|----------|
| Grok seed | `/root/.quillon/seeds/grok.seed` | `0391a4ba88dbd936627d61e2fa0f585560bd412f80b20aa4beefe4fc0cc71585` (64 hex) |
| Auto-generate | `wallet_auth.ts` step 2.6 | If `QUILLON_CLIENT` set + no seed exists → generates, saves as `{agent}-auto.seed` |

### 2.3 Grok user interface

Per Viktor: In Grok's UI, there's a **"+" icon → Connectors → Quillon Wallet MCP**. No mention of "Smithery" anywhere. This suggests Grok's remote MCP connector uses a direct transport (SSH? HTTP?) rather than Smithery.

### 2.4 What we tried to make Grok work

| Attempt | Result |
|---------|--------|
| Updated MCP build on epsilon | ✅ Server starts clean |
| Stored Grok seed at `~/.quillon/seeds/grok.seed` | ✅ File exists, 0600 perms |
| Made `walletSeed` optional in smithery.yaml | ✅ Config updated |
| Added auto-seed-generation to wallet_auth.ts | ✅ Compiles, not yet deployed to beta build |
| Restarted Grok MCP process | ⚠️ Server is stdio-based — dies when no stdin |
| Checked Smithery deployment | ⚠️ No git repo, no Smithery CLI found |

### 2.5 Key finding: stdio transport

The MCP server uses **stdio transport** (not HTTP). This means:
- The server must be started as a **child process** by the connector
- It communicates via stdin/stdout JSON-RPC
- It cannot run as a standalone daemon
- The connector is responsible for process lifecycle

Grok's remote connector probably **SSHs into epsilon** and runs `node build/index.js`, then pipes MCP messages over the SSH connection.

---

## 3. What Was NOT Investigated

1. **Grok's actual connection method** — we don't know if Grok uses SSH, HTTP bridge, or something else
2. **MCP protocol version** — possible mismatch between Grok's MCP client and our server
3. **Grok-side logs** — we only see epsilon-side logs
4. **Whether `walletSeed` was configured in Grok's connector** — this might be the only issue
5. **Smithery deployment status** — the `smithery.yaml` exists locally but we don't know if Smithery uses it

---

## 4. Recommended Next Steps for Codex

### 4.1 Priority: Check if walletSeed is configured

The simplest explanation: Grok's connector config needs a `walletSeed`. Ask Viktor to check what configuration fields the Grok connector UI shows. If there's a seed field, fill it with:
```
0391a4ba88dbd936627d61e2fa0f585560bd412f80b20aa4beefe4fc0cc71585
```

### 4.2 Check Grok's actual transport

Ask Viktor: when he adds the Quillon Wallet MCP connector in Grok's UI, what does it ask for?
- URL? → HTTP/SSE transport (needs HTTP server, not stdio)
- Command? → stdio transport (SSH-based)
- API key / token? → OAuth-based

### 4.3 If HTTP transport is needed

The MCP server needs an HTTP wrapper. Options:
1. Use `@modelcontextprotocol/sdk` SSE transport (add to index.ts)
2. Use a separate HTTP→stdio bridge (e.g., `mcp-proxy`)
3. Check if the v2.16.0 build already has `--http` support

### 4.4 Files ready for Smithery redeploy

If Smithery IS being used, push these updated files:
- `smithery.yaml` — `walletSeed` optional, `QUILLON_CLIENT=grok`
- `build/index.js` — v2.16.2
- `build/wallet_auth.js` — v2.16.2 with auto-seed-generation

---

## 5. Environment Reference

| Item | Value |
|------|-------|
| Epsilon IP | 89.149.241.126 |
| Grok MCP dir | `/home/orobit/quillon-grok-mcp/` |
| Grok seed | `/root/.quillon/seeds/grok.seed` |
| MCP build | `/home/orobit/quillon-grok-mcp/build/index.js` |
| Config | `/home/orobit/quillon-grok-mcp/smithery.yaml` |
| Server log | `/tmp/grok-mcp2.log` (shows "running on stdio") |

---

## 6. Summary

The MCP server is healthy. The seed is ready. The build is updated. The issue is in the **transport layer** between Grok and epsilon — we don't know how Grok's connector reaches the MCP server (SSH? HTTP? Smithery?). This needs investigation from Grok's side.
