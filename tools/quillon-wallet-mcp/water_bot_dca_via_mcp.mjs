#!/usr/bin/env node
// water_bot_dca_via_mcp.mjs
// DCA test that goes THROUGH the Quillon MCP server (stdio, JSON-RPC).
// Not raw HTTP — this exercises the MCP's actual dex_swap tool layer.
//
// Usage:
//   node water_bot_dca_via_mcp.mjs [count=60] [intervalMs=1000] [perTxQug=0.05]

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const COUNT = Number(process.argv[2] ?? 60);
const INTERVAL_MS = Number(process.argv[3] ?? 1000);
const PER_TX_QUG = Number(process.argv[4] ?? 0.05);

const MCP_BIN = process.env.MCP_BIN
  || "/opt/orobit/shared/q-narwhalknight/tools/quillon-wallet-mcp/build/index.js";

// Use SYMBOL names — MCP dex_swap tool's findTokenBySymbol gate runs BEFORE
// the v2.10.2 address-resolver. Pass "SCALPEL"/"PACI" and the tool internally
// converts to the qnk-address for the API call. (Real MCP UX bug: callers
// who only know the contract hex get rejected at validation gate.)
const SCALPEL = "SCALPEL";
const PACI    = "PACI";

(async () => {
  console.log(`water_bot_dca_via_mcp: ${COUNT} swaps via MCP stdio, ~${INTERVAL_MS}ms apart, ${PER_TX_QUG} QUG each`);
  console.log(`  MCP binary: ${MCP_BIN}`);
  console.log(`  Alternating QUG→PACI and QUG→SCALPEL`);
  console.log(``);

  const transport = new StdioClientTransport({
    command: "node",
    args: [MCP_BIN],
    env: { ...process.env },
  });
  const client = new Client({ name: "water-bot-dca", version: "0.1.0" }, { capabilities: {} });
  await client.connect(transport);
  console.log("MCP connected. Listing available tools...");
  const tools = await client.listTools();
  console.log(`  ${tools.tools.length} tools available`);
  console.log(`  dex_swap present: ${tools.tools.some(t => t.name === "dex_swap")}`);
  console.log(``);

  const start = Date.now();
  const results = [];
  for (let i = 0; i < COUNT; i++) {
    const useScalpel = i % 2 === 0;
    const targetSymbol = useScalpel ? "SCALPEL" : "PACI";
    const targetAddr = useScalpel ? SCALPEL : PACI;
    const t0 = Date.now();
    let ok = false;
    let bodyHint = "";
    try {
      const res = await client.callTool({
        name: "dex_swap",
        arguments: {
          from_token: "QUG",
          to_token: targetAddr,
          amount: PER_TX_QUG,
          slippage_percent: 10,
          confirm: true,
          endpoint: "epsilon",
        },
      });
      const text = res?.content?.[0]?.text ?? "";
      ok = text.startsWith("✅") || text.includes("Swap submitted");
      bodyHint = text.slice(0, 100).replace(/\n/g, " ");
    } catch (e) {
      bodyHint = String(e).slice(0, 120);
    }
    const dt = Date.now() - t0;
    results.push({ ok, dt, label: targetSymbol, bodyHint });
    const tag = ok ? "✅" : "❌";
    console.log(`[${(i + 1).toString().padStart(2, "0")}/${COUNT}] ${tag} ${targetSymbol.padEnd(7)} ${dt.toString().padStart(5)}ms ${bodyHint.slice(0, 90)}`);
    const sleepFor = Math.max(0, INTERVAL_MS - dt);
    if (i < COUNT - 1 && sleepFor > 0) await new Promise(r => setTimeout(r, sleepFor));
  }

  const elapsed = (Date.now() - start) / 1000;
  const ok = results.filter(r => r.ok).length;
  const fail = results.length - ok;
  const avgDt = results.reduce((s, r) => s + r.dt, 0) / results.length;
  console.log(``);
  console.log(`=== Summary (MCP stdio path) ===`);
  console.log(`  Total elapsed:    ${elapsed.toFixed(1)}s`);
  console.log(`  Successful:       ${ok} / ${COUNT}`);
  console.log(`  Failed:           ${fail}`);
  console.log(`  Success rate:     ${(ok / COUNT * 100).toFixed(1)}%`);
  console.log(`  Avg call latency: ${avgDt.toFixed(0)}ms`);
  console.log(`  QUG spent:        ~${(PER_TX_QUG * ok).toFixed(4)}`);
  console.log(`  LP fees to Rocky's PACI pool:    ~${(PER_TX_QUG * 0.003 * Math.floor(ok / 2)).toFixed(6)} QUG`);
  console.log(`  LP fees to Codex's SCALPEL pool: ~${(PER_TX_QUG * 0.003 * Math.ceil(ok / 2)).toFixed(6)} QUG`);

  await client.close();
  process.exit(0);
})().catch(e => { console.error("FATAL:", e); process.exit(1); });
