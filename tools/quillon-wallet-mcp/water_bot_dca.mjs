#!/usr/bin/env node
// water_bot_dca.mjs
// DCA water-bot test: N tiny dex_swaps over a fixed time window.
// Alternates QUG→PACI and QUG→SCALPEL each iteration.
// Each swap is signed with the wallet seed and sent directly to /api/v1/dex/swap.
//
// Usage:
//   node water_bot_dca.mjs [count=60] [intervalMs=1000] [perTxQug=0.05]
//
// Honest expectations under v10.11.18/19 chain thrash:
//   - Some calls will hit 502/timeouts during OOM-restart windows
//   - Total elapsed may stretch past targetWindow if chain stalls
//   - Each successful call gives ~0.3% LP fee to the pool's LP holder

import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
import { readFileSync } from "node:fs";

const COUNT = Number(process.argv[2] ?? 60);
const INTERVAL_MS = Number(process.argv[3] ?? 1000);
const PER_TX_QUG = Number(process.argv[4] ?? 0.05);

const SEED_FILE = process.env.QNK_SEED_FILE
  || `${process.env.HOME ?? "/root"}/.claude/quillon-agent-seed`;
const API = process.env.QUILLON_API_URL ?? "https://quillon.xyz/api/v1";

// Token addresses for SCALPEL + PACI (Custom-type tokens need full qnk address per dex_pool_lookup_symbol_vs_address memory)
const SCALPEL_ADDR = "qnk3d879e69a57bcb5af685ac517c2092fee125c32fb103599a53c8c8fc2bbcc086";
const PACI_ADDR    = "qnkc722f70148faf360a342578f8c8d28da5dba4d4dc52e3d5ef4e08857767263d3";

// 24-decimal QUG base-unit amount
const amountBase = (BigInt(Math.floor(PER_TX_QUG * 1_000_000)) * 10n ** 18n).toString();
// Naive: assume 1 QUG → ~150-180 SCALPEL or ~150-180 PACI based on current pool prices.
// Set min_amount_out to a conservative 50% of expected = high slippage tolerance for the test.
const minOutBase = ((BigInt(Math.floor(PER_TX_QUG * 50 * 1_000_000)) * 10n ** 18n) / 100n).toString();

function loadSeed() {
  const raw = readFileSync(SEED_FILE, "utf8").trim();
  if (raw.startsWith("{")) {
    return JSON.parse(raw).seed;
  }
  return raw;
}

function signXWalletAuth(path, seed) {
  const priv = sha3_256(utf8ToBytes(seed));
  const pub = ed25519.getPublicKey(priv);
  const addr = "qnk" + bytesToHex(pub);
  const ts = Math.floor(Date.now() / 1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts);
  for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
  const sigBuf = new Uint8Array(40 + path.length);
  sigBuf.set(pub, 0);
  sigBuf.set(tsBuf, 32);
  sigBuf.set(utf8ToBytes(path), 40);
  const header = JSON.stringify({
    address: addr,
    timestamp: ts,
    scheme: "Ed25519",
    signature: bytesToHex(ed25519.sign(sha3_256(sigBuf), priv)),
    public_key: bytesToHex(pub),
  });
  return { header, addr };
}

async function dexSwap(toAddr, label) {
  const path = "/api/v1/dex/swap";
  const seed = loadSeed();
  const { header, addr } = signXWalletAuth(path, seed);
  const body = JSON.stringify({
    from_token: "QUG",
    to_token: toAddr,
    amount_in: amountBase,
    min_amount_out: minOutBase,
    wallet_address: addr,
    slippage_tolerance: 10.0,
  });
  const t0 = Date.now();
  try {
    const r = await fetch(API.replace(/\/api\/v1$/, "") + path, {
      method: "POST",
      headers: { "X-Wallet-Auth": header, "Content-Type": "application/json" },
      body,
      signal: AbortSignal.timeout(10_000),
    });
    const text = await r.text();
    const dt = Date.now() - t0;
    return { ok: r.ok, status: r.status, dt, body: text.slice(0, 200), label };
  } catch (e) {
    return { ok: false, status: 0, dt: Date.now() - t0, body: String(e), label };
  }
}

(async () => {
  console.log(`water_bot_dca: ${COUNT} swaps, ~${INTERVAL_MS}ms apart, ${PER_TX_QUG} QUG each`);
  console.log(`  alternating QUG→PACI and QUG→SCALPEL`);
  console.log(`  wallet seed source: ${SEED_FILE}`);
  console.log(`  API: ${API}`);
  console.log(``);

  const start = Date.now();
  const results = [];
  for (let i = 0; i < COUNT; i++) {
    const useScalpel = i % 2 === 0;
    const targetAddr = useScalpel ? SCALPEL_ADDR : PACI_ADDR;
    const label = useScalpel ? "SCALPEL" : "PACI";
    const r = await dexSwap(targetAddr, label);
    results.push(r);
    const tag = r.ok ? "✅" : "❌";
    console.log(`[${(i + 1).toString().padStart(2, "0")}/${COUNT}] ${tag} ${label.padEnd(7)} ${r.status} ${r.dt}ms ${r.body.slice(0, 80).replace(/\n/g, " ")}`);
    // Sleep INTERVAL_MS minus time-elapsed-in-call, floor at 0
    const sleepFor = Math.max(0, INTERVAL_MS - r.dt);
    if (i < COUNT - 1 && sleepFor > 0) await new Promise(res => setTimeout(res, sleepFor));
  }
  const elapsed = (Date.now() - start) / 1000;
  const ok = results.filter(r => r.ok).length;
  const fail = results.length - ok;
  console.log(``);
  console.log(`=== Summary ===`);
  console.log(`  Total elapsed:     ${elapsed.toFixed(1)}s`);
  console.log(`  Successful swaps:  ${ok} / ${COUNT}`);
  console.log(`  Failed swaps:      ${fail}`);
  console.log(`  Success rate:      ${(ok / COUNT * 100).toFixed(1)}%`);
  console.log(`  Total QUG spent:   ~${(PER_TX_QUG * ok).toFixed(4)} (only on successful swaps)`);
  console.log(``);
  console.log(`Per-pool fee revenue earned by LP-holders:`);
  console.log(`  PACI/QUG pool LP (Rocky):    ~${(PER_TX_QUG * 0.003 * Math.floor(ok / 2)).toFixed(6)} QUG`);
  console.log(`  SCALPEL/QUG pool LP (Codex): ~${(PER_TX_QUG * 0.003 * Math.ceil(ok / 2)).toFixed(6)} QUG`);
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
