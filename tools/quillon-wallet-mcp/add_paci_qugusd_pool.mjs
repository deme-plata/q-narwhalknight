// Create the QUG-USD ↔ PACI AMM pool — 2026-05-21
// 22,054 QUGUSD ↔ 220,540 PACI → 1 PACI = $0.10 implied
// Submitted against Epsilon direct (89.149.241.126:8080) to dodge LB routing.
import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
import { readFileSync } from "node:fs";

const SEED = readFileSync("/root/.claude/quillon-agent-seed", "utf8").trim();
const PRIV = sha3_256(utf8ToBytes(SEED));
const PUB = ed25519.getPublicKey(PRIV);
const ME = "qnk" + bytesToHex(PUB);

const PACI_ADDR = "qnkc722f70148faf360a342578f8c8d28da5dba4d4dc52e3d5ef4e08857767263d3";
const BASE = "http://89.149.241.126:8080";

function sig(path) {
  const ts = Math.floor(Date.now() / 1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts); for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
  const buf = new Uint8Array(40 + path.length);
  buf.set(PUB, 0); buf.set(tsBuf, 32); buf.set(utf8ToBytes(path), 40);
  return JSON.stringify({
    address: ME, timestamp: ts, scheme: "Ed25519",
    signature: bytesToHex(ed25519.sign(sha3_256(buf), PRIV)),
    public_key: bytesToHex(PUB),
  });
}

// 24-decimal AMM-base universal scale.
const POW24 = 10n ** 24n;
const AMT_QUGUSD_RAW = (22054n * POW24).toString();   // 22,054 QUGUSD
const AMT_PACI_RAW   = (220540n * POW24).toString();  // 220,540 PACI

const path = "/api/v1/liquidity/add";
// CRITICAL: u128 amounts as JSON STRINGS, not bare integers. The server's
// deserialize_u128_from_any accepts both, but bare integers > 2^53 get
// parsed as f64 and lose precision (server logs WARN deserialize_u128:
// received f64 = … PRECISION MAY BE LOST!). For first-LP pool create
// the exact ratio sets the price, so we cannot afford precision drift.
const body = `{"token0":"QUGUSD","token1":"${PACI_ADDR}","amount0":"${AMT_QUGUSD_RAW}","amount1":"${AMT_PACI_RAW}","provider":"${ME}"}`;

console.log("Submitting first-LP for QUGUSD ↔ PACI:");
console.log(`  token0 = QUGUSD       amount0 = 22,054 (raw ${AMT_QUGUSD_RAW.slice(0, 10)}…)`);
console.log(`  token1 = ${PACI_ADDR.slice(0, 12)}…  amount1 = 220,540 (raw ${AMT_PACI_RAW.slice(0, 10)}…)`);
console.log(`  implied: 1 PACI = $0.10 USD (1 QUGUSD = 10 PACI)`);
console.log(`  provider: ${ME.slice(0, 18)}…`);
console.log();

const t0 = Date.now();
// 90s timeout — first-LP pool create can take 30-60s on Epsilon when busy
// (writes new pool to RocksDB, mints LP token credit, broadcasts).
const ctl = new AbortController();
const tm = setTimeout(() => ctl.abort(), 90_000);
let r, txt;
try {
  r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
    body,
    signal: ctl.signal,
  });
  txt = await r.text();
} finally {
  clearTimeout(tm);
}
const dur = Date.now() - t0;
console.log(`HTTP ${r.status}  ${dur}ms`);
console.log(txt);
