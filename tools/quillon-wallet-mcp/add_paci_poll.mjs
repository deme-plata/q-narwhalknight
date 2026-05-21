import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
import { readFileSync } from "node:fs";

const SEED = readFileSync("/root/.claude/quillon-agent-seed", "utf8").trim();
const PRIV = sha3_256(utf8ToBytes(SEED));
const PUB = ed25519.getPublicKey(PRIV);
const ME = "qnk" + bytesToHex(PUB);
const PACI = "qnkc722f70148faf360a342578f8c8d28da5dba4d4dc52e3d5ef4e08857767263d3";
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

const POW24 = 10n ** 24n;
const A0 = (22054n * POW24).toString();
const A1 = (220540n * POW24).toString();
const path = "/api/v1/liquidity/add";
const body = `{"token0":"QUGUSD","token1":"${PACI}","amount0":"${A0}","amount1":"${A1}","provider":"${ME}"}`;

console.log(`[${new Date().toISOString()}] launching POST /liquidity/add (NOT awaiting response)`);
// Fire-and-forget: kick off the fetch but don't await the body
const submission = fetch(BASE + path, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
  body,
}).then(async (r) => {
  const txt = await r.text().catch(() => "(no body)");
  console.log(`\n[${new Date().toISOString()}] FETCH FINISHED: HTTP ${r.status} body[0..200]: ${txt.slice(0, 200)}`);
}).catch((e) => {
  console.log(`\n[${new Date().toISOString()}] FETCH FAILED: ${e.message}`);
});

// Poll pools every 8s, max 12 iterations = 96s
for (let i = 1; i <= 12; i++) {
  await new Promise((r) => setTimeout(r, 8000));
  const r = await fetch(BASE + "/api/v1/liquidity/pools", { signal: AbortSignal.timeout(5000) }).catch((e) => null);
  if (!r) { console.log(`  poll[${i}] FAILED to fetch pools`); continue; }
  const j = await r.json();
  const pools = j?.data ?? [];
  const paciPool = pools.find((p) => (p.token0 || "").includes("c722f7") || (p.token1 || "").includes("c722f7"));
  if (paciPool) {
    console.log(`\n[${new Date().toISOString()}] ✅ POOL CREATED on poll #${i}`);
    console.log(`  pool_id:  ${paciPool.pool_id}`);
    console.log(`  token0:   ${paciPool.token0}`);
    console.log(`  token1:   ${paciPool.token1}`);
    console.log(`  reserve0: ${paciPool.reserve0}`);
    console.log(`  reserve1: ${paciPool.reserve1}`);
    console.log(`  provider: ${paciPool.provider}`);
    break;
  } else {
    console.log(`  poll[${i.toString().padStart(2)}] ${pools.length} pools total, no PACI yet`);
  }
}

// Make sure submission finishes before we exit
await submission;
