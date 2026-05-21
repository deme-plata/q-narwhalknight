// Query my agent-wallet balance from all reachable Quillon nodes.
// Same X-Wallet-Auth signature works on any node — signature signs
// SHA3(pub || ts || path) where path is the same /api/v1/wallets/<addr>/balance
// regardless of which node we hit. Each node looks up its own DB.

import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";

const seed = "9c83a476b9c1ba558429058ffb2297dfa0cb0284f96c48c661fab6f93cd1ee41";
const priv = sha3_256(utf8ToBytes(seed));
const pub  = ed25519.getPublicKey(priv);
const addr = "qnk" + bytesToHex(pub);
const path = "/api/v1/wallets/" + addr + "/balance";

function authHeader(path) {
  const ts = Math.floor(Date.now() / 1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts);
  for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
  const buf = new Uint8Array(40 + path.length);
  buf.set(pub, 0); buf.set(tsBuf, 32); buf.set(utf8ToBytes(path), 40);
  const sig = ed25519.sign(sha3_256(buf), priv);
  return JSON.stringify({ address: addr, timestamp: ts, scheme: "Ed25519", signature: bytesToHex(sig) });
}

const NODES = [
  { name: "Epsilon (prod via quillon.xyz)", url: "https://quillon.xyz" },
  { name: "Epsilon (direct :8080)",          url: "http://89.149.241.126:8080" },
  { name: "Gamma (:8808)",                   url: "http://109.205.176.60:8808" },
  { name: "Delta (:8080)",                   url: "http://5.79.79.158:8080" },
];

const QUG_SCALE = 1e24;

console.log("Wallet:", addr);
console.log();
console.log("node                                | balance (QUG)            | height        | version    | latency");
console.log("------------------------------------|--------------------------|---------------|------------|--------");

const results = [];
for (const n of NODES) {
  const t0 = Date.now();
  try {
    const r = await fetch(n.url + path, {
      headers: { "X-Wallet-Auth": authHeader(path) },
      signal: AbortSignal.timeout(8000),
    });
    const latency = Date.now() - t0;
    if (!r.ok) {
      console.log(`${n.name.padEnd(36)}| HTTP ${r.status}                | -             | -          | ${latency}ms`);
      results.push({ name: n.name, balance: null, error: `HTTP ${r.status}` });
      continue;
    }
    const j = await r.json();
    const raw = typeof j.data?.balance === "number" ? j.data.balance
              : typeof j.data?.balance === "string" ? parseFloat(j.data.balance)
              : typeof j.balance === "number" ? j.balance
              : null;
    const display = raw !== null ? raw / QUG_SCALE : null;

    // Also fetch /status for height + version
    const sr = await fetch(n.url + "/api/v1/status", { signal: AbortSignal.timeout(5000) });
    const sj = await sr.json();
    const h = sj.data?.upgrades?.current_height ?? "?";
    const v = sj.data?.version ?? "?";

    if (display !== null) {
      console.log(`${n.name.padEnd(36)}| ${display.toFixed(8).padEnd(25)}| ${String(h).padEnd(14)}| ${String(v).padEnd(11)}| ${latency}ms`);
      results.push({ name: n.name, balance: display, height: h, version: v });
    } else {
      console.log(`${n.name.padEnd(36)}| (unparseable response)   | ${String(h).padEnd(14)}| ${String(v).padEnd(11)}| ${latency}ms`);
      results.push({ name: n.name, balance: null, error: "unparseable", raw_response: j });
    }
  } catch (e) {
    const latency = Date.now() - t0;
    console.log(`${n.name.padEnd(36)}| ERROR: ${e.message.slice(0,18).padEnd(18)}| -             | -          | ${latency}ms`);
    results.push({ name: n.name, balance: null, error: e.message });
  }
}

console.log();
const balances = results.filter(r => r.balance !== null).map(r => r.balance);
if (balances.length >= 2) {
  const min = Math.min(...balances);
  const max = Math.max(...balances);
  const diff = max - min;
  if (diff < 0.0001) {
    console.log(`✅ All ${balances.length} reachable nodes agree on balance: ${min.toFixed(8)} QUG (diff < 0.0001)`);
  } else {
    console.log(`⚠️  DISCREPANCY: max=${max.toFixed(8)} min=${min.toFixed(8)} diff=${diff.toFixed(8)} QUG`);
    console.log("Investigate per-node sync state via quillon-sync-probe skill.");
  }
} else {
  console.log("Not enough reachable nodes to compare.");
}
