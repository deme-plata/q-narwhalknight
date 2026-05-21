// TPS test with REAL money — extremely low, random amounts in [1, 10000] base
// units (1e-8 to 1e-4 QUG). All txs go to Viktor's admin wallet as a real-money
// throughput probe across v10.9.49+ send_signed endpoint.
//
// N=20 txs in parallel.  Total spend bounded above by 20 × 10000 = 200,000 base
// units = 0.002 QUG.  Tiny relative to wallet balance.
import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";

const seed = "9c83a476b9c1ba558429058ffb2297dfa0cb0284f96c48c661fab6f93cd1ee41";
const priv = sha3_256(utf8ToBytes(seed));
const pub  = ed25519.getPublicKey(priv);
const addr = "qnk" + bytesToHex(pub);
const RECIPIENT = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";
// v10.10.15 fix: was `const API = "https://quillon.xyz/api/v1"` + `path = "/api/v1/..."`
// → URL became "https://quillon.xyz/api/v1/api/v1/transactions/send_signed" (404),
// my parser treated 404+non-JSON as ok and reported false "31.7 TPS / 20/20 OK".
// Now API holds only the host; path holds the full route. Fixes URL doubling.
const API = "https://quillon.xyz";
const path = "/api/v1/transactions/send_signed";
const N = 20;

function auth(path) {
  const ts = Math.floor(Date.now() / 1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts);
  for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
  const buf = new Uint8Array(40 + path.length);
  buf.set(pub, 0); buf.set(tsBuf, 32); buf.set(utf8ToBytes(path), 40);
  const sig = ed25519.sign(sha3_256(buf), priv);
  return JSON.stringify({ address: addr, timestamp: ts, scheme: "Ed25519", signature: bytesToHex(sig) });
}

function rand(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }

console.log("TPS test: " + N + " parallel signed sends, random amounts in [1, 10000] base units (1e-8 to 1e-4 QUG)");
console.log("Sender:    " + addr);
console.log("Recipient: " + RECIPIENT);
console.log("Endpoint:  " + API + path);
console.log("---");

const start = Date.now();
const results = await Promise.allSettled(
  Array.from({ length: N }, (_, i) => {
    const amount = rand(1, 10000);
    const body = { from: addr, to: RECIPIENT, amount, token_type: "QUG", memo: `tps-tiny-${i}-rand${amount}` };
    return fetch(API + path, {
      method: "POST",
      headers: { "X-Wallet-Auth": auth(path), "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(async r => ({ amount, status: r.status, body: (await r.text()).slice(0, 200) }));
  }),
);
const elapsed = (Date.now() - start) / 1000;

let ok = 0, fail = 0;
const errMap = new Map();
const samples = [];
for (const r of results) {
  if (r.status === "fulfilled" && r.value.status >= 200 && r.value.status < 300) {
    try {
      const j = JSON.parse(r.value.body);
      if (j.success === false) { fail++; const k = (j.error ?? "unknown").slice(0, 60); errMap.set(k, (errMap.get(k) ?? 0) + 1); }
      else { ok++; if (samples.length < 3) samples.push({ amount: r.value.amount, tx: j.data?.transaction_id ?? "?" }); }
    } catch { ok++; }
  } else {
    fail++;
    const k = r.status === "rejected" ? `network: ${r.reason?.message?.slice(0, 40)}` : `HTTP ${r.value.status}: ${r.value.body.slice(0,80)}`;
    errMap.set(k, (errMap.get(k) ?? 0) + 1);
  }
}

console.log(`OK:     ${ok}/${N}`);
console.log(`FAIL:   ${fail}/${N}`);
console.log(`Time:   ${elapsed.toFixed(2)}s   ⇒ ${(ok / elapsed).toFixed(1)} ok TPS`);
if (samples.length) {
  console.log("\nSample transaction IDs:");
  for (const s of samples) console.log(`  amount=${s.amount}  tx=${s.tx}`);
}
if (errMap.size) {
  console.log("\nError reasons:");
  for (const [k, v] of [...errMap.entries()].sort((a,b) => b[1] - a[1])) {
    console.log(`  ${v}× ${k}`);
  }
}
