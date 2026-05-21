// One-shot: send 0.01 QUGUSD to Viktor with a memo about today's work,
// then run a 5-tx batch self-loop to measure send_batch TPS.
//
// Pure HTTP — no MCP, no browser-auth. Uses the same X-Wallet-Auth
// signature pattern as multi_node_balance.mjs.

import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";

const SEED = "9c83a476b9c1ba558429058ffb2297dfa0cb0284f96c48c661fab6f93cd1ee41";
const PRIV = sha3_256(utf8ToBytes(SEED));
const PUB  = ed25519.getPublicKey(PRIV);
const ME   = "qnk" + bytesToHex(PUB);
const VIKTOR = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

const API = "https://quillon.xyz";
const QUGUSD_DECIMALS = 24n;
const QUG_SCALE = 10n ** QUGUSD_DECIMALS;

function signedAuthHeader(path) {
  const ts = Math.floor(Date.now() / 1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts);
  for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
  const buf = new Uint8Array(40 + path.length);
  buf.set(PUB, 0); buf.set(tsBuf, 32); buf.set(utf8ToBytes(path), 40);
  const sig = ed25519.sign(sha3_256(buf), PRIV);
  return JSON.stringify({ address: ME, timestamp: ts, scheme: "Ed25519", signature: bytesToHex(sig) });
}

function toBase(displayAmount, decimals = QUGUSD_DECIMALS) {
  const [whole, frac = ""] = String(displayAmount).split(".");
  const padded = (frac + "0".repeat(Number(decimals))).slice(0, Number(decimals));
  return (BigInt(whole) * (10n ** decimals) + BigInt(padded)).toString();
}

async function call(path, method, body) {
  const r = await fetch(API + path, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-Wallet-Auth": signedAuthHeader(path),
    },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(15000),
  });
  const j = await r.json().catch(() => ({}));
  return { status: r.status, body: j };
}

console.log("=== Quillon micro-send + batch demo ===");
console.log(`from:    ${ME}`);
console.log(`to:      ${VIKTOR}`);
console.log();

// ─── Step 1: single send_signed of 0.01 QUGUSD with the work memo ────────────
const MEMO =
  "v10.11.0 ships 2026-05-21: /api/v1/integrity/balance-root live; " +
  "send_signed fee fix (no more ghost-confirms at 0); send_batch endpoint live; " +
  "BlockStreamBar wicked-cool block-counter in topbar; Crown & Ash agent-detail tab; " +
  "7 new Crown & Ash MCP tools (world/realm/join/propose_alliance/accept_treaty/action/turn). " +
  "Slint Windows EXE rebuilding in background. Cluster docker sync regression universal across v10.10.9/.10/.12/.13 (publisher dies). " +
  "Today we shipped data integrity + agentic gameplay.";

const sendBody = {
  from: ME,
  to: VIKTOR,
  amount: Number(toBase("0.01")), // u128 in body — JS Number is fine up to 2^53; 10^22 exceeds. Use string.
  memo: MEMO,
  token_type: "QUGUSD",
};
// IMPORTANT: u128 amount in JSON. 10^22 > 2^53 so we MUST send as a string in JSON.
// However serde u128 accepts both number and string in axum.
// To be safe, use string.
sendBody.amount = toBase("0.01");

console.log("--- step 1: send_signed 0.01 QUGUSD with memo (length=" + MEMO.length + " chars) ---");
const t0 = Date.now();
const r1 = await call("/api/v1/transactions/send_signed", "POST", sendBody);
const dur1 = Date.now() - t0;
console.log(`status: ${r1.status}  (${dur1}ms)`);
console.log(JSON.stringify(r1.body, null, 2));
console.log();

// ─── Step 2: send_batch of 5 × 0.0001 QUGUSD to myself (self-loop, no loss) ──
console.log("--- step 2: send_batch 5 × 0.0001 QUGUSD self-loop (TPS test) ---");
const N = 5;
const txs = [];
for (let i = 0; i < N; i++) {
  txs.push({
    to: ME,
    amount: toBase("0.0001"),
    token_type: "QUGUSD",
    memo: `batch test tx ${i + 1}/${N} from v10.11.0 send_batch endpoint`,
  });
}
const batchBody = { from: ME, transactions: txs };
const tb0 = Date.now();
const r2 = await call("/api/v1/transactions/send_batch", "POST", batchBody);
const durb = Date.now() - tb0;
console.log(`status: ${r2.status}  (${durb}ms total, ${(durb / N).toFixed(1)}ms/tx amortized)`);
console.log(JSON.stringify(r2.body, null, 2));

if (r2.body?.data) {
  const accepted = r2.body.data.accepted_count ?? r2.body.data.results?.filter(x => x.accepted)?.length ?? 0;
  const submission_ms = r2.body.data.submission_time_ms ?? durb;
  const tps = (N * 1000 / submission_ms).toFixed(1);
  console.log();
  console.log(`📊 batch TPS estimate: ${tps} tx/s  (${accepted}/${N} accepted, server reports ${submission_ms.toFixed(2)}ms submission)`);
}
