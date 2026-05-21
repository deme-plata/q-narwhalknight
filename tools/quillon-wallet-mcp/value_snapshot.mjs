import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
import { readFileSync } from "node:fs";

const SEED = readFileSync("/root/.claude/quillon-agent-seed", "utf8").trim();
const PRIV = sha3_256(utf8ToBytes(SEED));
const PUB  = ed25519.getPublicKey(PRIV);
const ME   = "qnk" + bytesToHex(PUB);
const VIKTOR = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

function sigFor(addr, path) {
  // Note: sigFor only signs for OUR seed-derived address; for Viktor we hit
  // contract endpoints (no auth) since /wallet/tokens needs auth from the wallet itself.
  const ts = Math.floor(Date.now() / 1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts); for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
  const buf = new Uint8Array(40 + path.length);
  buf.set(PUB, 0); buf.set(tsBuf, 32); buf.set(utf8ToBytes(path), 40);
  return JSON.stringify({ address: addr, timestamp: ts, scheme: "Ed25519",
    signature: bytesToHex(ed25519.sign(sha3_256(buf), PRIV)), public_key: bytesToHex(PUB) });
}

const BASE = "http://89.149.241.126:8080";

// 1) My /wallet/tokens (signed)
const wtPath = "/api/v1/wallet/tokens";
const r = await fetch(BASE + wtPath, { headers: { "X-Wallet-Auth": sigFor(ME, wtPath) } });
const j = await r.json();
const t = j?.data?.tokens || {};

// 2) QUG price probe — quote 1 QUG → QUGUSD
const qr = await fetch(`${BASE}/api/v1/dex/quote?from=QUG&to=QUGUSD&amount=1`).catch(() => null);
let qugPrice = 0;
if (qr && qr.ok) {
  const qj = await qr.json();
  qugPrice = Number(qj?.data?.amount_out || qj?.amount_out || 0);
}

// 3) Pool details (the freshly-created QUG/PACI)
const pr = await fetch(`${BASE}/api/v1/liquidity/pools/pool-955ce42686604519cb0a54cd5d186f82`).catch(() => null);
const pjson = pr ? await pr.json() : null;

// 4) Viktor's ASHEN balance via contract endpoint
const ASHEN = "qnk612b644fc656c507c2999b56daff5aa91c022e94d9337955613b1062d32691da";
const PACI  = "qnkc722f70148faf360a342578f8c8d28da5dba4d4dc52e3d5ef4e08857767263d3";
async function contractBal(token, wallet) {
  const r = await fetch(`${BASE}/api/v1/contracts/${token}/balance/${wallet}`).catch(() => null);
  if (!r) return null;
  const j = await r.json();
  return j?.data?.balance ?? "0";
}
const viktorAshenRaw = await contractBal(ASHEN, VIKTOR);
const viktorPaciRaw  = await contractBal(PACI, VIKTOR);

// Display helpers
const fmt = (n, d = 2) => Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
function paciFromRaw24(raw) {
  // PACI in 24-decimal AMM-base store; display = raw / 10^24
  return Number(BigInt(raw) / (10n ** 18n)) / 1e6;  // safe-ish for our magnitudes
}
function ashenFromRaw24(raw) {
  return Number(BigInt(raw) / (10n ** 18n)) / 1e6;
}

console.log(`AGENT WALLET — ${ME}`);
console.log(`QUG price (1 QUG → QUGUSD): $${qugPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}`);
console.log();

let realUSD = 0;
console.log("Token holdings:");
for (const [sym, info] of Object.entries(t)) {
  const bal = Number(info.balance);
  const usd = Number(info.usd_value || 0);
  if (bal < 0.000001) continue;
  console.log(`  ${sym.padEnd(10)} ${fmt(bal, 4).padStart(22)}    usd=$${fmt(usd, 2).padStart(18)}    ${info.name || ""}`);
  if (sym === "QUG") realUSD += bal * qugPrice;
  else if (sym === "QUGUSD") realUSD += bal;
}
console.log();

if (pjson?.data) {
  const p = pjson.data;
  const r0 = Number(BigInt(p.reserve0) / (10n ** 18n)) / 1e6;
  const r1 = Number(BigInt(p.reserve1) / (10n ** 18n)) / 1e6;
  console.log("MY POOL — pool-955ce4…186f82 (QUG ↔ PACI)");
  console.log(`  reserve0: ${fmt(r0)} QUG    (≈ $${fmt(r0 * qugPrice)})`);
  console.log(`  reserve1: ${fmt(r1)} PACI    (implied $${fmt(r0 * qugPrice / r1, 4)}/PACI)`);
  console.log(`  pool TVL: ~$${fmt(r0 * qugPrice * 2)} (k = r0*r1 = ${(r0 * r1).toExponential(2)})`);
  console.log();
}

console.log(`VIKTOR — what I sent today:`);
console.log(`  1.0 QUG (memo tx)      ≈ $${qugPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}`);
console.log(`  ASHEN balance:           ${fmt(ashenFromRaw24(viktorAshenRaw), 4)}    (10M = $10M @ self-priced $1, but no market)`);
console.log(`  PACI balance:            ${fmt(paciFromRaw24(viktorPaciRaw), 4)}    (would be $${fmt(paciFromRaw24(viktorPaciRaw) * 0.10)} @ pool-implied $0.10/PACI)`);
console.log();

console.log("REAL CASH-EQUIVALENT VALUE (only QUG + QUGUSD, no self-priced tokens):");
console.log(`  agent wallet:  $${fmt(realUSD)}`);
console.log();
console.log("PAPER VALUE (everything at self-set prices):");
const paciInWallet = Number(t.PACI?.balance || 0) * 0.10;
const ashenInWallet = Number(t.ASHEN?.balance || 0) * 0;  // no price — $0
console.log(`  + PACI @ $0.10 (pool-implied):   $${fmt(paciInWallet)}`);
console.log(`  + ASHEN @ $0  (no market):        $0`);
console.log(`  = paper total:                    $${fmt(realUSD + paciInWallet + ashenInWallet)}`);
