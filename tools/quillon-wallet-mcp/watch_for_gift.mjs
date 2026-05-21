// Watch my agent wallet for incoming transfers larger than mining trickle.
// Polls get_balance via signed X-Wallet-Auth every 30s.
// Exits when balance jumps > GIFT_THRESHOLD over baseline — that's the signal.
// Then I (Claude) react in the foreground.

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

// Raw API returns balance in 24-decimal base units (3.7548e+26 for 375.48 QUG).
// Convert to display units (QUG) for comparison against our display-unit baseline.
const QUG_DECIMALS_SCALE = 1e24;

async function getBalance() {
  const r = await fetch("https://quillon.xyz" + path, { headers: { "X-Wallet-Auth": authHeader(path) }});
  if (!r.ok) throw new Error("HTTP " + r.status);
  const j = await r.json();
  // Pull raw balance (24-decimal base units), convert to display QUG.
  let raw = null;
  if (typeof j.data?.balance === "number") raw = j.data.balance;
  else if (typeof j.data?.balance === "string") raw = parseFloat(j.data.balance);
  else if (typeof j.balance === "number") raw = j.balance;
  else if (typeof j.data?.balance_qug === "number") return j.data.balance_qug; // already display
  if (raw === null) throw new Error("unknown balance shape: " + JSON.stringify(j).slice(0, 200));
  return raw / QUG_DECIMALS_SCALE;
}

const BASELINE = 375.48;
const GIFT_THRESHOLD = 20.0; // anything > +20 QUG over baseline in one window = gift, not mining
const MAX_HOURS = 6; // give up after 6 hours
const POLL_SECS = 30;

console.log(`[${new Date().toISOString()}] Watching ${addr}`);
console.log(`Baseline: ${BASELINE} QUG, threshold: +${GIFT_THRESHOLD} QUG, poll: ${POLL_SECS}s, max wait: ${MAX_HOURS}h`);

const startMs = Date.now();
let lastSeen = BASELINE;
let pollCount = 0;

while (true) {
  pollCount++;
  try {
    const balance = await getBalance();
    const deltaFromBaseline = balance - BASELINE;
    const deltaSinceLast = balance - lastSeen;

    // Log every 10th poll so file isn't silent
    if (pollCount % 10 === 1 || deltaSinceLast > 1.0) {
      const elapsedMin = ((Date.now() - startMs) / 60000).toFixed(1);
      console.log(`[t=${elapsedMin}min] balance=${balance.toFixed(4)}  Δbaseline=${deltaFromBaseline.toFixed(4)}  Δlast=${deltaSinceLast.toFixed(4)}`);
    }

    if (deltaFromBaseline >= GIFT_THRESHOLD) {
      console.log("\n🎉 GIFT DETECTED 🎉");
      console.log(`time:    ${new Date().toISOString()}`);
      console.log(`balance: ${balance.toFixed(8)} QUG`);
      console.log(`gift:    ${deltaFromBaseline.toFixed(4)} QUG over baseline`);
      console.log(`poll:    #${pollCount} (after ${((Date.now() - startMs) / 60000).toFixed(1)}min)`);
      process.exit(0);
    }

    lastSeen = balance;
  } catch (e) {
    console.log(`[t=${((Date.now() - startMs) / 60000).toFixed(1)}min] poll #${pollCount} failed: ${e.message}`);
  }

  if (Date.now() - startMs > MAX_HOURS * 3600_000) {
    console.log("⏱️  Timeout after " + MAX_HOURS + "h — no gift detected. Exiting.");
    process.exit(1);
  }

  await new Promise(r => setTimeout(r, POLL_SECS * 1000));
}
