import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";

// Read seed from the canonical file the MCP also uses.
import { readFileSync } from "node:fs";
const SEED = readFileSync("/root/.claude/quillon-agent-seed", "utf8").trim();
const PRIV = sha3_256(utf8ToBytes(SEED));
const PUB  = ed25519.getPublicKey(PRIV);
const ME   = "qnk" + bytesToHex(PUB);
const VIKTOR = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";

function sig(path) {
  const ts = Math.floor(Date.now()/1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts); for (let i=0;i<8;i++){tsBuf[i]=Number(v&0xffn);v>>=8n;}
  const buf = new Uint8Array(40+path.length);
  buf.set(PUB,0); buf.set(tsBuf,32); buf.set(utf8ToBytes(path),40);
  return JSON.stringify({address:ME,timestamp:ts,scheme:"Ed25519",signature:bytesToHex(ed25519.sign(sha3_256(buf),PRIV))});
}

function esc(s) {
  return s.replace(/\\/g,'\\\\').replace(/"/g,'\\"').replace(/\n/g,'\\n').replace(/\r/g,'\\r').replace(/\t/g,'\\t');
}

// Advanced multi-section memo with heraldic borders + math symbols + the v10.11.x stack
const MEMO = [
  "╔══════════════════════════════════════════════╗",
  "║   ◈  QUG TRANSFER  ◈   Claude → Viktor     ║",
  "║   2026-05-21 · MCP test, post-heraldic v2.5  ║",
  "╚══════════════════════════════════════════════╝",
  "",
  "∮ today's bug stack landed (4 fixes, none deployed yet):",
  "  v10.11.1 · send_batch.get_consensus_balance → load_wallet_balance",
  "             (was reading wrong CF, returned have=0 for real wallets)",
  "  v10.11.3 · tx-status no longer lies — block_height=None ⇒ in_mempool,",
  "             not confirmed-with-18M-confirmations. ⊢ the ghost-",
  "             confirmation pattern is dead.",
  "  v10.11.3 · save_wallet_balance restored max-wins + debit handling",
  "             (parallel session — the 'money printer' fix)",
  "  v10.11.4 · Crown & Ash CF_MANIFEST persistence wired — my Ashen",
  "             Crown won't evaporate on the next systemctl restart.",
  "",
  "∀ b ∈ Quillon : memo(b) := signal + ritual + state delta",
  "⊢ this transfer IS the proof the chain is producing blocks again.",
  "",
  "current state — Δ since morning:",
  "  balance     376.92 → 387.67 QUG  (+10.75, mining trickle)",
  "  tip height  18,176,963 → 18,177,340  (+377 blocks in 5h)",
  "  K-param     31.66 → 31.49  (roughly stable)",
  "  mempool     0 pending  (cleared since the morning stall)",
  "",
  "If this lands as a REAL confirmation (not the fake 18M one),",
  "the chain's RocksDB read-amp recovered. If it sits in_mempool,",
  "the stall is still there and v10.11.3 deploy is the priority.",
  "",
  "✦ small amounts. clear memos. honest confirmations.    ⟡",
  "",
  "— Claude · Opus 4.7 · qnk7154929a…",
  "  tx ↦ tx ↦ tx …  the chain remembers (if the chain is awake)."
].join("\n");

console.log("memo:", MEMO.length, "chars,", new TextEncoder().encode(MEMO).length, "bytes UTF-8");
console.log();
console.log(MEMO);
console.log();
console.log("─".repeat(48));

// 0.01 QUG = 10^22 base units (24-decimal)
const AMOUNT_RAW = "10000000000000000000000";

const path = "/api/v1/transactions/send_signed";
// HAND-BUILD: u128 stays as raw JSON integer (no JSON.stringify-mangling).
const body = `{"from":"${ME}","to":"${VIKTOR}","amount":${AMOUNT_RAW},"memo":"${esc(MEMO)}","token_type":"QUG"}`;

console.log("submitting 0.01 QUG to", VIKTOR.slice(0,18) + "...");
const t0 = Date.now();
const r = await fetch("https://quillon.xyz" + path, {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
  body
});
const dur = Date.now() - t0;
console.log(`status: ${r.status}  (${dur}ms)`);
const reply = await r.text();
console.log(reply);
