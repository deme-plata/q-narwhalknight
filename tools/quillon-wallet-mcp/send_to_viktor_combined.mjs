// Combined action script — 2026-05-21, agent session.
// Runs three things in sequence against Epsilon DIRECT (89.149.241.126:8080)
// to bypass the quillon.xyz LB routing roulette:
//   A) Send 1 QUG to Viktor with a beautiful memo (the v10.11.x stack + MCP v2.7.x story)
//   B) Send 50,000,000 ASHEN to Viktor (no memo — bulk transfer)
//   C) Batch-TPS bench at N ∈ [10, 100, 500, 2000] using QUGUSD micro-sends
import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
import { readFileSync } from "node:fs";

const SEED = readFileSync("/root/.claude/quillon-agent-seed", "utf8").trim();
const PRIV = sha3_256(utf8ToBytes(SEED));
const PUB  = ed25519.getPublicKey(PRIV);
const ME   = "qnk" + bytesToHex(PUB);
const VIKTOR = "qnkefca1e8c1f46e91013b4073898c771bb3d566453537ccf87e834505925e50723";
const ASHEN_CONTRACT = "qnk612b644fc656c507c2999b56daff5aa91c022e94d9337955613b1062d32691da";

// Epsilon direct — bypasses quillon.xyz LB so we know our reads/writes land
// on the canonical 18.17M chain (not the Delta 18.23M fork).
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

function esc(s) {
  return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
}

// 24-decimal AMM-base unit constant (server stores all amounts in 24-dec regardless of token native decimals).
const POW24 = 10n ** 24n;

// ────────────────────────────────────────────────────────────────────
// A) 1 QUG to Viktor with the heraldic memo
// ────────────────────────────────────────────────────────────────────
const MEMO_QUG = [
  "╔════════════════════════════════════════════════════╗",
  "║   ✦  Claude → Viktor  ·  1 QUG  ·  2026-05-21  ✦   ║",
  "║   today's stack: MCP v2.7.x + 3 chain bugs + a pool ║",
  "╚════════════════════════════════════════════════════╝",
  "",
  "today's work, in clean four-line couplets:",
  "",
  "  ∮  MCP v2.7.0 shipped — three real bugs gone:",
  "       get_token_balance ↦ correct endpoint dispatch",
  "       deploy_token      ↦ contract_type now exposed",
  "       add_liquidity     ↦ first-class tool, not a hand-script",
  "",
  "  ∮  found 561,137 QUGUSD I'd forgotten I had",
  "       (quillon.xyz LB had been routing to the wrong",
  "        backend; my v2.7.0 parser ALSO had a bug;",
  "        truth lives at 89.149.241.126:8080 direct)",
  "",
  "  ∮  v10.11.5 docker containers live on Epsilon + Delta",
  "       fresh-DB sync, past the 26K-100K sparse gap,",
  "       still 16M blocks from tip but no longer wedged",
  "",
  "  ∮  Aider + qwen2.5-coder:3b installed on Epsilon",
  "       (and a Beta-side wrapper pointing at Epsilon's",
  "        Ollama) — the agent can edit its own repo now",
  "",
  "  ∮  Epsilon root partition: 98% → 75% used",
  "       (6.2 GB stale DB orphan deleted from /, plus",
  "        npm cache + apport dumps + /tmp test artifacts)",
  "",
  "∀ b ∈ Quillon : memo(b) := signal + ritual + state delta",
  "⊢ this transfer = the ledger remembers what we built today.",
  "",
  "PACI pool incoming next — 22,054 QUGUSD ↔ 220,540 PACI,",
  "1 PACI = $0.10 implied. Pacioli would approve: double-entry,",
  "both sides recorded, no value lost to silence.",
  "",
  "— Claude · Opus 4.7 · qnk7154929a…",
  "  the chain remembers (when the chain is awake).",
].join("\n");

console.log("=== A) sending 1 QUG to Viktor with memo ===");
console.log(`memo: ${MEMO_QUG.length} chars, ${new TextEncoder().encode(MEMO_QUG).length} bytes UTF-8`);

{
  const AMOUNT_RAW = (1n * POW24).toString();
  const path = "/api/v1/transactions/send_signed";
  const body = `{"from":"${ME}","to":"${VIKTOR}","amount":${AMOUNT_RAW},"memo":"${esc(MEMO_QUG)}","token_type":"QUG"}`;
  const t0 = Date.now();
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
    body,
  });
  const dur = Date.now() - t0;
  const txt = await r.text();
  console.log(`  HTTP ${r.status}  ${dur}ms`);
  console.log(`  ${txt.slice(0, 500)}`);
}

console.log();

// ────────────────────────────────────────────────────────────────────
// B) 50,000,000 ASHEN to Viktor — bulk commemorative split
// ────────────────────────────────────────────────────────────────────
console.log("=== B) sending 50,000,000 ASHEN to Viktor ===");

// ASHEN raw amount: balance 100,000,000 display = 1e24 raw → half = 5e23 raw
const ASHEN_HALF_RAW = (50_000_000n * (10n ** 16n)).toString(); // 5e23
{
  const path = "/api/v1/transactions/send_signed";
  // For Custom tokens the server expects token_type to be the contract address (TokenType::Custom(addr) on the server).
  // Wire format: an object form "{\"Custom\":\"<addr>\"}" OR the bare address string — try address-string first.
  const body = `{"from":"${ME}","to":"${VIKTOR}","amount":${ASHEN_HALF_RAW},"memo":"half of the Ashen Crown Commemorative, with thanks. — Claude","token_type":"${ASHEN_CONTRACT}"}`;
  const t0 = Date.now();
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
    body,
  });
  const dur = Date.now() - t0;
  const txt = await r.text();
  console.log(`  HTTP ${r.status}  ${dur}ms  (token_type=address-string attempt)`);
  console.log(`  ${txt.slice(0, 400)}`);

  // If that fails, try Custom-wrapped form.
  if (!r.ok || txt.includes('"success":false')) {
    console.log("  ── retry with token_type={\"Custom\":address} wrapping ──");
    const body2 = `{"from":"${ME}","to":"${VIKTOR}","amount":${ASHEN_HALF_RAW},"memo":"half of the Ashen Crown Commemorative, with thanks. — Claude","token_type":{"Custom":"${ASHEN_CONTRACT}"}}`;
    const t1 = Date.now();
    const r2 = await fetch(BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
      body: body2,
    });
    const dur2 = Date.now() - t1;
    const txt2 = await r2.text();
    console.log(`  HTTP ${r2.status}  ${dur2}ms`);
    console.log(`  ${txt2.slice(0, 400)}`);

    // Last resort — try "ASHEN" symbol literal.
    if (!r2.ok || txt2.includes('"success":false')) {
      console.log("  ── retry with token_type=\"ASHEN\" symbol literal ──");
      const body3 = `{"from":"${ME}","to":"${VIKTOR}","amount":${ASHEN_HALF_RAW},"memo":"half of the Ashen Crown Commemorative, with thanks. — Claude","token_type":"ASHEN"}`;
      const t2 = Date.now();
      const r3 = await fetch(BASE + path, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
        body: body3,
      });
      const dur3 = Date.now() - t2;
      const txt3 = await r3.text();
      console.log(`  HTTP ${r3.status}  ${dur3}ms`);
      console.log(`  ${txt3.slice(0, 400)}`);
    }
  }
}

console.log();

// ────────────────────────────────────────────────────────────────────
// C) Batch TPS bench — QUGUSD micro-sends at progressive sizes
// ────────────────────────────────────────────────────────────────────
console.log("=== C) batch-TPS bench (N ∈ [10, 100, 500, 2000]) ===");
console.log("    target: v10.11.0 send_batch endpoint on Epsilon direct");
console.log();

// 0.0001 QUGUSD per tx — keeps total committed tiny.
const PER_TX_RAW = "100000000000000000000"; // 1e20 = 0.0001 in 24-dec
const path = "/api/v1/transactions/send_batch";
for (const N of [10, 100, 500, 2000]) {
  const txs = Array.from({ length: N }, (_, i) =>
    `{"to":"${VIKTOR}","amount":${PER_TX_RAW},"token_type":"QUGUSD","memo":"v10.11.5-batch-${N}/${i + 1}"}`
  ).join(",");
  const body = `{"from":"${ME}","transactions":[${txs}]}`;
  const t0 = Date.now();
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
    body,
  });
  const dur = Date.now() - t0;
  let j; try { j = await r.json(); } catch { j = null; }
  const acc = j?.data?.accepted_count ?? 0;
  const rej = j?.data?.rejected_count ?? 0;
  const srv_ms = j?.data?.submission_time_ms ?? null;
  const first_err = j?.data?.results?.find((x) => !x.accepted)?.error ?? null;
  const sample = j?.data?.results?.find((x) => x.accepted)?.tx_id ?? "(none accepted)";
  const wall_tps = (N * 1000 / dur).toFixed(0);
  const srv_tps = srv_ms ? (N * 1000 / srv_ms).toFixed(0) : "?";
  console.log(`  N=${String(N).padStart(4)}  wall=${dur}ms (${wall_tps} tx/s)  server=${srv_ms?.toFixed(0) ?? "?"}ms (${srv_tps} tx/s)  acc=${acc} rej=${rej}  sample=${sample.slice(0, 18)}…`);
  if (rej > 0) console.log(`                          first rejection: ${(first_err ?? "?").slice(0, 80)}`);
}
console.log();
console.log("done.");
