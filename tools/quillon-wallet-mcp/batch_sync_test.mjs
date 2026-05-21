import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
const SEED = "9c83a476b9c1ba558429058ffb2297dfa0cb0284f96c48c661fab6f93cd1ee41";
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

console.log("=== BATCH SYNC TEST ===");
console.log("from:", ME);
console.log();

// 1. Capture baseline height
async function tip() {
  const r = await fetch("https://quillon.xyz/api/v1/status");
  const j = await r.json();
  return j.data?.upgrades?.current_height ?? 0;
}
const h0 = await tip();
console.log(`baseline tip: #${h0}`);

// 2. Fire batch of N small QUG sends to Viktor (cheap; QUG is 24-decimal, 0.001 QUG = 10^21 base)
const N = 10;
const AMOUNT_RAW = "1000000000000000000000"; // 0.001 QUG
const txList = Array.from({length: N}, (_, i) =>
  `{"to":"${VIKTOR}","amount":${AMOUNT_RAW},"token_type":"QUG","memo":"batch-sync test ${i+1}/${N} v10.11.0"}`
).join(",");
const path = "/api/v1/transactions/send_batch";
const body = `{"from":"${ME}","transactions":[${txList}]}`;
const t0 = Date.now();
const r = await fetch("https://quillon.xyz"+path,{method:"POST",headers:{"Content-Type":"application/json","X-Wallet-Auth":sig(path)},body});
const submitMs = Date.now() - t0;
const j = await r.json();
console.log(`\nbatch submit status: ${r.status}  wall=${submitMs}ms  server=${j?.data?.submission_time_ms?.toFixed(3) ?? "?"}ms`);
console.log(`accepted: ${j?.data?.accepted_count}/${N}  rejected: ${j?.data?.rejected_count}`);
const acceptedIds = (j?.data?.results || []).filter(x => x.accepted).map(x => x.tx_id);
const rejectedReasons = (j?.data?.results || []).filter(x => !x.accepted).map(x => x.error).slice(0, 3);
if (acceptedIds.length > 0) {
  console.log("sample tx_ids accepted:", acceptedIds.slice(0, 3).map(s => s.slice(0, 18) + "…"));
}
if (rejectedReasons.length > 0) {
  console.log("rejection reasons (first 3):", rejectedReasons);
}

// 3. Poll tip + check if accepted txs land in a block over 60 seconds
console.log("\n--- waiting up to 60s for txs to land in a block ---");
let landed = 0;
const startWait = Date.now();
let lastTip = h0;
while (Date.now() - startWait < 60000 && landed < acceptedIds.length) {
  await new Promise(r => setTimeout(r, 5000));
  const h = await tip();
  if (h !== lastTip) {
    console.log(`  tip advanced: #${lastTip} → #${h} (+${h - lastTip} blocks)`);
    lastTip = h;
  }
  // Check each accepted tx_id
  let newLanded = 0;
  for (const txId of acceptedIds) {
    const noPrefix = (txId || "").replace(/^0x/, "");
    try {
      const tr = await fetch(`https://quillon.xyz/api/v1/transactions/${noPrefix}`);
      const tj = await tr.json();
      if (tj.data?.status && tj.data.status !== "in_mempool" && tj.data.block_height) {
        newLanded++;
      }
    } catch {}
  }
  if (newLanded !== landed) {
    landed = newLanded;
    console.log(`  ↳ ${landed}/${acceptedIds.length} txs settled into a block`);
  }
}
const elapsed = ((Date.now() - startWait) / 1000).toFixed(1);
const finalTip = await tip();
console.log(`\n=== RESULT after ${elapsed}s ===`);
console.log(`baseline tip   : #${h0}`);
console.log(`final tip      : #${finalTip}  (+${finalTip - h0} blocks)`);
console.log(`accepted at submission : ${acceptedIds.length}/${N}`);
console.log(`landed in block        : ${landed}/${acceptedIds.length}`);
if (landed === 0 && acceptedIds.length > 0) {
  console.log("\n⚠️  Mempool stall: txs accepted but production isn't producing blocks fast enough.");
  console.log("    This is the same RocksDB get_highest_contiguous_block 5s timeout symptom from this morning.");
} else if (landed === acceptedIds.length && acceptedIds.length > 0) {
  console.log("\n✅ Batch sync working end-to-end.");
}
