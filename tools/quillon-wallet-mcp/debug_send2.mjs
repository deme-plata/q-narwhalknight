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
// Hand-build the body string so u128 stays as a raw JSON number literal.
function esc(s) { return s.replace(/\\/g,'\\\\').replace(/"/g,'\\"'); }
const MEMO = "v10.11.0 ships 2026-05-21: /api/v1/integrity/balance-root live; send_signed fee fix; send_batch endpoint live; BlockStreamBar topbar; Crown & Ash agent tab + 7 MCP tools; Slint Win EXE rebuilding. Data integrity + agentic gameplay shipped.";
const AMOUNT_RAW = "10000000000000000000000"; // 0.01 QUGUSD in 24-decimal base units

const path = "/api/v1/transactions/send_signed";
const bodyStr = `{"from":"${ME}","to":"${VIKTOR}","amount":${AMOUNT_RAW},"memo":"${esc(MEMO)}","token_type":"QUGUSD"}`;
const r = await fetch("https://quillon.xyz"+path,{method:"POST",headers:{"Content-Type":"application/json","X-Wallet-Auth":sig(path)},body:bodyStr});
console.log("send_signed status:", r.status);
console.log(await r.text());

// Batch test (5 self-loops)
const ME_TO = ME;
const BATCH_AMOUNT = "100000000000000000000"; // 0.0001 QUGUSD
const N = 5;
const txList = Array.from({length: N}, (_, i) =>
  `{"to":"${ME_TO}","amount":${BATCH_AMOUNT},"token_type":"QUGUSD","memo":"batch ${i+1}/${N}"}`
).join(",");
const batchPath = "/api/v1/transactions/send_batch";
const batchBody = `{"from":"${ME}","transactions":[${txList}]}`;
const t0 = Date.now();
const rb = await fetch("https://quillon.xyz"+batchPath,{method:"POST",headers:{"Content-Type":"application/json","X-Wallet-Auth":sig(batchPath)},body:batchBody});
const dur = Date.now() - t0;
console.log(`\nsend_batch status: ${rb.status}  (wall: ${dur}ms, ${(dur/N).toFixed(1)}ms/tx)`);
console.log(await rb.text());
