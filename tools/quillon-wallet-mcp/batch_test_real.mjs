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

// Two batch tests at different N to see the TPS curve.
for (const N of [5, 25, 100]) {
  const BATCH_AMOUNT = "1000000000000000000000"; // 0.001 QUGUSD per tx
  const txList = Array.from({length: N}, (_, i) =>
    `{"to":"${VIKTOR}","amount":${BATCH_AMOUNT},"token_type":"QUGUSD","memo":"v10.11.0 batch ${i+1}/${N}"}`
  ).join(",");
  const path = "/api/v1/transactions/send_batch";
  const body = `{"from":"${ME}","transactions":[${txList}]}`;
  const t0 = Date.now();
  const r = await fetch("https://quillon.xyz"+path,{method:"POST",headers:{"Content-Type":"application/json","X-Wallet-Auth":sig(path)},body});
  const dur = Date.now() - t0;
  const j = await r.json();
  const acc = j?.data?.accepted_count ?? 0;
  const rej = j?.data?.rejected_count ?? 0;
  const srv_ms = j?.data?.submission_time_ms ?? null;
  const first_err = j?.data?.results?.find(x => !x.accepted)?.error ?? "(none)";
  const sample_tx = j?.data?.results?.find(x => x.accepted)?.tx_id ?? "(none accepted)";
  console.log(`N=${String(N).padStart(3)}  wall=${dur}ms  amortized=${(dur/N).toFixed(2)}ms/tx  ${acc}/${N} accepted  server=${srv_ms?.toFixed(2)}ms`);
  console.log(`         server_tps=${srv_ms ? (N*1000/srv_ms).toFixed(0) : '?'}  wall_tps=${(N*1000/dur).toFixed(0)}  sample_tx=${sample_tx?.slice(0,18)}…  first_rej=${first_err.slice(0,60)}`);
}
