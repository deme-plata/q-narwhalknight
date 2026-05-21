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

// Try string amount first
const path = "/api/v1/transactions/send_signed";
const body1 = {from:ME, to:VIKTOR, amount: "10000000000000000000000", memo:"test", token_type:"QUGUSD"};
const r1 = await fetch("https://quillon.xyz"+path,{method:"POST",headers:{"Content-Type":"application/json","X-Wallet-Auth":sig(path)},body:JSON.stringify(body1)});
console.log("[string amount] status:", r1.status, " body:", (await r1.text()).slice(0,400));

// Try numeric amount (will lose precision but tests handler)
const body2 = {from:ME, to:VIKTOR, amount: 10000000000000000000000, memo:"test", token_type:"QUGUSD"};
const r2 = await fetch("https://quillon.xyz"+path,{method:"POST",headers:{"Content-Type":"application/json","X-Wallet-Auth":sig(path)},body:JSON.stringify(body2)});
console.log("[number amount] status:", r2.status, " body:", (await r2.text()).slice(0,400));
