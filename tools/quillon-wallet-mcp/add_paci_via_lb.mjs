import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
import { readFileSync } from "node:fs";
const SEED = readFileSync("/root/.claude/quillon-agent-seed", "utf8").trim();
const PRIV = sha3_256(utf8ToBytes(SEED));
const PUB = ed25519.getPublicKey(PRIV);
const ME = "qnk" + bytesToHex(PUB);
const PACI = "qnkc722f70148faf360a342578f8c8d28da5dba4d4dc52e3d5ef4e08857767263d3";

function sig(path) {
  const ts = Math.floor(Date.now()/1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts); for (let i=0;i<8;i++){tsBuf[i]=Number(v&0xffn);v>>=8n;}
  const buf = new Uint8Array(40+path.length);
  buf.set(PUB,0); buf.set(tsBuf,32); buf.set(utf8ToBytes(path),40);
  return JSON.stringify({address:ME,timestamp:ts,scheme:"Ed25519",signature:bytesToHex(ed25519.sign(sha3_256(buf),PRIV)),public_key:bytesToHex(PUB)});
}

const POW24 = 10n**24n;
const A0 = (22054n*POW24).toString();
const A1 = (220540n*POW24).toString();
const path = "/api/v1/liquidity/add";
const body = `{"token0":"QUGUSD","token1":"${PACI}","amount0":"${A0}","amount1":"${A1}","provider":"${ME}"}`;

console.log("Submitting via quillon.xyz LB (HTTPS):");
const t0 = Date.now();
try {
  const r = await fetch("https://quillon.xyz" + path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Wallet-Auth": sig(path) },
    body,
    signal: AbortSignal.timeout(60000),
  });
  const txt = await r.text();
  console.log(`HTTP ${r.status}  ${Date.now()-t0}ms`);
  console.log(txt.slice(0, 1000));
} catch (e) {
  console.log(`FAILED ${Date.now()-t0}ms: ${e.message}`);
}
