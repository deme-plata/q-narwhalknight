import { ed25519 } from '@noble/curves/ed25519.js';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils.js';
import { readFileSync } from 'node:fs';

const SEED = readFileSync('/root/.claude/quillon-agent-seed', 'utf8').trim();
const priv = sha3_256(utf8ToBytes(SEED));
const pub = ed25519.getPublicKey(priv);
const addr = 'qnk' + bytesToHex(pub);

const path = '/api/v1/wallet/tokens';
const ts = Math.floor(Date.now() / 1000);
const tsBuf = new Uint8Array(8);
let v = BigInt(ts);
for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
const sigBuf = new Uint8Array(40 + path.length);
sigBuf.set(pub, 0); sigBuf.set(tsBuf, 32); sigBuf.set(utf8ToBytes(path), 40);
const hdr = JSON.stringify({
  address: addr, timestamp: ts, scheme: 'Ed25519',
  signature: bytesToHex(ed25519.sign(sha3_256(sigBuf), priv)),
  public_key: bytesToHex(pub),
});
const r = await fetch('http://89.149.241.126:8080/api/v1/wallet/tokens', { headers: { 'X-Wallet-Auth': hdr } });
const j = await r.json();
const t = j?.data?.tokens || {};
for (const [sym, info] of Object.entries(t)) {
  if (sym === 'ASHEN') {
    console.log('ASHEN entry:');
    console.log(JSON.stringify(info, null, 2));
  }
}
