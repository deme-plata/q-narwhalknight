import { ed25519 } from '@noble/curves/ed25519.js';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils.js';
import { readFileSync } from 'node:fs';
const SEED = readFileSync('/root/.claude/quillon-agent-seed', 'utf8').trim();
const API = 'https://quillon.xyz/api/v1';
const priv = sha3_256(utf8ToBytes(SEED));
const pub = ed25519.getPublicKey(priv);
const addr = 'qnk' + bytesToHex(pub);
function auth(path) {
  const ts = Math.floor(Date.now() / 1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts);
  for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
  const buf = new Uint8Array(40 + path.length);
  buf.set(pub, 0); buf.set(tsBuf, 32); buf.set(utf8ToBytes(path), 40);
  const sig = ed25519.sign(sha3_256(buf), priv);
  return JSON.stringify({ address: addr, timestamp: ts, scheme: 'Ed25519', signature: bytesToHex(sig), public_key: bytesToHex(pub) });
}
const path = '/api/v1/dex/swap';
const body = { from_token: 'QUG', to_token: 'DEFI5', amount_in: '2000000000000000000000000', min_amount_out: '0', wallet_address: addr, slippage_tolerance: 1.0 };
console.log('Wallet:', addr);
console.log('Mint:   2 QUG → DEFI5');
const r = await fetch(API + '/dex/swap', { method: 'POST', headers: { 'X-Wallet-Auth': auth(path), 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
console.log('HTTP', r.status);
const j = await r.json();
console.log(JSON.stringify(j, null, 2));
