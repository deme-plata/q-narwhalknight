import { ed25519 } from '@noble/curves/ed25519.js';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils.js';
import { readFileSync } from 'node:fs';

const SEED = readFileSync('/root/.claude/quillon-agent-seed', 'utf8').trim();
const priv = sha3_256(utf8ToBytes(SEED));
const pub = ed25519.getPublicKey(priv);
const addr = 'qnk' + bytesToHex(pub);

const BACKENDS = [
  { name: 'quillon.xyz (MCP default — LB-routed)', base: 'https://quillon.xyz/api/v1' },
  { name: 'Epsilon direct (canonical 89.149.241.126)', base: 'http://89.149.241.126:8080/api/v1' },
  { name: 'Delta direct (fork 5.79.79.158)',          base: 'http://5.79.79.158:8080/api/v1' },
];

const path = '/api/v1/wallet/tokens';
const ts = Math.floor(Date.now() / 1000);
const tsBuf = new Uint8Array(8);
let v = BigInt(ts);
for (let i = 0; i < 8; i++) { tsBuf[i] = Number(v & 0xffn); v >>= 8n; }
const sigBuf = new Uint8Array(40 + path.length);
sigBuf.set(pub, 0);
sigBuf.set(tsBuf, 32);
sigBuf.set(utf8ToBytes(path), 40);
const hdr = JSON.stringify({
  address: addr,
  timestamp: ts,
  scheme: 'Ed25519',
  signature: bytesToHex(ed25519.sign(sha3_256(sigBuf), priv)),
  public_key: bytesToHex(pub),
});

console.log(`Probing addr ${addr.slice(0, 20)}… across 3 backends:`);
console.log();

for (const b of BACKENDS) {
  process.stdout.write(`── ${b.name}\n`);
  try {
    const r = await fetch(b.base + '/wallet/tokens', {
      headers: { 'X-Wallet-Auth': hdr, 'Content-Type': 'application/json' },
    });
    const text = await r.text();
    let parsed;
    try { parsed = JSON.parse(text); } catch { parsed = text; }
    process.stdout.write(`   HTTP ${r.status}\n`);
    if (typeof parsed === 'object') {
      // Find any non-zero token entries
      const list = parsed?.data?.tokens ?? parsed?.tokens ?? parsed?.data ?? parsed;
      if (Array.isArray(list)) {
        for (const e of list) {
          const sym = e.symbol ?? e.token ?? '?';
          const bal = e.balance ?? e.amount ?? '0';
          if (bal !== '0' && bal !== 0 && bal !== '' && bal !== null) {
            process.stdout.write(`   ${sym.padEnd(8)} balance: ${bal}\n`);
          }
        }
        if (list.length === 0) process.stdout.write('   (empty token list)\n');
      } else {
        process.stdout.write('   raw response: ' + JSON.stringify(parsed).slice(0, 300) + '\n');
      }
    } else {
      process.stdout.write('   raw: ' + text.slice(0, 200) + '\n');
    }
  } catch (e) {
    process.stdout.write(`   ERROR: ${e.message}\n`);
  }
  process.stdout.write('\n');
}

// Also check yesterday's swap via tx history
console.log('── Recent QUGUSD-related TXs (yesterday) — via /transactions/recent on quillon.xyz');
try {
  const r = await fetch(BACKENDS[0].base + '/transactions/recent?limit=200', {
    headers: { 'X-Wallet-Auth': hdr },
  });
  const j = await r.json();
  const txs = j?.data?.transactions ?? j?.transactions ?? j?.data ?? [];
  const mine = txs.filter(t => (t.from === addr || t.to === addr) && (
    (t.tx_type || '').toLowerCase().includes('swap') ||
    (t.token_type || '').toUpperCase() === 'QUGUSD' ||
    JSON.stringify(t).toUpperCase().includes('QUGUSD')
  ));
  console.log(`   Total recent txs touching this wallet: ${txs.filter(t => t.from === addr || t.to === addr).length}`);
  console.log(`   Swap-like or QUGUSD-tagged: ${mine.length}`);
  for (const t of mine.slice(0, 10)) {
    console.log(`     ${(t.timestamp ?? '?')}  ${(t.tx_type || t.transaction_type || '?')}  ${(t.amount ?? '?')}  ${(t.token_type || '?')}  hash=${(t.hash || t.tx_hash || '?').slice(0, 16)}`);
  }
} catch (e) {
  console.log(`   ERROR: ${e.message}`);
}
