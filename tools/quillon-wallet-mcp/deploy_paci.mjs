// One-off: deploy PACIOLI (PACI) as ContractType::AdvancedToken with
// reflection + staking enabled. The MCP's deploy_token hardcodes
// contract_type: "TOKEN" which only matches the basic SecureToken template;
// this script sends the full AdvancedToken parameter set.

import { ed25519 } from '@noble/curves/ed25519.js';
import { sha3_256 } from '@noble/hashes/sha3.js';
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils.js';
import { readFileSync } from 'node:fs';

const API = 'https://quillon.xyz/api/v1';
const SEED = readFileSync('/root/.claude/quillon-agent-seed', 'utf8').trim();
const priv = sha3_256(utf8ToBytes(SEED));
const pub = ed25519.getPublicKey(priv);
const addr = 'qnk' + bytesToHex(pub);
const ownerHex = bytesToHex(pub);                       // server wants raw hex, no qnk prefix

const path = '/api/v1/contracts/deploy';
const body = JSON.stringify({
  contract_type: 'advanced_token',
  owner: ownerHex,
  parameters: {
    name: 'Pacioli',
    symbol: 'PACI',
    decimals: 24,
    // 100M PACI total supply, scaled to 24 decimals.
    initial_supply: (100_000_000n * 10n ** 24n).toString(),
    description: 'Pacioli (PACI) — the first QUGUSD-paired token on Quillon Graph. Named for Luca Pacioli, the 1494 Franciscan friar whose Summa codified double-entry bookkeeping. Reflection rewards holders on every transfer; staking earns from network activity. See papers/five-mirrors-2026 for the historical thread.',
    // Feature toggles (advanced_token template defaults at orobit_smart_contracts.rs:752+).
    mintable: true,
    burnable: true,
    reflection: true,
    staking: true,
    governance: false,
    airdrops: true,
    upgrades: false,
    // Reflection / fee parameters — keep modest so trading isn't painful.
    reflection_fee_bps: 100,   // 1.0% of every transfer redistributes to holders
    burn_fee_bps: 0,
    liquidity_fee_bps: 0,
    max_tx_bps: 10_000,        // 100% — no anti-bot cap on first day
    max_wallet_bps: 10_000,    // 100% — no whale cap on first day
  },
  deployment_options: {
    test_deployment: false,
    auto_verify: true,
    enable_governance: false,
    enable_upgrades: false,
  },
});

// X-Wallet-Auth = Ed25519(SHA3(pub || ts_le8 || path)).
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

console.log(`[deploy-paci] from=${addr.slice(0, 20)}…  POSTing to ${API}/contracts/deploy ...`);
console.log(`[deploy-paci] body bytes=${body.length}`);

const t0 = Date.now();
const r = await fetch(API + '/contracts/deploy', {
  method: 'POST',
  headers: { 'X-Wallet-Auth': hdr, 'Content-Type': 'application/json' },
  body,
});
const text = await r.text();
console.log(`[deploy-paci] HTTP ${r.status} in ${Date.now() - t0}ms`);
console.log(text.slice(0, 1500));
