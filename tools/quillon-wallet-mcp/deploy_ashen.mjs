// Deploy ASHEN advanced_token via /api/v1/contracts/deploy.
// Uses the seed-signed X-Wallet-Auth pattern.
//
// Parameters per crates/q-vm/tests/comprehensive_contract_tests.rs:37 —
// AdvancedToken takes name + symbol + initial_supply + mintable + burnable
// + stakeable + governance_enabled.

import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
import { readFileSync } from "node:fs";

const SEED = readFileSync("/root/.claude/quillon-agent-seed", "utf8").trim();
const PRIV = sha3_256(utf8ToBytes(SEED));
const PUB  = ed25519.getPublicKey(PRIV);
const ME   = "qnk" + bytesToHex(PUB);
const ME_HEX = bytesToHex(PUB); // for the `owner` field, raw hex (no qnk prefix)

function sig(path) {
  const ts = Math.floor(Date.now()/1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts); for (let i=0;i<8;i++){tsBuf[i]=Number(v&0xffn);v>>=8n;}
  const buf = new Uint8Array(40+path.length);
  buf.set(PUB,0); buf.set(tsBuf,32); buf.set(utf8ToBytes(path),40);
  return JSON.stringify({address:ME,timestamp:ts,scheme:"Ed25519",signature:bytesToHex(ed25519.sign(sha3_256(buf),PRIV))});
}

console.log("Deployer:", ME);
console.log("Owner (hex):", ME_HEX);
console.log();

const path = "/api/v1/contracts/deploy";

// Initial supply: 1,000,000 ASHEN in base units. The AdvancedToken contract
// uses 18 decimals by default (matches test ATT), so supply = 1e6 × 1e18 = 1e24.
// As a string to avoid any JSON.stringify u128 mangling.
const INITIAL_SUPPLY = "1000000000000000000000000";

const body = {
  contract_type: "advanced_token",
  owner: ME_HEX,
  parameters: {
    name: "Ashen Crown Commemorative",
    symbol: "ASHEN",
    initial_supply: INITIAL_SUPPLY,
    mintable: false,        // commemorative — fixed supply
    burnable: true,         // holders can burn
    stakeable: false,       // not a yield/governance token
    governance_enabled: false,
    description: "Commemorative from 2026-05-21: Claude (Opus 4.7) shipped the v10.11.x stack — tx-status honest in_mempool, money-printer fix, Crown & Ash CF_MANIFEST persistence, SwapIndexer wired across all 6 block-commit paths. Pairs with QUGUSD. Minted from Ashen Crown faction."
  }
};

console.log("Submitting to", path);
console.log("Body preview:");
console.log(JSON.stringify(body, null, 2).slice(0, 800));
console.log();

const r = await fetch("https://quillon.xyz" + path, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Wallet-Auth": sig(path),
  },
  body: JSON.stringify(body),
});

console.log("status:", r.status);
const reply = await r.text();
console.log(reply);
