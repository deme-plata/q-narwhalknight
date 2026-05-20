// quillon-wallet-mcp v2.0.0 — X-Wallet-Auth signing.
//
// One job: take a seed string, produce a signed `X-Wallet-Auth` header for a
// given API path. Mirrors the canonical pattern in raw_balance.mjs that's been
// proven against the live v10.9.55+ Rust server (q-api-server::wallet_auth at
// crates/q-api-server/src/wallet_auth.rs:209-237).
//
// Algorithm (must match server byte-for-byte):
//   priv      = SHA3-256(utf8(seed))                       // 32 bytes
//   pub       = Ed25519.publicKey(priv)                    // 32 bytes
//   address   = "qnk" + hex(pub)                           // string
//   ts_le     = timestamp as u64 little-endian             // 8 bytes
//   challenge = SHA3-256(pub || ts_le || utf8(path))       // 32 bytes
//   signature = Ed25519.sign(priv, challenge)              // 64 bytes
//   header    = JSON {address, timestamp, scheme:"Ed25519", signature:hex, public_key:hex}
//
// Seed resolution order (loadSeed):
//   1. opts.seedArg (per-tool override)
//   2. ~/.claude/quillon-agent-seed (file — same one used by the qwallet helper)
//   3. process.env.QNK_SEED
// Throws SeedNotFoundError if all three are empty.

import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

export class SeedNotFoundError extends Error {
  constructor(public readonly checked: string[]) {
    super(
      "No wallet seed found. Checked (in order): " +
        checked.join(", ") +
        ". Provide a `seed` argument, set QNK_SEED env, or write the 64-char hex seed to ~/.claude/quillon-agent-seed.",
    );
    this.name = "SeedNotFoundError";
  }
}

export class SignatureError extends Error {
  constructor(message: string, public readonly address?: string) {
    super(message);
    this.name = "SignatureError";
  }
}

const SEED_FILE = path.join(os.homedir(), ".claude", "quillon-agent-seed");

export interface LoadSeedResult {
  seed: string;
  source: string; // "argument" | path to file | "QNK_SEED env"
}

export function loadSeed(opts?: { seedArg?: string }): LoadSeedResult {
  const checked: string[] = [];

  // 1. Per-tool argument
  if (opts?.seedArg && opts.seedArg.trim().length > 0) {
    return { seed: opts.seedArg.trim(), source: "argument" };
  }
  checked.push("argument");

  // 2. ~/.claude/quillon-agent-seed
  try {
    const fileSeed = fs.readFileSync(SEED_FILE, "utf8").trim();
    if (fileSeed.length > 0) {
      return { seed: fileSeed, source: SEED_FILE };
    }
    checked.push(SEED_FILE + " (empty)");
  } catch {
    checked.push(SEED_FILE + " (missing)");
  }

  // 3. QNK_SEED env
  const envSeed = process.env.QNK_SEED?.trim();
  if (envSeed && envSeed.length > 0) {
    return { seed: envSeed, source: "QNK_SEED env" };
  }
  checked.push("QNK_SEED env");

  throw new SeedNotFoundError(checked);
}

export interface DerivedKeys {
  priv: Uint8Array;
  pub: Uint8Array;
  address: string;
}

// Memoize per seed string. The agent typically uses one wallet per session,
// so this hits on every call after the first.
const keyCache = new Map<string, DerivedKeys>();

export function deriveKeys(seed: string): DerivedKeys {
  const cached = keyCache.get(seed);
  if (cached) return cached;
  const priv = sha3_256(utf8ToBytes(seed));
  const pub = ed25519.getPublicKey(priv);
  const address = "qnk" + bytesToHex(pub);
  const result = { priv, pub, address };
  keyCache.set(seed, result);
  return result;
}

export interface SignedAuth {
  header: string; // value for the X-Wallet-Auth HTTP header
  address: string; // derived qnk... address
  timestamp: number; // unix seconds we signed with
  source: string; // where the seed came from (for diagnostics)
}

export function signXWalletAuth(
  reqPath: string,
  opts?: { seedArg?: string; timestamp?: number },
): SignedAuth {
  const { seed, source } = loadSeed({ seedArg: opts?.seedArg });
  const { priv, pub, address } = deriveKeys(seed);

  const ts = opts?.timestamp ?? Math.floor(Date.now() / 1000);
  const tsBuf = new Uint8Array(8);
  let v = BigInt(ts);
  for (let i = 0; i < 8; i++) {
    tsBuf[i] = Number(v & 0xffn);
    v >>= 8n;
  }

  const pathBytes = utf8ToBytes(reqPath);
  const buf = new Uint8Array(40 + pathBytes.length);
  buf.set(pub, 0);
  buf.set(tsBuf, 32);
  buf.set(pathBytes, 40);

  const challenge = sha3_256(buf);
  const signature = ed25519.sign(challenge, priv);

  const header = JSON.stringify({
    address,
    timestamp: ts,
    scheme: "Ed25519",
    signature: bytesToHex(signature),
    public_key: bytesToHex(pub),
  });

  return { header, address, timestamp: ts, source };
}
