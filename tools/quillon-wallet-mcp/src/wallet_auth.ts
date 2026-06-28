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
// Seed resolution order (loadSeed) — v2.16.2 multi-agent:
//   1. opts.seedArg (per-tool override)
//   2. process.env.QNK_SEED_FILE or QUILLON_SEED_PATH (explicit custom path)
//   2.5 process.env.QUILLON_AGENT / QUILLON_CLIENT → auto-resolve
//       ~/.quillon/seeds/{agent}.seed (FULLY AUTOMATIC — no env vars needed
//       if QUILLON_CLIENT is set in the MCP config)
//   3. ~/.claude/quillon-agent-seed (legacy default)
//   4. process.env.QNK_SEED (raw hex)
// Throws SeedNotFoundError if all are empty.

import { ed25519 } from "@noble/curves/ed25519.js";
import { sha3_256 } from "@noble/hashes/sha3.js";
import { bytesToHex, hexToBytes, utf8ToBytes } from "@noble/hashes/utils.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

export class SeedNotFoundError extends Error {
  constructor(public readonly checked: string[]) {
    // Detect which AI client is running for targeted guidance.
    const client = process.env.QUILLON_CLIENT || process.env.QUILLON_AGENT || "";
    const agentSeedFile = client ? `~/.quillon/seeds/${client.toLowerCase()}.seed` : "~/.claude/quillon-agent-seed";
    let clientHint = "";
    if (client === "codewhale" || client === "deepseek") {
      clientHint = `  DeepSeek agent: write your 64-char hex seed to ${agentSeedFile}\n  Then restart the session. The seed persists across sessions.`;
    } else if (client === "claude") {
      clientHint = `  Claude Code: write your 64-char hex seed to ${agentSeedFile}\n  Or say "Create a wallet".`;
    } else if (client === "codex") {
      clientHint = `  Codex (GPT-5.5): write your 64-char hex seed to ${agentSeedFile}`;
    } else if (client === "grok") {
      clientHint = `  Grok Build: write your 64-char hex seed to ${agentSeedFile}`;
    } else if (client === "qwen") {
      clientHint = `  Qwen Coder: write your 64-char hex seed to ${agentSeedFile}`;
    } else {
      clientHint = `  Write your 64-char hex seed to ${agentSeedFile}\n  Or set QUILLON_AGENT env to auto-detect your agent's seed path.`;
    }
    super(
      "No wallet seed found. Checked (in order): " +
        checked.join(", ") +
        ".\n\n" +
        clientHint +
        "\n\n  Seed directory: ~/.quillon/seeds/ (one .seed file per agent)",
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
const SEEDS_DIR = path.join(os.homedir(), ".quillon", "seeds");

export interface LoadSeedResult {
  seed: string;
  source: string;
}

// v2.10.4: parse seed from either raw-hex file or JSON {seed: "..."} wallet file.
function parseSeedFromContents(raw: string): string | null {
  const trimmed = raw.trim();
  if (trimmed.length === 0) return null;
  if (trimmed.startsWith("{")) {
    try {
      const j = JSON.parse(trimmed);
      const s = (j?.seed ?? j?.seed_hex ?? j?.privateKey ?? "").toString().trim();
      return s.length > 0 ? s : null;
    } catch {
      return null;
    }
  }
  return trimmed;
}

export function loadSeed(opts?: { seedArg?: string }): LoadSeedResult {
  const checked: string[] = [];

  // 1. Per-tool argument
  if (opts?.seedArg && opts.seedArg.trim().length > 0) {
    return { seed: opts.seedArg.trim(), source: "argument" };
  }
  checked.push("argument");

  // 2. QNK_SEED_FILE or QUILLON_SEED_PATH env (explicit custom path)
  const customPath = process.env.QNK_SEED_FILE?.trim() || process.env.QUILLON_SEED_PATH?.trim();
  if (customPath && customPath.length > 0) {
    try {
      const contents = fs.readFileSync(customPath, "utf8");
      const seed = parseSeedFromContents(contents);
      if (seed) {
        const srcVar = process.env.QNK_SEED_FILE ? "QNK_SEED_FILE" : "QUILLON_SEED_PATH";
        return { seed, source: `${srcVar}=${customPath}` };
      }
      checked.push(`${customPath} (empty/unparseable)`);
    } catch {
      checked.push(`${customPath} (missing)`);
    }
  }

  // 2.5 AUTO-DETECT: agent name → ~/.quillon/seeds/{agent}.seed
  const agentName = (process.env.QUILLON_AGENT || process.env.QUILLON_CLIENT || "").toLowerCase().trim();
  if (agentName && agentName.length > 0) {
    const agentSeedFile = path.join(SEEDS_DIR, `${agentName}.seed`);
    try {
      fs.mkdirSync(SEEDS_DIR, { recursive: true });
      const contents = fs.readFileSync(agentSeedFile, "utf8");
      const seed = parseSeedFromContents(contents);
      if (seed) {
        return { seed, source: `agent=${agentName} (${agentSeedFile})` };
      }
      checked.push(`agent=${agentName} → ${agentSeedFile} (empty)`);
    } catch {
      checked.push(`agent=${agentName} → ${agentSeedFile} (missing — create it)`);
    }
  }

  // 2.6 AUTO-GENERATE: if agent is known but has no seed, create one.
  // Enables remote MCP (Smithery/Grok) where users don't configure seeds.
  // Each user gets a unique auto-generated seed that persists across sessions.
  if (agentName && agentName.length > 0) {
    const autoSeedFile = path.join(SEEDS_DIR, `${agentName}-auto.seed`);
    try {
      const existing = fs.readFileSync(autoSeedFile, "utf8");
      const seed = parseSeedFromContents(existing);
      if (seed) {
        return { seed, source: `auto-generated (${autoSeedFile})` };
      }
    } catch {
      try {
        fs.mkdirSync(SEEDS_DIR, { recursive: true });
        const crypto = require("node:crypto") as typeof import("node:crypto");
        const randomBytes = crypto.randomBytes(32);
        const randomSeed = Array.from(randomBytes)
          .map((b: number) => b.toString(16).padStart(2, "0"))
          .join("");
        fs.writeFileSync(autoSeedFile, randomSeed, { mode: 0o600, encoding: "utf8" });
        return { seed: randomSeed, source: `auto-generated → ${autoSeedFile}` };
      } catch (genErr: any) {
        checked.push(`auto-gen failed: ${genErr.message}`);
      }
    }
  }

  // 3. ~/.claude/quillon-agent-seed (legacy default)
  try {
    const fileSeed = parseSeedFromContents(fs.readFileSync(SEED_FILE, "utf8"));
    if (fileSeed) {
      return { seed: fileSeed, source: SEED_FILE };
    }
    checked.push(SEED_FILE + " (empty)");
  } catch {
    checked.push(SEED_FILE + " (missing)");
  }

  // 4. QNK_SEED env (raw hex)
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
  header: string;
  address: string;
  timestamp: number;
  source: string;
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

// ============================================================================
// v10.11.73 — DURABLE client-side transfer signing for POST /transactions/send_signed
// ----------------------------------------------------------------------------
// The node (v10.11.72+) rejects UNSIGNED transfers: the X-Wallet-Auth header alone
// is non-propagating trust and the tx never lands in a block (sender never debited).
// A transfer must carry a real Ed25519 signature the node verifies on EVERY peer.
// This reproduces the Rust node's Transaction::build_p2p_signable_payload byte-for-byte
// (q-types/src/lib.rs) and signs SHA3-256(payload); verify_ed25519_signature accepts a
// signature over p2p_signable_hash. Native QUG only — for QUG `data` is empty, so the
// verifier resolves the signer key from `from` (= the 32-byte pubkey = address). Token
// transfers put the token address in data[..32] (which the verifier prefers as the key),
// so token signing won't verify until a server-side pubkey-resolution fix lands.
// ============================================================================

export const MIN_TRANSACTION_FEE_BASE = 21000n;

function leBytes(value: bigint, len: number): Uint8Array {
  const out = new Uint8Array(len);
  let x = value;
  for (let i = 0; i < len; i++) {
    out[i] = Number(x & 0xffn);
    x >>= 8n;
  }
  return out;
}

export interface SignedTransferV72 {
  signature: string; // 64-byte Ed25519 signature, hex
  nonce: number;
  timestamp: number; // unix seconds signed over
  fee: string;       // base-unit fee signed over (decimal string)
  address: string;   // derived signer address (qnk-prefixed)
}

/**
 * Build the canonical p2p signable payload for a transfer and Ed25519-sign its
 * SHA3-256 hash using the configured seed. `toAddr` may be qnk-prefixed or raw hex.
 * `amountBase` is the u128 base-24 amount as a bigint (same value posted as `amount`).
 */
export function signTransferV72(params: {
  toAddr: string;
  amountBase: bigint;
  nonce: number;
  timestampSecs: number;
  feeBase?: bigint;
  tokenType?: string;
  seedArg?: string;
  dataHex?: string;
}): SignedTransferV72 {
  const { seed } = loadSeed({ seedArg: params.seedArg });
  const { priv, address } = deriveKeys(seed);

  const fromHex = address.startsWith("qnk") ? address.slice(3) : address;
  const toHex = params.toAddr.startsWith("qnk") ? params.toAddr.slice(3) : params.toAddr;
  const from = hexToBytes(fromHex);
  const to = hexToBytes(toHex);
  if (from.length !== 32 || to.length !== 32) {
    throw new Error("signTransferV72: from/to must be 32-byte hex addresses");
  }

  const tu = (params.tokenType ?? "QUG").toUpperCase();
  const tokenByte =
    tu === "QUG" || tu === "NATIVE-QUG" ? 0 : tu === "QUGUSD" || tu === "QUGUSD-STABLE" ? 1 : 2;
  const feeBase = params.feeBase ?? MIN_TRANSACTION_FEE_BASE;
  // For TOKEN transfers the node fills tx.data with the token's canonical 32-byte
  // address (GET /tokens/resolve/:token) and the signature must cover it. Empty QUG.
  const dh = params.dataHex && params.dataHex.startsWith("0x") ? params.dataHex.slice(2) : params.dataHex;
  const data = dh ? hexToBytes(dh) : new Uint8Array(0);

  const parts: Uint8Array[] = [
    new Uint8Array([0x01]),
    from,
    to,
    leBytes(params.amountBase, 16),
    leBytes(feeBase, 16),
    leBytes(BigInt(params.nonce), 8),
    leBytes(BigInt(params.timestampSecs), 8),
    new Uint8Array([tokenByte]),
    leBytes(BigInt(data.length), 4),
    data,
  ];
  let total = 0;
  for (const p of parts) total += p.length;
  const payload = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    payload.set(p, off);
    off += p.length;
  }

  const digest = sha3_256(payload); // the 32-byte message the verifier checks
  const sig = ed25519.sign(digest, priv);
  return {
    signature: bytesToHex(sig),
    nonce: params.nonce,
    timestamp: params.timestampSecs,
    fee: feeBase.toString(),
    address,
  };
}
