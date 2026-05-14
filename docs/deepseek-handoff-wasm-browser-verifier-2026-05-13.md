# DeepSeek/Claude Code Agent Handoff — WASM Browser Verifier for Recursive Proof Bootstrap

**Date:** 2026-05-13
**Project:** Quillon Graph — Live Mainnet, ~$2 B USD Market Cap
**Owner track:** Claude Code agent + Beta engineer (NOT DeepSeek's BLAKE3 critical path)
**Depends on:** Phase 2 Nova IVC wrapper completion (months out). Scaffolding can land NOW.
**Companion:** `docs/parallel-work-coordination-deepseek-claude-2026-05-13.md` (Track E)

---

# 🚨 SCOPE: WASM BROWSER VERIFIER ONLY

This handoff is for the **verifier** side of the recursive proof bootstrap — specifically the **browser/JS-libp2p** path. It is NOT:

- The prover. Provers run on genesis nodes (Epsilon, Beta, Gamma, Delta) — full-fat Rust binaries with the proving key in memory.
- The native node verifier. That's already going to be a Rust binary (the verifier function lives in the same crate as the prover).
- A new SNARK construction. Same Nova/lattice scheme as the native verifier — just compiled to WASM and exposed via JS bindings.

The deliverable: a JS-libp2p browser wallet can fetch `π_tip` from `GET /api/v1/proof/tip`, verify it locally in ≤ 250 ms on a 2024 laptop, cache the verified state-root in IndexedDB, and treat the chain tip as cryptographically trusted — without trusting any specific server or signer.

---

# WHY THIS MATTERS

Right now, the browser wallet at `quillon.xyz` trusts the bootstrap server's `state_root` because it has no way to verify it independently. If a malicious operator served a wrong state-root through `/api/v1/status`, the wallet would happily believe it. The recursive proof bootstrap closes that gap — but only if the wallet can actually verify the proof. WASM is the only way to get the verifier into the browser.

Without WASM verification, the whitepaper's claim of "trustless light-client" is incomplete for the largest user surface (browser wallet users). A non-browser-verified proof bootstrap is just "trust an HTTP server's signature" with extra cryptographic steps.

---

# WHAT EXISTS TODAY

## Browser wallet (production)

- **Path:** `gui/quantum-wallet/`
- **Stack:** React + TypeScript + Vite + js-libp2p
- **P2P deps already loaded:** `@chainsafe/libp2p-gossipsub`, `@chainsafe/libp2p-noise`, `@chainsafe/libp2p-yamux`, `@libp2p/bootstrap`, `@libp2p/kad-dht`, etc.
- **No WASM verifier dep yet.** No `@wasm-bindgen-...` package, no `q-ivc-verifier-wasm` crate.

## Other relevant scaffolding

- `gui/quantum-wallet/src/libp2p/` has the entire P2P client — `blockCache.ts`, `blockRequest.ts`, `handshake.ts`, etc.
- `tools/quillon-wallet-mcp/` is the Node.js MCP server that bridges Claude to the wallet. Has TypeScript build pipeline already (`tsc`).
- No existing WASM module in the browser wallet. Pure JS today.

## Backend (still to come, but specced)

- `GET /api/v1/proof/tip` (Blueprint 5 in `docs/blueprints-ivc-snark-2026-05-13.md`) — returns the proof bytes for the WASM verifier to consume.
- Will be added once Nova wrapper lands.

---

# THE TWO-PHASE SCAFFOLDING APPROACH

Because the actual recursive proof system (Nova) is months out, but the WASM build pipeline + JS API + IndexedDB caching can be built TODAY, we deliver in two phases:

## Phase 1 — scaffolding lands now (this week, ~3 days)

- Create `crates/q-ivc-verifier-wasm/` — a Rust crate that compiles to wasm32-unknown-unknown
- Use `wasm-bindgen` to expose a single function: `verify_proof_bytes(state_root: Uint8Array, height: u64, proof_bytes: Uint8Array) -> bool`
- **The verifier function is a no-op placeholder for Phase 1.** It returns `true` if `proof_bytes.length > 0` and `false` otherwise. This lets the JS side wire everything up without depending on Nova.
- TypeScript wrapper at `gui/quantum-wallet/src/ivc/` exposing a clean API:
  - `loadVerifier(): Promise<Verifier>` — async WASM init
  - `Verifier.verify(stateRoot, height, proofBytes): Promise<boolean>`
  - `cacheVerifiedTip(stateRoot, height): void` — IndexedDB write
  - `getCachedTip(): Promise<VerifiedTip | null>` — IndexedDB read with TTL
- Demo HTML page that fetches a fake proof, verifies it (placeholder), shows "✓ verified in N ms"

**Goal of Phase 1:** wire pipeline works end-to-end. Build artifact exists. JS API stable. Cache works. The verify function lies in a known place where Phase 2 swaps in real Nova verification.

## Phase 2 — real Nova verification (months out, after Nova IVC wrapper)

- Replace the placeholder `verify_proof_bytes` body with `nova_snark::RecursiveSNARK::verify(...)` (or whichever crate is chosen)
- Compile public parameters into the WASM binary (~5 MB) OR load on first launch + cache in IndexedDB
- Validate ≤ 250 ms verification on commodity laptops via a benchmark page
- Wire into the wallet's bootstrap path: on first launch, fetch tip + proof, verify, cache, treat as authoritative

## Phase 3 — lattice migration (much later, Phase 4 in the main roadmap)

- Swap Nova for LatticeFold / LaBRADOR / Greyhound
- WASM binary size may grow (lattice crypto is heavier) but proof verification stays ≤ 250 ms target
- The JS API does NOT change. Same `verify(stateRoot, height, proofBytes)` signature.

---

# § A — PHASE 1 DELIVERABLES (READY TO START TODAY)

## A.1 — New Rust crate `crates/q-ivc-verifier-wasm/`

### `crates/q-ivc-verifier-wasm/Cargo.toml`

```toml
[package]
name = "q-ivc-verifier-wasm"
version = "0.1.0"
edition = "2021"
description = "WASM-bindgen wrapper for the recursive proof verifier (browser wallets)"
license = "Apache-2.0"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console", "Performance", "Window"] }
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"

# Phase 1: no actual crypto deps. Phase 2 adds nova-snark or arkworks/nova.
# Phase 4: swaps for lattice scheme dep.
hex = "0.4"

[profile.release]
opt-level = "z"        # size-optimize the wasm output
lto = true
codegen-units = 1
panic = "abort"
strip = true

[lints.rust]
unsafe_code = "forbid"
```

### `crates/q-ivc-verifier-wasm/src/lib.rs`

```rust
//! WASM bindings for the Quillon Graph recursive proof verifier (browser wallets).
//!
//! Phase 1 (this file): scaffolding only. `verify_proof_bytes` is a placeholder
//! that returns true for any non-empty proof. The Rust crate compiles to WASM
//! and is exposed to JS via wasm-bindgen. The wire-protocol shape and the JS
//! API are stable from day 1; only the body of `verify_proof_bytes` changes
//! when Nova lands in Phase 2.
//!
//! Phase 2: replace the body with `nova_snark::RecursiveSNARK::verify(...)`
//! against pre-loaded public parameters.
//!
//! Phase 4: swap Nova for the chosen lattice scheme. JS API unchanged.

#![forbid(unsafe_code)]

use wasm_bindgen::prelude::*;

/// Verifier version. Bumped when the underlying proof system changes.
/// Phase 1: `"placeholder-v0"`.
/// Phase 2: `"nova-bn254-v1"`.
/// Phase 4: `"latticefold-modulesis-v1"` (or chosen lattice variant).
#[wasm_bindgen]
pub fn verifier_version() -> String {
    "placeholder-v0".to_string()
}

/// Verify a recursive proof against an expected state-root and tip height.
///
/// Inputs:
/// * `state_root_bytes` — 32 bytes (BLAKE3 SMT root v2)
/// * `tip_height` — u64 from the block header
/// * `proof_bytes` — serialized recursive proof
///
/// Returns true iff the proof verifies against `state_root_bytes` at `tip_height`.
///
/// Phase 1 PLACEHOLDER: returns true if proof_bytes is non-empty.
/// Replace the body in Phase 2 with real Nova verification.
#[wasm_bindgen]
pub fn verify_proof_bytes(
    state_root_bytes: &[u8],
    tip_height: u64,
    proof_bytes: &[u8],
) -> bool {
    // Defensive input checks — these stay regardless of Phase 1/2/4.
    if state_root_bytes.len() != 32 {
        return false;
    }
    if proof_bytes.is_empty() {
        return false;
    }
    let _ = tip_height; // currently unused; Phase 2 uses it as public input

    // PHASE 2 TODO: replace this line with real Nova verification.
    // let recursive_snark: nova_snark::RecursiveSNARK<...> =
    //     bincode::deserialize(proof_bytes).map_err(...)?;
    // let public_input = pack_state_root(state_root_bytes);
    // recursive_snark.verify(&PUBLIC_PARAMS, tip_height as usize, public_input)
    //     .map(|_| true)
    //     .unwrap_or(false)
    true
}

/// Hex-encode a byte slice. Helper exposed for JS-side debugging.
#[wasm_bindgen]
pub fn hex_encode(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Decode a hex string to bytes. Helper for JS-side use.
/// Returns empty Vec on invalid input.
#[wasm_bindgen]
pub fn hex_decode(s: &str) -> Vec<u8> {
    hex::decode(s).unwrap_or_default()
}

/// Quick health check the JS side can call after WASM load.
#[wasm_bindgen]
pub fn verifier_ready() -> bool {
    true
}
```

### Add the crate to the workspace

In `Cargo.toml` (root, workspace members list around line 17):

```toml
members = [
    ...existing entries...
    "crates/q-ivc-verifier-wasm",
]
```

### Build command (Phase 1)

```bash
cd /opt/orobit/shared/q-narwhalknight/crates/q-ivc-verifier-wasm
cargo install wasm-pack          # one-time, if not installed
wasm-pack build --target web --release
# Output: pkg/ directory with q_ivc_verifier_wasm.wasm + q_ivc_verifier_wasm.js
```

### Distribution

Copy the build output into the wallet's static directory:

```bash
cp -r pkg/* /opt/orobit/shared/q-narwhalknight/gui/quantum-wallet/src/ivc/wasm/
```

Or wire as an npm dependency via `wasm-pack publish` — for Phase 1 just copy directly.

## A.2 — TypeScript wrapper `gui/quantum-wallet/src/ivc/`

### File structure

```
gui/quantum-wallet/src/ivc/
├── README.md                  # what this is + how to use it
├── wasm/                      # built from crates/q-ivc-verifier-wasm
│   ├── q_ivc_verifier_wasm.js
│   ├── q_ivc_verifier_wasm.wasm
│   └── q_ivc_verifier_wasm.d.ts
├── verifier.ts                # high-level Verifier class
├── cache.ts                   # IndexedDB cache for verified state-roots
├── bootstrap.ts               # one-shot: fetch proof from /api/v1/proof/tip + verify
└── types.ts                   # shared types
```

### `types.ts`

```typescript
export interface ProofTipResponse {
  tip_height: number;
  state_root: string;            // hex, 0x prefix
  block_header: {
    height: number;
    state_root: string;
    parent_hash: string;
    timestamp: number;
    proposer: string;
    [key: string]: unknown;
  };
  proof_version: string;          // "placeholder-v0" | "nova-bn254-v1" | ...
  proof_size_bytes: number;
  proof_b64: string;              // base64-encoded proof bytes
}

export interface VerifiedTip {
  state_root_hex: string;
  tip_height: number;
  verified_at: number;            // unix epoch ms
  verify_duration_ms: number;
  proof_version: string;
}

export interface VerifyResult {
  valid: boolean;
  duration_ms: number;
  error?: string;
}
```

### `verifier.ts`

```typescript
import init, {
  verify_proof_bytes,
  verifier_version,
  verifier_ready,
} from "./wasm/q_ivc_verifier_wasm.js";
import type { VerifyResult } from "./types";

let initialized = false;

/**
 * Load and initialize the WASM verifier. Idempotent — safe to call multiple times.
 * Returns the proof-system version string once ready.
 */
export async function loadVerifier(): Promise<string> {
  if (initialized) {
    return verifier_version();
  }
  await init();
  if (!verifier_ready()) {
    throw new Error("WASM verifier failed self-check");
  }
  initialized = true;
  return verifier_version();
}

/**
 * Verify a recursive proof against an expected state-root and height.
 * Returns { valid, duration_ms, error? }.
 */
export async function verifyProof(
  stateRootHex: string,
  tipHeight: number,
  proofBytes: Uint8Array,
): Promise<VerifyResult> {
  if (!initialized) {
    return { valid: false, duration_ms: 0, error: "verifier not initialized" };
  }
  const stateRoot = hexToBytes(stateRootHex);
  if (stateRoot.length !== 32) {
    return { valid: false, duration_ms: 0, error: "state_root must be 32 bytes" };
  }
  const start = performance.now();
  const valid = verify_proof_bytes(stateRoot, BigInt(tipHeight), proofBytes);
  const duration_ms = performance.now() - start;
  return { valid, duration_ms };
}

function hexToBytes(hex: string): Uint8Array {
  const clean = hex.startsWith("0x") ? hex.slice(2) : hex;
  if (clean.length % 2 !== 0) return new Uint8Array(0);
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(clean.substr(i * 2, 2), 16);
  }
  return out;
}
```

### `cache.ts`

```typescript
import type { VerifiedTip } from "./types";

const DB_NAME = "quillon-ivc-cache";
const STORE = "verified-tips";
const DB_VERSION = 1;

export const CACHE_TTL_MS = 5 * 60 * 1000; // 5-minute soft TTL

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "state_root_hex" });
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

export async function cacheVerifiedTip(tip: VerifiedTip): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(tip);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function getMostRecentVerifiedTip(): Promise<VerifiedTip | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).getAll();
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const all = (req.result || []) as VerifiedTip[];
      if (all.length === 0) {
        resolve(null);
        return;
      }
      // Return the highest-height tip
      all.sort((a, b) => b.tip_height - a.tip_height);
      resolve(all[0]);
    };
  });
}

export async function isFreshCache(tip: VerifiedTip): boolean {
  return Date.now() - tip.verified_at < CACHE_TTL_MS;
}
```

### `bootstrap.ts`

```typescript
import { loadVerifier, verifyProof } from "./verifier";
import { cacheVerifiedTip, getMostRecentVerifiedTip, isFreshCache } from "./cache";
import type { ProofTipResponse, VerifiedTip } from "./types";

const PROOF_TIP_ENDPOINT = "/api/v1/proof/tip";
const PEER_FALLBACK = "https://quillon.xyz";

/**
 * Bootstrap the wallet to a cryptographically verified tip.
 *
 * Tries cache first (fresh), then fetches the proof from a peer, verifies
 * locally in-browser via WASM, and caches the verified state-root.
 *
 * Returns the verified tip or throws if no usable proof can be obtained.
 */
export async function bootstrapVerifiedTip(): Promise<VerifiedTip> {
  await loadVerifier();

  // 1. Try cache
  const cached = await getMostRecentVerifiedTip();
  if (cached && (await isFreshCache(cached))) {
    return cached;
  }

  // 2. Fetch from peer
  const url = `${PEER_FALLBACK}${PROOF_TIP_ENDPOINT}`;
  const resp = await fetch(url, { cache: "no-cache" });
  if (!resp.ok) {
    throw new Error(`fetch ${url}: HTTP ${resp.status}`);
  }
  const body = (await resp.json()) as ProofTipResponse;
  if (!body.proof_b64 || !body.state_root || typeof body.tip_height !== "number") {
    throw new Error("malformed /api/v1/proof/tip response");
  }
  const proofBytes = base64ToBytes(body.proof_b64);

  // 3. Verify
  const result = await verifyProof(body.state_root, body.tip_height, proofBytes);
  if (!result.valid) {
    throw new Error(`proof verification failed: ${result.error ?? "unknown"}`);
  }

  // 4. Cache + return
  const verified: VerifiedTip = {
    state_root_hex: body.state_root,
    tip_height: body.tip_height,
    verified_at: Date.now(),
    verify_duration_ms: result.duration_ms,
    proof_version: body.proof_version,
  };
  await cacheVerifiedTip(verified);
  return verified;
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}
```

### `README.md`

```markdown
# Quillon Graph IVC Browser Verifier

WASM-based recursive-proof verifier for js-libp2p browser wallets.

## Status

- **Phase 1 (current):** scaffolding with a placeholder verifier. WASM build pipeline
  works, JS API is stable, IndexedDB caching is implemented. The verify function
  is a no-op stub that returns true for any non-empty proof.
- **Phase 2 (months out):** real Nova/BN254 verification. Replace the body of
  `verify_proof_bytes` in `crates/q-ivc-verifier-wasm/src/lib.rs`. Public parameters
  bundled into the WASM binary or loaded once and cached in IndexedDB.
- **Phase 4 (~2028):** swap Nova for lattice scheme (LatticeFold / LaBRADOR /
  Greyhound). JS API unchanged. Same `verify(stateRoot, height, proofBytes)`.

## Usage

```typescript
import { bootstrapVerifiedTip } from "./ivc/bootstrap";

const tip = await bootstrapVerifiedTip();
console.log(`verified tip ${tip.tip_height} in ${tip.verify_duration_ms} ms`);
console.log(`state_root: ${tip.state_root_hex}`);
```

## Build

```bash
cd crates/q-ivc-verifier-wasm
wasm-pack build --target web --release
cp -r pkg/* ../../gui/quantum-wallet/src/ivc/wasm/
```

## Verifier version detection

`loadVerifier()` returns `"placeholder-v0"` in Phase 1, `"nova-bn254-v1"` in Phase 2.
Wallets can branch on this to show different UI ("⚠️ placeholder verifier — do not trust"
in dev vs ✅ verified in prod).
```

## A.3 — Demo page

Create a simple demo HTML at `gui/quantum-wallet/public/ivc-demo.html` that loads the verifier, fetches a hardcoded proof body (or runs against the placeholder), shows the verify result and duration. Useful for sanity-checking the pipeline.

---

# § B — PHASE 1 ACCEPTANCE CRITERIA

A PR that delivers Phase 1 must satisfy:

- [ ] `crates/q-ivc-verifier-wasm/` exists with `Cargo.toml` + `src/lib.rs` as above
- [ ] Crate is added to root workspace `members` list
- [ ] `cargo build --release --target wasm32-unknown-unknown --package q-ivc-verifier-wasm` succeeds (no `wasm-pack` required to verify)
- [ ] `wasm-pack build --target web --release` produces `pkg/` with `.wasm` + `.js` + `.d.ts`
- [ ] `pkg/q_ivc_verifier_wasm.wasm` is **under 100 KB** (Phase 1 has no crypto deps so should be tiny — closer to 20-40 KB)
- [ ] `gui/quantum-wallet/src/ivc/` files exist as specced
- [ ] `npm run build` in `gui/quantum-wallet/` completes clean with the new ivc module imported
- [ ] Demo page loads, shows `verifier_version()` returning `"placeholder-v0"`
- [ ] Demo page successfully calls `verifyProof("0x" + "00".repeat(32), 0, new Uint8Array([1,2,3]))` and gets `valid: true`
- [ ] Demo page successfully calls `verifyProof("0x" + "00".repeat(32), 0, new Uint8Array())` and gets `valid: false` (empty proof rejected)
- [ ] `cacheVerifiedTip` + `getMostRecentVerifiedTip` round-trips in IndexedDB

If any checkbox fails, the PR is rejected.

---

# § C — WHAT THIS UNLOCKS

Once Phase 1 lands:

1. **Wallet has a placeholder bootstrap path** that fetches `/api/v1/proof/tip` and runs `verifyProof()`. The actual security comes later, but the wire is in place.
2. **Phase 2 is a single-file change** in `lib.rs` — no JS or TypeScript work needed once Nova lands.
3. **Wallet UI can branch on `verifier_version()`** to show different trust states. While `placeholder-v0` is active, the UI can warn users "⚠️ proof verification not yet active." Once `nova-bn254-v1` is active, the UI shows the green ✅ "verified by recursive zk-SNARK in 187 ms" state from Blueprint 7's FAST-READY banner.
4. **Demonstrates the architecture to users + reviewers** — the whitepaper's claim of "trustless browser bootstrap" has tangible code behind it.

---

# § D — WHAT THIS DOES NOT DO

- **No real cryptographic security in Phase 1.** Placeholder returns true for any non-empty proof. Anyone could feed garbage bytes and pass. Production deploy of Phase 1 alone is **not** a substitute for the existing checkpoint-bootstrap trust model. Document this loudly in the README + UI.
- **No public parameters in Phase 1.** Phase 2 needs to figure out whether to bundle ~5 MB of Nova public params into the WASM or load them on first launch. Loading is cleaner but slower first-time UX.
- **No mobile-first optimization yet.** Phase 1 targets desktop browsers. Mobile Safari has known WASM threading quirks; revisit in Phase 2.
- **No Service Worker prefetch.** Once Nova lands, the wallet could prefetch + verify `π_tip` in a service worker so it's already verified when the user opens the app. Future enhancement.

---

# § E — INTEGRATION WITH OTHER TRACKS

This work sits in **Track E** of the parallel-work coordination doc. It is **largely independent**:

- **Does NOT depend** on DeepSeek's multi-block BLAKE3 work (that's for native circuit, not WASM verify)
- **Does NOT depend** on Merkle gadget or δ-circuit work (verifier doesn't need the circuit; just the verification key)
- **Does depend** on the wire-protocol Blueprint 5 (`/api/v1/proof/tip` endpoint) — but only at runtime, and only the JSON schema. Phase 1 can use a mocked response.
- **Will eventually depend** on Phase 2 Nova IVC wrapper (Claude Code agent track) for the real verifier body.

So an external implementer or Claude Code agent can start Phase 1 today **in parallel** with DeepSeek's BLAKE3 work and with the Nova wrapper work. Three streams in flight at once.

---

# § F — START-HERE CHECKLIST

1. Read this doc end-to-end (this is the spec)
2. Read `gui/quantum-wallet/package.json` to confirm the wallet's build tooling (vite, tsc)
3. Verify `wasm-pack` is installed: `wasm-pack --version` (install via `cargo install wasm-pack` if missing)
4. Create `crates/q-ivc-verifier-wasm/` files exactly as specced in §A.1
5. Add the crate to workspace `members` in root `Cargo.toml`
6. Build with `wasm-pack build --target web --release`
7. Copy `pkg/*` into `gui/quantum-wallet/src/ivc/wasm/`
8. Create `gui/quantum-wallet/src/ivc/` files exactly as specced in §A.2
9. Run `npm run build` in `gui/quantum-wallet/` — confirm clean compile
10. Create `gui/quantum-wallet/public/ivc-demo.html` per §A.3
11. Manually test the demo page in a browser: verify `verifier_version() == "placeholder-v0"`, verify the round-trip cache
12. Push branch to `code.quillon.xyz` as `claude-agent/wasm-verifier-scaffold` (or equivalent)
13. Notify Beta in `#dev-coordination` for review

Estimated total time for Phase 1: **2-3 working days** for someone familiar with wasm-bindgen + React.

---

# § G — KILL-SWITCH CONDITIONS

The PR is rejected immediately if any of these happen:

1. **Phase 1 ships without the placeholder warning in the README and UI.** Users must know that until Phase 2 lands, the verifier is theater.
2. **The WASM binary includes real cryptographic dependencies** (nova-snark, arkworks, etc.) in Phase 1. Keep the dep list minimal so the binary is small and the boundary between scaffolding and real crypto is sharp.
3. **The JS API is broken in a way that Phase 2 can't slot in without changes.** The `verify(stateRoot, height, proofBytes) -> Promise<{valid, duration_ms}>` signature is contractual.
4. **Any file outside `crates/q-ivc-verifier-wasm/`, `gui/quantum-wallet/src/ivc/`, `gui/quantum-wallet/public/ivc-demo.html`, and the root `Cargo.toml` workspace-members entry is modified.** Scope creep is the enemy here.

— Quillon Graph maintainers, 2026-05-13
