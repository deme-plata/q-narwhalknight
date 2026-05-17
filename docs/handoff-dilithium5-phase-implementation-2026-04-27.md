# Dilithium5 Phase Implementation — Handoff Document

**Date:** 2026-04-27  
**Status:** Phase 0 complete. Phase A blocked on WASM dependency.  
**Full spec:** `docs/technical-review-pq-key-derivation-2026-04-27.md` (v2.1)

---

## What Is Done

### Phase 0 — Simulated Verifier Hardened ✅
**File:** `gui/quantum-wallet/src/libp2p/postQuantumCrypto.ts`

The simulated backend (fallback when real WASM is absent) was accepting any correctly-sized buffer as a valid Dilithium5 signature — a critical security hole. Fixed:

- `verify()` now always returns `false` in production (was: `signature.length === DILITHIUM5_SIGNATURE_BYTES`)
- `keyGen()` now throws in production (was: silently returned random bytes)
- `loadPQCrypto()` fallback path now dispatches a `pq-backend-failed` CustomEvent so UI can show a warning
- `isRealBackendLoaded()` exported — Phase A uses this to gate keygen on real WASM presence

```typescript
// New export for Phase A gating
export function isRealBackendLoaded(): boolean {
  return wasmLoaded && pqcrypto !== null && pqcrypto.type !== 'simulated'
}
```

**Mainnet risk:** Zero. Client-side only. No blocks, no consensus affected.

---

## What Is Pending

### Phase A — Deterministic Dilithium5 from Mnemonic

**Blocker:** Need a Dilithium5 WASM that exposes `keyPairFromSeed(seed: Uint8Array)`.

Two options (from spec doc Section 3):
- **Option 1 (preferred):** Build `pqcrypto-dilithium` Rust crate to WASM with seeded keygen exposed
- **Option 2:** Fork/patch the existing `dilithium-crystals` npm package to accept a seed

Until the WASM exists, Phase A cannot be completed — `isRealBackendLoaded()` will return false for all users.

**Files to change when WASM is ready:**

1. **`gui/quantum-wallet/src/libp2p/postQuantumCrypto.ts`**  
   Wire the WASM `keyPairFromSeed(seed)` into `loadPQCrypto()`.

2. **`gui/quantum-wallet/src/services/walletAuth.ts:670–700`**  
   Fix `generateDilithium5KeyPairFromMnemonic()` — currently computes seed then discards it:
   ```typescript
   // BUG: seed computed but never passed
   const keypair = await pqCrypto.dilithium5KeyGen();  // ← ignores seed
   
   // FIX: pass seed to deterministic keygen
   const keypair = await pqCrypto.dilithium5KeyGenFromSeed(seed);
   ```

3. **`gui/quantum-wallet/src/services/walletAuth.ts` (lazy re-derive logic)**  
   Add Phase A3 migration: when `walletEncryptedDilithium5Key` is absent from localStorage
   and mnemonic is in session, silently derive and store the key:
   ```typescript
   if (!localStorage.getItem('walletEncryptedDilithium5Key') && sessionMnemonic) {
     const dil5Keys = await generateDilithium5KeyPairFromMnemonic(sessionMnemonic);
     // encrypt and store
     localStorage.setItem('walletEncryptedDilithium5Key', encryptedKey);
     localStorage.setItem('walletDilithium5PublicKey', hex(dil5Keys.publicKey));
     localStorage.setItem('walletDilithium5KeyVersion', '2');
   }
   ```

**Migration note:** Old random Dilithium5 keys in localStorage are unrecoverable and intentionally superseded by the new deterministic keys.

---

### Phase B — Remove SQIsign Dead Code ✅ (easy, no dependencies)

**Files:**

1. **`gui/quantum-wallet/src/services/walletAuth.ts`**  
   - Line ~681: delete `buildSignedTransaction()` (zero callers, dead SQIsign pipeline)  
   - Line ~41, 127–165: `generateAuthHeader()` declares `'Dilithium5'` scheme but never handles it — implement or throw

2. **`crates/q-types/src/lib.rs`**  
   - Remove `Phase2SQIsign` and `HybridEd25519SQIsign` from `sign_block_with_keypair()` match arms (lines ~192–200)  
   - Remove `preferred_phase = Phase2SQIsign` default from ephemeral keypair generation (line ~225)

3. **`crates/q-sqisign-sys/`** — leave as-is but add `[features] sqisign-V2 = []` and gate behind it so it compiles as a research artifact only.

**Mainnet risk:** Zero. Removing dead code that currently produces invalid signatures anyway.

---

### Phase C — Server-Side Dilithium5 Seeded Keygen

**File:** `crates/q-types/src/lib.rs`

Add `ValidatorKeypair::generate_from_seed(seed: [u8; 32])` using domain-separated SHAKE-256:

```rust
pub fn generate_from_seed(seed: [u8; 32]) -> Self {
    let dil_seed = shake256_domain(b"qnk_dilithium5_v1", &seed);
    let (pk, sk) = dilithium5::keypair_from_seed(&dil_seed);
    // ...
}
```

Used by: node restart determinism, validator identity derivation.

---

### Phase D — Wire Dilithium5 into HTTP Transaction Signing

**File:** `gui/quantum-wallet/src/api.ts:~1260`

`sendTransaction()` currently uses Ed25519 only. When PQ enforcement is live, this path must also sign with Dilithium5.

```typescript
// Add after existing Ed25519 signing:
if (isRealBackendLoaded() && walletDilithium5Key) {
  tx.dilithium5_signature = await signWithDilithium5(txBytes, walletDilithium5Key);
}
```

**Mainnet risk:** Zero during Phase D. Upgrade gate is still `u64::MAX` so the signature field is ignored by all nodes.

---

### Phase E — Activate the Upgrade Gate (Final Step)

**File:** `crates/q-consensus-guard/src/upgrade_gate.rs:111`

```rust
// Change from:
activation_height: u64::MAX,
// To a real height ~20,000 blocks (~5.5 hours) from deployment:
activation_height: CURRENT_HEIGHT + 20_000,
```

**Prerequisites before Phase E:**
1. All of Phases A–D deployed and stable on mainnet
2. Public announcement (Discord + BitcoinTalk) with activation height and upgrade instructions
3. Canary validation on Delta/Docker per checklist in Section 8 of the spec doc
4. User migration edge case handled: post-Phase-E, new device without mnemonic shows:  
   *"Your funds are safe and accessible. However, sending transactions now requires your 24-word recovery phrase..."*

**This is the only step that changes consensus rules.** Do not rush it.

---

## Key Files Quick Reference

| Phase | File | What to change |
|-------|------|---------------|
| A1 | WASM build pipeline | New: `dilithium5KeyGenFromSeed(seed)` export |
| A2 | `walletAuth.ts:670` | Pass seed to WASM keygen |
| A3 | `walletAuth.ts` (session logic) | Lazy re-derive missing Dilithium5 key |
| B | `walletAuth.ts:41,681` | Delete dead SQIsign code |
| B | `q-types/src/lib.rs:192,225` | Remove SQIsign match arms |
| C | `q-types/src/lib.rs` | Add `generate_from_seed()` |
| D | `api.ts:~1260` | Dual-sign HTTP transactions |
| E | `upgrade_gate.rs:111` | Set real activation height |

## Constants (Dilithium5)

```typescript
const DILITHIUM5_PUBLIC_KEY_BYTES = 2592;
const DILITHIUM5_SECRET_KEY_BYTES = 4864;
const DILITHIUM5_SIGNATURE_BYTES  = 4627;
```

```rust
// Domain separator for seed derivation
const DOMAIN_SEP: &[u8] = b"qnk_dilithium5_v1";
```

## Identifier Conventions

Use these exact names throughout (from spec doc Section 4):

- WASM function: `dilithium5KeyGenFromSeed`
- Wallet type: `HybridEd25519Dilithium5`
- LocalStorage key (encrypted secret): `walletEncryptedDilithium5Key`
- LocalStorage key (public): `walletDilithium5PublicKey`
- LocalStorage key (version): `walletDilithium5KeyVersion`
- Rust field name: `dilithium5_public` / `dilithium5_secret`
- Session variable: `dil5Keys`
- Auth scheme string: `qnk_dilithium5_v1`
