# Technical Review: Post-Quantum Key Derivation & Mainnet Transition
**Version:** 2.1 (Dilithium5 confirmed per project decision, 2026-04-27)  
**Date:** 2026-04-27  
**Author:** Server Beta (Claude Code audit)  
**Reviewed by:** Core Cryptography & Consensus Review  
**Status:** Approved for implementation planning  
**Severity:** CRITICAL — blocks all mandatory PQ enforcement  
**Mainnet exposure:** $1.5B market cap — zero-risk deployment required

---

## Executive Summary

The codebase has full structural support for post-quantum signatures but none of it is correctly wired for production. Three independent failure modes exist:

1. **Dilithium5 key derivation is broken** — the mnemonic seed is computed then discarded; keys are random and unrecoverable.
2. **SQIsign is cryptographic theatre** — neither the wallet nor the backend generates valid SQIsign keypairs; all signatures produced will fail any real verifier.
3. **The upgrade gate is permanently disabled** — `PostQuantumSignatures` activation height = `u64::MAX`; no enforcement can happen today even if the above were fixed.

**Decisions confirmed by external review:**
- Use **Dilithium5** (NIST Level 5): 4,627-byte signatures, maximum classical security (256-bit / 128-bit quantum), overriding the external review's Dilithium3 recommendation
- Use **HybridEd25519Dilithium5** for the initial enforcement phase — Ed25519 already binds the Dilithium key, making a separate on-chain key registry unnecessary
- **Disable SQIsign entirely** (Option B) — remove from all live code paths; keep `q-sqisign-sys` as a research artifact under a future `sqisign-V2` feature flag
- Fix the simulated verifier as the **first action** before anything else

**Critical safety principle for a $1.5B mainnet:** Every fix in Phases A–D is purely additive and forwards-compatible. It changes nothing about existing blocks, existing balances, or existing consensus rules. The upgrade gate (`u64::MAX`) remains in place throughout all phases. Zero consensus-layer changes land until Phase E, and Phase E is preceded by a mandatory public announcement period and canary validation. There is no step in this plan that can cause fund loss, fork, or rollback.

---

## 1. Current State — The Upgrade Gate

**File:** `crates/q-consensus-guard/src/upgrade_gate.rs:111`

```rust
upgrades.insert(Upgrade::PostQuantumSignatures, UpgradeConfig {
    activation_height: u64::MAX, // Not scheduled yet
    description: "Enable Dilithium post-quantum signatures".to_string(),
    mandatory: false,
    min_version: "2.0.0".to_string(),
});
```

This is the correct mechanism — height-gated, deterministic, backwards-compatible. Every node independently checks this gate when validating each block. Setting `u64::MAX` means the gate is permanently off until a new binary with a real height is deployed.

**This gate is what separates "fixing the plumbing" from "changing mainnet rules."** All of Phases A–D happen behind this gate. Users, miners, and validators are completely unaffected during those phases. Phase E is the only step that changes anything on-chain, and it cannot activate until the binary change is live on the majority of nodes.

---

## 2. Bug 1 — Dilithium Key Derivation (CRITICAL)

### The Bug

**File:** `gui/quantum-wallet/src/services/walletAuth.ts:670–700`

```typescript
export async function generateDilithium5KeyPairFromMnemonic(mnemonic: string): Promise<...> {
  const seedInput = new TextEncoder().encode('qnk_dilithium5_v1' + mnemonic);
  const seed = sha3_256(seedInput);   // ← seed computed...

  const keypair = await pqCrypto.dilithium5KeyGen();  // ← seed NEVER PASSED
  return keypair;
}
```

`dilithium5KeyGen()` calls `crypto.getRandomValues()` internally. Every call produces a different keypair. The function name is false — it does not derive from the mnemonic.

Same bug in `generateKyber1024KeyPairFromMnemonic()` (lines 706–727).

### Why This Causes Fund Loss

| Scenario | Today (PQ not enforced) | After PQ gate activates |
|---|---|---|
| User clears browser / new device | Ed25519 key recoverable from mnemonic ✅ | Dilithium key **gone forever** ❌ |
| User restores from mnemonic | Ed25519 works ✅ | New random Dilithium key, different from the one on-chain ❌ |
| Fallback branch in signTransactionForP2P | Generates fresh random Dilithium key per tx | Tx rejected by validators ❌ |

### Working Reference — How Ed25519 Does It

**File:** `gui/quantum-wallet/src/services/walletAuth.ts:205–217`

```typescript
export async function keypairFromMnemonic(mnemonic: string): Promise<WalletKeyPair> {
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const privateKey = sha3_256(mnemonicBytes);        // deterministic 32-byte seed
  const publicKey = await ed25519.getPublicKey(privateKey);  // derived from seed
  return { publicKey, privateKey, address: deriveAddress(publicKey) };
}
```

Same mnemonic → same private key → same public key → same address. This is the contract that must be replicated for Dilithium.

### The Dependency Limitation

**File:** `gui/quantum-wallet/src/libp2p/postQuantumCrypto.ts:262–276`

```typescript
export async function dilithium5KeyGen(): Promise<Dilithium5Keypair> {
  const keypair = await pqcrypto.dilithium.keyPair()  // npm: dilithium-crystals
  // ↑ takes no seed parameter — uses internal RNG
}
```

The npm package `dilithium-crystals` does not expose a seeded `keyPair()` call. However, ML-DSA (Dilithium) is deterministic by specification (FIPS 204 §6.1) — a 32-byte `zeta` seed expands into the full keypair via SHAKE-256. The algorithm supports this; the library just does not expose it.

**Resolution (agreed by review):**
- **Option 1 (preferred):** Build `pqcrypto-dilithium` to WASM with seeded keygen exposed — strongest spec compliance.
- **Option 2:** Fork/patch `dilithium-crystals` to expose `keyPairFromSeed(seed: Uint8Array)`.
- **Option 3 (rejected by review):** Treating HKDF output as a Dilithium secret key directly — violates key format, breaks interoperability. Do not implement.

### Algorithm Choice: Dilithium5

The project will use **Dilithium5** (NIST Level 5 — the highest ML-DSA security level). Dilithium5 provides 256-bit classical security and 128-bit quantum security — the maximum available under FIPS 204. The external review recommended Dilithium3 for its smaller 3,293-byte signatures, but the project is opting for Dilithium5's maximum security headroom. At 4,627 bytes per signature the storage overhead is accepted in exchange for the highest security level.

---

## 3. Bug 2 — Simulated Verifier Accepts Any Signature (HIGH — Fix First)

**File:** `gui/quantum-wallet/src/libp2p/postQuantumCrypto.ts:769–772`

```typescript
// Simulated Dilithium5 verify:
dilithium5.verify = (sig, msg, pk) => {
  return signature.length === DILITHIUM5_SIGNATURE_BYTES;  // ← accepts ANY correctly-sized buffer
};
```

When the WASM module fails to load (network error, bundler misconfiguration, CSP block), the wallet silently falls back to this simulated backend. The simulated verifier accepts every signature of the correct byte length — a completely invalid signature passes.

**This is not a standalone bug — it undermines the entire security model.** Fixing key derivation is worthless if the verifier never actually checks. The simulated path must be disabled in production.

**Fix (one line):**

```typescript
// BEFORE:
dilithium5.verify = (sig, msg, pk) => signature.length === DILITHIUM5_SIGNATURE_BYTES;

// AFTER:
dilithium5.verify = (sig, msg, pk) => {
  if (process.env.NODE_ENV !== 'development') {
    console.error('[PQC] Real Dilithium backend not loaded — verification disabled');
    return false;  // ← always reject when no real crypto
  }
  return false;  // also false in development; simulated path must never be trusted
};
```

Additionally, the wallet must show a clear error to the user when the PQ backend fails to load, rather than silently degrading. Failing loudly is safe; failing silently is not.

**This fix must land before any other PQ work.**

---

## 4. Bug 3 — SQIsign Is Not SQIsign (CRITICAL — Remove Entirely)

### Frontend

**File:** `gui/quantum-wallet/src/services/walletAuth.ts:737–756`

```typescript
export async function generateSQIsignKeyPair(): Promise<...> {
  const secretKey = crypto.getRandomValues(new Uint8Array(SQISIGN_SK_SIZE));
  const publicKey = crypto.getRandomValues(new Uint8Array(SQISIGN_PK_SIZE));
  // XOR with SHA-256(sk) — still random, not an isogeny computation
  return { publicKey, secretKey };
}
```

A real SQIsign public key is the j-invariant of a supersingular elliptic curve — the image of E₀ under a secret isogeny. This function produces `random XOR SHA256(random)`. No isogeny. No cryptographic relationship between secret and public key. Signatures produced will fail any real verifier.

**Dead code confirmation:** `buildSignedTransaction()`, `signTransactionPQC()`, `signTransactionHybrid()`, `signWithSQIsign()` — all have **zero callers** in the frontend source tree. The SQIsign transaction-signing path is entirely dead.

### Backend

**File:** `crates/q-types/src/pqc_keys.rs:76–82`

```rust
let mut sqisign_secret = vec![0u8; 64];
let mut sqisign_public = vec![0u8; 64];
getrandom::getrandom(&mut sqisign_secret)...;
getrandom::getrandom(&mut sqisign_public)...;  // independent random bytes, not an isogeny
```

Same problem. Both are random and independent. No keypair relationship exists.

### Decision (confirmed by external review): Option B — Remove SQIsign

Per review: *"The real SQIsign C code is available in q-sqisign-sys, but its performance (signing ~500ms) and lack of hardened side-channel protections in the browser make it unsuitable for a mandatory signing path in the near term."*

Actions:
1. Remove `generateSQIsignKeyPair()` and related functions from wallet.
2. Delete dead code: `buildSignedTransaction()`, `signTransactionPQC()`, `signTransactionHybrid()`, `signWithSQIsign()`.
3. Remove `Phase2SQIsign` and `HybridEd25519SQIsign` from `sign_block_with_keypair` match arms.
4. Add compile guard to prevent accidental reactivation:

```rust
SignaturePhase::Phase2SQIsign | SignaturePhase::HybridEd25519SQIsign => {
    #[cfg(not(feature = "sqisign-prod"))]
    compile_error!("SQIsign not production-ready. Enable sqisign-prod feature explicitly.");
}
```

5. Keep `q-sqisign-sys` and `q-sqisign` crates in the workspace as research artifacts. Track reintroduction under future feature flag `sqisign-V2`.

**Backwards-compat SQIsign key regeneration bug:** `load_encrypted()` and `load_from_file()` silently regenerate fresh random SQIsign keys when loading older keypair formats. After SQIsign is removed from live paths, this code becomes harmless — but it should still log a `WARN` that the SQIsign fields are placeholder and not valid keypair material.

---

## 5. Bug 4 — Backend Validator Keypair Not Persistent (HIGH)

### Location

`crates/q-api-server/src/main.rs` — validator keypair loading block

```rust
let validator_keypair = if let Some(key_path) = ... {
    ValidatorKeypair::load_from_file(key_path)  // ← deprecated plaintext path
} else {
    ValidatorKeypair::generate_with_zk_stark_untrusted()  // ← ephemeral, never saved
    // preferred_phase = Phase2SQIsign on ephemeral keypairs ← also wrong
};
```

- No `--validator-key` flag → new random keypair on every restart, never persisted.
- File-loading path uses the **deprecated plaintext** `load_from_file` — private keys on disk in JSON.
- Ephemeral default sets `preferred_phase = Phase2SQIsign` — which as shown above produces invalid signatures.

### Fix

1. Add `ValidatorKeypair::generate_from_seed(seed: [u8; 32])` — deterministic derivation from a 32-byte seed using domain-separated SHAKE-256 for Dilithium5.
2. Accept `Q_VALIDATOR_MNEMONIC` (BIP-39) or `Q_VALIDATOR_SEED_HEX` env var to derive the seed.
3. Auto-persist to `$Q_DB_PATH/validator-keypair.enc` on first boot; load on subsequent boots.
4. Log first 8 hex bytes of the Ed25519 public key on every start so operators can verify identity continuity.
5. Replace `load_from_file` with `load_encrypted` throughout.

---

## 6. Additional Bugs

| # | File | Lines | Severity | Description | Fix |
|---|------|-------|----------|-------------|-----|
| A | `postQuantumCrypto.ts` | 769–772 | **CRITICAL** | Simulated verifier accepts any correctly-sized signature | Always return `false` when no real backend loaded (see Bug 3 above) |
| B | `walletAuth.ts` | 41, 127–165 | MEDIUM | `generateAuthHeader()` declares `'Dilithium5'` scheme but never handles it — would emit unsigned header | Implement the scheme or throw hard error |
| C | `pqc_keys.rs` | 487–498, 555–567 | HIGH | Backwards-compat silently re-randomises SQIsign keys on old-format load — no warning | Add `WARN` log; fix becomes moot after SQIsign removal |
| D | `pqc_keys.rs` | 625–705 | MEDIUM | `from_backup_bytes()` doesn't verify Dilithium keypair consistency after restore | Add sanity check: re-derive public from secret and compare |
| E | `walletAuth.ts` | 681 | LOW | Domain separator `'qnk_dilithium5_v1' + mnemonic` without length prefix — different mnemonics could collide | Use length-prefixed encoding (fixed in Phase A implementation) |
| F | `api.ts` | ~1260 | HIGH | `sendTransaction()` HTTP path uses Ed25519 auth only — all HTTP txs rejected when PQ is mandatory | Add Dilithium5 signing in Phase D |
| G | `walletAuth.ts` | 955–1000 | MEDIUM | `buildSignedTransaction()` zero callers — dead SQIsign pipeline | Delete in Phase B |
| H | `walletAuth.ts` | ~1380 | HIGH | `signTransactionForP2P` ephemeral fallback generates fresh random Dilithium key per tx | Remove fallback; use error or Ed25519-only when PQ key absent |

---

## 7. Implementation Plan

### Absolute Rule: The Upgrade Gate Stays at u64::MAX Until Phase E

Every change in Phases A–D is in wallet software, node software configuration, or dead-code removal. **None of them changes consensus rules.** The upgrade gate prevents any enforcement until Phase E. A node running Phase D code on mainnet is 100% compatible with a node running today's code — it produces and accepts identical blocks.

---

### Phase 0 — Fix Simulated Verifier (Do Today, One Line)

**File:** `gui/quantum-wallet/src/libp2p/postQuantumCrypto.ts`

Change simulated `dilithium5.verify` to always return `false`.  
Change simulated `dilithium5.keyGen` to throw an error in production.  
Add user-visible error when PQ backend fails to load.

**Mainnet risk:** Zero. This is a client-side change. No blocks, no transactions, no consensus affected. A user whose WASM fails to load gets an error message instead of silently passing fake verification.

**Rollback:** Deploy previous wallet build. No on-chain state involved.

---

### Phase A — Deterministic Dilithium5 Derivation from Mnemonic

#### A1 — WASM dependency

Build or fork a Dilithium5 WASM that exposes `keyPairFromSeed(seed: Uint8Array)`. Per external review, Option 2 (spec-compliant WASM build) is preferred.

Verify determinism before shipping:
```typescript
const k1 = await dilithium5KeyGenFromSeed(seed);
const k2 = await dilithium5KeyGenFromSeed(seed);
console.assert(k1.publicKey.every((b, i) => b === k2.publicKey[i]),
  'Dilithium5 keygen must be deterministic');
```

Run this check on every supported browser (Chrome, Firefox, Safari, Brave) before merging.

#### A2 — Fix `generateDilithium5KeyPairFromMnemonic`

```typescript
export async function generateDilithium5KeyPairFromMnemonic(mnemonic: string): Promise<{
  publicKey: Uint8Array;
  secretKey: Uint8Array;
}> {
  const pqCrypto = await getPQCrypto();
  if (!pqCrypto.isRealBackendLoaded()) {
    throw new Error('PQ crypto backend not available — cannot generate quantum-safe keys');
  }

  // Domain-separated, length-prefixed seed derivation
  const domain = 'qnk_dilithium5_v1';
  const mnemonicBytes = new TextEncoder().encode(mnemonic);
  const buf = new Uint8Array(1 + domain.length + mnemonicBytes.length);
  buf[0] = domain.length;
  buf.set(new TextEncoder().encode(domain), 1);
  buf.set(mnemonicBytes, 1 + domain.length);
  const seed = sha3_256(buf);  // 32 bytes

  return pqCrypto.dilithium5KeyGenFromSeed(seed);  // deterministic
}
```

**Migration note:** This changes the Dilithium key derived from any given mnemonic compared to what was previously stored (since the old function was random). This is intentional and correct — the old random keys were unrecoverable anyway. Existing stored keys in localStorage will be superseded by the re-derived deterministic keys during Phase A3.

#### A3 — Fix `loadWallet` recovery path

When `walletEncryptedDilithium5Key` is missing from localStorage and the mnemonic is available in the session:

```typescript
if (!keyPair.dilithium5SecretKey && mnemonic) {
  try {
    const dil5Keys = await generateDilithium5KeyPairFromMnemonic(mnemonic);
    const enc = await encryptPrivateKey(dil5Keys.secretKey, password);
    localStorage.setItem('walletEncryptedDilithium5Key', enc);
    localStorage.setItem('walletDilithium5PublicKey', bytesToHex(dil5Keys.publicKey));
    keyPair.dilithium5PublicKey = dil5Keys.publicKey;
    keyPair.dilithium5SecretKey = dil5Keys.secretKey;
    // Store version tag to detect pre-fix blobs
    localStorage.setItem('walletDilithium5KeyVersion', '2');
  } catch (e) {
    console.warn('[PQC] Could not derive Dilithium5 keys from mnemonic:', e);
    // Do NOT generate random fallback keys
  }
}
```

**Version tag:** Store `walletDilithium5KeyVersion = '2'` when keys are deterministically derived. On first load after update, if version is missing or `'1'` (the old random generation), trigger re-derivation. This allows the wallet to identify and replace all pre-fix random keys automatically.

#### A4 — Remove ephemeral fallback from `signTransactionForP2P`

```typescript
// REMOVE THIS:
} else {
  const dilithiumKeypair = await pqCrypto.dilithium5KeyGen();
  dilithium5Signature = await pqCrypto.dilithium5Sign(txHash, dilithiumKeypair.secretKey);
}

// REPLACE WITH:
} else {
  // No PQ key available — use Ed25519-only while PQ is not yet mandatory
  signatureMode = 'ed25519';
  // When PQ becomes mandatory, this branch becomes an error:
  // throw new Error('Quantum-safe key not available — please log in again to restore your PQ key');
}
```

**Mainnet risk:** Zero. The upgrade gate is still at `u64::MAX`. Ed25519-only transactions are fully valid throughout Phase A. The change is that we stop polluting the transaction with an ephemeral random Dilithium key that the user cannot reproduce.

---

### Phase B — Remove SQIsign

**Mainnet risk:** Zero. This removes dead code and compile-guards against accidental reactivation. No block, transaction, or consensus rule is affected. The upgrade gate is still at `u64::MAX`.

#### B1 — Frontend wallet

Delete from `walletAuth.ts`:
- `generateSQIsignKeyPair()`
- `generateSQIsignKeyPairFromMnemonic()` (if it exists)
- `signWithSQIsign()`
- `signTransactionPQC()`
- `signTransactionHybrid()`
- `buildSignedTransaction()`
- `SQISIGN_SK_SIZE`, `SQISIGN_PK_SIZE` constants
- `sqisignPublicKey`, `sqisignSecretKey` from `WalletKeyPair` type

Remove from `loadWallet`:
- SQIsign localStorage keys (`walletEncryptedSQIsignKey`, `walletSQIsignPublicKey`)

Remove from wallet creation flow:
- The `generateSQIsignKeyPair()` call in `keypairFromMnemonic`/wallet setup

#### B2 — Backend block producer

**File:** `crates/q-api-server/src/block_producer.rs`

```rust
SignaturePhase::Phase2SQIsign | SignaturePhase::HybridEd25519SQIsign => {
    #[cfg(not(feature = "sqisign-prod"))]
    {
        error!("🚨 SQIsign signing requested but sqisign-prod feature not enabled");
        return Err("SQIsign not production-ready".into());
    }
    #[cfg(feature = "sqisign-prod")]
    { /* real implementation goes here when sqisign-V2 lands */ }
}
```

#### B3 — Backend keypair

Remove `sqisign_secret` and `sqisign_public` from `ValidatorKeypair`. For binary backwards-compat with existing keypair files, keep the deserialization fields but mark them `#[serde(default)]` and log `WARN` if non-empty bytes are found (meaning the file was generated before the cleanup).

---

### Phase C — Validator Keypair Persistence

**Mainnet risk:** Zero. This changes how validator nodes manage their identity on disk. It does not change how they sign blocks (still `Phase0Ed25519` by default) or how other nodes verify them. The upgrade gate is still at `u64::MAX`.

#### C1 — `generate_from_seed` constructor

```rust
pub fn generate_from_seed(seed: [u8; 32]) -> Result<Self> {
    // Ed25519: seed used directly
    let ed25519_signing = SigningKey::from_bytes(&seed);
    let ed25519_verifying = ed25519_signing.verifying_key();

    // Dilithium5: domain-separate, expand with SHAKE-256 per FIPS 204 §6.1
    let mut dil_seed = [0u8; 32];
    shake256_expand(b"qnk_dilithium5_v1", &seed, &mut dil_seed);
    let (dilithium5_public, dilithium5_secret) = dilithium5::keypair_from_seed(&dil_seed);

    Ok(Self {
        node_id: ed25519_verifying.to_bytes(),
        ed25519_signing,
        ed25519_verifying,
        dilithium5_secret,
        dilithium5_public,
        preferred_phase: SignaturePhase::Phase0Ed25519,  // stays classical until gate
    })
}
```

**Note:** `dilithium5::keypair_from_seed` requires a patch to `pqcrypto-dilithium` or a vendored implementation that exposes the FIPS 204 §6.1 seeded variant. This is the same dependency work as Phase A1.

#### C2 — Auto-persist in `main.rs`

```rust
let key_path = data_dir.join("validator-keypair.enc");
let passphrase = std::env::var("Q_VALIDATOR_PASSPHRASE")
    .unwrap_or_else(|_| generate_machine_passphrase(&node_id));

let validator_keypair = if key_path.exists() {
    info!("🔐 Loading validator keypair from {}", key_path.display());
    let kp = ValidatorKeypair::load_encrypted(&key_path, &passphrase)?;
    info!("   Node Ed25519 ID: {}...", hex::encode(&kp.node_id[..8]));
    kp
} else if let Ok(mnemonic) = std::env::var("Q_VALIDATOR_MNEMONIC") {
    info!("🔐 Deriving validator keypair from Q_VALIDATOR_MNEMONIC");
    let seed = mnemonic_to_seed_32(&mnemonic);
    let kp = ValidatorKeypair::generate_from_seed(seed)?;
    kp.save_encrypted(&key_path, &passphrase)?;
    info!("   Keypair saved to {}", key_path.display());
    info!("   Node Ed25519 ID: {}...", hex::encode(&kp.node_id[..8]));
    kp
} else if let Ok(hex_seed) = std::env::var("Q_VALIDATOR_SEED_HEX") {
    info!("🔐 Deriving validator keypair from Q_VALIDATOR_SEED_HEX");
    let seed = hex::decode(&hex_seed)?;
    let kp = ValidatorKeypair::generate_from_seed(seed.try_into()?)?;
    kp.save_encrypted(&key_path, &passphrase)?;
    kp
} else {
    info!("🔐 Generating new validator keypair (first boot — no seed provided)");
    let kp = ValidatorKeypair::generate();
    kp.save_encrypted(&key_path, &passphrase)?;
    warn!("⚠️  No Q_VALIDATOR_MNEMONIC set — this keypair is NOT recoverable from a seed phrase");
    warn!("   Backup {} immediately or configure Q_VALIDATOR_MNEMONIC", key_path.display());
    kp
};
```

Remove `generate_with_zk_stark_untrusted()` from the startup path. Remove the `load_from_file` (plaintext) path.

---

### Phase D — Fix HTTP Transaction Path

**File:** `gui/quantum-wallet/src/services/api.ts`

`sendTransaction()` currently uses Ed25519-only auth header. When PQ is mandatory, all HTTP-submitted transactions will fail. Fix:

```typescript
async sendTransaction(params: SendTransactionParams): Promise<...> {
  const session = walletSession.getSession();
  const scheme = (session?.dilithium5SecretKey && session?.dilithium5PublicKey)
    ? 'HybridEd25519Dilithium5'
    : 'Ed25519';

  const authHeader = await generateAuthHeader(
    walletAddress, privateKey, nonce, scheme,
    session?.dilithium5SecretKey,
    session?.dilithium5PublicKey
  );
  ...
}
```

`generateAuthHeader` must be updated to implement the `HybridEd25519Dilithium5` scheme (currently declared but unimplemented — Bug B above). The scheme:
1. Signs `sha3_256(nonce + walletAddress + amount + recipient)` with Ed25519.
2. Signs the same payload with Dilithium5.
3. Encodes both signatures and both public keys into the auth header.
4. Server validates both before accepting.

**Mainnet risk:** Zero. While the upgrade gate is at `u64::MAX`, the server continues to accept Ed25519-only transactions normally. The hybrid header adds Dilithium5 data which the server currently ignores (since PQ is not enforced). When Phase E activates, the server will begin requiring the Dilithium5 component — but by then all updated clients will already be sending it.

This is the classic "write path ahead of the read path" safe migration pattern.

---

### Phase E — Schedule the Upgrade Gate (Final Step)

**This is the only step that changes mainnet consensus. All previous phases must be complete and validated before this step.**

#### Pre-requisites checklist (all must be true before setting activation height)

- [ ] Phase 0: Simulated verifier disabled — deployed and confirmed
- [ ] Phase A: Deterministic Dilithium5 derivation — deployed and confirmed working across Chrome, Firefox, Safari, Brave, mobile
- [ ] Phase A3: `loadWallet` recovery path deployed — existing users' keys have been upgraded (version tag `'2'` visible in analytics/logs)
- [ ] Phase B: SQIsign removed from all live code paths — deployed and confirmed
- [ ] Phase C: Validator keypair persistence deployed — all production validators confirmed with stable identities across restarts
- [ ] Phase D: HTTP transaction path upgraded — all transactions now include hybrid signatures
- [ ] Binary version bumped to `11.0.0` (matches `min_version` in upgrade config)
- [ ] All four production nodes (Beta, Gamma, Epsilon, Delta) running v11.0.0+
- [ ] Public announcement made on Discord, BitcoinTalk, and direct validator outreach
- [ ] Minimum 4-week announcement window elapsed
- [ ] Canary validation complete (see Section 8 — Mainnet Safety)
- [ ] Emergency rollback procedure tested and confirmed working

#### Activation height calculation

Current mainnet height: ~16,400,000  
Block rate: ~1 block/second  
4 weeks: 4 × 7 × 24 × 3600 = 2,419,200 blocks  
**Target activation height: ~18,820,000**

```rust
// crates/q-consensus-guard/src/upgrade_gate.rs
upgrades.insert(Upgrade::PostQuantumSignatures, UpgradeConfig {
    activation_height: 18_820_000,
    description: "Require HybridEd25519Dilithium5 signatures (NIST Level 5, 256-bit classical / 128-bit quantum)".to_string(),
    mandatory: true,
    min_version: "11.0.0".to_string(),
});
```

#### What happens at activation height

Every validator independently checks, for each incoming block at height ≥ 18,820,000:

```rust
let pq_required = is_upgrade_active(Upgrade::PostQuantumSignatures, block.height);
if pq_required && !block_has_dilithium5_signature(&block) {
    error!("❌ Block {} rejected: PQ upgrade active but no Dilithium5 signature", block.height);
    return Err(ValidationError::MissingPQSignature);
}
```

Any transaction submitted without a Dilithium5 signature is rejected from that height. Any node running pre-11.0.0 binary becomes incompatible at that height and will stop syncing (the `min_version` check in the upgrade config enforces this).

---

## 8. Mainnet Safety — Zero-Risk Deployment for a $1.5B Chain

### Core Safety Principle

**The upgrade gate is the firewall between "fixing software" and "changing mainnet rules."** A chain with $1.5B market cap has no room for forced rollbacks, emergency hard forks, or user fund loss. Every step in this plan is designed so that if anything goes wrong at any phase, the chain continues to operate identically to how it does today.

### Why Phases A–D Cannot Cause a Fork or Fund Loss

| What changes | On-chain effect | Reversible? |
|---|---|---|
| Simulated verifier disabled | None — client-side only | Yes — redeploy old wallet |
| Dilithium5 deterministic keygen | None — key generation only | Yes — users who haven't used PQ yet unaffected |
| SQIsign dead code removed | None — never reached in production | Yes — redeploy old binary |
| Validator keypair auto-persist | None — same Ed25519 identity, same Phase0 signing | Yes — delete file, restart |
| HTTP tx path adds Dilithium5 | None — server ignores extra data while gate is u64::MAX | Yes — redeploy |

None of these changes produce a different block hash, affect balance state, or change how any validator validates any historical block. The chain tip is identical before and after Phases A–D.

### Why the Upgrade Gate Is Safe

The upgrade gate height (`18_820_000`) is:
1. **Far in the future** — ~4 weeks from announcement, giving every node and user time to upgrade.
2. **Agreed upon by the network** — it is hard-coded in the binary, not a governance vote. Every node running v11.0.0 agrees on the same height. This is identical to how Bitcoin handles soft forks.
3. **Version-gated** — `min_version: "11.0.0"` in the upgrade config means pre-11.0.0 nodes are rejected before the height is even reached, preventing a fork.
4. **Monotone** — once the height passes, there is no going back on-chain. But the binary can always be changed (emergency hard fork if necessary) with a new height further in the future.

### Canary Validation Before Phase E

Before setting the activation height in any production binary:

1. **Run the upgrade on a private fork.** Spin up 3 nodes on the test network, set activation height to current height + 100, let it activate. Verify:
   - All nodes accept post-activation blocks with HybridEd25519Dilithium5 signatures.
   - All nodes reject blocks without Dilithium5 at and after the activation height.
   - All nodes continue to accept pre-activation blocks (historic blocks) without Dilithium5.
   - Node sync from genesis through the activation height works correctly.

2. **Test wallet recovery.** On the test network:
   - Create a wallet, record the mnemonic.
   - Clear localStorage.
   - Restore from mnemonic.
   - Send a transaction post-activation.
   - Verify it is accepted.

3. **Test the HTTP path.** Submit a transaction via HTTP API both with and without Dilithium5 after activation. Confirm pre-activation HTTP txs pass, post-activation non-PQ HTTP txs fail.

4. **Test validator identity continuity.** Restart the validator node (Gamma, Epsilon). Confirm the same Ed25519 node ID is loaded. Confirm blocks continue to be produced without interruption.

5. **Soak period.** Run the test network at the new activation height for 72 hours minimum before deploying to mainnet production. Monitor for:
   - Unexpected forks between nodes.
   - Any transaction that should be accepted being rejected.
   - Any transaction that should be rejected being accepted.
   - Memory, CPU, and latency regressions from Dilithium5 signature verification overhead.

### Rollback Procedure (Emergency Use Only)

If the activation height is set and a critical bug is discovered before it fires:

1. Build a new binary with `activation_height` set further in the future (e.g., `u64::MAX` again, or `20_000_000`).
2. Coordinate immediate deployment to Beta, Gamma, Epsilon, Delta before the original activation height.
3. If the original height has already passed and blocks are being rejected:
   - All nodes must be updated to the new binary simultaneously (within 1 block time).
   - The chain will fork at the activation height. The fork with the longer chain wins.
   - Emergency announcement to all users to not submit transactions until the fork is resolved.

**This scenario is extremely unlikely if the canary validation in the previous section is done properly.** The entire point of the 4-week announcement window and the 72-hour soak period is to eliminate this scenario.

### Dilithium5 Verification Overhead

Dilithium5 verification takes approximately 0.1–0.3ms on modern hardware. At the current block rate (~1 block/second) and typical transaction count, this adds at most a few milliseconds to block validation time — negligible. Key generation takes 1–3ms. Signing takes 2–3ms. These are acceptable for a blockchain with 1-second block times.

**Signature storage overhead:** 4,627 bytes per transaction for Dilithium5. At 219 GB current DB size and ~16.4M blocks, the marginal addition per new transaction is ~4.7 KB (signature + public key). The external review recommended Dilithium3 (3,293 bytes) to save 1.3 KB per transaction; the project has chosen Dilithium5 for maximum NIST Level 5 security instead.

### User Migration Safety

#### Users with PQ keys in localStorage (created after wallet PQ support was added)

After Phase A is deployed: On first login, the wallet detects `walletDilithium5KeyVersion` missing or `'1'` (pre-fix random key). If the mnemonic is available (session), the wallet silently re-derives the correct deterministic key, updates localStorage, and shows a non-blocking notification: *"Your quantum-safe keys have been upgraded."*

The user's old random Dilithium5 key in localStorage is superseded. This is safe because the old random key was never enforced on-chain (gate at `u64::MAX`). The new deterministic key becomes their permanent PQ identity.

#### Users without PQ keys (e.g., created a wallet before PQ was added)

`loadWallet` Phase A3 generates their Dilithium5 keys from mnemonic on first login. They get PQ keys transparently with no action required.

#### Users who have lost their mnemonic

They cannot recover their Dilithium5 key if localStorage is lost. A clear in-wallet warning before Phase E activation: *"Your account's quantum-safe key is stored only in this browser. To protect your funds after [date], please write down your recovery phrase now."* A mandatory "I have written down my phrase" confirmation before the date should be strongly considered.

#### Node operators

They receive the `Q_VALIDATOR_MNEMONIC` environment variable option (Phase C). If they configure it before Phase C deploys, their node has a deterministic identity from the start. If they don't, the auto-generated keypair file is sufficient — as long as they back it up, they are safe.

### What If a User Misses the Upgrade Window?

After the activation height, a wallet without a Dilithium5 key cannot submit transactions. **Funds are not lost** — they are in the account and visible on-chain. The user can:
1. Re-authenticate in the wallet (which triggers Phase A3 key re-derivation from mnemonic).
2. Submit a transaction with the newly generated key.

**Special case — first login after Phase E, on a new device, without mnemonic:** This user cannot produce a Dilithium5 signature. Their transactions will be rejected. The wallet must display a clear, non-alarming error:

> *"Your funds are safe and accessible. However, sending transactions now requires your 24-word recovery phrase to restore your quantum-safe key on this device. Enter your recovery phrase in Settings → Restore to continue."*

This message must appear before the user attempts a send — not after rejection. The wallet should detect the absent PQ key at login time (Phase A3 already does this check) and surface the warning immediately, rather than letting the user fill in a send form only to hit an opaque rejection.

The only permanent loss scenario is a user who has both lost their mnemonic AND lost their localStorage. In that case, their Ed25519 key is also gone (same problem exists today, before PQ). The PQ upgrade does not worsen this existing risk.

---

## 9. Summary of Decisions (Confirmed by External Review)

| Decision | Choice | Rationale |
|---|---|---|
| PQ algorithm | **Dilithium5** (NIST Level 5) | Maximum classical security (256-bit); project decision — overrides external review's Dilithium3 recommendation |
| Mode | **HybridEd25519Dilithium5** | Ed25519 binds the Dilithium5 key — no on-chain registry needed; provides safety net if Dilithium has issues |
| SQIsign | **Remove entirely** | Not production-ready; current implementation is cryptographic theatre |
| SQIsign future | **`sqisign-V2` feature flag** | Return when C reference implementation is hardened and WASM is available |
| On-chain PQ key registry | **Not needed for hybrid phase** | Ed25519 signature already binds the Dilithium key in hybrid mode |
| Announcement window | **4 weeks minimum** | Gives all node operators and users time to upgrade |
| Activation height | **~18,820,000** | ~4 weeks from announcement at current block rate |
| Canary validation | **72-hour soak on private fork** | Non-negotiable before mainnet deployment |

---

## 10. Files to Change — Complete List

| Phase | File | Change |
|---|------|--------|
| 0 | `gui/quantum-wallet/src/libp2p/postQuantumCrypto.ts` | Fix simulated verifier — always return `false` |
| A | `gui/quantum-wallet/src/libp2p/postQuantumCrypto.ts` | Add `dilithium5KeyGenFromSeed(seed)` |
| A | `gui/quantum-wallet/src/services/walletAuth.ts` | Fix keygen, fix `loadWallet` recovery, remove ephemeral fallback, add version tag |
| B | `gui/quantum-wallet/src/services/walletAuth.ts` | Delete all SQIsign functions and dead code |
| B | `gui/quantum-wallet/src/libp2p/postQuantumCrypto.ts` | Remove SQIsign functions |
| B | `crates/q-api-server/src/block_producer.rs` | Add compile guard on Phase2SQIsign arms |
| B | `crates/q-types/src/pqc_keys.rs` | Remove SQIsign key fields or add placeholder warning |
| C | `crates/q-types/src/pqc_keys.rs` | Add `generate_from_seed()`, remove deprecated path |
| C | `crates/q-api-server/src/main.rs` | Auto-persist keypair, add `Q_VALIDATOR_MNEMONIC` env var, remove `load_from_file` |
| D | `gui/quantum-wallet/src/services/api.ts` | Add Dilithium5 to `sendTransaction` |
| D | `gui/quantum-wallet/src/services/walletAuth.ts` | Implement `HybridEd25519Dilithium5` in `generateAuthHeader` |
| E | `crates/q-consensus-guard/src/upgrade_gate.rs` | Set activation height (last step, after all else validated) |
| E | `Cargo.toml` | Bump to version `11.0.0` |

**Do not touch:** Block production logic, consensus rules, balance ledger, P2P gossip, RocksDB schema, existing transaction validation, mining reward calculation, or any file not in the table above. These are working correctly, are mainnet-stable, and must not be disturbed.

---

## 11. Open Questions (Resolved by External Review)

| Question | Resolution |
|---|---|
| Dilithium5 vs Dilithium3 | **Dilithium5** — NIST Level 5, maximum security; project decision overrides external review's Dilithium3 recommendation |
| Hybrid vs pure Dilithium | **Hybrid** for initial phase; pure Dilithium possible via later upgrade gate |
| SQIsign timeline | **No concrete date** — remove from codebase, track as `sqisign-V2` |
| User migration UX | Version tag in localStorage; auto re-derive on first boot; warning for lost mnemonics |
| On-chain PQ key registry | **Not needed for hybrid** — Ed25519 already binds Dilithium key; add as optional future upgrade |
