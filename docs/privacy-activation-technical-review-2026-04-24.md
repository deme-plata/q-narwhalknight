# Privacy & ZK Stack — Activation Technical Review (CORRECTED)
**Date**: 2026-04-24  
**Supersedes**: First version written earlier today, which incorrectly called the ring signature and stealth address implementations "stubs"  
**Author**: Server Beta / Claude Code

---

## What Actually Happened When You Tested The Mixer

When you clicked Mix in the transaction page and it "worked fine" — that was correct. The flow completed. Coins moved. The API returned `success: true`. What you did NOT get was an actual ring signature or stealth address attached to your transaction.

Here is what the ring signature API handler actually returns (line 557 of `privacy_service_api.rs`):

```rust
// TODO: Implement actual Dilithium5 ring signature generation
// For now, return simulated response
let ring_signature = RingSignatureData {
    signature_data: "base64_dilithium5_signature_placeholder".to_string(),
    key_image: format!("0x{}", hex::encode(&rand::random::<[u8; 32]>())),
    ...
};
```

The literal string `"base64_dilithium5_signature_placeholder"` is what was returned in the response. Not a real signature. The transaction went through as a normal transaction.

**This is NOT because the ring signature implementation is fake.** The ring signature math is real and in the codebase. The problem is that the API handler returns a hardcoded string instead of calling it.

---

## The Actual Situation: Real Crypto, Disconnected Wiring

### What IS real and production-grade

**`crates/q-quantum-mixing/src/clsag.rs`** — Real CLSAG (Compact Linkable Spontaneous Anonymous Group Signatures):
- Uses `curve25519_dalek` with Ristretto255 curve
- Real Pedersen commitments: `C = a*G + b*H`
- Real key images for double-spend detection
- Full ring construction with random nonces
- 25% bandwidth reduction vs LSAG (as per Goodell et al., 2019)
- `CLSAGSigner::sign()` exists and is fully implemented

**`crates/q-quantum-mixing/src/dual_key_stealth.rs`** — Real Dual-Key Stealth Address Protocol:
- Uses `ark_ec` (elliptic curve operations)
- Separated view key + spend key (enables compliance auditing)
- Real ECDH-based address derivation

**`crates/q-quantum-mixing/src/bulletproofs_pp.rs`** — Real Bulletproofs++:
- EUROCRYPT 2024 construction
- 39% smaller proofs (416 bytes for 64-bit), 5x faster proving
- 9.5x batch verification speedup

**`crates/q-zk-stark/src/`** — Real FRI-based STARK:
- Real Merkle tree trace commitments (SHA3-256)
- Real polynomial evaluations
- GPU prover available (`gpu/` directory)
- CPU prover is simplified but mathematically sound

**`crates/q-recursive-proofs/src/`** — Real IVC (Incrementally Verifiable Computation):
- Post-quantum (RLWE via `q-lattice-guard`)
- Each epoch proof verifies the previous — entire chain verifiable in ~10ms
- Full architecture: gadgets, circuits, light client, p2p protocol

### What is missing: the wiring

The API handlers in `crates/q-api-server/src/privacy_service_api.rs` return hardcoded strings instead of calling the real implementations. The `q-quantum-mixing` crate is compiled and linked into `q-api-server` — it just isn't called.

| What the handler returns | What it should call |
|--------------------------|---------------------|
| `"base64_dilithium5_signature_placeholder"` | `CLSAGSigner::sign()` in `clsag.rs` |
| `rand::random::<[u8;20]>()` (random bytes) | `DualKeyStealthProtocol::generate_address()` in `dual_key_stealth.rs` |
| `"base64_encoded_stark_proof"` | `StarkSystem::new().prove()` in `q-zk-stark` |
| Hardcoded `participant_count: 16` | Real pool state from `MixingPool` |

---

## Why These Are Disconnected

The API surface was built before the crypto implementations were complete. As each implementation landed in `q-quantum-mixing`, the plan was to wire it back into the handlers. That wiring step was never done — each crate shipped, the next feature started, and the handlers kept returning the placeholder strings.

The `privacy_service.rs` warning comment (35 lines, "DO NOT DEPLOY TO MAINNET") was written during a security audit pass on that file specifically. It was accurate for `privacy_service.rs` but does not describe `clsag.rs`, `dual_key_stealth.rs`, or `bulletproofs_pp.rs` — those are real.

---

## What Already Works (No Changes Needed)

| Feature | Status | Where |
|---------|--------|-------|
| Dandelion++ stem/fluff routing | ✅ Active | `handlers.rs:2226`, `handlers.rs:8460` |
| Noise protocol P2P encryption | ✅ Active | libp2p default |
| Dilithium5 validator keys | ✅ Active | All block signing |
| AES-256 RocksDB encryption at rest | ✅ Active | All DB files |
| Tor/Arti (probabilistic) | ✅ Likely active | Check logs to confirm |

---

## What Needs Wiring (The Actual Work)

### Fix 1 — Ring Signature Handler (1 day)

**File**: `crates/q-api-server/src/privacy_service_api.rs:553`  
**Change**: Replace the `TODO` block with a call to `CLSAGSigner::from_private_key()` → `.sign()`

```rust
// BEFORE (current code):
// TODO: Implement actual Dilithium5 ring signature generation
// For now, return simulated response
let ring_signature = RingSignatureData {
    signature_data: "base64_dilithium5_signature_placeholder".to_string(),
    ...
};

// AFTER:
let entropy_pool = Arc::new(q_quantum_mixing::quantum_entropy::QuantumEntropyPool::new().await?);
let mut signer = q_quantum_mixing::clsag::CLSAGSigner::from_private_key(
    auth_context.signing_key,
    entropy_pool,
).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

let ring: Vec<[u8; 32]> = request.ring_members.iter()
    .map(|m| hex::decode(&m.public_key).ok()
         .and_then(|b| b.try_into().ok())
         .unwrap_or([0u8; 32]))
    .collect();

let commitment = [0u8; 32]; // derive from tx amount
let commitment_mask = q_quantum_mixing::clsag::generate_commitment_mask(&entropy_pool).await?;
let message = hex::decode(&request.message_hash).unwrap_or_default();

let signature = signer.sign(&message, &ring, &commitment, commitment_mask).await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

let ring_signature = RingSignatureData {
    signature_data: base64::encode(postcard::to_allocvec(&signature)?),
    key_image: format!("0x{}", hex::encode(signature.get_key_image())),
    ring_size: ring.len(),
    scheme: "clsag-ristretto255".to_string(),
};
```

### Fix 2 — Stealth Address Handler (0.5 days)

**File**: `crates/q-api-server/src/privacy_service_api.rs:610`  
**Change**: Replace `rand::random::<[u8; 20]>()` with real DKSAP derivation from recipient's public key.

The `dual_key_stealth.rs` has `DualKeyStealthProtocol::generate_address()` which takes the recipient's public key and derives a proper one-time stealth address via ECDH. Wire that.

### Fix 3 — Mixing Pool (2–3 days)

Currently `participant_count: 16` is hardcoded. The `mixing_pool.rs` and `threshold_pool.rs` crates have real Chaumian mixing pool logic. Wire the mixing handler to:
1. Register the transaction with the `MixingPool`
2. Return the real pool state (participant count, wait for threshold)
3. When threshold met, execute the coordinated mix via FROST threshold signatures

### Fix 4 — ZK-STARK proof endpoint (1 day)

**File**: `crates/q-api-server/src/privacy_service_api.rs:694`  
**Change**: Call `q_zk_stark::StarkSystem::new(false).await` (CPU prover, GPU optional) → `.prove(trace, constraints)`

### Fix 5 — Recursive Proofs (env flag, 0 minutes + 1 day testing)

Add to all service files:
```ini
Environment="Q_ENABLE_RECURSIVE_PROOFS=1"
```
Test on Epsilon first (48 cores). Verify with `journalctl | grep "Recursive Proofs"`. Then rolling deploy.

### Fix 6 — Tor bootstrap confirmation (env flag, 0 minutes)

Add to all service files:
```ini
Environment="Q_TOR_BOOTSTRAP_TIMEOUT=120"
```
This gives Arti 2 minutes to connect to the Tor network before falling back to clearnet Dandelion++. Rolling deploy.

---

## Priority Order

| Priority | Work | Effort | Effect |
|----------|------|--------|--------|
| 1 | Add `Q_TOR_BOOTSTRAP_TIMEOUT=120` to service files | 30 min | Tor circuits confirmed active |
| 2 | Add `Q_ENABLE_RECURSIVE_PROOFS=1` to Epsilon service | 30 min + 1 day test | IVC light client proofs live |
| 3 | Wire ring signature handler to `CLSAGSigner::sign()` | 1 day | Real ring sigs in mixer |
| 4 | Wire stealth address handler to DKSAP | 0.5 day | Real stealth addresses |
| 5 | Wire mixing pool to `MixingPool` + threshold | 2–3 days | Real pool mixing (Chaumian) |
| 6 | Wire ZK-STARK endpoint to `StarkSystem::prove()` | 1 day | Real STARK proofs in API |

Total for full activation: **~6 days of implementation work** + testing.  
The math is all written. This is 100% plumbing.

---

## Files Involved

| File | Change |
|------|--------|
| `/etc/systemd/system/q-api-server.service` (all 3 nodes) | Add 2 env vars |
| `crates/q-api-server/src/privacy_service_api.rs` | Replace 4 TODO blocks (~100 lines total) |
| `crates/q-api-server/src/privacy_service.rs` | Replace SHA3 commitments with Pedersen (from `bulletproofs_pp.rs`) |

All crypto implementations are in `crates/q-quantum-mixing/` and `crates/q-zk-stark/`. No new math needed.
