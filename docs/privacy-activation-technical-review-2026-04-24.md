# Privacy & ZK Stack — Activation Technical Review
**Date**: 2026-04-24  
**Author**: Server Beta / Claude Code  
**Context**: Full audit of ZK-STARK, Tor, Dandelion++, ring signatures, recursive proofs — what is genuinely active, what is disabled, and what is a placeholder that cannot be safely activated.

---

## Executive Summary

After a full audit of the codebase, the privacy stack has **three distinct tiers**:

| Tier | Status | Examples |
|------|--------|---------|
| **A — Working, active in prod** | ✅ Running now | Dandelion++, Noise encryption, Dilithium5 keys |
| **B — Working code, disabled by env flag** | 🔒 One env var away | Recursive proofs, Tor with Arti |
| **C — Placeholder — explicitly not production-ready** | ❌ Cannot activate safely | Ring signatures (XOR), stealth addresses (no ECDH), ZK via privacy service |

The reason these aren't all active is simple: Tier C was written as scaffolding with the intent to replace the internals later. The code compiles and returns plausible-looking results — but provides **zero actual cryptographic privacy guarantees** for those paths. `privacy_service.rs` says exactly this in a 35-line warning comment at the top of the file, authored when the decision was made not to complete them.

---

## Tier A — Active Right Now (Nothing to do)

### 1. Dandelion++ Transaction Privacy ✅
**Where**: `crates/q-tor-client/src/dandelion.rs`, wired via `handlers.rs:2226` and `handlers.rs:8460`

Every transaction submitted to the API goes through stem→fluff routing before gossipsub broadcast. The stem phase uses `.onion`-only relay candidates (IP leak fix, v3.4.2). Quantum-seeded ChaChaRng timing obfuscation is active. This is genuine and running on every transaction on every node.

**Nothing to do. Already live.**

### 2. Noise Protocol Encryption (all P2P) ✅
Every libp2p connection uses Noise XX handshake. All gossipsub traffic is encrypted in transit between nodes. Standard libp2p — always active.

### 3. Dilithium5 Post-Quantum Validator Keys ✅
Validator keypairs generated with `generate_with_zk_stark_untrusted()` use Dilithium5 for signing. This is the CRYSTALS-Dilithium lattice signature scheme — real post-quantum security, standardized by NIST. Active on all nodes.

### 4. AES-256 RocksDB Encryption at Rest ✅
All database files are encrypted at rest via the RocksDB encryption layer. Keys are auto-generated per node on first boot. Active.

---

## Tier B — Real Code, Disabled by Environment Variable

### 5. Tor (embedded Arti) — Probabilistically Active 🔒
**How to verify**: Run on Beta/Epsilon/Gamma:
```bash
journalctl -u q-api-server --since "2 hours ago" | grep -E "Tor client|🧅" | head -5
```
Look for `✅ Tor client initialized successfully`.

**If not active**, the blocker is Tor bootstrap timing. Fix:
```ini
# Add to /etc/systemd/system/q-api-server.service
Environment="Q_TOR_BOOTSTRAP_TIMEOUT=120"
```
Then `systemctl daemon-reload && systemctl restart q-api-server` via ha-deploy.

**What Tor adds when active**: Every Dandelion++ stem hop goes through a real 3-hop Tor circuit. Without it, stem hops are direct P2P connections — still private routing, but IPs visible to each relay node.

**Risk**: None. The existing code already handles Tor failure gracefully (falls back to clearnet Dandelion++). Adding the timeout just gives Arti more time to bootstrap.

### 6. Recursive IVC Proofs — Disabled 🔒
**Flag**: `Q_ENABLE_RECURSIVE_PROOFS=1`  
**Sub-flag**: `Q_ENABLE_PROVER=1` (makes THIS node participate in proof generation)

**What it does**: Post-quantum recursive SNARKs using `q-lattice-guard` (RLWE). Each epoch proof verifies the previous one. Light clients can verify the entire chain in ~10ms instead of replaying history. Uses `LatticeGuardProof` — quantum-resistant.

**Caveats before enabling**:
- The recursive proof service was added in v1.4.0-beta and has not been load-tested on mainnet height (~16M blocks). The epoch proof generation is CPU-heavy.
- The `Q_ENABLE_PROVER=1` sub-flag should only be set on high-CPU nodes (Epsilon, 48 cores). Bootstrap nodes should run with `Q_ENABLE_RECURSIVE_PROOFS=1` but NOT `Q_ENABLE_PROVER=1` — they verify only.
- Need to confirm `q-lattice-guard` compiles and initializes without panic on live data before enabling in prod service files.

**Activation plan**:
```bash
# Step 1: Test on Epsilon only (48 cores, can absorb the CPU cost)
ssh root@89.149.241.126 "
  Q_ENABLE_RECURSIVE_PROOFS=1 Q_ENABLE_PROVER=0 \
  /opt/orobit/shared/q-narwhalknight/q-api-server-v889 --port 8090
" 2>&1 | grep -E "Recursive|LatticeGuard|ERROR|panic" | head -20
```
If clean, add to service file and redeploy.

---

## Tier C — Placeholder Implementations (Cannot Activate Safely)

These are the ones that look like they should work but provide **zero actual cryptographic guarantees**. The `privacy_service.rs` file has a 35-line CRITICAL SECURITY WARNING at the top explaining each failure. They are documented here to explain why they are NOT being activated.

### 7. Ring Signatures — XOR, Not Elliptic Curve ❌
**Location**: `crates/q-api-server/src/privacy_service.rs`, `q-quantum-mixing` crate  
**What the code does**: Takes a list of "decoys" and performs byte-level XOR operations to produce a "ring signature".  
**What ring signatures should do**: Use elliptic curve operations (Schnorr or MLSAG/CLSAG) so that a verifier cannot determine which key in the ring actually signed. XOR provides no unlinkability — a trivial statistical analysis can identify the real signer.  
**To make it real**: Implement MLSAG or CLSAG using `curve25519-dalek` or `k256`. Estimated effort: 3–5 days for a competent cryptographer.

### 8. Stealth Addresses — No ECDH ❌
**Location**: `privacy_service.rs`, stealth address service  
**What the code does**: SHA3-hashes the recipient's public key to produce a "stealth address".  
**What stealth addresses should do**: Use ECDH (Elliptic Curve Diffie-Hellman) so the sender can derive a one-time address only the recipient can spend. SHA3 of the public key is just a deterministic alias — anyone who knows the public key can link all payments to it.  
**To make it real**: Implement Monero-style dual-key stealth addresses using `curve25519-dalek`. Estimated effort: 2–3 days.

### 9. ZK-STARK Balance Commitments — SHA3, Not Pedersen ❌
**Location**: `privacy_service.rs` balance commitments  
**What the code does**: Commits to a balance using `SHA3(amount || blinding_factor)`.  
**What Pedersen commitments provide**: Homomorphic hiding — you can prove sums and ranges without revealing values. SHA3 commitments are not homomorphic and can be brute-forced for small amounts (under ~1B QUG: ~2^64 operations, feasible).  
**To make it real**: Use the `bulletproofs` crate (based on Ristretto255) or `halo2`. Estimated effort: 1 week including range proof integration.

### 10. ZK-STARK Proof Service — Not Implemented in Privacy API ❌
**Location**: `privacy_service_api.rs` → `zk_stark_proof_service` handler  
**Important distinction**: The `q-zk-stark` crate has a real FRI-based STARK prover (with real Merkle commitments and polynomial evaluations). The **privacy service API endpoint** wraps a different, unfinished path that does not connect to the real STARK prover.  
**To make it real**: Wire `privacy_service_api::zk_stark_proof_service` to call `q_zk_stark::StarkSystem::new(false).await?.prove(trace, constraints)` instead of the current placeholder path. Estimated effort: 1–2 days (the hard math is already in `q-zk-stark`).

### 11. SQIsign / AES-GCM in Tor Init — Placeholders ❌
**Location**: `crates/q-tor-client/src/lib.rs:614–650`  
**What the code does**:  
- "SQIsign signature": `sqisign_signature[i] = sig_seed[i % 32] ^ (i as u8)` — deterministic XOR, not SQIsign  
- "AES-256-GCM": Simple XOR with key — comment literally says "Simple XOR encryption as placeholder"  
**What these protect**: The ZK proof attached to Tor initialization (proves correct setup without revealing keys).  
**To make it real**: SQIsign (NIST Round 2 candidate) — use the `sqisign` or `pqcrypto-sqisign` crate; AES-GCM — use the `aes-gcm` crate (already in Cargo.toml elsewhere). Estimated effort: 1 day (crypto plumbing, not new math).

---

## What To Do, In Priority Order

### Phase 1 — Activate now, zero code changes (1 hour)
1. **Confirm Tor is bootstrapping**: Check logs on all 3 nodes. If not:
   ```bash
   # Add to service files on Beta, Gamma, Epsilon:
   Environment="Q_TOR_BOOTSTRAP_TIMEOUT=120"
   ```
   Then rolling deploy via ha-deploy.sh.

2. **Deploy v10.4.1** (currently building — includes emission fallback fixes): Rolling deploy once binary is ready.

### Phase 2 — Activate with one flag, low risk (1 day, test on Epsilon first)
3. **Recursive Proofs** (`Q_ENABLE_RECURSIVE_PROOFS=1`): Test on Epsilon with a canary run first. If clean, add to all service files and rolling deploy. Verify with:
   ```bash
   journalctl -u q-api-server | grep "Recursive Proofs"
   ```

### Phase 3 — Wire real STARK prover to privacy API (2 days)
4. **ZK-STARK proof service**: Remove the stub path in `privacy_service_api.rs::zk_stark_proof_service`. Replace with a call to `q_zk_stark::StarkSystem`. The math is written — this is plumbing only.

### Phase 4 — Implement real cryptographic primitives (2–3 weeks total)
5. **AES-GCM in Tor init**: Replace XOR with `aes-gcm` crate. 1 day.
6. **SQIsign in Tor init**: Replace XOR padding with `pqcrypto-sqisign`. 1 day.
7. **Stealth addresses**: Implement ECDH with `curve25519-dalek`. 2–3 days.
8. **Ring signatures**: Implement MLSAG/CLSAG. 3–5 days (requires cryptographer review).
9. **Pedersen commitments / bulletproofs**: Replace SHA3 commitments with `bulletproofs` crate. 5–7 days including range proof integration.

### Phase 5 — Protocol upgrade (6 weeks, after Phase 2B Bracha BRB)
10. **Emission state gossipsub sync** (Bracha BRB, already designed): Wire `EmissionSyncBrb` into startup, add 4 gossipsub topics, testnet-validate.
11. **Embed `emission_cumulative` in block headers**: Consensus-finalized emission checkpoint on every block.

---

## Why These Were Not Done Before

The honest answer: the privacy service scaffolding was written to define the API surface and data types first, with the intent to fill in real cryptographic implementations incrementally. This is a standard approach for rapid prototyping — define the interface, ship the structure, implement the math later. The code even says so with `// placeholder` and `// Simple XOR as placeholder (real implementation would use AES-GCM)`.

The risk is that the scaffolding looks working from the outside (the endpoints respond, the proofs serialize and return data) but provides no actual security. This is caught here rather than in production.

**Nothing broken. No security incident.** The active path (Dandelion++, Noise, Dilithium5, AES-256 DB encryption) is genuine. The placeholder paths are gated behind API endpoints that require authentication and are not part of the core transaction flow. The chain's consensus, block validation, and fund security are unaffected.

---

## Files to Change for Each Phase

| Change | File | Lines |
|--------|------|-------|
| Tor bootstrap timeout | `/etc/systemd/system/q-api-server.service` | Add 1 env var |
| Recursive proofs enable | `/etc/systemd/system/q-api-server.service` | Add 1 env var |
| Wire STARK to privacy API | `crates/q-api-server/src/privacy_service_api.rs` | ~20 lines |
| AES-GCM in Tor init | `crates/q-tor-client/src/lib.rs:645–660` | ~15 lines |
| SQIsign in Tor init | `crates/q-tor-client/src/lib.rs:603–617` | ~20 lines + crate dep |
| Stealth addresses | `crates/q-quantum-mixing/src/stealth.rs` | ~100 lines |
| Ring signatures | `crates/q-quantum-mixing/src/ring.rs` | ~200 lines |
| Bulletproofs | `crates/q-api-server/src/privacy_service.rs` | ~300 lines |
