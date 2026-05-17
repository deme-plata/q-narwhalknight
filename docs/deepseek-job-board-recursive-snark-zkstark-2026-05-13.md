# DeepSeek Job Board — Recursive SNARK + zk-STARK Track

**Date:** 2026-05-13
**Project:** Quillon Graph — Live Mainnet, ~$2 B USD Market Cap
**Purpose:** A focused list of self-contained code jobs DeepSeek can take in parallel. Each job has a clear file path, public API, acceptance criteria, and a "do not touch" boundary.

---

# 🚨 MAINNET CONSTRAINTS — UNCHANGED

The rules from prior handoffs apply unmodified. Read them first if you haven't:

1. **No DB schema changes.** `cf_balance_smt` (already shipped) is the only new CF; do not add others. Do not touch existing CFs.
2. **No `balance_root_v1` modification anywhere.**
3. **No `save_wallet_balance` / `save_wallet_balances` modification without explicit Beta sign-off.**
4. **No `unsafe`, no `unwrap()` in non-test code.**
5. **Use `code.quillon.xyz` — NOT GitHub.** Branch name: `deepseek/<job-letter>-<description>`. Run `git update-server-info` after push.
6. **One Job at a time.** File-ownership matrix is per-job; don't grab a job already in flight.
7. **Push to integration branch `ivc/v1`** after Beta reviews.
8. **Soak before activation.** Code lands → ≥ 1 week of test-suite soak → mainnet activation gated on `BALANCE_ROOT_V2_HEIGHT`-style height upgrade.

If any of these are unclear, **ask in `#dev-coordination` before coding**. The cost of a clarifying question is zero; the cost of a misplaced edit on a $2 B chain is enormous.

---

# JOB DEPENDENCY GRAPH

```
                                 ┌─── E (Nova-SRS AIR) ──┐
                                 │                       │
                                 ├─── F (Batch-Dilithium │
                                 │     AIR)              │
                                 │                       ├──> J (Cross-chain
   A (Multi-block BLAKE3) ──┐    ├─── G (Wire protocol   │      bridge AIR)
                            │    │     scaffold)         │
                            ▼    │                       ├──> K (Activation-
   B (Merkle gadget) ───────────►├─── I (Mobile-light    │     height AIR)
                            │    │     STARK)            │
                            ▼    └───────────────────────┘
   C (δ-circuit) ───────────►
                            │
                            ▼
   D (Nova IVC wrapper) ──────────────────────► H (WASM Phase 2)
```

**Jobs on the same horizontal line are independent.** A-B-C-D-H is the critical path. E/F/G/I are parallel-able and can start TODAY.

---

# JOB A — Multi-block BLAKE3 Gadget (RETRY)

**Status:** Specced in `docs/deepseek-handoff-blake3-multiblock-2026-05-13.md`. Two prior DeepSeek iterations produced placeholders. **The handoff doc still applies unchanged.**

**Why this is still Job A:** without `hash_message`, Jobs B/C/D all block. The Merkle gadget needs to hash 75-byte node messages; the δ-circuit needs to hash block headers > 1 KB. The single-block `verify_hash` in `crates/q-ivc/src/gadgets/blake3.rs` doesn't do either.

**Re-read this section of the handoff doc** before retry: §C ("What `compress` does, exactly") and §D ("The 10 most common ways this will go wrong"). The pattern of producing `// implement BLAKE3 here ...` placeholder code is rejected on sight; if you can't write the multi-block tree mode, escalate rather than pretend.

**Acceptance criteria** (copy into your PR description):
- [ ] Empty input cross-checks against `blake3::hash(b"")`
- [ ] 64-byte input cross-checks against `blake3::hash(...)` AND matches `Blake3Gadget::verify_hash` output byte-for-byte
- [ ] 75-byte (Merkle-node-shape) input cross-checks
- [ ] 1024-byte (boundary) input cross-checks
- [ ] 1025-byte (multi-chunk tree mode) input cross-checks
- [ ] 4097-byte (multi-level tree, odd-bubble-up) input cross-checks
- [ ] Constraint count for 75-byte input is between 50K and 150K
- [ ] No file outside `crates/q-ivc/src/gadgets/blake3.rs` modified

**Branch:** `deepseek/a-blake3-multiblock-retry`

---

# JOB B — Merkle-path Gadget

**Status:** Blocked on Job A. Specced in `docs/deepseek-handoff-merkle-and-delta-2026-05-13.md` Blueprint 1B. **Reference that doc and do not re-spec.**

**Update vs earlier:** the underlying `BalanceSmt` (in `crates/q-storage/src/balance_smt.rs`) is shipped and production-grade. Tests for this Merkle gadget MUST call `BalanceSmt::prove()` to generate real fixtures — do not mock them. Add `q-storage` and `tempfile` as dev-dependencies to `crates/q-ivc/Cargo.toml`. Production deps unchanged.

**Acceptance criteria:**
- [ ] All 8 tests from Blueprint 1B's required-tests list pass
- [ ] Cross-check test: produce a `SmtProof` via `BalanceSmt::prove()`, feed into `MerklePathGadget::enforce_membership`, assert satisfied
- [ ] Adversarial: tamper one sibling byte → assert UNSAT
- [ ] Adversarial: lie about balance → assert UNSAT
- [ ] Constraint count for one 256-depth path is between 100K and 600K (sanity)

**Branch:** `deepseek/b-merkle-gadget`

---

# JOB C — δ-circuit Composition

**Status:** Blocked on Job B. Specced in `docs/deepseek-handoff-merkle-and-delta-2026-05-13.md` Blueprint 2. **Reference that doc.**

**Update vs earlier:** the corrections from my second review of DeepSeek's submission must be applied:

1. `nonce`, `from_addr`, `to_addr`, `fee` are **witnesses**, not constants. The earlier version made them `UInt8::constant(b)` which freezes the circuit per-transaction — non-functional for Nova IVC. Use `UInt8::new_witness` for all per-tx fields.
2. **Drop the postcard-offset extraction of `state_root` and `height` from the header bytes.** The `Blake3Gadget::verify_hash` over the full header bytes already binds them. Mark this as `// PHASE 1.5: TODO` if defensive belt-and-braces is wanted later.
3. Use `F::from(...)` not `Fr::from(...)` in the generic context. Missing `ConstraintSynthesizer` import.

**Acceptance criteria:**
- [ ] All 9 tests from Blueprint 2's required-tests list pass, including 5 adversarial cases
- [ ] Empty block (coinbase-only) → satisfied
- [ ] Single transfer A→B with valid Merkle paths → satisfied
- [ ] Adversarial: amount > balance → UNSAT
- [ ] Adversarial: forged signature → UNSAT
- [ ] Adversarial: state_root_next inconsistent with computed → UNSAT
- [ ] Constraint count for 5-transaction block is between 25M and 150M

**Branch:** `deepseek/c-delta-circuit`

---

# JOB D — Nova IVC Wrapper

**Status:** Blocked on Job C. Specced in `docs/blueprints-ivc-snark-2026-05-13.md` Blueprint 3.

**Pick the Nova implementation first.** Two candidates:
- `nova-snark` (Microsoft) — used in production by Lurk, Hyle. Recommended.
- `arkworks-rs/nova` (community) — more arkworks-native.

Spend ~1 working day prototyping a minimal `StepCircuit` against a toy circuit on each, then commit to one. Document the choice in the PR description with benchmarks (prover time per fold, verifier time, proof size).

**File layout:**
```
crates/q-ivc/src/recursion/
├── mod.rs           ~50 LOC
├── step.rs          ~400 LOC — StepCircuit impl wrapping DeltaBlockCircuit
├── folder.rs        ~300 LOC — Fold orchestrator (QnkFolder struct)
└── verify.rs        ~150 LOC — Verifier helper
```

**Public API contract** (stable across all Phase 2-3-4 deployments):
```rust
pub struct QnkFolder { /* opaque */ }
impl QnkFolder {
    pub fn setup() -> Result<Self>;
    pub async fn fold_block(&mut self, delta: DeltaBlockCircuit<F>) -> Result<()>;
    pub fn current_proof(&self) -> Result<Vec<u8>>;
    pub fn verify(proof_bytes: &[u8], expected_state_root: &[u8; 32], height: u64) -> Result<bool>;
}
```

The `verify` function must be the same byte-shape the WASM verifier (Job H) consumes.

**Acceptance criteria:**
- [ ] Fold 0 blocks → trivial proof verifies with genesis state-root
- [ ] Fold 1 block → verify succeeds with `state_root_1`
- [ ] Fold 100 blocks → verify succeeds; serialized proof size constant
- [ ] Adversarial: tamper with serialized proof → verify returns false
- [ ] Verifier latency: < 50ms on Beta (Intel Xeon)
- [ ] Proof size: < 100 KB

**Branch:** `deepseek/d-nova-wrapper`

---

# JOB E — AIR for Nova-SRS Generator (TRANSPARENT SETUP) — START NOW

**Status:** Independent of Jobs A-D. Can start TODAY. Uses existing `crates/q-zk-stark/` infrastructure.

**Why this matters:** Phase 2 deployment ships Nova with STARK-attested transparent setup, removing the multi-party-ceremony requirement (see whitepaper v2 §5). The STARK attests that the Nova SRS was generated correctly from a public seed. The AIR description of the SRS generator is the missing piece.

**File:** `crates/q-zk-stark/src/nova_srs_generator_air.rs` (new, ~400-500 LOC)

**Background reading:**
- `crates/q-zk-stark/src/air.rs` — the AIR trait + framework (429 LOC). **Do not modify.** Implement against the existing trait.
- `crates/q-zk-stark/src/blockchain_state_circuit.rs` — example AIR pattern (410 LOC). Use as the template.
- `crates/q-storage/src/encryption_zkstark.rs` — the production reference for STARK-attested setup (562 LOC). Use as the structural template for `EncryptionKeyProof` analogue, here `NovaSrsProof`.

**What the AIR proves:**
For Nova-on-BN254, the SRS is a sequence of group elements `[G, αG, α²G, ..., αⁿG]` where `α` is the trapdoor. We want a STARK proof that asserts: **"The provided SRS = `gen_srs(seed)` where `gen_srs` is a deterministic public algorithm that maps `seed → α → SRS`."**

The AIR trace columns:
- `seed_bytes[]` — public input (32 bytes)
- `alpha_bits[]` — derived from `seed` via BLAKE3 (witness)
- `g_pow_alpha_i[]` — sequence of accumulated group elements (witness)
- `valid_step[i]` — boolean asserting `g_pow_alpha_{i+1} == α · g_pow_alpha_i`

The trace is `n+1` rows tall where `n` is the SRS size. For Nova on BN254 with circuit size ~10⁶, `n ≈ 10⁶`. Trace length feasible because we batch via the existing `batch_prover.rs`.

**Public API contract:**
```rust
pub struct NovaSrsProof {
    pub seed: [u8; 32],
    pub srs_root: [u8; 32],        // BLAKE3 of the serialized SRS
    pub stark_bytes: Vec<u8>,
}

impl NovaSrsProof {
    /// Generate the SRS deterministically from `seed`, build the STARK attestation.
    /// Run ONCE at genesis (or at SRS rotation). Output committed to chain at
    /// `data-<network>/nova-srs.zkstark`.
    pub fn generate(seed: &[u8; 32], srs_size: usize) -> Result<(Self, Vec<u8> /* the SRS bytes */)>;

    /// Verify that the STARK attests `gen_srs(seed) == srs_root`.
    /// Called by every validator on bootstrap. Verification cost ≈ 5-50 ms.
    pub fn verify(&self) -> Result<bool>;
}
```

**Acceptance criteria:**
- [ ] AIR compiles against `crates/q-zk-stark/src/air.rs::Air` trait
- [ ] Generator round-trips: `generate(seed, 1024)` → `verify()` returns true
- [ ] Adversarial: flip one byte of `srs_root` → `verify()` returns false
- [ ] Verification latency: < 100 ms for srs_size = 1024 on Beta (sanity; production uses larger)
- [ ] No file outside `crates/q-zk-stark/src/nova_srs_generator_air.rs` modified except adding `pub mod nova_srs_generator_air;` to `lib.rs`

**Branch:** `deepseek/e-nova-srs-air`

---

# JOB F — AIR for Batched Dilithium Signature Verification — START NOW

**Status:** Independent. Can start TODAY. Uses existing `crates/q-zk-stark/src/batch_prover.rs` (662 LOC, already implements batched STARK proving).

**Why this matters:** In-circuit Dilithium5 signature verification costs ~1.5M R1CS constraints per signature. A 100-transaction block accumulates 150M constraints from signatures alone — dominating the δ-circuit cost. Replacing in-circuit verification with a single batched STARK proof reduces δ-circuit constraint count by ~30-50% (whitepaper v2 §7.2). The δ-circuit consumes the STARK as a single boolean assertion.

**File:** `crates/q-zk-stark/src/dilithium_batch_air.rs` (new, ~500-700 LOC)

**What the AIR proves:**
For K signatures `(pk_i, msg_i, sig_i)` for `i = 1..K`, the AIR asserts: **"All K signatures verify under Dilithium5."**

Trace columns:
- `pk_i_bytes` (public input per signature)
- `msg_i_hash` (public input, BLAKE3 of msg)
- `sig_components` — the c, z, h parts of the signature (witness)
- `verify_step_valid[i]` — boolean asserting Dilithium verification step at position `i` is correct
- All verify steps must be true for the proof to be valid

The STARK output is a single boolean (folded into the constraint trace). The δ-circuit consumes this as `assert_eq(batch_verify_result, true_boolean)`.

**Reference:**
- `crates/q-ivc/src/gadgets/dilithium.rs` — the existing in-circuit Dilithium primitives. The STARK AIR replicates the same verification logic in the AIR trace form.
- `crates/q-zk-stark/src/batch_prover.rs` — existing batched prover. The new AIR uses this directly.

**Public API contract:**
```rust
pub struct DilithiumBatchProof {
    pub n_signatures: u32,
    pub batched_msg_hash: [u8; 32],  // BLAKE3 of concatenated (pk, msg, sig) tuples
    pub stark_bytes: Vec<u8>,
}

impl DilithiumBatchProof {
    /// Generate a STARK proof for K signatures. Run by the prover/genesis nodes
    /// once per block (before the δ-circuit). K is bounded by the batch_prover's
    /// max trace length (~2^20 = 1M rows for ~K = 1000 signatures).
    pub fn generate(signatures: &[(Vec<u8>, Vec<u8>, Vec<u8>) /* (pk, msg, sig) */]) -> Result<Self>;

    /// Verify the STARK in pure native Rust. The δ-circuit gadget calls this
    /// via FFI or inline assertion.
    pub fn verify(&self, signatures: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> Result<bool>;
}
```

**Acceptance criteria:**
- [ ] AIR compiles and integrates with `batch_prover.rs`
- [ ] Generate + verify round-trip for K = 1, 10, 100, 500 signatures
- [ ] Adversarial: corrupt one signature byte in the witness → verify returns false
- [ ] Per-signature batch cost: < 200μs prover, < 2ms verifier (sanity bound)
- [ ] No file outside `crates/q-zk-stark/src/dilithium_batch_air.rs` modified except adding `pub mod dilithium_batch_air;` to `lib.rs`

**Branch:** `deepseek/f-dilithium-batch-air`

---

# JOB G — Wire Protocol Scaffold: `GET /api/v1/proof/tip` — START NOW

**Status:** Independent. Can start TODAY. Lands a placeholder endpoint that wallets can integrate against immediately.

**Why this matters:** The browser wallet (`gui/quantum-wallet/`) and the WASM verifier (Job H) both consume `GET /api/v1/proof/tip`. Until Nova lands (Job D), the endpoint returns a fixed-shape placeholder body so the JS-side proof-bootstrap path (in `gui/quantum-wallet/src/ivc/`) can be wired and tested end-to-end. When Job D ships, the endpoint body is updated in one place; the JS API contract is unchanged.

**File:** `crates/q-api-server/src/handlers.rs` (add ~80 LOC). Route entry in `crates/q-api-server/src/main.rs` (1 line).

**Required public JSON schema** (stable across Phase 1/2/4):
```json
{
  "tip_height": 11400000,
  "state_root": "0xabcd1234...",
  "block_header": {
    "height": 11400000,
    "parent_hash": "0x...",
    "tx_root": "0x...",
    "state_root": "0xabcd1234...",
    "timestamp": 1771761600,
    "producer_id": 123
  },
  "proof_version": "placeholder-v0",
  "proof_size_bytes": 32,
  "proof_b64": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
}
```

For Phase 1 (this Job G):
- `proof_version` = `"placeholder-v0"`
- `proof_b64` is a fixed 32-byte all-zero array, base64-encoded
- `state_root` and `block_header` come from real AppState (existing fields: `current_height_atomic`, get the latest block header via `app_state.storage_engine.get_latest_block_header().await`)

For Phase 2 (after Job D):
- `proof_version` = `"nova-bn254-v1"`
- `proof_b64` is the real Nova proof bytes
- Same JSON shape, same fields

**Acceptance criteria:**
- [ ] Endpoint returns HTTP 200 with the schema above
- [ ] `state_root` and `tip_height` reflect live AppState values
- [ ] `proof_version` literal equals `"placeholder-v0"`
- [ ] HTTP latency < 50 ms p99
- [ ] Wallet integration: bootstrap.ts (`gui/quantum-wallet/src/ivc/bootstrap.ts`) can fetch + call WASM placeholder verifier, get back a valid (placebo) tip
- [ ] No file outside `handlers.rs` + `main.rs` (single route line) modified

**Branch:** `deepseek/g-proof-tip-endpoint`

---

# JOB H — WASM Browser Verifier Phase 2

**Status:** Blocked on Job D. Specced in `docs/deepseek-handoff-wasm-browser-verifier-2026-05-13.md`.

**The change is one file:** `crates/q-ivc-verifier-wasm/src/lib.rs`. Replace the `verify_proof_bytes` body's Phase 1 placeholder `return true;` with a real `nova_snark::RecursiveSNARK::verify` call against bundled public parameters.

**Reference:** the Phase 2 swap-in comment block in the current `lib.rs` shows the exact location:
```rust
// PHASE 2 REPLACE: real Nova/BN254 verification.
//
// let recursive_snark: nova_snark::RecursiveSNARK<G1, G2, C1, C2> =
//     bincode::deserialize(proof_bytes).map_err(...)?;
// ...
// PHASE 1 placeholder:
true
```

Update `verifier_version()` to return `"nova-bn254-v1"`. JS API contract unchanged.

**Acceptance criteria:**
- [ ] Loading the WASM in a real browser succeeds (test via `gui/quantum-wallet/public/ivc-demo.html`)
- [ ] `verifier_version()` returns `"nova-bn254-v1"`
- [ ] Round-trip: fetch real Nova proof from `/api/v1/proof/tip`, call `verifyProof()`, get `valid: true`
- [ ] Adversarial: tamper with proof_b64 → `valid: false` with `error: ...`
- [ ] WASM binary size < 10 MB after `wasm-pack build --release` (with all Nova deps; Phase 1 was ~30 KB)
- [ ] Verifier latency on M2 / 8-core x86 desktop ≤ 50 ms in browser
- [ ] No JS/TypeScript file modification (the API stays stable from Phase 1)

**Branch:** `deepseek/h-wasm-verifier-phase2`

---

# JOB I — Mobile Light-Client STARK Path — START NOW

**Status:** Independent. Can start TODAY. Uses existing `crates/q-zk-stark/src/wallet_privacy_stark.rs` (530 LOC, production-deployed) as the template.

**Why this matters:** Even after Nova-WASM Phase 2 lands, very resource-constrained clients (low-end mobile, IoT, sub-2GB-RAM devices) may struggle with 250 ms WASM verification. A STARK proof of "the recent N block headers are valid" is verifiable in **pure JavaScript without WASM** in tens of milliseconds. Weaker guarantee than the full IVC proof (only validates last N blocks, not entire chain) but acceptable for current-state-only wallets.

**Files:**
- `crates/q-zk-stark/src/light_client_stark.rs` (new, ~400 LOC)
- `gui/quantum-wallet/src/ivc/light_client.ts` (new, ~150 LOC) — JS-side verifier in pure-JS BLAKE3 + FRI

**What the AIR proves:**
For the most recent N block headers `H_{tip-N+1}, ..., H_tip`, prove:
- Each header's BLAKE3 hash matches the published block hash
- Each `H_i.parent_hash == hash(H_{i-1})` (chain continuity)
- Each header is signed correctly by the producer for that height

The N is parameterized; typical: N = 100 (last ~100 seconds of chain).

**Public API:**
```rust
pub struct LightClientStarkProof {
    pub from_height: u64,
    pub to_height: u64,
    pub root_hash: [u8; 32],  // BLAKE3 over the N headers
    pub stark_bytes: Vec<u8>,
}

impl LightClientStarkProof {
    pub fn generate(headers: &[BlockHeader]) -> Result<Self>;
    pub fn verify(&self) -> Result<bool>;
    /// JS-friendly: serialize as JSON with hex/base64 fields.
    pub fn to_json(&self) -> serde_json::Value;
}
```

**JS-side verifier (`light_client.ts`):**
```typescript
import { lightClientVerify } from "./light_client";

const proof = await fetch("/api/v1/light-client-stark").then(r => r.json());
const valid = await lightClientVerify(proof);  // pure JS, no WASM
```

The JS verifier should use a pure-JS BLAKE3 (e.g., `blake3` npm package) and a pure-JS FRI verifier. No WASM dependencies. The verifier is what makes mobile / IoT / restricted-environment use cases viable.

**Acceptance criteria:**
- [ ] Generate STARK from N=100 real headers; round-trip verify in Rust
- [ ] JS-side `lightClientVerify` returns true for the same proof in node + browser
- [ ] Verifier latency in pure JS: ≤ 50 ms on M2 / 8-core x86 laptop
- [ ] WASM-free: the JS bundle doesn't add any WASM byte (verify via vite build report)

**Branch:** `deepseek/i-mobile-light-client-stark`

---

# JOB J — Cross-Chain Bridge Attestation (FUTURE)

**Status:** Future work. Specced in whitepaper v2 §7.4.

**Goal:** A STARK proof of "the Quillon Graph state root at height H is R" that can be verified by another chain (Ethereum, Bitcoin via Tapscript, any chain with hash evaluation).

**Implementation hint:** uses `crates/q-zk-stark/src/blockchain_state_circuit.rs` as the underlying AIR. The bridge proof is a specialization of the existing blockchain-state circuit with destination-chain-specific output formatting (Cairo-style verification key for Ethereum, Tapscript leaves for Bitcoin, etc.).

**Branch:** `deepseek/j-cross-chain-bridge-air` (don't start until Beta authorizes — this is destination-chain-specific and needs coordination with bridge-design discussion)

---

# JOB K — Activation-Height Attestations (FUTURE)

**Status:** Future work. Specced in whitepaper v2 §7.5.

**Goal:** A STARK proof of "the state transition at activation height correctly applies the consensus rule activation." Gives validators an independent audit trail; defends against adversarial activations.

**Implementation hint:** also uses `blockchain_state_circuit.rs` but at a single specific block height. Less complex than the bridge case.

**Branch:** `deepseek/k-activation-height-air` (don't start until Beta authorizes — activation-height design is the prerequisite)

---

# COORDINATION RULES (REPEATED FOR EMPHASIS)

## File ownership matrix for this Job Board

| Job | Files in scope | Owner branch |
|---|---|---|
| A | `crates/q-ivc/src/gadgets/blake3.rs` | `deepseek/a-blake3-multiblock-retry` |
| B | `crates/q-ivc/src/gadgets/merkle.rs` (new) | `deepseek/b-merkle-gadget` |
| C | `crates/q-ivc/src/circuits/delta_block.rs` (new) | `deepseek/c-delta-circuit` |
| D | `crates/q-ivc/src/recursion/` (new module) | `deepseek/d-nova-wrapper` |
| E | `crates/q-zk-stark/src/nova_srs_generator_air.rs` (new) + `crates/q-zk-stark/src/lib.rs` (1 line) | `deepseek/e-nova-srs-air` |
| F | `crates/q-zk-stark/src/dilithium_batch_air.rs` (new) + `crates/q-zk-stark/src/lib.rs` (1 line) | `deepseek/f-dilithium-batch-air` |
| G | `crates/q-api-server/src/handlers.rs` + `crates/q-api-server/src/main.rs` (1 line) | `deepseek/g-proof-tip-endpoint` |
| H | `crates/q-ivc-verifier-wasm/src/lib.rs` (one file, one function body) | `deepseek/h-wasm-verifier-phase2` |
| I | `crates/q-zk-stark/src/light_client_stark.rs` (new) + `gui/quantum-wallet/src/ivc/light_client.ts` (new) + `crates/q-zk-stark/src/lib.rs` (1 line) | `deepseek/i-mobile-light-client-stark` |
| J | TBD — coordination required | `deepseek/j-cross-chain-bridge-air` |
| K | TBD — coordination required | `deepseek/k-activation-height-air` |

**Beta owns:** `crates/q-storage/src/balance_smt.rs`, `crates/q-storage/src/lib.rs`, `crates/q-types/src/block.rs`, root `Cargo.toml`, `crates/q-zk-stark/src/{air,stark_prover,stark_verifier,batch_prover,polynomials,blockchain_state_circuit,wallet_privacy_stark}.rs`, `crates/q-storage/src/encryption_zkstark.rs`. **Do not touch.**

**Claude Code agents own:** `crates/q-tui/`, `crates/q-api-server/src/main.rs` (route wiring), the TUI Blueprint 7 work, network instrumentation, dev-tooling scripts. Coordinate via `#dev-coordination` before overlapping.

## Daily mechanics

1. Morning standup in `#dev-coordination`: post (a) shipped yesterday, (b) working on today, (c) blockers.
2. Push to `code.quillon.xyz` on the branch listed above. Run `git update-server-info` after push.
3. Tag Beta for review.
4. Friday integration: Beta merges `ivc/v1` and runs the full test suite on Epsilon Docker.
5. Mainnet deploy: only after `safe-deploy.sh` runs 4000+ mainnet-safety tests clean.

## What gets your PR rejected immediately

1. Touching `balance_root_v1` computation.
2. Modifying `save_wallet_balance` / `save_wallet_balances` without Beta sign-off.
3. `unsafe`, `unwrap()` outside `#[cfg(test)]`, or `panic!()` outside test code.
4. Submitting code with `// TODO implement here` or `// see spec for details` placeholder bodies in non-test functions.
5. Pushing to GitHub.
6. Modifying a file outside your Job's file-ownership scope.
7. Inventing citations or APIs that don't exist (verify everything via `grep` before claiming).

If any of these happen, the PR is rejected without negotiation. Repeated violations move you off the SNARK track.

---

# WHAT'S ALREADY SHIPPED (FOR REFERENCE — DO NOT REIMPLEMENT)

Verify each of these exists with `ls` and `wc -l` before doing anything in the related job:

| Component | Path | LOC | Status |
|---|---|---|---|
| BalanceSmt (SMT for balance_root_v2) | `crates/q-storage/src/balance_smt.rs` | ~743 | Shipped, shadow mode |
| BLAKE3 single-block gadget | `crates/q-ivc/src/gadgets/blake3.rs` | 551 | Shipped. Single-block only. Job A extends. |
| Poseidon gadget | `crates/q-ivc/src/gadgets/poseidon.rs` | 329 | Shipped |
| NTT butterfly | `crates/q-ivc/src/gadgets/ntt.rs` | 769 | Shipped |
| Dilithium5 primitives | `crates/q-ivc/src/gadgets/dilithium.rs` | 902 | Shipped (3 test fixtures still need fix, not blocking) |
| AIR framework | `crates/q-zk-stark/src/air.rs` | 429 | Shipped |
| STARK prover | `crates/q-zk-stark/src/stark_prover.rs` | 442 | Shipped |
| STARK verifier | `crates/q-zk-stark/src/stark_verifier.rs` | 548 | Shipped |
| Batch STARK prover | `crates/q-zk-stark/src/batch_prover.rs` | 662 | Shipped. Job F uses this. |
| Polynomial/FRI | `crates/q-zk-stark/src/polynomials.rs` | 549 | Shipped |
| Blockchain state circuit (AIR) | `crates/q-zk-stark/src/blockchain_state_circuit.rs` | 410 | Shipped. Jobs J/K use this. |
| Wallet privacy STARK | `crates/q-zk-stark/src/wallet_privacy_stark.rs` | 530 | **Production-deployed.** Job I template. |
| Encryption SRS attestation | `crates/q-storage/src/encryption_zkstark.rs` | 562 | **Production**, artifact at `data-mainnet-genesis/encryption.zkstark` (176,832 bytes). Job E follows the same pattern. |
| WASM verifier scaffold | `crates/q-ivc-verifier-wasm/src/lib.rs` | ~150 | Shipped (Phase 1 placeholder). Job H upgrades. |
| Archive status endpoint | `crates/q-api-server/src/handlers.rs::archive_status` | ~80 | Shipped v10.9.18 |
| Engine pulse endpoint | `crates/q-api-server/src/handlers.rs::engine_pulse` | ~120 | Shipping v10.9.19 |
| TUI readiness banner | `crates/q-tui/src/ui/dashboard.rs::draw_readiness_banner` | ~80 | Shipping v10.9.19 |

---

# RECOMMENDED PARALLELISM FOR THE NEXT 2 WEEKS

If you have multiple developers / can spawn multiple instances:

**Day 1-7:**
- Developer X: Job A (multi-block BLAKE3 retry)
- Developer Y: Job E (Nova-SRS AIR)
- Developer Z: Job G (proof-tip endpoint scaffold) — **2 days max, then move to Job I**

**Day 8-14:**
- Developer X: Job B (Merkle gadget, once Job A lands)
- Developer Y: Job F (Dilithium-batch AIR)
- Developer Z: Job I (mobile light-client STARK)

**Week 3+:**
- Job C (δ-circuit) blocks on Job B
- Job D (Nova wrapper) blocks on Job C
- Job H (WASM Phase 2) blocks on Job D

Beta engineer (in parallel): SMT shadow-mode wiring + activation-height definition + soak coordination.

---

# REPORTING BACK

After each Job's PR is opened:
1. PR description includes the acceptance-criteria checklist with each box ticked
2. `cargo test --package <crate>` output included in PR
3. Constraint count / verification latency / proof size numbers reported where applicable
4. Specific files touched, line counts, no scope creep
5. Beta reviews + tags `ivc/v1` for integration

Beta will run the integration test suite weekly on Epsilon Docker. Anything that breaks the suite gets reverted from `ivc/v1` until fixed.

— Quillon Graph maintainers, 2026-05-13
