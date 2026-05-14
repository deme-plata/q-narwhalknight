# DeepSeek + ChatGPT Job Board — Nova Wrapper, Phase 2

**Date:** 2026-05-14
**Project:** Quillon Graph — live mainnet, ~$2 B USD market cap
**Track:** Recursive SNARK Phase 2 — Nova folding wrapper around the δ-circuit
**Companion docs:**
- `papers/quillon-recursive-lattice-snark-whitepaper-v2-2026-05-13.pdf` (architecture)
- `docs/deepseek-job-board-recursive-snark-zkstark-2026-05-13.md` (the broader track; Jobs A–K still apply)
- `crates/q-zk-stark/src/nova_srs_generator_air.rs` (Job E — SRS attestation, Phase 1 shipped)

**Purpose.** A focused, self-contained job board for the Nova folding wrapper. Phase 2 is the recursion layer that turns the δ-circuit (one R1CS proof per block) into a single constant-size proof over the entire chain. Without it the 10-millisecond bootstrap promise from the whitepaper does not exist. Jobs are scoped so that ChatGPT or DeepSeek can take one in parallel without stepping on the others.

---

# 🚨 MAINNET CONSTRAINTS — UNCHANGED

Same rules as prior handoffs. Re-read if unsure:

1. **No DB schema changes.** No new column families. No modifications to existing CFs.
2. **No `balance_root_v1` modification anywhere.** The Phase 2 work uses `balance_root_v2` (sparse Merkle tree, already shipped in `crates/q-storage/src/balance_smt.rs`).
3. **No `save_wallet_balance` / `save_wallet_balances` modification without explicit Beta sign-off.**
4. **No `unsafe`, no `unwrap()` in non-test code.** `expect()` is acceptable only with a message explaining why the panic is unreachable.
5. **Use `code.quillon.xyz` — NOT GitHub.** Branch name: `deepseek/n<job-number>-<short-description>`. Run `git update-server-info` after push so Epsilon and Beta can pull.
6. **One job at a time.** File-ownership matrix below is per-job; don't grab a job already in flight.
7. **Push to integration branch `ivc/nova-phase2`** after a Beta reviewer signs off.
8. **Soak before activation.** Code lands → at least one week on the test rig with synthetic block load → mainnet flag `NOVA_PHASE2_ACTIVATION_HEIGHT` set in the upgrade gate (similar pattern to `BALANCE_ROOT_V2_HEIGHT`).
9. **Phase 2 proofs are ADVISORY** in the activation window. Blocks are still validated block-by-block by every node. The Nova proof is published alongside the chain tip but failure of recursive verification does not roll back the chain. This is the same safety pattern used for `BalanceRootV1` shadow mode.

If any of these are unclear, **ask in #dev-coordination before coding**. Cost of a question = zero. Cost of a misplaced edit on a $2 B chain = enormous.

---

# WHAT EXISTS TODAY (read before starting)

## Cryptographic gadgets — in production, `crates/q-ivc/src/gadgets/`

| Gadget | File | Status | LOC |
|--------|------|--------|-----|
| BLAKE3 single-block | `blake3.rs` (`verify_hash`) | Shipped | ~600 |
| BLAKE3 multi-block | `blake3.rs` (`hash_message`) | **NOT IMPLEMENTED — Job A in companion board** | — |
| Poseidon transcript | `poseidon.rs` | Shipped | ~400 |
| NTT butterfly + signed-norm | `ntt.rs` | Shipped, signed-norm hardened today | ~700 |
| Dilithium5 (`verify_structured`) | `dilithium.rs` | Shipped today with hint-weight + q-range + μ-prefix gates | ~1500 |
| Merkle-path gadget | `merkle.rs` | **NOT IMPLEMENTED — Job B in companion board** | — |

## STARK infrastructure — in production, `crates/q-zk-stark/`

- 4,469 LOC of Rust
- AIR framework, prover, verifier, batch prover, FRI commitments
- `nova_srs_generator_air.rs` — **Job E, shipped today**. Generates a deterministic SRS chain and proves correctness with a transparent STARK. This is the transparent-setup mechanism described in §4 of the whitepaper. Phase 1 uses simplified field arithmetic over a Mersenne prime; Phase 2 must lift this to real BN254 group exponentiation. **N4 below.**

## State commitment — in production, `crates/q-storage/src/balance_smt.rs`

- `BalanceSmt` — sparse Merkle tree of depth 256, BLAKE3 leaf/node hashing with explicit domain separators
- 743 LOC, twelve test cases
- Shadow mode active on mainnet since v10.9.x: every block updates both `balance_root_v1` (legacy flat hash) and the SMT, RocksDB write batch asserts they don't diverge

## Upgrade gate — in production, `crates/q-consensus-guard/src/upgrade_gate.rs`

- Use `Upgrade::HybridSignaturesV1` (added v10.9.20) as the template for adding `Upgrade::NovaPhase2`
- Mainnet activation height: `u64::MAX` (dormant) until you set it
- Testnet activation: 0 (active immediately for canary nodes)

## What does NOT exist yet

- **The δ-circuit composition** itself (Job C in companion board). The gadgets exist; the single R1CS predicate that composes them does not.
- **Any Nova folding driver.** No `crates/q-ivc/src/recursion/` directory yet. **You're creating it.**
- **The wire protocol.** `GET /api/v1/proof/tip` returns a placeholder.

---

# JOB DEPENDENCY GRAPH — Phase 2

```
   N1 (Nova crate selection + spike)
            │
            ▼
   N2 (δ-circuit as StepCircuit) ──┐
            │                       │
            ▼                       │
   N3 (RecursiveSNARK driver) ──────┼──── N7 (Compression SNARK)
            │                       │              │
            ▼                       │              ▼
   N4 (SRS generation +             │     N8 (Benchmarks)
       BN254 lift of Job E)         │
            │                       │
            ├───────────────────────┴───── N5 (Wire protocol /api/v1/proof/tip)
            │
            └───────── N6 (Client bootstrap — Algorithm 1)
```

**Critical path:** N1 → N2 → N3 → N4 → N7. Each of these blocks the next.
**Parallelizable from day one:** N5 (wire protocol scaffold), N6 (client algorithm against a mock proof).
**N8 needs N3 + N4** before it can produce meaningful numbers.

---

# JOB N1 — Nova crate selection and integration spike

**Branch:** `deepseek/n1-nova-spike`
**Files you own:**
- `crates/q-ivc/Cargo.toml` (add dep)
- `crates/q-ivc/src/recursion/mod.rs` (new — module root)
- `crates/q-ivc/src/recursion/spike.rs` (new — Fibonacci hello-world)
- `crates/q-ivc/src/lib.rs` (add `pub mod recursion;`)
**Files you must NOT modify:** anything outside the above list.

**Task.** Evaluate the two production-maintained Nova implementations and pick one. Add it as a workspace dependency. Implement a trivial `StepCircuit` (Fibonacci: `(a, b) → (b, a+b)`) and run a `RecursiveSNARK` for 100 steps. Verify. Commit benchmark numbers in the test output.

**The choice (you decide, with rationale):**

| Crate | Curve cycle | Maturity | Notes |
|-------|-------------|----------|-------|
| `nova-snark` (microsoft/Nova) | Pasta (Pallas/Vesta) or BN256/Grumpkin | Production-mature, used by Lurk, EZKL | Original Nova implementation. **Likely choice.** |
| `arkworks-nova` (community fork) | Configurable via ark-ec | Less mature | Better fit if you already use arkworks elsewhere (we do) |
| Custom | — | — | **Out of scope.** Don't even consider. |

The whitepaper says BN254. Microsoft's nova-snark supports a BN256/Grumpkin cycle which is what you want. Pasta works for Phase 2 too but doesn't match the paper's curve choice — if you pick Pasta, you must justify in the PR description why the curve mismatch is OK.

**Acceptance criteria** (copy into PR description):
- [ ] One Nova crate is in `[workspace.dependencies]` in the root `Cargo.toml` with an exact version pin (no `*`, no caret-only)
- [ ] `crates/q-ivc/src/recursion/spike.rs` contains a Fibonacci `StepCircuit` impl and a test `test_nova_fibonacci_100_steps` that:
  - Creates the public parameters
  - Folds 100 steps of Fibonacci starting from `(0, 1)`
  - Verifies the resulting `RecursiveSNARK`
  - Asserts the output is `Fibonacci(100)` (use a u128 to avoid overflow)
- [ ] Test passes: `cargo test --package q-ivc --lib recursion::spike 2>&1 | tail -10` shows `1 passed`
- [ ] Test output includes timings: setup time, per-step prove time, total prove time, verify time
- [ ] PR description includes which crate you chose, why, and the four timing numbers from the test run
- [ ] No file outside the four files listed above is modified

**Expected timing (on commodity hardware, for sanity):**
- Setup: 1-5 sec
- Per-step prove: 100-500 ms (Fibonacci is trivial; real δ-circuit will be slower)
- 100-step total prove: 10-50 sec
- Verify: 5-50 ms

If your numbers are an order of magnitude off, something is wrong. Stop and investigate.

---

# JOB N2 — δ-circuit as StepCircuit

**Status:** Blocked on N1 (need the Nova trait imports). Can start spec work immediately.
**Branch:** `deepseek/n2-delta-as-step-circuit`
**Files you own:**
- `crates/q-ivc/src/circuits/delta.rs` (new — the δ-circuit composition)
- `crates/q-ivc/src/circuits/mod.rs` (add `pub mod delta;`)
- `crates/q-ivc/Cargo.toml` (no new deps — reuse the gadget library)
**Files you must NOT modify:** any file under `crates/q-ivc/src/gadgets/`. If a gadget is missing or broken, raise it in #dev-coordination; do not patch in-line.

**Task.** Implement the δ-circuit defined in §3.2 of the whitepaper as a Nova `StepCircuit`. The step is "apply one block to the previous state root and produce the next state root, enforcing every consensus rule along the way."

**Inputs / outputs of the step:**

Public inputs (`z`):
- `state_root_prev: [u8; 32]` (32 bytes packed into 1 BN256 field element via low-128/high-128 split; or 4 elements of 64 bits — your call, document it)
- `block_height: u64` (1 field element)
- `header_hash: [u8; 32]` (same packing as state root)

Public output (`z_next`):
- `state_root_next: [u8; 32]`
- `block_height + 1: u64`
- `next_header_hash: [u8; 32]` — this is `header_hash` of `B_{n+2}` if available, else zero; allows chaining headers across folds

Private witnesses (provided by the producer):
- The full block body `B_{n+1}` (transactions, coinbase, timestamp, signatures, NTT anchor proof)
- For every wallet touched in the block: Merkle path in `state_root_prev` from the SMT, intermediate state root after the from-side update, Merkle path in the intermediate root
- For every transaction: Dilithium5 public key, signature `(z, h, c̃, c_poly)`, message hash
- The NTT anchor election witness

**Constraints to enforce (from whitepaper §3.2 step list):**

1. **Header hash:** `header_hash = BLAKE3(header_bytes)` via `Blake3Gadget::verify_hash` (or `hash_message` if multi-block — Job A). Block header is < 1 KB so single-block path is borderline; assume `hash_message` for safety.
2. **For every transaction:**
   - `Verify_Dilithium5(pk, msg_tx, sig)` via `DilithiumVerifierGadget::verify_structured`
   - Membership of `(from, b_from)` in `state_root_prev` via `MerklePathGadget::enforce_membership`
   - Membership of `(to, b_to)` in `state_root_prev` via the same gadget
   - Balance sufficiency: `b_from >= a + f` (range check on `b_from - a - f >= 0` via `enforce_norm_bound` with bound `u128::MAX`)
   - Updated `(from, b_from - a - f)` membership in the intermediate root
   - Updated `(to, b_to + a)` membership in the root that follows the from-side update
3. **Coinbase emission:** `e <= R(height)` where R is the four-year halving schedule. Implement R as a piecewise-constant lookup over height boundaries (`HALVING_BLOCK_INTERVAL` constants from `crates/q-types/src/emission.rs`).
4. **NTT anchor election:** validate the block producer's claim of being the anchor via `NttVerifierGadget`.
5. **Final state-root equality:** after sequentially applying all transactions and the coinbase, the resulting root must equal the public `state_root_next`.

**Acceptance criteria:**

- [ ] `crates/q-ivc/src/circuits/delta.rs` defines `pub struct DeltaCircuit { /* witness fields */ }` and `impl StepCircuit<F> for DeltaCircuit`
- [ ] `synthesize` body enforces all five constraint groups above using existing gadgets — **no inline cryptography**, no copy-paste of gadget bodies
- [ ] Test `test_delta_one_transaction_satisfiable` builds a one-transaction block, generates witnesses, calls `cs.is_satisfied()`, asserts true
- [ ] Test `test_delta_invalid_signature_rejected` flips one bit in a transaction signature, asserts `cs.is_satisfied()` is false
- [ ] Test `test_delta_double_spend_rejected` constructs two transactions from the same sender that together exceed the sender's balance, asserts unsatisfied
- [ ] Test `test_delta_wrong_state_root_rejected` modifies the public `state_root_next` from its correct value, asserts unsatisfied
- [ ] Constraint count is reported in test output. Expected order of magnitude per block: 50-100M (paper says ~440M at full Dilithium5 scale with many txs; one-transaction block is much smaller)
- [ ] PR description lists constraint counts for the four tests

**Critical: do not invent new gadgets.** If you need a primitive that doesn't exist, raise it in #dev-coordination and let Beta either add it or split it off as Job N2b. The cost of "make a placeholder Merkle gadget" is that the δ-circuit becomes unsound and gets shipped to mainnet sounding fine. **The dependency graph protects this property — respect it.**

---

# JOB N3 — RecursiveSNARK driver

**Status:** Blocked on N1 + N2.
**Branch:** `deepseek/n3-recursive-snark-driver`
**Files you own:**
- `crates/q-ivc/src/recursion/driver.rs` (new — the folding loop)
- `crates/q-ivc/src/recursion/mod.rs` (add `pub mod driver;`)
**Files you must NOT modify:** anything in `crates/q-ivc/src/circuits/` or `crates/q-ivc/src/gadgets/`.

**Task.** Wrap Nova's `RecursiveSNARK::{new, prove_step, verify}` in an API the rest of the codebase can use without knowing about Nova internals.

**Public API:**

```rust
pub struct NovaFolder {
    pp: PublicParams,  // produced once at genesis or boot
    state: RecursiveSNARK,
}

impl NovaFolder {
    /// Initialize at genesis with the initial state root.
    pub fn new(pp: PublicParams, initial_state_root: [u8; 32]) -> Result<Self>;

    /// Fold one block into the running proof. Called per block by the producer.
    pub fn fold_block(
        &mut self,
        block: &Block,
        witnesses: DeltaWitness,
    ) -> Result<()>;

    /// Serialize the current proof for wire transport. Postcard binary.
    pub fn serialize_proof(&self) -> Result<Vec<u8>>;

    /// Deserialize and verify a proof received from a peer. Returns the
    /// asserted state root and block height if valid.
    pub fn verify_received(
        pp: &PublicParams,
        proof_bytes: &[u8],
    ) -> Result<(StateRoot, BlockHeight)>;
}
```

**Witness assembly is YOUR responsibility.** N2 defines what `DeltaWitness` needs to contain; the driver is the place that walks the block, queries the storage layer for Merkle paths, and assembles the witness struct. Use the existing `BalanceSmt::prove` to generate Merkle proofs (synchronous, single-allocation). Use `q_types::transaction::Transaction` as the block's tx representation.

**Acceptance criteria:**

- [ ] `NovaFolder::new` works in under 5 seconds for a fresh genesis
- [ ] `fold_block` takes a real `Block` (use the one-tx block from N2's test fixture) and produces a valid proof — verified inline with the existing Nova `verify`
- [ ] Folding 10 blocks in sequence works; serialized proof size grows by less than 5% across the 10 blocks (constant-ish)
- [ ] `verify_received` round-trips: take a proof from `serialize_proof`, pass it through `verify_received`, get back the expected `(state_root, block_height)`
- [ ] Bad proof bytes (random / truncated / tampered) cause `verify_received` to return `Err`, not panic
- [ ] All five paths covered by tests:
  - `test_genesis_then_fold_one_block`
  - `test_fold_ten_blocks_proof_size_constant`
  - `test_verify_received_round_trip`
  - `test_verify_received_rejects_tampered_proof`
  - `test_verify_received_rejects_wrong_pp` (different public params should not verify)

---

# JOB N4 — SRS generation and BN254 lift of Job E

**Status:** Blocked on N1.
**Branch:** `deepseek/n4-srs-bn254`
**Files you own:**
- `crates/q-zk-stark/src/nova_srs_generator_bn254.rs` (new — Phase 2 SRS gen)
- `crates/q-zk-stark/src/lib.rs` (add `pub mod nova_srs_generator_bn254;`)
- `crates/q-ivc/src/recursion/srs.rs` (new — Nova-side adapter)
**Files you must NOT modify:** `crates/q-zk-stark/src/nova_srs_generator_air.rs` (Phase 1 reference — leave intact for backwards compatibility).

**Task.** The Phase 1 `nova_srs_generator_air.rs` does the SRS-generation math over a 61-bit Mersenne prime field. Phase 2 needs the same shape of attestation but over the BN254 curve where group exponentiations actually live. The δ-circuit's SRS is a sequence of BN254 group elements `[G, αG, α²G, ..., α^{n-1}G]` (and corresponding G2 elements for the pairing side).

The transparent-setup story is:
1. A future block hash `s` becomes public after the SRS-generation transaction is mined (RANDAO-style — see Section 4.2 of the whitepaper).
2. The deterministic algorithm `gen_srs(s)` is BN254 multi-scalar-multiplication: derive `α = HashToField(s)`, then compute the chain via repeated point addition / doubling.
3. A zk-STARK attests that the SRS was produced by running `gen_srs` on `s` correctly. Use the same `q-zk-stark` AIR framework as Phase 1, but the trace is now sequences of BN254 affine coordinates.

**Acceptance criteria:**

- [ ] `nova_srs_generator_bn254.rs` exposes `pub struct BN254SrsProof { seed, srs_size, generator: BN254Affine, final_power: BN254Affine, srs_root: [u8; 32], stark_bytes: Vec<u8> }`
- [ ] `BN254SrsProof::generate(seed: &[u8; 32], srs_size: usize) -> Result<(Self, Vec<BN254Affine>)>` — returns the proof AND the actual SRS points
- [ ] `BN254SrsProof::verify(&self) -> Result<bool>` — re-derives α, recomputes chain root, verifies STARK
- [ ] Test `test_bn254_srs_generate_and_verify_small` for `srs_size = 8`
- [ ] Test `test_bn254_srs_generate_and_verify_realistic` for `srs_size = 2^20` (matches the δ-circuit's constraint count target)
- [ ] Tampering tests for each field of `BN254SrsProof` (seed, generator, final_power, srs_root, stark_bytes) — five rejection tests minimum
- [ ] `crates/q-ivc/src/recursion/srs.rs` provides `pub fn nova_pp_from_srs(srs: &[BN254Affine]) -> PublicParams` that converts the BN254 points into the Nova crate's expected `PublicParams` shape

**Performance target:** SRS generation for `srs_size = 2^20` must complete in < 10 minutes on commodity hardware. STARK proof size < 200 KB. STARK verification time < 100 ms.

---

# JOB N5 — Wire protocol: `/api/v1/proof/tip`

**Status:** Parallel to N1-N4. Can start TODAY against a mock proof.
**Branch:** `deepseek/n5-proof-tip-endpoint`
**Files you own:**
- `crates/q-api-server/src/handlers.rs` (extend the existing `proof_tip` placeholder)
- `crates/q-api-server/src/main.rs` (route wiring — minimal)
- `crates/q-types/src/proof_wire.rs` (new — wire format)
**Files you must NOT modify:** anything in `crates/q-ivc/` or `crates/q-zk-stark/`.

**Task.** Replace the v10.9.19 placeholder `proof_tip` handler with one that serves a real Nova proof when one is available, and returns a structured "not yet" response when not. The handler should not block on proof generation — if no proof is cached, respond with `HTTP 503 + Retry-After + reason`.

**Wire format** (postcard binary, content-type `application/octet-stream`):

```rust
pub struct ProofTipResponse {
    /// 1 byte: proof_version. `0x00` = placeholder, `0x01` = Nova Phase 2.
    pub version: u8,
    /// Current chain tip — block height the proof corresponds to.
    pub height: u64,
    /// State root the proof attests to.
    pub state_root: [u8; 32],
    /// Block header hash at the tip.
    pub header_hash: [u8; 32],
    /// Serialized Nova proof. Empty when version == 0x00.
    pub proof_bytes: Vec<u8>,
    /// Network ID for cross-network safety (`mainnet2026.1` etc).
    pub network_id: String,
}
```

**HTTP response headers** (parsed by the bootstrap client — see N6):
- `X-QNK-Proof-Version: 1` (or 0)
- `X-QNK-Proof-Height: 17742000`
- `X-QNK-Proof-Size: 5234` (bytes; lets the client decide whether to download)
- `Content-Type: application/octet-stream`

**Acceptance criteria:**

- [ ] `GET /api/v1/proof/tip` returns 200 + a valid `ProofTipResponse` (postcard binary) when a proof is cached
- [ ] Returns 503 + `Retry-After: 60` + JSON body `{"reason": "no proof available yet"}` when none is cached
- [ ] Returns 410 + reason when the proof is too stale (height < tip - 1000) — the client should fall back to legacy sync in that case
- [ ] All five headers above are set correctly
- [ ] Test using `axum::TestClient` covers all three response paths (200, 503, 410)
- [ ] No new auth required — this is a public read endpoint, like `/api/v1/status`

---

# JOB N6 — Client bootstrap (Algorithm 1)

**Status:** Parallel to N1-N5. Can start against the wire format from N5.
**Branch:** `deepseek/n6-bootstrap-client`
**Files you own:**
- `crates/q-api-server/src/main.rs` (the `--bootstrap-from-proof` CLI flag and its handler)
- `crates/q-api-server/src/bootstrap.rs` (new — Algorithm 1 implementation)
**Files you must NOT modify:** `crates/q-storage/`, `crates/q-network/`, `crates/q-types/` (use what's there).

**Task.** Implement Algorithm 1 from the whitepaper:

```
Require: Peer URL p, expected genesis state root s_0
1: (s_t, π_t, H_t) ← HTTP_GET(p, /api/v1/proof/tip)
2: valid ← VerifyRecursive(π_t, s_t, H_t.height)
3: if ¬valid then retry with a different peer
4: assert H_t.state_root = s_t
5: accept s_t as canonical; enable mining and transaction acceptance
6: begin archive backfill in background (not on critical path)
```

**CLI:**
```bash
./q-api-server --bootstrap-from-proof <bootstrap-peer-url> --port 8080
```

**Acceptance criteria:**

- [ ] Algorithm 1 implemented in `bootstrap.rs::bootstrap_from_proof`
- [ ] Verify call uses `NovaFolder::verify_received` from N3
- [ ] Retry with three different peers on failure before giving up
- [ ] On success: write `s_t` to RocksDB as the accepted state root, set the safe-floor pointer, kick off the archive backfill task in the background
- [ ] On total failure (all three peers refused/served invalid proofs): fall back to legacy sync from genesis. Log loudly.
- [ ] CLI flag `--bootstrap-from-proof <url>` accepted; integration test exercises it against a mock proof server
- [ ] Time from `--bootstrap-from-proof` invocation to "mining-ready" log line: target < 1 second on commodity hardware

**Failure modes to handle** (the part where placeholder code usually goes wrong):
- Peer returns 503: try next peer immediately
- Peer returns 200 but proof verification fails: ban this peer for 1 hour, try next
- All peers return 503: wait 30 sec, retry the first peer; after three rounds give up and fall back to legacy sync
- Network timeout (peer unreachable): treat as 503

---

# JOB N7 — Compression SNARK

**Status:** Blocked on N3. The "decompression" step Nova requires before mainstream verification.
**Branch:** `deepseek/n7-compression-snark`
**Files you own:**
- `crates/q-ivc/src/recursion/compression.rs` (new)
**Files you must NOT modify:** anything outside `crates/q-ivc/src/recursion/`.

**Task.** Nova's `RecursiveSNARK` is a *relaxed* R1CS instance. To get a fixed-size SNARK that any verifier can check in milliseconds, the relaxed instance must be "decompressed" through a Spartan-style proof. The Nova crate exposes `CompressedSNARK` for this; wire it in.

**Public API:**

```rust
pub struct CompressedProof {
    /// Same shape as RecursiveSNARK serialized output, but ~5x smaller.
    pub bytes: Vec<u8>,
}

impl CompressedProof {
    /// Produced once per chain tip, expensive (~30 sec).
    pub fn from_recursive(folder: &NovaFolder) -> Result<Self>;

    /// Verifies in 5-10 ms.
    pub fn verify(&self, pp: &PublicParams) -> Result<(StateRoot, BlockHeight)>;
}
```

**Acceptance criteria:**

- [ ] `CompressedProof` round-trips against `NovaFolder` from N3
- [ ] Compression takes < 60 seconds on commodity hardware
- [ ] Verification takes < 50 ms (Phase 2 target was 5-10 ms; 50 ms is acceptable for first integration, we'll optimize later)
- [ ] Compressed proof size is < 100 KB
- [ ] Tampered compressed proof is rejected (5 tampering tests, one per field of the proof structure)

---

# JOB N8 — Benchmarks

**Status:** Blocked on N3 + N4 (need real δ-circuit SRS to measure realistic numbers).
**Branch:** `deepseek/n8-bench`
**Files you own:**
- `crates/q-ivc/benches/nova_phase2.rs` (new — criterion benchmark)
- `Cargo.toml` workspace bench entries (minimal)
**Files you must NOT modify:** any source file (this is pure measurement).

**Task.** Produce the numbers the whitepaper §5 promises. Use `criterion` for reproducible measurement.

**Benchmark scenarios:**

1. `nova_fold_one_block` — single fold operation on a realistic block (100 transactions, 256-wallet SMT)
2. `nova_fold_10_blocks` — 10 sequential folds, report cumulative time and proof size
3. `nova_verify_relaxed` — `RecursiveSNARK::verify` time on a 1000-block-old proof
4. `nova_compress_proof` — `CompressedSNARK::prove` time
5. `nova_verify_compressed` — `CompressedSNARK::verify` time (the user-facing 10 ms number)

**Acceptance criteria:**

- [ ] All five benchmarks compile and run with `cargo bench --package q-ivc`
- [ ] Benchmark report file (`benches/nova_phase2_results.md`) committed alongside the code with:
  - Hardware spec the numbers came from (CPU model, RAM, OS)
  - Cargo profile (release with `lto=true` ideally)
  - Median + p95 for each benchmark
- [ ] Numbers are within an order of magnitude of whitepaper §5 targets. If not, file an issue and document where the gap is.

**Whitepaper §5 targets** (for cross-reference):
- Per-block prove: not specified (it's prover infrastructure, runs once per block)
- Verify relaxed: not specified
- Compress: not specified
- Verify compressed: 5-10 ms ← this is the load-bearing number for §1 of the paper

---

# CODING RULES — Apply to ALL JOBS

1. **Read the existing code before writing new code.** Many gadgets and helpers already exist. Reuse them. If a primitive is missing, raise it before implementing — duplicating logic creates surface area for bugs.

2. **No placeholders or stubs that "look right."** If you can't implement something correctly, leave the function `unimplemented!("Job X — blocked on Y")` with a clear blocker note. A panic is a hundred times better than a silent placeholder that returns `Ok(true)` for everything.

3. **No `unwrap()`** in non-test code. Use `?` or pattern-match. `expect("...")` is allowed only with a message that explains why the panic is unreachable.

4. **Real cryptography only.** No `Ok(true)` returns from verify functions. No `vec![0u8; 64]` as a "signature." If a primitive needs a real implementation, write it or block.

5. **Tests exercise the failure path, not just the happy path.** Every verify test must have at least one tampering test that asserts rejection. Soundness regression catches are how you prevent quiet breakage.

6. **Constraint counts in test output.** `println!("constraints: {}", cs.num_constraints());` after every meaningful gadget call. We track these across PRs.

7. **No file outside your job's owned-files list is modified.** Period. If you need a shared helper that's missing, raise it as a new sub-job; don't patch sideways.

8. **PR description includes:**
   - Branch name (matches naming convention)
   - Files modified (full list)
   - Test results (`cargo test --package <pkg> --lib <prefix> 2>&1 | tail -20` output)
   - Performance numbers if benchmarks are part of the job
   - Anything you couldn't complete and why

9. **Push to `code.quillon.xyz`. NEVER to GitHub.** Then `git update-server-info`. Then announce in #dev-coordination so Beta can review.

---

# WHAT BETA WILL CHECK ON REVIEW

For every PR landing in `ivc/nova-phase2`:

- [ ] No `unwrap()` outside `#[cfg(test)]` modules
- [ ] No `unsafe` blocks
- [ ] No new direct `pqcrypto-dilithium` calls in production code (use `q-types::pqc_keys::ValidatorKeypair` for any signing; use `DilithiumVerifierGadget` for in-circuit verification)
- [ ] No new RocksDB column families
- [ ] No `save_wallet_balance` / `save_wallet_balances` calls outside `q-storage`
- [ ] Cargo.toml dependency additions are exact-pinned (no `*`, no caret-only)
- [ ] Constraint counts in test output match what the PR description claims
- [ ] If the PR claims a verification time < 10 ms, the benchmark proves it

---

# COMMUNICATION PROTOCOL

- **Coordination channel:** `#dev-coordination` on Discord
- **Status:** Update the Phase 2 board (`docs/phase2-status.md` — create if doesn't exist) with `[in-progress / blocked / ready-for-review]` for each job you take
- **Block on missing primitives early.** If Job N2 needs `MerklePathGadget::enforce_membership` and it doesn't exist (companion-board Job B is not done), say so on day one — don't wait until day five to mention it
- **Push partial work.** Branches don't have to be PR-ready to be visible. A "WIP, blocked on X" push to `deepseek/n2-delta-as-step-circuit` is better than a missing branch
- **One-page summary at the end of each job.** What you did, what you didn't, what surprised you, what you'd warn the next person about. Saved to `docs/nova-phase2-postmortems/n<job-number>.md`

---

**This board is intentionally over-specified.** Phase 2 is the load-bearing piece of the whitepaper's 10-millisecond promise. The δ-circuit can be re-derived from gadgets; the Nova wrapper is the part where a small mistake silently breaks the bootstrap guarantee for every node that ever runs the binary. Be paranoid, test failure paths, and ask before guessing.

If you're a human reading this for the first time, the order to tackle items is **N1 → (N5 + N6 in parallel) → N2 → N4 → N3 → N7 → N8**. Critical path is N1-N2-N3 to first proof; N4 to make it trustless; N7 to make it fast to verify.

Good luck.
