# IVC SNARK Blueprints — Engineering Specs for Outside Implementation

**Date:** 2026-05-13
**Companion to:** `technical-plan-instant-bootstrap-recursive-snark-v2-2026-05-13.md`
**Audience:** External implementer (e.g. DeepSeek) given the Q-NarwhalKnight codebase
**Purpose:** Six standalone engineering specs — Merkle gadget, δ-circuit, Nova wrapper, lattice swap, wire protocol, test debt — each detailed enough to implement without ambiguity.

---

## Conventions Used Throughout

- **Crate root:** `crates/q-ivc/`
- **Field:** `F: PrimeField` (parametric; concrete instantiation is BN254 `Fr` for Phase 2, lattice modulus for Phase 4)
- **Hash:** `crates/q-ivc/src/gadgets/blake3.rs::Blake3Gadget` — `Vec<UInt32<F>>` for hash inputs/outputs (32-byte hash = 8 × UInt32)
- **Signatures:** `crates/q-ivc/src/gadgets/dilithium.rs::DilithiumVerifierGadget`
- **Existing skeleton:** `crates/q-ivc/src/circuits/epoch_transition.rs::EpochTransitionCircuit`
- **R1CS layer:** arkworks 0.5 (`ark-r1cs-std`, `ark-relations`)
- **State-root primitive in production:** BLAKE3, 32 bytes, domain-separated by `"balance_root_v1"` (`crates/q-storage/src/lib.rs`)
- **Wallet address:** `[u8; 32]` (BLAKE3 of pubkey)
- **Balance amount:** `u128` (16 bytes, little-endian)

---

# Blueprint 1: Merkle-Path Gadget + balance_root_v2 (Sparse Merkle Tree)

## Why this is two pieces of work, not one

The current `balance_root_v1` is a *flat hash* over sorted `(addr, balance)` pairs (`crates/q-storage/src/lib.rs`):

```rust
let mut root_hasher = blake3::Hasher::new();
root_hasher.update(b"balance_root_v1");
for leaf in &leaf_hashes {
    root_hasher.update(leaf);   // O(N) in N wallets
}
*root_hasher.finalize().as_bytes()
```

A ZK proof of "balance(addr) = b" against this commitment is O(N) hashes in the circuit — infeasible (N is currently ~10⁵ wallets, will grow to 10⁷+).

**Required migration:** introduce `balance_root_v2` as a **sparse Merkle tree** (SMT) keyed by the 32-byte address. Path proofs are O(log₂(2²⁵⁶)) = 256 hashes — but with key compaction tricks, in practice ~30-40 hashes per proof. ZK-feasible.

## File Layout

```
crates/q-storage/src/balance_smt.rs            new, ~800 LOC
crates/q-ivc/src/gadgets/merkle.rs             new, ~600 LOC
crates/q-types/src/upgrades.rs                 modify — add BALANCE_ROOT_V2_HEIGHT
```

## balance_root_v2 (storage-side, `crates/q-storage/src/balance_smt.rs`)

### Tree structure

- Depth 256, keyed by full 32-byte address (256 bits = depth)
- Leaf value: `BLAKE3("smt_leaf_v2" || addr[..] || balance.to_le_bytes())`
- Empty subtree hashes precomputed at every depth (`empty_subtree[d]` for d = 0..256)
- Path-compaction: when a sibling at depth d is `empty_subtree[d]`, that step of the path can be encoded as a single bit and reconstructed
- Internal node: `BLAKE3("smt_node_v2" || left || right)`

### Public API

```rust
pub struct BalanceSmt {
    db: Arc<RocksDB>,
    column_family: ColumnFamilyHandle,   // "cf_balance_smt"
    cached_root: parking_lot::RwLock<[u8; 32]>,
    empty_subtree: [[u8; 32]; 257],      // precomputed at startup
}

impl BalanceSmt {
    pub fn new(db: Arc<RocksDB>) -> Result<Self>;

    /// Replay every wallet balance into the SMT at startup.
    /// Idempotent — safe to call after a checkpoint snapshot.
    pub async fn rebuild_from_balances(&self, balances: &HashMap<[u8;32], u128>) -> Result<[u8;32]>;

    /// Insert or update (addr, balance). Updates root. O(log N) RocksDB writes.
    pub async fn update(&self, addr: &[u8;32], balance: u128) -> Result<[u8;32]>;

    /// Batched update — atomic, returns final root. Used after each block.
    pub async fn batch_update(&self, updates: &[([u8;32], u128)]) -> Result<[u8;32]>;

    /// Generate a Merkle path proof: returns 256 sibling hashes + a 256-bit
    /// "is_empty_sibling" bitmap to compact the empties.
    pub async fn prove(&self, addr: &[u8;32]) -> Result<SmtProof>;

    /// Current root (cached, updated atomically with batch_update).
    pub fn root(&self) -> [u8;32];
}

pub struct SmtProof {
    pub addr: [u8;32],
    pub balance: u128,
    pub siblings: [[u8;32]; 256],
    pub empty_bitmap: [u8; 32],   // bit i = 1 if siblings[i] is empty_subtree[i]
}

impl SmtProof {
    pub fn verify(&self, expected_root: &[u8;32]) -> bool;
}
```

### Rules

- `cf_balance_smt` is a NEW column family, NOT a migration of `cf_wallet_balances`
- Both `balance_root_v1` and `balance_root_v2` are maintained in parallel until `BALANCE_ROOT_V2_HEIGHT` activation (mainnet-safety pattern from CLAUDE.md)
- Block header gains a new field `state_root_v2: [u8;32]` for blocks ≥ activation height (defaulted to `[0u8;32]` before)
- `save_wallet_balances` and `save_wallet_balances_batch` MUST update the SMT inside the same RocksDB write-batch (atomicity required)
- Genesis: SMT root = `BLAKE3("smt_node_v2" || empty_subtree[1] || empty_subtree[1])` recursively up — precomputed constant

### Test plan
- Empty tree → known constant root
- Insert one entry → matches handcomputed root
- Insert 1000 random entries, query each, verify proof
- Batch update vs sequential update → identical root
- Adversarial: invalid path with one sibling tampered → verify returns false

## Merkle-path gadget (circuit-side, `crates/q-ivc/src/gadgets/merkle.rs`)

### Goal
Prove inside an R1CS circuit: given `(addr, balance, path)` and a public `root`, that the Merkle path is consistent — i.e., `compute_root(addr, balance, path) == root`.

### Public API

```rust
pub struct MerklePathGadget;

impl MerklePathGadget {
    /// Compute leaf hash for (addr, balance).
    /// Returns BLAKE3("smt_leaf_v2" || addr || balance_le) as Vec<UInt32<F>>.
    pub fn leaf_hash<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        addr_bits: &[Boolean<F>; 256],
        balance: &FpVar<F>,            // constrained ≤ 2^128 by caller
    ) -> Result<Vec<UInt32<F>>, SynthesisError>;

    /// Compute root from (leaf, addr_bits, siblings, empty_bitmap).
    /// Returns Vec<UInt32<F>> (8 words = 256 bits = BLAKE3 output).
    pub fn compute_root<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        leaf_hash: &[UInt32<F>],
        addr_bits: &[Boolean<F>; 256],
        siblings: &[Vec<UInt32<F>>; 256],
        empty_bitmap: &[Boolean<F>; 256],
        empty_subtree_hashes: &[Vec<UInt32<F>>; 257],   // public constants
    ) -> Result<Vec<UInt32<F>>, SynthesisError>;

    /// Enforce: compute_root(...) == expected_root.
    pub fn enforce_membership<F: PrimeField>(
        cs: ConstraintSystemRef<F>,
        addr_bits: &[Boolean<F>; 256],
        balance: &FpVar<F>,
        siblings: &[Vec<UInt32<F>>; 256],
        empty_bitmap: &[Boolean<F>; 256],
        empty_subtree_hashes: &[Vec<UInt32<F>>; 257],
        expected_root: &[UInt32<F>],
    ) -> Result<(), SynthesisError>;
}
```

### Constraint sketch (per path)

For `d` in `0..256` (bottom-to-top):
1. Compute `effective_sibling` = `select(empty_bitmap[d], empty_subtree_hashes[d], siblings[d])`
2. Compute `parent` = `select(addr_bits[d], BLAKE3(effective_sibling, current), BLAKE3(current, effective_sibling))`
3. `current = parent`

After loop: `enforce_equal(current, expected_root)`.

### Cost
- 256 × BLAKE3-compression invocations (existing `Blake3Gadget::compress`)
- Each BLAKE3-compression in the existing gadget is ~2,300 constraints (per blake3.rs audit)
- Total: ~590,000 constraints per path proof
- One block transition with K transactions touches 2K paths (sender + receiver) + 1 coinbase path
- For K=100 transactions: ~120 million constraints per block

That's large but tractable for Nova folding (Nova's relaxed R1CS handles arbitrary-size single-step circuits — the work is per-block, not amortized).

### Optimization opportunities (future work, NOT in initial implementation)
- Replace BLAKE3 with Poseidon (already have the gadget — `gadgets/poseidon.rs`, 329 LOC). Poseidon is ~100× cheaper in R1CS. Trade-off: `state_root_v2` would no longer be BLAKE3-computable outside the circuit, which means light clients without an arkworks engine can't verify roots. Decision: stay BLAKE3 for v2; consider Poseidon-rooted `state_root_v3` later.

### Test plan
- Single-leaf tree of depth 256, prove leaf membership
- 1000-leaf adversarial tree: any sibling bit flipped → proof fails
- Cross-check: gadget result == `BalanceSmt::prove()` then verified outside the circuit
- `empty_bitmap` consistency: if bit set, sibling slot must equal `empty_subtree_hashes[d]`

---

# Blueprint 2: δ-Circuit Composition (`crates/q-ivc/src/circuits/delta_block.rs`)

## Goal

One R1CS circuit, the **block transition circuit**, that takes:
- Public: `state_root_prev`, `state_root_next`, `block_header_hash`, `block_height`
- Private (witness): the entire block body and its Merkle paths
- Output: 1 if the block validly transitions `state_root_prev → state_root_next`, else infeasibility

## Existing primitives we wire together

| Constraint | Gadget |
|------------|--------|
| `block_header_hash = BLAKE3(header_bytes)` | `Blake3Gadget::verify_hash` |
| Per-tx signature: `Verify_Dilithium5(pk, msg, sig)` | `DilithiumVerifierGadget::verify` |
| Per-tx amount range: `0 ≤ amt < 2^128` | `enforce_norm_bound` (small wrapper) |
| Per-tx Merkle: `(from, bal_old)` ∈ `state_root_prev` | `MerklePathGadget::enforce_membership` |
| Per-tx Merkle: `(to, bal_to_old)` ∈ `state_root_prev` | same |
| Per-tx Merkle: updated `(from, bal_old - amt - fee)` ∈ `state_root_next` | same |
| Per-tx Merkle: updated `(to, bal_to_old + amt)` ∈ `state_root_next` | same |
| Sufficient balance: `bal_old ≥ amt + fee` | `enforce_norm_bound` on `bal_old - amt - fee` |
| Coinbase emission: `emission ≤ scheduled_rate(height)` | `enforce_norm_bound` + lookup |
| NTT anchor election | `NttVerifierGadget` (already done) |

## File layout

```
crates/q-ivc/src/circuits/delta_block.rs        new, ~1500 LOC
crates/q-ivc/src/circuits/mod.rs                modify — add pub mod delta_block
```

## Public API

```rust
pub struct DeltaBlockCircuit<F: PrimeField> {
    // Public inputs
    pub state_root_prev: Option<[u8; 32]>,
    pub state_root_next: Option<[u8; 32]>,
    pub block_header_hash: Option<[u8; 32]>,
    pub block_height: Option<u64>,

    // Private witness
    pub block_header_bytes: Option<Vec<u8>>,
    pub transactions: Option<Vec<TxWitness>>,
    pub coinbase: Option<CoinbaseWitness>,
    pub anchor_witness: Option<AnchorWitness>,    // for NTT-based anchor election
    pub empty_subtree_hashes: Vec<[u8; 32]>,      // 257 entries, public constants
}

pub struct TxWitness {
    pub from_addr: [u8; 32],
    pub to_addr: [u8; 32],
    pub amount: u128,
    pub fee: u128,
    pub nonce: u64,
    pub sig: DilithiumSigBytes,
    pub from_pk: DilithiumPubKey,

    // SMT proofs against state_root_prev
    pub from_path_prev: SmtProof,
    pub to_path_prev: SmtProof,
    // SMT proofs against state_root_next (with updated balances)
    pub from_path_next: SmtProof,
    pub to_path_next: SmtProof,
}

pub struct CoinbaseWitness {
    pub miner_addr: [u8; 32],
    pub emission: u128,
    pub scheduled_rate: u128,
    pub miner_path_prev: SmtProof,
    pub miner_path_next: SmtProof,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for DeltaBlockCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // Implementation below
    }
}
```

## Constraint generation order

1. **Allocate public inputs.** state_root_prev/next as `[UInt32<F>; 8]`, block_height as `FpVar<F>` (bounded ≤ 2⁶⁴).
2. **Allocate empty_subtree_hashes as constants** (no allocation cost, but they're inputs to MerklePathGadget).
3. **Verify block header hash:**
   ```rust
   let header_bytes_var = UInt8::new_witness_vec(cs.clone(), &self.block_header_bytes.unwrap_or_default())?;
   Blake3Gadget::verify_hash(&header_bytes_var, &public_header_hash)?;
   ```
4. **Loop over transactions** (sequential — Nova's strength is folding so this is fine even at 100 txs):
   ```rust
   let mut current_root = state_root_prev_var.clone();
   for tx in self.transactions.unwrap_or_default() {
       // Allocate witness
       let from_pk_var = Vec::<UInt8<F>>::new_witness(...);
       let sig_var     = ... ;
       let msg_var     = tx_signed_message(&tx);

       // 1. Signature verifies
       DilithiumVerifierGadget::verify(&from_pk_var, &msg_var, &sig_var)?;

       // 2. Range check amount and fee
       enforce_norm_bound(&amount_var, 128)?;
       enforce_norm_bound(&fee_var, 64)?;

       // 3. From balance Merkle prev
       MerklePathGadget::enforce_membership(
           &from_addr_bits, &from_bal_old, &from_path_prev_siblings,
           &from_path_prev_empty_bitmap, &empty_subtree, &current_root)?;

       // 4. To balance Merkle prev (against same current_root)
       MerklePathGadget::enforce_membership(
           &to_addr_bits, &to_bal_old, &to_path_prev_siblings,
           &to_path_prev_empty_bitmap, &empty_subtree, &current_root)?;

       // 5. Balance sufficiency: from_bal_new = from_bal_old - amount - fee, ≥ 0
       let from_bal_new = &from_bal_old - &amount_var - &fee_var;
       enforce_norm_bound(&from_bal_new, 128)?;

       // 6. To gets credit: to_bal_new = to_bal_old + amount
       let to_bal_new = &to_bal_old + &amount_var;
       enforce_norm_bound(&to_bal_new, 128)?;

       // 7. From-path against post-tx intermediate root (use the from_path_next siblings)
       //    — this gives us the new root after the from-side update
       let intermediate_root = MerklePathGadget::compute_root(
           &from_bal_new_leaf, &from_addr_bits,
           &from_path_next_siblings, &from_path_next_empty_bitmap, &empty_subtree)?;

       // 8. To-path against intermediate_root, yields current_root := next_root
       let next_root = MerklePathGadget::compute_root(
           &to_bal_new_leaf, &to_addr_bits,
           &to_path_next_siblings, &to_path_next_empty_bitmap, &empty_subtree)?;

       current_root = next_root;
   }
   ```
5. **Coinbase emission:**
   ```rust
   let coinbase = self.coinbase.unwrap_or_default();
   let scheduled = scheduled_emission_lookup(&block_height_var)?;
   enforce_emission_lte_scheduled(&coinbase.emission_var, &scheduled)?;
   // Update miner balance via Merkle paths as above
   ```
6. **Anchor election check:**
   ```rust
   NttVerifierGadget::verify_anchor(&anchor_witness, &block_height_var)?;
   ```
7. **Final equality:** `current_root` after all txs and coinbase MUST equal `state_root_next_var`.
   ```rust
   for (a, b) in current_root.iter().zip(state_root_next_var.iter()) {
       a.conditional_enforce_equal(b, &Boolean::TRUE)?;
   }
   ```

## Scheduled emission lookup

The emission schedule (Era 0: 2,625,000 QUG/year, 4-year halving, see `crates/q-emission/`) needs an in-circuit lookup. Two options:

- **Option A (preferred):** Precompute a piecewise-linear table at compile time. The schedule is era-step-function: emission/block is constant within an era, halves at era boundary. The circuit input `block_height` selects an era index via range checks, then a fixed `era_emission_rate[era]` constant table lookup.
- **Option B:** Compute the schedule inside the circuit. Requires power-of-2 division which is ZK-expensive. Avoid.

Define era boundaries at compile time:
```rust
const ERA_BOUNDARIES: [(u64, u128); 8] = [
    (0,         0x2624A5F86FA000_u128),  // era 0: 2,625,000/year / 31,536,000 sec * scale
    (BLOCKS_PER_4_YEARS,     half_of_above),
    ...
];
```

Inside the circuit, do range-membership checks: `era == 0 iff height ∈ [0, BLOCKS_PER_4_YEARS)`, etc., then select the right rate.

## Cost estimate (per block, K=100 txs)

| Component | Constraints |
|-----------|-------------|
| Block-header BLAKE3 | ~50,000 |
| Per-tx Dilithium5 verify | ~1,500,000 × 100 = 150 M |
| Per-tx 4 Merkle paths × ~590K | 4 × 590K × 100 = 236 M |
| Per-tx balance arithmetic | ~10,000 × 100 = 1 M |
| Coinbase | ~5 M |
| Anchor verification (NTT) | ~50 M |
| **Total** | **~440 million constraints** |

This is large. Nova's relaxed-R1CS folding handles it but proving time per block will be 30-90 seconds on commodity hardware. **Acceptable for the genesis-node prover.** Verifier cost (after folding) is constant ~5-10 ms — that's what matters for users.

## Sanity tests
1. Empty block (no txs, just coinbase) → circuit satisfied iff coinbase respects schedule
2. Single transfer A → B with valid Merkle paths → satisfied
3. Adversarial: amount > balance → unsatisfiable (constraint failure)
4. Adversarial: forged signature → unsatisfiable
5. Adversarial: state_root_next inconsistent with computed root → unsatisfiable
6. Replay attack: same nonce reused → must be caught by signed-message convention (nonce binding)

---

# Blueprint 3: Nova IVC Recursion Wrapper (`crates/q-ivc/src/recursion/`)

## Crate choice

**Recommendation: `nova-snark` (Microsoft, latest 0.36+).**

- Production track record (Lurk, Hyle, others)
- Actively maintained
- Native R1CS / `StepCircuit` trait
- Works with arkworks-style `ark-bn254` / pasta curves
- Verifier ~5 ms

Alternative: `arkworks-rs/nova` (community) — lower maturity, more arkworks-native. Spend ~3 days prototyping a minimal step circuit on both, pick based on dev-experience.

## File layout

```
crates/q-ivc/src/recursion/mod.rs              new, ~50 LOC
crates/q-ivc/src/recursion/step.rs             new, ~400 LOC — StepCircuit impl
crates/q-ivc/src/recursion/folder.rs           new, ~300 LOC — Fold orchestrator
crates/q-ivc/src/recursion/verify.rs           new, ~150 LOC — Verifier helper
crates/q-ivc/Cargo.toml                        modify — add nova-snark dep
```

## StepCircuit implementation

```rust
use nova_snark::traits::circuit::StepCircuit;
use nova_snark::traits::Group;

pub struct QnkStepCircuit<F: PrimeField> {
    pub delta_circuit: DeltaBlockCircuit<F>,
}

impl<G: Group, F: PrimeField> StepCircuit<F> for QnkStepCircuit<F> {
    fn arity(&self) -> usize {
        // We track 1 piece of state across folds: state_root.
        // (Block height is implicit in the fold count i — Nova exposes it as the "step" counter.)
        // state_root as F via felt-encoding: pack 32 bytes into 2 F field elements.
        2
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z_in: &[FpVar<F>],   // z_in[0..2] = state_root_prev as 2 field elements
    ) -> Result<Vec<FpVar<F>>, SynthesisError> {
        // Decompose z_in into state_root_prev bytes
        let state_root_prev_bytes = unpack_state_root_from_field_pair(cs, &z_in[..2])?;

        // Run the δ-circuit. It outputs state_root_next as bytes.
        let state_root_next_bytes = self.delta_circuit
            .generate_constraints_with_output(cs.namespace(|| "delta"), &state_root_prev_bytes)?;

        // Pack state_root_next into 2 F field elements
        let z_out = pack_state_root_to_field_pair(cs, &state_root_next_bytes)?;
        Ok(z_out)
    }
}
```

Important: the existing `DeltaBlockCircuit::generate_constraints` takes `state_root_prev` and `state_root_next` as **both public**. To use it as a `StepCircuit`, refactor to make `state_root_next` an *output* computed from `state_root_prev` and the block witness, instead of a public input we equality-check. (The equality-check still happens implicitly because Nova binds the output to the next step's input.)

## Folder orchestrator

```rust
pub struct QnkFolder<F: PrimeField> {
    pub pp: nova_snark::PublicParams<...>,   // generated once at setup
    pub recursive_snark: Option<RecursiveSNARK<...>>,
    pub current_state_root: [u8; 32],
    pub current_block_height: u64,
}

impl QnkFolder<F> {
    /// Set up Nova public parameters. Called once per genesis-node startup.
    pub fn setup() -> Result<Self>;

    /// Fold one block. Called per block produced.
    pub async fn fold_block(&mut self, delta: DeltaBlockCircuit<F>) -> Result<()>;

    /// Get the current folded proof. Serialize for /api/v1/proof/tip.
    pub fn current_proof(&self) -> Result<Vec<u8>>;

    /// Verify a serialized proof (verifier-only path; doesn't need the prover key).
    pub fn verify(serialized_proof: &[u8], expected_state_root: &[u8; 32], height: u64) -> Result<bool>;
}
```

## Setup performance (one-time)

Nova `PublicParams` for a 440 M-constraint step circuit will take 10-30 minutes to generate on first run. Bake this into a genesis ceremony or distribute as a downloadable file (~100 MB). Verify the parameters' hash matches a committed-in-code constant.

## Fold performance (per block)

Target: ≤ 60 seconds per fold on a 48-core Epsilon-class machine.

Profiling will likely show Dilithium5 sig verification dominates. Optimization paths:
- Batch sig verification (one Nova step proves K signatures in parallel, K=4-8) — reduces per-tx cost
- Hardware-accel BLAKE3 (already exists in the gadget but worth checking the constraint-count is minimal)
- FFI to a C-NTT library if Poseidon migration becomes viable

## Verifier API

```rust
pub fn verify_recursive_proof(
    proof_bytes: &[u8],
    public_state_root: &[u8; 32],
    public_tip_height: u64,
) -> Result<bool> {
    let proof: RecursiveSNARK<...> = bincode::deserialize(proof_bytes)?;
    let z_final = pack_state_root_to_field_pair(state_root);
    proof.verify(&pp, public_tip_height, vec![z_final])
        .map(|_| true)
        .or(Ok(false))
}
```

Target: 5 ms on M2 / 8-core x86, 10 ms on commodity laptop, 250 ms in browser WASM.

## Sanity tests
1. Fold 0 blocks → trivial proof, verify succeeds with genesis state-root
2. Fold 1 block → verify succeeds with `state_root_1`
3. Fold 1000 blocks → verify succeeds; serialized proof size constant
4. Adversarial: swap one block's witness for an invalid one mid-fold → folder should fail at that step
5. Adversarial: tamper with serialized proof → verify returns false

---

# Blueprint 4: Lattice Flavor Swap (Phase 4, R&D track)

## Why this is research-grade

No production Rust implementation of a lattice IVC exists today (2026). The Q1 2028 target is aggressive but realistic if we start scoping now.

## Candidate evaluation

### Candidate A: LatticeFold (Boneh-Chen-Tairi 2024)
- **Pros:** Designed specifically for IVC. Module-SIS hardness (PQ). Sub-linear verification.
- **Cons:** No reference impl. Paper is from 2024 — implementation work is ground-up.
- **Estimated effort:** 12 person-months for a reference implementation.

### Candidate B: LaBRADOR + custom recursion wrapper
- **Pros:** Reference implementation exists (Beullens/Seiler 2023 Rust prototype). Mature crypto.
- **Cons:** Native LaBRADOR is *not* IVC — it's an argument of knowledge for R1CS. We'd need to write the recursion wrapper from scratch.
- **Estimated effort:** 8 person-months — 4 for wrapping LaBRADOR in an IVC harness, 4 for production hardening.

### Candidate C: Greyhound (Nguyen-Seiler 2024)
- **Pros:** Reference impl exists, batch-friendly (could prove multiple blocks per fold).
- **Cons:** Not designed for IVC primarily, would need adaptation.
- **Estimated effort:** 10 person-months.

## Decision pathway (NOT decided yet — instrumentation first)

1. **Late 2026:** Read each paper, prototype a *non-IVC* lattice proof of single-block validity using each scheme. Measure: verify time, proof size, prover time.
2. **Q1 2027 (Nova in production already by then):** Pick the candidate with the best Phase 2/3 prover-cost trajectory.
3. **Q1-Q3 2027:** Build lattice IVC harness for chosen scheme.
4. **Q4 2027:** Side-by-side Nova/lattice testnet, identical δ-circuit, compare proof times and verifier times.
5. **Q1 2028:** Production swap.

## What stays the same across the swap

- The **δ-circuit definition** in `crates/q-ivc/src/circuits/delta_block.rs` — unchanged.
- The **storage layer** (`balance_root_v2`) — unchanged.
- The **wire protocol** (`/api/v1/proof/tip`) — only the proof encoding changes; the API shape is the same.
- The **consensus rules** — the proof is advisory through Phase 3; the swap is invisible to users.

## What changes

- `Cargo.toml`: replace `nova-snark` with `q-lattice-ivc` (in-house crate or wrapping the chosen library)
- `crates/q-ivc/src/recursion/`: re-implement `QnkFolder` against the lattice scheme's API
- `crates/q-ivc/src/recursion/verify.rs`: re-implement verifier
- Public parameters file (`nova-params.bin` → `lattice-params.bin`)

## Risk surface for the lattice swap
- Soundness bugs in lattice proof systems are subtle — most published schemes have had 1-2 soundness errata before settling
- Side-channel attacks on lattice operations (NTT timing variations) — must use constant-time implementations
- Specifying the *exact* security level (e.g., 128-bit post-quantum vs 192-bit) requires careful parameter selection per the NIST PQC analyses

---

# Blueprint 5: Wire Protocol — `GET /api/v1/proof/tip`

## Endpoints

### 5.1 `GET /api/v1/proof/tip`
**Purpose:** New node fetches the latest folded proof from a peer.

**Request:** No body. Optional query `?since_height=N` to get a delta proof (not implemented in v1).

**Response:**
```json
{
  "tip_height": 11400000,
  "state_root": "0xabcd…",
  "block_header": {
    "height": 11400000,
    "parent_hash": "0x…",
    "tx_root": "0x…",
    "state_root": "0xabcd…",
    "timestamp": 1771761600,
    "producer_id": 123,
    "anchor_validator": "..."
  },
  "proof_version": "nova-bn254-v1",       // or "latticefold-modulesis-v1" later
  "proof_size_bytes": 51232,
  "proof_b64": "<base64-encoded proof bytes>"
}
```

**Auth:** None (proof is publicly verifiable; this is bootstrap discovery, no authority).
**Caching:** `Cache-Control: max-age=2` (block time is ~1s, so 2s cache acceptable).

### 5.2 `POST /api/v1/proof/verify`
**Purpose:** Optional helper for clients without a verifier — submits proof to be verified by the server. Returns bool. Intended for browser/WASM clients without arkworks compiled in.

**Request body:**
```json
{
  "proof_b64": "...",
  "state_root": "0x...",
  "height": 11400000
}
```

**Response:**
```json
{
  "valid": true,
  "verified_in_ms": 4.7
}
```

### 5.3 Gossipsub topic
New topic: `/qnk/mainnet-genesis/recursive-proof`

**Message format:** Same as response 5.1 but with `tip_height` as routing key. Nodes maintain a "latest seen proof" and forward higher-height proofs only.

**Rate limit:** 1 message per block per peer.

## Implementation surface

```
crates/q-api-server/src/handlers.rs            modify — add 2 handlers
crates/q-api-server/src/main.rs                modify — wire new route + gossip topic
crates/q-network/src/gossipsub_topics.rs       modify — add topic
crates/q-ivc/src/recursion/verify.rs           used by handler 5.2
```

## Verification on the new node side

Pseudocode for `/bootstrap` flow when `--bootstrap-mode=proof` is set:

```rust
async fn bootstrap_via_proof(peer_url: &str) -> Result<BootstrappedState> {
    // 1. Fetch proof from any peer
    let r: ProofTipResponse = http_get(format!("{peer_url}/api/v1/proof/tip")).await?;

    // 2. Verify the proof locally (≤10 ms)
    let proof_bytes = base64::decode(&r.proof_b64)?;
    let valid = verify_recursive_proof(&proof_bytes, &r.state_root_bytes, r.tip_height)?;
    if !valid {
        return Err("Peer returned invalid proof — try another peer.");
    }

    // 3. Sanity-check block header binds to state_root
    let header_hash = blake3_hash(&postcard::to_vec(&r.block_header)?);
    if r.block_header.state_root != r.state_root {
        return Err("Header inconsistent with claimed state_root");
    }

    // 4. Accept state — node is now able to mine/transact immediately
    Ok(BootstrappedState {
        tip_height: r.tip_height,
        state_root: r.state_root,
        block_header: r.block_header,
        // Backfill of historical blocks starts in parallel,
        // independent of consensus participation
    })
}
```

## Bootstrap-mode flag

Add to `crates/q-api-server/src/main.rs` CLI:
```
--bootstrap-mode=proof    # New: download proof, verify, start instant
--bootstrap-mode=checkpoint  # Existing default
--bootstrap-mode=genesis  # Existing — full sync from height 1
```

## Tests

1. Local proof fetch + verify smoke test (single test container against mainnet Epsilon)
2. Adversarial: peer returns corrupted proof bytes → bootstrap rejects, tries next peer
3. Adversarial: peer returns valid proof but wrong block header → bootstrap rejects
4. Latency: verify time on `bzImage = 11.4M` height proof ≤ 10 ms on Epsilon, ≤ 50 ms on a Raspberry Pi 5

---

# Blueprint 6: Test Debt — Specific Fixes

This is small but blocking task #22. Three concrete fixes.

## 6.1 Three dilithium helpers using stale `neg_one` for `roots[1]`

Per the test convention fix already documented in MEMORY (`v10_2_9_fixes.md` and the NTT v3 doc), for `n=2` the correct value of `roots[1]` is `Fr::one()`, **not** `Fr::neg_one()`.

**Fix:** in `crates/q-ivc/src/gadgets/dilithium.rs`, find the three tests that still set `roots[1] = neg_one`. Probably grep for `neg_one` in the test module:

```bash
grep -n 'neg_one' crates/q-ivc/src/gadgets/dilithium.rs
```

For each match in a `#[test]` function:
```rust
// BEFORE
let roots = NttRoots { values: vec![Fr::one(), Fr::neg_one()] };
// AFTER
let roots = NttRoots { values: vec![Fr::one(), Fr::one()] };
```

Note: this is *only* for the n=2 fixture. Larger n's use the proper roots of unity from the field characteristic — those tests are already correct. Verify by running the test and checking it passes against a known good vector (compute roots externally with sage/python and compare).

## 6.2 `test_use_hint_with_bias` math error

This test exercises `use_hint(coeff, hint, alpha)` from FIPS 204 §6.5.2. The expected output value was computed wrong. Re-derive:

Given α = 2γ₂ = 95,232 (Dilithium5 parameters), the use_hint operation:

```
m = (q-1) / alpha    where q = 2^23 - 2^13 + 1 = 8,380,417
                     m = 87
HighBits(r, alpha) = ((r % q) + alpha/2) / alpha   (rounded)
LowBits(r, alpha) = r - HighBits(r, alpha) * alpha

UseHint(r, h, alpha):
  if h == 0:
    return HighBits(r, alpha)
  else:
    let r1 = HighBits(r, alpha)
    if LowBits(r, alpha) > 0: r1 + 1 mod m
    else: r1 - 1 mod m
```

For the "with bias" test case in the file, the test sets:
- `r = some specific value near a step boundary`
- `h = 1`
- expected `r1' = computed value`

The current expected value is off-by-one (suspect: forgot to apply `mod m`).

**Fix:** locate the test (`grep -n 'test_use_hint_with_bias' crates/q-ivc/src/gadgets/dilithium.rs`), recompute the expected output using the formula above (a small Python script suffices), update the assertion.

## 6.3 `enforce_signed_norm_bound` arkworks `is_cmp` limitation

The current implementation is positive-only with a documented CAVEAT:
```rust
// CAVEAT: arkworks FpVar::is_cmp is broken for values near p.
// This implementation only handles positive coefficients [0, bound).
// Full signed [-bound, bound] support requires either:
// - Patched arkworks (upstream PR pending), or
// - Manual range-decomposition (~3× more constraints)
```

**Not a fix; status documentation.** This is acceptable because:
1. Dilithium5 polynomial coefficients used in `Az - ct` are always in `[0, q)` (centered representation done at the gadget boundary)
2. Full signed support is needed only for adversarial-input scenarios that the protocol doesn't admit

No code fix required. Confirm the CAVEAT block is present and accurate.

---

# Cross-Cutting: Acceptance Criteria

A complete handoff from DeepSeek/outside-implementer should satisfy:

1. ✅ All existing q-ivc tests pass (`cargo test --package q-ivc`)
2. ✅ New `balance_smt` tests pass against handcomputed reference vectors
3. ✅ `MerklePathGadget::enforce_membership` test: produce 100 random SMT instances, prove + verify each in-circuit
4. ✅ `DeltaBlockCircuit` test on synthetic block: 10 random transfers, all valid → satisfied
5. ✅ `DeltaBlockCircuit` adversarial test: 5 well-defined fail modes (bad sig, insufficient balance, wrong state_root, etc.) → unsatisfiable in each case
6. ✅ Nova `QnkFolder::fold_block` × 100 sequential blocks completes, final proof verifies
7. ✅ Verifier `verify_recursive_proof` against a height-100 proof: < 10 ms on Beta (Intel Xeon)
8. ✅ `/api/v1/proof/tip` endpoint serves valid proofs, `/api/v1/proof/verify` returns correct booleans
9. ✅ `--bootstrap-mode=proof` Docker integration test: fresh container with no data dir, points at Epsilon, bootstraps + verifies + accepts transactions within 30 seconds
10. ✅ All existing mainnet-safety tests still pass (sync_down_protection, balance_propagation, fork_reorg, etc.)
11. ✅ No regression on v10.9.14 soak metrics (memory, sync throughput)

---

# Summary Table — What Goes Where

| Blueprint | Crate / Files | Effort | Dependencies |
|-----------|--------------|--------|--------------|
| 1A. balance_root_v2 SMT | `crates/q-storage/src/balance_smt.rs`, `crates/q-types/src/upgrades.rs` | 3 wk | BLAKE3 lib, existing storage layer |
| 1B. Merkle gadget | `crates/q-ivc/src/gadgets/merkle.rs` | 3 wk | Blueprint 1A (for fixture generation), existing Blake3Gadget |
| 2. δ-circuit | `crates/q-ivc/src/circuits/delta_block.rs` | 4 wk | Blueprints 1B, all existing gadgets |
| 3. Nova IVC wrapper | `crates/q-ivc/src/recursion/` | 2 wk | Blueprint 2, `nova-snark` crate |
| 4. Lattice swap | TBD | 8-12 mo R&D | Blueprint 3 production-validated |
| 5. Wire protocol | `crates/q-api-server/src/handlers.rs` | 1 wk | Blueprint 3 |
| 6. Test debt fixes | `crates/q-ivc/src/gadgets/dilithium.rs` | 1.5 d | None |

Total to working **advisory** mainnet recursive proof (Blueprints 1-3, 5, 6): **~13 weeks of focused engineering**. Lattice swap is additive on top.

— Server Beta, 2026-05-13
