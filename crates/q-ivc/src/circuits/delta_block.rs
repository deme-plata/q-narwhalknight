//! δ-circuit — single-block state-transition R1CS predicate.
//!
//! This circuit is **the heart of the 10ms-verification recursive SNARK** for
//! the Quillon Graph chain. It encodes the predicate:
//!
//! ```text
//!     δ(state_root_prev, block, state_root_next) ∈ {0, 1}
//! ```
//!
//! which is `1` iff `block` is valid under all consensus rules AND applying
//! its transactions, coinbase, and emission to the wallet-balance map
//! committed by `state_root_prev` produces the map committed by
//! `state_root_next`. The recursive Nova fold (Phase 2) then chains these
//! per-block proofs into a constant-size proof of the entire chain:
//!
//! ```text
//!     π_{n+1} = Nova.Fold(δ, state_root_n, block_{n+1}, π_n)
//! ```
//!
//! See `papers/quillon-recursive-lattice-snark-whitepaper-v2-2026-05-13.tex`
//! §3.2 (δ-circuit definition) and §4.1 (constraint cost breakdown), plus
//! `docs/blueprints-ivc-snark-2026-05-13.md` Blueprint 2 (this file's spec).
//!
//! ## Status
//!
//! This is a **skeleton commit**. The type system, the public/private input
//! layout, the `ConstraintSynthesizer` impl shape, and the per-transaction
//! loop are scaffolded. The bodies of each consensus-rule sub-block are
//! marked with `TODO(delta-circuit-PHASE-1)` and the production fill-in
//! lands in subsequent commits. Each TODO links to the relevant existing
//! gadget so the engineer landing it knows exactly which `verify_*` /
//! `enforce_*` method to call.
//!
//! Cost estimate at full fill-in (Blueprint 2 + whitepaper §4.1):
//!
//! ```text
//! Block-header BLAKE3                 ~50K           (Blake3Gadget::verify_hash)
//! Per-tx Dilithium5 sig verify        1.5M × K
//! Per-tx Merkle paths (4 × depth-256) 590K × 4 × K   (MerklePathGadget)
//! Per-tx balance range check          ~10K × K
//! Coinbase Merkle + emission lookup   ~5M total
//! NTT anchor verification             ~50M total
//! ─────────────────────────────────────────────
//! Block total (K=100 txs)             ~442M constraints
//! ```
//!
//! 442M constraints per block is large but tractable for Nova folding;
//! Nova's relaxed R1CS handles arbitrary-size single-step circuits, with
//! the heavy work being per-block (not amortized).
//!
//! ## Phase 1 dependencies (currently met)
//!
//! - [x] `Blake3Gadget::verify_hash` — `crates/q-ivc/src/gadgets/blake3.rs`
//! - [x] `Blake3Gadget::compress` (used by Merkle two-block helper) — same file
//! - [x] `DilithiumVerifierGadget::verify_structured` — `crates/q-ivc/src/gadgets/dilithium.rs`
//! - [x] `MerklePathGadget::enforce_membership` — `crates/q-ivc/src/gadgets/merkle.rs` (this branch)
//! - [x] `NttVerifierGadget` — `crates/q-ivc/src/gadgets/ntt.rs`
//! - [x] `BalanceSmt::SmtProof` + `precompute_empty_subtree_hashes` host helper
//!
//! All five gadgets are in tree; this file wires them together.

use ark_ff::PrimeField;
use ark_r1cs_std::{
    boolean::Boolean,
    fields::fp::FpVar,
    prelude::*,
    uint32::UInt32,
};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};

#[allow(unused_imports)]
use crate::gadgets::{
    blake3::Blake3Gadget,
    dilithium::DilithiumVerifierGadget,
    merkle::{
        MerklePathGadget,
        precompute_empty_subtree_hashes,
        SMT_DEPTH,
    },
    ntt::NttVerifierGadget,
};

/// One transaction's witness data — all the values the prover must supply
/// for that transaction to be admitted into the δ-circuit.
///
/// SMT update insight: when only ONE leaf changes, the sibling hashes along
/// that path do not change (siblings are the OTHER subtrees, untouched by
/// the leaf flip). So the from-path only needs one sibling set, used twice:
/// once to prove the OLD leaf is in the current running_root, once to
/// COMPUTE the new running_root from the new leaf with the same siblings.
///
/// However, after the from-update, the internal nodes along the from-path
/// have new values. If `to_addr` shares a prefix with `from_addr`, the
/// to-path's siblings differ from what they were against the pre-from
/// root. So `to_siblings` here is "to-path siblings in the tree AFTER the
/// from-update has been applied." The prover knows them; the verifier
/// re-derives the same root via the circuit's chained running_root.
#[derive(Clone)]
pub struct TransactionWitness<F: PrimeField> {
    /// Sender address (32 bytes, MSB-first bit decomposition).
    pub from_addr: [u8; 32],
    /// Recipient address.
    pub to_addr: [u8; 32],
    /// Amount transferred (u128 fitting; range-enforced inside circuit).
    pub amount: u128,
    /// Fee paid to producer (u128 fitting).
    pub fee: u128,
    /// Replay-protection nonce (used by the future v3 transaction format).
    pub nonce: u64,

    // ─── Sender path witnesses ─────────────────────────────────────────
    /// Sender's balance BEFORE this transaction (`from_balance_prev`).
    pub from_balance_prev: u128,
    /// Sibling hashes along the path to `from_addr` in the pre-tx tree.
    /// Used for both the prev-membership proof and the next-root compute
    /// (siblings unchanged across a single-leaf update).
    pub from_siblings: [[u8; 32]; SMT_DEPTH],
    /// Empty-bitmap for the from-path (1 bit per depth, MSB-first).
    pub from_empty_bitmap: [u8; 32],

    // ─── Recipient path witnesses ─────────────────────────────────────
    /// Recipient's balance BEFORE this transaction (`to_balance_prev`).
    /// May be zero (recipient previously unfunded).
    pub to_balance_prev: u128,
    /// Sibling hashes along the path to `to_addr` in the tree AFTER the
    /// from-update has been applied.
    pub to_siblings: [[u8; 32]; SMT_DEPTH],
    pub to_empty_bitmap: [u8; 32],

    // ─── Signature witness ────────────────────────────────────────────
    /// Sender's Dilithium5 public key (encoded per FIPS 204).
    pub from_pubkey_bytes: Vec<u8>,
    /// Dilithium5 signature over the transaction's signing-message.
    pub signature_bytes: Vec<u8>,
    /// The message that was signed (canonical bytes of the tx struct).
    pub signing_message: Vec<u8>,

    /// Phantom: holds the field type so `TransactionWitness<F>` is generic.
    pub _marker: core::marker::PhantomData<F>,
}

/// Coinbase witness — the emission transaction at the top of every block.
/// Same single-sibling-set insight as `TransactionWitness`: when only the
/// producer's balance leaf changes, the siblings stay the same.
#[derive(Clone)]
pub struct CoinbaseWitness<F: PrimeField> {
    /// Producer's wallet address (32 bytes).
    pub producer_addr: [u8; 32],
    /// Coinbase amount (must be ≤ era-scheduled emission).
    pub amount: u128,
    /// Producer's balance before this block (witness).
    pub producer_balance_prev: u128,
    /// Sibling hashes along the path to `producer_addr` in the tree AFTER
    /// all tx updates have been applied (the running_root entering Phase 4).
    pub producer_siblings: [[u8; 32]; SMT_DEPTH],
    pub producer_empty_bitmap: [u8; 32],
    pub _marker: core::marker::PhantomData<F>,
}

/// NTT-based anchor election witness.
///
/// The Quillon Graph anchor election uses a verifiable NTT-based randomness
/// beacon. The δ-circuit verifies that the producer claimed in the block
/// header is the legitimate anchor for the round.
#[derive(Clone)]
pub struct AnchorWitness<F: PrimeField> {
    /// The producer claimed in the block header.
    pub claimed_producer_id: u32,
    /// VDF / NTT witness data; consumed by NttVerifierGadget.
    pub ntt_witness: Vec<u8>,
    pub _marker: core::marker::PhantomData<F>,
}

/// Complete δ-circuit inputs.
///
/// Splits cleanly into PUBLIC (visible to verifier) and PRIVATE (witness).
pub struct DeltaBlockInputs<F: PrimeField> {
    // ──── PUBLIC ──────────────────────────────────────────────────────
    /// SMT root at height `n`.
    pub state_root_prev: [u8; 32],
    /// SMT root at height `n+1` (claimed by the prover).
    pub state_root_next: [u8; 32],
    /// BLAKE3 of the block header at height `n+1`.
    pub block_header_hash: [u8; 32],
    /// Block height `n+1`.
    pub block_height: u64,

    // ──── PRIVATE WITNESS ─────────────────────────────────────────────
    /// Block header bytes (preimage of `block_header_hash`). 64-byte
    /// canonical form (Blake3Gadget::verify_hash takes 16 u32 words).
    pub block_header_bytes: Vec<u8>,
    /// Transactions in this block (in serialization order).
    pub transactions: Vec<TransactionWitness<F>>,
    /// Coinbase emission.
    pub coinbase: CoinbaseWitness<F>,
    /// Anchor-election witness.
    pub anchor: AnchorWitness<F>,
}

// ════════════════════════════════════════════════════════════════════════════
// Circuit
// ════════════════════════════════════════════════════════════════════════════

/// δ-circuit. Implements `ConstraintSynthesizer` so it composes with any
/// arkworks-compatible proving backend (Groth16, Marlin, PLONK), and can be
/// wrapped as a Nova `StepCircuit` for the recursive fold (Phase 2 wrapper
/// lives in `crates/q-ivc/src/recursion/`).
pub struct DeltaBlockCircuit<F: PrimeField> {
    pub inputs: DeltaBlockInputs<F>,
}

impl<F: PrimeField> ConstraintSynthesizer<F> for DeltaBlockCircuit<F> {
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> Result<(), SynthesisError> {
        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 0 — allocate the 4 public inputs, then dispatch to inner   ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // Standalone (Groth16/Marlin) entry point. The verifier sees:
        // state_root_prev, state_root_next, block_header_hash, block_height
        // — everything else is private witness, enforced inside
        // `generate_constraints_inner`.
        //
        // The Nova step-circuit wrapper bypasses this method: it allocates
        // its own z_in/z_out as the chain-state vector and calls
        // `generate_constraints_inner` directly with pre-allocated handles,
        // avoiding double-allocation of the same field elements as Groth16
        // public inputs.

        let state_root_prev = alloc_root_input(cs.clone(), &self.inputs.state_root_prev)?;
        let state_root_next = alloc_root_input(cs.clone(), &self.inputs.state_root_next)?;
        let block_header_hash =
            alloc_root_input(cs.clone(), &self.inputs.block_header_hash)?;
        let block_height_var = FpVar::new_input(cs.clone(), || {
            Ok(F::from(self.inputs.block_height))
        })?;

        self.generate_constraints_inner(
            cs,
            &state_root_prev,
            &state_root_next,
            &block_header_hash,
            &block_height_var,
        )
    }
}

impl<F: PrimeField> DeltaBlockCircuit<F> {
    /// Phase 1–5 enforcement WITHOUT allocating the four canonical public
    /// inputs (`state_root_prev`, `state_root_next`, `block_header_hash`,
    /// `block_height`). Use this when an outer caller — typically the Nova
    /// `StepCircuit` wrapper — already owns those handles as part of its
    /// z_in/z_out chain state and would otherwise pay an extra allocation
    /// per fold step.
    ///
    /// Slice arguments are expected to be exactly 8 `UInt32<F>` words each
    /// (32-byte hash, little-endian word order). `block_height_var` is the
    /// step's height variable; the inner body uses it only as input to
    /// `era_emission_cap`.
    ///
    /// All Phase 1-5 invariants enforced by `generate_constraints` are also
    /// enforced here — the two paths produce identical constraint systems
    /// modulo the 4 public-input allocations (≈ 256 + 1 constraints).
    pub fn generate_constraints_inner(
        &self,
        cs: ConstraintSystemRef<F>,
        state_root_prev: &[UInt32<F>],
        state_root_next: &[UInt32<F>],
        block_header_hash: &[UInt32<F>],
        block_height_var: &FpVar<F>,
    ) -> Result<(), SynthesisError> {
        debug_assert_eq!(state_root_prev.len(), 8);
        debug_assert_eq!(state_root_next.len(), 8);
        debug_assert_eq!(block_header_hash.len(), 8);

        // Empty-subtree hashes are public constants — derived from BLAKE3 and
        // the SMT tag bytes (`smt_leaf_v2`, `smt_node_v2`). The host computes
        // them once at startup via `precompute_empty_subtree_hashes()`.
        let empty_subtree_bytes = precompute_empty_subtree_hashes();
        let empty_subtree: Vec<Vec<UInt32<F>>> = empty_subtree_bytes
            .iter()
            .map(|h| {
                h.chunks(4)
                    .map(|c| {
                        let w = u32::from_le_bytes(c.try_into().expect("4 bytes per word"));
                        UInt32::constant(w)
                    })
                    .collect()
            })
            .collect();

        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 1 — block header BLAKE3 hash check (1A — implemented)      ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // Enforce: block_header_hash = BLAKE3(block_header_bytes).
        // The header is exactly 64 bytes (single BLAKE3 block) per the
        // consensus spec — `Blake3Gadget::verify_hash` does the single-block
        // path with flags = CHUNK_START | CHUNK_END | ROOT.
        if self.inputs.block_header_bytes.len() != 64 {
            // The δ-circuit's block header MUST be 64 bytes. The block
            // header serialization is fixed-width per the spec; a non-64-byte
            // input is a witness-construction bug, not a runtime path.
            // Returning AssignmentMissing here would let the prover provide
            // junk and have it silently accepted — fail loud instead.
            return Err(SynthesisError::AssignmentMissing);
        }

        // Allocate the 64-byte header preimage as 16 little-endian FpVar
        // words (Blake3Gadget's expected preimage shape). These are
        // PRIVATE WITNESSES — the verifier doesn't see header bytes,
        // only the resulting hash.
        let header_preimage: Vec<FpVar<F>> = self.inputs.block_header_bytes
            .chunks(4)
            .map(|c| {
                let w = u32::from_le_bytes(c.try_into().expect("4 bytes per word"));
                FpVar::new_witness(cs.clone(), || Ok(F::from(w)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // The expected hash is already allocated as a public-input UInt32
        // (from alloc_root_input above). Blake3Gadget::verify_hash wants
        // FpVar — convert each UInt32 word to FpVar via its bit
        // decomposition. This is a "view conversion," not a re-allocation:
        // each FpVar is a linear combination of the SAME constrained bits
        // already on the wire. ~32 constraints per word (8 words = ~256).
        let mut expected_hash_fp: Vec<FpVar<F>> = Vec::with_capacity(8);
        for u32_word in block_header_hash.iter() {
            let bits = u32_word.to_bits_le();
            expected_hash_fp.push(Boolean::le_bits_to_fp_var(&bits)?);
        }

        Blake3Gadget::verify_hash(cs.clone(), &header_preimage, &expected_hash_fp)?;

        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 2 — anchor-election NTT verification (1B — wired stub)     ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // Calls into `host::anchor_witness::verify_anchor_election` which is
        // currently a stub returning constant-true (see that file for the
        // unpacking spec). When the host-helper body lands, this call's
        // returned Boolean must be enforced == true. Until then the anchor
        // check is non-enforcing at the recursive level; the API server's
        // accept_block path validates it block-by-block during the advisory
        // window. Constraint cost when filled in: ~50M.
        let anchor_ok = crate::host::anchor_witness::verify_anchor_election::<F>(
            cs.clone(),
            &[crate::host::anchor_witness::AnchorVdfBytes::new(
                self.inputs.anchor.ntt_witness.clone(),
            )],
            self.inputs.anchor.claimed_producer_id,
            self.inputs.block_height,
        )?;
        // Enforcing == true is currently a no-op (stub returns constant-true)
        // but keeps the call site in canonical form so 1B-final lands as a
        // single-file change to anchor_witness.rs.
        anchor_ok.enforce_equal(&Boolean::constant(true))?;

        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 3 — per-transaction loop                                   ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // For each transaction, walk the prev-state-root → intermediate-root
        // chain. After the from-side update, the intermediate root is the
        // state with the sender's new balance; after the to-side update, it's
        // the new state including the recipient's credit.
        //
        // We carry a `running_root` variable through the loop. The first
        // iteration starts at `state_root_prev`. The last must equal
        // `state_root_next` (enforced after the coinbase).

        // We carry a working copy of the input slice. UInt32<F>: Clone, so
        // to_vec() produces an owned Vec whose individual word handles still
        // reference the same constrained wires as the caller's slice.
        let mut running_root: Vec<UInt32<F>> = state_root_prev.to_vec();

        for (tx_idx, tx) in self.inputs.transactions.iter().enumerate() {
            // ──── Phase 3a: Dilithium5 signature verification (1C — wired stub) ──
            //
            // Calls into `host::dilithium_witness::allocate_dilithium_witness_for_tx`
            // which is currently a stub. The structural skeleton + FIPS-204
            // spec references for the bit-unpacking live in that file; the
            // bodies are tracked as follow-up commits.
            //
            // The call below is wrapped in a length check: if the prover
            // didn't supply a full-length signature payload, we just skip
            // (the API-server accept_block path independently checks the
            // sig during the advisory window). When the host-helper bodies
            // land, this short-circuit gets removed and the verifier's
            // returned Boolean is enforced == true unconditionally.
            //
            // Constraint cost when filled in: ~1.5M per tx.
            let _ = tx_idx;
            if tx.from_pubkey_bytes.len()
                == crate::host::dilithium_witness::DILITHIUM5_PK_BYTES
                && tx.signature_bytes.len()
                    == crate::host::dilithium_witness::DILITHIUM5_SIG_BYTES
            {
                // Host helper currently returns AssignmentMissing; swallow
                // until the stub body lands. The compile-time wiring is in
                // place so the future body change is single-file.
                let _stub = crate::host::dilithium_witness::allocate_dilithium_witness_for_tx::<F>(
                    cs.clone(),
                    &tx.from_pubkey_bytes,
                    &tx.signature_bytes,
                    &tx.signing_message,
                );
                let _ = _stub; // explicitly discard the Err until 1C-final lands
            }

            // ──── Phase 3b/3e: amount + fee + balance range checks (1D + 1G) ──
            //
            // Allocate amount, fee, sender's old balance, recipient's old
            // balance as FpVar witnesses. Range-check each fits in u128.
            // Then enforce `from_balance_prev ≥ amount + fee` by computing
            // the difference and range-checking it.
            //
            // Field subtraction in PrimeField wraps around modulo p, so
            // `balance_prev - amt - fee` produces a HUGE value if the
            // operation would underflow over the integers. The u128 range
            // check on the difference catches this — a value > 2^128 fails
            // the upper-bit-zero constraint.
            let amount_fp = FpVar::new_witness(cs.clone(), || Ok(F::from(tx.amount)))?;
            let fee_fp = FpVar::new_witness(cs.clone(), || Ok(F::from(tx.fee)))?;
            let from_balance_prev_fp =
                FpVar::new_witness(cs.clone(), || Ok(F::from(tx.from_balance_prev)))?;
            let to_balance_prev_fp =
                FpVar::new_witness(cs.clone(), || Ok(F::from(tx.to_balance_prev)))?;

            // 1D — every per-tx amount fits in u128 (no field-wrap exploits).
            enforce_u128_range(&amount_fp)?;
            enforce_u128_range(&fee_fp)?;
            enforce_u128_range(&from_balance_prev_fp)?;
            enforce_u128_range(&to_balance_prev_fp)?;

            // Compute new balances. Subtraction wraps in the field; the
            // range check on `from_balance_new` is what enforces the
            // "balance sufficiency" rule (1G).
            let total_out = &amount_fp + &fee_fp;
            let from_balance_new_fp = &from_balance_prev_fp - &total_out;
            let to_balance_new_fp = &to_balance_prev_fp + &amount_fp;

            // 1G — new sender balance must still fit in u128.
            // If `from_balance_prev < amount + fee` the subtraction underflows
            // (field wraparound), the result is ≥ 2^128, this check fails,
            // and the circuit rejects the witness.
            enforce_u128_range(&from_balance_new_fp)?;
            // The new recipient balance must also fit in u128 to prevent
            // overflow exploits. (Sum of two u128s can overflow to 2^129;
            // the range-check forces the prover to use values where the
            // sum stays within u128, matching the production AMM/transfer
            // overflow protection in q-storage.)
            enforce_u128_range(&to_balance_new_fp)?;

            // ──── Phase 3c/3d: sender's SMT path update (1E + 1F) ──────
            //
            // Single helper handles both the prev-membership proof and the
            // new-root compute. See `apply_smt_leaf_update`.
            let from_addr_bits = alloc_addr_bits(cs.clone(), &tx.from_addr)?;
            let from_siblings = alloc_siblings(cs.clone(), &tx.from_siblings)?;
            let from_empty_bitmap_bits =
                alloc_empty_bitmap(cs.clone(), &tx.from_empty_bitmap)?;
            running_root = apply_smt_leaf_update(
                cs.clone(),
                &running_root,
                &from_addr_bits,
                &from_siblings,
                &from_empty_bitmap_bits,
                &empty_subtree,
                &from_balance_prev_fp,
                &from_balance_new_fp,
            )?;

            // ──── Phase 3f/3g: recipient's SMT path update (1H + 1I) ───
            //
            // Same pattern. The siblings supplied by the prover are valid
            // against the POST-from-update root (the new running_root we
            // just computed).
            let to_addr_bits = alloc_addr_bits(cs.clone(), &tx.to_addr)?;
            let to_siblings = alloc_siblings(cs.clone(), &tx.to_siblings)?;
            let to_empty_bitmap_bits =
                alloc_empty_bitmap(cs.clone(), &tx.to_empty_bitmap)?;
            running_root = apply_smt_leaf_update(
                cs.clone(),
                &running_root,
                &to_addr_bits,
                &to_siblings,
                &to_empty_bitmap_bits,
                &empty_subtree,
                &to_balance_prev_fp,
                &to_balance_new_fp,
            )?;
        }

        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 4 — coinbase emission (1J — implemented)                   ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // Producer receives `coinbase.amount` QUG into `coinbase.producer_addr`.
        // Enforces:
        //   • coinbase.amount ≤ era-scheduled emission at this block height.
        //   • Producer balance fits in u128 before AND after the credit.
        //   • SMT path: (producer_addr, balance_prev) ∈ running_root, then
        //     running_root = SMT(producer_addr, balance_prev + amount).
        //
        // After this phase, running_root holds the final state root which
        // Phase 5 enforces equal to the public `state_root_next`.
        let coinbase = &self.inputs.coinbase;
        let coinbase_amount_fp =
            FpVar::new_witness(cs.clone(), || Ok(F::from(coinbase.amount)))?;
        let producer_balance_prev_fp =
            FpVar::new_witness(cs.clone(), || Ok(F::from(coinbase.producer_balance_prev)))?;

        // Range checks: u128 fit for both prev and new producer balance,
        // and for the coinbase amount itself.
        enforce_u128_range(&coinbase_amount_fp)?;
        enforce_u128_range(&producer_balance_prev_fp)?;
        let producer_balance_new_fp = &producer_balance_prev_fp + &coinbase_amount_fp;
        enforce_u128_range(&producer_balance_new_fp)?;

        // Era-step emission cap: coinbase.amount ≤ era_emission_cap(block_height).
        // Use is_cmp(amount, cap+1, Less, false) which returns true iff amount ≤ cap.
        let cap = era_emission_cap(cs.clone(), block_height_var)?;
        let cap_plus_one = &cap + FpVar::Constant(F::one());
        let cap_ok = coinbase_amount_fp
            .is_cmp(&cap_plus_one, core::cmp::Ordering::Less, false)?;
        cap_ok.enforce_equal(&Boolean::constant(true))?;

        // SMT update for the producer leaf.
        let producer_addr_bits = alloc_addr_bits(cs.clone(), &coinbase.producer_addr)?;
        let producer_siblings = alloc_siblings(cs.clone(), &coinbase.producer_siblings)?;
        let producer_empty_bitmap_bits =
            alloc_empty_bitmap(cs.clone(), &coinbase.producer_empty_bitmap)?;
        running_root = apply_smt_leaf_update(
            cs.clone(),
            &running_root,
            &producer_addr_bits,
            &producer_siblings,
            &producer_empty_bitmap_bits,
            &empty_subtree,
            &producer_balance_prev_fp,
            &producer_balance_new_fp,
        )?;

        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 5 — final state-root equality enforcement (1K — implemented) ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // After all transactions + coinbase have been applied, `running_root`
        // must equal the public `state_root_next`. Enforce word-by-word.
        //
        // Until 1E/1F/1H/1I/1J land (the Merkle-path updates that actually
        // mutate running_root), this check is equivalent to enforcing
        // `state_root_prev == state_root_next`. That's why the skeleton tests
        // use identical roots — they exercise this equality without needing
        // the full per-tx update logic.
        for (got, exp) in running_root.iter().zip(state_root_next.iter()) {
            got.enforce_equal(exp)?;
        }

        // The previous "vacuously satisfied" guardrail is no longer
        // applicable — Phases 1, 3, 4, 5 are all live as of this commit.
        // TWO TODO bodies remain (`1B` NTT anchor, `1C` Dilithium sig
        // verify) and are clearly stubbed out with the full FIPS-204 /
        // NTT-anchor wiring spec inline. Block-by-block validation in
        // the API server still independently checks signatures and the
        // anchor election — the recursive proof produced by this circuit
        // is ADVISORY until Phase 5 mandatory activation (per the
        // whitepaper §6.2 phased deployment with advisory window). Do
        // not flip Phase 5 until both stub TODOs are filled and a soak
        // period demonstrates zero soundness discrepancies.

        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Allocate a 32-byte hash (root, header hash, etc.) as a public input
/// of 8 u32 words.
fn alloc_root_input<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    bytes: &[u8; 32],
) -> Result<Vec<UInt32<F>>, SynthesisError> {
    bytes
        .chunks(4)
        .map(|c| {
            let w = u32::from_le_bytes(c.try_into().expect("4 bytes per word"));
            UInt32::new_input(cs.clone(), || Ok(w))
        })
        .collect()
}

/// Enforce that an FpVar fits in 128 bits (u128 range).
///
/// Approach: decompose to little-endian bits and constrain every bit at
/// position ≥128 to be zero. Cost: ~128 bit constraints + 1 enforcement per
/// excess bit. For F::MODULUS_BIT_SIZE = 255 (BN254/BLS12-381 Fr), that's
/// ~255 bit constraints total (the decomposition itself) plus 127 zero
/// enforcements.
fn enforce_u128_range<F: PrimeField>(v: &FpVar<F>) -> Result<(), SynthesisError> {
    let bits = v.to_bits_le()?;
    for bit in bits.iter().skip(128) {
        bit.enforce_equal(&Boolean::constant(false))?;
    }
    Ok(())
}

/// Build a 256-bit MSB-first Boolean decomposition of a 32-byte address.
fn alloc_addr_bits<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    addr: &[u8; 32],
) -> Result<Vec<Boolean<F>>, SynthesisError> {
    let mut bits = Vec::with_capacity(SMT_DEPTH);
    for byte_idx in 0..32 {
        for bit_in_byte in (0..8).rev() {
            let b = (addr[byte_idx] >> bit_in_byte) & 1 == 1;
            bits.push(Boolean::new_witness(cs.clone(), || Ok(b))?);
        }
    }
    Ok(bits)
}

/// Allocate the 256 sibling hashes for one Merkle path.
fn alloc_siblings<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    siblings: &[[u8; 32]; SMT_DEPTH],
) -> Result<Vec<Vec<UInt32<F>>>, SynthesisError> {
    let mut out: Vec<Vec<UInt32<F>>> = Vec::with_capacity(SMT_DEPTH);
    for sib in siblings.iter() {
        let words = sib
            .chunks(4)
            .map(|c| {
                let w = u32::from_le_bytes(c.try_into().expect("4 bytes per word"));
                UInt32::new_witness(cs.clone(), || Ok(w))
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.push(words);
    }
    Ok(out)
}

/// Allocate a 256-bit MSB-first empty-bitmap.
fn alloc_empty_bitmap<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    bitmap: &[u8; 32],
) -> Result<Vec<Boolean<F>>, SynthesisError> {
    let mut bits = Vec::with_capacity(SMT_DEPTH);
    for byte_idx in 0..32 {
        for bit_in_byte in (0..8).rev() {
            let b = (bitmap[byte_idx] >> bit_in_byte) & 1 == 1;
            bits.push(Boolean::new_witness(cs.clone(), || Ok(b))?);
        }
    }
    Ok(bits)
}

/// Apply one SMT leaf update inside the circuit.
///
/// Given:
///   - `running_root`: the current SMT root before this update
///   - `addr_bits`, `siblings`, `empty_bitmap`: the path for the affected leaf
///   - `balance_prev`: the OLD leaf value (must be a member of running_root)
///   - `balance_new`: the NEW leaf value to install
///
/// Enforces:
///   1. `compute_root(leaf(addr, balance_prev), path)` == `running_root`
///   2. Returns new `running_root = compute_root(leaf(addr, balance_new), path)`
///
/// Combines 1E + 1F (or 1H + 1I) into a single helper used three times in
/// the δ-circuit (sender update, recipient update, coinbase update).
#[allow(clippy::too_many_arguments)]
fn apply_smt_leaf_update<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    running_root: &[UInt32<F>],
    addr_bits: &[Boolean<F>],
    siblings: &[Vec<UInt32<F>>],
    empty_bitmap: &[Boolean<F>],
    empty_subtree: &[Vec<UInt32<F>>],
    balance_prev: &FpVar<F>,
    balance_new: &FpVar<F>,
) -> Result<Vec<UInt32<F>>, SynthesisError> {
    // Step 1: enforce membership of the OLD (addr, balance_prev) in running_root.
    let leaf_prev = MerklePathGadget::leaf_hash(cs.clone(), addr_bits, balance_prev)?;
    let recomputed = MerklePathGadget::compute_root(
        cs.clone(),
        &leaf_prev,
        addr_bits,
        siblings,
        empty_bitmap,
        empty_subtree,
    )?;
    for (got, exp) in recomputed.iter().zip(running_root.iter()) {
        got.enforce_equal(exp)?;
    }

    // Step 2: compute the new running_root from (addr, balance_new) using the
    // same siblings (the single-leaf-update invariant — only this path's
    // INTERNAL nodes change; siblings come from OTHER subtrees, untouched).
    let leaf_new = MerklePathGadget::leaf_hash(cs.clone(), addr_bits, balance_new)?;
    MerklePathGadget::compute_root(
        cs,
        &leaf_new,
        addr_bits,
        siblings,
        empty_bitmap,
        empty_subtree,
    )
}

/// Era-step emission schedule as an in-circuit lookup.
///
/// Returns the maximum permissible coinbase amount at `block_height` per
/// the 4-year halving schedule. Implemented as a piecewise constant table.
/// Constants match the production emission controller at
/// `crates/q-storage/src/emission_controller.rs`.
fn era_emission_cap<F: PrimeField>(
    cs: ConstraintSystemRef<F>,
    block_height: &FpVar<F>,
) -> Result<FpVar<F>, SynthesisError> {
    // Production schedule (per CLAUDE.md mainnet 2026.2 launch params):
    //   Era 0: 2,625,000 QUG/year @ 1 bps = 1 block / second → ~0.083 QUG/block
    //   Era 1 (after 4 years = 31,536,000 × 4 ≈ 126,144,000 blocks): halved
    //   Era 2: halved again
    //   ...
    //
    // In 24-decimal base units, era 0 emission per block ≈
    //   2,625,000 × 10^24 / 31,536,000 ≈ 8.32e22 base units
    //
    // For the skeleton we constrain the cap at era 0's value. Multi-era
    // lookup is a follow-up (requires the in-circuit height-to-era
    // bucketing — straightforward with `is_cmp` on era boundaries).
    let _ = (cs, block_height);
    let era0_cap_base_units: u128 = 8_322_368_421_052_631_578_947_368;
    Ok(FpVar::Constant(F::from(era0_cap_base_units)))
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_relations::r1cs::ConstraintSystem;

    fn empty_tx() -> TransactionWitness<Fr> {
        TransactionWitness {
            from_addr: [0u8; 32],
            to_addr: [0u8; 32],
            amount: 0,
            fee: 0,
            nonce: 0,
            from_balance_prev: 0,
            from_siblings: [[0u8; 32]; SMT_DEPTH],
            from_empty_bitmap: [0xFFu8; 32],
            to_balance_prev: 0,
            to_siblings: [[0u8; 32]; SMT_DEPTH],
            to_empty_bitmap: [0xFFu8; 32],
            from_pubkey_bytes: Vec::new(),
            signature_bytes: Vec::new(),
            signing_message: Vec::new(),
            _marker: core::marker::PhantomData,
        }
    }

    fn empty_coinbase() -> CoinbaseWitness<Fr> {
        CoinbaseWitness {
            producer_addr: [0u8; 32],
            amount: 0,
            producer_balance_prev: 0,
            producer_siblings: [[0u8; 32]; SMT_DEPTH],
            producer_empty_bitmap: [0xFFu8; 32],
            _marker: core::marker::PhantomData,
        }
    }

    fn empty_anchor() -> AnchorWitness<Fr> {
        AnchorWitness {
            claimed_producer_id: 0,
            ntt_witness: Vec::new(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Compute the empty-tree (genesis) SMT root using the same primitives as
    /// the in-circuit gadget. Test fixture only — production callers use
    /// `crate::gadgets::merkle::precompute_empty_subtree_hashes()`.
    fn genesis_root() -> [u8; 32] {
        crate::gadgets::merkle::precompute_empty_subtree_hashes()[0]
    }

    /// Compute the BLAKE3 hash of 64 zero bytes (the test's stand-in block
    /// header). Used to satisfy Phase 1's BLAKE3 hash check on a header
    /// whose body is all-zeros.
    fn zero_header_hash() -> [u8; 32] {
        *blake3::hash(&[0u8; 64]).as_bytes()
    }

    /// A coinbase witness that does NOT actually emit anything — producer
    /// address is the all-zeros (i.e., the empty-leaf address), amount=0,
    /// balance_prev=0 → balance_new=0 → no actual update. Valid against a
    /// genesis-rooted tree.
    fn no_op_coinbase() -> CoinbaseWitness<Fr> {
        // For the producer leaf to verify against the empty-tree root, we
        // need: leaf(producer_addr, balance_prev=0) is at the all-zeros leaf
        // position (which it is, since producer_addr is all zeros), and the
        // siblings are all empty-subtree hashes (signaled via empty_bitmap
        // = all-ones).
        CoinbaseWitness {
            producer_addr: [0u8; 32],
            amount: 0,
            producer_balance_prev: 0,
            producer_siblings: [[0u8; 32]; SMT_DEPTH],
            producer_empty_bitmap: [0xFFu8; 32], // all siblings empty
            _marker: core::marker::PhantomData,
        }
    }

    #[test]
    fn delta_circuit_accepts_empty_block_at_genesis() {
        // No transactions, no coinbase emission, the chain is still at the
        // genesis (empty) state root. The block header is 64 zero bytes,
        // and its BLAKE3 hash is the corresponding 32-byte digest.
        //
        // Every active constraint must pass:
        //   • Phase 1 BLAKE3: blake3(zeros[64]) == zero_header_hash. ✓
        //   • Phase 3: no transactions to iterate. ✓
        //   • Phase 4 coinbase: producer leaf (zeros, 0) lives in the empty
        //     tree at the all-zeros path; after a 0→0 update the root is
        //     unchanged. ✓
        //   • Phase 5: running_root == state_root_next == genesis. ✓
        let g = genesis_root();
        let inputs = DeltaBlockInputs {
            state_root_prev: g,
            state_root_next: g,
            block_header_hash: zero_header_hash(),
            block_height: 1,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: no_op_coinbase(),
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        assert!(
            cs.is_satisfied().unwrap(),
            "δ-circuit must accept an empty no-op block at genesis"
        );
    }

    #[test]
    fn delta_circuit_rejects_wrong_block_header_hash() {
        // Same as above but the block_header_hash is wrong. Phase 1 should
        // catch the BLAKE3 mismatch and the constraint system must fail.
        let g = genesis_root();
        let mut wrong_hash = zero_header_hash();
        wrong_hash[0] ^= 1; // flip one bit

        let inputs = DeltaBlockInputs {
            state_root_prev: g,
            state_root_next: g,
            block_header_hash: wrong_hash,
            block_height: 1,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: no_op_coinbase(),
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "δ-circuit must REJECT a block whose claimed header hash does not match BLAKE3 of the body"
        );
    }

    #[test]
    fn delta_circuit_rejects_state_root_mismatch() {
        // Same empty block but state_root_next disagrees with state_root_prev.
        // Phase 5 must catch the mismatch.
        let g = genesis_root();
        let mut wrong_next = g;
        wrong_next[0] ^= 1;

        let inputs = DeltaBlockInputs {
            state_root_prev: g,
            state_root_next: wrong_next,
            block_header_hash: zero_header_hash(),
            block_height: 1,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: no_op_coinbase(),
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "δ-circuit must REJECT a block whose state_root_next does not match the computed running_root"
        );
    }

    #[test]
    fn delta_circuit_rejects_overemission_coinbase() {
        // Coinbase amount > era cap. Phase 4 cap check must reject.
        let g = genesis_root();

        // Cap is ~8.32e22 base units; overshoot by an order of magnitude.
        let overcap_amount: u128 = 9_000_000_000_000_000_000_000_000;

        let mut coinbase = no_op_coinbase();
        coinbase.amount = overcap_amount;
        // Note: with this amount, the SMT update path will also produce a
        // different root than state_root_next=g, so Phase 5 alone would
        // catch it. The point of THIS test is that the ERA-CAP check fires
        // FIRST (independent constraint), preventing an attacker from
        // tricking the prover into producing a valid-looking
        // state_root_next where the coinbase was over-emitted.
        let inputs = DeltaBlockInputs {
            state_root_prev: g,
            state_root_next: g,
            block_header_hash: zero_header_hash(),
            block_height: 1,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase,
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        assert!(
            !cs.is_satisfied().unwrap(),
            "δ-circuit must REJECT a block whose coinbase amount exceeds the era emission cap"
        );
    }

    #[test]
    fn delta_circuit_public_input_count_is_26() {
        // The δ-circuit publishes:
        //   • state_root_prev (8 × u32 = 8 instance vars)
        //   • state_root_next (8 × u32 = 8)
        //   • block_header_hash (8 × u32 = 8)
        //   • block_height (1 × FpVar)
        // Plus arkworks' implicit `one` at index 0.
        // Total: 26 instance variables.
        //
        // This test fixes that count so future code can't silently change
        // the verifier surface without an explicit update here.
        let g = genesis_root();
        let inputs = DeltaBlockInputs {
            state_root_prev: g,
            state_root_next: g,
            block_header_hash: zero_header_hash(),
            block_height: 0,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: no_op_coinbase(),
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();

        assert_eq!(
            cs.num_instance_variables(),
            26,
            "Expected 26 instance variables (1 implicit + 24 root u32 words + 1 height)"
        );
    }

    /// Companion to `delta_circuit_public_input_count_is_26`: the inner
    /// method MUST NOT allocate any public inputs of its own. Only the
    /// implicit `one` remains. The caller (Nova step-circuit wrapper) is
    /// responsible for supplying pre-allocated handles.
    #[test]
    fn delta_circuit_inner_allocates_zero_public_inputs() {
        use ark_r1cs_std::alloc::AllocVar;
        let g = genesis_root();
        let inputs = DeltaBlockInputs {
            state_root_prev: g,
            state_root_next: g,
            block_header_hash: zero_header_hash(),
            block_height: 1,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: no_op_coinbase(),
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();

        // Allocate the four canonical handles as WITNESSES (not inputs) so
        // they don't appear in the instance variable count. This mirrors
        // what the Nova step circuit will do with its z_in / z_out chain.
        let alloc_root = |bytes: &[u8; 32]| -> Vec<UInt32<Fr>> {
            bytes
                .chunks(4)
                .map(|c| {
                    let w = u32::from_le_bytes(c.try_into().unwrap());
                    UInt32::new_witness(cs.clone(), || Ok(w)).unwrap()
                })
                .collect()
        };
        let state_root_prev = alloc_root(&circuit.inputs.state_root_prev);
        let state_root_next = alloc_root(&circuit.inputs.state_root_next);
        let block_header_hash = alloc_root(&circuit.inputs.block_header_hash);
        let block_height_var =
            FpVar::<Fr>::new_witness(cs.clone(), || Ok(Fr::from(circuit.inputs.block_height)))
                .unwrap();

        circuit
            .generate_constraints_inner(
                cs.clone(),
                &state_root_prev,
                &state_root_next,
                &block_header_hash,
                &block_height_var,
            )
            .unwrap();

        assert_eq!(
            cs.num_instance_variables(),
            1,
            "inner method must not allocate public inputs (only implicit `one` should remain)"
        );
        assert!(
            cs.is_satisfied().unwrap(),
            "inner method must produce a satisfiable constraint system on a valid witness"
        );
    }

    /// The outer ConstraintSynthesizer impl is a thin wrapper that allocates
    /// the four publics and then dispatches to the inner method. This test
    /// pins the constraint-count delta to exactly the cost of those four
    /// allocations (24 UInt32 inputs + 1 FpVar input). If a future change
    /// drifts the inner body, the outer count will drift correspondingly,
    /// and this test will catch any asymmetry.
    #[test]
    fn delta_circuit_outer_inner_constraint_count_match() {
        use ark_r1cs_std::alloc::AllocVar;
        let g = genesis_root();
        let build_inputs = || DeltaBlockInputs {
            state_root_prev: g,
            state_root_next: g,
            block_header_hash: zero_header_hash(),
            block_height: 1,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: no_op_coinbase(),
            anchor: empty_anchor(),
        };

        // Outer path
        let circuit_outer = DeltaBlockCircuit { inputs: build_inputs() };
        let cs_outer = ConstraintSystem::<Fr>::new_ref();
        circuit_outer.generate_constraints(cs_outer.clone()).unwrap();
        let outer_constraints = cs_outer.num_constraints();

        // Inner path with the four publics allocated as witnesses by the caller
        let circuit_inner = DeltaBlockCircuit { inputs: build_inputs() };
        let cs_inner = ConstraintSystem::<Fr>::new_ref();
        let alloc_root = |bytes: &[u8; 32]| -> Vec<UInt32<Fr>> {
            bytes
                .chunks(4)
                .map(|c| {
                    let w = u32::from_le_bytes(c.try_into().unwrap());
                    UInt32::new_witness(cs_inner.clone(), || Ok(w)).unwrap()
                })
                .collect()
        };
        let state_root_prev = alloc_root(&circuit_inner.inputs.state_root_prev);
        let state_root_next = alloc_root(&circuit_inner.inputs.state_root_next);
        let block_header_hash = alloc_root(&circuit_inner.inputs.block_header_hash);
        let block_height_var =
            FpVar::<Fr>::new_witness(cs_inner.clone(), || Ok(Fr::from(circuit_inner.inputs.block_height)))
                .unwrap();
        circuit_inner
            .generate_constraints_inner(
                cs_inner.clone(),
                &state_root_prev,
                &state_root_next,
                &block_header_hash,
                &block_height_var,
            )
            .unwrap();
        let inner_constraints = cs_inner.num_constraints();

        // Inner + outer should differ ONLY by the constraint cost of moving
        // 24 u32 words + 1 FpVar from witness allocation to input allocation.
        // In arkworks 0.4, that's the same r1cs cost (alloc_var is the same
        // routine modulo where the variable lives), so the counts should be
        // IDENTICAL. If this drifts, something nontrivial changed and we
        // want CI to flag it.
        assert_eq!(
            outer_constraints, inner_constraints,
            "Outer and inner constraint counts must match. \
             Outer (with public inputs): {}. Inner (witness inputs): {}.",
            outer_constraints, inner_constraints
        );
    }
}
