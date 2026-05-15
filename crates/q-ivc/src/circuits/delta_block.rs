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
/// Public roots and signatures are private witnesses (the verifier sees only
/// the state roots and block-header hash). Merkle paths are also witnesses.
#[derive(Clone)]
pub struct TransactionWitness<F: PrimeField> {
    /// Sender address (32 bytes, MSB-first bit decomposition supplied separately).
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
    /// Sibling hashes along the path to `from_addr` in `state_root_prev`.
    pub from_siblings_prev: [[u8; 32]; SMT_DEPTH],
    /// Empty-bitmap for the prev path.
    pub from_empty_bitmap_prev: [u8; 32],
    /// Sibling hashes along the path to `from_addr` in `state_root_next`.
    pub from_siblings_next: [[u8; 32]; SMT_DEPTH],
    /// Empty-bitmap for the next path.
    pub from_empty_bitmap_next: [u8; 32],

    // ─── Recipient path witnesses ─────────────────────────────────────
    /// Recipient's balance BEFORE this transaction (`to_balance_prev`).
    /// May be zero (recipient previously unfunded).
    pub to_balance_prev: u128,
    pub to_siblings_prev: [[u8; 32]; SMT_DEPTH],
    pub to_empty_bitmap_prev: [u8; 32],
    pub to_siblings_next: [[u8; 32]; SMT_DEPTH],
    pub to_empty_bitmap_next: [u8; 32],

    // ─── Signature witness ────────────────────────────────────────────
    /// Sender's Dilithium5 public key (encoded per FIPS 204).
    pub from_pubkey_bytes: Vec<u8>,
    /// Dilithium5 signature over the transaction's signing-message.
    pub signature_bytes: Vec<u8>,
    /// The message that was signed (canonical bytes of the tx struct).
    pub signing_message: Vec<u8>,
}

/// Coinbase witness — the emission transaction at the top of every block.
#[derive(Clone)]
pub struct CoinbaseWitness<F: PrimeField> {
    /// Producer's wallet address (32 bytes).
    pub producer_addr: [u8; 32],
    /// Coinbase amount (must be ≤ era-scheduled emission).
    pub amount: u128,
    /// Producer's balance before this block (witness).
    pub producer_balance_prev: u128,
    /// Sibling hashes along the path to `producer_addr` in `state_root_prev`.
    pub producer_siblings_prev: [[u8; 32]; SMT_DEPTH],
    pub producer_empty_bitmap_prev: [u8; 32],
    /// Sibling hashes in `state_root_next`.
    pub producer_siblings_next: [[u8; 32]; SMT_DEPTH],
    pub producer_empty_bitmap_next: [u8; 32],
    // PhantomData kept implicit; F is on the type for future signed-amount fields.
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
        // ║  PHASE 0 — allocate public inputs and the empty-subtree constants ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // The verifier sees: state_root_prev, state_root_next, block_header_hash,
        // block_height. Everything else is private witness.

        let state_root_prev = alloc_root_input(cs.clone(), &self.inputs.state_root_prev)?;
        let state_root_next = alloc_root_input(cs.clone(), &self.inputs.state_root_next)?;
        let block_header_hash =
            alloc_root_input(cs.clone(), &self.inputs.block_header_hash)?;
        let block_height_var = FpVar::new_input(cs.clone(), || {
            Ok(F::from(self.inputs.block_height))
        })?;

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
        // ║  PHASE 1 — block header BLAKE3 hash check                         ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // Enforce: block_header_hash = BLAKE3(block_header_bytes).
        // The header is 64 bytes (single BLAKE3 block) per the consensus spec.
        //
        // TODO(delta-circuit-PHASE-1A): Call `Blake3Gadget::verify_hash` with
        // the 16-u32-word allocation of `self.inputs.block_header_bytes` and
        // the 8-u32-word allocation of `block_header_hash`. Reuse the helper
        // pattern in tests/blake3_gadget_compiles_with_real_input.
        //
        // Constraint cost: ~50K.
        let _ = block_header_hash; // suppress unused-variable until wired
        let _ = self.inputs.block_header_bytes; // ditto

        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 2 — anchor-election NTT verification                       ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // Enforce: the producer claimed in the block header is the legitimate
        // anchor for this round under the NTT-based randomness beacon.
        //
        // TODO(delta-circuit-PHASE-1B): Call into NttVerifierGadget. The
        // existing gadget already supports the FIPS 204 negacyclic convention;
        // we need the wiring that consumes `self.inputs.anchor.ntt_witness` and
        // produces a Boolean that must be `true`.
        //
        // Constraint cost: ~50M.

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

        let mut running_root = state_root_prev.clone();

        for (tx_idx, tx) in self.inputs.transactions.iter().enumerate() {
            // ──── Phase 3a: Dilithium5 signature verification ─────────
            //
            // TODO(delta-circuit-PHASE-1C): Call
            // `DilithiumVerifierGadget::verify_structured(cs, pubkey, msg, sig)`.
            // The pubkey/sig/msg come from `tx.from_pubkey_bytes`,
            // `tx.signature_bytes`, `tx.signing_message`.
            //
            // Constraint cost: ~1.5M per tx.
            let _ = tx_idx;
            let _ = &tx.from_pubkey_bytes;
            let _ = &tx.signature_bytes;
            let _ = &tx.signing_message;

            // ──── Phase 3b: amount + fee range check ──────────────────
            //
            // TODO(delta-circuit-PHASE-1D): Wrap `dilithium::enforce_norm_bound`
            // (or a dedicated u128 range gadget) to assert both `amount` and
            // `fee` fit in u128 and that `amount + fee` does not overflow.
            //
            // Constraint cost: ~10K per tx.

            // ──── Phase 3c: sender's prev-state Merkle membership ──────
            //
            // Prove (tx.from_addr, tx.from_balance_prev) ∈ running_root.
            //
            // TODO(delta-circuit-PHASE-1E): Build `AllocatedMerkleWitness`
            // from `tx.from_addr`, `tx.from_balance_prev`,
            // `tx.from_siblings_prev`, `tx.from_empty_bitmap_prev`,
            // `running_root` and call `MerklePathGadget::enforce_membership`.
            //
            // Constraint cost: 256 × ~90K = ~23M per Merkle path.

            // ──── Phase 3d: sender's NEXT-state Merkle update ──────────
            //
            // After this iteration, running_root becomes the SMT root
            // committing to the sender's updated balance:
            //   from_balance_new = tx.from_balance_prev - tx.amount - tx.fee
            //
            // We use `MerklePathGadget::compute_root` (NOT enforce — we WANT
            // the new root, we don't have it yet) with the same address but
            // the new balance and the prover-supplied `siblings_next` /
            // `empty_bitmap_next`. The result is the new running_root.
            //
            // TODO(delta-circuit-PHASE-1F): Implement the compute_root call
            // and reassign `running_root` to its output.
            //
            // Constraint cost: 256 × ~90K = ~23M per Merkle path.

            // ──── Phase 3e: balance sufficiency check ──────────────────
            //
            // Enforce: from_balance_prev ≥ amount + fee.
            //
            // TODO(delta-circuit-PHASE-1G): Range check on
            // `from_balance_prev - amount - fee ≥ 0`. Uses the same gadget
            // as 3b but with the subtraction result.
            //
            // Constraint cost: ~10K per tx.

            // ──── Phase 3f: recipient's prev-state Merkle membership ───
            //
            // Same pattern as 3c but for `tx.to_addr`. After 3d's update we
            // have a new running_root; the to-side path must be a member of
            // THAT root (because the sender's balance update has already been
            // applied — the SMT state is post-sender, pre-recipient).
            //
            // TODO(delta-circuit-PHASE-1H): `enforce_membership` against the
            // current `running_root`.
            //
            // Constraint cost: ~23M per path.

            // ──── Phase 3g: recipient's NEXT-state Merkle update ───────
            //
            // to_balance_new = tx.to_balance_prev + tx.amount.
            //
            // TODO(delta-circuit-PHASE-1I): `compute_root` and reassign
            // `running_root`.
            //
            // Constraint cost: ~23M per path.
        }

        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 4 — coinbase emission                                      ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // Producer receives `coinbase.amount` QUG into `coinbase.producer_addr`.
        //
        // Constraints:
        //   • coinbase.amount ≤ R(block_height) where R is the era-step schedule
        //     (piecewise lookup over 4-year halving boundaries).
        //   • Merkle membership of (producer_addr, producer_balance_prev) in
        //     the current running_root.
        //   • Merkle membership of (producer_addr,
        //     producer_balance_prev + coinbase.amount) in the new running_root.
        //
        // After this phase, `running_root` should equal `state_root_next`.
        //
        // TODO(delta-circuit-PHASE-1J): Implement the coinbase path. The
        // era-step lookup table can be a `match` on `(block_height /
        // BLOCKS_PER_ERA)` returning a constant FpVar.
        //
        // Constraint cost: ~5M total (one Merkle path pair + range check).

        // ╔═══════════════════════════════════════════════════════════════════╗
        // ║  PHASE 5 — final state-root equality enforcement                  ║
        // ╚═══════════════════════════════════════════════════════════════════╝
        //
        // The running_root after all txs + coinbase MUST equal the public
        // `state_root_next`.
        //
        // TODO(delta-circuit-PHASE-1K): for each of the 8 u32 words, call
        // `running_root[i].enforce_equal(&state_root_next[i])`.

        let _ = (state_root_next, running_root, empty_subtree, block_height_var);
        // ─── Until the TODOs are filled in, the circuit is vacuously
        // satisfied (no constraints applied). This is INTENTIONAL: the
        // skeleton must compile and the test harness must be able to call
        // `generate_constraints` so subsequent commits can land each TODO
        // body incrementally and validate via diff. Calling the circuit on
        // production data BEFORE all TODOs are filled would erroneously
        // accept any block — DO NOT use this circuit for consensus until
        // every TODO is removed.

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
            from_siblings_prev: [[0u8; 32]; SMT_DEPTH],
            from_empty_bitmap_prev: [0xFFu8; 32],
            from_siblings_next: [[0u8; 32]; SMT_DEPTH],
            from_empty_bitmap_next: [0xFFu8; 32],
            to_balance_prev: 0,
            to_siblings_prev: [[0u8; 32]; SMT_DEPTH],
            to_empty_bitmap_prev: [0xFFu8; 32],
            to_siblings_next: [[0u8; 32]; SMT_DEPTH],
            to_empty_bitmap_next: [0xFFu8; 32],
            from_pubkey_bytes: Vec::new(),
            signature_bytes: Vec::new(),
            signing_message: Vec::new(),
        }
    }

    fn empty_coinbase() -> CoinbaseWitness<Fr> {
        CoinbaseWitness {
            producer_addr: [0u8; 32],
            amount: 0,
            producer_balance_prev: 0,
            producer_siblings_prev: [[0u8; 32]; SMT_DEPTH],
            producer_empty_bitmap_prev: [0xFFu8; 32],
            producer_siblings_next: [[0u8; 32]; SMT_DEPTH],
            producer_empty_bitmap_next: [0xFFu8; 32],
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

    #[test]
    fn skeleton_circuit_compiles_with_zero_txs() {
        // Empty block (no transactions, no coinbase output, no signatures).
        // The skeleton's TODO bodies are vacuously satisfied — every phase
        // is a no-op until the production logic lands. This test confirms
        // the skeleton's ConstraintSynthesizer impl compiles and runs to
        // completion without panicking.
        let inputs = DeltaBlockInputs {
            state_root_prev: [0x11u8; 32],
            state_root_next: [0x11u8; 32], // same root — no transitions
            block_header_hash: [0x22u8; 32],
            block_height: 1,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: empty_coinbase(),
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        assert!(
            cs.is_satisfied().unwrap(),
            "Skeleton circuit should be vacuously satisfied"
        );
    }

    #[test]
    fn skeleton_circuit_compiles_with_one_dummy_tx() {
        // Single transaction with zero-everything witnesses. Same vacuity
        // applies — until the TODOs are filled, the constraints are inert.
        let inputs = DeltaBlockInputs {
            state_root_prev: [0x33u8; 32],
            state_root_next: [0x33u8; 32],
            block_header_hash: [0x44u8; 32],
            block_height: 100,
            block_header_bytes: vec![0u8; 64],
            transactions: vec![empty_tx()],
            coinbase: empty_coinbase(),
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap());
    }

    #[test]
    fn skeleton_circuit_public_input_count_is_correct() {
        // The δ-circuit publishes:
        //   • state_root_prev (8 × u32 = 8 inputs)
        //   • state_root_next (8 × u32 = 8 inputs)
        //   • block_header_hash (8 × u32 = 8 inputs)
        //   • block_height (1 × FpVar = 1 input)
        // Total: 25 public inputs (the verifier supplies these).
        //
        // This test fixes that count so future code can't silently change
        // the verifier surface without an explicit update here.
        let inputs = DeltaBlockInputs {
            state_root_prev: [0u8; 32],
            state_root_next: [0u8; 32],
            block_header_hash: [0u8; 32],
            block_height: 0,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: empty_coinbase(),
            anchor: empty_anchor(),
        };
        let circuit = DeltaBlockCircuit { inputs };
        let cs = ConstraintSystem::<Fr>::new_ref();
        circuit.generate_constraints(cs.clone()).unwrap();

        let public_inputs = cs.num_instance_variables();
        // arkworks counts an implicit "one" instance at index 0, so the
        // public-input count we expect is 1 (implicit) + 24 (u32 roots) +
        // 1 (block_height) = 26.
        assert_eq!(
            public_inputs, 26,
            "Expected 26 instance variables (1 implicit + 24 root u32 words + 1 height), got {}",
            public_inputs
        );
    }
}
