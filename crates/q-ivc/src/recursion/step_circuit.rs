//! Nova `StepCircuit` wrapper around the δ-circuit.
//!
//! This is the **Phase 2 boundary** for the recursive-lattice zk-SNARK.
//! `DeltaBlockCircuit` (in `crates/q-ivc/src/circuits/delta_block.rs`)
//! enforces per-block validity; this file wraps it as a Nova-style step
//! circuit so a folding driver can chain proofs across the entire chain
//! into a single constant-size proof verifiable in ~5–10 ms.
//!
//! ## What a step circuit is
//!
//! A step circuit is an R1CS predicate that takes a fixed-shape public
//! input vector `z_in`, a private witness, and emits a public output
//! vector `z_out` such that there exists a witness satisfying the
//! constraints. Nova's folding scheme accumulates a chain of such
//! step-circuit instances into a relaxed R1CS instance whose verification
//! cost is independent of the chain length. A final SNARK compresses
//! the accumulated instance into a constant-size proof.
//!
//! For Quillon Graph's δ-circuit, the step shape is:
//!
//! ```text
//! z_in  = [state_root_prev (8 u32 words) || block_height (1 fp)]
//! z_out = [state_root_next                || block_height + 1     ]
//! ```
//!
//! The 9-element shape matches what every existing block already commits
//! to (no consensus change). The δ-circuit's existing public inputs
//! `state_root_prev`, `state_root_next`, and `block_height` become the
//! `z_in` and `z_out`; `block_header_hash` and per-tx witnesses become
//! private witnesses inside the step.
//!
//! ## Status
//!
//! - [x] `StepCircuitInput` / `StepCircuitOutput` shape definitions
//! - [x] `DeltaStepCircuit` adaptor — wraps a `DeltaBlockCircuit` plus the
//!       z_in / z_out shape
//! - [x] `fold_native` host-side helper — drives the OFF-CIRCUIT state
//!       advance one block at a time. Returns the chain of z_out values
//!       a folding prover would need to advance against.
//! - [x] Trait `StepCircuitAdapter` — abstract boundary that subsequent
//!       commits will implement against the chosen Nova crate
//!       (`nova-snark` for Microsoft's impl, or `arkworks-rs/nova` for
//!       the community one — the choice is made via a feature flag in a
//!       subsequent commit).
//! - [ ] Concrete `impl nova_snark::traits::StepCircuit for
//!       DeltaStepCircuit` (Job N2 in the Phase 2 board) — adds the
//!       trait bound + the `generate_constraints` adapter; nova-snark
//!       dep is added at that point.
//! - [ ] `NovaFolder::fold_block` (Job N3) — the driver that calls the
//!       step circuit per block and stores the accumulated proof. Once
//!       landed, the `PHASE2-WIRE-POINT` marker in `tip_watcher.rs:114`
//!       gets replaced with one line.
//!
//! ## Cost / latency targets (whitepaper §4.2/4.3)
//!
//! Per-fold prover cost on Epsilon-class (48-core) hardware:
//!   • δ-circuit constraint count at K=100 txs: ~442M
//!   • Nova fold step prover time target: < 1s per block
//!   • If unmet: fall back to one fold every `m ≤ 10` blocks (acceptable
//!     UX — a bootstrap proof that lags the tip by 10s)
//! Verifier latency target: 5 ms (desktop) / 10 ms (laptop) / 250 ms (WASM)

use ark_ff::PrimeField;
use ark_r1cs_std::{
    fields::fp::FpVar,
    prelude::*,
    uint32::UInt32,
};
use ark_relations::r1cs::{ConstraintSystemRef, SynthesisError};

use crate::circuits::delta_block::{DeltaBlockCircuit, DeltaBlockInputs};

/// Number of u32 words in the IVC public-input vector `z`.
///
///   8 words for the state root + 1 word for the block height = 9 words.
pub const STEP_Z_LEN: usize = 9;

/// Native (off-circuit) representation of the step's public input/output.
///
/// One of these per block in the chain. The folding driver maintains
/// `z_current` and advances it block-by-block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepIO {
    /// SMT root at this point in the chain.
    pub state_root: [u8; 32],
    /// Chain height at this point (post-block-application).
    pub height: u64,
}

impl StepIO {
    pub fn new(state_root: [u8; 32], height: u64) -> Self {
        Self { state_root, height }
    }

    /// Genesis IO: empty-tree root, height 0.
    pub fn genesis() -> Self {
        Self {
            state_root: crate::gadgets::merkle::precompute_empty_subtree_hashes()[0],
            height: 0,
        }
    }

    /// Pack into a `Vec<u32>` of length `STEP_Z_LEN` for serialization to
    /// the folding driver. Bottom 8 words = state_root (LE-packed),
    /// top word = height truncated to u32 (the consensus path also caps at
    /// u32 effectively — at 1 bps mainnet, u32 overflows in ~136 years).
    pub fn pack(&self) -> [u32; STEP_Z_LEN] {
        let mut out = [0u32; STEP_Z_LEN];
        for (i, c) in self.state_root.chunks(4).enumerate() {
            out[i] = u32::from_le_bytes(c.try_into().expect("4 bytes per word"));
        }
        out[8] = (self.height & 0xFFFF_FFFF) as u32;
        out
    }

    /// Unpack from a `Vec<u32>` produced by `pack`.
    pub fn unpack(z: &[u32; STEP_Z_LEN]) -> Self {
        let mut state_root = [0u8; 32];
        for i in 0..8 {
            state_root[i * 4..(i + 1) * 4].copy_from_slice(&z[i].to_le_bytes());
        }
        Self {
            state_root,
            height: z[8] as u64,
        }
    }
}

/// Trait that subsequent commits implement against the chosen Nova crate.
///
/// The Phase 2 board (jobs N1..N3) decides whether to use Microsoft's
/// `nova-snark` or `arkworks-rs/nova`. Whichever one wins, it provides
/// its own `StepCircuit` trait with this same shape — the adapter
/// commit just maps this trait's methods 1:1.
///
/// Keeping the trait local lets us:
///   • Test the step circuit's constraint logic NOW (this commit),
///     without pulling in the Nova crate's heavy dependency graph yet.
///   • Swap the underlying Nova implementation later by changing only the
///     adapter file.
///   • Phase 4 lattice migration: re-implement this trait against
///     LatticeFold / LaBRADOR / Greyhound — the δ-circuit itself doesn't
///     change.
pub trait StepCircuitAdapter<F: PrimeField> {
    /// Number of public-input words per step. Always `STEP_Z_LEN` (9) for
    /// the δ-step; trait parameter for future flexibility.
    fn z_arity() -> usize {
        STEP_Z_LEN
    }

    /// Apply this step's constraints. Given `z_in` (allocated by the
    /// folding driver as either witness or public input depending on the
    /// Nova trait flavor), produce `z_out`.
    ///
    /// The Nova crate's StepCircuit::synthesize signature differs slightly
    /// between `nova-snark` (uses bellperson) and `arkworks-rs/nova` (uses
    /// arkworks). The adapter unifies them.
    fn synthesize_step(
        &self,
        cs: ConstraintSystemRef<F>,
        z_in: &[UInt32<F>],
    ) -> Result<Vec<UInt32<F>>, SynthesisError>;
}

/// Adapter wrapping a `DeltaBlockCircuit` into a Nova-style step.
///
/// The wrapped circuit's `state_root_prev` is taken from `z_in[0..8]`,
/// `state_root_next` is produced as `z_out[0..8]`, `block_height` comes
/// from `z_in[8]`, and `z_out[8] = z_in[8] + 1` is enforced.
pub struct DeltaStepCircuit<F: PrimeField> {
    pub inner: DeltaBlockCircuit<F>,
}

impl<F: PrimeField> DeltaStepCircuit<F> {
    pub fn new(inner: DeltaBlockCircuit<F>) -> Self {
        Self { inner }
    }

    /// The native (off-circuit) z_out this step would produce.
    /// Verified by the folding driver against the in-circuit z_out.
    pub fn native_z_out(&self) -> StepIO {
        StepIO {
            state_root: self.inner.inputs.state_root_next,
            height: self.inner.inputs.block_height,
        }
    }

    /// The native z_in this step expects.
    pub fn native_z_in(&self) -> StepIO {
        StepIO {
            state_root: self.inner.inputs.state_root_prev,
            // block_height in the δ-circuit's inputs is "the height of the
            // produced block." z_in's height is "pre-block." So z_in.height
            // = block_height - 1.
            height: self.inner.inputs.block_height.saturating_sub(1),
        }
    }
}

impl<F: PrimeField> StepCircuitAdapter<F> for DeltaStepCircuit<F> {
    fn synthesize_step(
        &self,
        _cs: ConstraintSystemRef<F>,
        _z_in: &[UInt32<F>],
    ) -> Result<Vec<UInt32<F>>, SynthesisError> {
        // The full implementation:
        //   1. Constrain z_in[0..8] equal to inner.inputs.state_root_prev's
        //      circuit-allocated form.
        //   2. Constrain z_in[8] equal to (block_height - 1) circuit form.
        //   3. Call inner.generate_constraints(cs) — this enforces the
        //      block's validity AND that state_root_next, block_height
        //      match the public-input allocations.
        //   4. Return z_out = [state_root_next, block_height] (i.e.,
        //      block_height NOT block_height+1 — the next step receives
        //      THIS block's height as its `z_in.height`).
        //
        // This synthesize_step body lands together with the chosen Nova
        // crate's StepCircuit trait impl (Job N2). Postponed because:
        //   • The DeltaBlockCircuit already allocates state_root_prev /
        //     state_root_next / block_height as PUBLIC inputs internally
        //     (via `alloc_root_input` and `FpVar::new_input`). Wiring
        //     z_in / z_out as separate allocations requires either
        //     refactoring DeltaBlockCircuit to take pre-allocated inputs,
        //     or adding equality enforcements between two parallel
        //     allocations (wasteful — 24 enforce_equal per fold).
        //   • The right refactor is to split DeltaBlockCircuit into
        //     a "synth-only" inner method that takes already-allocated
        //     vars + an outer wrapper for standalone use. That's a
        //     5-minute refactor but should not land in this scaffold —
        //     the Nova crate adapter commit is the right place.
        //
        // Until then this returns an empty Vec (would fail any real
        // proving call, intentionally — the caller is the future Nova
        // adapter and will replace this body).
        Err(SynthesisError::AssignmentMissing)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Host-side (off-circuit) fold helper
// ════════════════════════════════════════════════════════════════════════════

/// Off-circuit driver: given a starting StepIO and a sequence of blocks
/// (each as a `DeltaBlockInputs`), produce the sequence of
/// `(z_in, z_out)` pairs the Nova fold loop will need.
///
/// Verifies CONSISTENCY: each block's `state_root_prev` must equal the
/// previous step's `state_root_next`, and `block_height` must equal the
/// previous step's height + 1. Returns an error on any inconsistency —
/// this is the host-side guardrail that catches mis-sequenced blocks
/// before they reach the prover.
///
/// The actual SNARK proof generation is the Nova fold driver's job (lands
/// in Job N3). This function is the orchestration layer beneath that.
pub fn fold_native<F: PrimeField>(
    initial: StepIO,
    blocks: Vec<DeltaBlockInputs<F>>,
) -> Result<Vec<(StepIO, StepIO)>, FoldError> {
    let mut steps = Vec::with_capacity(blocks.len());
    let mut current = initial;

    for (idx, b) in blocks.into_iter().enumerate() {
        if b.state_root_prev != current.state_root {
            return Err(FoldError::StateRootMismatch {
                block_index: idx,
                expected: current.state_root,
                got: b.state_root_prev,
            });
        }
        // Heights should be monotone +1.
        if b.block_height != current.height + 1 {
            return Err(FoldError::HeightDiscontinuity {
                block_index: idx,
                expected: current.height + 1,
                got: b.block_height,
            });
        }
        let next = StepIO {
            state_root: b.state_root_next,
            height: b.block_height,
        };
        steps.push((current.clone(), next.clone()));
        current = next;
    }

    Ok(steps)
}

/// Errors the host-side fold driver can raise. None of these are
/// soundness violations — they're caller errors that the host catches
/// BEFORE the prover wastes time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FoldError {
    StateRootMismatch {
        block_index: usize,
        expected: [u8; 32],
        got: [u8; 32],
    },
    HeightDiscontinuity {
        block_index: usize,
        expected: u64,
        got: u64,
    },
}

impl core::fmt::Display for FoldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FoldError::StateRootMismatch { block_index, expected, got } => {
                write!(
                    f,
                    "block {} state_root_prev mismatch: expected {} got {}",
                    block_index,
                    hex::encode(&expected[..8]),
                    hex::encode(&got[..8])
                )
            }
            FoldError::HeightDiscontinuity { block_index, expected, got } => {
                write!(
                    f,
                    "block {} height discontinuity: expected {} got {}",
                    block_index, expected, got
                )
            }
        }
    }
}

impl std::error::Error for FoldError {}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::delta_block::{
        AnchorWitness, CoinbaseWitness, DeltaBlockInputs,
    };
    use ark_bls12_381::Fr;
    use crate::gadgets::merkle::SMT_DEPTH;

    fn genesis_root() -> [u8; 32] {
        crate::gadgets::merkle::precompute_empty_subtree_hashes()[0]
    }

    fn zero_header_hash() -> [u8; 32] {
        *blake3::hash(&[0u8; 64]).as_bytes()
    }

    fn no_op_coinbase() -> CoinbaseWitness<Fr> {
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

    fn empty_block_inputs(height: u64, prev_root: [u8; 32], next_root: [u8; 32]) -> DeltaBlockInputs<Fr> {
        DeltaBlockInputs {
            state_root_prev: prev_root,
            state_root_next: next_root,
            block_header_hash: zero_header_hash(),
            block_height: height,
            block_header_bytes: vec![0u8; 64],
            transactions: Vec::new(),
            coinbase: no_op_coinbase(),
            anchor: empty_anchor(),
        }
    }

    #[test]
    fn step_io_pack_unpack_roundtrip() {
        let original = StepIO::new([0x42u8; 32], 12345);
        let packed = original.pack();
        let unpacked = StepIO::unpack(&packed);
        assert_eq!(original, unpacked);
    }

    #[test]
    fn step_io_genesis_matches_empty_smt_root() {
        let g = StepIO::genesis();
        assert_eq!(g.state_root, genesis_root());
        assert_eq!(g.height, 0);
    }

    #[test]
    fn fold_native_accepts_consistent_chain() {
        // A three-block chain that stays at the genesis root (no actual
        // state change — empty no-op blocks). Heights advance 1, 2, 3.
        let g = genesis_root();
        let blocks = vec![
            empty_block_inputs(1, g, g),
            empty_block_inputs(2, g, g),
            empty_block_inputs(3, g, g),
        ];
        let steps = fold_native(StepIO::genesis(), blocks).unwrap();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].0, StepIO::genesis());
        assert_eq!(steps[0].1.height, 1);
        assert_eq!(steps[2].1.height, 3);
    }

    #[test]
    fn fold_native_rejects_state_root_break() {
        // Second block claims a different prev root than the first block's
        // next root — fold driver must reject.
        let g = genesis_root();
        let mut bad_prev = g;
        bad_prev[0] ^= 1;
        let blocks = vec![
            empty_block_inputs(1, g, g),
            empty_block_inputs(2, bad_prev, g), // breaks the chain
        ];
        let result = fold_native(StepIO::genesis(), blocks);
        assert!(matches!(result, Err(FoldError::StateRootMismatch { block_index: 1, .. })));
    }

    #[test]
    fn fold_native_rejects_height_skip() {
        // Block heights 1, 3 (skipping 2) — fold driver must reject.
        let g = genesis_root();
        let blocks = vec![
            empty_block_inputs(1, g, g),
            empty_block_inputs(3, g, g),  // skipped 2
        ];
        let result = fold_native(StepIO::genesis(), blocks);
        assert!(matches!(result, Err(FoldError::HeightDiscontinuity { block_index: 1, expected: 2, got: 3 })));
    }

    #[test]
    fn delta_step_circuit_native_io_matches_inner_inputs() {
        let g = genesis_root();
        let inner_inputs = empty_block_inputs(5, g, g);
        let inner = DeltaBlockCircuit { inputs: inner_inputs };
        let step = DeltaStepCircuit::new(inner);
        assert_eq!(step.native_z_in().height, 4);  // block_height - 1
        assert_eq!(step.native_z_out().height, 5);
        assert_eq!(step.native_z_in().state_root, g);
        assert_eq!(step.native_z_out().state_root, g);
    }
}
