//! `q-tip-proof-stir` — windowed FRI tip proof for Quillon.
//!
//! ## Layering
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  TipProofStir (wire object, ~3-5 KB)                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   ┌─ binding_commitment ──────────────────────────────────────┐  │
//! │   │ BLAKE3-keyed(anchor||tip||window_root||anchor_chain_bytes)│  │
//! │   └────────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! │   ┌─ anchor_chain (q_recursive_proofs::LatticeTipProof, 176 B) │  │
//! │   │   v1 BLAKE3-FS hash-chain : anchor → window_start          │  │
//! │   └────────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! │   ┌─ window_fri_proof (q_zk_stark::StarkProof, ~2-4 KB) ──────┐  │
//! │   │   Real FRI over the trace of the K headers in the window. │  │
//! │   │   Header-chain hash links are verified at commit time     │  │
//! │   │   (prover refuses to build the trace if any prev_hash     │  │
//! │   │   link breaks), and the trace_commitment in the FRI proof │  │
//! │   │   binds that committed chain.                              │  │
//! │   └────────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## What this proves
//!
//! Given a verifier-known `(anchor_height, anchor_state)`, the proof attests:
//!
//! 1. The v1 BLAKE3-FS transcript chain links the anchor to
//!    `window_start_height / window_start_state` (DeepSeek-§0 fixed).
//! 2. The prover possesses a sequence of `K ≤ 65,536` block headers
//!    starting at `window_start_height + 1` that form a valid hash chain
//!    (each header's `prev_block_hash == hash(prev_header)`), and whose
//!    final header has the claimed `tip_state`. This is committed via
//!    `q_zk_stark::StarkSystem`'s real FRI commitment to the trace.
//! 3. The outer binding commitment ties (anchor, tip, window_root,
//!    anchor_chain) together — same DeepSeek-§0 fix pattern.
//!
//! ## What it does NOT prove
//!
//! - That the headers in the window are part of the canonical chain
//!   (an attacker with full prover access can build a valid hash-chain
//!   from any genesis-anchored prefix). For consensus-bound proof, we
//!   add BFT signature folding in v3.
//! - Anything about transaction execution, state-root correctness, or
//!   double-spends. Those live in the consensus layer (BFT signatures
//!   already cover them at the block level).
//!
//! ## Honest naming
//!
//! `proof_version = "tip-stir-fri-v2"`. The underlying FRI is from
//! `q-zk-stark`, which is a real Reed-Solomon proximity proof but does
//! NOT yet have the STIR query-count reduction (Arnon-Chiesa 2024 —
//! integration tracked in a separate ticket). When `q-zk-stark` adopts
//! STIR proper, this version stays the same; only the inner proof
//! shrinks. The wire-format and verifier API are stable.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Hash32 = [u8; 32];
pub type StateRoot = Hash32;
pub type HeaderHash = Hash32;

/// Minimal block-header shape committed to inside the FRI trace.
/// Byte-for-byte compatible with `q_types::block::QBlockHeader` under
/// bincode — the verifier must reproduce the same hash.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HeaderChainStep {
    pub height: u64,
    pub prev_block_hash: HeaderHash,
    pub state_root: StateRoot,
    pub tx_root: Hash32,
    pub timestamp: u64,
    pub producer_id: u8,
}

impl HeaderChainStep {
    /// Canonical header hash, used for both Merkle-style commitment and
    /// hash-chain integrity at commit time.
    pub fn hash(&self) -> HeaderHash {
        let mut h = blake3::Hasher::new();
        h.update(b"qnk-header-v1");
        h.update(&self.height.to_le_bytes());
        h.update(&self.prev_block_hash);
        h.update(&self.state_root);
        h.update(&self.tx_root);
        h.update(&self.timestamp.to_le_bytes());
        h.update(&[self.producer_id]);
        *h.finalize().as_bytes()
    }
}

/// Window length cap. 2¹⁶ for a clean log₂ FRI domain shape and ~18 hr
/// of Quillon chain (1 bps).
pub const WINDOW_SIZE: usize = 65_536;

/// The wire proof. Bincode-serialised size: ~3-5 KB depending on
/// q-zk-stark's FRI parameters (currently ~2 KB FRI body + 176 B anchor
/// chain + ~200 B framing = ~2.5 KB observed in tests).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TipProofStir {
    /// Inner FRI proof from `q-zk-stark::StarkSystem`. Contains
    /// `execution_trace_commitment`, `fri_proof: Vec<u8>`, public inputs.
    pub window_proof: q_zk_stark::StarkProof,
    /// Inclusive height range covered by the window.
    pub window_start_height: u64,
    pub window_end_height: u64, // == tip_height
    pub window_start_state: StateRoot,
    pub window_end_state: StateRoot, // == tip_state

    /// v1 BLAKE3-FS chain linking the trusted anchor (genesis or earlier
    /// window's tip) to `window_start`. 176 bytes.
    pub anchor_chain: q_recursive_proofs::LatticeTipProof,

    /// Outer binding commitment over public inputs + window trace
    /// commitment + anchor chain. Closes DeepSeek-§0 anchor-swap forgery.
    pub binding_commitment: Hash32,

    /// Public inputs (mirrored from the request so the proof is
    /// self-contained on the wire).
    pub anchor_height: u64,
    pub anchor_state: StateRoot,
    pub tip_height: u64,
    pub tip_state: StateRoot,
}

impl TipProofStir {
    pub const VERSION: &'static str = "tip-stir-fri-v2";
    pub fn version_str() -> &'static str { Self::VERSION }
}

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("anchor mismatch — proof's anchor does not match verifier's trust root")]
    AnchorMismatch,
    #[error("anchor-chain (v1 BLAKE3-FS) failed to verify")]
    AnchorChainInvalid,
    #[error("window start does not equal anchor-chain tip — provenance gap")]
    WindowStartGap,
    #[error("binding commitment mismatch — public-input swap or transcript forge")]
    BindingMismatch,
    #[error("inner FRI proof rejected by q_zk_stark verifier")]
    FriRejected,
    #[error("window range invalid (end != tip_height or end < start)")]
    WindowRangeInvalid,
    #[error("q_zk_stark prover error: {0}")]
    Prover(String),
}

pub mod prover;
pub mod verifier;
pub mod trace;

pub use prover::{anchor, extend, WindowBuilder};
pub use verifier::verify;

// ─── Helpers shared by prover and verifier ───────────────────────────────────

/// Compute the binding commitment. Both prover and verifier call this.
pub(crate) fn binding_commitment(
    anchor_height: u64,
    anchor_state: &StateRoot,
    tip_height: u64,
    tip_state: &StateRoot,
    window_trace_commitment: &Hash32,
    anchor_chain: &q_recursive_proofs::LatticeTipProof,
) -> Hash32 {
    let key = blake3::hash(b"qnk-tip-stir-binding-v1");
    let mut h = blake3::Hasher::new_keyed(key.as_bytes());
    h.update(&[0x01]); // version byte (DeepSeek §2)
    h.update(&anchor_height.to_le_bytes());
    h.update(anchor_state);
    h.update(&tip_height.to_le_bytes());
    h.update(tip_state);
    h.update(window_trace_commitment);
    // Bincode-serialize the anchor chain. `LatticeTipProof` uses
    // `#[serde(default)]` on additive fields so this is stable forward.
    let chain_bytes = bincode::serialize(anchor_chain).unwrap_or_default();
    h.update(&chain_bytes);
    *h.finalize().as_bytes()
}
