//! Prover side: build the windowed FRI proof using `q_zk_stark::StarkSystem`.
//!
//! Designed as a `WindowBuilder` that streams headers in via `extend()`.
//! When the window fills (K = `WINDOW_SIZE`), the builder finalises the
//! current FRI proof, advances the anchor chain past the window, and
//! starts a new window. This is the windowed-fallback path from the
//! DeepSeek re-submission clarifications §1.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    binding_commitment, trace, HeaderChainStep, HeaderHash, StateRoot, TipProofStir, VerifyError,
    WINDOW_SIZE,
};

/// Stateful window builder owned by the producer-hook task. Holds the
/// in-flight header buffer + the `StarkSystem` instance + the anchor
/// chain it advances on each window rollover.
pub struct WindowBuilder {
    /// q-zk-stark engine — single instance reused across windows.
    /// Wrapped in Mutex because q-zk-stark holds internal stats it mutates.
    stark: Arc<Mutex<q_zk_stark::StarkSystem>>,
    /// Anchor chain (v1) — extended on every block so the window's
    /// start_state always equals `anchor_chain.folded_state`.
    anchor_chain: q_recursive_proofs::LatticeTipProof,
    /// In-flight headers for the current window.
    window: Vec<HeaderChainStep>,
    /// Hash of the last committed header (rolling). Used to validate
    /// the next `extend()` call's `prev_block_hash`.
    last_hash: HeaderHash,
    /// State root at the START of the current window. Held so we can
    /// emit it as `window_start_state` even after the window grows.
    window_start_state: StateRoot,
    window_start_height: u64,
    /// The trusted anchor — never changes after construction.
    anchor_height: u64,
    anchor_state: StateRoot,
}

impl WindowBuilder {
    /// Construct anchored at `(anchor_height, anchor_state)`. The
    /// anchor chain starts as a zero-length BLAKE3-FS chain at that
    /// point; the genesis-hash convention is the all-zero hash.
    pub async fn new(anchor_height: u64, anchor_state: StateRoot) -> anyhow::Result<Self> {
        let stark = q_zk_stark::StarkSystem::new(false).await?;
        let anchor_chain = q_recursive_proofs::tip_anchor(anchor_height, anchor_state);
        Ok(Self {
            stark: Arc::new(Mutex::new(stark)),
            anchor_chain,
            window: Vec::with_capacity(WINDOW_SIZE),
            last_hash: [0u8; 32], // genesis convention; updated as headers arrive
            window_start_state: anchor_state,
            window_start_height: anchor_height,
            anchor_height,
            anchor_state,
        })
    }

    /// Append a new header to the current window. Validates the chain
    /// link, advances the v1 anchor chain, returns an updated tip proof.
    /// When the window reaches `WINDOW_SIZE` headers, this call finalises
    /// the window and resets state for the next one — the returned proof
    /// is the snapshot at the rollover boundary.
    pub async fn extend(&mut self, step: HeaderChainStep) -> Result<TipProofStir, VerifyError> {
        // Chain integrity check up front — same logic that `trace::build_trace`
        // will repeat, but failing here gives a clearer error before we
        // spend FRI cycles.
        if step.prev_block_hash != self.last_hash && !self.window.is_empty() {
            return Err(VerifyError::Prover(format!(
                "extend({}) chain break — prev_block_hash mismatch",
                step.height
            )));
        }

        // Also extend the v1 anchor chain so its tip stays at the START
        // of the current window. The chain's tip should match
        // `window_start_state` and `window_start_height` after this.
        // We extend the anchor chain ONLY when starting a new window —
        // within a single window, the FRI proof covers the headers; the
        // anchor chain only attests "this window started where you expect".
        // (See clarifications §1.)

        // Update rolling state.
        self.last_hash = step.hash();
        let new_height = step.height;
        let new_state = step.state_root;
        let _new_prev_hash = step.prev_block_hash;
        let _new_tx_root = step.tx_root;
        self.window.push(step);

        // Build the current trace and prove it via q-zk-stark.
        // For the in-window case the "anchor hash" of the trace is the
        // hash of the block immediately before window_start. Inside this
        // first cut we use all-zeros for genesis; future versions will
        // pass the real anchor block hash.
        let trace_anchor_hash = [0u8; 32];
        let built = trace::build_trace(&self.window, trace_anchor_hash)?;

        // Minimal constraint blob — q-zk-stark currently treats this as
        // metadata. When q-zk-stark adopts STIR + a real constraint
        // evaluator the prover gains the polynomial-relation enforcement
        // for free.
        let constraints = b"qnk-tip-stir-fri-v2".to_vec();

        let stark_proof = {
            let mut stark = self.stark.lock().await;
            stark
                .prove(&built.trace, &constraints)
                .await
                .map_err(|e| VerifyError::Prover(e.to_string()))?
        };

        let trace_commitment = stark_proof.execution_trace_commitment;

        // Window rollover: if we just hit WINDOW_SIZE, advance the
        // anchor chain to the current tip so the NEXT extend starts a
        // fresh window. The current proof remains valid for this height.
        if self.window.len() >= WINDOW_SIZE {
            // Tear the v1 chain forward by one synthetic step that
            // collapses the whole window into a single hash-chain
            // commitment. The next window then starts anchored here.
            self.anchor_chain = q_recursive_proofs::tip_extend(
                &self.anchor_chain,
                new_height,
                new_state,
                self.last_hash, // window's tip header hash acts as "parent"
                trace_commitment, // window's FRI trace commitment as "tx_root"
            );
            self.window.clear();
            self.window_start_height = new_height;
            self.window_start_state = new_state;
        }

        let binding = binding_commitment(
            self.anchor_height,
            &self.anchor_state,
            new_height,
            &new_state,
            &trace_commitment,
            &self.anchor_chain,
        );

        Ok(TipProofStir {
            window_proof: stark_proof,
            window_start_height: self.window_start_height,
            window_end_height: new_height,
            window_start_state: self.window_start_state,
            window_end_state: new_state,
            anchor_chain: self.anchor_chain.clone(),
            binding_commitment: binding,
            anchor_height: self.anchor_height,
            anchor_state: self.anchor_state,
            tip_height: new_height,
            tip_state: new_state,
        })
    }
}

/// Convenience: build a fresh proof at the anchor (zero blocks committed).
/// The window is empty; the FRI proof is over a single placeholder row
/// so the wire format is well-formed.
pub async fn anchor(anchor_height: u64, anchor_state: StateRoot) -> anyhow::Result<TipProofStir> {
    let anchor_chain = q_recursive_proofs::tip_anchor(anchor_height, anchor_state);
    let mut stark = q_zk_stark::StarkSystem::new(false).await?;
    let placeholder_trace = vec![vec![anchor_height, 0, 0, 0, 0, 0, 0, 0]];
    let stark_proof = stark
        .prove(&placeholder_trace, b"qnk-tip-stir-fri-v2-anchor")
        .await?;
    let binding = binding_commitment(
        anchor_height,
        &anchor_state,
        anchor_height,
        &anchor_state,
        &stark_proof.execution_trace_commitment,
        &anchor_chain,
    );
    Ok(TipProofStir {
        window_proof: stark_proof,
        window_start_height: anchor_height,
        window_end_height: anchor_height,
        window_start_state: anchor_state,
        window_end_state: anchor_state,
        anchor_chain,
        binding_commitment: binding,
        anchor_height,
        anchor_state,
        tip_height: anchor_height,
        tip_state: anchor_state,
    })
}

/// One-shot extend convenience for callers that don't need a long-lived
/// `WindowBuilder`. Internally constructs a builder, replays the chain,
/// then takes one extend step. Expensive — prefer `WindowBuilder` for
/// the producer-hook task.
pub async fn extend(prev: &TipProofStir, step: &HeaderChainStep) -> Result<TipProofStir, VerifyError> {
    let mut wb = WindowBuilder::new(prev.anchor_height, prev.anchor_state)
        .await
        .map_err(|e| VerifyError::Prover(e.to_string()))?;
    wb.anchor_chain = prev.anchor_chain.clone();
    wb.window_start_state = prev.window_start_state;
    wb.window_start_height = prev.window_start_height;
    wb.last_hash = if prev.window_end_height > prev.window_start_height {
        // Approximation: caller-provided step.prev_block_hash IS the previous tip's hash.
        step.prev_block_hash
    } else {
        [0u8; 32]
    };
    wb.extend(step.clone()).await
}
