/// P2P bridge — wires the Dark Knight bot to q-narwhal-core's Bracha RBB.
///
/// How it works:
///   - Each Quillon node running the trading bot instantiates a `BotP2pBridge`
///   - Before executing a swap, the bot calls `propose_trade()` which:
///       1. Packs swap params into a q-types `Vertex` with payload type `TradeProposal`
///       2. Calls `reliable_broadcast.broadcast_vertex(vertex).await`
///       3. Waits for BRB delivery (2f+1 echoes received = Byzantine fault tolerant)
///       4. Returns `Delivered` only if enough honest nodes confirmed the proposal
///   - This extends the local 6-species swarm vote to a network-wide BFT vote:
///       * Each node runs its own IndicatorSet + resonance gate locally
///       * The Bracha echo/ready messages carry Ed25519 signatures from each node
///       * A swap only executes after >(2n/3 + 1) nodes have delivered the proposal
///
/// Current status: ARCHITECTURE DEFINED, not yet wired.
/// To enable:
///   1. Add `q-narwhal-core` to Cargo.toml dependencies (already done)
///   2. Spin up a `NarwhalNode` in main.rs with `NarwhalConfig { ... }`
///   3. Pass `node.reliable_broadcast.clone()` to `BotP2pBridge::new()`
///   4. Set `DarkKnightConfig::p2p_bracha = true`
///
/// Reference: Bracha (1987) "Asynchronous Byzantine Agreement Protocols"

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Proposal broadcast via Bracha before a swap executes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeProposal {
    pub token_in: String,
    pub token_out: String,
    pub amount_display: f64,
    pub resonance_eta: f64,
    pub swarm_yes_pct: f64,
    pub rsi: f64,
    pub adx: f64,
    pub proposed_at_unix: u64,
    /// Ed25519 public key of the proposing node (for accountability)
    pub proposer_pubkey_hex: String,
}

/// Outcome of a Bracha BRB round.
#[derive(Debug)]
pub enum BrbOutcome {
    /// Quorum reached — proceed with execution.
    Delivered,
    /// Insufficient echoes within timeout — skip this cycle.
    Timeout { received: usize, needed: usize },
    /// Not connected to P2P network.
    Disconnected,
}

/// Bridge to the narwhal-core ReliableBroadcast instance.
/// Wire this up by passing the narwhal node's `reliable_broadcast` field.
pub struct BotP2pBridge {
    // NOTE: In production, hold `Arc<q_narwhal_core::ReliableBroadcast>` here.
    // Placeholder until the narwhal node is instantiated.
    node_id: String,
    validator_count: usize,
}

impl BotP2pBridge {
    pub fn new(node_id: String, validator_count: usize) -> Self {
        BotP2pBridge { node_id, validator_count }
    }

    /// Broadcast `proposal` via Bracha RBB and wait for quorum.
    ///
    /// BFT threshold: needs >(2n/3 + 1) = `threshold_2f_plus_1` echos.
    /// With 6 Quillon nodes: threshold = 5 (tolerates 1 Byzantine).
    pub async fn propose_trade(&self, proposal: &TradeProposal) -> Result<BrbOutcome> {
        let threshold = (2 * self.validator_count / 3) + 1;

        info!(
            "📡 BRB proposal: {} {} → {} (need {}/{} echoes) | proposer={}",
            proposal.amount_display, proposal.token_in, proposal.token_out,
            threshold, self.validator_count, &self.node_id[..8.min(self.node_id.len())]
        );

        // ── PRODUCTION IMPLEMENTATION ─────────────────────────────────────
        // let vertex_id = blake3::hash(&serde_json::to_vec(proposal)?);
        // self.narwhal_brb.broadcast_vertex(Vertex {
        //     id: vertex_id.as_bytes().to_vec(),
        //     payload: serde_json::to_vec(proposal)?,
        //     author: self.node_id.clone(),
        //     round: 0,
        //     ..Default::default()
        // }).await?;
        //
        // // Poll for delivery (Bracha converges in O(n²) messages)
        // let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        // loop {
        //     if self.narwhal_brb.is_delivered(&vertex_id).await {
        //         return Ok(BrbOutcome::Delivered);
        //     }
        //     if tokio::time::Instant::now() > deadline {
        //         return Ok(BrbOutcome::Timeout { received: 0, needed: threshold });
        //     }
        //     tokio::time::sleep(Duration::from_millis(100)).await;
        // }
        // ── END PRODUCTION IMPLEMENTATION ────────────────────────────────

        // Stub: return Disconnected so bot falls back to local-only mode
        Ok(BrbOutcome::Disconnected)
    }
}
