//! 2026-08-17 — light-client balance proof endpoint.
//!
//! `GET /api/v1/proof/balance/:address` — the missing piece that turns the
//! balance_root_v2 SMT (crates/q-storage/src/balance_smt.rs) from "a fingerprint
//! only this node's own code can use" into something anyone can independently
//! check. Response is fully self-contained: address, balance, the Merkle proof
//! siblings, and the root the proof is valid against. A caller verifies with
//! ONLY that JSON — via `SmtProof::verify()` locally, or by re-deriving the
//! same BLAKE3 chain by hand — without trusting this node any further than
//! trusting the root itself (which is independently comparable across nodes
//! via `/api/v1/integrity/balance-root`).
//!
//! Public, read-only, no auth — same posture as `/api/v1/integrity/balance-root`.
//! Never exposes anything beyond what `get_balance` already does; this only
//! adds the proof that the returned balance is really what the tree commits to.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use std::sync::Arc;

use crate::handlers::{parse_wallet_address, ApiResponse};
use crate::AppState;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BalanceProofResponse {
    /// Hex-encoded 32-byte wallet address this proof is for.
    pub address: String,
    /// The balance this proof attests to, in base units (24 decimals — divide
    /// by 10^24 for QUG). MUST match what `/api/v1/wallets/:address/balance`
    /// (or equivalent) reports for the same node/height.
    pub balance: String,
    /// v2 SMT root this proof verifies against. Compare against
    /// `/api/v1/integrity/balance-root`'s `root_v2_smt` on ANY node — a
    /// mismatch there (not here) is what would indicate a real problem.
    pub root_v2_smt: String,
    /// Merkle proof siblings, root-to-leaf order, hex-encoded, depth entries.
    pub siblings: Vec<String>,
    /// Bitmap marking which sibling levels are the canonical empty-subtree
    /// hash (lets a verifier skip re-deriving those — same encoding the
    /// node's own `SmtProof::verify()` uses).
    pub empty_bitmap: String,
    /// `"missing"` — the SMT has no entries yet on this node (never expected
    /// once balance_root_v2 activation lands, but honest right now since the
    /// SMT is not yet consensus-enforced). `"ready"` — proof is meaningful.
    pub smt_state: String,
}

/// GET /api/v1/proof/balance/:address
pub async fn balance_proof(
    State(state): State<Arc<AppState>>,
    Path(address): Path<String>,
) -> Result<Json<ApiResponse<BalanceProofResponse>>, StatusCode> {
    let addr = match parse_wallet_address(&address) {
        Ok(a) => a,
        Err(e) => return Ok(Json(ApiResponse::error(format!("Invalid address: {}", e)))),
    };

    let balance_opt = state
        .storage_engine
        .load_wallet_balance(&addr)
        .await
        .map_err(|e| {
            tracing::warn!("[BALANCE-PROOF] load_wallet_balance failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // 2026-08-17: an address that has NEVER been written to the wallet table
    // is not stored in the tree as leaf_hash(addr, 0) — its position defaults
    // structurally to a shared, address-independent "empty subtree" constant.
    // Calling prove(addr, 0) for it would build a proof claiming the WRONG
    // leaf value and can never verify, even though nothing is actually wrong.
    // Only a wallet that was genuinely written (even if later drained to
    // exactly 0) has a real leaf_hash(addr, 0) on disk that prove() can attest
    // to. Be honest about the difference rather than return a proof that
    // silently fails verification.
    let balance = match balance_opt {
        Some(b) => b,
        None => {
            return Ok(Json(ApiResponse::error(
                "This address has never appeared in the balance table — its balance is \
                 implicitly 0, but a cryptographic non-membership proof for never-touched \
                 addresses isn't supported yet (only addresses with at least one recorded \
                 write can be proven). This is not an error with the address."
                    .to_string(),
            )))
        }
    };

    let smt_root = state.storage_engine.balance_smt.root();
    let smt_genesis = state.storage_engine.balance_smt.genesis_root();
    let smt_state = if smt_root == smt_genesis {
        "missing"
    } else {
        "ready"
    };

    let proof = state
        .storage_engine
        .balance_smt
        .prove(&addr, balance)
        .map_err(|e| {
            tracing::warn!("[BALANCE-PROOF] prove() failed for {}: {}", address, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(ApiResponse::success(BalanceProofResponse {
        address: hex::encode(addr),
        balance: balance.to_string(),
        root_v2_smt: hex::encode(smt_root),
        siblings: proof.siblings.iter().map(hex::encode).collect(),
        empty_bitmap: hex::encode(proof.empty_bitmap),
        smt_state: smt_state.to_string(),
    })))
}
