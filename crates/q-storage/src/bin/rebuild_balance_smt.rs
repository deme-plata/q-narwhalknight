//! One-shot: rebuild the balance_root_v2 Sparse Merkle Tree from the wallet table.
//!
//! 2026-06-25 (rocky) — after the phantom-supply clawback, the primary balance root
//! (v1) reflects the corrected balances but the dormant v2 SMT was never populated
//! (roots_agree=false, smt_state=missing). This repopulates the SMT from the current
//! wallet table so /api/v1/integrity/balance-root reports roots_agree=true.
//!
//! SAFETY: reads wallet_balances, writes ONLY the SMT column family — it does NOT
//! modify any wallet balance. The node MUST be stopped (exclusive DB lock).
//!
//! Usage: rebuild_balance_smt [data_dir]   (default /home/orobit/data-mainnet-genesis)

use anyhow::Result;
use q_storage::StorageEngine;

#[tokio::main]
async fn main() -> Result<()> {
    let data_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/orobit/data-mainnet-genesis".to_string());
    println!("📊 [SMT-REBUILD] Opening StorageEngine at {} (node MUST be stopped)…", data_dir);

    // node_id is irrelevant for an offline SMT rebuild (used only for logging on open).
    let node_id: [u8; 32] = [0u8; 32];
    let engine = StorageEngine::open(&data_dir, node_id).await?;

    println!("📊 [SMT-REBUILD] Rebuilding balance_root_v2 SMT from the wallet table…");
    let root = engine.rebuild_balance_smt_from_wallet_table().await?;
    println!("✅ [SMT-REBUILD] Done. balance_root_v2 = {}", hex::encode(root));
    println!("   Restart the node and check /api/v1/integrity/balance-root → roots_agree.");
    Ok(())
}
