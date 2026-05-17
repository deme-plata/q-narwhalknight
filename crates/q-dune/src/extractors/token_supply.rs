use crate::csv_formatter::TokenSupplyCsv;
use q_storage::BalanceConsensusEngine;
use q_storage::emission_controller::{self, QUG_MAX_SUPPLY};
use std::sync::Arc;
use tracing::debug;

/// Take a point-in-time supply snapshot.
pub async fn extract_token_supply(
    bce: &Arc<BalanceConsensusEngine>,
    now_ts: u64,
) -> anyhow::Result<String> {
    let mut csv = TokenSupplyCsv::new();

    let summary = bce.get_emission_summary().await?;

    let total_supply = summary.total_supply;
    let pct_mined = if QUG_MAX_SUPPLY > 0 {
        (total_supply as f64 / QUG_MAX_SUPPLY as f64) * 100.0
    } else {
        0.0
    };

    // TODO: integrate real stablecoin/qcredit supply queries when available
    csv.add_row(
        now_ts,
        total_supply,
        QUG_MAX_SUPPLY,
        pct_mined,
        summary.current_era,
        emission_controller::annual_emission(summary.current_era),
        0.0, // qugusd_supply placeholder
        0.0, // qcredit_supply placeholder
        0.0, // qusd_supply placeholder
    );

    debug!("[Dune] Supply snapshot: {:.6} QUG ({:.4}%)", summary.total_supply as f64 / 1e24, pct_mined);
    Ok(csv.finish())
}
