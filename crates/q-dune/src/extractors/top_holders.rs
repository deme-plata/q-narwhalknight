use crate::csv_formatter::TopHoldersCsv;
use q_storage::StorageEngine;
use std::sync::Arc;
use tracing::debug;

/// Extract top 100 holders snapshot for a given date.
pub async fn extract_top_holders(
    storage: &Arc<StorageEngine>,
    date: &str,
) -> anyhow::Result<String> {
    let mut csv = TopHoldersCsv::new();

    let balances = storage.load_wallet_balances().await?;

    // Sort by balance descending
    let mut entries: Vec<([u8; 32], u128)> = balances.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));

    // Compute total supply from all balances
    let total_supply: u128 = entries.iter().map(|(_, b)| *b).sum();

    for (rank_idx, (addr, balance)) in entries.iter().take(100).enumerate() {
        let pct = if total_supply > 0 {
            (*balance as f64 / total_supply as f64) * 100.0
        } else {
            0.0
        };

        csv.add_row(
            date,
            (rank_idx + 1) as u32,
            &hex::encode(addr),
            *balance,
            pct,
        );
    }

    debug!("[Dune] Top holders snapshot: {} entries for {}", entries.len().min(100), date);
    Ok(csv.finish())
}
