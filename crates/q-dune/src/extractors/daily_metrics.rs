use crate::csv_formatter::DailyMetricsCsv;
use q_storage::StorageEngine;
use q_types::TransactionType;
use std::collections::HashSet;
use std::sync::Arc;
use tracing::debug;

/// Aggregate a single day's metrics from blocks.
/// `date` is in YYYY-MM-DD format. We scan blocks whose timestamp falls on that day.
/// `circulating_supply` is the current total supply in base units (for velocity/NVT).
pub async fn extract_daily_metrics(
    storage: &Arc<StorageEngine>,
    start_height: u64,
    end_height: u64,
    date: &str,
    circulating_supply: u128,
) -> anyhow::Result<String> {
    let mut csv = DailyMetricsCsv::new();

    // Parse the target date boundaries
    let day_start = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")?;
    let ts_start = day_start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp() as u64;
    let ts_end = ts_start + 86400;

    let mut block_count: u64 = 0;
    let mut tx_count: u64 = 0;
    let mut total_volume: u128 = 0;
    let mut total_fees: u128 = 0;
    let mut active_addresses = HashSet::new();
    let mut total_emission: u128 = 0;
    let mut miners: HashSet<[u8; 32]> = HashSet::new();
    let mut swap_count: u64 = 0;

    for height in start_height..=end_height {
        let block = match storage.get_qblock_by_height(height).await? {
            Some(b) => b,
            None => continue,
        };

        if block.header.timestamp < ts_start || block.header.timestamp >= ts_end {
            continue;
        }

        block_count += 1;
        miners.insert(block.header.proposer);

        if let Some(reward) = block.header.total_coinbase_reward {
            total_emission = total_emission.saturating_add(reward);
        }

        for tx in &block.transactions {
            tx_count += 1;
            total_volume = total_volume.saturating_add(tx.amount);
            total_fees = total_fees.saturating_add(tx.fee);
            active_addresses.insert(tx.from);
            active_addresses.insert(tx.to);

            if tx.effective_tx_type() == TransactionType::Swap {
                swap_count += 1;
            }
        }
    }

    if block_count > 0 {
        csv.add_row(
            date,
            block_count,
            tx_count,
            total_volume,
            total_fees,
            active_addresses.len() as u64,
            total_emission,
            miners.len() as u64,
            swap_count,
            circulating_supply,
        );
    }

    debug!("[Dune] Daily metrics for {}: {} blocks, {} txs", date, block_count, tx_count);
    Ok(csv.finish())
}
