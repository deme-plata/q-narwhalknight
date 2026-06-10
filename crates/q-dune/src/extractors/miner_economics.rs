use crate::csv_formatter::MinerEconomicsCsv;
use q_storage::StorageEngine;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// Per-miner daily economics: blocks mined, rewards earned, dominance %.
/// Scans all blocks in the given height range that fall on `date`.
pub async fn extract_miner_economics(
    storage: &Arc<StorageEngine>,
    start_height: u64,
    end_height: u64,
    date: &str,
) -> anyhow::Result<String> {
    let mut csv = MinerEconomicsCsv::new();

    let day_start = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")?;
    let ts_start = day_start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp() as u64;
    let ts_end = ts_start + 86400;

    // Track per-miner stats: (blocks_mined, total_reward, timestamps_vec)
    let mut miner_stats: HashMap<[u8; 32], (u64, u128, Vec<u64>)> = HashMap::new();
    let mut total_blocks: u64 = 0;

    for height in start_height..=end_height {
        let block = match storage.get_qblock_by_height(height).await? {
            Some(b) => b,
            None => continue,
        };

        if block.header.timestamp < ts_start || block.header.timestamp >= ts_end {
            continue;
        }

        total_blocks += 1;
        let proposer = block.header.proposer;
        let reward = block.header.total_coinbase_reward.unwrap_or(0);

        let entry = miner_stats.entry(proposer).or_insert((0, 0, Vec::new()));
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(reward);
        entry.2.push(block.header.timestamp);
    }

    // Also compute cumulative rewards per miner (all-time, from height 1 to end)
    // This is expensive, so we approximate using the daily data only
    // (cumulative tracking would need persistent state per miner)

    // Sort miners by blocks_mined descending for consistent output
    let mut miners: Vec<([u8; 32], (u64, u128, Vec<u64>))> = miner_stats.into_iter().collect();
    miners.sort_by(|a, b| b.1.0.cmp(&a.1.0));

    for (addr, (blocks_mined, total_reward, timestamps)) in &miners {
        let pct = if total_blocks > 0 {
            (*blocks_mined as f64 / total_blocks as f64) * 100.0
        } else {
            0.0
        };

        // Average block interval: sort timestamps, compute mean gap
        let avg_interval = if timestamps.len() > 1 {
            let mut sorted = timestamps.clone();
            sorted.sort();
            let gaps: Vec<f64> = sorted.windows(2)
                .map(|w| (w[1] - w[0]) as f64)
                .collect();
            if gaps.is_empty() { 0.0 } else { gaps.iter().sum::<f64>() / gaps.len() as f64 }
        } else {
            0.0
        };

        csv.add_row(
            date,
            &hex::encode(addr),
            *blocks_mined,
            *total_reward,
            pct,
            *total_reward, // cumulative = daily for now (would need persistent tracking)
            avg_interval,
        );
    }

    debug!("[Dune] Miner economics for {}: {} miners, {} blocks", date, miners.len(), total_blocks);
    Ok(csv.finish())
}
