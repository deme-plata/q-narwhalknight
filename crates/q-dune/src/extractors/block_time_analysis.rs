use crate::csv_formatter::BlockTimeAnalysisCsv;
use q_storage::StorageEngine;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::debug;

/// Hourly block time analysis: throughput, variability, emission rate.
/// Groups blocks by hour and computes statistics for each hourly bucket.
pub async fn extract_block_time_analysis(
    storage: &Arc<StorageEngine>,
    start_height: u64,
    end_height: u64,
    date: &str,
) -> anyhow::Result<String> {
    let mut csv = BlockTimeAnalysisCsv::new();

    let day_start = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d")?;
    let ts_start = day_start.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp() as u64;
    let ts_end = ts_start + 86400;

    // Collect all block timestamps and data for the day, grouped by hour
    // Key: hour_ts (start of hour), Value: (timestamps, rewards, tx_counts, miners)
    let mut hourly: HashMap<u64, (Vec<u64>, Vec<u128>, Vec<u64>, HashSet<[u8; 32]>)> = HashMap::new();

    for height in start_height..=end_height {
        let block = match storage.get_qblock_by_height(height).await? {
            Some(b) => b,
            None => continue,
        };

        let ts = block.header.timestamp;
        if ts < ts_start || ts >= ts_end {
            continue;
        }

        // Round down to the start of the hour
        let hour_ts = (ts / 3600) * 3600;

        let entry = hourly.entry(hour_ts).or_insert_with(|| (Vec::new(), Vec::new(), Vec::new(), HashSet::new()));
        entry.0.push(ts);
        entry.1.push(block.header.total_coinbase_reward.unwrap_or(0));
        entry.2.push(block.transactions.len() as u64);
        entry.3.insert(block.header.proposer);
    }

    // Process each hour
    let mut hours: Vec<u64> = hourly.keys().copied().collect();
    hours.sort();

    for hour_ts in hours {
        let (timestamps, rewards, tx_counts, miners) = hourly.get(&hour_ts).unwrap();

        let blocks_produced = timestamps.len() as u64;
        if blocks_produced == 0 {
            continue;
        }

        // Compute block times (gaps between consecutive block timestamps)
        let mut sorted_ts = timestamps.clone();
        sorted_ts.sort();

        let block_times: Vec<f64> = sorted_ts.windows(2)
            .map(|w| (w[1] as f64) - (w[0] as f64))
            .filter(|&bt| bt >= 0.0) // safety: no negative intervals
            .collect();

        let (avg_bt, min_bt, max_bt, stddev_bt) = if !block_times.is_empty() {
            let n = block_times.len() as f64;
            let sum: f64 = block_times.iter().sum();
            let avg = sum / n;
            let min = block_times.iter().cloned().fold(f64::MAX, f64::min);
            let max = block_times.iter().cloned().fold(0.0_f64, f64::max);

            let variance = block_times.iter()
                .map(|&bt| (bt - avg) * (bt - avg))
                .sum::<f64>() / n;
            let stddev = variance.sqrt();

            (avg, min, max, stddev)
        } else {
            // Only 1 block in the hour — no interval data
            (0.0, 0.0, 0.0, 0.0)
        };

        let total_emission: u128 = rewards.iter().copied().sum();
        let total_tx_count: u64 = tx_counts.iter().sum();
        let avg_txs = total_tx_count as f64 / blocks_produced as f64;

        csv.add_row(
            hour_ts,
            blocks_produced,
            avg_bt,
            min_bt,
            max_bt,
            stddev_bt,
            total_emission,
            miners.len() as u64,
            total_tx_count,
            avg_txs,
        );
    }

    let result = csv.finish();
    debug!("[Dune] Block time analysis for {}: {} hourly buckets", date, result.lines().count().saturating_sub(1));
    Ok(result)
}
