use crate::csv_formatter::NetworkStatsCsv;
use tracing::debug;

/// Snapshot current network stats. Takes pre-fetched values from AppState.
/// Now includes blocks_per_minute and nakamoto_coefficient for enhanced analytics.
pub fn extract_network_stats(
    now_ts: u64,
    block_height: u64,
    peer_count: u32,
    active_miners: u32,
    total_hashrate_khs: f64,
    difficulty: f64,
    blocks_per_minute: f64,
    nakamoto_coefficient: u32,
) -> String {
    let mut csv = NetworkStatsCsv::new();
    csv.add_row(now_ts, block_height, peer_count, active_miners, total_hashrate_khs, difficulty, blocks_per_minute, nakamoto_coefficient);
    debug!("[Dune] Network stats snapshot: height={}, peers={}, miners={}, bpm={:.2}, nakamoto={}", block_height, peer_count, active_miners, blocks_per_minute, nakamoto_coefficient);
    csv.finish()
}
