use crate::csv_formatter::BlocksCsv;
use q_storage::StorageEngine;
use std::sync::Arc;
use tracing::debug;

/// Extract blocks from `start_height..=end_height` into CSV.
pub async fn extract_blocks(
    storage: &Arc<StorageEngine>,
    start_height: u64,
    end_height: u64,
) -> anyhow::Result<String> {
    let mut csv = BlocksCsv::new();

    for height in start_height..=end_height {
        let block = match storage.get_qblock_by_height(height).await? {
            Some(b) => b,
            None => continue,
        };

        let hash = hex::encode(block.calculate_hash());
        let reward = block.header.total_coinbase_reward.unwrap_or(0);
        let proposer = hex::encode(block.header.proposer);

        csv.add_row(
            height,
            block.header.timestamp,
            &hash,
            &proposer,
            block.transactions.len(),
            reward,
            block.size_bytes,
            block.header.dag_round,
        );
    }

    debug!("[Dune] Extracted {} block rows ({}-{})", csv.row_count(), start_height, end_height);
    Ok(csv.finish())
}
