use crate::csv_formatter::DexSwapsCsv;
use q_storage::StorageEngine;
use q_types::TransactionType;
use std::sync::Arc;
use tracing::debug;

/// Extract DEX swap transactions from blocks.
pub async fn extract_dex_swaps(
    storage: &Arc<StorageEngine>,
    start_height: u64,
    end_height: u64,
) -> anyhow::Result<String> {
    let mut csv = DexSwapsCsv::new();

    for height in start_height..=end_height {
        let block = match storage.get_qblock_by_height(height).await? {
            Some(b) => b,
            None => continue,
        };

        let block_ts = block.header.timestamp;

        for tx in &block.transactions {
            if tx.effective_tx_type() == TransactionType::Swap {
                let tx_hash = hex::encode(tx.id);
                let wallet = hex::encode(tx.from);
                // For swaps: `from` is the trader, `to` is the pool.
                // token_in/token_out can be derived from tx.data, but for now
                // we use "QUG" as input and the pool address as output token marker.
                let token_in = "QUG";
                let token_out = hex::encode(tx.to);

                csv.add_row(
                    &tx_hash,
                    height,
                    block_ts,
                    &wallet,
                    token_in,
                    &token_out,
                    tx.amount,
                    0, // amount_out is not directly in tx fields; would need pool logic
                );
            }
        }
    }

    debug!("[Dune] Extracted {} DEX swap rows ({}-{})", csv.row_count(), start_height, end_height);
    Ok(csv.finish())
}
