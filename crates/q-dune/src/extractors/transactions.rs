use crate::csv_formatter::TransactionsCsv;
use q_storage::StorageEngine;
use q_types::TransactionType;
use std::sync::Arc;
use tracing::debug;

/// Extract transactions from blocks in `start_height..=end_height` into CSV.
pub async fn extract_transactions(
    storage: &Arc<StorageEngine>,
    start_height: u64,
    end_height: u64,
) -> anyhow::Result<String> {
    let mut csv = TransactionsCsv::new();

    for height in start_height..=end_height {
        let block = match storage.get_qblock_by_height(height).await? {
            Some(b) => b,
            None => continue,
        };

        let block_ts = block.header.timestamp;

        for tx in &block.transactions {
            let tx_hash = hex::encode(tx.id);
            let tx_type_name = tx.effective_tx_type().name();
            let from_addr = hex::encode(tx.from);
            let to_addr = hex::encode(tx.to);
            let is_coinbase = tx.effective_tx_type() == TransactionType::Coinbase;

            csv.add_row(
                &tx_hash,
                height,
                block_ts,
                tx_type_name,
                &from_addr,
                &to_addr,
                tx.amount,
                tx.fee,
                is_coinbase,
            );
        }
    }

    debug!("[Dune] Extracted {} transaction rows ({}-{})", csv.row_count(), start_height, end_height);
    Ok(csv.finish())
}
