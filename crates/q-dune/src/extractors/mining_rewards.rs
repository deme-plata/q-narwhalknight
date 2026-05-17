use crate::csv_formatter::MiningRewardsCsv;
use q_storage::StorageEngine;
use q_storage::emission_controller;
use q_types::TransactionType;
use std::sync::Arc;
use tracing::debug;

/// Extract mining reward entries from coinbase transactions in blocks.
pub async fn extract_mining_rewards(
    storage: &Arc<StorageEngine>,
    start_height: u64,
    end_height: u64,
    genesis_timestamp: u64,
) -> anyhow::Result<String> {
    let mut csv = MiningRewardsCsv::new();

    for height in start_height..=end_height {
        let block = match storage.get_qblock_by_height(height).await? {
            Some(b) => b,
            None => continue,
        };

        let block_ts = block.header.timestamp;
        let elapsed = block_ts.saturating_sub(genesis_timestamp);
        let era = elapsed / emission_controller::SECONDS_PER_HALVING;

        for tx in &block.transactions {
            if tx.effective_tx_type() == TransactionType::Coinbase {
                let miner_addr = hex::encode(tx.to);
                csv.add_row(height, block_ts, &miner_addr, tx.amount, era);
            }
        }

        // Also check balance_updates for mining rewards if no coinbase tx
        if block.transactions.iter().all(|t| t.effective_tx_type() != TransactionType::Coinbase) {
            if let Some(reward) = block.header.total_coinbase_reward {
                if reward > 0 {
                    let proposer = hex::encode(block.header.proposer);
                    csv.add_row(height, block_ts, &proposer, reward, era);
                }
            }
        }
    }

    debug!("[Dune] Extracted {} mining reward rows ({}-{})", csv.row_count(), start_height, end_height);
    Ok(csv.finish())
}
