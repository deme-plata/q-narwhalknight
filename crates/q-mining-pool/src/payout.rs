//! Payout processing for mining pool

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::PayoutConfig;
use crate::error::{PayoutError, PoolResult};
use crate::pplns::RewardEntry;

/// Payout status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PayoutStatus {
    /// Pending payout (accumulating)
    Pending,
    /// Queued for processing
    Queued,
    /// Transaction submitted
    Processing,
    /// Payout completed
    Completed,
    /// Payout failed
    Failed,
}

/// Individual payout record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payout {
    /// Unique payout ID
    pub id: u64,

    /// Recipient wallet address
    pub wallet_address: String,

    /// Amount (atomic units)
    pub amount: u64,

    /// Status
    pub status: PayoutStatus,

    /// Round ID this payout is from
    pub round_id: u64,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,

    /// Transaction hash (if completed)
    pub tx_hash: Option<String>,

    /// Error message (if failed)
    pub error: Option<String>,
}

/// Pending balance for a wallet
#[derive(Debug)]
pub struct PendingBalance {
    /// Wallet address
    pub wallet_address: String,

    /// Pending amount
    pub amount: AtomicU64,

    /// Last updated
    pub last_updated: RwLock<DateTime<Utc>>,
}

impl PendingBalance {
    /// Create new pending balance
    pub fn new(wallet_address: String) -> Self {
        Self {
            wallet_address,
            amount: AtomicU64::new(0),
            last_updated: RwLock::new(Utc::now()),
        }
    }

    /// Add to balance
    pub fn add(&self, amount: u64) {
        self.amount.fetch_add(amount, Ordering::Relaxed);
        *self.last_updated.write() = Utc::now();
    }

    /// Get current balance
    pub fn balance(&self) -> u64 {
        self.amount.load(Ordering::Relaxed)
    }

    /// Reset balance (after payout)
    pub fn reset(&self) -> u64 {
        self.amount.swap(0, Ordering::Relaxed)
    }
}

/// Batch payout transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPayout {
    /// Batch ID
    pub id: u64,

    /// Individual payouts in this batch
    pub payouts: Vec<Payout>,

    /// Total amount
    pub total_amount: u64,

    /// Status
    pub status: PayoutStatus,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,

    /// Transaction hash
    pub tx_hash: Option<String>,
}

/// Payout processor
pub struct PayoutProcessor {
    /// Configuration
    config: PayoutConfig,

    /// Pending balances by wallet
    pending_balances: DashMap<String, PendingBalance>,

    /// Completed payouts history
    payout_history: RwLock<VecDeque<Payout>>,

    /// Batch payout history
    batch_history: RwLock<VecDeque<BatchPayout>>,

    /// Payout ID counter
    payout_counter: AtomicU64,

    /// Batch ID counter
    batch_counter: AtomicU64,

    /// Total paid out
    total_paid: AtomicU64,

    /// Maximum history size
    max_history: usize,
}

impl PayoutProcessor {
    /// Create new payout processor
    pub fn new(config: PayoutConfig) -> Self {
        Self {
            config,
            pending_balances: DashMap::new(),
            payout_history: RwLock::new(VecDeque::new()),
            batch_history: RwLock::new(VecDeque::new()),
            payout_counter: AtomicU64::new(1),
            batch_counter: AtomicU64::new(1),
            total_paid: AtomicU64::new(0),
            max_history: 10_000,
        }
    }

    /// Credit rewards to pending balances
    pub fn credit_rewards(&self, rewards: &[RewardEntry], round_id: u64) {
        for reward in rewards {
            let balance = self.pending_balances
                .entry(reward.wallet_address.clone())
                .or_insert_with(|| PendingBalance::new(reward.wallet_address.clone()));

            balance.add(reward.amount);

            tracing::debug!(
                wallet = %reward.wallet_address,
                amount = reward.amount,
                round = round_id,
                "Credited reward to pending balance"
            );
        }
    }

    /// Get pending balance for wallet
    pub fn get_pending_balance(&self, wallet: &str) -> u64 {
        self.pending_balances
            .get(wallet)
            .map(|b| b.balance())
            .unwrap_or(0)
    }

    /// Get all pending balances above threshold
    pub fn get_payable_balances(&self) -> Vec<(String, u64)> {
        self.pending_balances
            .iter()
            .filter(|b| b.balance() >= self.config.min_payout)
            .map(|b| (b.wallet_address.clone(), b.balance()))
            .collect()
    }

    /// Process pending payouts
    pub async fn process_payouts<F, Fut>(&self, send_transaction: F) -> PoolResult<Option<BatchPayout>>
    where
        F: Fn(Vec<(String, u64)>) -> Fut,
        Fut: std::future::Future<Output = PoolResult<String>>,
    {
        let payable = self.get_payable_balances();

        if payable.is_empty() {
            return Ok(None);
        }

        // Limit batch size
        let batch: Vec<_> = payable.into_iter()
            .take(self.config.max_batch_size)
            .collect();

        let total_amount: u64 = batch.iter().map(|(_, amount)| *amount).sum();

        tracing::info!(
            count = batch.len(),
            total = total_amount,
            "Processing batch payout"
        );

        // Create batch
        let batch_id = self.batch_counter.fetch_add(1, Ordering::Relaxed);
        let mut payouts = Vec::new();

        for (wallet, amount) in &batch {
            let payout_id = self.payout_counter.fetch_add(1, Ordering::Relaxed);
            payouts.push(Payout {
                id: payout_id,
                wallet_address: wallet.clone(),
                amount: *amount,
                status: PayoutStatus::Queued,
                round_id: 0, // Multiple rounds aggregated
                created_at: Utc::now(),
                completed_at: None,
                tx_hash: None,
                error: None,
            });
        }

        let mut batch_payout = BatchPayout {
            id: batch_id,
            payouts: payouts.clone(),
            total_amount,
            status: PayoutStatus::Processing,
            created_at: Utc::now(),
            completed_at: None,
            tx_hash: None,
        };

        // Send transaction
        match send_transaction(batch.clone()).await {
            Ok(tx_hash) => {
                // Success - update statuses
                batch_payout.status = PayoutStatus::Completed;
                batch_payout.completed_at = Some(Utc::now());
                batch_payout.tx_hash = Some(tx_hash.clone());

                for payout in &mut batch_payout.payouts {
                    payout.status = PayoutStatus::Completed;
                    payout.completed_at = Some(Utc::now());
                    payout.tx_hash = Some(tx_hash.clone());
                }

                // Clear pending balances
                for (wallet, _) in &batch {
                    if let Some(balance) = self.pending_balances.get(wallet) {
                        balance.reset();
                    }
                }

                // Update total paid
                self.total_paid.fetch_add(total_amount, Ordering::Relaxed);

                // Store in history
                {
                    let mut history = self.payout_history.write();
                    for payout in &batch_payout.payouts {
                        history.push_back(payout.clone());
                        if history.len() > self.max_history {
                            history.pop_front();
                        }
                    }
                }

                {
                    let mut batch_history = self.batch_history.write();
                    batch_history.push_back(batch_payout.clone());
                    if batch_history.len() > 1000 {
                        batch_history.pop_front();
                    }
                }

                tracing::info!(
                    batch_id = batch_id,
                    tx_hash = %tx_hash,
                    amount = total_amount,
                    recipients = batch.len(),
                    "Batch payout completed"
                );

                Ok(Some(batch_payout))
            }
            Err(e) => {
                // Failed
                batch_payout.status = PayoutStatus::Failed;

                for payout in &mut batch_payout.payouts {
                    payout.status = PayoutStatus::Failed;
                    payout.error = Some(e.to_string());
                }

                tracing::error!(
                    batch_id = batch_id,
                    error = %e,
                    "Batch payout failed"
                );

                Err(PayoutError::BatchFailed(e.to_string()).into())
            }
        }
    }

    /// Get payout history for wallet
    pub fn get_wallet_history(&self, wallet: &str) -> Vec<Payout> {
        self.payout_history
            .read()
            .iter()
            .filter(|p| p.wallet_address == wallet)
            .cloned()
            .collect()
    }

    /// Get total paid out
    pub fn total_paid(&self) -> u64 {
        self.total_paid.load(Ordering::Relaxed)
    }

    /// Get payout statistics
    pub fn stats(&self) -> PayoutStats {
        let pending_total: u64 = self.pending_balances
            .iter()
            .map(|b| b.balance())
            .sum();

        let pending_count = self.pending_balances
            .iter()
            .filter(|b| b.balance() > 0)
            .count();

        PayoutStats {
            total_paid: self.total_paid.load(Ordering::Relaxed),
            pending_total,
            pending_wallets: pending_count,
            min_payout: self.config.min_payout,
            payouts_completed: self.payout_history.read().len(),
            batches_completed: self.batch_history.read().len(),
        }
    }

    /// Get recent batch payouts
    pub fn recent_batches(&self, count: usize) -> Vec<BatchPayout> {
        self.batch_history
            .read()
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
}

/// Payout statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutStats {
    pub total_paid: u64,
    pub pending_total: u64,
    pub pending_wallets: usize,
    pub min_payout: u64,
    pub payouts_completed: usize,
    pub batches_completed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PayoutInterval;

    fn test_config() -> PayoutConfig {
        PayoutConfig {
            min_payout: 1_000_000, // 0.001 QUG
            interval: PayoutInterval::Immediate,
            max_batch_size: 100,
            auto_payout: true,
        }
    }

    #[test]
    fn test_credit_rewards() {
        let processor = PayoutProcessor::new(test_config());

        let rewards = vec![
            RewardEntry {
                worker_id: crate::worker::WorkerId::new("qnk123", "rig1"),
                wallet_address: "qnk123abc".to_string(),
                amount: 1_000_000,
                proportion: 0.5,
                difficulty_contribution: 1.0,
            },
            RewardEntry {
                worker_id: crate::worker::WorkerId::new("qnk456", "rig1"),
                wallet_address: "qnk456def".to_string(),
                amount: 1_000_000,
                proportion: 0.5,
                difficulty_contribution: 1.0,
            },
        ];

        processor.credit_rewards(&rewards, 1);

        assert_eq!(processor.get_pending_balance("qnk123abc"), 1_000_000);
        assert_eq!(processor.get_pending_balance("qnk456def"), 1_000_000);
    }

    #[test]
    fn test_min_payout_threshold() {
        let processor = PayoutProcessor::new(test_config());

        // Credit below threshold
        let rewards = vec![
            RewardEntry {
                worker_id: crate::worker::WorkerId::new("qnk123", "rig1"),
                wallet_address: "qnk123abc".to_string(),
                amount: 500_000, // Below 1_000_000 threshold
                proportion: 0.5,
                difficulty_contribution: 1.0,
            },
        ];

        processor.credit_rewards(&rewards, 1);

        let payable = processor.get_payable_balances();
        assert!(payable.is_empty());

        // Credit more to exceed threshold
        processor.credit_rewards(&rewards, 2);

        let payable = processor.get_payable_balances();
        assert_eq!(payable.len(), 1);
        assert_eq!(payable[0].1, 1_000_000);
    }
}
