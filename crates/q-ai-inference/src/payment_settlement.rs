//! Payment Settlement for Decentralized AI Inference
//!
//! v6.0.0: Implements the economic loop: users pay, workers earn, verifiers get bounties.
//! All payments are escrowed during inference and settled atomically after verification.
//!
//! ## Payment Flow
//!
//! ```text
//! 1. User commits escrow: price_per_token × max_tokens (debit wallet)
//! 2. Worker executes inference (N tokens)
//! 3a. Verification passes → Worker paid N × price_per_token
//!     Verifier paid small_fee, remainder refunded to user
//! 3b. Dispute + worker loses → Worker gets nothing, slashed
//!     Verifier gets inference_fee + slash_bounty
//!     User refunded full escrow
//! 4. All payments recorded as DAG transactions
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Default price per token in QUG base units (24-decimal)
/// 0.0001 QUG = ~$0.30 at $3000/QUG
pub const DEFAULT_PRICE_PER_TOKEN: u128 = 100_000_000_000_000_000_000; // 0.0001 * 1e24

/// Verifier reward percentage of total inference fee (basis points)
pub const VERIFIER_REWARD_BPS: u64 = 500; // 5%

/// Escrow entry for an in-progress inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscrowEntry {
    /// Unique request identifier
    pub request_id: [u8; 32],
    /// User who is paying for inference
    pub user_address: [u8; 32],
    /// Worker assigned to execute
    pub worker_address: [u8; 32],
    /// Escrowed amount (price_per_token × max_tokens)
    pub escrow_amount: u128,
    /// Price per token agreed upon
    pub price_per_token: u128,
    /// Maximum tokens user is paying for
    pub max_tokens: u32,
    /// Block height when escrow was created
    pub created_at: u64,
    /// Whether this escrow has been settled
    pub settled: bool,
}

/// Settlement result after inference completes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementResult {
    pub request_id: [u8; 32],
    /// Amount paid to worker
    pub worker_payment: u128,
    /// Amount paid to verifier (if verification happened)
    pub verifier_payment: u128,
    /// Amount refunded to user (unused tokens)
    pub user_refund: u128,
    /// Addresses and amounts for each transfer
    pub transfers: Vec<PaymentTransfer>,
}

/// Individual payment transfer (to be recorded on-chain)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentTransfer {
    pub from: [u8; 32],
    pub to: [u8; 32],
    pub amount: u128,
    pub reason: PaymentReason,
}

/// Reason for a payment transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaymentReason {
    /// User escrowing funds for inference
    InferenceEscrow,
    /// Worker payment for completed inference
    InferencePayment { tokens_generated: u32 },
    /// Verifier reward for successful verification
    VerifierReward,
    /// Refund to user for unused tokens
    UserRefund,
    /// Slash bounty paid to dispute winner
    SlashBounty,
    /// Full refund on worker fraud
    FraudRefund,
}

/// Payment Settlement Manager
pub struct PaymentSettlement {
    /// Active escrows: request_id -> escrow entry
    escrows: Arc<RwLock<HashMap<[u8; 32], EscrowEntry>>>,
    /// Settlement history (last 1000)
    history: Arc<RwLock<Vec<SettlementResult>>>,
    /// Statistics
    stats: Arc<RwLock<PaymentStats>>,
}

/// Payment statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PaymentStats {
    pub total_escrows_created: u64,
    pub total_settled: u64,
    pub total_refunded: u64,
    pub total_worker_earnings: u128,
    pub total_verifier_earnings: u128,
    pub total_user_refunds: u128,
    pub total_escrowed_volume: u128,
}

impl PaymentSettlement {
    pub fn new() -> Self {
        info!("💰 Initializing Payment Settlement for AI inference");
        Self {
            escrows: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(PaymentStats::default())),
        }
    }

    /// Create escrow for an inference request.
    ///
    /// The caller MUST debit `escrow_amount` from the user's wallet_balances
    /// BEFORE calling this method.
    pub async fn create_escrow(
        &self,
        request_id: [u8; 32],
        user_address: [u8; 32],
        worker_address: [u8; 32],
        price_per_token: u128,
        max_tokens: u32,
        current_height: u64,
    ) -> Result<EscrowEntry> {
        let escrow_amount = price_per_token
            .checked_mul(max_tokens as u128)
            .ok_or_else(|| anyhow!("Escrow amount overflow"))?;

        let entry = EscrowEntry {
            request_id,
            user_address,
            worker_address,
            escrow_amount,
            price_per_token,
            max_tokens,
            created_at: current_height,
            settled: false,
        };

        self.escrows.write().await.insert(request_id, entry.clone());

        let mut stats = self.stats.write().await;
        stats.total_escrows_created += 1;
        stats.total_escrowed_volume += escrow_amount;

        info!(
            "🔐 Escrow created: {} QUG for up to {} tokens",
            escrow_amount / 10u128.pow(24),
            max_tokens
        );

        Ok(entry)
    }

    /// Settle escrow after successful inference (verification passed or not selected).
    ///
    /// Returns transfers that the caller must execute against wallet_balances.
    pub async fn settle_success(
        &self,
        request_id: &[u8; 32],
        tokens_generated: u32,
        verifier_address: Option<[u8; 32]>,
    ) -> Result<SettlementResult> {
        let mut escrows = self.escrows.write().await;
        let escrow = escrows.get_mut(request_id)
            .ok_or_else(|| anyhow!("No escrow for request"))?;

        if escrow.settled {
            return Err(anyhow!("Escrow already settled"));
        }

        let actual_tokens = tokens_generated.min(escrow.max_tokens);
        let worker_payment = escrow.price_per_token * actual_tokens as u128;

        let verifier_payment = if verifier_address.is_some() {
            worker_payment * VERIFIER_REWARD_BPS as u128 / 10_000
        } else {
            0
        };

        let total_cost = worker_payment + verifier_payment;
        let user_refund = escrow.escrow_amount.saturating_sub(total_cost);

        let mut transfers = Vec::new();

        // Worker payment
        transfers.push(PaymentTransfer {
            from: escrow.user_address,
            to: escrow.worker_address,
            amount: worker_payment,
            reason: PaymentReason::InferencePayment { tokens_generated: actual_tokens },
        });

        // Verifier reward
        if let Some(verifier) = verifier_address {
            if verifier_payment > 0 {
                transfers.push(PaymentTransfer {
                    from: escrow.user_address,
                    to: verifier,
                    amount: verifier_payment,
                    reason: PaymentReason::VerifierReward,
                });
            }
        }

        // User refund
        if user_refund > 0 {
            transfers.push(PaymentTransfer {
                from: [0u8; 32], // From escrow (virtual)
                to: escrow.user_address,
                amount: user_refund,
                reason: PaymentReason::UserRefund,
            });
        }

        escrow.settled = true;

        let result = SettlementResult {
            request_id: *request_id,
            worker_payment,
            verifier_payment,
            user_refund,
            transfers,
        };

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_settled += 1;
        stats.total_worker_earnings += worker_payment;
        stats.total_verifier_earnings += verifier_payment;
        stats.total_user_refunds += user_refund;

        // Store in history
        let mut history = self.history.write().await;
        history.push(result.clone());
        if history.len() > 1000 {
            let excess = history.len() - 1000;
            history.drain(0..excess);
        }

        Ok(result)
    }

    /// Settle escrow after worker fraud (dispute lost by worker).
    ///
    /// Full refund to user. Worker gets nothing (slashing handled by opml_verifier).
    pub async fn settle_fraud(
        &self,
        request_id: &[u8; 32],
        slash_bounty: u128,
        verifier_address: [u8; 32],
    ) -> Result<SettlementResult> {
        let mut escrows = self.escrows.write().await;
        let escrow = escrows.get_mut(request_id)
            .ok_or_else(|| anyhow!("No escrow for request"))?;

        if escrow.settled {
            return Err(anyhow!("Escrow already settled"));
        }

        let mut transfers = Vec::new();

        // Full refund to user
        transfers.push(PaymentTransfer {
            from: [0u8; 32],
            to: escrow.user_address,
            amount: escrow.escrow_amount,
            reason: PaymentReason::FraudRefund,
        });

        // Slash bounty to verifier (from worker's slashed stake, not from escrow)
        if slash_bounty > 0 {
            transfers.push(PaymentTransfer {
                from: escrow.worker_address,
                to: verifier_address,
                amount: slash_bounty,
                reason: PaymentReason::SlashBounty,
            });
        }

        escrow.settled = true;

        let result = SettlementResult {
            request_id: *request_id,
            worker_payment: 0,
            verifier_payment: slash_bounty,
            user_refund: escrow.escrow_amount,
            transfers,
        };

        let mut stats = self.stats.write().await;
        stats.total_settled += 1;
        stats.total_refunded += 1;
        stats.total_user_refunds += escrow.escrow_amount;
        stats.total_verifier_earnings += slash_bounty;

        let mut history = self.history.write().await;
        history.push(result.clone());
        if history.len() > 1000 {
            let excess = history.len() - 1000;
            history.drain(0..excess);
        }

        Ok(result)
    }

    /// Expire stale escrows (no settlement after 500 blocks) — refund to user
    pub async fn expire_stale(&self, current_height: u64) -> Vec<SettlementResult> {
        let mut escrows = self.escrows.write().await;
        let stale: Vec<[u8; 32]> = escrows.iter()
            .filter(|(_, e)| !e.settled && current_height.saturating_sub(e.created_at) > 500)
            .map(|(id, _)| *id)
            .collect();

        let mut results = Vec::new();
        for id in stale {
            if let Some(mut escrow) = escrows.get_mut(&id) {
                escrow.settled = true;
                results.push(SettlementResult {
                    request_id: id,
                    worker_payment: 0,
                    verifier_payment: 0,
                    user_refund: escrow.escrow_amount,
                    transfers: vec![PaymentTransfer {
                        from: [0u8; 32],
                        to: escrow.user_address,
                        amount: escrow.escrow_amount,
                        reason: PaymentReason::UserRefund,
                    }],
                });
            }
        }

        results
    }

    /// Get payment statistics
    pub async fn get_stats(&self) -> PaymentStats {
        self.stats.read().await.clone()
    }

    /// Get escrow for a request
    pub async fn get_escrow(&self, request_id: &[u8; 32]) -> Option<EscrowEntry> {
        self.escrows.read().await.get(request_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_escrow_lifecycle() {
        let settlement = PaymentSettlement::new();
        let request_id = [1u8; 32];
        let user = [10u8; 32];
        let worker = [20u8; 32];
        let price = DEFAULT_PRICE_PER_TOKEN;

        // Create escrow for 100 tokens
        let escrow = settlement.create_escrow(request_id, user, worker, price, 100, 1000).await.unwrap();
        assert_eq!(escrow.escrow_amount, price * 100);

        // Settle with 50 tokens generated
        let result = settlement.settle_success(&request_id, 50, None).await.unwrap();
        assert_eq!(result.worker_payment, price * 50);
        assert_eq!(result.user_refund, price * 50); // 50 unused tokens refunded
    }

    #[tokio::test]
    async fn test_fraud_settlement() {
        let settlement = PaymentSettlement::new();
        let request_id = [1u8; 32];
        let user = [10u8; 32];
        let worker = [20u8; 32];
        let verifier = [30u8; 32];
        let price = DEFAULT_PRICE_PER_TOKEN;

        settlement.create_escrow(request_id, user, worker, price, 100, 1000).await.unwrap();

        let slash_bounty = 500 * 10u128.pow(24); // 500 QUG bounty
        let result = settlement.settle_fraud(&request_id, slash_bounty, verifier).await.unwrap();

        assert_eq!(result.worker_payment, 0);
        assert_eq!(result.user_refund, price * 100); // Full refund
        assert_eq!(result.verifier_payment, slash_bounty);
    }

    #[tokio::test]
    async fn test_verifier_gets_cut() {
        let settlement = PaymentSettlement::new();
        let request_id = [1u8; 32];
        let user = [10u8; 32];
        let worker = [20u8; 32];
        let verifier = [30u8; 32];
        let price = DEFAULT_PRICE_PER_TOKEN;

        settlement.create_escrow(request_id, user, worker, price, 100, 1000).await.unwrap();

        let result = settlement.settle_success(&request_id, 100, Some(verifier)).await.unwrap();

        let expected_worker = price * 100;
        let expected_verifier = expected_worker * VERIFIER_REWARD_BPS as u128 / 10_000;
        assert_eq!(result.worker_payment, expected_worker);
        assert_eq!(result.verifier_payment, expected_verifier);
    }
}
