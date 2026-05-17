//! Failsafe Timer for Dandelion++ Protocol
//!
//! Prevents transaction loss by forcing fluff phase after timeout.
//! This is critical for ensuring transactions are never dropped even
//! if stem relay fails or peers become unavailable.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, watch};
use tracing::{debug, info, warn};

use crate::{DandelionMessage, DandelionPhase};

/// Tracked transaction for failsafe monitoring
#[derive(Debug, Clone)]
pub struct TrackedTransaction {
    /// The Dandelion message
    pub message: DandelionMessage,
    /// When we first received this transaction
    pub received_at: Instant,
    /// Number of relay attempts
    pub relay_attempts: u8,
    /// Current state
    pub state: TransactionState,
    /// Selected stem peer (if any)
    pub stem_peer: Option<[u8; 32]>,
}

/// Transaction state in Dandelion++ lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Just received, deciding phase
    Pending,
    /// In stem phase, being relayed
    Stemming { hops_completed: u8 },
    /// Transitioning to fluff
    FluffPending,
    /// Broadcasting via gossipsub
    Fluffing,
    /// Successfully delivered
    Delivered,
    /// Failed after retries
    Failed,
    /// Forced fluff due to timeout
    TimedOut,
}

/// Event types for state machine
#[derive(Debug, Clone)]
pub enum FailsafeEvent {
    /// Transaction timed out in stem phase
    StemTimeout { message_id: [u8; 32] },
    /// Stem relay succeeded
    StemSuccess { message_id: [u8; 32] },
    /// Stem relay failed
    StemFailed { message_id: [u8; 32], error: String },
    /// Fluff broadcast complete
    FluffComplete { message_id: [u8; 32] },
}

/// Failsafe timer configuration
#[derive(Debug, Clone)]
pub struct FailsafeConfig {
    /// Maximum time in stem phase before forced fluff
    pub stem_timeout: Duration,
    /// Check interval for timeouts
    pub check_interval: Duration,
    /// Maximum relay retries before giving up
    pub max_retries: u8,
    /// Delay between retries
    pub retry_delay: Duration,
    /// Maximum tracked transactions
    pub max_tracked: usize,
}

impl Default for FailsafeConfig {
    fn default() -> Self {
        Self {
            // v8.6.0: reduced from 30s to 20s — with max_stem_hops reduced to 5,
            // stems complete faster; 20s is sufficient before forcing fluff fallback
            stem_timeout: Duration::from_secs(20),
            check_interval: Duration::from_secs(1),
            max_retries: 3,
            // v8.6.0: reduced from 500ms to 300ms — faster retries between hops
            retry_delay: Duration::from_millis(300),
            max_tracked: 10000,
        }
    }
}

/// Failsafe timer for Dandelion++ transactions
pub struct FailsafeTimer {
    /// Configuration
    config: FailsafeConfig,
    /// Tracked transactions
    transactions: Arc<RwLock<HashMap<[u8; 32], TrackedTransaction>>>,
    /// Event sender for timeout notifications
    event_tx: mpsc::UnboundedSender<FailsafeEvent>,
    /// Shutdown signal
    shutdown_tx: watch::Sender<bool>,
    /// Shutdown receiver
    shutdown_rx: watch::Receiver<bool>,
}

impl FailsafeTimer {
    /// Create new failsafe timer
    pub fn new(config: FailsafeConfig) -> (Self, mpsc::UnboundedReceiver<FailsafeEvent>) {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let timer = Self {
            config,
            transactions: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            shutdown_tx,
            shutdown_rx,
        };

        (timer, event_rx)
    }

    /// Start the failsafe timer background task
    pub fn start(&self) -> tokio::task::JoinHandle<()> {
        let transactions = self.transactions.clone();
        let config = self.config.clone();
        let event_tx = self.event_tx.clone();
        let mut shutdown_rx = self.shutdown_rx.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.check_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::check_timeouts(&transactions, &config, &event_tx).await;
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            info!("Failsafe timer shutting down");
                            break;
                        }
                    }
                }
            }
        })
    }

    /// Check for timed out transactions
    async fn check_timeouts(
        transactions: &Arc<RwLock<HashMap<[u8; 32], TrackedTransaction>>>,
        config: &FailsafeConfig,
        event_tx: &mpsc::UnboundedSender<FailsafeEvent>,
    ) {
        let now = Instant::now();
        let mut timed_out = Vec::new();

        {
            let txs = transactions.read().await;
            for (id, tx) in txs.iter() {
                // Only check pending or stemming transactions
                if matches!(tx.state, TransactionState::Pending | TransactionState::Stemming { .. }) {
                    if now.duration_since(tx.received_at) > config.stem_timeout {
                        timed_out.push(*id);
                    }
                }
            }
        }

        // Process timeouts
        for message_id in timed_out {
            warn!(
                "Transaction {} exceeded failsafe timeout ({:?})",
                hex::encode(message_id),
                config.stem_timeout
            );

            // Update state
            {
                let mut txs = transactions.write().await;
                if let Some(tx) = txs.get_mut(&message_id) {
                    tx.state = TransactionState::TimedOut;
                }
            }

            // Send timeout event
            let _ = event_tx.send(FailsafeEvent::StemTimeout { message_id });
        }
    }

    /// Track a new transaction
    pub async fn track_transaction(&self, message: DandelionMessage) -> Result<()> {
        let mut transactions = self.transactions.write().await;

        // Check capacity
        if transactions.len() >= self.config.max_tracked {
            // Remove oldest completed transactions
            let mut to_remove = Vec::new();
            for (id, tx) in transactions.iter() {
                if matches!(tx.state, TransactionState::Delivered | TransactionState::Failed) {
                    to_remove.push(*id);
                }
            }
            for id in to_remove.into_iter().take(100) {
                transactions.remove(&id);
            }
        }

        let tracked = TrackedTransaction {
            message: message.clone(),
            received_at: Instant::now(),
            relay_attempts: 0,
            state: TransactionState::Pending,
            stem_peer: None,
        };

        transactions.insert(message.id, tracked);

        debug!(
            "Tracking transaction {} for failsafe",
            hex::encode(message.id)
        );

        Ok(())
    }

    /// Update transaction state
    pub async fn update_state(&self, message_id: [u8; 32], state: TransactionState) {
        let mut transactions = self.transactions.write().await;
        if let Some(tx) = transactions.get_mut(&message_id) {
            tx.state = state;
            debug!(
                "Updated transaction {} state to {:?}",
                hex::encode(message_id),
                state
            );
        }
    }

    /// Record stem relay attempt
    pub async fn record_relay_attempt(&self, message_id: [u8; 32], success: bool) {
        let mut transactions = self.transactions.write().await;
        if let Some(tx) = transactions.get_mut(&message_id) {
            tx.relay_attempts += 1;

            if success {
                if let TransactionState::Stemming { hops_completed } = tx.state {
                    tx.state = TransactionState::Stemming {
                        hops_completed: hops_completed + 1,
                    };
                }
                let _ = self.event_tx.send(FailsafeEvent::StemSuccess { message_id });
            } else if tx.relay_attempts >= self.config.max_retries {
                let _ = self.event_tx.send(FailsafeEvent::StemFailed {
                    message_id,
                    error: format!("Max retries ({}) exceeded", self.config.max_retries),
                });
            }
        }
    }

    /// Mark transaction as delivered
    pub async fn mark_delivered(&self, message_id: [u8; 32]) {
        self.update_state(message_id, TransactionState::Delivered).await;
        let _ = self.event_tx.send(FailsafeEvent::FluffComplete { message_id });
    }

    /// Get transaction state
    pub async fn get_state(&self, message_id: &[u8; 32]) -> Option<TransactionState> {
        let transactions = self.transactions.read().await;
        transactions.get(message_id).map(|tx| tx.state)
    }

    /// Get all pending transactions (for recovery)
    pub async fn get_pending_transactions(&self) -> Vec<DandelionMessage> {
        let transactions = self.transactions.read().await;
        transactions
            .values()
            .filter(|tx| {
                matches!(
                    tx.state,
                    TransactionState::Pending | TransactionState::Stemming { .. }
                )
            })
            .map(|tx| tx.message.clone())
            .collect()
    }

    /// Get statistics
    pub async fn get_stats(&self) -> FailsafeStats {
        let transactions = self.transactions.read().await;

        let mut pending = 0;
        let mut stemming = 0;
        let mut fluffing = 0;
        let mut delivered = 0;
        let mut failed = 0;
        let mut timed_out = 0;

        for tx in transactions.values() {
            match tx.state {
                TransactionState::Pending => pending += 1,
                TransactionState::Stemming { .. } => stemming += 1,
                TransactionState::FluffPending | TransactionState::Fluffing => fluffing += 1,
                TransactionState::Delivered => delivered += 1,
                TransactionState::Failed => failed += 1,
                TransactionState::TimedOut => timed_out += 1,
            }
        }

        FailsafeStats {
            total_tracked: transactions.len(),
            pending,
            stemming,
            fluffing,
            delivered,
            failed,
            timed_out,
        }
    }

    /// Shutdown the failsafe timer
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }
}

/// Statistics from failsafe timer
#[derive(Debug, Clone)]
pub struct FailsafeStats {
    pub total_tracked: usize,
    pub pending: usize,
    pub stemming: usize,
    pub fluffing: usize,
    pub delivered: usize,
    pub failed: usize,
    pub timed_out: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_failsafe_tracking() {
        let config = FailsafeConfig {
            stem_timeout: Duration::from_millis(100),
            check_interval: Duration::from_millis(10),
            ..Default::default()
        };

        let (timer, _rx) = FailsafeTimer::new(config);

        let message = DandelionMessage {
            id: [1u8; 32],
            payload: vec![1, 2, 3],
            phase: crate::DandelionPhase::Stem,
            hop_count: 0,
            vrf_proof: None,
            timestamp: 0,
            quantum_nonce: [0u8; 16],
        };

        timer.track_transaction(message).await.unwrap();

        let state = timer.get_state(&[1u8; 32]).await;
        assert!(matches!(state, Some(TransactionState::Pending)));
    }

    #[tokio::test]
    async fn test_failsafe_stats() {
        let config = FailsafeConfig::default();
        let (timer, _rx) = FailsafeTimer::new(config);

        let stats = timer.get_stats().await;
        assert_eq!(stats.total_tracked, 0);
    }
}
