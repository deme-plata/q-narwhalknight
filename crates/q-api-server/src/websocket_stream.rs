//! High-Performance WebSocket Transaction Streaming
//!
//! This module provides WebSocket-based transaction streaming for ultra-high TPS:
//! - Zero HTTP overhead (persistent connection)
//! - Binary MessagePack protocol
//! - Optimized parallel processing with rayon
//! - Batch signature verification
//! - Target: 1M+ TPS
//!
//! Architecture:
//! - Client opens WebSocket connection
//! - Client streams batches of transactions continuously
//! - Server processes with optimized parallel workers
//! - Server sends batch acknowledgments

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use ed25519_dalek::{Signature as DalekSignature, Verifier, VerifyingKey};
use futures::{SinkExt, StreamExt};
use q_types::Transaction;
use rayon::prelude::*;
use rmp_serde;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// Transaction batch for WebSocket streaming
#[derive(Debug, Serialize, Deserialize)]
pub struct TransactionBatch {
    pub batch_id: u64,
    pub transactions: Vec<Transaction>,
}

/// Batch acknowledgment
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchAck {
    pub batch_id: u64,
    pub accepted: usize,
    pub rejected: usize,
    pub processing_time_us: u64,
}

/// WebSocket transaction processor with optimized parallel processing
pub struct WebSocketProcessor {
    /// Number of parallel worker threads
    worker_count: usize,
    /// Atomic statistics (lock-free)
    total_batches: AtomicU64,
    total_transactions: AtomicU64,
    total_processing_time_us: AtomicU64,
}

#[derive(Debug, Default)]
pub struct ProcessorStats {
    pub total_batches: u64,
    pub total_transactions: u64,
    pub total_processing_time_us: u64,
}

impl WebSocketProcessor {
    pub fn new(worker_count: usize) -> Self {
        // Initialize rayon thread pool for CPU-bound work
        rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build_global()
            .ok();

        Self {
            worker_count,
            total_batches: AtomicU64::new(0),
            total_transactions: AtomicU64::new(0),
            total_processing_time_us: AtomicU64::new(0),
        }
    }

    /// Process a batch of transactions with optimized parallel processing
    pub async fn process_batch(&self, batch: TransactionBatch) -> BatchAck {
        let start = std::time::Instant::now();
        let tx_count = batch.transactions.len();

        debug!(
            "Processing batch {} with {} transactions",
            batch.batch_id, tx_count
        );

        // Use rayon for CPU-bound parallel processing (faster than tokio::spawn for CPU work)
        let (accepted, rejected) = tokio::task::spawn_blocking(move || {
            // Process transactions in parallel using rayon
            batch
                .transactions
                .par_iter()
                .map(|tx| Self::validate_transaction_fast(tx))
                .fold(
                    || (0usize, 0usize),
                    |(acc, rej), valid| {
                        if valid {
                            (acc + 1, rej)
                        } else {
                            (acc, rej + 1)
                        }
                    },
                )
                .reduce(
                    || (0, 0),
                    |(acc1, rej1), (acc2, rej2)| (acc1 + acc2, rej1 + rej2),
                )
        })
        .await
        .unwrap_or((0, tx_count));

        let processing_time_us = start.elapsed().as_micros() as u64;

        // Update statistics (lock-free atomic operations)
        self.total_batches.fetch_add(1, Ordering::Relaxed);
        self.total_transactions
            .fetch_add(tx_count as u64, Ordering::Relaxed);
        self.total_processing_time_us
            .fetch_add(processing_time_us, Ordering::Relaxed);

        if batch.batch_id % 10 == 0 {
            info!(
                "Batch {} complete: {}/{} accepted in {}μs ({:.2} TPS)",
                batch.batch_id,
                accepted,
                tx_count,
                processing_time_us,
                (tx_count as f64 / (processing_time_us as f64 / 1_000_000.0))
            );
        }

        BatchAck {
            batch_id: batch.batch_id,
            accepted,
            rejected,
            processing_time_us,
        }
    }

    /// Fast transaction validation with real signature verification
    fn validate_transaction_fast(tx: &Transaction) -> bool {
        // Basic checks first (fail fast)
        if tx.amount == 0 || tx.signature.len() != 64 {
            return false;
        }

        // Verify Ed25519 signature
        // Reconstruct the message that was signed
        let mut hasher = Sha3_256::new();
        hasher.update(&tx.id);
        hasher.update(&tx.from);
        hasher.update(&tx.to);
        hasher.update(&tx.amount.to_le_bytes());
        hasher.update(&tx.fee.to_le_bytes());
        hasher.update(&tx.nonce.to_le_bytes());
        let message = hasher.finalize();

        // Verify signature
        // Extract fixed-size arrays from slices
        let public_key_bytes: [u8; 32] = match tx.from[..32].try_into() {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };

        let signature_bytes: [u8; 64] = match tx.signature[..64].try_into() {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };

        // Try to create verifying key and signature
        let public_key = match VerifyingKey::from_bytes(&public_key_bytes) {
            Ok(pk) => pk,
            Err(_) => return false,
        };

        let signature = match DalekSignature::try_from(&signature_bytes[..]) {
            Ok(sig) => sig,
            Err(_) => return false,
        };

        // Verify the signature
        public_key.verify(&message, &signature).is_ok()
    }

    /// Get current statistics
    pub fn get_stats(&self) -> ProcessorStats {
        ProcessorStats {
            total_batches: self.total_batches.load(Ordering::Relaxed),
            total_transactions: self.total_transactions.load(Ordering::Relaxed),
            total_processing_time_us: self.total_processing_time_us.load(Ordering::Relaxed),
        }
    }
}

impl Clone for ProcessorStats {
    fn clone(&self) -> Self {
        Self {
            total_batches: self.total_batches,
            total_transactions: self.total_transactions,
            total_processing_time_us: self.total_processing_time_us,
        }
    }
}

/// WebSocket handler for transaction streaming
pub async fn ws_transaction_stream(ws: WebSocketUpgrade) -> Response {
    // Create processor with 16 workers (matches main.rs configuration)
    let processor = Arc::new(WebSocketProcessor::new(16));
    ws.on_upgrade(|socket| handle_socket(socket, processor))
}

/// Handle WebSocket connection
async fn handle_socket(socket: WebSocket, processor: Arc<WebSocketProcessor>) {
    let (sender, mut receiver) = socket.split();
    let sender = Arc::new(tokio::sync::Mutex::new(sender));
    let (ack_tx, mut ack_rx) = mpsc::channel::<BatchAck>(100);

    info!("WebSocket connection established for transaction streaming");

    // Spawn task to send acknowledgments back to client
    let ack_sender_handle = {
        let sender = sender.clone();
        tokio::spawn(async move {
            while let Some(ack) = ack_rx.recv().await {
                // Serialize acknowledgment to MessagePack
                if let Ok(packed) = rmp_serde::to_vec(&ack) {
                    if let Err(e) = sender.lock().await.send(Message::Binary(packed)).await {
                        error!("Failed to send acknowledgment: {}", e);
                        break;
                    }
                }
            }
        })
    };

    // Process incoming transaction batches
    let mut batch_count = 0;
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Binary(data)) => {
                batch_count += 1;

                // Deserialize batch from MessagePack
                match rmp_serde::from_slice::<TransactionBatch>(&data) {
                    Ok(batch) => {
                        debug!(
                            "Received batch {} with {} transactions",
                            batch.batch_id,
                            batch.transactions.len()
                        );

                        // Process batch with parallel workers
                        let ack = processor.process_batch(batch).await;

                        // Send acknowledgment back to client
                        if let Err(e) = ack_tx.send(ack).await {
                            error!("Failed to queue acknowledgment: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Failed to deserialize batch: {}", e);
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!(
                    "WebSocket connection closed by client after {} batches",
                    batch_count
                );
                break;
            }
            Ok(Message::Ping(data)) => {
                // Respond to ping
                if let Err(e) = sender.lock().await.send(Message::Pong(data)).await {
                    error!("Failed to send pong: {}", e);
                    break;
                }
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    // Wait for acknowledgment sender to finish
    ack_sender_handle.abort();

    let stats = processor.get_stats();
    info!(
        "WebSocket session complete: {} batches, {} transactions, avg {:.2}μs per tx",
        stats.total_batches,
        stats.total_transactions,
        stats.total_processing_time_us as f64 / stats.total_transactions as f64
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_processor_creation() {
        let processor = WebSocketProcessor::new(16);
        assert_eq!(processor.worker_count, 16);
    }

    #[tokio::test]
    async fn test_batch_processing() {
        use chrono::Utc;

        let processor = WebSocketProcessor::new(4);

        // Create test transactions
        let mut transactions = Vec::new();
        for i in 0..1000 {
            transactions.push(Transaction {
                id: [i as u8; 32],
                from: [1; 32],
                to: [2; 32],
                amount: 1000,
                fee: 1,
                nonce: i,
                signature: vec![0; 64],
                timestamp: Utc::now(),
                data: vec![],
                token_type: q_types::TokenType::QUG,
                fee_token_type: q_types::TokenType::QUGUSD,
                tx_type: q_types::TransactionType::Transfer,
                pqc_signature: None,
                signature_phase: q_types::TxSignaturePhase::Phase0Ed25519,
                pqc_public_key: None,
            });
        }

        let batch = TransactionBatch {
            batch_id: 1,
            transactions,
        };

        let ack = processor.process_batch(batch).await;

        assert_eq!(ack.batch_id, 1);
        assert_eq!(ack.accepted, 1000);
        assert_eq!(ack.rejected, 0);
        assert!(ack.processing_time_us > 0);
    }
}
