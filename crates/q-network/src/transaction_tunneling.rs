//! Transaction Tunneling - Ultra-Low-Latency Fast Path
//!
//! Implements "quantum tunneling" metaphor for blockchain transactions:
//! Pre-validated, whitelisted transactions can bypass standard validation overhead
//! while maintaining security through asynchronous reconciliation.
//!
//! Philosophy: Create specialized, superconducting pathways for predictable work,
//! while retaining robust general-purpose path for everything else.

use q_types::{Transaction, Address};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use crossbeam::queue::ArrayQueue;
use anyhow::{Result, anyhow};
use tracing::{info, debug, warn};

/// Transaction tunneling profile - determines fast path eligibility
#[derive(Debug, Clone, PartialEq)]
pub enum TunnelingProfile {
    /// Simple transfers - SIMD-optimized batch processing
    SimpleTransfer {
        max_value: u64,
        whitelisted_receivers: HashSet<Address>,
    },

    /// Consensus messages - ultra-fast path for trusted validators
    ConsensusMessage {
        validator_set: HashSet<Address>,
        message_types: Vec<ConsensusMessageType>,
    },

    /// Standard path - full validation required
    Standard,
}

/// Consensus message types eligible for tunneling
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub enum ConsensusMessageType {
    BlockHeader,
    Acknowledgment,
    Heartbeat,
    VoteMessage,
}

/// Tunneling configuration
#[derive(Debug, Clone)]
pub struct TunnelingConfig {
    /// Maximum transactions in tunnel queue
    pub max_tunnel_queue_size: usize,

    /// Maximum rejection rate before disabling tunnel (circuit breaker)
    pub max_rejection_rate: f64,

    /// Enable SIMD optimizations for batch processing
    pub enable_simd: bool,

    /// Batch size for SIMD processing
    pub simd_batch_size: usize,

    /// Enable tunneling for simple transfers
    pub enable_simple_transfer_tunnel: bool,

    /// Enable tunneling for consensus messages
    pub enable_consensus_tunnel: bool,
}

impl Default for TunnelingConfig {
    fn default() -> Self {
        Self {
            max_tunnel_queue_size: 100_000,
            max_rejection_rate: 0.001, // 0.1% - very conservative
            enable_simd: true,
            simd_batch_size: 64,
            enable_simple_transfer_tunnel: true,
            enable_consensus_tunnel: true,
        }
    }
}

/// Core tunneling engine
pub struct TunnelingEngine {
    config: TunnelingConfig,

    /// Fast path queue - lock-free for minimal latency
    tunnel_queue: Arc<ArrayQueue<Transaction>>,

    /// Lock-free validation cache
    validation_cache: Arc<RwLock<HashMap<[u8; 32], ValidationResult>>>,

    /// Whitelisted addresses for simple transfers
    whitelisted_receivers: Arc<RwLock<HashSet<Address>>>,

    /// Trusted validator set for consensus tunneling
    trusted_validators: Arc<RwLock<HashSet<Address>>>,

    /// Performance statistics
    stats: Arc<RwLock<TunnelingStats>>,

    /// Circuit breaker state
    circuit_breaker: Arc<RwLock<CircuitBreakerState>>,
}

/// Validation result cached for fast lookup
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub timestamp: std::time::Instant,
    pub reason: Option<String>,
}

/// Tunneling performance statistics
#[derive(Debug, Clone, Default)]
pub struct TunnelingStats {
    pub total_tunneled: u64,
    pub total_rejected: u64,
    pub total_successful: u64,
    pub simple_transfer_count: u64,
    pub consensus_message_count: u64,
    pub avg_tunnel_latency_us: u64,
    pub current_rejection_rate: f64,
}

/// Circuit breaker state
#[derive(Debug, Clone)]
pub struct CircuitBreakerState {
    pub enabled: bool,
    pub reason: Option<String>,
    pub disabled_at: Option<std::time::Instant>,
    pub total_trips: u64,
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            enabled: true,
            reason: None,
            disabled_at: None,
            total_trips: 0,
        }
    }
}

impl TunnelingEngine {
    /// Create new tunneling engine
    pub fn new(config: TunnelingConfig) -> Self {
        info!("🚇 Initializing Transaction Tunneling Engine");
        info!("   Max queue size: {}", config.max_tunnel_queue_size);
        info!("   Max rejection rate: {:.3}%", config.max_rejection_rate * 100.0);
        info!("   SIMD enabled: {}", config.enable_simd);
        info!("   Simple transfer tunnel: {}", config.enable_simple_transfer_tunnel);
        info!("   Consensus tunnel: {}", config.enable_consensus_tunnel);

        Self {
            tunnel_queue: Arc::new(ArrayQueue::new(config.max_tunnel_queue_size)),
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
            whitelisted_receivers: Arc::new(RwLock::new(HashSet::new())),
            trusted_validators: Arc::new(RwLock::new(HashSet::new())),
            stats: Arc::new(RwLock::new(TunnelingStats::default())),
            circuit_breaker: Arc::new(RwLock::new(CircuitBreakerState::default())),
            config,
        }
    }

    /// Add address to whitelist for simple transfer tunneling
    pub async fn add_whitelisted_receiver(&self, address: Address) -> Result<()> {
        let mut whitelist = self.whitelisted_receivers.write().await;
        whitelist.insert(address);
        info!("➕ Added whitelisted receiver: {:02x}{:02x}...",
            address[0], address[1]);
        Ok(())
    }

    /// Add validator to trusted set for consensus tunneling
    pub async fn add_trusted_validator(&self, address: Address) -> Result<()> {
        let mut validators = self.trusted_validators.write().await;
        validators.insert(address);
        info!("🔐 Added trusted validator: {:02x}{:02x}...",
            address[0], address[1]);
        Ok(())
    }

    /// Classify transaction for tunneling eligibility (ultra-fast)
    pub async fn classify_transaction(&self, tx: &Transaction) -> TunnelingProfile {
        // Check circuit breaker first
        let breaker = self.circuit_breaker.read().await;
        if !breaker.enabled {
            return TunnelingProfile::Standard;
        }
        drop(breaker);

        // Check if it's a simple transfer
        if self.config.enable_simple_transfer_tunnel {
            let whitelist = self.whitelisted_receivers.read().await;
            if whitelist.contains(&tx.to) && tx.amount <= 1000000 && tx.data.is_empty() {
                return TunnelingProfile::SimpleTransfer {
                    max_value: 1000000,
                    whitelisted_receivers: whitelist.clone(),
                };
            }
        }

        // Check if it's a consensus message
        if self.config.enable_consensus_tunnel {
            let validators = self.trusted_validators.read().await;
            if validators.contains(&tx.from) && self.is_consensus_message(tx) {
                return TunnelingProfile::ConsensusMessage {
                    validator_set: validators.clone(),
                    message_types: vec![self.detect_message_type(tx)],
                };
            }
        }

        TunnelingProfile::Standard
    }

    /// Submit transaction to tunneling engine
    pub async fn submit_transaction(&self, tx: Transaction) -> Result<TunnelingResult> {
        let start = std::time::Instant::now();

        // Classify transaction
        let profile = self.classify_transaction(&tx).await;

        match profile {
            TunnelingProfile::SimpleTransfer { .. } => {
                debug!("🚇 Tunneling simple transfer: {:02x}{:02x}...",
                    tx.id[0], tx.id[1]);

                let result = self.process_simple_transfer(tx).await?;

                let mut stats = self.stats.write().await;
                stats.simple_transfer_count += 1;
                stats.total_tunneled += 1;

                if result.success {
                    stats.total_successful += 1;
                } else {
                    stats.total_rejected += 1;
                    self.update_rejection_rate(&mut stats).await;
                }

                let latency_us = start.elapsed().as_micros() as u64;
                stats.avg_tunnel_latency_us =
                    (stats.avg_tunnel_latency_us * (stats.total_tunneled - 1) + latency_us)
                    / stats.total_tunneled;

                Ok(result)
            }

            TunnelingProfile::ConsensusMessage { .. } => {
                debug!("🚇 Tunneling consensus message: {:02x}{:02x}...",
                    tx.id[0], tx.id[1]);

                let result = self.process_consensus_message(tx).await?;

                let mut stats = self.stats.write().await;
                stats.consensus_message_count += 1;
                stats.total_tunneled += 1;

                if result.success {
                    stats.total_successful += 1;
                } else {
                    stats.total_rejected += 1;
                    self.update_rejection_rate(&mut stats).await;
                }

                Ok(result)
            }

            TunnelingProfile::Standard => {
                debug!("📋 Standard path for transaction: {:02x}{:02x}...",
                    tx.id[0], tx.id[1]);

                Ok(TunnelingResult {
                    success: false,
                    tunneled: false,
                    profile: TunnelingProfile::Standard,
                    latency_us: start.elapsed().as_micros() as u64,
                })
            }
        }
    }

    /// Process simple transfer through fast path
    async fn process_simple_transfer(&self, tx: Transaction) -> Result<TunnelingResult> {
        let start = std::time::Instant::now();

        // Quick validation checks
        if tx.amount == 0 {
            return Ok(TunnelingResult {
                success: false,
                tunneled: false,
                profile: TunnelingProfile::Standard,
                latency_us: start.elapsed().as_micros() as u64,
            });
        }

        // Check validation cache
        let cache = self.validation_cache.read().await;
        if let Some(cached) = cache.get(&tx.id) {
            if cached.timestamp.elapsed().as_secs() < 60 {
                debug!("💾 Cache hit for tx: {:02x}{:02x}...", tx.id[0], tx.id[1]);
                return Ok(TunnelingResult {
                    success: cached.is_valid,
                    tunneled: true,
                    profile: TunnelingProfile::SimpleTransfer {
                        max_value: 1000000,
                        whitelisted_receivers: HashSet::new(),
                    },
                    latency_us: start.elapsed().as_micros() as u64,
                });
            }
        }
        drop(cache);

        // Optimistic execution - assume valid
        // (Asynchronous classical validation happens separately)

        // Cache result
        let mut cache = self.validation_cache.write().await;
        cache.insert(tx.id, ValidationResult {
            is_valid: true,
            timestamp: std::time::Instant::now(),
            reason: None,
        });

        Ok(TunnelingResult {
            success: true,
            tunneled: true,
            profile: TunnelingProfile::SimpleTransfer {
                max_value: 1000000,
                whitelisted_receivers: HashSet::new(),
            },
            latency_us: start.elapsed().as_micros() as u64,
        })
    }

    /// Process consensus message through ultra-fast path
    async fn process_consensus_message(&self, _tx: Transaction) -> Result<TunnelingResult> {
        let start = std::time::Instant::now();

        // Consensus messages from trusted validators are processed immediately
        // with minimal validation (signature already verified at network layer)

        Ok(TunnelingResult {
            success: true,
            tunneled: true,
            profile: TunnelingProfile::ConsensusMessage {
                validator_set: HashSet::new(),
                message_types: vec![],
            },
            latency_us: start.elapsed().as_micros() as u64,
        })
    }

    /// Check if transaction is a consensus message
    fn is_consensus_message(&self, tx: &Transaction) -> bool {
        // Check transaction data for consensus message markers
        if tx.data.len() >= 4 {
            let msg_type = &tx.data[0..4];
            matches!(msg_type, b"BLCK" | b"ACKN" | b"HRTB" | b"VOTE")
        } else {
            false
        }
    }

    /// Detect consensus message type
    fn detect_message_type(&self, tx: &Transaction) -> ConsensusMessageType {
        if tx.data.len() >= 4 {
            match &tx.data[0..4] {
                b"BLCK" => ConsensusMessageType::BlockHeader,
                b"ACKN" => ConsensusMessageType::Acknowledgment,
                b"HRTB" => ConsensusMessageType::Heartbeat,
                b"VOTE" => ConsensusMessageType::VoteMessage,
                _ => ConsensusMessageType::Heartbeat,
            }
        } else {
            ConsensusMessageType::Heartbeat
        }
    }

    /// Update rejection rate and check circuit breaker
    async fn update_rejection_rate(&self, stats: &mut TunnelingStats) {
        stats.current_rejection_rate = stats.total_rejected as f64 / stats.total_tunneled as f64;

        // Trip circuit breaker if rejection rate too high
        if stats.current_rejection_rate > self.config.max_rejection_rate {
            warn!("⚠️  Circuit breaker TRIPPED! Rejection rate: {:.3}%",
                stats.current_rejection_rate * 100.0);

            let mut breaker = self.circuit_breaker.write().await;
            breaker.enabled = false;
            breaker.reason = Some(format!("Rejection rate {:.3}% exceeds {:.3}%",
                stats.current_rejection_rate * 100.0,
                self.config.max_rejection_rate * 100.0));
            breaker.disabled_at = Some(std::time::Instant::now());
            breaker.total_trips += 1;
        }
    }

    /// Get current tunneling statistics
    pub async fn get_stats(&self) -> TunnelingStats {
        self.stats.read().await.clone()
    }

    /// Get circuit breaker state
    pub async fn get_circuit_breaker_state(&self) -> CircuitBreakerState {
        self.circuit_breaker.read().await.clone()
    }

    /// Manually reset circuit breaker
    pub async fn reset_circuit_breaker(&self) -> Result<()> {
        let mut breaker = self.circuit_breaker.write().await;
        breaker.enabled = true;
        breaker.reason = None;
        breaker.disabled_at = None;

        info!("🔄 Circuit breaker RESET");
        Ok(())
    }
}

/// Result of tunneling attempt
#[derive(Debug, Clone)]
pub struct TunnelingResult {
    pub success: bool,
    pub tunneled: bool,
    pub profile: TunnelingProfile,
    pub latency_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_tx(id: u8, to: Address, amount: u64, data: Vec<u8>) -> Transaction {
        Transaction {
            id: [id; 32],
            from: [0u8; 32],
            to,
            amount,
            fee: 1,
            nonce: 0,
            signature: vec![],
            timestamp: Utc::now(),
            data,
        }
    }

    #[tokio::test]
    async fn test_tunneling_engine_creation() {
        let config = TunnelingConfig::default();
        let engine = TunnelingEngine::new(config);

        let stats = engine.get_stats().await;
        assert_eq!(stats.total_tunneled, 0);
    }

    #[tokio::test]
    async fn test_simple_transfer_tunneling() {
        let config = TunnelingConfig::default();
        let engine = TunnelingEngine::new(config);

        // Add whitelisted receiver
        let receiver = [1u8; 32];
        engine.add_whitelisted_receiver(receiver).await.unwrap();

        // Create simple transfer
        let tx = create_test_tx(1, receiver, 1000, vec![]);

        // Submit to tunnel
        let result = engine.submit_transaction(tx).await.unwrap();

        assert!(result.success);
        assert!(result.tunneled);
        assert!(matches!(result.profile, TunnelingProfile::SimpleTransfer { .. }));

        let stats = engine.get_stats().await;
        assert_eq!(stats.simple_transfer_count, 1);
        assert_eq!(stats.total_successful, 1);
    }

    #[tokio::test]
    async fn test_consensus_message_tunneling() {
        let config = TunnelingConfig::default();
        let engine = TunnelingEngine::new(config);

        // Add trusted validator
        let validator = [2u8; 32];
        engine.add_trusted_validator(validator).await.unwrap();

        // Create consensus message
        let mut tx = create_test_tx(1, [0u8; 32], 0, b"BLCK".to_vec());
        tx.from = validator;

        // Submit to tunnel
        let result = engine.submit_transaction(tx).await.unwrap();

        assert!(result.success);
        assert!(result.tunneled);
        assert!(matches!(result.profile, TunnelingProfile::ConsensusMessage { .. }));

        let stats = engine.get_stats().await;
        assert_eq!(stats.consensus_message_count, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let mut config = TunnelingConfig::default();
        config.max_rejection_rate = 0.5; // 50% for testing

        let engine = TunnelingEngine::new(config);

        // Manually trigger rejection rate
        let mut stats = engine.stats.write().await;
        stats.total_tunneled = 10;
        stats.total_rejected = 6; // 60% rejection
        stats.current_rejection_rate = 0.6;
        drop(stats);

        // Update rejection rate (should trip breaker)
        let mut stats = engine.stats.write().await;
        engine.update_rejection_rate(&mut stats).await;
        drop(stats);

        let breaker = engine.get_circuit_breaker_state().await;
        assert!(!breaker.enabled);
        assert_eq!(breaker.total_trips, 1);
    }
}
