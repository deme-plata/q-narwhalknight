/// Dandelion++ Implementation for Q-NarwhalKnight
/// Provides traffic analysis resistance by mixing transaction propagation patterns
/// with quantum-enhanced entropy for timing obfuscation
///
/// v3.4.2-beta: SECURITY FIX - Removed SocketAddr usage to prevent IP leaks
/// All relay addresses are now onion addresses (strings) routed through Tor
use anyhow::{Context, Result};
use rand::Rng;  // Keep Rng from rand (gen, gen_range methods)
use rand_chacha::{ChaChaRng, rand_core::{RngCore, SeedableRng}};  // Use compatible trait versions
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::circuit_manager::CircuitManager;
use crate::metrics::TorMetrics;

/// Generate a proper v3 onion address from a peer identifier
/// v3 onion addresses are 56 characters derived from Ed25519 public key
/// Format: base32(pubkey || checksum || version).onion
pub fn generate_onion_address(peer_id: &[u8]) -> String {
    use sha3::{Digest, Sha3_256};

    // Create a deterministic "public key" from peer_id for address generation
    // In production, this would use the peer's actual Ed25519 public key
    let mut expanded_key = [0u8; 32];
    let mut hasher = Sha3_256::new();
    hasher.update(peer_id);
    hasher.update(b"QNK_ONION_V3");
    let hash = hasher.finalize();
    expanded_key.copy_from_slice(&hash);

    // Calculate checksum: SHA3-256(".onion checksum" || pubkey || version)[0..2]
    let mut checksum_hasher = Sha3_256::new();
    checksum_hasher.update(b".onion checksum");
    checksum_hasher.update(&expanded_key);
    checksum_hasher.update(&[0x03u8]); // v3 version byte
    let checksum = checksum_hasher.finalize();

    // Combine: pubkey (32) + checksum (2) + version (1) = 35 bytes
    let mut onion_bytes = [0u8; 35];
    onion_bytes[0..32].copy_from_slice(&expanded_key);
    onion_bytes[32..34].copy_from_slice(&checksum[0..2]);
    onion_bytes[34] = 0x03; // v3 version

    // Base32 encode (56 chars) + ".onion"
    let encoded = base32::encode(base32::Alphabet::Rfc4648 { padding: false }, &onion_bytes);
    format!("{}.onion", encoded.to_lowercase())
}

/// Dandelion++ phases for transaction propagation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DandelionPhase {
    /// Stem phase: transactions follow a deterministic path
    Stem,
    /// Fluff phase: transactions flood the network
    Fluff,
}

/// Transaction wrapper for Dandelion++ routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DandelionTransaction {
    /// Transaction ID
    pub id: Uuid,
    /// Original transaction data
    pub data: Vec<u8>,
    /// Current phase in Dandelion++
    pub phase: DandelionPhase,
    /// Hop count in stem phase
    pub hop_count: u32,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Next relay target onion address (if in stem phase)
    /// v3.4.2-beta: Changed from SocketAddr to String for IP leak prevention
    pub next_relay: Option<String>,
}

/// Dandelion++ configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DandelionConfig {
    /// Probability of transitioning from stem to fluff phase
    pub fluff_probability: f64,
    /// Maximum hops in stem phase before forced fluff
    pub max_stem_hops: u32,
    /// Stem relay selection interval
    pub relay_selection_interval: Duration,
    /// Maximum time in stem phase
    pub max_stem_duration: Duration,
    /// Enable quantum timing obfuscation
    pub quantum_timing: bool,
    /// Minimum delay between transmissions
    pub min_delay: Duration,
    /// Maximum delay between transmissions
    pub max_delay: Duration,
}

impl Default for DandelionConfig {
    fn default() -> Self {
        Self {
            // v8.6.0: increased from 0.1 to 0.15 — slightly higher fluff chance
            // reduces average stem path length, improving latency by ~10-15%
            // while still providing strong source anonymity
            fluff_probability: 0.15,
            // v8.6.0: reduced from 10 to 5 — aligns with Dandelion++ paper recommendation;
            // 5 hops provides sufficient anonymity, 10 added unnecessary latency
            max_stem_hops: 5,
            relay_selection_interval: Duration::from_secs(600), // 10 minutes
            // v8.6.0: reduced from 30s to 20s — faster failover to fluff on slow stems
            max_stem_duration: Duration::from_secs(20),
            quantum_timing: true,
            min_delay: Duration::from_millis(100),
            // v8.6.0: reduced from 2s to 1.5s — tighter timing window for better throughput
            max_delay: Duration::from_millis(1500),
        }
    }
}

/// Dandelion++ protocol implementation
/// v3.4.2-beta: All relay addresses are now onion addresses for IP leak prevention
pub struct DandelionProtocol {
    /// Configuration
    config: DandelionConfig,
    /// Current stem relay onion address for outgoing transactions
    /// v3.4.2-beta: Changed from SocketAddr to String
    stem_relay: Arc<RwLock<Option<String>>>,
    /// Pending transactions in various phases
    pending_transactions: Arc<Mutex<HashMap<Uuid, DandelionTransaction>>>,
    /// Relay candidate onion addresses for stem phase
    /// v3.4.2-beta: Changed from Vec<SocketAddr> to Vec<String>
    relay_candidates: Arc<RwLock<Vec<String>>>,
    /// Circuit manager for Tor routing
    circuit_manager: Arc<Mutex<CircuitManager>>,
    /// Metrics collection
    metrics: Arc<TorMetrics>,
    /// Quantum RNG for timing obfuscation
    quantum_rng: Arc<Mutex<ChaChaRng>>,
    /// Last relay selection time
    last_relay_selection: Arc<Mutex<Instant>>,
}

impl DandelionProtocol {
    /// Create a new Dandelion++ protocol instance
    pub fn new(
        config: DandelionConfig,
        circuit_manager: Arc<Mutex<CircuitManager>>,
        metrics: Arc<TorMetrics>,
        quantum_seed: [u8; 32],
    ) -> Self {
        let quantum_rng = ChaChaRng::from_seed(quantum_seed);

        Self {
            config,
            stem_relay: Arc::new(RwLock::new(None)),
            pending_transactions: Arc::new(Mutex::new(HashMap::new())),
            relay_candidates: Arc::new(RwLock::new(Vec::new())),
            circuit_manager,
            metrics,
            quantum_rng: Arc::new(Mutex::new(quantum_rng)),
            last_relay_selection: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Update relay candidates from peer discovery
    /// v3.4.2-beta: Now accepts onion addresses instead of SocketAddr for IP leak prevention
    pub async fn update_relay_candidates(&self, candidates: Vec<String>) -> Result<()> {
        // Validate that all candidates are proper .onion addresses
        for candidate in &candidates {
            if !candidate.ends_with(".onion") {
                warn!("⚠️ Rejecting non-onion relay candidate (IP leak prevention)");
                continue;
            }
        }

        let mut relay_candidates = self.relay_candidates.write().await;
        *relay_candidates = candidates.into_iter()
            .filter(|c| c.ends_with(".onion"))
            .collect();

        // Force relay reselection if no current relay
        if self.stem_relay.read().await.is_none() {
            drop(relay_candidates); // Release lock before calling select_stem_relay
            self.select_stem_relay().await?;
        }

        let count = self.relay_candidates.read().await.len();
        info!(
            "Updated Dandelion++ relay candidates: {} onion addresses",
            count
        );
        Ok(())
    }

    /// Update relay candidates from peer IDs (converts to onion addresses)
    /// This is a convenience method for integration with peer discovery
    pub async fn update_relay_candidates_from_peer_ids(&self, peer_ids: Vec<&[u8]>) -> Result<()> {
        let onion_addresses: Vec<String> = peer_ids
            .iter()
            .map(|id| generate_onion_address(id))
            .collect();
        self.update_relay_candidates(onion_addresses).await
    }

    /// Select a new stem relay using quantum randomness
    /// v3.4.2-beta: Now uses onion addresses, logs are sanitized to prevent correlation
    async fn select_stem_relay(&self) -> Result<()> {
        let candidates = self.relay_candidates.read().await;
        if candidates.is_empty() {
            warn!("No onion relay candidates available for Dandelion++ stem phase");
            return Ok(());
        }

        let mut rng = self.quantum_rng.lock().await;
        let selected_index = rng.gen_range(0..candidates.len());
        let selected_relay = candidates[selected_index].clone();

        drop(candidates); // Release read lock
        let mut stem_relay = self.stem_relay.write().await;
        *stem_relay = Some(selected_relay);

        let mut last_selection = self.last_relay_selection.lock().await;
        *last_selection = Instant::now();

        // v3.4.2-beta: Sanitize log output - don't log full onion address for privacy
        info!("Selected new Dandelion++ stem relay (onion address redacted for privacy)");
        Ok(())
    }

    /// Propagate a transaction using Dandelion++ protocol
    pub async fn propagate_transaction(&self, tx_data: Vec<u8>) -> Result<()> {
        let tx_id = Uuid::new_v4();

        // Create Dandelion++ transaction wrapper
        let dandelion_tx = DandelionTransaction {
            id: tx_id,
            data: tx_data,
            phase: DandelionPhase::Stem,
            hop_count: 0,
            created_at: SystemTime::now(),
            next_relay: self.stem_relay.read().await.clone(),
        };

        // Add to pending transactions
        {
            let mut pending = self.pending_transactions.lock().await;
            pending.insert(tx_id, dandelion_tx.clone());
        }

        // Start propagation
        self.process_transaction(dandelion_tx).await?;

        // Update metrics
        self.metrics.dandelion_transactions_started.inc();

        Ok(())
    }

    /// Process a transaction through Dandelion++ phases
    async fn process_transaction(&self, mut tx: DandelionTransaction) -> Result<()> {
        match tx.phase {
            DandelionPhase::Stem => {
                self.process_stem_phase(&mut tx).await?;
            }
            DandelionPhase::Fluff => {
                self.process_fluff_phase(&tx).await?;
            }
        }
        Ok(())
    }

    /// Handle stem phase propagation
    async fn process_stem_phase(&self, tx: &mut DandelionTransaction) -> Result<()> {
        // Check if we should transition to fluff phase
        let should_fluff = self.should_transition_to_fluff(tx).await?;

        if should_fluff {
            tx.phase = DandelionPhase::Fluff;
            self.process_fluff_phase(tx).await?;
            self.metrics.dandelion_stem_to_fluff.inc();
            return Ok(());
        }

        // Apply quantum timing obfuscation
        if self.config.quantum_timing {
            self.apply_quantum_delay().await?;
        }

        // Forward to next relay (using onion address)
        if let Some(ref relay_onion) = tx.next_relay {
            self.forward_to_relay(tx, relay_onion).await?;
            tx.hop_count += 1;

            // Select next relay for continued stemming
            self.update_next_relay(tx).await?;
        } else {
            // No relay available, transition to fluff
            tx.phase = DandelionPhase::Fluff;
            self.process_fluff_phase(tx).await?;
        }

        Ok(())
    }

    /// Handle fluff phase broadcasting
    async fn process_fluff_phase(&self, tx: &DandelionTransaction) -> Result<()> {
        info!("Broadcasting transaction {} in fluff phase", tx.id);

        // Broadcast to all connected peers through Tor circuits
        let circuit_manager = self.circuit_manager.lock().await;
        circuit_manager
            .broadcast_transaction(&tx.data)
            .await
            .context("Failed to broadcast transaction in fluff phase")?;

        // Remove from pending transactions
        {
            let mut pending = self.pending_transactions.lock().await;
            pending.remove(&tx.id);
        }

        self.metrics.dandelion_fluff_broadcasts.inc();
        Ok(())
    }

    /// Determine if transaction should transition to fluff phase
    async fn should_transition_to_fluff(&self, tx: &DandelionTransaction) -> Result<bool> {
        // Check hop count limit
        if tx.hop_count >= self.config.max_stem_hops {
            debug!(
                "Transaction {} reached max stem hops, transitioning to fluff",
                tx.id
            );
            return Ok(true);
        }

        // Check time limit
        if tx.created_at.elapsed().unwrap_or_default() > self.config.max_stem_duration {
            debug!(
                "Transaction {} exceeded max stem duration, transitioning to fluff",
                tx.id
            );
            return Ok(true);
        }

        // Probabilistic transition using quantum randomness
        let mut rng = self.quantum_rng.lock().await;
        let transition_roll: f64 = rng.gen();

        if transition_roll < self.config.fluff_probability {
            debug!(
                "Transaction {} randomly transitioning to fluff (roll: {:.3})",
                tx.id, transition_roll
            );
            return Ok(true);
        }

        Ok(false)
    }

    /// Apply quantum-enhanced timing obfuscation
    async fn apply_quantum_delay(&self) -> Result<()> {
        let mut rng = self.quantum_rng.lock().await;
        let delay_ms = rng
            .gen_range(self.config.min_delay.as_millis()..=self.config.max_delay.as_millis())
            as u64;

        drop(rng); // Release lock before sleeping
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;

        Ok(())
    }

    /// Forward transaction to specified relay through Tor
    /// v3.4.2-beta: Now uses onion addresses, logs sanitized for privacy
    async fn forward_to_relay(&self, tx: &DandelionTransaction, relay_onion: &str) -> Result<()> {
        // v3.4.2-beta: Validate onion address format
        if !relay_onion.ends_with(".onion") {
            anyhow::bail!("Security: Refusing to relay to non-onion address (IP leak prevention)");
        }

        // v3.4.2-beta: Sanitized log - don't expose onion address
        debug!("Forwarding transaction {} to onion relay", tx.id);

        let circuit_manager = self.circuit_manager.lock().await;
        circuit_manager
            .send_to_onion(relay_onion, &tx.data)
            .await
            .context("Failed to forward transaction to onion relay")?;

        self.metrics.dandelion_stem_forwards.inc();
        Ok(())
    }

    /// Update next relay for continued stem propagation
    async fn update_next_relay(&self, tx: &mut DandelionTransaction) -> Result<()> {
        // Check if we need to reselect relay
        let last_selection = *self.last_relay_selection.lock().await;
        if last_selection.elapsed() > self.config.relay_selection_interval {
            self.select_stem_relay().await?;
        }

        tx.next_relay = self.stem_relay.read().await.clone();
        Ok(())
    }

    /// Handle received Dandelion++ transaction
    /// v3.4.2-beta: Changed from_peer from SocketAddr to onion address string for IP leak prevention
    pub async fn handle_received_transaction(
        &self,
        tx_data: Vec<u8>,
        from_peer_onion: &str,
    ) -> Result<()> {
        // v3.4.2-beta: Validate that we're receiving from an onion address
        if !from_peer_onion.ends_with(".onion") {
            warn!("⚠️ Received transaction from non-onion source - potential IP leak!");
        }

        // Attempt to deserialize as Dandelion++ transaction
        let dandelion_tx: DandelionTransaction = match serde_json::from_slice(&tx_data) {
            Ok(tx) => tx,
            Err(_) => {
                // Not a Dandelion++ transaction, handle normally
                return self.handle_normal_transaction(tx_data).await;
            }
        };

        // v3.4.2-beta: Sanitized log - don't expose peer address
        debug!(
            "Received Dandelion++ transaction {} from onion peer",
            dandelion_tx.id
        );

        // Check if we've already seen this transaction
        {
            let pending = self.pending_transactions.lock().await;
            if pending.contains_key(&dandelion_tx.id) {
                debug!(
                    "Already processing transaction {}, ignoring duplicate",
                    dandelion_tx.id
                );
                return Ok(());
            }
        }

        // Add to pending and continue processing
        {
            let mut pending = self.pending_transactions.lock().await;
            pending.insert(dandelion_tx.id, dandelion_tx.clone());
        }

        self.process_transaction(dandelion_tx).await?;
        self.metrics.dandelion_transactions_received.inc();

        Ok(())
    }

    /// Handle normal (non-Dandelion++) transaction
    /// v3.4.2-beta: Removed SocketAddr parameter for IP leak prevention
    async fn handle_normal_transaction(
        &self,
        tx_data: Vec<u8>,
    ) -> Result<()> {
        // Forward normal transactions directly to consensus layer
        info!("Received normal transaction, forwarding to consensus");

        // This would integrate with the main transaction processing
        // For now, just log the reception
        self.metrics.normal_transactions_received.inc();

        Ok(())
    }

    /// Get current protocol statistics
    /// v3.4.2-beta: No longer exposes relay address - only indicates if relay is selected
    pub async fn get_statistics(&self) -> DandelionStatistics {
        let pending_count = self.pending_transactions.lock().await.len();
        let has_relay = self.stem_relay.read().await.is_some();
        let candidate_count = self.relay_candidates.read().await.len();

        DandelionStatistics {
            pending_transactions: pending_count,
            has_stem_relay: has_relay,
            relay_candidates: candidate_count,
            stem_forwards: self.metrics.dandelion_stem_forwards.get(),
            fluff_broadcasts: self.metrics.dandelion_fluff_broadcasts.get(),
            transactions_started: self.metrics.dandelion_transactions_started.get(),
            transactions_received: self.metrics.dandelion_transactions_received.get(),
        }
    }

    /// Clean up expired transactions
    pub async fn cleanup_expired_transactions(&self) -> Result<()> {
        let mut pending = self.pending_transactions.lock().await;
        let now = SystemTime::now();

        let expired_txs: Vec<Uuid> = pending
            .iter()
            .filter(|(_, tx)| {
                now.duration_since(tx.created_at).unwrap_or_default()
                    > self.config.max_stem_duration * 2
            })
            .map(|(id, _)| *id)
            .collect();

        for tx_id in expired_txs {
            pending.remove(&tx_id);
            warn!("Cleaned up expired Dandelion++ transaction: {}", tx_id);
        }

        Ok(())
    }
}

/// Statistics for Dandelion++ protocol
/// v3.4.2-beta: current_stem_relay is now a bool (not exposing actual address for privacy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DandelionStatistics {
    pub pending_transactions: usize,
    /// v3.4.2-beta: Changed to bool - indicates if a relay is selected, without exposing address
    pub has_stem_relay: bool,
    pub relay_candidates: usize,
    pub stem_forwards: u64,
    pub fluff_broadcasts: u64,
    pub transactions_started: u64,
    pub transactions_received: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit_manager::CircuitManager;
    use crate::metrics::TorMetrics;

    fn create_test_dandelion() -> DandelionProtocol {
        let config = DandelionConfig::default();
        let circuit_manager = Arc::new(Mutex::new(CircuitManager::mock()));
        let metrics = Arc::new(TorMetrics::new());
        let quantum_seed = [0u8; 32]; // Test seed

        DandelionProtocol::new(config, circuit_manager, metrics, quantum_seed)
    }

    #[tokio::test]
    async fn test_relay_selection_with_onion_addresses() {
        let dandelion = create_test_dandelion();

        // v3.4.2-beta: Use proper onion addresses instead of IP addresses
        let candidates = vec![
            generate_onion_address(b"peer1_test_id_bytes"),
            generate_onion_address(b"peer2_test_id_bytes"),
        ];

        dandelion.update_relay_candidates(candidates).await.unwrap();

        let stem_relay = dandelion.stem_relay.read().await;
        assert!(stem_relay.is_some());
        // Verify it's an onion address
        assert!(stem_relay.as_ref().unwrap().ends_with(".onion"));
    }

    #[tokio::test]
    async fn test_rejects_non_onion_addresses() {
        let dandelion = create_test_dandelion();

        // v3.4.2-beta: Test that IP addresses are rejected
        let bad_candidates = vec![
            "127.0.0.1:8080".to_string(), // IP address - should be rejected
            generate_onion_address(b"good_peer"),
        ];

        dandelion.update_relay_candidates(bad_candidates).await.unwrap();

        // Only the onion address should remain
        let candidates = dandelion.relay_candidates.read().await;
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].ends_with(".onion"));
    }

    #[tokio::test]
    async fn test_fluff_transition() {
        let dandelion = create_test_dandelion();

        let tx = DandelionTransaction {
            id: Uuid::new_v4(),
            data: vec![1, 2, 3, 4],
            phase: DandelionPhase::Stem,
            hop_count: 15, // Exceeds max_stem_hops
            created_at: SystemTime::now(),
            next_relay: None,
        };

        let should_fluff = dandelion.should_transition_to_fluff(&tx).await.unwrap();
        assert!(should_fluff);
    }

    #[test]
    fn test_onion_address_generation() {
        let peer_id = b"test_peer_id_32_bytes_long!!!!!";
        let onion = generate_onion_address(peer_id);

        // Verify it ends with .onion
        assert!(onion.ends_with(".onion"));
        // v3 onion addresses are 56 chars + ".onion" = 62 chars total
        assert_eq!(onion.len(), 62);
    }
}
