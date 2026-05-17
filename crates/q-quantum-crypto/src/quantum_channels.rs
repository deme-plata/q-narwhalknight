//! 📡 Quantum Channel Management System
//! Manages quantum communication channels for QKD and quantum signatures

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{mpsc, RwLock};
use tokio::time::sleep;

use crate::quantum_entropy::QuantumEntropySource;
use crate::{NodeId, PhotonState};

/// Quantum channel state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChannelState {
    /// Channel initialization
    Initializing,
    /// Channel calibration in progress
    Calibrating,
    /// Channel ready for quantum communication
    Active,
    /// Channel experiencing errors
    Degraded,
    /// Channel temporarily unavailable
    Maintenance,
    /// Channel permanently failed
    Failed,
}

/// Quantum channel quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelQuality {
    /// Bit error rate (QBER)
    pub bit_error_rate: f64,
    /// Channel loss rate
    pub loss_rate: f64,
    /// Signal-to-noise ratio
    pub snr_db: f64,
    /// Channel coherence time (microseconds)
    pub coherence_time_us: f64,
    /// Dark count rate (counts per second)
    pub dark_count_rate: f64,
    /// Measurement time
    pub measured_at: SystemTime,
}

impl ChannelQuality {
    /// Check if channel quality is acceptable for QKD
    pub fn is_acceptable_for_qkd(&self) -> bool {
        self.bit_error_rate < 0.11 && // Standard QKD threshold
        self.loss_rate < 0.5 &&       // Reasonable loss threshold
        self.snr_db > 10.0 // Minimum SNR
    }

    /// Get overall quality score (0.0 - 1.0)
    pub fn quality_score(&self) -> f64 {
        let ber_score = (0.11 - self.bit_error_rate.min(0.11)) / 0.11;
        let loss_score = (0.5 - self.loss_rate.min(0.5)) / 0.5;
        let snr_score = (self.snr_db.min(30.0)) / 30.0;

        (ber_score + loss_score + snr_score) / 3.0
    }
}

/// Quantum channel implementation
#[derive(Debug)]
pub struct QuantumChannel {
    /// Channel identifier
    pub channel_id: String,
    /// Source node
    pub source_node: NodeId,
    /// Target node  
    pub target_node: NodeId,
    /// Channel state
    pub state: Arc<RwLock<ChannelState>>,
    /// Channel quality metrics
    pub quality: Arc<RwLock<ChannelQuality>>,
    /// Photon transmission queue
    photon_tx: mpsc::Sender<PhotonMessage>,
    /// Photon reception queue
    photon_rx: Arc<RwLock<mpsc::Receiver<PhotonMessage>>>,
    /// Classical communication channel
    classical_tx: mpsc::Sender<ClassicalMessage>,
    classical_rx: Arc<RwLock<mpsc::Receiver<ClassicalMessage>>>,
    /// Channel statistics
    stats: Arc<RwLock<ChannelStats>>,
}

impl QuantumChannel {
    /// Create new quantum channel
    pub async fn new(
        channel_id: String,
        source_node: NodeId,
        target_node: NodeId,
        entropy_source: Arc<QuantumEntropySource>,
    ) -> Result<Self> {
        let (photon_tx, photon_rx) = mpsc::channel(10000);
        let (classical_tx, classical_rx) = mpsc::channel(1000);

        let channel = Self {
            channel_id,
            source_node,
            target_node,
            state: Arc::new(RwLock::new(ChannelState::Initializing)),
            quality: Arc::new(RwLock::new(ChannelQuality {
                bit_error_rate: 0.05,
                loss_rate: 0.1,
                snr_db: 15.0,
                coherence_time_us: 100.0,
                dark_count_rate: 100.0,
                measured_at: SystemTime::now(),
            })),
            photon_tx,
            photon_rx: Arc::new(RwLock::new(photon_rx)),
            classical_tx,
            classical_rx: Arc::new(RwLock::new(classical_rx)),
            stats: Arc::new(RwLock::new(ChannelStats {
                photons_sent: 0,
                photons_received: 0,
                classical_messages_sent: 0,
                classical_messages_received: 0,
                last_transmission: SystemTime::UNIX_EPOCH,
                last_reception: SystemTime::UNIX_EPOCH,
            })),
        };

        // Start channel calibration
        channel.calibrate_channel().await?;

        Ok(channel)
    }

    /// Calibrate quantum channel
    async fn calibrate_channel(&self) -> Result<()> {
        *self.state.write().await = ChannelState::Calibrating;

        // Simulate channel calibration process
        sleep(Duration::from_millis(100)).await;

        // Update channel quality based on calibration
        {
            let mut quality = self.quality.write().await;
            quality.bit_error_rate = 0.02 + (rand::random::<f64>() * 0.05);
            quality.loss_rate = 0.05 + (rand::random::<f64>() * 0.1);
            quality.snr_db = 15.0 + (rand::random::<f64>() * 10.0);
            quality.measured_at = SystemTime::now();
        }

        *self.state.write().await = ChannelState::Active;
        Ok(())
    }

    /// Send photons through quantum channel
    pub async fn send_photons(&self, photons: &[PhotonState]) -> Result<()> {
        let state = self.state.read().await.clone();
        if state != ChannelState::Active {
            return Err(anyhow::anyhow!("Channel not active: {:?}", state));
        }

        for photon in photons {
            let message = PhotonMessage {
                photon: photon.clone(),
                timestamp: SystemTime::now(),
                channel_id: self.channel_id.clone(),
            };

            // Simulate quantum channel transmission with errors
            let transmitted_photon = self.simulate_quantum_transmission(&message.photon).await?;

            let transmitted_message = PhotonMessage {
                photon: transmitted_photon,
                timestamp: message.timestamp,
                channel_id: message.channel_id,
            };

            self.photon_tx
                .send(transmitted_message)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send photon: {}", e))?;
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.photons_sent += photons.len() as u64;
            stats.last_transmission = SystemTime::now();
        }

        Ok(())
    }

    /// Receive photons from quantum channel
    pub async fn receive_photons(&self) -> Result<Vec<PhotonState>> {
        let mut photons = Vec::new();
        let mut receiver = self.photon_rx.write().await;

        // Collect all available photons
        while let Ok(message) = receiver.try_recv() {
            photons.push(message.photon);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.photons_received += photons.len() as u64;
            stats.last_reception = SystemTime::now();
        }

        Ok(photons)
    }

    /// Send classical message (for basis reconciliation, etc.)
    pub async fn send_classical_message(
        &self,
        message_type: ClassicalMessageType,
        data: Vec<u8>,
    ) -> Result<()> {
        let message = ClassicalMessage {
            message_type,
            data,
            timestamp: SystemTime::now(),
            source_node: self.source_node,
            target_node: self.target_node,
        };

        self.classical_tx
            .send(message)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send classical message: {}", e))?;

        Ok(())
    }

    /// Receive classical message
    pub async fn receive_classical_message(
        &self,
        expected_type: ClassicalMessageType,
    ) -> Result<Vec<u8>> {
        let mut receiver = self.classical_rx.write().await;

        // Wait for message with timeout
        let timeout = Duration::from_secs(30);
        let message = tokio::time::timeout(timeout, receiver.recv())
            .await
            .map_err(|_| anyhow::anyhow!("Timeout waiting for classical message"))?
            .ok_or_else(|| anyhow::anyhow!("Channel closed"))?;

        if message.message_type != expected_type {
            return Err(anyhow::anyhow!(
                "Unexpected message type: {:?}",
                message.message_type
            ));
        }

        Ok(message.data)
    }

    /// Simulate quantum transmission through noisy channel
    async fn simulate_quantum_transmission(&self, photon: &PhotonState) -> Result<PhotonState> {
        let quality = self.quality.read().await;

        // Simulate bit flip errors
        if rand::random::<f64>() < quality.bit_error_rate {
            let flipped_photon = match photon {
                PhotonState::Rectilinear(bit) => PhotonState::Rectilinear(!bit),
                PhotonState::Diagonal(bit) => PhotonState::Diagonal(!bit),
            };
            return Ok(flipped_photon);
        }

        // Simulate basis rotation errors (rare)
        if rand::random::<f64>() < 0.001 {
            let rotated_photon = match photon {
                PhotonState::Rectilinear(bit) => PhotonState::Diagonal(*bit),
                PhotonState::Diagonal(bit) => PhotonState::Rectilinear(*bit),
            };
            return Ok(rotated_photon);
        }

        Ok(photon.clone())
    }

    /// Monitor channel quality continuously
    pub async fn monitor_quality(&self) -> Result<()> {
        loop {
            sleep(Duration::from_secs(10)).await; // Monitor every 10 seconds

            // Simulate quality measurement
            let mut quality = self.quality.write().await;

            // Add some variation to simulate real channel fluctuations
            quality.bit_error_rate += (rand::random::<f64>() - 0.5) * 0.001;
            quality.bit_error_rate = quality.bit_error_rate.max(0.001).min(0.15);

            quality.loss_rate += (rand::random::<f64>() - 0.5) * 0.01;
            quality.loss_rate = quality.loss_rate.max(0.01).min(0.6);

            quality.snr_db += (rand::random::<f64>() - 0.5) * 2.0;
            quality.snr_db = quality.snr_db.max(5.0).min(35.0);

            quality.measured_at = SystemTime::now();

            // Update channel state based on quality
            let new_state = if quality.is_acceptable_for_qkd() {
                ChannelState::Active
            } else if quality.bit_error_rate > 0.15 {
                ChannelState::Failed
            } else {
                ChannelState::Degraded
            };

            *self.state.write().await = new_state;
        }
    }

    /// Get channel statistics
    pub async fn get_stats(&self) -> ChannelStats {
        self.stats.read().await.clone()
    }

    /// Get channel quality
    pub async fn get_quality(&self) -> ChannelQuality {
        self.quality.read().await.clone()
    }

    /// Get channel state
    pub async fn get_state(&self) -> ChannelState {
        self.state.read().await.clone()
    }
}

/// Quantum channel manager
#[derive(Debug)]
pub struct QuantumChannelManager {
    /// Node identifier
    node_id: NodeId,
    /// Entropy source
    entropy_source: Arc<QuantumEntropySource>,
    /// Active channels
    channels: Arc<RwLock<HashMap<String, Arc<QuantumChannel>>>>,
    /// Channel establishment history
    establishment_history: Arc<RwLock<Vec<ChannelEstablishmentRecord>>>,
}

impl QuantumChannelManager {
    /// Create new quantum channel manager
    pub async fn new(node_id: NodeId, entropy_source: Arc<QuantumEntropySource>) -> Result<Self> {
        Ok(Self {
            node_id,
            entropy_source,
            channels: Arc::new(RwLock::new(HashMap::new())),
            establishment_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Establish quantum channel with peer
    pub async fn establish_channel(&self, peer_id: NodeId) -> Result<Arc<QuantumChannel>> {
        let channel_id = format!(
            "{}_{}",
            hex::encode(&self.node_id[..8]),
            hex::encode(&peer_id[..8])
        );

        // Check if channel already exists
        {
            let channels = self.channels.read().await;
            if let Some(existing_channel) = channels.get(&channel_id) {
                let state = existing_channel.get_state().await;
                if state == ChannelState::Active || state == ChannelState::Degraded {
                    return Ok(existing_channel.clone());
                }
            }
        }

        // Create new channel
        let channel = Arc::new(
            QuantumChannel::new(
                channel_id.clone(),
                self.node_id,
                peer_id,
                self.entropy_source.clone(),
            )
            .await?,
        );

        // Start quality monitoring
        let monitor_channel = channel.clone();
        tokio::spawn(async move {
            let _ = monitor_channel.monitor_quality().await;
        });

        // Store channel
        self.channels
            .write()
            .await
            .insert(channel_id.clone(), channel.clone());

        // Record establishment
        let record = ChannelEstablishmentRecord {
            channel_id,
            peer_id,
            established_at: SystemTime::now(),
            initial_quality: channel.get_quality().await.quality_score(),
        };
        self.establishment_history.write().await.push(record);

        Ok(channel)
    }

    /// Get channel by peer ID
    pub async fn get_channel(&self, peer_id: NodeId) -> Option<Arc<QuantumChannel>> {
        let channel_id = format!(
            "{}_{}",
            hex::encode(&self.node_id[..8]),
            hex::encode(&peer_id[..8])
        );
        self.channels.read().await.get(&channel_id).cloned()
    }

    /// Get all active channels
    pub async fn get_active_channels(&self) -> Vec<Arc<QuantumChannel>> {
        let channels = self.channels.read().await;
        let mut active_channels = Vec::new();

        for channel in channels.values() {
            let state = channel.get_state().await;
            if state == ChannelState::Active || state == ChannelState::Degraded {
                active_channels.push(channel.clone());
            }
        }

        active_channels
    }

    /// Get active channel count
    pub async fn get_active_channel_count(&self) -> usize {
        self.get_active_channels().await.len()
    }

    /// Close channel
    pub async fn close_channel(&self, peer_id: NodeId) -> Result<()> {
        let channel_id = format!(
            "{}_{}",
            hex::encode(&self.node_id[..8]),
            hex::encode(&peer_id[..8])
        );

        if let Some(channel) = self.channels.write().await.remove(&channel_id) {
            *channel.state.write().await = ChannelState::Failed;
        }

        Ok(())
    }

    /// Health check for all channels
    pub async fn health_check(&self) -> Result<bool> {
        let channels = self.channels.read().await;

        for channel in channels.values() {
            let state = channel.get_state().await;
            let quality = channel.get_quality().await;

            if state == ChannelState::Failed || !quality.is_acceptable_for_qkd() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get channel manager statistics
    pub async fn get_manager_stats(&self) -> ChannelManagerStats {
        let channels = self.channels.read().await;
        let mut total_photons_sent = 0;
        let mut total_photons_received = 0;
        let mut active_count = 0;
        let mut failed_count = 0;

        for channel in channels.values() {
            let stats = channel.get_stats().await;
            let state = channel.get_state().await;

            total_photons_sent += stats.photons_sent;
            total_photons_received += stats.photons_received;

            match state {
                ChannelState::Active | ChannelState::Degraded => active_count += 1,
                ChannelState::Failed => failed_count += 1,
                _ => {}
            }
        }

        ChannelManagerStats {
            total_channels: channels.len(),
            active_channels: active_count,
            failed_channels: failed_count,
            total_photons_sent,
            total_photons_received,
            establishment_history: self.establishment_history.read().await.len(),
        }
    }
}

/// Photon message for quantum communication
#[derive(Debug, Clone)]
struct PhotonMessage {
    photon: PhotonState,
    timestamp: SystemTime,
    channel_id: String,
}

/// Classical message types for QKD protocol
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassicalMessageType {
    /// Basis announcement in BB84
    BasisAnnouncement,
    /// Test bits for error estimation
    TestBits,
    /// Error rate announcement
    ErrorRate,
    /// Matching basis indices
    MatchingIndices,
    /// Error correction data
    ErrorCorrection,
    /// Privacy amplification parameters
    PrivacyAmplification,
}

/// Classical message for QKD coordination
#[derive(Debug, Clone)]
struct ClassicalMessage {
    message_type: ClassicalMessageType,
    data: Vec<u8>,
    timestamp: SystemTime,
    source_node: NodeId,
    target_node: NodeId,
}

/// Channel statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelStats {
    pub photons_sent: u64,
    pub photons_received: u64,
    pub classical_messages_sent: u64,
    pub classical_messages_received: u64,
    pub last_transmission: SystemTime,
    pub last_reception: SystemTime,
}

/// Channel establishment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelEstablishmentRecord {
    pub channel_id: String,
    pub peer_id: NodeId,
    pub established_at: SystemTime,
    pub initial_quality: f64,
}

/// Channel manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelManagerStats {
    pub total_channels: usize,
    pub active_channels: usize,
    pub failed_channels: usize,
    pub total_photons_sent: u64,
    pub total_photons_received: u64,
    pub establishment_history: usize,
}

// Implement QKD channel trait for QuantumChannel
#[async_trait::async_trait]
impl crate::qkd::QKDChannel for QuantumChannel {
    async fn send_photons(&self, photons: &[PhotonState]) -> Result<()> {
        self.send_photons(photons).await
    }

    async fn receive_photons(&self) -> Result<Vec<PhotonState>> {
        self.receive_photons().await
    }

    async fn send_basis_announcement(&self, bases: &[bool]) -> Result<()> {
        let data = bases.iter().map(|&b| b as u8).collect();
        self.send_classical_message(ClassicalMessageType::BasisAnnouncement, data)
            .await
    }

    async fn receive_basis_announcement(&self) -> Result<Vec<bool>> {
        let data = self
            .receive_classical_message(ClassicalMessageType::BasisAnnouncement)
            .await?;
        Ok(data.into_iter().map(|b| b != 0).collect())
    }

    async fn send_test_bits(&self, bits: &[bool]) -> Result<()> {
        let data = bits.iter().map(|&b| b as u8).collect();
        self.send_classical_message(ClassicalMessageType::TestBits, data)
            .await
    }

    async fn receive_test_bits(&self, _count: usize) -> Result<Vec<bool>> {
        let data = self
            .receive_classical_message(ClassicalMessageType::TestBits)
            .await?;
        Ok(data.into_iter().map(|b| b != 0).collect())
    }

    async fn receive_matching_indices(&self) -> Result<Vec<usize>> {
        let data = self
            .receive_classical_message(ClassicalMessageType::MatchingIndices)
            .await?;
        Ok(data
            .chunks(8)
            .map(|chunk| {
                let mut bytes = [0u8; 8];
                bytes[..chunk.len()].copy_from_slice(chunk);
                usize::from_le_bytes(bytes)
            })
            .collect())
    }

    async fn receive_error_rate(&self) -> Result<f64> {
        let data = self
            .receive_classical_message(ClassicalMessageType::ErrorRate)
            .await?;
        if data.len() >= 8 {
            Ok(f64::from_le_bytes([
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ]))
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_entropy::QuantumEntropySource;

    #[tokio::test]
    async fn test_channel_quality() {
        let quality = ChannelQuality {
            bit_error_rate: 0.05,
            loss_rate: 0.1,
            snr_db: 15.0,
            coherence_time_us: 100.0,
            dark_count_rate: 100.0,
            measured_at: SystemTime::now(),
        };

        assert!(quality.is_acceptable_for_qkd());
        assert!(quality.quality_score() > 0.5);
    }

    #[tokio::test]
    async fn test_quantum_channel_creation() {
        let entropy_source = Arc::new(QuantumEntropySource::new().await.unwrap());
        let source_node = [1u8; 32];
        let target_node = [2u8; 32];

        let channel = QuantumChannel::new(
            "test_channel".to_string(),
            source_node,
            target_node,
            entropy_source,
        )
        .await
        .unwrap();

        assert_eq!(channel.source_node, source_node);
        assert_eq!(channel.target_node, target_node);
        assert_eq!(channel.get_state().await, ChannelState::Active);
    }

    #[tokio::test]
    async fn test_channel_manager() {
        let node_id = [1u8; 32];
        let peer_id = [2u8; 32];
        let entropy_source = Arc::new(QuantumEntropySource::new().await.unwrap());

        let manager = QuantumChannelManager::new(node_id, entropy_source)
            .await
            .unwrap();

        let channel = manager.establish_channel(peer_id).await.unwrap();
        assert_eq!(channel.source_node, node_id);
        assert_eq!(channel.target_node, peer_id);

        let retrieved_channel = manager.get_channel(peer_id).await.unwrap();
        assert_eq!(retrieved_channel.channel_id, channel.channel_id);
    }
}
