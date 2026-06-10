/// Persistent Tor Channel Management for Q-NarwhalKnight
/// Maintains dedicated circuits per validator with automatic rotation
use anyhow::{Context, Result};
use q_tor_client::QTorClient;
use q_types::ValidatorId;

// Temporary placeholder for TorCircuitConnection
#[derive(Debug)]
pub struct TorCircuitConnection {
    pub circuit_id: String,
    pub established_at: Instant,
}
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

/// Persistent Tor channel for a specific validator peer
#[derive(Debug, Clone)]
pub struct TorChannel {
    pub peer_validator_id: ValidatorId,
    pub onion_address: String,
    pub circuit_id: u64,
    pub connection_quality: ChannelQuality,
    pub last_used: Instant,
    pub created_at: Instant,
    pub rotation_count: u64,
}

/// Quality metrics for a Tor channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelQuality {
    pub latency_ms: u32,
    pub throughput_mbps: f32,
    pub success_rate: f64,
    pub total_messages: u64,
    pub failed_messages: u64,
    #[serde(
        serialize_with = "serialize_instant",
        deserialize_with = "deserialize_instant"
    )]
    pub last_latency_check: Instant,
}

impl ChannelQuality {
    pub fn new() -> Self {
        Self {
            latency_ms: 0,
            throughput_mbps: 0.0,
            success_rate: 100.0,
            total_messages: 0,
            failed_messages: 0,
            last_latency_check: Instant::now(),
        }
    }

    pub fn record_message_success(&mut self, latency_ms: u32) {
        self.total_messages += 1;
        self.latency_ms = latency_ms;
        self.last_latency_check = Instant::now();
        self.update_success_rate();
    }

    pub fn record_message_failure(&mut self) {
        self.total_messages += 1;
        self.failed_messages += 1;
        self.update_success_rate();
    }

    fn update_success_rate(&mut self) {
        if self.total_messages > 0 {
            self.success_rate = ((self.total_messages - self.failed_messages) as f64
                / self.total_messages as f64)
                * 100.0;
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.success_rate >= 90.0 && self.latency_ms < 500
    }
}

/// Message to be sent through a Tor channel
#[derive(Debug, Clone)]
pub struct ChannelMessage {
    pub data: Vec<u8>,
    pub message_type: MessageType,
    pub priority: MessagePriority,
    pub created_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Consensus,
    Mempool,
    StateSync,
    Heartbeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Manager for persistent Tor channels to validators
pub struct PersistentChannelManager {
    channels: RwLock<HashMap<ValidatorId, TorChannel>>,
    tor_client: Arc<QTorClient>,
    active_connections: Mutex<HashMap<ValidatorId, Arc<Mutex<TorCircuitConnection>>>>,
    channel_rotation_interval: Duration,
    max_channel_age: Duration,
    local_validator_id: ValidatorId,
}

impl PersistentChannelManager {
    pub fn new(
        tor_client: Arc<QTorClient>,
        local_validator_id: ValidatorId,
        rotation_interval_hours: u64,
    ) -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            tor_client,
            active_connections: Mutex::new(HashMap::new()),
            channel_rotation_interval: Duration::from_secs(rotation_interval_hours * 3600),
            max_channel_age: Duration::from_secs(rotation_interval_hours * 3600 * 2), // 2x rotation interval
            local_validator_id,
        }
    }

    /// Establish or retrieve persistent channel to a validator
    pub async fn get_channel(
        &self,
        validator_id: ValidatorId,
        onion_address: String,
    ) -> Result<Arc<Mutex<TorCircuitConnection>>> {
        // Check if we already have an active connection
        {
            let connections = self.active_connections.lock().await;
            if let Some(connection) = connections.get(&validator_id) {
                debug!(
                    "🔗 Reusing existing channel to {}",
                    hex::encode(validator_id)
                );
                return Ok(connection.clone());
            }
        }

        // Create new connection
        let connection = self.create_new_channel(validator_id, onion_address).await?;

        {
            let mut connections = self.active_connections.lock().await;
            connections.insert(validator_id, connection.clone());
        }

        info!(
            "✅ Established persistent channel to {}",
            hex::encode(validator_id)
        );
        Ok(connection)
    }

    /// Create a new Tor channel to a validator
    async fn create_new_channel(
        &self,
        validator_id: ValidatorId,
        onion_address: String,
    ) -> Result<Arc<Mutex<TorCircuitConnection>>> {
        info!(
            "🚀 Creating new Tor channel to {} via {}",
            hex::encode(validator_id),
            onion_address
        );

        let start_time = Instant::now();

        // Connect through Tor client
        let tor_connection = self
            .tor_client
            .connect_to_peer(&onion_address)
            .await
            .context("Failed to establish Tor connection")?;

        let latency = start_time.elapsed().as_millis() as u32;
        let circuit_id = tor_connection.get_circuit_id();

        // Create channel record
        let channel = TorChannel {
            peer_validator_id: validator_id,
            onion_address: onion_address.clone(),
            circuit_id,
            connection_quality: ChannelQuality::new(),
            last_used: Instant::now(),
            created_at: Instant::now(),
            rotation_count: 0,
        };

        // Store channel metadata
        {
            let mut channels = self.channels.write().await;
            channels.insert(validator_id, channel);
        }

        // Update initial latency
        self.update_channel_quality(validator_id, latency, true)
            .await?;

        // Convert TorConnection to TorCircuitConnection
        let circuit_connection = TorCircuitConnection {
            circuit_id: tor_connection.get_circuit_id().to_string(),
            established_at: Instant::now(),
        };

        let connection = Arc::new(Mutex::new(circuit_connection));
        Ok(connection)
    }

    /// Send message through persistent channel
    pub async fn send_message(
        &self,
        validator_id: ValidatorId,
        message: ChannelMessage,
    ) -> Result<()> {
        let channel_info = {
            let channels = self.channels.read().await;
            channels
                .get(&validator_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("No channel to validator {}", hex::encode(validator_id))
                })?
                .clone()
        };

        debug!(
            "📤 Sending {:?} message to {} via channel {}",
            message.message_type,
            hex::encode(validator_id),
            channel_info.circuit_id
        );

        let start_time = Instant::now();

        // Get connection and send message
        let connection = {
            let connections = self.active_connections.lock().await;
            connections
                .get(&validator_id)
                .ok_or_else(|| anyhow::anyhow!("No active connection to validator"))?
                .clone()
        };

        // TODO: Implement actual message sending through TorCircuitConnection
        // For now, simulate the operation
        let send_result = self.simulate_message_send(&message, &connection).await;

        let latency = start_time.elapsed().as_millis() as u32;

        match send_result {
            Ok(_) => {
                self.update_channel_quality(validator_id, latency, true)
                    .await?;
                debug!("✅ Message sent successfully in {}ms", latency);
            }
            Err(e) => {
                self.update_channel_quality(validator_id, latency, false)
                    .await?;
                error!("❌ Message send failed: {}", e);
                return Err(e);
            }
        }

        // Update last used timestamp
        {
            let mut channels = self.channels.write().await;
            if let Some(channel) = channels.get_mut(&validator_id) {
                channel.last_used = Instant::now();
            }
        }

        Ok(())
    }

    /// Simulate message sending (placeholder for actual implementation)
    async fn simulate_message_send(
        &self,
        _message: &ChannelMessage,
        _connection: &Arc<Mutex<TorCircuitConnection>>,
    ) -> Result<()> {
        // Simulate network delay
        sleep(Duration::from_millis(50)).await;

        // Simulate 5% failure rate for testing
        if rand::random::<f32>() < 0.05 {
            return Err(anyhow::anyhow!("Simulated network error"));
        }

        Ok(())
    }

    /// Update channel quality metrics
    async fn update_channel_quality(
        &self,
        validator_id: ValidatorId,
        latency_ms: u32,
        success: bool,
    ) -> Result<()> {
        let mut channels = self.channels.write().await;
        if let Some(channel) = channels.get_mut(&validator_id) {
            if success {
                channel
                    .connection_quality
                    .record_message_success(latency_ms);
            } else {
                channel.connection_quality.record_message_failure();
            }
        }
        Ok(())
    }

    /// Rotate channels (called periodically, e.g., every epoch)
    pub async fn rotate_channels(&self) -> Result<()> {
        info!("🔄 Starting channel rotation process");

        let channels_to_rotate: Vec<ValidatorId> = {
            let channels = self.channels.read().await;
            let now = Instant::now();

            channels
                .iter()
                .filter(|(_, channel)| {
                    now.duration_since(channel.created_at) > self.channel_rotation_interval
                        || !channel.connection_quality.is_healthy()
                })
                .map(|(validator_id, _)| *validator_id)
                .collect()
        };

        let mut rotated_count = 0;

        for validator_id in channels_to_rotate {
            match self.rotate_single_channel(validator_id).await {
                Ok(_) => {
                    rotated_count += 1;
                    debug!(
                        "✅ Rotated channel for validator {}",
                        hex::encode(validator_id)
                    );
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to rotate channel for validator {}: {}",
                        hex::encode(validator_id),
                        e
                    );
                }
            }
        }

        info!(
            "🔄 Channel rotation complete: {} channels rotated",
            rotated_count
        );
        Ok(())
    }

    /// Rotate a single channel
    async fn rotate_single_channel(&self, validator_id: ValidatorId) -> Result<()> {
        let onion_address = {
            let channels = self.channels.read().await;
            channels
                .get(&validator_id)
                .map(|c| c.onion_address.clone())
                .ok_or_else(|| anyhow::anyhow!("Channel not found for rotation"))?
        };

        // Close existing connection
        {
            let mut connections = self.active_connections.lock().await;
            connections.remove(&validator_id);
        }

        // Create new channel
        let new_connection = self.create_new_channel(validator_id, onion_address).await?;

        {
            let mut connections = self.active_connections.lock().await;
            connections.insert(validator_id, new_connection);
        }

        // Update rotation count
        {
            let mut channels = self.channels.write().await;
            if let Some(channel) = channels.get_mut(&validator_id) {
                channel.rotation_count += 1;
            }
        }

        Ok(())
    }

    /// Get channel statistics
    pub async fn get_channel_stats(&self) -> ChannelManagerStats {
        let channels = self.channels.read().await;
        let connections = self.active_connections.lock().await;

        let total_channels = channels.len();
        let active_connections = connections.len();

        let avg_latency = if total_channels > 0 {
            channels
                .values()
                .map(|c| c.connection_quality.latency_ms)
                .sum::<u32>()
                / total_channels as u32
        } else {
            0
        };

        let healthy_channels = channels
            .values()
            .filter(|c| c.connection_quality.is_healthy())
            .count();

        let total_messages = channels
            .values()
            .map(|c| c.connection_quality.total_messages)
            .sum();

        let total_failed = channels
            .values()
            .map(|c| c.connection_quality.failed_messages)
            .sum();

        ChannelManagerStats {
            total_channels,
            active_connections,
            healthy_channels,
            average_latency_ms: avg_latency,
            total_messages_sent: total_messages,
            total_failed_messages: total_failed,
            overall_success_rate: if total_messages > 0 {
                ((total_messages - total_failed) as f64 / total_messages as f64) * 100.0
            } else {
                100.0
            },
        }
    }

    /// Clean up stale channels
    pub async fn cleanup_stale_channels(&self) -> Result<()> {
        let now = Instant::now();
        let mut stale_validators = Vec::new();

        {
            let channels = self.channels.read().await;
            for (validator_id, channel) in channels.iter() {
                if now.duration_since(channel.last_used) > self.max_channel_age {
                    stale_validators.push(*validator_id);
                }
            }
        }

        if !stale_validators.is_empty() {
            info!("🧹 Cleaning up {} stale channels", stale_validators.len());

            let mut channels = self.channels.write().await;
            let mut connections = self.active_connections.lock().await;

            for validator_id in stale_validators {
                channels.remove(&validator_id);
                connections.remove(&validator_id);
            }
        }

        Ok(())
    }

    /// Get channel health report
    pub async fn get_health_report(&self) -> Vec<ChannelHealthReport> {
        let channels = self.channels.read().await;

        channels
            .iter()
            .map(|(validator_id, channel)| ChannelHealthReport {
                validator_id: *validator_id,
                onion_address: channel.onion_address.clone(),
                is_healthy: channel.connection_quality.is_healthy(),
                latency_ms: channel.connection_quality.latency_ms,
                success_rate: channel.connection_quality.success_rate,
                age: Instant::now().duration_since(channel.created_at),
                rotation_count: channel.rotation_count,
            })
            .collect()
    }
}

/// Statistics for the channel manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelManagerStats {
    pub total_channels: usize,
    pub active_connections: usize,
    pub healthy_channels: usize,
    pub average_latency_ms: u32,
    pub total_messages_sent: u64,
    pub total_failed_messages: u64,
    pub overall_success_rate: f64,
}

/// Health report for a single channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelHealthReport {
    pub validator_id: ValidatorId,
    pub onion_address: String,
    pub is_healthy: bool,
    pub latency_ms: u32,
    pub success_rate: f64,
    pub age: Duration,
    pub rotation_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use q_tor_client::QTorClient;

    #[tokio::test]
    async fn test_channel_manager_creation() {
        let tor_client = Arc::new(QTorClient::mock());
        let validator_id = [1u8; 32];
        let manager = PersistentChannelManager::new(tor_client, validator_id, 24);

        let stats = manager.get_channel_stats().await;
        assert_eq!(stats.total_channels, 0);
        assert_eq!(stats.active_connections, 0);
    }

    #[tokio::test]
    async fn test_channel_quality_metrics() {
        let mut quality = ChannelQuality::new();

        quality.record_message_success(100);
        assert_eq!(quality.latency_ms, 100);
        assert_eq!(quality.success_rate, 100.0);

        quality.record_message_failure();
        assert_eq!(quality.success_rate, 50.0);
    }
}

// Custom serialization for Instant
fn serialize_instant<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let duration = instant.elapsed();
    let secs = duration.as_secs();
    serializer.serialize_u64(secs)
}

fn deserialize_instant<'de, D>(deserializer: D) -> Result<Instant, D::Error>
where
    D: Deserializer<'de>,
{
    let secs = u64::deserialize(deserializer)?;
    let duration = Duration::from_secs(secs);
    // Use checked_sub to avoid panic on Windows where Instant is based on uptime
    Ok(Instant::now().checked_sub(duration).unwrap_or(Instant::now()))
}
