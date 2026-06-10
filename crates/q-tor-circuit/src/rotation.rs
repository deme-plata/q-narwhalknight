/// Circuit Rotation: Automatic circuit rotation for security and performance
/// Rotates circuits every epoch (5-minute default) for traffic analysis resistance
use anyhow::Result;
use q_types::{NodeId, Phase};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

use super::{CircuitInfo, CircuitPurpose};

/// Circuit rotation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationPolicy {
    /// Base rotation interval
    pub base_interval: Duration,
    /// Minimum rotation interval (security limit)
    pub min_interval: Duration,
    /// Maximum rotation interval
    pub max_interval: Duration,
    /// Randomization factor (0.0 - 1.0)
    pub randomization_factor: f64,
    /// Enable adaptive rotation based on usage
    pub adaptive_rotation: bool,
    /// High-usage threshold for faster rotation
    pub high_usage_threshold: u64,
    /// Enable quantum-entropy rotation timing
    pub quantum_entropy_timing: bool,
}

impl Default for RotationPolicy {
    fn default() -> Self {
        Self {
            base_interval: Duration::from_secs(300), // 5 minutes
            min_interval: Duration::from_secs(180),  // 3 minutes minimum
            max_interval: Duration::from_secs(600),  // 10 minutes maximum
            randomization_factor: 0.2,               // ±20% randomization
            adaptive_rotation: true,
            // v8.6.0: increased from 10MB to 25MB — block propagation can push
            // substantial data through a single circuit; 10MB triggered premature
            // rotations under normal load, degrading throughput
            high_usage_threshold: 25_000_000, // 25 MB threshold
            quantum_entropy_timing: true,
        }
    }
}

/// Rotation event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationEvent {
    /// Scheduled rotation based on time
    Scheduled {
        circuit_id: u64,
        purpose: CircuitPurpose,
        age: Duration,
    },
    /// Forced rotation due to performance issues
    PerformanceForced {
        circuit_id: u64,
        purpose: CircuitPurpose,
        reason: String,
    },
    /// Adaptive rotation due to high usage
    AdaptiveUsage {
        circuit_id: u64,
        purpose: CircuitPurpose,
        bytes_transferred: u64,
    },
    /// Security rotation (random timing)
    SecurityRotation {
        circuit_id: u64,
        purpose: CircuitPurpose,
        entropy_seed: [u8; 8],
    },
    /// Epoch-based rotation (synchronized across validators)
    EpochRotation { circuits: Vec<u64>, epoch: u64 },
}

/// Circuit rotation scheduler
pub struct CircuitRotator {
    /// Rotation policy configuration
    policy: RotationPolicy,
    /// Circuit rotation schedules
    schedules: Arc<RwLock<HashMap<u64, CircuitRotationSchedule>>>,
    /// Next scheduled rotation time
    next_rotation: Arc<Mutex<Instant>>,
    /// Rotation event history
    rotation_history: Arc<RwLock<Vec<RotationEvent>>>,
    /// Statistics
    stats: Arc<RwLock<RotationStats>>,
    /// Current epoch counter
    current_epoch: Arc<Mutex<u64>>,
    /// Quantum entropy source for timing variation
    quantum_entropy: Arc<RwLock<Vec<u8>>>,
}

/// Individual circuit rotation schedule
#[derive(Debug, Clone)]
pub struct CircuitRotationSchedule {
    pub circuit_id: u64,
    pub purpose: CircuitPurpose,
    pub created_at: Instant,
    pub scheduled_rotation: Instant,
    pub last_rotation: Option<Instant>,
    pub rotation_count: u32,
    pub bytes_since_rotation: u64,
    pub adaptive_factor: f64,
}

/// Rotation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationStats {
    pub total_rotations: u64,
    pub scheduled_rotations: u64,
    pub performance_rotations: u64,
    pub adaptive_rotations: u64,
    pub security_rotations: u64,
    pub epoch_rotations: u64,
    pub avg_circuit_lifetime: Duration,
    pub shortest_lifetime: Duration,
    pub longest_lifetime: Duration,
    #[serde(skip)]
    pub last_rotation: Option<Instant>,
    #[serde(skip)]
    pub next_scheduled_rotation: Option<Instant>,
}

impl CircuitRotator {
    /// Create new circuit rotator
    pub fn new(base_rotation_interval: Duration) -> Self {
        let policy = RotationPolicy {
            base_interval: base_rotation_interval,
            ..Default::default()
        };

        info!(
            "🔄 Initializing CircuitRotator with {}s base interval",
            base_rotation_interval.as_secs()
        );

        Self {
            policy,
            schedules: Arc::new(RwLock::new(HashMap::new())),
            next_rotation: Arc::new(Mutex::new(Instant::now() + base_rotation_interval)),
            rotation_history: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(RotationStats {
                total_rotations: 0,
                scheduled_rotations: 0,
                performance_rotations: 0,
                adaptive_rotations: 0,
                security_rotations: 0,
                epoch_rotations: 0,
                avg_circuit_lifetime: Duration::from_secs(0),
                shortest_lifetime: Duration::from_secs(u64::MAX),
                longest_lifetime: Duration::from_secs(0),
                last_rotation: None,
                next_scheduled_rotation: None,
            })),
            current_epoch: Arc::new(Mutex::new(0)),
            quantum_entropy: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Schedule a circuit for rotation
    pub async fn schedule_circuit(
        &mut self,
        circuit_id: u64,
        purpose: CircuitPurpose,
    ) -> Result<()> {
        let now = Instant::now();

        // Calculate rotation time with randomization
        let base_interval = self.purpose_rotation_interval(purpose);
        let rotation_time = now + self.randomize_interval(base_interval).await;

        let schedule = CircuitRotationSchedule {
            circuit_id,
            purpose,
            created_at: now,
            scheduled_rotation: rotation_time,
            last_rotation: None,
            rotation_count: 0,
            bytes_since_rotation: 0,
            adaptive_factor: 1.0,
        };

        {
            let mut schedules = self.schedules.write().await;
            schedules.insert(circuit_id, schedule);
        }

        // Update next rotation time if this is sooner
        {
            let mut next_rotation = self.next_rotation.lock().await;
            if rotation_time < *next_rotation {
                *next_rotation = rotation_time;

                let mut stats = self.stats.write().await;
                stats.next_scheduled_rotation = Some(rotation_time);
            }
        }

        debug!(
            "📅 Scheduled circuit {} for rotation in {}s",
            circuit_id,
            base_interval.as_secs()
        );

        Ok(())
    }

    /// Check if it's time to rotate circuits
    pub fn should_rotate_now(&self) -> bool {
        if let Ok(next_rotation) = self.next_rotation.try_lock() {
            Instant::now() >= *next_rotation
        } else {
            false // If lock is held, assume no rotation needed right now
        }
    }

    /// Get circuits that need rotation right now
    pub async fn get_circuits_for_rotation(&self) -> Vec<RotationEvent> {
        let now = Instant::now();
        let schedules = self.schedules.read().await;

        let mut rotation_events = Vec::new();

        for schedule in schedules.values() {
            // Check scheduled rotation
            if now >= schedule.scheduled_rotation {
                rotation_events.push(RotationEvent::Scheduled {
                    circuit_id: schedule.circuit_id,
                    purpose: schedule.purpose,
                    age: now.duration_since(schedule.created_at),
                });
            }
            // Check adaptive rotation (high usage)
            else if self.policy.adaptive_rotation
                && schedule.bytes_since_rotation > self.policy.high_usage_threshold
            {
                rotation_events.push(RotationEvent::AdaptiveUsage {
                    circuit_id: schedule.circuit_id,
                    purpose: schedule.purpose,
                    bytes_transferred: schedule.bytes_since_rotation,
                });
            }
        }

        if !rotation_events.is_empty() {
            info!(
                "🔄 Found {} circuits ready for rotation",
                rotation_events.len()
            );
        }

        rotation_events
    }

    /// Mark rotation as complete and reschedule
    pub async fn mark_rotation_complete(&mut self) {
        let now = Instant::now();
        let mut next_rotation_time = now + self.policy.max_interval;

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.last_rotation = Some(now);
            stats.total_rotations += 1;
        }

        // Find next earliest rotation
        {
            let schedules = self.schedules.read().await;
            for schedule in schedules.values() {
                if schedule.scheduled_rotation > now
                    && schedule.scheduled_rotation < next_rotation_time
                {
                    next_rotation_time = schedule.scheduled_rotation;
                }
            }
        }

        {
            let mut next_rotation = self.next_rotation.lock().await;
            *next_rotation = next_rotation_time;
        }

        debug!(
            "✅ Rotation complete, next rotation in {}s",
            next_rotation_time.duration_since(now).as_secs()
        );
    }

    /// Force immediate rotation for performance reasons
    pub async fn force_rotation(
        &mut self,
        circuit_id: u64,
        reason: String,
    ) -> Result<RotationEvent> {
        info!("⚠️ Forcing rotation of circuit {}: {}", circuit_id, reason);

        let schedules = self.schedules.read().await;

        if let Some(schedule) = schedules.get(&circuit_id) {
            let event = RotationEvent::PerformanceForced {
                circuit_id,
                purpose: schedule.purpose,
                reason,
            };

            // Record event
            {
                let mut history = self.rotation_history.write().await;
                history.push(event.clone());

                // Keep only last 1000 events
                if history.len() > 1000 {
                    history.drain(0..100);
                }
            }

            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.performance_rotations += 1;
                stats.total_rotations += 1;
            }

            Ok(event)
        } else {
            anyhow::bail!("Circuit {} not found in rotation schedule", circuit_id);
        }
    }

    /// Update circuit usage for adaptive rotation
    pub async fn update_circuit_usage(&mut self, circuit_id: u64, bytes_transferred: u64) {
        let mut schedules = self.schedules.write().await;

        if let Some(schedule) = schedules.get_mut(&circuit_id) {
            schedule.bytes_since_rotation += bytes_transferred;

            // Adaptive factor calculation based on usage
            if self.policy.adaptive_rotation {
                let usage_ratio =
                    schedule.bytes_since_rotation as f64 / self.policy.high_usage_threshold as f64;
                schedule.adaptive_factor = (2.0 - usage_ratio.min(1.0)).max(0.5); // Factor between 0.5 and 2.0

                debug!(
                    "📊 Circuit {} usage: {} bytes, adaptive factor: {:.2}",
                    circuit_id, schedule.bytes_since_rotation, schedule.adaptive_factor
                );
            }
        }
    }

    /// Perform epoch-based rotation (synchronized across network)
    pub async fn rotate_epoch(&mut self, epoch: u64) -> Result<RotationEvent> {
        info!("🌍 Performing epoch rotation for epoch {}", epoch);

        {
            let mut current_epoch = self.current_epoch.lock().await;
            *current_epoch = epoch;
        }

        let schedules = self.schedules.read().await;
        let circuits_for_rotation: Vec<u64> = schedules
            .values()
            .filter(|s| {
                // Rotate circuits based on epoch and circuit purpose
                (epoch + s.circuit_id) % self.epoch_rotation_modulus(s.purpose) == 0
            })
            .map(|s| s.circuit_id)
            .collect();

        if !circuits_for_rotation.is_empty() {
            info!(
                "🔄 Rotating {} circuits for epoch {}",
                circuits_for_rotation.len(),
                epoch
            );

            // Update statistics
            {
                let mut stats = self.stats.write().await;
                stats.epoch_rotations += 1;
                stats.total_rotations += circuits_for_rotation.len() as u64;
            }
        }

        let event = RotationEvent::EpochRotation {
            circuits: circuits_for_rotation,
            epoch,
        };

        // Record event
        {
            let mut history = self.rotation_history.write().await;
            history.push(event.clone());
        }

        Ok(event)
    }

    /// Add quantum entropy for rotation timing variation
    pub async fn add_quantum_entropy(&mut self, entropy: &[u8]) {
        if self.policy.quantum_entropy_timing {
            let mut quantum_entropy = self.quantum_entropy.write().await;
            quantum_entropy.extend_from_slice(entropy);

            // Keep only recent entropy (last 1KB)
            if quantum_entropy.len() > 1024 {
                quantum_entropy.drain(0..512);
            }

            debug!(
                "🎲 Added {} bytes of quantum entropy for rotation timing",
                entropy.len()
            );
        }
    }

    /// Get rotation statistics
    pub async fn get_rotation_stats(&self) -> RotationStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Get rotation event history
    pub async fn get_rotation_history(&self, limit: Option<usize>) -> Vec<RotationEvent> {
        let history = self.rotation_history.read().await;
        let limit = limit.unwrap_or(100);

        history.iter().rev().take(limit).cloned().collect()
    }

    /// Calculate purpose-specific rotation interval
    fn purpose_rotation_interval(&self, purpose: CircuitPurpose) -> Duration {
        match purpose {
            CircuitPurpose::Control => Duration::from_secs(600), // 10 minutes - less frequent
            // v8.6.0: reduced from 180s to 150s — quantum entropy is high-value;
            // tighter rotation limits the correlation window for timing attacks
            CircuitPurpose::QuantumBeacon => Duration::from_secs(150), // 2.5 minutes - more sensitive
            CircuitPurpose::BlockGossip => self.policy.base_interval, // Default interval
            CircuitPurpose::AckGossip => self.policy.base_interval, // Default interval
        }
    }

    /// Calculate epoch rotation modulus for synchronized rotation
    fn epoch_rotation_modulus(&self, purpose: CircuitPurpose) -> u64 {
        match purpose {
            CircuitPurpose::Control => 4,       // Every 4th epoch (20 minutes)
            CircuitPurpose::QuantumBeacon => 2, // Every 2nd epoch (10 minutes)
            CircuitPurpose::BlockGossip => 3,   // Every 3rd epoch (15 minutes)
            CircuitPurpose::AckGossip => 3,     // Every 3rd epoch (15 minutes)
        }
    }

    /// Randomize interval with quantum entropy if available
    async fn randomize_interval(&self, base_interval: Duration) -> Duration {
        let randomization = self.policy.randomization_factor;
        let base_ms = base_interval.as_millis() as f64;

        // Get randomness from quantum entropy if available
        let random_factor = if self.policy.quantum_entropy_timing {
            let quantum_entropy = self.quantum_entropy.read().await;
            if !quantum_entropy.is_empty() {
                // Use quantum entropy for randomization
                let entropy_bytes = &quantum_entropy[quantum_entropy.len().saturating_sub(8)..];
                let entropy_u64 = if entropy_bytes.len() >= 8 {
                    u64::from_be_bytes([
                        entropy_bytes[0],
                        entropy_bytes[1],
                        entropy_bytes[2],
                        entropy_bytes[3],
                        entropy_bytes[4],
                        entropy_bytes[5],
                        entropy_bytes[6],
                        entropy_bytes[7],
                    ])
                } else {
                    rand::random::<u64>()
                };

                // Convert to range [-1.0, 1.0]
                (entropy_u64 as f64 / u64::MAX as f64) * 2.0 - 1.0
            } else {
                (rand::random::<f64>() - 0.5) * 2.0
            }
        } else {
            (rand::random::<f64>() - 0.5) * 2.0
        };

        let variation = base_ms * randomization * random_factor;
        let final_ms = (base_ms + variation)
            .max(self.policy.min_interval.as_millis() as f64)
            .min(self.policy.max_interval.as_millis() as f64);

        Duration::from_millis(final_ms as u64)
    }

    /// Remove circuit from rotation schedule
    pub async fn remove_circuit(&mut self, circuit_id: u64) {
        let mut schedules = self.schedules.write().await;
        if schedules.remove(&circuit_id).is_some() {
            debug!("🗑️ Removed circuit {} from rotation schedule", circuit_id);
        }
    }

    /// Update rotation policy
    pub async fn update_policy(&mut self, new_policy: RotationPolicy) -> Result<()> {
        info!(
            "🔧 Updating rotation policy: base_interval={}s, adaptive={}",
            new_policy.base_interval.as_secs(),
            new_policy.adaptive_rotation
        );

        self.policy = new_policy;

        // Reschedule all circuits with new policy
        let circuit_ids: Vec<u64> = {
            let schedules = self.schedules.read().await;
            schedules.keys().cloned().collect()
        };

        for circuit_id in circuit_ids {
            let purpose = {
                let schedules = self.schedules.read().await;
                schedules.get(&circuit_id).map(|s| s.purpose)
            };

            if let Some(purpose) = purpose {
                self.schedule_circuit(circuit_id, purpose).await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rotator_creation() {
        let rotator = CircuitRotator::new(Duration::from_secs(300));
        let stats = rotator.get_rotation_stats().await;
        assert_eq!(stats.total_rotations, 0);
    }

    #[tokio::test]
    async fn test_circuit_scheduling() {
        let mut rotator = CircuitRotator::new(Duration::from_secs(300));

        let result = rotator.schedule_circuit(1, CircuitPurpose::Control).await;
        assert!(result.is_ok());

        let stats = rotator.get_rotation_stats().await;
        assert!(stats.next_scheduled_rotation.is_some());
    }

    #[tokio::test]
    async fn test_rotation_intervals() {
        let rotator = CircuitRotator::new(Duration::from_secs(300));

        let control_interval = rotator.purpose_rotation_interval(CircuitPurpose::Control);
        let quantum_interval = rotator.purpose_rotation_interval(CircuitPurpose::QuantumBeacon);

        assert_eq!(control_interval, Duration::from_secs(600));
        // v8.6.0: updated from 180s to 150s
        assert_eq!(quantum_interval, Duration::from_secs(150));
    }

    #[tokio::test]
    async fn test_usage_tracking() {
        let mut rotator = CircuitRotator::new(Duration::from_secs(300));

        rotator
            .schedule_circuit(1, CircuitPurpose::BlockGossip)
            .await
            .unwrap();
        rotator.update_circuit_usage(1, 1000000).await;

        // Verify usage is tracked (internal state test)
        let schedules = rotator.schedules.read().await;
        if let Some(schedule) = schedules.get(&1) {
            assert_eq!(schedule.bytes_since_rotation, 1000000);
        }
    }

    #[tokio::test]
    async fn test_forced_rotation() {
        let mut rotator = CircuitRotator::new(Duration::from_secs(300));

        rotator
            .schedule_circuit(1, CircuitPurpose::Control)
            .await
            .unwrap();

        let event = rotator
            .force_rotation(1, "Performance issue".to_string())
            .await;
        assert!(event.is_ok());

        let stats = rotator.get_rotation_stats().await;
        assert_eq!(stats.performance_rotations, 1);
    }

    #[tokio::test]
    async fn test_quantum_entropy() {
        let mut rotator = CircuitRotator::new(Duration::from_secs(300));

        rotator.add_quantum_entropy(&[1, 2, 3, 4, 5, 6, 7, 8]).await;

        let quantum_entropy = rotator.quantum_entropy.read().await;
        assert_eq!(quantum_entropy.len(), 8);
    }

    #[test]
    fn test_epoch_modulus() {
        let rotator = CircuitRotator::new(Duration::from_secs(300));

        assert_eq!(rotator.epoch_rotation_modulus(CircuitPurpose::Control), 4);
        assert_eq!(
            rotator.epoch_rotation_modulus(CircuitPurpose::QuantumBeacon),
            2
        );
        assert_eq!(
            rotator.epoch_rotation_modulus(CircuitPurpose::BlockGossip),
            3
        );
    }
}
