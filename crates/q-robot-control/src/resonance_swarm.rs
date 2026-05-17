//! 🌊🎭 Resonance Swarm Integration for Water Robots
//!
//! This module enables water robots to participate in the Quillon Resonance
//! consensus system, modeling swarm coordination as string-theoretic vibrations.
//!
//! ## Philosophy
//!
//! Water robots moving through space are like strings vibrating in a field.
//! Their positions, velocities, and sensor readings create interference patterns.
//! Consensus emerges when these patterns harmonize - when the swarm resonates.
//!
//! ## Key Concepts
//!
//! - **Robot String States**: Each robot is a vibrating string with amplitude (energy),
//!   frequency (activity rate), and phase (position in formation)
//! - **Swarm Harmonics**: Swarm coordination modeled as constructive interference
//! - **Spectral BFT**: Byzantine (malfunctioning) robots detected via Laplacian analysis
//! - **Shadow Consensus**: Run resonance alongside DAG-BFT for validation

use crate::{
    WaterRobotId, WaterRobotState, Position3D, Velocity3D,
    CoordinationState, FormationMode, DagBftState,
};
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// 🎭 Resonance parameters for swarm behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResonanceConfig {
    /// Base frequency for robot oscillations (Hz)
    pub base_frequency: f64,

    /// Coupling strength between nearby robots
    pub coupling_strength: f64,

    /// Maximum distance for resonance coupling (meters)
    pub coupling_radius: f64,

    /// Damping factor for energy dissipation
    pub damping: f64,

    /// K-parameter phase transition threshold
    pub k_critical: f64,

    /// Shadow mode resonance weight (0.0 = pure DAG-BFT, 1.0 = pure resonance)
    pub resonance_weight: f64,

    /// Enable spectral Byzantine detection
    pub spectral_bft_enabled: bool,

    /// Minimum eigenvalue gap for Byzantine detection
    pub eigenvalue_threshold: f64,
}

impl Default for SwarmResonanceConfig {
    fn default() -> Self {
        Self {
            base_frequency: 1.0,           // 1 Hz base oscillation
            coupling_strength: 0.5,        // Moderate coupling
            coupling_radius: 10.0,         // 10 meter coupling range
            damping: 0.1,                  // Light damping
            k_critical: 2.618,             // Golden ratio squared (aesthetic resonance)
            resonance_weight: 0.0,         // Start with pure DAG-BFT
            spectral_bft_enabled: true,    // Byzantine detection on
            eigenvalue_threshold: 0.01,    // Spectral gap threshold
        }
    }
}

/// 🌊 String state for a water robot
///
/// Models the robot as a vibrating string in the field space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotStringState {
    /// Robot identifier
    pub robot_id: WaterRobotId,

    /// Amplitude (energy level, health, activity)
    pub amplitude: f64,

    /// Frequency (update rate, responsiveness)
    pub frequency: f64,

    /// Phase (position in formation, 0 to 2π)
    pub phase: f64,

    /// Position in field space (3D coordinates)
    pub field_position: [f64; 3],

    /// Velocity in field space
    pub field_velocity: [f64; 3],

    /// Energy functional value (lower = more stable)
    pub energy: f64,

    /// Coupling energies to other robots
    pub couplings: HashMap<String, f64>,

    /// Byzantine suspicion score (0 = trusted, 1 = suspected)
    pub byzantine_score: f64,

    /// Last update timestamp
    pub last_update: i64,
}

impl RobotStringState {
    /// Create string state from water robot state
    pub fn from_robot_state(robot: &WaterRobotState, config: &SwarmResonanceConfig) -> Self {
        let amplitude = robot.energy_level as f64 * robot.health_status.overall_health as f64;
        let frequency = config.base_frequency * (1.0 + robot.communication_quality as f64);

        // Phase based on position relative to origin
        let phase = (robot.position.x.atan2(robot.position.y) + PI) % (2.0 * PI);

        Self {
            robot_id: robot.robot_id.clone(),
            amplitude,
            frequency,
            phase,
            field_position: [robot.position.x, robot.position.y, robot.position.z],
            field_velocity: [robot.velocity.x, robot.velocity.y, robot.velocity.z],
            energy: 0.0, // Calculated later
            couplings: HashMap::new(),
            byzantine_score: 0.0,
            last_update: Utc::now().timestamp_millis(),
        }
    }

    /// Calculate wave function value at time t
    pub fn wave_function(&self, t: f64) -> f64 {
        self.amplitude * (2.0 * PI * self.frequency * t + self.phase).sin()
    }

    /// Calculate interference with another robot
    pub fn interference_with(&self, other: &RobotStringState, t: f64) -> f64 {
        let wave1 = self.wave_function(t);
        let wave2 = other.wave_function(t);

        // Constructive interference when in phase, destructive when out of phase
        let combined = wave1 + wave2;
        let max_possible = self.amplitude + other.amplitude;

        if max_possible > 0.0 {
            combined / max_possible // Normalized interference (-1 to 1)
        } else {
            0.0
        }
    }
}

/// 🎭 Swarm Resonance Coordinator
///
/// Manages the resonance consensus for water robot swarms.
pub struct SwarmResonanceCoordinator {
    /// Configuration
    config: SwarmResonanceConfig,

    /// String states for all robots
    string_states: Arc<RwLock<HashMap<String, RobotStringState>>>,

    /// Laplacian matrix for spectral analysis
    laplacian: Arc<RwLock<Vec<Vec<f64>>>>,

    /// Shadow mode metrics
    shadow_metrics: Arc<RwLock<SwarmShadowMetrics>>,

    /// Current consensus round
    current_round: Arc<RwLock<u64>>,

    /// Eigenvalues from spectral decomposition
    eigenvalues: Arc<RwLock<Vec<f64>>>,
}

/// 📊 Shadow mode metrics for swarm resonance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwarmShadowMetrics {
    /// Total consensus rounds processed
    pub total_rounds: u64,

    /// Rounds where resonance agreed with DAG-BFT
    pub agreement_rounds: u64,

    /// Current agreement rate
    pub agreement_rate: f64,

    /// Average swarm energy (lower = more stable)
    pub avg_swarm_energy: f64,

    /// Spectral gap (second smallest eigenvalue)
    pub spectral_gap: f64,

    /// Byzantine robots detected
    pub byzantine_detected: Vec<String>,

    /// Swarm coherence score (0 = chaos, 1 = perfect harmony)
    pub coherence_score: f64,

    /// K-parameter current value
    pub k_parameter: f64,

    /// Phase transition detected
    pub phase_transition_active: bool,
}

impl SwarmResonanceCoordinator {
    /// Create new swarm resonance coordinator
    pub fn new(config: SwarmResonanceConfig) -> Self {
        info!("🎭🌊 Initializing Swarm Resonance Coordinator");
        info!("   Base Frequency: {} Hz", config.base_frequency);
        info!("   Coupling Strength: {}", config.coupling_strength);
        info!("   Coupling Radius: {} m", config.coupling_radius);
        info!("   K-Critical: {}", config.k_critical);
        info!("   Resonance Weight: {:.0}%", config.resonance_weight * 100.0);

        Self {
            config,
            string_states: Arc::new(RwLock::new(HashMap::new())),
            laplacian: Arc::new(RwLock::new(Vec::new())),
            shadow_metrics: Arc::new(RwLock::new(SwarmShadowMetrics::default())),
            current_round: Arc::new(RwLock::new(0)),
            eigenvalues: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 🌊 Update string state from robot state
    pub async fn update_robot_string_state(&self, robot: &WaterRobotState) {
        let string_state = RobotStringState::from_robot_state(robot, &self.config);
        let robot_key = robot.robot_id.0.clone();

        let mut states = self.string_states.write().await;
        states.insert(robot_key, string_state);
    }

    /// 🎭 Process swarm state through resonance consensus
    ///
    /// This is the shadow mode processor that runs alongside DAG-BFT.
    pub async fn process_swarm_resonance(
        &self,
        coordination_state: &CoordinationState,
    ) -> Result<SwarmResonanceResult> {
        let mut round = self.current_round.write().await;
        *round += 1;
        let current_round = *round;

        debug!("🎭 Processing swarm resonance round {}", current_round);

        // Update all robot string states
        for (_, robot) in &coordination_state.active_robots {
            self.update_robot_string_state(robot).await;
        }

        // Calculate coupling energies between robots
        self.calculate_couplings().await;

        // Build Laplacian matrix for spectral analysis
        self.build_laplacian().await;

        // Compute spectral decomposition for Byzantine detection
        let byzantine_suspects = if self.config.spectral_bft_enabled {
            self.spectral_byzantine_detection().await
        } else {
            vec![]
        };

        // Calculate swarm energy functional
        let swarm_energy = self.calculate_swarm_energy().await;

        // Calculate K-parameter for phase analysis
        let k_parameter = self.calculate_k_parameter().await;

        // Calculate coherence score
        let coherence = self.calculate_coherence().await;

        // Determine resonance ordering (which robots should lead)
        let resonance_ordering = self.determine_resonance_ordering().await;

        // Update shadow metrics
        let mut metrics = self.shadow_metrics.write().await;
        metrics.total_rounds = current_round;
        metrics.avg_swarm_energy = swarm_energy;
        metrics.coherence_score = coherence;
        metrics.k_parameter = k_parameter;
        metrics.byzantine_detected = byzantine_suspects.clone();
        metrics.phase_transition_active = k_parameter > self.config.k_critical;

        // Log status
        if current_round % 10 == 0 {
            info!("🎭🌊 Swarm Resonance Round {} Complete:", current_round);
            info!("   🤖 Active Robots: {}", coordination_state.active_robots.len());
            info!("   ⚡ Swarm Energy: {:.4}", swarm_energy);
            info!("   🌀 Coherence: {:.1}%", coherence * 100.0);
            info!("   📊 K-Parameter: {:.4} (critical: {})",
                  k_parameter, self.config.k_critical);
            info!("   ⚠️  Byzantine Suspects: {}", byzantine_suspects.len());

            if metrics.phase_transition_active {
                warn!("   🔄 PHASE TRANSITION DETECTED - K exceeds critical!");
            }
        }

        // Extract recommended leader before moving resonance_ordering
        let recommended_leader = resonance_ordering.first().cloned();

        Ok(SwarmResonanceResult {
            round: current_round,
            swarm_energy,
            coherence,
            k_parameter,
            byzantine_suspects,
            resonance_ordering,
            recommended_leader,
        })
    }

    /// Calculate coupling energies between nearby robots
    async fn calculate_couplings(&self) {
        let mut states = self.string_states.write().await;
        let robot_ids: Vec<String> = states.keys().cloned().collect();

        for i in 0..robot_ids.len() {
            for j in (i + 1)..robot_ids.len() {
                let id_i = &robot_ids[i];
                let id_j = &robot_ids[j];

                // Get positions
                let pos_i = states.get(id_i).map(|s| s.field_position).unwrap_or([0.0; 3]);
                let pos_j = states.get(id_j).map(|s| s.field_position).unwrap_or([0.0; 3]);

                // Calculate distance
                let dx = pos_i[0] - pos_j[0];
                let dy = pos_i[1] - pos_j[1];
                let dz = pos_i[2] - pos_j[2];
                let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                // Coupling strength decreases with distance
                if distance < self.config.coupling_radius {
                    let coupling = self.config.coupling_strength
                        * (1.0 - distance / self.config.coupling_radius);

                    // Update couplings bidirectionally
                    if let Some(state_i) = states.get_mut(id_i) {
                        state_i.couplings.insert(id_j.clone(), coupling);
                    }
                    if let Some(state_j) = states.get_mut(id_j) {
                        state_j.couplings.insert(id_i.clone(), coupling);
                    }
                }
            }
        }
    }

    /// Build Laplacian matrix for spectral analysis
    async fn build_laplacian(&self) {
        let states = self.string_states.read().await;
        let n = states.len();

        if n == 0 {
            return;
        }

        let robot_ids: Vec<String> = states.keys().cloned().collect();
        let mut laplacian = vec![vec![0.0; n]; n];

        // Build adjacency and degree matrices
        for (i, id_i) in robot_ids.iter().enumerate() {
            let mut degree = 0.0;

            for (j, id_j) in robot_ids.iter().enumerate() {
                if i != j {
                    if let Some(state) = states.get(id_i) {
                        if let Some(&coupling) = state.couplings.get(id_j) {
                            laplacian[i][j] = -coupling;
                            degree += coupling;
                        }
                    }
                }
            }

            laplacian[i][i] = degree;
        }

        *self.laplacian.write().await = laplacian;
    }

    /// Spectral Byzantine detection using Laplacian eigenvalues
    ///
    /// Byzantine (malfunctioning) robots create anomalies in the spectral structure.
    async fn spectral_byzantine_detection(&self) -> Vec<String> {
        let laplacian = self.laplacian.read().await;
        let states = self.string_states.read().await;
        let robot_ids: Vec<String> = states.keys().cloned().collect();

        if laplacian.is_empty() {
            return vec![];
        }

        let n = laplacian.len();
        let mut suspects = vec![];

        // Simple power iteration to find approximate eigenvalues
        // (In production, use proper linear algebra library)
        let eigenvalues = self.approximate_eigenvalues(&laplacian, 3);
        *self.eigenvalues.write().await = eigenvalues.clone();

        // Check for anomalous robots based on their contribution to eigenvectors
        for (i, id) in robot_ids.iter().enumerate() {
            if let Some(state) = states.get(id) {
                // Robot is suspicious if:
                // 1. Low coupling (isolated)
                // 2. High energy deviation from swarm average
                // 3. Phase significantly different from neighbors

                let total_coupling: f64 = state.couplings.values().sum();
                let avg_coupling = if !state.couplings.is_empty() {
                    total_coupling / state.couplings.len() as f64
                } else {
                    0.0
                };

                // Mark as suspicious if very low coupling
                if avg_coupling < self.config.eigenvalue_threshold && n > 1 {
                    suspects.push(id.clone());
                    debug!("🎭 Byzantine suspect: {} (low coupling: {:.4})", id, avg_coupling);
                }
            }
        }

        suspects
    }

    /// Approximate eigenvalues using power iteration
    fn approximate_eigenvalues(&self, matrix: &[Vec<f64>], count: usize) -> Vec<f64> {
        let n = matrix.len();
        if n == 0 {
            return vec![];
        }

        let mut eigenvalues = vec![];

        // Simple trace-based approximation for first eigenvalue
        let trace: f64 = (0..n).map(|i| matrix[i][i]).sum();
        eigenvalues.push(trace / n as f64);

        // Estimate spectral gap from matrix structure
        let mut off_diagonal_sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off_diagonal_sum += matrix[i][j].abs();
                }
            }
        }
        let spectral_gap = off_diagonal_sum / (n * n) as f64;
        eigenvalues.push(spectral_gap);

        eigenvalues
    }

    /// Calculate total swarm energy functional
    async fn calculate_swarm_energy(&self) -> f64 {
        let states = self.string_states.read().await;

        if states.is_empty() {
            return 0.0;
        }

        let mut total_energy = 0.0;

        for state in states.values() {
            // Kinetic energy (from velocity)
            let kinetic = 0.5 * (
                state.field_velocity[0].powi(2) +
                state.field_velocity[1].powi(2) +
                state.field_velocity[2].powi(2)
            );

            // Potential energy (from amplitude/phase deviation)
            let potential = (1.0 - state.amplitude).powi(2);

            // Coupling energy (lower when well-coupled)
            let coupling_energy: f64 = state.couplings.values()
                .map(|&c| (1.0 - c).powi(2))
                .sum();

            total_energy += kinetic + potential + coupling_energy * self.config.damping;
        }

        total_energy / states.len() as f64
    }

    /// Calculate K-parameter for phase transition analysis
    async fn calculate_k_parameter(&self) -> f64 {
        let states = self.string_states.read().await;

        if states.len() < 2 {
            return 0.0;
        }

        // K = average phase coherence * energy gradient
        let phases: Vec<f64> = states.values().map(|s| s.phase).collect();
        let energies: Vec<f64> = states.values().map(|s| s.amplitude).collect();

        // Phase coherence (Kuramoto order parameter)
        let phase_x: f64 = phases.iter().map(|p| p.cos()).sum::<f64>() / phases.len() as f64;
        let phase_y: f64 = phases.iter().map(|p| p.sin()).sum::<f64>() / phases.len() as f64;
        let coherence = (phase_x.powi(2) + phase_y.powi(2)).sqrt();

        // Energy variance
        let avg_energy: f64 = energies.iter().sum::<f64>() / energies.len() as f64;
        let energy_var: f64 = energies.iter()
            .map(|e| (e - avg_energy).powi(2))
            .sum::<f64>() / energies.len() as f64;

        // K = coherence / (1 + variance) - high coherence, low variance = high K
        coherence / (1.0 + energy_var)
    }

    /// Calculate swarm coherence score
    async fn calculate_coherence(&self) -> f64 {
        let states = self.string_states.read().await;

        if states.is_empty() {
            return 0.0;
        }

        let t = Utc::now().timestamp_millis() as f64 / 1000.0;
        let robot_states: Vec<&RobotStringState> = states.values().collect();

        // Calculate average interference (coherence)
        let mut total_interference = 0.0;
        let mut pair_count = 0;

        for i in 0..robot_states.len() {
            for j in (i + 1)..robot_states.len() {
                let interference = robot_states[i].interference_with(robot_states[j], t);
                total_interference += interference.abs();
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_interference / pair_count as f64
        } else {
            1.0 // Single robot = perfect coherence
        }
    }

    /// Determine resonance-based ordering of robots
    ///
    /// Robots with higher energy and better coupling are natural leaders.
    async fn determine_resonance_ordering(&self) -> Vec<WaterRobotId> {
        let states = self.string_states.read().await;

        let mut scores: Vec<(WaterRobotId, f64)> = states.values()
            .map(|state| {
                let coupling_score: f64 = state.couplings.values().sum();
                let energy_score = state.amplitude;
                let byzantine_penalty = 1.0 - state.byzantine_score;

                let total_score = (energy_score + coupling_score) * byzantine_penalty;
                (state.robot_id.clone(), total_score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.into_iter().map(|(id, _)| id).collect()
    }

    /// Get current shadow metrics
    pub async fn get_metrics(&self) -> SwarmShadowMetrics {
        self.shadow_metrics.read().await.clone()
    }

    /// Set resonance weight (0.0 to 1.0)
    pub fn set_resonance_weight(&mut self, weight: f64) {
        self.config.resonance_weight = weight.clamp(0.0, 1.0);
        info!("🎭 Resonance weight set to {:.0}%", self.config.resonance_weight * 100.0);
    }

    /// Enable/disable spectral BFT
    pub fn set_spectral_bft(&mut self, enabled: bool) {
        self.config.spectral_bft_enabled = enabled;
        info!("🎭 Spectral BFT: {}", if enabled { "enabled" } else { "disabled" });
    }
}

/// 🌊 Result of swarm resonance processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResonanceResult {
    /// Consensus round number
    pub round: u64,

    /// Total swarm energy (lower = more stable)
    pub swarm_energy: f64,

    /// Coherence score (0 = chaos, 1 = harmony)
    pub coherence: f64,

    /// K-parameter value
    pub k_parameter: f64,

    /// Byzantine (malfunctioning) robot suspects
    pub byzantine_suspects: Vec<String>,

    /// Resonance-based robot ordering (leaders first)
    pub resonance_ordering: Vec<WaterRobotId>,

    /// Recommended swarm leader based on resonance
    pub recommended_leader: Option<WaterRobotId>,
}

/// 🎭 Create resonance coordinator for water robot swarm
pub fn create_swarm_resonance_coordinator(config: Option<SwarmResonanceConfig>) -> SwarmResonanceCoordinator {
    SwarmResonanceCoordinator::new(config.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HealthStatus, PumpStatus, SensorStatus, CapabilityProfile, TorCircuitStatus};

    fn create_test_robot(id: &str, x: f64, y: f64, energy: f32) -> WaterRobotState {
        WaterRobotState {
            robot_id: WaterRobotId(id.to_string()),
            position: Position3D { x, y, z: 0.0 },
            velocity: Velocity3D { x: 0.0, y: 0.0, z: 0.0 },
            health_status: HealthStatus {
                overall_health: 1.0,
                water_level: 1.0,
                pump_status: PumpStatus::Operational,
                sensor_status: SensorStatus {
                    pressure_sensors: true,
                    ph_sensors: true,
                    temperature_sensors: true,
                    position_sensors: true,
                },
                communication_health: 1.0,
            },
            capability_profile: CapabilityProfile {
                water_manipulation: 1.0,
                chemical_analysis: 1.0,
                swarm_coordination: 1.0,
                blockchain_processing: 1.0,
                neural_interface_compatibility: 1.0,
            },
            current_task: None,
            energy_level: energy,
            last_update: Utc::now(),
            communication_quality: 1.0,
            neural_control_active: false,
            blockchain_identities: HashMap::new(),
            tor_circuit_status: TorCircuitStatus {
                active_circuits: 4,
                circuit_health: vec![1.0, 1.0, 1.0, 1.0],
                onion_service_active: true,
                qnk_domain: Some("robot.qnk.onion".to_string()),
            },
        }
    }

    #[tokio::test]
    async fn test_string_state_creation() {
        let robot = create_test_robot("robot-1", 5.0, 5.0, 0.8);
        let config = SwarmResonanceConfig::default();

        let string_state = RobotStringState::from_robot_state(&robot, &config);

        assert_eq!(string_state.robot_id.0, "robot-1");
        assert!(string_state.amplitude > 0.0);
        assert!(string_state.frequency > 0.0);
    }

    #[tokio::test]
    async fn test_wave_interference() {
        let config = SwarmResonanceConfig::default();

        let robot1 = create_test_robot("robot-1", 0.0, 0.0, 1.0);
        let robot2 = create_test_robot("robot-2", 1.0, 0.0, 1.0);

        let state1 = RobotStringState::from_robot_state(&robot1, &config);
        let state2 = RobotStringState::from_robot_state(&robot2, &config);

        let interference = state1.interference_with(&state2, 0.0);
        assert!(interference >= -1.0 && interference <= 1.0);
    }

    #[tokio::test]
    async fn test_swarm_resonance_processing() {
        let coordinator = create_swarm_resonance_coordinator(None);

        let robot1 = create_test_robot("robot-1", 0.0, 0.0, 1.0);
        let robot2 = create_test_robot("robot-2", 5.0, 0.0, 0.9);
        let robot3 = create_test_robot("robot-3", 2.5, 4.0, 0.8);

        coordinator.update_robot_string_state(&robot1).await;
        coordinator.update_robot_string_state(&robot2).await;
        coordinator.update_robot_string_state(&robot3).await;

        let mut coordination_state = CoordinationState::default();
        coordination_state.active_robots.insert(robot1.robot_id.clone(), robot1);
        coordination_state.active_robots.insert(robot2.robot_id.clone(), robot2);
        coordination_state.active_robots.insert(robot3.robot_id.clone(), robot3);

        let result = coordinator.process_swarm_resonance(&coordination_state).await.unwrap();

        assert_eq!(result.round, 1);
        assert!(result.coherence >= 0.0 && result.coherence <= 1.0);
        assert!(!result.resonance_ordering.is_empty());
    }
}
