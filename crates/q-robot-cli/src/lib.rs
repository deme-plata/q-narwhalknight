//! # Quantum Water Robot Control CLI
//! 
//! A comprehensive command-line interface for controlling quantum-enhanced water robots
//! integrated with the Q-NarwhalKnight consensus system.
//!
//! ## Features
//!
//! - **Robot Control**: Connect to and manage individual quantum water robots
//! - **Swarm Coordination**: Multi-robot swarms with quantum entanglement
//! - **Quantum Monitoring**: Real-time quantum state visualization and measurement
//! - **Environmental Monitoring**: Marine ecosystem tracking and conservation
//! - **Consensus Integration**: Secure data submission to Q-NarwhalKnight network
//! - **Interactive UI**: Full-featured terminal interface with real-time updates
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use q_robot_cli::{RobotManager, RobotConfig, RobotId};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = RobotConfig::load("robot-config.toml").await?;
//!     let mut robot_manager = RobotManager::new(config).await?;
//!     
//!     // Connect to robot
//!     let robot_id = RobotId::new("quantum_jelly_001");
//!     robot_manager.connect_robot(robot_id, Some("jellyfish".to_string())).await?;
//!     
//!     // Move robot
//!     robot_manager.move_robot("quantum_jelly_001", vec![10.0, 20.0, -5.0], 0.5).await?;
//!     
//!     Ok(())
//! }
//! ```

pub mod robot;
pub mod swarm;
pub mod quantum;
pub mod ui;
pub mod config;
pub mod consensus;
pub mod finance;

// Re-export main types for easy access
pub use robot::{RobotManager, RobotId, RobotType, RobotStatus, SensorData, ScanResults, WaterQuality, MarineLifeEntry};
pub use swarm::{SwarmController, SwarmFormation};
pub use quantum::{QuantumStateMonitor, QuantumState, QuantumObservable, BellStateType};
pub use ui::TerminalUI;
pub use config::{RobotConfig, RobotConfigEntry, NetworkConfig, QuantumConfig, SecurityConfig};
pub use consensus::{ConsensusIntegration, RobotConsensusData, RobotDataType, ConsensusEvent};
pub use finance::{
    FinancialIntelligenceEngine, KLawParameters, FlowWeights,
    QNKFlowDensity, ThreeLayerAdoption, KristensenRatio, AdoptionHealth,
    FinancialRobotRole, FinancialSnapshot, HolderDistribution,
    AdoptionCheckpoint, FinancialIntelligenceReport,
    get_financial_role_for_robot_type,
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Current API version
pub const API_VERSION: &str = "v1";

/// Default configuration file name
pub const DEFAULT_CONFIG_FILE: &str = "robot-config.toml";

/// Error types specific to robot control
#[derive(Debug, thiserror::Error)]
pub enum RobotError {
    #[error("Robot not found: {id}")]
    RobotNotFound { id: String },
    
    #[error("Invalid robot type: {robot_type}")]
    InvalidRobotType { robot_type: String },
    
    #[error("Robot connection failed: {reason}")]
    ConnectionFailed { reason: String },
    
    #[error("Authentication failed for robot: {id}")]
    AuthenticationFailed { id: String },
    
    #[error("Invalid coordinates: {coords:?}")]
    InvalidCoordinates { coords: Vec<f64> },
    
    #[error("Ability not supported by robot type: {ability}")]
    UnsupportedAbility { ability: String },
    
    #[error("Robot is offline: {id}")]
    RobotOffline { id: String },
    
    #[error("Battery level too low: {level}%")]
    LowBattery { level: f64 },
}

/// Error types for swarm operations
#[derive(Debug, thiserror::Error)]
pub enum SwarmError {
    #[error("Swarm not found: {name}")]
    SwarmNotFound { name: String },
    
    #[error("Invalid formation: {formation}")]
    InvalidFormation { formation: String },
    
    #[error("Insufficient robots for formation: need {required}, have {available}")]
    InsufficientRobots { required: u32, available: u32 },
    
    #[error("Mission deployment failed: {reason}")]
    MissionFailed { reason: String },
    
    #[error("Quantum entanglement lost: fidelity {fidelity:.3}")]
    EntanglementLost { fidelity: f64 },
    
    #[error("Swarm coordination timeout")]
    CoordinationTimeout,
}

/// Error types for quantum operations
#[derive(Debug, thiserror::Error)]
pub enum QuantumError {
    #[error("Quantum decoherence: coherence time {time:.6}s")]
    Decoherence { time: f64 },
    
    #[error("Invalid quantum state: {reason}")]
    InvalidState { reason: String },
    
    #[error("Measurement failed: {observable}")]
    MeasurementFailed { observable: String },
    
    #[error("Quantum random generation failed")]
    RandomGenerationFailed,
    
    #[error("Unsupported visualization type: {viz_type}")]
    UnsupportedVisualization { viz_type: String },
    
    #[error("Quantum network connection failed")]
    NetworkConnectionFailed,
}

/// Result type for robot operations
pub type RobotResult<T> = std::result::Result<T, RobotError>;

/// Result type for swarm operations
pub type SwarmResult<T> = std::result::Result<T, SwarmError>;

/// Result type for quantum operations
pub type QuantumResult<T> = std::result::Result<T, QuantumError>;

/// General result type for the library
pub type Result<T> = anyhow::Result<T>;

/// Utility functions for the library
pub mod utils {
    use nalgebra::Vector3;
    
    /// Calculate distance between two 3D points
    pub fn distance_3d(point1: (f64, f64, f64), point2: (f64, f64, f64)) -> f64 {
        let p1 = Vector3::new(point1.0, point1.1, point1.2);
        let p2 = Vector3::new(point2.0, point2.1, point2.2);
        (p2 - p1).norm()
    }
    
    /// Convert degrees to radians
    pub fn deg_to_rad(degrees: f64) -> f64 {
        degrees * std::f64::consts::PI / 180.0
    }
    
    /// Convert radians to degrees
    pub fn rad_to_deg(radians: f64) -> f64 {
        radians * 180.0 / std::f64::consts::PI
    }
    
    /// Normalize angle to [-π, π] range
    pub fn normalize_angle(angle: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        angle - two_pi * (angle / two_pi).floor()
    }
    
    /// Check if a point is within a bounding box
    pub fn point_in_bounds(
        point: (f64, f64, f64), 
        min: (f64, f64, f64), 
        max: (f64, f64, f64)
    ) -> bool {
        point.0 >= min.0 && point.0 <= max.0 &&
        point.1 >= min.1 && point.1 <= max.1 &&
        point.2 >= min.2 && point.2 <= max.2
    }
    
    /// Calculate quantum fidelity between two states
    pub fn quantum_fidelity(state1: &[num_complex::Complex64], state2: &[num_complex::Complex64]) -> f64 {
        if state1.len() != state2.len() {
            return 0.0;
        }
        
        let overlap: num_complex::Complex64 = state1.iter()
            .zip(state2.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        
        overlap.norm_sqr()
    }
    
    /// Generate UUID for robot operations
    pub fn generate_uuid() -> String {
        format!("{:x}", rand::random::<u128>())
    }
    
    /// Format timestamp for logging
    pub fn format_timestamp(timestamp: std::time::Instant) -> String {
        let elapsed = timestamp.elapsed();
        format!("{:.3}s ago", elapsed.as_secs_f64())
    }
}

/// Prelude module for common imports
pub mod prelude {
    pub use crate::{
        RobotManager, RobotId, RobotType, RobotStatus,
        SwarmController, SwarmFormation,
        QuantumStateMonitor, QuantumState, QuantumObservable,
        RobotConfig, RobotConfigEntry,
        ConsensusIntegration, RobotConsensusData,
        FinancialIntelligenceEngine, KLawParameters, KristensenRatio,
        FinancialRobotRole, FinancialSnapshot,
        Result, RobotResult, SwarmResult, QuantumResult,
        RobotError, SwarmError, QuantumError,
    };

    pub use anyhow::{Context, Result as AnyhowResult};
    pub use tokio;
    pub use tracing::{debug, info, warn, error};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(API_VERSION, "v1");
    }
    
    #[test]
    fn test_utility_functions() {
        use utils::*;
        
        // Test distance calculation
        let dist = distance_3d((0.0, 0.0, 0.0), (3.0, 4.0, 0.0));
        assert!((dist - 5.0).abs() < 1e-10);
        
        // Test angle conversion
        let radians = deg_to_rad(180.0);
        assert!((radians - std::f64::consts::PI).abs() < 1e-10);
        
        let degrees = rad_to_deg(std::f64::consts::PI);
        assert!((degrees - 180.0).abs() < 1e-10);
        
        // Test point in bounds
        assert!(point_in_bounds((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)));
        assert!(!point_in_bounds((3.0, 1.0, 1.0), (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)));
    }
    
    #[test]
    fn test_error_types() {
        let robot_error = RobotError::RobotNotFound { id: "test_robot".to_string() };
        assert!(robot_error.to_string().contains("Robot not found: test_robot"));
        
        let swarm_error = SwarmError::SwarmNotFound { name: "test_swarm".to_string() };
        assert!(swarm_error.to_string().contains("Swarm not found: test_swarm"));
        
        let quantum_error = QuantumError::Decoherence { time: 0.001 };
        assert!(quantum_error.to_string().contains("Quantum decoherence"));
    }
}