//! # Q-Resonance: String-Theoretic Consensus
//!
//! This crate implements the Quillon Resonance Consensus algorithm, which models
//! distributed consensus as physical resonance in multi-dimensional field space.
//!
//! ## Core Concepts
//!
//! - **String States**: Transactions as vibrating strings with amplitude, frequency, phase
//! - **Energy Functional**: Consensus emerges from energy minimization
//! - **Spectral BFT**: Byzantine detection via Laplacian eigenvalue analysis
//! - **Harmonic Convergence**: Agreement through constructive interference

// Import external crates to make them available
use serde as _serde;
use thiserror as _thiserror;

pub mod string_state;
pub mod energy;
pub mod vertex;
pub mod spectral_bft;
pub mod ordering;
pub mod integration;
pub mod gossip;
pub mod simd_acceleration;
pub mod shadow_mode;
pub mod k_parameter;  // Kristensen K-Parameter phase analysis
pub mod k_energy;     // K-Parameter enhanced energy functional
pub mod k_metrics;    // K-Parameter metrics and monitoring
pub mod k_effective;  // Effective K-Parameter with observer dependence (Harlow 2025)

pub use string_state::StringState;
pub use energy::EnergyFunctional;
pub use vertex::ResonanceVertex;
pub use spectral_bft::SpectralBFT;
pub use ordering::ResonanceOrdering;
pub use integration::{
    ResonanceCoordinator,
    ResonanceEnhancedVertex,
    ResonanceMetrics,
    NarwhalTransaction,
};

// Re-export AEGIS-QL types for convenience
pub use q_aegis_ql::{
    AegisQL,
    PublicKey as AegisPublicKey,
    SecretKey as AegisSecretKey,
    Signature as AegisSignature,
};
pub use gossip::{
    ResonanceMessage,
    ResonanceStateTracker,
    ConsensusInfo,
    ByzantineAlertInfo,
    RESONANCE_PROTOCOL,
    serialize_resonance_message,
    deserialize_resonance_message,
};
pub use simd_acceleration::{
    SimdEnergyComputer,
    SimdStats,
    BenchmarkResults,
    benchmark_simd_performance,
};
pub use shadow_mode::{
    ShadowModeCoordinator,
    ShadowModeConfig,
    ShadowModeMetrics,
    MigrationReport,
};
pub use k_parameter::{
    KParameterAnalyzer,
    PhaseTransition,
    ConsensusTuning,
};
pub use k_energy::{
    KEnhancedEnergy,
    PhaseAnalysis,
    PhaseRecommendation,
};
pub use k_metrics::{
    KParameterMetrics,
    TrendDirection,
};
pub use k_effective::{
    KEffectiveAnalyzer,
    KEffectiveResult,
    DecoherenceRates,
    ConsensusPhase,
    observer_factor,
    estimate_observer_entropy,
};

/// Resonance consensus error types
#[derive(Debug, thiserror::Error)]
pub enum ResonanceError {
    #[error("Energy minimization failed to converge")]
    ConvergenceError,

    #[error("Byzantine node detected: {0:?}")]
    ByzantineDetected([u8; 32]),

    #[error("Invalid string state: {0}")]
    InvalidState(String),

    #[error("Spectral analysis failed: {0}")]
    SpectralError(String),

    #[error("Ordering constraint violated")]
    OrderingViolation,
}

pub type Result<T> = std::result::Result<T, ResonanceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Verify all modules are accessible
    }
}
