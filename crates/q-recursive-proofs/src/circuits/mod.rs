//! High-level circuits for recursive proof construction
//!
//! This module provides the main circuits used in the recursive proof chain:
//!
//! - `LatticeGuardVerifierCircuit`: Verifies a LatticeGuard proof inside the circuit (recursion)
//! - `BFTSignatureCircuit`: Verifies BFT threshold signatures
//! - `StateTransitionCircuit`: Verifies epoch state transitions
//! - `EpochTransitionCircuit`: Complete epoch proof combining all components

pub mod bft_signature;
pub mod epoch_transition;
pub mod lattice_verifier;
pub mod state_transition;

pub use bft_signature::BFTSignatureCircuit;
pub use epoch_transition::EpochTransitionCircuit;
pub use lattice_verifier::LatticeGuardVerifierCircuit;
pub use state_transition::StateTransitionCircuit;
