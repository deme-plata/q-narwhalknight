//! zk-STARK proof system for TemporalShield
//!
//! Provides STARK proofs for Shamir secret sharing consistency.
//! NO TRUSTED SETUP - all randomness is derived from public transcript.

pub mod air;
pub mod prover;
pub mod verifier;
pub mod trace;
pub mod utils;

pub use air::ShamirConsistencyAir;
pub use prover::generate_proof;
pub use verifier::verify_proof;
