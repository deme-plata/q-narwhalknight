//! Circuit gadgets for recursive proof construction
//!
//! This module provides arithmetic circuit gadgets that are used as building blocks
//! for the recursive epoch transition circuits.

pub mod dilithium;
pub mod merkle;
pub mod poseidon;

pub use dilithium::DilithiumVerifierGadget;
pub use merkle::MerkleTreeGadget;
pub use poseidon::PoseidonGadget;
