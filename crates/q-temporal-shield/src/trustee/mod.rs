//! Trustee management for TemporalShield
//!
//! Handles key generation, storage, and HSM simulation.

pub mod keys;
pub mod hsm;

pub use keys::{TrusteePublicKey, TrusteePrivateKey, TrusteeKeyPair};
pub use hsm::{HsmSimulator, DistributedHsmNetwork, HsmAuditEntry, HsmOperation};
