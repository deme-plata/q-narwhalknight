//! P2P Protocol for Decentralized Proof Generation
//!
//! This module implements the libp2p-based protocol for decentralized
//! recursive proof generation. Multiple prover nodes compete to generate
//! epoch proofs, with the first valid proof being accepted.
//!
//! ## Gossipsub Topics
//!
//! - `/qnk/epoch-proof-task` - Announces when an epoch needs proving
//! - `/qnk/epoch-proofs` - Broadcasts completed epoch proofs
//! - `/qnk/proof-verification` - Verification results (optional)
//! - `/qnk/light-client-request` - Light client proof requests
//! - `/qnk/light-client-response` - Light client proof responses

pub mod messages;
pub mod prover_node;
pub mod topics;

pub use messages::{EpochProofSubmission, EpochProofTask, LightClientProofRequest, LightClientProofResponse};
pub use prover_node::ProverNode;
pub use topics::RecursiveProofTopics;
