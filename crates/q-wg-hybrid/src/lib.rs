//! Rosenpass-style hybrid PQ session-key augmentation for WireGuard.
//!
//! See `Cargo.toml` for the threat model. This file holds the rotation
//! state machine + public API. The Linux integration is in `wg_backend.rs`.
//!
//! ## API shape
//!
//! ```ignore
//! use q_wg_hybrid::{HybridRotator, RotatorConfig, KyberKeypair, WireGuardBackend};
//!
//! let our_keys = KyberKeypair::generate();
//! let mut rotator = HybridRotator::new(
//!     RotatorConfig { peer_pubkey: peer_kyber_pubkey, wg_iface: "wg0".into(), wg_peer: "ABC...".into() },
//!     our_keys,
//!     Box::new(LinuxWgCli),
//! );
//!
//! loop {
//!     tokio::time::sleep(EPOCH).await;
//!     let envelope = rotator.rotate()?;
//!     // publish `envelope.ciphertext` to peer via Quillon-mesh gossipsub
//!     gossipsub.publish("quillon-mesh/wg-rekey", envelope.ciphertext)?;
//! }
//! ```
//!
//! ## Why not a full Rosenpass implementation
//!
//! Rosenpass replaces the WireGuard handshake with a 3-message PQ exchange
//! (Classic McEliece + Kyber). That's a multi-week port and brings a much
//! larger Rosenpass-protocol surface area. This crate is the 80%-of-the-
//! benefit / 20%-of-the-cost path: we keep WireGuard's handshake and just
//! continuously rotate its PSK using Kyber-encapsulated entropy. An attacker
//! who breaks Curve25519 still needs to break Kyber-1024 to recover any
//! session that used a hybrid PSK.

pub mod rotator;
pub mod wg_backend;

pub use rotator::{HybridRotator, RotatorConfig, RotationEnvelope, RotationError};
pub use wg_backend::{WireGuardBackend, WgBackendError};

use pqcrypto_kyber::kyber1024;
use pqcrypto_traits::kem::{PublicKey, SecretKey};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Wire format for our Kyber-1024 public key. Stored on disk + announced
/// on gossipsub. Kyber-1024 pubkey is 1568 bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KyberPublicKey(pub Vec<u8>);

impl KyberPublicKey {
    pub fn as_bytes(&self) -> &[u8] { &self.0 }
}

/// Local Kyber-1024 keypair. The secret never leaves the host. Drops zeroed.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct KyberKeypair {
    pub pk_bytes: Vec<u8>,
    pub sk_bytes: Vec<u8>,
}

impl std::fmt::Debug for KyberKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KyberKeypair {{ pk_len: {}, sk_len: <redacted> }}", self.pk_bytes.len())
    }
}

impl KyberKeypair {
    /// Fresh keypair. Uses pqcrypto's RNG (which uses getrandom under the hood).
    pub fn generate() -> Self {
        let (pk, sk) = kyber1024::keypair();
        Self {
            pk_bytes: pk.as_bytes().to_vec(),
            sk_bytes: sk.as_bytes().to_vec(),
        }
    }

    pub fn public(&self) -> KyberPublicKey {
        KyberPublicKey(self.pk_bytes.clone())
    }
}

/// Public errors surfaced by the hybrid rotation logic. WireGuard backend
/// errors are wrapped here so callers see one error type.
#[derive(Debug, Error)]
pub enum HybridError {
    #[error("kyber operation failed: {0}")]
    Kyber(String),
    #[error("wireguard backend failed: {0}")]
    WgBackend(#[from] WgBackendError),
    #[error("rotation policy violation: {0}")]
    Policy(String),
}
